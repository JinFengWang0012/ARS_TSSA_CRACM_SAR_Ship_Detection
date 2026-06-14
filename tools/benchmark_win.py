# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import time
from functools import partial
from typing import List, Optional, Union

import torch
import torch.nn as nn
from mmcv.cnn import fuse_conv_bn
from mmengine import MMLogger
from mmengine.config import Config, DictAction
from mmengine.device import get_max_cuda_memory
from mmengine.dist import get_world_size, init_dist
from mmengine.runner import Runner, load_checkpoint
from mmengine.utils import mkdir_or_exist
from mmengine.utils.dl_utils import set_multi_processing
from torch.nn.parallel import DistributedDataParallel

from mmrotate.registry import MODELS
from mmrotate.utils import register_all_modules

try:
    import psutil
except ImportError:
    psutil = None


def parse_args():
    parser = argparse.ArgumentParser(description='MMRotate benchmark (Windows-safe)')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument(
        '--task',
        choices=['inference', 'dataloader'],
        default='inference',
        help='Which task do you want to benchmark')
    parser.add_argument(
        '--repeat-num',
        type=int,
        default=1,
        help='number of repeat times of measurement for averaging the results')
    parser.add_argument(
        '--max-iter', type=int, default=200, help='num of max iter')
    parser.add_argument(
        '--log-interval', type=int, default=50, help='interval of logging')
    parser.add_argument(
        '--num-warmup', type=int, default=20, help='Number of warmup')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn')
    parser.add_argument(
        '--dataset-type',
        choices=['train', 'val', 'test'],
        default='test',
        help='Benchmark dataset type')
    parser.add_argument(
        '--work-dir',
        help='the directory to save benchmark metrics')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override settings in the used config')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def custom_round(value: Union[int, float],
                 factor: Union[int, float],
                 precision: int = 2) -> float:
    return round(value / factor, precision)


gb_round = partial(custom_round, factor=1024**3)


def print_log(msg: str, logger: Optional[MMLogger] = None) -> None:
    if logger is None:
        print(msg, flush=True)
    else:
        logger.info(msg)


def _safe_mem_fields(mem_info):
    rss = getattr(mem_info, 'rss', 0)
    uss = getattr(mem_info, 'uss', rss)
    pss = getattr(mem_info, 'pss', uss)
    return rss, uss, pss


def print_process_memory(process, logger: Optional[MMLogger] = None) -> None:
    if psutil is None:
        print_log('psutil is not installed; skip process memory logging.', logger)
        return

    mem_used = gb_round(psutil.virtual_memory().used)
    rss, uss, pss = _safe_mem_fields(process.memory_full_info())
    uss_mem = gb_round(uss)
    pss_mem = gb_round(pss)

    for child in process.children():
        _, child_uss, child_pss = _safe_mem_fields(child.memory_full_info())
        uss_mem += gb_round(child_uss)
        pss_mem += gb_round(child_pss)

    process_count = 1 + len(process.children())
    print_log(
        f'(GB) mem_used: {mem_used:.2f} | uss: {uss_mem:.2f} | '
        f'pss: {pss_mem:.2f} | total_proc: {process_count}',
        logger)


class BaseBenchmark:

    def __init__(self,
                 max_iter: int,
                 log_interval: int,
                 num_warmup: int,
                 logger: Optional[MMLogger] = None):
        self.max_iter = max_iter
        self.log_interval = log_interval
        self.num_warmup = num_warmup
        self.logger = logger

    def run(self, repeat_num: int = 1) -> dict:
        results = []
        for _ in range(repeat_num):
            results.append(self.run_once())
        return self.average_multiple_runs(results)

    def run_once(self) -> dict:
        raise NotImplementedError()

    def average_multiple_runs(self, results: List[dict]) -> dict:
        raise NotImplementedError()


class InferenceBenchmark(BaseBenchmark):

    def __init__(self,
                 cfg: Config,
                 checkpoint: str,
                 distributed: bool,
                 is_fuse_conv_bn: bool,
                 max_iter: int = 200,
                 log_interval: int = 50,
                 num_warmup: int = 20,
                 logger: Optional[MMLogger] = None):
        super().__init__(max_iter, log_interval, num_warmup, logger)

        assert get_world_size() == 1, (
            'Inference benchmark does not allow distributed multi-GPU')

        self.cfg = copy.deepcopy(cfg)
        self.distributed = distributed

        if psutil is None:
            raise ImportError('psutil is not installed, please install it by: pip install psutil')

        self._process = psutil.Process()
        env_cfg = self.cfg.get('env_cfg')
        if env_cfg.get('cudnn_benchmark'):
            torch.backends.cudnn.benchmark = True

        mp_cfg = env_cfg.get('mp_cfg', {})
        set_multi_processing(**mp_cfg, distributed=self.distributed)

        print_log('before build:', self.logger)
        print_process_memory(self._process, self.logger)

        self.model = self._init_model(checkpoint, is_fuse_conv_bn)

        dataloader_cfg = cfg.test_dataloader
        if self.cfg.get('test_pipeline', None) is not None:
            dataloader_cfg['dataset']['pipeline'] = self.cfg.test_pipeline
        dataloader_cfg['num_workers'] = 0
        dataloader_cfg['batch_size'] = 1
        dataloader_cfg['persistent_workers'] = False
        self.data_loader = Runner.build_dataloader(dataloader_cfg)

        print_log('after build:', self.logger)
        print_process_memory(self._process, self.logger)

    def _init_model(self, checkpoint: str, is_fuse_conv_bn: bool) -> nn.Module:
        model = MODELS.build(self.cfg.model)
        if checkpoint:
            load_checkpoint(model, checkpoint, map_location='cpu')
        else:
            print_log('No checkpoint provided, benchmarking with current model weights.', self.logger)
        if is_fuse_conv_bn:
            model = fuse_conv_bn(model)
        model = model.cuda()

        if self.distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=False)

        model.eval()
        return model

    def run_once(self) -> dict:
        pure_inf_time = 0
        fps = 0

        for i, data in enumerate(self.data_loader):
            if (i + 1) % self.log_interval == 0:
                print_log('==================================', self.logger)

            torch.cuda.synchronize()
            start_time = time.perf_counter()

            with torch.no_grad():
                self.model.test_step(data)

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time

            if i >= self.num_warmup:
                pure_inf_time += elapsed
                if (i + 1) % self.log_interval == 0:
                    fps = (i + 1 - self.num_warmup) / pure_inf_time
                    cuda_memory = get_max_cuda_memory()
                    print_log(
                        f'Done image [{i + 1:<3}/{self.max_iter}], '
                        f'fps: {fps:.1f} img/s, '
                        f'times per image: {1000 / fps:.1f} ms/img, '
                        f'cuda memory: {cuda_memory} MB',
                        self.logger)
                    print_process_memory(self._process, self.logger)

            if (i + 1) == self.max_iter:
                fps = (i + 1 - self.num_warmup) / pure_inf_time
                break

        return {'fps': fps}

    def average_multiple_runs(self, results: List[dict]) -> dict:
        print_log('============== Done ==================', self.logger)

        fps_list = [round(result['fps'], 1) for result in results]
        avg_fps = sum(fps_list) / len(fps_list)
        outputs = {'avg_fps': avg_fps, 'fps_list': fps_list}

        if len(fps_list) > 1:
            times_list = [round(1000 / result['fps'], 1) for result in results]
            avg_times = sum(times_list) / len(times_list)
            print_log(
                f'Overall fps: {fps_list}[{avg_fps:.1f}] img/s, '
                f'times per image: {times_list}[{avg_times:.1f}] ms/img',
                self.logger)
        else:
            print_log(
                f'Overall fps: {fps_list[0]:.1f} img/s, '
                f'times per image: {1000 / fps_list[0]:.1f} ms/img',
                self.logger)

        print_log(f'cuda memory: {get_max_cuda_memory()} MB', self.logger)
        print_process_memory(self._process, self.logger)
        return outputs


class DataLoaderBenchmark(BaseBenchmark):

    def __init__(self,
                 cfg: Config,
                 distributed: bool,
                 dataset_type: str,
                 max_iter: int = 200,
                 log_interval: int = 50,
                 num_warmup: int = 20,
                 logger: Optional[MMLogger] = None):
        super().__init__(max_iter, log_interval, num_warmup, logger)

        assert dataset_type in ['train', 'val', 'test']
        assert get_world_size() == 1, (
            'Dataloader benchmark does not allow distributed multi-GPU')

        self.cfg = copy.deepcopy(cfg)
        self.distributed = distributed

        if psutil is None:
            raise ImportError('psutil is not installed, please install it by: pip install psutil')
        self._process = psutil.Process()

        mp_cfg = self.cfg.get('env_cfg', {}).get('mp_cfg')
        if mp_cfg is not None:
            set_multi_processing(distributed=self.distributed, **mp_cfg)
        else:
            set_multi_processing(distributed=self.distributed)

        print_log('before build:', self.logger)
        print_process_memory(self._process, self.logger)

        if dataset_type == 'train':
            self.data_loader = Runner.build_dataloader(cfg.train_dataloader)
        elif dataset_type == 'test':
            self.data_loader = Runner.build_dataloader(cfg.test_dataloader)
        else:
            self.data_loader = Runner.build_dataloader(cfg.val_dataloader)

        self.batch_size = self.data_loader.batch_size
        self.num_workers = self.data_loader.num_workers

        print_log('after build:', self.logger)
        print_process_memory(self._process, self.logger)

    def run_once(self) -> dict:
        pure_inf_time = 0
        fps = 0

        start_time = time.perf_counter()
        for i, _data in enumerate(self.data_loader):
            elapsed = time.perf_counter() - start_time

            if (i + 1) % self.log_interval == 0:
                print_log('==================================', self.logger)

            if i >= self.num_warmup:
                pure_inf_time += elapsed
                if (i + 1) % self.log_interval == 0:
                    fps = (i + 1 - self.num_warmup) / pure_inf_time
                    print_log(
                        f'Done batch [{i + 1:<3}/{self.max_iter}], '
                        f'fps: {fps:.1f} batch/s, '
                        f'times per batch: {1000 / fps:.1f} ms/batch, '
                        f'batch size: {self.batch_size}, num_workers: {self.num_workers}',
                        self.logger)
                    print_process_memory(self._process, self.logger)

            if (i + 1) == self.max_iter:
                fps = (i + 1 - self.num_warmup) / pure_inf_time
                break

            start_time = time.perf_counter()

        return {'fps': fps}

    def average_multiple_runs(self, results: List[dict]) -> dict:
        print_log('============== Done ==================', self.logger)

        fps_list = [round(result['fps'], 1) for result in results]
        avg_fps = sum(fps_list) / len(fps_list)
        outputs = {'avg_fps': avg_fps, 'fps_list': fps_list}

        if len(fps_list) > 1:
            times_list = [round(1000 / result['fps'], 1) for result in results]
            avg_times = sum(times_list) / len(times_list)
            print_log(
                f'Overall fps: {fps_list}[{avg_fps:.1f}] batch/s, '
                f'times per batch: {times_list}[{avg_times:.1f}] ms/batch',
                self.logger)
        else:
            print_log(
                f'Overall fps: {fps_list[0]:.1f} batch/s, '
                f'times per batch: {1000 / fps_list[0]:.1f} ms/batch',
                self.logger)

        print_process_memory(self._process, self.logger)
        return outputs


def inference_benchmark(args, cfg, distributed, logger):
    return InferenceBenchmark(
        cfg,
        args.checkpoint,
        distributed,
        args.fuse_conv_bn,
        args.max_iter,
        args.log_interval,
        args.num_warmup,
        logger=logger)


def dataloader_benchmark(args, cfg, distributed, logger):
    return DataLoaderBenchmark(
        cfg,
        distributed,
        args.dataset_type,
        args.max_iter,
        args.log_interval,
        args.num_warmup,
        logger=logger)


def main():
    register_all_modules()

    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    distributed = False
    if args.launcher != 'none':
        init_dist(args.launcher, **cfg.get('env_cfg', {}).get('dist_cfg', {}))
        distributed = True

    log_file = None
    if args.work_dir:
        log_file = os.path.join(args.work_dir, 'benchmark.log')
        mkdir_or_exist(args.work_dir)

    logger = MMLogger.get_instance(
        'mmrotate-benchmark', log_file=log_file, log_level='INFO')

    benchmark = eval(f'{args.task}_benchmark')(args, cfg, distributed, logger)
    benchmark.run(args.repeat_num)


if __name__ == '__main__':
    main()
