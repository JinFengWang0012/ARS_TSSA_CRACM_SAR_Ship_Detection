import argparse
import copy
import math
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.runner import Runner, load_checkpoint

from mmrotate.registry import MODELS
from mmrotate.utils import register_all_modules

ROOT = Path(__file__).resolve().parents[2]


class FooterProgressStream:

    def __init__(self, manager, wrapped):
        self.manager = manager
        self.wrapped = wrapped
        self.encoding = getattr(wrapped, 'encoding', None)
        self.errors = getattr(wrapped, 'errors', None)

    def write(self, text):
        if not text:
            return 0
        self.manager.before_external_write()
        written = self.wrapped.write(text)
        self.manager.after_external_write(text)
        return written

    def flush(self):
        return self.wrapped.flush()

    def isatty(self):
        return self.wrapped.isatty()

    def fileno(self):
        return self.wrapped.fileno()

    @property
    def buffer(self):
        return getattr(self.wrapped, 'buffer', None)


class FooterProgressManager:

    def __init__(self):
        self.orig_stdout = sys.stdout
        self.orig_stderr = sys.stderr
        self.enabled = bool(
            getattr(self.orig_stdout, 'isatty', lambda: False)()
            and getattr(self.orig_stderr, 'isatty', lambda: False)())
        self.lock = threading.RLock()
        self.footer_text = ''
        self.footer_visible = False
        self.cursor_at_line_start = True
        self.installed = False

    def install(self):
        if not self.enabled or self.installed:
            return
        sys.stdout = FooterProgressStream(self, self.orig_stdout)
        sys.stderr = FooterProgressStream(self, self.orig_stderr)
        self.installed = True

    def restore(self):
        if not self.installed:
            return
        with self.lock:
            self._clear_footer_locked()
        sys.stdout = self.orig_stdout
        sys.stderr = self.orig_stderr
        self.installed = False

    def set_footer(self, text):
        if not self.enabled:
            return
        with self.lock:
            self._clear_footer_locked()
            self.footer_text = text
            self._draw_footer_locked()
            self.orig_stdout.flush()

    def clear_footer(self):
        if not self.enabled:
            return
        with self.lock:
            self._clear_footer_locked()
            self.footer_text = ''
            self.orig_stdout.flush()

    def before_external_write(self):
        if not self.enabled:
            return
        with self.lock:
            self._clear_footer_locked()

    def after_external_write(self, text):
        if not self.enabled:
            return
        with self.lock:
            self.cursor_at_line_start = text.endswith('\n') or text.endswith('\r')
            self._draw_footer_locked()
            self.orig_stdout.flush()

    def _draw_footer_locked(self):
        if not self.footer_text:
            return
        if not self.cursor_at_line_start:
            self.orig_stdout.write('\n')
        self.orig_stdout.write(self.footer_text)
        self.footer_visible = True
        self.cursor_at_line_start = False

    def _clear_footer_locked(self):
        if not self.footer_visible:
            return
        self.orig_stdout.write('\r\x1b[2K')
        self.footer_visible = False
        self.cursor_at_line_start = True


FOOTER_PROGRESS = FooterProgressManager()

ABLATION_CONFIGS = {
    'A': ROOT / 'configs' / 'sar-wjf' / 'ablation_A_baseline_resnet50_fpn_rotatedfcos.py',
    'B': ROOT / 'configs' / 'sar-wjf' / 'ablation_B_attnres_backbone_fpn_rotatedfcos.py',
    'C': ROOT / 'configs' / 'sar-wjf' / 'ablation_C_attnres_backbone_pyramidattnneck_rotatedfcos.py',
    'D': ROOT / 'configs' / 'sar-wjf' / 'ablation_D_full_attnres_pyramidattnneck_acmconsistency.py',
}

COMPARISON_TABLES = {
    'backbone': {
        'SE': ROOT / 'configs' / 'sar-wjf' / 'rotated-fcos-r50_ResNet50_PyramidAttnFusion_SAACM_SE.py',
        'CBAM': ROOT / 'configs' / 'sar-wjf' / 'rotated-fcos-r50_ResNet50_PyramidAttnFusion_SAACM_CBAM.py',
        'ECA': ROOT / 'configs' / 'sar-wjf' / 'rotated-fcos-r50_ResNet50_PyramidAttnFusion_SAACM_ECA.py',
    },
    'neck': {
        'PAFPN': ROOT / 'configs' / 'sar-wjf' / 'rotated-fcos-r50_attnres_stage_backbone_SAACM_PAFPN.py',
        'NAS-FPN': ROOT / 'configs' / 'sar-wjf' / 'rotated-fcos-r50_attnres_stage_backbone_SAACM_NASFPN.py',
        'BiFPN': ROOT / 'configs' / 'sar-wjf' / 'rotated-fcos-r50_attnres_stage_backbone_SAACM_BiFPN.py',
        'AugFPN': ROOT / 'configs' / 'sar-wjf' / 'rotated-fcos-r50_attnres_stage_backbone_SAACM_AugFPN.py',
        'GraphFPN': ROOT / 'configs' / 'sar-wjf' / 'rotated-fcos-r50_attnres_stage_backbone_SAACM_GraphFPN.py',
        'RFFP-Neck': ROOT / 'configs' / 'sar-wjf' / 'rotated-fcos-r50_attnres_stage_backbone_SAACM_RFFPNeck.py',
        'SPAFPN': ROOT / 'configs' / 'sar-wjf' / 'rotated-fcos-r50_attnres_stage_backbone_SAACM_SPAFPN.py',
    },
    'angle': {
        'CSL': ROOT / 'configs' / 'sar-wjf' / 'rotated-fcos-rsdd-r50_attnres_pyramid_backbone_RotatedFCOSHead_CSL.py',
        'DCL-like': ROOT / 'configs' / 'sar-wjf' / 'rotated-fcos-rsdd-r50_attnres_pyramid_backbone_RotatedFCOSHead_DCLLikeCFUCR.py',
        'KLD-based Regression': ROOT / 'configs' / 'sar-wjf' / 'rotated-fcos-rsdd-r50_attnres_pyramid_backbone_RotatedFCOSHead_KLDCDM.py',
        'ACM': ROOT / 'configs' / 'sar-wjf' / 'rotated-fcos-r50_attnres_stage_backbone_PyramidAttnNeck_RotatedFCOSHead_ACM.py',
    }
    ,
    'module_ablation': {
        '-Backbone (ResNet50)': ROOT / 'configs' / 'sar-wjf' / 'ablation_minus_backbone_resnet50_pyramidattn_saacm.py',
        '-Neck (FPN)': ROOT / 'configs' / 'sar-wjf' / 'ablation_minus_neck_fpn_attnres_saacm.py',
    }
}


DATASET_PRESETS = {
    'rsdd': dict(
        data_root='data/rsdd/',
        train_ann='ImageSets/train.json',
        ann_file='ImageSets/test.json',
        img_prefix='JPEGImages/',
        scale=(1024, 1024)),
    'ssdd': dict(
        data_root='data/ssdd/',
        train_ann='annotations/train.json',
        ann_file='annotations/test.json',
        img_prefix='train2017/',
        scale=(512, 512)),
}


def parse_args():
    parser = argparse.ArgumentParser(description='Fast one-batch FPS benchmark')
    parser.add_argument('config', nargs='?', help='config file path')
    parser.add_argument('--checkpoint', default=None, help='checkpoint file')
    parser.add_argument(
        '--dataset',
        choices=['rsdd', 'ssdd'],
        help='dataset preset used to override test dataloader')
    parser.add_argument('--max-iter', type=int, default=20)
    parser.add_argument('--num-warmup', type=int, default=5)
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='number of test samples to cache and average over')
    parser.add_argument(
        '--mode',
        choices=['test_step', 'forward'],
        default='test_step',
        help='benchmark end-to-end test_step or pure network forward')
    parser.add_argument(
        '--suite-ablation',
        action='store_true',
        help='run A/B/C/D on both RSDD and SSDD and save a txt summary')
    parser.add_argument(
        '--output',
        default=str(ROOT / 'work_dirs' / 'ablation_fps_suite.txt'),
        help='txt path used when --suite-ablation is enabled')
    parser.add_argument(
        '--suite-train-benchmark',
        action='store_true',
        help='train each A/B/C/D on RSDD and SSDD for one epoch, then benchmark FPS and save txt')
    parser.add_argument(
        '--train-epochs',
        type=int,
        default=1,
        help='epochs used by --suite-train-benchmark')
    parser.add_argument(
        '--suite-comparisons',
        action='store_true',
        help='train and benchmark backbone/neck/angle comparison experiments')
    parser.add_argument(
        '--comparison-output',
        default=str(ROOT / 'work_dirs' / 'comparison_tables_90ep.txt'),
        help='txt output path used by --suite-comparisons')
    parser.add_argument(
        '--early-stop-patience',
        type=int,
        default=10,
        help='stop a training experiment when AP50 does not improve for N val epochs')
    parser.add_argument(
        '--val-interval',
        type=int,
        default=1,
        help='validation interval used by training suites')
    parser.add_argument(
        '--only-table',
        choices=['backbone', 'neck', 'angle', 'module_ablation'],
        default=None,
        help='run only one comparison table')
    parser.add_argument(
        '--run-tag',
        default='',
        help='optional suffix added to work_dirs so a rerun does not reuse old results')
    parser.add_argument(
        '--existing-results',
        nargs='*',
        default=[],
        help='existing comparison txt files whose rows should be reused directly')
    return parser.parse_args()


def _set_resize_scale(pipeline, scale):
    if pipeline is None:
        return
    for transform in pipeline:
        if transform.get('type') == 'mmdet.Resize':
            transform['scale'] = scale


def apply_dataset_preset_to_cfg(cfg, dataset_name):
    preset = DATASET_PRESETS[dataset_name]
    if cfg.get('train_pipeline', None) is not None:
        _set_resize_scale(cfg.train_pipeline, preset['scale'])
    if cfg.get('val_pipeline', None) is not None:
        _set_resize_scale(cfg.val_pipeline, preset['scale'])
    if cfg.get('test_pipeline', None) is not None:
        _set_resize_scale(cfg.test_pipeline, preset['scale'])

    if cfg.get('train_dataloader', None) is not None:
        train_dataset = cfg.train_dataloader['dataset']
        train_dataset['data_root'] = preset['data_root']
        train_dataset['ann_file'] = preset['train_ann']
        train_dataset['data_prefix']['img'] = preset['img_prefix']
        if cfg.get('train_pipeline', None) is not None:
            train_dataset['pipeline'] = cfg.train_pipeline
        _set_resize_scale(train_dataset.get('pipeline', None), preset['scale'])

    for loader_name in ['val_dataloader', 'test_dataloader']:
        if cfg.get(loader_name, None) is None:
            continue
        dataset_cfg = cfg[loader_name]['dataset']
        dataset_cfg['data_root'] = preset['data_root']
        dataset_cfg['ann_file'] = preset['ann_file']
        dataset_cfg['data_prefix']['img'] = preset['img_prefix']
        if loader_name == 'val_dataloader' and cfg.get('val_pipeline', None) is not None:
            dataset_cfg['pipeline'] = cfg.val_pipeline
        if loader_name == 'test_dataloader' and cfg.get('test_pipeline', None) is not None:
            dataset_cfg['pipeline'] = cfg.test_pipeline
        _set_resize_scale(dataset_cfg.get('pipeline', None), preset['scale'])


def build_test_dataloader(cfg, dataset_name):
    dataloader_cfg = copy.deepcopy(cfg.test_dataloader)
    preset = DATASET_PRESETS[dataset_name]
    dataset_cfg = dataloader_cfg['dataset']
    dataset_cfg['data_root'] = preset['data_root']
    dataset_cfg['ann_file'] = preset['ann_file']
    dataset_cfg['data_prefix']['img'] = preset['img_prefix']
    if cfg.get('test_pipeline', None) is not None:
        dataset_cfg['pipeline'] = cfg.test_pipeline
    _set_resize_scale(dataset_cfg.get('pipeline', None), preset['scale'])

    dataloader_cfg['batch_size'] = 1
    dataloader_cfg['num_workers'] = 0
    dataloader_cfg['persistent_workers'] = False
    return Runner.build_dataloader(dataloader_cfg)


def collect_batches(dataloader, num_samples):
    batches = []
    data_iter = iter(dataloader)
    for _ in range(num_samples):
        try:
            batches.append(next(data_iter))
        except StopIteration:
            break
    if not batches:
        raise RuntimeError('No batch collected from test dataloader.')
    return batches


def run_model_once(model, batch, mode):
    with torch.no_grad():
        if mode == 'test_step':
            model.test_step(batch)
        else:
            processed = model.data_preprocessor(batch, False)
            model(
                processed['inputs'],
                data_samples=processed.get('data_samples', None),
                mode='tensor')


def benchmark_model(config_path, checkpoint, dataset_name, max_iter, num_warmup,
                    num_samples, mode):
    cfg = Config.fromfile(config_path)
    apply_dataset_preset_to_cfg(cfg, dataset_name)
    model = MODELS.build(cfg.model)

    if checkpoint:
        load_checkpoint(model, checkpoint, map_location='cpu')

    model = model.cuda()
    model.eval()

    dataloader = build_test_dataloader(cfg, dataset_name)
    batches = collect_batches(dataloader, num_samples)

    for idx in range(num_warmup):
        batch = batches[idx % len(batches)]
        run_model_once(model, batch, mode)
        torch.cuda.synchronize()

    pure_inf_time = 0.0
    for idx in range(max_iter):
        batch = batches[idx % len(batches)]
        torch.cuda.synchronize()
        start = time.perf_counter()
        run_model_once(model, batch, mode)
        torch.cuda.synchronize()
        pure_inf_time += time.perf_counter() - start

    fps = max_iter / pure_inf_time
    return fps, 1000.0 / fps, len(batches)


def find_checkpoint(work_dir):
    work_dir = Path(work_dir)
    bests = sorted(work_dir.glob('best_r_coco_bbox_mAP_50_epoch_*.pth'))
    if bests:
        return str(bests[-1])
    epoch_ones = sorted(work_dir.glob('epoch_1.pth'))
    if epoch_ones:
        return str(epoch_ones[-1])
    epochs = sorted(work_dir.glob('epoch_*.pth'))
    if epochs:
        return str(epochs[-1])
    raise FileNotFoundError(f'No checkpoint found in {work_dir}')


def cleanup_checkpoints(work_dir, keep_checkpoint):
    work_dir = Path(work_dir)
    keep = Path(keep_checkpoint).resolve()
    for ckpt in work_dir.glob('*.pth'):
        if ckpt.resolve() != keep:
            ckpt.unlink(missing_ok=True)


def find_dumped_config(work_dir, run_name):
    work_dir = Path(work_dir)
    preferred = work_dir / f'{run_name}.py'
    if preferred.exists():
        return str(preferred)
    py_files = sorted(work_dir.glob('*.py'))
    if py_files:
        return str(py_files[0])
    raise FileNotFoundError(f'No dumped config found in {work_dir}')


def find_last_training_checkpoint(work_dir):
    work_dir = Path(work_dir)
    last_file = work_dir / 'last_checkpoint'
    if last_file.exists():
        ckpt_path = Path(last_file.read_text(encoding='utf-8').strip())
        if ckpt_path.exists():
            return str(ckpt_path)
    epochs = sorted(work_dir.glob('epoch_*.pth'))
    if epochs:
        return str(epochs[-1])
    return None


def extract_epoch_from_checkpoint(checkpoint_path):
    if checkpoint_path is None:
        return None
    stem = Path(checkpoint_path).stem
    marker = 'epoch_'
    if marker not in stem:
        return None
    suffix = stem.split(marker, 1)[1]
    digits = []
    for ch in suffix:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    if not digits:
        return None
    return int(''.join(digits))


def completion_marker_path(work_dir):
    return Path(work_dir) / 'comparison_done.txt'


def mark_experiment_done(work_dir, train_epochs, checkpoint):
    marker = completion_marker_path(work_dir)
    marker.write_text(
        f'train_epochs={train_epochs}\ncheckpoint={checkpoint}\n',
        encoding='utf-8')


def experiment_done(work_dir, train_epochs):
    marker = completion_marker_path(work_dir)
    if not marker.exists():
        return False
    text = marker.read_text(encoding='utf-8')
    return f'train_epochs={train_epochs}' in text


def experiment_has_reusable_best(work_dir, run_name):
    work_dir = Path(work_dir)
    if not work_dir.exists():
        return False
    try:
        find_dumped_config(work_dir, run_name)
        find_checkpoint(work_dir)
    except FileNotFoundError:
        return False
    return True


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f'{hours:02d}:{minutes:02d}:{secs:02d}'
    return f'{minutes:02d}:{secs:02d}'


def make_progress_bar(done, total, width=28):
    if total <= 0:
        total = 1
    ratio = min(max(done / total, 0.0), 1.0)
    filled = int(round(ratio * width))
    filled = min(filled, width)
    return '[' + '#' * filled + '-' * (width - filled) + ']'


def parse_existing_result_tables(paths):
    existing = {}
    for path in paths:
        file_path = Path(path)
        if not file_path.exists():
            continue
        current_table = None
        current_dataset = None
        for raw_line in file_path.read_text(encoding='utf-8').splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith('[') and line.endswith('TABLE]'):
                current_table = line.strip('[]').replace(' TABLE', '').lower()
                current_dataset = None
                continue
            if line.startswith('Dataset:'):
                current_dataset = line.split(':', 1)[1].strip().lower()
                continue
            if not line.startswith('|') or line.startswith('|---'):
                continue
            parts = [part.strip() for part in line.split('|')[1:-1]]
            if not parts or parts[0] == 'Method':
                continue
            if current_table is None or current_dataset is None:
                continue
            method_name = parts[0]
            existing[(current_table, current_dataset, method_name)] = line
    return existing


def count_params_million(config_path):
    cfg = Config.fromfile(config_path)
    model = MODELS.build(cfg.model)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    return params


def _metric_value(metrics, *candidate_keys):
    for key in candidate_keys:
        if key in metrics:
            return float(metrics[key])
    return math.nan


def evaluate_model(config_path, checkpoint, dataset_name, eval_work_dir=None):
    cfg = Config.fromfile(config_path)
    apply_dataset_preset_to_cfg(cfg, dataset_name)
    if eval_work_dir is None:
        eval_work_dir = ROOT / 'work_dirs' / '_tmp_eval'
    cfg.work_dir = str(eval_work_dir)
    cfg.load_from = checkpoint
    cfg.resume = False
    cfg.default_hooks = copy.deepcopy(cfg.get('default_hooks', {}))
    cfg.default_hooks.pop('checkpoint', None)
    runner = Runner.from_cfg(cfg)
    metrics = runner.val()
    return {
        'AP': _metric_value(metrics, 'r_coco/bbox_mAP', 'bbox_mAP'),
        'AP50': _metric_value(metrics, 'r_coco/bbox_mAP_50', 'bbox_mAP_50'),
        'AP75': _metric_value(metrics, 'r_coco/bbox_mAP_75', 'bbox_mAP_75'),
    }


def train_one_epoch(config_path, dataset_name, train_epochs, run_name, val_interval,
                    early_stop_patience):
    cfg = Config.fromfile(config_path)
    apply_dataset_preset_to_cfg(cfg, dataset_name)
    cfg.work_dir = str(ROOT / 'work_dirs' / run_name)
    cfg.train_cfg = dict(
        type='EpochBasedTrainLoop',
        max_epochs=train_epochs,
        val_interval=val_interval)
    cfg.default_hooks = copy.deepcopy(cfg.get('default_hooks', {}))
    cfg.default_hooks['checkpoint'] = dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=1,
        save_last=True,
        save_best='r_coco/bbox_mAP_50',
        rule='greater')
    cfg.custom_hooks = copy.deepcopy(cfg.get('custom_hooks', []))
    cfg.custom_hooks.append(
        dict(
            type='EarlyStoppingHook',
            monitor='r_coco/bbox_mAP_50',
            rule='greater',
            min_delta=0.0,
            patience=early_stop_patience))
    cfg.resume = False
    cfg.load_from = None
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    last_ckpt = find_last_training_checkpoint(cfg.work_dir)
    if last_ckpt is not None:
        cfg.resume = True
    dumped_cfg = Path(cfg.work_dir) / f'{run_name}.py'
    cfg.dump(str(dumped_cfg))
    runner = Runner.from_cfg(cfg)
    runner.train()
    checkpoint = find_checkpoint(cfg.work_dir)
    cleanup_checkpoints(cfg.work_dir, checkpoint)
    return str(dumped_cfg), checkpoint


def get_or_train_experiment(config_path, dataset_name, train_epochs, run_name,
                            val_interval, early_stop_patience):
    work_dir = ROOT / 'work_dirs' / run_name
    if work_dir.exists():
        last_ckpt = find_last_training_checkpoint(work_dir)
        best_ckpt = None
        if experiment_has_reusable_best(work_dir, run_name):
            dumped_cfg = find_dumped_config(work_dir, run_name)
            best_ckpt = find_checkpoint(work_dir)
            best_epoch = extract_epoch_from_checkpoint(best_ckpt)
            last_epoch = extract_epoch_from_checkpoint(last_ckpt)
            if best_epoch is not None and last_epoch is not None and best_epoch > last_epoch:
                mark_experiment_done(work_dir, train_epochs, best_ckpt)
                return dumped_cfg, best_ckpt, True

        if experiment_done(work_dir, train_epochs):
            try:
                dumped_cfg = find_dumped_config(work_dir, run_name)
                checkpoint = find_checkpoint(work_dir)
                return dumped_cfg, checkpoint, True
            except FileNotFoundError:
                pass

        # If a previous run already produced a best checkpoint and no active
        # training state is left, reuse it even when the explicit done marker
        # was never written because the script stopped later in the pipeline.
        if last_ckpt is None and experiment_has_reusable_best(work_dir, run_name):
            dumped_cfg = find_dumped_config(work_dir, run_name)
            checkpoint = find_checkpoint(work_dir)
            mark_experiment_done(work_dir, train_epochs, checkpoint)
            return dumped_cfg, checkpoint, True
    dumped_cfg, checkpoint = train_one_epoch(
        config_path,
        dataset_name,
        train_epochs,
        run_name,
        val_interval=val_interval,
        early_stop_patience=early_stop_patience)
    return dumped_cfg, checkpoint, False


def run_suite(args):
    lines = []
    lines.append('Ablation FPS Suite')
    lines.append(f'Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append(f'Max iter: {args.max_iter}')
    lines.append(f'Warmup: {args.num_warmup}')
    lines.append(f'Num samples: {args.num_samples}')
    lines.append(f'Mode: {args.mode}')
    lines.append('')
    lines.append('| Config | RSDD FPS | RSDD ms/img | SSDD FPS | SSDD ms/img |')
    lines.append('|---|---:|---:|---:|---:|')

    for name in ['A', 'B', 'C', 'D']:
        config_path = str(ABLATION_CONFIGS[name])
        print(f'Running {name}-RSDD...', flush=True)
        rsdd_fps, rsdd_ms, _ = benchmark_model(
            config_path, None, 'rsdd', args.max_iter, args.num_warmup,
            args.num_samples, args.mode)
        print(f'Running {name}-SSDD...', flush=True)
        ssdd_fps, ssdd_ms, _ = benchmark_model(
            config_path, None, 'ssdd', args.max_iter, args.num_warmup,
            args.num_samples, args.mode)
        lines.append(
            f'| {name} | {rsdd_fps:.2f} | {rsdd_ms:.2f} | {ssdd_fps:.2f} | {ssdd_ms:.2f} |'
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = '\n'.join(lines)
    output_path.write_text(text, encoding='utf-8')
    print(text)
    print(f'\nSaved to: {output_path}')


def run_train_benchmark_suite(args):
    lines = []
    lines.append('Ablation Train-Then-Benchmark FPS Suite')
    lines.append(f'Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append(f'Train epochs: {args.train_epochs}')
    lines.append(f'Max iter: {args.max_iter}')
    lines.append(f'Warmup: {args.num_warmup}')
    lines.append(f'Num samples: {args.num_samples}')
    lines.append(f'Mode: {args.mode}')
    lines.append('')
    lines.append('| Config | Dataset | Checkpoint | FPS | ms/img |')
    lines.append('|---|---|---|---:|---:|')

    for name in ['A', 'B', 'C', 'D']:
        for dataset_name in ['rsdd', 'ssdd']:
            run_name = f'fps_train1ep_{name}_{dataset_name}'
            config_path = str(ABLATION_CONFIGS[name])
            print(f'Training {name}-{dataset_name} for {args.train_epochs} epoch(s)...', flush=True)
            dumped_cfg, checkpoint = train_one_epoch(
                config_path, dataset_name, args.train_epochs, run_name)
            print(f'Benchmarking {name}-{dataset_name}...', flush=True)
            fps, ms_per_img, _ = benchmark_model(
                dumped_cfg, checkpoint, dataset_name, args.max_iter,
                args.num_warmup, args.num_samples, args.mode)
            lines.append(
                f'| {name} | {dataset_name.upper()} | {Path(checkpoint).name} | {fps:.2f} | {ms_per_img:.2f} |')

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = '\n'.join(lines)
    output_path.write_text(text, encoding='utf-8')
    print(text)
    print(f'\nSaved to: {output_path}')


def run_comparison_tables(args):
    selected_tables = COMPARISON_TABLES
    if args.only_table is not None:
        selected_tables = {args.only_table: COMPARISON_TABLES[args.only_table]}

    run_tag_suffix = ''
    if args.run_tag:
        run_tag_suffix = '_' + args.run_tag.replace(' ', '_').replace('-', '_')

    lines = []
    lines.append('Comparison Experiment Summary')
    lines.append(f'Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append(f'Train epochs: {args.train_epochs}')
    lines.append(f'Val interval: {args.val_interval}')
    lines.append(f'Early stop patience: {args.early_stop_patience}')
    lines.append(f'FPS mode: {args.mode}')
    lines.append(f'FPS max iter: {args.max_iter}')
    lines.append(f'FPS warmup: {args.num_warmup}')
    lines.append(f'FPS num samples: {args.num_samples}')
    lines.append('')

    total_experiments = sum(len(config_map) * 2 for config_map in selected_tables.values())
    completed_experiments = 0
    reused_experiments = 0
    comparison_start_time = time.perf_counter()
    table_totals = {
        table_name: len(config_map) * 2
        for table_name, config_map in selected_tables.items()
    }
    table_finished = {table_name: 0 for table_name in selected_tables}
    existing_rows = parse_existing_result_tables(args.existing_results)

    def update_footer(current_label, phase):
        elapsed = time.perf_counter() - comparison_start_time
        ratio = completed_experiments / total_experiments if total_experiments else 1.0
        progress_bar = make_progress_bar(completed_experiments, total_experiments)
        measured_done = max(completed_experiments - reused_experiments, 0)
        if measured_done > 0:
            avg_seconds = elapsed / measured_done
            remaining = avg_seconds * (total_experiments - completed_experiments)
            avg_text = format_duration(avg_seconds)
            eta_text = format_duration(remaining)
        elif reused_experiments > 0:
            avg_text = 'reused'
            eta_text = format_duration(0)
        else:
            avg_text = 'estimating...'
            eta_text = 'estimating...'
        summary_text = (
            f'Backbone: {table_finished.get("backbone", 0)}/{table_totals.get("backbone", 0)} | '
            f'Neck: {table_finished.get("neck", 0)}/{table_totals.get("neck", 0)} | '
            f'Angle: {table_finished.get("angle", 0)}/{table_totals.get("angle", 0)} | '
            f'Overall: {completed_experiments}/{total_experiments}'
        )
        FOOTER_PROGRESS.set_footer(
            f'{summary_text}\n'
            f'Progress {progress_bar} {completed_experiments}/{total_experiments} '
            f'({ratio * 100:.1f}%) | {phase}: {current_label} | '
            f'Elapsed {format_duration(elapsed)} | Avg/exp {avg_text} | ETA {eta_text}')

    for table_name, config_map in selected_tables.items():
        for dataset_name in ['ssdd', 'rsdd']:
            for method_name in config_map:
                if (table_name, dataset_name, method_name) in existing_rows:
                    reused_experiments += 1
                    completed_experiments += 1
                    table_finished[table_name] += 1
                    continue
                run_name = f'{table_name}_{method_name}_{dataset_name}{run_tag_suffix}'.replace(
                    ' ', '_').replace('-', '_')
                work_dir = ROOT / 'work_dirs' / run_name
                if work_dir.exists() and (
                        experiment_done(work_dir, args.train_epochs)
                        or (find_last_training_checkpoint(work_dir) is None
                            and experiment_has_reusable_best(work_dir, run_name))):
                    reused_experiments += 1
                    completed_experiments += 1
                    table_finished[table_name] += 1

    def flush_output():
        output_path = Path(args.comparison_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('\n'.join(lines), encoding='utf-8')

    FOOTER_PROGRESS.install()
    try:
        update_footer('initializing', 'setup')
        for table_name, config_map in selected_tables.items():
            lines.append(f'[{table_name.upper()} TABLE]')
            for dataset_name in ['ssdd', 'rsdd']:
                lines.append(f'Dataset: {dataset_name.upper()}')
                lines.append('| Method | AP | AP50 | AP75 | Params (M) | FPS | Best Checkpoint |')
                lines.append('|---|---:|---:|---:|---:|---:|---|')
                for method_name, config_path in config_map.items():
                    existing_key = (table_name, dataset_name, method_name)
                    if existing_key in existing_rows:
                        lines.append(existing_rows[existing_key])
                        flush_output()
                        continue
                    run_name = f'{table_name}_{method_name}_{dataset_name}{run_tag_suffix}'.replace(
                        ' ', '_').replace('-', '_')
                    current_idx = completed_experiments + 1
                    current_label = f'[{current_idx}/{total_experiments}] {table_name}-{method_name}-{dataset_name}'
                    update_footer(current_label, 'training/resume')
                    print(f'Starting {current_label}', flush=True)
                    dumped_cfg, checkpoint, reused = get_or_train_experiment(
                        str(config_path), dataset_name, args.train_epochs, run_name,
                        val_interval=args.val_interval,
                        early_stop_patience=args.early_stop_patience)
                    if reused:
                        mark_experiment_done(ROOT / 'work_dirs' / run_name, args.train_epochs, checkpoint)
                        print(
                            f'Reusing existing checkpoint for {table_name}-{method_name}-{dataset_name}: '
                            f'{Path(checkpoint).name}',
                            flush=True)
                    else:
                        print(
                            f'Trained {table_name}-{method_name}-{dataset_name} for {args.train_epochs} epochs.',
                            flush=True)
                    update_footer(current_label, 'evaluating')
                    print(f'Evaluating {table_name}-{method_name}-{dataset_name}...', flush=True)
                    eval_work_dir = ROOT / 'work_dirs' / f'{run_name}_eval'
                    metric_dict = evaluate_model(
                        dumped_cfg, checkpoint, dataset_name, eval_work_dir=eval_work_dir)
                    update_footer(current_label, 'benchmarking fps')
                    print(f'Benchmarking {table_name}-{method_name}-{dataset_name}...', flush=True)
                    fps, _, _ = benchmark_model(
                        dumped_cfg, checkpoint, dataset_name, args.max_iter,
                        args.num_warmup, args.num_samples, args.mode)
                    params_m = count_params_million(dumped_cfg)
                    lines.append(
                        f"| {method_name} | {metric_dict['AP']:.4f} | {metric_dict['AP50']:.4f} | "
                        f"{metric_dict['AP75']:.4f} | {params_m:.2f} | {fps:.2f} | {Path(checkpoint).name} |"
                    )
                    mark_experiment_done(ROOT / 'work_dirs' / run_name, args.train_epochs, checkpoint)
                    if not reused:
                        completed_experiments += 1
                        if table_finished[table_name] < table_totals[table_name]:
                            table_finished[table_name] += 1
                    update_footer(current_label, 'completed')
                    flush_output()
                lines.append('')
            lines.append('')

        text = '\n'.join(lines)
        flush_output()
        FOOTER_PROGRESS.clear_footer()
        print(text)
        print(f'\nSaved to: {args.comparison_output}')
    finally:
        FOOTER_PROGRESS.clear_footer()
        FOOTER_PROGRESS.restore()


def main():
    register_all_modules()
    args = parse_args()

    if args.suite_train_benchmark:
        run_train_benchmark_suite(args)
        return

    if args.suite_comparisons:
        if args.train_epochs == 1:
            args.train_epochs = 90
        run_comparison_tables(args)
        return

    if args.suite_ablation:
        run_suite(args)
        return

    if not args.config or not args.dataset:
        raise ValueError('Single-model mode requires config and --dataset.')

    fps, ms_per_img, sample_count = benchmark_model(
        args.config, args.checkpoint, args.dataset, args.max_iter,
        args.num_warmup, args.num_samples, args.mode)
    print('=' * 30)
    print(f'Config: {args.config}')
    print(f'Dataset: {args.dataset}')
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Mode: {args.mode}')
    print(f'Warmup: {args.num_warmup}')
    print(f'Iter: {args.max_iter}')
    print(f'Samples: {sample_count}')
    print(f'FPS: {fps:.2f} img/s')
    print(f'Time per image: {ms_per_img:.2f} ms/img')
    print('=' * 30)


if __name__ == '__main__':
    main()
