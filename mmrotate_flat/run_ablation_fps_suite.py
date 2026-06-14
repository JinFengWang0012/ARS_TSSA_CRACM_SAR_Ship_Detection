import argparse
import copy
import gc
from datetime import datetime
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.runner import Runner, load_checkpoint

from mmrotate.registry import MODELS
from mmrotate.utils import register_all_modules


ROOT = Path(__file__).resolve().parents[2]

CONFIGS = {
    'A': ROOT / 'configs' / 'sar-wjf' / 'ablation_A_baseline_resnet50_fpn_rotatedfcos.py',
    'B': ROOT / 'configs' / 'sar-wjf' / 'ablation_B_attnres_backbone_fpn_rotatedfcos.py',
    'C': ROOT / 'configs' / 'sar-wjf' / 'ablation_C_attnres_backbone_pyramidattnneck_rotatedfcos.py',
    'D': ROOT / 'configs' / 'sar-wjf' / 'ablation_D_full_attnres_pyramidattnneck_acmconsistency.py',
}

CHECKPOINTS = {
    'A': ROOT / 'work_dirs' / 'ablation_A_baseline_resnet50_fpn_rotatedfcos' /
    'best_r_coco_bbox_mAP_50_epoch_1.pth',
    'B': ROOT / 'work_dirs' / 'ablation_B_attnres_backbone_fpn_rotatedfcos' /
    'best_r_coco_bbox_mAP_50_epoch_1.pth',
    'C': None,
    'D': None,
}

DATASET_PRESETS = {
    'rsdd': dict(
        data_root='data/rsdd/',
        ann_file='ImageSets/test.json',
        img_prefix='JPEGImages/'),
    'ssdd': dict(
        data_root='data/ssdd/',
        ann_file='annotations/test.json',
        img_prefix='train2017/'),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run A/B/C/D FPS suite on RSDD and SSDD and save a txt report.')
    parser.add_argument('--max-iter', type=int, default=20)
    parser.add_argument('--num-warmup', type=int, default=5)
    parser.add_argument(
        '--use-checkpoint',
        action='store_true',
        help='Load default checkpoint when available.')
    parser.add_argument(
        '--output',
        default=str(ROOT / 'work_dirs' / 'ablation_fps_suite.txt'),
        help='Path to save the txt report.')
    return parser.parse_args()


def build_test_dataloader(cfg, dataset_name):
    dataloader_cfg = copy.deepcopy(cfg.test_dataloader)
    preset = DATASET_PRESETS[dataset_name]
    dataset_cfg = dataloader_cfg['dataset']
    dataset_cfg['data_root'] = preset['data_root']
    dataset_cfg['ann_file'] = preset['ann_file']
    dataset_cfg['data_prefix']['img'] = preset['img_prefix']
    if cfg.get('test_pipeline', None) is not None:
        dataset_cfg['pipeline'] = cfg.test_pipeline

    dataloader_cfg['batch_size'] = 1
    dataloader_cfg['num_workers'] = 0
    dataloader_cfg['persistent_workers'] = False
    return Runner.build_dataloader(dataloader_cfg)


def benchmark_one(cfg_path, dataset_name, checkpoint_path, max_iter, num_warmup):
    cfg = Config.fromfile(str(cfg_path))
    model = MODELS.build(cfg.model)

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        load_checkpoint(model, str(checkpoint_path), map_location='cpu')
        ckpt_label = str(checkpoint_path)
    else:
        ckpt_label = 'None'

    model = model.cuda()
    model.eval()

    dataloader = build_test_dataloader(cfg, dataset_name)
    batch = next(iter(dataloader))

    for _ in range(num_warmup):
        with torch.no_grad():
            model.test_step(batch)
        torch.cuda.synchronize()

    pure_inf_time = 0.0
    for _ in range(max_iter):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            model.test_step(batch)
        end.record()
        torch.cuda.synchronize()
        pure_inf_time += start.elapsed_time(end) / 1000.0

    fps = max_iter / pure_inf_time
    ms_per_img = 1000.0 / fps

    del batch
    del dataloader
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'dataset': dataset_name,
        'checkpoint': ckpt_label,
        'fps': fps,
        'ms_per_img': ms_per_img,
    }


def format_table(results):
    lines = []
    lines.append('| Config | RSDD FPS | RSDD ms/img | SSDD FPS | SSDD ms/img |')
    lines.append('|---|---:|---:|---:|---:|')
    for cfg_name in ['A', 'B', 'C', 'D']:
        rsdd = results[(cfg_name, 'rsdd')]
        ssdd = results[(cfg_name, 'ssdd')]
        lines.append(
            f'| {cfg_name} | {rsdd["fps"]:.2f} | {rsdd["ms_per_img"]:.2f} | '
            f'{ssdd["fps"]:.2f} | {ssdd["ms_per_img"]:.2f} |')
    return '\n'.join(lines)


def main():
    register_all_modules()
    args = parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {}
    log_lines = []
    log_lines.append('Ablation FPS Suite')
    log_lines.append(f'Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    log_lines.append(f'Max iter: {args.max_iter}')
    log_lines.append(f'Warmup: {args.num_warmup}')
    log_lines.append(f'Use checkpoint: {args.use_checkpoint}')
    log_lines.append('')

    for cfg_name, cfg_path in CONFIGS.items():
        for dataset_name in ['rsdd', 'ssdd']:
            checkpoint_path = None
            if args.use_checkpoint:
                checkpoint_path = CHECKPOINTS[cfg_name]
            print(f'Running {cfg_name} on {dataset_name} ...', flush=True)
            result = benchmark_one(
                cfg_path, dataset_name, checkpoint_path, args.max_iter,
                args.num_warmup)
            results[(cfg_name, dataset_name)] = result
            log_lines.append(
                f'{cfg_name}-{dataset_name}: fps={result["fps"]:.2f}, '
                f'ms/img={result["ms_per_img"]:.2f}, checkpoint={result["checkpoint"]}')

    log_lines.append('')
    log_lines.append(format_table(results))
    report = '\n'.join(log_lines)

    output_path.write_text(report, encoding='utf-8')
    print(report)
    print(f'\nSaved to: {output_path}')


if __name__ == '__main__':
    main()
