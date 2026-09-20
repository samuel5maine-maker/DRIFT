"""Predeclared H30 paired experiments; reuse the existing DRIFT evaluator."""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dgl
import numpy as np
import torch

from Baselines.context_teacher_model import NET
from experiments import scientific_gaussian as benchmark


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--dataset', choices=('CoraFull-CL', 'Arxiv-CL'), required=True)
    parser.add_argument('--sigmas', type=float, nargs='+', required=True)
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--stream-seed', type=int, default=1)
    parser.add_argument('--partition', choices=('validation', 'test'), default='validation')
    parser.add_argument('--arm', choices=('paired', 'control', 'teacher'), default='paired')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'results_h30')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    cli = parser.parse_args()
    if cli.partition == 'test':
        raise RuntimeError('H30 confirmation is locked until validation decision and source freeze.')
    if cli.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    benchmark.STREAM_SEED = cli.stream_seed
    args = benchmark.base_args(cli.dataset, ROOT / 'data', ROOT / 'data/raw', cli.device)
    dataset, task_train, merged, mapping, graphs, ids = benchmark.prepare(args, cli.partition)
    split = ROOT / 'data' / f'tr0.6_va0.2_te0.2_split_{cli.dataset}.pkl'
    source_paths = [split, Path(__file__), ROOT / 'gaussian_utils.py', ROOT / 'pipeline.py']
    for folder in ('Baselines', 'Backbones', 'training', 'dataset'):
        source_paths.extend(sorted((ROOT / folder).glob('*.py')))
    source_paths.extend([ROOT / 'experiments/scientific_gaussian.py', ROOT / 'experiments/drift_args.py'])
    fingerprint = {
        'files': {str(p.relative_to(ROOT)): benchmark.sha256(p) for p in source_paths},
        'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                    'dgl': dgl.__version__, 'numpy': np.__version__},
    }
    output = cli.output_root / cli.dataset / cli.partition / f'seed{cli.seed}_stream{cli.stream_seed}'
    arms = ('control', 'teacher') if cli.arm == 'paired' else (cli.arm,)
    results = []
    for arm in arms:
        enabled = arm == 'teacher'

        def build(model, learner_args, dataset=None):
            learner_args.context_ema_args['feedback_enabled'] = enabled
            return NET(model, learner_args, dataset=dataset)

        benchmark.NET = build
        config = {'name': f'h30_{arm}', 'buffer': 'cbrs', 'replay': 'context',
                  'ema_alpha': .995, 'classifier_norm': 'fixed',
                  'feedback_enabled': enabled, 'inference': 'working',
                  'distillation_temperature': 2., 'distillation_coefficient': 1.}
        for sigma in cli.sigmas:
            print(f'[start] {cli.dataset} {arm} seed={cli.seed} stream={cli.stream_seed} sigma={sigma:g}', flush=True)
            result = benchmark.run_one((dataset, task_train, merged, mapping),
                                       (graphs, ids, cli.partition), args, config,
                                       sigma, cli.seed, output, fingerprint)
            results.append(result)
    summary = {'dataset': cli.dataset, 'partition': cli.partition, 'seed': cli.seed,
               'stream_seed': cli.stream_seed, 'sigmas': cli.sigmas, 'arms': {}}
    for arm in arms:
        subset = [r for r in results if r['spec']['config']['name'] == f'h30_{arm}']
        summary['arms'][arm] = {key: float(np.mean([r['metrics'][key] for r in subset]))
                                for key in subset[0]['metrics']}
    benchmark.atomic_json(output / ('summary_' + cli.arm + '.json'), summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
