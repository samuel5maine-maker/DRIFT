"""Predeclared H32 paired experiments; reuse the existing DRIFT evaluator."""
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

from Baselines.scalefree_mas_model import NET
from experiments import scientific_gaussian as benchmark

SIGMAS = {'CoraFull-CL': (3., 10., 20.), 'Arxiv-CL': (60.,)}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--dataset', choices=('CoraFull-CL', 'Arxiv-CL'), required=True)
    parser.add_argument('--sigmas', type=float, nargs='+')
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--stream-seed', type=int, default=1)
    parser.add_argument('--partition', choices=('validation', 'test'), default='validation')
    parser.add_argument('--arm', choices=('paired', 'control', 'scalefree'), default='paired')
    parser.add_argument('--replicate', type=str, default='r1')
    parser.add_argument('--traces', action='store_true', help='store per-step loss traces')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'results_h32')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    cli = parser.parse_args()
    if cli.partition == 'test':
        raise RuntimeError('H32 confirmation is locked until the validation decision and source freeze.')
    if cli.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    sigmas = cli.sigmas or SIGMAS[cli.dataset]
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
    output = (cli.output_root / cli.dataset / cli.partition /
              f'seed{cli.seed}_stream{cli.stream_seed}' / cli.replicate)
    arms = ('control', 'scalefree') if cli.arm == 'paired' else (cli.arm,)
    traces = {}
    results = []
    for arm in arms:
        enabled = arm == 'scalefree'

        def build(model, learner_args, dataset=None):
            learner_args.mas_geometry_args['monitor_task_loss'] = enabled
            learner = NET(model, learner_args, dataset=dataset)
            traces['learner'] = learner
            return learner

        benchmark.MASGeometryNET = build
        config = {'name': f'h32_{arm}', 'learner': 'mas_geometry',
                  'project_classifier': False, 'monitor_task_loss': enabled}
        for sigma in sigmas:
            print(f'[start] {cli.dataset} {arm} seed={cli.seed} sigma={sigma:g} rep={cli.replicate}', flush=True)
            result = benchmark.run_one((dataset, task_train, merged, mapping),
                                       (graphs, ids, cli.partition), args, config,
                                       sigma, cli.seed, output, fingerprint)
            results.append(result)
            if cli.traces:
                learner = traces['learner']
                benchmark.atomic_json(
                    output / 'traces' / f'{arm}_sigma{sigma:g}.json',
                    {'task_losses': learner.task_losses, 'total_losses': learner.total_losses,
                     'window_means': learner.monitored_losses,
                     'consolidation_steps': learner.consolidation_steps,
                     'first_seen_steps': learner.first_seen_steps},
                )
    summary = {'dataset': cli.dataset, 'seed': cli.seed, 'stream_seed': cli.stream_seed,
               'replicate': cli.replicate, 'sigmas': sigmas, 'arms': {}}
    for arm in arms:
        subset = [r for r in results if r['spec']['config']['name'] == f'h32_{arm}']
        summary['arms'][arm] = {key: float(np.mean([r['metrics'][key] for r in subset]))
                                for key in subset[0]['metrics']}
        summary['arms'][arm]['consolidations'] = float(np.mean(
            [r['budget']['compute']['importance_updates'] for r in subset]))
        summary['arms'][arm]['consolidations_per_new_class'] = float(np.mean(
            [r['budget']['compute']['consolidations_per_new_class'] for r in subset]))
    if len(arms) == 2:
        summary['delta'] = {key: summary['arms']['scalefree'][key] - summary['arms']['control'][key]
                            for key in summary['arms']['control']}
    benchmark.atomic_json(output / f'summary_{cli.arm}.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
