"""Decompose DRIFT's prefix-restricted task accuracy (diagnostic, evaluation only).

For latent task t, DRIFT scores predictions over classes 0 .. max(classes of tasks <= t),
on task t's own evaluation subgraph.  Since classes arrive in index order, this prefix
removes every class that arrived after task t.  Each task's restricted accuracy factors as

    restricted = P(prediction in task t's own pair) * P(correct | prediction in pair)

and this script records both factors at every checkpoint, the task-identity-given accuracy
(argmax over the pair's two columns only), and at the final checkpoint the unrestricted
accuracy over every class.  Training is untouched: the standard evaluator still produces
the reported metrics, and this runs one extra forward per evaluation graph.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from Baselines.hybrid_model import NET
from experiments import scientific_gaussian as benchmark

ARMS = {'mas_relative': (False, True, 'relative'),
        'replay_only': (True, False, 'absolute'),
        'hybrid_absolute': (True, True, 'absolute')}


def decompose(model, graphs, ids, args):
    model.eval()
    seen, rows = set(), []
    with torch.no_grad():
        for t, (graph_cpu, task_ids) in enumerate(zip(graphs, ids)):
            graph = graph_cpu.to('cuda:0' if args.cuda else 'cpu')
            labels = graph.dstdata['label'].squeeze()[task_ids]
            seen.update(int(v) for v in labels.unique().tolist())
            offset = max(seen) + 1
            offset += offset % 2
            output, _ = model(graph, graph.srcdata['feat'])
            logits = output[task_ids]
            pair = torch.tensor(args.task_seq[t], device=logits.device)
            restricted = logits[:, :offset].argmax(dim=1)
            in_pair = (restricted.unsqueeze(1) == pair.unsqueeze(0)).any(dim=1)
            within = pair[logits.index_select(1, pair).argmax(dim=1)]
            full = logits[:, :args.n_cls].argmax(dim=1)
            rows.append({
                'n': int(labels.numel()),
                'restricted': float((restricted == labels).float().mean()),
                'in_pair': float(in_pair.float().mean()),
                'within_pair_given_task': float((within == labels).float().mean()),
                'unrestricted': float((full == labels).float().mean()),
                'prefix_classes': offset,
            })
    return rows


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--dataset', choices=('CoraFull-CL', 'Arxiv-CL'), required=True)
    parser.add_argument('--sigma', type=float, required=True)
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'results_decomposition')
    cli = parser.parse_args()
    benchmark.STREAM_SEED = 1
    args = benchmark.base_args(cli.dataset, ROOT / 'data', ROOT / 'data/raw', 'cuda')
    dataset, task_train, merged, mapping, graphs, ids = benchmark.prepare(args, 'validation')
    original_eval = benchmark.eval_tasks_cis
    for arm in cli.arms:
        replay_on, mas_on, trigger = ARMS[arm]
        records = []

        def recording_eval(model, eval_graphs, eval_ids, current_task, eval_args):
            records.append(decompose(model, eval_graphs, eval_ids, eval_args))
            return original_eval(model, eval_graphs, eval_ids, current_task, eval_args)

        def build(model, learner_args, dataset=None):
            learner_args.hybrid_args = {'replay_enabled': replay_on, 'mas_enabled': mas_on,
                                        'buffer': 'cbrs', 'trigger': trigger}
            learner_args.mas_geometry_args['monitor_task_loss'] = True
            return NET(model, learner_args, dataset=dataset)

        benchmark.eval_tasks_cis = recording_eval
        benchmark.NET = build
        config = {'name': f'decomp_{arm}', 'learner': 'hybrid', 'project_classifier': False,
                  'replay_enabled': replay_on, 'mas_enabled': mas_on, 'buffer': 'cbrs',
                  'trigger': trigger, 'monitor_task_loss': True}
        output = cli.output_root / cli.dataset / f'seed{cli.seed}'
        result = benchmark.run_one((dataset, task_train, merged, mapping), (graphs, ids, 'validation'),
                                   args, config, cli.sigma, cli.seed, output, {'files': {}, 'runtime': {}})
        weights = np.array([r['n'] for r in records[0]], dtype=float)
        weights /= weights.sum()

        def pooled(key, rows):
            return float(np.dot(weights, [r[key] for r in rows]))

        curve = {key: [pooled(key, rows) for rows in records]
                 for key in ('restricted', 'in_pair', 'within_pair_given_task')}
        summary = {
            'arm': arm, 'dataset': cli.dataset, 'sigma': cli.sigma, 'seed': cli.seed,
            'A_AUC_percent': result['metrics']['A_AUC_percent'],
            'AF_s_percent': result['metrics']['AF_s_percent'],
            'check_restricted_matches_reported': float(np.max(np.abs(
                np.array(curve['restricted']) - np.array(result['pooled_accuracy'])))),
            'mean_over_checkpoints': {k: float(np.mean(v)) for k, v in curve.items()},
            'final': {k: pooled(k, records[-1]) for k in
                      ('restricted', 'in_pair', 'within_pair_given_task', 'unrestricted')},
            'final_per_task': records[-1],
            'curve': curve,
        }
        benchmark.atomic_json(output / f'decomposition_{arm}_sigma{cli.sigma:g}.json', summary)
        print(json.dumps({k: summary[k] for k in ('arm', 'A_AUC_percent', 'AF_s_percent',
                                                  'check_restricted_matches_reported',
                                                  'mean_over_checkpoints', 'final')}, indent=1), flush=True)


if __name__ == '__main__':
    main()
