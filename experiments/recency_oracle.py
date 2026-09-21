"""Score a feature-free, training-free predictor under DRIFT's evaluator (diagnostic).

At each checkpoint, for latent task t the evaluator restricts predictions to classes
0 .. max(classes of tasks <= t).  This predictor ignores the node entirely: it predicts
the most recently *first-delivered* class inside that prefix, using only the arrival order
of labels that any online learner observes (class 0 if nothing in the prefix has arrived).
Its A_AUC measures how much of DRIFT's metric is available from arrival order alone.

A second variant additionally knows each evaluated node's true within-pair class whenever
it predicts inside the task's own pair, i.e. arrival order plus perfect task-IL skill; it is
the ceiling for "recency ordering + within-pair discrimination".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from experiments import scientific_gaussian as benchmark
from gaussian_utils import build_gaussian_stream


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--dataset', choices=('CoraFull-CL', 'Arxiv-CL'), required=True)
    parser.add_argument('--sigmas', type=float, nargs='+', required=True)
    parser.add_argument('--stream-seed', type=int, default=1)
    cli = parser.parse_args()
    args = benchmark.base_args(cli.dataset, ROOT / 'data', ROOT / 'data/raw', 'cpu')
    dataset, task_train, merged, mapping, graphs, ids = benchmark.prepare(args, 'validation')
    node_labels = merged.ndata['label'].squeeze().cpu().numpy()
    eval_labels = [g.dstdata['label'].squeeze().cpu().numpy()[np.asarray(i)] for g, i in zip(graphs, ids)]
    sizes = np.array([len(l) for l in eval_labels], dtype=float)
    weights = sizes / sizes.sum()
    out = {}
    for sigma in cli.sigmas:
        stream, _, _, total, _ = build_gaussian_stream(task_train, 10, sigma, seed=cli.stream_seed, replace=False)
        first = {}
        for step, (original_ids, _, _) in enumerate(stream):
            for nid in original_ids:
                first.setdefault(int(node_labels[mapping[int(nid)]]), step)
        checkpoints = list(range(0, total, 100)) + [total]
        curves = {'recency_only': [], 'recency_plus_within_pair': []}
        for c in checkpoints:
            arrived = {k: s for k, s in first.items() if s < c}
            acc_r, acc_rw = [], []
            seen = set()
            for t, labels in enumerate(eval_labels):
                seen.update(int(v) for v in np.unique(labels))
                offset = max(seen) + 1
                offset += offset % 2
                candidates = [k for k in arrived if k < offset]
                prediction = max(candidates, key=lambda k: arrived[k]) if candidates else 0
                acc_r.append(float(np.mean(labels == prediction)))
                pair = set(args.task_seq[t])
                acc_rw.append(1.0 if prediction in pair else float(np.mean(labels == prediction)))
            curves['recency_only'].append(float(np.dot(weights, acc_r)))
            curves['recency_plus_within_pair'].append(float(np.dot(weights, acc_rw)))
        out[str(sigma)] = {k: {'A_AUC_percent': 100 * float(np.mean(v)), 'final_percent': 100 * v[-1]}
                           for k, v in curves.items()}
        print(cli.dataset, 'sigma', sigma, json.dumps(out[str(sigma)]), flush=True)
    target = ROOT / 'results_decomposition' / cli.dataset / 'recency_oracle.json'
    benchmark.atomic_json(target, out)


if __name__ == '__main__':
    main()
