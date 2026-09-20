"""Offline no-forgetting, no-recency-bias ceiling reference for the DRIFT Gaussian benchmark.

Answers: "what A_AUC could a model reach if it never forgot and had no sequential-order
bias, given the same backbone and the same class-arrival schedule?"

Two independent pieces, both derived from the *same* Gaussian stream construction used by
``experiments/scientific_gaussian.py``:

  1. A fresh model trained offline and i.i.d. over ALL training nodes (union of all tasks),
     with a full ``n_cls``-way softmax head (no class-incremental masking). This removes
     forgetting and recency bias from the *model*.
  2. A per-checkpoint "ceiling" curve that credits a task's joint accuracy only once every
     class in that task has actually been delivered by the stream at least once by that
     checkpoint (mirroring the CIS eval's growing label space), so the reference respects
     the same class-arrival schedule as the online runs without forcing any particular
     training order.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dgl
import numpy as np
import torch
import torch.nn.functional as F

from Backbones.model_factory import get_model  # noqa: E402
from experiments import scientific_gaussian as benchmark  # noqa: E402
from gaussian_utils import build_gaussian_stream  # noqa: E402
from pipeline import eval_tasks_cis  # noqa: E402
from training.utils import set_seed  # noqa: E402


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--dataset', choices=('CoraFull-CL', 'Arxiv-CL'), required=True)
    parser.add_argument('--sigma', type=float, required=True)
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--stream-seed', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'results_ceiling')
    cli = parser.parse_args()

    if cli.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')

    benchmark.STREAM_SEED = cli.stream_seed
    args = benchmark.base_args(cli.dataset, ROOT / 'data', ROOT / 'data/raw', cli.device)
    dataset, task_train, merged, mapping, eval_graphs, eval_ids = benchmark.prepare(args, 'validation')

    device = torch.device('cuda:0' if cli.device == 'cuda' else 'cpu')

    # ---- Step 2: walk the stream once, untrained, to record each class's first delivery ----
    stream, centers, batch_counts, total, epochs_per_task = build_gaussian_stream(
        task_train, 10, cli.sigma, seed=cli.stream_seed, replace=False,
    )
    assert total == len(stream)
    labels_cpu = merged.ndata['label'].squeeze()
    first_delivery = {}
    for batch_index, (original_ids, _, weights) in enumerate(stream):
        local = [mapping[int(nid)] for nid in original_ids]
        for cls in labels_cpu[local].unique().tolist():
            cls = int(cls)
            if cls not in first_delivery:
                first_delivery[cls] = batch_index

    # ---- Step 3: train a FRESH model offline and i.i.d. over the union of all training nodes,
    # full n_cls-way softmax head (no class-incremental masking) ----
    args.seed = cli.seed
    set_seed(args)
    model = get_model(dataset, args)
    model.to(device)
    graph = merged.to(device)
    labels = graph.ndata['label'].squeeze()

    all_train_local = np.array(
        sorted({mapping[int(nid)] for task in task_train for nid in task}), dtype=np.int64,
    )
    rng = np.random.default_rng(cli.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)
    sampler = dgl.dataloading.NeighborSampler([10, 25])

    epoch_losses = []
    started = time.perf_counter()
    model.train()
    for epoch in range(cli.epochs):
        perm = rng.permutation(len(all_train_local))
        shuffled = all_train_local[perm]
        losses = []
        for start in range(0, len(shuffled), 10):
            batch_local = torch.tensor(shuffled[start:start + 10], dtype=torch.long, device=device)
            _, _, blocks = sampler.sample_blocks(graph, batch_local)
            logits, _ = model.forward_batch(blocks, blocks[0].srcdata['feat'])
            batch_labels = labels[batch_local]
            loss = F.cross_entropy(logits, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        mean_loss = float(np.mean(losses))
        epoch_losses.append(mean_loss)
        print(f'[offline_ceiling] {cli.dataset} sigma={cli.sigma:g} epoch {epoch + 1}/{cli.epochs} '
              f'mean_loss={mean_loss:.4f}', flush=True)
    train_seconds = time.perf_counter() - started

    # ---- Step 4: evaluate the trained model with the harness's own CIS evaluator ----
    joint_per_task, joint_pooled, _, _ = eval_tasks_cis(model, eval_graphs, eval_ids, args.n_tasks - 1, args)
    joint_per_task = [float(v) for v in joint_per_task]
    joint_pooled = float(joint_pooled)

    # ---- Step 5: per-task evaluation weights ----
    sizes = [len(e) for e in eval_ids]
    total_eval = sum(sizes)
    weights_k = [size / total_eval for size in sizes]

    # ---- Step 6: ceiling A_AUC curve ----
    checkpoints = list(range(0, total, 100)) + [total]
    ceiling_pooled_curve = []
    for c in checkpoints:
        pooled_value = 0.0
        for k, classes in enumerate(args.task_seq):
            ready = all(first_delivery.get(cls, total + 1) <= c for cls in classes)
            pooled_value += weights_k[k] * (joint_per_task[k] if ready else 0.0)
        ceiling_pooled_curve.append(pooled_value)
    a_auc_ceiling_percent = 100 * float(np.mean(ceiling_pooled_curve))

    result = {
        'dataset': cli.dataset,
        'sigma': cli.sigma,
        'seed': cli.seed,
        'stream_seed': cli.stream_seed,
        'epochs': cli.epochs,
        'total_batches': total,
        'checkpoints': checkpoints,
        'first_delivery': {str(k): v for k, v in sorted(first_delivery.items())},
        'task_weights': weights_k,
        'joint_per_task': joint_per_task,
        'joint_pooled_accuracy_percent': 100 * joint_pooled,
        'ceiling_pooled_curve': ceiling_pooled_curve,
        'A_AUC_ceiling_percent': a_auc_ceiling_percent,
        'epoch_losses': epoch_losses,
        'train_seconds': train_seconds,
        'runtime': {
            'python': platform.python_version(), 'torch': torch.__version__,
            'dgl': dgl.__version__, 'numpy': np.__version__,
        },
    }
    out_path = cli.output_root / cli.dataset / f'ceiling_sigma{cli.sigma:g}.json'
    atomic_json(out_path, result)
    print(f'[done] {cli.dataset} sigma={cli.sigma:g} seed={cli.seed} stream={cli.stream_seed} '
          f'joint_pooled={100 * joint_pooled:.2f}% A_AUC_ceiling={a_auc_ceiling_percent:.2f}% -> {out_path}',
          flush=True)


if __name__ == '__main__':
    main()
