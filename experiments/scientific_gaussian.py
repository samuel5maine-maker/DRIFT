"""Falsifiable CoraFull Gaussian experiments for context-aware replay."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import dgl
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Backbones.model_factory import get_model  # noqa: E402
from Baselines.context_ema_model import NET  # noqa: E402
from Baselines.mas_geometry_model import NET as MASGeometryNET  # noqa: E402
from dataset.utils import NodeLevelDataset  # noqa: E402
from experiments.drift_args import main_args  # noqa: E402
from gaussian_utils import build_gaussian_stream  # noqa: E402
from pipeline import eval_tasks_cis  # noqa: E402
from training.utils import set_seed  # noqa: E402

STREAM_SEED = 1
DATASET_SIGMAS = {
    'CoraFull-CL': (3.0, 10.0, 20.0),
    'Arxiv-CL': (60.0,),
}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def base_args(dataset_name, data_root, original_data_root, device):
    args = main_args([
        '--dataset', dataset_name, '--method', 'bare',
        '--setting', 'tfo_gaussian', '--seed', '0', '--epochs', '1',
        '--batch_size', '10', '--log_every', '100',
        '--cuda', 'yes' if device == 'cuda' else 'no',
        '--data_path', str(data_root),
        '--ori_data_path', str(original_data_root),
    ])
    args.n_cls_per_task = 2
    args.sample_nbs = True
    args.n_nbs_sample = [10, 25]
    args.eval_batch = False
    return args


def prepare(args, partition):
    dataset = NodeLevelDataset(
        args.dataset, ratio_valid_test=[0.2, 0.2], args=args,
    )
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    args.task_seq = [list(range(i, i + 2)) for i in range(0, args.n_cls - 1, 2)]
    args.n_tasks = len(args.task_seq)
    split_index = {'validation': 1, 'test': 2}[partition]
    task_train, eval_graphs, eval_ids = [], [], []
    task_test = []
    for classes in args.task_seq:
        train = [nid for cls in classes for nid in dataset.tr_va_te_split[cls][0]]
        test = [nid for cls in classes for nid in dataset.tr_va_te_split[cls][2]]
        evaluate = [nid for cls in classes for nid in dataset.tr_va_te_split[cls][split_index]]
        task_train.append(train)
        task_test.append(test)
        graph, (_, local_eval) = dataset.get_graph(node_ids=[train, evaluate])
        eval_graphs.append(graph)
        eval_ids.append(local_eval)
    all_train = [nid for task in task_train for nid in task]
    all_test = [nid for task in task_test for nid in task]
    merged, _ = dataset.get_graph(node_ids=[all_train, all_test])
    original = merged.ndata['_ID'].cpu().tolist()
    mapping = {int(nid): i for i, nid in enumerate(original)}
    return dataset, task_train, merged, mapping, eval_graphs, eval_ids


def configurations(experiment):
    if experiment == 'h1':
        return [
            {'name': 'isolated_reservoir', 'replay': 'isolated', 'buffer': 'reservoir', 'ema_alpha': 0.995},
            {'name': 'context_reservoir', 'replay': 'context', 'buffer': 'reservoir', 'ema_alpha': 0.995},
        ]
    if experiment == 'h2':
        return [
            {'name': 'context_cbrs', 'replay': 'context', 'buffer': 'cbrs', 'ema_alpha': 0.995},
        ]
    if experiment == 'h3':
        return [
            {'name': 'context_cbrs_adaptive_ema', 'replay': 'context', 'buffer': 'cbrs',
             'ema_alpha': 0.995, 'ema_fast': 0.99, 'ema_mode': 'adaptive'},
        ]
    if experiment == 'h4':
        return [
            {'name': 'context_cbrs_classifier_norm', 'replay': 'context', 'buffer': 'cbrs',
             'ema_alpha': 0.995, 'classifier_norm': True},
        ]
    if experiment == 'h5':
        return [
            {'name': 'context_cbrs_classifier_norm_replay_half', 'replay': 'context',
             'buffer': 'cbrs', 'ema_alpha': 0.995, 'classifier_norm': True,
             'replay_weight': 0.5},
        ]
    if experiment == 'h6':
        return [
            {'name': 'context_cbrs_classifier_mean_norm', 'replay': 'context',
             'buffer': 'cbrs', 'ema_alpha': 0.995, 'classifier_norm': 'mean'},
        ]
    if experiment == 'h7':
        return [
            {'name': 'context_cbrs_classifier_norm_prototype_eval', 'replay': 'context',
             'buffer': 'cbrs', 'ema_alpha': 0.995, 'classifier_norm': 'fixed',
             'prototype_eval': True},
        ]
    if experiment == 'h8':
        return [
            {'name': 'context_cbrs_classifier_norm_label_propagation', 'replay': 'context',
             'buffer': 'cbrs', 'ema_alpha': 0.995, 'classifier_norm': 'fixed',
             'label_propagation_eval': True, 'label_propagation_steps': 10},
        ]
    if experiment == 'h9':
        return [
            {'name': 'context_cbrs_classifier_norm_half_scale', 'replay': 'context',
             'buffer': 'cbrs', 'ema_alpha': 0.995, 'classifier_norm': 'fixed',
             'classifier_norm_scale': 0.5},
        ]
    if experiment == 'h10':
        return [
            {'name': 'context_full_cbrs_classifier_norm', 'replay': 'context',
             'buffer': 'cbrs', 'ema_alpha': 0.995, 'classifier_norm': 'fixed',
             'replay_neighbors': 'full'},
        ]
    if experiment == 'h11':
        return [
            {'name': 'context_cbrs_cosine_classifier', 'replay': 'context',
             'buffer': 'cbrs', 'ema_alpha': 0.995, 'classifier': 'cosine',
             'cosine_scale': 10.0},
        ]
    if experiment == 'h12':
        return [
            {'name': 'context_cbrs_classifier_norm_hard_ace', 'replay': 'context',
             'buffer': 'cbrs', 'ema_alpha': 0.995, 'classifier_norm': 'fixed',
             'ace': True},
        ]
    if experiment == 'h13':
        return [
            {'name': 'context_cbrs_classifier_norm_balanced_softmax', 'replay': 'context',
             'buffer': 'cbrs', 'ema_alpha': 0.995, 'classifier_norm': 'fixed',
             'balanced_softmax': True},
        ]
    if experiment == 'h14':
        return [
            {'name': 'context_cbrs_classifier_norm_smooth', 'replay': 'context',
             'buffer': 'cbrs', 'ema_alpha': 0.995, 'classifier_norm': 'fixed',
             'smoothness_weight': 0.1},
        ]
    if experiment == 'h15':
        return [
            {'name': 'context_degree_cbrs_classifier_norm', 'replay': 'context',
             'buffer': 'degree_cbrs', 'ema_alpha': 0.995, 'classifier_norm': 'fixed'},
        ]
    if experiment == 'h16':
        return [
            {'name': 'context_cbrs_classifier_norm_head_lr2', 'replay': 'context',
             'buffer': 'cbrs', 'ema_alpha': 0.995, 'classifier_norm': 'fixed',
             'head_lr_multiplier': 2.0},
        ]
    if experiment == 'h17':
        return [
            {'name': 'context_cbrs_classifier_norm_adaptive_ema', 'replay': 'context',
             'buffer': 'cbrs', 'ema_alpha': 0.995, 'ema_fast': 0.99,
             'ema_mode': 'adaptive', 'classifier_norm': 'fixed'},
        ]
    if experiment == 'h18':
        return [
            {'name': 'context_cbrs_classifier_norm_blend_eval', 'replay': 'context',
             'buffer': 'cbrs', 'ema_alpha': 0.995, 'classifier_norm': 'fixed',
             'blend_eval': True, 'blend_working_weight': 0.25},
        ]
    if experiment == 'h19':
        return [
            {'name': 'context_cbrs_classifier_norm_sync_new_rows', 'replay': 'context',
             'buffer': 'cbrs', 'ema_alpha': 0.995, 'classifier_norm': 'fixed',
             'sync_new_class_rows': True},
        ]
    if experiment == 'h20':
        return [
            {'name': 'context_cbrs_classifier_norm_partial_new_rows', 'replay': 'context',
             'buffer': 'cbrs', 'ema_alpha': 0.995, 'classifier_norm': 'fixed',
             'sync_new_class_rows': 0.05},
        ]
    if experiment == 'h21':
        return [
            {'name': 'context_cbrs_dual_classifier_norm', 'replay': 'context',
             'buffer': 'cbrs', 'ema_alpha': 0.995, 'classifier_norm': 'fixed',
             'project_ema_classifier': True},
        ]
    if experiment == 'h22':
        return [
            {'name': 'context_cbrs_classifier_norm_recency_blend', 'replay': 'context',
             'buffer': 'cbrs', 'ema_alpha': 0.995, 'classifier_norm': 'fixed',
             'blend_eval': True, 'blend_working_weight': 0.25,
             'blend_recency_horizon': 100},
        ]
    if experiment == 'h23':
        return [
            {'name': 'context_cbrs_classifier_norm_recency_blend_low', 'replay': 'context',
             'buffer': 'cbrs', 'ema_alpha': 0.995, 'classifier_norm': 'fixed',
             'blend_eval': True, 'blend_working_weight': 0.125,
             'blend_recency_horizon': 100},
        ]
    if experiment == 'h24':
        return [
            {'name': 'context_cbrs_norm_scaled_ema_recency_blend', 'replay': 'context',
             'buffer': 'cbrs', 'ema_alpha': 0.99966, 'classifier_norm': 'fixed',
             'blend_eval': True, 'blend_working_weight': 0.125,
             'blend_recency_horizon': 100},
        ]
    if experiment == 'h25':
        return [
            {'name': 'mas_fixed_classifier_norm', 'learner': 'mas_geometry',
             'project_classifier': True},
        ]
    if experiment == 'h26':
        return [
            {'name': 'mas_regenerated_split_control', 'learner': 'mas_geometry',
             'project_classifier': False},
        ]
    if experiment == 'h27':
        return [
            {'name': 'context_cbrs_partial_rows_recency_blend', 'replay': 'context',
             'buffer': 'cbrs', 'ema_alpha': 0.995, 'classifier_norm': 'fixed',
             'sync_new_class_rows': 0.05, 'blend_eval': True,
             'blend_working_weight': 0.125, 'blend_recency_horizon': 100},
        ]
    if experiment == 'h28':
        return [
            {'name': 'mas_stream_scaled_ema', 'learner': 'mas_geometry',
             'project_classifier': False, 'mas_ema_alpha': 0.99966},
        ]
    if experiment == 'h29':
        return [
            {'name': 'mas_working_slow_ema_blend', 'learner': 'mas_geometry',
             'project_classifier': False, 'mas_ema_alpha': 0.99966,
             'mas_eval_working_weight': 0.75},
        ]
    raise ValueError(experiment)


def run_one(prepared, eval_data, args, config, sigma, seed, output, fingerprint):
    dataset, task_train, merged_cpu, mapping = prepared
    eval_graphs, eval_ids, partition = eval_data
    spec = {
        'experiment': output.name, 'dataset': args.dataset, 'partition': partition, 'seed': seed,
        'stream_seed': STREAM_SEED, 'sigma': sigma, 'config': config,
        'buffer_budget': 100, 'replay_seed_rows': 10, 'optimizer_steps_per_batch': 1,
        'source': fingerprint,
    }
    path = output / 'runs' / f"{config['name']}_seed{seed}_sigma{sigma:g}.json"
    if path.exists():
        value = json.loads(path.read_text(encoding='utf-8'))
        if value.get('status') == 'complete' and value.get('spec') == spec:
            print('[resume]', path.name, flush=True)
            return value
        raise RuntimeError(f'incompatible result exists: {path}')

    args.seed = seed
    set_seed(args)
    stream, centers, counts, total, epochs = build_gaussian_stream(
        task_train, 10, sigma, seed=STREAM_SEED, replace=False,
    )
    assert total == len(stream)
    model = get_model(dataset, args)
    device = torch.device('cuda:0' if args.cuda else 'cpu')
    model.to(device)
    graph = merged_cpu.to(device)
    labels = graph.ndata['label'].squeeze()
    masked_labels = torch.full_like(labels, -1)
    learner_args = SimpleNamespace(
        epochs=1, lr=0.005, weight_decay=0.0005, batch_size=10,
        sample_nbs=True, n_nbs_sample=[10, 25], cuda=args.cuda, gpu=0,
        context_ema_args={
            'budget': 100, 'memory_proportion': 1,
            'buffer': config.get('buffer', 'cbrs'), 'replay': config.get('replay', 'context'),
            'ema_alpha': config.get('ema_alpha', 0.995),
            'ema_fast': config.get('ema_fast', config.get('ema_alpha', 0.995)),
            'ema_mode': config.get('ema_mode', 'constant'),
            'classifier_norm': config.get('classifier_norm', 'none'),
            'classifier_norm_scale': config.get('classifier_norm_scale', 1.0),
            'replay_weight': config.get('replay_weight', 1.0),
            'replay_neighbors': config.get('replay_neighbors', 'sampled'),
            'classifier': config.get('classifier', 'linear'),
            'cosine_scale': config.get('cosine_scale', 10.0),
            'ace': config.get('ace', False),
            'balanced_softmax': config.get('balanced_softmax', False),
            'smoothness_weight': config.get('smoothness_weight', 0.0),
            'head_lr_multiplier': config.get('head_lr_multiplier', 1.0),
            'sync_new_class_rows': config.get('sync_new_class_rows', False),
            'project_ema_classifier': config.get('project_ema_classifier', False),
        },
        mas_args={'memory_strength': 0.5},
        mas_geometry_args={
            'project_classifier': config.get('project_classifier', True),
            'ema_alpha': config.get('mas_ema_alpha'),
            'eval_working_weight': config.get('mas_eval_working_weight'),
        },
    )
    if config.get('learner', 'context_ema') == 'mas_geometry':
        learner = MASGeometryNET(model, learner_args, dataset=dataset)
    else:
        learner = NET(model, learner_args, dataset=dataset)
    checkpoints, pooled_curve, task_curve = [], [], []
    prototype_pooled_curve, prototype_task_curve = [], []
    propagation_pooled_curve, propagation_task_curve = [], []
    blend_pooled_curve, blend_task_curve = [], []
    current_task = 0
    started = time.perf_counter()

    def evaluate_extended(checkpoint):
        inference_model = learner.eval_model()
        inference_model.eval()
        prototype_matrix = None
        prototype_present = torch.zeros(args.n_cls, dtype=torch.bool, device=device)
        if config.get('prototype_eval', False):
            with torch.no_grad():
                inference_model(graph, graph.ndata['feat'])
                hidden = inference_model.second_last_h
                prototype_matrix = torch.zeros(args.n_cls, hidden.shape[1], device=device)
                if learner.buffer.ids:
                    buffer_local = torch.tensor(
                        [mapping[int(nid)] for nid in learner.buffer.ids], dtype=torch.long, device=device,
                    )
                    buffer_labels = torch.tensor(learner.buffer.labels, dtype=torch.long, device=device)
                    for class_id in buffer_labels.unique():
                        prototype_matrix[class_id] = hidden[buffer_local[buffer_labels == class_id]].mean(dim=0)
                        prototype_present[class_id] = True
                    prototype_matrix = torch.nn.functional.normalize(prototype_matrix, dim=1)

        standard_tasks, prototype_tasks, propagation_tasks, blend_tasks = [], [], [], []
        standard_correct = prototype_correct = propagation_correct = blend_correct = total = 0
        classes_seen = set()
        buffer_labels_by_id = {int(node): int(label) for node, label in zip(learner.buffer.ids, learner.buffer.labels)}
        for eval_graph_cpu, ids in zip(eval_graphs, eval_ids):
            eval_graph = eval_graph_cpu.to(device)
            eval_labels_all = eval_graph.dstdata['label'].squeeze()
            eval_index = torch.tensor(ids, dtype=torch.long, device=device)
            labels_here = eval_labels_all[eval_index]
            classes_seen.update(int(value) for value in labels_here.unique().tolist())
            offset = max(classes_seen) + 1
            offset += offset % 2
            with torch.no_grad():
                logits, _ = inference_model(eval_graph, eval_graph.srcdata['feat'])
                eval_hidden = inference_model.second_last_h[eval_index]
                standard_prediction = logits[eval_index, :offset].argmax(dim=1)
                blend_prediction = standard_prediction
                if config.get('blend_eval', False):
                    working_model = learner.alt_eval_models()['working']
                    working_logits, _ = working_model(eval_graph, eval_graph.srcdata['feat'])
                    working_weight = float(config['blend_working_weight'])
                    blended = logits.clone()
                    if 'blend_recency_horizon' in config:
                        horizon = int(config['blend_recency_horizon'])
                        recent = [
                            class_id for class_id, last_seen in learner.class_last_seen.items()
                            if class_id < offset and learner.optimizer_steps - last_seen <= horizon
                        ]
                        if recent:
                            columns = torch.tensor(recent, dtype=torch.long, device=device)
                            ema_columns = logits.index_select(1, columns)
                            working_columns = working_logits.index_select(1, columns)
                            blended.index_copy_(
                                1, columns, (1 - working_weight) * ema_columns + working_weight * working_columns,
                            )
                    else:
                        blended = (1 - working_weight) * logits + working_weight * working_logits
                    blend_prediction = blended[eval_index, :offset].argmax(dim=1)
                present = prototype_present[:offset]
                if config.get('prototype_eval', False) and bool(present.any()):
                    scores = torch.nn.functional.normalize(eval_hidden, dim=1) @ prototype_matrix[:offset].T
                    scores[:, ~present] = -torch.inf
                    prototype_prediction = scores.argmax(dim=1)
                else:
                    prototype_prediction = standard_prediction
                propagation_prediction = standard_prediction
                if config.get('label_propagation_eval', False) and buffer_labels_by_id:
                    original = eval_graph.ndata['_ID'].detach().cpu().tolist()
                    seed_rows, seed_labels = [], []
                    for row, node_id in enumerate(original):
                        if int(node_id) in buffer_labels_by_id:
                            label = buffer_labels_by_id[int(node_id)]
                            if label < offset:
                                seed_rows.append(row)
                                seed_labels.append(label)
                    if seed_rows:
                        seed_rows_t = torch.tensor(seed_rows, dtype=torch.long, device=device)
                        seed_labels_t = torch.tensor(seed_labels, dtype=torch.long, device=device)
                        seed_scores = torch.zeros(eval_graph.num_nodes(), offset, device=device)
                        seed_scores[seed_rows_t, seed_labels_t] = 1
                        scores = seed_scores
                        degrees = eval_graph.in_degrees().to(device).clamp_min(1).unsqueeze(1)
                        with eval_graph.local_scope():
                            for _ in range(int(config['label_propagation_steps'])):
                                eval_graph.ndata['_lp'] = scores
                                eval_graph.update_all(dgl.function.copy_u('_lp', '_m'), dgl.function.sum('_m', '_sum'))
                                scores = eval_graph.ndata['_sum'] / degrees
                                scores[seed_rows_t] = seed_scores[seed_rows_t]
                        propagated = scores[eval_index]
                        has_mass = propagated.sum(dim=1) > 0
                        propagation_prediction = standard_prediction.clone()
                        propagation_prediction[has_mass] = propagated[has_mass].argmax(dim=1)
            standard_hits = int((standard_prediction == labels_here).sum())
            prototype_hits = int((prototype_prediction == labels_here).sum())
            propagation_hits = int((propagation_prediction == labels_here).sum())
            blend_hits = int((blend_prediction == labels_here).sum())
            count = len(ids)
            standard_tasks.append(standard_hits / count)
            prototype_tasks.append(prototype_hits / count)
            propagation_tasks.append(propagation_hits / count)
            blend_tasks.append(blend_hits / count)
            standard_correct += standard_hits
            prototype_correct += prototype_hits
            propagation_correct += propagation_hits
            blend_correct += blend_hits
            total += count
        checkpoints.append(checkpoint)
        task_curve.append(standard_tasks)
        pooled_curve.append(standard_correct / total)
        if config.get('prototype_eval', False):
            prototype_task_curve.append(prototype_tasks)
            prototype_pooled_curve.append(prototype_correct / total)
        if config.get('label_propagation_eval', False):
            propagation_task_curve.append(propagation_tasks)
            propagation_pooled_curve.append(propagation_correct / total)
        if config.get('blend_eval', False):
            blend_task_curve.append(blend_tasks)
            blend_pooled_curve.append(blend_correct / total)

    def evaluate(checkpoint):
        if config.get('prototype_eval', False) or config.get('label_propagation_eval', False) or config.get('blend_eval', False):
            evaluate_extended(checkpoint)
            return
        per_task, pooled, _, _ = eval_tasks_cis(
            learner.eval_model(), eval_graphs, eval_ids, current_task, args,
        )
        checkpoints.append(checkpoint)
        pooled_curve.append(float(pooled))
        task_curve.append([float(v) for v in per_task])

    for batch, (original_ids, _, weights) in enumerate(stream):
        if batch % 100 == 0:
            evaluate(batch)
        current_task = int(np.argmax(weights))
        local = torch.tensor([mapping[int(nid)] for nid in original_ids], dtype=torch.long, device=device)
        masked_labels[local] = labels[local]
        learner.observe_cis(learner_args, graph, graph.ndata['feat'], masked_labels, local)
        masked_labels[local] = -1
    current_task = args.n_tasks - 1
    evaluate(len(stream))

    replay_rows = getattr(learner, 'replay_rows_per_update', [])
    memory = learner.memory_accounting()
    assertions = {
        'one_step_per_batch': learner.optimizer_steps == total,
        'expected_checkpoints': checkpoints == list(range(0, total, 100)) + [total],
    }
    if config.get('learner', 'context_ema') == 'mas_geometry':
        assertions.update({
            'no_replay_buffer': memory['buffer_bytes'] == 0,
            'no_replay_rows': replay_rows == [],
        })
    else:
        assertions.update({
            'replay_schedule': replay_rows == [0] + [10] * (total - 1),
            'buffer_exactly_100': len(learner.buffer) == 100,
            'buffer_storage_bounded': memory['buffer_bytes'] <= 2400,
            'one_ema_copy': memory['extra_param_count'] == sum(p.numel() for p in model.parameters()),
        })
    assert all(assertions.values()), assertions
    matrix = np.asarray(task_curve)
    prototype_matrix = np.asarray(prototype_task_curve) if prototype_task_curve else None
    propagation_matrix = np.asarray(propagation_task_curve) if propagation_task_curve else None
    blend_matrix = np.asarray(blend_task_curve) if blend_task_curve else None
    result = {
        'status': 'complete', 'spec': spec, 'checkpoints': checkpoints,
        'pooled_accuracy': pooled_curve, 'per_task_accuracy': task_curve,
        'metrics': {
            'A_AUC_percent': 100 * float(np.mean(pooled_curve)),
            'AF_s_percent': 100 * float(np.mean(matrix[-1] - matrix.max(axis=0))),
            'final_accuracy_percent': 100 * pooled_curve[-1],
        },
        'budget': {'memory': memory, 'compute': learner.compute_accounting(), 'assertions': assertions},
        'stream': {'centers': list(map(float, centers)), 'batch_counts': list(map(int, counts)), 'epochs': list(map(int, epochs))},
        'runtime_seconds': time.perf_counter() - started,
    }
    if hasattr(learner, 'ema_alphas'):
        result['ema'] = {
            'mean_alpha_after_initialization': float(np.mean(learner.ema_alphas[1:])),
            'min_alpha_after_initialization': float(np.min(learner.ema_alphas[1:])),
            'max_alpha_after_initialization': float(np.max(learner.ema_alphas[1:])),
        }
    if prototype_matrix is not None:
        result['prototype_pooled_accuracy'] = prototype_pooled_curve
        result['prototype_per_task_accuracy'] = prototype_task_curve
        result['prototype_metrics'] = {
            'A_AUC_percent': 100 * float(np.mean(prototype_pooled_curve)),
            'AF_s_percent': 100 * float(np.mean(prototype_matrix[-1] - prototype_matrix.max(axis=0))),
            'final_accuracy_percent': 100 * prototype_pooled_curve[-1],
            'full_training_graph_inferences': len(checkpoints),
            'stored_prototype_bytes': 0,
        }
    if propagation_matrix is not None:
        result['propagation_pooled_accuracy'] = propagation_pooled_curve
        result['propagation_per_task_accuracy'] = propagation_task_curve
        result['propagation_metrics'] = {
            'A_AUC_percent': 100 * float(np.mean(propagation_pooled_curve)),
            'AF_s_percent': 100 * float(np.mean(propagation_matrix[-1] - propagation_matrix.max(axis=0))),
            'final_accuracy_percent': 100 * propagation_pooled_curve[-1],
            'message_passing_iterations_per_eval_graph': int(config.get('label_propagation_steps', 0)),
            'stored_state_bytes': 0,
        }
    if blend_matrix is not None:
        result['blend_pooled_accuracy'] = blend_pooled_curve
        result['blend_per_task_accuracy'] = blend_task_curve
        result['blend_metrics'] = {
            'A_AUC_percent': 100 * float(np.mean(blend_pooled_curve)),
            'AF_s_percent': 100 * float(np.mean(blend_matrix[-1] - blend_matrix.max(axis=0))),
            'final_accuracy_percent': 100 * blend_pooled_curve[-1],
            'extra_full_eval_graph_inferences': len(checkpoints) * len(eval_graphs),
            'stored_state_bytes': 0,
        }
    atomic_json(path, result)
    print(f"[done] {path.name}: A_AUC={result['metrics']['A_AUC_percent']:.2f} AF_s={result['metrics']['AF_s_percent']:.2f}", flush=True)
    del learner, model, graph, labels, masked_labels
    gc.collect()
    if args.cuda:
        torch.cuda.empty_cache()
    return result


def summarize(output, results):
    groups = {}
    for result in results:
        groups.setdefault(result['spec']['config']['name'], []).append(result)
    summary = {'selection_metric': 'equal-sigma mean validation A_AUC', 'groups': {}}
    for name, runs in groups.items():
        runs.sort(key=lambda r: r['spec']['sigma'])
        summary['groups'][name] = {
            'mean_A_AUC_percent': float(np.mean([r['metrics']['A_AUC_percent'] for r in runs])),
            'mean_AF_s_percent': float(np.mean([r['metrics']['AF_s_percent'] for r in runs])),
            'by_sigma': {str(r['spec']['sigma']): r['metrics'] for r in runs},
            'compute': {str(r['spec']['sigma']): r['budget']['compute'] for r in runs},
        }
        if all('prototype_metrics' in run for run in runs):
            summary['groups'][name]['prototype_mean_A_AUC_percent'] = float(np.mean([
                run['prototype_metrics']['A_AUC_percent'] for run in runs
            ]))
            summary['groups'][name]['prototype_by_sigma'] = {
                str(run['spec']['sigma']): run['prototype_metrics'] for run in runs
            }
        if all('propagation_metrics' in run for run in runs):
            summary['groups'][name]['propagation_mean_A_AUC_percent'] = float(np.mean([
                run['propagation_metrics']['A_AUC_percent'] for run in runs
            ]))
            summary['groups'][name]['propagation_by_sigma'] = {
                str(run['spec']['sigma']): run['propagation_metrics'] for run in runs
            }
        if all('blend_metrics' in run for run in runs):
            summary['groups'][name]['blend_mean_A_AUC_percent'] = float(np.mean([
                run['blend_metrics']['A_AUC_percent'] for run in runs
            ]))
            summary['groups'][name]['blend_mean_AF_s_percent'] = float(np.mean([
                run['blend_metrics']['AF_s_percent'] for run in runs
            ]))
            summary['groups'][name]['blend_by_sigma'] = {
                str(run['spec']['sigma']): run['blend_metrics'] for run in runs
            }
    atomic_json(output / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--dataset', choices=tuple(DATASET_SIGMAS), default='CoraFull-CL')
    parser.add_argument('--experiment', choices=('h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10', 'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17', 'h18', 'h19', 'h20', 'h21', 'h22', 'h23', 'h24', 'h25', 'h26', 'h27', 'h28', 'h29'), required=True)
    parser.add_argument('--partition', choices=('validation', 'test'), default='validation')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--original-data-root', type=Path, default=ROOT / 'data' / 'raw')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'results_gaussian_science')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--dry-run', action='store_true')
    cli = parser.parse_args()
    split = cli.data_root / f'tr0.6_va0.2_te0.2_split_{cli.dataset}.pkl'
    if cli.dataset == 'CoraFull-CL' and not split.exists():
        raise FileNotFoundError(split)
    if cli.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    output = cli.output_root / cli.dataset / cli.partition / cli.experiment
    configs = configurations(cli.experiment)
    if cli.dry_run:
        sigmas = DATASET_SIGMAS[cli.dataset]
        print(json.dumps({'runs': len(configs) * len(sigmas), 'configs': configs, 'sigmas': sigmas}, indent=2))
        return
    args = base_args(cli.dataset, cli.data_root, cli.original_data_root, cli.device)
    prepared_full = prepare(args, cli.partition)
    if not split.exists():
        raise FileNotFoundError(f'dataset preparation did not create {split}')
    dataset, task_train, merged, mapping, eval_graphs, eval_ids = prepared_full
    prepared = (dataset, task_train, merged, mapping)
    eval_data = (eval_graphs, eval_ids, cli.partition)
    sources = [split, ROOT / 'Baselines/context_ema_model.py', ROOT / 'Baselines/replay_base.py', ROOT / 'gaussian_utils.py', Path(__file__)]
    fingerprint = {
        'files': {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        'runtime': {'python': platform.python_version(), 'torch': torch.__version__, 'dgl': dgl.__version__, 'numpy': np.__version__},
    }
    results = []
    for config in configs:
        for sigma in DATASET_SIGMAS[cli.dataset]:
            results.append(run_one(prepared, eval_data, args, config, sigma, cli.seed, output, fingerprint))
    summarize(output, results)


if __name__ == '__main__':
    main()
