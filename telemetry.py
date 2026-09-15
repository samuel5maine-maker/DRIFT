"""
Buffered CSV telemetry for continual-learning methods.

Enabled by setting the MAS_TELEMETRY_DIR environment variable. When it is unset,
`make_telemetry` returns None and callers skip all logging, so the training
path is unchanged. Rows are held in memory and flushed periodically and at exit.
"""

import atexit
import csv
import os


class CSVTelemetry:
    def __init__(self, out_dir, flush_every=500):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.flush_every = flush_every
        self._rows = {}      # name -> list of dicts
        self._fields = {}    # name -> list of column names (fixed by first row)
        self._started = set()
        atexit.register(self.close)

    def log(self, name, row):
        if name not in self._fields:
            self._fields[name] = list(row.keys())
            self._rows[name] = []
        self._rows[name].append(row)
        if len(self._rows[name]) >= self.flush_every:
            self._flush(name)

    def _flush(self, name):
        rows = self._rows.get(name)
        if not rows:
            return
        path = os.path.join(self.out_dir, f'{name}.csv')
        mode = 'a' if name in self._started else 'w'
        with open(path, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._fields[name], extrasaction='ignore')
            if mode == 'w':
                writer.writeheader()
            writer.writerows(rows)
        self._started.add(name)
        self._rows[name] = []

    def close(self):
        for name in list(self._rows):
            self._flush(name)


def regime_name(args):
    setting = args.setting
    if setting == 'tfo_gaussian':
        return f'gaussian_sigma{args.gaussian_sigma}'
    if setting == 'tfo_bb':
        return f'boundaryblurry_K{args.blurry_batch_count}_ratio{int(args.boundary_mix_ratio * 100)}'
    if setting == 'tfo_blurry':
        return f'blurry{int(round((1.0 - args.percentage) * 100))}'
    if setting == 'tfocis':
        return 'clsincre'
    return setting


def make_telemetry(args, run_name):
    root = os.environ.get('MAS_TELEMETRY_DIR')
    if not root:
        return None
    out_dir = os.path.join(root, args.dataset, args.backbone, regime_name(args), run_name, f'seed{args.seed}')
    return CSVTelemetry(out_dir)


def buffer_labels(life_model):
    """Labels currently held in a method's replay memory, or None if the method has no memory."""
    if hasattr(life_model, 'buffer_labels'):
        labels = life_model.buffer_labels()
        return None if labels is None else [int(y) for y in labels]
    aux = getattr(life_model, 'aux_labels', None)          # DRIFT's ER / A-GEM / DMSG keep labels on the aux graph
    if aux is not None:
        return [int(y) for y in aux.view(-1).tolist()]
    if hasattr(life_model, 'buffer_node_ids'):
        return []
    return None


def memory_accounting(life_model):
    """Bytes a method holds beyond the backbone: replay slots and any extra parameter copies."""
    if hasattr(life_model, 'memory_accounting'):
        return life_model.memory_accounting()
    labels = buffer_labels(life_model)
    buffer_bytes = 16 * len(labels) if labels is not None else 0     # int64 node id + int64 label per slot
    extra = 0
    for name in ('distingush_model',):                               # DMSG's adversarial discriminator
        mod = getattr(life_model, name, None)
        if mod is not None:
            extra += sum(p.numel() for p in mod.parameters())
    for name in ('omegas', 'star_variables', 'omega', 'theta_star'):   # MAS-style importance and anchor copies
        tensors = getattr(life_model, name, None)
        if tensors:
            extra += sum(t.numel() for t in tensors)
    return {'buffer_bytes': buffer_bytes, 'extra_param_count': extra, 'extra_param_bytes': 4 * extra}


class EvalTelemetry:
    """
    Evaluation-time logs (baselines spec §6): per_task_accuracy.csv at every evaluation point,
    buffer_occupancy.csv per class and per latent task, and at the end per_class_final.csv and
    memory_accounting.csv. Read-only: no RNG, no parameter updates. Disabled unless MAS_TELEMETRY_DIR is set.
    """

    def __init__(self, args):
        from training.utils import run_name
        self.args = args
        self.tel = make_telemetry(args, run_name(args))
        self.method = run_name(args)

    def _occupancy(self, step, life_model):
        labels = buffer_labels(life_model)
        if labels is None:
            return
        per_cls = int(self.args.n_cls_per_task)
        n_cls = int(self.args.n_cls)
        counts = [0] * n_cls
        for y in labels:
            counts[y] += 1
        for c in range(n_cls):
            self.tel.log('buffer_occupancy', {'step': step, 'method': self.method, 'seed': self.args.seed,
                                              'group_type': 'class', 'group_id': c, 'slot_count': counts[c]})
        for t in range((n_cls + per_cls - 1) // per_cls):
            self.tel.log('buffer_occupancy', {'step': step, 'method': self.method, 'seed': self.args.seed,
                                              'group_type': 'task', 'group_id': t,
                                              'slot_count': sum(counts[t * per_cls:(t + 1) * per_cls])})
        return counts

    def _alt_curves(self, step, life_model, eval_fn):
        """Accuracy of a method's other models (e.g. CLS-ER working/plastic) at the same evaluation point."""
        if eval_fn is None or not hasattr(life_model, 'alt_eval_models'):
            return
        for name, net in life_model.alt_eval_models().items():
            res, avg, _, _ = eval_fn(net)
            for k, acc in enumerate(res):
                self.tel.log('alt_model_accuracy', {'step': step, 'method': self.method, 'seed': self.args.seed,
                                                    'model': name, 'task': k, 'accuracy': float(acc)})
            self.tel.log('alt_model_accuracy', {'step': step, 'method': self.method, 'seed': self.args.seed,
                                                'model': name, 'task': -1, 'accuracy': float(avg)})

    def on_eval(self, step, res_per_t, life_model, eval_fn=None):
        if self.tel is None:
            return
        for k, acc in enumerate(res_per_t):
            self.tel.log('per_task_accuracy', {'step': step, 'method': self.method, 'seed': self.args.seed,
                                               'task': k, 'accuracy': float(acc)})
        self._occupancy(step, life_model)
        self._alt_curves(step, life_model, eval_fn)

    def on_final(self, step, model, subgraphs, tasks_te, life_model, res_per_t, masked=True, eval_fn=None):
        if self.tel is None:
            return
        self._alt_curves(step, life_model, eval_fn)
        for k, acc in enumerate(res_per_t):
            self.tel.log('per_task_accuracy', {'step': step, 'method': self.method, 'seed': self.args.seed,
                                               'task': k, 'accuracy': float(acc)})
        counts = self._occupancy(step, life_model)
        import torch
        model.eval()
        cls_seen = set()
        per_cls = int(self.args.n_cls_per_task)
        with torch.no_grad():
            for t, task_te in enumerate(tasks_te):
                g = subgraphs[t]
                labels = g.dstdata['label'].squeeze()
                output, _ = model(g, g.srcdata['feat'])
                logits = output[task_te]
                y = labels[task_te]
                if masked:                                   # same head restriction as pipeline.eval_tasks_cis
                    cls_seen.update(int(c) for c in y.unique())
                    top = max(cls_seen) + 1
                    top += top % 2
                    logits = logits[:, :top]
                pred = logits.argmax(1)
                for c in sorted(int(v) for v in y.unique()):
                    m = y == c
                    self.tel.log('per_class_final', {
                        'method': self.method, 'seed': self.args.seed, 'class': c, 'task': c // per_cls,
                        'accuracy': float((pred[m] == c).float().mean()), 'n_test': int(m.sum()),
                        'slot_count': counts[c] if counts is not None else ''})
        acc = memory_accounting(life_model)
        self.tel.log('memory_accounting', {'method': self.method, 'seed': self.args.seed, **acc,
                                           'total_bytes': acc['buffer_bytes'] + acc['extra_param_bytes']})
        self.tel.close()
