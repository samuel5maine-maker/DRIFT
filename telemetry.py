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
