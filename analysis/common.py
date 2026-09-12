"""Shared loaders and figure style for the MAS* telemetry analysis. Reads telemetry/ and results/ only."""
import glob
import math
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TELEMETRY = os.path.join(ROOT, 'telemetry')
RESULTS = os.path.join(ROOT, 'results')

# Reference palette (dataviz skill, light mode): categorical slots in fixed order, text inks, surface.
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
TEXT, TEXT2, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#8a8984', '#e6e5e1', '#fcfcfb'
SEQ_BLUE = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']

REGIME_ORDER = ['clsincre', 'gaussian_sigma3.0', 'gaussian_sigma10.0', 'gaussian_sigma20.0',
                'boundaryblurry_K5_ratio50', 'blurry30']
REGIME_LABEL = {'clsincre': 'hard (tfocis)', 'gaussian_sigma3.0': 'σ=3', 'gaussian_sigma10.0': 'σ=10',
                'gaussian_sigma20.0': 'σ=20', 'boundaryblurry_K5_ratio50': 'boundary K=5',
                'blurry30': 'global 30%'}


def style():
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': GRID, 'axes.labelcolor': TEXT2, 'xtick.color': TEXT2, 'ytick.color': TEXT2,
        'text.color': TEXT, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
        'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.5,
        'font.size': 9, 'axes.titlesize': 10, 'axes.titleweight': 'bold', 'legend.frameon': False,
    })


def run_dirs(dataset='CoraFull-CL', backbone='GCN'):
    """Yield dicts describing every telemetry run directory: regime, run name, seed, path."""
    for path in glob.glob(os.path.join(TELEMETRY, dataset, backbone, '*', '*', 'seed*')):
        parts = path.split(os.sep)
        yield {'regime': parts[-3], 'run': parts[-2], 'seed': int(parts[-1][4:]), 'path': path}


def parse_thresholds(run):
    m = re.search(r'lth(-?[\d.e+-]+)_sth(-?[\d.e+-]+)', run)
    return (float(m.group(1)), float(m.group(2))) if m else (math.nan, math.nan)


def load(path, name):
    p = os.path.join(path, f'{name}.csv')
    return pd.read_csv(p) if os.path.exists(p) else None


def task_boundaries(stream):
    """Steps where the ground-truth dominant task changes."""
    if stream is None or 'task' not in stream:
        return []
    t = stream['task'].to_numpy()
    return [int(stream['step'].iloc[i]) for i in range(1, len(t)) if t[i] != t[i - 1]]


def savefig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
