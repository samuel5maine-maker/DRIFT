"""
Missing-baselines study (baselines spec §6-§8), protocol t2 (fixed OMP_NUM_THREADS=2).

    python analysis/baselines_report.py select     # seed-0 sweeps -> experiments/selected_hparams.json + sweep tables
    python analysis/baselines_report.py report     # results tables, §6.3 diagnostic, memory accounting

Reads results_t2/*.pkl and telemetry_t2/**; writes analysis/baselines/*.
"""
import glob
import json
import math
import os
import pickle
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ROOT, SERIES, TEXT2, MUTED, savefig, style

RESULTS = os.path.join(ROOT, 'results_t2')
TELEMETRY = os.path.join(ROOT, 'telemetry_t2')
OUT = os.path.join(ROOT, 'analysis', 'baselines')
SELECTED = os.path.join(ROOT, 'experiments', 'selected_hparams.json')
MAIN = {'CoraFull-CL': 'gaussian_sigma20.0', 'Arxiv-CL': 'gaussian_sigma60.0'}

# DRIFT arXiv:2605.12998v3 Table 2, Gaussian mixing: (A_AUC, sd, AF_s, sd)
PAPER = {
    'CoraFull-CL': {'Bare': (21.9, 0.8, -60.5, 7.3), 'Joint': (86.3, 0.1, None, None), 'A-GEM': (29.9, 2.8, -53.9, 4.8),
                    'ER': (27.8, 0.6, -48.0, 5.2), 'GSS': (29.3, 1.4, -55.3, 0.1), 'MAS*': (29.8, 2.2, -44.8, 8.0),
                    'SSM': (25.4, 2.2, -58.4, 3.8), 'SEM': (27.5, 1.9, -58.4, 2.6), 'DMSG': (34.1, 1.0, -37.6, 3.5)},
    'Arxiv-CL': {'Bare': (18.5, 1.5, -65.4, 3.1), 'Joint': (71.6, 1.4, None, None), 'A-GEM': (34.1, 1.4, -48.6, 4.4),
                 'ER': (34.9, 0.8, -37.3, 2.8), 'GSS': (24.2, 3.2, -60.4, 5.6), 'MAS*': (38.4, 2.1, -22.2, 1.8),
                 'SSM': (28.0, 4.3, -64.5, 1.9), 'SEM': (26.8, 1.0, -59.7, 6.9), 'DMSG': (31.6, 0.9, -31.9, 2.9)},
}
LABEL = {'bare': 'Bare', 'er': 'ER', 'agem': 'A-GEM', 'tfmas': 'MAS*', 'dmsg': 'DMSG', 'er_cbrs': 'ER-CBRS',
         'der': 'DER', 'derpp': 'DER++', 'pdgnn': 'PDGNN', 'lwf_online': 'LwF-online', 'clser': 'CLS-ER',
         'dercls': 'DER++ + CLS-ER (CBRS)'}
EXISTING = {'bare', 'er', 'agem', 'tfmas', 'dmsg'}
NAME_RE = re.compile(r'^(?P<dataset>.+?)_(?P<backbone>GCN|GAT|GIN|SGC)_(?P<method>[a-z_]+?)_batch\d+_'
                     r'(?P<regime>gaussian_sigma[\d.]+|boundaryblurry_K\d+_ratio\d+|blurry\d+|clsincre)'
                     r'(?P<hp>.*?)_seed(?P<seed>\d+)$')


def load():
    rows = []
    for p in glob.glob(os.path.join(RESULTS, '*.pkl')):
        m = NAME_RE.match(os.path.basename(p)[:-4])
        if not m:
            continue
        with open(p, 'rb') as f:
            result_list, avg_acc_list = pickle.load(f)[:2]
        acc, res = np.asarray(avg_acc_list, float), np.asarray(result_list, float)
        d = m.groupdict()
        rows.append({**d, 'seed': int(d['seed']), 'AAUC': 100 * acc.mean(), 'AFs': 100 * (res[-1] - res.max(0)).mean()})
    return pd.DataFrame(rows)


def parse_hp(hp):
    """'_alpha0.5_beta1.0' -> {'alpha': 0.5, 'beta': 1.0} (keys are sorted in names; values float or str)."""
    out = {}
    for tok in hp.strip('_').split('_') if hp else []:
        m = re.match(r'([a-zA-Z]+(?:_[a-zA-Z]+)*?)(-?[\d.e+-]+|[a-z]+)$', tok)
        if m:
            k, v = m.groups()
            try:
                v = float(v)
            except ValueError:
                pass
            out[k] = v
    return out


def split_hp(hp_string, method):
    """Method names can contain underscores and so can keys, so parse against the known argument names."""
    keys = {'der': ['alpha'], 'derpp': ['alpha', 'beta'], 'lwf_online': ['T', 'lambda_dist', 'update_every'],
            'clser': ['plastic_alpha', 'plastic_update_freq', 'reg_weight', 'stable_alpha', 'stable_update_freq',
                      'eval_model'],
            'dercls': ['alpha', 'beta', 'ema_alpha', 'ema_update_freq', 'eval_model', 'gamma'],
            'er_cbrs': ['loss', 'replay', 'buffer'], 'pdgnn': ['buffer']}.get(method, [])
    out, s = {}, hp_string
    for k in sorted(keys, key=len, reverse=True):
        m = re.search(rf'_{k}(-?[\d.]+(?:e-?\d+)?|[a-z]+)(?=_|$)', s)
        if m:
            v = m.group(1)
            try:
                v = float(v)
            except ValueError:
                pass
            out[k] = v
            s = s[:m.start()] + s[m.end():]
    return out


def fmt(mean, sd, n):
    if n == 0 or mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return '–'
    return f'{mean:.1f} ± {sd:.1f}' if n > 1 else f'{mean:.1f}'


# ------------------------------------------------------------------ select
def select():
    df = load()
    tune = df[df['seed'] == 0]
    selected, lines = {}, ['# Seed-0 hyperparameter sweeps (protocol t2)', '',
                           'Selection: highest A_AUC on seed 0 per method and dataset (seed 0 is not an evaluation seed). '
                           'Single-seed selection is noisy; the full sweep is shown so the choice can be judged.', '']
    for dataset, regime in MAIN.items():
        selected[dataset] = {}
        for method in ('der', 'derpp', 'lwf_online', 'clser'):
            d = tune[(tune.dataset == dataset) & (tune.regime == regime) & (tune.method == method)].copy()
            if d.empty:
                continue
            d['hp_dict'] = d['hp'].map(lambda h, m=method: split_hp(h, m))
            d = d.sort_values('AAUC', ascending=False)
            best = d.iloc[0]
            selected[dataset][method] = best['hp_dict']
            lines += [f'## {dataset} — {LABEL[method]} ({len(d)} configs)', '', '| rank | hyperparameters | A_AUC | AF_s |',
                      '|---|---|---|---|']
            for i, (_, r) in enumerate(d.iterrows(), 1):
                hp = ', '.join(f'{k}={v:g}' if isinstance(v, float) else f'{k}={v}' for k, v in sorted(r['hp_dict'].items()))
                lines.append(f"| {i}{' ←' if i == 1 else ''} | {hp} | {r['AAUC']:.1f} | {r['AFs']:.1f} |")
            lines.append('')
    os.makedirs(OUT, exist_ok=True)
    with open(SELECTED, 'w') as f:
        json.dump(selected, f, indent=2, sort_keys=True)
    with open(os.path.join(OUT, 'tuning.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    sys.stdout.reconfigure(encoding='utf-8')
    print('\n'.join(lines))
    print(json.dumps(selected, indent=2))


# ------------------------------------------------------------------ report
def telemetry_frames(dataset, regime, name):
    frames = {}
    for seed_dir in glob.glob(os.path.join(TELEMETRY, dataset, '*', regime, name, 'seed*')):
        for f in glob.glob(os.path.join(seed_dir, '*.csv')):
            frames.setdefault(os.path.basename(f)[:-4], []).append(pd.read_csv(f))
    return {k: pd.concat(v, ignore_index=True) for k, v in frames.items()}


def coverage_diagnostic(lines):
    """Spec §6.3: final per-class accuracy vs final per-class buffer occupancy, ER (uniform reservoir) vs ER-CBRS."""
    style()
    datasets = [d for d in MAIN if any(telemetry_frames(d, MAIN[d], m).get('per_class_final') is not None
                                       for m in ('er', 'er_cbrs'))]
    if not datasets:
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 4.0), squeeze=False)
    lines += ['## §6.3 Coverage diagnostic: final per-class accuracy vs final buffer slots', '',
              'Seeds 1–3 pooled; one point per (class, seed). Pearson r and Spearman ρ over all points; '
              '"acc | 0 slots" vs "acc | ≥1 slot" compares classes the memory missed with classes it covered.', '',
              '| dataset | method | points | Pearson r | Spearman ρ | classes with 0 slots (mean per seed) | acc at 0 slots | acc at ≥1 slot |',
              '|---|---|---|---|---|---|---|---|']
    for ax, dataset in zip(axes[0], datasets):
        for color, method in zip((SERIES[0], SERIES[1]), ('er', 'er_cbrs')):
            pc = telemetry_frames(dataset, MAIN[dataset], method).get('per_class_final')
            if pc is None:
                continue
            pc = pc[pc['seed'] > 0]
            x, y = pc['slot_count'].astype(float), 100 * pc['accuracy']
            pear = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 else math.nan
            spear = float(pd.Series(x).rank().corr(pd.Series(y).rank())) if x.std() > 0 else math.nan
            zero = pc[pc['slot_count'] == 0]
            some = pc[pc['slot_count'] > 0]
            jitter = (np.random.default_rng(0).random(len(x)) - 0.5) * 0.3
            ax.scatter(x + jitter + (0.15 if method == 'er_cbrs' else -0.15), y, s=12, alpha=0.55, color=color,
                       label=f'{LABEL[method]}  r={pear:.2f}', edgecolor='none')
            if x.std() > 0:
                k, b = np.polyfit(x, y, 1)
                xs = np.linspace(x.min(), x.max(), 10)
                ax.plot(xs, k * xs + b, color=color, lw=2)
            lines.append(f"| {dataset} | {LABEL[method]} | {len(pc)} | {pear:.2f} | {spear:.2f} | "
                         f"{(pc['slot_count'] == 0).sum() / pc['seed'].nunique():.1f} | "
                         f"{100 * zero['accuracy'].mean() if len(zero) else math.nan:.1f} | "
                         f"{100 * some['accuracy'].mean() if len(some) else math.nan:.1f} |")
        ax.set_title(f'{dataset} ({MAIN[dataset].replace("gaussian_", "")})', loc='left')
        ax.set_xlabel('final buffer slots for the class')
        ax.set_ylabel('final class accuracy (%)')
        ax.legend(fontsize=8, loc='upper left')
    fig.suptitle('§6.3 — Does buffer coverage predict per-class accuracy?', x=0.01, ha='left', fontweight='bold')
    savefig(fig, os.path.join(OUT, 'coverage_diagnostic.png'))
    lines += ['', '![coverage](coverage_diagnostic.png)', '']


def report():
    df = load()
    ev = df[df['seed'].isin([1, 2, 3])]
    lines = ['# Missing continual-learning baselines on DRIFT — results (protocol t2)', '',
             'Mean ± std over seeds 1–3, CPU (DGL 1.1.2), fixed OMP_NUM_THREADS=2. Metrics recomputed from `results_t2/*.pkl` '
             'exactly as `metrics.tf_metrics`: A_AUC ↑ and AF_s = mean_k(final − peak) ↑ (≤ 0). '
             '**AF_s can be improved by underfitting (a task never learned has no peak to fall from): read it only with A_AUC.**', '',
             'source: **paper** = DRIFT Table 2; **reproduced** = DRIFT method re-run here; **new** = added here. '
             'Paper numbers come from a different environment and are not directly comparable to rows run here (see NOTES.md, Gate 0).', '']
    selected = json.load(open(SELECTED)) if os.path.exists(SELECTED) else {}
    os.makedirs(OUT, exist_ok=True)
    for dataset, regime in MAIN.items():
        rows = []
        for name, (a, asd, f, fsd) in PAPER[dataset].items():
            rows.append((name, f'{a} ± {asd}', '–' if f is None else f'{f} ± {fsd}', 'paper'))
        d = ev[(ev.dataset == dataset) & (ev.regime == regime)]
        for (method, hp, backbone), g in d.groupby(['method', 'hp', 'backbone']):
            label = LABEL.get(method, method)
            extra = split_hp(hp, method)
            if extra:
                label += ' (' + ', '.join(f'{k}={v:g}' if isinstance(v, float) else f'{k}={v}' for k, v in sorted(extra.items())) + ')'
            if backbone != 'GCN':
                label += f' [{backbone}]'
            n = len(g)
            rows.append((label, fmt(g.AAUC.mean(), g.AAUC.std(ddof=1), n) + f' (n={n})',
                         fmt(g.AFs.mean(), g.AFs.std(ddof=1), n), 'reproduced' if method in EXISTING else 'new'))
        lines += [f'## {dataset}, Gaussian mixing, {regime.replace("gaussian_", "")}', '', '| Method | A_AUC ↑ | AF_s ↑ | source |',
                  '|--------|---------|--------|--------|'] + [f'| {r[0]} | {r[1]} | {r[2]} | {r[3]} |' for r in rows] + ['']
        pd.DataFrame(rows, columns=['method', 'A_AUC', 'AF_s', 'source']).to_csv(
            os.path.join(OUT, f'{regime}_{dataset}.csv'), index=False)

    # memory accounting (spec §2)
    mem = []
    for f in glob.glob(os.path.join(TELEMETRY, '*', '*', '*', '*', 'seed[123]', 'memory_accounting.csv')):
        parts = f.split(os.sep)
        r = pd.read_csv(f)
        r['dataset'], r['backbone'], r['regime'] = parts[-6], parts[-5], parts[-4]
        mem.append(r)
    if mem:
        mem = pd.concat(mem).groupby(['dataset', 'backbone', 'method'])[
            ['buffer_bytes', 'extra_param_count', 'extra_param_bytes', 'total_bytes']].max().reset_index()
        mem.to_csv(os.path.join(OUT, 'memory_accounting.csv'), index=False)
        lines += ['## Memory beyond the backbone (spec §2)', '',
                  'buffer_bytes: node id + label per slot, plus stored logits (DER family) or embeddings (PDGNN). '
                  'extra params: parameter copies a method keeps (EMA models, teacher, importance weights, discriminator). '
                  'Replay graphs are materialised from dataset.graph at replay time and are not counted.', '',
                  '| dataset | method | backbone | buffer bytes | extra params | extra param bytes | total bytes |',
                  '|---|---|---|---|---|---|---|']
        for _, r in mem.sort_values(['dataset', 'total_bytes']).iterrows():
            lines.append(f"| {r.dataset} | {r.method} | {r.backbone} | {int(r.buffer_bytes):,} | {int(r.extra_param_count):,} | "
                         f"{int(r.extra_param_bytes):,} | {int(r.total_bytes):,} |")
        lines.append('')

    coverage_diagnostic(lines)
    with open(os.path.join(OUT, 'results.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    sys.stdout.reconfigure(encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    {'select': select, 'report': report}[sys.argv[1] if len(sys.argv) > 1 else 'report']()
