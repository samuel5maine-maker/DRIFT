"""
Phase T1/T2: loss scale and exact offline first-fire map from detector-off MAS* traces (calibration seed 0).

Before its first consolidation, MAS* has no detector side effects (Omega = 0, so no penalty; the buffer
update ignores the detector; the detector consumes no RNG). A detector-off run therefore follows exactly the
trajectory of any detector-on run up to that run's first firing, so the first-fire step for any (l_th, std_th)
is the first step with a buffer, win_mean < l_th and win_std < std_th.

    python analysis/calibration.py          -> analysis/calibration/*.png, calibration_t1t2.md,
                                               experiments/thresholds_t3.txt
"""
import math
import os

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

from common import (ROOT, REGIME_LABEL, REGIME_ORDER, SERIES, SEQ_BLUE, TEXT2, MUTED, load, run_dirs,
                    savefig, style, task_boundaries)

OUT = os.path.join(ROOT, 'analysis', 'calibration')
OFF_RUN = 'tfmas_star_lth-1.0_sth-1.0'
QUANTILES = [5, 15, 30, 50]
CALIB_SEED = 0


def first_fire(steps, l_th, std_th):
    ok = steps['cond_buffer'].astype(bool) & (steps['win_mean'] < l_th) & (steps['win_std'] < std_th)
    idx = np.flatnonzero(ok.to_numpy())
    return int(steps['step'].iloc[idx[0]]) if len(idx) else None


def sig(x, n=4):
    return float(f'{x:.{n}g}')


def main():
    style()
    os.makedirs(OUT, exist_ok=True)
    traces = {}
    for r in run_dirs():
        if r['run'] == OFF_RUN and r['seed'] == CALIB_SEED:
            traces[r['regime']] = (load(r['path'], 'steps'), load(r['path'], 'stream'))
    regimes = [g for g in REGIME_ORDER if g in traces]
    if 'clsincre' not in traces:
        raise SystemExit('need the detector-off clsincre trace (experiments/run_matrix.py --plan calib_t1)')

    # ---- T1: loss scale against thresholds-to-be and chance level
    fig, axes = plt.subplots(len(regimes), 1, figsize=(9, 2.3 * len(regimes)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, g in zip(axes, regimes):
        st, stream = traces[g]
        ax.plot(st['step'], st['win_mean'], color=SERIES[0], lw=1.5, label='window mean μ(W)')
        ax.plot(st['step'], st['win_std'], color=SERIES[1], lw=1.5, label='window std σ(W)')
        if (st['offset2'] > 0).any():
            chance = np.log(st['offset2'].clip(lower=2))
            ax.plot(st['step'], 2 * chance, color=MUTED, lw=1.2, ls='--', label='chance: 2·ln(head width)')
        for b in task_boundaries(stream):
            ax.axvline(b, color=MUTED, lw=0.4, alpha=0.35, zorder=0)
        ax.set_title(REGIME_LABEL.get(g, g), loc='left')
        ax.set_ylabel('loss')
    axes[0].legend(loc='upper right', ncol=3)
    axes[-1].set_xlabel('stream step (thin verticals: ground-truth dominant-task changes)')
    fig.suptitle('Detector-off MAS*: what the plateau detector would see (window = L(X,Y)+L(X_B,Y_B))', x=0.01,
                 ha='left', fontweight='bold')
    savefig(fig, os.path.join(OUT, 't1_loss_scale.png'))

    # ---- threshold grid from the hard-control trace
    ref = traces['clsincre'][0]
    ref = ref[ref['cond_buffer'].astype(bool) & (ref['win_len'] >= ref['win_len'].max())]
    mu_q = {q: sig(np.percentile(ref['win_mean'], q)) for q in QUANTILES}
    sd_q = {q: sig(np.percentile(ref['win_std'], q)) for q in QUANTILES}
    pairs = [(mu_q[a], sd_q[b]) for a in QUANTILES for b in QUANTILES]
    with open(os.path.join(ROOT, 'experiments', 'thresholds_t3.txt'), 'w') as f:
        f.write('# l_th std_th  (percentiles 5/15/30/50 of mu(W) x sigma(W), clsincre detector-off, seed 0)\n')
        for l, s in pairs:
            f.write(f'{l} {s}\n')

    # ---- T2: dense first-fire map per regime
    all_mean = pd.concat([traces[g][0]['win_mean'] for g in regimes])
    all_std = pd.concat([traces[g][0]['win_std'] for g in regimes])
    l_grid = np.geomspace(max(all_mean.min(), 1e-3), all_mean.max(), 60)
    s_grid = np.geomspace(max(all_std[all_std > 0].min(), 1e-4), all_std.max(), 60)
    n_steps = max(len(traces[g][0]) for g in regimes)
    cmap = ListedColormap(SEQ_BLUE)
    cmap.set_bad('#d9d8d4')
    fig, axes = plt.subplots(1, len(regimes), figsize=(3.2 * len(regimes), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, g in zip(axes, regimes):
        st = traces[g][0]
        # first step at which both conditions hold, vectorised over the grid
        mean = st['win_mean'].to_numpy()[:, None, None]
        std = st['win_std'].to_numpy()[:, None, None]
        buf = st['cond_buffer'].astype(bool).to_numpy()[:, None, None]
        ok = buf & (mean < l_grid[None, None, :]) & (std < s_grid[None, :, None])
        first = np.where(ok.any(0), ok.argmax(0), np.nan).astype(float)
        im = ax.pcolormesh(l_grid, s_grid, np.ma.masked_invalid(first), cmap=cmap, vmin=0, vmax=n_steps,
                           shading='auto')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.scatter([p[0] for p in pairs], [p[1] for p in pairs], s=14, facecolor='white', edgecolor=TEXT2, lw=0.8,
                   zorder=3)
        ax.set_title(REGIME_LABEL.get(g, g), loc='left')
        ax.set_xlabel('l_th (δμ)')
    axes[0].set_ylabel('std_th (δσ)')
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label('step of first consolidation (gray = never)')
    fig.suptitle('Exact first-fire map (white dots: T3 sweep grid)', x=0.01, ha='left', fontweight='bold')
    savefig(fig, os.path.join(OUT, 't2_first_fire_map.png'))

    # ---- report
    lines = ['# MAS* threshold calibration — T1/T2', '',
             f'Source: detector-off `tfmas_star` traces (`l_th=-1`), seed {CALIB_SEED}, CoraFull-CL, GCN.', '',
             '## Loss scale (steps with a full window and a buffer)', '',
             '| regime | μ(W) p5 | p15 | p30 | p50 | σ(W) p5 | p15 | p30 | p50 | ln(head) final |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for g in regimes:
        st = traces[g][0]
        st = st[st['cond_buffer'].astype(bool) & (st['win_len'] >= st['win_len'].max())]
        mq = [np.percentile(st['win_mean'], q) for q in QUANTILES]
        sq = [np.percentile(st['win_std'], q) for q in QUANTILES]
        head = st['offset2'].iloc[-1]
        head_str = f'{math.log(head):.2f}' if head > 0 else '–'
        lines.append(f"| {REGIME_LABEL.get(g, g)} | " + ' | '.join(f'{v:.3g}' for v in mq + sq) + f' | {head_str} |')
    lines += ['', '## T3 grid (percentiles of the hard-control trace) and offline first-fire step', '',
              '| l_th | std_th | ' + ' | '.join(REGIME_LABEL.get(g, g) for g in regimes) + ' |',
              '|---|---|' + '---|' * len(regimes)]
    for l, s in pairs:
        cells = []
        for g in regimes:
            f = first_fire(traces[g][0], l, s)
            cells.append('never' if f is None else str(f))
        lines.append(f'| {l} | {s} | ' + ' | '.join(cells) + ' |')
    lines += ['', '![loss scale](calibration/t1_loss_scale.png)', '', '![first fire](calibration/t2_first_fire_map.png)']
    with open(os.path.join(ROOT, 'analysis', 'calibration_t1t2.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
