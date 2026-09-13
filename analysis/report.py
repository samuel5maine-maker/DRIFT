"""
Phase T3 / Phase 2-3 analysis for MAS* on DRIFT. Reads results/*.pkl and telemetry/**; writes
analysis/report_tables.md and analysis/figures/*.png. Re-runnable without retraining.

    python analysis/report.py [--frozen "l_th std_th"]
"""
import argparse
import glob
import math
import os
import pickle
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (ROOT, RESULTS, REGIME_LABEL, REGIME_ORDER, SERIES, TEXT2, MUTED, GRID, load, run_dirs,
                    savefig, style, task_boundaries)

FIG = os.path.join(ROOT, 'analysis', 'figures')
SIGMAS = {'gaussian_sigma3.0': 3, 'gaussian_sigma10.0': 10, 'gaussian_sigma20.0': 20}
# Published CoraFull-CL sigma sweep AAUC (%), as quoted in the telemetry spec (DRIFT paper).
PUBLISHED = {'MAS* (DRIFT)': {3: 14.0, 10: 16.2, 20: 29.8}, 'Bare (DRIFT)': {3: 10.6, 10: 16.2, 20: 21.9}}
ARM_ORDER = ['tfmas_star', 'tfmas_star (no consolidation)', 'tfmas (DRIFT legacy)', 'bare']
ARM_COLOR = dict(zip(ARM_ORDER, SERIES))

NAME_RE = re.compile(
    r'^(?P<dataset>.+?)_(?P<backbone>GCN|GAT|GIN)_(?P<method>.+?)_batch(?P<bs>\d+)_'
    r'(?P<setting>gaussian_sigma[\d.]+|boundaryblurry_K\d+_ratio\d+|blurry\d+|clsincre)'
    r'(?:_lth(?P<lth>-?[\d.e+-]+)_sth(?P<sth>-?[\d.e+-]+))?(?P<extra>.*?)_seed(?P<seed>\d+)$')


# ---------------------------------------------------------------- results

def load_results():
    rows = []
    for p in glob.glob(os.path.join(RESULTS, '*.pkl')):
        name = os.path.basename(p)[:-4]
        m = NAME_RE.match(name)
        if not m:
            continue
        with open(p, 'rb') as f:
            result_list, avg_acc_list, _, _, _, time_spent = pickle.load(f)
        acc = np.asarray(avg_acc_list, dtype=float)
        res = np.asarray(result_list, dtype=float)
        d = m.groupdict()
        lth = float(d['lth']) if d['lth'] else math.nan
        method = d['method']
        if method == 'tfmas_star':
            arm = 'tfmas_star (no consolidation)' if lth < 0 else 'tfmas_star'
        else:
            arm = {'tfmas': 'tfmas (DRIFT legacy)'}.get(method, method)
        rows.append({'dataset': d['dataset'], 'backbone': d['backbone'], 'method': method, 'arm': arm,
                     'regime': d['setting'], 'l_th': lth, 'std_th': float(d['sth']) if d['sth'] else math.nan,
                     'extra': d['extra'], 'seed': int(d['seed']),
                     'AAUC': 100 * acc.mean(), 'AA_final': 100 * acc[-1],
                     'FM': 100 * (res[-1] - res.max(axis=0)).mean(), 'seconds': time_spent})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- telemetry scalars

def run_scalars(steps, stream, events):
    fired = steps['consolidated'].astype(bool)
    c_steps = steps.loc[fired, 'step'].to_numpy()
    gaps = np.diff(c_steps)
    fail = pd.DataFrame({'latch': ~steps['cond_latch'].astype(bool), 'mean': ~steps['cond_mean'].astype(bool),
                         'std': ~steps['cond_std'].astype(bool), 'buffer': ~steps['cond_buffer'].astype(bool)})
    nf = fail[~fired]
    n_fail = nf.sum(axis=1)
    denom = max(len(nf), 1)
    out = {
        'N_c': int(fired.sum()),
        'first_consolidation': int(c_steps[0]) if len(c_steps) else None,
        'interval_median': float(np.median(gaps)) if len(gaps) else math.nan,
        'interval_IQR': float(np.subtract(*np.percentile(gaps, [75, 25]))) if len(gaps) else math.nan,
        'pct_latch_armed': 100 * steps['cond_latch'].mean(),
        'block_latch_only': 100 * ((n_fail == 1) & nf['latch']).sum() / denom,
        'block_mean_only': 100 * ((n_fail == 1) & nf['mean']).sum() / denom,
        'block_std_only': 100 * ((n_fail == 1) & nf['std']).sum() / denom,
        'block_multiple': 100 * (n_fail > 1).sum() / denom,
        'block_buffer_only': 100 * ((n_fail == 1) & nf['buffer']).sum() / denom,
        'n_peaks': int(steps['peak_fired'].sum()),
        'penalty_mean': float(steps['penalty_value'].mean()),
        'pct_penalty_zero': 100 * float((steps['penalty_value'] < 1e-8).mean()),
        'omega_l1_final': float(steps['omega_l1'].iloc[-1]),
    }
    if events is not None and (events['event_type'] == 'consolidate').sum() > 1:
        cos = events.loc[events['event_type'] == 'consolidate', 'cos_estimate_vs_prev_omega'].dropna()
        out['omega_cos_median'] = float(cos.median()) if len(cos) else math.nan
    else:
        out['omega_cos_median'] = math.nan
    bounds = task_boundaries(stream)
    if bounds:
        segs = np.split(np.arange(len(steps)), bounds)
        out['pct_segments_with_consolidation'] = 100 * np.mean([fired.iloc[s].any() for s in segs if len(s)])
    else:
        out['pct_segments_with_consolidation'] = math.nan
    return out


def consolidation_lags(steps, stream, reference):
    c = steps.loc[steps['consolidated'].astype(bool), 'step'].to_numpy()
    lags = []
    for r in reference:
        nxt = c[c >= r]
        lags.append(float(nxt[0] - r) if len(nxt) else math.nan)
    return lags


def task_centres(stream):
    """Step at which each task's mixing weight peaks (used where no task dominates, e.g. sigma=20)."""
    cols = [c for c in stream.columns if c.startswith('alpha_') and c != 'alpha_max']
    return sorted(int(stream['step'].iloc[stream[c].to_numpy().argmax()]) for c in cols)


def alpha_velocity(stream):
    cols = [c for c in stream.columns if c.startswith('alpha_') and c != 'alpha_max']
    a = stream[cols].to_numpy()
    return np.concatenate([[0.0], np.abs(np.diff(a, axis=0)).sum(axis=1)])


def permutation_test(values_at_events, all_values, n=10000, seed=0):
    """One-sided: are events at lower alpha velocity than random steps? Returns (observed mean, null mean, p)."""
    rng = np.random.default_rng(seed)
    k = len(values_at_events)
    obs = float(np.mean(values_at_events))
    null = np.array([all_values[rng.choice(len(all_values), k, replace=False)].mean() for _ in range(n)])
    return obs, float(null.mean()), float((np.sum(null <= obs) + 1) / (n + 1))


# ---------------------------------------------------------------- figures

def fig_detector(runs, frozen, path):
    regs = [g for g in REGIME_ORDER if g in runs]
    fig, axes = plt.subplots(len(regs), 1, figsize=(10, 2.2 * len(regs)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, g in zip(axes, regs):
        st, stream = runs[g]['steps'], runs[g]['stream']
        ax.plot(st['step'], st['win_mean'], color=SERIES[0], lw=1.2, label='μ(W)')
        ax.plot(st['step'], st['win_std'], color=SERIES[1], lw=1.2, label='σ(W)')
        ax.axhline(frozen[0], color=SERIES[0], ls='--', lw=1, label=f'δμ = {frozen[0]:g}')
        ax.axhline(frozen[1], color=SERIES[1], ls='--', lw=1, label=f'δσ = {frozen[1]:g}')
        for s in st.loc[st['consolidated'].astype(bool), 'step']:
            ax.axvline(s, color=SERIES[2], lw=0.9, alpha=0.8, zorder=0)
        pk = st.loc[st['peak_fired'].astype(bool)]
        ax.scatter(pk['step'], np.full(len(pk), ax.get_ylim()[1] * 0.97), marker='v', s=16, color=SERIES[6],
                   zorder=3, label='peak (re-arm)')
        for b in task_boundaries(stream):
            ax.axvline(b, color=MUTED, lw=0.4, alpha=0.3, zorder=0)
        ax.set_yscale('symlog', linthresh=0.1)
        ax.set_title(f"{REGIME_LABEL.get(g, g)} — {int(st['consolidated'].sum())} consolidations (green lines)",
                     loc='left')
    axes[0].legend(ncol=5, loc='upper right', fontsize=8)
    axes[-1].set_xlabel('stream step (thin gray: ground-truth dominant-task changes)')
    fig.suptitle('Fig. 1 — Plateau detector against its thresholds (seed 1)', x=0.01, ha='left', fontweight='bold')
    savefig(fig, path)


def fig_alpha(runs, path):
    regs = [g for g in REGIME_ORDER if g in runs and 'alpha_max' in runs[g]['stream']]
    if not regs:
        return
    fig, axes = plt.subplots(len(regs), 1, figsize=(10, 1.6 * len(regs)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, g in zip(axes, regs):
        st, stream = runs[g]['steps'], runs[g]['stream']
        ax.fill_between(stream['step'], stream['alpha_max'], color=SERIES[0], alpha=0.25, lw=0)
        ax.plot(stream['step'], stream['alpha_max'], color=SERIES[0], lw=1.2)
        for s in st.loc[st['consolidated'].astype(bool), 'step']:
            ax.axvline(s, color=SERIES[2], lw=0.9, alpha=0.8)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel('max αₖ')
        ax.set_title(REGIME_LABEL.get(g, g), loc='left')
    axes[-1].set_xlabel('stream step (green: consolidations)')
    fig.suptitle('Fig. 2 — Ground-truth mixing curve with consolidation times (seed 1)', x=0.01, ha='left',
                 fontweight='bold')
    savefig(fig, path)


def fig_penalty(runs, path):
    regs = [g for g in REGIME_ORDER if g in runs]
    fig, axes = plt.subplots(len(regs), 2, figsize=(10, 1.8 * len(regs)), sharex=True)
    axes = np.atleast_2d(axes)
    for (a1, a2), g in zip(axes, regs):
        st = runs[g]['steps']
        a1.plot(st['step'], st['penalty_value'], color=SERIES[0], lw=1.2)
        a1.set_title(f'{REGIME_LABEL.get(g, g)}: penalty λ/2·ΣΩ(θ−θ*)²', loc='left')
        a2.plot(st['step'], st['omega_l1'], color=SERIES[1], lw=1.2)
        a2.set_title(f'{REGIME_LABEL.get(g, g)}: ‖Ω‖₁', loc='left')
    for a in axes[-1]:
        a.set_xlabel('stream step')
    fig.suptitle('Fig. 3 — Is MAS* regularizing at all? (seed 1)', x=0.01, ha='left', fontweight='bold')
    savefig(fig, path)


def fig_blocking(runs, path, window=50):
    regs = [g for g in REGIME_ORDER if g in runs]
    fig, axes = plt.subplots(len(regs), 1, figsize=(10, 1.7 * len(regs)), sharex=True)
    axes = np.atleast_1d(axes)
    labels = ['fired', 'latch only', 'mean only', 'std only', 'buffer only', 'multiple']
    colors = [SERIES[2], SERIES[6], SERIES[0], SERIES[1], MUTED, SERIES[4]]
    for ax, g in zip(axes, regs):
        st = runs[g]['steps']
        f = pd.DataFrame({'latch': ~st['cond_latch'].astype(bool), 'mean': ~st['cond_mean'].astype(bool),
                          'std': ~st['cond_std'].astype(bool), 'buffer': ~st['cond_buffer'].astype(bool)})
        n = f.sum(axis=1)
        cats = pd.DataFrame({
            'fired': st['consolidated'].astype(bool),
            'latch only': (n == 1) & f['latch'], 'mean only': (n == 1) & f['mean'],
            'std only': (n == 1) & f['std'], 'buffer only': (n == 1) & f['buffer'], 'multiple': n > 1,
        }).astype(float).rolling(window, min_periods=1).mean()
        ax.stackplot(st['step'], [cats[c] for c in labels], colors=colors, labels=labels, lw=0)
        ax.set_ylim(0, 1)
        ax.set_title(REGIME_LABEL.get(g, g), loc='left')
    axes[0].legend(ncol=6, loc='upper right', fontsize=8)
    axes[-1].set_xlabel(f'stream step ({window}-step rolling share of steps)')
    fig.suptitle('Fig. 4 — What blocked consolidation (seed 1)', x=0.01, ha='left', fontweight='bold')
    savefig(fig, path)


def fig_money(scal, res, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    s = scal[scal['regime'].isin(SIGMAS)].copy()
    s['sigma'] = s['regime'].map(SIGMAS)
    agg = s.groupby('sigma')['N_c'].agg(['mean', 'min', 'max'])
    a1.errorbar(agg.index, agg['mean'], yerr=[agg['mean'] - agg['min'], agg['max'] - agg['mean']], color=SERIES[0],
                marker='o', ms=6, lw=2, capsize=3)
    a1.set_xticks(list(SIGMAS.values()))
    a1.set_xlabel('σ (larger = smoother transitions)')
    a1.set_title('Consolidations per run (tfmas_star, mean, min–max over seeds)', loc='left')
    r = res[res['regime'].isin(SIGMAS)].copy()
    r['sigma'] = r['regime'].map(SIGMAS)
    for arm in ARM_ORDER:
        d = r[r['arm'] == arm].groupby('sigma')['AAUC'].agg(['mean', 'std'])
        if len(d):
            a2.errorbar(d.index, d['mean'], yerr=d['std'].fillna(0), color=ARM_COLOR[arm], marker='o', ms=6, lw=2,
                        capsize=3, label=arm)
    for (name, vals), c in zip(PUBLISHED.items(), [ARM_COLOR['tfmas (DRIFT legacy)'], ARM_COLOR['bare']]):
        a2.plot(list(vals), list(vals.values()), ls=':', marker='o', ms=7, mfc='none', color=c, lw=1.2, label=name)
    a2.set_xticks(list(SIGMAS.values()))
    a2.set_xlabel('σ')
    a2.set_title('AAUC (%) — this env (filled) vs published (hollow)', loc='left')
    a2.legend(fontsize=7.5, loc='best')
    fig.suptitle('Fig. 5 — Does consolidation frequency track performance across σ?', x=0.01, ha='left',
                 fontweight='bold')
    savefig(fig, path)


def fig_arms(res, path):
    regs = [g for g in REGIME_ORDER if g in set(res['regime'])]
    arms = [a for a in ARM_ORDER if a in set(res['arm'])]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    width = 0.8 / max(len(arms), 1)
    for i, arm in enumerate(arms):
        for j, g in enumerate(regs):
            v = res[(res['arm'] == arm) & (res['regime'] == g)]['AAUC']
            if not len(v):
                continue
            x = j + (i - (len(arms) - 1) / 2) * width
            ax.bar(x, v.mean(), width=width - 0.03, color=ARM_COLOR[arm], label=arm if j == 0 else None, zorder=2)
            ax.scatter(np.full(len(v), x), v, s=10, color=TEXT2, zorder=3)
    ax.set_xticks(range(len(regs)))
    ax.set_xticklabels([REGIME_LABEL.get(g, g) for g in regs])
    ax.set_ylabel('AAUC (%)')
    ax.legend(ncol=4, fontsize=8, loc='upper right')
    ax.grid(axis='x', visible=False)
    fig.suptitle('Fig. 6 — Consolidation vs buffer vs legacy vs bare (bars: mean; dots: seeds)', x=0.01, ha='left',
                 fontweight='bold')
    savefig(fig, path)


def fig_sweep(scal, res, path):
    s = scal[(scal['seed'] == 0) & (scal['l_th'] >= 0)]
    if s.empty:
        return
    regs = [g for g in REGIME_ORDER if g in set(s['regime'])]
    fig, axes = plt.subplots(1, 2 * len(regs), figsize=(4.2 * len(regs), 3.4))
    for k, g in enumerate(regs):
        d = s[s['regime'] == g]
        for m, (metric, title) in enumerate([('N_c', 'consolidations'), ('pct_penalty_zero', '% steps penalty≈0')]):
            ax = axes[2 * k + m]
            piv = d.pivot_table(index='std_th', columns='l_th', values=metric)
            im = ax.imshow(piv.to_numpy(), origin='lower', cmap='Blues' if m == 0 else 'Greys', aspect='auto')
            for (i, j), v in np.ndenumerate(piv.to_numpy()):
                ax.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=7,
                        color='white' if v > np.nanmax(piv.to_numpy()) * 0.6 else TEXT2)
            ax.set_xticks(range(piv.shape[1]))
            ax.set_xticklabels([f'{c:.3g}' for c in piv.columns], rotation=45, fontsize=7)
            ax.set_yticks(range(piv.shape[0]))
            ax.set_yticklabels([f'{c:.3g}' for c in piv.index], fontsize=7)
            ax.set_xlabel('δμ')
            ax.set_ylabel('δσ')
            ax.set_title(f'{REGIME_LABEL.get(g, g)}: {title}', loc='left', fontsize=9)
            ax.grid(False)
    fig.suptitle('Fig. 7 — T3 threshold sweep (seed 0)', x=0.01, ha='left', fontweight='bold')
    savefig(fig, path)


# ---------------------------------------------------------------- main

def fmt(v, nd=1):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return '–'
    return f'{v:.{nd}f}' if isinstance(v, float) else str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frozen', help='"l_th std_th" selected in T3')
    a = ap.parse_args()
    frozen = tuple(float(x) for x in a.frozen.split()) if a.frozen else None
    style()

    res = load_results()
    scal_rows, telem = [], {}
    for r in run_dirs():
        steps = load(r['path'], 'steps')
        if steps is None or not r['run'].startswith('tfmas_star'):
            continue
        stream, events = load(r['path'], 'stream'), load(r['path'], 'events')
        m = re.match(r'tfmas_star_lth(-?[\d.e+-]+)_sth(-?[\d.e+-]+)', r['run'])
        l_th, std_th = float(m.group(1)), float(m.group(2))
        row = {'regime': r['regime'], 'seed': r['seed'], 'l_th': l_th, 'std_th': std_th}
        row.update(run_scalars(steps, stream, events))
        scal_rows.append(row)
        telem[(r['regime'], r['seed'], l_th, std_th)] = {'steps': steps, 'stream': stream, 'events': events}
    scal = pd.DataFrame(scal_rows)
    lines = ['# MAS* on DRIFT — generated tables', '',
             'Metrics recomputed from `results/*.pkl`: AAUC = mean over evaluation checkpoints of average accuracy '
             'on seen tasks; AA_final = last checkpoint; FM = mean over tasks of (final − best) accuracy. '
             'Non-Gaussian pipelines add evaluation checkpoints at task changes, so AAUC is comparable within a '
             'regime, not across regime families.', '']

    # T3 sweep table
    sweep = scal[(scal['seed'] == 0) & (scal['l_th'] >= 0)] if len(scal) else scal
    if len(sweep):
        rs = res[(res['seed'] == 0) & (res['method'] == 'tfmas_star')][['regime', 'l_th', 'std_th', 'AAUC']]
        sw = sweep.merge(rs, on=['regime', 'l_th', 'std_th'], how='left')
        lines += ['## T3 threshold sweep (seed 0)', '',
                  '| regime | δμ | δσ | N_c | first | % segments w/ consolidation | % penalty≈0 | peaks | AAUC |',
                  '|---|---|---|---|---|---|---|---|---|']
        for _, x in sw.sort_values(['regime', 'l_th', 'std_th']).iterrows():
            lines.append(f"| {REGIME_LABEL.get(x.regime, x.regime)} | {x.l_th:g} | {x.std_th:g} | {x.N_c} | "
                         f"{fmt(x.first_consolidation)} | {fmt(x.pct_segments_with_consolidation)} | "
                         f"{fmt(x.pct_penalty_zero)} | {x.n_peaks} | {fmt(x.AAUC)} |")
        lines.append('')
        fig_sweep(scal, res, os.path.join(FIG, 'fig7_threshold_sweep.png'))

    if frozen is not None:
        main_s = scal[(scal['l_th'] == frozen[0]) & (scal['std_th'] == frozen[1]) & (scal['seed'] > 0)]
        if len(main_s):
            lines += [f'## Per-run detector scalars (tfmas_star, δμ={frozen[0]:g}, δσ={frozen[1]:g})', '',
                      '| regime | seed | N_c | first | interval med (IQR) | % latch armed | blocked: latch / mean / std '
                      '/ multiple (%) | peaks | penalty mean | % penalty≈0 | ‖Ω‖₁ final | Ω cos med | % segments |',
                      '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
            for _, x in main_s.sort_values(['regime', 'seed'], key=lambda c: c.map(
                    {g: i for i, g in enumerate(REGIME_ORDER)}) if c.name == 'regime' else c).iterrows():
                lines.append(
                    f"| {REGIME_LABEL.get(x.regime, x.regime)} | {x.seed} | {x.N_c} | {fmt(x.first_consolidation)} | "
                    f"{fmt(x.interval_median, 0)} ({fmt(x.interval_IQR, 0)}) | {fmt(x.pct_latch_armed)} | "
                    f"{fmt(x.block_latch_only)} / {fmt(x.block_mean_only)} / {fmt(x.block_std_only)} / "
                    f"{fmt(x.block_multiple)} | {x.n_peaks} | {x.penalty_mean:.3g} | {fmt(x.pct_penalty_zero)} | "
                    f"{x.omega_l1_final:.3g} | {fmt(x.omega_cos_median, 2)} | "
                    f"{fmt(x.pct_segments_with_consolidation)} |")
            lines.append('')

            # consolidation lag + alignment
            lines += ['## Consolidation lag and alignment (tfmas_star, frozen thresholds, seeds pooled)', '',
                      'Lag = steps from a reference event to the next consolidation. References: ground-truth '
                      'dominant-task changes; for Gaussian regimes also each task\'s mixing peak (the definition the '
                      'spec asks for at σ=20, where no task dominates). Alignment: mean α-velocity Σ|Δα| at '
                      'consolidation steps vs random steps, one-sided permutation test (10k), lower = consolidates '
                      'in stable stretches.', '',
                      '| regime | ref | n refs | lag median | lag IQR | % refs never followed | α-vel at consol. | '
                      'α-vel random | p |', '|---|---|---|---|---|---|---|---|---|']
            for g in REGIME_ORDER:
                keys = [k for k in telem if k[0] == g and k[1] > 0 and (k[2], k[3]) == frozen]
                if not keys:
                    continue
                for ref_name in ('task change', 'task mixing peak'):
                    lags = []
                    for k in keys:
                        st, stream = telem[k]['steps'], telem[k]['stream']
                        if ref_name == 'task change':
                            ref = task_boundaries(stream)
                        elif 'alpha_max' in stream:
                            ref = task_centres(stream)
                        else:
                            ref = []
                        lags += consolidation_lags(st, stream, ref)
                    if not lags:
                        continue
                    lg = np.array(lags)
                    ok = lg[~np.isnan(lg)]
                    vel_cells = '– | – | –'
                    if ref_name == 'task change' and 'alpha_max' in telem[keys[0]]['stream']:
                        ev, allv = [], []
                        for k in keys:
                            st, stream = telem[k]['steps'], telem[k]['stream']
                            v = alpha_velocity(stream)
                            ev += list(v[st['consolidated'].astype(bool).to_numpy()])
                            allv.append(v)
                        if ev:
                            obs, null, p = permutation_test(np.array(ev), np.concatenate(allv))
                            vel_cells = f'{obs:.4f} | {null:.4f} | {p:.4f}'
                    lines.append(f"| {REGIME_LABEL.get(g, g)} | {ref_name} | {len(lg)} | "
                                 f"{fmt(float(np.median(ok)) if len(ok) else math.nan, 0)} | "
                                 f"{fmt(float(np.subtract(*np.percentile(ok, [75, 25]))) if len(ok) else math.nan, 0)}"
                                 f" | {100 * np.isnan(lg).mean():.0f} | {vel_cells} |")
            lines.append('')

            seed1 = {k[0]: telem[k] for k in telem if k[1] == 1 and (k[2], k[3]) == frozen}
            if seed1:
                fig_detector(seed1, frozen, os.path.join(FIG, 'fig1_detector.png'))
                fig_alpha(seed1, os.path.join(FIG, 'fig2_alpha.png'))
                fig_penalty(seed1, os.path.join(FIG, 'fig3_penalty_omega.png'))
                fig_blocking(seed1, os.path.join(FIG, 'fig4_blocking.png'))
            fig_money(main_s, res[(res['seed'] > 0) & ((res['method'] != 'tfmas_star') |
                                                       ((res['l_th'] == frozen[0]) & (res['std_th'] == frozen[1])) |
                                                       (res['l_th'] < 0))], os.path.join(FIG, 'fig5_money.png'))

        # arm comparison
        arms = res[(res['seed'] > 0) & ((res['method'] != 'tfmas_star') | (res['l_th'] < 0) |
                                        ((res['l_th'] == frozen[0]) & (res['std_th'] == frozen[1])))]
        arms = arms[arms['extra'] == '']
        if len(arms):
            lines += ['## Arm comparison (seeds 1–3; mean ± std)', '',
                      '| regime | ' + ' | '.join(a for a in ARM_ORDER) + ' |', '|---|' + '---|' * len(ARM_ORDER)]
            for metric in ('AAUC', 'AA_final', 'FM'):
                lines.append(f'| **{metric}** |' + ' |' * len(ARM_ORDER))
                for g in REGIME_ORDER:
                    cells = []
                    for arm in ARM_ORDER:
                        v = arms[(arms['regime'] == g) & (arms['arm'] == arm)][metric]
                        cells.append(f'{v.mean():.1f} ± {v.std(ddof=1):.1f} (n={len(v)})' if len(v) > 1 else
                                     (f'{v.mean():.1f} (n=1)' if len(v) else '–'))
                    if any(c != '–' for c in cells):
                        lines.append(f'| {REGIME_LABEL.get(g, g)} | ' + ' | '.join(cells) + ' |')
            lines.append('')
            fig_arms(arms, os.path.join(FIG, 'fig6_arms.png'))

        # robustness slice
        rob = res[(res['seed'] == 1) & (res['method'] == 'tfmas_star') & (res['l_th'] >= 0) &
                  res['regime'].isin(SIGMAS)]
        if len(rob) and len(scal):
            rob = rob.merge(scal[scal['seed'] == 1][['regime', 'l_th', 'std_th', 'N_c', 'pct_penalty_zero']],
                            on=['regime', 'l_th', 'std_th'], how='left')
            lines += ['## Threshold robustness (seed 1)', '', '| regime | δμ | δσ | N_c | % penalty≈0 | AAUC | frozen |',
                      '|---|---|---|---|---|---|---|']
            for _, x in rob.sort_values(['regime', 'l_th', 'std_th']).iterrows():
                is_f = '←' if (x.l_th, x.std_th) == frozen else ''
                lines.append(f"| {REGIME_LABEL.get(x.regime, x.regime)} | {x.l_th:g} | {x.std_th:g} | {fmt(x.N_c)} | "
                             f"{fmt(x.pct_penalty_zero)} | {x.AAUC:.1f} | {is_f} |")
            lines.append('')

    # flag runs that measured the buffer baseline instead of MAS*
    if len(scal):
        bad = scal[(scal['l_th'] >= 0) & ((scal['N_c'] == 0) | (scal['pct_penalty_zero'] > 99.9))]
        lines += ['## Runs where MAS* never regularized (N_c = 0 or penalty ≈ 0 throughout)', '']
        if len(bad):
            for _, x in bad.iterrows():
                lines.append(f"- {REGIME_LABEL.get(x.regime, x.regime)}, seed {x.seed}, δμ={x.l_th:g}, δσ={x.std_th:g}: "
                             f"N_c={x.N_c}, penalty≈0 on {x.pct_penalty_zero:.1f}% of steps — this run measured the "
                             'buffer baseline, not MAS*')
        else:
            lines.append('- none')
        lines.append('')

    with open(os.path.join(ROOT, 'analysis', 'report_tables.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    sys.stdout.reconfigure(encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
