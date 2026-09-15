"""
Gate 0 (baselines spec §4): compare reproduced ER / A-GEM / MAS* / Bare against DRIFT Table 2.

"Within seed noise" is taken as |mean_ours - mean_paper| <= 2 * SE_diff, with
SE_diff = sqrt(sd_ours^2 / n_ours + sd_paper^2 / 3) (the paper reports mean ± std over 3 seeds).

    python analysis/gate0.py   -> analysis/gate0.md
"""
import math
import os
import sys

from report import load_results
from common import ROOT

# DRIFT arXiv:2605.12998v3, Table 2 (Gaussian mixing): (A_AUC mean, sd, AF_s mean, sd)
PAPER = {
    ('CoraFull-CL', 'gaussian_sigma20.0'): {'bare': (21.9, 0.8, -60.5, 7.3), 'er': (27.8, 0.6, -48.0, 5.2),
                                            'agem': (29.9, 2.8, -53.9, 4.8), 'tfmas': (29.8, 2.2, -44.8, 8.0)},
    ('Arxiv-CL', 'gaussian_sigma60.0'): {'bare': (18.5, 1.5, -65.4, 3.1), 'er': (34.9, 0.8, -37.3, 2.8),
                                         'agem': (34.1, 1.4, -48.6, 4.4), 'tfmas': (38.4, 2.1, -22.2, 1.8)},
}
LABEL = {'bare': 'Bare', 'er': 'ER', 'agem': 'A-GEM', 'tfmas': 'MAS* (DRIFT tfmas)'}


def verdict(m, s, n, pm, ps):
    if n < 2:
        return math.nan, 'n<2'
    se = math.sqrt(s ** 2 / n + ps ** 2 / 3)
    z = (m - pm) / se if se > 0 else math.inf
    return z, 'within noise' if abs(z) <= 2 else ('ABOVE paper' if z > 0 else 'BELOW paper')


def main():
    res = load_results()
    res = res[(res['seed'].isin([1, 2, 3])) & (res['extra'] == '')]
    lines = ['# Gate 0: reproduction against DRIFT Table 2 (Gaussian mixing)', '',
             'Seeds 1–3 in the `drift-cpu` env (CPU, DGL 1.1.2). Criterion: |Δ| ≤ 2·SE of the difference, '
             'SE = √(sd²/n + sd_paper²/3). Metrics recomputed from `results/*.pkl` exactly as `metrics.tf_metrics`.', '']
    for (dataset, regime), rows in PAPER.items():
        lines += [f'## {dataset}, {regime.replace("gaussian_", "")}', '',
                  '| method | n | A_AUC ours | A_AUC paper | Δ | z | verdict | AF_s ours | AF_s paper | Δ | z | verdict |',
                  '|---|---|---|---|---|---|---|---|---|---|---|---|']
        for method, (pa, pas, pf, pfs) in rows.items():
            d = res[(res['dataset'] == dataset) & (res['regime'] == regime) & (res['method'] == method)]
            n = len(d)
            if n == 0:
                lines.append(f'| {LABEL[method]} | 0 | – | {pa} ± {pas} | | | not run | – | {pf} ± {pfs} | | | |')
                continue
            a_m, a_s = d['AAUC'].mean(), d['AAUC'].std(ddof=1) if n > 1 else math.nan
            f_m, f_s = d['FM'].mean(), d['FM'].std(ddof=1) if n > 1 else math.nan
            za, va = verdict(a_m, a_s, n, pa, pas)
            zf, vf = verdict(f_m, f_s, n, pf, pfs)
            lines.append(f'| {LABEL[method]} | {n} | {a_m:.1f} ± {a_s:.1f} | {pa} ± {pas} | {a_m - pa:+.1f} | {za:+.1f} | '
                         f'{va} | {f_m:.1f} ± {f_s:.1f} | {pf} ± {pfs} | {f_m - pf:+.1f} | {zf:+.1f} | {vf} |')
        lines.append('')
    with open(os.path.join(ROOT, 'analysis', 'gate0.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    sys.stdout.reconfigure(encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
