# Gate 0: reproduction against DRIFT Table 2 (Gaussian mixing)

Seeds 1–3 in the `drift-cpu` env (CPU, DGL 1.1.2). Criterion: |Δ| ≤ 2·SE of the difference, SE = √(sd²/n + sd_paper²/3). Metrics recomputed from `results/*.pkl` exactly as `metrics.tf_metrics`.

## CoraFull-CL, sigma20.0

| method | n | A_AUC ours | A_AUC paper | Δ | z | verdict | AF_s ours | AF_s paper | Δ | z | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Bare | 3 | 24.8 ± 1.4 | 21.9 ± 0.8 | +2.9 | +3.1 | ABOVE paper | -53.6 ± 3.7 | -60.5 ± 7.3 | +6.9 | +1.4 | within noise |
| ER | 3 | 34.7 ± 0.0 | 27.8 ± 0.6 | +6.9 | +19.8 | ABOVE paper | -40.5 ± 4.3 | -48.0 ± 5.2 | +7.5 | +1.9 | within noise |
| A-GEM | 3 | 32.4 ± 1.7 | 29.9 ± 2.8 | +2.5 | +1.3 | within noise | -50.2 ± 3.7 | -53.9 ± 4.8 | +3.7 | +1.0 | within noise |
| MAS* (DRIFT tfmas) | 3 | 30.6 ± 2.6 | 29.8 ± 2.2 | +0.8 | +0.4 | within noise | -44.8 ± 12.5 | -44.8 ± 8.0 | +0.0 | +0.0 | within noise |

## Arxiv-CL, sigma60.0

| method | n | A_AUC ours | A_AUC paper | Δ | z | verdict | AF_s ours | AF_s paper | Δ | z | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Bare | 3 | 22.1 ± 1.1 | 18.5 ± 1.5 | +3.6 | +3.4 | ABOVE paper | -61.5 ± 1.2 | -65.4 ± 3.1 | +3.9 | +2.0 | ABOVE paper |
| ER | 3 | 35.6 ± 1.3 | 34.9 ± 0.8 | +0.7 | +0.8 | within noise | -37.4 ± 2.1 | -37.3 ± 2.8 | -0.1 | -0.1 | within noise |
| A-GEM | 3 | 35.0 ± 0.8 | 34.1 ± 1.4 | +0.9 | +1.0 | within noise | -48.9 ± 3.3 | -48.6 ± 4.4 | -0.3 | -0.1 | within noise |
| MAS* (DRIFT tfmas) | 3 | 43.5 ± 1.6 | 38.4 ± 2.1 | +5.1 | +3.4 | ABOVE paper | -16.6 ± 6.9 | -22.2 ± 1.8 | +5.6 | +1.4 | within noise |

