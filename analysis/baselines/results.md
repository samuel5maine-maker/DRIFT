# Missing continual-learning baselines on DRIFT — results (protocol t2)

Mean ± std over seeds 1–3, CPU (DGL 1.1.2), fixed OMP_NUM_THREADS=2. Metrics recomputed from `results_t2/*.pkl` exactly as `metrics.tf_metrics`: A_AUC ↑ and AF_s = mean_k(final − peak) ↑ (≤ 0). **AF_s can be improved by underfitting (a task never learned has no peak to fall from): read it only with A_AUC.**

source: **paper** = DRIFT Table 2; **reproduced** = DRIFT method re-run here; **new** = added here. Paper numbers come from a different environment and are not directly comparable to rows run here (see NOTES.md, Gate 0).

## CoraFull-CL, Gaussian mixing, sigma20.0

| Method | A_AUC ↑ | AF_s ↑ | source |
|--------|---------|--------|--------|
| Bare | 21.9 ± 0.8 | -60.5 ± 7.3 | paper |
| Joint | 86.3 ± 0.1 | – | paper |
| A-GEM | 29.9 ± 2.8 | -53.9 ± 4.8 | paper |
| ER | 27.8 ± 0.6 | -48.0 ± 5.2 | paper |
| GSS | 29.3 ± 1.4 | -55.3 ± 0.1 | paper |
| MAS* | 29.8 ± 2.2 | -44.8 ± 8.0 | paper |
| SSM | 25.4 ± 2.2 | -58.4 ± 3.8 | paper |
| SEM | 27.5 ± 1.9 | -58.4 ± 2.6 | paper |
| DMSG | 34.1 ± 1.0 | -37.6 ± 3.5 | paper |
| A-GEM | 32.1 ± 0.7 (n=3) | -54.5 ± 2.7 | reproduced |
| Bare | 24.2 ± 0.4 (n=3) | -56.3 ± 7.3 | reproduced |
| DMSG | 34.6 ± 0.7 (n=3) | -32.8 ± 4.4 | reproduced |
| ER | 32.8 ± 1.0 (n=3) | -42.4 ± 2.0 | reproduced |
| ER-CBRS | 33.3 ± 1.0 (n=3) | -47.3 ± 5.4 | new |
| MAS* | 29.7 ± 0.6 (n=3) | -47.0 ± 8.7 | reproduced |

## Arxiv-CL, Gaussian mixing, sigma60.0

| Method | A_AUC ↑ | AF_s ↑ | source |
|--------|---------|--------|--------|
| Bare | 18.5 ± 1.5 | -65.4 ± 3.1 | paper |
| Joint | 71.6 ± 1.4 | – | paper |
| A-GEM | 34.1 ± 1.4 | -48.6 ± 4.4 | paper |
| ER | 34.9 ± 0.8 | -37.3 ± 2.8 | paper |
| GSS | 24.2 ± 3.2 | -60.4 ± 5.6 | paper |
| MAS* | 38.4 ± 2.1 | -22.2 ± 1.8 | paper |
| SSM | 28.0 ± 4.3 | -64.5 ± 1.9 | paper |
| SEM | 26.8 ± 1.0 | -59.7 ± 6.9 | paper |
| DMSG | 31.6 ± 0.9 | -31.9 ± 2.9 | paper |
| A-GEM | 33.5 ± 1.6 (n=3) | -50.3 ± 6.2 | reproduced |
| Bare | 22.2 ± 1.8 (n=3) | -63.1 ± 4.9 | reproduced |
| DMSG | 32.3 ± 1.6 (n=3) | -29.5 ± 2.5 | reproduced |
| ER | 35.6 ± 1.7 (n=3) | -38.0 ± 0.8 | reproduced |
| ER-CBRS | 30.2 ± 1.2 (n=3) | -41.7 ± 3.2 | new |
| MAS* | 42.6 ± 1.1 (n=3) | -14.3 ± 2.1 | reproduced |

## Memory beyond the backbone (spec §2)

buffer_bytes: node id + label per slot, plus stored logits (DER family) or embeddings (PDGNN). extra params: parameter copies a method keeps (EMA models, teacher, importance weights, discriminator). Replay graphs are materialised from dataset.graph at replay time and are not counted.

| dataset | method | backbone | buffer bytes | extra params | extra param bytes | total bytes |
|---|---|---|---|---|---|---|
| Arxiv-CL | bare | GCN | 0 | 0 | 0 | 0 |
| Arxiv-CL | agem | GCN | 1,600 | 0 | 0 | 1,600 |
| Arxiv-CL | er | GCN | 1,600 | 0 | 0 | 1,600 |
| Arxiv-CL | er_cbrs | GCN | 1,600 | 0 | 0 | 1,600 |
| Arxiv-CL | dmsg | GCN | 1,600 | 16,513 | 66,052 | 67,652 |
| Arxiv-CL | tfmas | GCN | 0 | 86,016 | 344,064 | 344,064 |
| CoraFull-CL | bare | GCN | 0 | 0 | 0 | 0 |
| CoraFull-CL | agem | GCN | 1,600 | 0 | 0 | 1,600 |
| CoraFull-CL | er | GCN | 1,600 | 0 | 0 | 1,600 |
| CoraFull-CL | er_cbrs | GCN | 1,600 | 0 | 0 | 1,600 |
| CoraFull-CL | dmsg | GCN | 1,600 | 16,513 | 66,052 | 67,652 |
| CoraFull-CL | tfmas | GCN | 0 | 4,495,360 | 17,981,440 | 17,981,440 |

## §6.3 Coverage diagnostic: final per-class accuracy vs final buffer slots

Seeds 1–3 pooled; one point per (class, seed). Pearson r and Spearman ρ over all points; "acc | 0 slots" vs "acc | ≥1 slot" compares classes the memory missed with classes it covered.

| dataset | method | points | Pearson r | Spearman ρ | classes with 0 slots (mean per seed) | acc at 0 slots | acc at ≥1 slot |
|---|---|---|---|---|---|---|---|
| CoraFull-CL | ER | 210 | 0.38 | 0.42 | 23.7 | 27.6 | 52.0 |
| CoraFull-CL | ER-CBRS | 210 | 0.23 | 0.22 | 0.0 | nan | 44.5 |
| Arxiv-CL | ER | 120 | 0.52 | 0.59 | 8.0 | 13.2 | 51.3 |
| Arxiv-CL | ER-CBRS | 120 | 0.24 | 0.22 | 0.0 | nan | 48.3 |

![coverage](coverage_diagnostic.png)

