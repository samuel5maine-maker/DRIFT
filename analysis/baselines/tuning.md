# Seed-0 hyperparameter sweeps (protocol t2)

Selection: highest A_AUC on seed 0 per method and dataset (seed 0 is not an evaluation seed). Single-seed selection is noisy; the full sweep is shown so the choice can be judged.

## CoraFull-CL — DER (2 configs)

| rank | hyperparameters | A_AUC | AF_s |
|---|---|---|---|
| 1 ← | alpha=0.5 | 32.9 | -36.9 |
| 2 | alpha=1 | 32.0 | -39.6 |

## CoraFull-CL — DER++ (4 configs)

| rank | hyperparameters | A_AUC | AF_s |
|---|---|---|---|
| 1 ← | alpha=0.5, beta=1 | 35.0 | -40.1 |
| 2 | alpha=0.2, beta=1 | 34.6 | -36.9 |
| 3 | alpha=0.2, beta=0.5 | 32.6 | -45.0 |
| 4 | alpha=0.5, beta=0.5 | 31.9 | -44.2 |

## CoraFull-CL — LwF-online (27 configs)

| rank | hyperparameters | A_AUC | AF_s |
|---|---|---|---|
| 1 ← | T=2, lambda_dist=10, update_every=100 | 39.8 | -23.1 |
| 2 | T=20, lambda_dist=10, update_every=100 | 38.4 | -30.4 |
| 3 | T=2, lambda_dist=10, update_every=10 | 35.8 | -40.3 |
| 4 | T=2, lambda_dist=1, update_every=100 | 35.6 | -33.5 |
| 5 | T=20, lambda_dist=1, update_every=100 | 31.2 | -48.6 |
| 6 | T=20, lambda_dist=10, update_every=10 | 31.0 | -50.2 |
| 7 | T=0.2, lambda_dist=1, update_every=100 | 30.9 | -25.9 |
| 8 | T=2, lambda_dist=0.1, update_every=100 | 29.8 | -48.9 |
| 9 | T=0.2, lambda_dist=10, update_every=10 | 29.2 | -28.1 |
| 10 | T=2, lambda_dist=1, update_every=10 | 28.9 | -55.4 |
| 11 | T=0.2, lambda_dist=0.1, update_every=100 | 27.7 | -44.1 |
| 12 | T=20, lambda_dist=1, update_every=10 | 27.7 | -57.3 |
| 13 | T=0.2, lambda_dist=10, update_every=100 | 25.6 | -12.6 |
| 14 | T=0.2, lambda_dist=0.1, update_every=10 | 25.4 | -60.2 |
| 15 | T=20, lambda_dist=10, update_every=1 | 25.0 | -59.7 |
| 16 | T=2, lambda_dist=0.1, update_every=10 | 25.0 | -60.5 |
| 17 | T=20, lambda_dist=0.1, update_every=100 | 24.3 | -54.6 |
| 18 | T=0.2, lambda_dist=0.1, update_every=1 | 24.0 | -61.8 |
| 19 | T=0.2, lambda_dist=1, update_every=10 | 23.9 | -55.4 |
| 20 | T=20, lambda_dist=0.1, update_every=10 | 23.8 | -59.6 |
| 21 | T=20, lambda_dist=0.1, update_every=1 | 23.5 | -65.7 |
| 22 | T=0.2, lambda_dist=10, update_every=1 | 23.2 | -60.3 |
| 23 | T=2, lambda_dist=1, update_every=1 | 23.1 | -59.4 |
| 24 | T=2, lambda_dist=0.1, update_every=1 | 22.2 | -58.9 |
| 25 | T=0.2, lambda_dist=1, update_every=1 | 21.9 | -60.3 |
| 26 | T=20, lambda_dist=1, update_every=1 | 21.3 | -57.9 |
| 27 | T=2, lambda_dist=10, update_every=1 | 20.7 | -64.8 |

## CoraFull-CL — CLS-ER (6 configs)

| rank | hyperparameters | A_AUC | AF_s |
|---|---|---|---|
| 1 ← | plastic_alpha=0.95, plastic_update_freq=1, reg_weight=0.1, stable_alpha=0.99444, stable_update_freq=0.9 | 37.5 | -27.2 |
| 2 | plastic_alpha=0.95, plastic_update_freq=1, reg_weight=1.25, stable_alpha=0.99444, stable_update_freq=0.9 | 36.7 | -19.7 |
| 3 | plastic_alpha=0.95, plastic_update_freq=1, reg_weight=1.25, stable_alpha=0.98148, stable_update_freq=0.9 | 36.6 | -32.2 |
| 4 | plastic_alpha=0.999, plastic_update_freq=0.9, reg_weight=0.1, stable_alpha=0.999, stable_update_freq=0.7 | 34.7 | -4.5 |
| 5 | plastic_alpha=0.95, plastic_update_freq=1, reg_weight=0.1, stable_alpha=0.98148, stable_update_freq=0.9 | 34.5 | -37.4 |
| 6 | plastic_alpha=0.99, plastic_update_freq=1, reg_weight=1.25, stable_alpha=0.99, stable_update_freq=0.9 | 33.7 | -28.1 |

