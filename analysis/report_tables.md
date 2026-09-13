# MAS* on DRIFT — generated tables

Metrics recomputed from `results/*.pkl`: AAUC = mean over evaluation checkpoints of average accuracy on seen tasks; AA_final = last checkpoint; FM = mean over tasks of (final − best) accuracy. Non-Gaussian pipelines add evaluation checkpoints at task changes, so AAUC is comparable within a regime, not across regime families.

## T3 threshold sweep (seed 0)

| regime | δμ | δσ | N_c | first | % segments w/ consolidation | % penalty≈0 | peaks | AAUC |
|---|---|---|---|---|---|---|---|---|
| hard (tfocis) | 0.3779 | 0.06544 | 7 | 351.0 | 14.3 | 29.9 | 7 | 25.5 |
| hard (tfocis) | 0.3779 | 0.1317 | 9 | 349.0 | 17.1 | 29.9 | 9 | 22.2 |
| hard (tfocis) | 0.3779 | 0.2164 | 7 | 348.0 | 17.1 | 29.7 | 7 | 25.2 |
| hard (tfocis) | 0.3779 | 0.3732 | 7 | 348.0 | 17.1 | 29.7 | 7 | 25.2 |
| hard (tfocis) | 0.9404 | 0.06544 | 14 | 53.0 | 28.6 | 5.7 | 14 | 19.5 |
| hard (tfocis) | 0.9404 | 0.1317 | 18 | 51.0 | 34.3 | 5.8 | 18 | 18.9 |
| hard (tfocis) | 0.9404 | 0.2164 | 23 | 7.0 | 40.0 | 2.6 | 23 | 17.8 |
| hard (tfocis) | 0.9404 | 0.3732 | 24 | 7.0 | 45.7 | 2.7 | 24 | 18.6 |
| hard (tfocis) | 1.624 | 0.06544 | 13 | 37.0 | 28.6 | 4.2 | 13 | 17.3 |
| hard (tfocis) | 1.624 | 0.1317 | 22 | 34.0 | 45.7 | 4.8 | 22 | 19.6 |
| hard (tfocis) | 1.624 | 0.2164 | 21 | 5.0 | 51.4 | 2.2 | 21 | 18.8 |
| hard (tfocis) | 1.624 | 0.3732 | 22 | 2.0 | 57.1 | 2.1 | 22 | 18.5 |
| hard (tfocis) | 2.858 | 0.06544 | 13 | 37.0 | 28.6 | 4.2 | 13 | 20.1 |
| hard (tfocis) | 2.858 | 0.1317 | 23 | 30.0 | 51.4 | 4.5 | 23 | 18.3 |
| hard (tfocis) | 2.858 | 0.2164 | 22 | 5.0 | 60.0 | 2.3 | 22 | 17.2 |
| hard (tfocis) | 2.858 | 0.3732 | 26 | 2.0 | 65.7 | 2.4 | 26 | 17.7 |
| σ=20 | 0.3779 | 0.06544 | 0 | – | 0.0 | 100.0 | 0 | 28.1 |
| σ=20 | 0.3779 | 0.1317 | 0 | – | 0.0 | 100.0 | 0 | 26.1 |
| σ=20 | 0.3779 | 0.2164 | 0 | – | 0.0 | 100.0 | 0 | 26.1 |
| σ=20 | 0.3779 | 0.3732 | 0 | – | 0.0 | 100.0 | 0 | 26.1 |
| σ=20 | 0.9404 | 0.06544 | 4 | 541.0 | 5.7 | 45.5 | 4 | 30.8 |
| σ=20 | 0.9404 | 0.1317 | 6 | 541.0 | 11.4 | 45.7 | 6 | 32.4 |
| σ=20 | 0.9404 | 0.2164 | 6 | 541.0 | 11.4 | 45.7 | 6 | 32.1 |
| σ=20 | 0.9404 | 0.3732 | 6 | 541.0 | 11.4 | 45.7 | 6 | 32.1 |
| σ=20 | 1.624 | 0.06544 | 3 | 525.0 | 8.6 | 44.1 | 3 | 27.4 |
| σ=20 | 1.624 | 0.1317 | 13 | 20.0 | 25.7 | 2.8 | 13 | 30.4 |
| σ=20 | 1.624 | 0.2164 | 20 | 20.0 | 34.3 | 3.4 | 20 | 34.8 |
| σ=20 | 1.624 | 0.3732 | 22 | 20.0 | 40.0 | 3.6 | 22 | 31.2 |
| σ=20 | 2.858 | 0.06544 | 3 | 525.0 | 8.6 | 44.1 | 3 | 27.4 |
| σ=20 | 2.858 | 0.1317 | 16 | 12.0 | 31.4 | 2.4 | 16 | 34.7 |
| σ=20 | 2.858 | 0.2164 | 23 | 12.0 | 40.0 | 3.0 | 23 | 35.1 |
| σ=20 | 2.858 | 0.3732 | 37 | 8.0 | 68.6 | 3.8 | 37 | 34.3 |

## Runs where MAS* never regularized (N_c = 0 or penalty ≈ 0 throughout)

- σ=20, seed 0, δμ=0.3779, δσ=0.06544: N_c=0, penalty≈0 on 100.0% of steps — this run measured the buffer baseline, not MAS*
- σ=20, seed 0, δμ=0.3779, δσ=0.1317: N_c=0, penalty≈0 on 100.0% of steps — this run measured the buffer baseline, not MAS*
- σ=20, seed 0, δμ=0.3779, δσ=0.2164: N_c=0, penalty≈0 on 100.0% of steps — this run measured the buffer baseline, not MAS*
- σ=20, seed 0, δμ=0.3779, δσ=0.3732: N_c=0, penalty≈0 on 100.0% of steps — this run measured the buffer baseline, not MAS*

