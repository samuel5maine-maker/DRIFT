# Dual-dataset scientific continuation

**Branch:** `Codex/dual-dataset-scientific`

**Datasets:** CoraFull-CL and Arxiv-CL

**Goal:** exceed 42.75 A_AUC and improve AF_s above -13.46 without increasing the DRIFT replay budget or training passes

## Result

The accuracy target was beaten on CoraFull, but no frozen configuration beat both targets on both datasets.

| Dataset and frozen configuration | Selection result | Test seeds | Test A_AUC | Test AF_s | Decision |
|---|---|---:|---:|---:|---|
| CoraFull H23, sigmas 3/10/20 | 43.52 / -13.16 | 1--3 | **44.25** | **-14.00** | Accuracy target met; retention target missed |
| Arxiv H26 MAS, sigma 60 | 45.02 / -7.90 | 1--3 | **42.23** | **-17.03** | Both targets missed |

These test results are final and unbiased with respect to this experiment cycle. Later hypotheses returned to validation after the test results were known and are reported as research candidates only. They were not evaluated again on test.

## Fixed fairness conditions

All experiments retained the DRIFT online protocol:

- one pass through the stream;
- one optimizer update per incoming minibatch;
- minibatch size 10;
- latent task identity and transition weights withheld from the learner;
- no validation or test labels used for training or inference;
- CoraFull replay limited to 100 node-label pairs and 10 replay seeds per 10 incoming nodes;
- Arxiv MAS used no replay buffer;
- the same two-layer GCN, Adam learning rate 0.005, and neighborhood sampler `[10, 25]`;
- model selection on validation before the first test run;
- added state, graph work, and evaluation forwards recorded explicitly.

H23 stores at most 560 bytes of class last-seen counters in addition to CBRS-100 and one EMA parameter copy. Its inference uses one extra GCN forward on each evaluation graph to combine 12.5% working logits with 87.5% EMA logits only for classes delivered during the previous 100 updates. Old-class columns remain pure EMA. It uses delivered labels only and receives no boundary or task identifier.

Arxiv MAS stores parameter importance and anchor tensors but no examples. Its task-free loss-plateau detector controls consolidation. H26 is the repository's existing MAS rule wrapped only for uniform measurement; it is not claimed as a new algorithm.

The protocol follows the comparison constraints described by the [DRIFT paper](https://arxiv.org/pdf/2605.12998). Dataset-specific hyperparameters were selected on validation, as in the repository's existing baseline study.

## Observations carried forward from H1--H15

1. Graph-context replay was causally useful on CoraFull: it improved A_AUC by 2.69 points with the same stored nodes, replay seeds, and optimizer steps.
2. CBRS added 0.72 point after graph context was restored.
3. Fixed working-classifier row norms added 3.23 points, the largest first-cycle gain.
4. Final CoraFull accuracy was much higher than checkpoint-average accuracy, identifying slow acquisition as the primary A_AUC bottleneck.
5. Larger replay neighborhoods and degree-selected hubs increased graph work without improving A_AUC.
6. The retained Cora model did not transfer directly to Arxiv. Dataset scale changed the stability timescale, and even a corrected EMA remained acquisition-limited.

## Second-cycle hypotheses

Results through H23 use CoraFull validation seed 0 and equal means over sigmas 3, 10, and 20. Arxiv results use validation seed 0 at sigma 60. “Control” is the ordinary inference output from the same training run where available.

| ID | Hypothesis | Candidate A_AUC | Candidate AF_s | Decision and observation |
|---|---|---:|---:|---|
| H16 | Double only the projected classifier learning rate | 41.85 | -16.21 | Reject. Faster angular motion caused late instability. |
| H17 | Combine H4 with overlap-adaptive EMA | 42.76 | -14.18 | Reject. Marginal acquisition gain traded away retention. |
| H18 | Blend 25% working logits globally with slow EMA | 44.18 | -16.44 | Reject. Higher intermediate peaks were not preserved. |
| H19 | Hard-sync a classifier row into EMA on first class observation | 41.49 | -11.78 | Reject. Retention improved, but working-head/EMA-encoder mismatch hurt acquisition. |
| H20 | Use a 5% first-observation row interpolation | 42.40 | -12.91 | Reject for A_AUC. Partial synchronization retained the same tradeoff. |
| H21 | Normalize EMA classifier rows as well as working rows | 41.48 | -11.89 | Reject. EMA norm shrinkage carries useful class-age information. |
| H22 | Apply 25% working blend only to recently delivered classes | 43.75 | -13.75 | Reject narrowly for retention. Recency gating localized most of the damage. |
| H23 | Halve the gated working contribution to 12.5% | **43.52** | **-13.16** | Accept on selection; freeze for CoraFull test. |
| H23-Arxiv | Transfer H23 unchanged to Arxiv | 28.81 | -40.82 | Reject. Cora EMA timescale is too fast relative to Arxiv task spans. |
| H24 | Scale Arxiv EMA decay to alpha 0.99966 | 30.16 EMA / 32.34 gated | -13.41 / -22.49 | Retention hypothesis accepted, overall candidate rejected for acquisition. |
| H25 | Apply fixed classifier norms to Arxiv MAS | 16.80 | -79.55 | Reject decisively. MAS depends on unconstrained classifier magnitude. |
| H26 | Measure unmodified MAS on regenerated validation split | **45.02** | **-7.90** | Freeze as Arxiv configuration. |
| H27 | Combine H20 retention with H23 acquisition | 43.54 | -13.12 | Seed-0 success; multi-seed validation not robust. |
| H28 | Evaluate Arxiv MAS through slow EMA | 34.48 | -4.98 | Reject for acquisition. |
| H29 | Blend 75% working MAS with 25% slow EMA | 42.46 | -22.27 | Reject. Logit blending introduced peak/final mismatch. |

## Frozen CoraFull test

H23 was frozen after validation seed 0 and evaluated on learner seeds 1--3 at every requested sigma.

### Aggregate by learner seed

| Seed | A_AUC | AF_s |
|---:|---:|---:|
| 1 | 43.52 | -15.08 |
| 2 | 44.37 | -13.60 |
| 3 | 44.86 | -13.33 |
| **Mean** | **44.25** | **-14.00** |

### Aggregate by Gaussian width

| Sigma | A_AUC | AF_s |
|---:|---:|---:|
| 3 | 43.61 | -13.80 |
| 10 | 43.52 | -15.55 |
| 20 | 45.62 | -12.67 |

Mean final accuracy was 74.02%. H23 robustly improves checkpoint-average accuracy, but forgetting varies by seed and is worst at sigma 10. The final aggregate misses the AF_s target by 0.54 point.

## Frozen Arxiv test

H26 was frozen after its 45.02/-7.90 validation result and evaluated at sigma 60.

| Seed | A_AUC | AF_s | Final accuracy | MAS consolidations |
|---:|---:|---:|---:|---:|
| 1 | 42.21 | -19.87 | 71.84 | 161 |
| 2 | 42.35 | -17.94 | 76.67 | 134 |
| 3 | 42.13 | -13.28 | 79.97 | 135 |
| **Mean** | **42.23** | **-17.03** | **76.16** | **143.3** |

MAS is sensitive to the training seed and number/timing of loss-plateau consolidations. Its single validation run was optimistic. The three-seed test mean misses A_AUC by 0.52 point and AF_s by 3.57 points.

## Post-test Cora research candidate

H27 combines the previously tested 5% first-observation EMA-row interpolation with H23's 12.5% class-recency gate. Across validation learner seeds 0--3 and all three sigmas it achieved:

| Scope | A_AUC | AF_s |
|---|---:|---:|
| Seed 0 | 43.54 | -13.12 |
| Seed 1 | 42.91 | -14.73 |
| Seed 2 | 44.11 | -13.89 |
| Seed 3 | 44.19 | -13.70 |
| **Four-seed mean** | **43.69** | **-13.86** |

H27 confirms the acquisition gain across seeds but still misses the retention target by 0.40 point. It must not be described as test-confirmed because it was designed after H23 test results were available.

## Scientific conclusions

### CoraFull

The reliable improvement is faster acquisition. Contextual replay, fixed working-head geometry, and class-recency-gated inference raise test A_AUC from the first-cycle 42.75 validation baseline to 44.25 on the frozen test aggregate. The remaining limitation is seed-sensitive retention rather than insufficient replay volume. Partial classifier-row stabilization reduces forgetting on seed 0 but does not cross the AF_s threshold across seeds.

### Arxiv

The Cora replay/EMA design is not portable. Arxiv's much longer per-class spans make alpha 0.995 effectively fast, producing severe forgetting. Stream-scale alpha 0.99966 corrects forgetting but lags acquisition. MAS offers a better acquisition/stability balance, yet its plateau-triggered consolidations vary enough across seeds that the validation result does not hold on test. Fixed classifier norms and post-hoc EMA blending both make MAS worse.

### Fair next direction

Future work should improve event stability rather than increase memory or replay. For CoraFull, the next candidate should replace the hard 100-update recency cutoff with a task-free continuous confidence derived from delivered-label counts, but it requires a fresh held-out split because this test split has been opened. For Arxiv, the priority is making MAS consolidation less seed-sensitive using a predeclared, continuous loss-statistic update rather than hard plateau events. This should be selected on multiple validation seeds and confirmed on a new untouched split.

## Reproducibility

- CoraFull split SHA-256: `d9dcf71d86bfd4345f9d382164c142d9c4186109c53e61f06febd16294fd8027`
- Arxiv split SHA-256: `e5d14cb6f86148ae3e7ea1627f8538defcc58a53cfe586559f8fc2c0d8b19ecf`
- Full hypothesis declarations: [PLAN.md](PLAN.md)
- Cora learner and controls: `Baselines/context_ema_model.py`
- MAS control wrappers: `Baselines/mas_geometry_model.py`
- Dual-dataset runner: `experiments/scientific_gaussian.py`
- Unit tests: `tests/test_context_ema.py`
- Raw local results: `results_gaussian_science` (ignored by Git; every run contains its configuration, source hashes, runtime versions, curves, budgets, and assertions)
