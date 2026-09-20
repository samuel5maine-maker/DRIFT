# H32: MAS consolidation lock-out — mechanism confirmed, advance criterion missed

**Branch:** `Astra/h32-scalefree-consolidation`. Predeclared protocol: [H32_PLAN.md](H32_PLAN.md), committed as `9217db4` before any outcome was measured.

**Verdict: the mechanism is confirmed and the fix is real, but H32 misses its predeclared advance criterion and is rejected as a route to the target.** CoraFull needed +5.0 A_AUC and gained +3.78.

## Question

DRIFT's `tfmas` appends the **penalised** loss to its plateau window, so once `omega` is non-zero the quantity the detector monitors is inflated by the very penalty the previous anchor created. The hypothesis was that this locks the detector out, that the lock-out is what makes MAS untransferable, and that monitoring the task loss instead would recover CoraFull while leaving Arxiv intact.

## The change

One line of behaviour: the plateau window holds the task cross-entropy instead of the penalised loss. The thresholds .2 and .1, the peak latch, the window length of 5, the squared-output importance estimate, the cumulative average into `omega`, the anchor copy, the MAS weight of .5, the backbone and the optimiser are untouched. **No constant is introduced, removed or changed, and nothing was swept.**

## Verification

Six unit tests pass. With the change disabled the learner reproduces `mas_geometry` bit-for-bit over 60 steps on parameters, `omega`, `theta*`, consolidation count and step count. An offline replay of the detector over the saved control traces reproduces the observed consolidation counts **exactly** — 169, 3, 3 and 2 — when fed the penalised loss, which independently confirms both that the traces are genuine control traces and that the detector is understood. All budget assertions passed on every run.

## Result

Validation, learner seed 4, stream seed 1. CoraFull is the equal-sigma mean over 3, 10 and 20; Arxiv is sigma 60. Each replicate is an identical-configuration rerun.

| Dataset | Arm | A_AUC | AF_s | weighted peak | weighted final | consolidations | per class arrival |
|---|---|---:|---:|---:|---:|---:|---:|
| CoraFull | control | 24.22 | -52.85 | 91.81 | 38.48 | 2.6 | 0.037 |
| CoraFull | task-loss window | **28.00** | **-46.26** | 92.04 | 46.21 | 21.4 | 0.306 |
| CoraFull | delta | **+3.78** | +6.59 | +0.22 | **+7.73** | **x8.4** | |
| Arxiv | control | 45.21 | -16.83 | 93.89 | 76.01 | 169.0 | 4.225 |
| Arxiv | task-loss window | **45.54** | **-11.74** | 93.95 | 86.02 | 317.0 | 7.925 |
| Arxiv | delta | **+0.33** | +5.09 | +0.07 | **+10.01** | x1.9 | |

**The mechanism did exactly what was predicted.** CoraFull's consolidation rate rose 8.4-fold and its per-class-arrival rate went from .037 to .306; Arxiv, which was never locked out, roughly doubled.

**The gain is genuine retention, not the peak suppression that sank H31.** On both datasets the weighted peak is unchanged to within .25 of a point while the weighted final accuracy rises by 7.73 and 10.01 points. This is the guard the H31 report demanded, and it is the first mechanism in this project to pass it.

**Against the predeclared gate:** Arxiv had to lose no more than 1.5 A_AUC and instead gained .33, so it passes. CoraFull had to gain at least 5.0 and gained 3.78, so it fails. The gate stands as written; the bar is not moved after the fact, and no constant is introduced to rescue it.

**Against the project targets (44 A_AUC, -14 AF_s):** the fixed rule reaches **45.54 / -11.74 on Arxiv validation, clearing both**, and 28.00 / -46.26 on CoraFull, clearing neither. The single-rule goal remains unmet, and CoraFull is the reason.

*Replication caveat.* Nondeterminism here is intermittent rather than per-run: CoraFull replicates r1 and r3 were bit-identical to each other and r2 differed; both Arxiv replicates were identical. The effective number of independent samples is therefore smaller than the nominal 3 and 2, and the deltas above should be read as point estimates with the H31 variance figures (A_AUC sd near .6, AF_s sd near 5.5) as the relevant scale. The CoraFull A_AUC delta is consistent in sign across every replicate and at two of three widths.

## What this establishes

1. **A real defect in DRIFT's released MAS baseline, with a measured cost.** Fixing it is worth +3.78 A_AUC and +7.7 points of final accuracy on CoraFull, and +5.09 AF_s and +10.0 points of final accuracy on Arxiv. Anyone reproducing `tfmas` inherits the lock-out. The published CoraFull MAS numbers are measurements of a detector that stops working partway through the stream.
2. **The lock-out was not the reason MAS cannot do CoraFull.** Consolidation was restored eightfold and CoraFull still only reached 28.0, against 44.25 for the replay family. MAS keeps no examples, so on a stream delivering a new class every 17 steps there is nothing to recover a class from once it stops arriving. The deficit is structural, not a timing artifact. The plan's O4 "stale pin" story is therefore **partly right and insufficient**: unpinning helps by about four points and leaves sixteen.
3. **Two earlier readings of this detector were wrong and were discarded on evidence before declaring** — that it fires too often on CoraFull (it fires far too rarely), and that its absolute thresholds are the binding constraint (the task loss clears .2 on 15--27% of CoraFull steps; the penalty is what blocks it). Both are recorded in [H32_PLAN.md](H32_PLAN.md) so the record shows what was believed when.

## A harness fact worth keeping

DGL's CPU neighbour sampler does not fully reset under `dgl.random.seed`. The **first** learner constructed in a process draws a different neighbour stream from every later one in that process, so a naive two-arm comparison inside one script compares first-run against second-run and diverges for reasons that have nothing to do with the arms. The equivalence test burns one throwaway run before comparing. Any future paired comparison on this harness must do the same.

## Where this leaves the programme

Three hypotheses have now been rejected against the same wall, and the two families have been measured precisely on both datasets:

| | CoraFull | Arxiv |
|---|---:|---:|
| replay + EMA family (H23) | **44.25 / -14.00** (test) | ~29 |
| MAS family, with this fix | 28.00 / -46.26 | **45.54 / -11.74** (validation) |

Each family clears both targets on one dataset and fails badly on the other, and the failures have complementary, now-measured causes: MAS has no memory, so it cannot survive CoraFull's 17-step class turnover; replay has a 100-node buffer against 101,710 Arxiv training nodes, so it cannot survive Arxiv's length. Every single-mechanism repair tried so far — teacher distillation (H30), prior-shift correction (H31), consolidation timing (H32) — moves one dataset a few points and does not bridge them.

The next hypothesis should therefore be **one learner carrying both mechanisms at once**, contextual replay for class coverage and importance-based consolidation for stability, both always active with identical constants on both datasets. This is not a dataset switch: both terms run everywhere and the stream decides which carries the load. It has never been tried here — H25, H28 and H29 combined MAS with geometry or inference changes, never with replay — and the complementary failure modes are now quantified precisely enough to predict what each term should contribute.

## Reproduction

```powershell
.\.conda\science\python.exe -m unittest tests.test_scalefree_mas -q
.\.conda\science\python.exe experiments/h32_scalefree_mas.py --dataset CoraFull-CL --arm paired --replicate r1
.\.conda\science\python.exe experiments/h32_scalefree_mas.py --dataset Arxiv-CL --arm paired --replicate r1
```

Add `--traces` to store per-step task and penalised loss traces. Raw artifacts: `results_h32/` and `results_h32_diag/`, each run carrying split and source fingerprints, budgets, consolidation indices and protocol assertions. Final-test mode is locked and was never opened. Prior studies already opened this repository's test nodes; any later confirmation with fresh seeds is new stochastic replication, not an untouched node holdout. Arxiv validation sits about 3 points above the published Arxiv test aggregate for the same configuration, so the 45.54 above is a validation number and is not a test claim.
