# H32: MAS consolidation is locked out by its own penalty

Declared 2026-09-20 before any H32 outcome is measured. Branch `Astra/h32-scalefree-consolidation`, parent `fe8bde9`.

**Order of work, stated for the record.** The loss-trace diagnostics in O5 were collected first, from **control runs only**, and the mechanism below was chosen from them. No outcome metric (A_AUC, AF_s, accuracy) was consulted when choosing the mechanism, and two earlier guesses about this detector — that it fires too often on CoraFull, and that its absolute thresholds are the binding constraint — were both falsified by those traces and discarded before this plan was written.

## Objective

One identical learning rule with identical constants must exceed **44 A_AUC** and **-14 AF_s** (closer to zero) on CoraFull-CL and on Arxiv-CL, averaging equally over drift widths and seeds. A dataset-specific switch is not a solution.

## Observations

**O1. MAS already clears both targets on Arxiv, at every seed measured.** Validation, sigma 60, stream seed 1, unmodified `tfmas` through the `mas_geometry` wrapper with `project_classifier=False`:

| seed | A_AUC | AF_s | final accuracy | consolidations |
|---:|---:|---:|---:|---:|
| 4 | 45.95 | -11.01 | 84.01 | 144 |
| 5 | 47.52 | -5.95 | 88.31 | 168 |
| 6 | 45.26 | -14.18 | 78.86 | 169 |
| **mean** | **46.24** | **-10.38** | **83.73** | 160 |

**O2. The same rule is catastrophic on CoraFull.** Validation, learner seed 4, stream seed 1: sigma 3 gives 15.97 / -50.42, sigma 10 gives 24.05 / -48.84, sigma 20 gives 31.61 / -47.30, for an equal-sigma mean of **23.88 / -48.86** with final accuracy 36.64 and 1 to 4 consolidations per run. A final accuracy of 37% on a stream whose classes are all delivered is underfitting, not only forgetting.

**O3. The re-anchoring rate differs by two orders of magnitude in the units that matter.** CoraFull delivers a new class every 17.1 stream steps and Arxiv every 254.3, so MAS re-anchors about **4 times per new class arrival on Arxiv** and about **0.03 times on CoraFull**, a factor near 130.

**O4. The two rates are qualitatively different algorithms.** Refreshing `theta*` every ~64 steps makes the penalty a *moving trust region*: stay near where you recently were. Refreshing it 1 to 4 times in 1,200 steps makes it a *stale pin* to a state reached while the model had seen a handful of CoraFull's 70 classes. The Arxiv success is attributable to frequent re-anchoring, not to parameter importance as such.

**O5. The cause is that the plateau detector monitors the penalised loss.** `Baselines/tfmas_model.py` appends `loss` to `self.loss_window` *after* the MAS penalty has been added to it, so once `omega` is non-zero the monitored quantity is inflated by the very penalty the previous anchor created, and further plateaus become strictly harder to detect. Measured on control runs, fraction of steps whose value sits below the .2 threshold:

| run | task loss < .2 | penalised loss < .2 | consolidations | last consolidation |
|---|---:|---:|---:|---|
| CoraFull sigma 3 | .274 | **.013** | 2 | step 51 of 1200 |
| CoraFull sigma 10 | .243 | **.038** | 3 | step 166 of 1200 |
| CoraFull sigma 20 | .152 | **.031** | 3 | step 524 of 1200 |
| Arxiv sigma 60 | .431 | **.214** | 169 | step 10108 of 10172 |

Under the **task** loss the eligibility fraction is broadly comparable across both datasets (.15 to .43). Under the **penalised** loss it collapses on CoraFull by a factor of 5 to 20 but survives on Arxiv. CoraFull stops consolidating in the first half of its stream and never resumes; Arxiv consolidates to the last 64 steps. The absolute thresholds .2 and .1 are therefore **not** the binding constraint, and the earlier guess that they were is withdrawn.

**O6. Prior cycles never touched the trigger.** H25 applied fixed classifier norms to MAS (16.80, catastrophic), H28 evaluated MAS through a slow EMA (34.48), H29 blended MAS with an EMA (42.46). All changed geometry or inference. The consolidation schedule was never changed.

*Recorded, not load-bearing:* `loss_window_mean_threshold = 0.2,` and `loss_window_variance_threshold = 0.1,` carry trailing commas and are one-element tuples. NumPy broadcasts the comparisons so the arithmetic is still `< .2` and `< .1` and every run above is valid; it is noted only as evidence these constants were never revisited.

## Hypothesis

**MAS fails to transfer because its plateau detector monitors the penalised loss and therefore locks itself out after its first anchor. On CoraFull the lock-out fires early and freezes a 70-class stream onto an early state; on Arxiv the loss is low enough that the detector survives it. Monitoring the task loss instead will restore re-anchoring on CoraFull and recover a large part of its 22-point deficit, while leaving Arxiv near its current result.**

## Mechanism

Exactly one line changes in behaviour: the plateau window holds the **task cross-entropy** rather than the penalised loss. The thresholds .2 and .1, the peak latch, the window length of 5, the importance estimate from squared outputs, the cumulative average into `omega`, the anchor copy, the MAS weight of .5, the backbone, the optimiser and the class-incremental head are all untouched.

**No constant is introduced, removed or changed.** Nothing is fitted to either dataset and nothing is swept. This is a correction relative to the detector's stated purpose — it exists to notice that the current data has been learned, which the penalty term does not measure — and it is disclosed as a change to DRIFT's released code, with the original behaviour retained as the control arm.

## Fairness and budget

No stored examples, no replay buffer, no extra parameters beyond the `omega` and `theta*` MAS already keeps, no extra optimizer steps, no extra graph work. The only cost is that consolidation will fire more often, and each firing performs one extra forward and backward on the block already sampled for that step; this is the original method's own cost, and the count is recorded per run. The trigger reads only the learner's own training loss. It receives no dataset name, Gaussian width, mixture weight, latent task identity, boundary, evaluation label or future label. One pass, one Adam update per minibatch at lr .005 and weight decay .0005, batch 10, the two-layer GCN, fanouts [10, 25], evaluation every 100 updates.

## Falsifiable experiment

**Arms.** Control is the released rule (monitor the penalised loss). Treatment monitors the task loss. Both run through the same new learner, so the arms differ in one boolean.

**Equivalence requirement, already verified.** With the treatment disabled the learner reproduces `mas_geometry` bit-for-bit over 60 steps on parameters, `omega`, `theta*`, consolidation count and step count. Six unit tests pass. Verifying this exposed a harness fact worth recording: DGL's CPU sampler does not fully reset under `dgl.random.seed`, so the **first** learner constructed in a process draws a different neighbour stream from every later one; the equivalence test burns one run before comparing.

**Pilot.** Full-length validation runs, learner seed 4, stream seed 1, CoraFull sigmas 3/10/20 and Arxiv sigma 60, paired, **three identical-configuration replicates per arm** (the H31 variance result measured an A_AUC sd near .6 and an AF_s sd near 5.5 from CUDA nondeterminism alone, so no decision is taken on a single run).

**Advance criterion.** CoraFull equal-sigma mean A_AUC improves by at least **5.0** points over the 23.88 control **and** Arxiv A_AUC falls by no more than **1.5** points from the 46.24 control mean, with every budget assertion passing. The CoraFull bar is large because the measured deficit is 22 points.

**Predeclared falsifiers.**
- If CoraFull's consolidation count does not rise substantially, the mechanism did not do what it claims and the result is uninterpretable rather than negative; report it as such.
- If the count rises but CoraFull A_AUC does not, O4's stale-pin explanation is wrong and H32 is rejected.
- If Arxiv degrades by more than 1.5 A_AUC, more frequent re-anchoring over-rigidifies and H32 is rejected.
- In every case the next mechanism goes on a new branch. No constant is introduced or tuned on this branch to rescue a failure, and the thresholds .2 and .1 stay where DRIFT set them.

**Declared diagnostics, recorded but not used for any decision:** consolidation count and step indices, consolidations per new class arrival, the task- and penalised-loss traces, and weighted peak and weighted final accuracy per arm reported next to AF_s so a flattened curve cannot be mistaken for retention.

**If the pilot passes.** Validation learner seeds 4/5/6 at fixed stream seed 1, CoraFull widths 3/10/20 and Arxiv widths 9/30/60, same rule and constants. Freeze source and configuration, then confirm on learner/stream pairs 101/101, 102/102, 103/103, reporting per-dataset mean and standard deviation, every width, final accuracy, AF_s, weighted peak and paired control deltas.

**Disclosure.** Prior studies already opened this repository's test nodes, so new stochastic replicates are not an untouched node holdout and will not be presented as one. Arxiv validation runs sit about 3 points above the published Arxiv test aggregate for the same MAS configuration (46.24 here against 42.23 on test seeds 1--3), so any claim against the 44 target must be made on confirmation runs, not on validation.

## Implementation plan

1. `Baselines/scalefree_mas_model.py` inherits the existing MAS learner and overrides only the plateau bookkeeping. `Baselines/tfmas_model.py` and every other baseline are untouched. *(done)*
2. `experiments/h32_scalefree_mas.py` reuses the benchmark data path, evaluator, budget assertions and source fingerprints, with replicate and trace flags. *(done)*
3. Unit tests for disabled-arm equivalence, window source, unchanged thresholds and latch, monotone consolidation indices and unchanged budget. *(done, 6 passing)*
4. Commit plan, implementation and tests before running the experiment.
5. Run the replicated paired pilot, then decide against the criterion above.
