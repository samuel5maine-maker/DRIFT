# H31: instantaneous prior-shift correction (drift-adjusted softmax)

Declared 2026-09-20 before implementation and before any H31 outcome. Branch `Astra/h31-prior-shift-softmax`, parent `c1ca145` on `Codex/h30-contextual-teacher`.

## Objective

One identical learning rule with identical constants must exceed **44 A_AUC** and **-14 AF_s** (closer to zero) on CoraFull-CL and on Arxiv-CL, averaging equally over drift widths and seeds. A dataset-specific switch is not a solution.

## Observations that motivate this hypothesis

**O1. The stream is literally a label-prior shift process.** `gaussian_utils.gaussian_task_weights` defines a softmax-Gaussian mixture `alpha(t)` over latent tasks, and each minibatch draws its labels from that mixture. The training label prior is therefore an explicit, smoothly moving function of time, while evaluation pools all test nodes under one fixed global prior. This is textbook prior shift, and it is the *defining* property of the DRIFT Gaussian regime rather than an incidental one.

**O2. Retention, not acquisition, is where the remaining points are.** Recomputing A_AUC from the H30 paired runs after replacing each latent task's accuracy curve by its running maximum (that is, assuming nothing once learned is ever lost, with acquisition speed left exactly as measured) gives:

| Run | measured A_AUC | retention-oracle A_AUC | headroom |
|---|---:|---:|---:|
| CoraFull sigma10 control | 35.45 | 53.51 | +18.06 |
| Arxiv sigma60 control | 29.84 | 53.14 | +23.30 |

Task test-set weights were recovered exactly from the stored pooled and per-task curves by least squares (Arxiv residual 0.00000, Cora 0.057). Peak per-task accuracy already averages about 88% on Arxiv and about 88% on Cora; final per-task accuracy averages about 44% and 47%. **Every latent task is already acquired well and then lost.** This revises the earlier report's conclusion that acquisition was the primary bottleneck: that conclusion was drawn from an EMA-inference configuration, and does not hold for the current working-inference learner.

**O3. The targets and the headroom are quantitatively compatible.** Interpolating linearly between the measured point and the retention oracle, closing 66% of Arxiv's AF_s gap predicts A_AUC about 45.2, and closing 60% of Cora's predicts about 46.3. A mechanism that cuts forgetting by roughly three fifths, without slowing acquisition, is sufficient to clear both targets on both datasets. Nothing else in the repository's history is worth that many points.

**O4. Two rejected experiments bracket the mechanism proposed here.** H12 applied a hard asymmetric cross-entropy that removed absent old classes from the incoming softmax entirely. It produced the best AF_s in the whole first cycle (-10.71, past the target) and the worst A_AUC cost (-5.12). H13 applied a balanced softmax using *cumulative* delivered-label counts and lost 1.21 A_AUC. Cumulative counts estimate the stream's average prior, which is roughly stationary and therefore almost unrelated to the drift; they also actively handicap a newly arrived class, whose cumulative count is near zero precisely when it must be learned fastest. The instantaneous prior is the quantity the drift actually moves, and a smoothed correction is the principled interior of the interval whose endpoints H12 and H13 measured.

**O5. Forgetting worsens exactly where prior shift is sharper.** Arxiv's width relative to its stream length (60/10172) is about one third of CoraFull's (20/1200), so its prior moves through the class set proportionally faster. Arxiv also forgets more (AF_s -41.6 against -35.1). The framing predicts the sign of an already-measured difference it was not fitted to.

**O6. The per-class gradient budget is the concrete mechanism.** Each update applies one positive-class gradient per row and a negative-class gradient to every other column of the head. With ten incoming rows plus ten class-balanced replay rows and about forty seen classes, a class absent from the current window receives on the order of nineteen negative pushes for every quarter of a positive push. That ratio grows with the number of seen classes, which is why the damage compounds along the stream.

## Hypothesis

**The dominant cause of forgetting in DRIFT's Gaussian regime is that cross-entropy fits the posterior under the instantaneous training prior, which the mixture schedule drives away from the fixed evaluation prior. Correcting the logits by the instantaneous prior during training, and predicting with the uncorrected logits, will recover a large fraction of the retention headroom on both datasets with one rule and no dataset-specific constant.**

## Mechanism

Let the update at step `t` contain the ten incoming labels and the drawn replay labels. Let `n_c` be the number of those rows carrying class `c`, over the class-incremental head `[0, offset)` the benchmark already uses. Define the adjustment

```text
a_c = log(n_c + 1)
```

and train with

```text
loss = CE(z_incoming + a, y_incoming) + CE(z_replay + a, y_replay)
```

Inference uses `z` unchanged. The adjustment vector is rebuilt every update and stored nowhere.

`a` is the Laplace-smoothed log prior of the rows in this update, up to an additive constant that the softmax discards. Because the constant drops out, only `log(n_c + 1)` survives: the correction is free of any normalisation, any window length, any temperature, and any dependence on the number of classes or the batch size. A class with `n` rows in the update is handicapped by exactly `log(n + 1)` nats against a class with none, so the protection given to an absent class is a fixed, scale-free function of how strongly the present classes dominate this update.

This is [Logit Adjusted Softmax](https://arxiv.org/abs/2311.06460) (Hoang et al., TMLR) at its published defaults, tau = 1 and window length l = 1, applied for the first time to graph continual learning and to a stream whose prior shift is specified analytically rather than induced by task boundaries. The prior-shift correction itself is [Balanced Softmax / logit adjustment](https://arxiv.org/abs/2007.07314). This experiment does not claim to invent the estimator. It claims that DRIFT's Gaussian regime is a prior-shift problem and is therefore the setting the estimator should fit best.

**Predeclared handling of zero counts.** The source paper leaves it unspecified, and an unsmoothed `log(n_c / N)` sends absent classes to negative infinity, which is exactly H12's hard mask and is already measured to cost 5.12 A_AUC on Cora. Add-one (Laplace) smoothing is the canonical uninformative choice and is fixed at `eps = 1` here. **It will not be swept.** If `eps = 1` fails, H31 is rejected on this branch.

## Fairness and budget

The adjustment adds no parameters, no stored examples, no labels beyond those already delivered in the update, no optimizer steps, no forward passes and no graph work. It reads only the labels of rows the learner is already training on. It receives no dataset name, Gaussian width, mixture weight, latent task identity, task boundary, evaluation label or future label. Its entire state is one temporary vector of length `offset`, discarded before the next update. Every other element of the DRIFT protocol is unchanged: one pass, one Adam update per incoming minibatch at lr .005 and weight decay .0005, batch 10, 100-entry buffer, 10 replay seeds, the two-layer GCN and fanouts [10, 25], evaluation every 100 updates.

## Falsifiable experiment

**Base.** Exactly the H30 control learner: contextual replay, CBRS-100, EMA decay .995 maintained but used only for accounting, fixed working-classifier row norms, working-network inference. The base is held fixed so the pilot varies one thing.

**Equivalence requirement.** With the adjustment disabled the new learner must reproduce the stored H30 control runs bit-for-bit on both datasets. The adjustment must not consume random state, so all sampling and the replay draw keep their existing order and only loss construction changes. This is asserted in unit tests and checked against the stored control JSON.

**Pilot.** Full-length validation runs, learner seed 4, stream seed 1, CoraFull sigma 10 and Arxiv sigma 60, paired against the control in the same process.

**Advance criterion, both datasets simultaneously:** AF_s improves by at least **5.0** points **and** A_AUC improves by at least **2.0** points, with every budget assertion passing. The gate is deliberately large. The remaining distance to target is fourteen points, so a mechanism worth pursuing must move a substantial share of the measured headroom; a one-point effect would not be the answer even if it were real.

**Predeclared falsifiers.** If AF_s improves while A_AUC falls on either dataset, the mechanism sits on the same acquisition-retention frontier as H12 and is rejected, not re-weighted. If the Cora delta is positive and the Arxiv delta is not, the transfer claim fails and H31 is rejected. In either case the next mechanism goes on a new branch with a fresh declaration; no coefficient, smoothing constant or base component is tuned on this branch.

**Declared diagnostics, recorded but not used for any decision:** mean and maximum adjustment magnitude, mean distinct classes per update, and the per-run gradient-budget ratio from O6. A declared mechanistic ablation setting the replay rows' contribution to `n` aside is *not* run in this cycle.

**If the pilot passes.** Validation learner seeds 4/5/6 at fixed stream seed 1, CoraFull widths 3/10/20 and Arxiv widths 9/30/60, the same rule and constants everywhere. Arxiv 9 and 30 scale the paper's CoraFull sweep by the same proportions; the paper reports only Arxiv 60, so they are an extension and are not claimed as official widths. Source and configuration are frozen before confirmation. Confirmation uses learner/stream seed pairs 101/101, 102/102, 103/103 over the same widths and reports per-dataset mean and standard deviation, every width, final accuracy, AF_s, and the paired control deltas.

**Disclosure.** Prior studies already opened this repository's test nodes. New stochastic replicates are held out from this cycle but are not an untouched node holdout, and will not be presented as one. If confirmation fails it is reported as a failure; later reuse is not presented as independent confirmation.

## Implementation plan

1. Add `Baselines/prior_shift_model.py`, a small learner inheriting the H30 control and overriding only loss construction. Leave every existing baseline file untouched.
2. Add `experiments/h31_prior_shift.py`, reusing the benchmark data path, evaluator, budget assertions and source fingerprints.
3. Unit-check: adjustment equals `log(n+1)` over incoming and replay rows jointly, identical vector applied to both loss terms, no adjustment at inference, disabled-arm equivalence including random-state order through buffer overflow, no access to undelivered labels, and unchanged replay and step budgets.
4. Commit the plan and the implementation before running the pilot.
5. Run the paired pilot on both datasets, then decide against the criterion above.
