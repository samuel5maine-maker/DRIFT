# H31: instantaneous prior-shift correction — rejected

**Branch:** `Astra/h31-prior-shift-softmax`. Predeclared protocol: [H31_PLAN.md](H31_PLAN.md), committed as `8b7dc10` before any run.

**Verdict: rejected.** The predeclared falsifier fired on Arxiv. The mechanism is not the answer, but the runs produced three measurement results that matter more than the hypothesis did, and one of them invalidates an argument made in H31's own plan.

## Question

DRIFT's Gaussian schedule is an explicit label-prior shift: `gaussian_utils.gaussian_task_weights` defines a moving mixture over latent tasks and every minibatch draws its labels from it, while evaluation pools all test nodes under one fixed prior. Cross-entropy therefore fits the posterior of a distribution nobody scores. The hypothesis was that correcting the logits by this update's own label prior during training, and predicting with the uncorrected logits, would recover a large part of the forgetting gap on both datasets with one rule.

## The rule

For the ten incoming rows and the ten replay rows of one update, let `n_c` be how many carry class `c` over the class-incremental head. Training adds `a_c = log(n_c + 1)` to the logits of both loss terms; inference uses the raw logits.

`a` is the Laplace-smoothed log prior of this update's rows up to an additive constant that the softmax discards, so only `log(n_c + 1)` survives. A present class is handicapped by exactly `log(n_c + 1)` nats against an absent one, independent of batch size, class count and normalisation. Nothing is stored: no parameters, examples, optimizer steps, forward passes or graph work are added, and the vector is discarded each update. This is [Logit Adjusted Softmax](https://arxiv.org/abs/2311.06460) at its published defaults (tau = 1, window l = 1), on [Balanced Softmax / logit adjustment](https://arxiv.org/abs/2007.07314); the contribution claimed was the setting, not the estimator. Add-one smoothing was predeclared because the unsmoothed form sends absent classes to negative infinity, which is exactly H12's hard mask, already measured to cost 5.12 A_AUC.

**Base:** the H30 control, unchanged — contextual replay, CBRS-100, EMA .995 kept for accounting only, fixed working-classifier row norms, working-network inference. One variable moved.

## Verification

Twenty-six unit tests pass. The disabled arm reproduces the H30 control learner bit-for-bit over forty steps, past the buffer's first overflow, on parameters, EMA, buffer contents, replay schedule and graph work: the correction does not touch random state, because the sampling and replay-draw order are unchanged and only loss construction differs. Tests also pin the adjustment formula, the joint incoming-plus-replay counting, the reduced downward gradient on absent classes, the unadjusted inference path, and unchanged budgets. All budget assertions passed on every run.

The recorded diagnostics confirm the mechanism was active and sized as predicted: on Arxiv the mean adjustment was 2.24 nats over 10,172 updates, 9.0 distinct classes were present per update, and the negative-to-positive gradient-row ratio was 21.6, against the 19-or-so estimated in the plan.

## Result

Learner seed 4, stream seed 1, full-length validation runs.

| Dataset / width | A_AUC control | A_AUC adjusted | delta | AF_s control | AF_s adjusted | delta |
|---|---:|---:|---:|---:|---:|---:|
| CoraFull / 10 (means, n=5 and n=4) | 36.06 | 36.08 | **+0.02** | -37.55 | -35.40 | +2.15 |
| Arxiv / 60 (paired, n=1) | 30.32 | 27.99 | **-2.33** | -43.71 | -39.42 | +4.29 |

The gate required at least +2.0 A_AUC and at least +5.0 AF_s on **both** datasets. CoraFull returns an exact null on A_AUC. Arxiv **loses** 2.33 A_AUC, which is the plan's explicit falsifier: "If AF_s improves while A_AUC falls on either dataset, the mechanism sits on the same acquisition-retention frontier as H12 and is rejected, not re-weighted." No constant was swept, and `eps` stays at 1 as declared.

CoraFull numbers are means over repeated identical-configuration runs (5 control, 4 adjusted) because of the variance result below. Individual control A_AUCs were 36.32, 36.76, 35.45, 36.32, 35.45; adjusted were 36.05, 36.02, 36.24, 36.02.

## Three results that outlast the hypothesis

### 1. AF_s improves here by suppressing peaks, not by retaining knowledge

Decomposing AF_s into its two terms, weighted by each latent task's evaluation size:

| Run | weighted peak | weighted final | AF_s | A_AUC |
|---|---:|---:|---:|---:|
| Arxiv control | 89.57 | 41.20 | -43.71 | 30.32 |
| Arxiv adjusted | 83.72 | 40.14 | -39.42 | 27.99 |
| CoraFull control | 90.29 | 39.67 | -46.92 | 36.32 |
| CoraFull adjusted | 86.95 | 55.55 | -32.83 | 36.02 |

On Arxiv the entire 4.29-point AF_s "gain" is a 5.85-point **drop in peak** accompanied by a 1.06-point **drop in final**. Nothing was retained; the curve was flattened. On CoraFull the final genuinely rose, but the peak still fell 3.34 points, and against the replicate means the final delta is +2.2 against a standard deviation of about 5, so it is not established either.

**AF_s is therefore not a safe optimisation target on this benchmark, and a joint target of "A_AUC > 44 and AF_s > -14" can be approached from the wrong side.** The repository's baseline report warned about this qualitatively; this is a direct causal measurement of it inside one controlled pair.

### 2. The retention-oracle bound argued in H31's own plan is not attainable

The plan (O2/O3) computed A_AUC with every task held at its running maximum, obtaining 53.51 on CoraFull and 53.14 on Arxiv, and argued that closing about three fifths of the forgetting gap would clear both targets. That bound is **too optimistic, and the argument is withdrawn.** On Arxiv the twenty latent tasks reach their maxima at 19 and 20 *distinct* checkpoints in the two arms; on CoraFull, 12 distinct checkpoints for 35 tasks. The maxima are achieved at mutually exclusive moments, and a large part of each peak is recency bias: a task scores near 90% exactly when the model is biased toward the classes then arriving. No single model state realises all of them at once, so the oracle is not a reachable target and the linear interpolation from it to the goal has no force. A defensible ceiling needs an offline jointly-trained reference under the same evaluation protocol, which is being computed separately.

This does not rescue H31 — it failed its gate on measured numbers — but the plan's motivating arithmetic should not be reused.

### 3. Single-run deltas on this harness are noise-dominated

Five runs of the identical control configuration, same learner seed, same stream seed, same width, differing only in CUDA nondeterminism:

| Metric | values | mean | sd |
|---|---|---:|---:|
| A_AUC | 36.32, 36.76, 35.45, 36.32, 35.45 | 36.06 | **0.59** |
| AF_s | -33.16, -37.54, -35.06, -46.92, -35.06 | -37.55 | **5.46** |
| final accuracy | — | 46.53 | 6.47 |

`training.utils.set_seed` fixes every seed and sets `cudnn.deterministic`, but DGL's GPU sparse aggregation accumulates with atomics, so identical configurations do not reproduce.

The consequence is uncomfortable and should be acted on. The H1–H30 programme accepted and rejected hypotheses on **single-seed** deltas against a **0.5-point** A_AUC threshold. At an A_AUC sd near 0.6 per run per width (about 0.34 after averaging three widths), a 0.5-point decision is roughly a 1.5-sigma coin flip, and every AF_s decision at single seed is essentially unmeasured. H2 (+0.72), H8 (+0.58), H3 (-0.09), H5 (-0.29), H9 (-0.11), H10 (-0.51), H14 (-0.39), H17, the H22-versus-H23 choice (43.75 against 43.52), and H30's Cora AF_s gain (+3.52 against an AF_s sd of about 5) all sit inside the noise band. It also explains why validation results kept failing to transfer to test — H26 went from 45.02 on validation to 42.23 on test.

**Every future cycle in this project should run at least three identical-configuration replicates per arm and report the spread, and should report weighted peak and weighted final alongside AF_s so that a flattened curve cannot be mistaken for retention.**

## What this says about the mechanism

The correction behaved exactly as designed and still did not help. Three independent retention mechanisms have now produced the same signature on this base: H12's hard asymmetric cross-entropy (AF_s -10.71, A_AUC -5.12), H30's gated EMA-teacher distillation (AF_s better on both, A_AUC -1.58 on Arxiv), and H31 (AF_s better on both, A_AUC -2.33 on Arxiv). Given result 1, the most economical reading is that all three move along the same axis: they damp how strongly the model commits to whatever is currently arriving, which lowers peaks, raises or holds finals, improves AF_s, and leaves A_AUC flat or worse.

Two further specific readings, offered as hypotheses and not as established facts:

- The base may already absorb this correction. Fixed classifier-row norms remove the norm channel of recency bias and CBRS equalises replay exposure across classes, so the remaining softmax-denominator channel that H31 targets is the smaller one. A test on a bare contextual-replay base would separate these, and is not run here because this branch varies one thing.
- The Arxiv loss is consistent with the correction slowing acquisition on the dataset with the longer per-class span, where the plan predicted the opposite.

## Reproduction

```powershell
.\.conda\science\python.exe -m unittest tests.test_prior_shift tests.test_context_teacher tests.test_context_ema tests.test_mas_geometry -q
.\.conda\science\python.exe experiments/h31_prior_shift.py --dataset CoraFull-CL --sigmas 10 --seed 4 --arm paired
.\.conda\science\python.exe experiments/h31_prior_shift.py --dataset Arxiv-CL --sigmas 60 --seed 4 --arm paired
```

Replicates use `--arm control` or `--arm adjusted` with `--output-root results_h31_noise/<tag>`. Raw artifacts: `results_h31/` and `results_h31_noise/`, each run carrying its split and source fingerprints, budgets and protocol assertions. Final-test mode is locked and was never opened. Prior studies already opened this repository's test nodes; any later confirmation with fresh seeds is new stochastic replication, not an untouched node holdout.
