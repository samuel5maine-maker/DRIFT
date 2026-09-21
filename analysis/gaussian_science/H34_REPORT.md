# H34: scale-free trigger — rejected, and H33's explanation is falsified

**Branch:** `Astra/h34-relative-trigger`. Predeclared protocol: [H34_PLAN.md](H34_PLAN.md), committed as `b472afd` before any run.

**Verdict: rejected.** CoraFull fell 1.68 A_AUC against its 1.5 tolerance. The Arxiv half was uninterpretable in the pilot, because the trigger did not fire there. A diagnostic rerun then did fire it, and that rerun falsifies the explanation H33 was built on.

## Result

Validation, learner seed 4, stream seed 1.

| Dataset | Arm | A_AUC | AF_s | weighted final | anchors |
|---|---|---:|---:|---:|---:|
| CoraFull (2 reps) | hybrid, absolute trigger (H33) | 41.42 | -17.13 | 73.51 | 15.7 |
| CoraFull (2 reps) | hybrid, relative trigger | 39.74 | -20.66 | 69.74 | 55.0 |
| CoraFull (2 reps) | MAS only, relative trigger | 23.01 | -51.73 | 30.60 | 40.7 |
| Arxiv (pilot) | hybrid, absolute trigger | 30.30 | -35.78 | 52.44 | 13 |
| Arxiv (pilot) | hybrid, relative trigger | 32.95 | -32.62 | 57.96 | **12** |
| Arxiv (diagnostic rerun) | hybrid, relative trigger | 32.99 | -33.69 | 49.11 final | **269** |
| Arxiv (pilot) | **MAS only, relative trigger** | **47.18** | **-8.70** | 88.87 | 401 |

**Against the gate.** Arxiv needed at least 44.04 and got 32.95: fail, but the predeclared falsifier applies first — the online anchor count (12) did not rise above H33's 13, so the pilot did not test the mechanism. CoraFull needed to stay within 1.5 of the absolute-trigger hybrid and lost 1.68: fail.

## Why online and offline disagreed

The offline replay predicted about 260 anchors on the Arxiv hybrid; the pilot produced 12 and an identical-configuration rerun produced 269. Both online runs anchored at steps 0 and 2, during the initial descent, then went silent for thousands of steps. The relative plateau test admits anchoring while the loss is still high, and DRIFT's latch then refuses to re-arm until the window mean climbs back above that early anchor's level plus one standard deviation. Whether it ever does is path-dependent: the rerun re-armed at step 3,139 and anchored 267 more times; the pilot re-armed only near the end. Offline replays over a trajectory produced by a different trigger could not see this, because the latch's reference point is set by the rule under test. **Offline trigger replays are not a reliable guide to online anchor counts on this learner**, and H34's O3 should not be reused.

## What the rerun falsifies

With 267 anchors over the last 70% of the stream, at a rate close to MAS alone, the hybrid still scored 32.99 with a final accuracy of 49%. Its pooled curve tracks replay-only almost exactly, while MAS alone rises steadily to 89%:

| checkpoint (every 10th) | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid, 269 anchors | 3 | 10 | 17 | 29 | 30 | 30 | 39 | 42 | 48 | 51 | 46 |
| replay only | 3 | 10 | 17 | 31 | 31 | 26 | 41 | 43 | 47 | 52 | 52 |
| MAS only, 401 anchors | 3 | 10 | 19 | 36 | 38 | 42 | 55 | 60 | 70 | 80 | 89 |

**H33's O2 — that replay beat MAS on Arxiv by silencing the consolidation trigger — is therefore wrong, or at most a minor part of the story.** Restoring frequent anchoring does not restore MAS's behaviour. Replay itself changes what the network does on Arxiv.

Two secondary results:
- The relative trigger **improves MAS alone on Arxiv**, to 47.18 / -8.70 from H32's 45.54 / -11.74, the best Arxiv result in this project. It does nothing for MAS alone on CoraFull (23.01).
- On CoraFull the relative trigger fired 3.5 times as often in the hybrid and cost 1.68 A_AUC, consistent with over-rigidifying a fast-turnover stream.

## The question this raises

MAS alone ends Arxiv at 89% pooled accuracy. An offline, i.i.d.-trained copy of the same backbone reached 53%, and every replay learner here ends near 50%. A one-pass sequential learner should not beat i.i.d. training by 36 points on real 40-way discrimination. DRIFT's evaluator restricts each latent task's logits to the classes up to that task, and classes arrive in index order. A network whose logits are ordered by recency would find the evaluated task's own pair on top of that prefix almost for free. That would explain at once why every class-balancing intervention in this project hurt Arxiv (CBRS, H31's prior correction, replay in H33 and H34) and why fixed classifier norms, which erase recency in logit magnitude, collapsed MAS from 45 to 16.8 in H25.

This is a hypothesis about the metric, not a result. It is measured next, by decomposing each task's restricted accuracy into "prediction falls in the task's own pair" and "correct within the pair", before any further learner is designed.

## Reproduction

```powershell
.\.conda\science\python.exe -m unittest tests.test_hybrid -q
.\.conda\science\python.exe experiments/h34_relative_trigger.py --dataset CoraFull-CL --replicate r1
.\.conda\science\python.exe experiments/h34_relative_trigger.py --dataset Arxiv-CL --replicate r1
```

Raw artifacts: `results_h34/`, `results_h34_diag/` (with per-step loss traces). Final-test mode is locked and was never opened.
