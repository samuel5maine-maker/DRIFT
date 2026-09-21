# H34: a scale-free consolidation trigger for the replay + MAS hybrid

Declared 2026-09-21 before any H34 outcome. Branch `Astra/h34-relative-trigger`, parent `7a21315` (H33 report).

**Order of work.** The trigger was chosen from task-loss traces of H33 hybrid runs and H32 control runs, replayed offline through candidate rules and compared on **firing rate only**. No A_AUC, AF_s or accuracy was consulted in choosing it.

## Objective

One identical rule with identical constants exceeding **44 A_AUC** and **-14 AF_s** on CoraFull-CL and Arxiv-CL, averaged over widths and seeds.

## Observations

**O1. H33 solved CoraFull and lost Arxiv to one identified cause.** The hybrid (contextual CBRS replay + H32 unlatched MAS) reached 41.78 / -17.54 on CoraFull, +14.55 over MAS alone and +4.63 over replay alone, with genuine retention. On Arxiv it reached 30.30 against 45.54 for MAS alone, because it consolidated only **13 times, stopping at step 7,047**, against 317 for MAS alone.

**O2. The absolute threshold makes replay and consolidation antagonistic.** DRIFT fires when the incoming loss window's mean is under 0.2 and its variance under 0.1. Without replay the network collapses onto current classes and the incoming loss falls under 0.2 constantly; class-balanced replay keeps the head spread and the loss stays above it. MAS's Arxiv success is therefore a side effect of recency overconfidence, which is exactly what replay removes.

**O3. A median-relative trigger restores the MAS regime under replay and is regular across datasets.** Replaying the saved traces through the latch with "window mean at or below the median of all window means so far, and window variance at or below the median of all window variances so far":

| trace | absolute rule: fires / median gap | relative rule: fires / median gap |
|---|---|---|
| Arxiv, H33 hybrid | 13 / 30, last at step 7,047 | **260 / 15**, last at step 10,084 |
| Arxiv, MAS only (H32 control) | 366 / 14 | 474 / 13 |
| CoraFull sigma 3, H33 hybrid | 22 / 30 | 58 / 16 |
| CoraFull sigma 10, H33 hybrid | 23 / 22 | 50 / 20 |
| CoraFull sigma 20, H33 hybrid | 4 / 32 | 49 / 15 |

Under replay on Arxiv the relative rule recovers the rate at which MAS alone thrives (median gap 15 against 14) and keeps firing to the end of the stream. On CoraFull it makes the rate regular across widths instead of ranging from 4 to 23. Its median gap sits between 13 and 30 steps in every regime examined. These are offline replays over trajectories produced by the absolute rule, so they indicate rather than guarantee the online rate; the online rate is a declared diagnostic.

## Hypothesis

**Replacing DRIFT's absolute plateau thresholds with their scale-free median-relative form removes the antagonism between replay and consolidation, so the H33 hybrid keeps its CoraFull result and recovers MAS-level performance on Arxiv, with one rule and no dataset constant.**

## Mechanism

Exactly one change from the H33 hybrid: the plateau condition. With `m_t` and `v_t` the mean and variance of the five most recent incoming cross-entropies,

```text
plateau_t = m_t <= median(m_1 .. m_t)  and  v_t <= median(v_1 .. v_t)
```

replaces `m_t < 0.2 and v_t < 0.1`. The peak latch, window length 5, importance estimate, cumulative `omega`, anchor copy, MAS weight .5, contextual CBRS-100 replay at 1:1, the working-network inference and the optimiser are all unchanged. **The rule has no constant**: the two absolute thresholds are removed and nothing replaces them. Unit tests verify it is invariant to rescaling the loss.

The history it reads is the learner's own sequence of window statistics, one scalar pair per update; it is disclosed as learner state (at most 10,172 pairs on Arxiv, about 160 KB) and contains no example, label, feature or evaluation information.

## Falsifiable experiment

**Arms from one learner:** `hybrid_relative` (treatment), `hybrid_absolute` (H33 rule, rerun in the same processes as the paired control), and `mas_relative` (replay off, relative trigger), which attributes any Arxiv change to the trigger itself rather than to its interaction with replay. The H32 MAS-only result on Arxiv (45.54 / -11.74) is the fixed reference the Arxiv gate is written against.

**Pilot.** Validation, learner seed 4, stream seed 1, CoraFull widths 3/10/20 with two replicates, Arxiv width 60 with one. The single Arxiv replicate is justified by measurement: the Arxiv replicates in H32 and H33 were bit-identical in all five arm-pairs checked.

**Advance criterion, all required:**
1. Arxiv: `hybrid_relative` A_AUC at least **44.04** (MAS only minus 1.5).
2. CoraFull: `hybrid_relative` A_AUC at least `hybrid_absolute` minus **1.5**.
3. Wherever AF_s improves against the comparison arm, weighted final accuracy does not fall.

**Predeclared falsifiers.** If the online consolidation rate on Arxiv does not rise far above 13, the mechanism did not act and the result is uninterpretable. If it rises and Arxiv does not recover, O2's explanation is wrong and H34 is rejected. If CoraFull loses more than 1.5, the regular firing over-rigidifies fast-turnover streams and H34 is rejected. No constant is introduced to rescue either.

**Declared diagnostics:** online consolidation count, median gap and last anchor step; weighted peak and final; replay graph work.

**If the pilot passes.** Validation learner seeds 4/5/6 at stream seed 1, CoraFull widths 3/10/20 and Arxiv widths 9/30/60, same rule; freeze; confirm on learner/stream pairs 101/101, 102/102, 103/103.

**Disclosure.** The test nodes were opened in earlier cycles, so fresh seeds are stochastic replication, not an untouched holdout. Arxiv validation has run about 3 points above test for MAS, so claims against 44 rest on confirmation.
