# H33: contextual replay and unlatched MAS consolidation in one learner

Declared 2026-09-21 before any H33 outcome. Branch `Astra/h33-replay-consolidation`, parent `a875e3a` (H32).

## Objective

One identical learning rule with identical constants must exceed **44 A_AUC** and **-14 AF_s** on CoraFull-CL and Arxiv-CL, averaging equally over widths and seeds. A dataset-specific switch is not a solution.

## Observations

**O1. Each family clears both targets on exactly one dataset.** Validation, learner seed 4, stream seed 1 unless stated:

| | CoraFull | Arxiv |
|---|---:|---:|
| contextual replay + EMA (H23, test aggregate) | **44.25 / -14.00** | ~29 (H23 transferred, validation) |
| MAS with the H32 correction | 28.00 / -46.26 | **45.54 / -11.74** |

**O2. The failures are complementary and measured.** MAS keeps no examples. With H32's correction it consolidates eight times more often on CoraFull and still reaches only 28, so the CoraFull deficit is not consolidation timing: on a stream delivering a new class every 17 steps, nothing recovers a class once it stops arriving. Replay supplies exactly that. In the other direction, a 100-node buffer is 0.1% of Arxiv's 101,710 training nodes replayed across 10,172 updates, and every replay learner in this repository forgets heavily there (AF_s -37 to -44), while MAS's frequently re-anchored penalty acts as a moving trust region that holds Arxiv at a weighted final accuracy of 86%.

**O3. The combination has never been run.** H25, H28 and H29 combined MAS with classifier-norm projection, slow-EMA inference and logit blending respectively. None added replay. DRIFT's released baselines include no replay-plus-regularisation method either.

**O4. Two components of the Cora replay recipe are excluded on evidence, not to fit Arxiv.** Fixed classifier-row norms combined with MAS were catastrophic (H25: 16.80). An EMA inference model trades acquisition against retention on a timescale written in absolute steps, and on Arxiv every EMA tried lost acquisition (H24, H28). The hybrid therefore uses neither, and predicts with the working network, as MAS does.

## Hypothesis

**Replay and consolidation repair each other's measured failure. Running contextual CBRS replay and H32's unlatched MAS together in one learner, with identical constants on both datasets, will recover CoraFull well above either single mechanism while keeping Arxiv at the MAS level.**

## Mechanism

Per update, with ten incoming nodes:

```text
loss = CE(incoming)                                   over the class-incremental head
     + CE(replay)                                     ten seeds from CBRS-100, sampled
                                                      neighbourhoods on the training graph
     + 0.5/2 * sum_j omega_j (theta_j - theta*_j)^2   DRIFT's MAS penalty, once anchored
```

One Adam update. The plateau window holds the **incoming** cross-entropy only (H32's correction), and consolidation — importance from squared outputs on the incoming block, cumulative average into `omega`, anchor copy — is DRIFT's `tfmas` rule unchanged, with its thresholds .2 and .1, window 5 and weight .5. Incoming nodes are then offered to the CBRS memory. Inference uses the working network.

**Every constant is inherited, none is chosen here:** buffer 100, replay 1:1 and CBRS from the DRIFT protocol and H2; MAS weight, thresholds and window from DRIFT's released `tfmas`; the task-loss window from H32. Nothing is swept and nothing differs by dataset.

The CBRS choice is predeclared with its risk stated. Under isolated replay CBRS cost Arxiv 5.4 A_AUC relative to uniform reservoir, because Arxiv's pooled accuracy is dominated by a few very large classes. In this learner replay is hired for class **coverage** on fast-turnover streams, which is what CBRS provides and reservoir does not (24 of CoraFull's 70 classes end with no slot under reservoir), and Arxiv stability is expected to come from consolidation. If Arxiv is lost here, the buffer policy is the first suspect for a follow-up branch; it is not changed on this one.

## Fairness and budget

100 node IDs and labels in memory, ten replay seeds per ten incoming, one optimizer step per minibatch, the two-layer GCN and fanouts [10, 25]. The persistent extra state is MAS's own `omega` and `theta*`, two parameter copies, which is exactly what DRIFT's released MAS already keeps; the H23 learner kept one EMA copy instead. Contextual replay adds sampled graph work, recorded per run as in H1. No dataset name, width, mixture weight, task identity, boundary or future label reaches the learner. The harness asserts the replay schedule, a buffer of exactly 100, bounded storage, and at most two extra parameter copies.

## Falsifiable experiment

**Arms, all from one learner differing only in two booleans:** `hybrid` (replay on, MAS on), `replay_only` (MAS off: contextual CBRS replay, working inference, no EMA, no norms), and `mas_only` (replay off). Unit tests verify that `mas_only` reproduces H32's learner bit-for-bit, that `replay_only` never consolidates, that the window holds only the incoming loss, and the replay schedule and budget. 38 tests pass across H30–H33.

**Pilot.** Full-length validation runs, learner seed 4, stream seed 1, CoraFull sigmas 3/10/20 and Arxiv sigma 60, all three arms, two replicates each (the measured nondeterminism is intermittent, so a third replicate often duplicates one of the first two).

**Advance criterion, all three required:**
1. CoraFull: hybrid A_AUC at least **5.0** above `mas_only`, and not more than 1.5 below `replay_only`.
2. Arxiv: hybrid A_AUC not more than **1.5** below `mas_only`.
3. On both datasets, any AF_s improvement of the hybrid over the arm it is compared with is accompanied by a weighted final accuracy that does not fall — no gain from flattened peaks.

The project target (44 / -14 on both) is reported alongside but is not the pilot gate; a pilot on one seed is not a claim.

**Predeclared falsifiers.** If the hybrid is not better than the better single arm on at least one dataset, the mechanisms do not combine and H33 is rejected. If replay costs Arxiv more than 1.5 A_AUC against `mas_only`, the conflict between replay and the penalty is real on long streams and H33 is rejected. No constant, weight, buffer policy or component is changed on this branch to rescue a failure.

**Declared diagnostics:** consolidations and their rate per new class arrival, replay graph work, weighted peak and weighted final accuracy per arm.

**If the pilot passes.** Validation learner seeds 4/5/6 at stream seed 1 over CoraFull widths 3/10/20 and Arxiv widths 9/30/60, same rule and constants; freeze; confirm on learner/stream pairs 101/101, 102/102, 103/103, reporting per-dataset mean and sd, every width, final accuracy, AF_s, weighted peak and the single-arm deltas.

**Disclosure.** This repository's test nodes were already opened by earlier cycles; fresh stochastic replicates are not an untouched holdout. Arxiv validation has run about 3 points above test for the same MAS configuration, so any claim against 44 must rest on confirmation runs.
