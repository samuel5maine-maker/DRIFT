# Scientific report: CoraFull Gaussian mixing experiments

**Branch:** `Codex/gaussian-scientific-50`

**Selection data:** CoraFull-CL validation split only

**Transition settings:** Gaussian mixing with \(\sigma \in \{3,10,20\}\)

**Primary objective:** equal-sigma mean \(A_{AUC} > 50\) without exceeding the DRIFT replay budget

## Outcome

The target has not been met. The strongest training design reached **42.75 mean validation A_AUC** with **-13.46 AF_s**. It combines class-balanced context replay, a constant exponential moving average (EMA) model for inference, and fixed classifier-row norms. An optional transductive label-propagation inference step reached **43.32 A_AUC** and **-10.96 AF_s**, but it uses the benchmark's per-task evaluation graphs and adds substantial evaluation work. It is therefore reported separately rather than treated as the default fair candidate.

No test result was opened. The protocol required the design to reach the target on validation before freezing it and running seeds 1–3 on test. Stopping at validation prevents test-set tuning and means these numbers are exploratory single-seed estimates, not final benchmark claims.

## Starting observations

The existing CoraFull Gaussian results showed three useful patterns:

1. Plain experience replay was weak: the repository's reproduced ER result at \(\sigma=20\) was 32.8 A_AUC.
2. EMA-based inference was consistently useful. CLS-ER reached 38.6, while its working network reached only 33.8 in the existing ablation. Most of that gain came from parameter averaging rather than its consistency penalty.
3. Coverage alone was insufficient. ER-CBRS covered all classes but reached only 33.3 at \(\sigma=20\), close to ER's 32.8. This suggested that how replay nodes are presented to the GNN mattered more than simply retaining one node from every class.

The experiments therefore began with the hypothesis that replaying isolated nodes destroys the neighborhood information on which a GCN depends. The complete predeclared sequence and decision thresholds are in [PLAN.md](PLAN.md).

## Protocol and fairness controls

Every training experiment used the same constraints:

- CoraFull-CL, 35 latent tasks, two classes per task, with task identity withheld from the learner.
- Gaussian streams at \(\sigma=3,10,20\), generated with stream seed 1.
- Learner seed 0 for validation selection.
- 1,200 minibatches of 10 nodes, one pass through the stream.
- A replay buffer fixed at **100 node-label pairs**.
- At most 10 replay seeds per incoming minibatch, a 1:1 replay ratio.
- One Adam update per minibatch, learning rate 0.005.
- The same two-layer GCN and sampled neighborhood fanouts `[10, 25]`.
- Evaluation every 100 updates and at the end, matching the legacy A_AUC calculation.
- Labels outside the delivered minibatch were masked from the learner.
- Equal weight for each sigma in the selection mean.

These controls follow the core comparison conditions described by the [DRIFT paper](https://arxiv.org/pdf/2605.12998): online one-pass learning, latent tasks, a two-layer GCN, and a 100-node replay memory. Context replay stores no additional examples. It reconstructs neighborhoods from the same transductive training graph already used for incoming updates, while accounting for the extra sampled nodes and edges.

The working environment was rebuilt after the previous local workspace disappeared. It used Python 3.10, PyTorch 2.3.1+cu118, DGL 1.1.2+cu118, and an RTX 3070 Ti. The CoraFull split was regenerated with the documented seed; its SHA-256 is `d9dcf71d86bfd4345f9d382164c142d9c4186109c53e61f06febd16294fd8027`. Because this differs from the earlier local split, historical repository scores are motivation rather than matched controls.

## Hypothesis results

The acceptance threshold was a validation A_AUC improvement of at least 0.5 point unless the plan specified otherwise. Deltas below use the relevant contemporary control; H7 and H8 use the standard predictions from the exact same run.

| ID | Intervention and hypothesis | Mean A_AUC | AF_s | Delta | Decision |
|---|---|---:|---:|---:|---|
| H1 | Replay sampled graph context instead of isolated self-loop nodes | 38.81 | -12.91 | +2.69 | Accept |
| H2 | Replace reservoir memory with class-balanced reservoir sampling (CBRS) | 39.52 | -13.66 | +0.72 | Accept |
| H3 | Adapt EMA decay from consecutive minibatch label overlap | 39.44 | -12.41 | -0.09 | Reject |
| H4 | Project every classifier row to its initial mean norm | **42.75** | -13.46 | **+3.23** | Accept |
| H5 | Reduce replay loss weight from 1.0 to 0.5 | 42.46 | -14.65 | -0.29 | Reject |
| H6 | Project rows to the current mean norm, allowing global scale drift | 39.10 | -19.36 | -3.65 | Reject |
| H7 | Classify with buffer prototypes derived from EMA embeddings | 40.06 | -17.00 | -2.31 | Reject |
| H8 | Ten-step label propagation from buffer labels at evaluation | **43.32** | **-10.96** | +0.58 | Provisional; inference caveat |
| H9 | Halve the fixed classifier norm | 42.64 | -14.31 | -0.11 | Reject |
| H10 | Use full rather than sampled replay neighborhoods | 42.25 | -12.64 | -0.51 | Reject |
| H11 | Replace the linear head with a cosine classifier | 41.18 | -14.37 | -1.58 | Reject |
| H12 | Apply hard asymmetric cross-entropy to incoming samples | 37.63 | -10.71 | -5.12 | Reject |
| H13 | Apply online balanced softmax using delivered-label counts | 41.54 | -15.20 | -1.21 | Reject |
| H14 | Add hidden-state cosine smoothness over sampled incoming edges | 42.36 | -14.38 | -0.39 | Reject |
| H15 | Prefer high-degree nodes within class-balanced memory | 41.56 | -10.20 | -1.19 | Reject |

H12 illustrates why AF_s cannot be optimized alone: it produced much less measured forgetting but substantially worse A_AUC because weak acquisition leaves less learned performance to forget. This agrees with the warning already recorded in the repository's baseline report.

## Best model by drift width

The best retained training model is H4: sampled context replay, CBRS-100, EMA decay 0.995, and classifier rows projected to their initialization-time mean norm.

| Gaussian sigma | A_AUC | AF_s | Final accuracy |
|---:|---:|---:|---:|
| 3 | 42.96 | -13.90 | 70.35 |
| 10 | 42.66 | -11.27 | 75.51 |
| 20 | 42.64 | -15.22 | 72.42 |
| **Equal-sigma mean** | **42.75** | **-13.46** | **72.76** |

The result is unusually stable across transition widths: its A_AUC range is only 0.33 point. That suggests the remaining deficit is not specific to narrow or broad Gaussian overlap.

H8's optional propagation inference reached 42.29, 43.66, and 44.00 A_AUC at sigma 3, 10, and 20 respectively. It stores no extra state and uses no evaluation labels, but each checkpoint performs ten propagation iterations on each supplied task-specific evaluation graph. Because this graph partition exposes benchmark evaluation structure and adds 4,550 graph iterations per run, it should only be compared under an explicitly standardized transductive-inference protocol.

## What the experiments show

### 1. Neighborhood context is necessary for useful graph replay

H1 is the cleanest causal result. Isolated and contextual arms had the same 100 stored examples, 11,990 replay seed rows, 1,200 optimizer steps, EMA, and stream. Restoring sampled neighborhoods improved A_AUC from 36.11 to 38.81 and AF_s from -20.12 to -12.91.

The gain costs compute. Across sigmas, isolated replay processed 23,980 source-node occurrences and 23,980 edges per run. Context replay processed roughly 379,000–404,000 source-node occurrences and 560,000–607,000 edges. This is a compute-for-information trade rather than hidden memory expansion.

### 2. Class coverage helps once replay preserves graph structure

CBRS improved contextual replay by 0.72 point and improved every sigma. This differs from the small ER-to-ER-CBRS gap in the historical isolated-replay results. The combined evidence supports an interaction: balanced coverage becomes useful when retained nodes are replayed with neighborhoods that make them meaningful to the GCN.

### 3. Classifier-scale control produced the largest gain

Fixing all classifier rows at the initialization-time mean norm added 3.23 A_AUC points, the largest improvement in the study. The current-mean variant lost 3.65 points and the half-scale variant did not improve results. The mechanism therefore appears to be an absolute, stable logit scale together with equal class-row norms, rather than row equality alone.

A plausible explanation is that sequentially observed classes otherwise receive unequal weight norms and therefore unequal logit margins. Projection prevents norm growth from encoding arrival order. It adds no parameters, examples, labels, or optimizer steps.

### 4. The bottleneck is early acquisition, not only retention

Final accuracies were generally 70–76%, but checkpoint-averaged accuracy remained near 43%. A constructed causal reference makes this visible. If a system scored 0% on unseen classes and became perfect immediately when each class first appeared, its A_AUC would be:

| Sigma | Immediate-perfect causal reference A_AUC |
|---:|---:|
| 3 | 51.57 |
| 10 | 51.81 |
| 20 | 53.10 |
| **Mean** | **52.16** |

This is a reference, not a strict mathematical upper bound because forward transfer could yield nonzero unseen-class accuracy. Still, a score above 50 would require about 96% of this reference. H4 reaches about 82%. Its accuracy grows gradually from roughly 7% initially to the low-to-mid 70s, so the main opportunity is to learn newly arriving classes much faster without damaging old ones.

### 5. More replay computation is not automatically better

Full-neighborhood replay was worse than sampled replay despite processing roughly 0.61–0.69 million replay source-node occurrences and 0.93–1.08 million replay edges. Degree-prioritized CBRS was even more expensive because hubs generated large neighborhoods: about 1.25 million source-node occurrences and 2.13–2.15 million edges per run. It improved AF_s but reduced A_AUC. This rejects the idea that larger contextual reach alone closes the gap.

## Resource accounting

The retained buffer contains 100 node IDs and 100 labels, or 1,600 bytes with 64-bit values. The EMA keeps one additional GCN parameter copy: 2,247,680 parameters or 8,990,720 bytes in float32. The classifier projection adds no state. Degree-CBRS adds 800 bytes of degree metadata and was rejected.

All ordinary variants retain one optimizer update per incoming minibatch and 11,990 replay seed rows over the 1,200-step run. H4's contextual replay processed 352,836–410,725 replay source-node occurrences and 518,655–623,027 replay edges depending on sigma. These counters make the added graph compute explicit even though persistent replay capacity remains fixed.

## Limitations and status

- The study contains one validation learner seed. Small differences, especially H2's +0.72 and H8's +0.58, need replication before they are treated as reliable. Repeated same-configuration CUDA runs varied by several tenths of a point.
- The regenerated split prevents direct numerical comparison with older local runs. Every H1–H15 comparison in this report uses the new split and the same runtime.
- H8 depends on task-specific evaluation graph partitions and is not the recommended default until the benchmark defines that inference allowance uniformly.
- The experimental learner and harness are implemented and unit-tested, but the retained H4 configuration has not yet been wired into the repository's main benchmark method registry.
- Arxiv-CL remains a follow-up. CoraFull was intentionally handled first.

## Recommended next experiment

Freeze H4 as the current training baseline and replicate it on validation seeds before opening test data. The next mechanism should target fast class acquisition rather than add replay volume. A defensible direction is an online classifier update that uses the current minibatch and the same CBRS-100 memory to solve or approximate the linear head more rapidly while retaining the fixed row-norm constraint. It must preserve one stream pass, the 100-example memory, the 1:1 replay ratio, and task-free operation. Any candidate should first beat H4 by at least 0.5 validation point across multiple seeds; only then should the nine-run sigma-by-seed test aggregate be evaluated.

## Reproducibility pointers

- Scientific protocol and hypothesis declarations: [PLAN.md](PLAN.md)
- Experiment runner: `experiments/scientific_gaussian.py`
- Experimental learner: `Baselines/context_ema_model.py`
- Degree-balanced buffer: `Baselines/replay_buffers.py`
- Unit tests: `tests/test_context_ema.py`
- Raw local run artifacts: `results_gaussian_science/h1` through `results_gaussian_science/h15` (ignored by Git because they include bulky per-checkpoint output)
