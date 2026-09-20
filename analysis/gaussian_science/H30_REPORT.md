# H30: reliable contextual teacher feedback

## Question

Can an EMA preserve knowledge through training feedback, allowing the fast working network to retain graph classes without relying on slow inference averaging? This is a transferable-rule experiment: both datasets use the same implementation and constants.

The predeclared protocol is in [H30_PLAN.md](H30_PLAN.md). Implementation and plan were committed as `286849d` before the paired pilot finished. Existing baseline files remain unchanged.

## How the model works

The learner retains the benchmark two-layer GCN, a 100-entry class-balanced reservoir of node IDs and labels, and one non-trainable EMA copy. Each update processes ten incoming labeled nodes and draws ten buffered nodes. Replay neighborhoods are sampled from the graph already available to the benchmark; their labels are never read. Teacher and student process exactly the same replay blocks.

Let student logits be z and EMA teacher logits be u, with stored replay label y. The teacher is trusted only when argmax(u)=y and softmax(u)[y] > softmax(z)[y]. Both conditions are detached from gradients. New classes and poor teacher predictions therefore do not automatically become targets.

With a binary gate g per replay row and temperature T=2, the extra loss is:

```text
distillation = mean_over_all_replay_rows(g * T^2 * KL(softmax(u/T) || softmax(z/T)))
loss = incoming_cross_entropy + replay_cross_entropy + distillation
```

The denominator includes ungated rows. No coefficient was selected from a dataset sweep: the three terms have coefficient one. After one Adam update, the working classifier rows are projected to their initialization-time mean norm. The teacher follows EMA decay .995. The working GCN supplies primary predictions; the teacher does not choose inference results.

This is a modification of the prior contextual replay learner inspired by [CLS-ER](https://arxiv.org/abs/2201.12604), not a claim to have invented teacher consistency. Its specific proposed benefit is conditioning feedback on reliable predictions with graph context preserved.

Additional literature checked during the Arxiv pilot gives counterevidence: [Soutif-Cormerais et al., Appendix A.4](https://proceedings.mlr.press/v232/soutif-cormerais23a/soutif-cormerais23a.pdf) found that EMA distillation gains did not transfer robustly across their image datasets. They suggest feedback may reduce ensemble diversity. That result concerns a different formulation, but reinforces the need for the paired cross-dataset gate rather than treating a Cora gain as sufficient.

## Fairness and verification

Both arms use the same stream, seed, sampled replay policy, backbone, optimizer, 100 entries and one optimizer update per incoming batch. The control disables only teacher feedback. No dataset name, Gaussian width, latent task boundary, or future label enters the learner. The extra cost is a no-gradient teacher forward on existing replay blocks, plus the EMA copy already present in the control. Full costs are stored per run.

Nineteen relevant unit tests passed, including gate behavior, teacher detachment, exact disabled-control equivalence through buffer overflow, same block identities, and bounded replay state. Raw results include split/source fingerprints and protocol assertions.

The [DRIFT paper](https://arxiv.org/pdf/2605.12998) supplies the online GCN/buffer protocol. Contextual replay uses more graph computation than the original isolated-node replay, so its cost must accompany comparisons. This experiment holds that contextual computation identical between its causal arms.

## Validation pilot results

Full stream, learner seed4, stream seed1. These are development results, not final test claims.

| Dataset / width | Arm | A_AUC | AF_s | Final accuracy | Training seconds |
|---|---|---:|---:|---:|---:|
| CoraFull / 10 | Feedback off | 35.4468 | -35.0644 | 49.2677 | 38.05 |
| CoraFull / 10 | Reliable teacher | 36.1985 | -31.5453 | 55.1515 | 49.66 |
| Arxiv / 60 | Feedback off | 29.8392 | -41.5797 | 43.3979 | 275.92 |
| Arxiv / 60 | Reliable teacher | 28.2630 | -39.5913 | 46.2177 | 332.88 |

Cora feedback adds .7517 A_AUC and improves AF_s by 3.5190 points. The gate accepts 1,345 of 11,990 replay rows (11.22%); its mean gated temperature-scaled KL is .2982 per replay row. Arxiv feedback **loses 1.5762 A_AUC** and improves AF_s by 1.9884 points. The Arxiv gate accepts 35,868 of 101,710 replay rows (35.26%) at mean KL .2028. All budget assertions pass on all four runs.

## Decision: H30 is rejected

The predeclared gate required feedback to improve A_AUC by at least .5 point on **both** datasets. Arxiv A_AUC fell by 1.5762, so the criterion fails and the plan forbids coefficient tuning on this branch. H30 is rejected as declared. No sweep, no re-specification, and no full width/seed sweep was run.

The direction of the two deltas is consistent and worth carrying forward: gated teacher feedback improved AF_s on **both** datasets (+3.5190 Cora, +1.9884 Arxiv) while costing acquisition on the longer stream. That matches the counterevidence cited during the pilot from [Soutif-Cormerais et al., Appendix A.4](https://proceedings.mlr.press/v232/soutif-cormerais23a/soutif-cormerais23a.pdf). It also matches the repository's recurring pattern: every retention mechanism tried so far pays for retention with acquisition.

### Caveat on how much this result can be generalized

The Arxiv arm was run on a base learner that is itself known to be poor on Arxiv. The H30 control scores 29.84 A_AUC, whereas the repository's plain reproduced ER scores 35.6 on the same dataset and width. Two components of the Cora-derived base are independently documented to hurt Arxiv: CBRS costs 5.4 A_AUC there relative to uniform reservoir, and fixed classifier norms were catastrophic when combined with Arxiv MAS (H25). The negative Arxiv delta therefore falsifies "gated teacher feedback added to this base transfers", which is what was declared, but it does not establish that functional feedback is useless on Arxiv in general. Any future test of that broader claim needs a base that is not already crippled on the dataset.

## Reproduction

```powershell
.\.conda\science\python.exe -m unittest tests.test_context_teacher tests.test_context_ema tests.test_mas_geometry -q
.\.conda\science\python.exe experiments/h30_context_teacher.py --dataset CoraFull-CL --sigmas 10 --seed 4 --arm paired
.\.conda\science\python.exe experiments/h30_context_teacher.py --dataset Arxiv-CL --sigmas 60 --seed 4 --arm paired
```

Raw artifacts: `results_h30/<dataset>/validation/seed4_stream1/runs/`. Final-test mode remains locked. Prior studies already opened the repository test nodes; any future confirmation with fresh seeds must disclose that it is new stochastic replication, not an untouched node holdout.
