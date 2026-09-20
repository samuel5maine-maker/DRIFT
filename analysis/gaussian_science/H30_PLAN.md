# H30: reliable contextual teacher feedback

Declared 2026-09-20 before implementation or H30 outcomes. Branch: `Codex/h30-contextual-teacher`, parent `8dba7cf`.

## Objective and evidence

One identical learning rule and hyperparameter configuration must exceed 44 A_AUC and -14 AF_s (closer to zero) on each dataset, equally averaging drift widths and seeds. A dataset-specific switch between replay and MAS is not a solution.

Prior evidence: contextual replay improved Cora A_AUC by 2.69 points, fixed working classifier norms by 3.23; EMA helped retention but delayed acquisition. H23 reached 44.25/-14.00 on Cora test and failed to transfer to Arxiv (28.81/-40.82 validation). Slowing Arxiv EMA reduced forgetting but failed acquisition. MAS used a different rule and reached 42.23/-17.03 test. Hard asymmetric CE and full new-class row synchronization failed acquisition. Global inference blending damaged retention. Historical CLS-ER consistency operated on isolated replay nodes, so its weak contribution does not establish that contextual feedback is useless.

Sources: [DRIFT](https://arxiv.org/pdf/2605.12998), [CLS-ER](https://arxiv.org/abs/2201.12604), [Strong Experience Replay](https://arxiv.org/abs/2305.13622). These motivate functional preservation; they do not establish effectiveness of this proposed variant.

## Hypothesis and mechanism

An EMA used only at inference cannot prevent the working model from overwriting old graph decision boundaries. Feed reliable EMA predictions back during contextual replay. Gate a replay row on the teacher's correct argmax AND strictly higher probability for the buffered ground-truth label than the student. Compute the gate without gradients. Distill only those rows using T=2 KL(teacher || student), multiplied by T squared, averaged over all replay rows (including zero gated rows). Loss: incoming CE + replay CE + gated KL. Coefficient 1; no coefficient search.

Teacher and student use the identical sampled blocks. Teacher has no gradients; no neighborhood labels are consumed. EMA decay remains 0.995; working classifier rows keep the fixed initialization mean norm. One Adam update, lr .005, decay .0005, incoming batch10, CBRS100, replay10, existing 2-layer GCN and fanouts10/25. No latent tasks, width, total stream length, task boundaries, test labels, stored logits, extra trainable parameters, or extra replay seeds. A single EMA copy and one additional no-gradient replay forward are disclosed costs. Primary inference is the working network: fast acquisition is part of the hypothesis, not a post hoc output choice. EMA metrics may be diagnostic only.

## Falsifiable experiment

Pilot: full-length validation runs, learner seed4, stream seed1, Cora sigma10 and Arxiv sigma60. Paired control uses exactly the same contextual learner with feedback disabled and working inference. Do not reuse historical controls across source/seed changes. Record replay teacher gate fraction, distillation loss, graph work, runtime and budgets.

Advance only if feedback improves A_AUC by at least .5 point on both datasets, does not worsen AF_s by over .5 point on either, and keeps every budget assertion. Otherwise reject or explicitly revise the mechanism on a NEW branch, citing diagnostic evidence; do not tune coefficients on the same branch. A successful pilot is not a claim that the absolute targets are met.

If promising: validation learner seeds4/5/6, fixed stream1, Cora widths3/10/20 and Arxiv widths9/30/60, same rule everywhere. Arxiv9/30 are an extension scaled in the same proportions as the paper's Cora sweep; the paper reports only Arxiv60, so these are not claimed as official widths. Freeze all source/configuration before confirmation. Final confirmation uses learner/stream seed pairs101/101,102/102,103/103 with identical width sweeps, and reports per-dataset mean/std plus each width, final accuracy, AF_s, and causal control deltas.

The repository test nodes were already evaluated in prior studies. New stochastic replicates are held out from this cycle but are NOT a new untouched node test set; disclose this limitation. No repeated final-test selection. If final confirmation fails, report the failure and return to development without presenting later reuse as independent confirmation.

## Implementation plan

1. Add a small separate learner inheriting contextual replay; leave existing baselines unchanged.
2. Add a separate H30 runner reusing benchmark data/evaluation with explicit widths, seeds, working inference and full fingerprints.
3. Unit-check teacher detachment, gate behavior, same-block use, disabled-control equivalence, one-step/buffer limits and absence of unseen-label access.
4. Run paired pilots sequentially on the GPU, inspect outcomes and decide before full sweeps.
5. Record results and an explanation of the model; commit a reproducible, reversible hypothesis record.
