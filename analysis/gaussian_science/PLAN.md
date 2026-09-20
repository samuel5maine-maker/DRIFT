# Scientific plan: CoraFull Gaussian A_AUC > 50

Branch: `Codex/gaussian-scientific-50`. This branch was created before source changes. Raw results stay ignored; compact evidence and conclusions will be committed.

## Fixed protocol

The experiment follows DRIFT's CoraFull Gaussian protocol: sigma 3, 10, and 20; batch size 10; 2-layer GCN with hidden width 256; Adam at 0.005; one optimizer update per incoming batch; and at most 100 buffered nodes. Latent task identity, Gaussian weights, sigma, boundaries, future labels, validation labels, and test labels are unavailable to the learner. The fixed stream seed is 1. Validation uses learner seed 0; final tests use seeds 1–3 only after a design is frozen. Selection maximizes the equal-sigma mean validation A_AUC.

The original ignored data directory disappeared between sessions. The repository was restored from its recorded user-owned remote. The CoraFull split was regenerated once with the repository's documented seed 1 and is now frozen. Its SHA-256 is `d9dcf71d86bfd4345f9d382164c142d9c4186109c53e61f06febd16294fd8027`. It differs from the older laptop-era split used by prior local tables, so every scientific comparison here must be rerun on this split. Historical scores are observations that motivate hypotheses, not matched controls.

## Observations

1. Historical Cora sigma-20 A_AUC rises from ER 32.8 and ER-CBRS 33.3 to approximately 38–39 for EMA/distillation methods. Stabilizing parameters helps, while larger composite losses have not produced additive gains.
2. The previous isolated-branch study achieved 36.81 mean A_AUC over sigma 3/10/20 with CBRS plus one EMA. Validation selected rho=1, rejecting asymmetric stream-logit attenuation. Its per-sigma A_AUC was 31.52, 38.09, 40.82.
3. The shared replay implementation deliberately turns buffered graph nodes into isolated self-loops. Incoming nodes, in contrast, train through sampled neighborhoods. Thus replay changes both the temporal distribution and the input representation.
4. DRIFT's offline joint upper bound is 86.3 on CoraFull, so the target is not ruled out by the backbone. The gap is in online optimization and retention.

## Hypothesis cycle

### H1: relational replay

Hypothesis: much of replay's weakness comes from removing neighborhood context. Replaying the same ten buffered seed nodes through neighborhoods sampled from the already-available transductive training graph should improve validation A_AUC without increasing buffer capacity, replay seed count, forward count, or optimizer steps.

Experiment: compare `isolated` and `context` replay using the same reservoir policy and EMA alpha 0.995 across all three sigmas on validation seed 0. The context arm may read features/topology from the same merged graph used for every incoming update; it may read labels only for the ten buffered seeds. Record sampled source nodes and edges to make the compute difference explicit. Retain context only if the equal-sigma mean improves by at least 1 A_AUC point.

### H2: temporal representation versus class coverage

Conditional on H1 succeeding, compare reservoir sampling with CBRS under the retained replay representation. Reservoir approximates the stream distribution; CBRS protects rare classes but, with 70 classes and 100 slots, often leaves only one example per class. Retain CBRS only if it improves equal-sigma validation A_AUC by at least 0.5 point; otherwise prefer reservoir for lower policy complexity.

### H3: stability timescale

Conditional on the replay choice, compare one constant EMA with a task-free adaptive EMA driven only by observed minibatch-label turnover. The hypothesis is that a faster average helps during label novelty and a slower average helps during recurrence. The matched control and candidate keep one EMA parameter copy. Reject the adaptive rule unless it improves all-sigma mean validation A_AUC by at least 0.5 point.

### H4: classifier row-norm bias

Conditional on H3, test whether concentrated gradients for newly arriving classes create classifier row-norm imbalance and recency bias. After the existing optimizer step, project every final classifier row to the common mean norm measured at initialization. This adds no parameters, samples, labels, forward pass, optimizer step, or task signal. Reject it unless equal-sigma validation A_AUC improves by at least 0.5 point.

### H5: replay strength after geometric stabilization

If H4 succeeds, test whether an equal replay-loss coefficient over-constrains acquisition after classifier geometry already limits recency bias. Keep the same ten replay seeds and one update, but compare replay coefficient 0.5 to the retained coefficient 1.0. Reject 0.5 unless it improves equal-sigma validation A_AUC by at least 0.5 point.

### H6: preserve learned global classifier scale

If fixed row normalization succeeds, test whether fixing its common norm at random initialization unnecessarily suppresses the global logit scale. Project rows to their current mean norm instead: class-specific norms remain equal, while the optimizer may learn a shared scale. This has identical parameters and compute. Reject it unless validation A_AUC improves by at least 0.5 point.

### H7: buffer-derived prototype inference

If H6 fails, test whether hidden representations become discriminative before parametric classifier rows converge. At each evaluation checkpoint, recompute class means from current EMA embeddings of the same buffered nodes, then classify with cosine similarity. Store no prototypes or additional samples; use no labels beyond the buffer. Evaluate the ordinary and prototype heads from the identical training run. Count one full training-graph inference per checkpoint. Reject prototypes unless validation A_AUC improves by at least 0.5 point.

### H8: buffer-seeded graph label propagation

If prototypes fail, test whether the graph itself can carry the sparse buffer labels more rapidly than gradient updates. At evaluation only, clamp labels for buffered nodes present in the supplied transductive graph and run ten parameter-free mean-aggregation iterations. Never read evaluation labels into propagation. Fall back to parametric predictions for nodes receiving zero propagated mass. Store no additional state and report the inference passes. Reject unless validation A_AUC improves by at least 0.5 point, and disclose dependence on the benchmark's transductive task evaluation graphs.

### H9: fixed classifier temperature

H6 tests and rejects unconstrained growth of the common classifier norm. Test the opposite explanation: a smaller fixed norm may keep softmax gradients useful longer during one-pass learning. Project to 0.5 times the initialization mean norm after each existing update. This has no effect on argmax solely through rescaling; any change comes from training geometry. Reject unless validation A_AUC improves by at least 0.5 point.

### H10: remove replay neighborhood sampling noise

The retained context arm samples replay neighborhoods with DRIFT's `[10,25]` fanout. Test full-neighborhood replay for the same ten buffered seeds in the same single replay forward and optimizer update. Incoming nodes keep the benchmark sampler. This increases measured replay edges but not buffer entries, replay seeds, updates, or model state. Reject unless validation A_AUC improves by at least 0.5 point.

### H11: cosine classifier geometry

H4 shows that class-specific weight norms are harmful. Test the complete geometric hypothesis by normalizing both final classifier rows and the final aggregated node representations, then applying a fixed scale of 10. Keep the same two GCN message-passing layers and parameter count. Reject unless validation A_AUC improves by at least 0.5 point.

### H12: hard asymmetric incoming loss

The earlier finite attenuation was tested before relational replay and classifier normalization. Test the original hard asymmetric principle in the retained model: for incoming cross-entropy only, exclude previously observed classes absent from the current minibatch; keep replay cross-entropy over the full seen-class head. Use only delivered labels. Reject unless validation A_AUC improves by at least 0.5 point.

### H13: online balanced-softmax pressure

If excluding old classes fails, test the opposite correction. Maintain cumulative counts from delivered labels only. During incoming loss, add log(count) to logits for observed classes, leaving never-observed rows at zero; replay loss remains ordinary. Frequent old classes then exert more denominator pressure, requiring new class rows to learn competitive raw logits without removing old competition. Report the 70 integer counters as state. Reject unless validation A_AUC improves by at least 0.5 point.

### H14: topology smoothness from existing blocks

H1 shows useful signal in graph context. Test whether a small unlabeled representation-smoothness term can extract it earlier: on edges already sampled for the incoming GCN forward, minimize one minus cosine similarity between endpoint hidden states with weight 0.1. Add no labels, samples, forwards, parameters, or task signal. Reject unless validation A_AUC improves by at least 0.5 point.

### H15: degree-representative class-balanced memory

Random CBRS may retain poorly connected exemplars. Preserve class balancing but, within the required 100-node capacity, retain higher-degree delivered nodes and evict the lowest-degree exemplar from an overrepresented class. Degree comes from the same merged training graph. Persist one 64-bit degree per slot (800 bytes at capacity), disclosed separately. Replay ten seeds unchanged. Reject unless validation A_AUC improves by at least 0.5 point.

## Finalization

Freeze the best fully specified design and its ablation before opening test results. Run the candidate and matched control for seeds 1–3 at all three sigmas. Report all curves, per-sigma means, equal-sigma aggregate, signed AF_s, final accuracy, memory, replay seeds, sampled graph work, runtime, and every rejected hypothesis. A_AUC > 50 must hold on the nine-run frozen-test aggregate to claim success. If it does not, report the negative result and continue with a new mechanism rather than tuning on test data.

## Second cycle: dual-dataset target

Branch: `Codex/dual-dataset-scientific`. The new objective is stricter than the first cycle: a candidate must exceed H4's **42.75 equal-sigma validation A_AUC** and improve its **-13.46 AF_s** (move closer to zero). The CoraFull split, stream seed, learner seed, sigmas, buffer-100 limit, ten replay seeds, one optimizer update, one stream pass, latent-task restriction, and evaluation cadence remain frozen. Test data remain sealed. A surviving CoraFull mechanism will be transferred without dataset-specific tuning to Arxiv-CL; Arxiv validation will use the paper setting, Gaussian sigma 60, before any test run.

### Observation after H1-H15

H4's final accuracy is 70.35--75.51%, but its checkpoint average is only 42.75%. Its pooled curves rise gradually from 7% to the low-to-mid 70s. Context replay and fixed classifier geometry produced the two largest gains; increasing replay neighborhood size, preferring graph hubs, and adding topology smoothness failed. The evidence points to slow acquisition of classifier directions rather than too little replay compute. Because H4 projects classifier rows back to a fixed norm after every step, radial gradient is discarded and the remaining angular update may be too small at the shared learning rate.

### H16: faster classifier-direction learning

Give only the existing final GCN classifier layer a 2x learning-rate multiplier. The encoder remains at 0.005, weight decay is unchanged, classifier rows are projected to the same fixed initialization norm after the same single Adam step, and all data/memory/compute constraints remain unchanged. The multiplier is fixed at 2 before observing results; no grid is permitted. Accept only if the equal-sigma CoraFull validation result has both A_AUC > 42.75 and AF_s > -13.46. Otherwise reject the mechanism and diagnose whether the failure is acquisition or retention from its checkpoint curve.

**Result:** rejected. Mean A_AUC fell to 41.85 and AF_s to -16.21. At sigma 20, final accuracy fell to 66.21% and AF_s to -21.00. Increasing classifier angular step size causes late instability, especially under broad overlap; classifier speed is not the missing acquisition mechanism.

### H17: adaptive EMA after classifier stabilization

H3's overlap-driven EMA reduced forgetting from H2's -13.66 to -12.41 but did not improve A_AUC before classifier normalization. H4 subsequently changed the geometry and may remove the instability that prevented the faster average from helping. Combine H4 with the exact H3 rule: EMA alpha 0.990 when consecutive delivered-label sets do not overlap, increasing linearly to 0.995 at full overlap. The signal uses delivered labels only, keeps one EMA copy, and adds no forwards or updates. Accept only if both A_AUC > 42.75 and AF_s > -13.46.

**Result:** rejected under the joint objective. A_AUC was 42.76, but AF_s worsened to -14.18. The sigma-20 A_AUC rose to 43.21 while its AF_s fell to -16.17, indicating that a more plastic inference average trades late stability for acquisition.

### H18: fixed working/EMA inference blend

Test whether the working model contains useful recent-class signal that the slow EMA delays, while the EMA can suppress most working-model noise. From the identical H4 training run, blend logits as 75% constant-alpha EMA and 25% working network. The weight is fixed before results and no grid is permitted. The two networks already exist in H4, so this adds no persistent parameters, replay, labels, training forwards, or updates; it adds one full GCN inference per evaluation graph and will be reported. Compare ordinary EMA and blended predictions from the same run. Accept only if the blend has A_AUC > 42.75 and AF_s > -13.46, and improves A_AUC by at least 0.5 over the same-run EMA control.

**Result:** rejected under the joint objective. The blend raised A_AUC to 44.18 but worsened AF_s to -16.44. It raises intermediate task peaks without preserving them at the final checkpoint. The same-run EMA control was 42.74/-13.31.

### H19: first-observation EMA classifier-row synchronization

The H17/H18 results localize the stability--plasticity conflict. A globally faster or partially live inference model learns recent classes sooner but forgets more. Under H4, each new class's working classifier row receives a supervised update immediately, while the inference EMA incorporates only 0.5% of that row per step. On the first delivered example of each class, copy only that class's final classifier weight row from the post-update working network into the EMA; all encoder parameters, old rows, and subsequent updates retain alpha 0.995. Also copy the corresponding bias element if the backbone has one. First observation is derived from delivered labels, with no task identity or boundary. The existing observed-class set is the only state. Accept only if both A_AUC > 42.75 and AF_s > -13.46.

**Result:** rejected for A_AUC. Hard synchronization produced 41.49 A_AUC and -11.78 AF_s. It improved retention but reduced acquisition/final accuracy, consistent with a working classifier row becoming geometrically mismatched to the slow EMA encoder.

### H20: partial first-observation row synchronization

Revise H19 by interpolating 5% of the post-update working classifier row into the EMA row on first observation, rather than replacing it. This is ten times the ordinary 0.5% EMA incorporation but retains 95% of the row aligned with the EMA encoder. The strength is fixed before results and no sweep is permitted. All fairness and acceptance conditions remain unchanged: A_AUC > 42.75 and AF_s > -13.46.

**Result:** rejected for A_AUC. Partial synchronization reached 42.40 A_AUC and -12.91 AF_s. Both hard and partial synchronization improve the forgetting metric while reducing checkpoint-average acquisition, so this line is closed without tuning its strength further.

### H21: normalize the inference EMA classifier

H4 normalizes working classifier rows, but its reported predictions come from the EMA. An average of equal-norm vectors generally has a smaller norm, with shrinkage determined by each row's age and directional movement. EMA averaging can therefore recreate the class-dependent norm bias that H4 removes from the working model. After every ordinary EMA update, project every EMA classifier row to H4's same fixed initialization-time norm. This adds no parameters, labels, task signal, forward pass, or optimizer step. Accept only if both A_AUC > 42.75 and AF_s > -13.46.

**Result:** rejected for A_AUC. Normalizing the inference head reached 41.48 A_AUC and -11.89 AF_s. EMA norm shrinkage appears to encode useful class-age confidence even though removing it improves the forgetting metric.

### H22: class-recency-gated working/EMA blend

H18 shows that working-model logits improve acquisition but harm old-class retention when applied globally. Use the same fixed 25% working / 75% EMA blend only for output columns whose class has appeared in the last 100 optimizer updates; use pure EMA logits for older classes. The 100-update horizon equals the pre-existing evaluation interval and is fixed without a sweep. Recency comes only from delivered labels, not task boundaries. Store one 64-bit last-seen step per observed class (at most 560 bytes), add no training forward or update, and report one extra full-graph inference at evaluation. Compare against the same-run pure EMA control. Accept only if A_AUC > 42.75, AF_s > -13.46, and A_AUC improves by at least 0.5 over the same-run control.

**Result:** rejected narrowly for retention. The gated blend reached 43.75 A_AUC and -13.75 AF_s. It exceeded both targets at sigma 3 and 10, while sigma 20 AF_s was -17.65. Relative to its same-run EMA, it gained 1.37 A_AUC but lost 0.78 AF_s.

### H23: lower-strength class-recency blend

The H22 direction is successful for acquisition but its 25% working contribution overshoots the retention constraint. Perform one fixed bisection to 12.5% working / 87.5% EMA on the same 100-update class gate. No further strength search is permitted. Training, state, inference accounting, and acceptance criteria are identical to H22. Accept only if A_AUC > 42.75, AF_s > -13.46, and A_AUC improves by at least 0.5 over its same-run EMA control.

**CoraFull result:** accepted. H23 reached 43.52 A_AUC and -13.16 AF_s, versus 42.37/-12.97 for its identical-run pure-EMA control. It clears both original thresholds and gains 1.15 A_AUC from inference gating.

**Direct Arxiv transfer:** rejected. At the paper's sigma 60, the pure-EMA control reached 28.70/-40.70 and the H23 gate reached 28.81/-40.82. CoraFull has about 34 batches per latent two-class task, while Arxiv has about 509; alpha 0.995 spans roughly four Cora task widths but less than one-third of an Arxiv task width.

### H24: stream-scale-normalized EMA on Arxiv

Preserve H4's decay over a normalized latent class-pair span. Solve `alpha_arxiv^(10172/20) = 0.995^(1200/35)`, giving alpha approximately 0.99966. This is a constant dataset-level hyperparameter analogous to the dataset-specific EMA settings already used by benchmark baselines; the learner receives no boundary or task identity. Keep H23's classifier normalization, CBRS-100, context replay, 12.5% recency gate, and 100-update horizon unchanged. Compare gated and pure-EMA predictions from the same run. The Arxiv target remains A_AUC > 42.75 and AF_s > -13.46.

**Result:** the timescale hypothesis is accepted for retention but rejected overall. Pure EMA reached 30.16 A_AUC and -13.41 AF_s, versus 28.70/-40.70 at alpha 0.995. The recency gate reached 32.34/-22.49. Slow averaging can meet retention but cannot acquire Arxiv classes quickly enough.

### H25: fixed classifier geometry on Arxiv MAS

The repository's reproduced Arxiv MAS* result (42.6 A_AUC, -14.3 AF_s) is much closer to both targets than replay/EMA. H4 established that fixed classifier-row norms can remove sequential class bias without extra state or updates. Apply the same post-update projection to the existing task-free MAS learner, leaving its optimizer, importance estimates, consolidation rule, parameter anchors, and zero replay memory unchanged. This is a targeted cross-component test, not a change in buffer or training passes. Validate at sigma 60 seed 0 on the same regenerated split. Accept only if A_AUC > 42.75 and AF_s > -13.46.

**Result:** rejected decisively. Fixed norms produced 16.80 A_AUC and -79.55 AF_s. MAS relies on unconstrained classifier magnitude in its optimization/consolidation dynamics, so H4's intervention is not portable to this learner.

### H26: regenerated-split MAS control

Run the original task-free MAS rule without classifier projection through the identical Arxiv validation harness. This is a matched control rather than a candidate intervention. It establishes whether the older 42.6/-14.3 test result transfers to the regenerated split and seed-0 validation protocol before a new MAS hypothesis is formed.

**Result:** accepted as the frozen Arxiv configuration. The unmodified rule reached 45.02 A_AUC, -7.90 AF_s, and 85.92% final accuracy. It uses no replay buffer, 10,172 optimizer steps for 10,172 stream batches, and performed 148 task-free importance consolidations. Since it already clears both targets and H25 shows that importing the Cora geometry is harmful, no further Arxiv tuning is justified before test.

## Frozen dual-dataset evaluation

The scientific result is two dataset-specific configurations rather than a claim that one intervention transfers universally:

- **CoraFull-CL:** H23 — sampled context replay, CBRS-100, one replay seed per incoming row, fixed working-classifier row norm, constant EMA alpha 0.995, and a 12.5% working-logit blend limited to classes delivered in the last 100 updates.
- **Arxiv-CL:** H26 — the repository's unmodified task-free MAS training rule, with no replay memory and no classifier projection.

Both configurations were selected on validation seed 0. Freeze all implementation and hyperparameters, then evaluate learner seeds 1--3 on the test split. CoraFull uses sigmas 3, 10, and 20; Arxiv uses the paper setting sigma 60. Report per-seed and aggregate A_AUC/AF_s. Do not revise either configuration from test outcomes.

**Frozen test outcome:** neither configuration meets both aggregate targets. CoraFull H23 reached 44.25 A_AUC and -14.00 AF_s over nine runs. Arxiv MAS reached 42.23 A_AUC and -17.03 AF_s over three runs. These remain the final unbiased test results. Any subsequent experiments return to validation and are research candidates only; this test split cannot provide another clean confirmation.

### H27: combine Cora retention and acquisition mechanisms

H20's 5% first-observation row interpolation improved validation AF_s to -12.91 but left A_AUC at 42.40. H23's recency gate added 1.15 A_AUC to its same-run control. Combine the two already fixed mechanisms without changing either strength: H20 training plus H23's 12.5%/100-update inference gate. Accept on validation only if both targets are met; if seed 0 succeeds, assess validation learner seeds 1--3 without changing the design.

**Result:** seed 0 passed at 43.54/-13.12, but the four-seed validation aggregate was 43.69/-13.86. Acquisition generalizes; retention remains too seed-sensitive. Retain as a research candidate only.

### H28: stream-scale EMA over Arxiv MAS

Arxiv MAS has strong acquisition but seed-sensitive forgetting. Maintain one EMA copy of the same MAS network with alpha 0.99966, the stream-scale decay already justified in H24. MAS training, consolidation, zero replay memory, and one optimizer step remain unchanged. Evaluate the EMA directly. This adds one disclosed parameter copy but no training forward, replay sample, label, or optimizer update. Accept on validation only if both targets are met; if seed 0 succeeds, assess validation learner seeds 1--3 without changing the design.

**Result:** rejected for acquisition. The slow EMA reached 34.48 A_AUC and -4.98 AF_s. It stabilizes MAS strongly but lags too far behind the working network.

### H29: working/slow-EMA MAS ensemble

H26's working MAS reached 45.02/-7.90 on validation seed 0, while H28's slow EMA reached 34.48/-4.98. Blend logits with a fixed 75% working / 25% EMA weight, favoring acquisition while retaining a stabilizing contribution. The weight is fixed without a sweep. Both networks already exist in H28; this adds one evaluation forward but no parameters beyond H28, replay, labels, training forwards, or optimizer steps. Accept if both targets are met, then assess validation seeds 1--3 unchanged.

**Result:** rejected. The blend reached 42.46 A_AUC and -22.27 AF_s. It neither preserves working-model acquisition nor inherits slow-EMA retention.

## End state

No configuration met both targets across the frozen test seeds. H23 is the final unbiased Cora result and H26 the final unbiased Arxiv result. H27 is the strongest later Cora validation candidate but also misses the multi-seed retention threshold. Further work requires a new untouched test split.
