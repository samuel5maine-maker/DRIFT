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
