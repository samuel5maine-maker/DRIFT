# MAS* on DRIFT: faithful Algorithm 1, calibrated thresholds, consolidation telemetry

**Dataset and backbone:** CoraFull-CL, GCN. **Seeds:** calibration used seed 0; evaluation used seeds 1–3. **Runs:** 123 in total, all logged.

Generated tables are in [`report_tables.md`](report_tables.md) and calibration details in [`calibration_t1t2.md`](calibration_t1t2.md). Implementation choices are documented in [`../docs/mas_star_code_map.md`](../docs/mas_star_code_map.md).

## Bottom line

1. **DRIFT's own MAS* (`tfmas`) sometimes never consolidates, and then it is Bare.** Under 30% global mixing, `tfmas` and `bare` produce **bit-identical accuracy matrices on all three seeds**. That is the same pattern as the byte-identical MAS*/Bare rows in DRIFT's Table 4. For this setting at least, the published MAS* number measures Bare, not MAS*. Hypothesis D (thresholds miscalibrated for the loss scale) is confirmed for the legacy code in this setting.
2. **With thresholds calibrated to DRIFT's loss scale, a faithful MAS* consolidates steadily in every regime.** The count does not fall as transitions get smoother: roughly 28 consolidations under hard transitions, then 39, 43 and 38 at σ = 3, 10, 20. No run at the chosen thresholds starved, and none locked its latch.
3. **The consolidation mechanism helps a little, and it helps more as σ grows, not less.** Paired by seed, faithful MAS* beats the same method with consolidation turned off in **12 of 12** hard/Gaussian runs. The mean gain is +1.1, +1.8, +1.8 and +3.1 AAUC at hard, σ=3, σ=10 and σ=20.
4. **Most of MAS*'s benefit over Bare comes from the hard buffer, not from Ω.** The buffer alone adds +2 to +4 AAUC in the hard and Gaussian regimes, +17 at boundary K=5 and +29 at global 30%. Consolidation adds nothing measurable in those last two regimes.
5. **Starvation is real, but only at tight thresholds.** At δμ = 0.378 (5th percentile of the loss window mean), σ=20 never consolidates, while hard transitions still consolidate 7–9 times. "The detector fails under smooth drift" is therefore a statement about the threshold setting, not an intrinsic property of the detector.

## What was run

- **Method.** Algorithm 1 of arXiv:1812.03596v3, implemented line by line as `tfmas_star`. It has a 100-node hard buffer filled by prioritized keeping, Ω estimated per sample on that buffer and combined by cumulative moving average, a window of 5 cleared on consolidation, a σ threshold, and a latch initialized to P=0. Where the paper is ambiguous, the code map lists the interpretation chosen. Departures forced by DRIFT: Adam at lr 5e-3 instead of SGD, 1 pass per batch, class-IL head masking, and buffered nodes replayed as isolated nodes (DRIFT's ER convention).
- **Calibration** (seed 0, chosen without looking at accuracy):
  - *T1:* detector-off traces for the hard, σ=3, σ=10 and σ=20 regimes.
  - *T2:* an exact offline map of when the detector would first fire. This is exact because the detector has no side effects before its first firing.
  - *T3:* a 4×4 grid of percentile thresholds × {hard, σ=20}, 32 runs.
  - *Selection rule, written down before T3:* on the hard control, maximize the share of task segments that contain at least one consolidation. It picked **δμ = 2.858, δσ = 0.3732**, which covers 65.7% of segments.
- **Main matrix.** Six regimes (hard/`tfocis`, σ=3, σ=10, σ=20, boundary K=5, global 30%) × four arms (`tfmas_star`, `tfmas_star` with consolidation off, legacy `tfmas`, `bare`) × seeds 1–3, for 72 runs. A robustness slice adds 5 neighbouring threshold pairs × σ ∈ {3, 10, 20} on seed 1, including pairs beyond the grid corner.
- **Verification.**
  - 11 unit tests pass.
  - Turning telemetry on or off leaves AAUC/FM and every per-checkpoint accuracy bit-identical on real data.
  - The detector-off arm has Ω ≡ 0.
  - Runs are deterministic once DGL sampling is pinned to one thread.

## Detector telemetry (faithful MAS*, frozen thresholds, seeds 1–3)

| regime | consolidations | % steps latch armed | blocked by σ only | blocked by mean only | peaks / consolidations | median Ω-to-Ω cosine | α-velocity at consolidation vs random (p) |
|---|---|---|---|---|---|---|---|
| hard | 27–29 | 55 | 1.5–2.7% | 9.8–12.1% | 1.0 | 0.20–0.24 | n/a |
| σ=3 | 38–40 | 46–54 | 2.6–4.6% | 8.5–10.8% | 1.0 | 0.20–0.26 | 0.004 vs 0.057 (p=0.0002) |
| σ=10 | 39–46 | 53–58 | 11.5–12.8% | 0.5–2.2% | ~1.0 | 0.34–0.40 | 0.031 vs 0.053 (p=0.0001) |
| σ=20 | 33–46 | 76–77 | 20.5–24.5% | 0.3–1.1% | 1.0 | 0.53–0.58 | 0.037 vs 0.038 (p=0.11) |
| boundary K=5 | 35–40 | 38–44 | 10–14% | 0.5–1.5% | 1.0 | 0.38–0.46 | n/a |
| global 30% | 33–42 | 82–88 | 10–12% | 6–7% | 1.0 | 0.74–0.78 | n/a |

Most non-firing steps fail several conditions at once (38–76%). Figures 1–4 show the time courses.

### Reading against the four hypotheses

- **A. Starvation.** *Not observed at calibrated thresholds; present at tight ones.*
  - Consolidation count does not decline with σ, and the penalty is active on 95–98% of steps in every hard/Gaussian run.
  - A's mechanism is still visible: the share of steps blocked by **σ alone** rises monotonically from ~2% (hard) to ~22% (σ=20), while "mean only" blocking disappears. Under smooth drift the loss sits near its level threshold but wobbles more.
  - With a tight δμ that wobble is enough to starve σ=20 entirely (T3 sweep, Fig. 7).
- **B. Latch lock.** *Rejected.* Every consolidation is followed by a re-arming peak (peaks ≈ consolidations in every run). The latch is armed 38–88% of the time.
- **C. Mistiming.** *Rejected for σ ≤ 10; inconclusive at σ=20.*
  - At σ=3 and σ=10, consolidations happen at much lower mixing velocity than random steps (p ≤ 0.0002): the detector fires in stable stretches, as intended.
  - At σ=20 there is no difference (p=0.11). But velocity is uniformly low at σ=20, so the test has little to discriminate.
  - The lag from a dominant-task change to the next consolidation has median 14–25 steps, about half a task (~34 batches).
- **D. Threshold miscalibration.** *Confirmed for legacy `tfmas`; resolved for `tfmas_star` by calibration.*
  - Legacy thresholds are 0.2 on (CE + penalty) with a variance test, and the constants are tuples, typed with a trailing comma. That can only fire while the class-IL head is narrow.
  - Under global mixing, the non-class-IL `observe` path computes CE over all 70 classes, so the detector never fires and the run is bitwise Bare.
- **Fifth possibility (the buffer).** *This is where most of the performance lives.*
  - Ω estimates become **more** similar across consolidations as mixing increases (cosine 0.2 under hard transitions → 0.55 at σ=20 → 0.76 under global mixing). This is the opposite of the telemetry spec's expectation. Under hard transitions, each consolidation measures a different subnetwork on a buffer dominated by the latest hard examples. Under mixing, the buffer is a stable blend, so Ω accumulates consistently.
  - This is consistent with consolidation helping more at higher σ.

## Performance (AAUC %, mean ± std over seeds 1–3)

| regime | MAS* (faithful) | MAS* without consolidation | DRIFT `tfmas` | Bare |
|---|---|---|---|---|
| hard | 15.6 ± 2.6 | 14.5 ± 2.2 | 14.8 ± 1.1 | 10.3 ± 1.4 |
| σ=3 | 18.1 ± 1.4 | 16.4 ± 2.2 | 13.5 ± 2.0 | 13.0 ± 1.4 |
| σ=10 | 23.7 ± 0.9 | 21.9 ± 1.0 | **29.3 ± 2.8** | 19.8 ± 1.1 |
| σ=20 | 30.8 ± 2.4 | 27.7 ± 0.3 | 30.6 ± 2.6 | 24.8 ± 1.4 |
| boundary K=5 | 30.2 ± 2.0 | 31.4 ± 1.3 | 23.6 ± 0.7 | 14.0 ± 0.5 |
| global 30% | 37.6 ± 3.0 | 36.3 ± 0.4 | 7.6 ± 0.3 (= Bare) | 7.6 ± 0.3 |

- **Every method improves with σ.** Smoother mixing replays old tasks for free, so "MAS* degrades as transitions smooth" can only mean *relative to a baseline*. Relative to its own buffer-only ablation, faithful MAS* gains more at higher σ (Fig. 5).
- **Legacy `tfmas` beats faithful MAS* at σ=10**, by 1.8–8.2 AAUC on every seed. It also sits far above DRIFT's published 16.2 for the same cell. I have no explanation from the telemetry, because legacy `tfmas` is not instrumented; this is flagged below.
- AAUC is comparable within a regime, not across regime families: non-Gaussian pipelines add evaluation checkpoints at task changes.

**Threshold robustness** (seed 1, frozen pair and 5 neighbours):

| | AAUC across the 6 threshold pairs | consolidations |
|---|---|---|
| σ=3 | 15.8–19.8 | 32–41 |
| σ=10 | 22.9–27.4 | 28–46 |
| σ=20 | 29.3–35.8 | 17–70 |

- All 18 settings are at or above the seed-1 buffer-only arm (15.3 / 22.5 / 27.7), so the sign of the consolidation effect does not depend on the exact thresholds.
- At σ=20, the loosest pair (beyond the grid corner) consolidates 70 times and reaches 35.8. More consolidation is better under smooth drift.

## Caveats

- **Environment.** Runs used CPU with DGL 1.1.2 and torch 1.13.1, not the environment behind DRIFT's published numbers. Absolute AAUC differs from the published values: legacy `tfmas` at σ=10 gives 29.3 here against 16.2 published, while Bare is within about 3 points. Every arm was re-run here, so the comparisons between arms are like for like.
- **Scope.** Three seeds, one dataset, one backbone. The Arxiv-CL confirmation and the GAT (Table 4) run were not done.
- **Shared stream.** DRIFT's stream caches omit the seed, so all seeds share the seed-1 stream. Seeds vary initialization and neighbour sampling only.
- **Threshold choice.**
  - Thresholds were calibrated on the hard control only.
  - The selected pair is on the grid corner. The robustness slice suggests looser thresholds would score equal or better under the selection rule.
  - With these thresholds the first consolidation happens at step 1–8 in the hard, Gaussian and boundary regimes (step 122–317 under global mixing), while the window holds one or two losses. That is faithful to Algorithm 1, which sets no minimum window occupancy, but the first Ω estimate is taken almost immediately.
- **Interpretations.** Several paper ambiguities were resolved by interpretation: the value pushed to the window, eval-mode Ω, F = squared L2 of the output, and buffer scoring under the current θ. The code map lists them all; `window_push=both` is implemented but was not swept.

## Could not verify

- The CVPR camera-ready Algorithm 1, and the supplementary values of δμ, δσ and λ. With those it would be possible to test whether the original thresholds transfer.
- Whether MAS [1] defines F as squared or plain L2.
- Why legacy `tfmas` does well at σ=10. Instrumenting legacy `tfmas` with the same telemetry would show when and how often it consolidates.
- The DGL version and seeds behind DRIFT's published tables.

## Figures

1. [Detector against thresholds](figures/fig1_detector.png)
2. [Mixing curve with consolidations](figures/fig2_alpha.png)
3. [Penalty and ‖Ω‖₁](figures/fig3_penalty_omega.png)
4. [Blocking attribution](figures/fig4_blocking.png)
5. [Consolidations and AAUC vs σ](figures/fig5_money.png)
6. [Arm comparison](figures/fig6_arms.png)
7. [T3 threshold sweep](figures/fig7_threshold_sweep.png)
8. Calibration: [loss scale](calibration/t1_loss_scale.png), [first-fire map](calibration/t2_first_fire_map.png)
