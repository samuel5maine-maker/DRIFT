# H33: replay and consolidation in one learner — CoraFull solved, Arxiv lost, cause identified

**Branch:** `Astra/h33-replay-consolidation`. Predeclared protocol: [H33_PLAN.md](H33_PLAN.md), committed as `ab7c05d` before any run.

**Verdict: rejected.** The Arxiv falsifier fired: the hybrid lost 15.24 A_AUC against MAS alone, against a tolerance of 1.5. But the CoraFull half succeeded by a wide margin, and the Arxiv failure has one measured cause that points directly at the next hypothesis.

## Result

Validation, learner seed 4, stream seed 1, two replicates per arm. CoraFull is the equal-sigma mean over 3/10/20.

| Dataset | Arm | A_AUC | AF_s | weighted peak | weighted final | consolidations |
|---|---|---:|---:|---:|---:|---:|
| CoraFull | **hybrid** | **41.78** | **-17.54** | 90.06 | 72.93 | 16.2 |
| CoraFull | replay only | 37.15 | -29.05 | 89.45 | 59.82 | 0 |
| CoraFull | MAS only | 27.23 | -48.98 | 92.44 | 42.68 | 20.5 |
| Arxiv | hybrid | 30.30 | -35.78 | 86.03 | 52.44 | **13** |
| Arxiv | replay only | 33.12 | -36.16 | 90.08 | 51.81 | 0 |
| Arxiv | **MAS only** | **45.54** | **-11.74** | 93.95 | 86.02 | **317** |

Replicates: CoraFull hybrid 42.14 and 41.41; both Arxiv replicates were bit-identical. The MAS-only arm reproduces H32's learner exactly (27.55 on the first CoraFull replicate, 45.54 on Arxiv), as the unit tests require.

**Against the gate.**
1. CoraFull: hybrid **+14.55** over MAS only (needed +5.0) and **+4.63** over replay only (allowed -1.5). **Pass.**
2. Arxiv: hybrid **-15.24** against MAS only (allowed -1.5). **Fail.**
3. Guard: on CoraFull the AF_s gains ride on weighted final accuracy rising 30.25 and 13.11 points with peak within 2.4, so they are genuine retention. **Pass on CoraFull.**

CoraFull by width: 39.56 / -19.94 at sigma 3, 41.89 / -18.43 at sigma 10, 43.88 / -14.24 at sigma 20. With working-network inference and no EMA, the hybrid comes within 2.2 A_AUC of H23's CoraFull test result and is the best CoraFull learner this project has built without averaged inference.

## Why Arxiv failed: replay silences the consolidation trigger

The hybrid consolidated **13 times** on Arxiv and **stopped entirely at step 7,047**; MAS alone consolidated 317 times, to step 10,168, with a median gap of 14 steps. In the hybrid, MAS was effectively switched off for the last third of the stream and nearly absent before it, which is why the hybrid scores like replay only (30.30 against 33.12) rather than like MAS.

The mechanism is the absolute threshold in DRIFT's plateau detector — the same `window mean < 0.2` that H32 left untouched because it was not binding there. Without replay, the network collapses onto whatever classes are currently arriving, so the incoming cross-entropy falls below 0.2 constantly and the detector fires about once every 32 steps. Class-balanced replay keeps the head spread over the old classes, which is exactly its purpose, and that keeps the incoming loss above 0.2. On CoraFull the two arms consolidated at similar rates (16.2 against 20.5), so the effect was invisible there; on Arxiv replay cut the rate 24-fold.

This refines what H32 found. MAS's Arxiv success comes from frequent re-anchoring — a moving trust region — and that frequency is not a property of MAS. It is a side effect of the network being overconfident on current classes, which the detector reads as "learned". **Anything that reduces recency bias, including the replay this learner needs for CoraFull, suppresses the very consolidation that holds Arxiv.** The absolute threshold makes the two mechanisms antagonistic.

## What carries forward

- The hybrid structure works where its consolidation fires: +14.6 A_AUC over MAS and +4.6 over replay on CoraFull, with genuine retention.
- The remaining defect is identified and specific: a consolidation trigger written as an absolute loss level. It must be replaced by one whose firing rate does not depend on how confident the network happens to be — the scale-free trigger first sketched and set aside in H32, now with direct evidence that it is the binding constraint.
- The next branch changes only the trigger, keeps every other H33 component and constant, and should choose the trigger form from loss-trace statistics on both datasets before any outcome is measured.

## Reproduction

```powershell
.\.conda\science\python.exe -m unittest tests.test_hybrid tests.test_scalefree_mas -q
.\.conda\science\python.exe experiments/h33_hybrid.py --dataset CoraFull-CL --replicate r1
.\.conda\science\python.exe experiments/h33_hybrid.py --dataset Arxiv-CL --replicate r1
```

Raw artifacts: `results_h33/`, each run carrying split and source fingerprints, budgets, consolidation indices, replay graph work and protocol assertions. Final-test mode is locked and was never opened.
