# MAS* on DRIFT: code map

**Reference:** Aljundi, Kelchtermans & Tuytelaars, *Task-Free Continual Learning*, arXiv:1812.03596v3 (19 Aug 2019), Algorithm 1 (p. 4). I have not checked the CVPR camera-ready version.

**Implementation:** [`Baselines/tfmas_star_model.py`](../Baselines/tfmas_star_model.py). Run it with `--method tfmas_star`. DRIFT's existing `tfmas` is left unchanged so it can still be compared against the published numbers.

Status labels:
- **verified**: read from the paper or the code.
- **interpretation**: the paper is ambiguous here, and the chosen reading is stated.
- **departure**: this intentionally differs from the paper, and the reason is stated.

## Algorithm 1 → code

| Alg. 1 | Paper | Code | Status |
|---|---|---|---|
| input δμ, δσ | plateau thresholds; values not in the paper body | `l_th`, `std_th` in `--tfmas_star_args`, required, no default ([:64-66](../Baselines/tfmas_star_model.py#L64)) | verified that the paper gives no values; they are calibrated here (see `analysis/calibration_t1t2.md`) |
| input N | inner passes per batch ("2-3" or "10-15" in the paper's experiments) | `passes`, default 1; `--epochs` must be 1 | departure: DRIFT runs 1 epoch per batch |
| line 2 `B={}, W={}` | | `buffer_ids=[]`, `W=deque(maxlen=window)` ([:92](../Baselines/tfmas_star_model.py#L92)) | verified |
| line 3 `Ω=0, μ_old=σ_old=0, P=0` | | [:92-99](../Baselines/tfmas_star_model.py#L92) | verified. P=0 means consolidation is allowed from step 1 |
| window length | "fixed loss window size of 5" (§4.2) | `window=5` | verified |
| line 5 | K recent samples | DRIFT batch of 10 nodes plus sampled neighbourhoods | verified |
| line 7 | L_T = L(X,Y) + L(X_B,Y_B) + λ/2 Σ Ω(θ−θ*)² | [:211-227](../Baselines/tfmas_star_model.py#L211). The penalty is skipped until the first consolidation, where Ω=0 makes it exactly 0 anyway | verified |
| λ | not in the paper body | `lam=0.5` (DRIFT App. A.4) | departure (value from DRIFT) |
| line 8 SGD | SGD, lr 1e-4 | Adam, lr 5e-3, wd 5e-4 (DRIFT defaults) | departure: keeps results comparable with the benchmark |
| line 10 `W ← update(W, L(X,Y), L(X_B,Y_B))` | two losses | pushes one value per step, `L(X,Y)+L(X_B,Y_B)`, using the pre-update losses from line 7 (`window_push=sum`). `window_push=both` pushes them as two entries | interpretation |
| line 13 `¬P ∧ μ(W)<δμ ∧ σ(W)<δσ` | the algorithm says σ; the text says "variance" | `plateau_detector` ([:23](../Baselines/tfmas_star_model.py#L23)) uses population std | interpretation (follows the algorithm) |
| line 13 (buffer) | Ω is estimated on the buffer | consolidation also requires a non-empty buffer (only matters at step 0) | interpretation |
| line 14 `Ω ← update(Ω, θ, (X_B,Y_B))` | eq. 3: Ω_i = 1/N Σ_k ‖g_i(x_k)‖, g = ∂F/∂θ, computed on the buffer | `_omega_estimate` ([:159](../Baselines/tfmas_star_model.py#L159)): per-sample \|∂‖logits[:head]‖²/∂θ\|, averaged over the 100 buffered nodes, in eval mode | F = squared L2 of the output is inferred from MAS [1]; eval mode is an interpretation |
| Ω accumulation | "cumulative moving average" (§3) | `Ω ← Ω + (Ω_new − Ω)/k` ([:252](../Baselines/tfmas_star_model.py#L252)) | verified |
| line 15 `θ* ← θ` | | [:254](../Baselines/tfmas_star_model.py#L254) | verified |
| line 16 `μ_old, σ_old` | recorded before the clear | `plateau_detector` [:39](../Baselines/tfmas_star_model.py#L39) | verified |
| line 17 `W={}, P=1` | | [:40](../Baselines/tfmas_star_model.py#L40) | verified |
| line 19 `μ(W) > μ_old + σ_old → P=0` | "higher than 85% of a normal distribution estimated on the previous plateau" | [:43](../Baselines/tfmas_star_model.py#L43); skipped while W is empty | interpretation for the empty-window case |
| line 22 buffer update | "prioritized keeping": keep the highest-loss samples among the new samples and the current buffer; size 100 | [:264](../Baselines/tfmas_star_model.py#L264): per-sample CE for buffer ∪ batch under θ after the step, deduplicated, top-100 kept | scoring the old buffer entries under the current θ is an interpretation |
| graph data | the paper has no graphs | buffered nodes are replayed as isolated nodes with self-loops, taken from `dataset.graph` (DRIFT's ER convention, `Baselines/er_model.py`) | departure (GNN adaptation) |
| class-IL head | the paper uses a fixed output | CE and F are restricted to `logits[:, :offset2]`, the classes seen so far (as in every DRIFT `observe_cis`) | departure (DRIFT protocol) |

## Legacy `tfmas` vs `tfmas_star`

| | `Baselines/tfmas_model.py` (DRIFT) | `tfmas_star` (Alg. 1) |
|---|---|---|
| hard buffer | none | 100 nodes, prioritized keeping |
| Ω data | the current 10-node batch | the buffer |
| Ω estimator | \|grad of the batch-mean squared logit\| | per-sample \|grad\|, averaged |
| value pushed to W | CE + MAS penalty | L(X,Y) + L(X_B,Y_B) (no penalty) |
| spread test | variance < `(0.1,)` (a tuple, because of a trailing comma) | σ < δσ |
| mean threshold | `(0.2,)` (tuple), hardcoded | δμ, calibrated |
| window after consolidation | never cleared | cleared |
| latch initial state | armed | armed (P=0) |
| λ | 0.5 hardcoded; `--mas_args memory_strength` is read but never used | 0.5, configurable |

## Corrections to the earlier telemetry spec

- The spec says Algorithm 1 initializes its flag so that consolidation is impossible until the first peak. arXiv v3 initializes P=0, which permits consolidation immediately.
- The spec says line 15 clears W before line 16 computes the statistics. In arXiv v3, the statistics are computed (line 16) before the clear (line 17).

## Environment and reproducibility findings

- **Conda env.** The `drift` env could not import DGL on this machine: a CUDA DGL build with a CPU-only torch, no GPU, and numpy 2 breaking torch 1.13. All runs use `drift-cpu`, a clone of `drift`, with dgl 1.1.2 (CPU), numpy 1.26.4 and matplotlib added. The package list is in `telemetry/provenance/`.
- **DGL version.** DGL 0.9.1 was rejected: its block message passing crashes on duplicate seed nodes, and the Gaussian stream produces them (CoraFull σ=10, batch 811). `RomanEmpireDataset` is missing below DGL 2.0, so that import is now guarded.
- **Sampling determinism.** DGL's multithreaded CPU neighbour sampling is not reproducible under `dgl.random.seed`. `training/utils.set_seed` now calls `dgl.utils.set_num_threads(1)`. With that, runs are bit-identical, and telemetry on and off give identical parameters.
- **Stream caches.** `data_prepare_*` cache names leave out the seed, so every seed reuses the stream built first. `experiments/prepare_data.py` builds all caches with seed 1.
- **Published numbers.** Legacy `tfmas`, CoraFull σ=10, seed 1 in this env gives AAUC 0.274 and FM −0.576. The telemetry spec quotes a published AAUC of 16.2, so this env does not reproduce that number, and every arm is re-run here.

## Not verified

- The CVPR 2019 camera-ready Algorithm 1, which may differ from arXiv v3.
- The supplementary material's δμ, δσ and λ, which were not accessible.
- Whether MAS [1] defines F as the squared or the plain L2 norm of the output (squared is used here).
- Which DGL version and seed produced DRIFT's published results.
