# NOTES: missing continual-learning baselines on DRIFT

Working notes for the "Missing Continual-Learning Baselines" spec (Sam Schultz, 2026-09-05).

**Tags on every claim**
- **[code]**: read from this repo.
- **[paper]**: read from DRIFT arXiv:2605.12998v3.
- **[run]**: measured here.
- **[inferred]**: reasoning, not verified.

**Environment.** Everything runs in the `drift-cpu` conda env: a clone of `drift` with DGL 1.1.2 (CPU), torch 1.13.1+cpu and numpy 1.26.4. The machine has no GPU. `docs/mas_star_code_map.md` explains why the original `drift` env cannot run here, and why DGL's sampler is pinned to one thread for reproducibility.

---

## §1 Answers

### Q1. Does message passing see the full graph adjacency at step *t*?

It depends on the regime.

**Gaussian mixing (`tfo_gaussian`)** [code]
- The training graph is `merged_subgraph`: the induced subgraph over **all tasks' train and test nodes**, built once before the stream starts (`pipeline.py` `data_prepare_gaussian`, `dataset.get_graph(node_ids=[all_tr, all_te])`). Validation nodes are excluded.
- Every step calls `NeighborSampler([10, 25])` on that graph (`args.sample_nbs=True`, `n_nbs_sample=[10,25]`). So at step *t* a node's sampled neighbourhood can include nodes, features and edges belonging to tasks that have not yet appeared in the stream.
- It is not "edges among observed nodes only".
- Neighbours are **sampled**, not the full neighbourhood.

**Hard transitions (`tfocis`) and global mixing (`tfo_blurry`)** [code]
- Each batch uses the **current task's own subgraph**: the induced subgraph over that task's train and test nodes (`data_prepare`, `get_graph(tasks_to_retain=task_cls)`). Edges between tasks are absent.

**Replay (ER, A-GEM, DMSG)** [code]
- The buffer stores **node IDs only**.
- At replay, buffered nodes become an auxiliary graph with **all edges removed and self-loops added**, e.g. `er_model.py:222-228`. Replayed nodes are re-embedded under the current weights but **without any neighbourhood**.
- On the Gaussian path ER and A-GEM build the aux graph from the merged graph's local IDs. Elsewhere they build it from `dataset.graph` using original IDs.

**Consequences for the spec**
- Buffer entries can stay bare node IDs. Re-embedding is free, but only as isolated nodes.
- To match existing replay baselines, new replay methods should use the same isolated-node convention.
- §5.2 DER stores logits at insertion. Under the isolated convention, those logits must be computed **on the isolated node**, not with its sampled neighbourhood. Otherwise the stored logits and the replay-time logits come from different inputs. Record the insertion-time form in the DER implementation.

### Q2. Train/val/test split

- **Per class, 60/20/20 train/val/test** [code]: `--ratio_valid_test 0.2 0.2` goes to `train_valid_test_split`, which calls `sklearn.train_test_split` with **no `random_state`**.
- The split is cached as `data/tr0.6_va0.2_te0.2_split_{dataset}.pkl`. Because it is seeded from the global numpy state, the first run fixes it for all later runs.
- The stream draws from train nodes only; evaluation uses test nodes only; validation nodes are unused.
- Classes with fewer than 2 nodes are dropped.

**Stream length** [code, run]: `compute_centers` gives Σ_tasks ceil(N_train,t / 10) batches.
- CoraFull-CL: **~1,190 steps** at B=10, and the cached σ=20 stream has 1,200. This is not ~1,980 as in spec §5.5's table, which used all nodes instead of the 60% train split.
- Arxiv-CL: **~10,100** (≈ 0.6 × 169k / 10). Reddit-CL: **~14,000**. The Arxiv and Reddit figures are inferred the same way and not yet run.
- §5.5's EMA table needs these corrected lengths: CoraFull's λ=0.999 window (~1,000 steps) covers **~84%** of the stream, not ~50%.

### Q3. When does DRIFT's DMSG rebuild its buffer and refit its variational layer?

**Never at an interval: there is no task-boundary surrogate at all** [code, `Baselines/dmsg_model.py`].
- **Buffer.** Every step (`observe_cis` → `_update_buffer_batch_local` on the Gaussian path, `_update_buffer_batch` elsewhere), the DMSG sampler selects `batch_size // 2 = 5` nodes from the **current batch** by representation diversity. Each is then offered to the 100-node buffer by **reservoir sampling**, and the aux graph is rebuilt.
- **Variational layer.** It has no separate refit. Its KL + similarity loss (`loss_vae`) and the adversarial diversity loss (`loss_adv`) are added to the replay loss **every step**, with weights `[1, 20, 1, 1]` for [CE, replay CE, adv, VAE].
- **So:** the spec's worry about a fixed-interval stand-in for a task boundary does not apply. DRIFT's adaptation is fully per-step. The soft spot is elsewhere: diversity selection happens *within a 10-node batch*, and reservoir acceptance then discards most of what the sampler chose.

⚠ **CPU-only bug in the original code** [code]
- On the Gaussian path, `_update_buffer_batch_local` moved the aux graph with a hardcoded `.to('cuda:{gpu}')` inside a `try/except` that only prints `"Warning: Could not update aux graph"`.
- On a machine without CUDA, DMSG would therefore never form a buffer and would silently run as Bare-plus-extra-losses.
- ER (`er_model.py:226`) and A-GEM (`agem_model.py:292`) had the same cast without a `try`, so they crashed instead.
- All three now use `subg.to(g.device)`, which is identical on GPU. Guarded `if self.cuda:` branches are untouched.
- This only matters for CPU runs. It does not bear on the published (GPU) numbers.

### Q4. Seeds and reporting

- **Paper** [paper, §5.1 / A.4]: "averaged over 3 random seeds", "mean ± std over 3 runs". The paper does not say which seeds.
- **Code** [code]:
  - `main.py` runs one `--seed` per invocation. `--repeats` defaults to 1, and repeats inside one process are not re-seeded (`set_seed` is called once).
  - The repo has no aggregation script; the readme says "Aggregate metrics across runs from the saved `*.pkl` files".
  - The readme's example loops seeds **1-5**.
- **Used here: seeds {1, 2, 3}, mean ± sample std.** The seed identities are an assumption.
- **Stream caching** [code]: stream pickles omit the seed, so every seed reuses the stream built first. Seeds vary model initialization, neighbour sampling and buffer RNG, not the stream. `experiments/prepare_data.py` builds caches with seed 1.

### Q5. Evaluation interval Δ and the A_AUC definition

- **Δ = 100 batches** [code]: `--log_every 100`. The model is evaluated before batch *b* whenever `b % 100 == 0`, plus once after the stream (`pipeline_gaussian`). CoraFull-CL at σ=20 gets 13 evaluation points.
- **A_AUC** = `avg_acc_list.mean()` in `metrics.tf_metrics` [code].
- **What each evaluation computes** [code: `eval_tasks_cis`]
  - It evaluates **all K latent-task test sets at every point**, including tasks whose classes have not yet appeared in the stream.
  - For task *k*, logits are restricted to classes `0 … max(test classes of tasks ≤ k)`.
  - The returned average is **pooled over all test nodes** (`total_pred / total_size`), which is size-weighted.
  - The paper's Ā(t) is "mean accuracy across all K latent-task test sets". Pooled and task-averaged accuracy coincide only when tasks are equal-sized. Noted as a difference between code and prose; using the code's definition unchanged, as §7 requires.
- **AF_s** [code: `tf_metrics`] = `mean_k (A_final,k − max_i A_i,k)`, which is **≤ 0, higher is better**. The table's negative values match this.
  - Paper Eq. 8 writes `max − final`, and spec §7 says "peak − final". Both are sign-flipped relative to the code and to the published tables.
  - The underfitting caveat in §7 still holds in either sign convention.

### Q6. Are labels available at insertion time?

**Yes** [code]. Every `observe`/`observe_cis` receives `labels` and `train_ids`. ER, A-GEM and DMSG all read `labels[train_ids]` during the step, and the aux graph carries `label` from the stored graph. Class-balanced buffers (§5.1) can use them.

---

## §2 Protocol as it exists in code [code]

| Setting | Spec §2 | Code default | Match |
|---|---|---|---|
| Backbone | 2-layer GCN, 256, no dropout, no BN | `GCN_args={'h_dims':[256],'dropout':0.0,'batch_norm':False}` → 2 `GCNLayer`s | ✓ |
| Optimiser | Adam 5e-3 | Adam, lr 0.005, **weight_decay 5e-4** | the paper and spec leave out weight decay |
| B | 10 | `--batch_size 10` | ✓ |
| Passes | 1 | `--epochs 1` | ✓ |
| Memory | 100 | `er_args/agem_args budget [100,1000]` → first element | ✓ |
| Replay ratio | 1:1 | ER/A-GEM sample `batch_size × int(memory_proportion[0]) = 10`; DMSG samples `min(batch_size, buffer)` | ✓ |
| Neighbour sampling | not specified | `NeighborSampler([10, 25])` for training; full graph at evaluation | worth stating |

**Datasets that can run in this environment**
- **CoraFull-CL** ✓
- **Arxiv-CL** ✓. The OGB download goes to `data/raw`, and the directed citation graph is used as-is: `add_reverse_edges` is commented out at `dataset/utils.py:202`.
- **RomanEmpire-CL ✗ here.** `dgl.data.RomanEmpireDataset` is absent from DGL 0.9.1 and 1.1.2 [run], so it needs a newer DGL, which in turn needs a newer torch than 1.13 [inferred].
- **Reddit-CL**: not yet attempted. It has 233k nodes and 114M edges, and the stream cache holds the merged graph plus 20 per-task subgraphs, so memory on a CPU-only machine is the likely limit [inferred].

---

## §4 Gate 0

**Status: partial. 12 of 16 cells reproduce; the 4 that do not (Bare on both datasets, ER on CoraFull, MAS* on Arxiv) are all above the paper. Stopped before implementing new methods, per spec §4, pending a decision.**

[run] Seeds 1–3, `drift-cpu` env. Metrics come from `results/*.pkl` via `analysis/gate0.py`, which writes `analysis/gate0.md`. The criterion is |Δ| ≤ 2·SE with SE = √(sd²/n + sd_paper²/3).

**CoraFull-CL, Gaussian σ=20**

| method | A_AUC ours | A_AUC paper | Δ (z) | AF_s ours | AF_s paper | Δ (z) |
|---|---|---|---|---|---|---|
| Bare | 24.8 ± 1.4 | 21.9 ± 0.8 | **+2.9 (+3.1)** | −53.6 ± 3.7 | −60.5 ± 7.3 | +6.9 (+1.4) |
| ER | 34.7 ± 0.04 | 27.8 ± 0.6 | **+6.9 (+19.8)** | −40.5 ± 4.3 | −48.0 ± 5.2 | +7.5 (+1.9) |
| A-GEM | 32.4 ± 1.7 | 29.9 ± 2.8 | +2.5 (+1.3) | −50.2 ± 3.7 | −53.9 ± 4.8 | +3.7 (+1.0) |
| MAS* (tfmas) | 30.6 ± 2.6 | 29.8 ± 2.2 | +0.8 (+0.4) | −44.8 ± 12.5 | −44.8 ± 8.0 | 0.0 |

- **Every A_AUC comes out at or above the paper.** A-GEM and MAS* are within noise; Bare (+2.9) and ER (+6.9) are not. All AF_s values are within noise.
- **ER's three seeds are distinct runs** (final accuracy 0.51 / 0.47 / 0.46). Their AAUCs just happen to land close together (34.62 / 34.70 / 34.69).
- **The ER gap is not caused by the CPU device fix.** That fix only changes where the aux graph lives; on CPU the unfixed code crashed rather than producing different numbers.
- **Likely cause** [inferred]: the environment. The published numbers presumably came from a GPU run with a different DGL version and different neighbour-sampling RNG. Here, DGL 1.1.2 on CPU is used with sampling pinned to one thread. A roughly uniform upward offset would fit Bare and A-GEM, but ER's +6.9 is more than twice Bare's.
- **Cannot be resolved without the original environment.** The DGL version behind the published tables is unknown. Note that `dataset/utils.py` imports `RomanEmpireDataset`, which is absent from DGL 0.9.1 and 1.1.2, so the repo as published needs a newer DGL than either, whereas the local `drift` env had `dgl-cu113 0.9.1`.

**Arxiv-CL, Gaussian σ=60** [run], seeds 1–3, ~5–8 min per run on CPU:

| method | A_AUC ours | A_AUC paper | Δ (z) | AF_s ours | AF_s paper | Δ (z) |
|---|---|---|---|---|---|---|
| Bare | 22.1 ± 1.1 | 18.5 ± 1.5 | **+3.6 (+3.4)** | −61.5 ± 1.2 | −65.4 ± 3.1 | +3.9 (+2.0) |
| ER | 35.6 ± 1.3 | 34.9 ± 0.8 | +0.7 (+0.8) | −37.4 ± 2.1 | −37.3 ± 2.8 | −0.1 |
| A-GEM | 35.0 ± 0.8 | 34.1 ± 1.4 | +0.9 (+1.0) | −48.9 ± 3.3 | −48.6 ± 4.4 | −0.3 |
| MAS* (tfmas) | 43.5 ± 1.6 | 38.4 ± 2.1 | **+5.1 (+3.4)** | −16.6 ± 6.9 | −22.2 ± 1.8 | +5.6 (+1.4) |

**Where Gate 0 stands.** ER and A-GEM reproduce on Arxiv-CL. On CoraFull-CL, A-GEM and MAS* reproduce. Four cells fall outside noise, and all four are **above** the paper:
- **Bare**, on both datasets: +2.9 and +3.6.
- **ER**, on CoraFull: +6.9.
- **MAS***, on Arxiv: +5.1.

No cell comes out below. That points to something systematic about this environment rather than a broken method [inferred].

**MAS* discrepancy: the code produces 43.5 ± 1.6** (per seed 41.9 / 45.0 / 43.6).
- **Tables 2 and 10:** 38.4 ± 2.1. The code is +5.1 above this.
- **§5.4 prose:** 23.1. The code is +20.4 above this.

The code is much closer to the tables and gives no support for 23.1 under Gaussian mixing at σ=60, B=10. For the question to the authors: "the code produces 43.5; tables say 38.4; prose says 23.1."

Two related points:
- The same §5.4 sentence gives Bare 22.6. The code gives **22.1 ± 1.1**, matching the prose, while Table 2 says 18.5 ± 1.5. One possibility is that the prose was written from a different run (another σ or configuration) than the tables [inferred]. It would be worth asking which configuration the §5.4 numbers came from.
- MAS* and Bare accuracy matrices differ on every Arxiv seed, so DRIFT's MAS* does consolidate on Arxiv. The bit-identical MAS* = Bare result was specific to CoraFull under global mixing (`analysis/report.md`).

Published targets [paper, Table 2, Gaussian mixing, A_AUC / AF_s]:

| | Arxiv-CL (σ=60) | CoraFull-CL (σ=20) |
|---|---|---|
| Bare | 18.5±1.5 / −65.4±3.1 | 21.9±0.8 / −60.5±7.3 |
| ER | 34.9±0.8 / −37.3±2.8 | 27.8±0.6 / −48.0±5.2 |
| A-GEM | 34.1±1.4 / −48.6±4.4 | 29.9±2.8 / −53.9±4.8 |
| MAS* | 38.4±2.1 / −22.2±1.8 | 29.8±2.2 / −44.8±8.0 |

**MAS* on Arxiv-CL, Gaussian: three figures in the paper** [paper]
- Table 2 gives **38.4**.
- §5.4 prose: "MAS∗ drops from 55.5% under hard transitions to 23.1% under Gaussian mixing and 22.5% under 30% global mixing, approaching the Bare baseline (22.6%)".
- §5.2: "best continual learning method improves over Bare by +16.4% under Gaussian mixing (ER in Table 2)". That is ER 34.9 − Bare 18.5 = 16.4, consistent with Table 2's Bare but not with MAS* at 38.4.
- The §5.4 prose also puts Bare at 22.6, versus 18.5 in Table 2.
- Table 10 could not be read from the arXiv HTML extraction and is still unchecked.

---

## Spec claims contradicted by the code

1. **§5.5 stream lengths** use all nodes. The stream uses the 60% train split, so lengths are ~0.6× (see Q2).
2. **§7 AF_s = peak − final.** The code and tables use final − peak (see Q5).
3. **Q3's premise.** DRIFT's DMSG has no rebuild interval (see Q3).
4. **§2 datasets.** RomanEmpire-CL cannot run on torch 1.13 / DGL < 2 (see §2).

---

## Decision after Gate 0

Sam, 2026-09-15: **proceed with the new methods.**
- Every baseline used for comparison is re-run in this environment under a fixed protocol (next section).
- Published numbers appear with `source = paper` and are not directly comparable.

## Reproducibility protocol "t2" (all new-baseline runs)

[run] Results depend on the OpenMP thread count, and on oversubscription.
- DRIFT's ER, CoraFull-CL σ=20, seed 1 gave A_AUC 34.6 (Gate 0, 8 threads), 33.5 (8 threads, contended), 32.8 (4 threads), 32.7 (2 threads) and 32.7 (1 thread).
- At a fixed `OMP_NUM_THREADS=2`, two concurrent runs gave bit-identical accuracy matrices.

The protocol:
- All new-baseline runs use `experiments/run_matrix.py --threads 2 --results results_t2`, with telemetry in `telemetry_t2/`.
- DGL sampling is pinned to one thread in `set_seed`.
- Batches run from a frozen git worktree (`DRIFT_OUT_ROOT`), so edits in the main checkout cannot reach queued jobs. Four early jobs crashed when `pipeline.py` and `telemetry.py` were edited mid-batch; they are re-run.
- **Implication for Gate 0 and the MAS* study:** those runs mixed thread counts (8 for the first job per regime, 2 for the rest). Their seed-to-seed spread includes ~1–2 A_AUC points of thread variation.

## §11 report-back

### 3. Where a source paper and its reference implementation disagree, or the port departs from both

| Method | Source says | Reference code does | Followed | Why |
|---|---|---|---|---|
| Reservoir (Algorithm R) | keep item t with probability k/(t+1) (Vitter 1985) | Mammoth `utils/buffer.py`: `randint(0, num_seen + 1)` (matches Vitter). DRIFT `er_model.py` and OCGL `ReservoirSamplingBuffer`: `randint(0, n_seen + i)`, i.e. k/t, an off-by-one | Vitter/Mammoth in the new methods; DRIFT's ER unchanged | new methods follow the reference algorithm; existing baselines stay as published |
| CBRS (§5.1) | Algorithm 1 (population) **plus** weighted replay and the loss a·L_s + (1−a)·L_r with a = 1/n_c (Algorithm 2) | no official code | population only by default; `'replay':'weighted'` and `'loss':'convex'` are options | the spec describes population only, and keeping DRIFT's ER step isolates the memory policy |
| DER / DER++ | store logits at insertion | Mammoth stores `outputs.data` from the (augmented) training forward, before the step | pre-step logits computed **on the isolated node**; `'insert_logits':'stream'` stores Mammoth's version | DRIFT replays isolated nodes (Q1); stream logits carry a sampled neighbourhood the replay input never has |
| CLS-ER | Algorithm 1: target = plastic if σ(Z_P) > σ(Z_S), else stable; no warm-up ramp mentioned | `clser.py`: stable if stable_prob > plastic_prob, else plastic; α = min(1 − 1/(step+1), α_max) | code | reference implementation; the only difference is how ties break |
| CLS-ER inference | "For inference, we use the stable model" | no `forward` override in `clser.py` | stable model; working/plastic curves logged to `alt_model_accuracy.csv` | paper states it explicitly |
| PDGNN | KDD 2024: coverage-maximisation sampling for the memory | OCGL online port: ER over SGC features, reservoir | OCGL | the spec points to OCGL's online adaptation |
| PDGNN SGC normalisation | — | OCGL normalises by **block** degrees and also predicts on blocks | full-graph degrees on training blocks | DRIFT evaluates on the full graph; with block degrees, train and test embeddings would be on different scales. With a full-neighbourhood sampler the two now agree exactly (unit test) |
| LwF-online | — | OCGL re-initialises the LwF net with `kaiming_normal_init` | DRIFT's default initialisation | all methods start from identical backbone weights |

### 4. Methods that cannot run under §2's constraints unchanged

- **PDGNN** replaces the 2-layer GCN with SGC(k=2) propagation + MLP(256). This is inherent to the method (OCGL asserts `backbone == 'SGC'`). It is reported with an `[SGC]` tag. Its memory stores d_data-dimensional embeddings: 8,710 dims on CoraFull (~3.5 MB for 100 slots) and 128 on Arxiv.
- **CLS-ER** keeps two extra full model copies. **DER, DER++ and the combination** store n_cls logits per slot. The combination also keeps one EMA copy. All of this appears in `analysis/baselines/memory_accounting.csv`.
- **DRIFT's own replay baselines are not memory-equal either.** A-GEM holds two gradient-sized vectors per step. DMSG has an adversarial discriminator, which is counted as extra params.
- **RomanEmpire-CL** cannot run in this environment (DGL < 2.0; see §2).

### 5. Spec claims contradicted by code or measurement (additions)

5. **§5.1 occupancy table.**

   | | spec | exact (hypergeometric) | simulated, 300 trials |
   |---|---|---|---|
   | classes with 0 slots | 16.7 | **24.18** | 23.7 |
   | classes with ≤ 1 slot | 40.6 | **43.72** | 43.7 |
   | latent tasks with 0 slots | 1.9 | **4.37** | 4.2 |

   This is for Algorithm R with k=100 over CoraFull's real class sizes (all 19,793 nodes). The DRIFT train-split stream gives 24.2 / 43.6 / 4.3, and CBRS gives 0 / 40.0 / 0. The spec's percentages (23.8%, 57.9%) are consistent with its own counts, so the spec simulated a different class-size distribution. The real imbalance is worse than stated. `tests/test_replay_buffers.py` now checks against the exact expectation.
6. **§5.5 "λ = 0.999 window as % of stream".** With the real stream lengths (Q2), CoraFull is ~84%, not ~50%. CLS-ER's own paper uses α = 0.99 for its general-CL benchmark (MNIST-360), not 0.999. The tuning grid covers both, plus stream-scaled decays (`clser_grid` in `experiments/run_matrix.py`).

### Hyperparameter tuning protocol (new methods)

- **Seed 0 only**, disjoint from the evaluation seeds {1, 2, 3}; selection by highest A_AUC. Single-seed selection is noisy relative to the ~1-2 point thread/seed spread, so `analysis/baselines/tuning.md` prints every configuration, not just the winner.
- Grids come from each method's own source: DER α ∈ {0.5, 1.0} and DER++ α ∈ {0.2, 0.5}, β ∈ {0.5, 1.0} (DER paper Table 10, MNIST-360 — the general-continual benchmark); LwF-online λ ∈ {0.1, 1, 10}, T ∈ {0.2, 2, 20}, update_every ∈ {1, 10, 100} (spec §5.4 / OCGL); CLS-ER: the paper's MNIST-360 row, the official repo defaults, and stream-scaled decays (plastic window σ, stable window {3, 10}·σ).
- lr, batch size, replay ratio and memory size are fixed by DRIFT's protocol (§2) and are not tuned.
- **Deviation, for compute:** an Arxiv-CL run takes ~5x a CoraFull-CL run, so LwF-online's 27-point grid was not repeated on Arxiv. Its best three CoraFull configurations were carried over (`tune_arxiv_small`). DER, DER++ and CLS-ER keep their full grids on both datasets.
- ER-CBRS and PDGNN have no tuned hyperparameters: both inherit DRIFT's memory size and 1:1 replay ratio.
