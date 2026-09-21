# DRIFT's evaluator rewards arrival order, not class knowledge

**Branch:** `Astra/h34-relative-trigger`. Diagnostic, not a hypothesis test: no learner was changed and every number below comes from the unmodified training rules of earlier cycles. Scripts: `experiments/eval_decomposition.py`, `experiments/recency_oracle.py`. Results: `results_decomposition/`.

## Summary

DRIFT scores latent task *t* only over classes 0 … max(classes of tasks ≤ *t*), at every checkpoint including the last, on task *t*'s own evaluation subgraph (`pipeline.eval_tasks_cis`). Classes arri ve in index order, so that prefix deletes every class that arrived after task *t*: the evaluator hands the model part of the task identity. A network whose logits are ordered by recency finds task *t*'s own pair at the top of the prefix without knowing anything else.

Three measurements show this dominates the metric:

1. **A predictor with no features and no training scores 25.8 A_AUC on CoraFull**, above Bare (24.2), by naming the most recently arrived class in the prefix. Arrival order plus perfect within-pair discrimination reaches 52–54, which is exactly the "immediate-perfect reference" of 52.16 computed in the first cycle. That reference was the metric's leak-assisted ceiling, not a knowledge ceiling.
2. **The strongest learner on Arxiv under DRIFT's metric is at chance on the real problem.** MAS's final restricted accuracy is 78.4%; over all 40 classes it is **2.0%**, against a chance rate of 2.5%.
3. **The ranking of learners by DRIFT's metric is roughly inverted relative to their real multi-class accuracy.** The H33 hybrid is last of three on Arxiv under DRIFT and first on unrestricted accuracy.

## Method

`eval_decomposition.py` wraps the standard evaluator and, from one extra forward per evaluation graph, splits each task's restricted accuracy:

```text
restricted  =  P(prediction in task t's own pair)  x  P(correct | prediction in pair)
```

It also records task-given accuracy (argmax over the pair's two columns) and, at the final checkpoint, unrestricted accuracy over every class. Its restricted pooled curve reproduces the reported metric to within 2e-8 in every run.

`recency_oracle.py` scores two predictors that never look at a node: **recency only** predicts, for each task, the most recently first-delivered class inside the task's prefix; **recency + within pair** additionally gets the within-pair decision right whenever recency lands in the task's own pair.

## Results

Validation, learner seed 4, stream seed 1. Final checkpoint, size-weighted over tasks.

| Dataset | Learner | DRIFT A_AUC | restricted | in own pair | within pair | **unrestricted** |
|---|---|---:|---:|---:|---:|---:|
| CoraFull σ10 | MAS (relative trigger) | 26.92 | 32.5 | 36.1 | 80.3 | **1.8** |
| CoraFull σ10 | replay only | 37.14 | 53.7 | 57.3 | 89.4 | **20.5** |
| CoraFull σ10 | hybrid (H33) | 42.13 | 76.6 | 80.1 | 93.7 | **16.6** |
| Arxiv σ60 | MAS (relative trigger) | 38.39 | 78.4 | 84.9 | 90.0 | **2.0** |
| Arxiv σ60 | replay only | 32.60 | 52.7 | 61.5 | 82.3 | **15.1** |
| Arxiv σ60 | hybrid (H33) | 30.52 | 51.4 | 58.3 | 81.8 | **21.7** |

Chance for unrestricted accuracy is 1.4% on CoraFull (70 classes) and 2.5% on Arxiv (40).

Feature-free predictors, A_AUC:

| Dataset | recency only | recency + perfect within pair |
|---|---:|---:|
| CoraFull σ3 / σ10 / σ20 | 25.15 / 25.79 / 26.55 | 51.68 / 51.93 / 53.25 |
| Arxiv σ60 | 13.94 | 53.60 |

Recency alone scores lower on Arxiv because its classes are imbalanced and the most recent class in a pair is often the minority; the within-pair ceiling is the same on both.

## What this explains

Every otherwise puzzling result in this project fits:

- **MAS's Arxiv advantage.** MAS keeps no examples, so it collapses onto current classes and its logits end up ordered by arrival. The prefix turns that into 78–89% restricted accuracy while real 40-way accuracy is at chance. H33's hybrid and H34's reruns could not "recover MAS behaviour" by restoring consolidation, because what MAS had was recency order, not retained knowledge.
- **Every class-balancing intervention hurt Arxiv:** CBRS (−5.4 against reservoir), H31's prior correction (−2.33), replay in H33 and H34. Each one weakens recency order, which is what the metric pays for.
- **Fixed classifier-row norms collapsed MAS from 45 to 16.8 (H25)** by erasing recency from logit magnitude.
- **AF_s rewarded peak suppression (H31)** and **per-task peaks were reached at mutually exclusive checkpoints**: a task's peak is the moment it is most recent.
- **An offline i.i.d.-trained backbone scored 53% pooled on Arxiv while sequential MAS scored 89%.** The i.i.d. model has no arrival order to exploit.

It also means the earlier headline results are measurements of the same artifact. H23's 44.25 on CoraFull and H26's 42.23 on Arxiv are real under DRIFT's protocol, and so are the published DRIFT tables, which come from the same evaluator. None of them establishes multi-class knowledge at that level.

## A second, smaller finding

The same MAS configuration scored 47.18 in the H34 pilot and 38.39 here. The only difference is that here it was the first learner constructed in its process, so DGL's CPU sampler — which does not fully reset under `dgl.random.seed` — gave it a different neighbour stream. A 9-point swing from sampler state alone is further evidence that MAS's Arxiv number rests on a fragile collapse rather than on learned structure.

## Consequence for the project goal

"Beat 44 A_AUC and −14 AF_s on both datasets" under DRIFT's evaluator can now be approached from two directions:

- **Legitimately**, by learning more — which, as the table shows, does not currently move DRIFT's metric in the right direction on Arxiv.
- **By exploiting arrival order**, for example by biasing logits toward recently arrived classes and away from balanced replay. The recency-plus-within-pair ceiling of 52–54 shows how much of the metric that route could unlock without any additional knowledge.

The second route satisfies the letter of the target and defeats its purpose. It should not be pursued without an explicit decision, and a model built that way should not be described as succeeding in DRIFT's problem setting.

## Limits of this diagnostic

One learner seed and one stream seed per dataset, one width on each. Unrestricted accuracy is measured at the final checkpoint only; an unrestricted A_AUC needs each checkpoint's set of delivered classes and is the obvious next measurement. The validation split is used throughout; test was not opened.
