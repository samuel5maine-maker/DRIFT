"""Evaluation protocol: pipeline.eval_tasks_cis grades only delivered tasks, over one common head.

The fixture is a 3-task / 6-class stream whose logits always favour class 4, a late class. Before this
change class 4 was deleted from the head while earlier tasks were graded, so they scored 100%; with a
common head it can win, so they score 0%. That difference is the whole point of the change.
"""
import os
import sys
import types
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:                                  # the batch evaluator needs DGL; the full-graph one tested here does not
    import dgl                        # noqa: F401
    import ogb                        # noqa: F401
except ImportError:                   # pragma: no cover - only on machines without the drift env
    class _Any:                               # absorbs anything the dgl imports do at module import time
        def __call__(self, *args, **kwargs):
            return self

        def __getattr__(self, attribute):
            return self

    def _stub(name):
        module = types.ModuleType(name)
        module.__path__ = []

        def resolve(attribute):
            if attribute.startswith('__'):
                raise AttributeError(attribute)
            return _Any()

        module.__getattr__ = resolve          # any dgl.<thing> the imports ask for
        return module

    for name in ('dgl', 'dgl.base', 'dgl.nn', 'dgl.nn.pytorch', 'dgl.nn.functional', 'dgl.function',
                 'dgl.dataloading', 'dgl.utils', 'dgl.data', 'dgl.data.utils', 'dgl.sampling',
                 'ogb', 'ogb.nodeproppred', 'torch_geometric', 'torch_geometric.io',
                 'torch_geometric.data', 'torch_geometric.utils'):
        sys.modules.setdefault(name, _stub(name))
    sys.modules['dgl'].nn = sys.modules['dgl.nn']
    sys.modules['dgl.nn'].pytorch = sys.modules['dgl.nn.pytorch']

import pipeline                       # noqa: E402
from metrics import tf_metrics        # noqa: E402

N_CLS = 6
N_TASKS = 3


class FakeGraph:
    """Minimal stand-in for a task's evaluation subgraph."""

    def __init__(self, labels):
        self.dstdata = {'label': torch.tensor(labels).view(-1, 1)}
        self.srcdata = {'feat': torch.zeros(len(labels), 1)}

    def to(self, device=None):
        return self


class FakeModel:
    """Returns fixed logits, so accuracy depends only on which columns the evaluator keeps."""

    def __init__(self, logits):
        self.logits = logits

    def eval(self):
        return self

    def __call__(self, graph, features):
        return self.logits[id(graph)], None


class Args:
    setting = 'tfo_gaussian'
    cuda = False
    gpu = 0
    n_cls_per_task = 2


def fixture():
    """Task t holds classes 2t and 2t+1, one test node each. Every node's largest logit is class 4."""
    graphs, tasks_te, logits = [], [], {}
    for t in range(N_TASKS):
        labels = [2 * t, 2 * t + 1]
        g = FakeGraph(labels)
        row = torch.full((2, N_CLS), -1.0)
        row[0, 2 * t] = 0.5          # correct class: second best overall
        row[1, 2 * t + 1] = 0.5
        row[:, 4] = 0.9              # a late class always outranks it
        graphs.append(g)
        tasks_te.append(torch.tensor([0, 1]))
        logits[id(g)] = row
    return graphs, tasks_te, FakeModel(logits)


class TestCommonHead(unittest.TestCase):
    def test_undelivered_tasks_are_not_graded(self):
        graphs, tasks_te, model = fixture()
        result, _, current, _ = pipeline.eval_tasks_cis(model, graphs, tasks_te, 1, Args())
        self.assertEqual(len(result), N_TASKS)
        self.assertTrue(result[2] != result[2], 'task 2 was not delivered, so it must be nan')
        self.assertEqual(len(current), 2)

    def test_every_graded_task_sees_every_delivered_class(self):
        graphs, tasks_te, model = fixture()
        # cur_t=2: classes 0..5 delivered, so class 4 competes everywhere. Tasks 0 and 1 lose both nodes
        # to it; task 2 owns class 4, so its class-4 node is right and its class-5 node is not.
        result, avg, _, _ = pipeline.eval_tasks_cis(model, graphs, tasks_te, 2, Args())
        self.assertEqual(result, [0.0, 0.0, 0.5])
        self.assertAlmostEqual(avg, 1.0 / 6.0)

    def test_head_width_follows_delivered_classes(self):
        graphs, tasks_te, _ = fixture()
        self.assertEqual(pipeline._common_head(graphs, tasks_te, 0), 2)
        self.assertEqual(pipeline._common_head(graphs, tasks_te, 1), 4)
        self.assertEqual(pipeline._common_head(graphs, tasks_te, 2), 6)

    def test_pooled_accuracy_covers_only_graded_tasks(self):
        graphs, tasks_te, model = fixture()
        # cur_t=1: head is 4 wide, so class 4 cannot win and both graded tasks are perfect
        result, avg, _, current_avg = pipeline.eval_tasks_cis(model, graphs, tasks_te, 1, Args())
        self.assertEqual(avg, 1.0)
        self.assertEqual(current_avg, 1.0)
        self.assertEqual(result[:2], [1.0, 1.0])


class TestMetricsToleratesUngraded(unittest.TestCase):
    def test_peak_ignores_checkpoints_before_delivery(self):
        nan = float('nan')
        result_a = [[0.9, nan, nan],      # only task 0 delivered
                    [0.8, 0.7, nan],
                    [0.5, 0.6, 0.9]]      # final checkpoint: everything graded
        avg_acc = torch.tensor([0.9, 0.75, 0.66])
        aauc, af_s = tf_metrics(result_a, avg_acc)
        self.assertAlmostEqual(float(aauc), 0.77, places=6)
        # peaks: 0.9, 0.7, 0.9 -> drops: -0.4, -0.1, 0.0
        self.assertAlmostEqual(float(af_s), (-0.4 - 0.1 + 0.0) / 3, places=6)


if __name__ == '__main__':
    unittest.main()
