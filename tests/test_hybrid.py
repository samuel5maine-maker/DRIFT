import unittest

import dgl
import numpy as np
import torch

from Baselines import hybrid_model, scalefree_mas_model
from tests.test_new_baselines import GCN, make_args, make_dataset, pipeline_graph, run_steps
from tests.test_scalefree_mas import warm_up


def build(module, seed=0, **hybrid):
    torch.manual_seed(seed)
    dgl.random.seed(seed)
    dgl.utils.set_num_threads(1)
    args = make_args()
    args.mas_args = {'memory_strength': 0.5}
    args.mas_geometry_args = {'project_classifier': False, 'ema_alpha': None,
                              'eval_working_weight': None, 'monitor_task_loss': True}
    if hybrid:
        args.hybrid_args = hybrid
    dataset = make_dataset()
    return module.NET(GCN(args), args, dataset=dataset), args, dataset


def steps_on(net, args, g, n, seed=0):
    """run_steps rebuilds the graph each call; contextual replay needs one stable graph."""
    labels = g.ndata['label'].squeeze()
    rng = np.random.RandomState(seed)
    for _ in range(n):
        ids = torch.tensor(rng.choice(g.num_nodes(), args.batch_size, replace=False), dtype=torch.long)
        net.observe_cis(args, g, g.ndata['feat'], labels, ids)


def force_anchor(net):
    net.omegas = [torch.ones_like(p) * 0.01 for p in net.net.parameters()]
    net.star_variables = [p.detach().clone() + 1.0 for p in net.net.parameters()]


class TestHybrid(unittest.TestCase):
    def test_mas_only_arm_matches_the_h32_learner_exactly(self):
        warm_up()
        reference, args_a, data_a = build(scalefree_mas_model)
        run_steps(reference, args_a, data_a, 60)
        mas_only, args_b, data_b = build(hybrid_model, replay_enabled=False, mas_enabled=True)
        run_steps(mas_only, args_b, data_b, 60)
        for left, right in zip(reference.net.parameters(), mas_only.net.parameters()):
            self.assertTrue(torch.equal(left, right))
        self.assertEqual(reference.count_updates, mas_only.count_updates)
        self.assertEqual(reference.consolidation_steps, mas_only.consolidation_steps)
        self.assertEqual(len(mas_only.buffer), 0)
        self.assertEqual(sum(mas_only.replay_rows_per_update), 0)

    def test_replay_schedule_and_budget(self):
        net, args, dataset = build(hybrid_model)
        run_steps(net, args, dataset, 40)
        self.assertEqual(net.replay_rows_per_update[0], 0)
        self.assertTrue(all(r == args.batch_size for r in net.replay_rows_per_update[1:]))
        self.assertLessEqual(len(net.buffer), 100)
        self.assertLessEqual(net.memory_accounting()['buffer_bytes'], 2400)
        self.assertGreater(net.replay_source_nodes, sum(net.replay_rows_per_update))

    def test_replay_only_arm_never_consolidates(self):
        net, args, dataset = build(hybrid_model, replay_enabled=True, mas_enabled=False)
        run_steps(net, args, dataset, 60)
        self.assertEqual(net.count_updates, 0)
        self.assertEqual(net.omegas, [])

    def test_window_holds_incoming_loss_only(self):
        net, args, dataset = build(hybrid_model)
        g = pipeline_graph(dataset)
        steps_on(net, args, g, 10)
        force_anchor(net)
        steps_on(net, args, g, 10, seed=1)
        self.assertTrue(any(o > t + 1e-6 for t, o in zip(net.task_losses, net.total_losses)))
        expected = sum(net.task_losses[-5:]) / 5
        self.assertAlmostEqual(net.monitored_losses[-1], expected, places=4)

    def test_penalty_and_replay_both_reach_the_gradient(self):
        net, args, dataset = build(hybrid_model)
        g = pipeline_graph(dataset)
        steps_on(net, args, g, 15)
        force_anchor(net)
        before = [p.detach().clone() for p in net.net.parameters()]
        steps_on(net, args, g, 1, seed=2)
        moved = any(not torch.equal(a, b) for a, b in zip(before, net.net.parameters()))
        self.assertTrue(moved)
        self.assertEqual(net.replay_rows_per_update[-1], args.batch_size)

    def test_relative_trigger_is_scale_free(self):
        rng = np.random.RandomState(3)
        means = list(rng.gamma(2.0, 1.0, size=200))
        variances = list(rng.gamma(1.0, 0.5, size=200))
        for index in range(20, 200, 7):
            base = hybrid_model.NET.relative_plateau(
                means[index], variances[index], means[:index], variances[:index])
            for k in (0.01, 3.0, 250.0):
                scaled = hybrid_model.NET.relative_plateau(
                    k * means[index], k * k * variances[index],
                    [k * m for m in means[:index]], [k * k * v for v in variances[:index]])
                self.assertEqual(base, scaled)

    def test_relative_trigger_fires_where_the_absolute_one_cannot(self):
        warm_up()
        absolute, args_a, data_a = build(hybrid_model, trigger='absolute')
        g_a = pipeline_graph(data_a)
        steps_on(absolute, args_a, g_a, 60)
        relative, args_b, data_b = build(hybrid_model, trigger='relative')
        g_b = pipeline_graph(data_b)
        steps_on(relative, args_b, g_b, 60)
        self.assertGreater(relative.count_updates, absolute.count_updates)
        self.assertEqual(relative.optimizer_steps, absolute.optimizer_steps)
        self.assertEqual(relative.replay_rows_per_update, absolute.replay_rows_per_update)

    def test_default_trigger_is_the_released_absolute_rule(self):
        net, _, _ = build(hybrid_model)
        self.assertEqual(net.trigger, 'absolute')
        with self.assertRaises(ValueError):
            build(hybrid_model, trigger='median')

    def test_unknown_arguments_are_rejected(self):
        with self.assertRaises(ValueError):
            build(hybrid_model, replay_enabled=True, buffer_size=1000)


if __name__ == '__main__':
    unittest.main()
