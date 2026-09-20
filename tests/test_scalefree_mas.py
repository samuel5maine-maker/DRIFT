import unittest

import numpy as np
import torch

from Baselines import mas_geometry_model, scalefree_mas_model
from tests.test_new_baselines import make_args, make_dataset, run_steps, GCN
import dgl


def build_mas(module, monitor_task_loss=None, seed=0):
    torch.manual_seed(seed)
    dgl.random.seed(seed)
    dgl.utils.set_num_threads(1)
    args = make_args()
    args.mas_args = {'memory_strength': 0.5}
    geometry = {'project_classifier': False, 'ema_alpha': None, 'eval_working_weight': None}
    if monitor_task_loss is not None:
        geometry['monitor_task_loss'] = monitor_task_loss
    args.mas_geometry_args = geometry
    dataset = make_dataset()
    return module.NET(GCN(args), args, dataset=dataset), args, dataset


def warm_up():
    """DGL's CPU sampler does not fully reset on dgl.random.seed, so the first run in a
    process draws a different stream from every later one.  Burn one run before comparing."""
    net, args, dataset = build_mas(mas_geometry_model)
    run_steps(net, args, dataset, 2)


class TestScaleFreeMAS(unittest.TestCase):
    def test_disabled_arm_matches_the_released_mas_exactly(self):
        steps = 60
        warm_up()
        control, control_args, control_data = build_mas(mas_geometry_model)
        run_steps(control, control_args, control_data, steps)
        disabled, disabled_args, disabled_data = build_mas(
            scalefree_mas_model, monitor_task_loss=False)
        run_steps(disabled, disabled_args, disabled_data, steps)
        for left, right in zip(control.net.parameters(), disabled.net.parameters()):
            self.assertTrue(torch.equal(left, right))
        self.assertEqual(control.count_updates, disabled.count_updates)
        self.assertEqual(len(control.omegas), len(disabled.omegas))
        for left, right in zip(control.omegas, disabled.omegas):
            self.assertTrue(torch.equal(left, right))
        for left, right in zip(control.star_variables, disabled.star_variables):
            self.assertTrue(torch.equal(left, right))
        self.assertEqual(control.optimizer_steps, disabled.optimizer_steps)

    def test_monitored_window_excludes_the_penalty_only_when_enabled(self):
        warm_up()
        for monitor in (True, False):
            net, args, dataset = build_mas(scalefree_mas_model, monitor_task_loss=monitor)
            run_steps(net, args, dataset, 10)
            # force an active penalty: the toy graph rarely drives the loss under 0.2
            net.omegas = [torch.ones_like(p) * 0.01 for p in net.net.parameters()]
            net.star_variables = [p.detach().clone() + 1.0 for p in net.net.parameters()]
            run_steps(net, args, dataset, 10, seed=1)
            penalised = [t for t, o in zip(net.task_losses, net.total_losses) if o > t + 1e-9]
            self.assertTrue(penalised, 'expected the MAS penalty to be active')
            window_source = net.task_losses if monitor else net.total_losses
            self.assertAlmostEqual(
                net.monitored_losses[-1], float(np.mean(window_source[-5:])), places=4)

    def test_enabled_arm_consolidates_at_least_as_often(self):
        steps = 60
        warm_up()
        disabled, args_a, data_a = build_mas(scalefree_mas_model, monitor_task_loss=False)
        run_steps(disabled, args_a, data_a, steps)
        enabled, args_b, data_b = build_mas(scalefree_mas_model, monitor_task_loss=True)
        run_steps(enabled, args_b, data_b, steps)
        self.assertGreaterEqual(enabled.count_updates, disabled.count_updates)
        self.assertEqual(enabled.optimizer_steps, disabled.optimizer_steps)

    def test_thresholds_and_latch_are_unchanged(self):
        net, _, _ = build_mas(scalefree_mas_model)
        self.assertEqual(float(np.asarray(net.loss_window_mean_threshold).ravel()[0]), 0.2)
        self.assertEqual(float(np.asarray(net.loss_window_variance_threshold).ravel()[0]), 0.1)
        self.assertEqual(net.loss_window_length, 5)
        self.assertEqual(net.MAS_weight, 0.5)

    def test_latch_disarms_after_each_consolidation(self):
        net, args, dataset = build_mas(scalefree_mas_model)
        run_steps(net, args, dataset, 60)
        self.assertEqual(len(net.consolidation_steps), net.count_updates)
        self.assertEqual(len(set(net.consolidation_steps)), len(net.consolidation_steps))
        self.assertEqual(sorted(net.consolidation_steps), net.consolidation_steps)

    def test_no_replay_state_and_budget_is_unchanged(self):
        net, args, dataset = build_mas(scalefree_mas_model)
        run_steps(net, args, dataset, 60)
        memory = net.memory_accounting()
        self.assertEqual(memory['buffer_bytes'], 0)
        accounting = net.compute_accounting()
        self.assertEqual(accounting['replay_seed_rows'], 0)
        self.assertEqual(accounting['replay_edges'], 0)
        self.assertEqual(accounting['optimizer_steps'], 60)
        self.assertGreater(accounting['distinct_classes_delivered'], 0)


if __name__ == '__main__':
    unittest.main()
