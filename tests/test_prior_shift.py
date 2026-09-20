import math
import unittest

import torch

from Baselines import context_teacher_model, prior_shift_model
from tests.test_new_baselines import build, run_steps

BASE = dict(budget=100, buffer='cbrs', replay='context', ema_alpha=0.995,
            classifier_norm='fixed', feedback_enabled=False)


class TestPriorShift(unittest.TestCase):
    def test_adjustment_is_log_count_plus_one_over_incoming_and_replay(self):
        net, _, _ = build(prior_shift_model, **BASE)
        incoming = torch.tensor([0, 0, 0, 1])
        replay = torch.tensor([1, 3])
        adjustment = net._prior_adjustment(5, [incoming, replay], torch.device('cpu'))
        expected = torch.tensor([math.log(4), math.log(3), 0.0, math.log(2), 0.0])
        self.assertTrue(torch.allclose(adjustment, expected))
        # a present class is handicapped by exactly log(n+1) against an absent one
        self.assertAlmostEqual(float(adjustment[0] - adjustment[2]), math.log(4), places=6)
        self.assertEqual(net.distinct_class_sum, 3)
        self.assertEqual(net.positive_rows, 6)

    def test_absent_classes_receive_less_downward_gradient_than_plain_ce(self):
        logits = torch.zeros(4, 6)
        labels = torch.tensor([0, 0, 0, 0])
        plain = logits.clone().requires_grad_(True)
        torch.nn.functional.cross_entropy(plain, labels).backward()
        net, _, _ = build(prior_shift_model, **BASE)
        adjustment = net._prior_adjustment(6, [labels], torch.device('cpu'))
        adjusted = logits.clone().requires_grad_(True)
        torch.nn.functional.cross_entropy(adjusted + adjustment, labels).backward()
        absent = slice(1, 6)
        self.assertTrue(torch.all(adjusted.grad[:, absent] < plain.grad[:, absent]))
        self.assertTrue(torch.all(adjusted.grad[:, absent] > 0))

    def test_inference_logits_are_not_adjusted(self):
        net, args, dataset = build(prior_shift_model, **BASE)
        run_steps(net, args, dataset, 5)
        self.assertIs(net.eval_model(), net.net)

    def test_disabled_arm_matches_the_h30_control_exactly(self):
        steps = 40                                   # past the 100-slot buffer's first overflow
        control, control_args, control_data = build(context_teacher_model, **BASE)
        run_steps(control, control_args, control_data, steps)
        disabled, disabled_args, disabled_data = build(
            prior_shift_model, **dict(BASE, prior_shift_enabled=False))
        run_steps(disabled, disabled_args, disabled_data, steps)
        for left, right in zip(control.net.parameters(), disabled.net.parameters()):
            self.assertTrue(torch.equal(left, right))
        for left, right in zip(control.ema.parameters(), disabled.ema.parameters()):
            self.assertTrue(torch.equal(left, right))
        self.assertEqual(control.buffer.ids, disabled.buffer.ids)
        self.assertEqual(control.buffer.labels, disabled.buffer.labels)
        self.assertEqual(control.replay_rows_per_update, disabled.replay_rows_per_update)
        self.assertEqual(control.replay_source_nodes, disabled.replay_source_nodes)
        self.assertEqual(disabled.adjusted_updates, 0)

    def test_enabled_arm_changes_the_trajectory(self):
        steps = 40
        disabled, args_a, data_a = build(prior_shift_model, **dict(BASE, prior_shift_enabled=False))
        run_steps(disabled, args_a, data_a, steps)
        enabled, args_b, data_b = build(prior_shift_model, **BASE)
        run_steps(enabled, args_b, data_b, steps)
        self.assertEqual(enabled.adjusted_updates, steps)
        self.assertFalse(all(torch.equal(left, right) for left, right
                             in zip(disabled.net.parameters(), enabled.net.parameters())))
        self.assertEqual(enabled.replay_rows_per_update, disabled.replay_rows_per_update)
        self.assertEqual(enabled.optimizer_steps, disabled.optimizer_steps)
        self.assertEqual(enabled.buffer.ids, disabled.buffer.ids)

    def test_budget_is_unchanged_and_state_is_not_retained(self):
        net, args, dataset = build(prior_shift_model, **BASE)
        run_steps(net, args, dataset, 40)
        memory = net.memory_accounting()
        self.assertLessEqual(memory['buffer_bytes'], 2400)
        self.assertEqual(net.optimizer_steps, 40)
        self.assertEqual(net.replay_rows_per_update, [0] + [args.batch_size] * 39)
        self.assertIsNone(net._last_replay_blocks)
        accounting = net.compute_accounting()
        self.assertEqual(accounting['teacher_forward_calls'], 0)
        self.assertGreater(accounting['prior_shift_mean_adjustment'], 0)
        self.assertGreater(accounting['prior_shift_negative_to_positive_ratio'], 1)

    def test_feedback_cannot_be_enabled_alongside_the_correction(self):
        with self.assertRaises(ValueError):
            build(prior_shift_model, **dict(BASE, feedback_enabled=True))


if __name__ == '__main__':
    unittest.main()
