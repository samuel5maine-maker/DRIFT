import unittest

import dgl
import torch

from Baselines import context_ema_model, context_teacher_model
from tests.test_new_baselines import build, run_steps


class TestContextTeacher(unittest.TestCase):
    def test_gate_requires_correct_and_more_confident_teacher(self):
        net, _, _ = build(context_teacher_model)
        student = torch.tensor(
            [[1.0, 0.0], [0.2, 0.8], [0.9, 0.1], [3.0, 0.0]], requires_grad=True,
        )
        teacher = torch.tensor(
            [[2.0, 0.0], [0.1, 0.9], [0.1, 0.9], [2.0, 0.0]], requires_grad=True,
        )
        labels = torch.tensor([0, 1, 0, 0])
        loss = net._feedback_loss(lambda value: value, student, teacher, labels)
        loss.backward()
        self.assertEqual(net.teacher_gate_rows, 2)
        self.assertGreater(float(loss), 0)
        self.assertIsNone(teacher.grad)
        self.assertIsNotNone(student.grad)
        self.assertTrue(torch.equal(student.grad[2], torch.zeros_like(student.grad[2])))
        self.assertTrue(torch.equal(student.grad[3], torch.zeros_like(student.grad[3])))

    def test_feedback_does_not_consume_unseen_head_columns(self):
        net, _, _ = build(context_teacher_model)
        labels = torch.tensor([0, 1])
        student_a = torch.tensor([[0.0, 1.0, -2.0, 8.0], [1.0, 0.0, 7.0, -3.0]], requires_grad=True)
        student_b = student_a.detach().clone().requires_grad_(True)
        student_b.data[:, 2:] = torch.tensor([[100.0, -100.0], [-100.0, 100.0]])
        teacher_a = torch.tensor([[0.0, 2.0, 9.0, -4.0], [2.0, 0.0, -5.0, 9.0]])
        teacher_b = teacher_a.clone()
        teacher_b[:, 2:] *= -20
        head = lambda value: value[:, :2]
        loss_a = net._feedback_loss(head, student_a, teacher_a, labels)
        loss_b = net._feedback_loss(head, student_b, teacher_b, labels)
        loss_a.backward()
        loss_b.backward()
        self.assertTrue(torch.equal(loss_a, loss_b))
        self.assertTrue(torch.equal(student_a.grad[:, :2], student_b.grad[:, :2]))
        self.assertTrue(torch.equal(student_a.grad[:, 2:], torch.zeros_like(student_a.grad[:, 2:])))

    def test_teacher_reuses_student_blocks_and_is_detached(self):
        net, args, dataset = build(context_teacher_model)
        calls = []
        original = net._model_logits
        def recording_logits(model, blocks):
            calls.append((model, tuple(id(block) for block in blocks)))
            return original(model, blocks)
        net._model_logits = recording_logits
        run_steps(net, args, dataset, 2)
        self.assertEqual(net.teacher_forward_calls, 1)
        self.assertEqual(net.teacher_forward_source_nodes, net.replay_source_nodes)
        self.assertEqual(net.teacher_forward_edges, net.replay_edges)
        self.assertTrue(all(parameter.grad is None for parameter in net.ema.parameters()))
        self.assertIs(calls[-2][0], net.net)
        self.assertIs(calls[-1][0], net.ema)
        self.assertEqual(calls[-2][1], calls[-1][1])
        self.assertIsNone(net._last_replay_blocks)
        self.assertIsNone(net._last_replay_index)

    def test_disabled_feedback_exactly_matches_parent(self):
        parent, args_p, dataset_p = build(context_ema_model, seed=7)
        control, args_c, dataset_c = build(context_teacher_model, seed=7, feedback_enabled=False)
        torch.manual_seed(19)
        dgl.random.seed(19)
        run_steps(parent, args_p, dataset_p, 25, seed=3)
        torch.manual_seed(19)
        dgl.random.seed(19)
        run_steps(control, args_c, dataset_c, 25, seed=3)
        for expected, actual in zip(parent.net.parameters(), control.net.parameters()):
            self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(parent.buffer.ids, control.buffer.ids)
        self.assertEqual(parent.buffer.labels, control.buffer.labels)
        self.assertEqual(control.teacher_forward_calls, 0)

    def test_budget_one_step_and_working_inference(self):
        net, args, dataset = build(
            context_teacher_model, buffer='cbrs', classifier_norm='fixed', ema_alpha=0.995,
        )
        run_steps(net, args, dataset, 25)
        self.assertEqual(net.optimizer_steps, 25)
        self.assertEqual(len(net.buffer), 100)
        self.assertEqual(net.buffer.payload, {})
        self.assertEqual(net.memory_accounting()['buffer_bytes'], 1600)
        self.assertEqual(
            net.memory_accounting()['extra_param_count'],
            sum(parameter.numel() for parameter in net.net.parameters()),
        )
        self.assertEqual(net.replay_rows_per_update, [0] + [args.batch_size] * 24)
        self.assertIs(net.eval_model(), net._working_eval)
        self.assertEqual(set(net.alt_eval_models()), {'ema'})
        accounting = net.compute_accounting()
        self.assertEqual(accounting['teacher_replay_rows'], sum(net.replay_rows_per_update))
        self.assertGreaterEqual(accounting['teacher_mean_kl'], 0)


if __name__ == '__main__':
    unittest.main()
