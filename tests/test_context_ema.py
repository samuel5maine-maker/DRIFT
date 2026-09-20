import unittest

import torch

from Baselines import context_ema_model
from tests.test_new_baselines import build, pipeline_graph, run_steps


class TestContextEMA(unittest.TestCase):
    def test_context_and_isolated_obey_same_budget(self):
        for replay in ('context', 'isolated'):
            net, args, dataset = build(
                context_ema_model, budget=100, memory_proportion=1,
                buffer='reservoir', replay=replay, ema_alpha=0.995,
            )
            run_steps(net, args, dataset, 25)
            self.assertEqual(len(net.buffer), 100)
            self.assertEqual(net.buffer.payload, {})
            self.assertEqual(net.optimizer_steps, 25)
            self.assertEqual(net.replay_rows_per_update, [0] + [5] * 24)
            self.assertEqual(net.memory_accounting()['buffer_bytes'], 1600)
            self.assertGreater(net.replay_source_nodes, 0)
            self.assertGreater(net.replay_edges, 0)

    def test_context_uses_more_graph_work_than_isolated(self):
        context, args, dataset = build(context_ema_model, replay='context')
        run_steps(context, args, dataset, 5)
        isolated, args_i, dataset_i = build(context_ema_model, replay='isolated')
        run_steps(isolated, args_i, dataset_i, 5)
        self.assertGreater(context.replay_source_nodes, isolated.replay_source_nodes)
        self.assertGreater(context.replay_edges, isolated.replay_edges)

    def test_ema_object_is_stable_and_first_step_is_exact(self):
        net, args, dataset = build(context_ema_model, replay='context')
        evaluation_object = net.eval_model()
        graph = pipeline_graph(dataset)
        labels = graph.ndata['label'].squeeze()
        ids = torch.arange(args.batch_size)
        net.observe_cis(args, graph, graph.ndata['feat'], labels, ids)
        self.assertIs(net.eval_model(), evaluation_object)
        for student, average in zip(net.net.parameters(), net.ema.parameters()):
            self.assertTrue(torch.equal(student, average))

    def test_adaptive_ema_uses_only_consecutive_observed_label_overlap(self):
        net, _, _ = build(
            context_ema_model, ema_mode='adaptive', ema_alpha=0.995, ema_fast=0.99,
        )
        with torch.no_grad():
            for parameter in net.net.parameters():
                parameter.fill_(1)
        net._update_ema(torch.tensor([0, 1]))
        with torch.no_grad():
            for parameter in net.net.parameters():
                parameter.fill_(2)
        net._update_ema(torch.tensor([0, 1]))
        self.assertEqual(net.ema_alphas[-1], 0.995)
        net._update_ema(torch.tensor([2, 3]))
        self.assertEqual(net.ema_alphas[-1], 0.99)

    def test_rejects_unfair_budget_or_ratio(self):
        with self.assertRaises(ValueError):
            build(context_ema_model, budget=101)
        with self.assertRaises(ValueError):
            build(context_ema_model, memory_proportion=2)
        with self.assertRaises(ValueError):
            build(context_ema_model, replay_weight=-0.1)
        with self.assertRaises(ValueError):
            build(context_ema_model, classifier_norm_scale=0)
        with self.assertRaises(ValueError):
            build(context_ema_model, replay_neighbors='invalid')
        with self.assertRaises(ValueError):
            build(context_ema_model, cosine_scale=0)
        with self.assertRaises(ValueError):
            build(context_ema_model, smoothness_weight=-0.1)
        with self.assertRaises(ValueError):
            build(context_ema_model, head_lr_multiplier=0)

    def test_head_lr_multiplier_changes_only_optimizer_rate(self):
        net, _, _ = build(context_ema_model, head_lr_multiplier=2.0)
        self.assertEqual(len(net.opt.param_groups), 2)
        self.assertAlmostEqual(net.opt.param_groups[1]['lr'], 2 * net.opt.param_groups[0]['lr'])
        head_ids = {id(parameter) for parameter in net.net.gat_layers[-1].parameters()}
        optimized_head_ids = {id(parameter) for parameter in net.opt.param_groups[1]['params']}
        self.assertEqual(optimized_head_ids, head_ids)

    def test_new_class_sync_copies_only_new_ema_rows(self):
        net, _, _ = build(context_ema_model, sync_new_class_rows=True)
        net._update_ema(torch.tensor([0, 1]))
        with torch.no_grad():
            working = net.net.gat_layers[-1].linear.weight
            ema = net.ema.gat_layers[-1].linear.weight
            working[0].fill_(3)
            working[2].fill_(4)
            ema[0].zero_()
            ema[2].zero_()
        net._update_ema(torch.tensor([2]))
        working = net.net.gat_layers[-1].linear.weight
        ema = net.ema.gat_layers[-1].linear.weight
        self.assertTrue(torch.equal(ema[2], working[2]))
        self.assertFalse(torch.equal(ema[0], working[0]))

    def test_cosine_classifier_is_finite_and_keeps_parameter_count(self):
        linear, _, _ = build(context_ema_model, classifier='linear')
        cosine, args, dataset = build(context_ema_model, classifier='cosine', cosine_scale=10)
        self.assertEqual(
            sum(parameter.numel() for parameter in linear.net.parameters()),
            sum(parameter.numel() for parameter in cosine.net.parameters()),
        )
        graph = pipeline_graph(dataset)
        labels = graph.ndata['label'].squeeze()
        ids = torch.arange(args.batch_size)
        cosine.observe_cis(args, graph, graph.ndata['feat'], labels, ids)
        output, _ = cosine.eval_model()(graph, graph.ndata['feat'])
        self.assertEqual(output.shape, (graph.num_nodes(), args.n_cls))
        self.assertTrue(torch.isfinite(output).all())

    def test_hard_ace_runs_without_extra_replay(self):
        net, args, dataset = build(context_ema_model, ace=True, classifier_norm='fixed')
        run_steps(net, args, dataset, 8)
        self.assertEqual(net.optimizer_steps, 8)
        self.assertEqual(net.replay_rows_per_update, [0] + [args.batch_size] * 7)

    def test_degree_memory_keeps_capacity_and_metadata_bound(self):
        net, args, dataset = build(context_ema_model, buffer='degree_cbrs')
        run_steps(net, args, dataset, 25)
        self.assertEqual(len(net.buffer), 100)
        self.assertEqual(len(net.buffer.scores), 100)
        self.assertEqual(net.memory_accounting()['buffer_bytes'], 2400)

    def test_classifier_projection_equalizes_rows_without_extra_parameters(self):
        net, _, _ = build(context_ema_model, classifier_norm=True)
        before = sum(parameter.numel() for parameter in net.net.parameters())
        with torch.no_grad():
            weight = net.net.gat_layers[-1].linear.weight
            weight[0].mul_(3)
            weight[1].mul_(0.2)
        net._project_classifier()
        norms = net.net.gat_layers[-1].linear.weight.norm(dim=1)
        self.assertTrue(torch.allclose(norms, norms[0].expand_as(norms), atol=1e-6))
        self.assertEqual(before, sum(parameter.numel() for parameter in net.net.parameters()))

    def test_ema_classifier_projection_equalizes_inference_rows(self):
        net, _, _ = build(context_ema_model, classifier_norm='fixed', project_ema_classifier=True)
        with torch.no_grad():
            weight = net.ema.gat_layers[-1].linear.weight
            weight[0].mul_(4)
            weight[1].mul_(0.1)
        net._project_ema_classifier()
        norms = net.ema.gat_layers[-1].linear.weight.norm(dim=1)
        self.assertTrue(torch.allclose(norms, norms[0].expand_as(norms), atol=1e-6))


if __name__ == '__main__':
    unittest.main()
