"""
Checks for the replay baselines added to DRIFT (ER-CBRS, DER, DER++), on a toy graph with a real GCN backbone.

    python -m unittest tests.test_new_baselines -v
"""
import os
import sys
import unittest
from argparse import Namespace

import dgl
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Backbones.gnns import GCN  # noqa: E402
from Baselines import der_model, derpp_model, er_cbrs_model  # noqa: E402


def make_args(**method_args):
    a = Namespace(d_data=8, n_cls=6, GCN_args={'h_dims': [16], 'dropout': 0.0, 'batch_norm': False}, lr=0.01,
                  weight_decay=0.0, epochs=1, cuda=False, gpu=0, sample_nbs=True, n_nbs_sample=[3, 3], batch_size=5,
                  dataset='toy', backbone='GCN', setting='tfo_gaussian', seed=0)
    for k, v in method_args.items():
        setattr(a, k, v)
    return a


def make_dataset(n=40, seed=0):
    gen = torch.Generator().manual_seed(seed)
    g = dgl.graph((torch.randint(0, n, (120,), generator=gen), torch.randint(0, n, (120,), generator=gen)), num_nodes=n)
    g.ndata['feat'] = torch.randn(n, 8, generator=gen)
    g.ndata['label'] = torch.randint(0, 6, (n, 1), generator=gen)
    return Namespace(graph=g)


def pipeline_graph(dataset):
    sub = dgl.node_subgraph(dataset.graph, torch.arange(dataset.graph.num_nodes()), store_ids=True)
    return dgl.add_self_loop(sub)


def build(module, seed=0, **hp):
    torch.manual_seed(seed)
    dgl.random.seed(seed)
    dgl.utils.set_num_threads(1)
    args = make_args(**{module.NET.ARGS_ATTR: hp})
    dataset = make_dataset()
    return module.NET(GCN(args), args, dataset=dataset), args, dataset


def run_steps(net, args, dataset, n_steps, seed=0):
    g = pipeline_graph(dataset)
    labels = g.ndata['label'].squeeze()
    rng = np.random.RandomState(seed)
    for _ in range(n_steps):
        ids = torch.tensor(rng.choice(g.num_nodes(), args.batch_size, replace=False), dtype=torch.long)
        net.observe_cis(args, g, g.ndata['feat'], labels, ids)


class TestERCBRS(unittest.TestCase):
    def test_runs_and_fills_balanced_memory(self):
        net, args, dataset = build(er_cbrs_model, budget=12)
        run_steps(net, args, dataset, 30)
        self.assertEqual(len(net.buffer), 12)
        counts = net.buffer.class_counts()
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_weighted_replay_and_convex_loss_run(self):
        net, args, dataset = build(er_cbrs_model, budget=12, replay='weighted', loss='convex')
        run_steps(net, args, dataset, 10)
        self.assertEqual(len(net.buffer), 12)

    def test_memory_accounting(self):
        net, args, dataset = build(er_cbrs_model, budget=12)
        run_steps(net, args, dataset, 10)
        self.assertEqual(net.memory_accounting()['buffer_bytes'], 16 * 12)


class TestDER(unittest.TestCase):
    def test_stored_logits_are_prestep_isolated_logits(self):
        net, args, dataset = build(der_model, budget=200)          # large budget: every node stays in memory
        g = pipeline_graph(dataset)
        labels = g.ndata['label'].squeeze()
        ids = torch.tensor([3, 7, 11, 19, 23])
        with torch.no_grad():
            expected = net._isolated_logits(net.net, [3, 7, 11, 19, 23], torch.device('cpu')).clone()
        before = [p.detach().clone() for p in net.net.parameters()]
        net.observe_cis(args, g, g.ndata['feat'], labels, ids)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, net.net.parameters())))  # a step happened
        stored = torch.stack(net.buffer.payload['logits'])
        self.assertTrue(torch.allclose(stored, expected, atol=1e-6))
        self.assertEqual(stored.shape[1], 6)                     # full head, not the class-IL slice

    def test_mse_term_uses_alpha(self):
        net, args, dataset = build(der_model, budget=20, alpha=0.0)
        run_steps(net, args, dataset, 6)
        terms = net._replay_terms(lambda o: o, torch.device('cpu'))
        self.assertEqual(float(terms['mse']), 0.0)

    def test_isolated_logits_handle_duplicate_nodes(self):
        net, args, dataset = build(der_model)
        out = net._isolated_logits(net.net, [5, 2, 5, 9], torch.device('cpu'))
        self.assertTrue(torch.equal(out[0], out[2]))
        self.assertEqual(out.shape[0], 4)


class TestDERpp(unittest.TestCase):
    def test_two_independent_buffer_draws(self):
        net, args, dataset = build(derpp_model, budget=20)
        run_steps(net, args, dataset, 6)
        calls = []
        original = net._replay_draw
        net._replay_draw = lambda: calls.append(1) or original()
        net._replay_terms(lambda o: o, torch.device('cpu'))
        self.assertEqual(len(calls), 2)

    def test_has_both_terms(self):
        net, args, dataset = build(derpp_model, budget=20, alpha=0.2, beta=1.0)
        run_steps(net, args, dataset, 6)
        terms = net._replay_terms(lambda o: o, torch.device('cpu'))
        self.assertEqual(set(terms), {'mse', 'ce'})

    def test_deterministic_given_seed(self):
        a, args, ds = build(derpp_model, seed=4, budget=20)
        run_steps(a, args, ds, 12, seed=4)
        b, args_b, ds_b = build(derpp_model, seed=4, budget=20)
        run_steps(b, args_b, ds_b, 12, seed=4)
        for pa, pb in zip(a.net.parameters(), b.net.parameters()):
            self.assertTrue(torch.equal(pa, pb))


class TestLwFOnline(unittest.TestCase):
    def test_teacher_refreshes_every_update_every_batches(self):
        from Baselines import lwf_online_model
        net, args, dataset = build(lwf_online_model, update_every=3)
        run_steps(net, args, dataset, 2)
        self.assertIsNone(net.teacher)
        run_steps(net, args, dataset, 1)
        self.assertIsNotNone(net.teacher)
        for pt, ps in zip(net.teacher.parameters(), net.net.parameters()):
            self.assertTrue(torch.equal(pt, ps))
            self.assertFalse(pt.requires_grad)

    def test_distillation_ignores_batch_classes(self):
        from Baselines.lwf_online_model import multiclass_cross_entropy
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        self.assertAlmostEqual(float(multiclass_cross_entropy(logits, logits, 2.0)),
                               float(-(torch.softmax(logits / 2, 1) * torch.log_softmax(logits / 2, 1)).sum()), places=6)


class TestCLSER(unittest.TestCase):
    def test_ema_ramp_and_update(self):
        from Baselines import clser_model
        net, args, dataset = build(clser_model, budget=20, plastic_update_freq=1.0, stable_update_freq=0.0)
        run_steps(net, args, dataset, 1)
        # global_step=1 -> alpha = min(1 - 1/2, alpha_max) = 0.5: plastic = 0.5*init + 0.5*working; stable untouched
        self.assertEqual(net.global_step, 1)
        self.assertFalse(any(torch.equal(p, w) for p, w in zip(net.plastic.parameters(), net.net.parameters())))

    def test_consistency_target_selection(self):
        from Baselines import clser_model
        net, args, dataset = build(clser_model, budget=20)
        run_steps(net, args, dataset, 4)
        with torch.no_grad():
            for p in net.stable.parameters():
                p.mul_(3.0)                                # make the two memories disagree
        ids = net.buffer.ids[:6]
        y = torch.tensor(net.buffer.labels[:6])
        target = net._consistency_target(ids, y, torch.device('cpu'))
        s = net._isolated_logits(net.stable, ids, torch.device('cpu'))
        p = net._isolated_logits(net.plastic, ids, torch.device('cpu'))
        for i in range(6):
            s_better = torch.softmax(s[i], 0)[y[i]] > torch.softmax(p[i], 0)[y[i]]
            self.assertTrue(torch.equal(target[i], s[i] if s_better else p[i]))

    def test_eval_model_is_stable_by_default(self):
        from Baselines import clser_model
        net, args, dataset = build(clser_model)
        self.assertIs(net.eval_model(), net.stable)
        self.assertEqual(set(net.alt_eval_models()), {'working', 'plastic'})


class TestDERCLS(unittest.TestCase):
    def test_three_independent_draws_and_terms(self):
        from Baselines import dercls_model
        net, args, dataset = build(dercls_model, budget=20)
        run_steps(net, args, dataset, 6)
        calls = []
        original = net._replay_draw
        net._replay_draw = lambda: calls.append(1) or original()
        terms = net._replay_terms(lambda o: o, torch.device('cpu'))
        self.assertEqual(len(calls), 3)
        self.assertEqual(set(terms), {'ce', 'stored_logits', 'ema_logits'})

    def test_ablations_drop_terms(self):
        from Baselines import dercls_model
        for hp, expected in (({'gamma': 0.0}, {'ce', 'stored_logits'}), ({'beta': 0.0}, {'ce', 'ema_logits'})):
            net, args, dataset = build(dercls_model, budget=20, **hp)
            run_steps(net, args, dataset, 4)
            self.assertEqual(set(net._replay_terms(lambda o: o, torch.device('cpu'))), expected)

    def test_uses_cbrs_and_ema_eval(self):
        from Baselines import dercls_model
        from Baselines.replay_buffers import CBRSBuffer
        net, args, dataset = build(dercls_model)
        self.assertIsInstance(net.buffer, CBRSBuffer)
        self.assertIs(net.eval_model(), net.ema)


class TestPDGNN(unittest.TestCase):
    def _build(self, seed=0, **hp):
        from Backbones.sgc import SGC_MLP
        from Baselines import pdgnn_model
        torch.manual_seed(seed)
        dgl.random.seed(seed)
        args = make_args(pdgnn_args=hp, backbone='SGC', SGC_args={'h_dims': [16], 'k': 2})
        dataset = make_dataset()
        return pdgnn_model.NET(SGC_MLP(args), args, dataset=dataset), args, dataset

    def test_block_propagation_matches_full_graph(self):
        net, args, dataset = self._build()
        g = pipeline_graph(dataset)
        net.net.annotate_degrees(g)
        seeds = torch.tensor([1, 4, 9, 16])
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        _, _, blocks = sampler.sample_blocks(g, seeds)
        via_blocks = net.net.propagate_blocks(blocks, blocks[0].srcdata['feat'])
        via_graph = net.net.propagate_graph(g, g.ndata['feat'])[seeds]
        self.assertTrue(torch.allclose(via_blocks, via_graph, atol=1e-5))

    def test_memory_stores_embeddings_and_replay_uses_mlp_only(self):
        net, args, dataset = self._build(budget=15)
        run_steps(net, args, dataset, 8)
        self.assertEqual(len(net.buffer), 15)
        self.assertEqual(net.buffer.payload['emb'][0].shape, (8,))
        self.assertEqual(net.memory_accounting()['buffer_bytes'], 15 * (16 + 8 * 4))

    def test_requires_sgc_backbone(self):
        from Baselines import pdgnn_model
        args = make_args(pdgnn_args={})
        with self.assertRaises(ValueError):
            pdgnn_model.NET(GCN(args), args, dataset=make_dataset())


if __name__ == '__main__':
    unittest.main()
