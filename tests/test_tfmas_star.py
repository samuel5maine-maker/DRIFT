"""
Checks for Baselines/tfmas_star_model.py (MAS*, Algorithm 1 of arXiv:1812.03596v3).

Run from the repo root:  python -m unittest tests.test_tfmas_star -v
"""
import collections
import os
import sys
import tempfile
import unittest
from argparse import Namespace

import numpy as np
import torch
import dgl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Backbones.gnns import GCN  # noqa: E402
from Baselines import tfmas_star_model  # noqa: E402
from Baselines.tfmas_star_model import NET, plateau_detector  # noqa: E402


def make_args(**hp):
    return Namespace(
        d_data=8, n_cls=6, GCN_args={'h_dims': [16], 'dropout': 0.0, 'batch_norm': False},
        lr=0.01, weight_decay=0.0, epochs=1, cuda=False, gpu=0, sample_nbs=True, n_nbs_sample=[3, 3],
        tfmas_star_args=hp, dataset='toy', backbone='GCN', setting='tfocis', seed=0,
    )


def make_dataset(n=40, seed=0):
    gen = torch.Generator().manual_seed(seed)
    src = torch.randint(0, n, (120,), generator=gen)
    dst = torch.randint(0, n, (120,), generator=gen)
    g = dgl.graph((src, dst), num_nodes=n)
    g.ndata['feat'] = torch.randn(n, 8, generator=gen)
    g.ndata['label'] = torch.randint(0, 6, (n,), generator=gen)
    return Namespace(graph=g)


def run_steps(net, args, dataset, n_steps, seed=0):
    """Feed batches of 5 nodes from a node-subgraph (as the pipelines do) and return them."""
    sub = dgl.node_subgraph(dataset.graph, torch.arange(dataset.graph.num_nodes()), store_ids=True)
    sub = dgl.add_self_loop(sub)
    labels = sub.ndata['label']
    rng = np.random.RandomState(seed)
    for _ in range(n_steps):
        ids = torch.tensor(rng.choice(sub.num_nodes(), 5, replace=False), dtype=torch.long)
        net.observe_cis(args, sub, sub.ndata['feat'], labels, ids)


def build(hp, seed=0):
    torch.manual_seed(seed)
    dgl.random.seed(seed)
    dgl.utils.set_num_threads(1)   # as training.utils.set_seed: reproducible sampling
    args = make_args(**hp)
    dataset = make_dataset()
    return NET(GCN(args), args, dataset=dataset), args, dataset


class TestPlateauDetector(unittest.TestCase):
    def test_fires_on_plateau_and_clears_window(self):
        W = collections.deque([0.10, 0.11, 0.09, 0.10, 0.10], maxlen=5)
        d = plateau_detector(W, False, 0.0, 0.0, l_th=0.5, std_th=0.05)
        self.assertTrue(d['consolidate'])
        self.assertTrue(d['P'])
        self.assertEqual(len(W), 0)
        self.assertAlmostEqual(d['mu_old'], 0.10, places=6)
        self.assertAlmostEqual(d['sigma_old'], float(np.std([0.10, 0.11, 0.09, 0.10, 0.10])), places=9)

    def test_blocked_until_peak_then_rearmed(self):
        W = collections.deque(maxlen=5)
        P, mu_old, sigma_old = True, 0.10, 0.01
        for v in [0.10, 0.10, 0.10]:                       # still at plateau level: stays blocked
            W.append(v)
            d = plateau_detector(W, P, mu_old, sigma_old, l_th=0.5, std_th=0.05)
            self.assertFalse(d['consolidate'])
            self.assertFalse(d['peak_fired'])
            P = d['P']
        self.assertTrue(P)
        W.append(2.0)                                      # mean now > mu_old + sigma_old: peak
        d = plateau_detector(W, P, mu_old, sigma_old, l_th=0.5, std_th=0.05)
        self.assertTrue(d['peak_fired'])
        self.assertFalse(d['P'])
        self.assertFalse(d['consolidate'])                 # line 13 is evaluated before line 19

    def test_each_condition_blocks(self):
        plateau = [0.1] * 5
        self.assertFalse(plateau_detector(collections.deque(plateau), False, 0, 0, 0.05, 1.0)['consolidate'])  # mean
        self.assertFalse(plateau_detector(collections.deque([0.0, 0.2] * 2 + [0.1]), False, 0, 0, 1.0, 0.01)['consolidate'])  # std
        self.assertFalse(plateau_detector(collections.deque(plateau), True, 0.1, 0.0, 1.0, 1.0)['consolidate'])  # latch
        self.assertFalse(plateau_detector(collections.deque(plateau), False, 0, 0, 1.0, 1.0, has_buffer=False)['consolidate'])

    def test_uses_std_not_variance(self):
        W = collections.deque([0.0, 0.6] * 2 + [0.3])     # std ~0.27, variance ~0.072
        self.assertFalse(plateau_detector(W, False, 0, 0, l_th=1.0, std_th=0.1)['consolidate'])

    def test_empty_window_never_fires_or_peaks(self):
        d = plateau_detector(collections.deque(), False, 0.0, 0.0, 1.0, 1.0)
        self.assertFalse(d['consolidate'])
        self.assertFalse(d['peak_fired'])


class TestNET(unittest.TestCase):
    def test_requires_thresholds(self):
        args = make_args()
        with self.assertRaises(ValueError):
            NET(GCN(args), args, dataset=make_dataset())

    def test_detector_disabled_is_buffer_only(self):
        net, args, dataset = build({'l_th': -1.0, 'std_th': -1.0, 'buffer_size': 8})
        run_steps(net, args, dataset, 20)
        self.assertEqual(net.n_consolidations, 0)
        self.assertTrue(all(float(o.abs().sum()) == 0.0 for o in net.omega))
        self.assertEqual(len(net.buffer_ids), 8)

    def test_omega_cumulative_moving_average(self):
        net, args, dataset = build({'l_th': 1e9, 'std_th': 1e9, 'buffer_size': 8})
        estimates = iter([1.0, 2.0, 6.0])
        net._omega_estimate = lambda aux_blocks, head: [torch.full_like(p, next(estimates)) if i == 0
                                                         else torch.zeros_like(p)
                                                         for i, p in enumerate(net.net.parameters())]
        run_steps(net, args, dataset, 2)        # step 0 has no buffer; step 1 consolidates
        self.assertEqual(net.n_consolidations, 1)
        for _ in range(2):
            net.P = False                       # force re-arm to consolidate again
            run_steps(net, args, dataset, 1)
        self.assertEqual(net.n_consolidations, 3)
        self.assertTrue(torch.allclose(net.omega[0], torch.full_like(net.omega[0], 3.0)))  # mean(1, 2, 6)

    def test_anchor_and_penalty_after_consolidation(self):
        net, args, dataset = build({'l_th': 1e9, 'std_th': 1e9, 'buffer_size': 8})
        run_steps(net, args, dataset, 2)
        self.assertEqual(net.n_consolidations, 1)
        self.assertTrue(any(float(o.abs().sum()) > 0 for o in net.omega))
        for s, p in zip(net.theta_star, net.net.parameters()):
            self.assertTrue(torch.equal(s, p.detach()))
        self.assertEqual(len(net.W), 0)

    def test_prioritized_keeping_exact_on_one_step(self):
        net, args, dataset = build({'l_th': -1.0, 'std_th': -1.0, 'buffer_size': 3})
        sub = dgl.add_self_loop(dgl.node_subgraph(dataset.graph, torch.arange(dataset.graph.num_nodes()), store_ids=True))
        net.seen_classes.update(range(6))
        net.observe_cis(args, sub, sub.ndata['feat'], sub.ndata['label'], torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]))
        net.net.eval()
        g = net._isolated_graph(list(range(8)), torch.device('cpu'))
        blocks = net._full_blocks(g)
        with torch.no_grad():
            losses = torch.nn.functional.cross_entropy(net._logits(blocks)[:, :6],
                                                       blocks[-1].dstdata['label'], reduction='none')
        top3 = set(int(g.ndata['_ID'][i]) for i in torch.argsort(losses, descending=True)[:3])
        self.assertEqual(set(net.buffer_ids), top3)

    def test_telemetry_is_inert(self):
        hp = {'l_th': 1e9, 'std_th': 1e9, 'buffer_size': 8}
        net_a, args, dataset = build(hp, seed=3)
        run_steps(net_a, args, dataset, 15, seed=3)

        with tempfile.TemporaryDirectory() as tmp:
            os.environ['MAS_TELEMETRY_DIR'] = tmp
            try:
                net_b, args_b, dataset_b = build(hp, seed=3)
                self.assertIsNotNone(net_b.tel)
                run_steps(net_b, args_b, dataset_b, 15, seed=3)
                net_b.tel.close()
                out = os.path.join(tmp, 'toy', 'GCN', 'clsincre', 'tfmas_star_lth1000000000.0_sth1000000000.0', 'seed0')
                self.assertTrue(os.path.exists(os.path.join(out, 'steps.csv')))
                self.assertTrue(os.path.exists(os.path.join(out, 'events.csv')))
            finally:
                del os.environ['MAS_TELEMETRY_DIR']

        for pa, pb in zip(net_a.net.parameters(), net_b.net.parameters()):
            self.assertTrue(torch.equal(pa, pb))
        self.assertEqual(net_a.n_consolidations, net_b.n_consolidations)


if __name__ == '__main__':
    unittest.main()
