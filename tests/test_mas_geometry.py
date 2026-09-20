import unittest

import torch

from Backbones.gnns import GCN
from Baselines.mas_geometry_model import NET
from tests.test_new_baselines import make_args, make_dataset, pipeline_graph


def build(**geometry):
    torch.manual_seed(0)
    args = make_args(
        mas_args={'memory_strength': 0.5},
        mas_geometry_args=geometry,
    )
    dataset = make_dataset()
    return NET(GCN(args), args, dataset=dataset), args, dataset


class TestMASGeometry(unittest.TestCase):
    def test_first_ema_update_is_exact(self):
        net, args, dataset = build(project_classifier=False, ema_alpha=0.99966)
        graph = pipeline_graph(dataset)
        labels = graph.ndata['label'].squeeze()
        ids = torch.arange(args.batch_size)
        net.observe_cis(args, graph, graph.ndata['feat'], labels, ids)
        for working, average in zip(net.net.parameters(), net.ema.parameters()):
            self.assertTrue(torch.equal(working, average))
        self.assertEqual(net.optimizer_steps, 1)

    def test_projection_equalizes_classifier_rows(self):
        net, args, dataset = build(project_classifier=True, ema_alpha=None)
        graph = pipeline_graph(dataset)
        labels = graph.ndata['label'].squeeze()
        ids = torch.arange(args.batch_size)
        net.observe_cis(args, graph, graph.ndata['feat'], labels, ids)
        norms = net.net.gat_layers[-1].linear.weight.norm(dim=1)
        self.assertTrue(torch.allclose(norms, norms[0].expand_as(norms), atol=1e-6))
        self.assertEqual(net.memory_accounting()['buffer_bytes'], 0)


if __name__ == '__main__':
    unittest.main()
