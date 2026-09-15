"""
SGC feature propagation + MLP classifier, the backbone PDGNN runs on.

Ported from OCGL (github.com/giovannidonghi/OCGL): Backbones/gnns.py `SGC` (parameter-free propagation
(D^-1/2 A D^-1/2)^k X over sampled blocks) and Backbones/feedforward_models.py `SGC_MLP` (Linear layers with ReLU,
no dropout). OCGL defaults: SGC_args = {'h_dims': [256], 'k': 2}.

Exposes DRIFT's backbone interface: forward(g, feat) for full-graph evaluation, forward_batch(blocks, feat) for
training, both returning (logits, []), plus propagate()/classify() so PDGNN can store and replay the propagated
(topology-aware) embeddings. `gat_layers` has k parameter-free entries so DRIFT's samplers get the right depth.
"""
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F


class SGC_MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.k = int(args.SGC_args['k'])
        self.gat_layers = nn.ModuleList([nn.Identity() for _ in range(self.k)])   # sampler depth only
        h_dims = list(args.SGC_args['h_dims'])
        dims = [args.d_data] + h_dims + [args.n_cls]
        self.mlp_layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

    @staticmethod
    def _propagate_once(g, feat, src_deg, dst_deg):
        with g.local_scope():
            g.srcdata['h'] = feat * src_deg.float().clamp(min=1).pow(-0.5).to(feat.device).unsqueeze(1)
            g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return g.dstdata['h'] * dst_deg.float().clamp(min=1).pow(-0.5).to(feat.device).unsqueeze(1)

    @staticmethod
    def annotate_degrees(g):
        """Store the graph's full degrees on its nodes, so sampled blocks normalise with them (see propagate_blocks)."""
        if 'sgc_out_deg' not in g.ndata:
            g.ndata['sgc_out_deg'] = g.out_degrees().float()
            g.ndata['sgc_in_deg'] = g.in_degrees().float()

    def propagate_blocks(self, blocks, feat):
        """
        Departure from OCGL: OCGL normalises with each *block's* degrees (sampled edges only) and also predicts on
        blocks. DRIFT evaluates on the full graph, so when the parent graph carries full degrees (annotate_degrees)
        they are used here; with a full-neighbourhood sampler this equals propagate_graph exactly.
        """
        if len(blocks) != self.k:
            raise ValueError(f'SGC with k={self.k} needs a {self.k}-layer sampler, got {len(blocks)} blocks')
        for block in blocks:
            if 'sgc_out_deg' in block.srcdata:
                src_deg, dst_deg = block.srcdata['sgc_out_deg'], block.dstdata['sgc_in_deg']
            else:
                src_deg, dst_deg = block.out_degrees(), block.in_degrees()
            feat = self._propagate_once(block, feat, src_deg, dst_deg)
        return feat

    def propagate_graph(self, g, feat):
        for _ in range(self.k):
            feat = self._propagate_once(g, feat, g.out_degrees(), g.in_degrees())
        return feat

    def classify(self, x):
        for layer in self.mlp_layers[:-1]:
            x = F.relu(layer(x))
        return self.mlp_layers[-1](x)

    def forward(self, g, features):
        return self.classify(self.propagate_graph(g, features)), []

    def forward_batch(self, blocks, features):
        return self.classify(self.propagate_blocks(blocks, features)), []
