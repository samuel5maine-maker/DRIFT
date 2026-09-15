"""
PDGNN — Parameter-Decoupled GNN with Topology-aware Embedding Memory (baselines spec §5.6).

Zhang, Song, Chen, Tao, "Topology-aware Embedding Memory for Continual Learning on Expanding Networks", KDD 2024.
Ported from OCGL's online adaptation (github.com/giovannidonghi/OCGL, Baselines/pdgnn.py -> er.py -> base_learner.py):
PDGNN is ER on an SGC backbone (asserted there as args.backbone == 'SGC'): the parameter-free SGC propagation
produces a topology-aware embedding (TEM) per node; the memory stores these embeddings, and replay feeds them
straight to the MLP classifier, so replayed nodes keep their neighbourhood information without storing subgraphs.

Adaptations to DRIFT, all in NOTES.md:
  - backbone: SGC(k=2) + MLP(256) instead of DRIFT's 2-layer GCN — the spec's fixed-backbone rule cannot hold for
    PDGNN by construction; OCGL uses the same backbone for PDGNN;
  - embeddings are computed on DRIFT's sampled training blocks (NeighborSampler [10, 25]) at insertion time; DRIFT
    evaluates on the full graph, so a stored embedding is a sampled estimate of the full-neighbourhood embedding;
  - memory population: reservoir, as in OCGL's online PDGNN. The KDD paper's coverage-maximisation sampling is not
    part of OCGL's online port and is not implemented;
  - replay ratio 1:1 and loss CE(stream) + CE(replay), as DRIFT's ER (OCGL's memory_proportion grid {1,2,3}: 1 here).
"""
import torch
import torch.nn.functional as F

from .replay_base import ReplayNET


class NET(ReplayNET):
    ARGS_ATTR = 'pdgnn_args'
    DEFAULTS = {'budget': 100, 'memory_proportion': 1, 'buffer': 'reservoir'}

    def __init__(self, model, args, dataset=None):
        if args.backbone != 'SGC':
            raise ValueError('PDGNN runs on the SGC backbone (as in OCGL); use --backbone SGC')
        super().__init__(model, args, dataset)

    def _step(self, args, g, labels, train_ids, cis):
        self.net.train()
        device = next(self.net.parameters()).device
        head, _ = self._head_fn(labels, train_ids, cis)
        new_orig = self._orig_ids(g, train_ids)
        y_new = labels[train_ids]

        self.net.annotate_degrees(g)
        blocks = self._stream_blocks(args, g, train_ids)
        with torch.no_grad():                                   # SGC propagation has no parameters
            emb = self.net.propagate_blocks(blocks, blocks[0].srcdata['feat'])
        self.net.zero_grad()
        loss = F.cross_entropy(head(self.net.classify(emb)), y_new)
        if len(self.buffer) > 0:
            slots = self._replay_draw()
            x = torch.stack([self.buffer.payload['emb'][i] for i in slots]).to(device)
            y = torch.tensor([self.buffer.labels[i] for i in slots], dtype=torch.long, device=device)
            loss = loss + F.cross_entropy(head(self.net.classify(x)), y)
        loss.backward()
        self.opt.step()

        for nid, yy, e in zip(new_orig, y_new.tolist(), emb.cpu()):
            self.buffer.add(nid, int(yy), emb=e.clone())
