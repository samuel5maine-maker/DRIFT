"""
Common machinery for the replay baselines added to DRIFT (ER-CBRS, DER, DER++, CLS-ER, combination).

It follows DRIFT's existing replay conventions (Baselines/er_model.py) so that only the method-specific parts differ:
  - stream batch: sampled neighbourhood blocks on the pipeline graph, CE over the class-IL head [:offset2];
  - replay: buffered nodes are forwarded as *isolated* nodes with self-loops, taken from dataset.graph;
  - 1:1 replay ratio: batch_size replayed nodes per batch_size incoming nodes (DRIFT App. A.4);
  - Adam with args.lr and args.weight_decay, one pass per batch (--epochs 1).
"""
import dgl
import torch
import torch.nn.functional as F

from .replay_buffers import make_buffer


class ReplayNET(torch.nn.Module):
    ARGS_ATTR = None     # name of the args dict holding method hyperparameters, e.g. 'er_cbrs_args'
    DEFAULTS = {}

    def __init__(self, model, args, dataset=None):
        super().__init__()
        if dataset is None:
            raise ValueError(f'{type(self).__module__} needs dataset= (list the method in pipeline.NEEDS_DATASET)')
        if args.epochs != 1:
            raise ValueError('DRIFT protocol: one pass per batch; run with --epochs 1')
        self.hp = dict(self.DEFAULTS)
        self.hp.update(getattr(args, self.ARGS_ATTR, {}) or {})
        self.net = model
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.full_graph = dataset.graph
        self.seen_classes = set()
        self.replay_size = int(args.batch_size * self.hp.get('memory_proportion', 1))
        self.buffer = make_buffer(str(self.hp.get('buffer', 'reservoir')), int(self.hp.get('budget', 100)))

    def forward(self, features):
        return self.net(features)

    # ---- interface used by pipeline.py and telemetry.py
    def observe(self, args, g, features, labels, train_ids):
        self._step(args, g, labels, train_ids, cis=False)

    def observe_cis(self, args, g, features, labels, train_ids):
        self._step(args, g, labels, train_ids, cis=True)

    def buffer_labels(self):
        return list(self.buffer.labels)

    def memory_accounting(self):
        return {'buffer_bytes': self.buffer.nbytes(), 'extra_param_count': 0, 'extra_param_bytes': 0}

    # ---- helpers
    def _head_fn(self, labels, train_ids, cis):
        self.seen_classes.update(int(c) for c in labels[train_ids].unique().tolist())
        if not cis:
            return lambda out: out, None
        offset2 = max(self.seen_classes) + 1
        offset2 += offset2 % 2
        return (lambda out: out[:, :offset2]), offset2

    @staticmethod
    def _logits(net, blocks):
        out, _ = net.forward_batch(blocks, blocks[0].srcdata['feat'])
        return out[0] if isinstance(out, tuple) else out

    def _stream_blocks(self, args, g, train_ids):
        sampler = dgl.dataloading.NeighborSampler(args.n_nbs_sample) if args.sample_nbs else \
            dgl.dataloading.MultiLayerFullNeighborSampler(len(self.net.gat_layers))
        _, _, blocks = sampler.sample_blocks(g, train_ids)
        return blocks

    def _isolated_blocks(self, orig_ids, device):
        """Replay input for buffered nodes: no edges, self-loops only (DRIFT's er_model.py convention)."""
        # the same node can occupy two slots (streams revisit nodes), so build the graph on unique ids and map back
        unique = sorted(set(orig_ids))
        sub = dgl.node_subgraph(self.full_graph, torch.tensor(unique, dtype=torch.long), store_ids=True)
        sub = dgl.remove_edges(sub, torch.arange(sub.num_edges()))
        sub = dgl.add_self_loop(sub).to(device)
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(len(self.net.gat_layers))
        _, _, blocks = sampler.sample_blocks(sub, torch.arange(sub.num_nodes(), device=device))
        order = sub.ndata['_ID'].cpu().tolist()
        pos = {nid: i for i, nid in enumerate(order)}
        index = torch.tensor([pos[n] for n in orig_ids], dtype=torch.long, device=device)
        return blocks, index

    def _isolated_logits(self, net, orig_ids, device):
        blocks, index = self._isolated_blocks(orig_ids, device)
        return self._logits(net, blocks)[index]

    @staticmethod
    def _orig_ids(g, train_ids):
        ids = train_ids.cpu()
        return g.ndata['_ID'][ids].cpu().tolist() if '_ID' in g.ndata else ids.tolist()

    def _replay_draw(self):
        """Slot indices for one replay minibatch (uniform, without replacement)."""
        return self.buffer.sample(self.replay_size)

    def _step(self, args, g, labels, train_ids, cis):
        raise NotImplementedError


def ce(head, logits, y):
    return F.cross_entropy(head(logits), y)
