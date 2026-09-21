"""Contextual replay plus unlatched MAS consolidation in one learner (H33).

Each of the two mechanisms clears the project targets on exactly one dataset
and fails on the other, for measured and complementary reasons.  MAS keeps no
examples, so on CoraFull's 17-step class turnover nothing can recover a class
once it stops arriving.  A 100-node replay buffer has no way to hold Arxiv's
101,710 training nodes stable across 10,172 updates, while MAS's moving trust
region does.  This learner runs both at once, with identical constants on every
dataset; the stream, not a switch, decides which one carries the load.

Per update: incoming cross-entropy over the class-incremental head, plus
cross-entropy on ten replay seeds drawn from a CBRS-100 memory and embedded with
neighbourhoods sampled from the same training graph, plus the MAS penalty.  The
plateau window holds the incoming cross-entropy only (the H32 correction), and
importance is estimated on the incoming block exactly as in DRIFT's `tfmas`.
Inference uses the working network.  ``replay_enabled`` and ``mas_enabled``
switch off one mechanism each, giving the two single-mechanism control arms.
"""

import dgl
import numpy as np
import torch
import torch.nn.functional as F

from .replay_buffers import make_buffer
from .scalefree_mas_model import NET as UnlatchedMAS


class NET(UnlatchedMAS):
    BUDGET = 100

    def __init__(self, model, args, dataset=None):
        hybrid = dict(getattr(args, 'hybrid_args', {}) or {})
        unknown = set(hybrid) - {'replay_enabled', 'mas_enabled', 'buffer', 'trigger'}
        if unknown:
            raise ValueError('unsupported hybrid_args: ' + ', '.join(sorted(unknown)))
        super().__init__(model, args, dataset)
        self.replay_enabled = bool(hybrid.get('replay_enabled', True))
        self.mas_enabled = bool(hybrid.get('mas_enabled', True))
        self.trigger = str(hybrid.get('trigger', 'absolute'))
        if self.trigger not in ('absolute', 'relative'):
            raise ValueError("trigger must be 'absolute' or 'relative'")
        self.buffer = make_buffer(str(hybrid.get('buffer', 'cbrs')), self.BUDGET)
        self.replay_rows = int(args.batch_size)  # DRIFT's 1:1 replay ratio
        self.replay_rows_per_update = []
        self.replay_source_nodes = 0
        self.replay_edges = 0
        self._training_graph = None
        self._orig_to_local = None

    def memory_accounting(self):
        accounting = super().memory_accounting()
        accounting['buffer_bytes'] = self.buffer.nbytes()
        accounting['node_label_bytes'] = self.buffer.nbytes()
        return accounting

    def compute_accounting(self):
        accounting = super().compute_accounting()
        accounting.update({
            'replay_seed_rows': sum(self.replay_rows_per_update),
            'replay_source_nodes': self.replay_source_nodes,
            'replay_edges': self.replay_edges,
            'replay_enabled': self.replay_enabled,
            'mas_enabled': self.mas_enabled,
            'trigger': self.trigger,
        })
        return accounting

    def _remember_graph(self, g):
        if self._training_graph is not None:
            if g is not self._training_graph:
                raise ValueError('contextual replay requires one stable merged training graph')
            return
        if '_ID' not in g.ndata:
            raise ValueError('merged graph must retain original node IDs')
        original = g.ndata['_ID'].detach().cpu().tolist()
        self._orig_to_local = {int(nid): i for i, nid in enumerate(original)}
        self._training_graph = g

    @staticmethod
    def relative_plateau(mean, variance, past_means, past_variances):
        """Scale-free form of DRIFT's two plateau conditions (H34).

        DRIFT fires when the window mean is under 0.2 and its variance under 0.1, which
        are levels in loss units.  How often those levels are reached depends on how
        confident the network happens to be, so class-balanced replay silenced the
        detector on Arxiv (H33).  Here "low" and "flat" are judged against the stream's
        own history: the window mean and variance must each be at or below the median of
        every window so far, including this one.  No constant is involved.
        """
        return (mean <= float(np.median(past_means + [mean]))
                and variance <= float(np.median(past_variances + [variance])))

    def _consolidate(self, blocks, input_features, offset1, offset2):
        """DRIFT's importance estimate and anchor copy, unchanged."""
        self.count_updates += 1
        self.consolidation_steps.append(self.optimizer_steps)
        self.last_loss_window_mean = self.loss_window_mean
        self.last_loss_window_variance = self.loss_window_variance
        self.new_peak_detected = False
        gradients = [0 for _ in self.net.parameters()]
        self.net.zero_grad()
        output, _ = self.net.forward_batch(blocks, input_features)
        if isinstance(output, tuple):
            output = output[0]
        output = output[:, offset1:offset2]
        output.pow_(2)
        output.mean().backward()
        for index, parameter in enumerate(self.net.parameters()):
            gradients[index] += torch.abs(parameter.grad.data.clone())
        omegas_old = self.omegas[:]
        self.omegas = []
        self.star_variables = []
        for index, parameter in enumerate(self.net.parameters()):
            if len(omegas_old) != 0:
                self.omegas.append(1 / self.count_updates * gradients[index]
                                   + (1 - 1 / self.count_updates) * omegas_old[index])
            else:
                self.omegas.append(gradients[index])
            self.star_variables.append(parameter.data.clone().detach())

    def observe_cis(self, args, g, features, labels, train_ids):
        self.net.train()
        self.net.zero_grad()
        self._remember_graph(g)
        device = next(self.net.parameters()).device

        for label in labels[train_ids].unique():
            if label not in self.seen_classes:
                self.seen_classes.append(label)
            self.first_seen_steps.setdefault(int(label), self.optimizer_steps)
        offset1, offset2 = 0, int(max(self.seen_classes)) + 1
        if offset2 % 2 != 0:
            offset2 += 1

        sampler = dgl.dataloading.NeighborSampler(args.n_nbs_sample) if args.sample_nbs else \
            dgl.dataloading.MultiLayerFullNeighborSampler(len(self.net.gat_layers))
        if args.cuda:
            train_ids = train_ids.to(device='cuda:{}'.format(args.gpu))
        _, _, blocks = sampler.sample_blocks(g, train_ids)
        output_labels = labels[train_ids]
        input_features = blocks[0].srcdata['feat']
        output, _ = self.net.forward_batch(blocks, input_features)
        if isinstance(output, tuple):
            output = output[0]
        task_loss = self.ce(output[:, offset1:offset2], output_labels)
        loss = task_loss

        slots = self.buffer.sample(self.replay_rows) if self.replay_enabled else []
        self.replay_rows_per_update.append(len(slots))
        if slots:
            replay_local = torch.tensor(
                [self._orig_to_local[int(self.buffer.ids[i])] for i in slots],
                dtype=torch.long, device=device)
            replay_labels = torch.tensor([self.buffer.labels[i] for i in slots],
                                         dtype=torch.long, device=device)
            _, _, replay_blocks = sampler.sample_blocks(g, replay_local)
            self.replay_source_nodes += sum(int(b.num_src_nodes()) for b in replay_blocks)
            self.replay_edges += sum(int(b.num_edges()) for b in replay_blocks)
            replay_out, _ = self.net.forward_batch(replay_blocks, replay_blocks[0].srcdata['feat'])
            if isinstance(replay_out, tuple):
                replay_out = replay_out[0]
            loss = loss + F.cross_entropy(replay_out[:, offset1:offset2], replay_labels)

        if len(self.star_variables) != 0 and len(self.omegas) != 0:
            for index, parameter in enumerate(self.net.parameters()):
                loss = loss + self.MAS_weight / 2. * torch.sum(
                    self.omegas[index] * (parameter - self.star_variables[index]) ** 2)

        loss.backward()
        self.opt.step()

        self.task_losses.append(float(task_loss.detach()))
        self.total_losses.append(float(loss.detach()))
        self.loss_window.append(task_loss.cpu().detach().numpy())
        if len(self.loss_window) > self.loss_window_length:
            del self.loss_window[0]
        self.loss_window_mean = float(sum(self.loss_window) / len(self.loss_window))
        mean = self.loss_window_mean
        self.loss_window_variance = float(sum((v - mean) ** 2 for v in self.loss_window) / len(self.loss_window))
        self.monitored_losses.append(self.loss_window_mean)
        if not self.new_peak_detected and \
                self.loss_window_mean > self.last_loss_window_mean + self.last_loss_window_variance ** 0.5:
            self.new_peak_detected = True
        if self.trigger == 'relative':
            plateau = self.relative_plateau(self.loss_window_mean, self.loss_window_variance,
                                            self.loss_window_means, self.loss_window_variances)
        else:
            plateau = (self.loss_window_mean < 0.2 and self.loss_window_variance < 0.1)
        if self.mas_enabled and self.new_peak_detected and plateau:
            self._consolidate(blocks, input_features, offset1, offset2)

        self.loss_window_means.append(self.loss_window_mean)
        self.loss_window_variances.append(self.loss_window_variance)

        if self.replay_enabled:
            original = g.ndata['_ID'][train_ids].cpu().tolist()
            for node_id, label in zip(original, output_labels.detach().cpu().tolist()):
                self.buffer.add(int(node_id), int(label))
        self.optimizer_steps += 1
