"""Budget-matched replay with an experimentally controlled graph-context arm.

Both arms store the same 100 node IDs and labels, draw the same ten replay
seeds, perform one replay forward and one Adam update.  ``replay=context``
samples neighborhoods from the same merged transductive graph used for stream
updates; ``replay=isolated`` is the repository's self-loop replay control.
"""
import copy

import dgl
import torch
import torch.nn.functional as F

from .replay_base import ReplayNET


def _cosine_full(model, graph, features, scale):
    if len(model.gat_layers) != 2:
        raise ValueError('cosine classifier currently requires the fixed two-layer GCN')
    hidden, _ = model.gat_layers[0](graph, features)
    hidden = F.relu(hidden)
    local = graph.local_var().to(hidden.device)
    local.ndata['_cos_h'] = hidden
    local.update_all(dgl.function.copy_u('_cos_h', '_m'), dgl.function.sum('_m', '_sum'))
    aggregated = local.ndata['_sum']
    weight = model.gat_layers[-1].linear.weight
    model.second_last_h = aggregated
    return scale * F.linear(F.normalize(aggregated, dim=1), F.normalize(weight, dim=1))


def _cosine_batch(model, blocks, scale):
    if len(model.gat_layers) != 2:
        raise ValueError('cosine classifier currently requires the fixed two-layer GCN')
    hidden, _ = model.gat_layers[0].forward_batch(blocks[0], blocks[0].srcdata['feat'])
    hidden = F.relu(hidden)
    final_block = blocks[-1].local_var().to(hidden.device)
    final_block.srcdata['_cos_h'] = hidden
    final_block.update_all(dgl.function.copy_u('_cos_h', '_m'), dgl.function.sum('_m', '_sum'))
    aggregated = final_block.dstdata['_sum']
    weight = model.gat_layers[-1].linear.weight
    model.second_last_h = aggregated
    return scale * F.linear(F.normalize(aggregated, dim=1), F.normalize(weight, dim=1))


class _CosineEval(torch.nn.Module):
    def __init__(self, model, scale):
        super().__init__()
        self.model = model
        self.scale = scale

    @property
    def second_last_h(self):
        return self.model.second_last_h

    def forward(self, graph, features):
        return _cosine_full(self.model, graph, features, self.scale), []


class NET(ReplayNET):
    ARGS_ATTR = 'context_ema_args'
    DEFAULTS = {
        'budget': 100,
        'memory_proportion': 1,
        'buffer': 'reservoir',
        'replay': 'context',
        'ema_alpha': 0.995,
        'ema_mode': 'constant',
        'ema_fast': 0.99,
        'classifier_norm': 'none',
        'classifier_norm_scale': 1.0,
        'replay_weight': 1.0,
        'replay_neighbors': 'sampled',
        'classifier': 'linear',
        'cosine_scale': 10.0,
        'ace': False,
        'balanced_softmax': False,
        'smoothness_weight': 0.0,
        'head_lr_multiplier': 1.0,
        'sync_new_class_rows': False,
        'project_ema_classifier': False,
    }
    _ALLOWED = frozenset(DEFAULTS)

    def __init__(self, model, args, dataset=None):
        supplied = getattr(args, self.ARGS_ATTR, {}) or {}
        unknown = set(supplied) - self._ALLOWED
        if unknown:
            raise ValueError('unsupported context_ema_args: ' + ', '.join(sorted(unknown)))
        super().__init__(model, args, dataset)
        if int(self.hp['budget']) != 100:
            raise ValueError('DRIFT protocol fixes the buffer at 100 nodes')
        if float(self.hp['memory_proportion']) != 1:
            raise ValueError('DRIFT protocol fixes replay at one row per incoming row')
        if self.hp['replay'] not in ('isolated', 'context'):
            raise ValueError("replay must be 'isolated' or 'context'")
        self.ema_alpha = float(self.hp['ema_alpha'])
        self.ema_fast = float(self.hp['ema_fast'])
        if not 0 < self.ema_alpha <= 1:
            raise ValueError('ema_alpha must be in (0, 1]')
        if not 0 < self.ema_fast <= self.ema_alpha:
            raise ValueError('ema_fast must be in (0, ema_alpha]')
        if self.hp['ema_mode'] not in ('constant', 'adaptive'):
            raise ValueError("ema_mode must be 'constant' or 'adaptive'")
        if self.hp['classifier_norm'] is True:
            self.hp['classifier_norm'] = 'fixed'
        if self.hp['classifier_norm'] is False:
            self.hp['classifier_norm'] = 'none'
        if self.hp['classifier_norm'] not in ('none', 'fixed', 'mean'):
            raise ValueError("classifier_norm must be 'none', 'fixed', or 'mean'")
        self.classifier_norm_scale = float(self.hp['classifier_norm_scale'])
        if self.classifier_norm_scale <= 0:
            raise ValueError('classifier_norm_scale must be positive')
        self.replay_weight = float(self.hp['replay_weight'])
        if self.replay_weight < 0:
            raise ValueError('replay_weight must be non-negative')
        self.smoothness_weight = float(self.hp['smoothness_weight'])
        if self.smoothness_weight < 0:
            raise ValueError('smoothness_weight must be non-negative')
        self.head_lr_multiplier = float(self.hp['head_lr_multiplier'])
        if self.head_lr_multiplier <= 0:
            raise ValueError('head_lr_multiplier must be positive')
        self.sync_new_class_rows = float(self.hp['sync_new_class_rows'])
        if not 0 <= self.sync_new_class_rows <= 1:
            raise ValueError('sync_new_class_rows must be in [0, 1]')
        if self.hp['replay_neighbors'] not in ('sampled', 'full'):
            raise ValueError("replay_neighbors must be 'sampled' or 'full'")
        if self.hp['classifier'] not in ('linear', 'cosine'):
            raise ValueError("classifier must be 'linear' or 'cosine'")
        self.cosine_scale = float(self.hp['cosine_scale'])
        if self.cosine_scale <= 0:
            raise ValueError('cosine_scale must be positive')

        if self.head_lr_multiplier != 1:
            head_parameters = list(self.net.gat_layers[-1].parameters())
            head_ids = {id(parameter) for parameter in head_parameters}
            encoder_parameters = [parameter for parameter in self.net.parameters() if id(parameter) not in head_ids]
            self.opt = torch.optim.Adam([
                {'params': encoder_parameters, 'lr': args.lr},
                {'params': head_parameters, 'lr': args.lr * self.head_lr_multiplier},
            ], lr=args.lr, weight_decay=args.weight_decay)

        self.ema = copy.deepcopy(self.net).requires_grad_(False)
        self._ema_initialized = False
        self._training_graph = None
        self._orig_to_local = None
        self.optimizer_steps = 0
        self.replay_rows_per_update = []
        self.replay_source_nodes = 0
        self.replay_edges = 0
        self._previous_labels = None
        self.ema_alphas = []
        self.class_observation_counts = {}
        self.class_last_seen = {}
        self._ema_seen_classes = set()
        final_weight = self.net.gat_layers[-1].linear.weight
        self._classifier_target_norm = float(final_weight.detach().norm(dim=1).mean())
        self._ema_eval = _CosineEval(self.ema, self.cosine_scale) if self.hp['classifier'] == 'cosine' else self.ema
        self._working_eval = _CosineEval(self.net, self.cosine_scale) if self.hp['classifier'] == 'cosine' else self.net

    def eval_model(self):
        return self._ema_eval

    def alt_eval_models(self):
        return {'working': self._working_eval}

    def _model_logits(self, model, blocks):
        if self.hp['classifier'] == 'cosine':
            return _cosine_batch(model, blocks, self.cosine_scale)
        return self._logits(model, blocks)

    def memory_accounting(self):
        params = sum(p.numel() for p in self.ema.parameters())
        param_bytes = sum(p.numel() * p.element_size() for p in self.ema.parameters())
        return {
            'buffer_bytes': self.buffer.nbytes(),
            'node_label_bytes': self.buffer.nbytes(),
            'extra_param_count': params,
            'extra_param_bytes': param_bytes,
            'class_counter_bytes': 8 * len(self.class_observation_counts),
            'class_recency_bytes': 8 * len(self.class_last_seen),
        }

    def compute_accounting(self):
        return {
            'optimizer_steps': self.optimizer_steps,
            'replay_seed_rows': sum(self.replay_rows_per_update),
            'replay_source_nodes': self.replay_source_nodes,
            'replay_edges': self.replay_edges,
        }

    @torch.no_grad()
    def _update_ema(self, current_labels):
        current = set(int(value) for value in current_labels.detach().cpu().unique().tolist())
        new_classes = current - self._ema_seen_classes
        if not self._ema_initialized:
            for target, source in zip(self.ema.parameters(), self.net.parameters()):
                target.copy_(source)
            self._ema_initialized = True
            self._previous_labels = current
            self._ema_seen_classes.update(current)
            self.ema_alphas.append(0.0)
            return
        alpha = self.ema_alpha
        if self.hp['ema_mode'] == 'adaptive':
            overlap = len(current.intersection(self._previous_labels)) / max(len(current), 1)
            alpha = self.ema_fast + (self.ema_alpha - self.ema_fast) * overlap
        for target, source in zip(self.ema.parameters(), self.net.parameters()):
            target.mul_(alpha).add_(source, alpha=1 - alpha)
        if self.sync_new_class_rows and new_classes:
            ema_head = self.ema.gat_layers[-1].linear
            working_head = self.net.gat_layers[-1].linear
            rows = torch.tensor(sorted(new_classes), dtype=torch.long, device=working_head.weight.device)
            ema_rows = ema_head.weight.index_select(0, rows)
            working_rows = working_head.weight.index_select(0, rows)
            ema_head.weight.index_copy_(0, rows, torch.lerp(ema_rows, working_rows, self.sync_new_class_rows))
            if working_head.bias is not None:
                ema_bias = ema_head.bias.index_select(0, rows)
                working_bias = working_head.bias.index_select(0, rows)
                ema_head.bias.index_copy_(0, rows, torch.lerp(ema_bias, working_bias, self.sync_new_class_rows))
        self._project_ema_classifier()
        self._ema_seen_classes.update(current)
        self._previous_labels = current
        self.ema_alphas.append(alpha)

    def _remember_training_graph(self, graph):
        if self._training_graph is not None:
            if graph is not self._training_graph:
                raise ValueError('context replay requires one stable merged training graph')
            return
        if '_ID' not in graph.ndata:
            raise ValueError('merged graph must retain original node IDs')
        original = graph.ndata['_ID'].detach().cpu().tolist()
        self._orig_to_local = {int(nid): i for i, nid in enumerate(original)}
        self._training_graph = graph

    def _context_logits(self, args, original_ids, device):
        local = torch.tensor(
            [self._orig_to_local[int(nid)] for nid in original_ids],
            dtype=torch.long,
            device=device,
        )
        if self.hp['replay_neighbors'] == 'full':
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(len(self.net.gat_layers))
            _, _, blocks = sampler.sample_blocks(self._training_graph, local)
        else:
            blocks = self._stream_blocks(args, self._training_graph, local)
        self.replay_source_nodes += sum(int(block.num_src_nodes()) for block in blocks)
        self.replay_edges += sum(int(block.num_edges()) for block in blocks)
        return self._model_logits(self.net, blocks)

    def _replay_logits(self, args, replay_ids, device):
        if self.hp['replay'] == 'context':
            return self._context_logits(args, replay_ids, device)
        blocks, index = self._isolated_blocks(replay_ids, device)
        logits = self._model_logits(self.net, blocks)[index]
        self.replay_source_nodes += len(replay_ids) * len(self.net.gat_layers)
        self.replay_edges += len(replay_ids) * len(self.net.gat_layers)
        return logits

    @torch.no_grad()
    def _project_classifier(self):
        if self.hp['classifier_norm'] == 'none':
            return
        weight = self.net.gat_layers[-1].linear.weight
        norms = weight.norm(dim=1, keepdim=True).clamp_min(1e-12)
        target = norms.mean() if self.hp['classifier_norm'] == 'mean' else self._classifier_target_norm * self.classifier_norm_scale
        weight.mul_(target / norms)

    @torch.no_grad()
    def _project_ema_classifier(self):
        if not self.hp['project_ema_classifier']:
            return
        weight = self.ema.gat_layers[-1].linear.weight
        norms = weight.norm(dim=1, keepdim=True).clamp_min(1e-12)
        weight.mul_(self._classifier_target_norm / norms)

    def _step(self, args, g, labels, train_ids, cis):
        self.net.train()
        device = next(self.net.parameters()).device
        self._remember_training_graph(g)
        prior_seen = set(self.seen_classes)
        head, _ = self._head_fn(labels, train_ids, cis)
        original_ids = self._orig_ids(g, train_ids)
        incoming_labels = labels[train_ids].to(device)
        for value in incoming_labels.detach().cpu().tolist():
            value = int(value)
            self.class_observation_counts[value] = self.class_observation_counts.get(value, 0) + 1
            self.class_last_seen[value] = self.optimizer_steps + 1

        self.opt.zero_grad()
        stream_blocks = self._stream_blocks(args, g, train_ids)
        incoming_logits = head(self._model_logits(self.net, stream_blocks))
        incoming_hidden = self.net.second_last_h
        if self.hp['ace']:
            current = set(int(value) for value in incoming_labels.detach().cpu().unique().tolist())
            absent_prior = [value for value in prior_seen - current if value < incoming_logits.shape[1]]
            if absent_prior:
                incoming_logits = incoming_logits.clone()
                incoming_logits[:, torch.tensor(absent_prior, dtype=torch.long, device=device)] = -torch.inf
        if self.hp['balanced_softmax']:
            offsets = torch.zeros(incoming_logits.shape[1], dtype=incoming_logits.dtype, device=device)
            for class_id, count in self.class_observation_counts.items():
                if class_id < offsets.numel():
                    offsets[class_id] = torch.log(torch.tensor(float(count), device=device))
            incoming_logits = incoming_logits + offsets
        loss = F.cross_entropy(incoming_logits, incoming_labels)
        if self.smoothness_weight:
            source, destination = stream_blocks[-1].edges()
            if source.numel():
                source_hidden = F.normalize(incoming_hidden[source], dim=1)
                destination_hidden = F.normalize(incoming_hidden[destination], dim=1)
                smoothness = (1 - (source_hidden * destination_hidden).sum(dim=1)).mean()
                loss = loss + self.smoothness_weight * smoothness

        slots = self._replay_draw()
        self.replay_rows_per_update.append(len(slots))
        if slots:
            replay_ids = [self.buffer.ids[i] for i in slots]
            replay_labels = torch.tensor(
                [self.buffer.labels[i] for i in slots], dtype=torch.long, device=device,
            )
            replay_logits = self._replay_logits(args, replay_ids, device)
            loss = loss + self.replay_weight * F.cross_entropy(head(replay_logits), replay_labels)

        loss.backward()
        self.opt.step()
        self._project_classifier()
        self.optimizer_steps += 1
        for node_id, label in zip(original_ids, incoming_labels.detach().cpu().tolist()):
            if self.hp['buffer'] == 'degree_cbrs':
                local_id = self._orig_to_local[int(node_id)]
                degree = float(self._training_graph.in_degrees(local_id))
                self.buffer.add(int(node_id), int(label), score=degree)
            else:
                self.buffer.add(int(node_id), int(label))
        self._update_ema(incoming_labels)
