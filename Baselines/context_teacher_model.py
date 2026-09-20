"""Contextual replay with reliable EMA-teacher feedback (H30)."""

import dgl
import torch
import torch.nn.functional as F

from . import context_ema_model


class NET(context_ema_model.NET):
    ARGS_ATTR = 'context_ema_args'
    DEFAULTS = dict(context_ema_model.NET.DEFAULTS, feedback_enabled=True)
    _ALLOWED = frozenset(DEFAULTS)
    TEMPERATURE = 2.0

    def __init__(self, model, args, dataset=None):
        super().__init__(model, args, dataset)
        self.feedback_enabled = bool(self.hp['feedback_enabled'])
        self.teacher_gate_rows = 0
        self.teacher_replay_rows = 0
        self.teacher_kl_sum = 0.0
        self.teacher_forward_calls = 0
        self.teacher_forward_source_nodes = 0
        self.teacher_forward_edges = 0
        self._last_replay_blocks = None
        self._last_replay_index = None

    def eval_model(self):
        """H30 uses the working network for primary inference."""
        return self._working_eval

    def alt_eval_models(self):
        return {'ema': self._ema_eval}

    def compute_accounting(self):
        accounting = super().compute_accounting()
        accounting.update({
            'teacher_gate_rows': self.teacher_gate_rows,
            'teacher_replay_rows': self.teacher_replay_rows,
            'teacher_mean_kl': self.teacher_kl_sum / max(self.teacher_replay_rows, 1),
            'teacher_forward_calls': self.teacher_forward_calls,
            'teacher_forward_source_nodes': self.teacher_forward_source_nodes,
            'teacher_forward_edges': self.teacher_forward_edges,
        })
        return accounting

    def _replay_logits(self, args, replay_ids, device):
        """Run the student and retain its exact sampled blocks for the teacher."""
        if self.hp['replay'] == 'context':
            local = torch.tensor(
                [self._orig_to_local[int(nid)] for nid in replay_ids],
                dtype=torch.long, device=device,
            )
            if self.hp['replay_neighbors'] == 'full':
                sampler = dgl.dataloading.MultiLayerFullNeighborSampler(len(self.net.gat_layers))
                _, _, blocks = sampler.sample_blocks(self._training_graph, local)
            else:
                blocks = self._stream_blocks(args, self._training_graph, local)
            index = None
        else:
            blocks, index = self._isolated_blocks(replay_ids, device)
        self._last_replay_blocks = blocks
        self._last_replay_index = index
        self.replay_source_nodes += sum(int(block.num_src_nodes()) for block in blocks)
        self.replay_edges += sum(int(block.num_edges()) for block in blocks)
        logits = self._model_logits(self.net, blocks)
        return logits if index is None else logits[index]

    @torch.no_grad()
    def _teacher_logits(self):
        blocks = self._last_replay_blocks
        if blocks is None:
            raise RuntimeError('teacher forward requires a preceding replay forward')
        was_training = self.ema.training
        self.ema.eval()
        try:
            logits = self._model_logits(self.ema, blocks)
        finally:
            self.ema.train(was_training)
        index = self._last_replay_index
        self.teacher_forward_calls += 1
        self.teacher_forward_source_nodes += sum(int(block.num_src_nodes()) for block in blocks)
        self.teacher_forward_edges += sum(int(block.num_edges()) for block in blocks)
        return logits if index is None else logits[index]

    def _feedback_loss(self, head, student_logits, teacher_logits, replay_labels):
        student = head(student_logits)
        teacher = head(teacher_logits).detach()
        teacher_prob = torch.softmax(teacher, dim=1)
        student_prob = torch.softmax(student.detach(), dim=1)
        row = torch.arange(replay_labels.numel(), device=replay_labels.device)
        gate = ((teacher.argmax(dim=1) == replay_labels) &
                (teacher_prob[row, replay_labels] > student_prob[row, replay_labels])).detach()
        temperature = self.TEMPERATURE
        per_row = F.kl_div(
            F.log_softmax(student / temperature, dim=1),
            F.softmax(teacher / temperature, dim=1),
            reduction='none',
        ).sum(dim=1) * temperature ** 2
        gated = (per_row * gate.to(per_row.dtype)).sum() / replay_labels.numel()
        self.teacher_gate_rows += int(gate.sum())
        self.teacher_replay_rows += int(gate.numel())
        self.teacher_kl_sum += float(gated.detach()) * replay_labels.numel()
        return gated

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
                loss = loss + self.smoothness_weight * (
                    1 - (source_hidden * destination_hidden).sum(dim=1)
                ).mean()

        slots = self._replay_draw()
        self.replay_rows_per_update.append(len(slots))
        if slots:
            replay_ids = [self.buffer.ids[i] for i in slots]
            replay_labels = torch.tensor(
                [self.buffer.labels[i] for i in slots], dtype=torch.long, device=device,
            )
            replay_logits = self._replay_logits(args, replay_ids, device)
            loss = loss + self.replay_weight * F.cross_entropy(head(replay_logits), replay_labels)
            if self.feedback_enabled:
                teacher_logits = self._teacher_logits()
                loss = loss + self._feedback_loss(head, replay_logits, teacher_logits, replay_labels)

        loss.backward()
        self._last_replay_blocks = None
        self._last_replay_index = None
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
