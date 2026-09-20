"""Controlled task-free MAS geometry and averaging experiments."""

import copy
import torch

from .tfmas_model import NET as MASNET


class _BlendEval(torch.nn.Module):
    def __init__(self, working, average, working_weight):
        super().__init__()
        self.working = working
        self.average = average
        self.working_weight = working_weight

    def forward(self, graph, features):
        working_logits, _ = self.working(graph, features)
        average_logits, _ = self.average(graph, features)
        logits = self.working_weight * working_logits + (1 - self.working_weight) * average_logits
        return logits, []


class NET(MASNET):
    """Preserve MAS training and project classifier rows after each update."""

    def __init__(self, model, args, dataset=None):
        super().__init__(model, args)
        hp = getattr(args, 'mas_geometry_args', {})
        self.project_classifier = bool(hp.get('project_classifier', True))
        self.ema_alpha = hp.get('ema_alpha')
        self.eval_working_weight = hp.get('eval_working_weight')
        if self.ema_alpha is not None:
            self.ema_alpha = float(self.ema_alpha)
            if not 0 < self.ema_alpha < 1:
                raise ValueError('MAS ema_alpha must be in (0, 1)')
            self.ema = copy.deepcopy(self.net).requires_grad_(False)
            self._ema_initialized = False
        else:
            self.ema = None
        if self.eval_working_weight is not None:
            self.eval_working_weight = float(self.eval_working_weight)
            if self.ema is None or not 0 <= self.eval_working_weight <= 1:
                raise ValueError('MAS eval_working_weight requires EMA and must be in [0, 1]')
            self._eval_model = _BlendEval(self.net, self.ema, self.eval_working_weight)
        else:
            self._eval_model = self.ema if self.ema is not None else self.net
        weight = self.net.gat_layers[-1].linear.weight
        self._classifier_target_norm = float(weight.detach().norm(dim=1).mean())
        self.optimizer_steps = 0

    @torch.no_grad()
    def _project_classifier(self):
        weight = self.net.gat_layers[-1].linear.weight
        norms = weight.norm(dim=1, keepdim=True).clamp_min(1e-12)
        weight.mul_(self._classifier_target_norm / norms)

    def observe_cis(self, args, g, features, labels, train_ids):
        super().observe_cis(args, g, features, labels, train_ids)
        if self.project_classifier:
            self._project_classifier()
        if self.ema is not None:
            with torch.no_grad():
                if not self._ema_initialized:
                    for target, source in zip(self.ema.parameters(), self.net.parameters()):
                        target.copy_(source)
                    self._ema_initialized = True
                else:
                    for target, source in zip(self.ema.parameters(), self.net.parameters()):
                        target.mul_(self.ema_alpha).add_(source, alpha=1 - self.ema_alpha)
        self.optimizer_steps += 1

    def eval_model(self):
        return self._eval_model

    def memory_accounting(self):
        anchor_count = sum(value.numel() for value in self.star_variables)
        omega_count = sum(value.numel() for value in self.omegas)
        element_bytes = next(self.net.parameters()).element_size()
        ema_count = 0 if self.ema is None else sum(value.numel() for value in self.ema.parameters())
        return {
            'buffer_bytes': 0,
            'node_label_bytes': 0,
            'extra_param_count': anchor_count + omega_count + ema_count,
            'extra_param_bytes': (anchor_count + omega_count + ema_count) * element_bytes,
            'class_counter_bytes': 0,
            'class_recency_bytes': 0,
        }

    def compute_accounting(self):
        return {
            'optimizer_steps': self.optimizer_steps,
            'replay_seed_rows': 0,
            'replay_source_nodes': 0,
            'replay_edges': 0,
            'importance_updates': self.count_updates,
        }
