"""Instantaneous prior-shift correction on the H30 control learner (H31).

DRIFT's Gaussian regime moves the training label prior smoothly through the
class set while evaluation pools every test node under one fixed prior.  Plain
cross-entropy therefore fits the posterior of a distribution that is not the
one being scored.  This learner adds the Laplace-smoothed log prior of the rows
in the current update to the logits during training and predicts with the
unadjusted logits, so the network fits the prior-invariant conditional.

The adjustment is rebuilt each update, stored nowhere, and reads only labels the
learner is already training on.  With ``prior_shift_enabled=False`` the loss is
constructed by the same operations as the H30 control, in the same random-state
order, so the disabled arm reproduces the control exactly.
"""

import torch
import torch.nn.functional as F

from . import context_teacher_model


class NET(context_teacher_model.NET):
    ARGS_ATTR = 'context_ema_args'
    DEFAULTS = dict(context_teacher_model.NET.DEFAULTS,
                    feedback_enabled=False, prior_shift_enabled=True)
    _ALLOWED = frozenset(DEFAULTS)
    SMOOTHING = 1.0

    def __init__(self, model, args, dataset=None):
        super().__init__(model, args, dataset)
        if self.feedback_enabled:
            raise ValueError('H31 holds the H30 teacher feedback off; its base is the H30 control')
        self.prior_shift_enabled = bool(self.hp['prior_shift_enabled'])
        self.adjustment_sum = 0.0
        self.adjustment_max = 0.0
        self.adjusted_updates = 0
        self.distinct_class_sum = 0
        self.positive_rows = 0
        self.negative_rows = 0

    def compute_accounting(self):
        accounting = super().compute_accounting()
        updates = max(self.adjusted_updates, 1)
        accounting.update({
            'prior_shift_updates': self.adjusted_updates,
            'prior_shift_mean_adjustment': self.adjustment_sum / updates,
            'prior_shift_max_adjustment': self.adjustment_max,
            'prior_shift_mean_distinct_classes': self.distinct_class_sum / updates,
            'prior_shift_negative_to_positive_ratio':
                self.negative_rows / max(self.positive_rows, 1),
        })
        return accounting

    def _prior_adjustment(self, width, label_groups, device):
        """Laplace-smoothed log prior of this update's rows, up to a dropped constant."""
        counts = torch.zeros(width, dtype=torch.float32, device=device)
        for group in label_groups:
            counts.index_add_(0, group, torch.ones_like(group, dtype=counts.dtype))
        adjustment = torch.log(counts + self.SMOOTHING)
        present = int((counts > 0).sum())
        rows = int(counts.sum())
        self.adjusted_updates += 1
        self.adjustment_sum += float(adjustment.max())
        self.adjustment_max = max(self.adjustment_max, float(adjustment.max()))
        self.distinct_class_sum += present
        self.positive_rows += rows
        self.negative_rows += rows * max(width - 1, 0)
        return adjustment

    def _step(self, args, g, labels, train_ids, cis):
        self.net.train()
        device = next(self.net.parameters()).device
        self._remember_training_graph(g)
        head, _ = self._head_fn(labels, train_ids, cis)
        original_ids = self._orig_ids(g, train_ids)
        incoming_labels = labels[train_ids].to(device)
        for value in incoming_labels.detach().cpu().tolist():
            value = int(value)
            self.class_observation_counts[value] = self.class_observation_counts.get(value, 0) + 1
            self.class_last_seen[value] = self.optimizer_steps + 1

        self.opt.zero_grad()
        # Sampling order below is the H30 control's, so the disabled arm keeps its random state.
        stream_blocks = self._stream_blocks(args, g, train_ids)
        incoming_logits = head(self._model_logits(self.net, stream_blocks))

        slots = self._replay_draw()
        self.replay_rows_per_update.append(len(slots))
        replay_logits = replay_labels = None
        if slots:
            replay_ids = [self.buffer.ids[i] for i in slots]
            replay_labels = torch.tensor(
                [self.buffer.labels[i] for i in slots], dtype=torch.long, device=device,
            )
            replay_logits = head(self._replay_logits(args, replay_ids, device))

        if self.prior_shift_enabled:
            groups = [incoming_labels] if replay_labels is None else [incoming_labels, replay_labels]
            adjustment = self._prior_adjustment(incoming_logits.shape[1], groups, device)
            incoming_logits = incoming_logits + adjustment
            if replay_logits is not None:
                replay_logits = replay_logits + adjustment

        loss = F.cross_entropy(incoming_logits, incoming_labels)
        if replay_logits is not None:
            loss = loss + self.replay_weight * F.cross_entropy(replay_logits, replay_labels)

        loss.backward()
        self._last_replay_blocks = None
        self._last_replay_index = None
        self.opt.step()
        self._project_classifier()
        self.optimizer_steps += 1
        for node_id, label in zip(original_ids, incoming_labels.detach().cpu().tolist()):
            self.buffer.add(int(node_id), int(label))
        self._update_ema(incoming_labels)
