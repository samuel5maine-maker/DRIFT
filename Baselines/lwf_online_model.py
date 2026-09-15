"""
LwF-online — Learning without Forgetting with a periodically refreshed teacher (baselines spec §5.4).

Li & Hoiem, "Learning without Forgetting", TPAMI 40(12), 2018. Task-free adaptation ported from OCGL
(github.com/giovannidonghi/OCGL, Baselines/lwf.py; Donghi et al., arXiv:2508.03283 §IV.4):
  - every `update_every` batches the teacher becomes a frozen copy of the current network (OCGL `save_every`);
  - on the incoming batch, distil the teacher into the student on the output classes *not present in the batch*:
        loss = CE(stream) + lambda_dist · MultiClassCrossEntropy(student[:, mask], teacher[:, mask], T)
    with MultiClassCrossEntropy(l, t, T) = -mean_i Σ_c softmax(t/T)_c · log_softmax(l/T)_c (OCGL utils.py);
  - no replay memory.

Adaptations: OCGL restricts outputs to the observed classes (remapped); DRIFT's class-IL head is the contiguous
range [:offset2], so the distillation mask is that range minus the batch's labels. OCGL additionally re-initialises
the LwF network with kaiming_normal_init; DRIFT's backbone initialisation is kept so all methods start identically.
Teacher and student see the same sampled blocks.
Grid (spec §5.4, from OCGL): lambda_dist ∈ {0.1, 1, 10}, T ∈ {0.2, 2, 20}, update_every ∈ {1, 10, 100}.
"""
import copy

import torch
import torch.nn.functional as F

from .replay_base import ReplayNET


def multiclass_cross_entropy(logits, target_logits, T):
    return -(torch.softmax(target_logits / T, dim=1) * torch.log_softmax(logits / T, dim=1)).sum(1).mean()


class NET(ReplayNET):
    ARGS_ATTR = 'lwf_online_args'
    DEFAULTS = {'lambda_dist': 1.0, 'T': 2.0, 'update_every': 10}

    def __init__(self, model, args, dataset=None):
        super().__init__(model, args, dataset)
        # copied before any forward pass (DRIFT backbones cache non-leaf activations, which deepcopy rejects);
        # refreshed in place with load_state_dict
        self._teacher_net = copy.deepcopy(self.net).requires_grad_(False)
        self.teacher = None
        self.batch_count = 0

    def buffer_labels(self):
        return None

    def memory_accounting(self):
        n = sum(p.numel() for p in self.net.parameters()) if self.teacher is not None else 0
        return {'buffer_bytes': 0, 'extra_param_count': n, 'extra_param_bytes': 4 * n}

    def _step(self, args, g, labels, train_ids, cis):
        self.net.train()
        head, offset2 = self._head_fn(labels, train_ids, cis)
        y_new = labels[train_ids]
        blocks = self._stream_blocks(args, g, train_ids)

        self.net.zero_grad()
        logits = head(self._logits(self.net, blocks))
        loss = F.cross_entropy(logits, y_new)
        if self.teacher is not None:
            mask = torch.ones(logits.shape[1], dtype=torch.bool, device=logits.device)
            mask[y_new.unique()] = False
            if mask.any():
                with torch.no_grad():
                    target = head(self._logits(self.teacher, blocks))
                loss = loss + float(self.hp['lambda_dist']) * multiclass_cross_entropy(
                    logits[:, mask], target[:, mask], float(self.hp['T']))
        loss.backward()
        self.opt.step()

        self.batch_count += 1
        if self.batch_count % int(self.hp['update_every']) == 0:
            self._teacher_net.load_state_dict(self.net.state_dict())
            self._teacher_net.eval()
            self.teacher = self._teacher_net
