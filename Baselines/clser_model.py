"""
CLS-ER — Complementary Learning System based Experience Replay (baselines spec §5.5).

Arani, Sarfraz, Zonooz, "Learning Fast, Learning Slow: A General Continual Learning Method based on Complementary
Learning System", ICLR 2022, arXiv:2201.12604. Ported line by line from the official code
(github.com/NeurAI-Lab/CLS-ER, models/clser.py):
  - working model (trained) + plastic and stable semantic memories (EMA copies of the working model);
  - one buffer draw per step; for each buffered sample the EMA target is the stable logits if the stable model's
    probability for the true label exceeds the plastic model's, otherwise the plastic logits;
  - loss = reg_weight · mean MSE(f_θ(x_buf), target) + CE(f_θ([x_stream, x_buf]), [y_stream, y_buf])
    (CE over the concatenated batch, i.e. the mean over stream and buffer rows);
  - after the optimiser step, global_step += 1, then each EMA is updated with probability update_freq using
    alpha = min(1 - 1/(global_step + 1), alpha_max);
  - reservoir buffer (Mammoth utils/buffer.py).

Paper vs code (code followed): the alpha warm-up ramp min(1 - 1/(step+1), alpha) exists only in the code; Algorithm 1
line 8 breaks probability ties toward the stable model, the code toward the plastic model.

Adaptations: graph replay as isolated nodes (DRIFT convention); consistency and target selection over the full output
head (as the official code), CE over the class-IL head. Inference model: "For inference, we use the stable model as it
retains long-term memory across the tasks" (paper); eval_model() selects it and alt_eval_models() lets the telemetry
log the working and plastic curves too. 'reg_weight': 0 keeps the EMAs but removes their feedback — the spec's
required "EMA without consistency" ablation.
Paper Table S4, MNIST-360 (general continual learning, buffer 500): reg_weight 1.25, alpha_S = alpha_P = 0.99,
r_S 0.9, r_P 1.0. The official repo defaults (0.1 / 0.999 / 0.7 / 0.9) are the vision settings. EMA decay on DRIFT is
set from the stream scale (spec §5.5): see experiments/run_matrix.py plan clser_tune.
"""
import copy

import torch
import torch.nn.functional as F

from .replay_base import ReplayNET


class NET(ReplayNET):
    ARGS_ATTR = 'clser_args'
    DEFAULTS = {'budget': 100, 'memory_proportion': 1, 'buffer': 'reservoir', 'reg_weight': 0.1,
                'stable_update_freq': 0.7, 'stable_alpha': 0.999, 'plastic_update_freq': 0.9, 'plastic_alpha': 0.999,
                'eval_model': 'stable'}

    def __init__(self, model, args, dataset=None):
        super().__init__(model, args, dataset)
        self.plastic = copy.deepcopy(self.net).requires_grad_(False)
        self.stable = copy.deepcopy(self.net).requires_grad_(False)
        self.global_step = 0

    # ---- models used for evaluation
    def _models(self):
        return {'working': self.net, 'plastic': self.plastic, 'stable': self.stable}

    def eval_model(self):
        return self._models()[self.hp['eval_model']]

    def alt_eval_models(self):
        return {k: v for k, v in self._models().items() if k != self.hp['eval_model']}

    def memory_accounting(self):
        n = 2 * sum(p.numel() for p in self.net.parameters())
        return {'buffer_bytes': self.buffer.nbytes(), 'extra_param_count': n, 'extra_param_bytes': 4 * n}

    # ---- training
    @torch.no_grad()
    def _ema_update(self, ema, alpha_max):
        alpha = min(1 - 1 / (self.global_step + 1), alpha_max)
        for ema_p, p in zip(ema.parameters(), self.net.parameters()):
            ema_p.mul_(alpha).add_(p, alpha=1 - alpha)

    def _consistency_target(self, ids, y_buf, device):
        with torch.no_grad():
            s_logits = self._isolated_logits(self.stable, ids, device)
            p_logits = self._isolated_logits(self.plastic, ids, device)
            s_prob = F.softmax(s_logits, 1).gather(1, y_buf.unsqueeze(1))
            p_prob = F.softmax(p_logits, 1).gather(1, y_buf.unsqueeze(1))
            return torch.where(s_prob > p_prob, s_logits, p_logits)

    def _step(self, args, g, labels, train_ids, cis):
        self.net.train()
        device = next(self.net.parameters()).device
        head, _ = self._head_fn(labels, train_ids, cis)
        new_orig = self._orig_ids(g, train_ids)
        y_new = labels[train_ids]

        self.net.zero_grad()
        stream_logits = self._logits(self.net, self._stream_blocks(args, g, train_ids))
        loss = 0.0
        if len(self.buffer) > 0:
            slots = self._replay_draw()
            ids = [self.buffer.ids[i] for i in slots]
            y_buf = torch.tensor([self.buffer.labels[i] for i in slots], dtype=torch.long, device=device)
            target = self._consistency_target(ids, y_buf, device)
            buf_logits = self._isolated_logits(self.net, ids, device)
            loss = float(self.hp['reg_weight']) * F.mse_loss(buf_logits, target)
            ce = F.cross_entropy(head(torch.cat([stream_logits, buf_logits])), torch.cat([y_new, y_buf]))
        else:
            ce = F.cross_entropy(head(stream_logits), y_new)
        loss = loss + ce
        loss.backward()
        self.opt.step()

        for nid, y in zip(new_orig, y_new.tolist()):
            self.buffer.add(nid, int(y))

        self.global_step += 1
        if float(torch.rand(1)) < float(self.hp['plastic_update_freq']):
            self._ema_update(self.plastic, float(self.hp['plastic_alpha']))
        if float(torch.rand(1)) < float(self.hp['stable_update_freq']):
            self._ema_update(self.stable, float(self.hp['stable_alpha']))
