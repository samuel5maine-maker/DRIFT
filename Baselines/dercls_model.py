"""
DER++ + CLS-ER consistency over a class-balanced buffer (baselines spec §5.7).

Provenance, stated plainly: the loss is not novel. The first three terms are DER++ (Buzzega et al., NeurIPS 2020,
Algorithm 2); the fourth is CLS-ER's consistency to an EMA of the working model (Arani et al., ICLR 2022); the memory
is CBRS (Chrysakis & Moens, ICML 2020). Bhat, Zonooz & Arani (CoLLAs 2022, arXiv:2207.04998) treat DER++ and CLS-ER as
members of one family — consistency across time-separated views of buffered samples — so β and γ may be redundant.

    L = CE(f(x_new), y_new)
      + alpha · CE(f(x_buf1), y_buf1)
      + beta  · MSE(f(x_buf2), z_buf2)         stored logits at insertion   (DER++)
      + gamma · MSE(f(x_buf3), f_φ(x_buf3))    single EMA of the working model (CLS-ER)

Three independent buffer draws. Naming follows the spec (§5.7: α on the label term, β on stored logits), which is the
reverse of DER++'s paper naming (α logits, β labels). One EMA φ, updated like CLS-ER's semantic memories
(probability ema_update_freq, alpha ramp min(1 - 1/(step+1), ema_alpha)).
Required ablation: beta-only (gamma=0), gamma-only (beta=0), both. Evaluation model: 'eval_model' ∈ {ema, working};
the other one is logged by the telemetry.
"""
import copy

import torch
import torch.nn.functional as F

from .der_model import NET as DER


class NET(DER):
    ARGS_ATTR = 'dercls_args'
    DEFAULTS = dict(DER.DEFAULTS, buffer='cbrs', alpha=1.0, beta=0.5, gamma=0.1, ema_alpha=0.99, ema_update_freq=0.9,
                    eval_model='ema')

    def __init__(self, model, args, dataset=None):
        super().__init__(model, args, dataset)
        self.ema = copy.deepcopy(self.net).requires_grad_(False)
        self.global_step = 0

    def eval_model(self):
        return self.ema if self.hp['eval_model'] == 'ema' else self.net

    def alt_eval_models(self):
        return {'working': self.net} if self.hp['eval_model'] == 'ema' else {'ema': self.ema}

    def memory_accounting(self):
        n = sum(p.numel() for p in self.net.parameters())
        return {'buffer_bytes': self.buffer.nbytes(), 'extra_param_count': n, 'extra_param_bytes': 4 * n}

    def _draw_ids(self):
        slots = self._replay_draw()
        return slots, [self.buffer.ids[i] for i in slots]

    def _replay_terms(self, head, device):
        terms = {}
        a, b, c = float(self.hp['alpha']), float(self.hp['beta']), float(self.hp['gamma'])
        if a > 0:
            slots, ids = self._draw_ids()
            y = torch.tensor([self.buffer.labels[i] for i in slots], dtype=torch.long, device=device)
            terms['ce'] = a * F.cross_entropy(head(self._isolated_logits(self.net, ids, device)), y)
        if b > 0:
            slots, ids = self._draw_ids()
            z = torch.stack([self.buffer.payload['logits'][i] for i in slots]).to(device)
            terms['stored_logits'] = b * F.mse_loss(self._isolated_logits(self.net, ids, device), z)
        if c > 0:
            _, ids = self._draw_ids()
            with torch.no_grad():
                target = self._isolated_logits(self.ema, ids, device)
            terms['ema_logits'] = c * F.mse_loss(self._isolated_logits(self.net, ids, device), target)
        return terms

    def _step(self, args, g, labels, train_ids, cis):
        super()._step(args, g, labels, train_ids, cis)
        self.global_step += 1
        if float(torch.rand(1)) < float(self.hp['ema_update_freq']):
            alpha = min(1 - 1 / (self.global_step + 1), float(self.hp['ema_alpha']))
            with torch.no_grad():
                for ema_p, p in zip(self.ema.parameters(), self.net.parameters()):
                    ema_p.mul_(alpha).add_(p, alpha=1 - alpha)
