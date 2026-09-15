"""
DER++ — Dark Experience Replay with ground-truth labels (baselines spec §5.3).

Buzzega et al., NeurIPS 2020, arXiv:2004.07211, Algorithm 2; ported from Mammoth models/derpp.py:
    loss = CE(stream) + α·MSE(f(x'), z') + β·CE(f(x''), y'')
with x' and x'' two *independent* draws from the buffer (Mammoth calls buffer.get_data twice).
λ_D = 1 on the stream term, matching ER. Graph adaptation and logit storage as in der_model.py.
Hyperparameter grid (paper Table 10, MNIST-360): α ∈ {0.2, 0.5}, β ∈ {0.5, 1.0}.
"""
import torch
import torch.nn.functional as F

from .der_model import NET as DER


class NET(DER):
    ARGS_ATTR = 'derpp_args'
    DEFAULTS = dict(DER.DEFAULTS, alpha=0.5, beta=1.0)

    def _replay_terms(self, head, device):
        terms = super()._replay_terms(head, device)            # first draw: logit matching
        slots = self._replay_draw()                              # second, independent draw: labels
        ids = [self.buffer.ids[i] for i in slots]
        y = torch.tensor([self.buffer.labels[i] for i in slots], dtype=torch.long, device=device)
        terms['ce'] = float(self.hp['beta']) * F.cross_entropy(head(self._isolated_logits(self.net, ids, device)), y)
        return terms
