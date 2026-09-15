"""
DER — Dark Experience Replay (baselines spec §5.2).

Buzzega, Boschini, Porrello, Abati, Calderara, "Dark Experience for General Continual Learning: a Strong, Simple
Baseline", NeurIPS 2020, arXiv:2004.07211, Algorithm 1. Ported from Mammoth models/der.py / derpp.py
(github.com/aimagelab/mammoth): reservoir buffer of (input, label, logits); loss = CE(stream) + α·MSE(f(x_buf), z_buf),
F.mse_loss with mean reduction over the buffer minibatch and all output units; logits stored are the ones computed
before the optimiser step.

Graph adaptation (NOTES.md Q1). Replay forwards buffered nodes as isolated nodes (DRIFT's replay convention), so the
stored logits z are computed the same way — on the isolated node, under the pre-step weights — rather than taken
from the neighbourhood-sampled stream forward as Mammoth's `outputs.data` would be. 'insert_logits':'stream' stores
the stream-forward logits instead (Mammoth's literal behaviour), for measuring that mismatch.

Output units: Mammoth's logit matching covers the full head; here too (all n_cls units), while CE terms use the
class-IL head [:offset2] like every DRIFT observe_cis.
Hyperparameter grid (paper Table 10, MNIST-360, the general-continual benchmark): α ∈ {0.5, 1.0}.
"""
import torch
import torch.nn.functional as F

from .replay_base import ReplayNET


class NET(ReplayNET):
    ARGS_ATTR = 'der_args'
    DEFAULTS = {'budget': 100, 'memory_proportion': 1, 'buffer': 'reservoir', 'alpha': 0.5, 'insert_logits': 'isolated'}

    def _replay_terms(self, head, device):
        """Loss terms computed on the buffer; DER++ and the combination extend this."""
        slots = self._replay_draw()
        ids = [self.buffer.ids[i] for i in slots]
        z = torch.stack([self.buffer.payload['logits'][i] for i in slots]).to(device)
        out = self._isolated_logits(self.net, ids, device)
        return {'mse': float(self.hp['alpha']) * F.mse_loss(out, z)}

    def _step(self, args, g, labels, train_ids, cis):
        self.net.train()
        device = next(self.net.parameters()).device
        head, _ = self._head_fn(labels, train_ids, cis)
        new_orig = self._orig_ids(g, train_ids)
        y_new = labels[train_ids]

        self.net.zero_grad()
        stream_logits = self._logits(self.net, self._stream_blocks(args, g, train_ids))
        loss = F.cross_entropy(head(stream_logits), y_new)
        if self.hp['insert_logits'] == 'isolated':
            with torch.no_grad():
                insert_logits = self._isolated_logits(self.net, new_orig, device).detach()
        else:
            insert_logits = stream_logits.detach()
        if len(self.buffer) > 0:
            for term in self._replay_terms(head, device).values():
                loss = loss + term
        loss.backward()
        self.opt.step()

        for nid, y, z in zip(new_orig, y_new.tolist(), insert_logits.cpu()):
            self.buffer.add(nid, int(y), logits=z.clone())
