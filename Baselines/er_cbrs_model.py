"""
ER-CBRS: Experience Replay with a Class-Balancing Reservoir (baselines spec §5.1).

Memory population: CBRS, Chrysakis & Moens, ICML 2020, Algorithm 1 (Baselines/replay_buffers.py).
Everything else matches DRIFT's ER (Baselines/er_model.py): loss = CE(stream) + CE(replay), with batch_size
replayed nodes drawn uniformly from memory, so the comparison isolates the memory population policy.

The CBRS paper also proposes *weighted replay* (replay probability of a slot inversely proportional to its class's
stored count) and the convex loss a·L_stream + (1-a)·L_replay with a = 1/n_classes_seen (Algorithm 2).
Both are off by default to keep DRIFT's ER protocol; enable with 'replay':'weighted' and 'loss':'convex'.
"""
import torch
import torch.nn.functional as F

from .replay_base import ReplayNET


class NET(ReplayNET):
    ARGS_ATTR = 'er_cbrs_args'
    DEFAULTS = {'budget': 100, 'memory_proportion': 1, 'buffer': 'cbrs', 'replay': 'uniform', 'loss': 'sum'}

    def _replay_draw(self):
        if self.hp['replay'] != 'weighted':
            return super()._replay_draw()
        counts = self.buffer.class_counts()
        w = torch.tensor([1.0 / counts[y] for y in self.buffer.labels])
        n = min(self.replay_size, len(self.buffer))
        return torch.multinomial(w, n, replacement=False).tolist()

    def _step(self, args, g, labels, train_ids, cis):
        self.net.train()
        device = next(self.net.parameters()).device
        head, _ = self._head_fn(labels, train_ids, cis)
        new_orig = self._orig_ids(g, train_ids)
        y_new = labels[train_ids]

        self.net.zero_grad()
        loss_stream = F.cross_entropy(head(self._logits(self.net, self._stream_blocks(args, g, train_ids))), y_new)
        loss = loss_stream
        if len(self.buffer) > 0:
            slots = self._replay_draw()
            ids = [self.buffer.ids[i] for i in slots]
            y_buf = torch.tensor([self.buffer.labels[i] for i in slots], dtype=torch.long, device=device)
            loss_replay = F.cross_entropy(head(self._isolated_logits(self.net, ids, device)), y_buf)
            if self.hp['loss'] == 'convex':
                a = 1.0 / max(len(self.seen_classes), 1)
                loss = a * loss_stream + (1 - a) * loss_replay
            else:
                loss = loss_stream + loss_replay
        loss.backward()
        self.opt.step()

        for nid, y in zip(new_orig, y_new.tolist()):
            self.buffer.add(nid, int(y))
