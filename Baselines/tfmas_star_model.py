import collections
import math

import dgl
import numpy as np
import torch
import torch.nn.functional as F

from telemetry import make_telemetry

# Values stated in Aljundi, Kelchtermans & Tuytelaars, "Task-Free Continual Learning"
# (arXiv:1812.03596v3): loss window of 5, hard buffer of 100. lam follows DRIFT App. A.4.
# l_th / std_th (delta_mu / delta_sigma) are not given in the paper and must be passed explicitly.
DEFAULTS = {
    'window': 5,
    'buffer_size': 100,
    'lam': 0.5,
    'passes': 1,
    'window_push': 'sum',   # 'sum': one entry L(X,Y)+L(X_B,Y_B) per step; 'both': two entries
}


def plateau_detector(W, P, mu_old, sigma_old, l_th, std_th, has_buffer=True):
    """
    Lines 13 and 16-21 of Algorithm 1. Mutates W (cleared on consolidation) and returns the new latch state,
    plateau statistics and per-condition outcomes. P=True means consolidation is blocked until a peak.

    Interpretations: consolidation also requires a non-empty buffer (line 14 estimates Omega on it);
    the peak check (line 19) is skipped while W is empty, since its mean is undefined.
    """
    win_len = len(W)
    win_mean = float(np.mean(W)) if win_len else float('nan')
    win_std = float(np.std(W)) if win_len else float('nan')
    cond_latch = not P
    cond_mean = win_len > 0 and win_mean < l_th
    cond_std = win_len > 0 and win_std < std_th
    consolidate = cond_latch and cond_mean and cond_std and has_buffer
    if consolidate:
        mu_old, sigma_old = win_mean, win_std     # line 16
        W.clear()                                 # line 17
        P = True
    peak_fired = False
    if len(W) > 0 and float(np.mean(W)) > mu_old + sigma_old:   # line 19
        peak_fired = P
        P = False
    return {'consolidate': consolidate, 'peak_fired': peak_fired, 'P': P, 'mu_old': mu_old, 'sigma_old': sigma_old,
            'win_len': win_len, 'win_mean': win_mean, 'win_std': win_std, 'cond_latch': cond_latch,
            'cond_mean': cond_mean, 'cond_std': cond_std, 'cond_buffer': has_buffer}


class NET(torch.nn.Module):
    """
        MAS* — Algorithm 1 of Aljundi et al., "Task-Free Continual Learning" (arXiv:1812.03596v3).

        Differs from Baselines/tfmas_model.py (DRIFT's task-free MAS) in: a prioritized-keeping hard
        buffer, Omega estimated per sample on that buffer, the window cleared on consolidation,
        sigma (not variance) thresholded, and the task losses (not the penalty) pushed to the window.
        Interpretation choices where the paper is ambiguous are marked "Interpretation:".

        :param model: The backbone GNNs, e.g. GCN, GAT, GIN, etc.
        :param args: Experiment arguments; method hyperparameters come from args.tfmas_star_args.
        :param dataset: The NodeLevelDataset; its full graph is used to replay buffered nodes.
        """

    def __init__(self, model, args, dataset=None):
        super(NET, self).__init__()
        hp = dict(DEFAULTS)
        hp.update(args.tfmas_star_args)
        if 'l_th' not in hp or 'std_th' not in hp:
            raise ValueError("tfmas_star needs explicit thresholds, e.g. --tfmas_star_args \"'l_th':1.0;'std_th':0.1\" "
                             "(pass l_th < 0 to disable consolidation)")
        if args.epochs != 1:
            raise ValueError("tfmas_star runs its own inner passes (N in Algorithm 1) via tfmas_star_args 'passes'; use --epochs 1")
        if dataset is None:
            raise ValueError("tfmas_star needs dataset= to replay buffered nodes")

        self.l_th = float(hp['l_th'])
        self.std_th = float(hp['std_th'])
        self.lam = float(hp['lam'])
        self.window = int(hp['window'])
        self.buffer_size = int(hp['buffer_size'])
        self.passes = int(hp['passes'])
        self.window_push = str(hp['window_push'])
        if self.window_push not in ('sum', 'both'):
            raise ValueError(f"window_push must be 'sum' or 'both', got {self.window_push}")

        self.net = model
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.full_graph = dataset.graph

        # Algorithm 1, lines 2-3
        self.W = collections.deque(maxlen=self.window)
        self.P = False                      # True = consolidation blocked until a peak
        self.mu_old = 0.0
        self.sigma_old = 0.0
        self.omega = [torch.zeros_like(p) for p in self.net.parameters()]
        self.theta_star = [p.detach().clone() for p in self.net.parameters()]
        self.n_consolidations = 0
        self.buffer_ids = []                # original node ids in dataset.graph
        self.aux_g = None                   # buffer nodes as isolated nodes with self-loops

        self.seen_classes = set()
        self.step = 0
        self.tel = make_telemetry(args, f'tfmas_star_lth{self.l_th}_sth{self.std_th}')

    def forward(self, features):
        return self.net(features)

    def buffer_labels(self):
        if not self.buffer_ids:
            return []
        return self.full_graph.ndata['label'][torch.tensor(self.buffer_ids)].view(-1).tolist()

    def set_stream_info(self, task=None, weights=None):
        """Ground-truth stream state for the step about to be observed (telemetry only)."""
        if self.tel is None:
            return
        row = {'step': self.step, 'task': task}
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            row['task'] = int(np.argmax(w))
            row['alpha_max'] = float(w.max())
            for k, v in enumerate(w):
                row[f'alpha_{k}'] = float(v)
        self.tel.log('stream', row)

    def observe(self, args, g, features, labels, train_ids):
        """Task-free online step without output-head masking."""
        self._step(args, g, labels, train_ids, cis=False)

    def observe_cis(self, args, g, features, labels, train_ids):
        """Task-free online class-IL step: logits restricted to classes seen so far."""
        self._step(args, g, labels, train_ids, cis=True)

    # ------------------------------------------------------------------ helpers

    def _logits(self, blocks):
        output, _ = self.net.forward_batch(blocks, blocks[0].srcdata['feat'])
        if isinstance(output, tuple):
            output = output[0]
        return output

    def _full_blocks(self, graph):
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(len(self.net.gat_layers))
        seeds = torch.arange(graph.num_nodes(), device=graph.device)
        _, _, blocks = sampler.sample_blocks(graph, seeds)
        return blocks

    def _isolated_graph(self, orig_ids, device):
        # DRIFT's replay convention (Baselines/er_model.py): buffered nodes without their neighbourhoods.
        ids = torch.tensor(orig_ids, dtype=torch.long)
        sub = dgl.node_subgraph(self.full_graph, ids, store_ids=True)
        sub = dgl.remove_edges(sub, torch.arange(sub.num_edges()))
        sub = dgl.add_self_loop(sub)
        return sub.to(device)

    @staticmethod
    def _labels(t):
        return t.view(-1) if t.dim() > 1 else t

    def _flat(self, tensors):
        return torch.cat([t.reshape(-1) for t in tensors])

    def _omega_estimate(self, aux_blocks, head):
        """Eq. (3): Omega_i = 1/N sum_k |d F(x_k) / d theta_i|, F = squared L2 norm of the logits, over the buffer.
        Interpretation: F taken as in MAS [1]; evaluated in eval mode so dropout adds no noise."""
        params = list(self.net.parameters())
        was_training = self.net.training
        self.net.eval()
        out = head(self._logits(aux_blocks))
        n = out.shape[0]
        acc = [torch.zeros_like(p) for p in params]
        for k in range(n):
            grads = torch.autograd.grad(out[k].pow(2).sum(), params, retain_graph=k < n - 1, allow_unused=True)
            for a, gr in zip(acc, grads):
                if gr is not None:
                    a.add_(gr.abs())
        self.net.train(was_training)
        return [a / n for a in acc]

    # ------------------------------------------------------------------ Algorithm 1

    def _step(self, args, g, labels, train_ids, cis):
        self.net.train()
        device = next(self.net.parameters()).device
        if args.cuda:
            train_ids = train_ids.to(device='cuda:{}'.format(args.gpu))

        if cis:
            self.seen_classes.update(int(c) for c in labels[train_ids].unique().tolist())
            offset2 = max(self.seen_classes) + 1
            if offset2 % 2 != 0:
                offset2 += 1
            head = lambda out: out[:, 0:offset2]
        else:
            offset2 = None
            head = lambda out: out

        train_ids_cpu = train_ids.cpu()
        new_orig = g.ndata['_ID'][train_ids_cpu].cpu().tolist() if '_ID' in g.ndata else train_ids_cpu.tolist()

        # line 5: receive K recent samples
        nb_sampler = dgl.dataloading.NeighborSampler(args.n_nbs_sample) if args.sample_nbs else \
            dgl.dataloading.MultiLayerFullNeighborSampler(len(self.net.gat_layers))
        _, _, blocks = nb_sampler.sample_blocks(g, train_ids)
        y_new = labels[train_ids]

        aux_blocks, y_buf = None, None
        if self.aux_g is not None:
            aux_blocks = self._full_blocks(self.aux_g)
            y_buf = self._labels(aux_blocks[-1].dstdata['label'])

        params = list(self.net.parameters())
        loss_new_v, loss_buf_v, penalty_v = float('nan'), float('nan'), 0.0

        # lines 6-12
        for n in range(self.passes):
            self.net.zero_grad()
            loss_new = F.cross_entropy(head(self._logits(blocks)), y_new)
            loss_total = loss_new
            loss_buf = None
            if aux_blocks is not None:
                loss_buf = F.cross_entropy(head(self._logits(aux_blocks)), y_buf)
                loss_total = loss_total + loss_buf
            penalty = None
            if self.n_consolidations > 0:   # Omega = 0 before the first consolidation, so the penalty is exactly 0
                penalty = self.lam / 2. * sum(torch.sum(o * (p - s) ** 2)
                                              for o, p, s in zip(self.omega, params, self.theta_star))
                loss_total = loss_total + penalty
            loss_total.backward()
            self.opt.step()

            if n == 0:
                # line 10. Interpretation: the pre-update losses from line 7 are pushed.
                loss_new_v = float(loss_new.detach())
                loss_buf_v = float(loss_buf.detach()) if loss_buf is not None else float('nan')
                penalty_v = float(penalty.detach()) if penalty is not None else 0.0
                if self.window_push == 'both':
                    self.W.append(loss_new_v)
                    if loss_buf is not None:
                        self.W.append(loss_buf_v)
                else:
                    self.W.append(loss_new_v + (loss_buf_v if loss_buf is not None else 0.0))

        # lines 13, 16-21 (window, latch and plateau statistics)
        det = plateau_detector(self.W, self.P, self.mu_old, self.sigma_old, self.l_th, self.std_th,
                               has_buffer=aux_blocks is not None)
        self.P, self.mu_old, self.sigma_old = det['P'], det['mu_old'], det['sigma_old']
        win_len, win_mean, win_std = det['win_len'], det['win_mean'], det['win_std']
        cond_latch, cond_mean, cond_std, cond_buffer = det['cond_latch'], det['cond_mean'], det['cond_std'], det['cond_buffer']
        consolidated, peak_fired = det['consolidate'], det['peak_fired']
        event = None
        if consolidated:
            omega_before = self._flat(self.omega)
            estimate = self._omega_estimate(aux_blocks, head)                      # line 14
            k = self.n_consolidations + 1
            self.omega = [o + (e - o) / k for o, e in zip(self.omega, estimate)]   # cumulative moving average
            self.n_consolidations = k
            self.theta_star = [p.detach().clone() for p in params]                 # line 15
            if self.tel is not None:
                omega_after = self._flat(self.omega)
                est_flat = self._flat(estimate)
                cos_prev = float(F.cosine_similarity(est_flat, omega_before, dim=0)) if k > 1 else float('nan')
                event = {'omega_l1_before': float(omega_before.abs().sum()),
                         'omega_l1_after': float(omega_after.abs().sum()),
                         'omega_estimate_l1': float(est_flat.abs().sum()),
                         'cos_estimate_vs_prev_omega': cos_prev}

        # line 22: prioritized keeping — highest-loss nodes among buffer and new samples, scored under current theta.
        candidates = list(dict.fromkeys(self.buffer_ids + new_orig))
        cand_g = self._isolated_graph(candidates, device)
        cand_blocks = self._full_blocks(cand_g)
        was_training = self.net.training
        self.net.eval()
        with torch.no_grad():
            per_sample = F.cross_entropy(head(self._logits(cand_blocks)),
                                         self._labels(cand_blocks[-1].dstdata['label']), reduction='none')
        self.net.train(was_training)
        cand_orig = cand_g.ndata['_ID'].cpu()
        keep = torch.argsort(per_sample.cpu(), descending=True)[:self.buffer_size]
        self.buffer_ids = cand_orig[keep].tolist()
        self.aux_g = self._isolated_graph(self.buffer_ids, device)

        if self.tel is not None:
            kept_losses = per_sample.cpu()[keep]
            flat_params = self._flat([p.detach() for p in params])
            omega_flat = self._flat(self.omega)
            row = {
                'step': self.step, 'loss_new': loss_new_v, 'loss_buffer': loss_buf_v,
                'loss_total': loss_new_v + (0.0 if math.isnan(loss_buf_v) else loss_buf_v) + penalty_v,
                'win_len': win_len, 'win_mean': win_mean, 'win_std': win_std,
                'l_th': self.l_th, 'std_th': self.std_th,
                'P_blocked': int(not cond_latch), 'cond_latch': int(cond_latch), 'cond_mean': int(cond_mean),
                'cond_std': int(cond_std), 'cond_buffer': int(cond_buffer),
                'consolidated': int(consolidated), 'peak_fired': int(peak_fired),
                'n_consolidations': self.n_consolidations, 'mu_old': self.mu_old, 'sigma_old': self.sigma_old,
                'omega_l1': float(omega_flat.abs().sum()), 'omega_l2': float(omega_flat.norm()),
                'omega_max': float(omega_flat.abs().max()),
                'theta_anchor_dist': float((flat_params - self._flat(self.theta_star)).norm()),
                'penalty_value': penalty_v,
                'buffer_size': len(self.buffer_ids), 'buffer_loss_mean': float(kept_losses.mean()),
                'buffer_loss_min': float(kept_losses.min()), 'buffer_loss_max': float(kept_losses.max()),
                'offset2': offset2 if offset2 is not None else -1,
            }
            self.tel.log('steps', row)
            if consolidated or peak_fired:
                buf_labels = self.full_graph.ndata['label'][torch.tensor(self.buffer_ids)].view(-1).tolist()
                hist = collections.Counter(int(c) for c in buf_labels)
                ev = {'step': self.step, 'event_type': 'consolidate' if consolidated else 'peak',
                      'win_mean': win_mean, 'win_std': win_std, 'mu_old': self.mu_old, 'sigma_old': self.sigma_old,
                      'n_consolidations': self.n_consolidations,
                      'omega_l1_before': float('nan'), 'omega_l1_after': float('nan'),
                      'omega_estimate_l1': float('nan'), 'cos_estimate_vs_prev_omega': float('nan'),
                      'buffer_size': len(self.buffer_ids),
                      'buffer_label_hist': '|'.join(f'{c}:{n}' for c, n in sorted(hist.items()))}
                if event is not None:
                    ev.update(event)
                self.tel.log('events', ev)

        self.step += 1
