"""
Replay memories shared by the new DRIFT baselines (ER-CBRS, DER, DER++, PDGNN, CLS-ER, combination).

A buffer holds one entry per slot: the node id (original id in dataset.graph), its label, and optional
per-slot tensors (e.g. DER's logits, PDGNN's topology-aware embedding). Population policy is the only
thing that differs between ReservoirBuffer and CBRSBuffer; sampling for replay is shared.

RNG: all random draws use torch's global generator, which training.utils.set_seed seeds.
"""
import torch


class _SlotBuffer:
    def __init__(self, budget):
        self.budget = int(budget)
        self.ids = []           # original node ids
        self.labels = []        # int labels
        self.payload = {}       # name -> list of tensors, aligned with ids

    def __len__(self):
        return len(self.ids)

    def _write(self, slot, node_id, label, payload):
        if slot == len(self.ids):
            self.ids.append(node_id)
            self.labels.append(label)
            for k, v in payload.items():
                self.payload.setdefault(k, []).append(v)
        else:
            self.ids[slot] = node_id
            self.labels[slot] = label
            for k, v in payload.items():
                self.payload[k][slot] = v

    def sample(self, n):
        """Indices of n distinct slots drawn uniformly (as DRIFT's ER and Mammoth's get_data: no replacement)."""
        n = min(int(n), len(self))
        return torch.randperm(len(self))[:n].tolist()

    def class_counts(self):
        counts = {}
        for y in self.labels:
            counts[y] = counts.get(y, 0) + 1
        return counts

    def nbytes(self):
        b = 16 * len(self)  # int64 id + int64 label per slot
        for tensors in self.payload.values():
            b += sum(t.numel() * t.element_size() for t in tensors)
        return b


class ReservoirBuffer(_SlotBuffer):
    """
    Algorithm R (Vitter 1985), as in Mammoth utils/buffer.py `reservoir`: while filling, store; afterwards the
    t-th stream item (0-based count t = items seen before it) replaces a uniformly chosen slot with probability
    budget/(t+1).

    Note: DRIFT's er_model.py and OCGL's ReservoirSamplingBuffer draw randint(0, n_seen + i), i.e. over t items
    instead of t+1; the acceptance probability is budget/t. This class follows Vitter/Mammoth.
    """

    def __init__(self, budget):
        super().__init__(budget)
        self.n_seen = 0

    def add(self, node_id, label, **payload):
        if len(self) < self.budget:
            self._write(len(self), node_id, label, payload)
        else:
            j = int(torch.randint(0, self.n_seen + 1, (1,)))
            if j < self.budget:
                self._write(j, node_id, label, payload)
        self.n_seen += 1


class CBRSBuffer(_SlotBuffer):
    """
    Class-Balancing Reservoir Sampling, Chrysakis & Moens, "Online Continual Learning from Imbalanced Data",
    ICML 2020 (PMLR 119:1952-1961), Algorithm 1 — ported line by line.

    A class is *largest* if no class has more stored instances; it is *full* if it is, or has ever been, a
    largest class once the memory is filled ("Once a class becomes full, it remains so in the future").
      - memory not filled: store.
      - class c not full: overwrite a random stored instance of a largest class.
      - class c full: with probability m_c / n_c overwrite a random stored instance of class c.
    n_c counts every stream instance of c seen so far, including the current one.
    """

    def __init__(self, budget):
        super().__init__(budget)
        self.n_seen_class = {}
        self.full = set()
        self.n_seen = 0

    def _refresh_full(self):
        counts = self.class_counts()
        top = max(counts.values())
        self.full.update(c for c, m in counts.items() if m == top)

    def add(self, node_id, label, **payload):
        self.n_seen += 1
        self.n_seen_class[label] = self.n_seen_class.get(label, 0) + 1
        if len(self) < self.budget:                                   # lines 3-4
            self._write(len(self), node_id, label, payload)
            if len(self) == self.budget:
                self._refresh_full()
            return
        if label not in self.full:                                    # lines 6-9
            counts = self.class_counts()
            top = max(counts.values())
            largest = [c for c, m in counts.items() if m == top]
            # "find all instances of the largest class": pool instances of every class tied for largest
            candidates = [i for i, y in enumerate(self.labels) if y in largest]
            slot = candidates[int(torch.randint(0, len(candidates), (1,)))]
            self._write(slot, node_id, label, payload)
        else:                                                         # lines 11-19
            m_c = sum(1 for y in self.labels if y == label)
            n_c = self.n_seen_class[label]
            u = float(torch.rand(1))
            if m_c > 0 and u <= m_c / n_c:
                candidates = [i for i, y in enumerate(self.labels) if y == label]
                slot = candidates[int(torch.randint(0, len(candidates), (1,)))]
                self._write(slot, node_id, label, payload)
        self._refresh_full()


def make_buffer(kind, budget):
    if kind == 'reservoir':
        return ReservoirBuffer(budget)
    if kind == 'cbrs':
        return CBRSBuffer(budget)
    raise ValueError(f'unknown buffer kind {kind!r} (reservoir | cbrs)')
