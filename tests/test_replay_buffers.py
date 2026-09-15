"""
Replay buffer checks, including the spec §5.1 simulation: uniform reservoir (Algorithm R) vs CBRS occupancy on a
CoraFull-shaped stream with a 100-slot memory.

    python -m unittest tests.test_replay_buffers -v
    python tests/test_replay_buffers.py --table      # print the occupancy table (more trials)
"""
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Baselines.replay_buffers import CBRSBuffer, ReservoirBuffer  # noqa: E402

# CoraFull-CL per-class node counts (classes 0..69, from data/tr0.6_va0.2_te0.2_split_CoraFull-CL.pkl).
# ALL = train+val+test (19,793 nodes, the spec's simulation); TRAIN = the 60% split DRIFT actually streams (11,849).
CORAFULL_ALL = [257, 52, 243, 378, 63, 305, 404, 663, 240, 342, 141, 223, 102, 521, 341, 138, 115, 111, 80, 435, 420,
                254, 414, 196, 334, 315, 284, 783, 113, 466, 221, 376, 154, 855, 576, 84, 293, 163, 125, 564, 280, 205,
                94, 53, 129, 370, 122, 74, 557, 285, 72, 625, 501, 650, 99, 473, 324, 928, 212, 301, 116, 220, 165, 291,
                147, 91, 137, 84, 15, 29]
CORAFULL_TRAIN = [154, 31, 145, 226, 37, 183, 242, 397, 144, 205, 84, 133, 61, 312, 204, 82, 69, 66, 48, 261, 252, 152,
                  248, 117, 200, 189, 170, 469, 67, 279, 132, 225, 92, 513, 345, 50, 175, 97, 75, 338, 168, 123, 56, 31,
                  77, 222, 73, 44, 334, 171, 43, 375, 300, 390, 59, 283, 194, 556, 127, 180, 69, 132, 99, 174, 88, 54,
                  82, 50, 9, 17]
N_CLS_PER_TASK = 2
BUDGET = 100


def task_ordered_stream(class_sizes, rng):
    """Latent tasks in order (task t = classes 2t, 2t+1), nodes shuffled within each task."""
    stream = []
    for t in range(len(class_sizes) // N_CLS_PER_TASK):
        labels = [c for c in (N_CLS_PER_TASK * t, N_CLS_PER_TASK * t + 1) for _ in range(class_sizes[c])]
        rng.shuffle(labels)
        stream.extend(labels)
    return stream


def exact_reservoir_expectation(class_sizes, k):
    """Expected group counts for a uniform k-subset of the stream (hypergeometric; exact)."""
    from math import comb
    n_total = sum(class_sizes)
    total = comb(n_total, k)

    def p_at_most(n, j):
        return sum(comb(n, i) * comb(n_total - n, k - i) for i in range(j + 1)) / total

    tasks = [sum(class_sizes[N_CLS_PER_TASK * t:N_CLS_PER_TASK * (t + 1)])
             for t in range(len(class_sizes) // N_CLS_PER_TASK)]
    return {'task_zero': sum(p_at_most(n, 0) for n in tasks),
            'class_zero': sum(p_at_most(n, 0) for n in class_sizes),
            'class_le1': sum(p_at_most(n, 1) for n in class_sizes)}


def occupancy(buffer_cls, class_sizes, trials, seed=0):
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    n_cls = len(class_sizes)
    per_class, per_task = [], []
    for _ in range(trials):
        buf = buffer_cls(BUDGET)
        for i, y in enumerate(task_ordered_stream(class_sizes, rng)):
            buf.add(i, y)
        counts = np.zeros(n_cls, dtype=int)
        for y in buf.labels:
            counts[y] += 1
        per_class.append(counts)
        per_task.append(counts.reshape(-1, N_CLS_PER_TASK).sum(1))
    per_class, per_task = np.array(per_class), np.array(per_task)
    return {
        'task_mean': per_task.mean(), 'task_zero': (per_task == 0).sum(1).mean(),
        'class_mean': per_class.mean(), 'class_zero': (per_class == 0).sum(1).mean(),
        'class_le1': (per_class <= 1).sum(1).mean(), 'class_max': per_class.max(1).mean(),
        'class_min': per_class.min(1).mean(),
    }


class TestReservoir(unittest.TestCase):
    def test_fills_then_keeps_budget(self):
        torch.manual_seed(0)
        buf = ReservoirBuffer(10)
        for i in range(500):
            buf.add(i, i % 3)
        self.assertEqual(len(buf), 10)
        self.assertEqual(buf.n_seen, 500)

    def test_uniform_inclusion_probability(self):
        """Every stream position is retained with probability budget/n (Algorithm R's defining property)."""
        torch.manual_seed(0)
        n, k, trials = 50, 5, 4000
        hits = np.zeros(n)
        for _ in range(trials):
            buf = ReservoirBuffer(k)
            for i in range(n):
                buf.add(i, 0)
            hits[buf.ids] += 1
        freq = hits / trials
        self.assertAlmostEqual(freq.mean(), k / n, places=6)
        self.assertLess(np.abs(freq - k / n).max(), 0.035)   # ~6 binomial std at p=0.1, 4000 trials

    def test_payload_follows_slot(self):
        torch.manual_seed(0)
        buf = ReservoirBuffer(4)
        for i in range(40):
            buf.add(i, i % 2, logits=torch.full((3,), float(i)))
        for node_id, logits in zip(buf.ids, buf.payload['logits']):
            self.assertTrue(torch.all(logits == node_id))


class TestCBRS(unittest.TestCase):
    def test_small_classes_kept_entirely(self):
        """Paper property 1: a class with fewer than m/n_c instances is stored in its entirety."""
        torch.manual_seed(0)
        sizes = {0: 191, 1: 6742, 2: 51, 3: 654, 4: 1909}          # the paper's Figure 1 stream, m=1000
        labels = [c for c, n in sizes.items() for _ in range(n)]
        rng = np.random.RandomState(0)
        rng.shuffle(labels)
        buf = CBRSBuffer(1000)
        for i, y in enumerate(labels):
            buf.add(i, y)
        counts = buf.class_counts()
        self.assertEqual(counts[0], 191)
        self.assertEqual(counts[2], 51)
        self.assertEqual(sum(counts.values()), 1000)
        self.assertLessEqual(max(counts.values()) - min(counts[1], counts[3], counts[4]), 1)

    def test_full_classes_never_grow(self):
        torch.manual_seed(0)
        buf = CBRSBuffer(20)
        stream = [0] * 200 + [1] * 50 + [2] * 5
        for i, y in enumerate(stream):
            before = buf.class_counts().get(y, 0)
            was_full = y in buf.full
            buf.add(i, y)
            if was_full:
                self.assertLessEqual(buf.class_counts().get(y, 0), before)

    def test_payload_follows_slot(self):
        torch.manual_seed(0)
        buf = CBRSBuffer(6)
        for i in range(60):
            buf.add(i, i % 4, emb=torch.full((2,), float(i)))
        for node_id, emb in zip(buf.ids, buf.payload['emb']):
            self.assertTrue(torch.all(emb == node_id))


class TestSpecSimulation(unittest.TestCase):
    """Spec §5.1: Algorithm R over a CoraFull-shaped stream (19,793 nodes, 35 tasks, 70 classes, k=100)."""

    def test_reservoir_matches_exact_expectation(self):
        """
        Algorithm R keeps a uniform k-subset of the stream, so a group of n nodes gets a hypergeometric(N, n, k)
        number of slots. Exact expectations for CoraFull (all 19,793 nodes, k=100):
            tasks with 0 slots 4.37, classes with 0 slots 24.18, classes with <=1 slot 43.72.
        The spec's table (1.9 / 16.7 / 40.6) does not reproduce with CoraFull's real class sizes; see NOTES.md.
        """
        r = occupancy(ReservoirBuffer, CORAFULL_ALL, trials=60)
        exact = exact_reservoir_expectation(CORAFULL_ALL, BUDGET)
        self.assertAlmostEqual(r['task_mean'], 100 / 35, places=6)
        self.assertAlmostEqual(r['class_mean'], 100 / 70, places=6)
        self.assertLess(abs(r['task_zero'] - exact['task_zero']), 0.6)      # ~3 SE at 60 trials
        self.assertLess(abs(r['class_zero'] - exact['class_zero']), 1.2)
        self.assertLess(abs(r['class_le1'] - exact['class_le1']), 1.2)
        self.assertAlmostEqual(exact['class_zero'], 24.18, places=2)

    def test_cbrs_covers_every_class(self):
        r = occupancy(CBRSBuffer, CORAFULL_ALL, trials=5)
        self.assertEqual(r['class_zero'], 0.0)
        self.assertEqual(r['task_zero'], 0.0)
        self.assertLessEqual(r['class_max'], 2.0)


if __name__ == '__main__':
    if '--table' in sys.argv:
        print('| stream | buffer | tasks: mean slots | tasks with 0 | classes: mean | classes with 0 | classes with <=1 '
              '| max slots/class | trials |')
        print('|---|---|---|---|---|---|---|---|---|')
        for name, sizes in (('CoraFull all nodes (spec)', CORAFULL_ALL), ('CoraFull train split (DRIFT stream)',
                                                                          CORAFULL_TRAIN)):
            for label, cls, trials in (('Algorithm R', ReservoirBuffer, 300), ('CBRS', CBRSBuffer, 30)):
                r = occupancy(cls, sizes, trials)
                print(f"| {name} | {label} | {r['task_mean']:.2f} | {r['task_zero']:.1f} | {r['class_mean']:.2f} | "
                      f"{r['class_zero']:.1f} | {r['class_le1']:.1f} | {r['class_max']:.1f} | {trials} |", flush=True)
    else:
        unittest.main()
