"""
Runs a list of DRIFT experiments with telemetry, skipping any whose result file already exists.

    python experiments/run_matrix.py --plan calib_t1 --workers 2
    python experiments/run_matrix.py --plan list --plan-file jobs.txt

Every job gets MAS_TELEMETRY_DIR=telemetry and --cuda no. Logs go to experiments/logs/.
Data caches are built serially (one job per regime) before any parallel work, because
two processes preparing the same stream pickle would race.
"""
import argparse
import itertools
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drift_args import main_args  # noqa: E402
from training.utils import result_name  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # code that runs (may be a frozen worktree)
# data, results, telemetry and logs live here; set DRIFT_OUT_ROOT to the main checkout when running from a worktree,
# so code edits in the main checkout cannot affect jobs that are already queued
OUT_ROOT = os.path.abspath(os.environ.get('DRIFT_OUT_ROOT', ROOT))
PY = sys.executable

REGIMES = {
    'clsincre': ['--setting', 'tfocis'],
    'sigma3': ['--setting', 'tfo_gaussian', '--gaussian_sigma', '3'],
    'sigma10': ['--setting', 'tfo_gaussian', '--gaussian_sigma', '10'],
    'sigma20': ['--setting', 'tfo_gaussian', '--gaussian_sigma', '20'],
    'sigma60': ['--setting', 'tfo_gaussian', '--gaussian_sigma', '60'],
    'bb5': ['--setting', 'tfo_bb', '--blurry_batch_count', '5'],
    'blurry30': ['--setting', 'tfo_blurry', '--percentage', '0.7'],
}


DATASET_FLAGS = {'Arxiv-CL': ['--ori_data_path', os.path.join(OUT_ROOT, 'data', 'raw')]}


def job(regime, method, seed, dataset='CoraFull-CL', backbone='GCN', star_args=None, method_args=None):
    """One main.py invocation. method_args (or star_args, for tfmas_star) become --{method}_args."""
    method_args = star_args if star_args is not None else method_args
    cmd = ['--dataset', dataset, '--backbone', backbone, '--method', method, '--seed', str(seed), '--cuda', 'no',
           '--data_path', os.path.join(OUT_ROOT, 'data')]
    cmd += DATASET_FLAGS.get(dataset, [])
    cmd += REGIMES[regime]
    if method_args:
        cmd += [f'--{method}_args', ';'.join(f"'{k}':{v}" for k, v in method_args.items())]
    name = f'{dataset}_{backbone}_{regime}_{method}'
    if method_args:
        name += '_' + '_'.join(f'{k}{v}' for k, v in method_args.items())
    name += f'_seed{seed}'
    return {'name': name, 'regime': regime, 'args': cmd}


def result_prefix(j):
    args = main_args(j['args'])
    return os.path.join(OUT_ROOT, args.result_path, result_name(args))


def done(j):
    p = result_prefix(j)
    return os.path.exists(p + '_tm.txt') or os.path.exists(p + '_cfmat.txt')


def with_results_dir(jobs, results_dir):
    if results_dir == 'results':
        return jobs
    for j in jobs:
        j['args'] = j['args'] + ['--result_path', os.path.join(OUT_ROOT, results_dir)]
    return jobs


def run(j, threads):
    if done(j):
        return j['name'], 'skipped', 0.0
    results_dir = os.path.basename(os.path.normpath(main_args(j['args']).result_path))
    suffix = '' if results_dir == 'results' else results_dir[len('results'):] if results_dir.startswith('results') \
        else '_' + results_dir
    log_dir = os.path.join(OUT_ROOT, 'experiments', 'logs' + suffix)
    os.makedirs(log_dir, exist_ok=True)
    # telemetry and logs follow the results directory, so a new protocol never overwrites an earlier study's files
    env = dict(os.environ, MAS_TELEMETRY_DIR=os.path.join(OUT_ROOT, 'telemetry' + suffix), OMP_NUM_THREADS=str(threads),
               PYTHONWARNINGS='ignore')
    t0 = time.time()
    with open(os.path.join(log_dir, j['name'] + '.log'), 'w') as log:
        rc = subprocess.call([PY, 'main.py'] + j['args'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    status = 'ok' if rc == 0 and done(j) else f'FAILED rc={rc}'
    dt = time.time() - t0
    print(f'{status:>12}  {dt:7.0f}s  {j["name"]}', flush=True)
    return j['name'], status, dt


def clser_grid(sigma):
    """
    CLS-ER configurations (spec §5.5). With update probability r and decay a, an EMA averages over ~1/(r(1-a)) steps.
    Paper Table S4 MNIST-360 (general CL) and the official repo defaults are included as references; the stream-scaled
    configs set the plastic window to sigma and the stable window to k*sigma, k in {3, 10}.
    """
    def decay(window, r):
        return round(1 - 1 / (window * r), 5)
    cfgs = [('clser', {'reg_weight': 1.25, 'stable_alpha': 0.99, 'plastic_alpha': 0.99,
                       'stable_update_freq': 0.9, 'plastic_update_freq': 1.0}),                 # paper, MNIST-360
            ('clser', {'reg_weight': 0.1, 'stable_alpha': 0.999, 'plastic_alpha': 0.999,
                       'stable_update_freq': 0.7, 'plastic_update_freq': 0.9})]                 # repo defaults
    for reg in (0.1, 1.25):
        for k in (3, 10):
            cfgs.append(('clser', {'reg_weight': reg, 'stable_alpha': decay(k * sigma, 0.9),
                                   'plastic_alpha': decay(sigma, 1.0), 'stable_update_freq': 0.9,
                                   'plastic_update_freq': 1.0}))
    return cfgs


def top_seed0_configs(dataset, regime, method, keys, k):
    """Best k seed-0 configurations for a method, read from the result files' AAUC lines."""
    import glob
    import re
    pat = os.path.join(OUT_ROOT, 'results_t2', f'{dataset}_GCN_{method}_batch10_{regime}*_seed0_tm.txt')
    scored = []
    for f in glob.glob(pat):
        aauc = float(re.search(r'AAUC: ([\d.]+)', open(f).read()).group(1))
        name = os.path.basename(f)
        hp = {}
        for key in sorted(keys, key=len, reverse=True):
            m = re.search(rf'_{key}(-?[\d.]+)(?=_|$)', name)
            if m:
                hp[key] = float(m.group(1))
                name = name[:m.start()] + name[m.end():]
        if len(hp) == len(keys):
            scored.append((aauc, hp))
    scored.sort(key=lambda t: -t[0])
    return [hp for _, hp in scored[:k]]


def plan_jobs(plan, thresholds=None):
    if plan == 'calib_t1':   # detector-off traces, calibration seed 0
        return [job(r, 'tfmas_star', 0, star_args={'l_th': -1.0, 'std_th': -1.0})
                for r in ('clsincre', 'sigma3', 'sigma10', 'sigma20')]
    if plan == 'calib_t3':   # real threshold sweep, from experiments/thresholds_t3.txt ("l_th std_th" per line)
        return [job(r, 'tfmas_star', 0, star_args={'l_th': l, 'std_th': s})
                for r in ('clsincre', 'sigma20') for l, s in thresholds]
    if plan in ('main_core', 'main_rest'):
        (l_th, std_th), = thresholds
        regimes = ('clsincre', 'sigma3', 'sigma10', 'sigma20') if plan == 'main_core' else ('bb5', 'blurry30')
        jobs = []
        for r, seed in itertools.product(regimes, (1, 2, 3)):
            jobs.append(job(r, 'tfmas_star', seed, star_args={'l_th': l_th, 'std_th': std_th}))
            jobs.append(job(r, 'tfmas_star', seed, star_args={'l_th': -1.0, 'std_th': -1.0}))
            jobs.append(job(r, 'tfmas', seed))
            jobs.append(job(r, 'bare', seed))
        return jobs
    if plan == 'gate0_cora':   # spec Gate 0: ER and A-GEM, Gaussian mixing, published sigma (tfmas/bare exist)
        return [job('sigma20', m, s) for m in ('er', 'agem') for s in (1, 2, 3)]
    if plan == 'gate0_arxiv':  # ER, A-GEM, MAS* (legacy tfmas) and Bare on Arxiv-CL at published sigma=60
        return [job('sigma60', m, s, dataset='Arxiv-CL') for m in ('er', 'agem', 'tfmas', 'bare') for s in (1, 2, 3)]
    if plan == 'base_t2':      # baselines spec: existing DRIFT baselines + new methods at each dataset's Table-2 sigma
        methods = [(m, None) for m in ('bare', 'er', 'agem', 'tfmas', 'dmsg', 'er_cbrs')]
        return [job(r, m, s, dataset=d, method_args=hp) for d, r in (('CoraFull-CL', 'sigma20'), ('Arxiv-CL', 'sigma60'))
                for m, hp in methods for s in (1, 2, 3)]
    if plan == 'eval_t2':
        """All new methods at seeds 1-3 with the seed-0 selections, plus the ablations the spec requires."""
        import json
        sel = json.load(open(os.path.join(OUT_ROOT, 'experiments', 'selected_hparams.json')))
        jobs = []
        for dataset, regime in (('CoraFull-CL', 'sigma20'), ('Arxiv-CL', 'sigma60')):
            hp = sel[dataset]
            der, derpp, lwf, clser = hp['der'], hp['derpp'], hp['lwf_online'], hp['clser']
            # spec §5.7 names alpha for the label term and beta for the stored logits; DER++'s paper naming is the
            # other way round, so the selected DER++ weights are mapped across here
            combo = {'alpha': derpp['beta'], 'beta': derpp['alpha'], 'gamma': clser['reg_weight'],
                     'ema_alpha': clser['stable_alpha'], 'ema_update_freq': clser['stable_update_freq']}
            arms = [('der', der), ('derpp', derpp), ('lwf_online', lwf), ('clser', clser),
                    ('clser', dict(clser, reg_weight=0.0)),            # EMA without consistency (spec §5.5 ablation)
                    ('dercls', combo),
                    ('dercls', dict(combo, gamma=0.0)),                # beta-only  (spec §5.7 ablation)
                    ('dercls', dict(combo, beta=0.0))]                 # gamma-only (spec §5.7 ablation)
            for method, args_ in arms:
                jobs += [job(regime, method, s_, dataset=dataset, method_args=args_) for s_ in (1, 2, 3)]
            jobs += [job(regime, 'pdgnn', s_, dataset=dataset, backbone='SGC') for s_ in (1, 2, 3)]
        return jobs
    if plan == 'tune_arxiv_small':
        # Arxiv runs ~5x longer than CoraFull, so LwF-online's 27-point grid is not repeated there: its best three
        # CoraFull configurations are carried over. DER, DER++ and CLS-ER keep their full grids (documented in NOTES).
        grid = [('der', {'alpha': a}) for a in (0.5, 1.0)]
        grid += [('derpp', {'alpha': a, 'beta': b}) for a in (0.2, 0.5) for b in (0.5, 1.0)]
        grid += [('lwf_online', hp) for hp in top_seed0_configs(
            'CoraFull-CL', 'gaussian_sigma20.0', 'lwf_online', ['T', 'lambda_dist', 'update_every'], 3)]
        grid += clser_grid(60)
        return [job('sigma60', m, 0, dataset='Arxiv-CL', method_args=hp) for m, hp in grid]
    if plan in ('tune_cora', 'tune_arxiv'):
        dataset, regime, sigma = ('CoraFull-CL', 'sigma20', 20) if plan == 'tune_cora' else ('Arxiv-CL', 'sigma60', 60)
        grid = []
        grid += [('der', {'alpha': a}) for a in (0.5, 1.0)]                                    # DER paper Table 10
        grid += [('derpp', {'alpha': a, 'beta': b}) for a in (0.2, 0.5) for b in (0.5, 1.0)]  # DER paper Table 10
        grid += [('lwf_online', {'lambda_dist': l, 'T': t, 'update_every': u})               # spec §5.4 / OCGL
                 for l in (0.1, 1.0, 10.0) for t in (0.2, 2.0, 20.0) for u in (1, 10, 100)]
        grid += clser_grid(sigma)
        return [job(regime, m, 0, dataset=dataset, method_args=hp) for m, hp in grid]
    if plan == 'gate0_arxiv_timing':
        return [job('sigma60', 'er', 1, dataset='Arxiv-CL')]
    if plan == 'robust':
        return [job(r, 'tfmas_star', 1, star_args={'l_th': l, 'std_th': s})
                for r in ('sigma3', 'sigma10', 'sigma20') for l, s in thresholds]
    raise ValueError(plan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', required=True)
    ap.add_argument('--thresholds', help='file with "l_th std_th" per line')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--threads', type=int, default=2,
                    help='OMP_NUM_THREADS for every run. Results are bit-reproducible only at a fixed thread count '
                         'without oversubscription (workers * threads <= cores); the serial cache-building job uses '
                         'the same value.')
    ap.add_argument('--results', default='results', help='result directory (relative to the repo root)')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    thresholds = None
    if a.thresholds:
        with open(a.thresholds) as f:
            thresholds = [tuple(float(x) for x in line.split()) for line in f if line.strip() and not line.startswith('#')]
    jobs = with_results_dir(plan_jobs(a.plan, thresholds), a.results)
    pending = [j for j in jobs if not done(j)]
    print(f'{len(jobs)} jobs, {len(pending)} pending', flush=True)
    if a.dry_run:
        for j in pending:
            print(' '.join(j['args']))
        return

    # build each regime's data cache serially before parallel work
    first_per_regime = {}
    for j in pending:
        first_per_regime.setdefault(j['regime'], j)
    for j in first_per_regime.values():
        run(j, a.threads)
    rest = [j for j in pending if j not in first_per_regime.values()]

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        results = list(ex.map(lambda j: run(j, a.threads), rest))
    failed = [r for r in results if r[1].startswith('FAILED')]
    print(f'finished: {len(results)} run, {len(failed)} failed', flush=True)
    for r in failed:
        print('  ', r[0])


if __name__ == '__main__':
    main()
