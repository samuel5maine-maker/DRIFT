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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


DATASET_FLAGS = {'Arxiv-CL': ['--ori_data_path', os.path.join(ROOT, 'data', 'raw')]}


def job(regime, method, seed, dataset='CoraFull-CL', backbone='GCN', star_args=None):
    cmd = ['--dataset', dataset, '--backbone', backbone, '--method', method, '--seed', str(seed), '--cuda', 'no']
    cmd += DATASET_FLAGS.get(dataset, [])
    cmd += REGIMES[regime]
    if star_args is not None:
        cmd += ['--tfmas_star_args', ';'.join(f"'{k}':{v}" for k, v in star_args.items())]
    name = f'{dataset}_{backbone}_{regime}_{method}'
    if star_args is not None:
        name += '_' + '_'.join(f'{k}{v}' for k, v in star_args.items())
    name += f'_seed{seed}'
    return {'name': name, 'regime': regime, 'args': cmd}


def result_prefix(j):
    """Mirror of main.py's result naming, to detect finished jobs."""
    a = dict(zip(j['args'][::2], j['args'][1::2]))  # every flag takes exactly one value
    s = f"{a['--dataset']}_{a['--backbone']}_{a['--method']}_batch10"
    setting = a['--setting']
    if setting == 'tfo_blurry':
        s += f"_blurry{int(round((1.0 - float(a['--percentage'])) * 100))}"
    elif setting == 'tfo_bb':
        s += f"_boundaryblurry_K{a['--blurry_batch_count']}_ratio50"
    elif setting == 'tfo_gaussian':
        s += f"_gaussian_sigma{float(a['--gaussian_sigma'])}"
    elif setting == 'tfocis':
        s += '_clsincre'
    if a['--method'] == 'tfmas_star':
        hp = dict(kv.replace("'", '').split(':') for kv in a['--tfmas_star_args'].split(';'))
        s += f"_lth{float(hp['l_th'])}_sth{float(hp['std_th'])}"
        for k in ('window', 'buffer_size', 'lam', 'passes', 'window_push'):
            if k in hp:
                v = hp[k]
                try:
                    v = float(v)
                except ValueError:
                    pass
                s += f'_{k}{v}'
    s += f"_seed{a['--seed']}"
    return os.path.join(ROOT, 'results', s)


def done(j):
    p = result_prefix(j)
    return os.path.exists(p + '_tm.txt') or os.path.exists(p + '_cfmat.txt')


def run(j, threads):
    if done(j):
        return j['name'], 'skipped', 0.0
    os.makedirs(os.path.join(ROOT, 'experiments', 'logs'), exist_ok=True)
    env = dict(os.environ, MAS_TELEMETRY_DIR=os.path.join(ROOT, 'telemetry'), OMP_NUM_THREADS=str(threads),
               PYTHONWARNINGS='ignore')
    t0 = time.time()
    with open(os.path.join(ROOT, 'experiments', 'logs', j['name'] + '.log'), 'w') as log:
        rc = subprocess.call([PY, 'main.py'] + j['args'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    status = 'ok' if rc == 0 and done(j) else f'FAILED rc={rc}'
    dt = time.time() - t0
    print(f'{status:>12}  {dt:7.0f}s  {j["name"]}', flush=True)
    return j['name'], status, dt


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
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    thresholds = None
    if a.thresholds:
        with open(a.thresholds) as f:
            thresholds = [tuple(float(x) for x in line.split()) for line in f if line.strip() and not line.startswith('#')]
    jobs = plan_jobs(a.plan, thresholds)
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
    threads_serial = os.cpu_count() or 8
    for j in first_per_regime.values():
        run(j, threads_serial)
    rest = [j for j in pending if j not in first_per_regime.values()]

    threads = max(1, (os.cpu_count() or 8) // a.workers)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        results = list(ex.map(lambda j: run(j, threads), rest))
    failed = [r for r in results if r[1].startswith('FAILED')]
    print(f'finished: {len(results)} run, {len(failed)} failed', flush=True)
    for r in failed:
        print('  ', r[0])


if __name__ == '__main__':
    main()
