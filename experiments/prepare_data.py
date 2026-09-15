"""
Builds DRIFT's cached task/stream pickles for each regime with a fixed seed, without training.

DRIFT's cache names omit the seed (pipeline.py data_prepare_* functions), so whichever run builds a
cache first fixes the stream for every later seed. Building them here with --seed 1 makes that explicit.

    python experiments/prepare_data.py --seed 1
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from run_matrix import REGIMES  # noqa: E402
from drift_args import main_args  # noqa: E402


def main():
    seed = sys.argv[sys.argv.index('--seed') + 1] if '--seed' in sys.argv else '1'
    dataset_name = sys.argv[sys.argv.index('--dataset') + 1] if '--dataset' in sys.argv else 'CoraFull-CL'
    from training.utils import set_seed, mkdir_if_missing
    from dataset.utils import NodeLevelDataset
    from pipeline import data_prepare, data_prepare_blurry, data_prepare_boundaryblurry, data_prepare_gaussian

    dataset = None
    for regime, flags in REGIMES.items():
        args = main_args(['--dataset', dataset_name, '--seed', seed, '--cuda', 'no'] + flags)
        set_seed(args)
        mkdir_if_missing(args.data_path)
        if dataset is None:
            dataset = NodeLevelDataset(args.dataset, ratio_valid_test=args.ratio_valid_test, args=args)
        prep = {'tfo_blurry': data_prepare_blurry, 'tfo_bb': data_prepare_boundaryblurry,
                'tfo_gaussian': data_prepare_gaussian}.get(args.setting, data_prepare)
        print(f'[prepare] {regime}', flush=True)
        prep(dataset, args)


if __name__ == '__main__':
    main()
