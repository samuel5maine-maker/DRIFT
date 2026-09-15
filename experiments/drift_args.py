"""Parse a DRIFT command line with main.py's own argparse block, so experiment tooling cannot drift from main.py."""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main_args(argv):
    src = open(os.path.join(ROOT, 'main.py'), encoding='utf-8').read()
    block = src[src.index("    parser = argparse.ArgumentParser"):src.index("    args = parser.parse_args()")]
    block = re.sub(r'(?m)^    ', '', block)
    ns = {}
    exec('import argparse\nfrom distutils.util import strtobool\nfrom training.utils import str2dict\n' + block, ns)
    args = ns['parser'].parse_args(argv)
    args.ratio_valid_test = [float(i) for i in args.ratio_valid_test]
    args.cuda = args.cuda == 'yes'
    return args
