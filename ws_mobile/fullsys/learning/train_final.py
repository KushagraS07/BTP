#!/usr/bin/env python3
"""train_final.py -- C0 (FINDINGS C0): one FINAL model per learned component, trained on the dev sessions only.

Same training functions as LOBO, nothing held out: validation = first dev session (best-val checkpoint), fit = the
other dev sessions. Implemented by calling each module's train_fold with held-out name 'FINAL' while its run loader
returns only dev-session runs plus a relabelled copy of the validation session as a dummy 'FINAL' test set (its
metrics are meaningless and not reported). Output: <models>/{rel,slip,li,rlidar}_FINAL.pt
Usage: OMP_NUM_THREADS=1 python3 learning/train_final.py rel slip [li] [rlidar] [--jobs N]
"""
import argparse
import copy
import multiprocessing as mp
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from zupt import paths as _P  # noqa: E402

DEV_LIST = os.path.join(_P.CONFIG, 'dev_sessions.txt')


def dev_sessions():
    out = []
    for line in open(DEV_LIST):
        line = line.split('#')[0].strip()
        if line:
            out.append(line.split()[0])
    return sorted(out)


def wrap_loader(fn, dev):
    val = dev[0]

    def loader(*a, **k):
        runs = [r for r in fn(*a, **k) if r['session'] in dev]
        fake = []
        for r in runs:
            if r['session'] == val:
                c = copy.copy(r)
                c['session'] = 'FINAL'
                fake.append(c)
        return runs + fake
    return loader


def train(kind):
    dev = dev_sessions()
    sessions = dev + ['FINAL']
    if kind == 'rel':
        import learning.train as M
        M.load_runs = wrap_loader(M.load_runs, dev)
        return M.train_fold(('FINAL', sessions, 15, 5))
    if kind == 'slip':
        import learning.train_slip as M
        M.load_runs = wrap_loader(M.load_runs, dev)
        return M.train_fold(('FINAL', sessions))
    if kind == 'li':
        import learning.train_li as M
        M.load_runs = wrap_loader(M.load_runs, dev)
        return M.train_fold(('FINAL', sessions, _P.DATA))
    if kind == 'rlidar':
        import learning.lidar_rel as M
        M.load_lruns = wrap_loader(M.load_lruns, dev)
        return M.train_fold(('FINAL', sessions))
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kinds', nargs='+', choices=['rel', 'slip', 'li', 'rlidar'])
    ap.add_argument('--jobs', type=int, default=1)
    a = ap.parse_args()
    if os.environ.get('OMP_NUM_THREADS') != '1':
        sys.exit('ABORT: run with OMP_NUM_THREADS=1')
    from learning.faults import SUFFIX
    from learning.runcfg import SEED
    if SUFFIX or SEED != 42:
        sys.exit('ABORT: default fault set and seed 42 required')
    print(f'FINAL models {a.kinds}: dev sessions {dev_sessions()} (val = first)', flush=True)
    if a.jobs <= 1:
        res = [train(k) for k in a.kinds]
    else:
        with mp.get_context('fork').Pool(min(a.jobs, len(a.kinds))) as pool:
            res = pool.map(train, a.kinds)
    for k, r in zip(a.kinds, res):
        print(f'  {k}_FINAL.pt written; best epoch {r.get("best_epoch")}', flush=True)


if __name__ == '__main__':
    main()
