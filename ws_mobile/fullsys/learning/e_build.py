#!/usr/bin/env python3
"""
e_build.py -- E (FINDINGS_fullsys.md E): features for the unseen fault set, same pipeline as build_dataset.py.

Sessions: --list (default config/test_sessions.txt, "<session> <world>"); every one must pass fs_check (KEEP).
Output: <ws_mobile>/fullsys_logs/features_unseen/<name>.npz (separate folder: never seen by training loaders)
        and summary.csv. Prints an injection check per fault (share of samples changed after onset, mean error
in std) -- no method is evaluated here.
Usage: OMP_NUM_THREADS=1 python3 learning/e_build.py [--jobs N] [--list FILE] [--only SESSION ...]
"""
import argparse
import contextlib
import glob
import io
import json
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from zupt import paths as _P  # noqa: E402
from learning.build_dataset import DET, EKF9, keep, read_list  # noqa: E402

UFEAT_DIR = os.path.join(_P.WS, 'fullsys_logs', 'features_unseen')
TEST_LIST = os.path.join(_P.CONFIG, 'test_sessions.txt')


def work(args):
    session, data, name = args
    from zupt.core.config import load_config
    from zupt.ekf.ekf9 import load_ekf9_params
    from zupt.replay import load_session
    from learning.features import build, slip_label
    from learning.unseen_faults import all_unseen, inject_unseen
    with contextlib.redirect_stdout(io.StringIO()):
        sess = load_session(data, session)
    cfg, p9 = load_config(DET), load_ekf9_params(EKF9)
    gt = sess[3]
    f = {u.name: u for u in all_unseen(session)}[name]
    fr = inject_unseen(sess, f)
    d = build(fr, sess[2], gt, cfg, p9)
    d['slip'] = slip_label(d.t.values, sess[1], gt)
    d['r_odom'] = np.minimum(d.r_odom.values, 1.0 - d.slip.values)
    gx, gy = np.interp(d.t, gt.t, gt.gt_x), np.interp(d.t, gt.t, gt.gt_y)
    rmse = float(np.sqrt(np.mean((d.ekf_x - gx) ** 2 + (d.ekf_y - gy) ** 2)))
    man = dict(fr.manifest, session=session, name=name, rmse_fixed_ekf9=rmse)
    np.savez_compressed(os.path.join(UFEAT_DIR, f'{name}.npz'), columns=np.array(list(d.columns), dtype=str),
                        data=d.values.astype(np.float64), manifest=json.dumps(man))
    return dict(name=name, session=session, type=man['type'], severity=man['severity'],
                affected=man['affected'], mean_err_std=man['mean_err_std'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--data', default=_P.DATA)
    ap.add_argument('--list', default=TEST_LIST)
    ap.add_argument('--only', nargs='*')
    a = ap.parse_args()
    if os.environ.get('OMP_NUM_THREADS') != '1':
        sys.exit('ABORT: run with OMP_NUM_THREADS=1')
    from learning.unseen_faults import all_unseen
    sessions = read_list(a.list)
    if a.only:
        sessions = [x for x in sessions if x[0] in a.only]
    for s, w in sessions:
        ok, txt = keep(s, w, a.data)
        if not ok:
            sys.exit(f'ABORT: {s} does not pass fs_check (F5/F5a):\n{txt}')
    os.makedirs(UFEAT_DIR, exist_ok=True)
    tasks = [(s, a.data, u.name) for s, _ in sessions for u in all_unseen(s)]
    print(f'E: {len(sessions)} sessions, {len(tasks)} unseen-fault runs -> {UFEAT_DIR}', flush=True)
    if a.jobs <= 1:
        rows = [work(t) for t in tasks]
    else:
        with mp.get_context('fork').Pool(a.jobs) as pool:
            rows = pool.map(work, tasks)
    allrows = []
    for p in sorted(glob.glob(os.path.join(UFEAT_DIR, '*.npz'))):
        m = json.loads(str(np.load(p)['manifest']))
        allrows.append(dict(name=m['name'], session=m['session'], type=m['type'], severity=m['severity'],
                            affected=m['affected'], mean_err_std=m['mean_err_std']))
    pd.DataFrame(allrows).to_csv(os.path.join(UFEAT_DIR, 'summary.csv'), index=False)
    df = pd.DataFrame(rows)
    pd.set_option('display.width', 160)
    print('\ninjection check (share of samples changed after onset, mean |error| in std) -- no method evaluated:')
    print(df.groupby(['type', 'severity'])[['affected', 'mean_err_std']].mean().round(3).to_string())


if __name__ == '__main__':
    main()
