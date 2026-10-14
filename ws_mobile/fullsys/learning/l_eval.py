#!/usr/bin/env python3
"""l_eval.py -- L step (FINDINGS L): LiDAR faults, chi2 gate on the scan-match update, overlapping faults.
Sets per session: clean (1), LiDAR faults (15), overlap LiDAR x odom/IMU (9), and the 26 F14 faults for the
no-harm check of hybrid_chi2L. Writes results/fullsys_lidar_faults.csv and applies the pre-stated rules.
Usage: OMP_NUM_THREADS=1 python3 learning/l_eval.py [--jobs N]"""
import argparse
import contextlib
import glob
import io
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from learning.closed_loop import DET, EKF9, FEAT_DIR, MODEL_DIR, RES, predict_r  # noqa: E402
from learning.closed_loop_lidar import SEG  # noqa: E402
from learning.faults import SUFFIX  # noqa: E402
from zupt import paths as _P  # noqa: E402

METHODS = ['fixed', 'hybrid', 'hybrid_chi2L', 'hybrid_stale', 'hybrid_chi2L_stale', 'lidar_off', 'oracle_L']


def work(args):
    h, data = args
    torch.set_num_threads(1)
    import learning.baselines as B
    from zupt.core.config import load_config
    from zupt.ekf.ekf9 import load_ekf9_params
    from zupt.replay import load_scans, load_session
    from zupt.segments import load_segment_config
    from learning.faults import Fault, all_faults, clean, inject
    from learning.features import FEATURES
    from learning.lidar_faults import OVERLAP_L, OVERLAP_O, LFault, all_lfaults, inject_scans
    from learning.train_slip import predict_p
    ck = torch.load(os.path.join(MODEL_DIR, f'rel_{h}.pt'), map_location='cpu', weights_only=False)
    cs = torch.load(os.path.join(MODEL_DIR, f'slip_{h}.pt'), map_location='cpu', weights_only=False)
    with contextlib.redirect_stdout(io.StringIO()):
        sess = load_session(data, h)
    scans = load_scans(data, h)
    cfg, p9, seg = load_config(DET), load_ekf9_params(EKF9), load_segment_config(SEG)
    imu, odom = sess[0], sess[1]
    t0, t1 = min(imu.t.iloc[0], odom.t.iloc[0]), max(imu.t.iloc[-1], odom.t.iloc[-1])
    cache = {}

    def preds(feat_name):
        if feat_name not in cache:
            z = np.load(os.path.join(FEAT_DIR, f'{feat_name}.npz'))
            cols, d = list(z['columns']), z['data']
            Xf = d[:, [cols.index(c) for c in FEATURES]].astype(np.float32)
            rp = predict_r(ck, Xf)
            cache[feat_name] = (rp[:, 0], rp[:, 1], predict_p(cs, Xf))
        return cache[feat_name]

    def specs(feat_name, active):
        ri, ro, ps = preds(feat_name)
        one = np.ones(len(ri))
        hy = dict(r_imu=ri, r_odom_imu=ro, q=ps, s=ps, chi2=True)
        return dict(fixed=dict(r_imu=one, r_odom_imu=one), hybrid=hy, hybrid_chi2L=dict(hy, lidar='chi2'),
                    hybrid_stale=dict(hy, stale=True), hybrid_chi2L_stale=dict(hy, lidar='chi2', stale=True),
                    lidar_off=dict(hy, lidar='off'), oracle_L=dict(hy, scan_skip=active))

    rows = []
    no_active = np.zeros(len(scans), bool)
    jobs = [('clean', h, clean(sess), scans, h, no_active, {}, METHODS)]
    for lf in all_lfaults(h):
        sc, man, act = inject_scans(scans, lf, t0, t1)
        jobs.append(('lidar', lf.name, clean(sess), sc, h, act, man, METHODS))
    ofaults = {(f.sensor, f.ftype, f.severity): f for f in all_faults(h)}
    for lt, ls in OVERLAP_L:
        sc, man, act = inject_scans(scans, LFault(h, lt, ls), t0, t1)
        for key in OVERLAP_O:
            fo = ofaults[key]
            m2 = dict(man, osensor=key[0], otype=key[1], oseverity=key[2])
            jobs.append(('overlap', f'{h}_ovl_{lt}_{ls}__{key[0]}_{key[1]}_{key[2]}', inject(sess, fo), sc, fo.name,
                         act, m2, METHODS))
    for fo in all_faults(h):
        jobs.append(('f14', fo.name, inject(sess, fo), scans, fo.name, no_active, {}, ['hybrid_chi2L']))
    for setname, run, fr, sc, feat, act, man, meths in jobs:
        s2 = (fr.imu, fr.odom, sess[2], sess[3])
        sp = specs(feat, act)
        for m in meths:
            res = dict(model=h, set=setname, run=run, method=m, **man)
            res.update(B.with_5c(s2, fr.odom, sc, cfg, p9, seg, **sp[m]))
            rows.append(res)
    return rows


def med_change(a, b):
    return 100 * float(np.median(a / b - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--data', default=_P.DATA)
    a = ap.parse_args()
    if os.environ.get('OMP_NUM_THREADS') != '1':
        sys.exit('ABORT: run with OMP_NUM_THREADS=1')
    if SUFFIX:
        sys.exit('ABORT: run with the default fault set (no FULLSYS_FAULT_UNIT / FULLSYS_FAULT_WINDOW)')
    hs = sorted(os.path.basename(p)[5:-3] for p in glob.glob(os.path.join(MODEL_DIR, 'slip_*.pt')))
    print(f'L: {len(hs)} sessions, methods {METHODS}', flush=True)
    tasks = [(h, a.data) for h in hs]
    if a.jobs <= 1:
        out = [work(t) for t in tasks]
    else:
        with mp.get_context('fork').Pool(a.jobs) as pool:
            out = pool.map(work, tasks)
    df = pd.DataFrame([r for o in out for r in o])
    os.makedirs(RES, exist_ok=True)
    df.to_csv(os.path.join(RES, 'fullsys_lidar_faults.csv'), index=False)
    pd.set_option('display.width', 220)
    ref = os.path.join(RES, 'fullsys_baselines.csv')
    b = pd.read_csv(ref) if os.path.exists(ref) else None
    if b is not None:
        mm = df[df.set == 'clean'].merge(b[['run', 'method', 'rmse']], on=['run', 'method'], suffixes=('', '_f14'))
        print(f'regression clean fixed/hybrid vs F14: max |diff| = {np.abs(mm.rmse - mm.rmse_f14).max():.2e} m '
              f'over {len(mm)} rows')
    print('\nDEVELOPMENT L means (rmse / max_err / diverged) by set and method:')
    print(df.groupby(['set', 'method'])[['rmse', 'max_err', 'diverged']].mean().round(3).to_string())
    L = df[df.set == 'lidar']
    print('\nLiDAR-fault runs, mean RMSE by type / severity:')
    print(L.pivot_table(index=['ltype', 'lseverity'], columns='method', values='rmse').round(3)[METHODS].to_string())
    O = df[df.set == 'overlap']
    print('\nOverlap runs, mean RMSE by combination:')
    print(O.pivot_table(index=['ltype', 'osensor', 'otype'], columns='method', values='rmse').round(3)[METHODS]
          .to_string())
    pv = L.pivot_table(index='run', columns='method', values='rmse')
    pc = df[df.set == 'clean'].pivot_table(index='run', columns='method', values='rmse')
    fm = np.nan
    if b is not None:
        f = df[df.set == 'f14'][['run', 'rmse']].merge(b[b.method == 'hybrid'][['run', 'rmse']], on='run',
                                                         suffixes=('', '_h'))
        fm = med_change(f.rmse, f.rmse_h)
    w = int((pv.hybrid_chi2L < pv.hybrid).sum())
    lm, cm = med_change(pv.hybrid_chi2L, pv.hybrid), med_change(pc.hybrid_chi2L, pc.hybrid)
    ok1 = w >= int(np.ceil(0.6 * len(pv))) and lm <= -3 and cm <= 2 and fm <= 2
    print(f'\nRule (1a) hybrid_chi2L vs hybrid: LiDAR faults wins {w}/{len(pv)}, median {lm:+.1f} %; clean median '
          f'{cm:+.1f} %; F14 faults median {fm:+.1f} % -> {"ADOPT" if ok1 else "not adopted"}')
    fz = L[L.ltype == 'freeze'].pivot_table(index='run', columns='method', values='rmse')
    ws = int((fz.hybrid_stale < fz.hybrid).sum())
    sm, cs = med_change(fz.hybrid_stale, fz.hybrid), med_change(pc.hybrid_stale, pc.hybrid)
    nf = L[L.ltype != 'freeze'].pivot_table(index='run', columns='method', values='rmse')
    ns = med_change(nf.hybrid_stale, nf.hybrid)
    ok1b = ws >= int(np.ceil(0.6 * len(fz))) and sm <= -3 and cs <= 2 and ns <= 2
    print(f'Rule (1b) hybrid_stale vs hybrid: freeze runs wins {ws}/{len(fz)}, median {sm:+.1f} %; clean median '
          f'{cs:+.1f} %; other LiDAR faults median {ns:+.1f} % -> {"ADOPT" if ok1b else "not adopted"}')
    D = {(True, True): 'hybrid_chi2L_stale', (True, False): 'hybrid_chi2L', (False, True): 'hybrid_stale',
         (False, False): 'hybrid'}[(ok1, ok1b)]
    om = med_change(pv.oracle_L, pv[D])
    worse = float((pv[D] > pv.lidar_off).mean())
    need = om <= -10 or worse >= 0.25
    print(f'Rule (2) design D = {D}: oracle_L vs D median {om:+.1f} % (<= -10 %?); D worse than lidar_off on '
          f'{100 * worse:.0f} % of LiDAR-fault runs (>= 25 %?) -> {"BUILD learned r_LiDAR" if need else "r_LiDAR NOT needed"}')


if __name__ == '__main__':
    main()
