#!/usr/bin/env python3
"""cal_eval.py -- C1 (FINDINGS C1): calibration of the remaining parameters on the calibration sessions.

Sessions: config/cal_sessions.txt ("<session> <world>"). Models: *_FINAL.pt. Filter = design D (hybrid + stale-scan
check [+ r_LiDAR with --rlidar]). Runs per session: clean + 26 F7 faults [+ 15 LiDAR faults + 9 overlaps with
--rlidar]. Candidates per group (current first), one group changed at a time; then the combination of chosen values.
C1b (FINDINGS C1b): no-harm guard (clean median <= +2 %, diverged count not higher) and the baselines (chi2,
Sage-Husa, Li) calibrated with the same procedure, each against its own current value.
Writes results/fullsys_calibration.csv and results/fullsys_calibration_choice.json (frozen values).
Usage: OMP_NUM_THREADS=1 python3 learning/cal_eval.py [--jobs N] [--rlidar]
"""
import argparse
import contextlib
import copy
import dataclasses
import io
import json
import math
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

CAL_LIST = os.path.join(_P.CONFIG, 'cal_sessions.txt')
CUR = dict(gate=0.5, chi2='both99', pslip=0.5, k=50.0, rl_gate=0.5)
CHI2 = {'both99': {1: 6.635, 2: 9.210, 3: 11.345}, 'gyro99': {1: 6.635, 2: math.inf, 3: 11.345},
        'both95': {1: 3.841, 2: 5.991, 3: 7.815},
        'both999': {1: 10.828, 2: 13.816, 3: 16.266}, 'gyro999': {1: 10.828, 2: math.inf, 3: 16.266}}
GROUPS = {'G1_gate': ('gate', [0.3, 0.7]), 'G2_chi2': ('chi2', ['gyro99', 'both999', 'gyro999']),
          'G3_pslip': ('pslip', [0.3, 0.7]), 'G4_k': ('k', [0.0, 100.0]), 'G5_rl_gate': ('rl_gate', [0.3, 0.7])}
THRESH = -2.0     # % median change needed to move away from the current value
CLEAN_MAX = 2.0   # C1b guard: % median change allowed on clean runs
# C1b baselines: method -> (current values, groups)
BCUR = {'chi2': dict(chi2='both99'), 'sage_husa': dict(b=0.98), 'li': dict(gs=0.7, gt=0.6)}
BGROUPS = {'chi2': {'B_chi2': ('chi2', ['both95', 'both999'])},
           'sage_husa': {'B_sh_b': ('b', [0.95, 0.99])},
           'li': {'L1_gs': ('gs', [0.6, 0.8]), 'L2_gt': ('gt', [0.5, 0.7])}}


def read_list(p):
    out = []
    for line in open(p):
        line = line.split('#')[0].strip()
        if line:
            out.append(line.split()[0])
    return out


def ours_groups(rlidar):
    return {g: v for g, v in GROUPS.items() if rlidar or g != 'G5_rl_gate'}


def all_specs(rlidar):
    """method -> (current dict, groups dict)."""
    out = {'hybrid': (CUR, ours_groups(rlidar))}
    out.update({m: (BCUR[m], BGROUPS[m]) for m in BCUR})
    return out


def variants(cur, groups):
    v = {'current': dict(cur)}
    for g, (key, vals) in groups.items():
        for x in vals:
            v[f'{g}={x}'] = dict(cur, **{key: x})
    return v


def run_baseline(B, L, meth, s2, odom, sc, cfg0, p90, seg, li_pr, var):
    B.GATE = CUR['gate']
    B.CHI2_99 = dict(CHI2[CUR['chi2']])
    B.SH_B = 0.98
    L.GS, L.GT = 0.7, 0.6
    one = np.ones(len(s2[0]))
    if meth == 'chi2':
        B.CHI2_99 = dict(CHI2[var['chi2']])
        return B.with_5c(s2, odom, sc, cfg0, p90, seg, r_imu=one, r_odom_imu=one, chi2=True)
    if meth == 'sage_husa':
        B.SH_B = var['b']
        return B.with_5c(s2, odom, sc, cfg0, p90, seg, r_imu=one, r_odom_imu=one, sage=True)
    if meth == 'li':
        L.GS, L.GT = var['gs'], var['gt']
        return L.with_5c(s2, odom, sc, cfg0, p90, seg, ps=li_pr[:, 0], pt=li_pr[:, 2])
    raise ValueError(meth)


def run_variant(B, s2, odom, sc, cfg0, p90, seg, pre, var, rlidar):
    B.GATE = var['gate']
    B.CHI2_99 = dict(CHI2[var['chi2']])
    p9 = dataclasses.replace(p90, nhc_p_skip=var['pslip'], k_w_slip=var['k'])
    cfg = copy.copy(cfg0)
    cfg.p_slip_max = var['pslip']
    ri, ro, ps, rl = pre
    kw = dict(r_imu=ri, r_odom_imu=ro, q=ps, s=ps, chi2=True, stale=True)
    if rlidar:
        kw['scan_skip'] = rl < var['rl_gate']
    return B.with_5c(s2, odom, sc, cfg, p9, seg, **kw)


def work(args):
    h, data, mvars, rlidar = args
    torch.set_num_threads(1)
    import learning.baselines as B
    import learning.li_baseline as L
    from learning.train_li import INPUTS, predict_probs
    from zupt.core.config import load_config
    from zupt.ekf.ekf9 import load_ekf9_params
    from zupt.replay import load_scans, load_session
    from zupt.segments import load_segment_config
    from learning.faults import all_faults, clean, inject
    from learning.features import FEATURES
    from learning.lidar_faults import OVERLAP_L, OVERLAP_O, LFault, all_lfaults, inject_scans
    from learning.train_slip import predict_p
    ck = torch.load(os.path.join(MODEL_DIR, 'rel_FINAL.pt'), map_location='cpu', weights_only=False)
    cs = torch.load(os.path.join(MODEL_DIR, 'slip_FINAL.pt'), map_location='cpu', weights_only=False)
    cli = torch.load(os.path.join(MODEL_DIR, 'li_FINAL.pt'), map_location='cpu', weights_only=False)
    cl = None
    if rlidar:
        from learning.lidar_rel import predict_rl, scan_features
        cl = torch.load(os.path.join(MODEL_DIR, 'rlidar_FINAL.pt'), map_location='cpu', weights_only=False)
    with contextlib.redirect_stdout(io.StringIO()):
        sess = load_session(data, h)
    scans = load_scans(data, h)
    cfg0, p90, seg = load_config(DET), load_ekf9_params(EKF9), load_segment_config(SEG)
    imu, odom = sess[0], sess[1]
    t0, t1 = min(imu.t.iloc[0], odom.t.iloc[0]), max(imu.t.iloc[-1], odom.t.iloc[-1])
    faults = all_faults(h)
    jobs = [('clean', h, clean(sess), scans, h)] + [('fault', f.name, inject(sess, f), scans, f.name) for f in faults]
    if rlidar:
        for lf in all_lfaults(h):
            sc, man, act = inject_scans(scans, lf, t0, t1)
            jobs.append(('lidar', lf.name, clean(sess), sc, h))
        of = {(f.sensor, f.ftype, f.severity): f for f in faults}
        for lt, ls in OVERLAP_L:
            sc, man, act = inject_scans(scans, LFault(h, lt, ls), t0, t1)
            for key in OVERLAP_O:
                jobs.append(('overlap', f'{h}_ovl_{lt}_{ls}__{"_".join(key)}', inject(sess, of[key]), sc, of[key].name))
    rows = []
    for setname, run, fr, sc, feat in jobs:
        z = np.load(os.path.join(FEAT_DIR, f'{feat}.npz'))
        cols, d = list(z['columns']), z['data']
        Xf = d[:, [cols.index(c) for c in FEATURES]].astype(np.float32)
        rp, ps = predict_r(ck, Xf), predict_p(cs, Xf)
        rl = predict_rl(cl, scan_features(sc, fr.odom, fr.imu)) if rlidar else None
        s2 = (fr.imu, fr.odom, sess[2], sess[3])
        li_pr = None
        for meth, vdict in mvars.items():
            if meth != 'hybrid' and setname not in ('clean', 'fault'):
                continue      # baselines are calibrated on clean + F7 runs only (C1b)
            if meth == 'li' and li_pr is None:
                li_pr = predict_probs(cli, d[:, [cols.index(c) for c in INPUTS]].astype(np.float32))
            for vname, var in vdict.items():
                res = dict(model=h, set=setname, run=run, method=meth, variant=vname)
                if meth == 'hybrid':
                    res.update(run_variant(B, s2, fr.odom, sc, cfg0, p90, seg, (rp[:, 0], rp[:, 1], ps, rl), var,
                                           rlidar))
                else:
                    res.update(run_baseline(B, L, meth, s2, fr.odom, sc, cfg0, p90, seg, li_pr, var))
                rows.append(res)
    return rows


def evaluate(df, meth, vname):
    """C1b: score over all runs, clean-run median change, extra diverged runs, eligibility."""
    d = df[df.method == meth]
    p = d.pivot_table(index='run', columns='variant', values='rmse')
    c = d[d.set == 'clean'].pivot_table(index='run', columns='variant', values='rmse')
    v = d.pivot_table(index='run', columns='variant', values='diverged', aggfunc='max').astype(int)
    sc = 100 * float(np.median(p[vname] / p['current'] - 1))
    cl = 100 * float(np.median(c[vname] / c['current'] - 1)) if len(c) else 0.0
    dv = int(v[vname].sum() - v['current'].sum())
    return dict(score=sc, clean=cl, ddiv=dv, ok=sc <= THRESH and cl <= CLEAN_MAX and dv <= 0)


def fmt(e):
    return f"{e['score']:+.1f} % (clean {e['clean']:+.1f} %, div {e['ddiv']:+d}{'' if e['ok'] else ', not eligible'})"


def choose(df, meth, cur, groups):
    """Per group: lowest-score eligible candidate, else current. Returns (choice, best single (score, key, val))."""
    choice, best = dict(cur), None
    print(f'\n[{meth}] current {cur}')
    for g, (key, vals) in groups.items():
        ev = {x: evaluate(df, meth, f'{g}={x}') for x in vals}
        print(f'  {g}: ' + '  '.join(f'{key}={x}: {fmt(e)}' for x, e in ev.items()), end='')
        ok = {x: e for x, e in ev.items() if e['ok']}
        if ok:
            x = min(ok, key=lambda y: ok[y]['score'])
            choice[key] = x
            if best is None or ok[x]['score'] < best[0]:
                best = (ok[x]['score'], key, x)
        print(f' -> {key} = {choice[key]}')
    return choice, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--data', default=_P.DATA)
    ap.add_argument('--rlidar', action='store_true')
    a = ap.parse_args()
    if os.environ.get('OMP_NUM_THREADS') != '1':
        sys.exit('ABORT: run with OMP_NUM_THREADS=1')
    if SUFFIX:
        sys.exit('ABORT: default fault set required')
    hs = read_list(CAL_LIST)
    specs = all_specs(a.rlidar)
    mvars = {m: variants(c, g) for m, (c, g) in specs.items()}
    print(f'C1 calibration: sessions {hs}; variants ' + ', '.join(f'{m} {len(v)}' for m, v in mvars.items()) +
          f'; r_LiDAR {a.rlidar}', flush=True)

    def pmap(tasks):
        if a.jobs <= 1:
            return [r for t in tasks for r in work(t)]
        with mp.get_context('fork').Pool(a.jobs) as pool:
            return [r for o in pool.map(work, tasks) for r in o]

    df = pd.DataFrame(pmap([(h, a.data, mvars, a.rlidar) for h in hs]))
    pd.set_option('display.width', 200)
    print('\nCALIBRATION scores (median RMSE change vs current over all runs; guard: clean <= +2 %, no extra '
          'diverged runs):')
    chosen = {}
    for m, (cur, groups) in specs.items():
        chosen[m] = choose(df, m, cur, groups)
    combos = {m: {'current': dict(specs[m][0]), 'combined': ch} for m, (ch, _) in chosen.items()
              if sum(ch[k] != specs[m][0][k] for k in ch) >= 2}
    ecomb = {}
    if combos:
        df2 = pd.DataFrame(pmap([(h, a.data, combos, a.rlidar) for h in hs]))
        for m in combos:
            ecomb[m] = evaluate(df2, m, 'combined')
        df = pd.concat([df, df2[df2.variant == 'combined']])
    frozen = {}
    for m, (cur, _) in specs.items():
        ch, best = chosen[m]
        if m in ecomb:
            e = ecomb[m]
            print(f'  [{m}] combined {ch}: {fmt(e)} vs best single {best[0]:+.1f} %')
            frozen[m] = ch if e['ok'] and e['score'] <= best[0] else dict(cur, **{best[1]: best[2]})
        else:
            frozen[m] = ch
    os.makedirs(RES, exist_ok=True)
    df.to_csv(os.path.join(RES, 'fullsys_calibration.csv'), index=False)
    json.dump(dict(frozen=frozen['hybrid'], current=CUR, baselines={m: frozen[m] for m in BCUR},
                   baselines_current=BCUR, rlidar=a.rlidar, sessions=hs),
              open(os.path.join(RES, 'fullsys_calibration_choice.json'), 'w'), indent=2)
    print('\nFROZEN parameters:')
    for m, v in frozen.items():
        print(f'  {m}: {v}')


if __name__ == '__main__':
    main()
