#!/usr/bin/env python3
"""final_eval.py -- FR (FINDINGS FR): confirmatory closed-loop runs on the final-test sessions.

Sessions: config/test_sessions.txt ("<session> <world>"); aborts if any is a dev or calibration session.
Models: rel / slip / li _FINAL.pt. Parameters: results/fullsys_calibration_choice.json (frozen, C1 / C1b).
Sets (--set): main  = clean + 26 F7 faults   (primary methods + ablations)
              sigma = 26 A1 sigma-scaled faults   (run with FULLSYS_FAULT_UNIT=sigma)
              end   = 26 R2 ending faults         (run with FULLSYS_FAULT_WINDOW=end; adds rmse_after)
              lidar = 15 LiDAR faults + 9 overlaps (primary + abl_no_stale + abl_no_lidar)
              unseen= 18 E faults (features from fullsys_logs/features_unseen)
Output: results/final/final_<set>.csv (one row per session x run x method).
Usage: OMP_NUM_THREADS=1 python3 learning/final_eval.py --set main [--jobs N]
"""
import argparse
import contextlib
import copy
import dataclasses
import io
import json
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from learning.cal_eval import CHI2, read_list  # noqa: E402
from learning.closed_loop import DET, EKF9, FEAT_DIR, MODEL_DIR, RES, predict_r  # noqa: E402
from learning.closed_loop_lidar import SEG  # noqa: E402
from zupt import paths as _P  # noqa: E402

TEST_LIST = os.path.join(_P.CONFIG, 'test_sessions.txt')
DEV_LIST = os.path.join(_P.CONFIG, 'dev_sessions.txt')
CAL_LIST = os.path.join(_P.CONFIG, 'cal_sessions.txt')
CHOICE = os.path.join(RES, 'fullsys_calibration_choice.json')
OUT_DIR = os.path.join(RES, 'final')
PRIMARY = ['hybrid', 'fixed', 'chi2', 'sage_husa', 'li']
ABLATIONS = ['abl_no_rel', 'abl_no_pslip', 'abl_no_chi2', 'abl_no_stale', 'abl_no_5c', 'abl_no_lidar']
SET_METHODS = {'main': PRIMARY + ABLATIONS, 'sigma': PRIMARY, 'end': PRIMARY,
               'lidar': PRIMARY + ['abl_no_stale', 'abl_no_lidar'], 'unseen': PRIMARY}
SET_ENV = {'main': ('std', 'open'), 'sigma': ('sigma', 'open'), 'end': ('std', 'end'),
           'lidar': ('std', 'open'), 'unseen': ('std', 'open')}
KEEP = ['rmse', 'max_err', 'final_err', 'head_rmse_deg', 'rpe5s', 'diverged', 'nees_in95', 'odom_skip', 'gyro_skip']


def two_pass(mod, s2, odom, sc, cfg, p9, seg, two=True, **kw):
    """Same as baselines.with_5c / li_baseline.with_5c, but also returns the trajectory."""
    import learning.baselines as B
    from zupt.evaluation.eval_5c import mask_to_odom
    from zupt.segments import run_segments
    p1 = mod.run(s2, odom, sc, cfg, p9, **kw)
    o = p1
    if two:
        segs = run_segments(seg, p1['t'], p1['hx'], p1['gz'], p1['st'], odom.t.values, odom.linear_x.values)
        g = mask_to_odom(segs, odom.t.values)
        if g.any():
            o = mod.run(s2, odom, sc, cfg, p9, gate5c=g, **kw)
    return o, B.metrics(o, s2[3])


def run_method(meth, B, L, s2, odom, sc, cfg0, p90, seg, pre, li_pr, fz):
    """fz = frozen JSON. Sets every module-level parameter explicitly before each run."""
    B.GATE = fz['frozen']['gate']
    B.CHI2_99 = dict(CHI2[fz['frozen']['chi2']])
    B.SH_B = fz['baselines']['sage_husa']['b']
    L.GS, L.GT = fz['baselines']['li']['gs'], fz['baselines']['li']['gt']
    one = np.ones(len(s2[0]))
    if meth == 'fixed':
        return two_pass(B, s2, odom, sc, cfg0, p90, seg, r_imu=one, r_odom_imu=one)
    if meth == 'chi2':
        B.CHI2_99 = dict(CHI2[fz['baselines']['chi2']['chi2']])
        return two_pass(B, s2, odom, sc, cfg0, p90, seg, r_imu=one, r_odom_imu=one, chi2=True)
    if meth == 'sage_husa':
        return two_pass(B, s2, odom, sc, cfg0, p90, seg, r_imu=one, r_odom_imu=one, sage=True)
    if meth == 'li':
        return two_pass(L, s2, odom, sc, cfg0, p90, seg, ps=li_pr[:, 0], pt=li_pr[:, 2])
    v = fz['frozen']
    p9 = dataclasses.replace(p90, nhc_p_skip=v['pslip'], k_w_slip=v['k'])
    cfg = copy.copy(cfg0)
    cfg.p_slip_max = v['pslip']
    ri, ro, ps = pre
    kw = dict(r_imu=ri, r_odom_imu=ro, q=ps, s=ps, chi2=True, stale=True)
    two = True
    if meth == 'abl_no_rel':
        kw.update(r_imu=one, r_odom_imu=one)
    elif meth == 'abl_no_pslip':
        kw.update(q=None, s=None)
    elif meth == 'abl_no_chi2':
        kw.update(chi2=False)
    elif meth == 'abl_no_stale':
        kw.update(stale=False)
    elif meth == 'abl_no_5c':
        two = False
    elif meth == 'abl_no_lidar':
        kw.update(lidar='off')
    elif meth != 'hybrid':
        raise ValueError(meth)
    return two_pass(B, s2, odom, sc, cfg, p9, seg, two=two, **kw)


def jobs_for(setname, h, sess, scans):
    """(run name, faulted session result, scans, feature file name, feature dir, extra manifest)."""
    from learning.faults import all_faults, clean, inject
    imu, odom = sess[0], sess[1]
    t0, t1 = min(imu.t.iloc[0], odom.t.iloc[0]), max(imu.t.iloc[-1], odom.t.iloc[-1])
    out = []
    if setname == 'main':
        out.append((h, clean(sess), scans, h, FEAT_DIR, {}))
    if setname in ('main', 'sigma', 'end'):
        for f in all_faults(h):
            out.append((f.name, inject(sess, f), scans, f.name, FEAT_DIR, {}))
    if setname == 'lidar':
        from learning.lidar_faults import OVERLAP_L, OVERLAP_O, LFault, all_lfaults, inject_scans
        for lf in all_lfaults(h):
            sc, man, _ = inject_scans(scans, lf, t0, t1)
            out.append((lf.name, clean(sess), sc, h, FEAT_DIR, dict(man, lset='lidar')))
        of = {(f.sensor, f.ftype, f.severity): f for f in all_faults(h)}
        for lt, ls in OVERLAP_L:
            sc, man, _ = inject_scans(scans, LFault(h, lt, ls), t0, t1)
            for key in OVERLAP_O:
                fo = of[key]
                out.append((f'{h}_ovl_{lt}_{ls}__{"_".join(key)}', inject(sess, fo), sc, fo.name, FEAT_DIR,
                            dict(man, lset='overlap', osensor=key[0], otype=key[1], oseverity=key[2])))
    if setname == 'unseen':
        from learning.e_build import UFEAT_DIR
        from learning.unseen_faults import all_unseen, inject_unseen
        for u in all_unseen(h):
            out.append((u.name, inject_unseen(sess, u), scans, u.name, UFEAT_DIR, {}))
    return out


def work(args):
    setname, h, data, fz = args
    torch.set_num_threads(1)
    import learning.baselines as B
    import learning.li_baseline as L
    from zupt.core.config import load_config
    from zupt.ekf.ekf9 import load_ekf9_params
    from zupt.replay import load_scans, load_session
    from zupt.segments import load_segment_config
    from learning.features import FEATURES
    from learning.train_li import INPUTS, predict_probs
    from learning.train_slip import predict_p
    ld = lambda k: torch.load(os.path.join(MODEL_DIR, f'{k}_FINAL.pt'), map_location='cpu', weights_only=False)
    ck, cs, cli = ld('rel'), ld('slip'), ld('li')
    with contextlib.redirect_stdout(io.StringIO()):
        sess = load_session(data, h)
    scans = load_scans(data, h)
    if scans is None:
        raise SystemExit(f'ABORT: no scan file for {h}')
    cfg0, p90, seg = load_config(DET), load_ekf9_params(EKF9), load_segment_config(SEG)
    rows = []
    for run, fr, sc, feat, fdir, extra in jobs_for(setname, h, sess, scans):
        z = np.load(os.path.join(fdir, f'{feat}.npz'))
        cols, d = list(z['columns']), z['data']
        Xf = d[:, [cols.index(c) for c in FEATURES]].astype(np.float32)
        rp, ps = predict_r(ck, Xf), predict_p(cs, Xf)
        li_pr = predict_probs(cli, d[:, [cols.index(c) for c in INPUTS]].astype(np.float32))
        s2 = (fr.imu, fr.odom, sess[2], sess[3])
        m = fr.manifest
        base = dict(model=h, set=setname, run=run, type=m.get('type'), sensor=m.get('sensor'),
                    severity=m.get('severity'), onset=m.get('onset', np.nan), fault_end=m.get('fault_end', np.nan),
                    **extra)
        if setname == 'lidar':
            base['type'] = extra['lset']
        ti, gt = fr.imu.t.values, sess[3]
        gx, gy = np.interp(ti, gt.t.values, gt.gt_x.values), np.interp(ti, gt.t.values, gt.gt_y.values)
        for meth in SET_METHODS[setname]:
            o, met = run_method(meth, B, L, s2, fr.odom, sc, cfg0, p90, seg, (rp[:, 0], rp[:, 1], ps), li_pr, fz)
            res = dict(base, method=meth, **{k: met[k] for k in KEEP})
            if setname == 'end':
                fend = base['fault_end']
                e = np.hypot(o['X'] - gx, o['Y'] - gy)
                aft = ti >= fend if np.isfinite(fend) else np.zeros(len(ti), bool)
                res['rmse_after'] = float(np.sqrt(np.mean(e[aft] ** 2))) if aft.any() else np.nan
            rows.append(res)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', required=True, choices=list(SET_METHODS))
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--data', default=_P.DATA)
    ap.add_argument('--list', default=TEST_LIST)
    a = ap.parse_args()
    if os.environ.get('OMP_NUM_THREADS') != '1':
        sys.exit('ABORT: run with OMP_NUM_THREADS=1')
    from learning.faults import UNIT, WINDOW
    from learning.runcfg import SEED
    if (UNIT, WINDOW) != SET_ENV[a.set] or SEED != 42:
        sys.exit(f'ABORT: set {a.set} needs fault unit/window {SET_ENV[a.set]} and seed 42 '
                 f'(got {UNIT}/{WINDOW}, seed {SEED})')
    hs = read_list(a.list)
    used = set(read_list(DEV_LIST)) | (set(read_list(CAL_LIST)) if os.path.exists(CAL_LIST) else set())
    leak = sorted(set(hs) & used)
    if leak:
        sys.exit(f'ABORT: test sessions also used for development / calibration: {leak}')
    fz = json.load(open(CHOICE))
    print(f'FR {a.set}: {len(hs)} sessions; methods {SET_METHODS[a.set]}; frozen {fz["frozen"]}; '
          f'baselines {fz["baselines"]}', flush=True)
    tasks = [(a.set, h, a.data, fz) for h in hs]
    if a.jobs <= 1:
        out = [work(t) for t in tasks]
    else:
        with mp.get_context('fork').Pool(a.jobs) as pool:
            out = pool.map(work, tasks)
    df = pd.DataFrame([r for o in out for r in o])
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f'final_{a.set}.csv')
    df.to_csv(path, index=False)
    print(f'wrote {len(df)} rows -> {path}', flush=True)


if __name__ == '__main__':
    main()
