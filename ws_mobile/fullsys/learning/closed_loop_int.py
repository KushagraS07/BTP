#!/usr/bin/env python3
"""closed_loop_int.py -- F11 integration check (development): reliability r and P_slip in EKF9.

Per held-out session h (models rel_<h>.pt and slip_<h>.pt; clean + 26 faults), identical inputs:
  fixed        r = 1,          no P_slip        (= closed_loop.py 'fixed')
  pslip        r = 1,          P_slip learned   (constraint gating alone)
  lstm         r learned,      no P_slip        (= closed_loop.py 'lstm', F10)
  full         r learned,      P_slip learned
  oracle_full  r = labels,     P_slip = GT slip label
P_slip (per IMU row, causal, from the parallel fixed-R run's features) enters the filter through
Pratiyush's hooks: predict(t, p_slip), update_nhc(p_slip), detector Evidence.p_slip.
Writes results/fullsys_int_closed_loop.csv. Usage: OMP_NUM_THREADS=1 python3 learning/closed_loop_int.py [--jobs N]
"""
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
from learning.closed_loop import DET, EKF9, EPS, FEAT_DIR, GATE, MODEL_DIR, RES, predict_r  # noqa: E402,F401

VARIANTS = ['fixed', 'pslip', 'lstm', 'full', 'oracle_full']


def run_ekf9(imu, odom, cmd, gt, det_cfg, p9, r_imu, r_odom_imu, p_slip=None):
    """closed_loop.run_ekf9 + optional per-IMU-row P_slip (None = hooks fed nothing, as before)."""
    from zupt.core.detector import Evidence, ImuSample, StationarityDetector, gravity_compensated_horizontal
    from zupt.ekf.ekf9 import Ekf9
    from zupt.replay import quat_to_rpy
    roll, pitch, _ = quat_to_rpy(imu.orient_x, imu.orient_y, imu.orient_z, imu.orient_w)
    roll, pitch = roll.values, pitch.values
    ax, ay, az = imu.linear_acc_x.values, imu.linear_acc_y.values, imu.linear_acc_z.values
    gz = imu.angular_vel_z.values
    _, _, gyaw = quat_to_rpy(gt.gt_orient_x, gt.gt_orient_y, gt.gt_orient_z, gt.gt_orient_w)
    f = Ekf9(p9, gt.gt_x.iat[0], gt.gt_y.iat[0], float(gyaw.iat[0]))
    det = StationarityDetector(det_cfg, have_lidar=False, logger=lambda *_: None)
    ti = imu.t.values
    ov, ow, to = odom.linear_x.values, odom.angular_z.values, odom.t.values
    ev = sorted([(t, 0, k) for k, t in enumerate(ti)] + [(t, 1, k) for k, t in enumerate(to)])
    ct = cmd.t.values if cmd is not None and len(cmd) else np.array([])
    X, Y = np.zeros(len(imu)), np.zeros(len(imu))
    lo = None
    for tt, kind, k in ev:
        j = k if kind == 0 else np.searchsorted(ti, tt, side='right') - 1
        ps = None if p_slip is None else (float(p_slip[j]) if j >= 0 else 0.0)
        if ps is None:
            f.predict(tt)
        else:
            f.predict(tt, ps)
        if kind == 1:
            ro = float(r_odom_imu[j]) if j >= 0 else 1.0
            if ro >= GATE:
                f.update_odom(ov[k], ow[k], ro)
            lo = (ov[k], ow[k], ro)
            continue
        hx, hy = gravity_compensated_horizontal(ax[k], ay[k], az[k], roll[k], pitch[k])
        f.set_accel(hx, hy)
        ri = float(r_imu[k])
        if ri >= GATE:
            f.update_gyro(gz[k], ri)
        if ps is None:
            f.update_nhc(0.0)
        else:
            f.update_nhc(ps)
        e = Evidence()
        ci = np.searchsorted(ct, tt, side='right') - 1
        if ci >= 0:
            e.v_cmd, e.w_cmd = cmd.linear_x.iat[ci], cmd.angular_z.iat[ci]
        if lo is not None:
            e.v_odom, e.w_odom, e.r_odom = lo
        e.r_imu = ri
        if ps is not None:
            e.p_slip = ps
        o = det.step(ImuSample(tt, gz[k], ax[k], ay[k], az[k], roll[k], pitch[k]), e)
        if o.apply_update:
            f.update_zupt(o.r_scale)
            f.update_zaru(o.r_scale)
        X[k], Y[k] = f.x[0], f.x[1]
    gx, gy = np.interp(ti, gt.t.values, gt.gt_x.values), np.interp(ti, gt.t.values, gt.gt_y.values)
    return float(np.sqrt(np.mean((X - gx) ** 2 + (Y - gy) ** 2)))


def work(args):
    h, data = args
    torch.set_num_threads(1)
    from zupt.core.config import load_config
    from zupt.ekf.ekf9 import load_ekf9_params
    from zupt.replay import load_session
    from learning.faults import all_faults, clean, inject
    from learning.features import FEATURES
    from learning.train_slip import auc, predict_p
    ck = torch.load(os.path.join(MODEL_DIR, f'rel_{h}.pt'), map_location='cpu', weights_only=False)
    cs = torch.load(os.path.join(MODEL_DIR, f'slip_{h}.pt'), map_location='cpu', weights_only=False)
    with contextlib.redirect_stdout(io.StringIO()):
        sess = load_session(data, h)
    cfg, p9 = load_config(DET), load_ekf9_params(EKF9)
    faults = {f.name: f for f in all_faults(h)}
    rows = []
    for name in [h] + list(faults):
        z = np.load(os.path.join(FEAT_DIR, f'{name}.npz'))
        cols, d = list(z['columns']), z['data']
        Xf = d[:, [cols.index(c) for c in FEATURES]].astype(np.float32)
        lab_i, lab_o, slip = (d[:, cols.index(c)] for c in ('r_imu', 'r_odom', 'slip'))
        fr = clean(sess) if name == h else inject(sess, faults[name])
        rp, ps = predict_r(ck, Xf), predict_p(cs, Xf)
        one = np.ones(len(Xf))
        res = dict(model=h, run=name, type=fr.manifest.get('type'), sensor=fr.manifest.get('sensor'),
                   severity=fr.manifest.get('severity'), p_auc=auc(slip, ps),
                   p_high_pct=100 * float((ps >= 0.5).mean()), slip_pct=100 * float(slip.mean()))
        cfgs = dict(fixed=(one, one, None), pslip=(one, one, ps), lstm=(rp[:, 0], rp[:, 1], None),
                    full=(rp[:, 0], rp[:, 1], ps), oracle_full=(lab_i, lab_o, slip))
        for tag in VARIANTS:
            ri, ro, p = cfgs[tag]
            res[tag] = run_ekf9(fr.imu, fr.odom, sess[2], sess[3], cfg, p9, ri, ro, p)
        rows.append(res)
    return rows


def main():
    from zupt import paths as _P
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--data', default=_P.DATA)
    a = ap.parse_args()
    if os.environ.get('OMP_NUM_THREADS') != '1':
        sys.exit('ABORT: run with OMP_NUM_THREADS=1')
    hs = sorted(os.path.basename(p)[5:-3] for p in glob.glob(os.path.join(MODEL_DIR, 'slip_*.pt')))
    print(f'{len(hs)} sessions with rel + slip models', flush=True)
    tasks = [(h, a.data) for h in hs]
    if a.jobs <= 1:
        out = [work(t) for t in tasks]
    else:
        with mp.get_context('fork').Pool(a.jobs) as pool:
            out = pool.map(work, tasks)
    df = pd.DataFrame([r for rows in out for r in rows])
    os.makedirs(RES, exist_ok=True)
    df.to_csv(os.path.join(RES, 'fullsys_int_closed_loop.csv'), index=False)
    pd.set_option('display.width', 220)
    c, f = df[df.type == 'clean'], df[df.type != 'clean']
    print('\nDEVELOPMENT integration, CLEAN runs (EKF9 position RMSE, m):')
    print(c[['model', 'slip_pct', 'p_auc', 'p_high_pct'] + VARIANTS].round(3).to_string(index=False))
    print('means: ' + '  '.join(f'{v} {c[v].mean():.3f}' for v in VARIANTS))
    g = f.groupby(['sensor', 'type'])[VARIANTS].mean()
    print('\nDEVELOPMENT integration, FAULT runs (mean RMSE, m):')
    print(g.round(3).to_string())
    for a_, b_ in (('pslip', 'fixed'), ('full', 'lstm'), ('full', 'fixed')):
        for nm, s in (('clean', c), ('fault', f)):
            print(f'{a_} vs {b_} [{nm}]: wins {int((s[a_] < s[b_]).sum())}/{len(s)}, '
                  f'median change {100 * np.median(s[a_] / s[b_] - 1):+.1f} %')


if __name__ == '__main__':
    main()
