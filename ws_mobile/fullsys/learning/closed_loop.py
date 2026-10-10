#!/usr/bin/env python3
"""
closed_loop.py -- F7 step 5 (development check): does the EKF9 get better with learned reliability?

For each LOBO model rel_<h>.pt and every run of its held-out session h (clean + 26 faults, rebuilt
deterministically with learning.faults), three EKF9 runs on identical inputs:
  fixed   r_imu = r_odom = 1 (nominal R)
  lstm    r from the model: R = R0 / (r + EPS) and the update is SKIPPED when r < GATE (D6)
  oracle  r = the label, same rule (upper bound)
r is computed causally from the features of the parallel fixed-R run (learning.features), every row,
first 49 rows left-padded with row 0 -- so it can be precomputed per session (as Phase 3B P_slip).
Odometry updates use r_odom of the latest IMU row at or before the odometry time. The detector gets
r_imu / r_odom as evidence (its r_imu_min / r_odom_min checks).
Writes results/fullsys_rel_closed_loop.csv and prints wins and mean RMSE per sensor/fault type.
Development only. Usage: OMP_NUM_THREADS=1 python3 learning/closed_loop.py [--jobs N]
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
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from zupt import paths as _P  # noqa: E402

FEAT_DIR = os.path.join(_P.WS, 'fullsys_logs', 'features')
MODEL_DIR = os.path.join(_P.WS, 'fullsys_logs', 'models')
RES = os.path.join(_P.WS, 'results')
DET = os.path.join(_P.CONFIG, 'zupt_params_dev.yaml')
EKF9 = os.path.join(_P.CONFIG, 'ekf9_params_dev.yaml')
GATE = 0.5          # D6: update skipped below this reliability
EPS = 1e-3


def predict_r(ck, X):
    from learning.train import RelLSTM
    m = RelLSTM(n_in=len(ck['features']), hidden=ck['hidden'], layers=ck['layers'])
    m.load_state_dict(ck['model_state'])
    m.eval()
    w = ck['window']
    pad = np.concatenate([np.repeat(X[:1], w - 1, axis=0), X])
    Xn = (pad - ck['mean']) / ck['std']
    out = []
    with torch.no_grad():
        for k in range(0, len(X), 2048):
            idx = np.arange(k, min(k + 2048, len(X)))
            win = np.stack([Xn[i:i + w] for i in idx]).astype(np.float32)
            out.append(m(torch.from_numpy(win)).numpy())
    return np.concatenate(out)


def run_ekf9(imu, odom, cmd, gt, det_cfg, p9, r_imu, r_odom_imu):
    """EKF9 replay with per-IMU-row reliabilities (r_odom given on the IMU timeline)."""
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
        f.predict(tt)
        if kind == 1:
            j = np.searchsorted(ti, tt, side='right') - 1
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
        f.update_nhc(0.0)
        e = Evidence()
        ci = np.searchsorted(ct, tt, side='right') - 1
        if ci >= 0:
            e.v_cmd, e.w_cmd = cmd.linear_x.iat[ci], cmd.angular_z.iat[ci]
        if lo is not None:
            e.v_odom, e.w_odom, e.r_odom = lo
        e.r_imu = ri
        o = det.step(ImuSample(tt, gz[k], ax[k], ay[k], az[k], roll[k], pitch[k]), e)
        if o.apply_update:
            f.update_zupt(o.r_scale)
            f.update_zaru(o.r_scale)
        X[k], Y[k] = f.x[0], f.x[1]
    gx, gy = np.interp(ti, gt.t.values, gt.gt_x.values), np.interp(ti, gt.t.values, gt.gt_y.values)
    return float(np.sqrt(np.mean((X - gx) ** 2 + (Y - gy) ** 2)))


def work(args):
    model_path, data = args
    torch.set_num_threads(1)
    from zupt.core.config import load_config
    from zupt.ekf.ekf9 import load_ekf9_params
    from zupt.replay import load_session
    from learning.faults import all_faults, clean, inject
    from learning.features import FEATURES
    ck = torch.load(model_path, map_location='cpu', weights_only=False)
    h = ck['held_out']
    with contextlib.redirect_stdout(io.StringIO()):
        sess = load_session(data, h)
    cfg, p9 = load_config(DET), load_ekf9_params(EKF9)
    faults = {f.name: f for f in all_faults(h)}
    rows = []
    for name in [h] + list(faults):
        z = np.load(os.path.join(FEAT_DIR, f'{name}.npz'))
        cols = list(z['columns'])
        d = z['data']
        Xf = d[:, [cols.index(c) for c in FEATURES]].astype(np.float32)
        lab_imu, lab_odom = d[:, cols.index('r_imu')], d[:, cols.index('r_odom')]
        fr = clean(sess) if name == h else inject(sess, faults[name])
        rp = predict_r(ck, Xf)
        one = np.ones(len(Xf))
        res = dict(model=h, run=name, type=fr.manifest.get('type'), sensor=fr.manifest.get('sensor'),
                   severity=fr.manifest.get('severity'))
        for tag, ri, ro in (('fixed', one, one), ('lstm', rp[:, 0], rp[:, 1]), ('oracle', lab_imu, lab_odom)):
            res[tag] = run_ekf9(fr.imu, fr.odom, sess[2], sess[3], cfg, p9, ri, ro)
        rows.append(res)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--data', default=_P.DATA)
    a = ap.parse_args()
    if os.environ.get('OMP_NUM_THREADS') != '1':
        sys.exit('ABORT: run with OMP_NUM_THREADS=1')
    models = sorted(glob.glob(os.path.join(MODEL_DIR, 'rel_*.pt')))
    print(f'{len(models)} models', flush=True)
    tasks = [(m, a.data) for m in models]
    if a.jobs <= 1:
        out = [work(t) for t in tasks]
    else:
        with mp.get_context('fork').Pool(a.jobs) as pool:
            out = pool.map(work, tasks)
    df = pd.DataFrame([r for rows in out for r in rows])
    os.makedirs(RES, exist_ok=True)
    df.to_csv(os.path.join(RES, 'fullsys_rel_closed_loop.csv'), index=False)
    pd.set_option('display.width', 200)
    f = df[df.type != 'clean']
    g = f.groupby(['sensor', 'type']).apply(lambda x: pd.Series(dict(
        n=len(x), lstm_beats_fixed=int((x.lstm < x.fixed).sum()), oracle_beats_fixed=int((x.oracle < x.fixed).sum()),
        fixed=x.fixed.mean(), lstm=x.lstm.mean(), oracle=x.oracle.mean())), include_groups=False)
    print('\nDEVELOPMENT closed loop (EKF9 position RMSE, m; mean over held-out fault runs):')
    print(g.round(3).to_string())
    c = df[df.type == 'clean']
    print(f'\nclean runs: fixed {c.fixed.mean():.3f}  lstm {c.lstm.mean():.3f}  oracle {c.oracle.mean():.3f}  '
          f'(lstm/fixed {100 * (c.lstm.mean() / c.fixed.mean() - 1):+.1f} %)')
    print(f'fault runs: lstm beats fixed in {int((f.lstm < f.fixed).sum())}/{len(f)}; '
          f'median change {100 * np.median(f.lstm / f.fixed - 1):+.1f} %')


if __name__ == '__main__':
    main()
