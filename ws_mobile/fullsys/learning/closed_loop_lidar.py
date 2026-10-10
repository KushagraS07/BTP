#!/usr/bin/env python3
"""closed_loop_lidar.py -- F12 (development): reliability r with LiDAR scan matching + 5c in EKF9.

Same runs as closed_loop.py (held-out session h: clean + 26 gyro/odom faults; LiDAR itself clean).
Filter = Paper_ws change 1 (zupt.evaluation.eval_change1.pipeline): EkfLidar (scan matching by
stochastic cloning), detector with LiDAR evidence (Tier 3 + stuck path), NHC, ZUPT/ZARU; no accel
bias injected (as F10). 5c = two passes: run, find slip segments at stops (run_segments), skip the
wheel updates inside them, run again. Wheel claims for the LiDAR motion check come from the
(possibly faulted) odometry. Variants:
  fixed_L      r = 1                  (= eval_change1 'LiDAR' with bias 0 -- regression check)
  fixed_L5c    r = 1, + 5c            (= eval_change1 'LiDAR+5c')
  lstm_L       r = F10 LSTM
  lstm_L5c     r = F10 LSTM, + 5c
  oracle_L     r = labels (F10, slip-aware)
Odometry update skipped when r_odom < GATE or inside a 5c slip segment; gyro skipped when r_imu < GATE.
Writes results/fullsys_lidar_closed_loop.csv. Usage: OMP_NUM_THREADS=1 python3 learning/closed_loop_lidar.py [--jobs N]
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
from learning.closed_loop import DET, EKF9, FEAT_DIR, GATE, MODEL_DIR, RES, predict_r  # noqa: E402
from zupt import paths as _P  # noqa: E402

SEG = os.path.join(_P.CONFIG, 'segment_params_dev.yaml')
VARIANTS = ['fixed_L', 'fixed_L5c', 'lstm_L', 'lstm_L5c', 'oracle_L']


def run_lidar(sess, odom, scans, det_cfg, p9, r_imu, r_odom_imu, gate5c=None):
    """eval_change1.pipeline(use_lidar=True, bias=0) + per-row reliabilities. Returns dict with the
    position RMSE (closed_loop definition, IMU grid) and the traces 5c needs."""
    from zupt.core.detector import Evidence, ImuSample, StationarityDetector, gravity_compensated_horizontal
    from zupt.core.lidar import scan_points
    from zupt.core.scan_match import make_reference
    from zupt.ekf.ekf9 import ITH, IX, IY
    from zupt.ekf.ekf_lidar import EkfLidar
    from zupt.replay import lidar_evidence, quat_to_rpy
    imu, _, cmd, gt = sess
    roll, pitch, _ = quat_to_rpy(imu.orient_x, imu.orient_y, imu.orient_z, imu.orient_w)
    roll, pitch = roll.values, pitch.values
    ax, ay = imu.linear_acc_x.values, imu.linear_acc_y.values
    az, gz = imu.linear_acc_z.values, imu.angular_vel_z.values
    _, _, gyaw = quat_to_rpy(gt.gt_orient_x, gt.gt_orient_y, gt.gt_orient_z, gt.gt_orient_w)
    f = EkfLidar(p9, gt.gt_x.iat[0], gt.gt_y.iat[0], float(gyaw.iat[0]), None)
    ov, ow, to = odom.linear_x.values, odom.angular_z.values, odom.t.values
    g5 = np.zeros(len(odom), bool) if gate5c is None else gate5c
    lev = lidar_evidence(scans, odom)
    lt = np.array([e.t for e in lev])
    det = StationarityDetector(det_cfg, have_lidar=True, logger=lambda *_: None)
    ti = imu.t.values
    ev = ([(tt, 0, k) for k, tt in enumerate(ti)] + [(tt, 1, k) for k, tt in enumerate(to)] +
          [(tt, 2, k) for k, tt in enumerate(scans.t.values)])
    ev.sort()
    ct = cmd.t.values if cmd is not None and len(cmd) else np.array([])
    X, Y, TH, ST, HX = (np.zeros(len(ti)), np.zeros(len(ti)), np.zeros(len(ti)), np.zeros(len(ti), bool),
                         np.zeros(len(ti)))
    last = None
    for tt, kind, k in ev:
        f.predict(tt)
        if kind == 1:
            j = np.searchsorted(ti, tt, side='right') - 1
            ro = 0.0 if g5[k] else (float(r_odom_imu[j]) if j >= 0 else 1.0)
            last = (ov[k], ow[k], ro)
            if ro >= GATE:
                f.update_odom(ov[k], ow[k], ro)
            continue
        if kind == 2:
            r = np.array(scans.ranges.iat[k].split(';'), float)
            a = (scans.angle_min.iat[k], scans.angle_increment.iat[k],
                 scans.range_min.iat[k], scans.range_max.iat[k])
            pts, _, _ = scan_points(r, *a)
            f.scan(pts, make_reference(r, *a))
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
        if last is not None:
            e.v_odom, e.w_odom, e.r_odom = last
        e.r_imu = ri
        li = np.searchsorted(lt, tt, side='right') - 1
        if li >= 0:
            L = lev[li]
            e.lidar_changed, e.lidar_age = L.changed, tt - L.t
            e.lidar_out0, e.lidar_out1 = L.out0, L.out1
            e.lidar_ref_age, e.lidar_claim = L.ref_age, L.claim
        o = det.step(ImuSample(tt, gz[k], ax[k], ay[k], az[k], roll[k], pitch[k]), e)
        if o.apply_update:
            f.update_zupt(o.r_scale)
            f.update_zaru(o.r_scale)
        X[k], Y[k], TH[k], ST[k], HX[k] = f.x[IX], f.x[IY], f.x[ITH], o.stationary, hx
    gx, gy = np.interp(ti, gt.t.values, gt.gt_x.values), np.interp(ti, gt.t.values, gt.gt_y.values)
    return dict(rmse=float(np.sqrt(np.mean((X - gx) ** 2 + (Y - gy) ** 2))), t=ti, hx=HX, gz=gz, st=ST, X=X, Y=Y, TH=TH)


def with_5c(sess, odom, scans, det_cfg, p9, seg_cfg, r_imu, r_odom):
    from zupt.evaluation.eval_5c import mask_to_odom
    from zupt.segments import run_segments
    p1 = run_lidar(sess, odom, scans, det_cfg, p9, r_imu, r_odom)
    segs = run_segments(seg_cfg, p1['t'], p1['hx'], p1['gz'], p1['st'], odom.t.values, odom.linear_x.values)
    g = mask_to_odom(segs, odom.t.values)
    p2 = run_lidar(sess, odom, scans, det_cfg, p9, r_imu, r_odom, gate5c=g) if g.any() else p1
    return p1, p2, float(g.mean())


def work(args):
    h, data = args
    torch.set_num_threads(1)
    from zupt.core.config import load_config
    from zupt.ekf.ekf9 import load_ekf9_params
    from zupt.replay import load_scans, load_session
    from zupt.segments import load_segment_config
    from learning.faults import all_faults, clean, inject
    from learning.features import FEATURES
    ck = torch.load(os.path.join(MODEL_DIR, f'rel_{h}.pt'), map_location='cpu', weights_only=False)
    with contextlib.redirect_stdout(io.StringIO()):
        sess = load_session(data, h)
    scans = load_scans(data, h)
    if scans is None:
        raise SystemExit(f'ABORT: no scan file for {h}')
    cfg, p9, seg = load_config(DET), load_ekf9_params(EKF9), load_segment_config(SEG)
    faults = {f.name: f for f in all_faults(h)}
    rows = []
    for name in [h] + list(faults):
        z = np.load(os.path.join(FEAT_DIR, f'{name}.npz'))
        cols, d = list(z['columns']), z['data']
        Xf = d[:, [cols.index(c) for c in FEATURES]].astype(np.float32)
        fr = clean(sess) if name == h else inject(sess, faults[name])
        s2 = (fr.imu, fr.odom, sess[2], sess[3])
        rp = predict_r(ck, Xf)
        one = np.ones(len(Xf))
        res = dict(model=h, run=name, type=fr.manifest.get('type'), sensor=fr.manifest.get('sensor'),
                   severity=fr.manifest.get('severity'))
        a, b, res['seg5c_fixed'] = with_5c(s2, fr.odom, scans, cfg, p9, seg, one, one)
        res['fixed_L'], res['fixed_L5c'] = a['rmse'], b['rmse']
        a, b, res['seg5c_lstm'] = with_5c(s2, fr.odom, scans, cfg, p9, seg, rp[:, 0], rp[:, 1])
        res['lstm_L'], res['lstm_L5c'] = a['rmse'], b['rmse']
        res['oracle_L'] = run_lidar(s2, fr.odom, scans, cfg, p9, d[:, cols.index('r_imu')],
                                    d[:, cols.index('r_odom')])['rmse']
        rows.append(res)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--data', default=_P.DATA)
    a = ap.parse_args()
    if os.environ.get('OMP_NUM_THREADS') != '1':
        sys.exit('ABORT: run with OMP_NUM_THREADS=1')
    hs = sorted(os.path.basename(p)[4:-3] for p in glob.glob(os.path.join(MODEL_DIR, 'rel_*.pt')))
    print(f'{len(hs)} sessions with rel models', flush=True)
    tasks = [(h, a.data) for h in hs]
    if a.jobs <= 1:
        out = [work(t) for t in tasks]
    else:
        with mp.get_context('fork').Pool(a.jobs) as pool:
            out = pool.map(work, tasks)
    df = pd.DataFrame([r for rows in out for r in rows])
    ref = os.path.join(RES, 'fullsys_rel_closed_loop.csv')       # F10 no-LiDAR reference
    if os.path.exists(ref):
        df = df.merge(pd.read_csv(ref)[['run', 'fixed', 'lstm']], on='run', how='left')
    os.makedirs(RES, exist_ok=True)
    df.to_csv(os.path.join(RES, 'fullsys_lidar_closed_loop.csv'), index=False)
    pd.set_option('display.width', 220)
    cols = [c for c in ['fixed', 'lstm'] if c in df] + VARIANTS
    c, f = df[df.type == 'clean'], df[df.type != 'clean']
    print('\nDEVELOPMENT LiDAR, CLEAN runs (EKF9 position RMSE, m; fixed/lstm = no LiDAR, F10):')
    print(c[['model'] + cols + ['seg5c_fixed', 'seg5c_lstm']].round(3).to_string(index=False))
    print('means: ' + '  '.join(f'{v} {c[v].mean():.3f}' for v in cols))
    print('\nDEVELOPMENT LiDAR, FAULT runs (mean RMSE, m):')
    print(f.groupby(['sensor', 'type'])[cols].mean().round(3).to_string())
    pairs = [('fixed_L', 'fixed'), ('lstm_L', 'fixed_L'), ('lstm_L5c', 'fixed_L5c'), ('lstm_L5c', 'lstm_L'),
             ('fixed_L5c', 'fixed_L')]
    for a_, b_ in pairs:
        if b_ not in df:
            continue
        for nm, s in (('clean', c), ('fault', f)):
            print(f'{a_} vs {b_} [{nm}]: wins {int((s[a_] < s[b_]).sum())}/{len(s)}, '
                  f'median change {100 * np.median(s[a_] / s[b_] - 1):+.1f} %')


if __name__ == '__main__':
    main()
