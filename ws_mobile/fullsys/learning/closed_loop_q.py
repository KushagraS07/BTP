#!/usr/bin/env python3
"""closed_loop_q.py -- F13 (development): Q(t) adaptation on top of reliability + LiDAR + 5c.

Same runs as F12. Filter = closed_loop_lidar.run_lidar (Paper_ws change 1 + F10 reliability gating) with
Pratiyush's hooks fed: q[k] -> Ekf9.predict(t, q) (Q_w *= 1 + k_w_slip*q), s[k] -> NHC + detector p_slip.
  base   q = none,          s = none        (= F12 lstm_L5c, regression)
  A_q    q = P_slip,        s = none
  A_all  q = P_slip,        s = P_slip      (Q + NHC + ZUPT hooks)
  B_q    q = 1 - r_imu,     s = none
All with 5c (two passes). Writes results/fullsys_q_closed_loop.csv.
Usage: OMP_NUM_THREADS=1 python3 learning/closed_loop_q.py [--jobs N]
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
from learning.closed_loop_lidar import SEG  # noqa: E402
from zupt import paths as _P  # noqa: E402

VARIANTS = ['base', 'A_q', 'A_all', 'B_q']


def run(sess, odom, scans, det_cfg, p9, r_imu, r_odom_imu, gate5c=None, q=None, s=None):
    from zupt.core.detector import Evidence, ImuSample, StationarityDetector, gravity_compensated_horizontal
    from zupt.core.lidar import scan_points
    from zupt.core.scan_match import make_reference
    from zupt.ekf.ekf9 import IX, IY
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
    X, Y = np.zeros(len(ti)), np.zeros(len(ti))
    ST, HX = np.zeros(len(ti), bool), np.zeros(len(ti))
    last = None
    for tt, kind, k in ev:
        j = k if kind == 0 else np.searchsorted(ti, tt, side='right') - 1
        if q is None:
            f.predict(tt)
        else:
            f.predict(tt, float(q[j]) if j >= 0 else 0.0)
        if kind == 1:
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
        f.update_nhc(0.0 if s is None else float(s[k]))
        e = Evidence()
        ci = np.searchsorted(ct, tt, side='right') - 1
        if ci >= 0:
            e.v_cmd, e.w_cmd = cmd.linear_x.iat[ci], cmd.angular_z.iat[ci]
        if last is not None:
            e.v_odom, e.w_odom, e.r_odom = last
        e.r_imu = ri
        if s is not None:
            e.p_slip = float(s[k])
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
        X[k], Y[k], ST[k], HX[k] = f.x[IX], f.x[IY], o.stationary, hx
    gx, gy = np.interp(ti, gt.t.values, gt.gt_x.values), np.interp(ti, gt.t.values, gt.gt_y.values)
    return dict(rmse=float(np.sqrt(np.mean((X - gx) ** 2 + (Y - gy) ** 2))), t=ti, hx=HX, gz=gz, st=ST)


def with_5c(sess, odom, scans, det_cfg, p9, seg_cfg, r_imu, r_odom, q=None, s=None):
    from zupt.evaluation.eval_5c import mask_to_odom
    from zupt.segments import run_segments
    p1 = run(sess, odom, scans, det_cfg, p9, r_imu, r_odom, q=q, s=s)
    segs = run_segments(seg_cfg, p1['t'], p1['hx'], p1['gz'], p1['st'], odom.t.values, odom.linear_x.values)
    g = mask_to_odom(segs, odom.t.values)
    return (run(sess, odom, scans, det_cfg, p9, r_imu, r_odom, gate5c=g, q=q, s=s) if g.any() else p1)['rmse']


def work(args):
    h, data = args
    torch.set_num_threads(1)
    from zupt.core.config import load_config
    from zupt.ekf.ekf9 import load_ekf9_params
    from zupt.replay import load_scans, load_session
    from zupt.segments import load_segment_config
    from learning.faults import all_faults, clean, inject
    from learning.features import FEATURES
    from learning.train_slip import predict_p
    ck = torch.load(os.path.join(MODEL_DIR, f'rel_{h}.pt'), map_location='cpu', weights_only=False)
    cs = torch.load(os.path.join(MODEL_DIR, f'slip_{h}.pt'), map_location='cpu', weights_only=False)
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
        rp, ps = predict_r(ck, Xf), predict_p(cs, Xf)
        ri, ro = rp[:, 0], rp[:, 1]
        res = dict(model=h, run=name, type=fr.manifest.get('type'), sensor=fr.manifest.get('sensor'),
                   severity=fr.manifest.get('severity'))
        res['base'] = with_5c(s2, fr.odom, scans, cfg, p9, seg, ri, ro)
        res['A_q'] = with_5c(s2, fr.odom, scans, cfg, p9, seg, ri, ro, q=ps)
        res['A_all'] = with_5c(s2, fr.odom, scans, cfg, p9, seg, ri, ro, q=ps, s=ps)
        res['B_q'] = with_5c(s2, fr.odom, scans, cfg, p9, seg, ri, ro, q=1.0 - ri)
        rows.append(res)
    return rows


def main():
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
    ref = os.path.join(RES, 'fullsys_lidar_closed_loop.csv')
    if os.path.exists(ref):
        r12 = pd.read_csv(ref)[['run', 'lstm_L5c']]
        m = df.merge(r12, on='run')
        print(f'regression vs F12 lstm_L5c: max |diff| = {np.abs(m.base - m.lstm_L5c).max():.2e} m '
              f'over {len(m)} runs', flush=True)
    os.makedirs(RES, exist_ok=True)
    df.to_csv(os.path.join(RES, 'fullsys_q_closed_loop.csv'), index=False)
    pd.set_option('display.width', 200)
    c, f = df[df.type == 'clean'], df[df.type != 'clean']
    print('\nDEVELOPMENT Q(t), CLEAN runs (EKF9 + reliability + LiDAR + 5c, position RMSE, m):')
    print(c[['model'] + VARIANTS].round(3).to_string(index=False))
    print('means: ' + '  '.join(f'{v} {c[v].mean():.3f}' for v in VARIANTS))
    print('\nDEVELOPMENT Q(t), FAULT runs (mean RMSE, m):')
    print(f.groupby(['sensor', 'type'])[VARIANTS].mean().round(3).to_string())
    print('\nDecision rule (F13): adopt only if clean wins >= 6/8, fault wins >= 125/208 (60 %), '
          'median change <= -3 % on both.')
    for v in VARIANTS[1:]:
        cw, fw = int((c[v] < c.base).sum()), int((f[v] < f.base).sum())
        cm, fm = 100 * np.median(c[v] / c.base - 1), 100 * np.median(f[v] / f.base - 1)
        ok = cw >= 6 and fw >= int(np.ceil(0.6 * len(f))) and cm <= -3 and fm <= -3
        print(f'{v:6s} vs base: clean {cw}/{len(c)} median {cm:+.1f} %; faults {fw}/{len(f)} median {fm:+.1f} % '
              f'-> {"ADOPT" if ok else "not adopted"}')


if __name__ == '__main__':
    main()
