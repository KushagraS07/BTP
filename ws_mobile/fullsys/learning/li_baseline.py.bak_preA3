#!/usr/bin/env python3
"""li_baseline.py -- F14b (development): Li et al. (IEEE Sensors J. 2026)-style baseline on our filter.

Motion-state probabilities (li_<h>.pt, causal) drive the constraints (paper eqs. 17-18):
  ZUPT only if P_s >= 0.7, R_zupt = R0*(1 + 49*((1-P_s)/0.3)**2); no ZARU; the classifier replaces the
  stationarity detector for ZUPT; 5c uses P_s >= 0.7 as stationary flag.
  NHC: R_nhc = R0*exp(2.5*max(0, P_t-0.6)/0.4).
Sensor updates fixed R. EKF9 + LiDAR + 5c as F14. With li=False the loop equals F14 'fixed'.
Writes results/fullsys_li_baseline.csv (same metrics as F14; 'ours' merged from fullsys_baselines.csv).
Usage: OMP_NUM_THREADS=1 python3 learning/li_baseline.py [--jobs N]
"""
import argparse
import contextlib
import glob
import io
import math
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from learning.closed_loop import DET, EKF9, FEAT_DIR, MODEL_DIR, RES  # noqa: E402
from learning.closed_loop_lidar import SEG  # noqa: E402
from learning.baselines import metrics  # noqa: E402
from zupt import paths as _P  # noqa: E402

GS, GT, BETA, ALPHA, RATIO = 0.7, 0.6, 2.0, 2.5, 50.0


def run(sess, odom, scans, det_cfg, p9, gate5c=None, ps=None, pt=None):
    from zupt.core.detector import Evidence, ImuSample, StationarityDetector, gravity_compensated_horizontal
    from zupt.core.lidar import scan_points
    from zupt.core.scan_match import make_reference
    from zupt.ekf.ekf9 import EPS, IS, ITH, IW, IX, IY
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
    n = len(ti)
    X, Y, TH = np.zeros(n), np.zeros(n), np.zeros(n)
    PP = np.zeros((n, 3, 3))
    ST, HX = np.zeros(n, bool), np.zeros(n)
    nis_o, nis_g = [], []
    cnt = dict(odom=0, odom_skip=0, gyro=0, gyro_skip=0)
    idx3 = [IX, IY, ITH]
    last = None
    for tt, kind, k in ev:
        f.predict(tt)
        if kind == 1:
            cnt['odom'] += 1
            if g5[k]:
                cnt['odom_skip'] += 1
                last = (ov[k], ow[k], 0.0)
                continue
            last = (ov[k], ow[k], 1.0)
            _, v = f.update_odom(ov[k], ow[k], 1.0)
            nis_o.append(v)
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
        cnt['gyro'] += 1
        _, v = f.update_gyro(gz[k], 1.0)
        nis_g.append(v)
        if pt is None:
            f.update_nhc(0.0)
        else:
            H = np.zeros((1, f.n)); H[0, IS] = 1; H[0, IW] = -p9.lever_l
            R = [[p9.r_nhc * math.exp(ALPHA * max(0.0, float(pt[k]) - GT) / (1 - GT)) / (1.0 + EPS)]]
            f._update('nhc', [0.0], [f.x[IS] - p9.lever_l * f.x[IW]], H, R)
        if ps is None:
            e = Evidence()
            ci = np.searchsorted(ct, tt, side='right') - 1
            if ci >= 0:
                e.v_cmd, e.w_cmd = cmd.linear_x.iat[ci], cmd.angular_z.iat[ci]
            if last is not None:
                e.v_odom, e.w_odom, e.r_odom = last
            e.r_imu = 1.0
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
            stat = o.stationary
        else:
            p = float(ps[k])
            stat = p >= GS
            if stat:
                f.update_zupt(1.0 + (RATIO - 1.0) * ((1.0 - p) / (1.0 - GS)) ** BETA)
        X[k], Y[k], TH[k], ST[k], HX[k] = f.x[IX], f.x[IY], f.x[ITH], stat, hx
        PP[k] = f.P[np.ix_(idx3, idx3)]
    return dict(t=ti, hx=HX, gz=gz, st=ST, X=X, Y=Y, TH=TH, PP=PP, nis_o=np.array(nis_o),
                nis_g=np.array(nis_g), cnt=cnt)


def with_5c(sess, odom, scans, det_cfg, p9, seg_cfg, **kw):
    from zupt.evaluation.eval_5c import mask_to_odom
    from zupt.segments import run_segments
    p1 = run(sess, odom, scans, det_cfg, p9, **kw)
    segs = run_segments(seg_cfg, p1['t'], p1['hx'], p1['gz'], p1['st'], odom.t.values, odom.linear_x.values)
    g = mask_to_odom(segs, odom.t.values)
    return metrics(run(sess, odom, scans, det_cfg, p9, gate5c=g, **kw) if g.any() else p1, sess[3])


def work(args):
    h, data, with_fixed = args
    torch.set_num_threads(1)
    from zupt.core.config import load_config
    from zupt.ekf.ekf9 import load_ekf9_params
    from zupt.replay import load_scans, load_session
    from zupt.segments import load_segment_config
    from learning.faults import all_faults, clean, inject
    from learning.train_li import INPUTS, predict_probs
    ck = torch.load(os.path.join(MODEL_DIR, f'li_{h}.pt'), map_location='cpu', weights_only=False)
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
        pr = predict_probs(ck, d[:, [cols.index(c) for c in INPUTS]].astype(np.float32))
        fr = clean(sess) if name == h else inject(sess, faults[name])
        s2 = (fr.imu, fr.odom, sess[2], sess[3])
        base = dict(model=h, run=name, type=fr.manifest.get('type'), sensor=fr.manifest.get('sensor'),
                    severity=fr.manifest.get('severity'))
        res = dict(base, method='li', share_ps=float((pr[:, 0] >= GS).mean()),
                   share_pt=float((pr[:, 2] >= GT).mean()))
        res.update(with_5c(s2, fr.odom, scans, cfg, p9, seg, ps=pr[:, 0], pt=pr[:, 2]))
        rows.append(res)
        if with_fixed:
            r2 = dict(base, method='fixed_check')
            r2.update(with_5c(s2, fr.odom, scans, cfg, p9, seg))
            rows.append(r2)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--data', default=_P.DATA)
    ap.add_argument('--check', action='store_true', help='also run hooks-off variant (regression vs F14 fixed)')
    a = ap.parse_args()
    if os.environ.get('OMP_NUM_THREADS') != '1':
        sys.exit('ABORT: run with OMP_NUM_THREADS=1')
    hs = sorted(os.path.basename(p)[3:-3] for p in glob.glob(os.path.join(MODEL_DIR, 'li_*.pt')))
    print(f'{len(hs)} sessions with Li-style models', flush=True)
    tasks = [(h, a.data, a.check) for h in hs]
    if a.jobs <= 1:
        out = [work(t) for t in tasks]
    else:
        with mp.get_context('fork').Pool(a.jobs) as pool:
            out = pool.map(work, tasks)
    df = pd.DataFrame([r for rows in out for r in rows])
    os.makedirs(RES, exist_ok=True)
    df.to_csv(os.path.join(RES, 'fullsys_li_baseline.csv'), index=False)
    bp = os.path.join(RES, 'fullsys_baselines.csv')
    if not os.path.exists(bp):
        print('fullsys_baselines.csv not found: comparison with ours skipped')
        return
    b = pd.read_csv(bp)
    if a.check:
        m = df[df.method == 'fixed_check'].merge(b[b.method == 'fixed'][['run', 'rmse']], on='run', suffixes=('', '_f14'))
        print(f'regression hooks-off vs F14 fixed: max |diff| = {np.abs(m.rmse - m.rmse_f14).max():.2e} m '
              f'over {len(m)} runs')
    li = df[df.method == 'li'][['run', 'type', 'sensor', 'rmse', 'max_err', 'head_rmse_deg', 'diverged']]
    allm = pd.concat([b[b.method.isin(['fixed', 'ours', 'hybrid'])], df[df.method == 'li']])
    agg = allm.assign(group=np.where(allm.type == 'clean', 'clean', 'fault')).groupby(['group', 'method']).agg(
        rmse=('rmse', 'mean'), max_err=('max_err', 'mean'), head_deg=('head_rmse_deg', 'mean'),
        rpe5s=('rpe5s', 'mean'), diverged=('diverged', 'mean'), nees_in95=('nees_in95', 'mean'))
    pd.set_option('display.width', 200)
    print('\nDEVELOPMENT F14b Li-style vs fixed vs ours (means over runs):')
    print(agg.round(3).to_string())
    for meth in ('ours', 'hybrid'):
        o = b[b.method == meth][['run', 'rmse']].rename(columns={'rmse': 'ref'})
        m = li.merge(o, on='run')
        for nm, s in (('clean', m[m.type == 'clean']), ('fault', m[m.type != 'clean'])):
            print(f'{meth} vs li [{nm}]: {meth} wins {int((s.ref < s.rmse).sum())}/{len(s)}, '
                  f'median change {100 * np.median(s.ref / s.rmse - 1):+.1f} %')
    print(f"li: P_s>=0.7 share {df[df.method=='li'].share_ps.mean():.3f}, P_t>=0.6 share "
          f"{df[df.method=='li'].share_pt.mean():.3f}")


if __name__ == '__main__':
    main()
