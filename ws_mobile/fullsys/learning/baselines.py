#!/usr/bin/env python3
"""baselines.py -- F14 (development): baselines, ablations and the metrics package (FINDINGS F14).

One filter infrastructure for every method: EKF9 + LiDAR scan matching (EkfLidar) + stationarity detector
+ 5c two-pass, as closed_loop_q.run; methods differ only in how sensor trust is set.
Writes results/fullsys_baselines.csv (one row per run x method, all metrics).
Usage: OMP_NUM_THREADS=1 python3 learning/baselines.py [--jobs N]
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
from learning.closed_loop import DET, EKF9, FEAT_DIR, GATE, MODEL_DIR, RES, predict_r  # noqa: E402
from learning.closed_loop_lidar import SEG  # noqa: E402
from zupt import paths as _P  # noqa: E402

CHI2_99 = {1: 6.635, 2: 9.210, 3: 11.345}
CHI2_95 = {1: 3.841, 2: 5.991}
NEES_BAND = (0.216, 9.348)       # chi2(3) 2.5 % / 97.5 %
SH_B, SH_LO, SH_HI = 0.98, 0.1, 1000.0
HEUR_W = 0.30                    # rad/s, Phase 3B frozen slip threshold
DIVERGE_M = 2.0
RPE_WIN, RPE_STEP = 250, 50      # 5 s windows every 1 s at 50 Hz
METHODS = ['fixed', 'chi2', 'sage_husa', 'heuristic', 'ours', 'hybrid', 'oracle',
           'odom_only', 'imu_only', 'no_pslip', 'no_5c', 'hybrid_bq']


class SageHusa:
    def __init__(self, r0):
        self.r0 = np.asarray(r0, float)
        self.r = self.r0.copy()
        self.k = 0

    def step(self, nu, hph):
        d = (1 - SH_B) / (1 - SH_B ** (self.k + 1))
        self.k += 1
        self.r = (1 - d) * self.r + d * (nu ** 2 - hph)
        self.r = np.clip(self.r, SH_LO * self.r0, SH_HI * self.r0)
        return self.r


def run(sess, odom, scans, det_cfg, p9, r_imu, r_odom_imu, gate5c=None, q=None, s=None,
        chi2=False, sage=False, lidar='on', scan_skip=None, stale=False):
    from zupt.core.detector import Evidence, ImuSample, StationarityDetector, gravity_compensated_horizontal
    from zupt.core.lidar import scan_points
    from zupt.core.scan_match import make_reference
    from zupt.ekf.ekf9 import EPS, IBG, ITH, IU, IW, IX, IY
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
    lev = lidar_evidence(scans, odom) if lidar != 'off' else []
    skipm = None if scan_skip is None else np.asarray(scan_skip, bool)
    if stale:       # a scan identical to the previous one is stale (real returns always carry noise)
        rv = scans.ranges.values
        st = np.r_[False, rv[1:] == rv[:-1]]
        skipm = st if skipm is None else (skipm | st)
    if skipm is not None and lidar != 'off':
        lev = [e for e, sk in zip(lev, skipm) if not sk]     # skipped scans: LiDAR evidence ignored too
    lt = np.array([e.t for e in lev])
    det = StationarityDetector(det_cfg, have_lidar=lidar != 'off', logger=lambda *_: None)
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
    cnt = dict(odom=0, odom_skip=0, gyro=0, gyro_skip=0, lidar=0, lidar_skip=0)
    sh_o = SageHusa([p9.r_odom_v, p9.r_odom_w]) if sage else None
    sh_g = SageHusa([p9.r_gyro]) if sage else None
    idx3 = [IX, IY, ITH]
    last = None

    def nis_of(H, nu, R):
        S = H @ f.P @ H.T + R
        return float(nu @ np.linalg.solve(S, nu))

    for tt, kind, k in ev:
        j = k if kind == 0 else np.searchsorted(ti, tt, side='right') - 1
        if q is None:
            f.predict(tt)
        else:
            f.predict(tt, float(q[j]) if j >= 0 else 0.0)
        if kind == 1:
            cnt['odom'] += 1
            ro = 0.0 if g5[k] else (float(r_odom_imu[j]) if j >= 0 else 1.0)
            last = (ov[k], ow[k], ro)
            if ro < GATE:
                cnt['odom_skip'] += 1
                continue
            H = np.zeros((2, f.n)); H[0, IU] = 1; H[1, IW] = 1
            nu = np.array([ov[k] - f.x[IU], ow[k] - f.x[IW]])
            if sage:
                R = np.diag(sh_o.step(nu, np.diag(H @ f.P @ H.T)))
            else:
                R = np.diag([p9.r_odom_v, p9.r_odom_w]) / (ro + EPS)
            v = nis_of(H, nu, R)
            if chi2 and v > CHI2_99[2]:
                cnt['odom_skip'] += 1
                continue
            nis_o.append(v)
            if sage:
                f._update('odom', [ov[k], ow[k]], f.x[[IU, IW]], H, R)
            else:
                f.update_odom(ov[k], ow[k], ro)
            continue
        if kind == 2:
            r = np.array(scans.ranges.iat[k].split(';'), float)
            a = (scans.angle_min.iat[k], scans.angle_increment.iat[k],
                 scans.range_min.iat[k], scans.range_max.iat[k])
            pts, _, _ = scan_points(r, *a)
            if lidar == 'off':
                continue
            if lidar == 'on' and skipm is None:
                f.scan(pts, make_reference(r, *a))
                continue
            from zupt.core.scan_match import match
            res = None
            if f.ref is not None and len(pts):
                res = match(f.ref, pts, init=f.predicted_relative(), cfg=f.sm_cfg)
                if res.ok:
                    cnt['lidar'] += 1
                    skip = skipm is not None and bool(skipm[k])
                    if not skip and lidar == 'chi2':
                        skip = lidar_nis(f, res.pose, res.cov) > CHI2_99[3]
                    if skip:
                        cnt['lidar_skip'] += 1
                    else:
                        f.update_relative_pose(res.pose, res.cov)
            f.last_match = res
            f.ref = make_reference(r, *a)
            f._clone()
            continue
        hx, hy = gravity_compensated_horizontal(ax[k], ay[k], az[k], roll[k], pitch[k])
        f.set_accel(hx, hy)
        ri = float(r_imu[k])
        cnt['gyro'] += 1
        if ri >= GATE:
            H = np.zeros((1, f.n)); H[0, IW] = 1; H[0, IBG] = 1
            nu = np.array([gz[k] - f.x[IW] - f.x[IBG]])
            if sage:
                R = np.diag(sh_g.step(nu, np.diag(H @ f.P @ H.T)))
            else:
                R = np.array([[p9.r_gyro / (ri + EPS)]])
            v = nis_of(H, nu, R)
            if chi2 and v > CHI2_99[1]:
                cnt['gyro_skip'] += 1
            else:
                nis_g.append(v)
                if sage:
                    f._update('gyro', [gz[k]], [f.x[IW] + f.x[IBG]], H, R)
                else:
                    f.update_gyro(gz[k], ri)
        else:
            cnt['gyro_skip'] += 1
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
        X[k], Y[k], TH[k], ST[k], HX[k] = f.x[IX], f.x[IY], f.x[ITH], o.stationary, hx
        PP[k] = f.P[np.ix_(idx3, idx3)]
    return dict(t=ti, hx=HX, gz=gz, st=ST, X=X, Y=Y, TH=TH, PP=PP, nis_o=np.array(nis_o),
                nis_g=np.array(nis_g), cnt=cnt)


def lidar_nis(f, z, R):
    """NIS of the scan-match relative-pose measurement (same h, H as EkfLidar.update_relative_pose)."""
    from zupt.ekf.ekf9 import ITH, IX, IY, wrap
    from zupt.ekf.ekf_lidar import ITHC, IXC, IYC
    h = f.predicted_relative()
    dx, dy = f.x[IX] - f.x[IXC], f.x[IY] - f.x[IYC]
    c, s = math.cos(f.x[ITHC]), math.sin(f.x[ITHC])
    H = np.zeros((3, f.n))
    H[0, IX], H[0, IY], H[0, IXC], H[0, IYC], H[0, ITHC] = c, s, -c, -s, -s * dx + c * dy
    H[1, IX], H[1, IY], H[1, IXC], H[1, IYC], H[1, ITHC] = -s, c, s, -c, -c * dx - s * dy
    H[2, ITH], H[2, ITHC] = 1.0, -1.0
    nu = np.asarray(z, float) - h
    nu[2] = wrap(nu[2])
    S = H @ f.P @ H.T + np.asarray(R, float)
    return float(nu @ np.linalg.solve(S, nu))


def metrics(o, gt):
    from zupt.replay import quat_to_rpy
    ti = o['t']
    gx, gy = np.interp(ti, gt.t.values, gt.gt_x.values), np.interp(ti, gt.t.values, gt.gt_y.values)
    _, _, gyaw = quat_to_rpy(gt.gt_orient_x, gt.gt_orient_y, gt.gt_orient_z, gt.gt_orient_w)
    gth = np.interp(ti, gt.t.values, np.unwrap(np.asarray(gyaw, float)))
    ex, ey = o['X'] - gx, o['Y'] - gy
    eth = (o['TH'] - gth + math.pi) % (2 * math.pi) - math.pi
    ep = np.hypot(ex, ey)
    i0 = np.arange(0, len(ti) - RPE_WIN, RPE_STEP)
    i1 = i0 + RPE_WIN
    rpe = np.hypot((o['X'][i1] - o['X'][i0]) - (gx[i1] - gx[i0]), (o['Y'][i1] - o['Y'][i0]) - (gy[i1] - gy[i0]))
    e3 = np.stack([ex, ey, eth], axis=1)
    try:
        nees = np.einsum('ni,ni->n', e3, np.linalg.solve(o['PP'], e3[..., None])[..., 0])
    except np.linalg.LinAlgError:
        nees = np.full(len(ti), np.nan)
    c = o['cnt']
    return dict(rmse=float(np.sqrt(np.mean(ep ** 2))), max_err=float(ep.max()), final_err=float(ep[-1]),
                head_rmse_deg=float(np.degrees(np.sqrt(np.mean(eth ** 2)))),
                rpe5s=float(np.sqrt(np.mean(rpe ** 2))) if len(rpe) else np.nan,
                diverged=bool(ep.max() > DIVERGE_M),
                nees_mean=float(np.nanmean(nees)),
                nees_in95=float(np.nanmean((nees >= NEES_BAND[0]) & (nees <= NEES_BAND[1]))),
                nis_odom_mean=float(o['nis_o'].mean()) if len(o['nis_o']) else np.nan,
                nis_odom_hi=float((o['nis_o'] > CHI2_95[2]).mean()) if len(o['nis_o']) else np.nan,
                nis_gyro_mean=float(o['nis_g'].mean()) if len(o['nis_g']) else np.nan,
                nis_gyro_hi=float((o['nis_g'] > CHI2_95[1]).mean()) if len(o['nis_g']) else np.nan,
                odom_skip=c['odom_skip'] / max(c['odom'], 1), gyro_skip=c['gyro_skip'] / max(c['gyro'], 1))


def with_5c(sess, odom, scans, det_cfg, p9, seg_cfg, two_pass=True, **kw):
    from zupt.evaluation.eval_5c import mask_to_odom
    from zupt.segments import run_segments
    p1 = run(sess, odom, scans, det_cfg, p9, **kw)
    if not two_pass:
        return metrics(p1, sess[3])
    segs = run_segments(seg_cfg, p1['t'], p1['hx'], p1['gz'], p1['st'], odom.t.values, odom.linear_x.values)
    g = mask_to_odom(segs, odom.t.values)
    return metrics(run(sess, odom, scans, det_cfg, p9, gate5c=g, **kw) if g.any() else p1, sess[3])


def work(args):
    h, data, methods = args
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
        one = np.ones(len(Xf))
        ti, to = fr.imu.t.values, fr.odom.t.values
        oi = np.clip(np.searchsorted(to, ti, side='right') - 1, 0, len(to) - 1)
        heur = np.where(np.abs(fr.odom.angular_z.values[oi] - fr.imu.angular_vel_z.values) >= HEUR_W, 0.0, 1.0)
        lab_i, lab_o, slip = (d[:, cols.index(c)] for c in ('r_imu', 'r_odom', 'slip'))
        spec = dict(
            fixed=dict(r_imu=one, r_odom_imu=one),
            chi2=dict(r_imu=one, r_odom_imu=one, chi2=True),
            sage_husa=dict(r_imu=one, r_odom_imu=one, sage=True),
            heuristic=dict(r_imu=one, r_odom_imu=heur),
            ours=dict(r_imu=ri, r_odom_imu=ro, q=ps, s=ps),
            hybrid=dict(r_imu=ri, r_odom_imu=ro, q=ps, s=ps, chi2=True),
            hybrid_bq=dict(r_imu=ri, r_odom_imu=ro, q=np.maximum(ps, 1.0 - ri), s=ps, chi2=True),
            oracle=dict(r_imu=lab_i, r_odom_imu=lab_o, q=slip, s=slip),
            odom_only=dict(r_imu=one, r_odom_imu=ro, q=ps, s=ps),
            imu_only=dict(r_imu=ri, r_odom_imu=one, q=ps, s=ps),
            no_pslip=dict(r_imu=ri, r_odom_imu=ro),
            no_5c=dict(r_imu=ri, r_odom_imu=ro, q=ps, s=ps, two_pass=False))
        for m in methods:
            res = dict(model=h, run=name, method=m, type=fr.manifest.get('type'),
                       sensor=fr.manifest.get('sensor'), severity=fr.manifest.get('severity'))
            res.update(with_5c(s2, fr.odom, scans, cfg, p9, seg, **spec[m]))
            rows.append(res)
    return rows


def summary(df):
    pd.set_option('display.width', 220)
    piv = df.pivot_table(index='run', columns='method', values='rmse').reset_index()
    piv = piv.merge(df.drop_duplicates('run')[['run', 'type', 'sensor']], on='run')
    c, f = piv[piv.type == 'clean'], piv[piv.type != 'clean']
    agg = df.assign(group=np.where(df.type == 'clean', 'clean', 'fault')).groupby(['group', 'method']).agg(
        rmse=('rmse', 'mean'), max_err=('max_err', 'mean'), final=('final_err', 'mean'),
        head_deg=('head_rmse_deg', 'mean'), rpe5s=('rpe5s', 'mean'), diverged=('diverged', 'mean'),
        nees=('nees_mean', 'median'), nees_in95=('nees_in95', 'mean'), nis_odom_hi=('nis_odom_hi', 'mean'),
        nis_gyro_hi=('nis_gyro_hi', 'mean'), odom_skip=('odom_skip', 'mean'), gyro_skip=('gyro_skip', 'mean'))
    print('\nDEVELOPMENT F14 summary (means over runs; nees = median of run means; diverged/skip = share):')
    print(agg.round(3).to_string())
    print('\nDEVELOPMENT F14 FAULT runs, mean RMSE by fault group:')
    print(df[df.type != 'clean'].groupby(['sensor', 'type', 'method']).rmse.mean().unstack().round(3).to_string())
    print('\nDEVELOPMENT F14 ours vs each method (wins of ours, median change of ours):')
    for m in [x for x in METHODS if x != 'ours' and x in piv]:
        for nm, sdf in (('clean', c), ('fault', f)):
            print(f'  ours vs {m:10s} [{nm}]: ours wins {int((sdf.ours < sdf[m]).sum())}/{len(sdf)}, '
                  f'median change {100 * np.median(sdf.ours / sdf[m] - 1):+.1f} %')
    if 'hybrid' in piv:
        cw, fw = int((c.hybrid < c.ours).sum()), int((f.hybrid < f.ours).sum())
        cm, fm = 100 * np.median(c.hybrid / c.ours - 1), 100 * np.median(f.hybrid / f.ours - 1)
        ok = cw >= 6 and fw >= int(np.ceil(0.6 * len(f))) and cm <= -3 and fm <= -3
        print(f'\nDecision rule (F14) hybrid vs ours: clean {cw}/{len(c)} {cm:+.1f} %; faults {fw}/{len(f)} '
              f'{fm:+.1f} % -> {"ADOPT" if ok else "not adopted"}')
    if 'hybrid_bq' in piv and 'hybrid' in piv:
        cw, fw = int((c.hybrid_bq < c.hybrid).sum()), int((f.hybrid_bq < f.hybrid).sum())
        cm, fm = 100 * np.median(c.hybrid_bq / c.hybrid - 1), 100 * np.median(f.hybrid_bq / f.hybrid - 1)
        ok = cw >= 6 and fw >= int(np.ceil(0.6 * len(f))) and cm <= -3 and fm <= -3
        print(f'Decision rule (A1-Q) hybrid_bq vs hybrid: clean {cw}/{len(c)} {cm:+.1f} %; faults {fw}/{len(f)} '
              f'{fm:+.1f} % -> {"ADOPT" if ok else "not adopted"}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--data', default=_P.DATA)
    ap.add_argument('--methods', nargs='*', default=METHODS)
    a = ap.parse_args()
    if os.environ.get('OMP_NUM_THREADS') != '1':
        sys.exit('ABORT: run with OMP_NUM_THREADS=1')
    hs = sorted(os.path.basename(p)[5:-3] for p in glob.glob(os.path.join(MODEL_DIR, 'slip_*.pt')))
    print(f'{len(hs)} sessions; methods: {a.methods}', flush=True)
    tasks = [(h, a.data, a.methods) for h in hs]
    if a.jobs <= 1:
        out = [work(t) for t in tasks]
    else:
        with mp.get_context('fork').Pool(a.jobs) as pool:
            out = pool.map(work, tasks)
    df = pd.DataFrame([r for rows in out for r in rows])
    os.makedirs(RES, exist_ok=True)
    from learning.faults import SUFFIX
    df.to_csv(os.path.join(RES, f'fullsys_baselines{SUFFIX}.csv'), index=False)
    for ref, col, m in (('fullsys_lidar_closed_loop.csv', 'fixed_L5c', 'fixed'),
                        ('fullsys_lidar_closed_loop.csv', 'lstm_L5c', 'no_pslip'),
                        ('fullsys_q_closed_loop.csv', 'A_all', 'ours')):
        p = os.path.join(RES, ref)
        if os.path.exists(p) and m in a.methods:
            mm = df[df.method == m].merge(pd.read_csv(p)[['run', col]], on='run')
            print(f'regression {m} vs {ref}:{col}: max |diff| = {np.abs(mm.rmse - mm[col]).max():.2e} m '
                  f'over {len(mm)} runs', flush=True)
    summary(df)


if __name__ == '__main__':
    main()
