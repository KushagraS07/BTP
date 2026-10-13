#!/usr/bin/env python3
"""r2_eval.py -- R2 (FINDINGS R2): faults that END. Run with FULLSYS_FAULT_WINDOW=end after build_dataset.
Methods fixed, chi2, sage_husa, ours, hybrid (F14 definitions, LiDAR + 5c); per run: whole-run metrics, RMSE after the
fault (fault end -> session end), final error; for the learned r of the faulted sensor: detection delay and recovery
delay (fault end -> r >= 0.5 held >= 1 s). Writes results/fullsys_r2.csv.
Usage: FULLSYS_FAULT_WINDOW=end OMP_NUM_THREADS=1 python3 learning/r2_eval.py [--jobs N]"""
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
from learning.closed_loop import DET, EKF9, FEAT_DIR, GATE, MODEL_DIR, RES, predict_r  # noqa: E402
from learning.closed_loop_lidar import SEG  # noqa: E402
from learning.faults import SUFFIX, WINDOW  # noqa: E402
from zupt import paths as _P  # noqa: E402

METHODS = ['fixed', 'chi2', 'sage_husa', 'ours', 'hybrid']
HOLD = 1.0      # s, recovery = r >= gate held this long


def traj_5c(s2, odom, scans, cfg, p9, seg, **kw):
    import learning.baselines as B
    from zupt.evaluation.eval_5c import mask_to_odom
    from zupt.segments import run_segments
    p1 = B.run(s2, odom, scans, cfg, p9, **kw)
    segs = run_segments(seg, p1['t'], p1['hx'], p1['gz'], p1['st'], odom.t.values, odom.linear_x.values)
    g = mask_to_odom(segs, odom.t.values)
    o = B.run(s2, odom, scans, cfg, p9, gate5c=g, **kw) if g.any() else p1
    return o, B.metrics(o, s2[3])


def delays(t, r, onset, fend):
    det = np.where((t >= onset) & (t < fend) & (r < GATE))[0]
    d_det = float(t[det[0]] - onset) if len(det) else np.nan
    if not len(det):
        return d_det, np.nan, np.nan
    after = np.where(t >= fend)[0]
    ok = r >= GATE
    dt = np.median(np.diff(t))
    need = max(1, int(round(HOLD / dt)))
    rec = np.nan
    for i in after:
        if i + need <= len(r) and ok[i:i + need].all():
            rec = float(t[i] - fend)
            break
    gated_after = float((r[after] < GATE).mean()) if len(after) else np.nan
    return d_det, rec, gated_after


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
    cfg, p9, seg = load_config(DET), load_ekf9_params(EKF9), load_segment_config(SEG)
    faults = {f.name: f for f in all_faults(h)}
    rows = []
    for name in [h] + list(faults):
        z = np.load(os.path.join(FEAT_DIR, f'{name}.npz'))
        cols, d = list(z['columns']), z['data']
        Xf = d[:, [cols.index(c) for c in FEATURES]].astype(np.float32)
        fr = clean(sess) if name == h else inject(sess, faults[name])
        m = fr.manifest
        s2 = (fr.imu, fr.odom, sess[2], sess[3])
        rp, ps = predict_r(ck, Xf), predict_p(cs, Xf)
        ri, ro = rp[:, 0], rp[:, 1]
        one = np.ones(len(Xf))
        spec = dict(fixed=dict(r_imu=one, r_odom_imu=one), chi2=dict(r_imu=one, r_odom_imu=one, chi2=True),
                    sage_husa=dict(r_imu=one, r_odom_imu=one, sage=True),
                    ours=dict(r_imu=ri, r_odom_imu=ro, q=ps, s=ps),
                    hybrid=dict(r_imu=ri, r_odom_imu=ro, q=ps, s=ps, chi2=True))
        ti = fr.imu.t.values
        gt = sess[3]
        gx, gy = np.interp(ti, gt.t.values, gt.gt_x.values), np.interp(ti, gt.t.values, gt.gt_y.values)
        fend = m.get('fault_end', np.nan)
        base = dict(model=h, run=name, type=m.get('type'), sensor=m.get('sensor'), severity=m.get('severity'),
                    onset=m.get('onset', np.nan), fault_end=fend)
        if m.get('type') != 'clean':
            r_s = ri if m['sensor'] == 'imu' else ro
            dd, rc, ga = delays(d[:, cols.index('t')], r_s, m['onset'], fend)
            base.update(det_delay=dd, rec_delay=rc, gated_after=ga)
        for meth in METHODS:
            o, met = traj_5c(s2, fr.odom, scans, cfg, p9, seg, **spec[meth])
            e = np.hypot(o['X'] - gx, o['Y'] - gy)
            aft = ti >= fend if np.isfinite(fend) else np.zeros(len(ti), bool)
            res = dict(base, method=meth, rmse=met['rmse'], max_err=met['max_err'], final_err=met['final_err'],
                       diverged=met['diverged'],
                       rmse_after=float(np.sqrt(np.mean(e[aft] ** 2))) if aft.any() else np.nan,
                       err_at_end=float(e[np.argmax(aft)]) if aft.any() else np.nan)
            rows.append(res)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--data', default=_P.DATA)
    a = ap.parse_args()
    if os.environ.get('OMP_NUM_THREADS') != '1':
        sys.exit('ABORT: run with OMP_NUM_THREADS=1')
    if WINDOW != 'end':
        sys.exit('ABORT: run with FULLSYS_FAULT_WINDOW=end')
    hs = sorted(os.path.basename(p)[5:-3] for p in glob.glob(os.path.join(MODEL_DIR, 'slip_*.pt')))
    print(f'R2: {len(hs)} sessions, fault set "{SUFFIX}", methods {METHODS}', flush=True)
    tasks = [(h, a.data) for h in hs]
    if a.jobs <= 1:
        out = [work(t) for t in tasks]
    else:
        with mp.get_context('fork').Pool(a.jobs) as pool:
            out = pool.map(work, tasks)
    df = pd.DataFrame([r for o in out for r in o])
    os.makedirs(RES, exist_ok=True)
    df.to_csv(os.path.join(RES, f'fullsys_r2{SUFFIX}.csv'), index=False)
    ref = os.path.join(RES, 'fullsys_baselines.csv')
    if os.path.exists(ref):
        b = pd.read_csv(ref)
        mm = df[df.type == 'clean'].merge(b[['run', 'method', 'rmse']], on=['run', 'method'], suffixes=('', '_f14'))
        print(f'regression clean runs vs F14: max |diff| = {np.abs(mm.rmse - mm.rmse_f14).max():.2e} m over {len(mm)} rows')
    pd.set_option('display.width', 200)
    f = df[df.type != 'clean']
    print('\nDEVELOPMENT R2 (faults end after <= 15 s) means over fault runs:')
    print(f.groupby('method')[['rmse', 'rmse_after', 'err_at_end', 'final_err', 'diverged']].mean().round(3)
          .reindex(METHODS).to_string())
    print('\nRMSE after the fault, by sensor/type (mean):')
    print(f.pivot_table(index=['sensor', 'type'], columns='method', values='rmse_after').round(3)[METHODS].to_string())
    print('\nRMSE after the fault, by session (mean; D6 = 150047, D7 = 150428):')
    print(f.pivot_table(index='model', columns='method', values='rmse_after').round(3)[METHODS].to_string())
    o = f[f.method == 'ours']
    g = o.groupby(['sensor', 'type'])
    tab = pd.DataFrame({'detected': g.det_delay.apply(lambda v: f'{int(v.notna().sum())}/{len(v)}'),
                        'det_delay_med_s': g.det_delay.median().round(2),
                        'recovered': g.rec_delay.apply(lambda v: f'{int(v.notna().sum())}/{len(v)}'),
                        'rec_delay_med_s': g.rec_delay.median().round(2),
                        'gated_after_mean': g.gated_after.mean().round(3)})
    print('\nLearned r of the faulted sensor (ours): detection and recovery')
    print(tab.to_string())
    pv = f.pivot_table(index='run', columns='method', values='rmse_after')
    w = int((pv.hybrid < pv.fixed).sum())
    print(f'\nlook criteria: median recovery delay {o.rec_delay.median():.2f} s (<= 2 s?); hybrid after-fault RMSE below '
          f'fixed in {w}/{len(pv)} fault runs ({100 * w / len(pv):.0f} %, >= 60 %?)')


if __name__ == '__main__':
    main()
