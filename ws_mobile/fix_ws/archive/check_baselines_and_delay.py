#!/usr/bin/env python3
"""
check_baselines_and_delay.py -- READ-ONLY on ~/ws_mobile/sensor_logs.
Lives in and writes only to ~/ws_mobile/fix_ws.

Part A: why do some "good" baselines have EKF RMSE of 37-261 m?
        Prints the per-baseline extents of ground truth, odometry, cmd_vel
        and the EKF, plus timestamp and row-count health and when the EKF
        error first exceeds 1 m.
Part B: are the delay fault sessions on disk really from the REDESIGNED
        (raw time_sec shift + re-preprocess) mechanism?
        Checks the manifest mechanism field, the time ranges, and whether the
        faulted sensor file actually differs from the baseline file.
"""
import json
import os

import numpy as np
import pandas as pd

LOG = os.path.expanduser('~/ws_mobile/sensor_logs')
FIX = os.path.expanduser('~/ws_mobile/fix_ws')
BASELINES = pd.read_csv(os.path.join(FIX, 'imu_baseline_profile.csv'))['baseline'].astype(str).tolist()
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 50)


def rd(name):
    p = os.path.join(LOG, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def rng_str(s):
    return f'[{s.min():.2f},{s.max():.2f}]'


# ---------------- Part A ----------------
rows = []
for b in BASELINES:
    gt, od = rd(f'ground_truth_filtered_{b}.csv'), rd(f'odom_filtered_{b}.csv')
    imu, cmd = rd(f'imu_filtered_{b}.csv'), rd(f'cmd_vel_filtered_{b}.csv')
    ekf = rd(f'ekf_fused_output_{b}.csv')
    r = {'baseline': b,
         'rows_imu/odom/gt/cmd': '/'.join(str(len(d)) if d is not None else '-' for d in (imu, od, gt, cmd))}
    if gt is not None:
        r['gt_x'], r['gt_y'] = rng_str(gt.gt_x), rng_str(gt.gt_y)
        r['gt_path_m'] = float(np.hypot(np.diff(gt.gt_x), np.diff(gt.gt_y)).sum())
    if od is not None:
        r['odom_x'], r['odom_y'] = rng_str(od.pos_x), rng_str(od.pos_y)
        steps = np.hypot(np.diff(od.pos_x), np.diff(od.pos_y))
        r['odom_path_m'] = float(steps.sum())
        r['odom_max_step_m'] = float(steps.max())
        dt = np.diff(od.timestamp.values)
        r['dt_min/max'] = f'{dt.min():.4f}/{dt.max():.4f}'
        r['n_dt_clipped'] = int(((dt > 0.5) | (dt < 0.001)).sum())
    if cmd is not None:
        r['cmd_v'], r['cmd_w'] = rng_str(cmd.linear_x), rng_str(cmd.angular_z)
    if ekf is not None:
        r['ekf_x'], r['ekf_y'] = rng_str(ekf.ekf_x), rng_str(ekf.ekf_y)
        if gt is not None:
            n = min(len(ekf), len(gt))
            err = np.hypot(ekf.ekf_x.values[:n] - gt.gt_x.values[:n], ekf.ekf_y.values[:n] - gt.gt_y.values[:n])
            t = ekf.timestamp.values[:n] - ekf.timestamp.values[0]
            over = np.where(err > 1.0)[0]
            r['t_err>1m_s'] = float(t[over[0]]) if len(over) else np.nan
            r['err_25/50/75/100%'] = '/'.join(f'{err[int(q * (n - 1))]:.1f}' for q in (.25, .5, .75, 1.0))
    rows.append(r)
print('\n=== PART A: baseline sanity ===')
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f'{x:.2f}'))

# ---------------- Part B ----------------
rows = []
for b in BASELINES:
    for sensor, prefix, col in (('imu', 'imu', 'orient_z'), ('odom', 'odom', 'pos_x')):
        base = rd(f'{prefix}_filtered_{b}.csv')
        for sev in ('small', 'medium', 'large'):
            fs = f'{b}_fault_{sensor}_delay_{sev}'
            mp = os.path.join(LOG, f'fault_meta_{fs}.json')
            r = {'fault_session': fs}
            if os.path.exists(mp):
                m = json.load(open(mp))
                r['redesign_in_manifest'] = 'delay_raw_redesign' in str(m.get('mechanism', ''))
                r['created_utc'] = str(m.get('created_utc', ''))[:19]
            else:
                r['redesign_in_manifest'] = 'NO MANIFEST'
            flt = rd(f'{prefix}_filtered_{fs}.csv')
            if flt is not None and base is not None:
                r['rows_base/fault'] = f'{len(base)}/{len(flt)}'
                r['t0_shift_s'] = float(flt.timestamp.iloc[0] - base.timestamp.iloc[0])
                n = min(len(base), len(flt))
                r[f'max|d {col}|_rowwise'] = float(np.abs(flt[col].values[:n] - base[col].values[:n]).max())
            mpath = os.path.join(LOG, f'ekf_metrics_{fs}.csv')
            fpath = os.path.join(LOG, f'{prefix}_filtered_{fs}.csv')
            if os.path.exists(mpath) and os.path.exists(fpath):
                r['metrics_older_than_data'] = os.path.getmtime(mpath) < os.path.getmtime(fpath)
            rows.append(r)
print('\n=== PART B: delay fault sessions ===')
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f'{x:.6f}'))
