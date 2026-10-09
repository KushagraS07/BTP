#!/usr/bin/env python3
"""
audit_imu_faults.py -- item 1.2 / 4.1 pre-check. READ-ONLY on ~/ws_mobile.
Lives in and writes only to ~/fix_ws.

Tests two alternative explanations for the IMU validation failures BEFORE
running any extreme-severity diagnostic:
  H1 (severity scale): IMU faults are injected on yaw with magnitude
      = mult x std(yaw over the session). That std measures how much the
      robot TURNED, not sensor noise, so straight/short sessions get tiny
      faults.
  H2 (2*pi wrap): a constant yaw bias near a multiple of 2*pi is invisible
      after quat re-encoding. bias 'large' = 6 x yaw_std, so if yaw_std is
      about 1.05 rad then the offset is about 2*pi, which has zero effect.

Uses the existing manifests + ekf_metrics files (last-written values).
"""
import glob
import json
import os

import numpy as np
import pandas as pd

LOG_DIR = os.path.expanduser('~/ws_mobile/sensor_logs')
OUT_FAULTS = os.path.expanduser('~/ws_mobile/fix_ws/imu_fault_audit.csv')
OUT_BASE = os.path.expanduser('~/ws_mobile/fix_ws/imu_baseline_profile.csv')
SEV_ORDER = ['low', 'small', 'slow', 'short', 'medium', 'high', 'large', 'fast', 'long', 'extreme']


def rmse(session):
    p = os.path.join(LOG_DIR, f'ekf_metrics_{session}.csv')
    if not os.path.exists(p):
        return np.nan
    m = pd.read_csv(p)
    return float(m.loc[m['metric'] == 'RMSE', 'value_m'].iloc[0])


def wrapped_abs(a):
    return float(abs(np.arctan2(np.sin(a), np.cos(a))))


def yaw_profile(baseline):
    df = pd.read_csv(os.path.join(LOG_DIR, f'imu_filtered_{baseline}.csv'),
                     usecols=['timestamp', 'orient_z', 'orient_w'])
    yaw = np.arctan2(2 * df.orient_w * df.orient_z, 1 - 2 * df.orient_z ** 2).to_numpy()
    yaw_std = pd.Series(yaw).std()  # ddof=1, wrapped yaw: identical to inject_*()
    return {
        'baseline': baseline,
        'duration_s': float(df.timestamp.iloc[-1] - df.timestamp.iloc[0]),
        'yaw_std_rad': float(yaw_std),
        'total_turn_rad': float(np.ptp(np.unwrap(yaw))),
        'per_step_dyaw_std_rad': float(np.std(np.diff(np.unwrap(yaw)), ddof=1)),
        'bias_mult_hitting_2pi': float(2 * np.pi / yaw_std) if yaw_std > 0 else np.nan,
        'rmse_baseline_m': rmse(baseline),
    }


rows = []
for mp in sorted(glob.glob(os.path.join(LOG_DIR, 'fault_meta_*_fault_imu_*.json'))):
    with open(mp) as f:
        m = json.load(f)
    if m.get('sensor') != 'imu':
        continue
    s = m['injected_stats'].get('_imu_yaw_temp', {})
    mag = s.get('injected_noise_std', s.get('injected_offset', s.get('final_injected_offset', np.nan)))
    rb, rf = rmse(m['baseline_session']), rmse(m['fault_session'])
    rows.append({
        'baseline': m['baseline_session'],
        'fault': m['fault_type'],
        'sev': m['severity_label'],
        'mult': m['severity_value'],
        'yaw_std_used': s.get('baseline_std', np.nan),
        'injected_rad': mag,
        'bias_after_wrap_rad': wrapped_abs(mag) if m['fault_type'] == 'bias' else np.nan,
        'rmse_base': rb,
        'rmse_fault': rf,
        'ratio': rf / rb if rb else np.nan,
    })

faults = pd.DataFrame(rows)
faults['_o'] = faults['sev'].map({k: i for i, k in enumerate(SEV_ORDER)})
faults = faults.sort_values(['fault', 'baseline', '_o']).drop(columns='_o')
base = pd.DataFrame([yaw_profile(b) for b in sorted(faults['baseline'].unique())])

pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 500)
print('\n=== Baseline yaw profile ===')
print(base.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
print('\n=== IMU fault sessions ===')
print(faults.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

faults.to_csv(OUT_FAULTS, index=False)
base.to_csv(OUT_BASE, index=False)
print(f'\nSaved: {OUT_FAULTS}\nSaved: {OUT_BASE}')
