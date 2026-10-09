#!/usr/bin/env python3
"""
delay_labels_d2.py -- D2 delay-fault reliability labels (fix_ws/PREREG_heldout.md,
Addendum A, frozen 2026-10-03 15:50 IST). Formula fixed there; do not tune.

    e_c(t) = | x_c^fault(t) - x_c^baseline(t) |     (each column, at EKF row time t)
    r_c(t) = 1 / (1 + e_c(t) / sigma_c)
    r(t)   = min_c r_c(t)

  columns c : keys of injected_stats in fault_meta_{baseline}_fault_{sensor}_noise_low.json
  sigma_c   : injected_noise_std / severity_value from that same manifest
              (= the baseline column std the injector itself used)
  signals   : {sensor}_filtered_{fault}.csv and {sensor}_filtered_{baseline}.csv,
              np.interp onto ekf_fused_output_{fault}.csv timestamps
  angles    : columns containing 'yaw' or 'theta' -> wrapped difference
  No smoothing, no clipping, no constants.
"""
import json
import os

import numpy as np
import pandas as pd

LOG_DIR = os.path.expanduser('~/ws_mobile/sensor_logs')


def _t(df):
    if 'timestamp' in df.columns:
        return df['timestamp'].values.astype(float)
    return df['time_sec'].values + df['time_nsec'].values * 1e-9


def delay_r(manifest: dict) -> np.ndarray:
    fault = manifest['fault_session']
    base = manifest['baseline_session']
    sensor = manifest['sensor']
    if manifest['fault_type'] != 'delay' or sensor not in ('imu', 'odom'):
        raise ValueError(f'delay_r called for {fault}')

    nm = json.load(open(os.path.join(LOG_DIR, f'fault_meta_{base}_fault_{sensor}_noise_low.json')))
    sev = float(nm['severity_value'])
    cols = {c: float(s['injected_noise_std']) / sev for c, s in nm['injected_stats'].items()}
    if not cols:
        raise ValueError(f'no columns in noise_low manifest for {base}/{sensor}')
    for c, sg in cols.items():
        if not np.isfinite(sg) or sg <= 0:
            raise ValueError(f'bad sigma for {base}/{sensor}/{c}: {sg}')

    t = pd.read_csv(os.path.join(LOG_DIR, f'ekf_fused_output_{fault}.csv'))['timestamp'].values.astype(float)
    ff = pd.read_csv(os.path.join(LOG_DIR, f'{sensor}_filtered_{fault}.csv'))
    fb = pd.read_csv(os.path.join(LOG_DIR, f'{sensor}_filtered_{base}.csv'))
    tf, tb = _t(ff), _t(fb)

    def col_at(df, tt, c):
        # The IMU injector perturbs a TEMPORARY column '_imu_yaw_temp' =
        # quat_to_yaw(orient_z, orient_w) (fault_injection.py), which is never saved.
        # Derive it exactly as the injector does: interpolate the quaternion
        # components (no +/-pi jump), then convert.
        if c == '_imu_yaw_temp':
            oz = np.interp(t, tt, df['orient_z'].values.astype(float))
            ow = np.interp(t, tt, df['orient_w'].values.astype(float))
            return np.arctan2(2.0 * ow * oz, 1.0 - 2.0 * oz * oz)
        return np.interp(t, tt, df[c].values.astype(float))

    curves = []
    for c, sg in cols.items():
        d = col_at(ff, tf, c) - col_at(fb, tb, c)
        if 'yaw' in c or 'theta' in c:
            d = np.arctan2(np.sin(d), np.cos(d))
        curves.append(1.0 / (1.0 + np.abs(d) / sg))
    return np.min(np.vstack(curves), axis=0)
