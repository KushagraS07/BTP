#!/usr/bin/env python3
"""
spike_diagnostic.py -- READ-ONLY. Do the sensor signals contain real spikes (isolated
single-sample outliers) that a proper local outlier filter would need to remove?

Method (local, so real motion is NOT flagged -- the flaw of global IQR):
  for each numeric signal x (yaw from quaternions, unwrapped):
    r_i  = x_i - (x_{i-1} + x_{i+1}) / 2         (distance from the line through its neighbours)
    s    = 1.4826 * MAD(r)                        (robust noise scale of r, whole session)
    spike if |r_i| > Z * s                        (Z = 6)
  Run on RAW logs and on the *_filtered_* data the EKF actually uses.

Decision rule (fixed BEFORE looking at the output):
  filtered data, key signals:
    all < 0.1 % spikes                         -> DON'T implement (state it in Methods)
    any >= 0.5 % spikes AND median spike > 10 x noise -> CONSIDER (later, pre-registered)
    otherwise                                  -> limitation only, no change
Writes only results/spike_diagnostic.csv.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import ws_paths as W  # noqa: E402

K, Z = 5, 6.0
DONT, CONSIDER, BIG = 0.1, 0.5, 10.0
SKIP = ('time', 'stamp', 'sec', 'nsec', 'seq', 'frame', 'range_max', 'range_min',
        'angle_min', 'angle_max', 'angle_increment', 'scan_time', 'time_increment', '_raw')
KEY = {'odom': ('pos_x', 'pos_y', 'yaw', 'linear_x', 'angular_z'),
       'imu': ('yaw', 'angular_vel_z', 'angular_velocity_z', 'linear_acc_x', 'linear_acc_y',
               'linear_acceleration_x', 'linear_acceleration_y')}


def sessions():
    reg = json.load(open(os.path.join(W.LOG_DIR, 'session_registry.json')))['good_sessions']
    ho_p = os.path.join(W.LOG_DIR, 'heldout_sessions.json')
    ho = json.load(open(ho_p))['heldout_sessions'] if os.path.exists(ho_p) else []
    return sorted(set(reg) | set(ho))


def spikes(x):
    """Isolated-spike test: r_i = x_i - (x_{i-1} + x_{i+1}) / 2 (distance from the line
    through the two neighbours). Smooth motion -> small r (curvature); noise -> noise-sized r;
    a single bad sample -> large r. Scale s = 1.4826 * MAD(r) over the session."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 50 or np.nanstd(x) == 0:
        return None
    r = x[1:-1] - 0.5 * (x[:-2] + x[2:])
    s = 1.4826 * np.median(np.abs(r - np.median(r)))
    floor = 1e-9 + 1e-9 * (np.nanmax(np.abs(x)) + 1)
    if s < floor:
        s = floor
    a = np.abs(r)
    peak = np.r_[True, a[1:] >= a[:-1]] & np.r_[a[:-1] >= a[1:], True]   # count each spike once
    hit = (a > Z * s) & peak
    mag = float(np.median(np.abs(r[hit])) / s) if hit.any() else 0.0
    return 100.0 * hit.mean(), int(hit.sum()), mag


def signals(df):
    out = {}
    if {'orient_z', 'orient_w'} <= set(df.columns):
        out['yaw'] = np.unwrap(np.arctan2(2 * df.orient_w * df.orient_z, 1 - 2 * df.orient_z ** 2))
    for c in df.columns:
        if c.startswith('range_') and c.split('_', 1)[1].isdigit():
            continue
        if any(k in c for k in SKIP) or c in ('orient_x', 'orient_y', 'orient_z', 'orient_w'):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out[c] = df[c].values
    return out


def main():
    rows = []
    for s in sessions():
        for sensor in ('odom', 'imu', 'cmd_vel'):
            for stage, pat in (('raw', f'{sensor}_{s}.csv'), ('filtered', f'{sensor}_filtered_{s}.csv')):
                p = os.path.join(W.LOG_DIR, pat)
                if not os.path.exists(p):
                    continue
                for name, x in signals(pd.read_csv(p)).items():
                    res = spikes(x)
                    if res:
                        rows.append(dict(session=s, sensor=sensor, stage=stage, signal=name,
                                         spike_pct=res[0], n_spikes=res[1], median_mag_x_noise=res[2]))
        # LiDAR: non-finite beams + spikes per beam, aggregated
        for stage, pat in (('raw', f'scan_{s}.csv'), ('filtered', f'scan_filtered_{s}.csv')):
            p = os.path.join(W.LOG_DIR, pat)
            if not os.path.exists(p):
                continue
            df = pd.read_csv(p)
            beams = [c for c in df.columns if c.startswith('range_') and c.split('_', 1)[1].isdigit()]
            arr = df[beams].values.astype(float)
            nonfin = 100.0 * (~np.isfinite(arr)).mean()
            pct = [spikes(arr[:, j])[0] for j in range(0, len(beams), 10) if spikes(arr[:, j])]
            rows.append(dict(session=s, sensor='scan', stage=stage, signal=f'beams (non-finite {nonfin:.2f}%)',
                             spike_pct=float(np.mean(pct)) if pct else 0.0, n_spikes=-1, median_mag_x_noise=np.nan))
    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(W.WS, 'results'), exist_ok=True)
    df.to_csv(os.path.join(W.WS, 'results', 'spike_diagnostic.csv'), index=False)

    pd.set_option('display.width', 200)
    summ = (df.groupby(['sensor', 'signal', 'stage'])
              .agg(mean_pct=('spike_pct', 'mean'), max_pct=('spike_pct', 'max'),
                   median_mag=('median_mag_x_noise', 'median'))
              .unstack('stage').round(3))
    print('\n=== Spike % per signal (mean / max over sessions; magnitude in x local noise) ===')
    print(summ.to_string())

    f = df[(df.stage == 'filtered')]
    key = f[f.apply(lambda r: r.sensor in KEY and r.signal in KEY[r.sensor], axis=1)]
    worst = key.groupby(['sensor', 'signal']).agg(pct=('spike_pct', 'max'), mag=('median_mag_x_noise', 'median'))
    print('\n=== Decision (filtered data, key signals; worst session) ===')
    print(worst.round(3).to_string())
    if worst.empty:
        print('No key signals found -- check column names (paste the summary).')
    elif (worst.pct < DONT).all():
        print(f'\nVERDICT: all key signals < {DONT}% spikes -> DON\'T implement an outlier filter.')
    elif ((worst.pct >= CONSIDER) & (worst.mag > BIG)).any():
        print(f'\nVERDICT: some key signal >= {CONSIDER}% spikes larger than {BIG}x noise -> CONSIDER a '
              f'local filter later (pre-registered, fresh test sessions).')
    else:
        print('\nVERDICT: small spike rates -> report as a limitation; no change.')
    print('Saved: results/spike_diagnostic.csv')


if __name__ == '__main__':
    main()
