#!/usr/bin/env python3
"""
fs_check.py -- full system: recording check (read-only). Reports, for one full-system session:
  rates     median rate and gaps of imu / odom / scan / ground truth / cmd_vel (raw CSVs)
  imu       check_imu_integration pass rule (|∫a dt − Δv_gt| ≤ 0.02 m/s, straight starts/stops)
  bias      mean gyro_z over ground-truth-still samples (≈ gyro bias), with count
F5 dev-session rule: all topics, IMU check PASS, IMU missing < 1 %, duration 60-120 s.
F5a (amendment 2026-10-10): the IMU check verifies the RECORDING, so it is judged on straight
  starts/stops on flat floor (incl. low-friction patches); events whose 1.1 s window touches a
  rumble-strip or ramp zone of the terrain world (--world terrain) are excluded and listed;
  at least 3 flat events are required.
Decision rules fixed BEFORE the recordings (FINDINGS_fullsys.md F4):
  odom_rate 100 adopted if odom median ≥ 90 Hz AND the IMU check passes AND IMU gaps < 1 %
  gyro-bias variant (gyro_bias_z:=0.01) works if the still-period gyro mean is in [0.009, 0.011]
Usage: python3 tools/fs_check.py SESSION [--data <ws_mobile>/fullsys_logs/raw]
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from zupt import paths as _P  # noqa: E402
from zupt.replay import gt_truth, load_session  # noqa: E402


# terrain_world (tools/gen_terrain_world.py): lane i centre y = 3.8 i, half width 1.8 m;
# rough / tilted zones in x (m): lane 3 rumble strips 4-12, lane 4 ramp up / plateau / ramp down
# 4-12, lane 5 rumble strips 6-8. Margin 0.3 m.
LANE_PITCH, LANE_HALF, ZONE_MARGIN = 3.8, 1.9, 0.3
ROUGH = {3: [(4.0, 12.0)], 4: [(4.0, 12.0)], 5: [(6.0, 8.0)]}


def in_rough(x, y):
    i = int(round(y / LANE_PITCH))
    if abs(y - LANE_PITCH * i) > LANE_HALF:
        return False
    return any(a - ZONE_MARGIN <= x <= b + ZONE_MARGIN for a, b in ROUGH.get(i, []))


def rate(path):
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path, usecols=['time_sec', 'time_nsec'])
    t = d.time_sec.values + d.time_nsec.values * 1e-9
    if len(t) < 3:
        return len(t), float('nan'), 0, 0.0
    dt = np.diff(t)
    r = 1.0 / float(np.median(dt))
    gaps = int((dt > 2.0 / r).sum())
    expected = (t[-1] - t[0]) * r + 1
    return len(t), r, gaps, 100.0 * max(0.0, 1 - len(t) / expected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session')
    ap.add_argument('--data', default=_P.DATA)
    ap.add_argument('--world', choices=('indoor', 'terrain'), default='indoor')
    a = ap.parse_args()
    s = a.session
    print(f'session {s}  ({a.data})')
    res = {}
    for name in ('imu', 'odom', 'scan', 'ground_truth', 'cmd_vel'):
        r = rate(os.path.join(a.data, f'{name}_{s}.csv'))
        res[name] = r
        if r is None:
            print(f'  {name:13s} MISSING')
        elif name == 'cmd_vel':
            print(f'  {name:13s} {r[0]:6d} msgs (event-driven)')
        else:
            print(f'  {name:13s} {r[0]:6d} msgs  median {r[1]:7.1f} Hz  gaps {r[2]:4d}  missing ~{r[3]:4.1f} %')

    p = subprocess.run([sys.executable, os.path.join(HERE, 'check_imu_integration.py'), '--session', s,
                        '--data', a.data], capture_output=True, text=True)
    rows = []
    for l in p.stdout.splitlines():
        f = l.split()
        if len(f) == 5:
            try:
                rows.append([float(v) for v in f])
            except ValueError:
                pass
    g = pd.read_csv(os.path.join(a.data, f'ground_truth_{s}.csv'))
    g = g[~g['child_frame_id'].isin(['base_footprint', 'base_link'])] if 'child_frame_id' in g else g
    tg = g.time_sec.values + g.time_nsec.values * 1e-9
    flat, rough = [], []
    for t0, _, _, _, err in rows:
        m = (tg >= t0 - 0.1) & (tg <= t0 + 1.0)
        r = a.world == 'terrain' and any(in_rough(x, y) for x, y in zip(g.gt_x.values[m], g.gt_y.values[m]))
        (rough if r else flat).append((t0, err))
    worst = max((abs(e) for _, e in flat), default=float('nan'))
    imu_ok = len(flat) >= 3 and worst <= 0.02
    print(f'  imu check     {len(flat)} flat-floor starts/stops, worst {worst:.3f} m/s -> '
          f'{"PASS" if imu_ok else "FAIL"} (<= 0.02, >= 3 events)')
    if rough:
        print('                excluded (rough/ramp zone, reported only): '
              + ', '.join(f't={t:.1f} err={e:+.3f}' for t, e in rough))
    dur = float(tg.max() - tg.min())
    ok = all(res[k] is not None for k in res) and imu_ok and res['imu'][3] < 1.0 and 60.0 <= dur <= 120.0
    print(f'  duration      {dur:.1f} s (simulation time)')
    print(f'  F5/F5a rule   (all topics, flat IMU check PASS, IMU missing < 1 %, 60-120 s): '
          f'{"KEEP" if ok else "FAIL -> re-record"}')

    imu, odom, cmd, gt = load_session(a.data, s)
    still, _ = gt_truth(imu.t.values, gt)
    if still.sum() >= 50:
        g = imu.angular_vel_z.values[still]
        print(f'  gyro_z while GT-still: mean {g.mean():+.5f} rad/s  (std of mean {g.std() / np.sqrt(len(g)):.5f}, '
              f'n = {len(g)} samples at 50 Hz)')
    else:
        print('  gyro_z while GT-still: too few still samples')


if __name__ == '__main__':
    main()
