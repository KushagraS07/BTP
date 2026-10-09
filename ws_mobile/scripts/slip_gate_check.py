#!/usr/bin/env python3
"""
slip_gate_check.py -- Phase 3B step B2: does the slip world really produce wheel slip in OUR
setup (ROS 2 Jazzy, Gazebo Harmonic 8.11, mobile_robot)? Read-only.

Patches (world frame, = indoor_world_slip.sdf):
  corridor  mu 0.10  x 2..4  y -1.5..1.5
  north     mu 0.30  x 3..6  y  3..6
  south     mu 0.05  x 3..6  y -6..-3
Events from the executed command (cmd_vel_filtered, linear_x):
  STOP   |v_cmd| from >= 0.10 to < 0.01 m/s, staying < 0.01 for >= 1 s.
         slide = ground-truth path length in the 2 s after the stop command;
         wheel = odometry path length in the same 2 s.
  START  |v_cmd| from < 0.01 (for >= 0.5 s) to >= 0.10 m/s.
         ratio = odometry / ground-truth path length in the 1.5 s after the command
         (only if ground truth moved >= 2 cm).
Location of an event = ground-truth position at the command time.

GATE RULE (fixed before the first slip recording):
  PASS if, on the corridor patch (mu 0.10), at least one STOP has slide >= 0.05 m
       OR at least one START has ratio >= 1.20.
  CONTROL (reported, WARN if violated): median slide of off-patch STOPs < 0.03 m.

Usage: python3 slip_gate_check.py SESSION
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ws_paths as W  # noqa: E402

LOG = W.LOG_DIR
PATCHES = {'corridor_mu0.10': (2.0, 4.0, -1.5, 1.5), 'north_mu0.30': (3.0, 6.0, 3.0, 6.0),
           'south_mu0.05': (3.0, 6.0, -6.0, -3.0)}
V_MOVE, V_STILL = 0.10, 0.01
STOP_HOLD, START_QUIET = 1.0, 0.5
STOP_WIN, START_WIN = 2.0, 1.5
SLIDE_PASS, RATIO_PASS, MIN_GT_MOVE, CONTROL_SLIDE = 0.05, 1.20, 0.02, 0.03


def tstamp(df):
    if 'timestamp' in df.columns:
        return df['timestamp'].values.astype(float)
    return df['time_sec'].values + df['time_nsec'].values * 1e-9


def where(x, y):
    for name, (x0, x1, y0, y1) in PATCHES.items():
        if x0 <= x <= x1 and y0 <= y <= y1:
            return name
    return 'off_patch'


def path_len(x, y):
    return float(np.sum(np.hypot(np.diff(x), np.diff(y)))) if len(x) > 1 else 0.0


def main(s):
    g = pd.read_csv(os.path.join(LOG, f'ground_truth_filtered_{s}.csv'))
    o = pd.read_csv(os.path.join(LOG, f'odom_filtered_{s}.csv'))
    c = pd.read_csv(os.path.join(LOG, f'cmd_vel_filtered_{s}.csv'))
    tg, to, tc = tstamp(g), tstamp(o), tstamp(c)
    gx, gy = g.gt_x.values, g.gt_y.values
    ox, oy = np.interp(tg, to, o.pos_x.values), np.interp(tg, to, o.pos_y.values)
    v = np.abs(np.interp(tg, tc, c.linear_x.values))
    dt = np.median(np.diff(tg))
    print(f'session {s}: {tg[-1] - tg[0]:.1f} s, GT path {path_len(gx, gy):.2f} m, '
          f'odom path {path_len(ox, oy):.2f} m')
    visited = sorted({where(a, b) for a, b in zip(gx, gy)})
    print(f'areas visited: {visited}')

    rows = []
    n = len(tg)
    for i in range(1, n):
        if v[i - 1] >= V_MOVE and v[i] < V_STILL:                      # STOP
            j = i + int(STOP_HOLD / dt)
            if j < n and np.all(v[i:j] < V_STILL):
                w = (tg >= tg[i]) & (tg <= tg[i] + STOP_WIN)
                rows.append(dict(event='STOP', t=tg[i] - tg[0], area=where(gx[i], gy[i]),
                                 gt_m=path_len(gx[w], gy[w]), wheel_m=path_len(ox[w], oy[w])))
        if v[i - 1] < V_STILL and v[i] >= V_MOVE:                      # START
            k = i - int(START_QUIET / dt)
            if k >= 0 and np.all(v[k:i] < V_STILL):
                w = (tg >= tg[i]) & (tg <= tg[i] + START_WIN)
                rows.append(dict(event='START', t=tg[i] - tg[0], area=where(gx[i], gy[i]),
                                 gt_m=path_len(gx[w], gy[w]), wheel_m=path_len(ox[w], oy[w])))
    ev = pd.DataFrame(rows)
    if ev.empty:
        sys.exit('No STOP/START events found -- drive with clear stops (k) and restarts (i).')
    ev['ratio'] = np.where(ev.gt_m >= MIN_GT_MOVE, ev.wheel_m / ev.gt_m.clip(lower=1e-9), np.nan)
    print('\nevents (gt_m / wheel_m = ground-truth / odometry path length in m over the window):')
    print(ev.round(3).to_string(index=False))

    cor = ev[ev.area == 'corridor_mu0.10']
    slide_ok = bool(((cor.event == 'STOP') & (cor.gt_m >= SLIDE_PASS)).any())
    spin_ok = bool(((cor.event == 'START') & (cor.ratio >= RATIO_PASS)).any())
    off = ev[(ev.area == 'off_patch') & (ev.event == 'STOP')]
    ctrl = float(off.gt_m.median()) if len(off) else float('nan')
    print(f'\ncorridor patch: {int((cor.event == "STOP").sum())} stops, {int((cor.event == "START").sum())} starts')
    print(f'  slide >= {SLIDE_PASS} m on a stop : {"YES" if slide_ok else "no"}')
    print(f'  wheel/GT >= {RATIO_PASS} on a start: {"YES" if spin_ok else "no"}')
    print(f'control: median off-patch stop slide = {ctrl:.3f} m '
          f'({"OK" if ctrl < CONTROL_SLIDE else "WARN" if ctrl == ctrl else "no off-patch stops"})')
    print(f'\nGATE: {"PASS" if (slide_ok or spin_ok) else "FAIL"}')
    if cor.empty:
        print('  (no events on the corridor patch: stop and restart ON the blue patch at x = 2..4 m)')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
