#!/usr/bin/env python3
"""
slip_explore.py -- Phase 3B, DEVELOPMENT ONLY (read-only; no decisions are final here).
For a slip session: what fraction of time steps would be labelled "slip" under candidate label
rules, and where (on/off patch)? Used to choose the label threshold BEFORE PREREG_slip.md is
frozen. Never run on test sessions.

Wheel speeds: odometry linear_x / angular_z if present, else derived from odom pose.
True speeds: forward speed and yaw rate from ground-truth pose (projected on GT heading).
Usage: python3 slip_explore.py SESSION [SESSION ...]
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ws_paths as W  # noqa: E402
from slip_gate_check import PATCHES, tstamp, where  # noqa: E402

LOG = W.LOG_DIR
V_THR = [0.05, 0.10, 0.20]     # m/s   |v_wheel - v_true|
W_THR = [0.10, 0.20, 0.30]     # rad/s |w_wheel - w_true|


def yaw(qz, qw):
    return np.unwrap(2 * np.arctan2(qz, qw))


def speeds(s):
    g = pd.read_csv(os.path.join(LOG, f'ground_truth_filtered_{s}.csv'))
    o = pd.read_csv(os.path.join(LOG, f'odom_filtered_{s}.csv'))
    tg, to = tstamp(g), tstamp(o)
    th = yaw(g.gt_orient_z.values, g.gt_orient_w.values)
    dt = np.gradient(tg)
    vx, vy = np.gradient(g.gt_x.values) / dt, np.gradient(g.gt_y.values) / dt
    v_true = vx * np.cos(th) + vy * np.sin(th)
    w_true = np.gradient(th) / dt
    if {'linear_x', 'angular_z'} <= set(o.columns):
        v_wh = np.interp(tg, to, o.linear_x.values)
        w_wh = np.interp(tg, to, o.angular_z.values)
        src = 'odom linear_x/angular_z'
    else:
        tho = yaw(o.orient_z.values, o.orient_w.values)
        dto = np.gradient(to)
        ovx, ovy = np.gradient(o.pos_x.values) / dto, np.gradient(o.pos_y.values) / dto
        v_wh = np.interp(tg, to, ovx * np.cos(tho) + ovy * np.sin(tho))
        w_wh = np.interp(tg, to, np.gradient(tho) / dto)
        src = 'derived from odom pose'
    area = np.array([where(a, b) for a, b in zip(g.gt_x.values, g.gt_y.values)])
    return v_true, w_true, v_wh, w_wh, area, src


def main(sessions):
    for s in sessions:
        v_t, w_t, v_w, w_w, area, src = speeds(s)
        dv, dw = np.abs(v_w - v_t), np.abs(w_w - w_t)
        on = area != 'off_patch'
        print(f'\n=== {s}  ({len(dv)} steps, {100 * on.mean():.1f}% on a patch; wheel speeds: {src})')
        print(f'  |dv| median off-patch {np.median(dv[~on]):.3f}, on-patch {np.median(dv[on]) if on.any() else float("nan"):.3f} m/s;'
              f'  |dw| median off {np.median(dw[~on]):.3f}, on {np.median(dw[on]) if on.any() else float("nan"):.3f} rad/s')
        print(f'  {"rule":28s} {"slip %":>7s} {"% of slip on patch":>19s} {"off-patch slip %":>17s}')
        for vt in V_THR:
            for wt in W_THR:
                lab = (dv >= vt) | (dw >= wt)
                share_on = 100 * (lab & on).sum() / max(lab.sum(), 1)
                off_rate = 100 * lab[~on].mean() if (~on).any() else float('nan')
                print(f'  |dv|>={vt:.2f} or |dw|>={wt:.2f}    {100 * lab.mean():6.1f} {share_on:18.1f} {off_rate:16.1f}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
