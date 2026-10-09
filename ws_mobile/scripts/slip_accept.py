#!/usr/bin/env python3
"""
slip_accept.py -- Phase 3B session acceptance: enforces the driving protocol of PREREG_slip.md §3.
Read-only. Run AFTER `final_analysis.py check-sessions` and `slip_sessions.py add`, BEFORE any EKF or
model output exists for the session. Exit code 0 = ACCEPT, 1 = REJECT.

Protocol checks (all must hold; thresholds inclusive):
  P1 duration 60.0–100.0 s (last − first odometry timestamp, raw odom_<s>.csv)
  P2 >= 2 different patches visited (ground-truth centre inside a patch for >= 1.0 s in total each)
  P3 >= 2 STOP events on a patch      (event definitions = slip_gate_check.py)
  P4 >= 2 START events on a patch
  P5 >= 1 TURN on a patch: |angular_z command| >= 0.5 rad/s held >= 0.5 s, GT centre on a patch
     at the turn's first sample
  P6 >= 1 STOP event off-patch
Any missing/unreadable file or error => REJECT (never "undetermined").
Usage: python3 slip_accept.py SESSION
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ws_paths as W  # noqa: E402
from slip_gate_check import (START_QUIET, STOP_HOLD, V_MOVE, V_STILL, tstamp,  # noqa: E402
                             where)

LOG = W.LOG_DIR
DUR_MIN, DUR_MAX = 60.0, 100.0
VISIT_MIN = 1.0
TURN_W, TURN_HOLD = 0.5, 0.5


def evaluate(s):
    odom_raw = pd.read_csv(os.path.join(LOG, f'odom_{s}.csv'))
    g = pd.read_csv(os.path.join(LOG, f'ground_truth_filtered_{s}.csv'))
    c = pd.read_csv(os.path.join(LOG, f'cmd_vel_filtered_{s}.csv'))
    tr = tstamp(odom_raw)
    dur = float(tr[-1] - tr[0])
    tg, tc = tstamp(g), tstamp(c)
    gx, gy = g.gt_x.values, g.gt_y.values
    area = np.array([where(a, b) for a, b in zip(gx, gy)])
    # zero-order hold of the (already zero-order-held) command onto the GT grid
    idx = np.clip(np.searchsorted(tc, tg, side='right') - 1, 0, len(tc) - 1)
    v = np.abs(c.linear_x.values[idx])
    w = np.abs(c.angular_z.values[idx])
    dt = float(np.median(np.diff(tg)))
    n = len(tg)

    visits = {}
    for a in set(area) - {'off_patch'}:
        visits[a] = float((area == a).sum() * dt)
    patches_visited = [a for a, sec in visits.items() if sec >= VISIT_MIN]

    stops_on = stops_off = starts_on = turns_on = 0
    for i in range(1, n):
        if v[i - 1] >= V_MOVE and v[i] < V_STILL:
            j = i + int(STOP_HOLD / dt)
            if j < n and np.all(v[i:j] < V_STILL):
                if area[i] == 'off_patch':
                    stops_off += 1
                else:
                    stops_on += 1
        if v[i - 1] < V_STILL and v[i] >= V_MOVE:
            k = i - int(START_QUIET / dt)
            if k >= 0 and np.all(v[k:i] < V_STILL) and area[i] != 'off_patch':
                starts_on += 1
        if w[i - 1] < TURN_W <= w[i]:
            j = i + int(TURN_HOLD / dt)
            if j <= n and np.all(w[i:j] >= TURN_W) and area[i] != 'off_patch':
                turns_on += 1

    checks = [
        ('P1 duration 60–100 s', DUR_MIN <= dur <= DUR_MAX, f'{dur:.1f} s'),
        ('P2 >= 2 patches visited (>= 1 s each)', len(patches_visited) >= 2,
         ', '.join(f'{a} {sec:.1f}s' for a, sec in sorted(visits.items())) or 'none'),
        ('P3 >= 2 stops on a patch', stops_on >= 2, str(stops_on)),
        ('P4 >= 2 starts on a patch', starts_on >= 2, str(starts_on)),
        ('P5 >= 1 turn on a patch', turns_on >= 1, str(turns_on)),
        ('P6 >= 1 stop off-patch', stops_off >= 1, str(stops_off)),
    ]
    return checks


def main(s):
    try:
        checks = evaluate(s)
    except Exception as e:  # missing / unreadable data => REJECT
        print(f'[REJECT] {s}: could not evaluate ({type(e).__name__}: {e})')
        return 1
    ok = all(c[1] for c in checks)
    for name, passed, val in checks:
        print(f'  [{"ok" if passed else "FAIL"}] {name}: {val}')
    print(f'[{"ACCEPT" if ok else "REJECT"}] {s}')
    return 0 if ok else 1


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
