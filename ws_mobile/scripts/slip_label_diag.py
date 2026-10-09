#!/usr/bin/env python3
"""
slip_label_diag.py -- Phase 3B, DEVELOPMENT ONLY, read-only. Describes WHERE the fixed slip label
(PREREG_slip.md §4) fires off-patch. It changes nothing: the label rule is frozen in Part A and is
not tuned from this output. Used only to describe the label honestly in the findings / paper.

For every off-patch step labelled slip = 1 it reports which of these hold (not exclusive):
  near_edge   GT centre within EDGE_M of a patch rectangle (robot body still on / just left it)
  turning     |commanded w| >= 0.5 rad/s (zero-order-held cmd_vel_filtered)
  transient   within TRANS_S of a command change (|dv_cmd| >= 0.05 m/s or |dw_cmd| >= 0.25 rad/s)
  reversing   commanded v < -0.01 m/s
  dv / dw     which threshold triggered (|dv| >= 0.10, |dw| >= 0.30)
  none        none of near_edge / turning / transient / reversing
Usage: python3 slip_label_diag.py SESSION [SESSION ...]
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ws_paths as W  # noqa: E402
from slip_gate_check import PATCHES, tstamp  # noqa: E402

LOG = W.LOG_DIR
EDGE_M, TRANS_S = 0.30, 0.50     # descriptive only
V_THR, W_THR = 0.10, 0.30        # = PREREG_slip.md §4


def dist_to_patches(x, y):
    d = np.full(len(x), np.inf)
    for x0, x1, y0, y1 in PATCHES.values():
        dx = np.maximum.reduce([x0 - x, np.zeros_like(x), x - x1])
        dy = np.maximum.reduce([y0 - y, np.zeros_like(y), y - y1])
        d = np.minimum(d, np.hypot(dx, dy))
    return d


def one(s):
    L = pd.read_csv(os.path.join(LOG, f'slip_labels_{s}.csv'))
    c = pd.read_csv(os.path.join(LOG, f'cmd_vel_filtered_{s}.csv'))
    t, tc = L.timestamp.values, tstamp(c)
    idx = np.clip(np.searchsorted(tc, t, side='right') - 1, 0, len(tc) - 1)
    v, w = c.linear_x.values[idx], c.angular_z.values[idx]
    dt = float(np.median(np.diff(t)))
    k = max(1, int(round(TRANS_S / dt)))
    chg = (np.abs(np.diff(v, prepend=v[0])) >= 0.05) | (np.abs(np.diff(w, prepend=w[0])) >= 0.25)
    trans = np.convolve(chg.astype(int), np.ones(2 * k + 1, int), mode='same') > 0

    off = (L.area.values == 'off_patch') & (L.slip.values == 1)
    n_off_steps = int((L.area.values == 'off_patch').sum())
    near = dist_to_patches(L.gt_x.values, L.gt_y.values) <= EDGE_M
    turn = np.abs(w) >= 0.5
    rev = v < -0.01
    dv_t, dw_t = L.dv.values >= V_THR, L.dw.values >= W_THR
    m = off
    n = int(m.sum())

    def pct(a):
        return round(100 * float((a & m).sum()) / max(n, 1), 1)
    none = ~(near | turn | trans | rev)
    return dict(session=s, offpatch_steps=n_off_steps, offpatch_slip_steps=n,
                near_edge=pct(near), turning=pct(turn), transient=pct(trans), reversing=pct(rev),
                none_of_these=pct(none), dv_trig=pct(dv_t), dw_trig=pct(dw_t),
                offpatch_rate_excl_edge_turn_trans=round(
                    100 * float((m & none).sum()) / max(int(((L.area.values == 'off_patch') & none).sum()), 1), 2))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    df = pd.DataFrame([one(s) for s in sys.argv[1:]])
    print('% of off-patch slip steps with each property (not exclusive):')
    print(df.to_string(index=False))
