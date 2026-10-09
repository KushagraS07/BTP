#!/usr/bin/env python3
"""
slip_prepare.py -- Phase 3B step B5: inputs and labels for the slip sessions.

For every selected slip session, in this order, using the UNCHANGED Phase 2 code:
  1. ekf_fusion.py <s>                                   -> ekf_fused_output_<s>.csv (fixed EKF)
  2. generate_reliability_labels.process_baseline_session -> reliability_labels_<s>.csv (r = 1;
     needed only because build_session_features checks it; never used by P_slip)
  3. build_training_dataset.build_session_features       -> features_<s>.csv (the 13 P_slip inputs)
  4. slip label of PREREG_slip.md §4                     -> slip_labels_<s>.csv
Nothing of Phase 2 (models, labels, results, good_sessions) is read for writing or changed.

Session selection (PREREG_slip.md Part B rule): a session is used only if its role in
sensor_logs/slip_sessions.json matches AND `slip_accept.py <s>` exits 0. The gate session is
taken by role only (it was recorded before the protocol). Expected counts are asserted:
gate 1, dev 8, calib 3, test 6 -- any other count => ABORT, nothing written.

Usage:
  python3 slip_prepare.py                       # roles gate dev calib (default)
  python3 slip_prepare.py --roles test          # only after Part B is frozen
  python3 slip_prepare.py --labels-only         # step 4 only (steps 1-3 must exist)
Writes results/slip_prepare_summary_<roles>.csv.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ws_paths as W  # noqa: E402
from slip_gate_check import tstamp, where  # noqa: E402

S, LOG, RES = W.SCRIPTS_DIR, W.LOG_DIR, os.path.join(W.WS, 'results')
PY = sys.executable
SLIP_JSON = os.path.join(LOG, 'slip_sessions.json')
EXPECTED = {'gate': 1, 'dev': 8, 'calib': 3, 'test': 6}
# --- PREREG_slip.md §4 (fixed) ---
V_THR, W_THR = 0.10, 0.30   # m/s, rad/s


def accepted(s):
    p = subprocess.run([PY, os.path.join(S, 'slip_accept.py'), s], cwd=S,
                       capture_output=True, text=True, timeout=300)
    return p.returncode == 0


def select(roles):
    info = json.load(open(SLIP_JSON))['info']
    chosen, rejected = {}, []
    for r in roles:
        cand = sorted(s for s, i in info.items() if i.get('role') == r)
        if r == 'gate':
            ok = cand
        else:
            ok = [s for s in cand if accepted(s)]
            rejected += [(s, r) for s in cand if s not in ok]
        if len(ok) != EXPECTED[r]:
            sys.exit(f'ABORT: role {r}: {len(ok)} accepted sessions, expected {EXPECTED[r]}: {ok}. '
                     'Nothing written.')
        chosen[r] = ok
    for s, r in rejected:
        print(f'  excluded (role {r}, slip_accept REJECT): {s}')
    return chosen


def yaw(qz, qw):
    return np.unwrap(2 * np.arctan2(qz, qw))


def slip_labels(s):
    """PREREG_slip.md §4, on the rows of ekf_fused_output_<s>.csv."""
    grid = pd.read_csv(os.path.join(LOG, f'ekf_fused_output_{s}.csv'), usecols=['timestamp'])
    t = grid.timestamp.values.astype(float)
    g = pd.read_csv(os.path.join(LOG, f'ground_truth_filtered_{s}.csv'))
    o = pd.read_csv(os.path.join(LOG, f'odom_filtered_{s}.csv'))
    tg, to = tstamp(g), tstamp(o)

    th = yaw(g.gt_orient_z.values, g.gt_orient_w.values)
    vx = np.gradient(g.gt_x.values, tg)
    vy = np.gradient(g.gt_y.values, tg)
    v_true = np.interp(t, tg, vx * np.cos(th) + vy * np.sin(th))
    w_true = np.interp(t, tg, np.gradient(th, tg))
    v_wh = np.interp(t, to, o.linear_x.values.astype(float))
    w_wh = np.interp(t, to, o.angular_z.values.astype(float))
    gx, gy = np.interp(t, tg, g.gt_x.values), np.interp(t, tg, g.gt_y.values)

    dv, dw = np.abs(v_wh - v_true), np.abs(w_wh - w_true)
    valid = ((t >= tg[0]) & (t <= tg[-1]) & (t >= to[0]) & (t <= to[-1])
             & np.isfinite(dv) & np.isfinite(dw))
    slip = np.where(valid, ((dv >= V_THR) | (dw >= W_THR)).astype(float), np.nan)
    area = np.array([where(a, b) for a, b in zip(gx, gy)])

    out = pd.DataFrame({'timestamp': t, 'v_wheel': v_wh, 'v_true': v_true, 'w_wheel': w_wh,
                        'w_true': w_true, 'dv': dv, 'dw': dw, 'gt_x': gx, 'gt_y': gy,
                        'area': area, 'slip': slip})
    out.to_csv(os.path.join(LOG, f'slip_labels_{s}.csv'), index=False)

    on = area != 'off_patch'
    lab = slip == 1
    return dict(rows=len(t), invalid=int((~valid).sum()),
                slip_pct=round(float(100 * np.nanmean(slip)), 2),
                slip_on_patch_pct=round(float(100 * (lab & on).sum() / max(lab.sum(), 1)), 1),
                offpatch_slip_rate_pct=round(float(100 * np.nanmean(slip[~on])), 2) if (~on).any() else np.nan,
                onpatch_slip_rate_pct=round(float(100 * np.nanmean(slip[on])), 2) if on.any() else np.nan)


def build_inputs(s):
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    p = subprocess.run([PY, os.path.join(S, 'ekf_fusion.py'), s], cwd=S, env=env,
                       capture_output=True, text=True, timeout=600)
    if p.returncode != 0 or not os.path.exists(os.path.join(LOG, f'ekf_fused_output_{s}.csv')):
        sys.exit(f'ABORT ekf_fusion {s}: exit {p.returncode}\n{p.stderr[-800:]}')
    import generate_reliability_labels as G
    G.process_baseline_session(s)
    import build_training_dataset as BT
    _, n = BT.build_session_features(s)
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--roles', nargs='+', default=['gate', 'dev', 'calib'], choices=list(EXPECTED))
    ap.add_argument('--labels-only', action='store_true')
    a = ap.parse_args()

    chosen = select(a.roles)
    for r, ss in chosen.items():
        print(f'{r}: {len(ss)} sessions: {" ".join(ss)}')

    rows = []
    for r, ss in chosen.items():
        for s in ss:
            n_feat = None
            if not a.labels_only:
                print(f'\n[{r}] {s}: EKF + features ...', flush=True)
                n_feat = build_inputs(s)
            st = slip_labels(s)
            n_f = len(pd.read_csv(os.path.join(LOG, f'features_{s}.csv'), usecols=['timestamp']))
            if n_f != st['rows']:
                sys.exit(f'ABORT {s}: features_{s}.csv has {n_f} rows, labels {st["rows"]}')
            rows.append(dict(session=s, role=r, **st))
            print(f'[{r}] {s}: rows {st["rows"]} (features {n_f}{"" if n_feat is None else ", built"}), '
                  f'invalid {st["invalid"]}, slip {st["slip_pct"]}%, slip on patch {st["slip_on_patch_pct"]}%, '
                  f'on-patch rate {st["onpatch_slip_rate_pct"]}%, off-patch rate {st["offpatch_slip_rate_pct"]}%',
                  flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(RES, exist_ok=True)
    out = os.path.join(RES, f'slip_prepare_summary_{"_".join(a.roles)}.csv')
    df.to_csv(out, index=False)
    print('\n' + df.to_string(index=False))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
