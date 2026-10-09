#!/usr/bin/env python3
"""
verify_corruption_indices.py -- item 4.2. READ-ONLY on ~/ws_mobile.
Lives in and writes only to ~/fix_ws.

For every EXISTING LiDAR corruption fault session, recovers which rows were
corrupted in two independent ways and checks that they agree:
  (a) DIFF:   rows where any range_* beam differs from the baseline scan file
  (b) REPLAY: np.random.default_rng(seed).choice(n, n_corrupt, replace=False)
              -- the exact call inject_lidar_scan_corruption() makes, which is
              the FIRST use of the rng created in make_fault_session()

If they agree for every session, the exact per-row labels for Phase 3 can be
recovered without regenerating any fault session. The indices are written to
~/ws_mobile/fix_ws/corrupted_row_indices.json. Nothing in ws_mobile is modified.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

LOG_DIR = os.path.expanduser('~/ws_mobile/sensor_logs')
OUT = os.path.expanduser('~/ws_mobile/fix_ws/corrupted_row_indices.json')


def range_cols(df):
    return sorted([c for c in df.columns
                   if c.startswith('range_') and c.split('_', 1)[1].isdigit()],
                  key=lambda c: int(c.split('_')[1]))


metas = sorted(glob.glob(os.path.join(LOG_DIR, 'fault_meta_*_fault_lidar_corruption_*.json')))
if not metas:
    sys.exit(f'No LiDAR corruption manifests found in {LOG_DIR}')

recovered, n_agree, n_total = {}, 0, 0
print(f'{"fault_session":60s} {"n_rows":>7s} {"n_corrupt":>9s} {"n_diff":>7s}  agree')
for mp in metas:
    with open(mp) as f:
        m = json.load(f)
    fs, b, st = m['fault_session'], m['baseline_session'], m['injected_stats']
    n_total += 1

    base_p = os.path.join(LOG_DIR, f'scan_filtered_{b}.csv')
    flt_p = os.path.join(LOG_DIR, f'scan_filtered_{fs}.csv')
    if not (os.path.exists(base_p) and os.path.exists(flt_p)):
        print(f'{fs:60s}  MISSING scan file(s)')
        continue
    base, flt = pd.read_csv(base_p), pd.read_csv(flt_p)
    rc = range_cols(base)
    if len(base) != len(flt) or rc != range_cols(flt):
        print(f'{fs:60s}  SHAPE MISMATCH base={base.shape} fault={flt.shape}')
        continue

    a, c = base[rc].to_numpy(float), flt[rc].to_numpy(float)
    same = np.isclose(a, c, rtol=0.0, atol=1e-9, equal_nan=True)
    diff_idx = np.where(~same.all(axis=1))[0]

    n, n_corrupt = len(base), int(st['n_frames_corrupted'])
    rng = np.random.default_rng(m['seed'])
    replay_idx = np.sort(rng.choice(n, size=n_corrupt, replace=False))

    agree = np.array_equal(diff_idx, replay_idx)
    n_agree += agree
    print(f'{fs:60s} {n:7d} {n_corrupt:9d} {len(diff_idx):7d}  {"YES" if agree else "NO"}')

    recovered[fs] = {
        'baseline_session': b,
        'n_rows': n,
        'n_frames_corrupted_manifest': n_corrupt,
        'diff_and_replay_agree': bool(agree),
        'corrupted_row_indices': diff_idx.tolist(),
        'replay_row_indices': None if agree else replay_idx.tolist(),
    }

with open(OUT, 'w') as f:
    json.dump(recovered, f, indent=1)
print(f'\n{n_agree}/{n_total} sessions: diff == replay. Saved: {OUT}')
