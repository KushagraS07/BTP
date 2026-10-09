#!/usr/bin/env python3
"""
apply_42_patch.py -- deferred item 4.2: inject_lidar_scan_corruption() records WHICH
frames it corrupted ('corrupted_row_indices') in its stats/manifest, not only how many.

Effect: only NEW LiDAR-corruption fault sessions get the extra field. The corruption
itself is unchanged (same RNG calls, same rows, same values), so no existing result changes.

Safety: anchor checked first (nothing changed on mismatch); backup to
fix_ws/fault_injection.py.bak_pre42; py_compile; then a self-test on a small synthetic
scan proving (a) the recorded indices are exactly the rows that changed and (b) the
corrupted output is identical to the unpatched function's output for the same seed.
"""
import importlib
import os
import py_compile
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
import ws_paths as W  # noqa: E402

P = os.path.join(W.SCRIPTS_DIR, 'fault_injection.py')
BAK = os.path.join(W.FIX_DIR, 'fault_injection.py.bak_pre42')

OLD = """    stats = {
        'fraction_frames_corrupted': float(frac),
        'n_frames_corrupted': int(n_corrupt),
        'range_max_valid_used': float(range_max_valid),
    }
"""
NEW = """    stats = {
        'fraction_frames_corrupted': float(frac),
        'n_frames_corrupted': int(n_corrupt),
        'range_max_valid_used': float(range_max_valid),
        # item 4.2: exact frames corrupted, for per-row r_LiDAR labels (Phase 3)
        'corrupted_row_indices': sorted(int(i) for i in corrupt_idx),
    }
"""

s = open(P).read()
if 'corrupted_row_indices' in s:
    sys.exit('fault_injection.py already has corrupted_row_indices. Nothing done.')
if s.count(OLD) != 1:
    sys.exit(f'ABORT: anchor found {s.count(OLD)} times (need 1). Nothing changed.')


def run(fi):
    n, beams = 200, 8
    rng0 = np.random.default_rng(1)
    df = pd.DataFrame(rng0.uniform(0.5, 2.0, size=(n, beams)), columns=[f'range_{i}' for i in range(beams)])
    df['range_max'] = 2.5
    cols = [f'range_{i}' for i in range(beams)]
    out, st = fi.inject_lidar_scan_corruption(df, cols, 0.3, np.random.default_rng(7))
    return df, cols, out, st


sys.path.insert(0, W.SCRIPTS_DIR)
import fault_injection as FI  # noqa: E402
_, _, out_before, _ = run(FI)

shutil.copy2(P, BAK)
open(P, 'w').write(s.replace(OLD, NEW))
try:
    py_compile.compile(P, doraise=True)
    FI = importlib.reload(FI)
    df, cols, out_after, st = run(FI)
    changed = sorted(np.where((df[cols].values != out_after[cols].values).any(axis=1))[0].tolist())
    assert st['corrupted_row_indices'] == changed, 'recorded indices != changed rows'
    assert st['n_frames_corrupted'] == len(changed), 'count mismatch'
    assert np.array_equal(out_before[cols].values, out_after[cols].values), 'output changed vs unpatched'
except Exception as e:
    shutil.copy2(BAK, P)
    sys.exit(f'ABORT: self-test failed ({e}). fault_injection.py restored from backup.')

print('Patched scripts/fault_injection.py (backup: fix_ws/fault_injection.py.bak_pre42)')
print(f'Self-test OK: {len(changed)} corrupted rows recorded exactly; corruption output identical to before.')
