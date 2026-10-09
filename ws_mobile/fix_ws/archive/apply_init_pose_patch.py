#!/usr/bin/env python3
"""
apply_init_pose_patch.py -- one-time patch to the Phase 1 / Phase 2 gate scripts:
  ~/ws_mobile/scripts/compare_adaptive_vs_fixed.py   (Phase 1)
  ~/ws_mobile/scripts/lstm_adaptive_ekf.py           (Phase 2)

Problem: both gate scripts start every filter (fixed, heuristic, LSTM) at the
origin (0, 0, 0), while the validated ekf_fusion.py starts at the ground-truth
first pose. The EKF has no absolute position sensor, so a wrong start position is
a permanent offset: on the 5 sessions that don't start at the origin, "fixed"
RMSE was 2.3-5.4 m instead of the validated 0.6-2.3 m, swamping real differences
between methods.

Fix: add load_init_pose() to compare_adaptive_vs_fixed.py -- ground-truth first
row (x, y, yaw), computed exactly as ekf_fusion.py does -- and use it as the
initial state for all filters in both gate scripts.

Every edit is an exact match with a required count; aborts and changes nothing
otherwise. Backups go to ~/ws_mobile/fix_ws/*.bak_preInitPose
"""
import os
import shutil
import sys

S = os.path.expanduser('~/ws_mobile/scripts')
FIX = os.path.expanduser('~/ws_mobile/fix_ws')

INIT_FN = '''def load_init_pose(session_id):
    """Ground-truth first pose (x, y, yaw) -- the same initial state
    ekf_fusion.py uses (gt_x/gt_y row 0, yaw = atan2(2wz, 1-2z^2)).
    Decision 2026-10-03 (fix_ws): the gate scripts previously started every
    filter at the origin, which is a permanent position offset for sessions
    that don't start there."""
    gt_df = pd.read_csv(os.path.join(LOG_DIR, f"ground_truth_filtered_{session_id}.csv"), nrows=1)
    oz, ow = float(gt_df["gt_orient_z"].iloc[0]), float(gt_df["gt_orient_w"].iloc[0])
    return (float(gt_df["gt_x"].iloc[0]), float(gt_df["gt_y"].iloc[0]),
            float(np.arctan2(2.0 * ow * oz, 1.0 - 2.0 * oz * oz)))


def rmse(x, y, gt_x, gt_y):
'''

PATCHES = {
    'compare_adaptive_vs_fixed.py': [
        ('rmse def', 'def rmse(x, y, gt_x, gt_y):\n', INIT_FN, 1),
        ('fixed call', '    fixed_out = from_ekf_core(*arrays, 0.0, 0.0, 0.0)\n',
         '    init = load_init_pose(session_id)\n'
         '    fixed_out = from_ekf_core(*arrays, *init)\n', 1),
        ('adaptive call', '    adaptive_out = from_adaptive_core(*arrays, 0.0, 0.0, 0.0)\n',
         '    adaptive_out = from_adaptive_core(*arrays, *init)\n', 1),
    ],
    'lstm_adaptive_ekf.py': [
        ('import', 'from compare_adaptive_vs_fixed import load_session_arrays, load_ground_truth, rmse  # noqa: E402\n',
         'from compare_adaptive_vs_fixed import load_session_arrays, load_ground_truth, rmse, load_init_pose  # noqa: E402\n', 1),
        ('init line', '    n = len(arrays[-1])  # timestamps array length\n',
         '    n = len(arrays[-1])  # timestamps array length\n'
         '    init = load_init_pose(session)  # ground-truth first pose, as ekf_fusion.py\n', 1),
        ('fixed call', 'run_ekf_core(*arrays, 0.0, 0.0, 0.0)', 'run_ekf_core(*arrays, *init)', 1),
        ('heur call', 'run_adaptive_ekf_core(*arrays, 0.0, 0.0, 0.0)', 'run_adaptive_ekf_core(*arrays, *init)', 1),
        ('lstm call', '        *arrays, 0.0, 0.0, 0.0, imu_raw, odom_raw,\n',
         '        *arrays, *init, imu_raw, odom_raw,\n', 1),
    ],
}

# Check every anchor in every file before touching anything.
texts = {}
for fname, edits in PATCHES.items():
    s = open(os.path.join(S, fname)).read()
    if 'load_init_pose' in s:
        sys.exit(f'{fname} already patched. Nothing done.')
    for name, old, _, need in edits:
        c = s.count(old)
        if c != need:
            sys.exit(f'ABORT: {fname} [{name}] anchor found {c} times (need {need}). Nothing changed.')
    texts[fname] = s

for fname, edits in PATCHES.items():
    p = os.path.join(S, fname)
    shutil.copy2(p, os.path.join(FIX, f'{fname}.bak_preInitPose'))
    s = texts[fname]
    for _, old, new, _ in edits:
        s = s.replace(old, new)
    open(p, 'w').write(s)
    print(f'Patched {p}  (backup: fix_ws/{fname}.bak_preInitPose)')
