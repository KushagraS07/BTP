#!/usr/bin/env python3
"""
apply_runtime_features_patch.py -- one-time patch to ~/ws_mobile/scripts/lstm_adaptive_ekf.py

Problem: the LSTM checkpoints are trained (build_training_dataset.py, Phase 3 B.4)
on 13 features including wheel_imu_disagreement and wheel_lidar_disagreement, but
lstm_adaptive_ekf.py's runtime (Phase 2 evaluation) can only build the original 11,
so load_model() refuses every checkpoint.

Fix: both features are EXOGENOUS (they depend only on recorded sensor data, never on
any EKF/LSTM decision), and build_training_dataset.py --stage1 already computed them
for every session into features_{session}.csv. The runtime now reads those two
columns from that file, so deployment sees byte-identical values to training -- no
re-implementation, no train/deploy mismatch. A row-count check guards alignment.

Every edit is an exact single-place match; aborts and changes nothing otherwise.
Backup: ~/ws_mobile/fix_ws/lstm_adaptive_ekf.py.bak_preB4runtime
"""
import os
import shutil
import sys

P = os.path.expanduser('~/ws_mobile/scripts/lstm_adaptive_ekf.py')
BAK = os.path.expanduser('~/ws_mobile/fix_ws/lstm_adaptive_ekf.py.bak_preB4runtime')
s = open(P).read()
if 'B4_EXO_FEATURES' in s:
    sys.exit('Already patched. Nothing done.')

EDITS = [
    ('AVAILABLE_FEATURES',
     '    "odom_linear_x", "odom_angular_z",\n}\n',
     '    "odom_linear_x", "odom_angular_z",\n'
     '    # Phase 3 B.4 exogenous features, read from features_{session}.csv\n'
     '    # (computed once by build_training_dataset.py --stage1).\n'
     '    "wheel_imu_disagreement", "wheel_lidar_disagreement",\n}\n'
     'B4_EXO_FEATURES = ["wheel_imu_disagreement", "wheel_lidar_disagreement"]\n'),
    ('row_feats',
     '            "odom_angular_z": odom_raw["angular_z"][i],\n        }\n',
     '            "odom_angular_z": odom_raw["angular_z"][i],\n'
     '            "wheel_imu_disagreement": odom_raw["wheel_imu_disagreement"][i],\n'
     '            "wheel_lidar_disagreement": odom_raw["wheel_lidar_disagreement"][i],\n'
     '        }\n'),
    ('loader',
     '    odom_raw = {c: odom_df[c].values for c in ["linear_x", "angular_z"]}\n'
     '    return imu_raw, odom_raw\n',
     '    odom_raw = {c: odom_df[c].values for c in ["linear_x", "angular_z"]}\n'
     '    # Phase 3 B.4: exogenous features, identical to the training values.\n'
     '    feat_path = SENSOR_LOGS_DIR / f"features_{session}.csv"\n'
     '    if not feat_path.exists():\n'
     '        raise FileNotFoundError(f"{session}: missing {feat_path.name} "\n'
     '                                f"(run build_training_dataset.py --stage1)")\n'
     '    feat_df = pd.read_csv(feat_path, usecols=B4_EXO_FEATURES)\n'
     '    if len(feat_df) != n:\n'
     '        raise ValueError(f"{session}: {feat_path.name} has {len(feat_df)} rows, "\n'
     '                         f"runtime expects {n}")\n'
     '    for c in B4_EXO_FEATURES:\n'
     '        odom_raw[c] = feat_df[c].values\n'
     '    return imu_raw, odom_raw\n'),
]

for name, old, _ in EDITS:
    c = s.count(old)
    if c != 1:
        sys.exit(f'ABORT: {name} anchor found {c} times (need exactly 1). Nothing changed.')

shutil.copy2(P, BAK)
for _, old, new in EDITS:
    s = s.replace(old, new)
open(P, 'w').write(s)
print(f'Patched {P}\nBackup: {BAK}')
