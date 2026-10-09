#!/usr/bin/env python3
"""
check_init_invariance.py -- READ-ONLY. Run AFTER apply_init_pose_patch.py.

1. Fixed EKF with the ground-truth initial pose must now reproduce
   ekf_fusion.py's validated RMSE (ekf_metrics_<session>.csv) exactly.
2. The LSTM training features odom_scale / imu_scale were computed in
   build_training_dataset.py with an ORIGIN initial pose. Check how much they
   differ from a ground-truth start. The filter's innovations depend only on
   relative motion and heading, so a pure position offset should not change
   them; only a brief heading transient at the start might. If the difference
   is limited to a handful of initial rows, the trained LSTM is still valid and
   no retraining is needed.
"""
import os
import sys

import numpy as np
import pandas as pd

S = os.path.expanduser('~/ws_mobile/scripts')
LOG = os.path.expanduser('~/ws_mobile/sensor_logs')
sys.path.insert(0, S)
from compare_adaptive_vs_fixed import load_session_arrays, load_ground_truth, rmse, load_init_pose  # noqa: E402
from ekf_fusion import run_ekf_core  # noqa: E402
from heuristic_adaptive_ekf import run_adaptive_ekf_core  # noqa: E402

B = ['20260712_141740', '20260712_144326', '20260824_094119', '20260824_095427',
     '20260824_095712', '20260824_095915', '20260824_100318']
rows = []
for b in B:
    arrays = load_session_arrays(b)
    gt_x, gt_y = load_ground_truth(b)
    init = load_init_pose(b)
    fixed = run_ekf_core(*arrays, *init)
    m = pd.read_csv(os.path.join(LOG, f'ekf_metrics_{b}.csv'))
    validated = float(m.loc[m.metric == 'RMSE', 'value_m'].iloc[0])

    a0 = run_adaptive_ekf_core(*arrays, 0.0, 0.0, 0.0)
    a1 = run_adaptive_ekf_core(*arrays, *init)
    r = {'baseline': b,
         'init_x,y,yaw': f'{init[0]:.2f},{init[1]:.2f},{init[2]:.2f}',
         'fixed_gt_init': rmse(fixed[0], fixed[1], gt_x, gt_y),
         'validated': validated}
    for name, k in (('odom_scale', 7), ('imu_scale', 8)):
        d = np.abs(np.asarray(a0[k]) - np.asarray(a1[k]))
        idx = np.where(d > 1e-9)[0]
        r[f'{name}_rows_diff'] = len(idx)
        r[f'{name}_last_diff_row'] = int(idx[-1]) if len(idx) else -1
        r[f'{name}_max_diff'] = float(d.max())
    r['n_rows'] = len(a0[7])
    rows.append(r)

pd.set_option('display.width', 250)
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f'{x:.4f}'))
