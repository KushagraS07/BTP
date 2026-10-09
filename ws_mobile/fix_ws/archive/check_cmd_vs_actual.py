#!/usr/bin/env python3
# READ-ONLY. Does the robot actually do what cmd_vel says, or does it saturate?
import os
import numpy as np, pandas as pd

LOG = os.path.expanduser('~/ws_mobile/sensor_logs')
FIX = os.path.expanduser('~/ws_mobile/fix_ws')
B = pd.read_csv(os.path.join(FIX, 'imu_baseline_profile.csv'))['baseline'].astype(str).tolist()

rows = []
for b in B:
    gt = pd.read_csv(os.path.join(LOG, f'ground_truth_filtered_{b}.csv'))
    cmd = pd.read_csv(os.path.join(LOG, f'cmd_vel_filtered_{b}.csv'))
    od = pd.read_csv(os.path.join(LOG, f'odom_filtered_{b}.csv'))
    dt = np.diff(gt.timestamp.values)
    gt_v = np.hypot(np.diff(gt.gt_x), np.diff(gt.gt_y)) / dt
    yaw = np.unwrap(np.arctan2(2*gt.gt_orient_w*gt.gt_orient_z, 1 - 2*gt.gt_orient_z**2))
    gt_w = np.abs(np.diff(yaw)) / dt
    cv, cw = np.abs(cmd.linear_x.values), np.abs(cmd.angular_z.values)
    v99, w99 = np.percentile(gt_v, 99), np.percentile(gt_w, 99)
    r = {'baseline': b,
         'gt_v_p99': v99, 'gt_v_max': gt_v.max(),
         'cmd_v_max': cv.max(), 'frac_cmd_v>1.5x_gt_p99': float((cv > 1.5 * v99).mean()),
         'gt_w_p99': w99, 'cmd_w_max': cw.max(), 'frac_cmd_w>1.5x_gt_p99': float((cw > 1.5 * w99).mean())}
    if 'linear_x' in od.columns:
        r['odom_v_max'] = np.abs(od.linear_x).max()
    if 'angular_z' in od.columns:
        r['odom_w_max'] = np.abs(od.angular_z).max()
    rows.append(r)

pd.set_option('display.width', 250)
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f'{x:.3f}'))
