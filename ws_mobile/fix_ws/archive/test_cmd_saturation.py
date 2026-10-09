#!/usr/bin/env python3
"""
test_cmd_saturation.py -- READ-ONLY. Writes nothing anywhere.

Question: if cmd_vel is replaced by what the robot can ACTUALLY execute,
do the broken baselines become sane?

Saturation model (from the robot's own model files, nothing tuned):
  - robot.xacro: wheel joint <limit velocity="10"/> -> 10 rad/s per wheel
  - wheel radius r, half-separation s4 = c/2 + d/2 (read from robot.xacro)
  - each wheel speed = v +/- w*s4, clipped to +/- 10*r (= 1.5 m/s if r=0.15)
  - executed v = (vr+vl)/2, executed w = (vr-vl)/(2*s4)
  (DiffDrive's max_linear_acceleration=10 m/s^2 is ignored: reaching
   1.5 m/s at that rate takes 0.15 s.)

For each baseline, runs the REAL, unmodified ekf_fusion.run_ekf_core in
memory with (1) raw cmd_vel and (2) saturated cmd_vel, and compares the
executed speed against ground-truth speed.
"""
import os
import re
import sys

import numpy as np
import pandas as pd

WS = os.path.expanduser('~/ws_mobile')
LOG = os.path.join(WS, 'sensor_logs')
sys.path.insert(0, os.path.join(WS, 'scripts'))
import ekf_fusion as E  # noqa: E402  (no argv -> SESSION=None, nothing runs on import)

WHEEL_JOINT_LIMIT = 10.0  # rad/s, robot.xacro <limit velocity="10"/>


def xacro_props(path):
    txt = open(path).read()
    raw = dict(re.findall(r'<xacro:property\s+name="(\w+)"\s+value="([^"]+)"', txt))
    vals = {}
    for _ in range(10):  # resolve simple ${...} expressions that reference other props
        for k, v in raw.items():
            if k in vals:
                continue
            expr = v[2:-1] if v.startswith('${') and v.endswith('}') else v
            try:
                vals[k] = float(eval(expr, {'__builtins__': {}, 'pi': np.pi}, dict(vals)))
            except Exception:
                pass
    return vals


P = xacro_props(os.path.join(WS, 'src/mobile_robot/model/robot.xacro'))
r, s4 = P['r'], P['s4']
VW_MAX = WHEEL_JOINT_LIMIT * r
print(f'From robot.xacro: r={r}  s4(half-separation)={s4:.4f}  -> wheel speed limit {VW_MAX:.3f} m/s, '
      f'max pure-rotation w = {VW_MAX / s4:.3f} rad/s\n')


def yaw(z, w):
    return np.arctan2(2 * np.asarray(w) * np.asarray(z), 1 - 2 * np.asarray(z) ** 2)


def saturate(v, w):
    vr = np.clip(v + w * s4, -VW_MAX, VW_MAX)
    vl = np.clip(v - w * s4, -VW_MAX, VW_MAX)
    return (vr + vl) / 2, (vr - vl) / (2 * s4)


def stored_rmse(b):
    m = pd.read_csv(os.path.join(LOG, f'ekf_metrics_{b}.csv'))
    return float(m.loc[m['metric'] == 'RMSE', 'value_m'].iloc[0])


B = pd.read_csv(os.path.join(WS, 'fix_ws/imu_baseline_profile.csv'))['baseline'].astype(str).tolist()
rows = []
for b in B:
    imu = pd.read_csv(os.path.join(LOG, f'imu_filtered_{b}.csv'))
    od = pd.read_csv(os.path.join(LOG, f'odom_filtered_{b}.csv'))
    n = min(len(imu), len(od))
    imu, od = imu.iloc[:n], od.iloc[:n]
    cmd = pd.read_csv(os.path.join(LOG, f'cmd_vel_filtered_{b}.csv')).iloc[:n]
    gt = pd.read_csv(os.path.join(LOG, f'ground_truth_filtered_{b}.csv')).iloc[:n]

    odom_th = yaw(od.orient_z, od.orient_w)
    imu_th = yaw(imu.orient_z, imu.orient_w)
    gt_th = yaw(gt.gt_orient_z, gt.gt_orient_w)
    ts = od.timestamp.values
    v_raw, w_raw = cmd.linear_x.values, cmd.angular_z.values
    v_sat, w_sat = saturate(v_raw, w_raw)

    def run(v, w):
        out = E.run_ekf_core(od.pos_x.values, od.pos_y.values, odom_th, v, w, imu_th, ts,
                             float(gt.gt_x.iloc[0]), float(gt.gt_y.iloc[0]), float(gt_th[0]))
        return E.compute_metrics(out[0], out[1], gt.gt_x.values, gt.gt_y.values)[0]

    gt_v = np.hypot(np.diff(gt.gt_x), np.diff(gt.gt_y)) / np.diff(ts)
    rows.append({
        'baseline': b,
        'rmse_stored': stored_rmse(b),
        'rmse_raw_recomputed': run(v_raw, w_raw),
        'rmse_saturated': run(v_sat, w_sat),
        'frac_rows_saturated': float((np.abs(v_raw - v_sat) + np.abs(w_raw - w_sat) > 1e-9).mean()),
        'corr(|v_raw|,gt_v)': float(np.corrcoef(np.abs(v_raw[1:]), gt_v)[0, 1]),
        'corr(|v_sat|,gt_v)': float(np.corrcoef(np.abs(v_sat[1:]), gt_v)[0, 1]),
        'max|v_sat|': float(np.abs(v_sat).max()),
        'gt_v_max': float(gt_v.max()),
    })

pd.set_option('display.width', 250)
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f'{x:.3f}'))
