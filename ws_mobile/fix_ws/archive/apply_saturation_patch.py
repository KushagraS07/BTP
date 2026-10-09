#!/usr/bin/env python3
"""
apply_saturation_patch.py -- one-time patch to ~/ws_mobile/scripts/preprocess_data.py

Adds wheel-level actuator saturation to process_cmd_vel():
  - the executed command replaces linear_x / angular_z
  - the original commands are kept as linear_x_raw / angular_z_raw
Every edit is an exact single-place match. If any anchor is not found
exactly once, the script stops and changes nothing.
A backup is written to ~/ws_mobile/fix_ws/preprocess_data.py.bak_presaturation
"""
import os
import shutil
import sys

P = os.path.expanduser('~/ws_mobile/scripts/preprocess_data.py')
BAK = os.path.expanduser('~/ws_mobile/fix_ws/preprocess_data.py.bak_presaturation')

s = open(P).read()
if 'def saturate_cmd_vel' in s:
    sys.exit('Already patched (saturate_cmd_vel found). Nothing done.')

CONFIG_ANCHOR = "N_BEAMS   = 540       # must match your LiDAR config\n"
CONFIG_ADD = CONFIG_ANCHOR + '''
# ── Actuator saturation (cmd_vel -> executable command) ──────────────────────
# Decision 2026-10-02 (see fix_ws/SATURATION_FIX.md). Teleop commands in
# several sessions far exceed what the robot can execute (up to 114 m/s
# commanded vs ground-truth max exactly 1.500 m/s). The EKF predict step uses
# cmd_vel as control input, so unexecutable commands made it diverge
# (RMSE up to 261 m). Values below come ONLY from the robot model, not tuning:
#   src/mobile_robot/model/robot.xacro: r = 0.15 m,
#   wheel joints <limit velocity="10"/> rad/s, s4 = c/2 + d/2 = 0.35 m
#   (DiffDrive wheel_separation = 2*s4).
APPLY_CMD_SATURATION   = True
WHEEL_RADIUS           = 0.15   # m
WHEEL_JOINT_VEL_LIMIT  = 10.0   # rad/s
HALF_WHEEL_SEPARATION  = 0.35   # m (s4)
'''

FUNC_ANCHOR = "def process_cmd_vel():\n"
FUNC_ADD = '''def saturate_cmd_vel(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Replace commanded (v, w) with what the differential drive can execute:
    each wheel's linear speed v +/- w*s4 is clipped to +/- r*wheel_limit,
    then executed v = (vr+vl)/2, w = (vr-vl)/(2*s4). Originals are kept as
    linear_x_raw / angular_z_raw for traceability. Pointwise, so it commutes
    with the later zero-order-hold resampling. Known limitation: this models
    the speed limit, not Gazebo's full dynamics (e.g. DiffDrive's
    max_linear_acceleration=10 m/s^2 is not modeled).
    """
    if not APPLY_CMD_SATURATION:
        return df
    missing = [c for c in ("linear_x", "angular_z") if c not in df.columns]
    if missing:
        raise ValueError(f"[{label}] saturate_cmd_vel: missing column(s) {missing}; "
                         f"got {list(df.columns)}")
    vmax = WHEEL_RADIUS * WHEEL_JOINT_VEL_LIMIT
    v = df["linear_x"].astype(float).values
    w = df["angular_z"].astype(float).values
    vr = np.clip(v + w * HALF_WHEEL_SEPARATION, -vmax, vmax)
    vl = np.clip(v - w * HALF_WHEEL_SEPARATION, -vmax, vmax)
    df["linear_x_raw"] = v
    df["angular_z_raw"] = w
    df["linear_x"] = (vr + vl) / 2.0
    df["angular_z"] = (vr - vl) / (2.0 * HALF_WHEEL_SEPARATION)
    n_sat = int(((np.abs(df["linear_x"].values - v) + np.abs(df["angular_z"].values - w)) > 1e-9).sum())
    print(f"  [{label}] Actuator saturation: {n_sat}/{len(df)} raw samples clipped "
          f"(wheel limit {vmax:.3f} m/s; raw |v|max={np.abs(v).max():.3f}, "
          f"raw |w|max={np.abs(w).max():.3f})")
    return df


''' + FUNC_ANCHOR

CALL_ANCHOR = '    df = sort_by_time(df, "CmdVel")\n'
CALL_ADD = CALL_ANCHOR + '    df = saturate_cmd_vel(df, "CmdVel")\n'

for name, anchor in (('config', CONFIG_ANCHOR), ('function', FUNC_ANCHOR), ('call', CALL_ANCHOR)):
    c = s.count(anchor)
    if c != 1:
        sys.exit(f'ABORT: {name} anchor found {c} times (need exactly 1). Nothing changed.')

shutil.copy2(P, BAK)
s = s.replace(CONFIG_ANCHOR, CONFIG_ADD).replace(FUNC_ANCHOR, FUNC_ADD).replace(CALL_ANCHOR, CALL_ADD)
open(P, 'w').write(s)
print(f'Patched {P}\nBackup: {BAK}')
