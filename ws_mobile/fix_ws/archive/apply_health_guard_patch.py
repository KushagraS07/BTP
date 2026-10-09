#!/usr/bin/env python3
"""
apply_health_guard_patch.py -- one-time patch to ~/ws_mobile/scripts/health_check.py

Adds the actuator-saturation guard (decision 2026-10-02, fix_ws/SATURATION_FIX.md):
  FAIL : the EXECUTED commands in cmd_vel_filtered (linear_x/angular_z)
         exceed the wheel limits. This means the session was preprocessed
         WITHOUT the saturation patch, so the EKF would get unexecutable
         control input.
  WARN : more than CMD_SATURATION_WARN_FRAC of the RAW commands
         (linear_x_raw/angular_z_raw) were clipped. The session stays usable,
         but the executed-command model is approximate (see the caveat in
         SATURATION_FIX.md). The guard warns rather than fails here because
         the agreed decision is to KEEP corrected sessions, not exclude them.
The wheel constants are imported from preprocess_data.py (single source of truth).
Every edit is an exact single-place match. If any anchor does not match
exactly once, the script aborts and changes nothing.
Backup: ~/ws_mobile/fix_ws/health_check.py.bak_presaturation
"""
import os
import shutil
import sys

P = os.path.expanduser('~/ws_mobile/scripts/health_check.py')
BAK = os.path.expanduser('~/ws_mobile/fix_ws/health_check.py.bak_presaturation')
s = open(P).read()
if 'def check_cmd_saturation' in s:
    sys.exit('Already patched. Nothing done.')

IMPORT_ANCHOR = "from session_registry import (\n"
IMPORT_ADD = ("# Wheel limits: single source of truth is preprocess_data.py (robot.xacro values).\n"
              "from preprocess_data import (WHEEL_RADIUS, WHEEL_JOINT_VEL_LIMIT,\n"
              "                             HALF_WHEEL_SEPARATION)\n" + IMPORT_ANCHOR)

FUNC_ANCHOR = "def check_session(session: str):\n"
FUNC_ADD = '''# Actuator-saturation guard (decision 2026-10-02, fix_ws/SATURATION_FIX.md)
CMD_SATURATION_WARN_FRAC = 0.10   # warn if >10% of raw command rows were clipped
CMD_SATURATION_REL_TOL = 1e-6     # float tolerance on the wheel limit


def check_cmd_saturation(cmd):
    """
    Returns (exec_frac, raw_frac, has_raw): fraction of rows whose wheel
    speeds (v +/- w*s4) exceed the wheel limit, for the executed commands
    (linear_x/angular_z) and the original raw commands (linear_x_raw/
    angular_z_raw, present only if preprocess_data.py's saturation ran).
    """
    vmax = WHEEL_RADIUS * WHEEL_JOINT_VEL_LIMIT * (1.0 + CMD_SATURATION_REL_TOL)

    def frac_over(v_col, w_col):
        v = pd.to_numeric(cmd[v_col], errors='coerce').to_numpy()
        w = pd.to_numeric(cmd[w_col], errors='coerce').to_numpy()
        ok = np.isfinite(v) & np.isfinite(w)
        v, w = v[ok], w[ok]
        if len(v) == 0:
            return float('nan')
        vr, vl = v + w * HALF_WHEEL_SEPARATION, v - w * HALF_WHEEL_SEPARATION
        return float(((np.abs(vr) > vmax) | (np.abs(vl) > vmax)).mean())

    exec_frac = frac_over('linear_x', 'angular_z')
    has_raw = 'linear_x_raw' in cmd.columns and 'angular_z_raw' in cmd.columns
    raw_frac = frac_over('linear_x_raw', 'angular_z_raw') if has_raw else exec_frac
    return exec_frac, raw_frac, has_raw


''' + FUNC_ANCHOR

CALL_ANCHOR = ("    # -----------------------------------------------------------------\n"
               "    # 4. IMU orientation health\n")
CALL_ADD = '''    # -----------------------------------------------------------------
    # 3b. Actuator saturation guard (decision 2026-10-02)
    # -----------------------------------------------------------------
    if cmd is not None:
        if 'linear_x' not in cmd.columns:
            reasons.append(f"cmd_vel_filtered_{session}.csv has no 'linear_x' column")
        else:
            exec_frac, raw_frac, has_raw = check_cmd_saturation(cmd)
            print(f"  cmd_vel wheel-limit check: executed over-limit={exec_frac:.1%}, "
                  f"raw clipped={raw_frac:.1%} "
                  f"({'raw columns present' if has_raw else 'NO raw columns'})")
            if exec_frac > 0:
                reasons.append(
                    f"cmd_vel exceeds wheel limits on {exec_frac:.1%} of rows: actuator "
                    f"saturation was NOT applied. Re-run the patched preprocess_data.py "
                    f"for this session before using it.")
            elif raw_frac > CMD_SATURATION_WARN_FRAC:
                print(f"  WARNING: {raw_frac:.1%} of raw commands were clipped to the wheel "
                      f"limits (> {CMD_SATURATION_WARN_FRAC:.0%}). Session kept; the "
                      f"executed-command model is approximate here (SATURATION_FIX.md).")

''' + CALL_ANCHOR

for name, a in (('import', IMPORT_ANCHOR), ('function', FUNC_ANCHOR), ('call', CALL_ANCHOR)):
    c = s.count(a)
    if c != 1:
        sys.exit(f'ABORT: {name} anchor found {c} times (need exactly 1). Nothing changed.')

shutil.copy2(P, BAK)
s = s.replace(IMPORT_ANCHOR, IMPORT_ADD).replace(FUNC_ANCHOR, FUNC_ADD).replace(CALL_ANCHOR, CALL_ADD)
open(P, 'w').write(s)
print(f'Patched {P}\nBackup: {BAK}')
