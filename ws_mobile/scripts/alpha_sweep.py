#!/usr/bin/env python3
"""
alpha_sweep.py — empirically finds the best LiDAR blend weight (ALPHA).

The paper does not specify a numeric value for this (checked the paper
text directly — the complementary fusion stage is described only
conceptually, with no formula or blend weight given). The current
ALPHA=0.98 in lidar_fusion_v2.py was chosen empirically, but that test
included session 20260714_014615 — which we since found has a dead IMU
sensor, producing an unreliable RNN pose that made trusting LiDAR more
look worse than it might actually be. Re-testing now, on the two
verified-good sessions with a genuinely improved RNN pose (0.29-0.41m,
vs the old 1.3-1.9m).

Reuses the REAL pipeline's own functions (estimate_position_correction,
sync_to_grid, quat_to_yaw, load_csv, get_timestamp, compute_metrics) by
importing lidar_fusion_v2.py as a module — same pattern noise_evaluation.py
already uses — not a reimplementation that could silently drift from the
real logic.

Read-only with respect to the pipeline: does NOT write final_fused_output_v2
or any file lidar_fusion_v2.py itself produces. Only writes its own
alpha_sweep_results.csv for reference.
"""

import os
import sys
import importlib.util
import numpy as np
import pandas as pd

LOG_DIR     = os.path.expanduser('~/ws_mobile/sensor_logs')
SCRIPTS_DIR = os.path.expanduser('~/ws_mobile/scripts')
SESSIONS    = ['20260712_141740', '20260712_144326']  # same as the current
                                                        # verified-good list
ALPHAS_TO_TEST = [0.98, 0.90, 0.80, 0.70, 0.50, 0.30]


def _load_module(mod_name, filename):
    path = os.path.join(SCRIPTS_DIR, filename)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


print("  Loading real lidar_fusion_v2.py module (reusing its real "
      "functions, not reimplementing them)...")
lidar_mod = _load_module('lidar_fusion_real', 'lidar_fusion_v2.py')


def run_single_alpha(session: str, alpha: float):
    """Same math as lidar_fusion_v2.py's run_lidar_fusion_single(), just
    parameterized on alpha instead of reading the module-level ALPHA
    constant, and without writing any output files."""
    rnn_path  = os.path.join(LOG_DIR, f'rnn_smoothed_output_{session}.csv')
    scan_path = os.path.join(LOG_DIR, f'scan_filtered_{session}.csv')
    odom_path = os.path.join(LOG_DIR, f'odom_filtered_{session}.csv')
    gt_path   = os.path.join(LOG_DIR, f'ground_truth_filtered_{session}.csv')
    for p in [rnn_path, scan_path, odom_path, gt_path]:
        if not os.path.exists(p):
            print(f"  ⚠️  Missing {p} — skipping session {session}")
            return None

    rnn_df  = lidar_mod.load_csv(rnn_path)
    scan_df = lidar_mod.load_csv(scan_path)
    odom_df = lidar_mod.load_csv(odom_path)
    gt_df   = pd.read_csv(gt_path)

    angle_increment = float(scan_df['angle_increment'].iloc[0])
    range_max       = float(scan_df['range_max'].iloc[0])

    rnn_t  = lidar_mod.get_timestamp(rnn_df)
    t_grid = rnn_t

    odom_sync      = lidar_mod.sync_to_grid(odom_df, t_grid, ['orient_z', 'orient_w'])
    scan_angle_min = lidar_mod.sync_to_grid(scan_df, t_grid, ['angle_min'])[:, 0]

    range_cols = sorted(
        [c for c in scan_df.columns
         if c.startswith('range_') and c.split('_', 1)[1].isdigit()],
        key=lambda c: int(c.split('_')[1]))
    scan_sync = lidar_mod.sync_to_grid(scan_df, t_grid, range_cols)

    n = len(t_grid)
    fused_x, fused_y = [], []

    for i in range(n):
        rnn_x = float(rnn_df['rnn_x'].iloc[i])
        rnn_y = float(rnn_df['rnn_y'].iloc[i])
        oz    = float(odom_sync[i, 0])
        ow    = float(odom_sync[i, 1])
        theta = lidar_mod.quat_to_yaw(oz, ow)
        if 'rnn_theta' in rnn_df.columns:
            theta = float(rnn_df['rnn_theta'].iloc[i])

        ranges    = scan_sync[i]
        angle_min = float(scan_angle_min[i])

        err_x, err_y, conf = lidar_mod.estimate_position_correction(
            rnn_x, rnn_y, theta, ranges, angle_min, angle_increment, range_max)

        lidar_x = rnn_x + err_x
        lidar_y = rnn_y + err_y

        w       = (1 - alpha) * conf
        final_x = (1 - w) * rnn_x + w * lidar_x
        final_y = (1 - w) * rnn_y + w * lidar_y

        fused_x.append(final_x)
        fused_y.append(final_y)

    gt_sync = lidar_mod.sync_to_grid(gt_df, t_grid, ['gt_x', 'gt_y'])
    rmse, mae, std = lidar_mod.compute_metrics(
        np.array(fused_x), np.array(fused_y), gt_sync[:, 0], gt_sync[:, 1])
    return rmse, mae, std, n


def main():
    print("\n" + "=" * 60)
    print("  ALPHA SWEEP — LiDAR blend weight, real data, real functions")
    print(f"  Sessions: {', '.join(SESSIONS)}")
    print(f"  Testing alpha in: {ALPHAS_TO_TEST}")
    print("=" * 60)

    rows = []
    for alpha in ALPHAS_TO_TEST:
        print(f"\n  --- alpha={alpha} ---")
        per_session_sq_err = []
        per_session_n = []
        for session in SESSIONS:
            result = run_single_alpha(session, alpha)
            if result is None:
                continue
            rmse, mae, std, n = result
            print(f"    Session {session}: RMSE={rmse:.4f}m  MAE={mae:.4f}m  Std={std:.4f}m")
            per_session_sq_err.append(rmse**2 * n)
            per_session_n.append(n)
            rows.append(dict(alpha=alpha, session=session, rmse=rmse, mae=mae, std=std))

        pooled_rmse = float(np.sqrt(sum(per_session_sq_err) / sum(per_session_n)))
        print(f"    POOLED RMSE @ alpha={alpha}: {pooled_rmse:.4f}m")
        rows.append(dict(alpha=alpha, session='POOLED', rmse=pooled_rmse, mae=None, std=None))

    df = pd.DataFrame(rows)
    out_path = os.path.join(LOG_DIR, 'alpha_sweep_results.csv')
    df.to_csv(out_path, index=False)

    pooled = df[df['session'] == 'POOLED'].sort_values('rmse')
    print("\n" + "=" * 60)
    print("  RESULTS — pooled RMSE by alpha, best first")
    print("=" * 60)
    for _, row in pooled.iterrows():
        print(f"    alpha={row['alpha']:<6}  pooled RMSE={row['rmse']:.4f}m")
    best_alpha = pooled.iloc[0]['alpha']
    current_row = pooled[pooled['alpha'] == 0.98]
    current_rmse = current_row.iloc[0]['rmse'] if len(current_row) else None
    print(f"\n  Best alpha: {best_alpha}  (RMSE={pooled.iloc[0]['rmse']:.4f}m)")
    if current_rmse is not None and best_alpha != 0.98:
        improvement = (current_rmse - pooled.iloc[0]['rmse']) / current_rmse * 100
        print(f"  Current alpha=0.98 gives RMSE={current_rmse:.4f}m — "
              f"switching to {best_alpha} would improve pooled RMSE by "
              f"{improvement:.1f}%")
    elif best_alpha == 0.98:
        print(f"  Current alpha=0.98 IS the best tested value — no change needed.")
    print(f"\n  Saved: alpha_sweep_results.csv")


if __name__ == '__main__':
    main()
