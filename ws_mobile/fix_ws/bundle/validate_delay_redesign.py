#!/usr/bin/env python3
"""
validate_delay_redesign.py

Systematic test of the redesigned raw-level delay fault injection
(make_delay_fault_session_raw(), routed through make_fault_session()'s
fault_type=='delay' branch) across ALL good baselines, both sensors
(imu, odom), and all 3 severities -- 42 total fault sessions.

Purpose: a single-baseline pilot (20260712_141740) showed a surprising,
consistent, monotonic pattern -- delay faults measurably IMPROVE final
EKF position RMSE rather than degrade it, for both sensors, at every
severity. This was confirmed NOT to be an artifact of grid trimming or
edge-clamping (raw sensor data comfortably spans the grid window before
and after the shift; row counts and time ranges are identical between
baseline and fault). The actual fused heading (ekf_theta) does diverge
substantially at specific points, confirming the fault genuinely feeds
stale data into the EKF -- so the "improvement" is a real dynamical
property of this specific Fixed EKF's response to stale-but-smooth
heading input, not a bug in the injection mechanism itself.

This script exists to test whether that pattern REPLICATES across all 7
baselines (same discipline as every other systematic validation in this
project) before treating it as a reportable finding.
"""

import sys
from pathlib import Path
import subprocess

import pandas as pd

SCRIPTS_DIR = Path.home() / "ws_mobile" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from fault_injection import make_fault_session, DELAY_SEVERITY  # noqa: E402
from health_check import check_session  # noqa: E402

SENSORS = ["imu", "odom"]


def discover_good_baselines() -> list[str]:
    SENSOR_LOGS_DIR = Path.home() / "ws_mobile" / "sensor_logs"
    candidates = set()
    for f in SENSOR_LOGS_DIR.glob("ekf_fused_output_*.csv"):
        session = f.stem[len("ekf_fused_output_"):]
        if "_fault_" not in session:
            candidates.add(session)
    good = []
    for c in sorted(candidates):
        passed, reasons = check_session(c)
        if passed:
            good.append(c)
        else:
            print(f"excluding baseline {c}: {reasons}")
    return good


def get_rmse(session: str) -> float:
    """Runs ekf_fusion.py for a session (if metrics don't already exist)
    and returns its RMSE. Raises on any failure -- no silent swallowing."""
    metrics_path = Path.home() / "ws_mobile" / "sensor_logs" / f"ekf_metrics_{session}.csv"
    if not metrics_path.exists():
        proc = subprocess.run(
            [sys.executable, "ekf_fusion.py", session],
            cwd=SCRIPTS_DIR, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            raise RuntimeError(f"ekf_fusion.py failed for {session} "
                                f"(exit {proc.returncode}): {proc.stderr[-800:]}")
    m = pd.read_csv(metrics_path)
    return float(m.loc[m["metric"] == "RMSE", "value_m"].iloc[0])


def main():
    good_baselines = discover_good_baselines()
    print(f"\n{len(good_baselines)} good baselines: {good_baselines}\n")

    rows = []
    for baseline in good_baselines:
        try:
            baseline_rmse = get_rmse(baseline)
        except Exception as e:
            print(f"SKIPPING baseline {baseline}: {e}")
            continue

        for sensor in SENSORS:
            for severity in DELAY_SEVERITY:
                try:
                    fault_session = make_fault_session(baseline, sensor, "delay", severity, seed=0)
                except Exception as e:
                    print(f"  FAILED to create {baseline}/{sensor}/delay/{severity}: {e}")
                    continue
                try:
                    fault_rmse = get_rmse(fault_session)
                except Exception as e:
                    print(f"  FAILED ekf_fusion for {fault_session}: {e}")
                    continue

                ratio = fault_rmse / baseline_rmse
                rows.append({
                    "baseline": baseline, "sensor": sensor, "severity": severity,
                    "baseline_rmse": baseline_rmse, "fault_rmse": fault_rmse, "ratio": ratio,
                })
                direction = "DEGRADED" if ratio > 1.0 else "improved"
                print(f"  {fault_session}: ratio={ratio:.4f} ({direction})")

    if not rows:
        print("\nno results")
        return

    df = pd.DataFrame(rows)
    out_path = Path.home() / "ws_mobile" / "sensor_logs" / "delay_redesign_validation_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")

    print("\n=== summary (ratio > 1.0 = degraded, as expected for a real fault) ===")
    print(f"overall: mean ratio={df['ratio'].mean():.4f}, "
          f"degraded={100*(df['ratio']>1.0).mean():.0f}% of {len(df)} cases, "
          f"improved={100*(df['ratio']<1.0).mean():.0f}% of cases")
    print()
    for sensor, g in df.groupby("sensor"):
        print(f"  {sensor}: mean ratio={g['ratio'].mean():.4f}, "
              f"degraded={100*(g['ratio']>1.0).mean():.0f}%, n={len(g)}")
    print()
    for severity, g in df.groupby("severity"):
        print(f"  severity={severity}: mean ratio={g['ratio'].mean():.4f}, "
              f"degraded={100*(g['ratio']>1.0).mean():.0f}%, n={len(g)}")
    print()
    for baseline, g in df.groupby("baseline"):
        print(f"  baseline={baseline}: mean ratio={g['ratio'].mean():.4f}, "
              f"degraded={100*(g['ratio']>1.0).mean():.0f}%, n={len(g)}")


if __name__ == "__main__":
    main()
