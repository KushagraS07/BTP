#!/usr/bin/env python3
"""
run_a2_sanity_check.py — A.2's cheap pre-training sanity check.

Compares RMSE using the real lidar_fusion.py, with the confidence term
swapped between:
  (1) the EXISTING valid-beam-count confidence (baseline)
  (2) the roughness proxy from lidar_confidence_proxy.py (hand-favorable
      heuristic, VERIFIED via testing to distinguish 'corruption' faults
      specifically -- proxy_confidence 0.11 for corrupted frames vs 0.96
      for clean)

Run at TWO alpha settings:
  DEPLOYED_ALPHA = 0.98  (real production value, from lidar_fusion.py)
  DIAGNOSTIC_ALPHA = 0.70 (substantially higher LiDAR influence, per
      A.2's explicit instruction to test detectability independent of
      the deployed configuration's max-2%-influence ceiling)

SCOPE: 'corruption' fault sessions ONLY -- the roughness proxy does NOT
meaningfully detect return_loss/scale_error (verified: 0.955/0.954
proxy confidence, indistinguishable from clean's 0.960).

PRE-REGISTERED PASS/FAIL THRESHOLD (fixed BEFORE running):
  >= 5% RMSE improvement (proxy vs baseline confidence) at
  DIAGNOSTIC_ALPHA on corruption sessions.

Requires: patched lidar_fusion.py, rnn_fusion.py's output already
generated for each session tested, session_registry.json. Place in
~/ws_mobile/scripts/ and run there.

Usage:
    python3 run_a2_sanity_check.py
"""
import sys
import numpy as np

sys.path.insert(0, '.')
from lidar_confidence_proxy import compute_roughness_proxy
from lidar_fusion import run_lidar_fusion_single

DEPLOYED_ALPHA = 0.98
DIAGNOSTIC_ALPHA = 0.70
PASS_THRESHOLD_PCT = 5.0  # pre-registered, fixed before running

CORRUPTION_SESSIONS = [
    "20260712_141740_fault_lidar_corruption_mild",
    "20260712_141740_fault_lidar_corruption_moderate",
    "20260712_141740_fault_lidar_corruption_severe",
    "20260712_144326_fault_lidar_corruption_mild",
    "20260712_144326_fault_lidar_corruption_moderate",
    "20260712_144326_fault_lidar_corruption_severe",
    "20260824_094119_fault_lidar_corruption_mild",
    "20260824_094119_fault_lidar_corruption_moderate",
    "20260824_094119_fault_lidar_corruption_severe",
    "20260824_095427_fault_lidar_corruption_mild",
    "20260824_095427_fault_lidar_corruption_moderate",
    "20260824_095427_fault_lidar_corruption_severe",
    "20260824_095712_fault_lidar_corruption_mild",
    "20260824_095712_fault_lidar_corruption_moderate",
    "20260824_095712_fault_lidar_corruption_severe",
    "20260824_095915_fault_lidar_corruption_mild",
    "20260824_095915_fault_lidar_corruption_moderate",
    "20260824_095915_fault_lidar_corruption_severe",
    "20260824_100318_fault_lidar_corruption_mild",
    "20260824_100318_fault_lidar_corruption_moderate",
    "20260824_100318_fault_lidar_corruption_severe",
]


def roughness_confidence_fn(ranges, range_max):
    return compute_roughness_proxy(np.asarray(ranges), range_max)


def run_at_alpha(session, alpha, confidence_fn, label):
    import lidar_fusion
    original_alpha = lidar_fusion.ALPHA
    lidar_fusion.ALPHA = alpha
    try:
        result = run_lidar_fusion_single(session, confidence_fn=confidence_fn)
    finally:
        lidar_fusion.ALPHA = original_alpha

    if result is None:
        print(f"    {label}: session {session} skipped (missing files)")
        return None
    rmse = result[0]
    return rmse


def main():
    print(f"Pre-registered pass threshold: >= {PASS_THRESHOLD_PCT}% RMSE "
          f"improvement at diagnostic alpha={DIAGNOSTIC_ALPHA}\n")

    results = []
    for session in CORRUPTION_SESSIONS:
        print(f"Session: {session}")
        baseline_deployed = run_at_alpha(session, DEPLOYED_ALPHA, None, "baseline/deployed")
        proxy_deployed = run_at_alpha(session, DEPLOYED_ALPHA, roughness_confidence_fn, "proxy/deployed")
        baseline_diagnostic = run_at_alpha(session, DIAGNOSTIC_ALPHA, None, "baseline/diagnostic")
        proxy_diagnostic = run_at_alpha(session, DIAGNOSTIC_ALPHA, roughness_confidence_fn, "proxy/diagnostic")

        if None in (baseline_deployed, proxy_deployed, baseline_diagnostic, proxy_diagnostic):
            continue

        pct_improvement_diagnostic = 100.0 * (baseline_diagnostic - proxy_diagnostic) / baseline_diagnostic
        pct_improvement_deployed = 100.0 * (baseline_deployed - proxy_deployed) / baseline_deployed

        print(f"    deployed  (alpha={DEPLOYED_ALPHA}):  baseline={baseline_deployed:.4f}  "
              f"proxy={proxy_deployed:.4f}  improvement={pct_improvement_deployed:+.2f}%")
        print(f"    diagnostic(alpha={DIAGNOSTIC_ALPHA}):  baseline={baseline_diagnostic:.4f}  "
              f"proxy={proxy_diagnostic:.4f}  improvement={pct_improvement_diagnostic:+.2f}%\n")

        results.append({
            'session': session,
            'improvement_deployed_pct': pct_improvement_deployed,
            'improvement_diagnostic_pct': pct_improvement_diagnostic,
        })

    if not results:
        print("No sessions produced results -- check file availability (rnn_smoothed_output_*.csv).")
        return

    mean_diagnostic_improvement = np.mean([r['improvement_diagnostic_pct'] for r in results])
    print(f"=== Mean diagnostic-weighting improvement across {len(results)} "
          f"corruption sessions: {mean_diagnostic_improvement:+.2f}% ===")

    if mean_diagnostic_improvement >= PASS_THRESHOLD_PCT:
        print(f"PASS: >= {PASS_THRESHOLD_PCT}% threshold met.")
    else:
        print(f"STOP: < {PASS_THRESHOLD_PCT}% threshold. Legitimate stop signal.")


if __name__ == '__main__':
    main()
