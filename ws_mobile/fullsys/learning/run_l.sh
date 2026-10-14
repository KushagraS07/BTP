#!/usr/bin/env bash
# L step development (FINDINGS L): LiDAR faults, chi2 gate on scan matching, stale-scan check, overlapping faults.
# Usage (from ~/ws_mobile/fullsys):  nohup bash learning/run_l.sh 3 > ../results/fullsys_l_run.log 2>&1 &
set -euo pipefail
JOBS="${1:-3}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
unset FULLSYS_FAULT_UNIT FULLSYS_FAULT_WINDOW FULLSYS_SEED || true
echo "== $(date '+%F %T') L closed loop (jobs $JOBS)"; python3 learning/l_eval.py --jobs "$JOBS"
echo "== $(date '+%F %T') ALL DONE"
