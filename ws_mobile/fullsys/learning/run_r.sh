#!/usr/bin/env bash
# R1 + R2 development (FINDINGS R1, R2): detector quality + runtime; faults that end (detection, recovery, after-fault error).
# Usage (from ~/ws_mobile/fullsys):  nohup bash learning/run_r.sh 3 > ../results/fullsys_r_run.log 2>&1 &
set -euo pipefail
JOBS="${1:-3}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "== $(date '+%F %T') R1 std faults + runtime"; python3 learning/r1_quality.py --jobs "$JOBS"
echo "== $(date '+%F %T') R1 sigma faults"; FULLSYS_FAULT_UNIT=sigma python3 learning/r1_quality.py --jobs "$JOBS" --no-runtime
echo "== $(date '+%F %T') R2 build features (faults that end)"; FULLSYS_FAULT_WINDOW=end python3 learning/build_dataset.py --jobs "$JOBS"
echo "== $(date '+%F %T') R2 closed loop"; FULLSYS_FAULT_WINDOW=end python3 learning/r2_eval.py --jobs "$JOBS"
echo "== $(date '+%F %T') R1 on ending faults"; FULLSYS_FAULT_WINDOW=end python3 learning/r1_quality.py --jobs "$JOBS" --no-runtime
echo "== $(date '+%F %T') ALL DONE"
