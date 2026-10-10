#!/usr/bin/env bash
# F13 development: Q(t) variants on top of reliability + LiDAR + 5c (one run; decision rule in FINDINGS F13).
# Usage (from ~/ws_mobile/fullsys):  nohup bash learning/run_q.sh 3 > ../results/fullsys_q_run.log 2>&1 &
set -euo pipefail
JOBS="${1:-3}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "== $(date '+%F %T') closed loop Q(t) (jobs $JOBS)"; python3 learning/closed_loop_q.py --jobs "$JOBS"
echo "== $(date '+%F %T') ALL DONE"
