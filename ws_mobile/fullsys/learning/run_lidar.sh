#!/usr/bin/env bash
# F12 development: reliability r (F10 models) + LiDAR scan matching + 5c in EKF9 (closed loop only).
# Usage (from ~/ws_mobile/fullsys):  nohup bash learning/run_lidar.sh 3 > ../results/fullsys_lidar_run.log 2>&1 &
set -euo pipefail
JOBS="${1:-3}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "== $(date '+%F %T') closed loop LiDAR (jobs $JOBS)"; python3 learning/closed_loop_lidar.py --jobs "$JOBS"
echo "== $(date '+%F %T') ALL DONE"
