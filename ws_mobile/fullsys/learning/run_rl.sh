#!/usr/bin/env bash
# L2 development (FINDINGS L2): learned r_LiDAR -- features, LOBO training, closed loop + rule (3).
# Usage (from ~/ws_mobile/fullsys):  nohup bash learning/run_rl.sh 3 > ../results/fullsys_rl_run.log 2>&1 &
set -euo pipefail
JOBS="${1:-3}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
unset FULLSYS_FAULT_UNIT FULLSYS_FAULT_WINDOW FULLSYS_SEED || true
echo "== $(date '+%F %T') build LiDAR features"; python3 learning/lidar_rel.py build --jobs "$JOBS"
echo "== $(date '+%F %T') train r_LiDAR"; python3 learning/lidar_rel.py train --jobs "$JOBS"
echo "== $(date '+%F %T') closed loop"; python3 learning/lidar_rel.py eval --jobs "$JOBS"
echo "== $(date '+%F %T') ALL DONE"
