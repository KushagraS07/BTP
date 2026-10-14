#!/usr/bin/env bash
# C1: calibration (FINDINGS C1). Needs config/cal_sessions.txt and the *_FINAL.pt models.
# Usage (from ~/ws_mobile/fullsys):  nohup bash learning/run_cal.sh 3 [--rlidar] > ../results/fullsys_cal_run.log 2>&1 &
set -euo pipefail
JOBS="${1:-3}"
RL="${2:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
unset FULLSYS_FAULT_UNIT FULLSYS_FAULT_WINDOW FULLSYS_SEED FULLSYS_LI_VARIANT || true
echo "== $(date '+%F %T') calibration features"; python3 learning/build_dataset.py --jobs "$JOBS" --list config/cal_sessions.txt
echo "== $(date '+%F %T') calibration closed loop $RL"; python3 learning/cal_eval.py --jobs "$JOBS" $RL
echo "== $(date '+%F %T') ALL DONE"
