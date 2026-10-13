#!/usr/bin/env bash
# A3 development: retrain reliability + P_slip with seeds 1 and 2; closed loop ours/hybrid/no_pslip; summary.
# Usage (from ~/ws_mobile/fullsys):  nohup bash learning/run_a3.sh 3 > ../results/fullsys_a3_run.log 2>&1 &
set -euo pipefail
JOBS="${1:-3}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for SD in 1 2; do
  export FULLSYS_SEED=$SD
  echo "== $(date '+%F %T') seed $SD: train reliability"; python3 learning/train.py --jobs "$JOBS"
  echo "== $(date '+%F %T') seed $SD: train P_slip"; python3 learning/train_slip.py --jobs "$JOBS"
  echo "== $(date '+%F %T') seed $SD: closed loop"; python3 learning/baselines.py --jobs "$JOBS" --methods ours hybrid no_pslip
done
unset FULLSYS_SEED
echo "== $(date '+%F %T') summary"; python3 learning/a3_summary.py --seeds 42 1 2
echo "== $(date '+%F %T') ALL DONE"
