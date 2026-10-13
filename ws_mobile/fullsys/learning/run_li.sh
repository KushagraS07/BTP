#!/usr/bin/env bash
# F14b development: Li et al.-style classifier (LOBO) + closed loop with hooks-off regression vs F14 fixed.
# Needs results/fullsys_baselines.csv from F14. Usage (from ~/ws_mobile/fullsys):
#   nohup bash learning/run_li.sh 3 > ../results/fullsys_li_run.log 2>&1 &
set -euo pipefail
JOBS="${1:-3}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "== $(date '+%F %T') train_li (jobs $JOBS)"; python3 learning/train_li.py --jobs "$JOBS"
echo "== $(date '+%F %T') li_baseline (jobs $JOBS)"; python3 learning/li_baseline.py --jobs "$JOBS" --check
echo "== $(date '+%F %T') ALL DONE"
