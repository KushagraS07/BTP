#!/usr/bin/env bash
# F14 development: baselines + ablations + metrics package (decision rule in FINDINGS F14).
# Usage (from ~/ws_mobile/fullsys):  nohup bash learning/run_baselines.sh 3 > ../results/fullsys_baselines_run.log 2>&1 &
set -euo pipefail
JOBS="${1:-3}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "== $(date '+%F %T') baselines (jobs $JOBS)"; python3 learning/baselines.py --jobs "$JOBS"
echo "== $(date '+%F %T') ALL DONE"
