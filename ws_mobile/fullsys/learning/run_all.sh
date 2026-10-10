#!/usr/bin/env bash
# F7 pipeline, development: features -> LOBO training -> closed-loop check. Stops at the first error.
# Usage (from ~/ws_mobile/fullsys):  nohup bash learning/run_all.sh 3 > ../results/fullsys_rel_run.log 2>&1 &
set -euo pipefail
JOBS=${1:-3}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")/.."
echo "== $(date '+%F %T') build_dataset (jobs $JOBS)"; python3 learning/build_dataset.py --jobs "$JOBS"
echo "== $(date '+%F %T') train (jobs $JOBS)";         python3 learning/train.py --jobs "$JOBS"
echo "== $(date '+%F %T') closed loop (jobs $JOBS)";   python3 learning/closed_loop.py --jobs "$JOBS"
echo "== $(date '+%F %T') ALL DONE"
