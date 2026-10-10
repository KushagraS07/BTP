#!/usr/bin/env bash
# F11 integration step 1, development: P_slip LOBO training -> closed loop (fixed/pslip/lstm/full/oracle_full).
# Needs the F10 feature files and rel_<h>.pt models. Stops at the first error.
# Usage (from ~/ws_mobile/fullsys):  nohup bash learning/run_int.sh 3 > ../results/fullsys_int_run.log 2>&1 &
set -euo pipefail
JOBS="${1:-3}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "== $(date '+%F %T') train_slip (jobs $JOBS)"; python3 learning/train_slip.py --jobs "$JOBS"
echo "== $(date '+%F %T') closed loop integration (jobs $JOBS)"; python3 learning/closed_loop_int.py --jobs "$JOBS"
echo "== $(date '+%F %T') ALL DONE"
