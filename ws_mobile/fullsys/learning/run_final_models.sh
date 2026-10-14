#!/usr/bin/env bash
# C0: FINAL models on the dev sessions only (FINDINGS C0). Run BEFORE recording (low load needed while recording).
# Usage (from ~/ws_mobile/fullsys):  nohup bash learning/run_final_models.sh "rel slip rlidar" 3 > ../results/fullsys_final_models.log 2>&1 &
set -euo pipefail
KINDS="${1:-rel slip rlidar}"
JOBS="${2:-3}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
unset FULLSYS_FAULT_UNIT FULLSYS_FAULT_WINDOW FULLSYS_SEED || true
echo "== $(date '+%F %T') FINAL models: $KINDS"; python3 learning/train_final.py $KINDS --jobs "$JOBS"
echo "== $(date '+%F %T') ALL DONE"
