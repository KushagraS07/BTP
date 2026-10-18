#!/usr/bin/env bash
# FR: confirmatory run (FINDINGS FR). Needs config/test_sessions.txt, *_FINAL.pt, results/fullsys_calibration_choice.json.
# Usage (from ~/ws_mobile/fullsys):  nohup bash learning/run_final.sh 3 > ../results/fullsys_final_run.log 2>&1 &
# Resumable: a set whose results/final/final_<set>.csv exists is skipped (delete it to redo that set).
set -euo pipefail
JOBS="${1:-3}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
unset FULLSYS_FAULT_UNIT FULLSYS_FAULT_WINDOW FULLSYS_SEED FULLSYS_LI_VARIANT || true
L=config/test_sessions.txt
M=../fullsys_logs/models
R=../results/final
mkdir -p $R
echo "== $(date '+%F %T') provenance"
echo "commit $(git rev-parse --short HEAD); uncommitted files: $(git status --porcelain | wc -l)"
sha256sum $L $M/rel_FINAL.pt $M/slip_FINAL.pt $M/li_FINAL.pt ../results/fullsys_calibration_choice.json | cut -c1-16,65-
echo "== $(date '+%F %T') features (std / sigma / end / unseen)"
python3 learning/build_dataset.py --jobs "$JOBS" --list $L
FULLSYS_FAULT_UNIT=sigma python3 learning/build_dataset.py --jobs "$JOBS" --list $L
FULLSYS_FAULT_WINDOW=end python3 learning/build_dataset.py --jobs "$JOBS" --list $L
python3 learning/e_build.py --jobs "$JOBS" --list $L
for S in main lidar unseen sigma end; do
  if [ -f $R/final_$S.csv ]; then echo "== $S exists, skipped"; continue; fi
  echo "== $(date '+%F %T') set $S"
  case $S in
    sigma) FULLSYS_FAULT_UNIT=sigma python3 learning/final_eval.py --set $S --jobs "$JOBS" ;;
    end)   FULLSYS_FAULT_WINDOW=end python3 learning/final_eval.py --set $S --jobs "$JOBS" ;;
    *)     python3 learning/final_eval.py --set $S --jobs "$JOBS" ;;
  esac
done
echo "== $(date '+%F %T') statistics"; python3 learning/final_stats.py
echo "== $(date '+%F %T') FINAL ALL DONE"
