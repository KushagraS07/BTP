#!/usr/bin/env bash
# F14b-det development: Li-style with OUR detector for ZUPT (classifier only for NHC); models li_<h>.pt reused.
# Usage (from ~/ws_mobile/fullsys):  nohup bash learning/run_li_det.sh 2 > ../results/fullsys_li_det_run.log 2>&1 &
set -euo pipefail
JOBS="${1:-2}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 FULLSYS_LI_VARIANT=li_det
echo "== $(date '+%F %T') li_det (jobs $JOBS)"; python3 learning/li_baseline.py --jobs "$JOBS"
echo "== $(date '+%F %T') stats"
python3 learning/stats.py ../results/fullsys_baselines.csv --ref hybrid --extra ../results/fullsys_li_baseline.csv ../results/fullsys_li_det.csv --out ../results/fullsys_li_det_stats.csv
echo "== $(date '+%F %T') ALL DONE"
