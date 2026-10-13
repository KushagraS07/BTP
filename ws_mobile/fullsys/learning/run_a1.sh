#!/usr/bin/env bash
# A1 development: physically scaled (sigma) faults, models unchanged (FINDINGS A1).
# Usage (from ~/ws_mobile/fullsys):  nohup bash learning/run_a1.sh 3 > ../results/fullsys_a1_run.log 2>&1 &
set -euo pipefail
JOBS="${1:-3}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 FULLSYS_FAULT_UNIT=sigma
echo "== $(date '+%F %T') build sigma features (jobs $JOBS)"; python3 learning/build_dataset.py --jobs "$JOBS"
echo "== $(date '+%F %T') baselines on sigma faults (jobs $JOBS)"
python3 learning/baselines.py --jobs "$JOBS" --methods fixed chi2 sage_husa heuristic ours hybrid no_pslip hybrid_bq
echo "== $(date '+%F %T') A2 statistics"
python3 learning/stats.py ../results/fullsys_baselines_sigma.csv --ref hybrid --out ../results/fullsys_baselines_sigma_stats.csv
echo "== $(date '+%F %T') ALL DONE"
