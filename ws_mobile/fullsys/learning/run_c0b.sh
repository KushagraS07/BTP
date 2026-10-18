#!/usr/bin/env bash
# C0b + C1/C1b overnight (FINDINGS C0b): backup C0 models -> D9-D14 features -> FINAL models on D1-D14 -> calibration.
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
unset FULLSYS_FAULT_UNIT FULLSYS_FAULT_WINDOW FULLSYS_SEED FULLSYS_LI_VARIANT || true
M=../fullsys_logs/models
for k in rel slip li; do [ -f $M/${k}_FINAL_c0.pt ] || cp $M/${k}_FINAL.pt $M/${k}_FINAL_c0.pt; done
echo "== $(date '+%F %T') C0 models backed up"; ls $M/*_FINAL_c0.pt
echo "== $(date '+%F %T') features D9-D14"
python3 learning/build_dataset.py --jobs 3 --only 20261017_155638 20261017_160044 20261017_161215 20261017_161659 20261017_163826 20261017_164350
echo "== $(date '+%F %T') FINAL models on D1-D14"; python3 learning/train_final.py rel slip li --jobs 3
echo "== $(date '+%F %T') calibration"; bash learning/run_cal.sh 3
echo "== $(date '+%F %T') C0B ALL DONE"
