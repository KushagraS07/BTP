#!/usr/bin/env python3
"""
run_pipeline_iqroff.py -- full downstream re-run after USE_IQR=False.

Run ONLY after the 7 baselines have been re-preprocessed + EKF-checked by hand.
Stops at the first failing step. Per-step logs: ~/ws_mobile/fix_ws/iqroff_logs/

Usage (from anywhere):
    nohup python3 ~/ws_mobile/fix_ws/run_pipeline_iqroff.py > ~/ws_mobile/fix_ws/iqroff_logs/_master.log 2>&1 &
    disown
Progress:
    cat ~/ws_mobile/fix_ws/iqroff_logs/_master.log
Resume from a later step after fixing a failure (e.g. step 5):
    python3 ~/ws_mobile/fix_ws/run_pipeline_iqroff.py --from 5
"""
import argparse
import filecmp
import glob
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

WS = os.path.expanduser('~/ws_mobile')
SCRIPTS = os.path.join(WS, 'scripts')
LOG_DIR = os.path.join(WS, 'sensor_logs')
LOGS = os.path.join(WS, 'fix_ws', 'iqroff_logs')
PY = sys.executable
MIN_FREE_GB = 4
N_FAULTS = 364      # 7 baselines x (16 IMU + 16 odom + 20 LiDAR)
N_LABELS = 231      # 7 baselines + 224 IMU/odom fault sessions


def log(msg):
    print(f'[{datetime.now():%F %T}] {msg}', flush=True)


def abort(msg):
    log(f'ABORT: {msg}')
    sys.exit(1)


def check_disk():
    free_gb = shutil.disk_usage(os.path.expanduser('~')).free / 1e9
    log(f'disk free: {free_gb:.1f} GB')
    if free_gb < MIN_FREE_GB:
        abort(f'less than {MIN_FREE_GB} GB free')


def run(name, cmd, cwd):
    """Run cmd (list), write stdout+stderr to LOGS/name.log, abort on non-zero exit."""
    path = os.path.join(LOGS, f'{name}.log')
    log(f'START {name}')
    with open(path, 'w') as f:
        rc = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        abort(f'{name} exited with code {rc} (see {path})')
    log(f'OK    {name}')
    return path


# ---------------------------------------------------------------- steps
def s01_faults():
    p = run('01_faults',
            [PY, '-u', '-c', 'from fault_injection import generate_all_faults; generate_all_faults()'],
            SCRIPTS)
    txt = open(p).read()
    if f'Done: {N_FAULTS}/{N_FAULTS}' not in txt:
        abort(f'fault regeneration not {N_FAULTS}/{N_FAULTS}')
    if 'SKIPPED' in txt:
        abort('SKIPPED sessions in fault regeneration')


def s02_validate_faults():
    run('02_validate_faults', [PY, '-u', 'scripts/validate_faults.py'], WS)


def s03_rnn():
    run('03_rnn', [PY, '-u', 'rnn_fusion_v2.py'], SCRIPTS)


def s04_rnn_refresh():
    log('START 04_rnn_refresh')
    n, missing = 0, []
    for mp in sorted(glob.glob(os.path.join(LOG_DIR, 'fault_meta_*.json'))):
        m = json.load(open(mp))
        src = os.path.join(LOG_DIR, f"rnn_smoothed_output_{m['baseline_session']}.csv")
        dst = os.path.join(LOG_DIR, f"rnn_smoothed_output_{m['fault_session']}.csv")
        if not os.path.exists(src):
            missing.append(src)
            continue
        shutil.copy2(src, dst)
        if not filecmp.cmp(src, dst, shallow=False):
            abort(f'copy mismatch: {dst}')
        n += 1
    log(f'refreshed {n}; missing {sorted(set(missing))}')
    if n != N_FAULTS or missing:
        abort('RNN refresh incomplete')
    log('OK    04_rnn_refresh')


def s05_lidar():
    run('05_lidar', [PY, '-u', 'lidar_fusion_v2.py'], SCRIPTS)


def s06_lidar_faults():
    run('06_lidar_faults', [PY, '-u', 'validate_lidar_faults.py'], SCRIPTS)


def s07_labels():
    run('07_labels', [PY, '-u', 'generate_reliability_labels.py'], SCRIPTS)
    n = len(glob.glob(os.path.join(LOG_DIR, 'reliability_labels_*.csv')))
    if n != N_LABELS:
        abort(f'expected {N_LABELS} label files, got {n}')


def s08_dataset_stage1():
    check_disk()
    run('08_dataset_stage1', [PY, '-u', 'scripts/build_training_dataset.py', '--stage1'], WS)


def s09_dataset_stage2():
    run('09_dataset_stage2', [PY, '-u', 'scripts/build_training_dataset.py', '--stage2'], WS)


def s10_lstm_train():
    # exactly the reported configuration
    run('10_lstm_train', [PY, '-u', 'scripts/train_reliability_lstm.py',
                          '--all-folds', '--closed-loop', '--baseline-weight', '5.0'], WS)


def s11_compare_phase1():
    run('11_compare_phase1', [PY, '-u', 'scripts/compare_adaptive_vs_fixed.py'], WS)


def s12_lstm_adaptive_ekf():
    run('12_lstm_adaptive_ekf', [PY, '-u', 'scripts/lstm_adaptive_ekf.py', '--all'], WS)


STEPS = [s01_faults, s02_validate_faults, s03_rnn, s04_rnn_refresh, s05_lidar,
         s06_lidar_faults, s07_labels, s08_dataset_stage1, s09_dataset_stage2,
         s10_lstm_train, s11_compare_phase1, s12_lstm_adaptive_ekf]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from', dest='start', type=int, default=1,
                    help=f'step number to start from (1-{len(STEPS)})')
    args = ap.parse_args()
    if not 1 <= args.start <= len(STEPS):
        abort(f'--from must be 1-{len(STEPS)}')

    os.makedirs(LOGS, exist_ok=True)
    log(f'pipeline start (from step {args.start}); python={PY}')
    check_disk()
    for step in STEPS[args.start - 1:]:
        step()
    log('ALL STEPS DONE')


if __name__ == '__main__':
    main()
