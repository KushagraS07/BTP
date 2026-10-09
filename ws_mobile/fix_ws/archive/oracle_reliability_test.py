#!/usr/bin/env python3
"""
oracle_reliability_test.py -- READ-ONLY on ~/ws_mobile (writes only to ~/ws_mobile/fix_ws).

Question: the LSTM predicts r_imu well (LOBO RMSE 0.255 vs 0.343 fit-mean), yet the
LSTM-adaptive EKF loses to the fixed EKF on 71% of IMU-fault sessions. Is the
limit the LSTM's prediction error, or the way reliability is used (R / (r + eps))?

Test: run the EXACT same adaptive EKF (lstm_adaptive_ekf.evaluate_session, unmodified)
twice per session:
  - lstm   : the trained LOBO model's r predictions (as in Phase 2)
  - oracle : the TRUE reliability labels (reliability_labels_<session>.csv)
The oracle is a stand-in "model" object that returns the true label for step i on
its i-th call. The script verifies it was called exactly once per row, so the
oracle is fed in the same place and order as the LSTM.

Interpretation:
  oracle beats fixed on IMU faults, LSTM doesn't -> prediction error is the limit
  oracle ALSO loses on IMU faults               -> the R-scaling design is the limit
"""
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

WS = os.path.expanduser('~/ws_mobile')
sys.path.insert(0, os.path.join(WS, 'scripts'))
import lstm_adaptive_ekf as L  # noqa: E402
from build_training_dataset import discover_all_sessions, session_to_baseline, parse_fault_info  # noqa: E402

OUT = os.path.join(WS, 'fix_ws', 'oracle_results.csv')


class OracleModel:
    """Returns the true (r_imu, r_odom) label for the current step on each call."""

    def __init__(self, labels):
        self.r = np.asarray(labels, dtype=np.float32)
        self.i = 0

    def __call__(self, x):
        out = torch.from_numpy(self.r[min(self.i, len(self.r) - 1)][None, :].copy())
        self.i += 1
        return out

    def eval(self):
        return self


def main():
    sessions = discover_all_sessions()
    print(f'evaluating {len(sessions)} sessions (LSTM + oracle)', flush=True)
    cache, rows = {}, []
    for k, s in enumerate(sessions, 1):
        b = session_to_baseline(s)
        if b not in cache:
            cache[b] = L.load_model(b) if (L.SENSOR_LOGS_DIR / f'model_{b}.pt').exists() else None
        if cache[b] is None:
            continue
        model, mean, std, feats, win = cache[b]
        lab_path = L.SENSOR_LOGS_DIR / f'reliability_labels_{s}.csv'
        if not lab_path.exists():
            print(f'  SKIP {s}: no labels', flush=True)
            continue
        labels = pd.read_csv(lab_path)[['r_imu', 'r_odom']].values

        fixed, heur, lstm = L.evaluate_session(s, b, model, mean, std, feats, win)
        oracle = OracleModel(labels)
        _, _, orc = L.evaluate_session(s, b, oracle, mean, std, feats, win)
        if oracle.i != len(labels):
            sys.exit(f'ABORT {s}: oracle called {oracle.i} times but labels have {len(labels)} rows '
                     f'-- the oracle is not aligned with the LSTM call pattern')

        sensor, ftype, sev = parse_fault_info(s, b)
        rows.append(dict(session=s, baseline=b, sensor=sensor, fault_type=ftype, severity=sev,
                         fixed=fixed, heuristic=heur, lstm=lstm, oracle=orc))
        print(f'[{k}/{len(sessions)}] {s}: fixed={fixed:.4f} lstm={lstm:.4f} oracle={orc:.4f}', flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    flt = df[df['sensor'].notna() & (df['sensor'].isin(['imu', 'odom']))]
    print('\n=== beats FIXED, IMU/odom fault sessions ===')
    agg = defaultdict(lambda: [0, 0, 0, 0])
    for _, r in flt.iterrows():
        for key in (r.sensor, f'{r.sensor}/{r.fault_type}'):
            a = agg[key]
            a[0] += r.lstm < r.fixed
            a[1] += r.oracle < r.fixed
            a[2] += r.heuristic < r.fixed
            a[3] += 1
    print(f'{"group":16s} {"heuristic":>10s} {"lstm":>10s} {"oracle":>10s}')
    for key in sorted(agg, key=lambda x: (x.count('/'), x)):
        a = agg[key]
        print(f'{key:16s} {a[2]:>4d}/{a[3]:<5d} {a[0]:>4d}/{a[3]:<5d} {a[1]:>4d}/{a[3]:<5d}')

    print('\n=== mean RMSE (m) ===')
    print(flt.groupby('sensor')[['fixed', 'heuristic', 'lstm', 'oracle']].mean().round(3).to_string())
    base = df[df['sensor'].isna() | ~df['sensor'].isin(['imu', 'odom', 'lidar'])]
    if len(base):
        print('\nno-fault baselines:', base[['fixed', 'heuristic', 'lstm', 'oracle']].mean().round(4).to_dict())
    print(f'\nSaved: {OUT}')


if __name__ == '__main__':
    main()
