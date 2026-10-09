#!/usr/bin/env python3
"""
oracle_mapping_test.py -- READ-ONLY on ~/ws_mobile (writes only to ~/ws_mobile/fix_ws).

Question: with PERFECT reliability (true labels), can ANY way of using r_imu make the
adaptive EKF beat the fixed EKF on IMU faults? The linear mapping R/(r+eps) reached
only 56/112 wins and no mean-RMSE gain (oracle_reliability_test.py).

The adaptive EKF (lstm_adaptive_ekf.evaluate_session) is used UNMODIFIED. Each
variant is produced by transforming the true r_imu label that the oracle feeds in,
so that the core's own R/(r+eps) yields the intended mapping. r_odom always stays the
true label, so only the IMU mapping changes.

PRE-REGISTERED variants (fixed before running; decision 2026-10-03):
  linear     r' = r                          (baseline: what Phase 2 uses)
  squared    r' = r^2                        -> R / r^2   (option b)
  gate05     r' = 1 if r >= 0.5 else 0       -> hard gate at 0.5 (option a)
  exclude    r' = 1 if r >= 0.999 else 0     -> ideal fault exclusion: IMU ignored
                                                whenever its label shows ANY fault
"0" means R_imu x 1/eps = x1000 (sigma ~2.2 rad), i.e. the IMU update is effectively
skipped (Kalman gain ~1e-3). Pass rule: a variant "can beat fixed on IMU faults" only if
it wins on >50% of the 112 IMU-fault sessions AND has lower mean RMSE than fixed.
If none passes, no reliability->R mapping can fix IMU faults in this filter, and only
an alternative heading source (option c) remains.

Runs on the 112 IMU-fault sessions + 7 no-fault baselines.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

WS = os.path.expanduser('~/ws_mobile')
sys.path.insert(0, os.path.join(WS, 'scripts'))
import lstm_adaptive_ekf as L  # noqa: E402
from build_training_dataset import discover_all_sessions, session_to_baseline, parse_fault_info  # noqa: E402

OUT = os.path.join(WS, 'fix_ws', 'oracle_mapping_results.csv')
VARIANTS = {
    'linear':  lambda r: r,
    'squared': lambda r: r ** 2,
    'gate05':  lambda r: (r >= 0.5).astype(np.float32),
    'exclude': lambda r: (r >= 0.999).astype(np.float32),
}


class OracleModel:
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
    sessions = []
    for s in discover_all_sessions():
        b = session_to_baseline(s)
        sensor, _, _ = parse_fault_info(s, b)
        if sensor in (None, 'imu') or s == b:
            sessions.append(s)
    print(f'evaluating {len(sessions)} sessions x {len(VARIANTS)} variants', flush=True)

    cache, rows = {}, []
    for k, s in enumerate(sessions, 1):
        b = session_to_baseline(s)
        if b not in cache:
            cache[b] = L.load_model(b) if (L.SENSOR_LOGS_DIR / f'model_{b}.pt').exists() else None
        if cache[b] is None:
            continue
        _, mean, std, feats, win = cache[b]
        lab = pd.read_csv(L.SENSOR_LOGS_DIR / f'reliability_labels_{s}.csv')[['r_imu', 'r_odom']].values
        sensor, ftype, sev = parse_fault_info(s, b)
        row = dict(session=s, baseline=b, sensor=sensor, fault_type=ftype, severity=sev,
                   frac_imu_faulty=float((lab[:, 0] < 0.999).mean()))
        for name, f in VARIANTS.items():
            l2 = lab.copy()
            l2[:, 0] = f(lab[:, 0])
            orc = OracleModel(l2)
            fixed, heur, rm = L.evaluate_session(s, b, orc, mean, std, feats, win)
            if orc.i != len(lab):
                sys.exit(f'ABORT {s}/{name}: oracle called {orc.i}x, labels {len(lab)} rows')
            row['fixed'], row['heuristic'], row[name] = fixed, heur, rm
        rows.append(row)
        print(f'[{k}/{len(sessions)}] {s}: fixed={row["fixed"]:.3f} ' +
              ' '.join(f'{n}={row[n]:.3f}' for n in VARIANTS), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    imu = df[df['sensor'] == 'imu']
    base = df[df['sensor'] != 'imu']

    print(f'\n=== IMU-fault sessions (n={len(imu)}): beats fixed, mean RMSE ===')
    print(f'{"variant":10s} {"wins":>9s} {"mean RMSE":>10s}   pass?')
    print(f'{"fixed":10s} {"-":>9s} {imu["fixed"].mean():10.3f}')
    for n in ['heuristic'] + list(VARIANTS):
        w = int((imu[n] < imu['fixed']).sum())
        m = imu[n].mean()
        ok = (w > len(imu) / 2) and (m < imu['fixed'].mean())
        print(f'{n:10s} {w:>4d}/{len(imu):<4d} {m:10.3f}   {"PASS" if ok else "no"}')

    print('\n=== by IMU fault type: wins vs fixed ===')
    cols = ['heuristic'] + list(VARIANTS)
    print(f'{"fault":10s} ' + ' '.join(f'{c:>10s}' for c in cols))
    for ft, g in imu.groupby('fault_type'):
        print(f'{ft:10s} ' + ' '.join(f'{int((g[c] < g["fixed"]).sum()):>5d}/{len(g):<4d}' for c in cols))

    print(f'\nno-fault baselines (n={len(base)}) mean RMSE: ' +
          ' '.join(f'{c}={base[c].mean():.4f}' for c in ['fixed'] + cols))
    print(f'\nSaved: {OUT}')


if __name__ == '__main__':
    main()
