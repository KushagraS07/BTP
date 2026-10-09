#!/usr/bin/env python3
"""
lstm_gate_test.py -- READ-ONLY on ~/ws_mobile (writes only to ~/ws_mobile/fix_ws).

The oracle mapping test showed that, with PERFECT reliability, hard-gating the IMU
update (skip when r_imu < 0.5) beats the fixed EKF on 73/112 IMU-fault sessions and
cuts mean RMSE 5.706 -> 4.861 m, while the current linear R/(r+eps) mapping
reaches only 56/112 and 5.734 m.

This test answers the real question: does the gate still help with the trained
LSTM's PREDICTED r_imu (not the labels)?

Method: the trained LOBO LSTM is wrapped; its r_imu output is transformed before the
UNMODIFIED adaptive EKF (lstm_adaptive_ekf.evaluate_session) uses it. r_odom is passed
through unchanged (the linear mapping already wins 89/112 on odom faults).

PRE-REGISTERED (decision 2026-10-03, fixed before running):
  lstm_linear : r_imu' = r_imu                    (current Phase 2; should reproduce
                                                    32/112 IMU, 89/112 odom)
  lstm_gate05 : r_imu' = 1 if r_imu >= 0.5 else 0 (gate threshold taken from the oracle
                                                    test, which used labels, not this
                                                    model's predictions)
Pass rule for lstm_gate05: on the 112 IMU-fault sessions it must beat fixed on >50%
AND have lower mean RMSE than fixed; on the 7 no-fault baselines its mean RMSE must be
no more than 1% worse than fixed (gating must not hurt clean data); and on the 112
odom-fault sessions it must not lose wins vs lstm_linear by more than 5.

Runs all 231 sessions (IMU, odom, LiDAR faults and baselines) with both variants.
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

OUT = os.path.join(WS, 'fix_ws', 'lstm_gate_results.csv')
GATE = 0.5


class Wrapped:
    """Calls the real LSTM, then transforms its r_imu output (column 0)."""

    def __init__(self, model, transform):
        self.model, self.transform, self.n = model, transform, 0

    def __call__(self, x):
        out = self.model(x).clone()
        out[:, 0] = self.transform(out[:, 0])
        self.n += 1
        return out

    def eval(self):
        return self


VARIANTS = {
    'lstm_linear': lambda r: r,
    'lstm_gate05': lambda r: (r >= GATE).to(r.dtype),
}


def main():
    sessions = discover_all_sessions()
    print(f'evaluating {len(sessions)} sessions x {len(VARIANTS)} variants', flush=True)
    cache, rows = {}, []
    for k, s in enumerate(sessions, 1):
        b = session_to_baseline(s)
        if b not in cache:
            cache[b] = L.load_model(b) if (L.SENSOR_LOGS_DIR / f'model_{b}.pt').exists() else None
        if cache[b] is None:
            continue
        model, mean, std, feats, win = cache[b]
        sensor, ftype, sev = parse_fault_info(s, b)
        row = dict(session=s, baseline=b, sensor=sensor, fault_type=ftype, severity=sev)
        for name, f in VARIANTS.items():
            w = Wrapped(model, f)
            with torch.no_grad():
                fixed, heur, rm = L.evaluate_session(s, b, w, mean, std, feats, win)
            if w.n == 0:
                sys.exit(f'ABORT {s}/{name}: wrapped model was never called')
            row['fixed'], row['heuristic'], row[name] = fixed, heur, rm
        rows.append(row)
        print(f'[{k}/{len(sessions)}] {s}: fixed={row["fixed"]:.3f} ' +
              ' '.join(f'{n}={row[n]:.3f}' for n in VARIANTS), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    cols = ['heuristic'] + list(VARIANTS)

    for sens in ('imu', 'odom'):
        g = df[df['sensor'] == sens]
        print(f'\n=== {sens.upper()}-fault sessions (n={len(g)}) ===')
        print(f'{"variant":12s} {"wins vs fixed":>14s} {"mean RMSE":>10s}')
        print(f'{"fixed":12s} {"-":>14s} {g["fixed"].mean():10.3f}')
        for c in cols:
            print(f'{c:12s} {int((g[c] < g["fixed"]).sum()):>8d}/{len(g):<5d} {g[c].mean():10.3f}')
        print('  by fault type (wins vs fixed): ' + '; '.join(
            f'{ft}: ' + ', '.join(f'{c.replace("lstm_", "")} {int((gg[c] < gg["fixed"]).sum())}/{len(gg)}' for c in cols)
            for ft, gg in g.groupby('fault_type')))

    base = df[~df['sensor'].isin(['imu', 'odom', 'lidar'])]
    print(f'\n=== no-fault baselines (n={len(base)}) mean RMSE ===')
    print('  ' + ' '.join(f'{c}={base[c].mean():.4f}' for c in ['fixed'] + cols))

    imu = df[df['sensor'] == 'imu']
    odom = df[df['sensor'] == 'odom']
    g, ln = 'lstm_gate05', 'lstm_linear'
    checks = {
        'IMU wins > 50%': (imu[g] < imu['fixed']).sum() > len(imu) / 2,
        'IMU mean RMSE < fixed': imu[g].mean() < imu['fixed'].mean(),
        'baselines <= 1% worse than fixed': base[g].mean() <= 1.01 * base['fixed'].mean(),
        'odom wins not down >5 vs linear': (odom[g] < odom['fixed']).sum() >= (odom[ln] < odom['fixed']).sum() - 5,
    }
    print('\n=== PRE-REGISTERED PASS RULE (lstm_gate05) ===')
    for k2, v in checks.items():
        print(f'  [{"PASS" if v else "FAIL"}] {k2}')
    print(f'  OVERALL: {"PASS" if all(checks.values()) else "FAIL"}')
    print(f'\nSaved: {OUT}')


if __name__ == '__main__':
    main()
