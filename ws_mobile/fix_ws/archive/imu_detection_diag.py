#!/usr/bin/env python3
"""
imu_detection_diag.py -- READ-ONLY on ~/ws_mobile (writes only to ~/ws_mobile/fix_ws).

Why did gating help with the TRUE labels (73/112 IMU wins) but not with the LSTM's
predicted r_imu (38/112)? This records the LSTM's actual r_imu predictions inside the
unmodified adaptive EKF run (linear mapping, exactly Phase 2) and scores them as a
fault DETECTOR against the labels (faulty row = label r_imu < 0.999):

  per IMU fault type and overall:
    detect@0.5  share of faulty rows the gate would catch   (pred < 0.5)
    false@0.5   share of clean rows the gate would wrongly drop (pred < 0.5)
    AUC         threshold-free separability of faulty vs clean rows
    pred mean on faulty / clean rows
  plus the same false-alarm rate on the 7 no-fault baselines.

Descriptive only: NO threshold is chosen from this output (that would be tuning on
the evaluation data).
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

OUT = os.path.join(WS, 'fix_ws', 'imu_detection_rows.csv')


class Recorder:
    def __init__(self, model):
        self.model, self.r = model, []

    def __call__(self, x):
        out = self.model(x)
        self.r.append(float(out[0, 0]))
        return out

    def eval(self):
        return self


def auc(score_pos, score_neg):
    """P(faulty row scores LOWER than clean row) -- low r means 'faulty'."""
    if len(score_pos) == 0 or len(score_neg) == 0:
        return float('nan')
    s = np.concatenate([score_pos, score_neg])
    ranks = pd.Series(s).rank().values
    rp = ranks[:len(score_pos)].sum()
    n1, n0 = len(score_pos), len(score_neg)
    return 1.0 - (rp - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    sessions = []
    for s in discover_all_sessions():
        b = session_to_baseline(s)
        sensor, _, _ = parse_fault_info(s, b)
        if s == b or sensor == 'imu':
            sessions.append(s)
    print(f'recording r_imu on {len(sessions)} sessions', flush=True)
    cache, parts = {}, []
    for k, s in enumerate(sessions, 1):
        b = session_to_baseline(s)
        if b not in cache:
            cache[b] = L.load_model(b) if (L.SENSOR_LOGS_DIR / f'model_{b}.pt').exists() else None
        if cache[b] is None:
            continue
        model, mean, std, feats, win = cache[b]
        lab = pd.read_csv(L.SENSOR_LOGS_DIR / f'reliability_labels_{s}.csv')['r_imu'].values
        rec = Recorder(model)
        with torch.no_grad():
            L.evaluate_session(s, b, rec, mean, std, feats, win)
        if len(rec.r) != len(lab):
            sys.exit(f'ABORT {s}: {len(rec.r)} predictions vs {len(lab)} labels')
        sensor, ftype, _ = parse_fault_info(s, b)
        parts.append(pd.DataFrame({'session': s, 'fault_type': ftype if s != b else 'none',
                                   'label': lab, 'pred': rec.r}))
        print(f'[{k}/{len(sessions)}] {s}', flush=True)

    d = pd.concat(parts, ignore_index=True)
    d.to_csv(OUT, index=False)
    d['faulty'] = d['label'] < 0.999

    print('\n=== LSTM r_imu as an IMU-fault detector (rows) ===')
    print(f'{"group":10s} {"faulty_rows":>11s} {"detect@0.5":>10s} {"false@0.5":>10s} '
          f'{"AUC":>6s} {"pred|faulty":>11s} {"pred|clean":>10s}')
    groups = [('ALL_IMU', d[d.fault_type != 'none'])] + \
             [(ft, g) for ft, g in d[d.fault_type != 'none'].groupby('fault_type')] + \
             [('baselines', d[d.fault_type == 'none'])]
    for name, g in groups:
        f, c = g[g.faulty], g[~g.faulty]
        det = (f.pred < 0.5).mean() if len(f) else float('nan')
        fa = (c.pred < 0.5).mean() if len(c) else float('nan')
        print(f'{name:10s} {len(f):>11d} {det:>10.1%} {fa:>10.1%} {auc(f.pred.values, c.pred.values):>6.3f} '
              f'{f.pred.mean() if len(f) else float("nan"):>11.3f} {c.pred.mean():>10.3f}')
    print(f'\nSaved: {OUT}')


if __name__ == '__main__':
    main()
