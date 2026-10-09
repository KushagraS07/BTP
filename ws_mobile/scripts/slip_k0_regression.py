#!/usr/bin/env python3
"""
slip_k0_regression.py -- Phase 3B equivalence check (PREREG_slip.md §5). Read-only except the
report it writes. Must PASS before any development result of Q(t) is used.

  (a) frozen Phase 2:  lstm_adaptive_ekf.run_lstm_adaptive_ekf_core, unmodified (slip_q.run, k=None)
  (b) Q(t) code, k=0:  the same call with the Q(t) filter substituted (slip_q.run, k=0), fed a
                       non-trivial p sequence (uniform [0,1], seed 0) so that k=0 really has to
                       cancel it
  PASS  <=> for every session x every Phase 2 model: equal row counts and numpy.array_equal on
            x, y, theta (also r_imu, r_odom). No tolerance.
Sessions: the gate session (role gate in slip_sessions.json) + the first (sorted) Phase 2 final
session from results/final_models.csv. Models: the 7 frozen D2 LOBO checkpoints
(fix_ws/final_training_list.txt).
Also checked (pre-conditions; any failure => ABORT):
  - sha256 of the frozen Phase 2 files used here = fix_ws/final_freeze.sha256
  - OMP_NUM_THREADS = MKL_NUM_THREADS = 1 and torch.get_num_threads() == 1
Also reported (informational, not part of PASS):
  - liveness: k = 10 with the same p changes the trajectory (the hook is really in the loop)
  - (a) on the final session reproduces lstm_linear of results/final_models.csv bit-exactly
Writes results/slip_k0_regression.csv and prints PASS/FAIL. Exit code 0 = PASS.
"""
import ast
import hashlib
import json
import os
import sys

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    if os.environ.get(_v) != '1':
        sys.exit(f'ABORT: run with OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ({_v}={os.environ.get(_v)})')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ws_paths as W  # noqa: E402

LOG, WS, FIX = W.LOG_DIR, W.WS, W.FIX_DIR
RES = os.path.join(WS, 'results')
FROZEN = ['scripts/lstm_adaptive_ekf.py', 'scripts/heuristic_adaptive_ekf.py', 'scripts/ekf_fusion.py',
          'scripts/compare_adaptive_vs_fixed.py', 'scripts/train_reliability_lstm.py',
          'scripts/build_training_dataset.py']
K_LIVE, SEED = 10.0, 0


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def check_frozen(models):
    ref = {}
    for line in open(os.path.join(FIX, 'final_freeze.sha256')):
        if line.strip() and not line.startswith('#'):
            hsh, path = line.split(None, 1)
            ref[path.strip()] = hsh
    bad = []
    for rel in FROZEN + [f'sensor_logs/model_{m}.pt' for m in models]:
        if rel not in ref:
            bad.append(f'{rel}: not in final_freeze.sha256')
        elif sha(os.path.join(WS, rel)) != ref[rel]:
            bad.append(f'{rel}: sha256 differs from the Phase 2 freeze')
    if bad:
        sys.exit('ABORT: frozen Phase 2 files changed:\n  ' + '\n  '.join(bad))
    print(f'frozen files OK ({len(FROZEN)} scripts + {len(models)} models match final_freeze.sha256)')


def main():
    torch.set_num_threads(1)
    if torch.get_num_threads() != 1:
        sys.exit('ABORT: torch is not single-threaded')
    models = ast.literal_eval(open(os.path.join(FIX, 'final_training_list.txt')).read().strip())
    check_frozen(models)

    import slip_q as Q
    import lstm_adaptive_ekf as LA
    from compare_adaptive_vs_fixed import load_ground_truth, rmse

    info = json.load(open(os.path.join(LOG, 'slip_sessions.json')))['info']
    gate = sorted(s for s, i in info.items() if i.get('role') == 'gate')
    # round_trip: pandas' default fast float parser can be 1 ulp off the written value, which
    # would make the exact table comparison below fail spuriously (seen 2026-10-09: 2 of 7).
    fm = pd.read_csv(os.path.join(RES, 'final_models.csv'), float_precision='round_trip')
    final = sorted(fm.baseline.unique())[0]
    if len(gate) != 1:
        sys.exit(f'ABORT: expected 1 gate session, found {gate}')
    sessions = [gate[0], final]
    print(f'sessions: {sessions}; models: {len(models)}')

    rows, all_ok = [], True
    for s in sessions:
        inp = Q.session_inputs(s)
        n = inp[-1]
        p = np.random.default_rng(SEED).uniform(0.0, 1.0, n)
        for m in models:
            model, mean, std, feats, win = LA.load_model(m)
            with torch.no_grad():
                a = Q.run(s, model, mean, std, feats, win, k=None, inputs=inp)
                b = Q.run(s, model, mean, std, feats, win, k=0.0, p_slip=p, inputs=inp)
                c = Q.run(s, model, mean, std, feats, win, k=K_LIVE, p_slip=p, inputs=inp)
            same_rows = all(len(u) == len(v) == n for u, v in zip(a, b))
            eq = [bool(np.array_equal(u, v)) for u, v in zip(a, b)]
            ok = same_rows and all(eq)
            all_ok &= ok
            live = not (np.array_equal(a[0], c[0]) and np.array_equal(a[1], c[1]))
            row = dict(session=s, model=m, rows=n, rows_equal=same_rows, x_equal=eq[0], y_equal=eq[1],
                       theta_equal=eq[2], r_imu_equal=eq[3], r_odom_equal=eq[4],
                       max_abs_dx=float(np.max(np.abs(a[0] - b[0]))),
                       max_abs_dy=float(np.max(np.abs(a[1] - b[1]))),
                       max_abs_dtheta=float(np.max(np.abs(a[2] - b[2]))),
                       live_k10_changes=live, PASS=ok)
            gt = load_ground_truth(s)
            if s == final and gt is not None:
                ref = fm[(fm.model.astype(str) == m) & (fm.session == s)].lstm_linear
                r_a = rmse(a[0], a[1], *gt)
                row['rmse_L'] = r_a
                row['phase2_table_equal'] = bool(len(ref) == 1 and r_a == float(ref.iloc[0]))
            rows.append(row)
            print(f'  {s} model {m}: rows {n}, x/y/theta equal {eq[0]}/{eq[1]}/{eq[2]}, '
                  f'r equal {eq[3]}/{eq[4]}, k=10 changes output: {live}'
                  + (f', reproduces Phase 2 table: {row["phase2_table_equal"]}' if 'phase2_table_equal' in row else '')
                  + f'  -> {"PASS" if ok else "FAIL"}', flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(RES, exist_ok=True)
    out = os.path.join(RES, 'slip_k0_regression.csv')
    df.to_csv(out, index=False)
    print(f'\nwrote {out}')
    if not df.live_k10_changes.all():
        print('WARNING: k=10 did not change some trajectory -- the Q(t) hook may not be in the loop')
    print(f'\nk=0 REGRESSION: {"PASS" if all_ok else "FAIL"} ({int(df.PASS.sum())}/{len(df)} session x model runs identical)')
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
