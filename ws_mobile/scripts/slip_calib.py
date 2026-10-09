#!/usr/bin/env python3
"""
slip_calib.py -- Phase 3B Part B calibration on the 3 calib sessions (PREREG_slip.md §7). Uses ONLY
the calib sessions (role calib AND slip_accept ACCEPT, 3 asserted). Never reads a test session.

  L  frozen Phase 2 LSTM-R filter (slip_q.run, k=None)                     -> 7 Phase 2 models
  O  L + Q(t) with oracle p = slip label of §4 (NaN -> 0), same 1-step delay -> 7 models per k
  S  L + Q(t) with learned p = P_slip of one dev LOBO model                  -> 7 x 8 = 56 per k
Per session: RMSE(L), RMSE(O) = mean over the 7 Phase 2 models; RMSE(S) = mean over the 56
combinations. Metric = mean over the 3 calib sessions (equal weight).

  Gate 1  any k in K_GRID with mean RMSE(O) < mean RMSE(L)?  no -> STOP (S is not run)
          oracle-best k reported, NOT used.
  k       only if Gate 1 passes: k with the lowest mean RMSE(S); ties (equal at 4 decimals in m)
          -> smallest k.
  Gate 2  any k with mean RMSE(S) < mean RMSE(L)?  no -> STOP before test sessions.
Writes results/slip_calib_runs.csv (every run) and results/slip_calib_summary.csv; exit code
0 = both gates pass (k selected), 2 = Gate 1 STOP, 3 = Gate 2 STOP.
Usage: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 slip_calib.py [--jobs N]
"""
import argparse
import ast
import os
import sys

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    if os.environ.get(_v) != '1':
        sys.exit(f'ABORT: run with OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ({_v}={os.environ.get(_v)})')

import multiprocessing as mp  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ws_paths as W  # noqa: E402

LOG, FIX, RES = W.LOG_DIR, W.FIX_DIR, os.path.join(W.WS, 'results')
K_GRID = [1.0, 3.0, 10.0, 30.0, 100.0]   # PREREG_slip.md §7 -- fixed
TIE_DECIMALS = 4

_C = {}   # per-process caches


def _init():
    torch.set_num_threads(1)


def _get(key, make):
    if key not in _C:
        _C[key] = make()
    return _C[key]


def phase2_models():
    return ast.literal_eval(open(os.path.join(FIX, 'final_training_list.txt')).read().strip())


def dev_models():
    import slip_train as T
    return T.dev_sessions()


def p_for(kind, s, slip_model=None):
    import slip_train as T
    if kind == 'oracle':
        y = pd.read_csv(os.path.join(LOG, f'slip_labels_{s}.csv')).slip.values.astype(float)
        return np.where(np.isfinite(y), y, 0.0)
    X = T.load_session(s)[0]
    return T.predict_p(T.load_ckpt(slip_model), X)


def run_task(t):
    import lstm_adaptive_ekf as LA
    import slip_q as Q
    from compare_adaptive_vs_fixed import load_ground_truth, rmse
    system, k, s, m2, ms = t
    inp = _get(('inp', s), lambda: Q.session_inputs(s))
    gt = _get(('gt', s), lambda: load_ground_truth(s))
    mdl = _get(('m2', m2), lambda: LA.load_model(m2))
    with torch.no_grad():
        if system == 'L':
            out = Q.run(s, *mdl, k=None, inputs=inp)
        else:
            p = _get(('p', system, s, ms), lambda: p_for('oracle' if system == 'O' else 'learned', s, ms))
            out = Q.run(s, *mdl, k=k, p_slip=p, inputs=inp)
    return dict(system=system, k=k, session=s, phase2_model=m2, slip_model=ms,
                rmse=rmse(out[0], out[1], *gt))


def run_all(tasks, jobs):
    if jobs <= 1:
        _init()
        res = []
        for i, t in enumerate(tasks, 1):
            res.append(run_task(t))
            if i % 10 == 0 or i == len(tasks):
                print(f'    {i}/{len(tasks)}', flush=True)
        return res
    with mp.get_context('fork').Pool(jobs, initializer=_init) as pool:
        res = []
        for i, r in enumerate(pool.imap(run_task, tasks, chunksize=1), 1):
            res.append(r)
            if i % 10 == 0 or i == len(tasks):
                print(f'    {i}/{len(tasks)}', flush=True)
        return res


def per_session(df, system, k=None):
    d = df[df.system == system] if k is None else df[(df.system == system) & (df.k == k)]
    return d.groupby('session').rmse.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(1)

    from slip_prepare import select
    calib = select(['calib'])['calib']
    m2s = phase2_models()
    devm = dev_models()
    missing = [h for h in devm if not os.path.exists(os.path.join(LOG, f'slip_model_{h}.pt'))]
    if len(calib) != 3 or len(m2s) != 7 or len(devm) != 8 or missing:
        sys.exit(f'ABORT: calib {len(calib)}, Phase 2 models {len(m2s)}, dev {len(devm)}, '
                 f'missing P_slip models {missing}')
    print(f'calib: {calib}\nPhase 2 models: {len(m2s)}; P_slip models: {len(devm)}; k grid {K_GRID}\n', flush=True)
    os.makedirs(RES, exist_ok=True)
    runs_csv = os.path.join(RES, 'slip_calib_runs.csv')

    # ---- L and O (Gate 1) ----
    tasks = [('L', np.nan, s, m, '') for s in calib for m in m2s]
    tasks += [('O', k, s, m, 'oracle') for k in K_GRID for s in calib for m in m2s]
    print(f'L + O: {len(tasks)} filter runs', flush=True)
    df = pd.DataFrame(run_all(tasks, a.jobs))
    df.to_csv(runs_csv, index=False)

    L = per_session(df, 'L')
    mL = float(L.mean())
    summ = [dict(system='L', k=np.nan, **{s: L[s] for s in calib}, mean=mL)]
    print(f'\nmean RMSE(L) = {mL:.5f} m  ' + '  '.join(f'{s}: {L[s]:.5f}' for s in calib))
    mO = {}
    for k in K_GRID:
        O = per_session(df, 'O', k)
        mO[k] = float(O.mean())
        summ.append(dict(system='O', k=k, **{s: O[s] for s in calib}, mean=mO[k]))
        print(f'  k={k:>5g}: mean RMSE(O) = {mO[k]:.5f} m ({100 * (mO[k] - mL) / mL:+.2f}% vs L)')
    k_or = min(K_GRID, key=lambda k: (round(mO[k], TIE_DECIMALS), k))
    gate1 = any(mO[k] < mL for k in K_GRID)
    print(f'\nGATE 1 (oracle feasibility): {"PASS" if gate1 else "STOP"}; oracle-best k = {k_or:g} '
          f'(reported only, not used)')
    if not gate1:
        pd.DataFrame(summ).to_csv(os.path.join(RES, 'slip_calib_summary.csv'), index=False)
        print('STOP: Q(t) from slip shows no benefit in this EKF even with perfect slip knowledge (§6a).')
        sys.exit(2)

    # ---- S (k selection, Gate 2) ----
    tasks = [('S', k, s, m, h) for k in K_GRID for s in calib for m in m2s for h in devm]
    print(f'\nS: {len(tasks)} filter runs', flush=True)
    dS = pd.DataFrame(run_all(tasks, a.jobs))
    df = pd.concat([df, dS], ignore_index=True)
    df.to_csv(runs_csv, index=False)
    mS = {}
    for k in K_GRID:
        S = per_session(df, 'S', k)
        mS[k] = float(S.mean())
        summ.append(dict(system='S', k=k, **{s: S[s] for s in calib}, mean=mS[k]))
        print(f'  k={k:>5g}: mean RMSE(S) = {mS[k]:.5f} m ({100 * (mS[k] - mL) / mL:+.2f}% vs L)')
    k_sel = min(K_GRID, key=lambda k: (round(mS[k], TIE_DECIMALS), k))
    gate2 = any(mS[k] < mL for k in K_GRID)
    pd.DataFrame(summ).to_csv(os.path.join(RES, 'slip_calib_summary.csv'), index=False)
    print(f'\nselected k = {k_sel:g} (lowest mean RMSE(S); ties at {TIE_DECIMALS} decimals -> smallest k)')
    print(f'GATE 2 (learned integration): {"PASS" if gate2 else "STOP"}')
    if not gate2:
        print('STOP before test sessions: the learned detector is not good enough for this integration (§6a).')
        sys.exit(3)
    print(f'\nBoth gates pass. Part B value: k = {k_sel:g}. Wrote {runs_csv} and slip_calib_summary.csv')


if __name__ == '__main__':
    main()
