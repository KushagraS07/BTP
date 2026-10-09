#!/usr/bin/env python3
"""
slip_test.py -- Phase 3B confirmatory evaluation (PREREG_slip.md §6, Part B §7/§7a). Runs ONCE, on the
6 test sessions recorded after Part B was frozen, and on the 5 Phase 2 final sessions (S3).

Refuses to run unless every file listed in fix_ws/slip_partB.sha256 matches its hash (Part B frozen).
k is NOT typed here: it is re-derived from results/slip_calib_summary.csv (hashed in Part B) with the
pre-registered rule (lowest mean RMSE(S), ties at 4 decimals -> smallest k).

Systems (per session):
  F  fixed EKF (ekf_fusion.run_ekf_core)                          1 run
  L  frozen Phase 2 LSTM-R (slip_q.run, k=None)                    7 Phase 2 models -> mean
  S  L + Q(t), learned P_slip (k selected)                         7 x 8 = 56 -> mean
  O  L + Q(t), oracle slip label, same 1-step delay (NaN -> 0)     7 -> mean        (reported)
  C  L + constant Q multiplier 1 + k * mean_t(P_slip of that P_slip model on that session)
     -- same average Q inflation as S, no timing information       7 x 8 = 56 -> mean (reported, §7a)
Tests (pass/fail, §6):
  S1 for each of the 8 P_slip models: ROC-AUC over all valid steps of the 6 test sessions pooled;
     PASS if the mean of the 8 AUCs >= 0.80
  S2 RMSE(S) < RMSE(L) in >= 5 of 6 test sessions AND median of 100*(S-L)/L < 0
  S3 on the 5 Phase 2 final sessions: mean RMSE(S) <= 1.01 * mean RMSE(L)
Reported only: O vs L (ceiling), S vs C (slip-specific part), S vs F, differences in mm, per-area
position RMSE, slip share and invalid steps per session, off-patch false-alarm rate (P >= 0.5 on
off-patch steps labelled 0), Brier score per P_slip model.
Writes results/slip_test_runs.csv, slip_test_sessions.csv, slip_test_s1.csv, slip_test_verdicts.csv.
Usage: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 slip_test.py [--jobs N]
"""
import argparse
import hashlib
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

LOG, FIX, WS, RES = W.LOG_DIR, W.FIX_DIR, W.WS, os.path.join(W.WS, 'results')
# ---------------- PREREG_slip.md §6 -- fixed ----------------
S1_AUC = 0.80
S2_MIN_WINS, N_TEST = 5, 6
S3_RATIO = 1.01
N_FINAL = 5
P_ALARM = 0.5
TIE_DECIMALS = 4
# -------------------------------------------------------------
_C = {}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def check_partB():
    path = os.path.join(FIX, 'slip_partB.sha256')
    if not os.path.exists(path):
        sys.exit('ABORT: fix_ws/slip_partB.sha256 missing -- Part B is not frozen.')
    bad, n = [], 0
    for line in open(path):
        if line.strip() and not line.startswith('#'):
            hsh, rel = line.split(None, 1)
            rel = rel.strip()
            n += 1
            full = os.path.join(WS, rel)
            if not os.path.exists(full) or sha(full) != hsh:
                bad.append(rel)
    if bad or n == 0:
        sys.exit(f'ABORT: Part B hash check failed ({n} entries): {bad}')
    print(f'Part B frozen files OK ({n} entries match fix_ws/slip_partB.sha256)')


def selected_k():
    d = pd.read_csv(os.path.join(RES, 'slip_calib_summary.csv'), float_precision='round_trip')
    s = d[d.system == 'S']
    if s.empty:
        sys.exit('ABORT: no S rows in slip_calib_summary.csv (Gate 1/2 did not pass)')
    L = float(d[d.system == 'L']['mean'].iloc[0])
    if not (s['mean'] < L).any():
        sys.exit('ABORT: Gate 2 did not pass in calibration')
    return float(min(zip(s['mean'], s.k), key=lambda t: (round(t[0], TIE_DECIMALS), t[1]))[1])


def _init():
    torch.set_num_threads(1)


def _get(key, make):
    if key not in _C:
        _C[key] = make()
    return _C[key]


def p_learned(s, h):
    import slip_train as T
    X = T.load_session(s)[0] if os.path.exists(os.path.join(LOG, f'slip_labels_{s}.csv')) else \
        pd.read_csv(os.path.join(LOG, f'features_{s}.csv'))[T.FEATURES].values.astype(np.float32)
    return T.predict_p(T.load_ckpt(h), X)


def p_oracle(s):
    y = pd.read_csv(os.path.join(LOG, f'slip_labels_{s}.csv')).slip.values.astype(float)
    return np.where(np.isfinite(y), y, 0.0)


def run_task(t):
    import lstm_adaptive_ekf as LA
    import slip_q as Q
    from compare_adaptive_vs_fixed import load_ground_truth
    from ekf_fusion import run_ekf_core
    system, k, s, m2, h = t
    inp = _get(('inp', s), lambda: Q.session_inputs(s))
    gt = _get(('gt', s), lambda: load_ground_truth(s))
    if system == 'F':
        arrays, init = inp[0], inp[1]
        out = run_ekf_core(*arrays, *init)
    else:
        mdl = _get(('m2', m2), lambda: LA.load_model(m2))
        with torch.no_grad():
            if system == 'L':
                out = Q.run(s, *mdl, k=None, inputs=inp)
            elif system == 'O':
                out = Q.run(s, *mdl, k=k, p_slip=_get(('po', s), lambda: p_oracle(s)), inputs=inp)
            else:
                p = _get(('pl', s, h), lambda: p_learned(s, h))
                if system == 'C':
                    p = np.full(len(p), float(np.mean(p)))
                out = Q.run(s, *mdl, k=k, p_slip=p, inputs=inp)
    x, y = np.asarray(out[0]), np.asarray(out[1])
    n = min(len(x), len(gt[0]))
    e2 = (x[:n] - gt[0][:n]) ** 2 + (y[:n] - gt[1][:n]) ** 2
    row = dict(system=system, k=k, session=s, phase2_model=m2, slip_model=h, rmse=float(np.sqrt(e2.mean())))
    lab = os.path.join(LOG, f'slip_labels_{s}.csv')
    if os.path.exists(lab):
        area = _get(('area', s), lambda: pd.read_csv(lab).area.values)[:n]
        for a in sorted(set(area)):
            row[f'rmse_{a}'] = float(np.sqrt(e2[area == a].mean()))
    return row


def run_all(tasks, jobs):
    res = []
    if jobs <= 1:
        _init()
        it = map(run_task, tasks)
        pool = None
    else:
        pool = mp.get_context('fork').Pool(jobs, initializer=_init)
        it = pool.imap(run_task, tasks, chunksize=1)
    for i, r in enumerate(it, 1):
        res.append(r)
        if i % 20 == 0 or i == len(tasks):
            print(f'    {i}/{len(tasks)}', flush=True)
    if pool is not None:
        pool.close()
        pool.join()
    return res


def auc(y, p):
    from slip_train import auc as _auc
    return _auc(y, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(1)
    check_partB()
    k = selected_k()

    import ast
    from slip_prepare import select
    import slip_train as T
    tests = select(['test'])['test']
    m2s = ast.literal_eval(open(os.path.join(FIX, 'final_training_list.txt')).read().strip())
    devm = T.dev_sessions()
    fm = pd.read_csv(os.path.join(RES, 'final_models.csv'))
    finals = sorted(fm.baseline.unique())
    if len(tests) != N_TEST or len(m2s) != 7 or len(devm) != 8 or len(finals) != N_FINAL:
        sys.exit(f'ABORT: test {len(tests)}, Phase 2 models {len(m2s)}, P_slip models {len(devm)}, finals {len(finals)}')
    overlap = set(tests) & (set(devm) | set(finals))
    if overlap:
        sys.exit(f'ABORT: test sessions overlap dev/final: {overlap}')
    print(f'k = {k:g} (from slip_calib_summary.csv)\ntest: {tests}\nfinal (S3): {finals}\n', flush=True)

    # ---------- S1 (detection) + reported detection numbers ----------
    s1, sess_rows = [], []
    for h in devm:
        ys, ps, offs = [], [], []
        for s in tests:
            lab = pd.read_csv(os.path.join(LOG, f'slip_labels_{s}.csv'))
            y = lab.slip.values.astype(float)
            p = p_learned(s, h)
            v = np.isfinite(y)
            ys.append(y[v]); ps.append(p[v]); offs.append((lab.area.values != 'off_patch')[v])
        y, p, on = np.concatenate(ys), np.concatenate(ps), np.concatenate(offs)
        s1.append(dict(slip_model=h, auc=auc(y, p), brier=float(np.mean((p - y) ** 2)),
                       false_alarm_off_patch=float(((p >= P_ALARM) & ~on & (y == 0)).sum() /
                                                   max(int((~on & (y == 0)).sum()), 1))))
    s1 = pd.DataFrame(s1)
    for s in tests:
        lab = pd.read_csv(os.path.join(LOG, f'slip_labels_{s}.csv'))
        y = lab.slip.values.astype(float)
        sess_rows.append(dict(session=s, rows=len(y), invalid=int((~np.isfinite(y)).sum()),
                              slip_share=float(np.nanmean(y))))

    # ---------- filter runs ----------
    tasks = []
    for s in tests + finals:
        tasks.append(('F', np.nan, s, '', ''))
        tasks += [('L', np.nan, s, m, '') for m in m2s]
        tasks += [(sy, k, s, m, h) for sy in ('S', 'C') for m in m2s for h in devm]
    tasks += [('O', k, s, m, 'oracle') for s in tests for m in m2s]
    print(f'filter runs: {len(tasks)}', flush=True)
    runs = pd.DataFrame(run_all(tasks, a.jobs))
    os.makedirs(RES, exist_ok=True)
    runs.to_csv(os.path.join(RES, 'slip_test_runs.csv'), index=False)

    per = runs.groupby(['system', 'session']).rmse.mean().unstack(0)
    T6, F5 = per.loc[tests], per.loc[finals]
    pct = 100 * (T6.S - T6.L) / T6.L
    s2_wins = int((T6.S < T6.L).sum())
    s2 = s2_wins >= S2_MIN_WINS and float(pct.median()) < 0
    s3_ratio = float(F5.S.mean() / F5.L.mean())
    s3 = s3_ratio <= S3_RATIO
    s1_mean = float(s1.auc.mean())
    s1_pass = s1_mean >= S1_AUC

    sess = pd.DataFrame(sess_rows).set_index('session')
    for col in ('F', 'L', 'S', 'O', 'C'):
        sess[f'rmse_{col}'] = T6[col]
    sess['S_vs_L_pct'] = pct
    sess['S_vs_L_mm'] = 1000 * (T6.S - T6.L)
    sess['O_vs_L_pct'] = 100 * (T6.O - T6.L) / T6.L
    sess['S_vs_C_pct'] = 100 * (T6.S - T6.C) / T6.C
    sess['S_vs_F_pct'] = 100 * (T6.S - T6.F) / T6.F
    sess.to_csv(os.path.join(RES, 'slip_test_sessions.csv'))
    s1.to_csv(os.path.join(RES, 'slip_test_s1.csv'), index=False)

    verdicts = pd.DataFrame([
        dict(test='S1', value=f'mean AUC {s1_mean:.4f} (min {s1.auc.min():.4f})', rule=f'>= {S1_AUC}', result='PASS' if s1_pass else 'FAIL'),
        dict(test='S2', value=f'S<L in {s2_wins}/{N_TEST}; median {pct.median():+.3f}%', rule=f'>= {S2_MIN_WINS}/{N_TEST} and median < 0', result='PASS' if s2 else 'FAIL'),
        dict(test='S3', value=f'mean S / mean L = {s3_ratio:.5f} on {N_FINAL} final sessions', rule=f'<= {S3_RATIO}', result='PASS' if s3 else 'FAIL'),
    ])
    verdicts.to_csv(os.path.join(RES, 'slip_test_verdicts.csv'), index=False)

    pd.set_option('display.width', 200)
    print('\n=== per test session (RMSE m; S, C = mean of 56; L, O = mean of 7) ===')
    print(sess.round(5).to_string())
    print('\n=== S1 per P_slip model ===')
    print(s1.round(4).to_string(index=False))
    print('\n=== S3 (Phase 2 final sessions) ===')
    print(F5[['F', 'L', 'S', 'C']].round(5).to_string())
    print(f'\nreported: mean O vs L {sess.O_vs_L_pct.mean():+.3f}% (ceiling); mean S vs C {sess.S_vs_C_pct.mean():+.3f}% '
          f'(slip-specific part); mean S vs L {pct.mean():+.3f}% = {sess.S_vs_L_mm.mean():+.2f} mm')
    print('\n=== VERDICTS ===')
    print(verdicts.to_string(index=False))


if __name__ == '__main__':
    main()
