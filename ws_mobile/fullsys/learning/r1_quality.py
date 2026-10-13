#!/usr/bin/env python3
"""r1_quality.py -- R1 (FINDINGS R1): learned reliability as a fault DETECTOR vs the NIS threshold detector, + runtime.
Held-out models (LOBO, current seed) on the feature files of the current fault set (FULLSYS_FAULT_UNIT / _WINDOW).
Usage: OMP_NUM_THREADS=1 python3 learning/r1_quality.py [--jobs N] [--data DIR] [--no-runtime]
Writes results/fullsys_r1_rows.csv (per sensor / detector / type) and prints the tables."""
import argparse
import contextlib
import glob
import io
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from learning.closed_loop import DET, EKF9, FEAT_DIR, GATE, MODEL_DIR, RES, predict_r  # noqa: E402
from learning.closed_loop_lidar import SEG  # noqa: E402
from learning.faults import SUFFIX  # noqa: E402
from zupt import paths as _P  # noqa: E402

CHI2_99 = {'imu': 6.63, 'odom': 9.21}
NIS_COL = {'imu': 'gyro_nis', 'odom': 'odom_nis'}
RCOL = {'imu': 0, 'odom': 1}


def auc(score, y):
    """Mann-Whitney AUC (ties averaged)."""
    y = np.asarray(y, bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(score).rank().values
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def active_window(m):
    """(start, end) of the injected fault from a feature manifest (works for files built before R2)."""
    on, t0, t1 = m['onset'], m['session_t0'], m['session_t1']
    if 'fault_end' in m:
        return on, m['fault_end']
    if m['type'] == 'dropout':
        return on, min(on + m['value'] * (t1 - t0), t1)
    return on, t1


def work(h):
    torch.set_num_threads(1)
    from learning.features import FEATURES
    ck = torch.load(os.path.join(MODEL_DIR, f'rel_{h}.pt'), map_location='cpu', weights_only=False)
    out = []
    for p in sorted(glob.glob(os.path.join(FEAT_DIR, f'{h}*.npz'))):
        z = np.load(p)
        m = json.loads(str(z['manifest']))
        if m['session'] != h:
            continue
        cols, d = list(z['columns']), z['data']
        t = d[:, cols.index('t')]
        r = predict_r(ck, d[:, [cols.index(c) for c in FEATURES]].astype(np.float32))
        typ = m.get('type')
        for sen in ('imu', 'odom'):
            act = np.zeros(len(t), bool)
            if typ != 'clean' and m.get('sensor') == sen:
                a, b = active_window(m)
                act = (t >= a) & (t < b)
            gate_lab = d[:, cols.index('r_odom' if sen == 'odom' else 'r_imu')] < GATE
            out.append(dict(session=h, run=m['name'], type=typ, fsensor=m.get('sensor'), severity=m.get('severity'),
                            sensor=sen, t=t, onset=m.get('onset', np.nan), act=act, gate_lab=gate_lab,
                            r=r[:, RCOL[sen]], nis=d[:, cols.index(NIS_COL[sen])]))
    return out


def first_delay(t, flag, act, onset):
    idx = np.where(flag & act)[0]
    return float(t[idx[0]] - onset) if len(idx) else np.nan


def runtime(data):
    from learning.features import FEATURES
    from learning.train_slip import SlipLSTM
    from learning.train import RelLSTM
    torch.set_num_threads(1)
    hs = sorted(os.path.basename(p)[4:-3] for p in glob.glob(os.path.join(MODEL_DIR, 'rel_*.pt')))
    h = hs[0]
    ck = torch.load(os.path.join(MODEL_DIR, f'rel_{h}.pt'), map_location='cpu', weights_only=False)
    cs = torch.load(os.path.join(MODEL_DIR, f'slip_{h}.pt'), map_location='cpu', weights_only=False)
    mr = RelLSTM(n_in=len(ck['features']), hidden=ck['hidden'], layers=ck['layers'])
    mr.load_state_dict(ck['model_state']); mr.eval()
    ms = SlipLSTM(); ms.load_state_dict(cs['model_state']); ms.eval()
    x = torch.from_numpy(np.random.default_rng(0).normal(size=(1, ck['window'], len(FEATURES))).astype(np.float32))
    res = {}
    for name, mod in (('rel_lstm', mr), ('slip_lstm', ms)):
        ts = []
        with torch.no_grad():
            for _ in range(50):
                mod(x)
            for _ in range(2000):
                a = time.perf_counter(); mod(x); ts.append(1e3 * (time.perf_counter() - a))
        res[name] = (float(np.median(ts)), float(np.percentile(ts, 95)))
    import learning.baselines as B
    from zupt.core.config import load_config
    from zupt.ekf.ekf9 import load_ekf9_params
    from zupt.replay import load_scans, load_session
    with contextlib.redirect_stdout(io.StringIO()):
        sess = load_session(data, h)
    scans = load_scans(data, h)
    cfg, p9 = load_config(DET), load_ekf9_params(EKF9)
    one = np.ones(len(sess[0]))
    for name, kw in (('filter_step_fixed', {}), ('filter_step_chi2', dict(chi2=True))):
        a = time.perf_counter()
        B.run(sess, sess[1], scans, cfg, p9, one, one, **kw)
        res[name] = (1e3 * (time.perf_counter() - a) / len(sess[0]), np.nan)
    return h, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--data', default=_P.DATA)
    ap.add_argument('--no-runtime', action='store_true')
    a = ap.parse_args()
    if os.environ.get('OMP_NUM_THREADS') != '1':
        sys.exit('ABORT: run with OMP_NUM_THREADS=1')
    hs = sorted(os.path.basename(p)[4:-3] for p in glob.glob(os.path.join(MODEL_DIR, 'rel_*.pt')))
    print(f'R1 on feature set "{SUFFIX or "std"}" ({FEAT_DIR}); {len(hs)} held-out models', flush=True)
    if a.jobs <= 1:
        out = [work(h) for h in hs]
    else:
        with mp.get_context('fork').Pool(a.jobs) as pool:
            out = pool.map(work, hs)
    runs = [x for o in out for x in o]
    rows = []
    for sen in ('imu', 'odom'):
        R = [x for x in runs if x['sensor'] == sen]
        act = np.concatenate([x['act'] for x in R])
        gl = np.concatenate([x['gate_lab'] for x in R])
        sr = 1.0 - np.concatenate([x['r'] for x in R])
        sn = np.concatenate([x['nis'] for x in R])
        clean = np.concatenate([np.full(len(x['t']), x['type'] == 'clean') for x in R])
        for det, score, flag in (('learned_r', sr, sr > 1.0 - GATE), ('nis_chi2', sn, sn > CHI2_99[sen])):
            for truth_name, y in (('fault', act), ('gate_label', gl)):
                if sen == 'imu' and truth_name == 'gate_label':
                    continue
                tp, fp, fn = int((flag & y).sum()), int((flag & ~y).sum()), int((~flag & y).sum())
                prec = tp / max(tp + fp, 1)
                rec = tp / max(tp + fn, 1)
                rows.append(dict(sensor=sen, detector=det, truth=truth_name, auc=auc(score, y), precision=prec,
                                 recall=rec, f1=2 * prec * rec / max(prec + rec, 1e-12),
                                 clean_false_alarm=float(flag[clean].mean())))
    q = pd.DataFrame(rows)
    pd.set_option('display.width', 200)
    print('\nDEVELOPMENT R1 detection quality (rows pooled over held-out sessions; flag: r < 0.5 / NIS > chi2 99 %):')
    print(q.round(3).to_string(index=False))
    drows = []
    for x in runs:
        if x['type'] in (None, 'clean') or x['fsensor'] != x['sensor'] or not x['act'].any():
            continue
        for det, flag in (('learned_r', x['r'] < GATE), ('nis_chi2', x['nis'] > CHI2_99[x['sensor']])):
            drows.append(dict(sensor=x['sensor'], type=x['type'], severity=x['severity'], detector=det,
                              session=x['session'], delay=first_delay(x['t'], flag, x['act'], x['onset']),
                              gated_share=float(flag[x['act']].mean())))
    dd = pd.DataFrame(drows)
    g = dd.groupby(['sensor', 'type', 'detector'])
    tab = pd.DataFrame({'detected': g.delay.apply(lambda v: f'{int(v.notna().sum())}/{len(v)}'),
                        'median_delay_s': g.delay.median().round(2), 'flagged_share_in_fault': g.gated_share.mean().round(3)})
    print('\nDEVELOPMENT R1 detection delay (onset -> first flag; median over detected runs):')
    print(tab.unstack('detector').to_string())
    os.makedirs(RES, exist_ok=True)
    q.to_csv(os.path.join(RES, f'fullsys_r1_quality{SUFFIX}.csv'), index=False)
    dd.to_csv(os.path.join(RES, f'fullsys_r1_delay{SUFFIX}.csv'), index=False)
    if not a.no_runtime:
        h, rt = runtime(a.data)
        print(f'\nDEVELOPMENT R1 runtime (session {h}, 1 thread; ms per step, median / p95):')
        for k, (med, p95) in rt.items():
            print(f'  {k:18s} {med:7.3f} / {p95:7.3f}' if not np.isnan(p95) else f'  {k:18s} {med:7.3f} (mean)')
        tot = rt['rel_lstm'][1] + rt['slip_lstm'][1] + rt['filter_step_chi2'][0]
        print(f'  2 LSTMs (p95) + hybrid filter step (mean) = {tot:.3f} ms vs 20 ms budget at 50 Hz -> '
              f'{"OK" if tot < 20 else "OVER"}')
    for sen in ('imu', 'odom'):
        x = q[(q.sensor == sen) & (q.truth == 'fault')].set_index('detector')
        print(f'look criteria [{sen}]: AUC learned {x.auc.learned_r:.3f} vs NIS {x.auc.nis_chi2:.3f}; clean false alarm '
              f'learned {x.clean_false_alarm.learned_r:.4f} vs NIS {x.clean_false_alarm.nis_chi2:.4f}')


if __name__ == '__main__':
    main()
