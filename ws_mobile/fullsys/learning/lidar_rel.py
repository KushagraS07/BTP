#!/usr/bin/env python3
"""lidar_rel.py -- L2 (FINDINGS L2): learned LiDAR reliability r_L.

  build : per-scan features (scan-to-scan ICP initialised with wheel odometry; no filter feedback) + label
          r_L = 0 while a LiDAR fault is active, for clean + 15 LiDAR faults + 9 overlaps (+ 26 F14 faults for
          inference only) per session -> fullsys_logs/lfeatures/<run>.npz
  train : LOBO LSTM (window 10 scans, hidden 32, BCE with pos_weight, 15 epochs, best-val, seed 42)
          -> fullsys_logs/models/rlidar_<h>.pt, results/fullsys_rlidar_lobo.csv
  eval  : closed loop hybrid_stale_rL (scan update + detector evidence skipped when r_L < 0.5) on clean / LiDAR /
          overlap / F14 runs; compared with hybrid_stale (L run) and F14 hybrid; applies rule (3).
Usage: OMP_NUM_THREADS=1 python3 learning/lidar_rel.py {build|train|eval} [--jobs N]
"""
import argparse
import contextlib
import glob
import io
import json
import math
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from learning.closed_loop import DET, EKF9, FEAT_DIR, MODEL_DIR, RES, predict_r  # noqa: E402
from learning.closed_loop_lidar import SEG  # noqa: E402
from learning.faults import SUFFIX  # noqa: E402
from zupt import paths as _P  # noqa: E402

LFEAT_DIR = os.path.join(_P.WS, 'fullsys_logs', 'lfeatures')
LFEATURES = ['ok', 'rms', 'inlier', 'degenerate', 'iters', 'd_trans_odom', 'd_rot_odom', 'd_rot_gyro',
             'finite', 'short', 'mean_dr', 'stale']
WINDOW, HIDDEN, LR, BATCH, EPOCHS, SEED = 10, 32, 1e-3, 256, 15, 42
GATE = 0.5


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def rel_motion(t, v, w, ta, tb):
    """Wheel-odometry relative pose (x, y, th) from ta to tb in the frame at ta (unicycle integration)."""
    i = (t > ta) & (t <= tb)
    x = y = th = 0.0
    prev = ta
    for tt, vv, ww in zip(t[i], v[i], w[i]):
        dt = tt - prev
        x += vv * dt * math.cos(th)
        y += vv * dt * math.sin(th)
        th += ww * dt
        prev = tt
    return np.array([x, y, th])


def scan_features(scans, odom, imu):
    from zupt.core.lidar import scan_points
    from zupt.core.scan_match import ScanMatchConfig, make_reference, match
    cfg = ScanMatchConfig()
    ts, ot = scans.t.values, odom.t.values
    ov, ow = odom.linear_x.values, odom.angular_z.values
    it, gz = imu.t.values, imu.angular_vel_z.values
    rv = scans.ranges.values
    F = np.zeros((len(scans), len(LFEATURES)))
    ref, prev_r = None, None
    for k in range(len(scans)):
        r = np.array(rv[k].split(';'), float)
        a = (scans.angle_min.iat[k], scans.angle_increment.iat[k], scans.range_min.iat[k], scans.range_max.iat[k])
        pts, _, okm = scan_points(r, *a)
        fin = float(okm.mean())
        short = float((okm & (r < 0.3)).mean())
        stale = float(k > 0 and rv[k] == rv[k - 1])
        f = dict(ok=0.0, rms=1.0, inlier=0.0, degenerate=0.0, iters=0.0, d_trans_odom=0.0, d_rot_odom=0.0,
                 d_rot_gyro=0.0, finite=fin, short=short, mean_dr=0.0, stale=stale)
        if k > 0:
            both = np.isfinite(r) & np.isfinite(prev_r)
            f['mean_dr'] = float(np.mean(np.abs(r[both] - prev_r[both]))) if both.any() else 1.0
            init = rel_motion(ot, ov, ow, ts[k - 1], ts[k])
            gi = (it > ts[k - 1]) & (it <= ts[k])
            grot = float(np.sum(gz[gi]) * (np.median(np.diff(it)) if len(it) > 1 else 0.0))
            if ref is not None and len(pts):
                res = match(ref, pts, init=init, cfg=cfg)
                f['iters'] = float(res.iters)
                f['degenerate'] = float(res.degenerate)
                f['inlier'] = float(res.n) / len(r)
                if res.ok:
                    f['ok'] = 1.0
                    f['rms'] = float(min(res.rms, 1.0))
                    f['d_trans_odom'] = float(np.hypot(*(res.pose[:2] - init[:2])))
                    f['d_rot_odom'] = abs(wrap(res.pose[2] - init[2]))
                    f['d_rot_gyro'] = abs(wrap(res.pose[2] - grot))
        F[k] = [f[c] for c in LFEATURES]
        ref = make_reference(r, *a)
        prev_r = r
    return F


def session_runs(h, data):
    """Yields (set, run, frame-result, faulted scans, F7 feature name, active mask, manifest)."""
    from zupt.replay import load_scans, load_session
    from learning.faults import all_faults, clean, inject
    from learning.lidar_faults import OVERLAP_L, OVERLAP_O, LFault, all_lfaults, inject_scans
    with contextlib.redirect_stdout(io.StringIO()):
        sess = load_session(data, h)
    scans = load_scans(data, h)
    imu, odom = sess[0], sess[1]
    t0, t1 = min(imu.t.iloc[0], odom.t.iloc[0]), max(imu.t.iloc[-1], odom.t.iloc[-1])
    none = np.zeros(len(scans), bool)
    yield 'clean', h, clean(sess), scans, h, none, {}, sess
    for lf in all_lfaults(h):
        sc, man, act = inject_scans(scans, lf, t0, t1)
        yield 'lidar', lf.name, clean(sess), sc, h, act, man, sess
    of = {(f.sensor, f.ftype, f.severity): f for f in all_faults(h)}
    for lt, ls in OVERLAP_L:
        sc, man, act = inject_scans(scans, LFault(h, lt, ls), t0, t1)
        for key in OVERLAP_O:
            fo = of[key]
            yield ('overlap', f'{h}_ovl_{lt}_{ls}__{key[0]}_{key[1]}_{key[2]}', inject(sess, fo), sc, fo.name, act,
                   dict(man, osensor=key[0], otype=key[1], oseverity=key[2]), sess)
    for fo in all_faults(h):
        yield 'f14', fo.name, inject(sess, fo), scans, fo.name, none, {}, sess


def build_work(args):
    h, data = args
    os.makedirs(LFEAT_DIR, exist_ok=True)
    n = 0
    for setname, run, fr, sc, feat, act, man, _ in session_runs(h, data):
        F = scan_features(sc, fr.odom, fr.imu)
        np.savez_compressed(os.path.join(LFEAT_DIR, f'{run}.npz'), X=F, y=(~act).astype(np.float32),
                            t=sc.t.values, meta=json.dumps(dict(man, set=setname, run=run, session=h, feat=feat)))
        n += 1
    return h, n


class RLNet(nn.Module):
    def __init__(self, n_in=len(LFEATURES), hidden=HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, 1, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(o[:, -1, :]).squeeze(-1)


def load_lruns(only_train=True):
    runs = []
    for p in sorted(glob.glob(os.path.join(LFEAT_DIR, '*.npz'))):
        z = np.load(p)
        m = json.loads(str(z['meta']))
        if only_train and m['set'] == 'f14':
            continue
        runs.append(dict(session=m['session'], set=m['set'], run=m['run'], X=z['X'].astype(np.float32),
                         y=z['y'].astype(np.float32)))
    return runs


def stack_windows(runs, mean, std):
    """All causal windows (rows < WINDOW-1 left-padded with row 0), normalised: (N, WINDOW, F), labels (N,)."""
    from numpy.lib.stride_tricks import sliding_window_view
    W, Y = [], []
    for r in runs:
        X = r['X']
        pad = np.concatenate([np.repeat(X[:1], WINDOW - 1, axis=0), X])
        pad = ((pad - mean) / std).astype(np.float32)
        W.append(sliding_window_view(pad, (WINDOW, pad.shape[1]))[:, 0])
        Y.append(r['y'])
    return np.ascontiguousarray(np.concatenate(W)), np.concatenate(Y).astype(np.float32)


def predict_rl(ck, X):
    m = RLNet()
    m.load_state_dict(ck['model_state'])
    m.eval()
    pad = np.concatenate([np.repeat(X[:1], WINDOW - 1, axis=0), X])
    pad = ((pad - ck['mean']) / ck['std']).astype(np.float32)
    w = np.stack([pad[i:i + WINDOW] for i in range(len(X))])
    with torch.no_grad():
        return torch.sigmoid(m(torch.from_numpy(w))).numpy()


def auc(score, y):
    y = np.asarray(y, bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(score).rank().values
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def train_fold(args):
    h, sessions = args
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    runs = load_lruns()
    rest = sorted(s for s in sessions if s != h)
    val_s, fit_s = rest[0], rest[1:]
    fit = [r for r in runs if r['session'] in fit_s]
    val = [r for r in runs if r['session'] == val_s]
    test = [r for r in runs if r['session'] == h]
    allf = np.concatenate([r['X'] for r in fit])
    mean, std = allf.mean(0), allf.std(0)
    std[std == 0] = 1.0
    Wf, yfit = stack_windows(fit, mean, std)
    Wv, yv = stack_windows(val, mean, std)
    pos_w = float((yfit == 1).sum() / max((yfit == 0).sum(), 1))      # positive class = "unreliable" (y = 0)
    model = RLNet()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lf = nn.BCEWithLogitsLoss(reduction='none')
    rng = np.random.default_rng(SEED)
    best, best_state, best_ep = float('inf'), None, -1

    def loss_of(X, y):
        logit = model(torch.from_numpy(X))
        yt = torch.from_numpy(y)
        wgt = torch.where(yt < 0.5, torch.tensor(pos_w), torch.tensor(1.0))
        return (lf(logit, yt) * wgt).mean()

    for ep in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(len(Wf))
        for k in range(0, len(order), BATCH):
            X, y = Wf[order[k:k + BATCH]], yfit[order[k:k + BATCH]]
            opt.zero_grad()
            loss = loss_of(X, y)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl, n = 0.0, 0
            for k in range(0, len(Wv), 4096):
                X, y = Wv[k:k + 4096], yv[k:k + 4096]
                vl += float(loss_of(X, y)) * len(y)
                n += len(y)
        vl /= n
        print(f'  [rL {h}] epoch {ep:2d}/{EPOCHS} val {vl:.4f}', flush=True)
        if vl < best:
            best, best_ep = vl, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    ck = dict(model_state=best_state, mean=mean, std=std, features=LFEATURES, window=WINDOW, held_out=h, val=val_s,
              best_epoch=best_ep, seed=SEED)
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(ck, os.path.join(MODEL_DIR, f'rlidar_{h}.pt'))
    out = dict(held_out=h, best_epoch=best_ep)
    for setname in ('clean', 'lidar', 'overlap'):
        rs = [r for r in test if r['set'] == setname]
        p = np.concatenate([predict_rl(ck, r['X']) for r in rs])
        y = np.concatenate([r['y'] for r in rs])
        out[f'auc_{setname}'] = auc(1 - p, y < 0.5)
        out[f'flag_{setname}'] = float((p < GATE).mean())
        if (y < 0.5).any():
            out[f'recall_{setname}'] = float((p[y < 0.5] < GATE).mean())
            out[f'false_alarm_{setname}'] = float((p[y >= 0.5] < GATE).mean())
    return out


def eval_work(args):
    h, data = args
    torch.set_num_threads(1)
    import learning.baselines as B
    from zupt.core.config import load_config
    from zupt.ekf.ekf9 import load_ekf9_params
    from zupt.segments import load_segment_config
    from learning.features import FEATURES
    from learning.train_slip import predict_p
    ck = torch.load(os.path.join(MODEL_DIR, f'rel_{h}.pt'), map_location='cpu', weights_only=False)
    cs = torch.load(os.path.join(MODEL_DIR, f'slip_{h}.pt'), map_location='cpu', weights_only=False)
    cl = torch.load(os.path.join(MODEL_DIR, f'rlidar_{h}.pt'), map_location='cpu', weights_only=False)
    cfg, p9, seg = load_config(DET), load_ekf9_params(EKF9), load_segment_config(SEG)
    rows = []
    for setname, run, fr, sc, feat, act, man, sess in session_runs(h, data):
        z = np.load(os.path.join(FEAT_DIR, f'{feat}.npz'))
        cols, d = list(z['columns']), z['data']
        Xf = d[:, [cols.index(c) for c in FEATURES]].astype(np.float32)
        rp, ps = predict_r(ck, Xf), predict_p(cs, Xf)
        lz = np.load(os.path.join(LFEAT_DIR, f'{run}.npz'))
        rl = predict_rl(cl, lz['X'].astype(np.float32))
        s2 = (fr.imu, fr.odom, sess[2], sess[3])
        kw = dict(r_imu=rp[:, 0], r_odom_imu=rp[:, 1], q=ps, s=ps, chi2=True, stale=True, scan_skip=rl < GATE)
        res = dict(model=h, set=setname, run=run, method='hybrid_stale_rL', rl_flag=float((rl < GATE).mean()),
                   rl_flag_active=float((rl[act] < GATE).mean()) if act.any() else np.nan, **man)
        res.update(B.with_5c(s2, fr.odom, sc, cfg, p9, seg, **kw))
        rows.append(res)
    return rows


def med(a, b):
    return 100 * float(np.median(a / b - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['build', 'train', 'eval'])
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--data', default=_P.DATA)
    a = ap.parse_args()
    if os.environ.get('OMP_NUM_THREADS') != '1':
        sys.exit('ABORT: run with OMP_NUM_THREADS=1')
    if SUFFIX:
        sys.exit('ABORT: default fault set required')
    hs = sorted(os.path.basename(p)[5:-3] for p in glob.glob(os.path.join(MODEL_DIR, 'slip_*.pt')))
    pool = mp.get_context('fork').Pool(a.jobs) if a.jobs > 1 else None
    mapf = pool.map if pool else map
    pd.set_option('display.width', 220)
    if a.cmd == 'build':
        for h, n in mapf(build_work, [(h, a.data) for h in hs]):
            print(f'  {h}: {n} runs', flush=True)
        runs = load_lruns(only_train=True)
        y = np.concatenate([r['y'] for r in runs])
        print(f'L2 features: {len(runs)} training runs, {len(y)} scans, unreliable share {100 * (y < 0.5).mean():.1f} %')
    elif a.cmd == 'train':
        df = pd.DataFrame(list(mapf(train_fold, [(h, hs) for h in hs])))
        os.makedirs(RES, exist_ok=True)
        df.to_csv(os.path.join(RES, 'fullsys_rlidar_lobo.csv'), index=False)
        print('\nDEVELOPMENT r_LiDAR LOBO (held out):')
        print(df.round(3).to_string(index=False))
        print('means: ' + '  '.join(f'{c} {df[c].mean():.3f}' for c in df.columns if c not in ('held_out', 'best_epoch')))
    else:
        df = pd.DataFrame([r for o in mapf(eval_work, [(h, a.data) for h in hs]) for r in o])
        df.to_csv(os.path.join(RES, 'fullsys_rlidar_closed_loop.csv'), index=False)
        L = pd.read_csv(os.path.join(RES, 'fullsys_lidar_faults.csv'))
        B = pd.read_csv(os.path.join(RES, 'fullsys_baselines.csv'))
        ref = pd.concat([L[(L.method == 'hybrid_stale')][['run', 'rmse']],
                         B[B.method == 'hybrid'][['run', 'rmse']].assign(run=lambda x: x.run)])
        ref = ref.drop_duplicates('run')
        m = df.merge(ref, on='run', suffixes=('', '_ref'))
        print('\nDEVELOPMENT r_LiDAR closed loop (hybrid_stale_rL vs hybrid_stale; F14 set vs F14 hybrid):')
        print(m.groupby('set')[['rmse', 'rmse_ref', 'diverged', 'rl_flag', 'rl_flag_active']].mean().round(3).to_string())
        lid = m[m.set == 'lidar']
        print('\nLiDAR-fault runs by type (mean RMSE rL / ref):')
        print(lid.groupby('ltype')[['rmse', 'rmse_ref']].mean().round(3).to_string())
        ov = m[m.set == 'overlap']
        print('\nOverlap runs (mean RMSE rL / ref):')
        print(ov.groupby(['ltype', 'osensor', 'otype'])[['rmse', 'rmse_ref']].mean().round(3).to_string())
        res = {}
        for s in ('clean', 'lidar', 'overlap', 'f14'):
            x = m[m.set == s]
            res[s] = (int((x.rmse < x.rmse_ref).sum()), len(x), med(x.rmse, x.rmse_ref))
            print(f'  {s:8s}: rL wins {res[s][0]}/{res[s][1]}, median change {res[s][2]:+.1f} %')
        gain = any(res[s][0] >= math.ceil(0.6 * res[s][1]) and res[s][2] <= -3 for s in ('lidar', 'overlap'))
        ok = gain and res['clean'][2] <= 2 and res['f14'][2] <= 2
        print(f'\nRule (3) r_LiDAR: gain on LiDAR or overlap runs: {gain}; clean {res["clean"][2]:+.1f} %; '
              f'F14 {res["f14"][2]:+.1f} % -> {"ADOPT" if ok else "not adopted"}')
    if pool:
        pool.close()


if __name__ == '__main__':
    main()
