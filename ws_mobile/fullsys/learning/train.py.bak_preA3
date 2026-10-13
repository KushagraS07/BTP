#!/usr/bin/env python3
"""
train.py -- F7 step 4: LOBO training of the EKF9 reliability LSTM (r_imu, r_odom).

Protocol = Phase 2 (scripts/train_reliability_lstm.py, FINDINGS 2026-10-03 §: --baseline-weight 5,
lr 1e-3, dropout 0, seed 42, 15 epochs, best-validation checkpoint), on the F7 features:
  window 50 rows (1 s at 50 Hz), causal, label = (r_imu, r_odom) of the last row
  LSTM(len(FEATURES) -> 64, 1 layer) -> two sigmoid heads; MSE (mean of the two heads)
  folds: held-out dev session h (its clean run + 26 faults); validation = first of the remaining
  sessions (sorted); fit = the rest. Normalisation mean/std on fit rows only (std 0 -> 1).
  clean-run windows weighted x5 in the fit loss (Phase 2 --baseline-weight 5).
Deviation from Phase 2, logged in F8 before training: training windows taken every STRIDE rows
(default 5) instead of every row -- consecutive windows overlap 49/50 rows, and stride 1 would need
~20 h of CPU for 8 folds on the VM. Validation and test use every row.
Outputs: <ws>/fullsys_logs/models/rel_<h>.pt, results/fullsys_rel_lobo.csv (open-loop test RMSE of
r vs trivial predictors 'always 1' and 'fit mean'; development numbers).
Usage: OMP_NUM_THREADS=1 python3 learning/train.py [--jobs N] [--epochs 15] [--stride 5]
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from zupt import paths as _P  # noqa: E402
from learning.features import FEATURES  # noqa: E402

from learning.faults import SUFFIX as _FS  # noqa: E402  (A1: features_sigma for the sigma set)
FEAT_DIR = os.path.join(_P.WS, 'fullsys_logs', 'features' + _FS)
MODEL_DIR = os.path.join(_P.WS, 'fullsys_logs', 'models')
RES = os.path.join(_P.WS, 'results')
WINDOW, HIDDEN, LAYERS, LR, BATCH, SEED, CLEAN_W = 50, 64, 1, 1e-3, 256, 42, 5.0
LABELS = ['r_imu', 'r_odom']


class RelLSTM(nn.Module):
    def __init__(self, n_in=len(FEATURES), hidden=HIDDEN, layers=LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, layers, batch_first=True)
        self.head_imu = nn.Linear(hidden, 1)
        self.head_odom = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        return torch.cat([torch.sigmoid(self.head_imu(h)), torch.sigmoid(self.head_odom(h))], dim=1)


def load_runs(feat_dir=FEAT_DIR):
    runs = []
    for p in sorted(glob.glob(os.path.join(feat_dir, '*.npz'))):
        z = np.load(p, allow_pickle=False)
        cols = list(z['columns'])
        d = z['data']
        man = json.loads(str(z['manifest']))
        runs.append(dict(name=man['name'], session=man['session'], clean=man.get('type') == 'clean',
                         X=d[:, [cols.index(c) for c in FEATURES]].astype(np.float32),
                         Y=d[:, [cols.index(c) for c in LABELS]].astype(np.float32)))
    return runs


def window_index(runs, stride):
    """(run, end_row) pairs for full windows ending at end_row, every `stride` rows."""
    idx = []
    for i, r in enumerate(runs):
        idx += [(i, e) for e in range(WINDOW - 1, len(r['X']), stride)]
    return np.array(idx, dtype=np.int64)


def gather(runs, idx, mean, std):
    X = np.stack([runs[i]['X'][e - WINDOW + 1:e + 1] for i, e in idx])
    Y = np.stack([runs[i]['Y'][e] for i, e in idx])
    return (X - mean) / std, Y


def mse_eval(model, runs, idx, mean, std, bs=4096):
    model.eval()
    se, n = 0.0, 0
    preds = []
    with torch.no_grad():
        for k in range(0, len(idx), bs):
            X, Y = gather(runs, idx[k:k + bs], mean, std)
            p = model(torch.from_numpy(X.astype(np.float32))).numpy()
            preds.append(p)
            se += float(((p - Y) ** 2).mean(axis=1).sum())
            n += len(Y)
    return se / max(n, 1), (np.concatenate(preds) if preds else np.zeros((0, 2)))


def train_fold(args):
    h, sessions, epochs, stride = args
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    runs = load_runs()
    rest = sorted(s for s in sessions if s != h)
    val_s, fit_s = rest[0], rest[1:]
    fit = [r for r in runs if r['session'] in fit_s]
    val = [r for r in runs if r['session'] == val_s]
    test = [r for r in runs if r['session'] == h]
    allfit = np.concatenate([r['X'] for r in fit])
    mean, std = allfit.mean(axis=0), allfit.std(axis=0)
    std[std == 0] = 1.0
    fi, vi, ti = window_index(fit, stride), window_index(val, 1), window_index(test, 1)
    w_fit = np.where([fit[i]['clean'] for i in fi[:, 0]], CLEAN_W, 1.0).astype(np.float32)
    model = RelLSTM()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(SEED)
    best, best_ep, best_state, hist = float('inf'), -1, None, []
    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(fi))
        tot = 0.0
        for k in range(0, len(order), BATCH):
            b = order[k:k + BATCH]
            X, Y = gather(fit, fi[b], mean, std)
            w = torch.from_numpy(w_fit[b])
            opt.zero_grad()
            per = ((model(torch.from_numpy(X.astype(np.float32))) - torch.from_numpy(Y)) ** 2).mean(dim=1)
            loss = (per * w).sum() / w.sum()
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        vl, _ = mse_eval(model, val, vi, mean, std)
        hist.append(vl)
        print(f'  [{h}] epoch {ep:2d}/{epochs} fit {tot / len(fi):.5f} val {vl:.5f}', flush=True)
        if vl < best:
            best, best_ep = vl, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    _, p = mse_eval(model, test, ti, mean, std)
    Yt = np.stack([test[i]['Y'][e] for i, e in ti])
    fitY = np.concatenate([r['Y'] for r in fit])
    rm = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    row = dict(held_out=h, val=val_s, best_epoch=best_ep, best_val=best,
               rmse_imu=rm(p[:, 0], Yt[:, 0]), rmse_odom=rm(p[:, 1], Yt[:, 1]),
               one_imu=rm(1.0, Yt[:, 0]), one_odom=rm(1.0, Yt[:, 1]),
               mean_imu=rm(fitY[:, 0].mean(), Yt[:, 0]), mean_odom=rm(fitY[:, 1].mean(), Yt[:, 1]),
               n_fit_windows=len(fi), n_test_windows=len(ti))
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(dict(model_state=best_state, mean=mean, std=std, features=FEATURES, labels=LABELS,
                    window=WINDOW, hidden=HIDDEN, layers=LAYERS, held_out=h, val=val_s, fit=fit_s,
                    stride=stride, epochs=epochs, best_epoch=best_ep, val_history=hist, seed=SEED),
               os.path.join(MODEL_DIR, f'rel_{h}.pt'))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--stride', type=int, default=5)
    a = ap.parse_args()
    if os.environ.get('OMP_NUM_THREADS') != '1':
        sys.exit('ABORT: run with OMP_NUM_THREADS=1')
    sessions = sorted({r['session'] for r in load_runs()})
    print(f'dev sessions ({len(sessions)}): {sessions}', flush=True)
    if len(sessions) < 3:
        sys.exit('ABORT: need >= 3 sessions')
    tasks = [(h, sessions, a.epochs, a.stride) for h in sessions]
    if a.jobs <= 1:
        rows = [train_fold(t) for t in tasks]
    else:
        with mp.get_context('fork').Pool(a.jobs) as pool:
            rows = pool.map(train_fold, tasks)
    df = pd.DataFrame(rows)
    os.makedirs(RES, exist_ok=True)
    df.to_csv(os.path.join(RES, 'fullsys_rel_lobo.csv'), index=False)
    pd.set_option('display.width', 200)
    print('\n' + df.round(4).to_string(index=False))
    print('\nDEVELOPMENT (open loop, r error): mean RMSE r_imu {:.4f} (always-1 {:.4f}, fit-mean {:.4f}); '
          'r_odom {:.4f} (always-1 {:.4f}, fit-mean {:.4f})'.format(
              df.rmse_imu.mean(), df.one_imu.mean(), df.mean_imu.mean(),
              df.rmse_odom.mean(), df.one_odom.mean(), df.mean_odom.mean()))


if __name__ == '__main__':
    main()
