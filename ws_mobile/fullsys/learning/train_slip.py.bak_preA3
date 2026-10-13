#!/usr/bin/env python3
"""train_slip.py -- F11: LOBO training of P_slip on the F7 EKF9 features (Phase 3B protocol).
LSTM(14->64) -> logit; BCE pos_weight = no-slip/slip of the fit split; Adam 1e-3, batch 256,
15 epochs, seed 42; val = first remaining session; best-val ckpt; norm on fit rows; stride 5.
Label: 'slip' column of the F10 feature files (all 27 runs per session).
Outputs <ws>/fullsys_logs/models/slip_<h>.pt, results/fullsys_slip_lobo.csv (held-out AUC/Brier).
Usage: OMP_NUM_THREADS=1 python3 learning/train_slip.py [--jobs N]
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
from learning.features import FEATURES  # noqa: E402
from learning.train import FEAT_DIR, MODEL_DIR, RES  # noqa: E402

WINDOW, HIDDEN, LR, BATCH, EPOCHS, SEED, STRIDE = 50, 64, 1e-3, 256, 15, 42, 5


class SlipLSTM(nn.Module):
    def __init__(self, n_in=len(FEATURES), hidden=HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, 1, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def logit(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])[:, 0]

    def forward(self, x):
        return torch.sigmoid(self.logit(x))


def load_runs():
    runs = []
    for p in sorted(glob.glob(os.path.join(FEAT_DIR, '*.npz'))):
        z = np.load(p)
        cols, d = list(z['columns']), z['data']
        if 'slip' not in cols:
            sys.exit(f'ABORT: {p} has no slip column (rebuild features with F10)')
        m = json.loads(str(z['manifest']))
        runs.append(dict(name=m['name'], session=m['session'], clean=m.get('type') == 'clean',
                         X=d[:, [cols.index(c) for c in FEATURES]].astype(np.float32),
                         y=d[:, cols.index('slip')].astype(np.float32)))
    return runs


def windows(runs, stride):
    return np.array([(i, e) for i, r in enumerate(runs) for e in range(WINDOW - 1, len(r['X']), stride)],
                    dtype=np.int64)


def gather(runs, idx, mean, std):
    X = np.stack([runs[i]['X'][e - WINDOW + 1:e + 1] for i, e in idx])
    return ((X - mean) / std).astype(np.float32), np.array([runs[i]['y'][e] for i, e in idx], np.float32)


def predict_p(ck, X):
    """Causal P_slip for every row (rows < 49 left-padded with row 0), as Phase 3B run time."""
    m = SlipLSTM()
    m.load_state_dict(ck['model_state'])
    m.eval()
    pad = (np.concatenate([np.repeat(X[:1], WINDOW - 1, axis=0), X]) - ck['mean']) / ck['std']
    out = []
    with torch.no_grad():
        for k in range(0, len(X), 2048):
            w = np.stack([pad[i:i + WINDOW] for i in range(k, min(k + 2048, len(X)))]).astype(np.float32)
            out.append(m(torch.from_numpy(w)).numpy())
    return np.concatenate(out).astype(float)


def auc(y, p):
    from scipy.stats import rankdata
    y = np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = rankdata(p)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def vloss(model, runs, idx, mean, std, lf):
    model.eval()
    tot = 0.0
    with torch.no_grad():
        for k in range(0, len(idx), 4096):
            X, y = gather(runs, idx[k:k + 4096], mean, std)
            tot += float(lf(model.logit(torch.from_numpy(X)), torch.from_numpy(y))) * len(y)
    return tot / len(idx)


def train_fold(args):
    h, sessions = args
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
    fi, vi = windows(fit, STRIDE), windows(val, 1)
    fy = np.array([fit[i]['y'][e] for i, e in fi])
    n_pos, n_neg = float(fy.sum()), float(len(fy) - fy.sum())
    lf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg / max(n_pos, 1.0)))
    model = SlipLSTM()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(SEED)
    best, best_ep, best_state, hist = float('inf'), -1, None, []
    for ep in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(len(fi))
        tot = 0.0
        for k in range(0, len(order), BATCH):
            X, y = gather(fit, fi[order[k:k + BATCH]], mean, std)
            opt.zero_grad()
            loss = lf(model.logit(torch.from_numpy(X)), torch.from_numpy(y))
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
        vl = vloss(model, val, vi, mean, std, lf)
        hist.append(vl)
        print(f'  [slip {h}] epoch {ep:2d}/{EPOCHS} fit {tot / len(fi):.4f} val {vl:.4f}', flush=True)
        if vl < best:
            best, best_ep = vl, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    ck = dict(model_state=best_state, mean=mean, std=std, features=FEATURES, window=WINDOW, hidden=HIDDEN,
              held_out=h, val=val_s, fit=fit_s, stride=STRIDE, epochs=EPOCHS, best_epoch=best_ep,
              val_history=hist, seed=SEED, pos_weight=n_neg / max(n_pos, 1.0))
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(ck, os.path.join(MODEL_DIR, f'slip_{h}.pt'))
    c = [r for r in test if r['clean']][0]
    pc = predict_p(ck, c['X'])
    pa = np.concatenate([predict_p(ck, r['X']) for r in test])
    ya = np.concatenate([r['y'] for r in test])
    return dict(held_out=h, val=val_s, best_epoch=best_ep, slip_share_clean=float(c['y'].mean()),
                auc_clean=auc(c['y'], pc), brier_clean=float(np.mean((pc - c['y']) ** 2)),
                auc_all=auc(ya, pa), brier_all=float(np.mean((pa - ya) ** 2)),
                false_alarm_clean=float(((pc >= 0.5) & (c['y'] == 0)).sum() / max((c['y'] == 0).sum(), 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=1)
    a = ap.parse_args()
    if os.environ.get('OMP_NUM_THREADS') != '1':
        sys.exit('ABORT: run with OMP_NUM_THREADS=1')
    sessions = sorted({r['session'] for r in load_runs()})
    print(f'P_slip LOBO over {len(sessions)} sessions: {sessions}', flush=True)
    tasks = [(h, sessions) for h in sessions]
    if a.jobs <= 1:
        rows = [train_fold(t) for t in tasks]
    else:
        with mp.get_context('fork').Pool(a.jobs) as pool:
            rows = pool.map(train_fold, tasks)
    df = pd.DataFrame(rows)
    os.makedirs(RES, exist_ok=True)
    df.to_csv(os.path.join(RES, 'fullsys_slip_lobo.csv'), index=False)
    pd.set_option('display.width', 200)
    print('\n' + df.round(4).to_string(index=False))
    print(f'\nDEVELOPMENT P_slip (held out): AUC clean mean {df.auc_clean.mean():.3f} '
          f'(min {df.auc_clean.min():.3f}); AUC all runs mean {df.auc_all.mean():.3f}; '
          f'false alarm (P>=0.5, no slip) clean mean {100 * df.false_alarm_clean.mean():.1f} %')


if __name__ == '__main__':
    main()
