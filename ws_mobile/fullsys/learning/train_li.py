#!/usr/bin/env python3
"""train_li.py -- F14b: LOBO training of the Li et al.-style motion-state BiLSTM (stationary/straight/turning).
Architecture as the paper (3-layer BiLSTM 128/64/32, FC 64, softmax), window 30, causal window, 5-D input;
labels from ground truth (FINDINGS F14b). Protocol as ours (LOBO, val = first remaining, 15 epochs,
best-val ckpt, stride 5, seed 42). Outputs <ws>/fullsys_logs/models/li_<h>.pt, results/fullsys_li_lobo.csv.
Usage: OMP_NUM_THREADS=1 python3 learning/train_li.py [--jobs N]
"""
import argparse
import contextlib
import glob
import io
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
from learning.train import FEAT_DIR, MODEL_DIR, RES  # noqa: E402
from zupt import paths as _P  # noqa: E402

INPUTS = ['acc_x', 'acc_y', 'acc_z', 'gyro_z', 'odom_v']
WINDOW, LR, BATCH, EPOCHS, SEED, STRIDE = 30, 1e-3, 256, 15, 42, 5
V_STILL, W_STILL, W_TURN = 0.02, 0.02, 0.10


class LiNet(nn.Module):
    def __init__(self, n_in=len(INPUTS)):
        super().__init__()
        self.l1 = nn.LSTM(n_in, 128, batch_first=True, bidirectional=True)
        self.l2 = nn.LSTM(256, 64, batch_first=True, bidirectional=True)
        self.l3 = nn.LSTM(128, 32, batch_first=True, bidirectional=True)
        self.d1, self.d2, self.d3 = nn.Dropout(0.2), nn.Dropout(0.2), nn.Dropout(0.3)
        self.fc = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        self.out = nn.Linear(64, 3)

    def forward(self, x):
        h, _ = self.l1(x)
        h, _ = self.l2(self.d1(h))
        h, _ = self.l3(self.d2(h))
        return self.out(self.d3(self.fc(h[:, -1, :])))     # logits


def motion_label(t, gt):
    from zupt.replay import quat_to_rpy
    tg = gt.t.values
    _, _, th = quat_to_rpy(gt.gt_orient_x, gt.gt_orient_y, gt.gt_orient_z, gt.gt_orient_w)
    th = np.unwrap(np.asarray(th, float))
    vx, vy = np.gradient(gt.gt_x.values, tg), np.gradient(gt.gt_y.values, tg)
    v = np.interp(t, tg, np.hypot(vx, vy))
    w = np.interp(t, tg, np.gradient(th, tg))
    lab = np.ones(len(t), np.int64)
    lab[(v < V_STILL) & (np.abs(w) < W_STILL)] = 0
    lab[np.abs(w) > W_TURN] = 2
    return lab


def load_runs(data):
    from zupt.replay import load_session
    labels, runs = {}, []
    for p in sorted(glob.glob(os.path.join(FEAT_DIR, '*.npz'))):
        z = np.load(p)
        cols, d = list(z['columns']), z['data']
        m = json.loads(str(z['manifest']))
        s = m['session']
        if s not in labels:
            with contextlib.redirect_stdout(io.StringIO()):
                gt = load_session(data, s)[3]
            labels[s] = gt
        runs.append(dict(name=m['name'], session=s, clean=m.get('type') == 'clean',
                         X=d[:, [cols.index(c) for c in INPUTS]].astype(np.float32),
                         y=motion_label(d[:, cols.index('t')], labels[s])))
    return runs


def windows(runs, stride):
    return np.array([(i, e) for i, r in enumerate(runs) for e in range(WINDOW - 1, len(r['X']), stride)],
                    dtype=np.int64)


def gather(runs, idx, mean, std):
    X = np.stack([runs[i]['X'][e - WINDOW + 1:e + 1] for i, e in idx])
    return ((X - mean) / std).astype(np.float32), np.array([runs[i]['y'][e] for i, e in idx], np.int64)


def predict_probs(ck, X):
    """Causal class probabilities for every row (rows < 29 left-padded with row 0)."""
    m = LiNet()
    m.load_state_dict(ck['model_state'])
    m.eval()
    pad = (np.concatenate([np.repeat(X[:1], WINDOW - 1, axis=0), X]) - ck['mean']) / ck['std']
    out = []
    with torch.no_grad():
        for k in range(0, len(X), 2048):
            w = np.stack([pad[i:i + WINDOW] for i in range(k, min(k + 2048, len(X)))]).astype(np.float32)
            out.append(torch.softmax(m(torch.from_numpy(w)), dim=1).numpy())
    return np.concatenate(out)


def train_fold(args):
    h, sessions, data = args
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    runs = load_runs(data)
    rest = sorted(s for s in sessions if s != h)
    val_s, fit_s = rest[0], rest[1:]
    fit = [r for r in runs if r['session'] in fit_s]
    val = [r for r in runs if r['session'] == val_s]
    test = [r for r in runs if r['session'] == h]
    allfit = np.concatenate([r['X'] for r in fit])
    mean, std = allfit.mean(axis=0), allfit.std(axis=0)
    std[std == 0] = 1.0
    fi, vi = windows(fit, STRIDE), windows(val, STRIDE)
    lf = nn.CrossEntropyLoss()
    model = LiNet()
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
            loss = lf(model(torch.from_numpy(X)), torch.from_numpy(y))
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
        model.eval()
        vl, n = 0.0, 0
        with torch.no_grad():
            for k in range(0, len(vi), 4096):
                X, y = gather(val, vi[k:k + 4096], mean, std)
                vl += float(lf(model(torch.from_numpy(X)), torch.from_numpy(y))) * len(y)
                n += len(y)
        vl /= n
        hist.append(vl)
        print(f'  [li {h}] epoch {ep:2d}/{EPOCHS} fit {tot / len(fi):.4f} val {vl:.4f}', flush=True)
        if vl < best:
            best, best_ep = vl, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    ck = dict(model_state=best_state, mean=mean, std=std, inputs=INPUTS, window=WINDOW, held_out=h,
              val=val_s, fit=fit_s, best_epoch=best_ep, val_history=hist, seed=SEED)
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(ck, os.path.join(MODEL_DIR, f'li_{h}.pt'))
    c = [r for r in test if r['clean']][0]
    p = predict_probs(ck, c['X']).argmax(axis=1)
    acc = float((p == c['y']).mean())
    rec = {k: float((p[c['y'] == k] == k).mean()) if (c['y'] == k).any() else np.nan for k in range(3)}
    return dict(held_out=h, val=val_s, best_epoch=best_ep, acc_clean=acc, recall_still=rec[0],
                recall_straight=rec[1], recall_turn=rec[2], share_still=float((c['y'] == 0).mean()),
                share_turn=float((c['y'] == 2).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--data', default=_P.DATA)
    a = ap.parse_args()
    if os.environ.get('OMP_NUM_THREADS') != '1':
        sys.exit('ABORT: run with OMP_NUM_THREADS=1')
    sessions = sorted({json.loads(str(np.load(p)['manifest']))['session']
                       for p in glob.glob(os.path.join(FEAT_DIR, '*.npz'))})
    print(f'Li-style classifier LOBO over {len(sessions)} sessions', flush=True)
    tasks = [(h, sessions, a.data) for h in sessions]
    if a.jobs <= 1:
        rows = [train_fold(t) for t in tasks]
    else:
        with mp.get_context('fork').Pool(a.jobs) as pool:
            rows = pool.map(train_fold, tasks)
    df = pd.DataFrame(rows)
    os.makedirs(RES, exist_ok=True)
    df.to_csv(os.path.join(RES, 'fullsys_li_lobo.csv'), index=False)
    pd.set_option('display.width', 200)
    print('\n' + df.round(3).to_string(index=False))
    print(f'\nDEVELOPMENT Li-style classifier (held out, clean runs): accuracy mean {df.acc_clean.mean():.3f}; '
          f'recall stationary {df.recall_still.mean():.3f}, straight {df.recall_straight.mean():.3f}, '
          f'turning {df.recall_turn.mean():.3f}')


if __name__ == '__main__':
    main()
