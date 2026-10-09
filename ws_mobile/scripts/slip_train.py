#!/usr/bin/env python3
"""
slip_train.py -- Phase 3B: train the P_slip LSTM, LOBO over the 8 dev sessions (PREREG_slip.md §5).

Fixed protocol (no setting here is chosen from results):
  inputs   13 features of features_<s>.csv, order FEATURES below
  window   50 rows, causal, ending at row t; label = slip(t) of slip_labels_<s>.csv (NaN dropped)
           training windows: t >= 49 (full windows only, as Phase 2 make_windows)
           run time (predict_p): every row; rows t < 49 are left-padded with row 0 (as the
           Phase 2 run-time LSTM does)
  network  LSTM(13 -> 64, 1 layer, dropout 0) -> Linear(64 -> 1) on the last step -> sigmoid
  loss     BCE (implemented as BCEWithLogitsLoss on the logit: same loss, numerically stable),
           pos_weight = #non-slip / #slip windows of the FIT split only
  optim    Adam lr 1e-3, no weight decay, batch 256 shuffled, 15 epochs, seed 42 (re-set at the start
           of every fold, so a fold does not depend on the folds trained before it)
  folds    held-out h; validation = first of the remaining 7 (sorted); fit = the other 6
  ckpt     lowest validation loss (same weighted BCE, fit-split pos_weight); ties -> earlier epoch
  norm     mean/std over the fit windows (std = 0 -> 1), as Phase 2
Only dev sessions are read (role dev AND slip_accept ACCEPT, 8 asserted). calib/test never.

Writes sensor_logs/slip_model_<h>.pt (8 files) and results/slip_train_lobo.csv (development
LOBO numbers: AUC / Brier on the held-out dev session -- descriptive, NOT a test result).
Usage: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 slip_train.py
"""
import os
import sys

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    if os.environ.get(_v) != '1':
        sys.exit(f'ABORT: run with OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ({_v}={os.environ.get(_v)})')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ws_paths as W  # noqa: E402

LOG, RES = W.LOG_DIR, os.path.join(W.WS, 'results')
# ---------------- PREREG_slip.md §5 -- do not change ----------------
FEATURES = ['innov_s', 'innov_phi', 'innov_imu_theta', 'odom_scale', 'imu_scale',
            'imu_angular_vel_z', 'imu_linear_acc_x', 'imu_linear_acc_y', 'imu_linear_acc_z',
            'odom_linear_x', 'odom_angular_z', 'wheel_imu_disagreement', 'wheel_lidar_disagreement']
WINDOW, HIDDEN, LAYERS, DROPOUT = 50, 64, 1, 0.0
LR, BATCH, EPOCHS, SEED = 1e-3, 256, 15, 42
N_DEV = 8
# ---------------------------------------------------------------------


class SlipLSTM(nn.Module):
    def __init__(self, n_in=len(FEATURES), hidden=HIDDEN, layers=LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def logit(self, x):
        out, _ = self.lstm(x)
        return self.head(self.dropout(out[:, -1, :]))[:, 0]

    def forward(self, x):
        return torch.sigmoid(self.logit(x))


def load_session(s):
    f = pd.read_csv(os.path.join(LOG, f'features_{s}.csv'))
    lab = pd.read_csv(os.path.join(LOG, f'slip_labels_{s}.csv'))
    if len(f) != len(lab) or not np.allclose(f.timestamp.values, lab.timestamp.values, atol=1e-9, rtol=0):
        sys.exit(f'ABORT {s}: features and slip labels are not row-aligned')
    X = f[FEATURES].values.astype(np.float32)
    if not np.all(np.isfinite(X)):
        sys.exit(f'ABORT {s}: non-finite feature values')
    return X, lab.slip.values.astype(float), lab.area.values


def full_windows(X, y):
    idx = np.arange(WINDOW - 1, len(X))
    Xw = np.stack([X[i - WINDOW + 1:i + 1] for i in idx])
    keep = np.isfinite(y[idx])
    return Xw[keep], y[idx][keep].astype(np.float32)


def all_windows(X):
    pad = np.concatenate([np.repeat(X[:1], WINDOW - 1, axis=0), X])
    return np.stack([pad[i:i + WINDOW] for i in range(len(X))])


def predict_p(ckpt, X):
    """P_slip for every row of one session (run-time definition, causal)."""
    model = SlipLSTM()
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    Xw = (all_windows(X) - ckpt['mean']) / ckpt['std']
    out = []
    with torch.no_grad():
        for i in range(0, len(Xw), 2048):
            out.append(model(torch.from_numpy(Xw[i:i + 2048].astype(np.float32))).numpy())
    return np.concatenate(out).astype(float)


def load_ckpt(h):
    return torch.load(os.path.join(LOG, f'slip_model_{h}.pt'), map_location='cpu', weights_only=False)


def auc(y, p):
    y = np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = rankdata(p)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def dev_sessions():
    from slip_prepare import select
    return select(['dev'])['dev']


def train_fold(h, dev, data):
    rest = sorted(s for s in dev if s != h)
    val, fit = rest[0], rest[1:]
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    fx, fy = map(np.concatenate, zip(*[full_windows(*data[s][:2]) for s in fit]))
    vx, vy = full_windows(*data[val][:2])
    flat = fx.reshape(-1, fx.shape[-1])
    mean, std = flat.mean(axis=0).astype(np.float32), flat.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0
    fx, vx = ((fx - mean) / std).astype(np.float32), ((vx - mean) / std).astype(np.float32)
    n_pos, n_neg = int(fy.sum()), int(len(fy) - fy.sum())
    pos_w = torch.tensor(n_neg / n_pos, dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    g = torch.Generator().manual_seed(SEED)
    loader = DataLoader(TensorDataset(torch.from_numpy(fx), torch.from_numpy(fy)),
                        batch_size=BATCH, shuffle=True, generator=g)
    model = SlipLSTM()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    vxt, vyt = torch.from_numpy(vx), torch.from_numpy(vy)
    best, best_ep, best_state, hist = float('inf'), -1, None, []
    for ep in range(1, EPOCHS + 1):
        model.train()
        tot = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model.logit(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            vl = float(sum(loss_fn(model.logit(vxt[i:i + 4096]), vyt[i:i + 4096]).item() * len(vyt[i:i + 4096])
                           for i in range(0, len(vyt), 4096)) / len(vyt))
        hist.append(vl)
        print(f'  [{h}] epoch {ep:2d}/{EPOCHS} fit_loss {tot / len(fy):.4f} val_loss {vl:.4f}', flush=True)
        if vl < best:
            best, best_ep = vl, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    ckpt = dict(model_state=best_state, mean=mean, std=std, feature_names=FEATURES, window_size=WINDOW,
                hidden_size=HIDDEN, num_layers=LAYERS, held_out=h, val_session=val, fit_sessions=fit,
                pos_weight=float(pos_w), n_fit_windows=len(fy), fit_slip_share=n_pos / len(fy),
                best_epoch=best_ep, best_val_loss=best, val_loss_history=hist, seed=SEED)
    torch.save(ckpt, os.path.join(LOG, f'slip_model_{h}.pt'))
    return ckpt


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    dev = dev_sessions()
    if len(dev) != N_DEV:
        sys.exit(f'ABORT: {len(dev)} dev sessions, expected {N_DEV}')
    data = {s: load_session(s) for s in dev}
    print(f'dev sessions ({len(dev)}): {" ".join(dev)}\n', flush=True)

    rows = []
    for h in dev:
        ck = train_fold(h, dev, data)
        X, y, area = data[h]
        p = predict_p(ck, X)
        v = np.isfinite(y)
        on = area != 'off_patch'
        rows.append(dict(held_out=h, val=ck['val_session'], best_epoch=ck['best_epoch'],
                         best_val_loss=round(ck['best_val_loss'], 4), pos_weight=round(ck['pos_weight'], 3),
                         test_slip_share=round(float(y[v].mean()), 4),
                         auc=round(auc(y[v], p[v]), 4), brier=round(float(np.mean((p[v] - y[v]) ** 2)), 4),
                         mean_p_on_patch=round(float(p[on].mean()), 3) if on.any() else np.nan,
                         mean_p_off_patch=round(float(p[~on].mean()), 3),
                         false_alarm_off_patch_p05=round(float(((p >= 0.5) & ~on & (y == 0)).sum() /
                                                               max(int((~on & (y == 0)).sum()), 1)), 4)))
        r = rows[-1]
        print(f'[{h}] best epoch {r["best_epoch"]} (val {r["val"]}), held-out AUC {r["auc"]}, '
              f'Brier {r["brier"]}, mean P on/off patch {r["mean_p_on_patch"]}/{r["mean_p_off_patch"]}\n',
              flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(RES, exist_ok=True)
    out = os.path.join(RES, 'slip_train_lobo.csv')
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f'\nDEVELOPMENT LOBO (not a test result): mean AUC {df.auc.mean():.4f}, '
          f'min {df.auc.min():.4f}; mean Brier {df.brier.mean():.4f}')
    print(f'wrote {out} and {len(df)} sensor_logs/slip_model_*.pt')


if __name__ == '__main__':
    main()
