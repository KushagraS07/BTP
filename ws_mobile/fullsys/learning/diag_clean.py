#!/usr/bin/env python3
"""diag_clean.py -- D-F9a (development, read-only): LSTM r on clean runs and pre-onset rows."""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from learning.closed_loop import FEAT_DIR, GATE, MODEL_DIR, RES, predict_r  # noqa: E402
from learning.features import FEATURES  # noqa: E402
torch.set_num_threads(1)
cl = pd.read_csv(os.path.join(RES, 'fullsys_rel_closed_loop.csv'))
rows = []
for mp_ in sorted(glob.glob(os.path.join(MODEL_DIR, 'rel_*.pt'))):
    ck = torch.load(mp_, map_location='cpu', weights_only=False)
    h = ck['held_out']
    z = np.load(os.path.join(FEAT_DIR, f'{h}.npz'))
    cols = list(z['columns'])
    r = predict_r(ck, z['data'][:, [cols.index(c) for c in FEATURES]].astype(np.float32))
    pre = []
    for p in sorted(glob.glob(os.path.join(FEAT_DIR, f'{h}_*.npz'))):
        zf = np.load(p)
        m = json.loads(str(zf['manifest']))
        c2 = list(zf['columns'])
        d = zf['data']
        t = d[:, c2.index('t')]
        on = m.get('onset')
        if on is None:
            continue
        rp = predict_r(ck, d[:, [c2.index(c) for c in FEATURES]].astype(np.float32))
        pre.append(rp[t < on])
    pre = np.concatenate(pre) if pre else np.zeros((0, 2))
    c = cl[(cl.model == h) & (cl.type == 'clean')].iloc[0]
    rows.append(dict(session=h, fixed=c.fixed, lstm=c.lstm, pct=100 * (c.lstm / c.fixed - 1),
                     r_imu_mean=r[:, 0].mean(), imu_below_gate=100 * (r[:, 0] < GATE).mean(),
                     imu_below_09=100 * (r[:, 0] < 0.9).mean(),
                     r_odom_mean=r[:, 1].mean(), odom_below_gate=100 * (r[:, 1] < GATE).mean(),
                     odom_below_09=100 * (r[:, 1] < 0.9).mean(),
                     pre_imu_below_gate=100 * (pre[:, 0] < GATE).mean() if len(pre) else np.nan,
                     pre_odom_below_gate=100 * (pre[:, 1] < GATE).mean() if len(pre) else np.nan))
df = pd.DataFrame(rows)
pd.set_option('display.width', 220)
print('D-F9a clean runs (development). below_* = % of rows; pre_* = pre-onset rows of fault runs')
print(df.round(3).to_string(index=False))
