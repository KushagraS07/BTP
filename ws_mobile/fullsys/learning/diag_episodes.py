#!/usr/bin/env python3
"""diag_episodes.py -- D-F9b (development, read-only): where/when the LSTM r drops below the gate on CLEAN runs."""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from learning.closed_loop import FEAT_DIR, GATE, MODEL_DIR, predict_r  # noqa: E402
from learning.features import FEATURES  # noqa: E402

torch.set_num_threads(1)
for h in sys.argv[1:]:
    ck = torch.load(os.path.join(MODEL_DIR, f'rel_{h}.pt'), map_location='cpu', weights_only=False)
    z = np.load(os.path.join(FEAT_DIR, f'{h}.npz'))
    cols, d = list(z['columns']), z['data']
    col = lambda c: d[:, cols.index(c)]
    r = predict_r(ck, d[:, [cols.index(c) for c in FEATURES]].astype(np.float32))
    t = col('t') - col('t')[0]
    print(f'\n{h}  ({t[-1]:.1f} s)  sensor  start_s  dur_s  min_r  x  y  odom_v  |wheel-imu|')
    for j, name in ((0, 'imu'), (1, 'odom')):
        low = np.r_[False, r[:, j] < GATE, False].astype(int)
        for a, b in zip(np.where(np.diff(low) == 1)[0], np.where(np.diff(low) == -1)[0]):
            s = slice(a, b)
            print(f'  {name:4s} {t[a]:6.1f} {t[b - 1] - t[a] + 0.02:5.2f} {r[s, j].min():.2f} '
                  f'{col("ekf_x")[a]:6.2f} {col("ekf_y")[a]:6.2f} {col("odom_v")[s].mean():5.2f} '
                  f'{np.abs(col("wheel_imu_dis")[s]).mean():.3f}')
