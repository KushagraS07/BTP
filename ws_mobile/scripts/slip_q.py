#!/usr/bin/env python3
"""
slip_q.py -- Phase 3B: process noise Q(t) = Q0 * (1 + k * P_slip(t-1)) on the FROZEN Phase 2 filter.

ONE code path (PREREG_slip.md §5). The frozen Phase 2 function
`lstm_adaptive_ekf.run_lstm_adaptive_ekf_core` is called unchanged. It builds its filter as
`DiffDriveEKF(...)` from its own module namespace. For S/O runs only, that name is replaced for
the duration of one call by a subclass whose predict() sets Q = Q0 * (1 + k * p[t-1]) and then
calls the unchanged DiffDriveEKF.predict(). No frozen file is edited (their sha256 stay as in
fix_ws/final_freeze.sha256).

  k = None            -> the frozen Phase 2 call, no substitution at all (system L)
  k = 0               -> substitution active, factor exactly 1.0 (regression test, must equal L)
  k > 0, p = P_slip   -> system S;  p = oracle slip label (NaN -> 0)  -> system O

Timing: predict(t) uses p[t-1]; p[-1] = 0 (PREREG_slip.md §5). p must have one finite value in
[0, 1] per filter step (same rows as features_<s>.csv).
"""
import contextlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lstm_adaptive_ekf as LA  # noqa: E402
from compare_adaptive_vs_fixed import load_init_pose, load_session_arrays  # noqa: E402

_FROZEN_EKF = LA.DiffDriveEKF


def q_ekf_class(k, p_slip):
    k = float(k)
    p = np.asarray(p_slip, dtype=float)
    if not np.all(np.isfinite(p)) or p.min() < 0.0 or p.max() > 1.0:
        raise ValueError('p_slip must be finite and in [0, 1] (map oracle NaN labels to 0 first)')
    made = []

    class QSlipEKF(_FROZEN_EKF):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._Q0 = self.Q.copy()
            self._i = 0
            made.append(self)

        def predict(self, v, omega, dt):
            p_prev = p[self._i - 1] if self._i > 0 else 0.0
            self.Q = self._Q0 * (1.0 + k * p_prev)
            self._i += 1
            super().predict(v, omega, dt)

    QSlipEKF.instances = made
    return QSlipEKF


@contextlib.contextmanager
def _substituted(cls):
    if LA.DiffDriveEKF is not _FROZEN_EKF:
        raise RuntimeError('lstm_adaptive_ekf.DiffDriveEKF already substituted (nested call?)')
    LA.DiffDriveEKF = cls
    try:
        yield
    finally:
        LA.DiffDriveEKF = _FROZEN_EKF


def session_inputs(s):
    arrays = load_session_arrays(s)
    n = len(arrays[-1])
    init = load_init_pose(s)
    imu_raw, odom_raw = LA.load_raw_features(s, n)
    return arrays, init, imu_raw, odom_raw, n


def run(s, model, mean, std, feats, win, k=None, p_slip=None, inputs=None):
    """Returns (x, y, theta, r_imu, r_odom) of the frozen Phase 2 LSTM-R filter, with Q(t) if k given."""
    arrays, init, imu_raw, odom_raw, n = inputs if inputs is not None else session_inputs(s)
    args = (*arrays, *init, imu_raw, odom_raw, model, mean, std, feats, win)
    if k is None:
        return LA.run_lstm_adaptive_ekf_core(*args)
    if p_slip is None or len(p_slip) != n:
        raise ValueError(f'{s}: p_slip needs {n} values, got {None if p_slip is None else len(p_slip)}')
    cls = q_ekf_class(k, p_slip)
    with _substituted(cls):
        out = LA.run_lstm_adaptive_ekf_core(*args)
    if len(cls.instances) != 1 or cls.instances[0]._i != n:
        raise RuntimeError(f'{s}: Q(t) filter built {len(cls.instances)}x / predict called '
                           f'{cls.instances[0]._i if cls.instances else 0}x, expected 1x / {n}x')
    return out
