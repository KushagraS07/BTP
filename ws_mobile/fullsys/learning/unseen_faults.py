"""
unseen_faults.py -- E (FINDINGS_fullsys.md E): fault KINDS never used in training, for the final test only.

6 types x 3 severities on the 50 Hz signals the filter consumes (same place as learning/faults.py):
    odom_zero   0.2 s slots after onset read v = w = 0 with probability 10 / 25 / 50 %
    odom_scale  v and w x (1 + 5 / 10 / 20 %)
    odom_spike  single-sample outliers +-10 x channel std at 1 / 2 / 5 per second (v and w)
    gyro_scale  gyro z x (1 + 5 / 15 / 30 %)
    gyro_sat    |gyro z| clipped at 70 / 50 / 30 % of the clean session's 99th percentile |gyro z|
    gyro_spike  as odom_spike, on gyro z
Onset uniform in [0.20, 0.60] of the session (F7), seed from the fault name; the fault stays on to the end.
Labels (analysis only): r = 1 / (1 + |faulted - clean| / std) per sample; odom = worst of v, w.
Returns learning.faults.FaultResult, so learning.features.build works unchanged.
"""
from dataclasses import dataclass
import zlib

import numpy as np

from learning.faults import ONSET_HI, ONSET_LO, FaultResult

USEVERITY = {
    'odom_zero': {'low': 0.10, 'medium': 0.25, 'high': 0.50},
    'odom_scale': {'low': 0.05, 'medium': 0.10, 'high': 0.20},
    'odom_spike': {'low': 1.0, 'medium': 2.0, 'high': 5.0},
    'gyro_scale': {'low': 0.05, 'medium': 0.15, 'high': 0.30},
    'gyro_sat': {'low': 0.70, 'medium': 0.50, 'high': 0.30},
    'gyro_spike': {'low': 1.0, 'medium': 2.0, 'high': 5.0},
}
SLOT = 0.2          # s, odom_zero slot length
SPIKE_K = 10.0      # spike amplitude in channel std
SAT_Q = 99.0        # percentile of |gyro z| (clean session) for the saturation limit
COLS = {'odom': ('odom', ['linear_x', 'angular_z']), 'gyro': ('imu', ['angular_vel_z'])}


@dataclass
class UFault:
    session: str
    ftype: str
    severity: str

    @property
    def sensor(self):
        return 'odom' if self.ftype.startswith('odom') else 'imu'

    @property
    def name(self):
        return f'{self.session}_unseen_{self.ftype}_{self.severity}'

    @property
    def value(self):
        return USEVERITY[self.ftype][self.severity]

    @property
    def seed(self):
        return zlib.crc32(self.name.encode())


def all_unseen(session):
    return [UFault(session, f, v) for f in USEVERITY for v in USEVERITY[f]]


def inject_unseen(sess, fault):
    """sess = (imu, odom, cmd, gt) from zupt.replay.load_session (clean); inputs are not modified."""
    imu, odom, cmd, gt = sess
    imu, odom = imu.copy(), odom.copy()
    frames = {'imu': imu, 'odom': odom}
    t0 = min(imu.t.iloc[0], odom.t.iloc[0])
    t1 = max(imu.t.iloc[-1], odom.t.iloc[-1])
    rng = np.random.default_rng(fault.seed)
    onset = t0 + rng.uniform(ONSET_LO, ONSET_HI) * (t1 - t0)
    df_name, cols = COLS[fault.ftype.split('_')[0]]
    df = frames[df_name]
    t = df.t.values
    after = t >= onset
    kind = fault.ftype.split('_')[1]
    sev = fault.value
    if kind == 'zero':
        slot = np.floor((t - onset) / SLOT).astype(int)
        nslot = int(slot.max()) + 1 if after.any() else 0
        off = rng.random(max(nslot, 1)) < sev
        zmask = after & off[np.clip(slot, 0, max(nslot - 1, 0))]
    if kind == 'spike':
        dt = float(np.median(np.diff(t)))
        smask = after & (rng.random(len(t)) < sev * dt)
        ssign = rng.choice([-1.0, 1.0], len(t))
    err = np.zeros(len(t))
    stats = {}
    for c in cols:
        x0 = df[c].values.astype(float)
        x = x0.copy()
        std = float(np.std(x0))
        if kind == 'zero':
            x[zmask] = 0.0
        elif kind == 'scale':
            x[after] *= 1.0 + sev
        elif kind == 'spike':
            x[smask] += ssign[smask] * SPIKE_K * std
        elif kind == 'sat':
            lim = sev * float(np.percentile(np.abs(x0), SAT_Q))
            x[after] = np.clip(x[after], -lim, lim)
            stats['limit'] = lim
        else:
            raise ValueError(fault.ftype)
        df[c] = x
        err = np.maximum(err, np.abs(x - x0) / max(std, 1e-12))
        stats[c] = {'std': std}
    r = 1.0 / (1.0 + err)
    r_imu = r if df_name == 'imu' else np.ones(len(imu))
    r_odom = r if df_name == 'odom' else np.ones(len(odom))
    manifest = dict(fault=fault.name, sensor=fault.sensor, type=fault.ftype, severity=fault.severity, value=sev,
                    onset=float(onset), session_t0=float(t0), session_t1=float(t1), seed=fault.seed,
                    columns=stats, set='unseen', affected=float(np.mean(err[after] > 0)) if after.any() else 0.0,
                    mean_err_std=float(err[after].mean()) if after.any() else 0.0)
    return FaultResult(imu, odom, r_imu, r_odom, manifest)
