"""
Fault injection + reliability labels for the EKF9 reliability LSTM (FINDINGS_fullsys.md F7).

Faults are applied offline to the signals the filter consumes, after zupt.replay.load_session
(IMU block-averaged to 50 Hz; odometry at its native rate):
    sensor 'imu'   -> imu.angular_vel_z                      (gyro z)
    sensor 'odom'  -> odom.linear_x and odom.angular_z       (wheel speeds)
Severity = multiple of the signal's own std in that (clean) session, as in Phase 2
(scripts/fault_injection.py). Every fault starts MID-SESSION (D6): onset uniform in
[ONSET_LO, ONSET_HI] of the session duration, from a seed derived from the fault name.

Labels (Phase 2 convention, scripts/generate_reliability_labels.py), r = 1 before onset and:
    noise, bias   r = 1 / (1 + severity)
    drift         r = 1 / (1 + severity * ramp),  ramp 0 -> 1 from onset to the session end
    dropout       r = 0 inside the hold window, 1 outside
The other sensor's label is 1 everywhere. Labels are returned on each sensor's own timeline.
"""
from dataclasses import dataclass, field
import zlib

import numpy as np

SEVERITY = {
    'noise': {'low': 1.0, 'medium': 3.0, 'high': 6.0, 'extreme': 12.0},
    'bias': {'small': 1.0, 'medium': 3.0, 'large': 6.0},
    'drift': {'slow': 1.0, 'medium': 3.0, 'fast': 6.0},
    'dropout': {'short': 0.05, 'medium': 0.15, 'long': 0.35},   # fraction of the session held
}
SIGNALS = {'imu': ('imu', ['angular_vel_z']), 'odom': ('odom', ['linear_x', 'angular_z'])}
ONSET_LO, ONSET_HI = 0.20, 0.60


@dataclass
class Fault:
    session: str
    sensor: str
    ftype: str
    severity: str

    @property
    def name(self):
        return f'{self.session}_fault_{self.sensor}_{self.ftype}_{self.severity}'

    @property
    def value(self):
        return SEVERITY[self.ftype][self.severity]

    @property
    def seed(self):
        return zlib.crc32(self.name.encode())


@dataclass
class FaultResult:
    imu: object
    odom: object
    r_imu: np.ndarray            # on imu.t
    r_odom: np.ndarray           # on odom.t
    manifest: dict = field(default_factory=dict)


def all_faults(session):
    return [Fault(session, s, f, v) for s in SIGNALS for f in SEVERITY for v in SEVERITY[f]]


def inject(sess, fault):
    """sess = (imu, odom, cmd, gt) from zupt.replay.load_session (clean). Returns FaultResult;
    the input frames are not modified."""
    imu, odom, cmd, gt = sess
    imu, odom = imu.copy(), odom.copy()
    frames = {'imu': imu, 'odom': odom}
    t0 = min(imu.t.iloc[0], odom.t.iloc[0])
    t1 = max(imu.t.iloc[-1], odom.t.iloc[-1])
    rng = np.random.default_rng(fault.seed)
    onset = t0 + rng.uniform(ONSET_LO, ONSET_HI) * (t1 - t0)
    df_name, cols = SIGNALS[fault.sensor]
    df = frames[df_name]
    t = df.t.values
    after = t >= onset
    sev = fault.value
    stats = {}
    for c in cols:
        x = df[c].values.astype(float).copy()
        std = float(np.std(x))
        if fault.ftype == 'noise':
            x[after] += rng.normal(0.0, sev * std, int(after.sum()))
        elif fault.ftype == 'bias':
            x[after] += sev * std
        elif fault.ftype == 'drift':
            ramp = np.clip((t - onset) / (t1 - onset), 0.0, 1.0)
            x += sev * std * ramp
        elif fault.ftype == 'dropout':
            end = onset + sev * (t1 - t0)
            hold = (t >= onset) & (t < end)
            if hold.any():
                x[hold] = x[np.argmax(hold)]
        else:
            raise ValueError(fault.ftype)
        df[c] = x
        stats[c] = {'std': std}

    def label(tt):
        r = np.ones(len(tt))
        a = tt >= onset
        if fault.ftype in ('noise', 'bias'):
            r[a] = 1.0 / (1.0 + sev)
        elif fault.ftype == 'drift':
            ramp = np.clip((tt - onset) / (t1 - onset), 0.0, 1.0)
            r = 1.0 / (1.0 + sev * ramp)
        elif fault.ftype == 'dropout':
            r[(tt >= onset) & (tt < onset + sev * (t1 - t0))] = 0.0
        return r

    r_imu = label(imu.t.values) if fault.sensor == 'imu' else np.ones(len(imu))
    r_odom = label(odom.t.values) if fault.sensor == 'odom' else np.ones(len(odom))
    manifest = dict(fault=fault.name, sensor=fault.sensor, type=fault.ftype, severity=fault.severity,
                    value=sev, onset=float(onset), session_t0=float(t0), session_t1=float(t1),
                    seed=fault.seed, columns=stats)
    return FaultResult(imu, odom, r_imu, r_odom, manifest)


def clean(sess):
    imu, odom, cmd, gt = sess
    return FaultResult(imu.copy(), odom.copy(), np.ones(len(imu)), np.ones(len(odom)),
                       dict(fault=None, sensor=None, type='clean'))
