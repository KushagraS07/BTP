"""lidar_faults.py -- L step (FINDINGS L): offline LiDAR scan faults.

A fault modifies the `ranges` strings of the scans from its onset (uniform in [20 %, 60 %] of the session, seed from
the fault name as learning.faults) to the session end; `freeze` holds the scan at onset for a window.
Values in physical units (robot.gazebo: 0.10-2.50 m, sigma 0.001 m):
    noise      Gaussian sd 0.01 / 0.03 / 0.10 m on valid returns
    loss       25 / 50 / 75 % of valid beams -> no return (inf)
    scale      ranges x (1 + e), e = 2 / 5 / 10 %
    spurious   5 / 10 / 20 % of beams -> uniform random range in [range_min, range_max]
    freeze     scan held for 5 / 15 / 35 % of the session
Returns beyond [range_min, range_max] become inf (no return), as the real driver reports.
"""
from dataclasses import dataclass
import zlib

import numpy as np

from learning.faults import ONSET_HI, ONSET_LO

LSEV = {
    'noise': {'low': 0.01, 'medium': 0.03, 'high': 0.10},
    'loss': {'low': 0.25, 'medium': 0.50, 'high': 0.75},
    'scale': {'low': 0.02, 'medium': 0.05, 'high': 0.10},
    'spurious': {'low': 0.05, 'medium': 0.10, 'high': 0.20},
    'freeze': {'short': 0.05, 'medium': 0.15, 'long': 0.35},
}
OVERLAP_L = [('noise', 'high'), ('scale', 'medium'), ('freeze', 'long')]
OVERLAP_O = [('odom', 'bias', 'medium'), ('odom', 'drift', 'medium'), ('imu', 'bias', 'medium')]


@dataclass
class LFault:
    session: str
    ftype: str
    severity: str

    @property
    def name(self):
        return f'{self.session}_lfault_{self.ftype}_{self.severity}'

    @property
    def value(self):
        return LSEV[self.ftype][self.severity]

    @property
    def seed(self):
        return zlib.crc32(self.name.encode())


def all_lfaults(session):
    return [LFault(session, f, v) for f in LSEV for v in LSEV[f]]


def inject_scans(scans, lf, t0, t1):
    """Returns (faulted copy of scans, manifest dict with onset / fault_end, per-scan bool 'active')."""
    rng = np.random.default_rng(lf.seed)
    onset = t0 + rng.uniform(ONSET_LO, ONSET_HI) * (t1 - t0)
    t = scans.t.values
    sev = lf.value
    end = min(onset + sev * (t1 - t0), t1) if lf.ftype == 'freeze' else t1
    active = (t >= onset) & (t < end) if lf.ftype == 'freeze' else t >= onset
    out = scans.copy()
    rr = list(out.ranges.values)
    held = None
    for k in np.where(active)[0]:
        r = np.array(rr[k].split(';'), float)
        lo, hi = float(out.range_min.iat[k]), float(out.range_max.iat[k])
        valid = np.isfinite(r) & (r >= lo) & (r <= hi)
        if lf.ftype == 'noise':
            r[valid] = r[valid] + rng.normal(0.0, sev, int(valid.sum()))
        elif lf.ftype == 'loss':
            drop = valid & (rng.random(len(r)) < sev)
            r[drop] = np.inf
        elif lf.ftype == 'scale':
            r[valid] = r[valid] * (1.0 + sev)
        elif lf.ftype == 'spurious':
            sp = rng.random(len(r)) < sev
            r[sp] = rng.uniform(lo, hi, int(sp.sum()))
        elif lf.ftype == 'freeze':
            if held is None:
                held = rr[k]
            rr[k] = held
            continue
        else:
            raise ValueError(lf.ftype)
        bad = np.isfinite(r) & ((r < lo) | (r > hi))
        r[bad] = np.inf
        rr[k] = ';'.join(repr(float(x)) for x in r)
    out['ranges'] = rr
    man = dict(name=lf.name, lsensor='lidar', ltype=lf.ftype, lseverity=lf.severity, lvalue=sev,
               lonset=float(onset), lend=float(end))
    return out, man, active
