"""Tests for learning/faults.py (F7): synthetic session, no data files needed."""
import numpy as np
import pandas as pd

from learning.faults import ONSET_HI, ONSET_LO, SEVERITY, Fault, all_faults, clean, inject


def _sess(n=3000, hz=50.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / hz + 10.0
    imu = pd.DataFrame({'t': t, 'angular_vel_z': rng.normal(0, 0.3, n), 'linear_acc_x': rng.normal(0, 1, n)})
    odom = pd.DataFrame({'t': t, 'linear_x': np.abs(rng.normal(0.3, 0.2, n)), 'angular_z': rng.normal(0, 0.3, n)})
    return imu, odom, None, None


def test_count_and_names_unique():
    fs = all_faults('S')
    assert len(fs) == 26
    assert len({f.name for f in fs}) == 26


def test_deterministic_and_input_untouched():
    s = _sess()
    before = s[0].angular_vel_z.values.copy()
    f = Fault('S', 'imu', 'noise', 'high')
    a, b = inject(s, f), inject(s, f)
    assert np.array_equal(a.imu.angular_vel_z.values, b.imu.angular_vel_z.values)
    assert np.array_equal(s[0].angular_vel_z.values, before)


def test_onset_in_range_and_clean_before():
    s = _sess()
    for f in all_faults('S'):
        r = inject(s, f)
        m = r.manifest
        frac = (m['onset'] - m['session_t0']) / (m['session_t1'] - m['session_t0'])
        assert ONSET_LO <= frac <= ONSET_HI
        df, orig = (r.imu, s[0]) if f.sensor == 'imu' else (r.odom, s[1])
        col = 'angular_vel_z' if f.sensor == 'imu' else 'linear_x'
        pre = df.t.values < m['onset']
        assert np.array_equal(df[col].values[pre], orig[col].values[pre])
        lab = r.r_imu if f.sensor == 'imu' else r.r_odom
        assert np.all(lab[pre] == 1.0)
        other = r.r_odom if f.sensor == 'imu' else r.r_imu
        assert np.all(other == 1.0)


def test_label_values():
    s = _sess()
    r = inject(s, Fault('S', 'odom', 'bias', 'medium'))
    a = r.odom.t.values >= r.manifest['onset']
    assert np.allclose(r.r_odom[a], 1 / (1 + 3.0))
    d = inject(s, Fault('S', 'imu', 'drift', 'fast'))
    assert abs(d.r_imu[-1] - 1 / (1 + 6.0)) < 1e-6 and np.all(np.diff(d.r_imu) <= 1e-12)
    h = inject(s, Fault('S', 'imu', 'dropout', 'medium'))
    held = h.r_imu == 0.0
    assert held.any() and np.ptp(h.imu.angular_vel_z.values[held]) == 0.0
    frac = held.sum() / len(held)
    assert abs(frac - SEVERITY['dropout']['medium']) < 0.01


def test_bias_magnitude():
    s = _sess()
    r = inject(s, Fault('S', 'imu', 'bias', 'large'))
    a = r.imu.t.values >= r.manifest['onset']
    diff = r.imu.angular_vel_z.values[a] - s[0].angular_vel_z.values[a]
    assert np.allclose(diff, 6.0 * np.std(s[0].angular_vel_z.values))


def test_clean():
    c = clean(_sess())
    assert np.all(c.r_imu == 1) and np.all(c.r_odom == 1)
