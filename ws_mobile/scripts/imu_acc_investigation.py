#!/usr/bin/env python3
"""
imu_acc_investigation.py -- READ-ONLY. What are the large IMU linear-acceleration samples?

Event: a raw IMU sample with |a_x| > 10 or |a_y| > 10 m/s^2 (consecutive samples above the
threshold form one event). For every event:
  gt_acc     ground-truth body acceleration (forward for x, centripetal v*omega for y),
             from ground_truth_filtered, interpolated to the event time
  cmd_step   largest |change| of commanded v (x) / omega*v (y) within +/-0.2 s
  dt_anom    raw IMU time step at the event < 0 or > 3x the session median
  wall_dist  distance from the true position to the nearest wall segment
  az_jolt    |a_z - g| at the event
  n_samples  event length (1 = isolated single sample)
  in_filtered  whether imu_filtered also exceeds 10 m/s^2 within +/-0.1 s
Filtered-only events (no raw event within +/-0.1 s) are counted separately.

Classification (fixed before looking at the results; first matching rule wins):
  discontinuity   dt_anom
  real_motion     |gt_acc| > 2 and same sign as the IMU value, or cmd_step present
  contact         wall_dist < 0.35 m or az_jolt > 5
  isolated_glitch n_samples == 1
  unexplained     everything else
Writes results/imu_acc_events.csv and results/imu_acc_examples.png only.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ws_paths as W  # noqa: E402

THR, WIN, G = 10.0, 0.2, 9.81
WALLS = [('y', 7.0, -10.0, 10.0), ('y', -7.0, -10.0, 10.0), ('x', 10.0, -7.0, 7.0), ('x', -10.0, -7.0, 7.0),
         ('y', 3.0, -8.0, -3.0), ('y', -3.0, -8.0, -3.0), ('x', -3.0, -3.0, 3.0), ('x', -8.0, -3.0, 3.0),
         ('y', 2.0, 0.0, 9.0), ('y', -2.0, 0.0, 9.0)]   # lidar_fusion_v2.WALLS


def tstamp(df):
    if 'timestamp' in df.columns:
        return df['timestamp'].values.astype(float)
    return df['time_sec'].values + df['time_nsec'].values * 1e-9


def wall_dist(x, y):
    d = []
    for ax, c, lo, hi in WALLS:
        if ax == 'y':
            px = min(max(x, lo), hi); d.append(np.hypot(x - px, y - c))
        else:
            py = min(max(y, lo), hi); d.append(np.hypot(x - c, y - py))
    return min(d)


def gt_kinematics(s):
    g = pd.read_csv(os.path.join(W.LOG_DIR, f'ground_truth_filtered_{s}.csv'))
    t = tstamp(g)
    th = np.unwrap(2 * np.arctan2(g.gt_orient_z.values, g.gt_orient_w.values))
    dt = np.gradient(t)
    vx, vy = np.gradient(g.gt_x.values) / dt, np.gradient(g.gt_y.values) / dt
    v_fwd = vx * np.cos(th) + vy * np.sin(th)
    om = np.gradient(th) / dt
    a_fwd = np.gradient(v_fwd) / dt
    return t, a_fwd, v_fwd * om, g.gt_x.values, g.gt_y.values


def events(t, a):
    above = np.abs(a) > THR
    ev, i = [], 0
    while i < len(a):
        if above[i]:
            j = i
            while j + 1 < len(a) and above[j + 1]:
                j += 1
            k = i + int(np.argmax(np.abs(a[i:j + 1])))
            ev.append((k, j - i + 1))
            i = j + 1
        else:
            i += 1
    return ev


def main():
    reg = json.load(open(os.path.join(W.LOG_DIR, 'session_registry.json')))['good_sessions']
    ho = json.load(open(os.path.join(W.LOG_DIR, 'heldout_sessions.json')))['heldout_sessions']
    rows, filt_only, totals = [], 0, {}
    examples = []
    for s in sorted(set(reg) | set(ho)):
        imu = pd.read_csv(os.path.join(W.LOG_DIR, f'imu_{s}.csv'))
        ti = tstamp(imu)
        order = np.argsort(ti, kind='stable')
        dts = np.diff(ti)
        med_dt = np.median(dts[dts > 0])
        imf = pd.read_csv(os.path.join(W.LOG_DIR, f'imu_filtered_{s}.csv'))
        tf = tstamp(imf)
        cmd = pd.read_csv(os.path.join(W.LOG_DIR, f'cmd_vel_filtered_{s}.csv'))
        tc = tstamp(cmd)
        tg, a_fwd, a_lat, gx, gy = gt_kinematics(s)
        totals[s] = (len(imu), ti[-1] - ti[0])
        raw_times = []
        for axis, col, gt_a, cmdsig in (('x', 'linear_acc_x', a_fwd, cmd.linear_x.values),
                                        ('y', 'linear_acc_y', a_lat, cmd.linear_x.values * cmd.angular_z.values)):
            a = imu[col].values.astype(float)
            for k, n in events(ti, a):
                tk = ti[k]
                raw_times.append(tk)
                m = (tc >= tk - WIN) & (tc <= tk + WIN)
                cstep = float(np.ptp(cmdsig[m])) if m.sum() > 1 else 0.0
                ga = float(np.interp(tk, tg, gt_a))
                dt_prev = ti[k] - ti[k - 1] if k > 0 else med_dt
                dt_next = ti[k + 1] - ti[k] if k + 1 < len(ti) else med_dt
                dt_anom = min(dt_prev, dt_next) <= 0 or max(dt_prev, dt_next) > 3 * med_dt
                wd = wall_dist(float(np.interp(tk, tg, gx)), float(np.interp(tk, tg, gy)))
                az = abs(float(imu.linear_acc_z.values[k]) - G) if 'linear_acc_z' in imu.columns else 0.0
                mf = (tf >= tk - 0.1) & (tf <= tk + 0.1)
                in_f = bool((np.abs(imf[col].values[mf]) > THR).any()) if mf.any() else False
                if dt_anom:
                    cls = 'discontinuity'
                elif (abs(ga) > 2 and np.sign(ga) == np.sign(a[k])) or cstep > 0.1:
                    cls = 'real_motion'
                elif wd < 0.35 or az > 5:
                    cls = 'contact'
                elif n == 1:
                    cls = 'isolated_glitch'
                else:
                    cls = 'unexplained'
                rows.append(dict(session=s, axis=axis, t=tk - ti[0], imu_value=a[k], n_samples=n, gt_acc=ga,
                                 cmd_step=cstep, dt_anom=dt_anom, wall_dist=wd, az_jolt=az,
                                 in_filtered=in_f, cls=cls))
                if len(examples) < 4 and cls in ('isolated_glitch', 'contact', 'unexplained', 'real_motion') \
                        and all(e[3] != cls for e in examples):
                    examples.append((s, tk, axis, cls, ti, a, tg, gt_a))
        for axis, col in (('x', 'linear_acc_x'), ('y', 'linear_acc_y')):
            for k, _ in events(tf, imf[col].values.astype(float)):
                if not raw_times or np.min(np.abs(np.array(raw_times) - tf[k])) > 0.1:
                    filt_only += 1

    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(W.WS, 'results'), exist_ok=True)
    df.to_csv(os.path.join(W.WS, 'results', 'imu_acc_events.csv'), index=False)
    dur = sum(v[1] for v in totals.values())
    nsamp = sum(v[0] for v in totals.values())
    print(f'Sessions: {len(totals)}, {dur / 60:.1f} min, {nsamp} raw IMU samples')
    print(f'Raw events (|a| > {THR} m/s^2): {len(df)}  ({100 * df.n_samples.sum() / nsamp:.3f}% of samples)'
          if len(df) else 'Raw events: 0')
    if len(df):
        print('\n=== Classification ===')
        c = df.groupby('cls').agg(events=('cls', 'size'), median_abs_value=('imu_value', lambda v: np.median(np.abs(v))),
                                  max_abs_value=('imu_value', lambda v: np.max(np.abs(v))),
                                  single_sample=('n_samples', lambda v: int((v == 1).sum())),
                                  also_in_filtered=('in_filtered', 'sum'))
        c['share_%'] = (100 * c.events / len(df)).round(1)
        print(c.round(2).to_string())
        print('\nper session:', df.groupby('session').size().to_dict())
        print('per axis   :', df.groupby('axis').size().to_dict())
        print(f'median |gt_acc| at events: {df.gt_acc.abs().median():.2f} m/s^2 ;'
              f' median event length {df.n_samples.median():.0f} sample(s)')
    print(f'\nFiltered-only events (not in raw within +/-0.1 s; preprocessing artifacts): {filt_only}')

    if examples:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(len(examples), 1, figsize=(10, 2.6 * len(examples)))
        axs = np.atleast_1d(axs)
        for ax, (s, tk, axis, cls, ti, a, tg, gt_a) in zip(axs, examples):
            m, mg = (ti > tk - 1) & (ti < tk + 1), (tg > tk - 1) & (tg < tk + 1)
            ax.plot(ti[m] - tk, a[m], lw=1, label=f'IMU a_{axis} (raw)')
            ax.plot(tg[mg] - tk, gt_a[mg], lw=1.5, label='ground-truth acceleration')
            ax.axhline(THR, ls=':', c='grey'); ax.axhline(-THR, ls=':', c='grey')
            ax.set_title(f'{s}  class: {cls}', fontsize=9); ax.set_ylabel('m/s²'); ax.legend(fontsize=7)
        axs[-1].set_xlabel('time relative to event (s)')
        fig.tight_layout()
        fig.savefig(os.path.join(W.WS, 'results', 'imu_acc_examples.png'), dpi=130)
    print('Saved: results/imu_acc_events.csv, results/imu_acc_examples.png')


if __name__ == '__main__':
    main()
