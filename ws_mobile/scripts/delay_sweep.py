#!/usr/bin/env python3
"""
delay_sweep.py -- item 4.3. Descriptive only (no pass/fail; no model trained or changed).

How does the FIXED EKF's error respond to sensor delay, from 0.05 s to 5 s?
Explains the open observation that small/medium delays sometimes LOWER the error
(4 of 6 held-out cases, FINDINGS §8.2).

  sessions : the 3 held-out sessions only (unsaturated; excluded from all training globs)
  delays   : 0.05 0.1 0.25 0.5 1 1.5 2 3 5 s, on IMU and on odometry (54 sweep sessions)
  mechanism: fault_injection.make_delay_fault_session_raw (unchanged raw-time shift +
             full re-preprocessing), then ekf_fusion.py (unchanged)
  reported (fixed now): ratio RMSE/baseline per delay; cases with ratio < 1 at <= 0.5 s;
             first delay from which ratio >= 1.05 for all larger delays
  check    : 0.1 / 0.5 / 2.0 s must reproduce the existing small / medium / large RMSE
  cleanup  : every sweep file is moved from sensor_logs/ to results/delay_sweep_files/;
             session_registry.json must be unchanged

Outputs: results/delay_sweep.csv, results/delay_sweep.png/.pdf
"""
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ws_paths as W  # noqa: E402

LOG, S = W.LOG_DIR, W.SCRIPTS_DIR
RES = os.path.join(W.WS, 'results')
MOVE_TO = os.path.join(RES, 'delay_sweep_files')
DELAYS = [0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
EXISTING = {0.1: 'small', 0.5: 'medium', 2.0: 'large'}
TAG = 'sw'


def key(d):
    return f'{TAG}{d:g}'.replace('.', 'p')          # 0.25 -> sw0p25


def rmse(s):
    m = pd.read_csv(os.path.join(LOG, f'ekf_metrics_{s}.csv'))
    return float(m.loc[m.metric == 'RMSE', 'value_m'].iloc[0])


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def main():
    os.chdir(S)
    sys.path.insert(0, S)
    import fault_injection as FI
    held = json.load(open(os.path.join(LOG, 'heldout_sessions.json')))['heldout_sessions']
    reg_path = os.path.join(LOG, 'session_registry.json')
    reg_sha = sha(reg_path)
    for d in DELAYS:
        FI.DELAY_SEVERITY[key(d)] = d

    rows = []
    for b in held:
        base = rmse(b)
        for sensor in ('imu', 'odom'):
            for d in DELAYS:
                s = FI.make_delay_fault_session_raw(b, sensor, key(d))
                p = subprocess.run([sys.executable, os.path.join(S, 'ekf_fusion.py'), s], cwd=S,
                                   capture_output=True, text=True, timeout=600)
                if p.returncode != 0:
                    sys.exit(f'ABORT: ekf_fusion failed on {s}\n{p.stderr[-600:]}')
                r = rmse(s)
                ref = rmse(f'{b}_fault_{sensor}_delay_{EXISTING[d]}') if d in EXISTING else np.nan
                rows.append(dict(session=b, sensor=sensor, delay_s=d, rmse=r, base=base, ratio=r / base,
                                 existing_rmse=ref))
                print(f'  {b} {sensor:4s} {d:>4g} s  RMSE {r:.4f}  ratio {r / base:.3f}', flush=True)

    # cleanup: move every sweep file out of sensor_logs
    os.makedirs(MOVE_TO, exist_ok=True)
    moved = 0
    for f in glob.glob(os.path.join(LOG, f'*_delay_{TAG}*')):
        shutil.move(f, os.path.join(MOVE_TO, os.path.basename(f)))
        moved += 1
    left = glob.glob(os.path.join(LOG, f'*_delay_{TAG}*'))
    print(f'\nMoved {moved} sweep files to results/delay_sweep_files/; left in sensor_logs: {len(left)}')
    print(f'session_registry.json unchanged: {sha(reg_path) == reg_sha}')

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, 'delay_sweep.csv'), index=False)

    chk = df.dropna(subset=['existing_rmse'])
    worst = float((chk.rmse - chk.existing_rmse).abs().max())
    print(f'Reproduction check (0.1/0.5/2.0 s vs existing small/medium/large): max |diff| = {worst:.2e} m '
          f'-> {"OK" if worst < 1e-6 else "MISMATCH"}')

    print('\n=== ratio RMSE / baseline (fixed EKF) ===')
    tab = df.pivot_table(index=['session', 'sensor'], columns='delay_s', values='ratio').round(3)
    print(tab.to_string())
    print('\n=== summary ===')
    for (b, sensor), g in df.groupby(['session', 'sensor']):
        g = g.sort_values('delay_s')
        below = g[(g.delay_s <= 0.5) & (g.ratio < 1)].delay_s.tolist()
        onset = next((d for d in g.delay_s if (g[g.delay_s >= d].ratio >= 1.05).all()), None)
        print(f'  {b} {sensor:4s}: lower-than-baseline at {below if below else "none"} ; '
              f'sustained >= 5% degradation from {onset if onset is not None else "never (<= 5 s)"} s ; '
              f'min ratio {g.ratio.min():.3f} at {g.loc[g.ratio.idxmin(), "delay_s"]:g} s')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    col = {'imu': '#2a78d6', 'odom': '#eb6834'}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, b in zip(axes, held):
        for sensor in ('imu', 'odom'):
            g = df[(df.session == b) & (df.sensor == sensor)].sort_values('delay_s')
            ax.plot(g.delay_s, g.ratio, 'o-', color=col[sensor], lw=2, ms=6, mec='white', label=sensor.upper())
        ax.axhline(1, color='#6b7280', lw=1)
        ax.axhline(1.05, color='#9ca3af', lw=0.8, ls=':')
        ax.set_xscale('log')
        ax.set_xticks(DELAYS)
        ax.set_xticklabels([f'{d:g}' for d in DELAYS], fontsize=8)
        ax.set_title(b, fontsize=10, loc='left')
        ax.set_xlabel('delay (s, log scale)', fontsize=9)
        ax.grid(color='#e5e7eb', lw=0.8)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel('RMSE / no-delay RMSE (fixed EKF)', fontsize=9)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(RES, f'delay_sweep.{ext}'), dpi=180 if ext == 'png' else None)
    print('Saved: results/delay_sweep.csv, results/delay_sweep.png/.pdf')


if __name__ == '__main__':
    main()
