#!/usr/bin/env python3
"""
final_analysis.py -- the pre-registered analysis for the FINAL run (PREREG_final.md).
Frozen (sha256 recorded) BEFORE any final session is recorded. Do not edit afterwards.

COMMANDS
  check-sessions S1 S2 ...   acceptance check for freshly recorded sessions (before mark-test).
                             Reads only the recording itself -- no EKF, no model, no result.
  analyse [--name final]     reads results/<name>_models.csv (written by
                             evaluate_heldout.py stage1 --name <name>) and prints every
                             pre-registered verdict; writes results/<name>_verdicts.csv,
                             results/<name>_effect_sizes.csv

Pre-registered settings are constants on purpose.
C1 counts wins per LOBO model and averages the 7 counts (the rule used in every earlier stage).
C2, P1, P2: unit = fault session; LSTM RMSE per fault session = mean over the 7 LOBO models (§15).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ws_paths as W  # noqa: E402

LOG = W.LOG_DIR
RESULTS = os.path.join(W.WS, 'results')

# ---------------- PRE-REGISTERED SETTINGS (PREREG_final.md) -- do not change -----------
MIN_DURATION = 45.0                # s
MAX_CLIPPED = 0                    # raw command samples clipped by preprocess_data.saturate_cmd_vel (F1)
N_MODELS = 7
B, SEED = 2000, 0                  # bootstrap resamples, seed
NOFAULT_TOL = 1.0                  # C3: LSTM no-fault mean RMSE at most 1% worse than fixed
IMU_NEGLIGIBLE = 1.0               # P1: |median IMU % change| <= 1%
# ----------------------------------------------------------------------------------------


def log(m=''):
    print(m, flush=True)


def tstamp(df):
    if 'timestamp' in df.columns:
        return df['timestamp'].values.astype(float)
    return df['time_sec'].values + df['time_nsec'].values * 1e-9


# ============================================================================ check-sessions
def n_clipped(cmd, label):
    """Raw command samples changed by the SAME actuator-saturation function the pipeline uses
    (preprocess_data.saturate_cmd_vel, fix F1). No limit is copied here."""
    import preprocess_data as PP
    if not getattr(PP, 'APPLY_CMD_SATURATION', False):
        sys.exit('ABORT: preprocess_data.APPLY_CMD_SATURATION is not True; the saturation check would be empty.')
    cols = ['linear_x', 'angular_z']
    sat = PP.saturate_cmd_vel(cmd.copy(), label)
    if len(sat) != len(cmd):
        sys.exit(f'ABORT {label}: saturate_cmd_vel changed the row count ({len(cmd)} -> {len(sat)})')
    a = cmd[cols].to_numpy(float)
    b = sat[cols].to_numpy(float)
    return int((~np.isclose(a, b, rtol=0, atol=1e-9)).any(axis=1).sum())


def cmd_check(sessions):
    reg = json.load(open(os.path.join(LOG, 'session_registry.json')))
    ok_all = True
    for s in sessions:
        problems = []
        for k in ('odom', 'imu'):
            if not os.path.exists(os.path.join(LOG, f'{k}_{s}.csv')):
                problems.append(f'missing {k}_{s}.csv')
        if not glob.glob(os.path.join(LOG, f'ground_truth*_{s}.csv')):
            problems.append('missing ground_truth file')
        # RAW command only: cmd_vel_filtered is actuator-limited (F1) and must not be used here
        cmd_p = os.path.join(LOG, f'cmd_vel_{s}.csv')
        if not os.path.exists(cmd_p):
            problems.append('missing raw cmd_vel file')
        if not problems:
            cmd = pd.read_csv(cmd_p)
            odom = pd.read_csv(os.path.join(LOG, f'odom_{s}.csv'))
            t = tstamp(odom)
            dur = float(t[-1] - t[0])
            vmax = float(cmd.linear_x.abs().max())
            wmax = float(cmd.angular_z.abs().max())
            n_clip = n_clipped(cmd, s)
            if dur < MIN_DURATION:
                problems.append(f'duration {dur:.1f} s < {MIN_DURATION} s')
            if n_clip > MAX_CLIPPED:
                problems.append(f'{n_clip} raw command samples clipped by the actuator model (must be {MAX_CLIPPED})')
            if s not in reg.get('good_sessions', []):
                problems.append('not in good_sessions (health check did not pass)')
            info = f'duration {dur:.1f} s, clipped {n_clip}/{len(cmd)}, max|v| {vmax:.4f}, max|w| {wmax:.4f}'
        else:
            info = ''
        ok = not problems
        ok_all &= ok
        log(f'  [{"ACCEPT" if ok else "REJECT"}] {s}  {info}' + ('' if ok else '  -> ' + '; '.join(problems)))
    log('\nAll accepted.' if ok_all else '\nAt least one REJECT: record a replacement (PREREG_final.md §2).')
    return ok_all


# ============================================================================ analyse
def ci_median(x, rng):
    x = np.asarray(x, float)
    m = np.median(rng.choice(x, size=(B, len(x)), replace=True), axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def ci_median_cluster(d, col, rng):
    """Sensitivity: resample BASELINE sessions (clusters), keep all their fault sessions."""
    groups = [g[col].values for _, g in d.groupby('baseline')]
    k = len(groups)
    meds = [np.median(np.concatenate([groups[i] for i in rng.integers(0, k, k)])) for _ in range(B)]
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def cmd_analyse(name):
    p = os.path.join(RESULTS, f'{name}_models.csv')
    df = pd.read_csv(p)
    if df.model.nunique() != N_MODELS:
        sys.exit(f'ABORT: expected {N_MODELS} models, found {df.model.nunique()}')
    df = df[df.fault_type != 'biasmid']
    rng = np.random.default_rng(SEED)
    base_ids = sorted(df.baseline.unique())
    log(f'input: {p}\nbaselines ({len(base_ids)}): {base_ids}')

    # per-model win counts
    def wins(g, col):
        return (g[col] < g.fixed).groupby(g.model).sum()

    odom, imu = df[df.sensor == 'odom'], df[df.sensor == 'imu']
    nofault = df[df.sensor.isna()]
    n_odom, n_imu = odom.session.nunique(), imu.session.nunique()

    # per fault session: LSTM averaged over the 7 models
    s = (df[df.sensor.notna()].groupby(['session', 'baseline', 'sensor', 'fault_type'])
         .agg(fixed=('fixed', 'first'), heur=('heuristic', 'first'), lstm=('lstm_linear', 'mean'))
         .reset_index())
    s['lstm_pct'] = 100 * (s.lstm - s.fixed) / s.fixed
    s['heur_pct'] = 100 * (s.heur - s.fixed) / s.fixed
    s['lvh_pct'] = 100 * (s.lstm - s.heur) / s.heur

    verdicts = []

    def verdict(code, text, value, rule, passed, kind):
        verdicts.append(dict(id=code, kind=kind, test=text, value=value, rule=rule,
                             result='PASS' if passed else 'FAIL'))

    # C1 odom wins
    lw, hw = wins(odom, 'lstm_linear'), int(wins(odom, 'heuristic').iloc[0])
    c1 = lw.mean() > n_odom / 2 and lw.mean() > hw
    verdict('C1', 'odom-fault wins vs fixed (wins counted per model, then averaged over the 7 models)',
            f'LSTM {lw.mean():.1f}+/-{lw.std():.1f}/{n_odom}; heuristic {hw}/{n_odom}',
            f'> {n_odom / 2:g} and > heuristic', c1, 'confirmatory')

    so = s[s.sensor == 'odom']
    # reported with C1 (not part of its rule): wins counted once per fault session on the 7-model mean RMSE
    c1_sess = (int((so.lstm < so.fixed).sum()), int((so.heur < so.fixed).sum()), len(so))
    # C2 odom effect size
    med = float(np.median(so.lstm_pct))
    lo, hi = ci_median(so.lstm_pct, rng)
    clo, chi = ci_median_cluster(so, 'lstm_pct', rng)
    verdict('C2', 'odom-fault median % change in RMSE, LSTM vs fixed',
            f'{med:+.1f}% [{lo:+.1f}, {hi:+.1f}] (cluster CI [{clo:+.1f}, {chi:+.1f}], reported)',
            'upper 95% CI < 0', hi < 0, 'confirmatory')

    # C3 no-fault
    mb = nofault.groupby('model')[['fixed', 'lstm_linear']].mean().mean()
    d3 = 100 * (mb.lstm_linear - mb.fixed) / mb.fixed
    verdict('C3', 'no-fault baselines: mean RMSE, LSTM vs fixed',
            f'fixed {mb.fixed:.4f}, LSTM {mb.lstm_linear:.4f} ({d3:+.2f}%)',
            f'<= +{NOFAULT_TOL:g}%', d3 <= NOFAULT_TOL, 'confirmatory')

    # P1 IMU negligible
    si = s[s.sensor == 'imu']
    mi = float(np.median(si.lstm_pct))
    ilo, ihi = ci_median(si.lstm_pct, rng)
    verdict('P1', 'IMU-fault median % change, LSTM vs fixed (prediction: negligible)',
            f'{mi:+.2f}% [{ilo:+.2f}, {ihi:+.2f}]', f'|median| <= {IMU_NEGLIGIBLE:g}%',
            abs(mi) <= IMU_NEGLIGIBLE, 'prediction')

    # P2 complementarity on odom noise
    sn = so[so.fault_type == 'noise']
    ml, mh = float(np.median(sn.lstm_pct)), float(np.median(sn.heur_pct))
    vlo, vhi = ci_median(sn.lvh_pct, rng)
    verdict('P2', 'odom noise: heuristic beats LSTM (prediction: complementarity)',
            f'LSTM {ml:+.1f}%, heuristic {mh:+.1f}%, LSTM vs heuristic median '
            f'{float(np.median(sn.lvh_pct)):+.1f}% [{vlo:+.1f}, {vhi:+.1f}]',
            'heuristic median < LSTM median and lower CI of LSTM-vs-heuristic > 0',
            mh < ml and vlo > 0, 'prediction')

    v = pd.DataFrame(verdicts)
    v['note'] = ''
    v.loc[v.id == 'C1', 'note'] = (f'reported, not part of the rule: per fault session on the 7-model mean RMSE '
                                   f'LSTM {c1_sess[0]}/{c1_sess[2]}, heuristic {c1_sess[1]}/{c1_sess[2]}')
    log('\n=== PRE-REGISTERED VERDICTS ===')
    for _, r in v.iterrows():
        log(f'  [{r.result}] {r.id} ({r.kind}) {r.test}\n         {r.value}   rule: {r.rule}')
        if r.note:
            log(f'         {r.note}')

    # descriptive tables
    log('\n=== Descriptive: by sensor x fault type ===')
    log(f'  {"":14s} {"n":>3s} {"LSTM wins":>10s} {"heur wins":>9s} {"LSTM % [95% CI]":>26s} {"heur %":>8s}')
    rows = []
    for (sen, ft), g in s.groupby(['sensor', 'fault_type']):
        gm = df[(df.sensor == sen) & (df.fault_type == ft)]
        m_, (a, b) = float(np.median(g.lstm_pct)), ci_median(g.lstm_pct, rng)
        hm = float(np.median(g.heur_pct))
        lwm, hwm = wins(gm, 'lstm_linear').mean(), int(wins(gm, 'heuristic').iloc[0])
        rows.append(dict(sensor=sen, fault=ft, n=len(g), lstm_wins=lwm, heur_wins=hwm,
                         lstm_pct=m_, lstm_lo=a, lstm_hi=b, heur_pct=hm))
        log(f'  {sen + " " + ft:14s} {len(g):>3d} {lwm:>6.1f}/{len(g):<3d} {hwm:>5d}/{len(g):<3d} '
            f'{m_:>+8.1f}% [{a:+.1f}, {b:+.1f}] {hm:>+7.1f}%')
    log(f'\n  IMU-fault wins (7-model mean): LSTM {wins(imu, "lstm_linear").mean():.1f}/{n_imu}, '
        f'heuristic {int(wins(imu, "heuristic").iloc[0])}/{n_imu}')
    os.makedirs(RESULTS, exist_ok=True)
    v.to_csv(os.path.join(RESULTS, f'{name}_verdicts.csv'), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, f'{name}_effect_sizes.csv'), index=False)
    log(f'\nSaved: results/{name}_verdicts.csv, results/{name}_effect_sizes.csv')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('check-sessions')
    p.add_argument('sessions', nargs='+')
    p = sub.add_parser('analyse')
    p.add_argument('--name', default='final')
    a = ap.parse_args()
    if a.cmd == 'check-sessions':
        sys.exit(0 if cmd_check(a.sessions) else 1)
    cmd_analyse(a.name)


if __name__ == '__main__':
    main()
