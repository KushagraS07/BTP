#!/usr/bin/env python3
"""
evaluate_heldout.py -- ONE entry point for every held-out evaluation in this project.
Merges the former fix_ws scripts heldout_stage1.py, heldout_stage2.py, rnn_lobo_test.py
and the A.4 check of heldout_stage3.py (those originals are archived in fix_ws/archive/).

Nothing is hard-coded:
  * workspace root   -> ws_paths.py ($WS_MOBILE_ROOT, else found automatically)
  * training sessions -> sensor_logs/session_registry.json (good_sessions)
  * test sessions     -> sensor_logs/heldout_sessions.json (or --sessions)
  * results           -> <root>/results/<name>_*.csv  (never overwrites fix_ws records)
Pre-registered thresholds are constants on purpose (PREREG_heldout.md / PREREG_rnn.md).

COMMANDS
  mark-test SESSION [...]   move freshly recorded sessions out of training and into the
                            test list (the logger's health_check adds every new recording
                            to good_sessions; test sessions must never stay there)
  stage1 [--from N] [--name NAME] [--sessions ...]
                            main held-out evaluation (= Stage 1 procedure):
                            1 faults  2 EKF  3 labels  4 features  5 H0/H1  6 7 LSTM models
                            (linear + D1-gate, descriptive) + report
  lidar [--name NAME]       A.2 roughness-proxy re-test on the test sessions (Stage 2 / D5)
  rnn   [--name NAME]       RNN generalisation test (PREREG_rnn.md): 7-fold LOBO + test set
  delay-check               D2 delay-label sanity checks (Addendum A.4), read-only
  verify NEW.csv REF.csv    prove two model-result tables are identical (merge check)

Usage (from anywhere):
  python3 <root>/scripts/evaluate_heldout.py stage1 --name final 2>&1 | tee <root>/results/final_run.log
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import ws_paths as W

S, LOG, ROOT = W.SCRIPTS_DIR, W.LOG_DIR, W.WS
RESULTS = os.path.join(ROOT, 'results')
sys.path.insert(0, S)
os.chdir(S)
PY = sys.executable

# ---------------- PRE-REGISTERED SETTINGS -- do not change ----------------------------
# PREREG_heldout.md (Stage 1, D1, D3, Stage 2)
GATE_ON, GATE_OFF, GATE_N = 0.5, 0.7, 25          # D1 hysteresis (rejected; reported only)
BIASMID_START = 0.4                                # D3 onset
H1_RATIO, H1_MIN_PASS, H1_DELAY_MIN = 1.05, 22, 5  # H1 (scaled below for other n)
DETECT_WINDOW = 100                                # D3: first 2 s at 50 Hz
DIAG_ALPHA, DEPLOYED_ALPHA, A2_PASS_PCT = 0.70, 0.98, 5.0
# Addendum A
D2_FAULTY = 0.5
# PREREG_rnn.md
RNN_SEED, RNN_MIN_FOLD_WINS, RNN_MIN_IMPROVE, RNN_HR_V3 = 42, 5, 5.0, 10.0
RNN_N_BLOCKS = 20
# ----------------------------------------------------------------------------------------

MONO = {'noise': ['low', 'medium', 'high', 'extreme'], 'bias': ['small', 'medium', 'large'],
        'drift': ['slow', 'medium', 'fast'], 'dropout': ['short', 'medium', 'long']}
SEV3 = ['small', 'medium', 'large']
LIDAR_SEVS = ['mild', 'moderate', 'severe']
HELDOUT_JSON = os.path.join(LOG, 'heldout_sessions.json')


def log(m=''):
    print(m, flush=True)


def train_sessions():
    from session_registry import get_good_sessions
    return sorted(get_good_sessions())


def test_sessions(override=None):
    if override:
        return list(override)
    return json.load(open(HELDOUT_JSON))['heldout_sessions']


def guard(tests):
    """Test sessions must never be training sessions."""
    from session_registry import is_heldout
    tr = set(train_sessions())
    bad = [s for s in tests if s in tr]
    if bad:
        sys.exit(f'ABORT: test session(s) also in good_sessions (training): {bad}. Run mark-test.')
    nh = [s for s in tests if not is_heldout(s)]
    if nh:
        sys.exit(f'ABORT: not listed in sensor_logs/heldout_sessions.json: {nh}. Run mark-test.')


def out(name, suffix):
    os.makedirs(RESULTS, exist_ok=True)
    return os.path.join(RESULTS, f'{name}_{suffix}')


def fs(b, sensor, ft, sev):
    return f'{b}_fault_{sensor}_{ft}_{sev}'


# ============================================================================ mark-test
def cmd_mark_test(sessions):
    reg_path = os.path.join(LOG, 'session_registry.json')
    os.makedirs(RESULTS, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    for p in (reg_path, HELDOUT_JSON):
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(RESULTS, f'{os.path.basename(p)}.bak_{stamp}'))
    missing = [s for s in sessions if not os.path.exists(os.path.join(LOG, f'odom_{s}.csv'))]
    if missing:
        sys.exit(f'ABORT: no recording found for {missing} (sensor_logs/odom_<session>.csv). Nothing changed.')
    reg = json.load(open(reg_path))
    reg['good_sessions'] = [s for s in reg.get('good_sessions', []) if s not in sessions]
    json.dump(reg, open(reg_path, 'w'), indent=2)
    ho = json.load(open(HELDOUT_JSON)) if os.path.exists(HELDOUT_JSON) else {'heldout_sessions': []}
    for s in sessions:
        if s not in ho['heldout_sessions']:
            ho['heldout_sessions'].append(s)
    json.dump(ho, open(HELDOUT_JSON, 'w'), indent=2)
    log(f'Marked as TEST (removed from training): {sessions}')
    log(f'Training sessions now: {len(reg["good_sessions"])}; test sessions now: {ho["heldout_sessions"]}')
    log(f'Backups of both json files: results/*.bak_{stamp}')


# ============================================================================ stage1
def fault_sessions(tests):
    o = []
    for b in tests:
        for sensor in ('imu', 'odom'):
            for ft, sevs in MONO.items():
                o += [fs(b, sensor, ft, v) for v in sevs]
            o += [fs(b, sensor, 'delay', v) for v in SEV3]
        o += [fs(b, 'imu', 'biasmid', v) for v in SEV3]
    return o


def rmse_metric(s):
    m = pd.read_csv(os.path.join(LOG, f'ekf_metrics_{s}.csv'))
    return float(m.loc[m.metric == 'RMSE', 'value_m'].iloc[0])


def s1_faults(tests):
    import fault_injection as FI
    created = FI.generate_all_faults(baseline_sessions=list(tests))
    if len(created) != 52 * len(tests):
        sys.exit(f'ABORT: generate_all_faults created {len(created)}, expected {52 * len(tests)}')
    FI.FAULT_FNS['biasmid'] = lambda df, cols, sev, sev_map, rng: FI.inject_bias(
        df, cols, sev_map[sev], start_frac=BIASMID_START)
    FI.FAULT_SEVERITY_MAPS['biasmid'] = FI.BIAS_SEVERITY
    for b in tests:
        for v in SEV3:
            FI.make_fault_session(b, 'imu', 'biasmid', v)
    log(f'step1 OK: {52 * len(tests)} standard + {3 * len(tests)} biasmid fault sessions')


def s2_ekf(tests):
    todo = fault_sessions(tests)
    for i, s in enumerate(todo, 1):
        p = subprocess.run([PY, os.path.join(S, 'ekf_fusion.py'), s], cwd=S,
                           capture_output=True, text=True, timeout=300)
        if p.returncode != 0 or not os.path.exists(os.path.join(LOG, f'ekf_metrics_{s}.csv')):
            sys.exit(f'ABORT step2 {s}: exit {p.returncode}\n{p.stderr[-800:]}')
        if i % 15 == 0:
            log(f'  ekf {i}/{len(todo)}')
    log(f'step2 OK: EKF on {len(todo)} fault sessions')


def s3_labels(tests):
    import generate_reliability_labels as G
    if isinstance(G.BASELINE_SESSIONS, list):
        G.BASELINE_SESSIONS.extend(b for b in tests if b not in G.BASELINE_SESSIONS)
    for b in tests:
        G.process_baseline_session(b)
    for s in fault_sessions(tests):
        if '_biasmid_' not in s:
            G.process_fault_session(Path(LOG) / f'fault_meta_{s}.json')
    for b in tests:  # D3: bias labels with r_imu = 1 before onset
        for v in SEV3:
            src = pd.read_csv(os.path.join(LOG, f'reliability_labels_{fs(b, "imu", "bias", v)}.csv'))
            onset = int(len(src) * BIASMID_START)
            src.loc[src.index[:onset], 'r_imu'] = 1.0
            src.to_csv(os.path.join(LOG, f'reliability_labels_{fs(b, "imu", "biasmid", v)}.csv'), index=False)
    missing = [s for s in list(tests) + fault_sessions(tests)
               if not os.path.exists(os.path.join(LOG, f'reliability_labels_{s}.csv'))]
    if missing:
        sys.exit(f'ABORT step3: {len(missing)} label files missing, e.g. {missing[:3]}')
    log('step3 OK: labels')


def s4_features(tests):
    import build_training_dataset as BT
    todo = list(tests) + fault_sessions(tests)
    for i, s in enumerate(todo, 1):
        BT.build_session_features(s)
        if i % 20 == 0:
            log(f'  features {i}/{len(todo)}')
    log(f'step4 OK: features for {len(todo)} sessions')


def s5_h0_h1(tests, name):
    rows, deg_pass, deg_total, delay_pass = [], 0, 0, 0
    log('\n=== H0: clean EKF RMSE ===')
    for b in tests:
        log(f'  {b}: {rmse_metric(b):.4f} m')
    log('\n=== H1: fault validation (worst severity >= 1.05x baseline) ===')
    for b in tests:
        base = rmse_metric(b)
        for sensor in ('imu', 'odom'):
            for ft, sevs in MONO.items():
                vals = [rmse_metric(fs(b, sensor, ft, v)) for v in sevs]
                ratio = vals[-1] / base
                ok = ratio >= H1_RATIO
                deg_pass += ok
                deg_total += 1
                rows.append(dict(baseline=b, sensor=sensor, fault=ft, ratio=ratio, passed=ok))
                log(f'  [{"PASS" if ok else "FAIL"}] {b}/{sensor}/{ft}: ' +
                    ', '.join(f'{v}={x:.3f}' for v, x in zip(sevs, vals)) + f'  ratio={ratio:.3f}')
            d = [rmse_metric(fs(b, sensor, 'delay', v)) for v in SEV3]
            ok = d[-1] > base
            delay_pass += ok
            rows.append(dict(baseline=b, sensor=sensor, fault='delay', ratio=d[-1] / base, passed=ok))
            log(f'  [{"UP" if ok else "NOT UP"}] {b}/{sensor}/delay: base={base:.3f} '
                f'small={d[0]:.3f} medium={d[1]:.3f} large={d[2]:.3f}')
    pd.DataFrame(rows).to_csv(out(name, 'h1.csv'), index=False)
    # pre-registered for 3 sessions: >= 22/24 and >= 5/6 -> same proportions for other n
    need_deg = int(np.ceil(H1_MIN_PASS / 24 * deg_total))
    need_delay = int(np.ceil(H1_DELAY_MIN / 6 * 2 * len(tests)))
    h1 = deg_pass >= need_deg and delay_pass >= need_delay
    log(f'\nH1: degradation {deg_pass}/{deg_total} (need >= {need_deg}), '
        f'delay up {delay_pass}/{2 * len(tests)} (need >= {need_delay}) -> {"PASS" if h1 else "FAIL"}')
    return h1


class Linear:
    def __init__(self, model):
        self.model, self.r, self.n = model, [], 0

    def __call__(self, x):
        o = self.model(x)
        self.r.append(float(o[0, 0]))
        self.n += 1
        return o

    def eval(self):
        return self


class Hysteresis:
    """D1 (rejected in Stage 1; descriptive only): gate r_imu'=0 when r_imu < GATE_ON;
    release after r_imu > GATE_OFF for GATE_N consecutive steps; starts released."""

    def __init__(self, model):
        self.model, self.gated, self.cnt, self.n = model, False, 0, 0

    def __call__(self, x):
        o = self.model(x).clone()
        r = float(o[0, 0])
        if not self.gated:
            if r < GATE_ON:
                self.gated, self.cnt = True, 0
        else:
            self.cnt = self.cnt + 1 if r > GATE_OFF else 0
            if self.cnt >= GATE_N:
                self.gated, self.cnt = False, 0
        o[0, 0] = 0.0 if self.gated else 1.0
        self.n += 1
        return o

    def eval(self):
        return self


def s6_models(tests, name):
    import torch
    import lstm_adaptive_ekf as L
    from build_training_dataset import parse_fault_info
    sessions = list(tests) + fault_sessions(tests)
    rows, det = [], []
    for mi, mb in enumerate(train_sessions(), 1):
        model, mean, std, feats, win = L.load_model(mb)
        log(f'\n--- model {mi}/7 (LOBO fold {mb}) ---')
        for s in sessions:
            b = s.split('_fault_')[0]
            n_lab = len(pd.read_csv(os.path.join(LOG, f'reliability_labels_{s}.csv')))
            lin, hyst = Linear(model), Hysteresis(model)
            with torch.no_grad():
                fixed, heur, r_lin = L.evaluate_session(s, b, lin, mean, std, feats, win)
                _, _, r_hys = L.evaluate_session(s, b, hyst, mean, std, feats, win)
            if lin.n != n_lab or hyst.n != n_lab:
                sys.exit(f'ABORT {s}: model called {lin.n}/{hyst.n}x, labels {n_lab}')
            sensor, ft, sev = (None, None, None) if s == b else parse_fault_info(s, b)
            rows.append(dict(model=mb, session=s, baseline=b, sensor=sensor, fault_type=ft,
                             severity=sev, fixed=fixed, heuristic=heur, lstm_linear=r_lin, lstm_d1=r_hys))
            if ft == 'biasmid':
                onset = int(n_lab * BIASMID_START)
                w = np.array(lin.r[onset:onset + DETECT_WINDOW])
                det.append(dict(model=mb, session=s, detect_2s=float((w < GATE_ON).mean()),
                                detect_after=float((np.array(lin.r[onset:]) < GATE_ON).mean()),
                                false_before=float((np.array(lin.r[:onset]) < GATE_ON).mean())))
        log(f'  done {len(sessions)} sessions')
    df, dt = pd.DataFrame(rows), pd.DataFrame(det)
    df.to_csv(out(name, 'models.csv'), index=False)
    dt.to_csv(out(name, 'd3_detect.csv'), index=False)
    return df, dt


def s_report(df, det):
    def wins(g, col):
        return (g[col] < g['fixed']).groupby(g['model']).sum()

    std_imu = df[(df.sensor == 'imu') & (df.fault_type != 'biasmid')]
    odom = df[df.sensor == 'odom']
    base = df[df.sensor.isna()]
    n_imu, n_odom = std_imu.session.nunique(), odom.session.nunique()
    log('\n=== Wins vs fixed (mean +/- std over the 7 models) ===')
    log(f'{"":10s} {"heuristic":>12s} {"lstm_linear":>16s} {"D1 (ref only)":>16s}')
    for nm, g, n in (('IMU', std_imu, n_imu), ('odom', odom, n_odom)):
        h = wins(g, 'heuristic').iloc[0]
        lw, dw = wins(g, 'lstm_linear'), wins(g, 'lstm_d1')
        log(f'{nm:10s} {h:>7d}/{n:<4d} {lw.mean():>9.1f}+/-{lw.std():<4.1f}/{n} '
            f'{dw.mean():>7.1f}+/-{dw.std():<4.1f}/{n}')
    for nm, g in (('IMU', std_imu), ('odom', odom)):
        log(f'\n  {nm} by fault type (7-model mean wins vs fixed):')
        for ft, gg in g.groupby('fault_type'):
            n = gg.session.nunique()
            log(f'    {ft:8s} heuristic {wins(gg, "heuristic").iloc[0]}/{n}  '
                f'linear {wins(gg, "lstm_linear").mean():.1f}/{n}  D1 {wins(gg, "lstm_d1").mean():.1f}/{n}')
    m = std_imu.groupby('model')[['fixed', 'lstm_linear', 'lstm_d1']].mean().mean()
    mo = odom.groupby('model')[['fixed', 'lstm_linear']].mean().mean()
    mb = base.groupby('model')[['fixed', 'lstm_linear', 'lstm_d1']].mean().mean()
    log(f'\n  IMU-fault mean RMSE:  fixed={m.fixed:.3f} linear={m.lstm_linear:.3f} '
        f'({100 * (m.lstm_linear - m.fixed) / m.fixed:+.1f}%)  D1={m.lstm_d1:.3f}')
    log(f'  odom-fault mean RMSE: fixed={mo.fixed:.3f} linear={mo.lstm_linear:.3f} '
        f'({100 * (mo.lstm_linear - mo.fixed) / mo.fixed:+.1f}%)')
    log(f'  no-fault baselines mean RMSE: fixed={mb.fixed:.4f} linear={mb.lstm_linear:.4f}')
    h_odom, l_odom = wins(odom, 'heuristic').iloc[0], wins(odom, 'lstm_linear').mean()
    log(f'\nOdom replication (Stage 1 H2 rule): LSTM {l_odom:.1f}/{n_odom} '
        f'(need > {n_odom / 2:.0f} and > heuristic {h_odom}) -> '
        f'{"PASS" if l_odom > n_odom / 2 and l_odom > h_odom else "FAIL"}')
    bm = df[df.fault_type == 'biasmid']
    if len(bm):
        mm = bm.groupby('model')[['fixed', 'lstm_linear', 'lstm_d1']].mean().mean()
        log(f'\n=== D3 mid-session IMU bias (reported) ===\n  wins: linear {wins(bm, "lstm_linear").mean():.1f}, '
            f'D1 {wins(bm, "lstm_d1").mean():.1f} /{bm.session.nunique()}; mean RMSE fixed={mm.fixed:.3f} '
            f'linear={mm.lstm_linear:.3f} D1={mm.lstm_d1:.3f}')
    if len(det):
        log(f'  r_imu < {GATE_ON}: first 2 s after onset {det.detect_2s.mean():.1%}, '
            f'whole post-onset {det.detect_after.mean():.1%}, pre-onset (false) {det.false_before.mean():.1%}')


def cmd_stage1(a):
    tests = test_sessions(a.sessions)
    guard(tests)
    log(f'test sessions: {tests}\ntraining models: {train_sessions()}\nresults: results/{a.name}_*')
    steps = {1: s1_faults, 2: s2_ekf, 3: s3_labels, 4: s4_features}
    for k in range(a.start, 5):
        steps[k](tests)
    s5_h0_h1(tests, a.name)
    df, det = s6_models(tests, a.name)
    s_report(df, det)
    log(f'\nSaved: results/{a.name}_h1.csv, {a.name}_models.csv, {a.name}_d3_detect.csv')


# ============================================================================ lidar (A.2)
def cmd_lidar(a):
    import torch
    import rnn_fusion_v2 as R
    import run_a2_sanity_check as A2
    tests = test_sessions(a.sessions)
    guard(tests)
    tr = train_sessions()
    model = R.DenoisingRNN(input_size=3, output_size=3)
    model.load_state_dict(torch.load(os.path.join(LOG, 'rnn_model_denoising.pth'), map_location='cpu'))
    model.eval()
    mean, std = np.load(os.path.join(LOG, 'rnn_norm_stats.npy'))
    res = np.load(os.path.join(LOG, 'rnn_residual_stats_per_session.npy'), allow_pickle=True).item()
    if sorted(res) != tr:
        sys.exit(f'ABORT: RNN residual stats {sorted(res)} != training sessions {tr}')
    res_mean = np.mean([res[s]['mean'] for s in tr], axis=0)
    res_std = np.mean([res[s]['std'] for s in tr], axis=0)
    rnn_rows = []
    for b in tests:
        ekf_df = R.load_ekf_output(b)
        ekf = ekf_df[['ekf_x', 'ekf_y', 'ekf_theta']].values.astype(np.float32)
        t = ekf_df['timestamp'].values.astype(float)
        gt, mask = R.load_ground_truth_aligned(t, b)
        ekf, t = ekf[mask], t[mask]
        pred, w = R.run_rnn_inference(model, mean, std, res_mean, res_std, ekf)
        n = len(pred)
        sm = R.smooth_trajectory(pred)
        for nm, arr in (('rnn_output', pred), ('rnn_smoothed_output', sm)):
            pd.DataFrame({'timestamp': t[w:w + n], 'rnn_x': arr[:, 0], 'rnn_y': arr[:, 1],
                          'rnn_theta': arr[:, 2]}).to_csv(os.path.join(LOG, f'{nm}_{b}.csv'), index=False)
        for v in LIDAR_SEVS:
            shutil.copy2(os.path.join(LOG, f'rnn_smoothed_output_{b}.csv'),
                         os.path.join(LOG, f'rnn_smoothed_output_{b}_fault_lidar_corruption_{v}.csv'))
        rnn_rows.append(dict(baseline=b, ekf_rmse=R.compute_rmse(ekf[w:w + n], gt[w:w + n])[0],
                             rnn_rmse=R.compute_rmse(sm, gt[w:w + n])[0]))
    rows = []
    for b in tests:
        for v in LIDAR_SEVS:
            s = f'{b}_fault_lidar_corruption_{v}'
            bd = A2.run_at_alpha(s, DEPLOYED_ALPHA, None, 'baseline/deployed')
            pdp = A2.run_at_alpha(s, DEPLOYED_ALPHA, A2.roughness_confidence_fn, 'proxy/deployed')
            bg = A2.run_at_alpha(s, DIAG_ALPHA, None, 'baseline/diagnostic')
            pg = A2.run_at_alpha(s, DIAG_ALPHA, A2.roughness_confidence_fn, 'proxy/diagnostic')
            rows.append(dict(session=s, severity=v, imp_diag_pct=100 * (bg - pg) / bg,
                             imp_deployed_pct=100 * (bd - pdp) / bd, base_diag=bg, proxy_diag=pg))
    df = pd.DataFrame(rows)
    df.to_csv(out(a.name, 'a2.csv'), index=False)
    pd.DataFrame(rnn_rows).to_csv(out(a.name, 'rnn.csv'), index=False)
    log('\n========== A.2 REPORT ==========')
    for r in rnn_rows:
        log(f'  RNN (descriptive) {r["baseline"]}: EKF {r["ekf_rmse"]:.4f} m  RNN {r["rnn_rmse"]:.4f} m')
    for v in LIDAR_SEVS:
        log(f'  {v:<9} {df[df.severity == v].imp_diag_pct.mean():+.2f}%')
    m = df.imp_diag_pct.mean()
    log(f'Mean improvement at ALPHA={DIAG_ALPHA}: {m:+.2f}%  (deployed {DEPLOYED_ALPHA}: {df.imp_deployed_pct.mean():+.2f}%)')
    log(f'Verdict (>= {A2_PASS_PCT}%): {"PASS" if m >= A2_PASS_PCT else "STOP"}')


# ============================================================================ rnn (PREREG_rnn.md)
def cmd_rnn(a):
    import torch
    import rnn_fusion_v2 as R
    tests = test_sessions(a.sessions)
    guard(tests)
    tr = train_sessions()
    work = os.path.join(RESULTS, f'{a.name}_rnn_lobo')
    os.makedirs(work, exist_ok=True)

    def sha(p):
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            for c in iter(lambda: f.read(1 << 20), b''):
                h.update(c)
        return h.hexdigest()

    def load_session(s):
        df = R.load_ekf_output(s)
        ekf = df[['ekf_x', 'ekf_y', 'ekf_theta']].values.astype(np.float32)
        gt, mask = R.load_ground_truth_aligned(df['timestamp'].values.astype(float), s)
        return ekf[mask], gt

    def own_stats(ekf, gt):
        edges = np.linspace(0, len(ekf), RNN_N_BLOCKS + 1).astype(int)
        e, g = [], []
        for b in range(RNN_N_BLOCKS):
            lo, hi = edges[b], edges[b + 1]
            if hi > lo and b % 7 not in (3, 6):
                e.append(ekf[lo:hi]); g.append(gt[lo:hi])
        r = R.compute_residual(np.concatenate(g), np.concatenate(e))
        return r.mean(axis=0), r.std(axis=0) + 1e-6

    def evaluate(model, mean, std, stats, s):
        assert s not in stats, f'leak: {s} in training stats'
        ekf, gt = load_session(s)
        m1 = np.mean([v['mean'] for v in stats.values()], axis=0)
        s1 = np.mean([v['std'] for v in stats.values()], axis=0)
        m3, s3 = own_stats(ekf, gt)
        o = {}
        for nm, rm, rs in (('V1', m1, s1), ('V2', np.zeros(3), s1), ('V3', m3, s3)):
            pred, w = R.run_rnn_inference(model, mean, std, rm, rs, ekf)
            n = len(pred)
            o[nm] = R.compute_rmse(R.smooth_trajectory(pred), gt[w:w + n])[0]
        o['V0'] = R.compute_rmse(ekf[w:w + n], gt[w:w + n])[0]
        return o

    def load_model(p):
        m = R.DenoisingRNN(input_size=3, output_size=3)
        m.load_state_dict(torch.load(p, map_location='cpu'))
        m.eval()
        return m

    V = ['V0', 'V1', 'V2', 'V3']
    model = load_model(os.path.join(LOG, 'rnn_model_denoising.pth'))
    mean, std = np.load(os.path.join(LOG, 'rnn_norm_stats.npy'))
    res = np.load(os.path.join(LOG, 'rnn_residual_stats_per_session.npy'), allow_pickle=True).item()
    sec = [dict(session=s, **evaluate(model, mean, std, res, s)) for s in tests]

    files = ['rnn_model_denoising.pth', 'rnn_norm_stats.npy', 'rnn_residual_stats_per_session.npy',
             'rnn_metrics.csv', 'rnn_training_curve.csv', 'rnn_denoising_loss_curve.png']
    for s in tr:
        files += [f'rnn_output_{s}.csv', f'rnn_smoothed_output_{s}.csv', f'rnn_trajectory_comparison_{s}.png']
    bak = os.path.join(work, '_prod_backup')
    os.makedirs(bak, exist_ok=True)
    hashes = {}
    for f in files:
        shutil.copy2(os.path.join(LOG, f), os.path.join(bak, f))
        hashes[f] = sha(os.path.join(LOG, f))
    prim = []
    try:
        for test in tr:
            log(f'\n=== LOBO fold: test = {test} ===')
            R.SESSIONS = [s for s in tr if s != test]
            torch.manual_seed(RNN_SEED)
            np.random.seed(RNN_SEED)
            R.train_denoising_rnn()
            fdir = os.path.join(work, test)
            os.makedirs(fdir, exist_ok=True)
            for f in ('rnn_model_denoising.pth', 'rnn_norm_stats.npy', 'rnn_residual_stats_per_session.npy'):
                shutil.copy2(os.path.join(LOG, f), os.path.join(fdir, f))
            fres = np.load(os.path.join(fdir, 'rnn_residual_stats_per_session.npy'), allow_pickle=True).item()
            if test in fres or len(fres) != len(tr) - 1:
                sys.exit(f'ABORT: fold {test} stats keys {sorted(fres)}')
            fm, fsd = np.load(os.path.join(fdir, 'rnn_norm_stats.npy'))
            prim.append(dict(test=test, **evaluate(load_model(os.path.join(fdir, 'rnn_model_denoising.pth')),
                                                    fm, fsd, fres, test)))
    finally:
        for f in files:
            shutil.copy2(os.path.join(bak, f), os.path.join(LOG, f))
        bad = [f for f in files if sha(os.path.join(LOG, f)) != hashes[f]]
        log(f'\nRestored {len(files)} production files; sha256 mismatches: {len(bad)}'
            + (f' -> {bad}' if bad else ' (all identical)'))
        if bad:
            sys.exit('ABORT: production RNN files not restored exactly.')
    P, Q = pd.DataFrame(prim), pd.DataFrame(sec)
    P.to_csv(out(a.name, 'rnn_primary.csv'), index=False)
    Q.to_csv(out(a.name, 'rnn_secondary.csv'), index=False)
    means = {v: P[v].mean() for v in V}
    log('\n========== RNN REPORT ==========')
    log(P.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    passed = {}
    for v in ('V1', 'V2'):
        w = int((P[v] < P['V0']).sum())
        imp = 100 * (means['V0'] - means[v]) / means['V0']
        passed[v] = w >= RNN_MIN_FOLD_WINS and imp >= RNN_MIN_IMPROVE
        log(f'{v}: beats V0 in {w}/{len(P)} folds; mean {imp:+.2f}% -> {"PASS" if passed[v] else "FAIL"}')
    adopted = 'V2' if passed['V2'] else ('V1' if passed['V1'] else None)
    log('Decision: ' + (f'adopt {adopted}' if adopted else 'neither passes -> RNN not generalising'))
    hr = 100 * (means['V1'] - means['V3']) / means['V1']
    log(f'H-R: V3 vs V1 {hr:+.2f}%; V2<V1 {means["V2"] < means["V1"]} -> '
        f'{"SUPPORTED" if hr >= RNN_HR_V3 and means["V2"] < means["V1"] else "NOT SUPPORTED"}')
    log('\nTest sessions (frozen production RNN):\n' + Q.to_string(index=False, float_format=lambda x: f'{x:.4f}'))


# ============================================================================ delay-check
def cmd_delay_check(a):
    from delay_labels_d2 import delay_r
    rows = []
    for group, bases in (('train', train_sessions()), ('test', test_sessions(a.sessions))):
        for b in bases:
            for sensor in ('imu', 'odom'):
                for v in SEV3:
                    s = f'{b}_fault_{sensor}_delay_{v}'
                    r = delay_r(json.load(open(os.path.join(LOG, f'fault_meta_{s}.json'))))
                    rows.append(dict(group=group, baseline=b, sensor=sensor, severity=v, mean_r=float(r.mean()),
                                     frac_lt1=float((r < 1 - 1e-12).mean()), n_faulty=int((r < D2_FAULTY).sum())))
    df = pd.DataFrame(rows)
    df.to_csv(out(a.name, 'delay_check.csv'), index=False)
    ok, n_mono, n_pairs = True, 0, 0
    for (grp, b, sensor), g in df.groupby(['group', 'baseline', 'sensor'], sort=False):
        mr = g.set_index('severity').loc[SEV3].mean_r.values
        mono = mr[0] >= mr[1] >= mr[2]
        log(f'{grp:<6}{b:<17}{sensor:<6}' + ''.join(f'{x:>8.3f}' for x in mr) + f'   {"yes" if mono else "NO"}')
        if grp == 'train':
            n_pairs += 1
            n_mono += mono
            ok &= mono
    zero = df[(df.group == 'train') & (df.frac_lt1 == 0)]
    ok &= zero.empty
    log(f'\nmonotonic train pairs {n_mono}/{n_pairs}; train sessions with no r<1: {len(zero)}; '
        f'test IMU-delay faulty rows: {int(df[(df.group == "test") & (df.sensor == "imu")].n_faulty.sum())}')
    log(f'A.4 verdict: {"PASS" if ok else "ABORT"}')


# ============================================================================ verify
def cmd_verify(a):
    new, ref = pd.read_csv(a.new), pd.read_csv(a.ref)
    key = ['model', 'session']
    cols = ['fixed', 'heuristic', 'lstm_linear', 'lstm_d1']
    m = new.merge(ref, on=key, suffixes=('_new', '_ref'), how='outer', indicator=True)
    only = m[m._merge != 'both']
    worst = max(float((m[f'{c}_new'] - m[f'{c}_ref']).abs().max()) for c in cols)
    log(f'rows: new {len(new)}, ref {len(ref)}, unmatched {len(only)}; max |difference| = {worst:.3e} m')
    ok = only.empty and worst < 1e-9
    log('VERIFY: ' + ('IDENTICAL -> merged code reproduces the reference exactly' if ok else 'DIFFERENT -> do not archive the old scripts yet'))
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('mark-test'); p.add_argument('sessions', nargs='+')
    for nm, default in (('stage1', 'heldout'), ('lidar', 'heldout'), ('rnn', 'heldout'), ('delay-check', 'heldout')):
        p = sub.add_parser(nm)
        p.add_argument('--name', default=default, help='results prefix (results/<name>_*)')
        p.add_argument('--sessions', nargs='+', help='test sessions (default: heldout_sessions.json)')
        if nm == 'stage1':
            p.add_argument('--from', dest='start', type=int, default=1, help='first step (1-5; 5 = skip data prep)')
    p = sub.add_parser('verify'); p.add_argument('new'); p.add_argument('ref')
    a = ap.parse_args()
    {'mark-test': lambda: cmd_mark_test(a.sessions), 'stage1': lambda: cmd_stage1(a),
     'lidar': lambda: cmd_lidar(a), 'rnn': lambda: cmd_rnn(a),
     'delay-check': lambda: cmd_delay_check(a), 'verify': lambda: cmd_verify(a)}[a.cmd]()


if __name__ == '__main__':
    main()
