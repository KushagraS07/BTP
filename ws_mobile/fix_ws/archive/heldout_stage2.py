#!/usr/bin/env python3
"""
heldout_stage2.py -- Stage 2 (D5) of the pre-registered held-out evaluation
(fix_ws/PREREG_heldout.md, frozen 2026-10-03 13:45 IST). A.2 re-test on the 9
held-out LiDAR corruption sessions. Nothing is retrained.

Pre-registered rules applied here (do not change):
  * RNN = the frozen rnn_model_denoising.pth (trained 2026-10-02 22:12 on the 7
    training baselines only).
  * Input normalisation = rnn_norm_stats.npy (fit on the training split of the 7
    training sessions only).
  * Residual de-normalisation = MEAN of the 7 training sessions' per-session
    residual stats (rnn_residual_stats_per_session.npy). Never stats computed from
    held-out ground truth.
  * A.2 itself: unchanged functions from run_a2_sanity_check.py
    (roughness_confidence_fn, run_at_alpha) and lidar_fusion_v2.run_lidar_fusion_single.
  * Pass rule: mean RMSE improvement (proxy vs baseline confidence) at
    ALPHA = 0.70 >= 5% across the 9 sessions.

Why the RNN step is needed: for training sessions, fault_injection.py copies the
baseline's rnn_smoothed_output into every fault session (the RNN never reads LiDAR).
The held-out baselines were never run through rnn_fusion_v2.py (it trains on
get_good_sessions(), which excludes them), so there was nothing to copy. This script
creates rnn_smoothed_output_<heldout>.csv with the frozen model, then copies it into
the corruption sessions exactly as fault_injection.py does for training sessions.

Writes: rnn_output_/rnn_smoothed_output_ for the 3 held-out baselines and their 9
corruption sessions, final_fused_output_v2_* for those 9 (side effect of the unchanged
lidar_fusion_v2), and fix_ws/heldout_stage2_a2.csv. Nothing is retrained or registered.

Usage:  cd ~/ws_mobile/scripts && python3 ../fix_ws/heldout_stage2.py 2>&1 | tee ../fix_ws/heldout_stage2.log
"""
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch

WS = os.environ.get('WS_MOBILE_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # automatic
S = os.path.join(WS, 'scripts')
LOG = os.path.join(WS, 'sensor_logs')
FIX = os.path.join(WS, 'fix_ws')
sys.path.insert(0, S)
os.chdir(S)

# ---- PRE-REGISTERED SETTINGS (PREREG_heldout.md, Stage 2) -- do not change ------
HELDOUT = json.load(open(os.path.join(FIX, 'heldout_sessions.json')))['heldout_sessions']
TRAIN = sorted(__import__('session_registry').get_good_sessions())  # automatic: the training baselines in session_registry.json
SEVS = ['mild', 'moderate', 'severe']
DIAG_ALPHA, DEPLOYED_ALPHA, PASS_PCT = 0.70, 0.98, 5.0
# -----------------------------------------------------------------------------------


def log(m=''):
    print(m, flush=True)


def main():
    import rnn_fusion_v2 as R
    import run_a2_sanity_check as A2
    from session_registry import get_good_sessions, is_heldout

    # ---- guards ---------------------------------------------------------------
    if any(b in get_good_sessions() for b in HELDOUT):
        sys.exit('ABORT: a held-out session is in good_sessions.')
    if not all(is_heldout(b) for b in HELDOUT):
        sys.exit('ABORT: is_heldout() does not recognise every held-out session.')
    if (A2.DIAGNOSTIC_ALPHA, A2.DEPLOYED_ALPHA, A2.PASS_THRESHOLD_PCT) != (DIAG_ALPHA, DEPLOYED_ALPHA, PASS_PCT):
        sys.exit('ABORT: run_a2_sanity_check constants differ from the pre-registration.')

    # ---- frozen RNN + training-only normalisation ----------------------------------
    mpath = os.path.join(LOG, 'rnn_model_denoising.pth')
    log(f'RNN model: {os.path.basename(mpath)}  modified '
        f'{time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(mpath)))} '
        f'(pre-registered: 2026-10-02 22:12)')
    model = R.DenoisingRNN(input_size=3, output_size=3)
    model.load_state_dict(torch.load(mpath, map_location='cpu'))
    model.eval()

    mean, std = np.load(os.path.join(LOG, 'rnn_norm_stats.npy'))
    res = np.load(os.path.join(LOG, 'rnn_residual_stats_per_session.npy'), allow_pickle=True).item()
    if sorted(res) != sorted(TRAIN):
        sys.exit(f'ABORT: residual stats keys {sorted(res)} != the 7 training sessions.')
    res_mean = np.mean([res[s]['mean'] for s in TRAIN], axis=0)
    res_std = np.mean([res[s]['std'] for s in TRAIN], axis=0)
    log(f'Residual stats = mean of 7 training sessions: mean={np.round(res_mean, 4)}, '
        f'std={np.round(res_std, 4)}')

    # ---- step 1: RNN output for each held-out baseline, copied to corruption sessions --
    log('\n=== Step 1: frozen-RNN inference on held-out baselines (descriptive) ===')
    rnn_rows = []
    for b in HELDOUT:
        ekf_df = R.load_ekf_output(b)
        ekf = ekf_df[['ekf_x', 'ekf_y', 'ekf_theta']].values.astype(np.float32)
        t = ekf_df['timestamp'].values.astype(float)
        gt, mask = R.load_ground_truth_aligned(t, b)
        ekf, t = ekf[mask], t[mask]
        pred, w = R.run_rnn_inference(model, mean, std, res_mean, res_std, ekf)
        n = len(pred)
        sm = R.smooth_trajectory(pred)
        gt_a = gt[w:w + n]
        ekf_rmse = R.compute_rmse(ekf[w:w + n], gt_a)[0]
        rnn_rmse = R.compute_rmse(sm, gt_a)[0]
        ts = t[w:w + n]
        for name, arr in (('rnn_output', pred), ('rnn_smoothed_output', sm)):
            pd.DataFrame({'timestamp': ts, 'rnn_x': arr[:, 0], 'rnn_y': arr[:, 1],
                          'rnn_theta': arr[:, 2]}).to_csv(
                os.path.join(LOG, f'{name}_{b}.csv'), index=False)
        src = os.path.join(LOG, f'rnn_smoothed_output_{b}.csv')
        for v in SEVS:
            fsess = f'{b}_fault_lidar_corruption_{v}'
            if not os.path.exists(os.path.join(LOG, f'scan_filtered_{fsess}.csv')):
                sys.exit(f'ABORT: scan_filtered_{fsess}.csv missing (Stage 1 step 1 output).')
            shutil.copy2(src, os.path.join(LOG, f'rnn_smoothed_output_{fsess}.csv'))
        rnn_rows.append(dict(baseline=b, ekf_rmse=ekf_rmse, rnn_rmse=rnn_rmse, rows=n))
        log(f'  {b}: EKF {ekf_rmse:.4f} m -> RNN (smoothed) {rnn_rmse:.4f} m  ({n} rows)')

    # ---- step 2: A.2, unchanged functions -------------------------------------------
    log('\n=== Step 2: A.2 on the 9 held-out corruption sessions ===')
    rows = []
    for b in HELDOUT:
        for v in SEVS:
            s = f'{b}_fault_lidar_corruption_{v}'
            bd = A2.run_at_alpha(s, DEPLOYED_ALPHA, None, 'baseline/deployed')
            pd_ = A2.run_at_alpha(s, DEPLOYED_ALPHA, A2.roughness_confidence_fn, 'proxy/deployed')
            bg = A2.run_at_alpha(s, DIAG_ALPHA, None, 'baseline/diagnostic')
            pg = A2.run_at_alpha(s, DIAG_ALPHA, A2.roughness_confidence_fn, 'proxy/diagnostic')
            if None in (bd, pd_, bg, pg):
                sys.exit(f'ABORT: A.2 returned no result for {s}.')
            rows.append(dict(session=s, baseline=b, severity=v,
                             base_deployed=bd, proxy_deployed=pd_,
                             imp_deployed_pct=100 * (bd - pd_) / bd,
                             base_diag=bg, proxy_diag=pg,
                             imp_diag_pct=100 * (bg - pg) / bg))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(FIX, 'heldout_stage2_a2.csv'), index=False)
    pd.DataFrame(rnn_rows).to_csv(os.path.join(FIX, 'heldout_stage2_rnn.csv'), index=False)

    # ---- report -----------------------------------------------------------------
    log('\n\n========== STAGE 2 REPORT ==========')
    log('RNN on held-out baselines (descriptive):')
    for r in rnn_rows:
        log(f'  {r["baseline"]}: EKF {r["ekf_rmse"]:.4f} m  RNN {r["rnn_rmse"]:.4f} m')
    log(f'\n{"session":<48}{"base@0.70":>10}{"proxy@0.70":>11}{"imp%":>8}{"imp%@0.98":>11}')
    for r in rows:
        log(f'  {r["session"]:<46}{r["base_diag"]:>10.4f}{r["proxy_diag"]:>11.4f}'
            f'{r["imp_diag_pct"]:>+8.2f}{r["imp_deployed_pct"]:>+11.2f}')
    log('\nBy severity (mean improvement at 0.70):')
    for v in SEVS:
        log(f'  {v:<9} {df[df.severity == v].imp_diag_pct.mean():+.2f}%')
    m = df.imp_diag_pct.mean()
    log(f'\nMean improvement at ALPHA=0.70 over {len(df)} sessions: {m:+.2f}%  '
        f'(original 7 sessions: +2.21%)')
    log(f'Mean improvement at ALPHA=0.98 (deployed): {df.imp_deployed_pct.mean():+.2f}%')
    log(f'D5 verdict (pre-registered >= {PASS_PCT}%): {"PASS" if m >= PASS_PCT else "STOP"}')
    log('\nSaved: fix_ws/heldout_stage2_a2.csv, fix_ws/heldout_stage2_rnn.csv')


if __name__ == '__main__':
    main()
