# Full system (Paper_ws filter in ws_mobile) — findings log

Development log. Nothing here is a paper result. Plan: `claude/solo_plan.md`, port plan
`claude/fullsystem_port_plan.md`.

## F0 — reproduction of Paper_ws (2026-10-09, cloud workspace)
- Paper_ws commit efe31ef: 66/66 unit tests pass; `zupt.evaluation.eval_change1` reproduces
  docs/change1.md exactly (36/36 cells). Same on the VM (`~/paper_ws_ref`, read-only clone): 66 passed.
- Logger CSV format of Paper_ws = ws_mobile (identical columns); only the output folder differs.

## F1 — Paper_ws filter on ws_mobile 50 Hz recordings (development check)
`eval_change1` on slip sessions 140649 / 150044 / 150546 (provisional Paper_ws parameters):
runs without error; best variant (LiDAR) RMSE 0.116 / 0.169 / 0.408 m. Indoor world is LiDAR-
degenerate along the corridor (674/674 scans degenerate in 140649), as Paper_ws found. 5c changed
nothing; oracle wheel gating did not help much → points to the 50 Hz IMU.

## F2 — IMU rate (pass rule fixed beforehand: Paper_ws `tools/check_imu_integration.py`,
## |∫a dt − Δv_gt| ≤ 0.02 m/s at every straight start/stop)
- ws_mobile 50 Hz sessions: **FAIL** 0.073 / 0.073 / 0.077 m/s (Paper_ws 50 Hz: FAIL 0.050).
- `apply_imu_rate_patch.py` (backups `*.bak_preImuRate`): launch args `imu_rate` (default 50),
  `gyro_bias_z` (default 0.0000075), `odom_rate` (default 50; misspelled, ignored tag
  `odom_publisher_frequency` replaced by `odom_publish_frequency`), `x y yaw`. Default xacro output
  identical to the original except that odom tag (checked tag by tag). colcon build OK.
- `scripts/fs_sessions.py`: full-system recordings moved to `fullsys_logs/raw/`, removed from
  good_sessions (Phase 2 list stays 7); refuses Phase 3B slip sessions.
- 1 kHz check 1 (20261009_215016): 33 553 msgs, 552 gaps (VM busy) → worst 0.021 → **FAIL**;
  7/10 events exact, 3 off by 0.012–0.021 (dropped messages on the ramps).
- 1 kHz check 2 (20261009_215432), Mac apps closed, Gazebo view untouched: 35 775 msgs, 22 gaps →
  **PASS, worst 0.000 m/s** (10/10 events exact).
- **Decision:** all full-system recordings use `imu_rate:=1000`, recorded with low load (Mac apps
  closed, Gazebo view not moved). Each recording's IMU gaps are reported (`check_imu_integration`).

## F3 — port into ws_mobile (2026-10-09 22:05–22:16 IST) — VERIFIED
- `fs_port.py`: Paper_ws commit efe31ef → `~/ws_mobile/fullsys/` (own git repo, commit 7bfa6d8).
  Copied zupt/, config/, tests/, tools (check_imu_integration, gen_terrain_world), docs_paperws/.
  Only edits: argparse default paths → `zupt/paths.py` (workspace found automatically, nothing
  hard-coded); terrain generator output → mobile_robot/model. `PORT.md` lists every edit.
- terrain_world.sdf copied byte-identical into mobile_robot/model (regenerated output identical too);
  colcon build OK.
- 66/66 tests pass in ws_mobile; `eval_change1` on Paper_ws data: **REPRODUCED_EXACTLY**
  (36/36 cells vs docs_paperws/change1.md).
- Git identity in fullsys/ copied from btp_backup (email is a placeholder; local repo only).

## F4 — remaining sim checks: decision rules written BEFORE recording (2026-10-09 22:25 IST)
Tool `fullsys/tools/fs_check.py` (read-only): topic rates/gaps, IMU integration check, gyro mean
while GT-still. Checked on Paper_ws data: gyro-bias session 20261005_000755 → +0.00988 rad/s
(Paper_ws decisions.md: 0.00988), 1 kHz session 201210 → PASS 0.011, 24.6 % IMU missing.
- **R1 terrain world** (`world:=terrain_world.sdf y:=3.8` = lane 1, ice μ 0.1 at x 4–12;
  `imu_rate:=1000 odom_rate:=100`): world loads and all topics present → terrain world OK.
  **Odom 100 Hz adopted** if odom median ≥ 90 Hz AND IMU check PASS AND IMU missing < 1 %;
  otherwise full-system recordings keep odom 50 Hz.
- **R2 gyro-bias variant** (indoor world, `imu_rate:=1000 gyro_bias_z:=0.01`): works if the
  GT-still gyro mean is in [0.009, 0.011] rad/s.

**Results (2026-10-09 22:26–22:30 IST):**
- R1 20261009_222654 (terrain lane 1): all topics present → **terrain world OK in Harmonic**.
  Odom 100.0 Hz (0 gaps), IMU check PASS 0.009 m/s, but IMU missing **3.9 %** (> 1 %) →
  **rule: odom 100 Hz NOT adopted; full-system recordings keep odom_rate 50 (default).**
  (Extra load from 100 Hz odom and/or the larger world; IMU check still passed.)
- R2 20261009_222927 (indoor, gyro_bias_z 0.01): GT-still gyro mean **+0.00999 rad/s**
  (± 0.00021) ∈ [0.009, 0.011] → **gyro-bias variant works** (Paper_ws: 0.00988). IMU missing 0.2 %,
  IMU check PASS 0.000.
- **Recording settings for the full system:** `imu_rate:=1000`, odom 50 Hz (default), low load.

## Weeks 2–3 of the plan: complete (2026-10-09)

## F5 — development sessions: plan (written 2026-10-10, before recording)
Purpose: development only (LSTM reliability retraining for EKF9 with LOBO, integration tuning).
Never used as test data. Recorded with `imu_rate:=1000`, odom 50 Hz, low load; moved with
`fs_sessions.py --role dev`.
- **8 dev sessions:** D1–D4 terrain world (lanes 1 ice, 3 rough strips, 4 ramp + icy descent,
  5 mixed: ice / strips / mild / severe), D5–D8 indoor slip world (corridor / north / south
  patches, stops, starts and turns on patches, as in the Phase 3B routes). Lanes 0 (smooth) and
  2 (mild) are kept for calibration sessions later.
- **Quality rule (fixed now):** a dev session is kept if `fs_check.py` shows IMU check PASS
  (≤ 0.02 m/s) AND IMU missing < 1 % AND all five topics present; otherwise re-recorded.
  Duration 60–120 s.
- **Known issue for the integration step (found while planning):** the Phase 3B P_slip model was
  trained on 50 Hz point-sampled IMU features (preprocess_data interpolates onto the 50 Hz grid).
  For 1 kHz recordings that interpolation would point-sample a noise level √20 ≈ 4.5× higher
  (noise is scaled up so that the AVERAGED 50 Hz noise is unchanged). P_slip inputs must therefore
  be built from the block-averaged 50 Hz IMU (as zupt.replay.decimate_imu), and the equivalence to
  the training distribution checked before P_slip is used in the full system.

## F5 — recording log (dev)
- D1 terrain lane 1 (ice): 121804 FAIL (IMU missing 3.0 %, 4 VM cores) → VM set to 6 cores;
  125637 FAIL (55 s); 130123 FAIL (56.3 s; terrain world at 1 kHz runs ~0.6× real time);
  **130530 KEEP** (66.9 s, IMU 0.2 %, check 0.000). Stopwatch target for the terrain world: 120 s.
- D2 terrain lane 3 (rumble strips): 131512 — 0.2 % missing, 65.4 s, IMU check 0.129 FAIL.
  Per-event table: all flat-floor starts/stops exact (0.000–0.002); errors 0.034–0.129 only for
  starts/stops on the strips (bumps/tilt; the check assumes flat ground).

## F5a — amendment to the F5 quality rule (2026-10-10 13:25 IST, development rule, logged)
**Reason:** the IMU check exists to verify the RECORDING (no dropped messages, IMU rate). On rumble
strips / ramps it fails for physical reasons (impacts, tilt), as already seen in Paper_ws (ramp
transitions 0.028–0.058). Re-recording cannot change that. **Change:** the check is judged on
straight starts/stops on flat floor (low-friction patches are flat and stay included); events whose
window (t−0.1 … t+1.0 s) touches a terrain-world rumble-strip or ramp zone (lane 3 x 4–12, lane 4
x 4–12, lane 5 x 6–8, ±0.3 m) are excluded and REPORTED; ≥ 3 flat events required.
`fs_check.py --world terrain` implements it. Does not affect D1 (ice is flat; all events checked).
This makes the rule apply to what it was meant to measure; it is not a test rule.

## F6 — development set COMPLETE (2026-10-10 15:23 IST)
All recorded with imu_rate 1000, odom 50 Hz, VM 6 cores, Mac apps closed. Checked with
`fs_check.py` (F5 + F5a); every failed attempt logged in `fix_ws/fullsys_session_log.txt`.

| ID | Session | World / route | Duration | IMU missing | Flat IMU check |
|---|---|---|---|---|---|
| D1 | 20261010_130530 | terrain lane 1 ice, there and back | 66.9 s | 0.2 % | 0.000 (9 ev) |
| D2 | 20261010_131512 | terrain lane 3 rumble strips, there and back | 65.4 s | 0.2 % | 0.000 (4 ev; 5 strip ev excl.) |
| D3 | 20261010_143058 | terrain lane 4 ramp up/plateau/icy ramp down (slide), one way | 75.0 s | 0.3 % | 0.000 (5 ev; 5 ramp ev excl.) |
| D4 | 20261010_143931 | terrain lane 5 mixed (ice/strips/mild/severe), there and back | 78.1 s | 0.3 % | 0.003 (9 ev; 2 strip ev excl.) |
| D5 | 20261010_145610 | indoor slip: corridor (2 stops, turn) → strip → north | 76.1 s | 0.1 % | 0.000 (7 ev) |
| D6 | 20261010_150047 | indoor slip: strip → south (2 stops, turn) → corridor | 85.3 s | 0.2 % | 0.003 (4 ev) |
| D7 | 20261010_150428 | indoor slip: strip → north (2 stops, turn) → corridor | 87.7 s | 0.4 % | 0.007 (6 ev) |
| D8 | 20261010_152122 | indoor slip: corridor (2 stops, turn) → strip → south | 76.4 s | 0.8 % | 0.002 (9 ev) |

Failed attempts (kept on disk, never used): D1 121804 (IMU 3.0 %), 125637 (55 s), 130123 (56.3 s);
D3 142525 (49.3 s); D8 151742 (IMU 1.8 %). Earlier check recordings (role check): 215016,
215432, 222654, 222927.
Observations (development): terrain world at 1 kHz runs ~0.6× real time (stopwatch 120–140 s →
65–78 s recorded); indoor ~0.8–0.9×. On the 5° icy descent (μ 0.1 ≈ tan 5.7°) the robot slides
to the bottom after a stop — wheels stopped, IMU accelerating: the case where a naive ZUPT fails.

## F7 — reliability LSTM for EKF9: design (written 2026-10-10, before any code or training)
Method = Phase 2 (same label convention, same training protocol), moved to the new signals.
Development only; final claims come from the later pre-registered run.

**Data:** D1–D8 (loader: role dev in `fullsys_logs/sessions.json` AND fs_check KEEP), IMU
block-averaged to 50 Hz (zupt.replay.load_session), odom 50 Hz.
**Faults (offline, on the 50 Hz streams the filter uses):**
| Sensor | Signal | Types × severities (multiples of the signal's own std in that session) |
|---|---|---|
| IMU | gyro_z | noise 1/3/6/12 · bias 1/3/6 · drift 1/3/6 · dropout (held value) 5/15/35 % |
| Wheels | odom linear_x, angular_z | same four types and severities |
- **D6:** every fault starts MID-SESSION: onset uniform in [20 %, 60 %] of the session (seeded),
  active to the end (dropout: one window of the given length starting at onset). Clean before onset.
- 13 faults × 2 sensors × 8 sessions = 208 fault sessions + 8 clean.
- Not in v1: delay faults, accelerometer faults (accel drives the prediction; it is protected by
  the ±25 m/s² clip, not by R), LiDAR faults (r_LiDAR comes later with scan-matching faults).
**Labels (Phase 2 convention, `generate_reliability_labels.py`):** r = 1 before onset; after onset
noise/bias r = 1/(1+severity); drift ramp r = 1/(1+offset/std); dropout r = 0 inside the window.
**Features (spec §4), from a PARALLEL fixed-R EKF9 run (no feedback of the LSTM's own output, as in
Phase 3B):** odom innovations (v, ω), gyro innovation, odom/gyro NIS; raw gyro_z; accel x/y/z
(clipped ±25 m/s², D6); odom v, ω; wheel–IMU disagreement (ω_odom − gyro_z); terrain cues:
windowed variance of accel_z and of high-passed accel_x. Excluded: r(t−1), cov_trace, terrain ID.
**Model / training (Phase 2):** LSTM window 50, hidden 64, 1 layer, two sigmoid heads (r_imu,
r_odom), MSE, Adam 1e-3, batch 256, 15 epochs, seed 42, LOBO over D1–D8 (val = first remaining),
normalisation on the fit split only, clean-window weight as Phase 2 final models.
**Use in the filter (integration step, later):** R = R₀/(r+ε); update SKIPPED when r < 0.5
(D6 gate; Paper_ws 5b finding 5: scaling alone does not remove a 100 Hz sensor).
**Development checks:** label sanity per fault; LOBO open-loop r error vs trivial predictors;
closed-loop EKF9 RMSE on held-out fault sessions vs fixed R and vs oracle r.

## F8 — F7 pipeline implemented (2026-10-10, cloud workspace; tested on Paper_ws data only)
`fullsys/learning/`: `faults.py` (26 faults/session, mid-session onset 20–60 %, Phase 2 labels;
6 unit tests, 72/72 total pass), `features.py` (parallel fixed-R EKF9 replay, 14 features),
`build_dataset.py` (sessions only from `config/dev_sessions.txt`, each must pass fs_check KEEP),
`train.py` (LOBO, Phase 2 protocol), `closed_loop.py` (fixed / LSTM / oracle EKF9),
`run_all.sh` (chain).
Checks on Paper_ws data (development, mechanics only):
- Clean fixed-EKF9 RMSE on S1 slip patch = 0.137 m = Paper_ws 5b value; pre-onset features are
  bit-identical to the clean run (no look-ahead).
- 3 sessions × 27 runs built; 2-epoch / 3-fold smoke training; closed loop runs (numbers not used).
**Design notes / deviations (logged before running on D1–D8):**
1. **Training stride 5** (windows every 5 rows) instead of Phase 2's 1: consecutive windows overlap
   49/50; stride 1 ≈ 20 h CPU for 8 folds on the VM. Validation/test use every row.
2. **Severity convention = Phase 2** (multiple of the signal's own std incl. motion). For gyro_z
   (std ≈ 0.45 rad/s, dominated by turns) "large" bias ≈ 2.7 rad/s, i.e. very large faults;
   fixed-EKF9 RMSE ratios reach 10²–10³. Kept for comparability with Phase 2; to be stated in the
   paper. A physically small fault set (scaled by sensor noise σ) is a possible addition later.
3. v1 features come from EKF9 without LiDAR (scan matching is slower and not fault-affected);
   LiDAR NIS and r_LiDAR come with the scan-matching faults later.
4. Accel clip ±25 m/s² is applied to the LSTM features only; the filter prediction is unchanged.

## F9 — reliability LSTM on D1–D8: first development result (2026-10-10, VM, run 15:51–17:39 IST)
`run_all.sh 3` (log `results/fullsys_rel_run.log`). Development numbers, not paper results.
- **Dataset:** 8/8 sessions KEEP; 216 runs. Integrity 63/64 — below 1.05: D3 (143058) IMU dropout
  ×1.01 (gyro dropout barely hurts EKF9 when wheels + NHC are healthy; noted, not a stop rule).
- **Training:** D1 and D2 folds have identical fit sets (val = first remaining session), so identical
  fit losses — a property of the split rule, not a bug. Val loss fell to ≈ 0.022 (fit ≈ 0.012).
- **Open loop (r error, mean of 8 folds):** r_imu 0.145 (always-1 0.364, fit-mean 0.321);
  r_odom 0.145 (always-1 0.359, fit-mean 0.317) → ≈ 55–60 % below the trivial predictors.
- **Closed loop (EKF9 position RMSE, m, held-out fault runs):**

| sensor | type | n | LSTM<fixed | oracle<fixed | fixed | LSTM | oracle |
|---|---|---|---|---|---|---|---|
| imu | bias | 24 | 23 | 21 | 11.908 | 2.332 | 2.926 |
| imu | drift | 24 | 20 | 20 | 5.799 | 3.004 | 2.495 |
| imu | dropout | 24 | 14 | 22 | 3.047 | 1.600 | 1.045 |
| imu | noise | 32 | 18 | 19 | 3.347 | 1.877 | 1.498 |
| odom | bias | 24 | 24 | 23 | 14.109 | 1.030 | 1.636 |
| odom | drift | 24 | 20 | 24 | 5.186 | 1.306 | 1.262 |
| odom | dropout | 24 | 15 | 19 | 1.908 | 2.082 | 0.705 |
| odom | noise | 32 | 18 | 26 | 1.278 | 0.842 | 0.739 |

  Fault runs: LSTM beats fixed in 152/208, median change −43.4 %.
  **Clean runs: fixed 0.670, LSTM 0.895 (+33.5 %), oracle 0.670.**
- **Interpretation (development):**
  1. Large gains on bias and drift (both sensors) — the expected strength of learned reliability.
  2. **Cost on clean data (+33.5 %)**: the LSTM lowers r without a fault (false alarms), so good
     updates are down-weighted or skipped. This also affects the pre-onset part of every fault run.
     Must be understood before integration (diagnosis D-F9a).
  3. Odom dropout: LSTM worse than fixed on average (2.08 vs 1.91; 15/24). Noise: wins only
     18/32 per run although means improve (means dominated by the extreme severities).
  4. The oracle is NOT an upper bound: LSTM < oracle on bias (imu, odom). The Phase 2 labels
     1/(1+sev) with the 0.5 gate are a heuristic weighting, not optimal R. Paper wording:
     "label-driven r", not "oracle upper bound".
  5. Fixed-R RMSE of 5–14 m reflects the very large Phase 2 severities (F8 note 2); headline gains
     must be reported per severity, and clean-run cost reported alongside.
- **D-F9a (diag_clean.py):** the clean-run cost comes from 3 of 8 sessions, all with wheel slip:
  D1 ice 0.238→0.677 (+184 %, r_imu < gate 3.5 % of rows), D6 indoor slip strip 1.373→2.537 (+85 %,
  r_imu 1.4 %, r_odom 4.8 %), D3 icy slide 2.173→2.370 (+9 %, r_odom 8.6 %). The other 5: ±0.6 %,
  ≈ 0 % below gate. Pre-onset rows of fault runs: ≈ 0 % (episodes fall later in the sessions).
- **D-F9b (diag_episodes.py):** every below-gate episode on a clean run coincides with physical
  slip. D1: 4 r_imu episodes (0.34–1.02 s) at x 5.4–8.3 on lane 1 = ON THE ICE, wheels v ≈ 0,
  |ω_wheel − ω_gyro| 0.31–0.81 rad/s → the LSTM blames the (correct) gyro. D6: r_imu and r_odom
  episodes 38–43 s at (4.9, −3.9), wheels v = 0, |ω_wheel − ω_gyro| 0.39–0.97 (slip during a turn);
  one r_odom episode 1.74 s at 69 s. D3: r_odom episodes (up to 4.3 s) with wheels stopped,
  |ω dis| 0.007 — wheels stopped on the ramp / icy descent (slide); gating odom there is physically
  plausible, so D3's +9 % is not clearly a false alarm.
- **Cause (development conclusion):** the labels encode only INJECTED faults; physical slip makes
  wheel odometry genuinely wrong while labelled r_odom = 1. A wheel–gyro disagreement is therefore
  ambiguous in training and the model sometimes blames the gyro. A persistence rule would hide
  short episodes but not the cause (episodes up to 1.7 s; it would also delay real detections).

## F10 — slip-aware reliability labels (decided 2026-10-10 18:30 IST, BEFORE re-running)
**Change:** r_odom label = min(fault label, 1 − slip), slip = Phase 3B frozen definition
(PREREG_slip.md §4: |v_wheel − v_true| ≥ 0.10 m/s or |ω_wheel − ω_true| ≥ 0.30 rad/s) computed from
the CLEAN wheel odometry vs ground truth on the 50 Hz IMU grid. r_imu labels unchanged.
Slip rows are treated like odom dropout (r = 0 → update skipped by the oracle; the LSTM learns it).
- No new parameter: thresholds are the frozen 3B values. One technical deviation from 3B code:
  GT yaw is unwrapped before differentiation (3B differentiated wrapped yaw → spikes at ±π).
- Ground truth is used for LABELS only (as in 3B); features and the filter never see it.
- Consequence for the paper: "reliability" = whether a measurement can currently be trusted,
  from sensor faults OR terrain; the closed-loop "oracle" (label-driven r) changes accordingly.
- Everything else as F7/F8 (features, LSTM, LOBO, stride 5, gate 0.5, seeds). v1 models/results
  kept as *_v1 for comparison. Code: `learning/patch_f10.py` (backups *.bak_preF10).
- **Pass/look criteria for the development re-run (written now):** (a) clean-run LSTM/fixed mean
  change ≤ +5 % (v1 +33.5 %); (b) fault-run gains stay comparable to v1 (LSTM beats fixed in
  152/208, median −43.4 %; reported either way, not tuned); (c) D1/D6 clean below-gate r_imu ≈ 0 %.
  If (a) fails, the next candidate (logged first) is P_slip as an LSTM input / gyro-gate override.

- **F10 run started 2026-10-10 18:24 IST** (commit 0005bf2; log `results/fullsys_rel_run_f10.log`;
  v1 kept as models_v1, *_v1.csv, fullsys_rel_run_v1.log). Integrity unchanged 63/64 (labels only).
  Slip label, % of rows on clean runs: D1 18.9, D2 1.1, D3 5.4, D4 13.0, D5 9.3, D6 23.9, D7 8.0,
  D8 10.0. Ice/slip-strip sessions highest, rumble strips lowest; indoor 8–10 % includes the slip
  strip and skid during turns (3B also found 5.6–16.3 % off-patch positives). Accepted as is.

- **F10 result (development, run 18:24–20:12 IST):**
  Open loop: r_imu 0.147 (always-1 0.364, fit-mean 0.321); r_odom 0.210 (always-1 0.471, fit-mean 0.397).

| sensor | type | n | LSTM<fixed | oracle<fixed | fixed | LSTM (v1) | oracle |
|---|---|---|---|---|---|---|---|
| imu | bias | 24 | 21 | 21 | 11.908 | 3.023 (2.332) | 2.241 |
| imu | drift | 24 | 19 | 21 | 5.799 | 3.201 (3.004) | 2.488 |
| imu | dropout | 24 | 22 | 24 | 3.047 | 1.699 (1.600) | 0.295 |
| imu | noise | 32 | 17 | 28 | 3.347 | 1.790 (1.877) | 0.691 |
| odom | bias | 24 | 24 | 22 | 14.109 | 1.357 (1.030) | 1.573 |
| odom | drift | 24 | 23 | 22 | 5.186 | 1.165 (1.306) | 1.072 |
| odom | dropout | 24 | 19 | 23 | 1.908 | 1.337 (2.082) | 0.243 |
| odom | noise | 32 | 21 | 29 | 1.278 | 0.863 (0.842) | 0.460 |

  Fault runs: LSTM beats fixed 166/208 (v1 152), median −47.4 % (v1 −43.4 %).
  **Clean runs: fixed 0.670, LSTM 0.378 (−43.6 %; v1 +33.5 %), label-driven r 0.131.**
  Criterion (a) clean ≤ +5 %: **MET** (−43.6 %). (b) fault gains kept: yes (166/208, −47.4 %).
  (c) see diag_clean (F10b).
- **Interpretation (development):** slip-aware labels turn the clean-run cost into a large gain: the
  LSTM now skips wheel updates during physical slip, which the fixed filter cannot do. Odom dropout
  flips from a loss to a gain. IMU bias slightly weaker than v1 (3.02 vs 2.33 m; 21 vs 23 wins);
  IMU noise still weak per run (17/32). Caveats for the paper: (1) the clean gain is a slip-handling
  effect, so the fair comparison needs baselines that also handle slip (Phase 3B P_slip gating,
  Sage–Husa adaptive R) — to be added before the final pre-registration; (2) the label-driven r
  (0.131) uses ground-truth slip and is a reference, not an achievable bound; (3) 8 dev sessions only.

- **F10b (diag_clean.py on F10 models):** every clean session improves: D1 −54 %, D2 −55 %,
  D3 −13 %, D4 −63 %, D5 −53 %, D6 −73 %, D7 −34 %, D8 −69 %. r_odom below gate tracks the slip
  label (D1 19.4 % vs label 18.9 %; D6 21.8 vs 23.9; D2 2.2 vs 1.1; D5 8.6 vs 9.3). Gyro false
  alarms: D1 3.5 → 0.45 %, D6 1.38 → 1.31 % (unchanged), all others 0 %; pre-onset 0 % everywhere.
  Criterion (c) PARTLY met (D6 residual gyro gating ≈ 1.3 %, not harmful in aggregate: D6 −73 %).
  To be watched in the final test; no further change now.
- **Decision:** F10 labels are the current reliability design. Next: commit + backup, then
  integration items (P_slip for NHC/ZUPT gating, Sage–Husa and P_slip-gating baselines).

- **Backup 2026-10-10 20:52 IST:** GitHub KushagraS07/BTP commit a54b470 (fullsys code + git bundle,
  models v1/F10, results, findings, session log, robot package). Raw logs and features not uploaded.

## F11 — integration step 1: learned P_slip into EKF9 (design written 2026-10-10 21:00 IST, before code)
**Why a new P_slip model:** the Phase 3B model takes features of the 3-state Phase 2 EKF (+ LiDAR
disagreement) at 50 Hz IMU; feeding it from EKF9 / 1 kHz-averaged data would be a distribution shift
(the open check of the port plan). Instead P_slip is RETRAINED on the F7 EKF9 features with the
Phase 3B protocol — the same move as the reliability LSTM (Phase 2 → F7). Phase 3B stays the
3-state result in the paper.
- **Model (= Phase 3B §5 protocol):** LSTM(14 → 64, 1 layer) → 1 logit → sigmoid; BCE with
  pos_weight = #no-slip / #slip of the fit split; Adam 1e-3, batch 256, 15 epochs, seed 42; LOBO,
  val = first remaining session, best-val checkpoint; normalisation on fit rows.
- **Inputs:** the 14 F7 features (same parallel fixed-R EKF9 run). **Label:** the F10 `slip` column
  (3B frozen definition, from clean odometry vs GT; GT for labels only).
- **Data:** all 27 runs per session (clean + 26 faults), so P_slip must tell physical slip from
  injected odom faults (label stays physical slip during faults). Stride 5 (as F8). Deviations from
  3B: inputs/data as stated; training stride.
- **Use in the filter (Pratiyush's hooks, provisional dev parameters unchanged, not tuned):**
  `predict(t, p_slip)` (Q_ω × (1 + 50·P)), `update_nhc(p_slip)` (R_nhc/(1−P), skipped at P ≥ 0.5),
  detector `Evidence.p_slip` (no ZUPT/ZARU at P ≥ 0.5). Division of roles: reliability r → sensor
  updates (R, skip); P_slip → physical constraints (NHC, ZUPT, Q).
- **Closed-loop variants (same inputs, held-out models):** fixed (r=1, P=0); pslip (r=1, P learned);
  lstm (r learned, P=0) = F10; **full (r learned, P learned)**; oracle_full (r labels, P = GT slip).
  fixed and lstm must reproduce F10 numbers exactly (regression check).
- **Look criteria (development, written now):** P_slip held-out AUC on clean runs ≥ 0.80 (3B rule
  S1); full vs lstm on clean runs and on fault runs reported (wins, median change); pslip vs fixed
  shows the value of constraint gating alone (an ablation and the "simple slip handling" baseline).
  Nothing is tuned on these numbers; parameter values come from calibration sessions later.

**F11 implemented (2026-10-10 21:40 IST, cloud workspace; mechanics tested on Paper_ws data only):**
`learning/train_slip.py`, `learning/closed_loop_int.py`, `learning/run_int.sh`. Regression:
closed_loop_int `fixed` and `lstm` equal closed_loop.py on all 81 mock runs (max diff 0.0).
1-epoch smoke training and the 5-variant closed loop run end to end (numbers not used).

- **F11 on the VM:** files verified by checksum, commit 788ebfd; `run_int.sh 3` started 2026-10-10
  22:07 IST (log `results/fullsys_int_run.log`).

- **F11 P_slip LOBO (development, training 22:06–00:15 IST):** held-out AUC on clean runs mean
  0.929 (min 0.861); AUC all runs 0.926; false alarm (P ≥ 0.5 without slip) on clean runs 7.1 %.
  Look criterion AUC ≥ 0.80: **MET** in every fold (min 0.861). Phase 3B 3-state model: 0.985 (test).

- **F11 closed loop (development, 00:15–00:24 IST, `results/fullsys_int_closed_loop.csv`):**

| clean run | slip % | P AUC | P≥0.5 % | fixed | pslip | lstm | full | oracle_full |
|---|---|---|---|---|---|---|---|---|
| D1 130530 | 18.9 | 0.941 | 20.1 | 0.238 | 0.280 | 0.109 | 0.127 | 0.112 |
| D2 131512 | 1.1 | 0.877 | 13.8 | 0.042 | 0.099 | 0.019 | 0.054 | 0.031 |
| D3 143058 | 5.4 | 0.861 | 18.6 | 2.173 | 2.162 | 1.898 | 1.858 | 0.119 |
| D4 143931 | 13.0 | 0.922 | 14.6 | 0.549 | 1.133 | 0.204 | 0.226 | 0.137 |
| D5 145610 | 9.3 | 0.965 | 11.0 | 0.307 | 0.452 | 0.146 | 0.104 | 0.136 |
| D6 150047 | 23.9 | 0.931 | 24.0 | 1.373 | 2.892 | 0.369 | 0.361 | 0.162 |
| D7 150428 | 8.0 | 0.959 | 8.2 | 0.191 | 0.176 | 0.126 | 0.099 | 0.086 |
| D8 152122 | 10.0 | 0.974 | 14.6 | 0.488 | 0.467 | 0.151 | 0.143 | 0.090 |
| mean | | | | 0.670 | 0.958 | 0.378 | 0.372 | 0.109 |

  Fault runs (mean RMSE): full ≈ lstm in every group (e.g. imu bias 3.09 vs 3.02, odom noise 0.72 vs 0.86).
  pslip vs fixed: clean 3/8, median +32.4 %; faults 86/208, +2.5 %.
  full vs lstm: clean 5/8, −2.0 %; faults 114/208, −1.0 %.
  full vs fixed: clean 7/8, −53.7 %; faults 163/208, −51.9 % (fixed and lstm columns = F10 exactly).
- **Interpretation (development):**
  1. **P_slip constraint gating adds ≈ nothing on top of slip-aware reliability** (−1 to −2 %, ~55 %
     wins). The F10 reliability already skips wheel updates during slip; gating NHC/ZUPT as well is
     largely redundant. Even ground-truth slip adds little: label-driven r alone 0.131 (F10) vs
     oracle_full 0.109.
  2. **P_slip alone HURTS** (pslip +32 % on clean runs): with provisional hook parameters, P ≥ 0.5
     switches NHC/ZUPT off 8–24 % of the time — including false alarms (D2 rumble strips: 13.8 % high
     with 1.1 % slip) — while slipping wheels are still trusted. Not a fair "slip baseline" as is
     (parameters untuned); a calibrated rule-based baseline is still needed (literature item).
  3. Remaining headroom is in r itself, mostly D3 (icy slide): lstm 1.90 vs oracle 0.12 m. Paper_ws
     fixed this case with LiDAR scan matching + 5c (ice slide 3.2 → 0.17 m / 0.01 m) — LiDAR is not
     yet in these closed loops.
  Same pattern as Phase 3B: slip knowledge alone does not help without a measurement that can use it.
- **Decision (2026-10-11 00:55 IST, development, reversible):** the system design goes forward with
  reliability r (F10) and WITHOUT learned P_slip gating of NHC/ZUPT/Q (hooks fed nothing, as F10).
  `full` and `pslip` stay as reported ablations. Code is kept. Revisit only if LiDAR integration
  changes the picture. Next: LiDAR scan matching + 5c in the closed loop, then LiDAR faults / r_LiDAR,
  then baselines (Sage–Husa; calibrated rule-based NHC/odom adaptation).

## F12 — LiDAR scan matching + 5c with learned reliability (design 2026-10-11 01:20 IST, before running)
**Why:** remaining gap after F10/F11 is mainly D3 (icy slide: lstm 1.90 m vs label-driven 0.12 m);
Paper_ws fixed slides with scan matching + 5c (change 1). LiDAR has not been in any F9–F11 loop.
- **Filter = Paper_ws change 1 unchanged** (`eval_change1.pipeline` logic): EkfLidar (stochastic
  cloning), detector with LiDAR evidence (Tier 3 + stuck path), NHC, ZUPT/ZARU, provisional dev
  parameters. No accel bias injected (as F10; Paper_ws injected 0.03/−0.02). 5c = two passes
  (segments from pass 1 gate the wheels in pass 2). LiDAR motion check uses the (faulted) odometry.
- **Reliability:** F10 models and features unchanged (features from the no-LiDAR parallel run;
  LiDAR-based features / r_LiDAR come with LiDAR faults later). Wheels skipped when r_odom < 0.5 or
  inside a 5c segment; gyro skipped when r_imu < 0.5.
- **Variants:** fixed_L, fixed_L5c, lstm_L, lstm_L5c, oracle_L (+ F10 no-LiDAR fixed/lstm merged).
- **Regression:** fixed_L and fixed_L5c equal Paper_ws pipeline (bias 0) exactly on mock S1
  (0.13728 / 0.08920 m; matches docs_paperws/change1.md S1 0.137 / 0.089).
- **Look criteria (development, written now):** (a) lstm_L5c vs fixed_L5c on clean and fault runs
  (wins, median change) — does learned reliability still help once LiDAR + 5c are in? (b) D3 clean:
  does LiDAR close the gap to the label-driven result? (c) lstm_L5c vs lstm_L (value of 5c with r).
  Reported, not tuned.

- **F12 on the VM:** files verified by checksum, commit 5d1ca01; `run_lidar.sh 3` started 2026-10-11
  01:03 IST (log `results/fullsys_lidar_run.log`).

- **Backup 2026-10-11 ~01:00 IST:** GitHub KushagraS07/BTP commit 2fb6c45 (F11 code, P_slip models
  `fullsys_logs/models_f11`, results). VM findings copy still ends at F10; project copy is complete.

- **F12 result (development, 01:03–01:47 IST, `results/fullsys_lidar_closed_loop.csv`):**

| clean run | fixed | lstm | fixed_L | fixed_L5c | lstm_L | lstm_L5c | oracle_L |
|---|---|---|---|---|---|---|---|
| D1 130530 | 0.238 | 0.109 | 0.232 | 0.191 | 0.136 | 0.127 | 0.143 |
| D2 131512 | 0.042 | 0.019 | 0.055 | 0.055 | 0.037 | 0.037 | 0.028 |
| D3 143058 | 2.173 | 1.898 | 1.551 | 0.114 | 0.422 | 0.077 | 0.090 |
| D4 143931 | 0.549 | 0.204 | 0.475 | 0.263 | 0.229 | 0.199 | 0.212 |
| D5 145610 | 0.307 | 0.146 | 0.323 | 0.323 | 0.091 | 0.091 | 0.116 |
| D6 150047 | 1.373 | 0.369 | 0.901 | 0.670 | 0.492 | 0.479 | 0.408 |
| D7 150428 | 0.191 | 0.126 | 0.207 | 0.207 | 0.134 | 0.134 | 0.122 |
| D8 152122 | 0.488 | 0.151 | 0.508 | 0.242 | 0.152 | 0.099 | 0.084 |
| mean | 0.670 | 0.378 | 0.532 | 0.258 | 0.212 | 0.155 | 0.150 |

  Fault runs (mean RMSE): fixed_L5c → lstm_L5c: imu bias 8.23 → 1.66, drift 3.94 → 2.01, dropout 2.19 → 0.75,
  noise 2.27 → 0.44; odom bias 5.46 → 0.58, drift 2.25 → 0.42, dropout 0.29 → 0.18, noise 0.96 → 0.48.
  lstm_L5c vs fixed_L5c: clean 8/8, median −33.6 %; faults 185/208, median −58.1 %.
  lstm_L vs fixed_L: clean 8/8, −48.6 %; faults 188/208, −67.4 %. lstm_L5c vs lstm_L: clean 5/8, −4.7 %; faults ±0.
  fixed_L vs fixed: clean 4/8, +0.8 %; faults 185/208, −29.5 %.
- **Interpretation (development):**
  (a) **Learned reliability keeps a large gain with LiDAR + 5c in the filter** (8/8 clean, 185/208 faults).
  (b) **D3 icy slide gap closed**: 1.898 (lstm, no LiDAR) → 0.077 (lstm_L5c), at the label-driven level (0.090);
      5c does most of it (fixed_L5c 0.114).
  (c) 5c adds a little on top of r (−4.7 % clean, 0 on faults). Full system lstm_L5c mean clean 0.155 m vs
      fixed no-LiDAR 0.670 (−77 %) and vs fixed_L5c 0.258.
  (d) LiDAR alone does not help clean runs (+0.8 %, corridor degeneracy) but helps under faults (−29.5 %).
  (e) Small regressions from adding LiDAR to lstm on some clean sessions (D1 0.109 → 0.127, D2 0.019 → 0.037,
      D6 0.369 → 0.479) — noted; LiDAR reliability (r_LiDAR) is the open item.
  (f) Remaining gaps to the label-driven reference: IMU dropout (0.75 vs 0.19) and IMU drift (2.01 vs 1.48);
      clean runs are already at the reference (0.155 vs 0.150) → little headroom for P_slip/Q(t) on clean
      runs; F13-B (r_imu → Q) targets the IMU gap.

## F13 — Q(t) with LiDAR + 5c (design 2026-10-11 02:55 IST, before code or running)
**Question:** does process-noise adaptation help once an absolute-type measurement (scan matching) and
5c are in the filter? (Phase 3B: Q(t) by P_slip did not help a filter WITHOUT absolute measurements.)
- **Base:** lstm_L5c of F12 (F10 reliability, gating, LiDAR scan matching, 5c). Must reproduce F12 exactly.
- **Mechanism (no new parameter):** Paper_ws hook `Ekf9.predict(t, x)`: Q_ω ← Q_ω·(1 + k_w_slip·x),
  k_w_slip = 50 (provisional dev value, unchanged). In EKF9 the gyro is a measurement and ω a random
  walk, so Q_ω sets how fast ω may follow the other heading sources (wheels, scan matching).
- **Variants:**
  A_q   x = P_slip (F11 models)                              — slip-driven Q only
  A_all x = P_slip in all hooks (Q, NHC skip/loosen, no ZUPT at P ≥ 0.5) — F11 'full' + LiDAR + 5c
  B_q   x = 1 − r_imu (F10 models)                            — IMU-unreliability-driven Q
- **Decision rule (written now):** a variant is ADOPTED into the design only if, vs base, it wins on
  clean runs ≥ 6/8 AND on fault runs ≥ 60 % of 208 AND median change ≤ −3 % on both. Otherwise Q(t)
  stays out (reported as negative). One run; no change of k or of the form after seeing results.
- Expectation stated beforehand: little headroom on clean runs (base 0.155 vs label-driven 0.150);
  B_q targets the IMU dropout/drift gap.

**F13 implemented (2026-10-11 03:05 IST, cloud workspace):** `learning/closed_loop_q.py`, `learning/run_q.sh`.
Regression: `base` equals F12 lstm_L5c exactly on mock S1 (max diff 0.0); all variants run (smoke test,
numbers not used). The script also prints the regression vs F12 on the VM and applies the decision rule.

- **Backup 2026-10-11 03:07 IST:** GitHub KushagraS07/BTP commit 272a30b (F12 code + results).

- **F13 on the VM:** files verified by checksum, commit 8edc6d1; `run_q.sh 3` started 2026-10-11 03:10 IST
  (log `results/fullsys_q_run.log`).

- **F13 result (development, 03:10–04:20 IST, `results/fullsys_q_closed_loop.csv`):** regression base vs
  F12 lstm_L5c max |diff| 4.4e-16 m over 216 runs.

| clean run | base | A_q | A_all | B_q |
|---|---|---|---|---|
| D1 130530 | 0.127 | 0.123 | 0.115 | 0.123 |
| D2 131512 | 0.037 | 0.040 | 0.044 | 0.042 |
| D3 143058 | 0.077 | 0.077 | 0.071 | 0.077 |
| D4 143931 | 0.199 | 0.197 | 0.123 | 0.214 |
| D5 145610 | 0.091 | 0.080 | 0.051 | 0.101 |
| D6 150047 | 0.479 | 0.483 | 0.361 | 0.472 |
| D7 150428 | 0.134 | 0.134 | 0.096 | 0.134 |
| D8 152122 | 0.099 | 0.113 | 0.092 | 0.098 |
| mean | 0.155 | 0.156 | 0.119 | 0.158 |

  Fault means (base → A_all / B_q): imu bias 1.656 → 1.286 / 1.304; imu drift 2.009 → 1.657 / 1.485;
  imu dropout 0.751 → 0.496 / 0.585; imu noise 0.438 → 0.399 / 0.361; odom groups ≈ unchanged.
  Decision rule (pre-stated): A_q clean 5/8, −0.1 %; faults 134/208, −0.4 % → not adopted.
  **A_all clean 7/8, −16.9 %; faults 158/208, −11.4 % → ADOPTED.** B_q clean 5/8, −0.0 %; faults 159/208,
  −0.7 % → not adopted.
- **Interpretation (development):**
  1. **With LiDAR + 5c in the filter, learned P_slip gating of the constraints helps** (mean clean 0.155 →
     0.119 m; 7/8 sessions), whereas without LiDAR it added ≈ 0 (F11). This supports the Phase 3B
     hypothesis: slip knowledge pays off only once an absolute-type measurement can correct the estimate.
  2. The gain comes from the NHC / ZUPT gating, not from Q alone: A_q (Q_ω only) ≈ 0. So "Q(t)" by itself
     stays out; P_slip acting on Q + NHC + ZUPT is adopted.
  3. B_q (1 − r_imu → Q_ω) lowers mean IMU-fault errors (bias −21 %, drift −26 %) but not the medians →
     not adopted by the rule; reported, possible future work.
  4. Cost: D2 (rumble strips, P_slip false alarms) 0.037 → 0.044.
- **Decision (2026-10-11 04:35 IST):** current design = F10 reliability (gating) + LiDAR scan matching + 5c
  + **learned P_slip on the constraints (Q_ω, NHC, ZUPT/ZARU via the detector)**. F11 decision ("go without
  P_slip") is superseded by this pre-stated rule, as anticipated in F11 ("revisit if LiDAR changes the
  picture"). Final confirmation only in the pre-registered test.

## F14 — baselines, sensor ablations and the metrics package (design 2026-10-12 00:45 IST, before code)
All variants on the SAME filter infrastructure as F13 (EKF9 + LiDAR scan matching + detector + 5c two-pass),
same runs (8 held-out sessions × 27), same models; only the way sensor trust is set differs.
- **Methods:**
  fixed       r = 1, no P_slip                                 (= F12 fixed_L5c; regression)
  chi2        r = 1; odometry / gyro update skipped if NIS > χ²(0.99): 9.21 (2 dof) / 6.63 (1 dof)
  sage_husa   r = 1; per-axis R of odometry and gyro estimated online (Sage–Husa, forgetting b = 0.98,
              R clamped to [0.1, 1000] × R₀); standard textbook values, not tuned
  heuristic   r_odom = 0 when |ω_wheel − ω_gyro| ≥ 0.30 rad/s (Phase 3B frozen slip threshold), else 1;
              r_imu = 1 (rule-based slip cue; no learning)
  ours        F10 r + P_slip on Q/NHC/ZUPT                     (= F13 A_all; regression)
  hybrid      ours + chi2 skip                                 (development variant)
  oracle      label r + ground-truth slip                      (reference, not deployable)
- **Ablations of ours:** odom_only (r_imu = 1), imu_only (r_odom = 1), no_pslip (= F12 lstm_L5c; regression),
  no_5c (single pass).
- **Metrics (defined now):** position RMSE (start pose known → no alignment, = ATE); max and final position
  error; heading RMSE (wrapped); RPE = RMSE of 5 s displacement error (windows every 1 s);
  **divergence** = max position error > 2.0 m; NEES on (x, y, θ) — mean and % inside the χ²(3) 95 % band
  [0.216, 9.35]; NIS of odometry and gyro updates — mean and % above χ²(0.95); share of skipped updates.
- **Decision rules (written now):** ours is reported against each baseline (clean / fault wins, median
  change, divergence). **hybrid adopted** only if vs ours: clean wins ≥ 6/8, fault wins ≥ 60 %, median ≤ −3 %
  on both; otherwise future work. Ablations are descriptive (no adoption decision).
- **F14b (separate step):** Li et al.-style baseline (motion-state classifier → ZUPT/NHC noise schedule),
  built from the paper's equations.

**F14 implemented (2026-10-12 01:30 IST, cloud workspace):** `learning/baselines.py`, `learning/run_baselines.sh`.
Regression on mock S1 (2 runs): fixed = F12 fixed_L5c, no_pslip = F12 lstm_L5c, ours = F13 A_all, all
max |diff| 0.0. All 11 methods and all metrics run (smoke numbers not used; note: χ² gating strong on the
mock slip session, 0.020 vs 0.089 m).

- **Backup:** GitHub KushagraS07/BTP commit 1c914c4 (F13 code + results), confirmed 2026-10-12 11:25 IST.

- **F14 on the VM:** files verified by checksum, commit 8d5b3b0; `run_baselines.sh 3` started 2026-10-12
  11:27 IST (log `results/fullsys_baselines_run.log`).

- **F14 result (development, 11:27–14:47 IST, `results/fullsys_baselines.csv`; summary
  `results/fullsys_baselines_summary.txt`).** Regressions: fixed vs F12 fixed_L5c 7.1e-15, no_pslip vs F12
  lstm_L5c 4.4e-16, ours vs F13 A_all 4.4e-16 m (216 runs). The summary printed by the run had a pivot bug
  (pivot over run × type × sensor with dropna=False → 648 "clean" rows, nan medians) in the comparison lines
  only; fixed (commit 53df5be, pivot by run) and the summary re-computed from the saved CSV — no re-run.

| method | clean RMSE | clean NEES in 95 % | fault RMSE | fault max err | fault diverged | fault NEES in 95 % |
|---|---|---|---|---|---|---|
| fixed | 0.258 | 0.388 | 3.076 | 7.101 | 0.673 | 0.308 |
| heuristic | 0.275 | 0.395 | 2.997 | 6.592 | 0.505 | 0.343 |
| sage_husa | 0.205 (div. 0.125) | 0.627 | 0.876 | 2.236 | 0.365 | 0.497 |
| chi2 | 0.183 | 0.658 | 0.895 | 2.253 | 0.341 | 0.565 |
| ours | 0.119 | 0.766 | 0.662 | 1.540 | 0.264 | 0.583 |
| **hybrid (ours + χ²)** | **0.104** | 0.768 | **0.565** | 1.349 | 0.255 | 0.646 |
| oracle (reference) | 0.089 | 0.763 | 0.560 | 1.258 | 0.197 | 0.666 |
| ablation odom_only (r_imu = 1) | 0.119 | 0.766 | 2.088 | 4.393 | 0.442 | 0.539 |
| ablation imu_only (r_odom = 1) | 0.272 | 0.397 | 1.685 | 4.177 | 0.505 | 0.338 |
| ablation no_pslip | 0.155 | 0.706 | 0.786 | 1.824 | 0.293 | 0.548 |
| ablation no_5c | 0.176 | 0.658 | 0.753 | 1.687 | 0.284 | 0.498 |

  ours vs (wins clean / faults, median change clean / faults): fixed 8/8, 190/208, −49.7 / −63.8 %;
  heuristic 8/8, 184/208, −52.3 / −46.5 %; chi2 6/8, 137/208, −27.8 / −19.8 %; sage_husa 5/8, 113/208,
  −20.3 / −8.1 %; no_pslip 7/8, 158/208, −16.9 / −11.4 %; no_5c 5/8, 98/208, −4.8 / 0.0 %;
  imu_only 8/8, 174/208, −48.8 / −40.6 %; odom_only 4/8, 136/208, −0.1 / −0.1 % (but IMU-fault means
  1.29 vs 8.44 m for bias, 0.50 vs 1.58 dropout: r_imu matters for IMU faults; medians dominated by odom faults).
  **Decision rule (pre-stated): hybrid vs ours clean 7/8, −11.3 %; faults 170/208, −7.6 % → ADOPTED.**
- **Interpretation (development):**
  1. Ours beats every non-learned baseline on both clean and fault runs; margin largest vs fixed and the
     rule-based slip heuristic (≈ −50 %), smaller vs χ² gating (−20 to −28 %) and smallest vs Sage–Husa
     (5/8 clean, 113/208 faults, −8 % median on faults). Sage–Husa is the closest baseline per run; it diverged
     on 1/8 clean runs (12.5 %) while ours diverged on none.
  2. χ² innovation gating is a strong, cheap baseline and is COMPLEMENTARY to learned reliability: the hybrid
     reaches the oracle level on faults (0.565 vs 0.560) and is the best deployable method on clean runs.
     Learned r acts before an innovation becomes large (slip, drift onset); χ² removes residual gross outliers.
  3. Consistency: fixed EKF9 is strongly over-confident (clean NEES median 30.5, 39 % inside the 95 % band);
     ours 3.7 / 77 %, hybrid 3.0 / 77 % → adaptive trust also makes the covariance credible (paper point).
  4. Ablations: r_odom drives the clean-run gain (slip); r_imu drives IMU-fault gains; P_slip adds −17 / −11 %;
     5c adds little on top (clean −4.8 %, faults 0).
  5. Divergence (> 2 m) under faults stays 20–26 % even for oracle → driven by the very large F7 severities
     (F8 note 2); must be reported per severity.
  6. χ² skips 19 % of gyro updates on fault runs (similar to ours 18 %); hybrid odom skip 32 % on faults.
- **Decision (2026-10-12 16:10 IST):** design = F10 reliability (gating) + χ² innovation gate (99 %) + LiDAR
  scan matching + 5c + learned P_slip on the constraints. "ours" in later steps = hybrid. Final confirmation
  only in the pre-registered test. F14b compares Li against both ours and hybrid (li_baseline.py amended
  before running; sha256 fbd967353f353795).

## F14b — Li et al. (IEEE Sensors J. 2026)-style baseline (design 2026-10-12 11:45 IST, before code)
Faithful to the paper where it gives values; adaptations for a differential-drive robot without GNSS are
stated here and fixed before any result.
- **Classifier (paper §III-A):** 3-layer BiLSTM (128, 64, 32 units; dropout 0.2 after layers 1–2), FC 64 +
  dropout 0.3, softmax over {stationary, straight, turning}; window 30 steps; cross-entropy, Adam 1e-3.
  Adaptations: (i) bidirectional over a CAUSAL window ending at t (no future samples — same information as
  our model); (ii) input 5-D (acc x/y/z clipped ±25, gyro z, wheel v) — gyro x/y omitted (planar robot);
  (iii) training protocol = ours (LOBO, val = first remaining, 15 epochs, best-val ckpt, stride 5, all 27
  runs per session, seed 42) instead of "stop at 95 % val accuracy".
- **Labels (paper eq. 15 adapted; GT for labels only):** stationary if v_h < 0.02 m/s and |ω| < 0.02 rad/s;
  turning if |ω| > 0.10 rad/s (in-place turns included: a differential-drive robot turns with v ≈ 0, the
  paper's car condition v_h > τ_v,min and centripetal acceleration does not apply); else straight.
- **Noise schedules (paper eqs. 17–18, constants as published):** ZUPT applied only if P_s ≥ γ_s = 0.7, with
  R_zupt = R0·[1 + 49·((1 − P_s)/(1 − γ_s))^β], β = 2 (ratio R_max/R_min = 0.5/0.01 = 50 kept; absolute
  values of our R0 kept since units/platform differ); otherwise ZUPT suspended. NHC: R_nhc = R0·exp(α·
  max(0, P_t − γ_t)/(1 − γ_t)), α = 2.5, γ_t = 0.6. No ZARU (paper has none). The motion classifier replaces
  our stationarity detector for ZUPT; 5c uses P_s ≥ 0.7 as its stationary flag.
- **Sensor updates:** fixed R (the paper adapts constraints only). Same EKF9 + LiDAR + 5c otherwise.
- **Reported:** same metrics as F14; ours vs li (wins, median change). Regression: with the hooks off the
  code equals F14 'fixed'.

**F14b implemented (2026-10-12, cloud workspace; mechanics tested on mock data only):** `learning/train_li.py`,
`learning/li_baseline.py`, `learning/run_li.sh`. Hooks-off regression vs baselines `fixed` on mock S1: clean and
IMU-noise fault run, max |diff| 0.0. Oracle motion labels (GT) as P_s/P_t give 0.025 m (fixed 0.023) → schedule
code sane. A 1-epoch smoke classifier (accuracy 0.60) gives 1.30 m: ZUPT fired during motion — shows the
method's sensitivity to classifier errors; smoke numbers not used. Note: IMU faults act on gyro_z only (F7), so
the Li classifier inputs (accel + gyro z + wheel v) change only through gyro z / wheel v under faults.
Package f14b.tgz sha256 be56234ad1ceb33c (train_li 5c7b257d146d26ac, li_baseline ccff48ce29f36b66,
run_li.sh 0a5be7cb231ccdd7). Runs on the VM after F14 (needs results/fullsys_baselines.csv).
- **F14b on the VM:** files verified by checksum (all 4 match), commit ff48e3b (2026-10-12 11:56 IST); run waits for F14.

- **F14b result (development; run 2026-10-12 16:01–00:11 IST; log `results/fullsys_li_run.log`, CSV
  `fullsys_li_baseline.csv`, stats `fullsys_li_stats.csv`).** Regression hooks-off vs F14 fixed: 7.1e-15 m (216 runs).
  Classifier (held out, clean runs): accuracy 0.939 (0.890–0.980), recall stationary 0.946, straight 0.958, turning
  0.842 (D2 rumble strips: turning recall 0.44); best epoch 2–4 in every fold (val loss rises after → best-val ckpt).
  P_s ≥ 0.7 on 33.8 % of rows, P_t ≥ 0.6 on 14.0 %.

| method | clean RMSE | clean diverged | clean NEES in 95 % | fault RMSE | fault diverged |
|---|---|---|---|---|---|
| fixed | 0.258 | 0.000 | 0.388 | 3.076 | 0.673 |
| **li (Li et al.-style)** | **0.460** | 0.125 | 0.406 | **4.086** | 0.663 |
| ours | 0.119 | 0.000 | 0.766 | 0.662 | 0.264 |
| hybrid | 0.104 | 0.000 | 0.768 | 0.565 | 0.255 |

  hybrid vs li: clean 8/8, −60.2 % [−86.4, −29.4]; faults 191/208 (8/8 sessions), −78.7 % [−89.2, −66.7];
  Wilcoxon p = 0.0078 each (Holm within this 2-test family 0.0156; in the full 22-test table 0.172).
  ours vs li: clean 8/8 −51.4 %, faults 187/208 −72.7 %.
- **Interpretation (development):**
  1. The motion classifier itself works (94 % accuracy), but the Li-style filter is WORSE than fixed: clean
     0.460 vs 0.258 m (1/8 runs diverged), faults 4.09 vs 3.08 m. Constraint scheduling from a motion classifier
     does not address sensor faults (fixed sensor R, as in the paper), and errors of a classifier-driven ZUPT
     (≈ 5 % of moving rows misclassified, ZUPT replaces our independent detector, no ZARU) inject false zero-
     velocity updates. Our detector uses LiDAR evidence and is conservative (0 false stationary in Paper_ws tests).
  2. Fairness caveats for the paper: this is an ADAPTATION (differential-drive robot, no GNSS, our EKF9, our
     training protocol, constants as published); Li et al. target vehicles with GNSS outages. The comparison
     shows that motion-state constraint scheduling is not a substitute for sensor-reliability estimation, not
     that the original method fails in its setting. Wording: "Li et al.-style constraint scheduling".
  3. Possible extra check (NOT run, would need its own pre-stated rule): li with our detector for ZUPT
     (classifier only for NHC) to separate the ZUPT-source effect.

## A2 — statistics for comparisons (rules written 2026-10-12 16:10 IST, before code or any output)
Runs within a session share the trajectory → not independent. Unit of inference = SESSION (n = 8 here).
- **Effect size:** per comparison (method X vs reference Y) and group (clean / fault): ratio
  q = median over runs of RMSE_X / RMSE_Y − 1 (same as the "median change" already reported).
- **95 % CI:** cluster bootstrap — resample the 8 sessions with replacement (all their runs together),
  recompute q; 10 000 resamples, seed 0, percentile interval.
- **Test:** per-session statistic d_s = median over that session's runs of log(RMSE_X / RMSE_Y); two-sided
  Wilcoxon signed-rank over the 8 d_s (exact). With n = 8 the smallest possible p is 0.0078.
- **Multiplicity:** Holm correction over the comparisons of one table (one family per step).
- Report: q, CI, sessions improved (d_s < 0) / 8, p, Holm-adjusted p. Descriptive in development; the same
  procedure is pre-registered for the final test. Applied to F14 (ours = hybrid vs every method, and
  ablations) and to F14b. No change to any result or method.

**A2 implemented (2026-10-12, cloud workspace):** `learning/stats.py` (104 lines, sha256 f9e0b39e78e49776).
Exact Wilcoxon equals scipy `wilcoxon(method='exact')` on 5 random cases; Holm checked by hand; synthetic
table runs (≈ 2 s per comparison). Note: on clean runs n = 1 run per session, so CI and test use 8 runs.

- **A2 on F14 (development, VM commit 22616e2; `results/fullsys_baselines_stats.{txt,csv}`), hybrid vs X,
  median change % [95 % CI], sessions better /8:**
  clean: fixed −56.8 [−65.8, −42.2] 8; chi2 −23.6 [−54.8, −10.7] 8; sage_husa −29.5 [−47.2, +9.9] 5;
  heuristic −57.3 [−68.5, −41.8] 8; ours −11.3 [−20.9, −1.1] 7; no_pslip −25.8 [−40.4, −12.8] 8;
  no_5c −24.2 [−44.2, −6.7] 7; oracle +3.8 [+0.3, +24.1] 1.
  faults: fixed −71.1 [−82.5, −58.9] 8; chi2 −32.0 [−39.4, −15.1] 8; sage_husa −21.0 [−35.2, −8.4] 7;
  heuristic −54.3 [−61.4, −47.6] 8; ours −7.6 [−11.0, −5.4] 8; no_pslip −25.5 [−33.8, −19.0] 8;
  no_5c −19.7 [−27.7, −12.6] 8; oracle +8.6 [−1.9, +20.9] 3.
  Wilcoxon p = 0.0078 (the minimum for n = 8) wherever all 8 sessions improve; **Holm over 20 comparisons →
  p_holm ≥ 0.156 everywhere** (20 × 0.0078). CIs exclude 0 for every baseline except Sage–Husa on clean runs.
- **Interpretation:** effects are consistent (8/8 sessions on most comparisons, CIs away from 0), but 8
  sessions cannot reach corrected significance for a 20-comparison family. **Consequence for the final
  pre-registration (design, not a result):** (1) a small PRIMARY family (e.g. hybrid vs fixed, χ², Sage–Husa,
  Li; faults and clean = 8 tests), ablations and oracle reported descriptively; (2) enough sessions: with m
  primary tests and Holm, need 2/2^n · m < 0.05 → m = 8 needs n ≥ 9 sessions; planned 12 final runs gives
  minimum p 0.00049 → 0.0039 after ×8. Sage–Husa on clean runs is the weakest comparison (5/8, CI crosses 0).

## A1 — physically scaled faults = held-out fault-magnitude test (design 2026-10-12 16:30 IST, before code)
**Why:** F7 severities are multiples of each signal's own std (gyro_z std ≈ 0.45 rad/s, dominated by turns), so
"small" faults are already large and even the oracle exceeds 2 m on 20 % of fault runs (F14). A reviewer will
ask how the method behaves on realistic, small faults. Because the models were trained ONLY on std-scaled faults,
testing them unchanged on σ-scaled faults is also the planned generalization test (held-out fault magnitude).
- **Fault set (no new parameter):** same 13 fault types × severities, same multipliers (noise 1/3/6/12, bias and
  drift 1/3/6, dropout 5/15/35 %), same onsets (seed from the F7 fault name). Only the UNIT changes:
  σ = √R0 of that channel from `ekf9_params_dev.yaml` (fixed before any result): gyro √r_gyro = 0.00873 rad/s
  (SDF noise); wheel v √r_odom_v = 0.05 m/s; wheel ω √r_odom_w = 0.10 rad/s. E.g. gyro bias 0.009 / 0.026 /
  0.052 rad/s; wheel v bias 0.05 / 0.15 / 0.30 m/s. Dropout unchanged (no scale). Run names get the suffix
  `_sigma`; features in `fullsys_logs/features_sigma/`; results `fullsys_baselines_sigma.csv`.
- **Models:** F10 reliability + F11 P_slip models UNCHANGED (no retraining, held-out sessions as before).
- **Methods:** fixed, chi2, sage_husa, heuristic, ours (= F13 A_all), hybrid, no_pslip. Oracle omitted (its
  1/(1+sev) labels have no meaning on the σ scale). Same filter, metrics and 5c as F14; A2 statistics.
- **Checks / look criteria (written now):** (a) clean runs identical to F14 (regression, max |diff| ≈ 0);
  (b) integrity reported: share of σ-fault runs where fixed RMSE ≥ 1.05 × clean (many small faults may be
  harmless — reported, not a stop rule); (c) **no-harm criterion: hybrid vs fixed on σ-fault runs median change
  ≤ 0 % AND wins ≥ 50 %** — learned reliability trained on large faults must not hurt on small ones;
  (d) hybrid vs χ² / Sage–Husa / heuristic reported with A2 statistics. Nothing tuned on these numbers.
- Runs on the VM after F14b. Default behaviour of every script unchanged (env var `FULLSYS_FAULT_UNIT=sigma`
  switches the set), so F7–F14 results stay reproducible.

**A1 implemented (2026-10-12 17:10 IST, cloud workspace):** `learning/patch_a1.py` (edits faults.py, build_dataset.py,
closed_loop.py, train.py, baselines.py; backups *.bak_preA1) and `learning/run_a1.sh` (features → 7 methods → A2 stats).
Tests on mock S1: default mode — all 26 faults byte-identical to pre-A1 (frames, labels, names, seeds); sigma mode —
onsets identical to the std set, bias added = sev × σ exactly (gyro 0.0087 / 0.0262 / 0.0523 rad/s, wheel v 0.05 / 0.15 /
0.30 m/s); end to end (features_sigma + baselines fixed/hybrid on 27 runs) runs; clean run identical to default mode
(fixed 0.022830, hybrid 0.077923 m). Mock numbers not used. Patch applied to pre-A1 copies reproduces the dev files
byte for byte. VM files must equal pre-A1 sha256 (faults c808033064728a3b, build_dataset 18182413723513df,
closed_loop 872c90d0291520a8, train 68bd9101d17ec0b2, baselines 41ee0e91debc22ba) → post (632a8e119bf377bf,
35b639c9522bdbb8, a61a5b7f2a1852c2, b9288a0dc277e5be, a9cda4e1f3dab6d6). Delivery after F14b (li_baseline imports
these modules when it starts).

**A1 amendment (2026-10-12 17:20 IST, before any A1 result): Q(t) retest `hybrid_bq`.** F13 B_q (q = 1 − r_imu)
lowered mean IMU-fault errors 21–26 % but failed the medians on std-scaled faults. Added as 8th A1 method:
hybrid + q = max(P_slip, 1 − r_imu) (Q_ω × (1 + 50 q); no new parameter; NHC/ZUPT still by P_slip).
**Rule (same as F13/F14):** adopt only if vs hybrid clean wins ≥ 6/8, fault wins ≥ 60 %, median ≤ −3 % on both;
otherwise reported as negative (Q alone not supported). Printed by baselines.py ("Decision rule (A1-Q)").
Tests: patch on pre-A1 copies reproduces dev files; mock hybrid unchanged (0.07792), hybrid_bq runs
(0.07791 clean). New sha256: baselines a1 5ed5d9b149d10e3c, patch_a1.py 6c92efcfd7aec5d8, run_a1.sh
9f1e692c866f4be1, package a1.tgz 65dbf29304118a65.

- **A1 result (development; VM commit 044d5be; run 2026-10-13 00:42–03:02 IST; `results/fullsys_a1_run.log`,
  `fullsys_baselines_sigma.csv`, `fullsys_baselines_sigma_stats.csv`).** (a) Clean runs identical to F14
  (fixed / no_pslip / ours vs F12/F13: 8.3e-17 m, 8 runs). (b) Integrity: 56/64 session × sensor × type groups
  raise fixed RMSE ≥ 1.05 × clean (below: mostly small IMU noise; some < 1, chance).

| method | clean RMSE | fault RMSE | fault max err | fault diverged | fault NEES in 95 % |
|---|---|---|---|---|---|
| fixed | 0.258 | 0.785 | 1.757 | 0.279 | 0.351 |
| heuristic | 0.275 | 0.666 | 1.430 | 0.173 | 0.365 |
| sage_husa | 0.205 | 0.626 | 1.519 | 0.240 | 0.511 |
| chi2 | 0.183 | 0.610 | 1.496 | 0.207 | 0.541 |
| no_pslip | 0.155 | 0.383 | 0.858 | 0.115 | 0.570 |
| ours | 0.119 | **0.324** | **0.750** | **0.091** | 0.612 |
| hybrid | **0.104** | 0.335 | 0.844 | 0.125 | 0.635 |
| hybrid_bq | 0.107 | 0.329 | 0.831 | 0.120 | 0.638 |

  A2 (hybrid vs X, σ faults, median % [95 % CI], sessions better /8): fixed −45.2 [−53.7, −39.1] 8; chi2 −23.9
  [−35.2, −14.8] 7; **sage_husa −15.1 [−32.1, +4.1] 4**; heuristic −44.8 [−49.7, −36.8] 8; ours −3.9 [−8.7, −1.5] 6;
  no_pslip −17.0 [−28.1, −10.6] 8; hybrid_bq −0.0 [−0.1, +0.3] 4. Holm (14 tests) ≥ 0.109.
  ours vs fixed on σ faults: 191/208 runs, −42.5 %. Rules: hybrid vs ours clean 7/8 −11.3 %, faults 147/208 −3.9 %
  → ADOPT (re-confirmed); **A1-Q hybrid_bq vs hybrid: clean 1/8 +1.0 %, faults 102/208 ±0.0 % → NOT adopted.**
- **Interpretation (development):**
  1. **(c) no-harm criterion MET by a wide margin:** on small, realistic faults learned reliability still cuts the
     error vs fixed by ≈ 45 % (8/8 sessions; ours 191/208 runs). **Generalization to a held-out (much smaller)
     fault magnitude holds** although the models saw only std-scaled faults (e.g. IMU dropout 2.19 → 0.42 m,
     IMU noise 0.40 → 0.14 m).
  2. **Q(t) on its own: final negative (development).** Retested on top of the hybrid with a pre-stated rule:
     no effect (±0 %). Q stays only inside the P_slip package (F13). Paper: "process-noise adaptation alone
     gives no measurable benefit once sensor reliability and constraint gating are present".
  3. **Sage–Husa is competitive on small faults** (4/8 sessions, CI crosses 0) — its mean is spoiled by IMU
     dropout (2.17 m: a held gyro value has small innovations, so covariance matching cannot see it), where
     learned reliability wins clearly (0.42). Report honestly: the gain over Sage–Husa is fault-type dependent.
  4. **Watch item — χ² on small odometry faults:** ours (no χ²) beats hybrid on means for odom bias / drift /
     noise (0.489 vs 0.558; 0.374 vs 0.472; 0.348 vs 0.407) and on divergence (0.091 vs 0.125); the χ² gate
     can reject valid wheel updates once a small bias has pulled the state away. Hybrid still wins the
     pre-stated rule (medians), so the design stays; the effect is to be reported and checked again in the
     final test (both ours and hybrid will be reported).

## A3 — seed robustness (design 2026-10-13 04:00 IST, before code was run on the VM)
Retrain reliability (F10 protocol) and P_slip (F11 protocol) with seeds 1 and 2 (seed 42 = existing models), same
data, LOBO, epochs, stride; closed loop ours / hybrid / no_pslip on the F14 (std) fault set. `fixed` is seed-free.
Seed switch `FULLSYS_SEED` (learning/runcfg.py; default 42 = unchanged): models → `fullsys_logs/models_seed<k>/`,
results → `results/seed<k>/`. **Rule (written now):** ROBUST if in every seed hybrid beats fixed on ≥ 7/8 clean and
≥ 80 % of fault runs AND the seed range of hybrid mean RMSE is < 25 % of the hybrid–fixed gap (clean and faults);
otherwise the paper reports seed variance as a limitation. Seed spread of ours / no_pslip reported descriptively.

## F14b-det — Li-style with our detector for ZUPT (design 2026-10-13 04:00 IST, before running)
Fairness check (reviewer question "is the Li loss caused by the classifier-driven ZUPT?"): same li models; ZUPT/ZARU
from OUR stationarity detector (as fixed), classifier P_t only for the NHC schedule (paper eq. 18). Descriptive, no
adoption decision; reported next to `li`. Env `FULLSYS_LI_VARIANT=li_det`; output `fullsys_li_det.csv`.

## D-A1 — divergence diagnostic (descriptive)
`learning/diag_diverge.py`: share of fault runs with max error > 2 m by sensor / type / severity and by session, for
fixed / ours / hybrid on the std (F14) and σ (A1) sets. Purpose: failure-case section; focus of the
"faults that end / recovery" step. No method change.

## Calibration design note (logged 2026-10-13 04:00 IST, from A1 finding 4; applied on day 6)
χ² gate settings become calibration CANDIDATES (selected on calibration sessions only, by the calibration rule):
gate on odometry + gyro (current) / gyro only; level 99 % / 99.9 %. Development design unchanged until then.

**A3 / F14b-det / D-A1 implemented (2026-10-13 04:20 IST, cloud workspace):** `learning/runcfg.py`, `patch_a3.py`
(train, train_slip, closed_loop, li_baseline; backups *.bak_preA3), `diag_diverge.py`, `a3_summary.py`, `run_a3.sh`,
`run_li_det.sh`. Tests on mock data: default paths/seed unchanged (hybrid 0.07792, li 1.29836 as before); seed 1
trains into models_seed1 / results/seed1 and closed loop runs; li_det runs; summary + rule tested on synthetic CSVs;
patch on pre-A3 copies reproduces dev files. VM pre-A3 sha256: train b9288a0dc277e5be, train_slip 0e24e437be967ea5,
closed_loop a61a5b7f2a1852c2, li_baseline fbd967353f353795; post: afbf3bf3e53e7dcd, 0a7c57f0fb9e2e3c,
125c502b03dd5473, 80ed228f012865f2; package a3.tgz 44fae8ccd9d1c028.

- **Backup 2026-10-13 15:07 IST:** GitHub KushagraS07/BTP commit b4b27f3 (fullsys code + git bundle up to VM 044d5be, all results incl. F14 / A2 / F14b / A1, all models). No file > 50 MB.

- **D-A1 result (2026-10-13 15:10 IST; descriptive).** Clean runs: 0 diverged for fixed / ours / hybrid in both sets.
  1. **Gate-boundary effect (std set):** IMU bias *small* diverges in 8/8 sessions for ours AND hybrid (fixed 8/8),
     while *medium* / *large* only 2/8. Severity 1 has label r = 1/(1+1) = 0.5 = exactly the skip gate, so the
     learned r hovers at the gate and a 0.45 rad/s gyro bias is often NOT skipped. Same pattern for IMU drift
     slow / medium (ramp labels cross 0.5 late): 0.62–0.88. A property of the Phase 2 label convention + gate 0.5,
     not of the network. On the σ set (physically small biases) IMU bias small / medium never diverge.
     → **Calibration candidates (logged now, decided on day 6):** gate 0.3 / 0.5 / 0.7. Paper: state the label–gate
     interaction; report per severity.
  2. **Session concentration:** hybrid divergences cluster in D6 (150047) 69 % std / 42 % σ and D7 (150428) 50 % /
     42 %; all other sessions ≤ 23 % std, ≤ 12 % σ. Both are indoor slip-world routes through the slip strip with
     turns (D6 also the heaviest slip, 23.9 %, and the residual gyro gating of F10b). Hypothesis (to check in the
     LiDAR-fault / recovery steps): corridor LiDAR degeneracy leaves no absolute heading correction once slip + a
     fault have rotated the estimate. No change now.
  3. **χ² on small wheel faults confirmed:** σ odom bias small diverges hybrid 3/8 vs ours 1/8 vs fixed 1/8; odom drift
     slow hybrid 2/8 vs ours 0/8 → supports the χ² calibration candidates (gyro-only gate / 99.9 %).

- **F14b-det result (development; VM b859b73; run 15:09–15:33 IST; `fullsys_li_det.csv`, `fullsys_li_det_stats.csv`).**
  li_det (our detector for ZUPT, classifier P_t for NHC): clean 0.260 m (fixed 0.258, li 0.460), 0 diverged;
  faults 3.045 m (fixed 3.076, li 4.086). hybrid vs li_det: clean 8/8, −55.7 % [−67.8, −43.1]; faults 189/208
  (8/8 sessions), −70.7 % [−82.4, −58.9]; p = 0.0078 each.
  **Interpretation:** the extra loss of `li` vs fixed came from the classifier-driven ZUPT (false stops); with our
  detector the Li-style filter equals fixed — the NHC schedule from the motion classifier adds ≈ nothing here.
  Either way constraint scheduling does not address sensor faults; the hybrid's advantage (−56 % / −71 %) is not an
  artefact of the ZUPT source. Paper: report li and li_det; claim "motion-state constraint scheduling is not a
  substitute for sensor-reliability estimation".
- **A3 progress:** seed 1 reliability training 15:33–17:15 IST (1 h 42 min); P_slip training started 17:15.

- **A3 result (development; VM b859b73; run 2026-10-13 15:33 – 2026-10-14 00:39 IST; `results/seed1/`, `results/seed2/`,
  `fullsys_a3_run.log`). Rule → ROBUST.**

| seed | hybrid clean | hybrid fault | ours clean | ours fault | no_pslip clean | no_pslip fault | hybrid wins vs fixed (clean / fault) | hybrid vs ours (clean / fault) |
|---|---|---|---|---|---|---|---|---|
| 42 | 0.104 | 0.565 | 0.119 | 0.662 | 0.155 | 0.786 | 8/8, 190/208 | 7/8 −11.3 %, 170/208 −7.6 % |
| 1 | 0.108 | 0.549 | 0.119 | 0.663 | 0.158 | 0.806 | 8/8, 190/208 | 7/8 −9.2 %, 167/208 −6.7 % |
| 2 | 0.107 | 0.549 | 0.117 | 0.673 | 0.153 | 0.826 | 8/8, 189/208 | 7/8 −12.5 %, 169/208 −8.9 % |

  Seed range of hybrid mean RMSE: clean 0.104–0.108 (gap to fixed 0.152), faults 0.549–0.565 (gap 2.52) → < 3 % of the
  gap. Hybrid ADOPT decision (vs ours) reproduced in all 3 seeds; P_slip gain (no_pslip → ours) in all 3 seeds.
  **Interpretation:** results do not depend on the training seed; seed variance is negligible relative to the effects.

- **Backup 2026-10-14 00:47 IST:** GitHub commit 8f7062d (A3 seeds + models_seed1/2, F14b-det, diagnostic; VM b859b73).

## R1 — reliability quality as a detector + runtime (design 2026-10-14 00:55 IST, before code)
Held-out models (seed 42, LOBO) on the existing feature files (std set; σ set reported too). Per sensor (gyro, wheels):
- **Truth:** T_fault = rows where the injected fault is active on that sensor (onset → end / dropout window); clean rows
  of fault runs before onset and all clean-run rows are negatives. For wheels also T_gate = label r_odom < 0.5
  (includes physical slip, F10).
- **Detectors:** learned score 1 − r (flag r < 0.5 = the gate); threshold detector = NIS of the parallel fixed-R EKF9
  (the F7 features `gyro_nis`, `odom_nis`), flag NIS > χ²(0.99) (6.63 / 9.21).
- **Metrics:** AUC (rank, threshold-free); precision / recall / F1 at the operating points; false-alarm share on clean
  runs; detection delay = time from onset to the first flagged row (median over fault runs, per type).
- **Runtime:** per-step inference of reliability LSTM + P_slip LSTM (one window, batch 1, 1 thread; 2000 calls) and
  filter step (closed-loop time / IMU rows, fixed and ours); median and 95th percentile vs the 20 ms budget at 50 Hz.
- **Look criteria (written now, descriptive):** learned r AUC ≥ NIS AUC on both sensors (T_fault); clean false-alarm
  share of r ≤ that of NIS; p95 (2 LSTMs + filter step) < 20 ms.

## R2 — faults that end: detection, recovery, after-fault error (design 2026-10-14 00:55 IST, before code)
Same 26 faults, same onsets (seed from the F7 name), std scale, but every fault ENDS: active for D = 15 s after onset
(dropout: its own window, capped at 15 s; drift ramps to full severity at the end of the window, then removed).
Sessions are 65–88 s and onsets ≤ 60 %, so ≥ 11 s remain after every fault. Env `FULLSYS_FAULT_WINDOW=end`; run
names suffix `_end`; features `features_end/`; models unchanged (never trained on ending faults).
- **Methods:** fixed, chi2, sage_husa, ours, hybrid (F14 definitions, LiDAR + 5c).
- **Metrics per run:** RMSE whole run; **RMSE after the fault** (fault end → session end); final error; for ours:
  detection delay (onset → first r < 0.5 of the faulted sensor), **recovery delay** (fault end → r ≥ 0.5 held ≥ 1 s),
  share of the window gated; for chi2 the same from its skip pattern is not logged → NIS-based delay from features.
- **Look criteria (written now, descriptive):** median recovery delay of r ≤ 2 s; hybrid after-fault RMSE below fixed
  in ≥ 60 % of fault runs; D6 / D7 after-fault error reported separately (D-A1 hypothesis: no recovery there).

**R1 / R2 implemented (2026-10-14 01:25 IST, cloud workspace):** `learning/patch_r.py` (faults.py: FULLSYS_FAULT_WINDOW;
backup .bak_preR), `r1_quality.py`, `r2_eval.py`, `run_r.sh`. Tests on mock data: default fault set byte-identical to
pre-R (26/26 frames + labels); ending faults: unchanged before onset and after the end, labels back to 1 after the
end, dropout capped at 15 s, onsets identical; R2 clean runs equal F14 (fixed 0.022830, hybrid 0.077923 m);
R1 metrics + runtime run (mock: LSTM 0.31 ms / 0.37 ms p95 each, filter step ≈ 3 ms). Mock numbers not used.
VM faults.py pre 632a8e119bf377bf → post ff785a65afa137a7; package r.tgz eacb84bff27ab066.

- **R1/R2 on the VM:** checksums verified (faults.py post ff785a65afa137a7), commit e4882c0; `run_r.sh 3` started 2026-10-14 01:15 IST (log `results/fullsys_r_run.log`).

- **R1 result (development; VM e4882c0; 2026-10-14 01:15–02:48 IST; `fullsys_r1_quality*.csv`, `fullsys_r1_delay*.csv`).**

| set | sensor | AUC learned / NIS | precision learned / NIS | recall learned / NIS | clean false alarm learned / NIS |
|---|---|---|---|---|---|
| std | gyro | **0.959 / 0.780** | 0.990 / 0.639 | 0.719 / 0.476 | **0.2 % / 1.4 %** |
| std | wheels (fault truth) | 0.876 / 0.860 | 0.674 / 0.532 | 0.789 / 0.753 | 12.0 % / 8.3 % |
| std | wheels (gate-label truth, incl. slip) | **0.942 / 0.887** | 0.813 / 0.626 | 0.882 / 0.821 | — |
| σ | gyro | 0.669 / 0.576 | 0.733 / 0.432 | 0.021 / 0.068 | 0.2 % / 1.4 % |
| σ | wheels (fault) | 0.800 / **0.864** | 0.568 / 0.634 | 0.505 / 0.456 | 12.0 % / 8.3 % |
| end | gyro | 0.915 / 0.773 | 0.862 / 0.485 | 0.637 / 0.469 | — |
| end | wheels (fault) | 0.844 / **0.875** | 0.360 / 0.337 | 0.750 / 0.714 | — |

  Delay (std, median onset → first flag): gyro bias 0.12 s vs 0.01 s, noise 0.05 vs 0.01, drift 7.5 vs 7.8, dropout
  2.3 s (13/24 detected) vs 3.7 s (6/24); wheels bias 0.04 vs 0.02, drift 1.15 vs 1.34, dropout 1.47 (23/24) vs 2.04
  (21/24). Share of the fault window flagged: learned r much more persistent (gyro bias 0.80 vs 0.28).
  **Runtime:** reliability LSTM 0.75 ms (p95), P_slip LSTM 0.74 ms, filter step 1.67 ms (hybrid, mean) → 3.2 ms per
  step vs 20 ms budget at 50 Hz (16 % of budget, VM, 1 thread).
  **Look criteria:** gyro AUC + clean false alarm MET (all sets); wheels AUC met on std (0.876 > 0.860) but NOT on σ
  (0.800 < 0.864) and end (0.844 < 0.875); wheels clean false alarm NOT met (12.0 % vs 8.3 %) — by design the learned
  r_odom also flags physical slip on clean runs (slip-aware labels, F10); against the gate-label truth that includes
  slip, learned r is better (AUC 0.942 vs 0.887, precision 0.81 vs 0.63). Runtime MET.
- **R2 result (development; `fullsys_r2_end.csv`).** Regression clean vs F14: 1.1e-16 m (40 rows).

| method | RMSE | RMSE after fault | error at fault end | final error | diverged |
|---|---|---|---|---|---|
| fixed | 1.962 | 2.791 | 2.374 | 3.331 | 0.389 |
| chi2 | 0.666 | 0.994 | 0.552 | 1.150 | 0.173 |
| sage_husa | 0.700 | 1.047 | 0.567 | 1.291 | 0.216 |
| ours | 0.443 | 0.628 | 0.470 | 0.709 | 0.144 |
| **hybrid** | **0.375** | **0.531** | **0.375** | **0.593** | **0.120** |

  Learned r of the faulted sensor: detection gyro bias 0.12 s, noise 0.05, drift 4.5, dropout 1.6 (10/24 detected);
  wheels 0.03–1.5 s. **Recovery after the fault ends: median 0.67 s** (gyro 0.06–0.43 s; wheels 1.0–1.5 s), all
  detected faults recovered; share of post-fault time still gated 1–14 %. Hybrid after-fault RMSE < fixed in 188/208
  (90 %). **Both look criteria MET.** Per session (hybrid after-fault): D6 1.53, D5 0.66, D7 0.58, others 0.16–0.43.
- **Interpretation (development):**
  1. As a detector, learned r is clearly better than a NIS threshold for the gyro (AUC 0.96 vs 0.78, 6× fewer false
     alarms, flags persist through the fault) and better for wheels when slip counts as "untrustworthy"; for small
     (σ) wheel faults a NIS threshold is the better pure detector. In closed loop the learned r still wins (A1) — the
     gain comes from slip handling and persistent gating, not from detecting tiny faults. Report both honestly.
  2. Small gyro faults (σ) are hardly flagged by either detector (recall 2–7 %); they are also mostly harmless (A1).
  3. Trust recovers quickly once a fault ends (median 0.67 s) — no lock-out. D6 keeps the largest after-fault error
     for every method although r recovers → supports D-A1 hypothesis 2 (the estimate cannot be pulled back without
     absolute heading information), not a reliability-recovery problem.
  4. Gyro-bias after-fault error: chi2 0.70 < hybrid 0.81 < ours 1.11 — consistent with D-A1 point 1 (label-gate
     boundary at small severity); calibration of the gate (0.3 / 0.5 / 0.7) covers it.
  5. Runtime is 16 % of the 50 Hz budget → real-time capable on a laptop-class CPU (VM, 1 thread).

- **Backup 2026-10-14 02:53 IST:** GitHub commit de0cfd0 (R1 + R2 code and results; VM e4882c0).

## L — LiDAR faults, LiDAR gating, overlapping faults (design 2026-10-14 16:45 IST, before code)
LiDAR (robot.gazebo): 10 Hz, 540 beams, 0.10–2.50 m, σ = 0.001 m. Scans are faulted offline (the scan matcher AND the
detector's LiDAR evidence see the faulted scans). Onset uniform in [20 %, 60 %] of the session (seed from the fault
name, as F7), active to the end (freeze: one window). Severity values fixed now (physical units; σ multiples would be
harmless at σ = 1 mm):
  noise (Gaussian on valid returns) 0.01 / 0.03 / 0.10 m · return loss (beams → no return) 25 / 50 / 75 % ·
  scale error (ranges × (1+e)) 2 / 5 / 10 % · spurious returns (uniform in range) 5 / 10 / 20 % of beams ·
  freeze (scan held) 5 / 15 / 35 % of the session.   → 15 LiDAR faults per session.
**Overlapping:** LiDAR {noise 0.10 m, scale 5 %, freeze 35 %} × {odom bias medium, odom drift medium, IMU bias medium}
(F7 faults, own onsets, std scale) = 9 per session. Reliability / P_slip inputs: LiDAR does not enter the features,
so LiDAR-only runs use the clean feature files and overlap runs the F7 file of their odom / IMU fault (no rebuild).
**Methods (LiDAR + 5c as F14):** fixed; hybrid (current design); **hybrid_chi2L** = hybrid + χ² gate on the
scan-match update (NIS > χ²(3 dof, 0.99) = 11.34 → update skipped; standard value, not tuned); lidar_off = hybrid
without any LiDAR (no scan updates, detector without LiDAR evidence); oracle_L = hybrid with scan updates skipped while
the LiDAR fault is active (reference). Also hybrid_chi2L on the 208 F14 fault runs (no-harm check vs F14 hybrid).
**Rules (written now):**
  (1) **hybrid_chi2L ADOPTED** only if vs hybrid: LiDAR-fault runs wins ≥ 60 % and median ≤ −3 %; clean runs median
      change ≤ +2 %; F14 fault runs median change ≤ +2 %.
  (2) **Learned r_LiDAR is built** (new LiDAR features + training) only if, for the design D after (1), on LiDAR-fault
      runs oracle_L beats D by a median ≥ 10 % OR D is worse than lidar_off on ≥ 25 % of runs. Otherwise r_LiDAR is
      not built and this is reported (consistent with the pre-registered Phase 3A negative r_LiDAR result).
  Overlap runs: descriptive (same methods). Regression: fixed and hybrid clean runs equal F14.

**L design amendment (2026-10-14 17:40 IST, before any VM run; mock smoke only):** (a) the mock smoke showed that a
FROZEN LiDAR also corrupts the detector's LiDAR evidence (identical scans = "no motion"), so oracle_L is defined as
"LiDAR ignored while the fault is active": scan updates AND detector LiDAR evidence skipped. (b) Added a parameter-free
**stale-scan check** as a second candidate: a scan byte-identical to the previous one is stale (a real LiDAR always has
range noise, σ = 1 mm) → its scan-match update and LiDAR evidence are skipped. Methods: fixed, hybrid, hybrid_chi2L,
hybrid_stale, hybrid_chi2L_stale, lidar_off, oracle_L. **Rules:** (1a) hybrid_chi2L vs hybrid as before; (1b)
**hybrid_stale ADOPTED** only if vs hybrid on freeze runs wins ≥ 60 % and median ≤ −3 %, clean median ≤ +2 %, other
LiDAR faults median ≤ +2 %; design D = hybrid + adopted parts (hybrid_chi2L_stale if both); (2) unchanged, with D.
**L implemented:** `learning/patch_l.py` (baselines.run: lidar='on'|'chi2'|'off', scan_skip, stale; default path
unchanged; backup .bak_preL), `lidar_faults.py`, `l_eval.py`, `run_l.sh`. Tests on mock data: default and the new code
path give identical RMSE (diff 0.0); faulted scans unchanged before onset; oracle on a frozen LiDAR 1.90 → 0.013 m
(fixed r); rules print. VM baselines.py pre 5ed5d9b149d10e3c → post 6afa0be18ab94d76; package l.tgz 0997a232e9e76169.

- **L on the VM:** checksums verified (baselines.py post 6afa0be18ab94d76), commit 254714b; `run_l.sh 3` started 2026-10-14 ~17:06 (VM clock) (log `results/fullsys_l_run.log`).

- **L result (development; VM 254714b; run 17:05–19:38 IST; `results/fullsys_lidar_faults.csv`).** Regression clean
  fixed / hybrid vs F14: 8.3e-17 m (16 rows).

| set | fixed | hybrid | hybrid_chi2L | hybrid_stale | hybrid_chi2L_stale | lidar_off | oracle_L |
|---|---|---|---|---|---|---|---|
| clean (mean RMSE) | 0.258 | 0.104 | 0.101 | 0.104 | 0.101 | 0.123 | 0.104 |
| LiDAR faults (120) | 0.840 | 0.626 | 0.412 | **0.125** | 0.112 | 0.123 | 0.123 |
| — of which freeze long | 3.803 | 4.142 | 2.158 | **0.165** | 0.110 | 0.123 | 0.165 |
| overlap (72) | 5.776 | 2.172 | 1.724 | **1.001** | 1.172 | 0.936 | 0.833 |
| F14 faults (208), chi2L only | | (0.565) | 1.114 | | | | |

  Diverged on LiDAR faults: fixed 22.5 %, hybrid 15 %, hybrid_stale 0.8 %. Non-freeze LiDAR faults (noise, loss,
  scale, spurious) hardly hurt hybrid (0.106–0.150 m vs clean 0.104); the frozen LiDAR is the dangerous fault (hybrid
  4.14 m on long freezes: identical scans tell scan matching AND the detector "no motion").
  Rule (1a) hybrid_chi2L: LiDAR faults 75/120, −3.1 %; clean −3.0 %; **F14 faults +20.8 %** → NOT adopted (gating the
  scan update when odom / IMU faults have pulled the state away removes the correction that the LiDAR provides).
  **Rule (1b) hybrid_stale: freeze runs 23/24, −94.9 %; clean ±0; other LiDAR faults ±0 → ADOPTED.**
  Rule (2) with D = hybrid_stale: oracle_L vs D median −0.0 %; D worse than lidar_off on 57 % of LiDAR-fault runs
  (≥ 25 %) → **BUILD learned r_LiDAR** (per the pre-stated rule).
- **Interpretation (development):** (i) the parameter-free stale-scan check removes the frozen-LiDAR failure (4.14 →
  0.17 m); (ii) on single LiDAR faults there is no headroom left (oracle = D); the 57 % "worse than lidar_off" are
  small differences (D 0.125 vs off 0.123 mean) because in these short-range (2.5 m) worlds LiDAR adds little per run;
  (iii) **overlap runs DO show headroom:** LiDAR noise 0.10 m + odom bias: D 0.961 vs oracle 0.255; + odom drift: 1.983
  vs 0.543 — with corrupted wheels, trusting noisy LiDAR hurts. So the rule's BUILD decision is consistent with the
  data; r_LiDAR is built as pre-stated. Design D for later steps: **hybrid + stale-scan check**.

## L2 — learned r_LiDAR (design 2026-10-14 22:10 IST, before code; triggered by rule (2))
- **Features (per scan, 10 Hz, no filter feedback):** scan-to-scan ICP of consecutive scans initialised with the
  wheel-odometry relative motion (as the filter would): ICP rms, inlier share n_used / beams, degeneracy, iterations,
  ok flag; finite-return share; share of returns < 0.3 m; mean |Δrange| vs previous scan; discrepancy between ICP
  relative pose and wheel-odometry relative pose (translation, rotation) and between ICP rotation and gyro-integrated
  rotation; stale flag. Window = 10 scans (1 s).
- **Label:** r_L = 0 while a LiDAR fault is active (any type / severity; = the oracle_L definition), else 1.
- **Data:** per session: clean + 15 LiDAR faults + 9 overlaps (25 runs); LOBO over D1–D8 as before (val = first
  remaining, 15 epochs, best-val, seed 42); LSTM(n_feat → 32) + sigmoid, BCE with pos_weight.
- **Use:** scan update and detector LiDAR evidence skipped when r_L < 0.5 (as oracle_L), on top of D (stale check
  kept). Method hybrid_stale_rL.
- **Rule (3) (written now):** r_LiDAR ADOPTED only if vs hybrid_stale: LiDAR-fault runs wins ≥ 60 % and median ≤ −3 %
  OR overlap runs wins ≥ 60 % and median ≤ −3 %; AND clean median ≤ +2 % AND F14 faults median ≤ +2 %. Otherwise
  reported as negative (as Phase 3A). Open-loop quality (AUC, false alarms on clean runs) reported.

**L2 implemented (2026-10-14 23:00 IST, cloud workspace):** `learning/lidar_rel.py` (build / train / eval),
`learning/run_rl.sh`. Mock smoke (2 LiDAR faults + 1 overlap + 1 F14 fault per session, 2 epochs): features build
(12 training runs, 36 % unreliable scans), LOBO training and closed loop run; frozen-LiDAR scans flagged 96 %, noisy
overlap scans 99.7 %, clean flagged 1.5 %. Mock numbers not used. sha256: lidar_rel.py 3da58b6e1c9971b3,
run_rl.sh fae2caa1385f25b8, package rl.tgz 7410818bb51f4f35.

## C0 — final models (design 2026-10-14 23:30 IST, before training)
Calibration and final-test sessions are new, so they need ONE model per learned component, trained on the dev sessions
only (D1–D8, filtered by config/dev_sessions.txt — calibration / final feature files are never used for training).
Protocol identical to LOBO, with nothing held out: validation = first dev session (D1, best-val checkpoint), fit =
D2–D8, 15 epochs, seed 42, same stride / windows / losses. Models saved as `<kind>_FINAL.pt` (rel, slip, li, rlidar).
No change to architecture or hyper-parameters.

## C1 — calibration (design 2026-10-14 23:30 IST, before recording)
**Sessions:** 3–4 new recordings, role "cal", same settings and fs_check rule as dev (F5 / F5a):
  C1 terrain lane 0 (smooth, y = 0): there and back, 2 stops, one 180° turn; C2 terrain lane 2 (mild μ 0.3, y = 7.6):
  there and back, stops and the turn ON the mild patch; C3 indoor slip world: corridor → south patch → slip strip →
  north (stops on the strip, turn on a patch) — a route not driven in D5–D8; optional C4 terrain lane 5 driven in
  REVERSE (start x = 15, yaw = π). Stopwatch 120 s each (terrain ≈ 65–80 s recorded).
**Runs:** clean + the 26 F7 faults (std scale) per calibration session; LiDAR-fault and overlap runs too if r_LiDAR is
adopted. Models: `*_FINAL.pt`. Filter: the design D after L2 (hybrid + stale-scan check [+ r_LiDAR]).
**Candidates (current value first, one group changed at a time from the current design):**
  G1 reliability gate 0.5 / 0.3 / 0.7 · G2 χ² gate: wheels+gyro @ 99 % / gyro only @ 99 % / wheels+gyro @ 99.9 % /
  gyro only @ 99.9 % · G3 P_slip threshold (NHC skip and detector p_slip_max together) 0.5 / 0.3 / 0.7 ·
  G4 k_w_slip 50 / 0 / 100 · G5 (only if r_LiDAR adopted) r_LiDAR gate 0.5 / 0.3 / 0.7.
**Rule (written now):** score(c) = median over all calibration runs (clean + faults [+ LiDAR / overlap]) of
RMSE_c / RMSE_current − 1. In each group the candidate with the lowest score is chosen if score ≤ −2 %, else the current
value stays. The combination of all chosen values is then run once: if its score ≤ the best single-group score, it is
frozen; otherwise only the best single-group change is frozen. Nothing else is tuned. Frozen values go into the
pre-registration; calibration sessions are never used for the final test.

**C0 / C1 implemented (2026-10-15 00:20 IST, cloud workspace):** `learning/train_final.py`, `cal_eval.py`,
`run_final_models.sh`, `run_cal.sh` (no change to existing files). Mock test: rel / slip / rlidar FINAL models written
(dev-list filter, dummy FINAL test set); calibration runs all variants on clean, fault, LiDAR and overlap runs, prints
group scores, applies the rule and writes the frozen-value JSON. sha256: train_final.py dc5319ab3ba9cfe6, cal_eval.py
a48b7dfdb29802f3, run_final_models.sh 3dd9d0c637690750, run_cal.sh 9eec375dcd15934d.

- **L2 result (development; VM c9f6e6c; run 22:20–23:17 IST; `fullsys_rlidar_lobo.csv`, `fullsys_rlidar_closed_loop.csv`).**
  Features: 200 training runs, 152 750 scans, 48.7 % unreliable. Open loop (LOBO): AUC LiDAR faults 0.958, overlap 0.967;
  recall 0.89 / 0.93; **false alarms: 17 % of clean-run scans flagged** (12.5 % / 15.7 % of non-fault scans in fault
  runs; D6 / D7 up to 31–35 %).
  Closed loop hybrid_stale_rL vs hybrid_stale: clean 3/8, +4.2 %; LiDAR faults 59/120, ±0.0 %; overlap 39/72, −1.9 %;
  F14 faults (vs F14 hybrid) 59/208, **+17.8 %**. Targeted cases improve strongly (overlap LiDAR noise + odom bias
  0.961 → 0.254 m, + odom drift 1.983 → 0.540 m) but false alarms remove good LiDAR updates elsewhere (scale + IMU bias
  0.644 → 1.490). **Rule (3) → r_LiDAR NOT adopted** (negative, consistent with the pre-registered Phase 3A result).
- **Final development design (2026-10-15 00:40 IST): hybrid + stale-scan check** = F10 reliability (gate) + χ² innovation
  gate (odom, gyro) + LiDAR scan matching + 5c + learned P_slip on Q / NHC / ZUPT + stale-LiDAR check. Calibration G5 is
  dropped (no r_LiDAR); FINAL models needed: rel, slip (li for the final-test baseline later). **Development complete.**

## C1b — calibration amendment (logged 2026-10-15 00:50 IST, before any calibration recording or result)
Two changes to the C1 rule; nothing else in C0 / C1 changes.
**(a) No-harm guard.** The C1 score pools 1 clean run with 26 fault runs per session, so clean runs carry almost no
weight. A candidate (single group or the combination) is now eligible only if ALL hold, each vs the current value on
the same calibration runs: (i) score (median over all runs of RMSE_c / RMSE_current − 1) ≤ −2 %; (ii) median over the
clean runs ≤ +2 %; (iii) number of diverged runs (max error > 2 m) not higher. Among eligible candidates the lowest
score wins; if the combination is not eligible, the best eligible single change is frozen; if none, current values.
**(b) Baselines calibrated with the same procedure** (fairness; same calibration sessions, same runs = clean + 26 F7
faults, same rule incl. the guard, each baseline scored against its own current value; current value first):
  χ² baseline: gate level 99 % / 95 % / 99.9 % (wheels + gyro) · Sage–Husa: forgetting factor b 0.98 / 0.95 / 0.99 ·
  Li (classifier ZUPT + NHC schedule, li_FINAL.pt): ZUPT threshold P_s 0.7 / 0.6 / 0.8 (group L1) and NHC threshold
  P_t 0.6 / 0.5 / 0.7 (group L2), combination checked as in C1 · fixed: no parameter (R₀ / Q are the shared EKF9
  values, identical for every method, not calibrated for anyone).
Frozen values for every method go into fullsys_calibration_choice.json and the pre-registration; the final test uses
each method at its frozen values.

**C1b implemented (2026-10-15 01:20 IST, cloud workspace):** `cal_eval.py` (guard + baseline groups; JSON gains
`baselines`), `run_cal.sh` (also unsets FULLSYS_LI_VARIANT). Mock test (1 session, 3 runs): guard rejected a candidate
with −7.8 % score but +1 diverged run; baselines at current values reproduce direct F14 / F14b runs exactly (chi2,
Sage–Husa, Li, max |diff| 0); combination path exercised. Mock numbers not used. Compute ≈ 2× the C1 estimate.
sha256: cal_eval.py a826bd884bf384b9, run_cal.sh 635a27d5dc2d94cd, package c1b.tgz 17a6c43a12149775.
VM commits: a283d6f (C0 / C1), 078628f (C1b); C0 FINAL training started 00:41 IST (rel, slip, li; dev list D1–D8, val D1).
- **C0 result (02:48 IST):** rel_FINAL.pt best epoch 7, slip_FINAL.pt best epoch 4, li_FINAL.pt best epoch 1 (val = D1, best-val checkpoint as in LOBO; li val loss rose after epoch 1 — protocol unchanged, noted for the Li baseline description). Models frozen.

## C0b — more training data for the FINAL models (logged 2026-10-15 03:05 IST, before any new recording)
**Reason:** the FINAL models learn from 8 sessions in 2 worlds while the final test uses a new world; development
showed signs of limited diversity (r_LiDAR 17 % clean false alarms; divergences concentrated in D6 / D7). Decision
taken from training-set size and diversity only, not from any calibration or final-test result.
**Sessions D9–D14 (role dev, same settings and F5 / F5a quality rule; lanes 0 and 2 stay reserved for calibration;
the final-test world is never used):** D9 terrain lane 1 ice reversed (x 15, yaw π), 2 stops on ice · D10 lane 3
strips at higher speed, turn, back · D11 lane 4 ramp reversed, one way · D12 lane 5 mixed with long stops (≥ 5 s)
and quick turns · D13 indoor north → strip → south, turns on patches, faster · D14 indoor long still periods and
spins in place on patches.
**Protocol:** dev_sessions.txt gains D9–D14; FINAL models (rel, slip, li) retrained exactly as C0 (val = D1, fit =
D2–D14, 15 epochs, seed 42); the C0 models (02:48) are kept as `*_FINAL_c0` backups and not used. Development
conclusions (F14 … L2, LOBO on D1–D8) are unchanged. Calibration (C1 / C1b) runs AFTER the retraining, on the new
FINAL models. Final-test sessions: target 15–16 (≥ 12 minimum).
**C0b recording log (2026-10-17):** D9 20261017_155638 KEEP (74.2 s, IMU 0.3 %, 8 flat ev 0.000); D10
20261017_160044 KEEP (90.6 s, 0.0 %, 5 ev 0.000); D11 20261017_160809 FAIL (50.3 s, 2 flat ev) — the robot cannot
climb the μ 0.1 icy 5° ramp. **D11 route amended (before re-recording, physical constraint only):** icy-ramp climb
attempt (~10 s wheel spin / stuck) → back down → flat stops → turn → long still to ≥ 60 s.
D11 (2nd) 20261017_161215 KEEP (74.3 s, 0.1 %, 4 flat ev 0.000); D12 20261017_161659 KEEP (94.3 s, 0.2 %, 8 ev
0.006); D13 20261017_162528 FAIL (IMU 1.2 %, 1 event), D13 (2nd) 20261017_163057 FAIL (0.0 %, 1 event). Cause of the
event count (from check_imu_integration.py): an event needs a single cmd change with ≥ 0.3 s to the next, GT yaw rate
≤ 0.05 rad/s over [t−0.1, t+1.0] and |Δv_gt| ≥ 0.05 — stops on low-friction patches / right after turns are not
counted. **Indoor routes amended (recording check only):** 3 straight corridor stop-and-go events (normal floor,
single key taps) before the slip part; slip content unchanged. Applies to D13 (3rd), D14, C3.
D13 (3rd) 20261017_163826 KEEP (104.5 s, 0.2 %, 6 ev 0.000); D14 20261017_164350 KEEP (119.3 s, 0.2 %, 9 ev 0.000).
**Calibration sessions (role calib):** C1 20261017_164819 terrain lane 0 KEEP (90.4 s, 0.1 %, 7 ev 0.000);
C2 20261017_165239 terrain lane 2 KEEP (65.4 s, 0.1 %, 9 ev 0.000); C3 20261017_165656 FAIL (121.4 s > 120, all else
PASS) → C3 (2nd) 20261017_170052 indoor KEEP (110.2 s, 0.4 %, 12 ev 0.000). C3 patch turn is 180° (patches are dead
ends). C4 not recorded.
**Overnight run (C0b + C1/C1b):** back up C0 models as *_FINAL_c0.pt → features for D9–D14 → FINAL rel / slip / li
on D1–D14 (val D1) → calibration features (C1–C3) → cal_eval (guard + baselines) → frozen JSON.
Run started 17:05 IST (VM 59c2f31): D9–D14 features 162/162 in 163 s; integrity check (development diagnostic, not a
gate) 109/112 pass — below 1.05: D3 imu dropout ×1.01 (known), D11 imu noise ×0.96, D13 odom dropout ×1.03; labels
unaffected, no action. FINAL training on 14 dev sessions started 17:08.
FINAL training done 01:16 IST (li ≈ 35 min/epoch alone on one thread). Calibration features 01:16–01:17 (3 × 27 runs;
features folder now 459 files = 14 dev + 3 cal sessions; integrity 133/136, all 24 calibration checks pass); clean-run
slip share C1 1.4 %, C2 4.1 %, C3 8.6 %. cal_eval started 01:17 (variants: hybrid 10, chi2 3, Sage–Husa 3, Li 5).
Note for the final run: the features folder now also holds calibration runs — every loader must keep filtering by an
explicit session list (train_final does; final-run scripts must too).
**C0b FINAL models (D1–D14, val D1):** rel_FINAL.pt best epoch 13, slip_FINAL.pt best epoch 7, li_FINAL.pt best epoch
1 (li val loss lowest at epoch 1 again, as in C0). cal_eval paused while the host Mac slept (VM suspended) and resumed;
no restart needed.
- **C1 / C1b RESULT (calibration; run done 2026-10-18 15:58 IST; `fullsys_calibration.csv`, `_choice.json`).**
  Sessions C1–C3, clean + 26 F7 faults each, FINAL models from D1–D14. Score = median RMSE change vs current (all runs);
  guard: clean ≤ +2 %, diverged not higher; threshold −2 %.
  Hybrid: G1 gate 0.3 +0.3 % (clean +2.8, div +4) / 0.7 −0.8 % (clean −1.2, div −4); G2 χ² gyro99 +1.3 % (clean +31.8,
  div +3) / both999 +1.2 % (clean +17.1, div +1) / gyro999 +1.6 % (clean +31.8, div +4); G3 P_slip 0.3 −0.1 % (clean
  −2.7, div −1) / 0.7 −0.0 % (clean +0.4, div −1); G4 k 0 **−2.1 %** (clean −2.3, **div +1 → not eligible**) / 100
  −0.0 % (clean +12.9, div +1). → **no change; hybrid frozen at gate 0.5, χ² wheels + gyro @ 99 %, P_slip 0.5, k 50.**
  χ² baseline: 95 % −3.9 % (clean −8.1, div −1) → **ADOPTED (χ² baseline frozen at 95 %)**; 99.9 % +0.5 %.
  Sage–Husa: b 0.95 −3.8 % (clean −9.3, **div +1 → not eligible**), 0.99 +7.2 % → **b = 0.98**.
  Li: P_s 0.6 +0.3 % (clean +4.4, div +4), 0.8 +0.2 %; P_t 0.5 / 0.7 ±0.0 % → **unchanged (0.7 / 0.6)**.
  Notes (for the paper): the development design was not improved by any candidate on new sessions; the C1b guard
  decided two cases (k = 0 for hybrid, b = 0.95 for Sage–Husa) — under the original C1 rule k = 0 would have been frozen;
  wheel χ² is essential on clean data (gyro-only +31.8 %). **Parameters FROZEN.**

## R — real-data check: decision (2026-10-17 17:11 IST)
**Skipped** by the author's decision (before any dataset was examined). The paper is simulation-only; this is stated as a
limitation (no real sensor noise, wheel slip, vibration or sensor quirks) and as future work. Generalization evidence
comes from the never-used final-test world, σ-scaled faults (A1), ending faults (R2) and — if adopted — unseen fault
types (E).

## E — unseen fault types (design 2026-10-17 17:20 IST, before any code or result)
**Purpose:** secondary generalization analysis for the final test — fault KINDS never used in training (training = the
26 F7 faults: noise / bias / drift / hold-dropout on gyro z and wheel v, ω). Not part of the primary claim.
**Set (6 types × 3 severities = 18 per session; onset uniform in 20–60 % of the session as F7, seed from the fault
name; fault stays on to the end):**
  odom_zero — intermittent encoder loss: 0.2 s slots after onset read v = ω = 0 with probability 10 / 25 / 50 %
  odom_scale — multiplicative wheel error: v and ω × (1 + 5 / 10 / 20 %)
  odom_spike — impulsive outliers: single samples ± 10 × std of the channel at 1 / 2 / 5 per second
  gyro_scale — gyro scale-factor error: ω_gyro × (1 + 5 / 15 / 30 %)
  gyro_sat — saturation: |ω_gyro| clipped at 70 / 50 / 30 % of the clean session's 99th percentile |ω_gyro|
  gyro_spike — impulsive outliers on ω_gyro, as odom_spike
Injected on the 50 Hz signals the filter consumes (same place as F7). **Labels** (analysis only, never trained on):
r = 1 / (1 + |faulted − clean| / std) per sample (odom: worst of v, ω), the F7 convention expressed per sample.
Features go to `fullsys_logs/features_unseen/` — a separate folder, so no training loader can see them.
**No development evaluation:** the code is checked for correct injection only (affected share, error size); method
performance on this set is first computed in the final run (decision rules in the pre-registration).
**E implemented (2026-10-17 17:35 IST, cloud workspace):** `learning/unseen_faults.py`, `learning/e_build.py` (no change
to existing files). Mock test (Paper_ws session 024700): 18 unique faults, pre-onset samples unchanged, other sensor
untouched, input frames unmodified, deterministic; injection shares as designed (odom_zero changes only moving samples:
5 / 20 / 44 %; gyro_spike 1.9 / 3.2 / 10.6 % of samples at 50 Hz); features build through features.build (20 columns).
No method evaluated. sha256: unseen_faults.py 2bf8ce3998f33f88, e_build.py bf85f34542ca5bf7, package e.tgz
f2252f42675112fa.
VM commit 2d0728a (E installed; features are built only for the final-test sessions).
VM injection-only check on C1 (20261017_164819, no features, no method): all 18 faults change the signal; mean error
rises with severity for every type; gyro_sat affects 6–8 % of samples (one turn in C1) → final routes need turns for E.

## Transfer to a second laptop (logged 2026-10-18 18:50 IST)
The author will do everything from the final-test world onward (recording, final run, analysis) on another laptop.
**GitHub inventory (KushagraS07/BTP 4b343c0):** fullsys code + git bundle, config (frozen calibration, session lists),
models incl. *_FINAL.pt (D1–D14) and *_FINAL_c0.pt, results (all fullsys_*.csv, calibration JSON), src/ (mobile_robot
incl. data_logger.py, worlds, launch), scripts/ (fs_sessions.py), fullsys_logs/sessions.json, session log, calibration
recordings C1–C3 (raw_cal/*.tgz, 43 / 30 / 56 MB). Development recordings (4.4 GB) are kept outside GitHub.
**VM environment (results/env_vm.txt, pip_freeze_vm.txt):** Ubuntu 24.04.4, aarch64, 6 cores, ROS 2 Jazzy, Gazebo 8.11.0
(Harmonic), Python 3.12.3, torch 2.12.1, numpy 1.26.4, pandas 3.0.3, scipy 1.11.4.
**Before any final data on the new laptop (rules to be fixed in writing at migration time):** T3 reproduction check
(re-run part of the calibration on C1 and compare RMSE with fullsys_calibration.csv) and T4 recording check (one
C1-route recording: fs_check KEEP, sensor noise / gyro bias similar to C1). All final sessions on that one laptop.

## FR — final-run pipeline: design (logged 2026-10-18 20:30 IST, before any final-test data exists)
Analysis code for the confirmatory run; the pre-registration document will reference it by commit hash and may only
add the route list / world details, not change the analysis.
**Inputs:** `config/test_sessions.txt` (explicit list; every session must pass fs_check; the script ABORTS if any
test session is also in dev_sessions.txt or cal_sessions.txt); models rel / slip / li `_FINAL.pt` (C0b); frozen
values from `results/fullsys_calibration_choice.json` (hybrid: gate 0.5, χ² wheels + gyro 99 %, P_slip 0.5, k 50;
χ² baseline 95 %; Sage–Husa b 0.98; Li 0.7 / 0.6). Seed 42 only.
**Methods:** primary — hybrid (+ stale-scan check), fixed, χ² (95 %), Sage–Husa, Li (classifier ZUPT + NHC schedule).
Ablations of hybrid, one part removed at a time: −reliability (r = 1), −P_slip (no Q / NHC / detector use), −χ²,
−stale check, −5c (one pass), −LiDAR (scan updates off).
**Sets per test session:** main = clean + 26 F7 (std) faults [primary + ablations]; sigma = 26 A1 σ-scaled faults;
end = 26 R2 ending faults (also RMSE after the fault ends); lidar = 15 LiDAR faults + 9 overlaps [primary + −stale,
−LiDAR]; unseen = 18 E faults. All methods run with identical inputs; output `results/final/final_<set>.csv`.
**Statistics (`final_stats.py`, A2 method):** q = median over runs of RMSE_hybrid / RMSE_X − 1; 95 % cluster bootstrap
over sessions (10 000, seed 0); exact two-sided Wilcoxon over per-session median log ratios. **Primary family** =
main-set FAULT runs, hybrid vs fixed / χ² / Sage–Husa / Li (4 tests, Holm). Per comparison: CONFIRMED if q < 0 AND
CI upper bound < 0 AND Holm-adjusted p < 0.05. No-harm check: main clean runs, hybrid vs fixed (reported with CI).
Everything else is secondary (CI + unadjusted p, no claims beyond description).
**FR implemented (2026-10-18 21:10 IST, cloud workspace):** `learning/final_eval.py`, `learning/final_stats.py`,
`learning/run_final.sh` (resumable; logs commit, uncommitted-file count and checksums of the test list, models and
frozen JSON first). Mock test (Paper_ws session 024700, reduced fault lists, mock models): all five sets run; regression
vs the development / calibration code on the same run — hybrid, fixed, χ² (95 %), Sage–Husa, Li, one-pass: max |diff|
2.2e-16 m; ablations change the result (−stale 0 on data without frozen scans, as expected). Statistics: synthetic
16-session data with hybrid 30 % better than fixed / χ² / Sage–Husa and tied with Li → CONFIRMED ×3, NOT CONFIRMED
(tie) ×1; exact Wilcoxon for 16 sessions ≈ 3 min. Mock numbers not used. sha256: final_eval.py 39e28f38c6cb1413,
final_stats.py e21e49122ced3607, run_final.sh 8205c68c6e287643, package fr.tgz c7e1382e32e0e230.
VM commit 2470c81 (FR installed); GitHub backup eaeafe7.

## Next
F14 result → log + backup; F14b run (run_li.sh) → log; generalization (held-out fault magnitude); reliability
vs threshold detector + latency; LiDAR faults + r_LiDAR, overlapping / ending faults; calibration sessions →
final-test world → pre-registration → final runs + ablations → writing. Update pipeline figure.
