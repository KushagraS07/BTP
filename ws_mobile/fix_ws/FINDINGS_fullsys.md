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

## Next
GitHub backup (code + docs), then integration: P_slip → NHC/ZUPT gating; baselines Sage–Husa and
fixed EKF9 + P_slip gating; LiDAR scan-matching faults + r_LiDAR; calibration sessions.
