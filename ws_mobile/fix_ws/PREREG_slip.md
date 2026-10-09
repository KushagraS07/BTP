# Pre-registration: Phase 3B — P_slip → Q(t)

**Written:** 2026-10-09 (DRAFT)
**Part A (design, labels, data plan, training protocol) confirmed:** 2026-10-09 14:27 IST (by the author, before any development session beyond the gate session is recorded)
**Part B (final parameters) frozen:** ____ (after development, before any test session is recorded)
**Status:** DRAFT. Part A fixes everything that can be fixed now. Part B lists what development may
still choose, and how; those values are written in and frozen (sha256) before the test sessions exist.
**Context:** `FINDINGS_slip.md` (B0–B2), `PREREG_final.md` (Phase 2, frozen).

---

## 1. Question

Does a learned slip probability P_slip(t), used to raise the EKF process noise Q(t), reduce
localization error on sessions with real wheel slip, without harming normal driving?

## 2. Setup (fixed)

- ws_mobile (ROS 2 Jazzy, Gazebo Harmonic 8.11), robot `mobile_robot` unchanged, logger and
  preprocessing unchanged (sync + Savitzky–Golay, no outlier filter).
- World `indoor_world_slip.sdf` (patches μ 0.10 corridor, μ 0.30 north, μ 0.05 south).
- **Phase 2 is frozen:** the 7 D2 LOBO reliability models, labels and EKF are not changed or
  retrained. P_slip is a **separate** model; the Phase 2 path is reproduced exactly when P_slip = 0.
- Slip sessions are registered in `sensor_logs/slip_sessions.json` (roles gate/dev/calib/test) and
  are excluded from every Phase 2 script.

## 3. Sessions

| Role | Number | Use |
|---|---|---|
| gate | 1 (20261009_140649) | slip check (done); development only |
| dev | 8 | label check, feature building, P_slip training (LOBO), development evaluation |
| calib | 3 | choosing k (Q(t) gain) with oracle P_slip — only these |
| test | 6, recorded only after Part B is frozen | confirmatory tests §6 |

**Driving protocol (every dev/calib/test session) — enforced by `slip_accept.py` (P1–P6):**
teleop at default speed (≤ 0.8 m/s, 1.0 rad/s), normal driving between patches, routes differ
between sessions, and:
- P1 duration **60.0–100.0 s** (raw odometry, inclusive);
- P2 **≥ 2 different patches** visited (ground-truth centre inside each for ≥ 1.0 s in total);
- P3 **≥ 2 STOPs on a patch**; P4 **≥ 2 STARTs on a patch** (event definitions below);
- P5 **≥ 1 TURN on a patch**: |commanded ω| ≥ 0.5 rad/s held ≥ 0.5 s, GT centre on a patch at its
  first sample;
- P6 **≥ 1 STOP off-patch**.

**Acceptance (fixed; checked before any EKF/model output exists for that session), in this order:**
1. `python3 scripts/final_analysis.py check-sessions <s>` — **before** step 2 (it needs the session
   in good_sessions). ACCEPT requires all of:
   - files `odom_<s>.csv`, `imu_<s>.csv`, a `ground_truth*_<s>.csv` and the **raw** `cmd_vel_<s>.csv` exist;
   - duration (last − first odometry timestamp) **≥ 45.0 s** (inclusive);
   - **0** raw command samples changed by the pipeline's own actuator model
     `preprocess_data.saturate_cmd_vel` (per-wheel limit on v ± ω·s; a sample counts as clipped if
     |changed − raw| > 1e-9 in linear_x or angular_z). There is **no separate speed or turn-rate
     threshold**: the per-wheel limit is the only command criterion;
   - the logger's health check passed (session listed in good_sessions).
2. `python3 scripts/slip_sessions.py add <s> --role <role> --note "..."` (moves it out of the Phase 2
   training list; records health_check_passed).
3. `python3 scripts/slip_accept.py <s>` must print ACCEPT (exit code 0): all of P1–P6 above.
   Event definitions (shared with `slip_gate_check.py`): STOP = |v_cmd| from ≥ 0.10 to < 0.01 m/s,
   held < 0.01 for ≥ 1 s; START = |v_cmd| < 0.01 for ≥ 0.5 s, then ≥ 0.10 m/s; the command is the
   zero-order-held `cmd_vel_filtered`; area = patch containing the GT centre at the event's first
   sample (patch bounds inclusive). There is no other acceptance rule and no discretion.
- If any required file is missing or unreadable, or any checker exits with an error, the session
  is **REJECTED** (not "undetermined").
- `slip_accept.py`, `slip_gate_check.py` and `slip_sessions.py` are part of the Part A hash: the
  acceptance code cannot change after Part A is confirmed.
- Rejected sessions are replaced (max 3 per role) and listed in the log with the reason.
  No exclusion after evaluation.

## 4. Slip label (fixed)

Per step of the EKF/feature grid (the 50 Hz rows of `ekf_fused_output_<s>.csv`, i.e. the same rows
the features and the EKF use), from ground truth — used for training and evaluation only, never as
a model input:

  slip(t) = 1  if |v_wheel(t) − v_true(t)| ≥ 0.10 m/s  or  |ω_wheel(t) − ω_true(t)| ≥ 0.30 rad/s, else 0

- **v_wheel, ω_wheel:** odometry `linear_x`, `angular_z` from `odom_filtered_<s>.csv`, linearly
  interpolated onto the grid timestamps.
- **v_true:** from `ground_truth_filtered_<s>.csv` on its own timestamps: θ = unwrapped yaw from
  (gt_orient_z, gt_orient_w); v_x, v_y = `numpy.gradient` of gt_x, gt_y over time (central
  differences, one-sided at the ends); v_true = v_x·cos θ + v_y·sin θ (signed forward speed);
  then linearly interpolated onto the grid timestamps.
- **ω_true:** `numpy.gradient` of the unwrapped GT yaw over time, interpolated onto the grid.
  Yaw RATES are compared directly (no angle wrapping needed; the yaw is unwrapped before
  differentiating).
- **Invalid samples:** a grid step whose timestamp lies outside the ground-truth or odometry time
  range, or where any input is NaN/inf, gets label NaN and is excluded from training loss, AUC and
  all label statistics. The count of such steps is reported per session.
- **What the label is:** disagreement between wheel-reported and true motion — a proxy for wheel
  slip, not a direct tyre-slip measurement. Gate-session check: 89 % of labelled steps lie on a
  patch and 1.2 % of off-patch steps are labelled; this shows separation in that session, not that
  every positive is physical slip.
**Reason (gate session, development):** 89 % of labelled steps on a patch, 1.2 % of off-patch steps
labelled; lower ω thresholds (0.10–0.20 rad/s) label normal off-patch turns (4.5–11.9 % off-patch).
Known limitation: the patch boundary is judged at the robot centre, so some edge steps count as
"off-patch".

## 5. Model and EKF integration

**P_slip model (fixed protocol; values copied from the Phase 2 trainer `train_reliability_lstm.py`):**
| Item | Value |
|---|---|
| Inputs | the 13 Phase 2 features, in this order: innov_s, innov_phi, innov_imu_theta, odom_scale, imu_scale, imu_angular_vel_z, imu_linear_acc_x, imu_linear_acc_y, imu_linear_acc_z, odom_linear_x, odom_angular_z, wheel_imu_disagreement, wheel_lidar_disagreement — read from `features_<s>.csv`, built by the unchanged Phase 2 feature code (see "Feature source" below) |
| Window | 50 rows (1 s at 50 Hz), causal, ending at the current row; label = slip(t) of the last row |
| Network | LSTM, hidden size 64, 1 layer, dropout 0.0; linear head on the last time step's hidden state → 1 output → sigmoid |
| Loss | binary cross-entropy, positive-class weight = #non-slip / #slip windows of the **fit split only** (never the validation or held-out session); NaN-label windows dropped |
| Optimiser | Adam, learning rate 1e-3, no weight decay; batch size 256, shuffled; 15 epochs; seed 42 |
| Folds | LOBO over the 8 dev sessions → 8 models |
| Fold roles | each LOBO fold uses the 8 dev sessions as: **1 held-out evaluation session** (the LOBO test of that fold); **1 validation session** = the first of the remaining 7 in sorted order (session-based, deterministic); **6 fit sessions** = the rest. The model is therefore fitted on 6 sessions per fold |
| Validation / checkpoint | the checkpoint with the lowest validation loss (same weighted BCE, using the fit-split class weight) is kept |
| Normalisation | mean/std of the **fit split only** (std = 0 → 1) |
| Isolation | the held-out session influences nothing in its fold: not fitting, normalisation, class weight or checkpoint selection. The validation session is used only for checkpoint selection |
| Runtime | `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` |
At test time each of the 8 models is evaluated separately; results are averaged (as in Phase 2).

**Feature source (fixed; identical in training and at run time):** P_slip always reads the
**parallel feature stream** `features_<s>.csv`, produced by the unchanged Phase 2 code
(`build_training_dataset.build_session_features`):
- innov_s, innov_phi, innov_imu_theta ← the **fixed EKF** (`ekf_fusion.py` → `ekf_fused_output_<s>.csv`;
  innov_imu_theta shifted by one row, as in Phase 2);
- odom_scale, imu_scale ← the **Phase 1 heuristic EKF** (`run_adaptive_ekf_core`; imu_scale shifted by one row);
- the other 8 features ← sensor signals (IMU, odometry, wheel–IMU and wheel–LiDAR disagreement).
None of these depends on the Phase 2 R-adaptation, on P_slip or on Q(t), so the slip estimator's
inputs cannot be changed by its own output, and the training and run-time inputs are the same
computation. Because every input is causal (rows ≤ t), P_slip(t) can be computed for a whole session
before the S/O EKF runs, which is equivalent to running the fixed and heuristic EKFs in parallel
online. Consequence: P_slip(t) depends only on the P_slip model, not on which Phase 2 model is used.
The Phase 2 R-LSTM keeps its own frozen run-time behaviour (innovations from the adaptive EKF it
drives, as documented in `lstm_adaptive_ekf.py`) — unchanged.

**Q(t) (fixed implementation):**
- Q_t = Q₀ · (1 + k · P_slip(t−1)), one scalar multiplier on all three diagonal entries of
  Q₀ = diag(0.1, 0.1, 0.05); applied in the existing `predict()` as `Q_t · dt` (as Q₀·dt now).
- **Timing:** in the existing loop, step t runs predict → features → LSTM → odom update → IMU
  update. P_slip(t) is therefore only known after predict(t); predict(t) uses P_slip(t−1)
  (one step = 0.02 s earlier). P_slip(−1) = 0.
- R adaptation = frozen Phase 2 LSTM (linear mapping), unchanged. Q₀, ε, initialisation,
  prediction/update order unchanged.
- **Equivalence check (regression test, before any development result is used):** with k = 0, the
  new code must reproduce the frozen Phase 2 L trajectories **exactly** (max |Δx| = max |Δy| =
  max |Δθ| = 0, bit-identical) on the gate session and on 1 Phase 2 final session, for all 7 Phase 2
  models. Both runs must use: the same initial state (GT start pose) and the same timestamps; the
  same Phase 2 checkpoints and normalisation (mean/std) files; the same update order, dtypes and
  numerical precision; the same trajectory rows (row counts equal, compared row by row, no
  interpolation, resampling or row dropping) and the same `OMP_NUM_THREADS=1` setting.
  **If any trajectory is not identical, the cause is found and fixed before any development result
  is used; the equality criterion is never relaxed.** Result logged in Part B.
- **Implementation of the check:** an automated script (`slip_k0_regression.py`, written in
  development, hashed in Part B) runs (a) the **unmodified** Phase 2 function
  `lstm_adaptive_ekf.run_lstm_adaptive_ekf_core` as frozen in Phase 2 (the version recorded in
  `fix_ws/final_freeze.sha256`) and (b) the new Q(t) code with k = 0, and compares every row of x, y
  and θ with exact equality (`numpy.array_equal`), plus row counts. The Q(t) code is added as an
  optional argument whose default leaves the Phase 2 computation untouched, so one code path serves
  both — not two parallel implementations.
- Note: raising Q does not by itself improve accuracy — it lets the measurements (odometry with
  r_odom, IMU heading with r_imu) pull the estimate more. Whether that helps is what S2 tests.

**Error metric (all tests):** position RMSE = √mean[(x − x_gt)² + (y − y_gt)²] over the session,
exactly as `compare_adaptive_vs_fixed.rmse()` (existing convention).

**Systems compared:**
| ID | System |
|---|---|
| F | fixed EKF |
| L | Phase 2 LSTM-R (frozen) |
| **S** | **L + learned P_slip → Q(t)** (proposed) |
| O | L + oracle P_slip → Q(t): P_slip(t) := slip(t), the ground-truth label of §4, used with **the same one-step delay as S** (predict(t) uses slip(t−1); slip(−1) = 0; an invalid (NaN) label counts as 0). Upper bound, reported only |

## 6. Confirmatory tests (on the 6 test sessions)

Per session: RMSE(L) = mean over the 7 Phase 2 models; RMSE(S) = mean over the 56 combinations
(7 Phase 2 models × 8 P_slip models); RMSE(O) = mean over the 7 Phase 2 models. Sessions weighted equally.

| ID | Claim | Pass rule |
|---|---|---|
| **S1** | P_slip detects slip | for each of the 8 P_slip models, ROC-AUC over all valid steps of the 6 test sessions pooled; the mean of the 8 AUCs ≥ 0.80 |
| **S2** | P_slip → Q(t) reduces error under slip | RMSE(S) < RMSE(L) in ≥ 5 of 6 test sessions **and** median over sessions of 100·(RMSE(S) − RMSE(L))/RMSE(L) < 0 |
| **S3** | No harm on normal driving | on the 5 Phase 2 final sessions (clean recordings, no slip world, never used in Phase 3B development): mean over sessions of RMSE(S) ≤ 1.01 × mean of RMSE(L), same metric and weighting |

Reported without pass/fail: O vs L (how much perfect slip knowledge would give); S vs F;
per-patch results; slip fraction and invalid-step count per test session; P_slip false-alarm rate
off-patch; **Brier score** of P_slip (calibration; the class-weighted loss means the output is a
ranking score, not a calibrated probability — AUC measures ranking only).

## 6a. Interpretation (fixed)

| Result | Interpretation |
|---|---|
| Calibration Gate 1: O not better than L for any k | Q(t) from slip has no demonstrated benefit in this EKF → stop; negative result |
| Calibration Gate 2: O helps, but S not better than L for any k | the process-noise idea may help, but the learned detector is inadequate for it → stop before test sessions |
| O helps, S1 fails | the process-noise idea may help, but the learned detector is inadequate |
| S1 passes, S2 fails | slip detection works, but this integration does not improve localization |
| S2 passes, S3 fails | helps under slip but harms normal driving → overall claim not confirmed |
| S1–S3 pass | learned P_slip → Q(t) confirmed under the stated simulation conditions |
Slip detection alone (S1) is never reported as evidence of better localization.

## 7. Part B — chosen in development, frozen before the test sessions

| Item | How it is chosen | Value |
|---|---|---|
| Gate 1 — feasibility (oracle) | grid k ∈ {1, 3, 10, 30, 100}; metric = mean over the 3 calib sessions (equal weight) of RMSE(O). If no k gives mean RMSE(O) < mean RMSE(L): STOP (negative result, §6a). The oracle-best k is reported but NOT used | ____ |
| k (Q gain) — selection (learned) | only if Gate 1 passes: same grid and metric, with the **learned** P_slip (the 8 dev LOBO models, none trained on calib sessions): metric = mean over the 3 calib sessions of RMSE(S); lowest wins; ties (equal to 4 decimals in m) → smallest k | ____ |
| Gate 2 — learned integration | if no k gives mean RMSE(S) < mean RMSE(L) on the calib sessions: STOP before recording test sessions (negative result: detector not good enough for this integration) | ____ |
| equivalence check (§5) | k = 0 reproduces L exactly on the gate session + 1 Phase 2 final session | ____ |
| P_slip model code + training settings | as §5; any change logged with reason | ____ |
| file hashes | models, scripts, this document (sha256) | ____ |

## 8. Interpretation

S1–S3 all pass → "learned P_slip → Q(t) confirmed on new slip sessions". Any failure is reported
as not confirmed; no retraining, re-recording or threshold change follows from test results.

## 9. Known limitations (stated now)

- **Offline evaluation only (causality).** The unchanged preprocessing applies a **centred**
  Savitzky–Golay filter (window 11, `scipy.signal.savgol_filter` default) to IMU and odometry, so a
  value at time t uses about 5 future raw samples (≈ 0.1 s); linear resampling onto the 50 Hz grid
  uses up to one future raw sample (LiDAR ≈ 10 Hz: up to 0.1 s). Commands use zero-order hold
  (causal). The LSTM windows, the P_slip(t−1) timing and the EKF are causal, but their inputs are
  not strictly so. This applies equally to F, L, S and O, and to all Phase 2 results. All claims are
  therefore stated for **offline evaluation on recorded, preprocessed data**, not for a real-time
  online system. A causal-preprocessing re-run is future work (it would change the frozen Phase 2
  pipeline).

Simulation only; friction patches (abrupt μ change), not continuous terrain; one robot; P_slip
labels need ground truth (5c-style self-supervised labels are future work); 6 test sessions;
3-state EKF (cmd_vel prediction) — Paper_ws's IMU-driven EKF would use P_slip differently (NHC).

## Deviations log
- *(none yet)*

**Draft revisions before Part A is confirmed (not deviations):**
- 2026-10-09 (4th review): causality checked in code — centred S-G filter and linear resampling
  look ≤ ~0.1 s ahead → claims limited to offline evaluation (§9); oracle given the same
  one-step delay as S; acceptance now enforces the full protocol (P1–P6) via `slip_accept.py`,
  included in the Part A hash; k = 0 check specified as an automated exact-equality regression
  against the unmodified Phase 2 function, with Q(t) as an optional argument on one code path.
- 2026-10-09 (3rd review): k = 0 equivalence made bit-exact with explicit conditions (same
  initial state, timestamps, checkpoints, normalisation, order, precision, rows, threads; never
  relaxed); fold roles (1 held-out, 1 validation, 6 fit) and isolation of the held-out session
  written out; class weight from the fit split only.
- 2026-10-09 (2nd review): feature source fixed to the parallel fixed/heuristic-EKF stream
  `features_<s>.csv` in training and at run time; k selection split into an oracle feasibility gate
  (Gate 1) and a learned-P_slip selection on the calib sessions, with Gate 2 (stop if learned P_slip
  cannot beat L for any k); acceptance thresholds, order and missing-data handling written out.
- 2026-10-09: after review — label computation fully specified (grid, interpolation, GT forward-speed projection, yaw-rate comparison, invalid samples); Q(t) timing fixed to P_slip(t−1) because the existing loop runs predict before the LSTM; k=0 equivalence check added; complete training protocol written in from `train_reliability_lstm.py`; RMSE definition, S1 pooling, S2/S3 aggregation and equal session weighting specified; calibration metric and tie-break fixed; interpretation table and Brier score added.
