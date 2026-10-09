# Pre-registration — Held-out Evaluation

**Written:** 2026-10-03, before any held-out result was computed
**Confirmed and frozen:** 2026-10-03 13:45 IST (confirmed by the author, before any held-out result was computed)
**Status:** FIXED. No setting below may be changed after held-out results are seen. Any later change must be logged as a deviation, with its reason.
**Context:** `FINDINGS_2026-10-03.md` (§3, §7)

## Held-out set

| Session | Duration | Saturation | Health |
|---|---|---|---|
| 20261003_132243 | 65 s | 0 clipped | PASS |
| 20261003_132822 | 47 s | 0 clipped | PASS |
| 20261003_133250 | 54 s | 0 clipped | PASS |

- **Recording conditions:** Gazebo, teleop at ≤ 0.805 m/s and ≤ 1.0 rad/s, so no actuator saturation.
- **Never used for training.** They are excluded by `is_heldout()` in every glob-based script, and are listed in `sensor_logs/heldout_sessions.json`. They are not in `good_sessions`.
- **Models are frozen:** the 7 LOBO LSTM checkpoints from 2026-10-03 05:30, the RNN from 2026-10-02 22:12, and the current EKF parameters. None has seen these sessions.
- **Faults are generated the same way as for training:** `generate_all_faults(baseline_sessions=held-out)`, the same matrix, seed 0. That's 52 fault sessions per held-out baseline, severity scaled to each session's own signal std.

**Caveat, stated up front:** n = 3 sessions. Results are a replication check, not a definitive estimate.

---

## Stage 1 — no retraining

### H0. Clean-data EKF baseline (reported, no pass/fail)
- EKF RMSE for each held-out baseline.

### H1. Do the saturated-session anomalies vanish on unsaturated data?
**Hypothesis:** the 3 Phase 0 odom failures and the "delay lowers RMSE" effect seen on the 08-24 sessions were caused by saturation.

**Pass if both hold:**
1. **Phase 0 degradation:** at least **22 of 24** checks pass. That's 3 sessions × {IMU, odom} × {noise, bias, drift, dropout}, with the unchanged rule that the worst severity is ≥ 1.05× baseline.
2. **Delay:** in at least **5 of 6** cases (3 sessions × {IMU, odom}), the RMSE at "large" delay **exceeds** the baseline RMSE.

### H2. Do the reported Phase 1 / Phase 2 results replicate?
- **Data:** the 96 IMU + odom fault sessions (3 × 32).
- **Models:** the Phase 2 LSTM with the linear mapping, evaluated with **each of the 7 LOBO models**. Report the mean ± std of wins vs fixed across models. The heuristic (Phase 1) is deterministic, so it's run once.
- **Replication criterion:** the 7-model mean LSTM odom-fault wins are **> 50%** of the 48 odom sessions **and** exceed the heuristic's odom wins.
- **IMU results:** reported only.

### D1. IMU gate with hysteresis
**Rule:**
- The IMU update is **gated off** (r_imu' = 0) when the LSTM's r_imu is **< 0.5**.
- It is **released** (r_imu' = 1) only after r_imu is **> 0.7 for 25 consecutive steps** (0.5 s).
- Otherwise the previous state holds. The state starts released.
- r_odom passes through unchanged, using the linear mapping.

**Pass if all hold** (on the 7-model mean):
1. IMU-fault wins vs fixed **> 50%** of the 48 IMU-fault sessions
2. mean IMU-fault RMSE **< fixed**
3. no-fault held-out baselines: mean RMSE **≤ 1% worse** than fixed
4. odom-fault wins **≥ (linear's odom wins − 2)**

*Note: the thresholds 0.5 / 0.7 / 25 were chosen after examining the original 7 sessions (the oracle and detection analyses). The held-out sessions are the clean test of that choice.*

### D3. IMU bias with mid-session onset (extra scenario; reported, no pass/fail)
- New fault: `imu_biasmid_{small, medium, large}` = the existing IMU bias severities with **onset at 40%** of the session (`start_frac = 0.4`). That's 3 × 3 = 9 extra sessions.
- **Reported:** wins vs fixed and mean RMSE for linear and D1; the LSTM r_imu detection rate in the first 2 s after onset (pred < 0.5).

---

## Stage 2 — A.2 re-test (D5)

- **Data:** the 9 held-out LiDAR corruption sessions (3 × mild/moderate/severe).
- **RNN inference on held-out sessions uses the frozen RNN.** Residual normalisation uses the **mean of the 7 training sessions' normalisation statistics**, never statistics computed from held-out ground truth.
- **Rule (unchanged):** mean RMSE improvement of the roughness proxy over baseline confidence at ALPHA = 0.70 must be **≥ 5%**.

---

## Stage 3 — D2 delay labels (requires retraining)

- **Change:** `generate_reliability_labels.py` gives IMU and odom delay faults r < 1 on the affected rows. The exact formula is to be fixed in a short addendum **before** this stage runs.
- **Retraining:** on the same 7 training baselines only (LOBO, same configuration).
- **Pass if both hold** on the held-out set:
  1. IMU-delay faulty rows > 0, and held-out IMU-delay wins vs fixed **≥** the Stage 1 linear wins
  2. no other IMU / odom fault type loses **more than 2** wins vs Stage 1 linear (7-model mean)

---

## Deviations log

*(Record any change made after this document is confirmed, with the date and reason.)*

- **2026-10-03, Stage 1 complete:** no deviations. Results: H1 PASS, H2 PASS, D1 FAIL. See `FINDINGS_2026-10-03.md` §8.
- **2026-10-03, Stage 2 complete:** no deviations. D5 = STOP (+0.96% vs ≥ 5%). See `FINDINGS_2026-10-03.md` §9.
  - **Implementation note** (the procedure the pre-registration implies, not a change to it): the held-out baselines had no RNN output, so `heldout_stage2.py` created it with the frozen RNN, then copied it into the corruption sessions the same way `fault_injection.py` does for training sessions.
- **2026-10-03 15:55, Addendum A, implementation clarification** (before any label was computed; the formula is unchanged):
  - **Problem:** the IMU column named in the noise manifest is `_imu_yaw_temp`. It is a temporary column that `fault_injection.py` derives as `quat_to_yaw(orient_z, orient_w)` and never saves. The first `--check` run stopped with a KeyError before producing any output.
  - **Fix:** `delay_labels_d2.py` now derives this column the same way. It interpolates orient_z and orient_w onto the EKF timestamps, converts them to yaw, and then applies the already-specified angle wrap (the column name contains "yaw").
- **2026-10-03 18:4x, Stage 3 procedural restart** (no results seen):
  - **What happened:** the first `--train` run was killed when its terminal closed, after 3 of 7 LSTM folds were trained. No evaluation had run, so no Stage 3 result was seen.
  - **Action:**
    1. production state restored with `--restore`
    2. fold files rebuilt and checked against their sha256 fingerprints
    3. Stage 3 restarted from scratch with `nohup`
  - **Unchanged:** the formula, the settings and the pass rule.
- **2026-10-03 ~19:35, Stage 3 second procedural interruption** (no results seen):
  - **What happened:** the `nohup` run was killed early in fold 2, with no error in the log. The VM did not reboot, the kernel shows no OOM kill, but 1.3 GB of swap was in use and `systemd-oomd` (userspace OOM killer) is active. That is the suspected cause, not yet confirmed. No evaluation had run, so no Stage 3 result was seen.
  - **Action:**
    1. production state restored again with `--restore`
    2. the restart is planned with `systemd-oomd` stopped for the run, and more VM memory if possible
  - **Unchanged:** the formula, the settings and the pass rule.

---

## Addendum A — D2 delay-label formula (required before Stage 3)

**Written:** 2026-10-03, after Stages 1 and 2. The Stage 3 data has not been run.
**Confirmed and frozen:** 2026-10-03 15:50 IST, before any delay label was regenerated. One change was made at confirmation: the A.4 monotonicity abort rule went from 12/14 to 14/14 (a stricter integrity check, not a performance criterion).
**Status:** FIXED, like the rest of this document.
**Procedural condition:** retraining (A.5) starts only after the author explicitly agrees (recorded by the `--kushagra-ok` command-line flag; the flag name is historical), because a pass replaces the reported Phase 2 models. The A.4 sanity checks do not modify any production file and may run before that.

### A.1 Problem
`generate_reliability_labels.py` sets r = 1.0 for every delay fault. This was justified by the *old* delay design, where the timestamp shift cancelled out (diff ~1e-14). The current raw-shift redesign *does* desynchronise the sensor: H1 shows large delay raises EKF RMSE in 6/6 held-out cases. So r = 1 is now a wrong label. It teaches the LSTM that a delayed sensor is reliable.

### A.2 Formula (same convention as the existing labels)
The existing labels all have the form **r = 1 / (1 + |injected error| / baseline column std)**:
- bias: offset = col_std × severity, so r = 1/(1 + severity)
- drift: r(t) = 1/(1 + offset(t)/baseline_std)

Delay uses the same form, with the error that the delay *actually* injected at each row:

```
e_c(t) = | x_c^fault(t) − x_c^baseline(t) |       (per column c, at EKF row time t)
r_c(t) = 1 / (1 + e_c(t) / σ_c)
r(t)   = min over c of r_c(t)                      (same column-combining rule as all other faults)
```

**Definitions, all fixed now:**
- **Signals:**
  - x^fault = `{sensor}_filtered_{fault_session}.csv`
  - x^baseline = `{sensor}_filtered_{baseline}.csv`
  - Both are linearly interpolated (`np.interp`) onto the fault session's `ekf_fused_output` timestamps, which are the rows the labels must match.
- **Columns c:** exactly the columns `fault_injection.py` perturbs for that sensor. These are the keys of `injected_stats` in the same baseline's `{sensor}_noise_low` manifest.
- **σ_c:**
  - The baseline column std the injector itself used: σ_c = `injected_noise_std` / `severity_value` from that same noise_low manifest.
  - This guarantees the same scale as the noise and bias labels.
- **Angle columns:** if a column name contains `yaw` or `theta`, the difference is wrapped: e = |atan2(sin d, cos d)|.
- **No smoothing, no clipping, no tuning constants.**
- **Consequence (intended):** r ≈ 1 when the robot is not moving, since a delayed signal equals the true one then. r drops during motion in proportion to how much the signal changes within the delay.

### A.3 Scope
- **Applies to:** IMU and odom delay faults only (small, medium and large), via `compute_sensor_r` for `fault_type == "delay"`.
- **Unchanged:**
  - all other labels
  - baseline labels
  - LiDAR (the LSTM does not use LiDAR faults)
  - the D3 mid-session-bias labels (Stage 1 only)

### A.4 Pre-run sanity checks (computed before any retraining; abort rules, not tuning)
1. For the 7 training baselines × {IMU, odom} × 3 delay severities (42 sessions), report the mean r and the fraction of rows with r < 0.5.
2. **Abort** if:
   - any delay session has 0 rows with r < 1, which would mean an alignment bug; **or**
   - mean r is not monotonic non-increasing small → medium → large (r_small ≥ r_medium ≥ r_large) in **any** of the 14 baseline-sensor pairs, i.e. all 14 must hold.
3. **"Faulty row" for the Stage 3 pass rule = r < 0.5**, the same threshold as D3 detection.
4. On abort: report the cause and **do not** adjust the formula silently. A fix becomes a logged deviation.

### A.5 Retraining (as in Stage 3 above; restated precisely)
- **Labels:** regenerate them for the training delay sessions only. Rebuild the training dataset.
- **Models:** retrain the 7 LOBO LSTMs with the identical configuration: 13 features, `--baseline-weight 5.0`, lr 1e-3, dropout 0, seed 42, same epochs and early stopping.
- **Data:** training uses the 7 training baselines only. Held-out data is only used for evaluation.
- **Backups:** before retraining, the current models, fold npz files and delay labels are backed up to `fix_ws/models_preD2/`.
  - **If Stage 3 FAILS,** they are restored and remain the reported models.
  - **If it PASSES,** the D2 models replace them, and all Phase 2 numbers are re-reported with them.

### A.6 Evaluation (Stage 3 rule, unchanged)
**Pass rule:** on held-out data, using the 7-model mean and the linear mapping:
1. IMU-delay faulty rows > 0 (held-out labels, A.4-3), **and** IMU-delay wins vs fixed **≥ 1.4/9** (Stage 1 linear)
2. no other IMU or odom fault type loses **more than 2** wins vs Stage 1 linear (Stage 1 values: IMU bias 1.9, drift 3.1, dropout 2.4, noise 4.0; odom bias 9.0, delay 3.6, drift 9.0, dropout 6.9, noise 12.0)

**Also reported** (outside the pass rule):
- odom-delay wins
- mean RMSE per fault type
- held-out no-fault baselines

**Known weakness, stated before the run:** criterion 1 is a low bar, since equal to the Stage 1 value counts as passing. It was fixed in the original pre-registration and is not changed here. The IMU-delay result will therefore also be reported against a ≥ 5/9 reference, **descriptively only**.

- **2026-10-05 ~00:30, Stage 3 complete:** **PASS, D2 models adopted.**
  - 1a: 923 faulty rows.
  - 1b: IMU-delay 2.00 ≥ 1.43.
  - 2: largest loss −0.29.
  - Descriptive ≥ 5/9 reference: not met.
  - This was the third attempt; the first two were killed by `systemd-oomd`, with no results seen. No other deviations.
  - Details: `FINDINGS_2026-10-03.md` §12.
