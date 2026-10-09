# Phase 3B (P_slip → Q(t)) — findings log

Workspace: ws_mobile (ROS 2 Jazzy, Gazebo Harmonic 8.11). Phase 2 models and results stay frozen.
Plan: `claude/solo_plan.md`. Development results below are NOT paper results.

## B0 — housekeeping (2026-10-09)
- ws_mobile backed up to GitHub `KushagraS07/BTP/ws_mobile/` (commit e1d10a7).
- `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` in ~/.bashrc.
- Disk: deleted 1,920 regenerable fault files of the 2026-10-03 held-out sessions (Stages 1–3
  finished, results saved; regenerable with seed 0) and the pip cache → 14 GB free.

## B1 — slip world (2026-10-09)
- `indoor_world.sdf` identical to Paper_ws's; `indoor_world_slip.sdf` taken from Paper_ws
  (68 lines added, 0 removed): 3 static friction patches, 2 mm thick —
  corridor μ 0.10 (x 2–4, y −1.5–1.5), north μ 0.30 (x 3–6, y 3–6), south μ 0.05 (x 3–6, y −6– −3).
- `gazebo_model.launch.py`: new `world:=` argument (default `indoor_world.sdf`, unchanged
  behaviour); backup `.bak_preWorldArg`.
- `session_registry.is_heldout()` also excludes sessions in `sensor_logs/slip_sessions.json`
  (backup `.bak_preSlip`); `scripts/slip_sessions.py` moves slip recordings out of
  good_sessions. Phase 2 training list verified = 7 sessions.

## B2 — slip gate (rule fixed before recording) — **PASS**
Session 20261009_140649 (role gate; 54.1 s; GT path 12.44 m, odometry 11.22 m).

| Event | Area | GT path (m) | Wheel path (m) | Ratio |
|---|---|---|---|---|
| STOP 5.6 s | corridor μ 0.10 | 0.231 | 0.005 | — (slide 23.1 cm) |
| START 11.9 s | corridor μ 0.10 | 0.519 | 0.740 | **1.43** |
| STOP 18.4 s | off patch | 0.017 | 0.005 | control |
| START 35.7 s | off patch | 0.732 | 0.740 | 1.01 |
| STOP 40.7 s | corridor μ 0.10 | 0.235 | 0.007 | — (slide 23.5 cm) |
| START 48.7 s | corridor μ 0.10 | 0.523 | 0.741 | **1.42** |

- Both slip modes occur: sliding after a stop (wheels at zero, body slides 23 cm) and wheel spin on
  starts (wheels report 1.42× the true distance). Off-patch control stop 1.7 cm.
- Consistent with Paper_ws on Gazebo 10 (slide 22–23 cm, off-patch 1.7 cm; spin 1.59×).

## B3 — label rule and Part A
- Label rule from the gate session: slip = |v_wheel − v_true| ≥ 0.10 m/s or |ω_wheel − ω_true| ≥ 0.30 rad/s
  (89 % of slip labels on patches, 1.2 % off-patch). PREREG_slip.md Part A confirmed 2026-10-09 14:27 IST.

## B4 — dev and calib recordings (2026-10-09) — complete
- dev (8): 150044, 150546, 151240, 151812, 152418, 152957, 153500, 154001 (all ACCEPT, 73–99 s).
- calib (3): 154452 (79.3 s), 155140 (73.5 s), 155922 (83.9 s; north visit 1.5 s, near the edge).
- Rejected: 145330 (dev, P1 117.8 s); replaced by 150044 (dev replacements used 1/3, calib 0/3).
- 145330 stays in `slip_sessions.json` with role=dev: `slip_sessions.py` is part of the Part A hash and
  has no "rejected" role, and re-adding it would overwrite health_check_passed. **Rule for every
  Part B script:** a session is used only if its role matches AND `slip_accept.py` exits 0 for it;
  the loader asserts exactly 8 dev and 3 calib sessions.
- Acceptance log corrected (route-5 row had the wrong session ID and no duration; route-9 row was
  missing; literal "—" text replaced by a dash). Backup kept.

## B5 — inputs and labels (`slip_prepare.py`, 2026-10-09) — done
- 12 sessions (gate + 8 dev + 3 calib); 145330 excluded automatically (slip_accept REJECT).
  Features and labels row-aligned with ekf_fused_output; invalid label steps 0 in every session.
- Gate session reproduces the exploration exactly (89.0 % of slip on patch, off-patch rate 1.19 %).
- dev/calib: slip 10.1–30.0 % of steps; 47–87 % of slip steps on a patch; on-patch rate 16.7–51.5 %;
  **off-patch rate 5.6–16.3 %** (gate 1.2 %). The protocol routes (turns on the spot, 0.5–0.9 m/s,
  reversing, more starts/stops) produce many more off-patch positives than the gate run.
  The label rule is frozen (Part A) and is **not** changed; `slip_label_diag.py` (read-only)
  describes where off-patch positives occur, for the label description only.
  Positive-class share (10–30 %) is handled by the fit-split class weight (§5).

- Off-patch positives (`slip_label_diag.py`, 12 sessions): ≥ 99 % of them lie near a patch edge
  (≤ 0.3 m), in an on-the-spot turn, or within 0.5 s of a command change (47–99 % transient,
  34–79 % turning, 0–63 % near edge; dw triggers 62–91 %, dv 46–87 %). Off-patch steps with none of
  these properties are labelled at 0.00–0.24 %. → Label description for the paper: in steady
  off-patch driving the label is ~0; off-patch positives are transients (accel/decel, turns) and
  edge effects — wheel/true speed mismatch that is not caused by a low-μ patch.

## B6 — Q(t) code and k = 0 regression (code written)
- `slip_q.py`: Q(t) = Q₀·(1 + k·p[t−1]) on the **unmodified** frozen
  `lstm_adaptive_ekf.run_lstm_adaptive_ekf_core`. For S/O runs only, the module name
  `DiffDriveEKF` is replaced for the duration of one call by a subclass whose `predict()` sets Q and
  calls the unchanged predict. Optional argument `k` (default None = Phase 2 call, no substitution).
  No frozen file is edited, so `final_freeze.sha256` stays valid. Implementation note for Part B
  (meets §5: optional argument, default untouched, one code path).
- `slip_k0_regression.py`: frozen-file hashes checked first; gate + first final session (175524) ×
  7 models; k = 0 fed uniform-random p (seed 0) so k = 0 must cancel it; exact equality of x, y, θ
  (and r_imu, r_odom); liveness check k = 10 must change the output; (a) compared with the frozen
  final table (lstm_linear). Tested on synthetic data in the cloud workspace: PASS, k = 10 live.

- **VM run 2026-10-09 16:33 IST: k = 0 REGRESSION PASS, 14/14** (gate 140649 + final 175524 ×
  7 models; x, y, θ, r_imu, r_odom array_equal; 2704 rows each; k = 10 changes output in all 14;
  frozen files match final_freeze.sha256).
- Informational check "reproduces Phase 2 table" printed False for 2 of 7 models (094119, 095915).
  Cause found: pandas' default fast float parser read those two `final_models.csv` values 1 ulp off
  (0x1.ef16a5450260e vs …0f; 0x1.efe717cf7659f vs …a0). The 17-digit values are identical
  (diff 0.0 when both sides are read the same way). Script fixed to `float_precision='round_trip'`;
  re-run so the logged output is the version hashed in Part B. Not part of the PASS criterion.
- **Re-run 2026-10-09 16:40 IST (fixed script): PASS 14/14, and the frozen Phase 2 call reproduces
  `final_models.csv` lstm_linear bit-exactly for all 7 models** (log `fix_ws/slip_k0_regression_log.txt`).

## B7 — P_slip training code (`slip_train.py`)
- Exactly §5. Implementation choices (not result-driven, logged for Part B): BCEWithLogitsLoss on the
  logit (= BCE on the sigmoid, numerically stable); seed 42 re-set at the start of each fold and a
  seeded DataLoader generator; `torch.use_deterministic_algorithms(True)`; training windows = full
  windows (t ≥ 49) as Phase 2 `make_windows`; run time (and every AUC/Brier) = every row, rows
  t < 49 left-padded with row 0 as the Phase 2 run-time LSTM; checkpoint ties → earlier epoch.
- Smoke-tested on synthetic data in the cloud workspace (mechanics only).

- **VM run 2026-10-09 16:42–17:42 IST (1 core): 8 models written.** Development LOBO only (not S1):
  held-out AUC 0.811–0.998, mean 0.965; Brier mean 0.036. Lowest AUC on 151812 (route 4, max speed
  0.886 m/s, above the ≤ 0.8 m/s protocol guidance — fastest session, least like the others).
  Mean P_slip on patch 0.19–0.53 vs off patch 0.07–0.19 (class-weighted → ranking score, not a
  calibrated probability, as stated in §6). Off-patch false-alarm rate at P ≥ 0.5: 0.9–4.8 %.
  Best epochs 9–15 (fold 150044 best at 15 = still improving at the fixed 15 epochs; fold 152957
  had a loss spike at epoch 15, not kept — best-validation checkpoint epoch 9). Nothing changed.

## B8 — calibration (`slip_calib.py`, 2026-10-09 17:50–18:15 IST, 4 cores, --jobs 3)
Calib sessions 154452, 155140, 155922; 7 Phase 2 models; 8 P_slip models.
Mean RMSE(L) = 0.47957 m (per session 0.46746 / 0.38620 / 0.58503).

| k | mean RMSE(O) | vs L | mean RMSE(S) | vs L |
|---|---|---|---|---|
| 1 | 0.47884 | −0.15 % | 0.47867 | −0.19 % |
| 3 | 0.47823 | −0.28 % | 0.47808 | −0.31 % |
| 10 | 0.47759 | −0.41 % | **0.47785** | **−0.36 %** |
| 30 | 0.47745 | −0.44 % | 0.47848 | −0.23 % |
| 100 | 0.47751 | −0.43 % | 0.47941 | −0.03 % |

- **Gate 1 PASS** (oracle-best k = 30, reported only). **k = 10 selected** (learned P_slip).
  **Gate 2 PASS.** By the pre-registered rules the test sessions go ahead with k = 10.
- **Effect size is very small:** the oracle ceiling is −0.44 % (≈ 2 mm on 0.48 m); S at k = 10 is
  −0.36 % (≈ 1.7 mm). Likely reason (hypothesis, not tested): the 3-state EKF has no absolute
  position measurement (odometry displacement + IMU heading only), so larger Q cannot pull back the
  position error slip causes.
- At k = 1 and 3, S is slightly better than O. The learned P_slip is a soft score that is also > 0
  off-patch, so part of the gain may be general Q inflation rather than slip-specific. A constant-Q
  control would separate the two (not in the prereg; to be decided before Part B is frozen).

## B9 — Part B written (2026-10-09, before any test session)
- Decisions (author delegated; chosen for publication strength): add constant-Q control C
  (reported only, §7a of `PREREG_slip_partB.md`); keep S1–S3 rules unchanged, report effect against
  the oracle ceiling and in mm ("consistent but practically negligible" if this small).
- Part B is a separate file `fix_ws/PREREG_slip_partB.md` (Part A file and hash untouched).
- `slip_test.py` written (refuses to run unless `slip_partB.sha256` matches; k re-derived from the
  hashed calibration summary); smoke-tested on synthetic data in the cloud workspace.

## B10 — Part B frozen 2026-10-09 18:26 IST; test sessions recorded 18:48–19:41 IST
Accepted (6): T1 190827 (66.5 s), T2 191915 (75.5 s), T3 192336 (75.9 s), T4 193020 (60.7 s),
T5 193520 (74.3 s), T6 193917 (77.1 s).
Rejected (5, all before any evaluation, by the fixed acceptance rules):
- 184810, 185245: logger started without `use_sim_time:=true` (**instruction error**: wrong command
  given to the author) → cmd_vel wall-clock stamps → preprocess clock-mismatch stop → health check
  failed; both also < 60 s (P1).
- 185847: P3/P4/P5 (second corridor stop and turn after leaving the 2 m patch).
- 190456: P1 55.9 s.
- 191524: P2 corridor visit just under 1.0 s.
**Deviation:** Part A allows max 3 replacements per role; the test role needed 5. Logged in
`slip_acceptance_log.md` at 19:05 IST, before any test evaluation. Justification: acceptance is
automatic and fixed and happens before any output exists, so extra replacements cannot bias S1–S3;
2 of the rejections were caused by the instruction error. Reported in the paper.
Rejected test sessions that were registered (185847, 190456, 191524) stay in `slip_sessions.json`
as role test and are excluded by the slip_accept rule (as 145330 for dev).
Driving was timed in seconds from T1 replacement 4 on (route geometry unchanged).

## B11 — CONFIRMATORY RESULT (`slip_test.py`, run once, 2026-10-09 19:45–≈21:00 IST, k = 10)
Part B hash check OK (20 files). Test inputs: 6 sessions, 0 invalid steps, slip share 9.3–21.7 %.

| Test | Value | Rule | Result |
|---|---|---|---|
| **S1** detection | mean AUC **0.9851** (8 models, 0.9810–0.9887); Brier 0.031–0.039; off-patch false alarm 2.3–2.7 % | ≥ 0.80 | **PASS** |
| **S2** localization under slip | S < L in **1/6**; median **+0.839 %** | ≥ 5/6 and median < 0 | **FAIL** |
| **S3** no harm on normal driving | mean S / mean L = **0.99163** (5 Phase 2 final sessions) | ≤ 1.01 | **PASS** |

Per test session (RMSE m): S − L = +0.06 / +1.70 / +0.49 / +1.19 / +2.97 / −2.32 % (+0.14 / +5.48 /
+1.86 / +3.15 / +10.60 / −8.78 mm). Mean S vs L **+0.68 % (+2.1 mm)**.
Reported only: **oracle O vs L +0.77 %** (perfect slip knowledge also makes the filter slightly worse
on the test sessions; calibration ceiling had been −0.44 %); **S vs C +0.035 %** (no slip-specific
effect beyond constant Q inflation); S vs F +0.6 % on average.

**Interpretation (pre-registered §6a: "S1 passes, S2 fails → slip detection works, but this
integration does not improve localization"):**
- The learned P_slip detects slip very well on new sessions (AUC 0.985, higher than development).
- Raising Q(t) with P_slip does not reduce error in this 3-state EKF — and neither does the oracle,
  so the failure is in the integration, not the detector. The tiny calibration gain (−0.36 %) did not
  generalize; effects in both directions are a few mm on ~0.3 m errors (practically negligible).
- Consistent with the hypothesis stated before the test: the 3-state EKF has no absolute position
  measurement (odometry displacement + IMU heading), so larger Q cannot pull back slip-induced
  position error. → Motivates the full system (ZUPT/NHC, scan matching = absolute constraints),
  where P_slip is used to gate those constraints instead of inflating Q.
- S3 PASS: no harm on clean driving (mean −0.8 %, mostly session 180357).
No re-run, no retraining, no threshold change follows from this result (§8).

## Next
Phase 3B complete. Back up ws_mobile to GitHub; then week 2 of the plan (port Paper_ws filter).
