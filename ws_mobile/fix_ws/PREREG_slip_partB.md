# Pre-registration Phase 3B — Part B (values chosen in development)

**Belongs to:** `fix_ws/PREREG_slip.md` (Part A, confirmed 2026-10-09 14:27 IST, hashes in
`fix_ws/slip_partA.sha256`). Part A is **not edited**: Part B is a separate document so the Part A hash
stays valid.
**Written:** 2026-10-09, after calibration, **before any test session is recorded**.
**Frozen:** 2026-10-09 18:26 IST (date, time IST; by the author, before the first test session is recorded)
**Hashes:** `fix_ws/slip_partB.sha256` (this document, all Phase 3B code, the 8 P_slip models, the
calibration and regression results). `slip_test.py` refuses to run unless every hash matches.

---

## 1. Values of PREREG_slip.md §7

| Item | Value | Evidence |
|---|---|---|
| Gate 1 — oracle feasibility | **PASS.** Mean RMSE(L) 0.47957 m; mean RMSE(O) 0.47884 / 0.47823 / 0.47759 / 0.47745 / 0.47751 m for k = 1 / 3 / 10 / 30 / 100 (−0.15 / −0.28 / −0.41 / −0.44 / −0.43 %). Oracle-best k = 30 (reported, not used) | `results/slip_calib_summary.csv`, `slip_calib.log` |
| k — selection (learned P_slip) | **k = 10.** Mean RMSE(S) 0.47867 / 0.47808 / **0.47785** / 0.47848 / 0.47941 m (−0.19 / −0.31 / **−0.36** / −0.23 / −0.03 %) | same |
| Gate 2 — learned integration | **PASS** (S < L for every k) | same |
| k = 0 equivalence (§5) | **PASS 14/14** (gate 140649 + final 175524 × 7 models; x, y, θ, r_imu, r_odom bit-identical; k = 10 changes every run; the frozen Phase 2 call reproduces `final_models.csv` lstm_linear exactly for all 7 models) | `results/slip_k0_regression.csv`, `fix_ws/slip_k0_regression_log.txt` |
| P_slip models | 8 dev-LOBO models `sensor_logs/slip_model_<s>.pt`, trained exactly as §5 (development LOBO AUC 0.811–0.998, mean 0.965 — development only) | `results/slip_train_lobo.csv`, `results/slip_train.log` |
| Calib sessions | 20261009_154452, 20261009_155140, 20261009_155922 | `slip_acceptance_log.md` |
| Dev sessions | 20261009_150044, 150546, 151240, 151812, 152418, 152957, 153500, 154001 (145330 rejected, P1) | same |

**Practical size (stated now, before test data):** in calibration the oracle ceiling is −0.44 %
(≈ 2 mm on 0.48 m) and S at k = 10 is −0.36 % (≈ 1.7 mm). The paper reports S2 together with the
oracle ceiling and the difference in mm. If the test effect is of this size, it is described as
**consistent but practically negligible**, whatever S2's pass/fail. The pass rules of §6 are **not**
changed (no minimum-effect rule is added after seeing calibration numbers).

## 2. Addition §7a — constant-Q control C (reported only; no pass/fail)

Decided after calibration, before any test data, because at k = 1 and 3 the learned S was slightly
better than the oracle O, which suggests part of the gain may come from a generally larger Q rather
than from slip timing.

- **C:** L + Q(t) = Q₀·(1 + k·c), with **c = the mean over all steps of that P_slip model's output on
  that session** (constant in time). Same k (= 10) and the same 7 × 8 combinations as S, so C has the
  same average Q inflation as S but no timing information.
- Reported: per test session and on the 5 final sessions, RMSE(C) and 100·(S − C)/C. "Slip-specific
  benefit" is claimed only if S < C on average; otherwise the gain is described as general Q inflation.
- C does not enter S1–S3.

## 3. Implementation notes (meet §5; logged for transparency)

1. **Q(t) hook** (`slip_q.py`): the frozen `lstm_adaptive_ekf.run_lstm_adaptive_ekf_core` is called
   unchanged; for S/O/C only, its module name `DiffDriveEKF` is replaced for one call by a subclass whose
   `predict()` sets Q = Q₀·(1 + k·p[t−1]) and calls the unchanged predict. Optional argument `k`
   (default None = Phase 2 call, no substitution). No frozen Phase 2 file is edited
   (`final_freeze.sha256` still valid, checked by the regression).
2. **P_slip training** (`slip_train.py`): BCEWithLogitsLoss on the logit (= BCE on the sigmoid);
   seed 42 re-set per fold with a seeded DataLoader generator; deterministic algorithms; training
   windows t ≥ 49 (as Phase 2 `make_windows`); at run time every row, rows t < 49 left-padded with
   row 0 (as the Phase 2 run-time LSTM); checkpoint ties → earlier epoch.
3. **Session selection** (`slip_prepare.py`): role in `slip_sessions.json` AND `slip_accept.py`
   exit 0; counts asserted (gate 1, dev 8, calib 3, test 6). 145330 stays in the json as role dev
   (`slip_sessions.py` is in the Part A hash) and is excluded by this rule.
4. **Test sessions** are prepared with `slip_prepare.py --roles test` (same code as dev/calib) and
   evaluated once with `slip_test.py` (k re-derived from the hashed calibration summary).
5. **S3 P_slip inputs** on the 5 final sessions: `features_<s>.csv` built in Phase 2 (same Phase 2
   feature code; no slip labels exist there, none are needed for S3).

## 4. Development observations relevant to interpretation (not used to change anything)

- Slip label (frozen): off-patch positives (5.6–16.3 % of off-patch steps in dev/calib) occur at
  patch edges, on-the-spot turns and within 0.5 s of command changes (≥ 99 %); steady off-patch
  driving is labelled 0.00–0.24 % (`fix_ws/slip_label_diag.txt`).
- Lowest development AUC (0.811) on the fastest session (151812, 0.886 m/s). Test sessions are driven
  at the protocol speed (default teleop, ≤ 0.8 m/s) as Part A already requires.

## 5. Test sessions

Recorded only after this document is frozen; logged in `fix_ws/slip_acceptance_log.md` (role test),
not here, so this document's hash does not change. Same protocol and acceptance as Part A §3;
replacements max 3, logged with reason; no session is excluded after evaluation.
Deviations found later are logged in `FINDINGS_slip.md` and reported in the paper.
