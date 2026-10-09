# Pre-registration — RNN generalisation test

**Written:** 2026-10-03, after Stage 2 (`FINDINGS_2026-10-03.md` §9.2)
**Confirmed and frozen:** 2026-10-03 15:38 IST, before any LOBO result was computed. Two wording changes were made at confirmation: the H-R criterion is now quantitative, and V1's leakage boundary is stated explicitly.
**Status:** FIXED. No setting below may change; any later change goes in the deviations log with its reason.

## Why
On the 3 held-out sessions, the frozen RNN made 2 of 3 trajectories worse than the EKF alone (0.415 → 0.788 m, 0.475 → 0.763 m; 1.287 → 1.230 m).

**Hypothesis (H-R):**
- `rnn_fusion_v2.py` de-normalises each session's predicted residual using statistics fit on **that session's own ground truth**.
- A deployed robot never has these statistics.
- Without them, the average residual mean (about +0.18 m in x and y) acts as a fixed wrong offset, and the RNN's reported gains (§2.3) don't transfer.

**Honesty caveat:** the 3 held-out sessions have now *revealed* this problem. Choosing a fix and then testing it on those same sessions would not be a clean test. So the **primary** test uses leave-one-session-out (LOBO) on the 7 training sessions, where the test session is never seen in any form. The held-out sessions are **secondary** only.

---

## Variants (this list is fixed; no others will be tried)

| ID | What | Deployable? |
|---|---|---|
| **V0** | EKF only (no RNN) | yes, reference |
| **V1** | RNN; residual de-normalised with the **mean and standard deviation of the training sessions' residual statistics, computed without using any samples from the test session**; what Stage 2 used | yes |
| **V2** | Same as V1, but residual **mean set to 0** (std still the mean of the training sessions') | yes |
| **V3** | RNN with the **test session's own** residual statistics, fit on its own ground truth (the current training-side procedure) | **no**; reported only, to measure how optimistic §2.3 is |

**Unchanged for all variants:**
- RNN architecture and all hyperparameters (hidden 32, window 10, lr 1e-3, dropout 0.2, weight decay 1e-4, ≤ 80 epochs, patience 10, 20-block split, seed 42)
- Savitzky-Golay smoothing of the output
- the RMSE definition

---

## Primary test: 7-fold LOBO on the training sessions

**For each of the 7 sessions:**
1. Train the RNN on the other 6 sessions. Use `rnn_fusion_v2.py`'s own training procedure with only the session list changed.
2. Fit the input normalisation on those 6 sessions only.
3. Run V0–V3 on the *whole* held-out session.

**Fold statistics:** V1 and V2 use statistics from the 6 training sessions only. V3 uses the test session's own statistics, and is diagnostic only.

**Metric:** position RMSE vs ground truth, on the same rows for every variant (RNN rows, i.e. starting 10 steps in).

**Pass rule for a deployable RNN (V1 or V2):**
1. RNN RMSE < V0 RMSE in **at least 5 of 7** folds, **and**
2. mean RMSE across the 7 folds **at least 5% lower** than V0's mean.

**Decision (fixed now):**
- **If V2 passes:** adopt V2. It's the simpler fix, because it uses no offset learned from ground truth.
- **Else, if V1 passes:** adopt V1.
- **If neither passes:** the RNN stage is reported as *not generalising* (a negative result). The paper's pipeline claim is then made on the EKF / adaptive-EKF output, and the RNN is described as an evaluated-and-rejected stage.

**H-R is supported if:**
- V3 mean RMSE is **at least 10% lower** than V1 mean RMSE, **and**
- V2 mean RMSE is **lower** than V1 mean RMSE.

(Mean = across the 7 LOBO folds.)

H-R is reported either way.

---

## Secondary check: the 3 held-out sessions (frozen production RNN; no retraining)
- Report V0–V3 for each session.
- If a variant was adopted above, its held-out mean RMSE must be **no worse than V0's**, or this is recorded as a failure to confirm.
- V1 has already been seen on these sessions (Stage 2). This check is confirmatory only.

---

## Safety
- **Production files are not changed.** The LOBO models and outputs go to `fix_ws/rnn_lobo/`. The script backs up and restores `rnn_model_denoising.pth`, `rnn_norm_stats.npy`, `rnn_residual_stats_per_session.npy` and every `rnn_output_*` / `rnn_smoothed_output_*` it might touch, then checks byte-for-byte that they are unchanged.
- **Single seed (42);** the variability across seeds is not measured.

## Deviations log
- **2026-10-03:** run completed with no deviations.
  - **Result:** V1 FAIL (1/7 folds, −12.4%); V2 FAIL (2/7, −8.4%) → negative-result fallback. H-R SUPPORTED (V3 +25.4% vs V1; V2 < V1).
  - **Details:** `FINDINGS_2026-10-03.md` §10.
