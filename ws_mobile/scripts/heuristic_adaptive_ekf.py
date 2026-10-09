"""
heuristic_adaptive_ekf.py — Phase 1: innovation-based adaptive covariance
(Mehra 1970 / Sage-Husa style), NOT a learned reliability model.

Reuses DiffDriveEKF's exact structure (imported unmodified from
ekf_fusion.py). ekf_fusion.py is NOT touched — it stays the Fixed EKF
baseline required by the framework's Sec22/Sec20 ablation table.

Per-update adaptive rule:
  1. Run the SAME H/innov computation as the base class (copied verbatim
     from DiffDriveEKF, not re-derived).
  2. Compute the innovation covariance S using the BASELINE R (R_odom or
     R_imu, unscaled).
  3. Compute the normalized innovation statistic:
       score = innov.T @ inv(S_baseline) @ innov
     This is the standard chi-squared test statistic (Mahalanobis^2) for
     the update's degrees of freedom:
       - odometry update: [s, phi] jointly, df=2, threshold = 5.99 (95%)
       - IMU theta update: scalar theta, df=1, threshold = 3.84 (95%)
  4. If score <= threshold: R unchanged, proceed exactly like the fixed
     baseline.
     If score > threshold: R_used = R_baseline * (score / threshold),
     i.e. inflate proportionally to how far over the threshold the
     innovation is. Recompute S, K, and the state/covariance update with
     R_used instead of R_baseline.

This uses the SAME odom/IMU decoupling as the baseline (two independent
updates per timestep, not a single joint 3-channel update), so the
adaptive logic can be swapped in per-channel without touching the
overall structure of run_ekf_core's predict/update/update sequence.

NOTE on a tried-and-reverted alternative: an EWMA-smoothed version of
the score (averaging the normalized-innovation score over recent steps
instead of gating on the instantaneous value) was tested and made
things WORSE, not better — it diluted the one signal that did react
(bias-onset transient) below threshold, and did not help drift at all,
since averaging a stationary small score just converges to that same
small score. A proper fix for slow drift needs a CUSUM-style test on
signed (not squared) innovation — out of scope for Phase 1's heuristic
by design; kept as a known limitation for the writeup, not solved here.

Known limitation of the instantaneous test kept here: it only reacts to
ABRUPT changes (bias onset, momentary noise bursts). Once EKF state
converges onto a sustained bias, that sensor's own innovation returns
to ~0 and the gate has nothing left to see — this is fundamental to any
single-sensor self-referential heuristic, not a bug, and is exactly
what Phase 2+'s learned, cross-sensor reliability model is meant to
address instead.
"""
import numpy as np
from ekf_fusion import DiffDriveEKF

CHI2_DF1_95 = 3.84   # IMU theta update (1-DOF)
CHI2_DF2_95 = 5.99   # odometry [s, phi] update (2-DOF)


def _wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


class AdaptiveDiffDriveEKF(DiffDriveEKF):
    """
    Same state/process model as DiffDriveEKF (predict() is inherited
    unchanged). Measurement updates are overridden to inflate R based on
    the instantaneous normalized innovation score for that update.
    """

    def update_odometry(self, x_prev, y_prev, theta_prev, s_meas, phi_meas):
        x, y, th = self.x
        dx, dy = x - x_prev, y - y_prev
        s_pred = np.hypot(dx, dy)
        phi_pred = th - theta_prev

        h_x = np.array([s_pred, phi_pred])
        z = np.array([s_meas, phi_meas])
        innov = z - h_x
        innov[1] = _wrap(innov[1])

        if s_pred > 1e-6:
            H = np.array([
                [dx / s_pred, dy / s_pred, 0.0],
                [0.0,          0.0,        1.0]
            ])
        else:
            H = np.array([
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0]
            ])

        # Step 1: baseline S, normalized innovation score (df=2)
        S_baseline = H @ self.P @ H.T + self.R_odom
        score = float(innov.T @ np.linalg.solve(S_baseline, innov))
        scale = max(1.0, score / CHI2_DF2_95)
        R_used = self.R_odom * scale

        # Step 2: real update using (possibly inflated) R
        S = H @ self.P @ H.T + R_used
        K = np.linalg.solve(S.T, (self.P @ H.T).T).T

        self.x = self.x + K @ innov
        self.x[2] = _wrap(self.x[2])

        I_KH = np.eye(3) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_used @ K.T

        return innov, K, score, scale

    def update_imu_theta(self, theta_meas):
        H = np.array([[0.0, 0.0, 1.0]])
        th = self.x[2]
        innov = np.array([_wrap(theta_meas - th)])

        S_baseline = H @ self.P @ H.T + self.R_imu
        score = float(innov @ np.linalg.solve(S_baseline, innov))
        scale = max(1.0, score / CHI2_DF1_95)
        R_used = self.R_imu * scale

        S = H @ self.P @ H.T + R_used
        K = np.linalg.solve(S.T, (self.P @ H.T).T).T

        self.x = self.x + (K @ innov)
        self.x[2] = _wrap(self.x[2])

        I_KH = np.eye(3) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_used @ K.T

        return innov, K, score, scale


def run_adaptive_ekf_core(odom_x, odom_y, odom_theta, v_lin, omega, imu_theta,
                           timestamps, init_x, init_y, init_theta):
    """
    Mirrors ekf_fusion.run_ekf_core's exact signature and per-step
    predict/update/update sequence, so it's a drop-in comparison against
    the fixed baseline on the same input arrays. Additionally returns
    the per-step R-scale factors (odom_scale, imu_scale) so inflation
    events can be inspected/plotted.
    """
    n = len(timestamps)
    odom_x = np.asarray(odom_x); odom_y = np.asarray(odom_y)
    odom_theta = np.asarray(odom_theta)
    v_lin = np.asarray(v_lin); omega = np.asarray(omega)
    imu_theta = np.asarray(imu_theta)
    timestamps = np.asarray(timestamps)

    d_odom_x = np.diff(odom_x, prepend=odom_x[0])
    d_odom_y = np.diff(odom_y, prepend=odom_y[0])
    s_meas = np.hypot(d_odom_x, d_odom_y)
    s_meas[0] = 0.0
    phi_raw = np.diff(odom_theta, prepend=odom_theta[0])
    phi_meas = _wrap(phi_raw)
    phi_meas[0] = 0.0

    ekf = AdaptiveDiffDriveEKF(init_x, init_y, init_theta)
    out_x, out_y, out_theta = [], [], []
    innov_s, innov_phi, innov_imu_theta = [], [], []
    cov_trace = []
    odom_scale, imu_scale = [], []

    x_prev, y_prev, theta_prev = ekf.x[0], ekf.x[1], ekf.x[2]

    for i in range(n):
        dt = float(timestamps[i] - timestamps[i-1]) if i > 0 else 0.02
        dt = np.clip(dt, 0.001, 0.5)

        ekf.predict(float(v_lin[i]), float(omega[i]), dt)
        innov_o, _, score_o, scale_o = ekf.update_odometry(
            x_prev, y_prev, theta_prev, float(s_meas[i]), float(phi_meas[i]))
        innov_i, _, score_i, scale_i = ekf.update_imu_theta(float(imu_theta[i]))

        x_prev, y_prev, theta_prev = ekf.x[0], ekf.x[1], ekf.x[2]

        out_x.append(ekf.x[0]); out_y.append(ekf.x[1]); out_theta.append(ekf.x[2])
        innov_s.append(innov_o[0]); innov_phi.append(innov_o[1])
        innov_imu_theta.append(innov_i[0])
        cov_trace.append(np.trace(ekf.P))
        odom_scale.append(scale_o); imu_scale.append(scale_i)

    return (np.array(out_x), np.array(out_y), np.array(out_theta),
            np.array(innov_s), np.array(innov_phi), np.array(innov_imu_theta),
            np.array(cov_trace), np.array(odom_scale), np.array(imu_scale))
