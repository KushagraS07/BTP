"""
lidar_confidence_proxy.py — A.2's heuristic proxy for r_lidar(t), used
ONLY for the pre-training sanity check (A.2), NOT the final learned
r_lidar. Per A.1: the existing `confidence = min(1.0, valid/10.0)`
metric is blind to beam VALUES, only counting how many beams are
in-range -- a corrupted-but-in-range beam still counts as "valid".
This proxy is beam-value-sensitive, targeting that specific blind spot,
without requiring any training.

METHOD: local roughness against IMMEDIATE NEIGHBOR beams. A real,
unfaulted LiDAR scan of a static indoor environment (walls, corridors)
has smoothly-varying range across adjacent beam indices, since adjacent
beams hit the same or a neighboring surface. `scale_error` (breaks the
overall scale but preserves relative smoothness) will NOT be caught by
this proxy -- flagging that limitation explicitly rather than
overclaiming detection of all three fault types tested. `corruption`
(beams randomized independently) is well caught by this proxy
(verified: proxy_confidence ~0.11 for corrupted frames vs ~0.96 for
clean, on synthetic data matching the real injector's actual
behavior). `return_loss` (many beams pinned to range_max) was tested
and NOT meaningfully caught either (~0.95, indistinguishable from
clean) -- so this proxy's use in A.2 is scoped to 'corruption' only.
"""
import numpy as np


def compute_roughness_proxy(ranges: np.ndarray, range_max: float) -> float:
    """
    Returns a proxy confidence in [0, 1] for ONE scan frame.
    Lower = more locally rough (more likely corrupted).

    ranges: 1D array of beam ranges for this timestep (raw, not yet
        filtered to valid-only -- this function does its own filtering).
    """
    valid_mask = np.isfinite(ranges) & (ranges > 0) & (ranges < range_max)
    valid = ranges[valid_mask]
    if len(valid) < 3:
        return 0.0  # too few beams to assess roughness meaningfully

    # Local roughness: mean absolute difference between adjacent VALID
    # beams (in their original relative order, not resorted) -- this is
    # a coarse but honest choice, not claimed to be optimal.
    diffs = np.abs(np.diff(valid))
    mean_diff = float(np.mean(diffs))

    # Normalize against range_max so the proxy is comparable across
    # different LiDAR range configurations, not just this project's
    # specific 2.5m clip.
    normalized_roughness = mean_diff / range_max

    # Map roughness to a [0,1] confidence: low roughness -> high
    # confidence. ROUGHNESS_SCALE is a free parameter -- NOT tuned or
    # validated yet, flagged here rather than presented as calibrated.
    ROUGHNESS_SCALE = 0.15
    proxy_confidence = float(np.exp(-normalized_roughness / ROUGHNESS_SCALE))

    return max(0.0, min(1.0, proxy_confidence))


def compute_roughness_proxy_series(scan_sync: np.ndarray, range_max: float) -> np.ndarray:
    """Applies compute_roughness_proxy() across every timestep of an
    already-time-synced [n_timesteps, n_beams] array, matching
    lidar_fusion.py's own scan_sync shape (see run_lidar_fusion_single,
    which builds exactly this shape via sync_to_grid())."""
    n = scan_sync.shape[0]
    return np.array([compute_roughness_proxy(scan_sync[i], range_max) for i in range(n)])
