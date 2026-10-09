"""
generate_r_lidar_labels.py — A.3: precise, per-timestep r_lidar(t)
ground-truth labels, derived by EXACTLY reproducing the injector's own
random draws from its recorded seed -- VERIFIED reproducible for the
'corruption' fault type (see verification test: independently
reproduced indices matched the injector's actual output exactly, using
only seed + severity_value + session row count -- no access to the
injector's internal state needed).

STATUS: 'corruption' only. return_loss/scale_error precise reproduction
NOT yet verified -- do not extend to those fault types without the same
verification test this file's corruption path was built from.

REQUIRES THE REAL row count of the baseline session's scan_filtered
CSV -- see get_session_row_count(). Do not substitute an assumed or
back-calculated row count for real use; only exact for this specific
manifest's fraction/count pair.
"""
import numpy as np
import pandas as pd
import json
import os


def get_session_row_count(scan_filtered_csv_path: str) -> int:
    """Reads the REAL row count directly from the actual file -- never
    infer this from injected_stats alone (n_frames_corrupted / frac can
    round to a range of possible n, not a single exact value)."""
    df = pd.read_csv(scan_filtered_csv_path)
    return len(df)


def reproduce_corruption_indices(seed: int, frac: float, n_rows: int) -> np.ndarray:
    """Exactly reproduces the injector's corrupt_idx for the
    'corruption' fault type. VERIFIED to match the real injector's
    output bit-for-bit when seed/frac/n_rows are correct. This function
    makes EXACTLY ONE rng call sequence (choice, once), matching the
    injector's own call order for this fault type with nothing
    preceding it."""
    rng = np.random.default_rng(seed)
    n_corrupt = max(1, int(n_rows * frac))
    corrupt_idx = rng.choice(n_rows, size=n_corrupt, replace=False)
    return corrupt_idx


def generate_r_lidar_label_for_corruption_session(
        manifest_path: str, baseline_scan_filtered_csv_path: str) -> np.ndarray:
    """
    Returns a [n_timesteps] array of r_lidar(t) in [0, 1] for a
    'corruption' fault session: 0.0 at exactly the corrupted frames
    (all range values in that frame were fully randomized -- genuinely
    unreliable), 1.0 everywhere else (frame untouched by this fault).

    This is a BINARY label by design -- corruption is a per-frame
    all-or-nothing event in the injector's own mechanism (the WHOLE
    frame gets randomized, not partially), so a binary label is the
    accurate reflection of that mechanism, not an arbitrary choice of
    simplicity.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    if manifest['fault_type'] != 'corruption':
        raise ValueError(
            f"This function is ONLY verified for fault_type='corruption', "
            f"got '{manifest['fault_type']}'. Do not use for other fault "
            f"types without their own verification test.")

    seed = manifest['seed']
    frac = manifest['severity_value']
    n_rows = get_session_row_count(baseline_scan_filtered_csv_path)

    corrupt_idx = reproduce_corruption_indices(seed, frac, n_rows)

    # Sanity cross-check against the manifest's own recorded aggregate --
    # if this doesn't match, STOP, don't silently trust the reproduction.
    expected_n = manifest['injected_stats']['n_frames_corrupted']
    if len(corrupt_idx) != expected_n:
        raise RuntimeError(
            f"Reproduction mismatch: computed {len(corrupt_idx)} corrupted "
            f"frames, manifest says {expected_n}. DO NOT TRUST this label -- "
            f"the row count (n_rows={n_rows}) is likely wrong. Re-check "
            f"baseline_scan_filtered_csv_path points at the EXACT file used "
            f"at injection time.")

    r_lidar = np.ones(n_rows, dtype=float)
    r_lidar[corrupt_idx] = 0.0
    return r_lidar


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: python3 generate_r_lidar_labels.py <manifest.json> <baseline_scan_filtered.csv>")
        sys.exit(1)
    labels = generate_r_lidar_label_for_corruption_session(sys.argv[1], sys.argv[2])
    print(f"Generated {len(labels)} r_lidar(t) labels. "
          f"Corrupted (r=0) timesteps: {int((labels == 0).sum())}. "
          f"Clean (r=1) timesteps: {int((labels == 1).sum())}.")
