"""
session_registry.py — single source of truth for which sessions are
usable, so rnn_fusion_v2.py / lidar_fusion_v2.py never need a hardcoded
SESSIONS list edited by hand. Lives in sensor_logs/ (with the data, not
the scripts), so it travels correctly if you ever record in a new
workspace instead of being wrong until someone remembers to edit code.

Bootstrapped with this project's own real, already-verified findings:
20260712_141740 and 20260712_144326 confirmed good; 20260714_014615
confirmed excluded (dead IMU sensor, diagnosed via flat orientation
signal — see health_check.py, which uses the exact same diagnostic).
"""

import ws_paths as _wsp  # noqa: E402  (automatic workspace paths, no hard-coding)
import json
import os

LOG_DIR = _wsp.LOG_DIR
REGISTRY_PATH = os.path.join(LOG_DIR, 'session_registry.json')

_DEFAULT_REGISTRY = {
    # Empty on purpose (automatic): a new machine starts with no sessions;
    # health_check.py adds each recorded session once it passes. It used to be
    # pre-filled with this project's session IDs, which don't exist elsewhere.
    'good_sessions': [],
    'excluded_sessions': {},
}


def load_registry() -> dict:
    if not os.path.exists(REGISTRY_PATH):
        save_registry(_DEFAULT_REGISTRY)
        return dict(_DEFAULT_REGISTRY)
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def save_registry(reg: dict):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(REGISTRY_PATH, 'w') as f:
        json.dump(reg, f, indent=2)


def get_good_sessions() -> list:
    return load_registry()['good_sessions']


def add_good_session(session: str):
    reg = load_registry()
    if session not in reg['good_sessions']:
        reg['good_sessions'].append(session)
    reg['excluded_sessions'].pop(session, None)
    save_registry(reg)


def add_excluded_session(session: str, reason: str):
    reg = load_registry()
    reg['excluded_sessions'][session] = reason
    if session in reg['good_sessions']:
        reg['good_sessions'].remove(session)
    save_registry(reg)


# Held-out test sessions (decision 2026-10-03, fix_ws/FINDINGS_2026-10-03.md).
# They must never be used for training or reported Phase 0-2 numbers; only the
# held-out evaluation scripts read them, by explicit name.
HELDOUT_PATH = os.path.join(LOG_DIR, 'heldout_sessions.json')


def is_heldout(session: str) -> bool:
    """True if `session`, or the baseline of a fault session
    ('{baseline}_fault_...'), is listed in sensor_logs/heldout_sessions.json."""
    try:
        with open(HELDOUT_PATH) as f:
            held = set(json.load(f).get('heldout_sessions', []))
    except FileNotFoundError:
        return False
    return session.split('_fault_')[0] in held


if __name__ == '__main__':
    reg = load_registry()
    print(f"Registry: {REGISTRY_PATH}\n")
    print(f"Good sessions ({len(reg['good_sessions'])}):")
    for s in reg['good_sessions']:
        print(f"  {s}")
    print(f"\nExcluded sessions ({len(reg['excluded_sessions'])}):")
    for s, reason in reg['excluded_sessions'].items():
        print(f"  {s}: {reason}")
