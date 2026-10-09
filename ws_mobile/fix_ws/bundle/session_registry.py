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

import json
import os

LOG_DIR = os.path.expanduser('~/ws_mobile/sensor_logs')
REGISTRY_PATH = os.path.join(LOG_DIR, 'session_registry.json')

_DEFAULT_REGISTRY = {
    'good_sessions': ['20260712_141740', '20260712_144326'],
    'excluded_sessions': {
        '20260714_014615': 'Dead IMU sensor — orientation flat at ~1e-7 rad '
                            'for the entire recording while ground truth '
                            'showed ~6 rad of real heading change. Confirmed '
                            'via imu_filtered orient_z/orient_w. Needs '
                            're-recording, not re-processing.'
    },
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


if __name__ == '__main__':
    reg = load_registry()
    print(f"Registry: {REGISTRY_PATH}\n")
    print(f"Good sessions ({len(reg['good_sessions'])}):")
    for s in reg['good_sessions']:
        print(f"  {s}")
    print(f"\nExcluded sessions ({len(reg['excluded_sessions'])}):")
    for s, reason in reg['excluded_sessions'].items():
        print(f"  {s}: {reason}")
