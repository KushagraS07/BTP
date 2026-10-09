#!/usr/bin/env python3
"""
apply_heldout_guard_patch.py -- keeps the held-out test sessions OUT of every
training/evaluation script that discovers sessions by globbing files.

Why: build_training_dataset.discover_all_sessions(), generate_reliability_labels,
validate_lidar_faults and compare_adaptive_vs_fixed (fault glob) find sessions from
files on disk (ekf_fused_output_*, cmd_vel_filtered_*_fault_*) plus health_check --
NOT from session_registry. The 3 held-out sessions have EKF output and pass the
health check, so the next pipeline run would silently train / evaluate on them.

Fix:
  1. session_registry.py gains is_heldout(session): True if the session (or the
     baseline of a fault session) is listed in sensor_logs/heldout_sessions.json.
  2. The 4 glob-based discovery points skip held-out sessions.
  3. sensor_logs/heldout_sessions.json is written (copied from fix_ws).
Registry-based scripts (validate_faults, fault_injection, rnn_fusion_v2,
lidar_fusion_v2) already use get_good_sessions(), which no longer lists them.

All anchors in all files are checked BEFORE anything is changed; any mismatch aborts
with nothing modified. Backups: ~/ws_mobile/fix_ws/<file>.bak_preHeldout
"""
import json
import os
import shutil
import sys

S = os.path.expanduser('~/ws_mobile/scripts')
LOG = os.path.expanduser('~/ws_mobile/sensor_logs')
FIX = os.path.expanduser('~/ws_mobile/fix_ws')

HELPER = '''# Held-out test sessions (decision 2026-10-03, fix_ws/FINDINGS_2026-10-03.md).
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


'''

PATCHES = {
    'session_registry.py': [
        ('helper', "if __name__ == '__main__':\n", HELPER + "if __name__ == '__main__':\n"),
    ],
    'build_training_dataset.py': [
        ('discover',
         '        if "_fault_lidar_" in session:\n            continue\n        sessions.append(session)\n',
         '        if "_fault_lidar_" in session:\n            continue\n'
         '        if is_heldout(session):  # held-out test set: never train/evaluate here\n'
         '            continue\n        sessions.append(session)\n'),
    ],
    'generate_reliability_labels.py': [
        ('discover',
         '        if "_fault_" in session:\n            continue\n        candidates.append(session)\n',
         '        if "_fault_" in session:\n            continue\n'
         '        if is_heldout(session):  # held-out test set\n'
         '            continue\n        candidates.append(session)\n'),
    ],
    'validate_lidar_faults.py': [
        ('discover',
         '        if "_fault_" not in session:\n            candidates.add(session)\n',
         '        if "_fault_" not in session and not is_heldout(session):  # held-out test set\n'
         '            candidates.add(session)\n'),
    ],
    'compare_adaptive_vs_fixed.py': [
        ('fault glob',
         '    if not fault_sessions:\n',
         '    fault_sessions = [s for s in fault_sessions if not is_heldout(s)]  # held-out test set\n'
         '    if not fault_sessions:\n'),
    ],
}
# import line for is_heldout, placed right after the file's first import of
# something from session_registry / health_check (both exist in each file)
IMPORT = 'from session_registry import is_heldout  # noqa: E402  (held-out guard)\n'

texts = {}
for fname, edits in PATCHES.items():
    s = open(os.path.join(S, fname)).read()
    if 'is_heldout' in s:
        sys.exit(f'{fname} already patched. Nothing done.')
    for name, old, _ in edits:
        c = s.count(old)
        if c != 1:
            sys.exit(f'ABORT: {fname} [{name}] anchor found {c} times (need 1). Nothing changed.')
    texts[fname] = s

reg = texts['session_registry.py']
if 'import json' not in reg or 'LOG_DIR' not in reg:
    sys.exit('ABORT: session_registry.py lacks "import json" or LOG_DIR. Nothing changed.')

for fname, edits in PATCHES.items():
    p = os.path.join(S, fname)
    shutil.copy2(p, os.path.join(FIX, f'{fname}.bak_preHeldout'))
    s = texts[fname]
    for _, old, new in edits:
        s = s.replace(old, new)
    if fname != 'session_registry.py':
        # put the import at the very top, after the shebang/docstring block is fine for
        # Python as long as it precedes use; prepend after the first line that imports.
        lines = s.split('\n')
        imp = [i for i, l in enumerate(lines)
               if (l.startswith('import ') or l.startswith('from ')) and '__future__' not in l]
        fut = [i for i, l in enumerate(lines) if l.startswith('from __future__')]
        idx = max(imp[0], fut[-1] + 1) if fut else imp[0]
        lines.insert(idx, IMPORT.rstrip('\n'))
        s = '\n'.join(lines)
    open(p, 'w').write(s)
    print(f'Patched {fname}')

src = os.path.join(FIX, 'heldout_sessions.json')
shutil.copy2(src, os.path.join(LOG, 'heldout_sessions.json'))
print('Copied heldout_sessions.json -> sensor_logs/')
print('Held-out:', json.load(open(src))['heldout_sessions'])
