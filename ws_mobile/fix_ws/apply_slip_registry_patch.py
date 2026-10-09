#!/usr/bin/env python3
"""
apply_slip_registry_patch.py -- Phase 3B, step B2 prerequisite.
Slip sessions (recorded in the slip world) must never enter Phase 2 training or the Phase 2
evaluations. Every glob-based Phase 2 script already skips sessions for which
session_registry.is_heldout() is True, so is_heldout() is extended to ALSO return True for
sessions listed in sensor_logs/slip_sessions.json. Nothing else changes; if that file does not
exist the behaviour is exactly as before.
Backup: session_registry.py.bak_preSlip ; refuses to run twice.
Run:  python3 apply_slip_registry_patch.py [path/to/session_registry.py]
"""
import os
import shutil
import sys

path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/ws_mobile/scripts/session_registry.py')
src = open(path).read()
if 'SLIP_PATH' in src:
    sys.exit('Already patched -- nothing changed.')

old = '''    try:
        with open(HELDOUT_PATH) as f:
            held = set(json.load(f).get('heldout_sessions', []))
    except FileNotFoundError:
        return False
    return session.split('_fault_')[0] in held
'''
new = '''    held = set()
    try:
        with open(HELDOUT_PATH) as f:
            held |= set(json.load(f).get('heldout_sessions', []))
    except FileNotFoundError:
        pass
    # Phase 3B: slip-world sessions are also kept out of every Phase 2 script
    try:
        with open(SLIP_PATH) as f:
            held |= set(json.load(f).get('slip_sessions', []))
    except FileNotFoundError:
        pass
    return session.split('_fault_')[0] in held
'''
anchor = "HELDOUT_PATH = os.path.join(LOG_DIR, 'heldout_sessions.json')\n"
if src.count(old) != 1 or src.count(anchor) != 1:
    sys.exit('ABORT: session_registry.py does not look as expected. Nothing changed.')
src = src.replace(old, new).replace(
    anchor, anchor + "SLIP_PATH = os.path.join(LOG_DIR, 'slip_sessions.json')  # Phase 3B slip-world sessions\n")
compile(src, path, 'exec')
shutil.copy2(path, path + '.bak_preSlip')
open(path, 'w').write(src)
print(f'Patched {path}\nBackup  {path}.bak_preSlip')
