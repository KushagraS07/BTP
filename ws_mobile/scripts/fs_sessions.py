#!/usr/bin/env python3
"""
fs_sessions.py -- full system: keep full-system recordings separate from Phase 2 / 3B.

The logger writes every recording to sensor_logs/ and its health check adds it to good_sessions
(the Phase 2 training list). Full-system recordings must not stay there. `add` MOVES every file of
the session (raw, stage, filtered, metadata, frequency log) to fullsys_logs/raw/, removes the
session from good_sessions, and records role + note in fullsys_logs/sessions.json.
Backups of both json files go to results/*.bak_<time> before every change.

  python3 fs_sessions.py add SESSION --role check|dev|calib|test [--note "..."]
  python3 fs_sessions.py list
Data folder for the zupt code afterwards:  --data ~/ws_mobile/fullsys_logs/raw
"""
import argparse
import glob
import json
import os
import shutil
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ws_paths as W  # noqa: E402

LOG = W.LOG_DIR
REG = os.path.join(LOG, 'session_registry.json')
FS = os.path.join(W.WS, 'fullsys_logs')
RAW = os.path.join(FS, 'raw')
FS_JSON = os.path.join(FS, 'sessions.json')
RES = os.path.join(W.WS, 'results')
ROLES = ('check', 'dev', 'calib', 'test')


def load_fs():
    return json.load(open(FS_JSON)) if os.path.exists(FS_JSON) else {'sessions': {}}


def cmd_add(a):
    s = a.session
    files = sorted(glob.glob(os.path.join(LOG, f'*_{s}.*')))
    if not any(os.path.basename(f) == f'imu_{s}.csv' for f in files):
        sys.exit(f'ABORT: no sensor_logs/imu_{s}.csv. Nothing changed.')
    slip = os.path.join(LOG, 'slip_sessions.json')
    if os.path.exists(slip) and s in json.load(open(slip)).get('slip_sessions', []):
        sys.exit(f'ABORT: {s} is a Phase 3B slip session; it stays where it is. Nothing changed.')
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    for p in (REG, FS_JSON):
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(RES, f'{os.path.basename(p)}.bak_{stamp}'))
    reg = json.load(open(REG))
    was_good = s in reg.get('good_sessions', [])
    reg['good_sessions'] = [x for x in reg.get('good_sessions', []) if x != s]
    json.dump(reg, open(REG, 'w'), indent=2)
    for f in files:
        shutil.move(f, os.path.join(RAW, os.path.basename(f)))
    d = load_fs()
    d['sessions'][s] = {'role': a.role, 'note': a.note, 'health_check_passed': was_good,
                        'files': len(files), 'added': datetime.now().isoformat(timespec='seconds')}
    json.dump(d, open(FS_JSON, 'w'), indent=2)
    print(f'{s}: role={a.role}; moved {len(files)} files to fullsys_logs/raw/; '
          f'removed from good_sessions (was there: {was_good}).')
    print(f'Phase 2 training sessions now: {len(reg["good_sessions"])} (must stay 7)')


def cmd_list(_):
    d = load_fs()
    reg = json.load(open(REG))
    print(f'Phase 2 training sessions: {len(reg.get("good_sessions", []))}')
    print(f'Full-system sessions: {len(d["sessions"])}')
    for s, i in sorted(d['sessions'].items()):
        print(f'  {s}  role={i["role"]}  files={i["files"]}  {i["note"]}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('add')
    p.add_argument('session')
    p.add_argument('--role', required=True, choices=ROLES)
    p.add_argument('--note', default='')
    sub.add_parser('list')
    a = ap.parse_args()
    {'add': cmd_add, 'list': cmd_list}[a.cmd](a)


if __name__ == '__main__':
    main()
