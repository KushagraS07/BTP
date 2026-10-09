#!/usr/bin/env python3
"""
slip_sessions.py -- Phase 3B: keep slip-world recordings separate from Phase 2.

The logger's health check adds EVERY new recording to good_sessions (= Phase 2 training list).
Slip sessions must not stay there. This tool moves them to sensor_logs/slip_sessions.json, with a
role and a free-text note (world, what was driven).

  python3 slip_sessions.py add SESSION --role gate|dev|calib|test [--note "..."]
  python3 slip_sessions.py list
Backups of both json files go to results/*.bak_<time> before every change.
"""
import argparse
import json
import os
import shutil
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ws_paths as W  # noqa: E402

LOG = W.LOG_DIR
REG = os.path.join(LOG, 'session_registry.json')
SLIP = os.path.join(LOG, 'slip_sessions.json')
RES = os.path.join(W.WS, 'results')
ROLES = ('gate', 'dev', 'calib', 'test')


def load_slip():
    if os.path.exists(SLIP):
        return json.load(open(SLIP))
    return {'slip_sessions': [], 'info': {}}


def cmd_add(a):
    if not os.path.exists(os.path.join(LOG, f'odom_{a.session}.csv')):
        sys.exit(f'ABORT: no recording sensor_logs/odom_{a.session}.csv. Nothing changed.')
    os.makedirs(RES, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    for p in (REG, SLIP):
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(RES, f'{os.path.basename(p)}.bak_{stamp}'))
    reg = json.load(open(REG))
    was_good = a.session in reg.get('good_sessions', [])
    reg['good_sessions'] = [s for s in reg.get('good_sessions', []) if s != a.session]
    json.dump(reg, open(REG, 'w'), indent=2)
    sl = load_slip()
    if a.session not in sl['slip_sessions']:
        sl['slip_sessions'].append(a.session)
    sl['info'][a.session] = {'role': a.role, 'note': a.note, 'health_check_passed': was_good,
                             'added': datetime.now().isoformat(timespec='seconds')}
    json.dump(sl, open(SLIP, 'w'), indent=2)
    print(f'{a.session}: role={a.role}, health check passed={was_good}; removed from good_sessions.')
    print(f'Phase 2 training sessions now: {len(reg["good_sessions"])} (must stay 7)')


def cmd_list(_):
    sl = load_slip()
    reg = json.load(open(REG))
    print(f'Phase 2 training sessions: {len(reg.get("good_sessions", []))}')
    print(f'Slip sessions: {len(sl["slip_sessions"])}')
    for s in sl['slip_sessions']:
        i = sl['info'].get(s, {})
        print(f'  {s}  role={i.get("role")}  health={i.get("health_check_passed")}  {i.get("note", "")}')


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
