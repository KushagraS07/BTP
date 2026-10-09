#!/usr/bin/env python3
"""
ws_paths.py -- the ONE place that decides where the workspace is.

Every pipeline script imports its folders from here instead of hard-coding
~/ws_mobile, so the workspace can live anywhere and have any name.

How the root is found (first match wins):
  1. environment variable WS_MOBILE_ROOT, if set
         export WS_MOBILE_ROOT=/path/to/workspace
  2. automatically: the folder that contains this file's folder
         (<root>/scripts/ws_paths.py  ->  <root>)

Nothing else needs configuring. The folder layout inside the root is unchanged:
  <root>/scripts, <root>/sensor_logs, <root>/fix_ws
"""
import os
from pathlib import Path

ROOT_PATH = Path(os.environ.get('WS_MOBILE_ROOT') or Path(__file__).resolve().parent.parent).resolve()
SCRIPTS_PATH = ROOT_PATH / 'scripts'
LOG_PATH = ROOT_PATH / 'sensor_logs'
FIX_PATH = ROOT_PATH / 'fix_ws'

# string versions, for code that uses os.path
WS = str(ROOT_PATH)
SCRIPTS_DIR = str(SCRIPTS_PATH)
LOG_DIR = str(LOG_PATH)
FIX_DIR = str(FIX_PATH)

if __name__ == '__main__':
    for k in ('WS', 'SCRIPTS_DIR', 'LOG_DIR', 'FIX_DIR'):
        print(f'{k:12s} = {globals()[k]}')
