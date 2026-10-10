"""
Workspace paths for the full-system code (no hard-coded locations).
<ws_mobile> = $WS_MOBILE_ROOT if set, else the first folder above this file that contains both
scripts/ and src/ (ws_mobile layout). Added by fs_port.py (port from Paper_ws).
"""
import os


def _ws_root():
    env = os.environ.get("WS_MOBILE_ROOT")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    d = os.path.dirname(os.path.realpath(__file__))
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "scripts")) and os.path.isdir(os.path.join(d, "src")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("ws_mobile root not found: set WS_MOBILE_ROOT")


WS = _ws_root()
FULLSYS = os.path.join(WS, "fullsys")
DATA = os.path.join(WS, "fullsys_logs", "raw")
CONFIG = os.path.join(FULLSYS, "config")
WS_SCRIPTS = os.path.join(WS, "scripts")
MODEL_DIR = os.path.join(WS, "src", "mobile_robot", "model")
