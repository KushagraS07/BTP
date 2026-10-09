#!/usr/bin/env python3
"""
apply_autopaths_patch.py -- removes hard-coded ~/ws_mobile paths and hard-coded
session lists from the pipeline and evaluation scripts.

  PATHS:    every '~/ws_mobile/...' in real code becomes an import from
            scripts/ws_paths.py (root = $WS_MOBILE_ROOT, else found automatically).
  SESSIONS: the training-session lists come from session_registry.get_good_sessions();
            run_a2_sanity_check builds its 21 corruption sessions from that list;
            session_registry's fallback default (used only when the .json is missing)
            becomes empty instead of naming this project's sessions.
            validate_faults.py's --dry-run self-test keeps its fixed synthetic values
            on purpose (they are test fixtures, never used on real data).

SAFETY
  * refuses to run while Stage 3 (or any training) is running
  * every anchor is checked BEFORE anything is changed; any mismatch -> nothing changed
  * on THIS machine every new value must equal the old hard-coded value
    (same paths, same session lists, same order), otherwise -> nothing changed
  * backups: fix_ws/<file>.bak_preAutoPaths ; py_compile of every patched file
  * comments / printed hints that mention ~/ws_mobile are left as they are (harmless)
  * old scripts (alpha_sweep, noise_evaluation, rnn_fusion1, validate_delay_redesign)
    and the one-off fix_ws archive tools are NOT touched

Usage:  cp ~/ws_mobile/fix_ws/ws_paths.py ~/ws_mobile/scripts/   (done by this script)
        python3 ~/ws_mobile/fix_ws/apply_autopaths_patch.py
"""
import os
import py_compile
import re
import shutil
import subprocess
import sys

HOME_WS = os.path.expanduser('~/ws_mobile')
S = os.path.join(HOME_WS, 'scripts')
FIX = os.path.join(HOME_WS, 'fix_ws')

if subprocess.run(['pgrep', '-f', 'heldout_stage3|train_reliability_lstm|heldout_stage1|rnn_lobo_test'],
                  capture_output=True).stdout.strip():
    sys.exit('ABORT: Stage 3 / an evaluation is still running. Wait until it has finished. Nothing changed.')

IMPORT = 'import ws_paths as _wsp  # noqa: E402  (automatic workspace paths, no hard-coding)'

# generic code-form replacements (comments / print strings never use these exact forms)
SUBS = [
    ("os.path.expanduser('~/ws_mobile/sensor_logs')", '_wsp.LOG_DIR'),
    ('os.path.expanduser("~/ws_mobile/sensor_logs")', '_wsp.LOG_DIR'),
    ("os.path.expanduser('~/ws_mobile/scripts')", '_wsp.SCRIPTS_DIR'),
    ('os.path.expanduser("~/ws_mobile/scripts")', '_wsp.SCRIPTS_DIR'),
    ('Path.home() / "ws_mobile" / "sensor_logs"', '_wsp.LOG_PATH'),
    ('Path.home() / "ws_mobile" / "scripts"', '_wsp.SCRIPTS_PATH'),
]
# file -> expected number of replacements (from the grep on Kushagra's VM, 2026-10-03)
SCRIPT_FILES = {
    'preprocess_data.py': 1, 'session_registry.py': 1, 'fault_injection.py': 2,
    'validate_faults.py': 2, 'validate_lidar_faults.py': 4, 'ekf_fusion.py': 1,
    'generate_reliability_labels.py': 2, 'delay_labels_d2.py': 1,
    'build_training_dataset.py': 2, 'train_reliability_lstm.py': 1,
    'lstm_adaptive_ekf.py': 2, 'compare_adaptive_vs_fixed.py': 2,
    'rnn_fusion_v2.py': 1, 'lidar_fusion_v2.py': 1, 'generate_paper_figures.py': 1,
}

# fix_ws evaluation scripts: root derived from their own location (fix_ws/..)
FIX_WS_OLD = "WS = os.path.expanduser('~/ws_mobile')"
FIX_WS_NEW = ("WS = os.environ.get('WS_MOBILE_ROOT') or "
              "os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # automatic")
FIX_FILES = ['heldout_stage1.py', 'heldout_stage2.py', 'heldout_stage3.py', 'rnn_lobo_test.py']
TRAIN_RE = re.compile(r"^(TRAIN(?:_MODELS)?) = \[('20\d{6}_\d{6}',\s*)+'20\d{6}_\d{6}'\]\n", re.M | re.S)
TRAIN_NEW = ("{name} = sorted(__import__('session_registry').get_good_sessions())"
             "  # automatic: the training baselines in session_registry.json\n")

A2_RE = re.compile(r'^CORRUPTION_SESSIONS = \[\n(?:\s*"[^"]+",\n)+\]\n', re.M)
A2_NEW = ('from session_registry import get_good_sessions  # noqa: E402\n'
          'CORRUPTION_SESSIONS = [f"{b}_fault_lidar_corruption_{v}"  # automatic: all registry baselines\n'
          '                       for b in get_good_sessions() for v in ("mild", "moderate", "severe")]\n')


def add_import(src):
    lines = src.split('\n')
    idx = next(i for i, l in enumerate(lines)
               if (l.startswith('import ') or l.startswith('from ')) and '__future__' not in l)
    lines.insert(idx, IMPORT)
    return '\n'.join(lines)


# ---------------- plan everything first (nothing written yet) ----------------
sys.path.insert(0, FIX)            # ws_paths.py ships in fix_ws until installed
import ws_paths as W               # noqa: E402
if W.WS != HOME_WS and not os.environ.get('WS_MOBILE_ROOT'):
    # imported from fix_ws/: its parent.parent is ws_mobile -> must equal the old root
    sys.exit(f'ABORT: automatic root {W.WS} != {HOME_WS}. Nothing changed.')
checks = {"os.path.expanduser('~/ws_mobile/sensor_logs')": W.LOG_DIR,
          "os.path.expanduser('~/ws_mobile/scripts')": W.SCRIPTS_DIR}
for old, new in checks.items():
    if eval(old) != new:
        sys.exit(f'ABORT: {old} = {eval(old)} but ws_paths gives {new}. Nothing changed.')

plan = {}
for f, n_expected in SCRIPT_FILES.items():
    p = os.path.join(S, f)
    if not os.path.exists(p):
        if f == 'delay_labels_d2.py':
            print('  (delay_labels_d2.py not in scripts/ -- Stage 3 did not adopt D2; skipped)')
            continue
        sys.exit(f'ABORT: {p} missing. Nothing changed.')
    s = open(p).read()
    if '_wsp.' in s:
        sys.exit(f'ABORT: {f} already patched. Nothing changed.')
    n = sum(s.count(a) for a, _ in SUBS)
    if n != n_expected:
        sys.exit(f'ABORT: {f}: found {n} hard-coded paths, expected {n_expected}. Nothing changed.')
    for a, b in SUBS:
        s = s.replace(a, b)
    plan[p] = add_import(s)

sys.path.insert(0, S)
from session_registry import get_good_sessions  # noqa: E402
good = sorted(get_good_sessions())

for f in FIX_FILES:
    p = os.path.join(FIX, f)
    s = open(p).read()
    if s.count(FIX_WS_OLD) != 1:
        sys.exit(f'ABORT: {f}: WS anchor found {s.count(FIX_WS_OLD)}x. Nothing changed.')
    s = s.replace(FIX_WS_OLD, FIX_WS_NEW)
    m = list(TRAIN_RE.finditer(s))
    if len(m) != 1:
        sys.exit(f'ABORT: {f}: training list found {len(m)}x. Nothing changed.')
    old_list = eval(m[0].group(0).split('=', 1)[1])
    if old_list != good:
        sys.exit(f'ABORT: {f}: hard-coded list {old_list} != registry {good}. Nothing changed.')
    s = s[:m[0].start()] + TRAIN_NEW.format(name=m[0].group(1)) + s[m[0].end():]
    plan[p] = s

p = os.path.join(S, 'run_a2_sanity_check.py')
s = plan.get(p, open(p).read())
m = list(A2_RE.finditer(s))
if len(m) != 1:
    sys.exit(f'ABORT: run_a2_sanity_check.py: CORRUPTION_SESSIONS block found {len(m)}x. Nothing changed.')
old_a2 = eval(m[0].group(0).split('=', 1)[1])
new_a2 = [f'{b}_fault_lidar_corruption_{v}' for b in get_good_sessions() for v in ('mild', 'moderate', 'severe')]
if old_a2 != new_a2:
    sys.exit(f'ABORT: run_a2 list differs from the automatic one (order or content). Nothing changed.\n'
             f' old={old_a2}\n new={new_a2}')
plan[p] = s[:m[0].start()] + A2_NEW + s[m[0].end():]

# session_registry.py: the default registry (used ONLY when session_registry.json is
# missing, i.e. on a new machine) must not name this project's sessions.
REG_RE = re.compile(r"^_DEFAULT_REGISTRY = \{\n.*?^\}\n", re.M | re.S)
REG_NEW = ("_DEFAULT_REGISTRY = {\n"
           "    # Empty on purpose (automatic): a new machine starts with no sessions;\n"
           "    # health_check.py adds each recorded session once it passes. It used to be\n"
           "    # pre-filled with this project's session IDs, which don't exist elsewhere.\n"
           "    'good_sessions': [],\n"
           "    'excluded_sessions': {},\n"
           "}\n")
if not os.path.exists(os.path.join(HOME_WS, 'sensor_logs', 'session_registry.json')):
    sys.exit('ABORT: sensor_logs/session_registry.json missing here -- the default would be used. Nothing changed.')
rp = os.path.join(S, 'session_registry.py')
m = list(REG_RE.finditer(plan[rp]))
if len(m) != 1:
    sys.exit(f'ABORT: session_registry.py: _DEFAULT_REGISTRY block found {len(m)}x. Nothing changed.')
plan[rp] = plan[rp][:m[0].start()] + REG_NEW + plan[rp][m[0].end():]

# ROS logger (recording side): src/mobile_robot/mobile_robot/data_logger.py
LOGGER = os.path.join(HOME_WS, 'src', 'mobile_robot', 'mobile_robot', 'data_logger.py')
LOGGER_SUBS = [
    ("os.path.expanduser('~/ws_mobile/sensor_logs')", "os.path.join(_ws_root(), 'sensor_logs')"),
    ("os.path.expanduser('~/ws_mobile/scripts/preprocess_data.py')",
     "os.path.join(_ws_root(), 'scripts', 'preprocess_data.py')"),
    ("os.path.expanduser('~/ws_mobile/scripts/health_check.py')",
     "os.path.join(_ws_root(), 'scripts', 'health_check.py')"),
]
LOGGER_HELPER = """import os


def _ws_root():
    \"\"\"Workspace root, found automatically (no hard-coded ~/ws_mobile):
    1. $WS_MOBILE_ROOT if set; 2. the first folder above this file that contains
    both scripts/ and src/ (works from src/, build/ or install/ after colcon build).\"\"\"
    env = os.environ.get('WS_MOBILE_ROOT')
    if env:
        return os.path.abspath(os.path.expanduser(env))
    d = os.path.dirname(os.path.realpath(__file__))
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, 'scripts')) and os.path.isdir(os.path.join(d, 'src')):
            return d
        d = os.path.dirname(d)
    raise RuntimeError('workspace root not found: set WS_MOBILE_ROOT')
"""
ls = open(LOGGER).read()
if '_ws_root' in ls:
    sys.exit('ABORT: data_logger.py already patched. Nothing changed.')
for old, _ in LOGGER_SUBS:
    if ls.count(old) != 1:
        sys.exit(f'ABORT: data_logger.py: {old} found {ls.count(old)}x (need 1). Nothing changed.')
if ls.count('\nimport os\n') != 1:
    sys.exit('ABORT: data_logger.py: "import os" line not found exactly once. Nothing changed.')
for old, new in LOGGER_SUBS:
    ls = ls.replace(old, new)
ls = ls.replace('\nimport os\n', '\n' + LOGGER_HELPER, 1)
# prove the automatic root equals the old one on this machine
_ns = {'__file__': LOGGER}
exec(LOGGER_HELPER, _ns)
if _ns['_ws_root']() != HOME_WS and not os.environ.get('WS_MOBILE_ROOT'):
    sys.exit(f"ABORT: logger root {_ns['_ws_root']()} != {HOME_WS}. Nothing changed.")
plan[LOGGER] = ls

# ---------------- write ----------------
shutil.copy2(os.path.join(FIX, 'ws_paths.py'), os.path.join(S, 'ws_paths.py'))
print('Installed scripts/ws_paths.py')
for p, new in plan.items():
    shutil.copy2(p, os.path.join(FIX, os.path.basename(p) + '.bak_preAutoPaths'))
    open(p, 'w').write(new)
    py_compile.compile(p, doraise=True)
    print(f'Patched {os.path.relpath(p, HOME_WS)}')
print(f'\nDone: {len(plan)} files. Backups: fix_ws/*.bak_preAutoPaths')
print('Check:  python3 ~/ws_mobile/scripts/ws_paths.py')
print('Logger changed: rebuild the ROS package before the next recording:')
print('  cd ~/ws_mobile && colcon build --packages-select mobile_robot && source install/setup.bash')
