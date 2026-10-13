#!/usr/bin/env python3
"""patch_r.py -- R2: faults that END (FULLSYS_FAULT_WINDOW=end, active 15 s after onset). Default unchanged.
Backs up learning/faults.py as faults.py.bak_preR. Usage (from ~/ws_mobile/fullsys): python3 learning/patch_r.py"""
import shutil

P = 'learning/faults.py'
PAIRS = [
    ("SUFFIX = '' if UNIT == 'std' else '_sigma'\n",
     "# R2: FULLSYS_FAULT_WINDOW=end -> every fault is active for END_DUR s after onset, then removed.\n"
     "WINDOW = os.environ.get('FULLSYS_FAULT_WINDOW', 'open')\n"
     "if WINDOW not in ('open', 'end'):\n"
     "    raise ValueError(f'FULLSYS_FAULT_WINDOW must be open or end, not {WINDOW}')\n"
     "END_DUR = 15.0\n"
     "SUFFIX = ('' if UNIT == 'std' else '_sigma') + ('' if WINDOW == 'open' else '_end')\n"),
    ("    after = t >= onset\n",
     "    tend = onset + END_DUR if WINDOW == 'end' else np.inf\n"
     "    after = (t >= onset) & (t < tend)\n"),
    ("            ramp = np.clip((t - onset) / (t1 - onset), 0.0, 1.0)\n            x += sev * std * ramp\n",
     "            ramp = np.clip((t - onset) / (min(t1, tend) - onset), 0.0, 1.0)\n"
     "            ramp[t >= tend] = 0.0\n"
     "            x += sev * std * ramp\n"),
    ("            end = onset + sev * (t1 - t0)\n            hold = (t >= onset) & (t < end)\n",
     "            end = min(onset + sev * (t1 - t0), tend)\n            hold = (t >= onset) & (t < end)\n"),
    ("        a = tt >= onset\n",
     "        a = (tt >= onset) & (tt < tend)\n"),
    ("            ramp = np.clip((tt - onset) / (t1 - onset), 0.0, 1.0)\n            r = 1.0 / (1.0 + sev * ramp)\n",
     "            ramp = np.clip((tt - onset) / (min(t1, tend) - onset), 0.0, 1.0)\n"
     "            ramp[tt >= tend] = 0.0\n"
     "            r = 1.0 / (1.0 + sev * ramp)\n"),
    ("            r[(tt >= onset) & (tt < onset + sev * (t1 - t0))] = 0.0\n",
     "            r[(tt >= onset) & (tt < min(onset + sev * (t1 - t0), tend))] = 0.0\n"),
    ("                    seed=fault.seed, columns=stats)\n",
     "                    seed=fault.seed, columns=stats, window=WINDOW, end=float(min(tend, t1)),\n"
     "                    fault_end=float(min(onset + sev * (t1 - t0), tend, t1)) if fault.ftype == 'dropout'\n"
     "                    else float(min(tend, t1)))\n"),
]
s = open(P).read()
for a, b in PAIRS:
    if s.count(a) != 1:
        raise SystemExit(f'ABORT (nothing changed): pattern not found once in {P}: {a[:60]!r}')
    s = s.replace(a, b)
shutil.copy2(P, P + '.bak_preR')
open(P, 'w').write(s)
print('patched', P)
