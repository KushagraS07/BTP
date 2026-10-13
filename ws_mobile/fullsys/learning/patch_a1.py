#!/usr/bin/env python3
"""patch_a1.py -- A1: add the sigma fault-unit switch (FULLSYS_FAULT_UNIT=sigma). Default behaviour unchanged.
Backs up each file as *.bak_preA1. Usage (from ~/ws_mobile/fullsys): python3 learning/patch_a1.py"""
import os
import shutil

FEAT_OLD = "FEAT_DIR = os.path.join(_P.WS, 'fullsys_logs', 'features')\n"
FEAT_NEW = ("from learning.faults import SUFFIX as _FS  # noqa: E402  (A1: features_sigma for the sigma set)\n"
            "FEAT_DIR = os.path.join(_P.WS, 'fullsys_logs', 'features' + _FS)\n")
BQ = [("           'odom_only', 'imu_only', 'no_pslip', 'no_5c']\n", "           'odom_only', 'imu_only', 'no_pslip', 'no_5c', 'hybrid_bq']\n"), ('            hybrid=dict(r_imu=ri, r_odom_imu=ro, q=ps, s=ps, chi2=True),\n', '            hybrid=dict(r_imu=ri, r_odom_imu=ro, q=ps, s=ps, chi2=True),\n            hybrid_bq=dict(r_imu=ri, r_odom_imu=ro, q=np.maximum(ps, 1.0 - ri), s=ps, chi2=True),\n'), ('              f\'{fm:+.1f} % -> {"ADOPT" if ok else "not adopted"}\')\n', '              f\'{fm:+.1f} % -> {"ADOPT" if ok else "not adopted"}\')\n    if \'hybrid_bq\' in piv and \'hybrid\' in piv:\n        cw, fw = int((c.hybrid_bq < c.hybrid).sum()), int((f.hybrid_bq < f.hybrid).sum())\n        cm, fm = 100 * np.median(c.hybrid_bq / c.hybrid - 1), 100 * np.median(f.hybrid_bq / f.hybrid - 1)\n        ok = cw >= 6 and fw >= int(np.ceil(0.6 * len(f))) and cm <= -3 and fm <= -3\n        print(f\'Decision rule (A1-Q) hybrid_bq vs hybrid: clean {cw}/{len(c)} {cm:+.1f} %; faults {fw}/{len(f)} \'\n              f\'{fm:+.1f} % -> {"ADOPT" if ok else "not adopted"}\')\n')]
PATCHES = {
    'learning/faults.py': [
        ("from dataclasses import dataclass, field\nimport zlib\n",
         "from dataclasses import dataclass, field\nimport math\nimport os\nimport zlib\n"),
        ("ONSET_LO, ONSET_HI = 0.20, 0.60\n",
         "ONSET_LO, ONSET_HI = 0.20, 0.60\n"
         "# A1: FULLSYS_FAULT_UNIT=sigma scales faults by sigma = sqrt(R0) of the channel (ekf9_params_dev.yaml)\n"
         "# instead of the signal's own std; names get SUFFIX, onsets stay those of the std set.\n"
         "UNIT = os.environ.get('FULLSYS_FAULT_UNIT', 'std')\n"
         "if UNIT not in ('std', 'sigma'):\n"
         "    raise ValueError(f'FULLSYS_FAULT_UNIT must be std or sigma, not {UNIT}')\n"
         "SUFFIX = '' if UNIT == 'std' else '_sigma'\n"
         "R0_KEY = {'angular_vel_z': 'r_gyro', 'linear_x': 'r_odom_v', 'angular_z': 'r_odom_w'}\n\n\n"
         "def sigma_r0():\n"
         "    from zupt import paths as _P\n"
         "    from zupt.ekf.ekf9 import load_ekf9_params\n"
         "    p9 = load_ekf9_params(os.path.join(_P.CONFIG, 'ekf9_params_dev.yaml'))\n"
         "    return {c: math.sqrt(getattr(p9, k)) for c, k in R0_KEY.items()}\n"),
        ("        return f'{self.session}_fault_{self.sensor}_{self.ftype}_{self.severity}'\n",
         "        return f'{self.session}_fault{SUFFIX}_{self.sensor}_{self.ftype}_{self.severity}'\n"),
        ("        return zlib.crc32(self.name.encode())\n",
         "        return zlib.crc32(f'{self.session}_fault_{self.sensor}_{self.ftype}_{self.severity}'.encode())\n"),
        ("    sev = fault.value\n    stats = {}\n",
         "    sev = fault.value\n    stats = {}\n    sig = sigma_r0() if UNIT == 'sigma' else None\n"),
        ("        std = float(np.std(x))\n", "        std = float(np.std(x)) if sig is None else sig[c]\n"),
        ("        stats[c] = {'std': std}\n", "        stats[c] = {'std': std, 'unit': UNIT}\n"),
    ],
    'learning/build_dataset.py': [(FEAT_OLD, FEAT_NEW)],
    'learning/closed_loop.py': [(FEAT_OLD, FEAT_NEW)],
    'learning/train.py': [(FEAT_OLD, FEAT_NEW)],
    'learning/baselines.py': [
        ("    df.to_csv(os.path.join(RES, 'fullsys_baselines.csv'), index=False)\n",
         "    from learning.faults import SUFFIX\n"
         "    df.to_csv(os.path.join(RES, f'fullsys_baselines{SUFFIX}.csv'), index=False)\n")] + BQ,
}
texts = {}
for p, pairs in PATCHES.items():
    s = open(p).read()
    for a, b in pairs:
        if s.count(a) != 1:
            raise SystemExit(f'ABORT (nothing changed): pattern not found once in {p}: {a[:60]!r}')
        s = s.replace(a, b)
    texts[p] = s
for p, s in texts.items():
    shutil.copy2(p, p + '.bak_preA1')
    open(p, 'w').write(s)
    print('patched', p)
