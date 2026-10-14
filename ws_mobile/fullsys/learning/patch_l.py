#!/usr/bin/env python3
"""patch_l.py -- L step: baselines.run gets lidar='on'|'chi2'|'off' and scan_skip (bool per scan). Default unchanged.
Backs up learning/baselines.py as baselines.py.bak_preL. Usage (from ~/ws_mobile/fullsys): python3 learning/patch_l.py"""
import shutil

P = 'learning/baselines.py'
PAIRS = [
    ("CHI2_99 = {1: 6.635, 2: 9.210}\n", "CHI2_99 = {1: 6.635, 2: 9.210, 3: 11.345}\n"),
    ("        chi2=False, sage=False):\n",
     "        chi2=False, sage=False, lidar='on', scan_skip=None, stale=False):\n"),
    ("    lev = lidar_evidence(scans, odom)\n",
     "    lev = lidar_evidence(scans, odom) if lidar != 'off' else []\n"
     "    skipm = None if scan_skip is None else np.asarray(scan_skip, bool)\n"
     "    if stale:       # a scan identical to the previous one is stale (real returns always carry noise)\n"
     "        rv = scans.ranges.values\n"
     "        st = np.r_[False, rv[1:] == rv[:-1]]\n"
     "        skipm = st if skipm is None else (skipm | st)\n"
     "    if skipm is not None and lidar != 'off':\n"
     "        lev = [e for e, sk in zip(lev, skipm) if not sk]     # skipped scans: LiDAR evidence ignored too\n"),
    ("    det = StationarityDetector(det_cfg, have_lidar=True, logger=lambda *_: None)\n",
     "    det = StationarityDetector(det_cfg, have_lidar=lidar != 'off', logger=lambda *_: None)\n"),
    ("    cnt = dict(odom=0, odom_skip=0, gyro=0, gyro_skip=0)\n",
     "    cnt = dict(odom=0, odom_skip=0, gyro=0, gyro_skip=0, lidar=0, lidar_skip=0)\n"),
    ("            pts, _, _ = scan_points(r, *a)\n            f.scan(pts, make_reference(r, *a))\n            continue\n",
     "            pts, _, _ = scan_points(r, *a)\n"
     "            if lidar == 'off':\n"
     "                continue\n"
     "            if lidar == 'on' and skipm is None:\n"
     "                f.scan(pts, make_reference(r, *a))\n"
     "                continue\n"
     "            from zupt.core.scan_match import match\n"
     "            res = None\n"
     "            if f.ref is not None and len(pts):\n"
     "                res = match(f.ref, pts, init=f.predicted_relative(), cfg=f.sm_cfg)\n"
     "                if res.ok:\n"
     "                    cnt['lidar'] += 1\n"
     "                    skip = skipm is not None and bool(skipm[k])\n"
     "                    if not skip and lidar == 'chi2':\n"
     "                        skip = lidar_nis(f, res.pose, res.cov) > CHI2_99[3]\n"
     "                    if skip:\n"
     "                        cnt['lidar_skip'] += 1\n"
     "                    else:\n"
     "                        f.update_relative_pose(res.pose, res.cov)\n"
     "            f.last_match = res\n"
     "            f.ref = make_reference(r, *a)\n"
     "            f._clone()\n"
     "            continue\n"),
    ("def metrics(o, gt):\n",
     "def lidar_nis(f, z, R):\n"
     "    \"\"\"NIS of the scan-match relative-pose measurement (same h, H as EkfLidar.update_relative_pose).\"\"\"\n"
     "    from zupt.ekf.ekf9 import ITH, IX, IY, wrap\n"
     "    from zupt.ekf.ekf_lidar import ITHC, IXC, IYC\n"
     "    h = f.predicted_relative()\n"
     "    dx, dy = f.x[IX] - f.x[IXC], f.x[IY] - f.x[IYC]\n"
     "    c, s = math.cos(f.x[ITHC]), math.sin(f.x[ITHC])\n"
     "    H = np.zeros((3, f.n))\n"
     "    H[0, IX], H[0, IY], H[0, IXC], H[0, IYC], H[0, ITHC] = c, s, -c, -s, -s * dx + c * dy\n"
     "    H[1, IX], H[1, IY], H[1, IXC], H[1, IYC], H[1, ITHC] = -s, c, s, -c, -c * dx - s * dy\n"
     "    H[2, ITH], H[2, ITHC] = 1.0, -1.0\n"
     "    nu = np.asarray(z, float) - h\n"
     "    nu[2] = wrap(nu[2])\n"
     "    S = H @ f.P @ H.T + np.asarray(R, float)\n"
     "    return float(nu @ np.linalg.solve(S, nu))\n\n\n"
     "def metrics(o, gt):\n"),
]
s = open(P).read()
for a, b in PAIRS:
    if s.count(a) != 1:
        raise SystemExit(f'ABORT (nothing changed): pattern not found once in {P}: {a[:60]!r}')
    s = s.replace(a, b)
shutil.copy2(P, P + '.bak_preL')
open(P, 'w').write(s)
print('patched', P)
