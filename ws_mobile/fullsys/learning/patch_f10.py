import os, shutil
L = os.path.expanduser('~/ws_mobile/fullsys/learning/') if os.path.isdir(os.path.expanduser('~/ws_mobile/fullsys/learning')) else 'learning/'
F, B = L + 'features.py', L + 'build_dataset.py'
for p in (F, B):
    if not os.path.exists(p + '.bak_preF10'):
        shutil.copy(p, p + '.bak_preF10')
f = open(F).read()
a = "def build(fr, cmd, gt, det_cfg, p9):"
assert f.count(a) == 1 and 'def slip_label' not in f
f = f.replace(a, '''SLIP_V, SLIP_W = 0.10, 0.30   # m/s, rad/s -- Phase 3B frozen slip label (PREREG_slip.md 4)


def slip_label(t, odom, gt):
    """F10: physical wheel slip on the grid t from CLEAN wheel odometry vs ground truth (Phase 3B
    definition: |v_wheel - v_true| >= 0.10 m/s or |w_wheel - w_true| >= 0.30 rad/s; GT yaw unwrapped).
    1 = slip, 0 = no slip or outside the odom/GT time range."""
    tg, to = gt.t.values, odom.t.values
    _, _, th = quat_to_rpy(gt.gt_orient_x, gt.gt_orient_y, gt.gt_orient_z, gt.gt_orient_w)
    th = np.unwrap(np.asarray(th, dtype=float))
    vx, vy = np.gradient(gt.gt_x.values, tg), np.gradient(gt.gt_y.values, tg)
    v_true = np.interp(t, tg, vx * np.cos(th) + vy * np.sin(th))
    w_true = np.interp(t, tg, np.gradient(th, tg))
    dv = np.abs(np.interp(t, to, odom.linear_x.values) - v_true)
    dw = np.abs(np.interp(t, to, odom.angular_z.values) - w_true)
    ok = (t >= max(tg[0], to[0])) & (t <= min(tg[-1], to[-1])) & np.isfinite(dv) & np.isfinite(dw)
    return np.where(ok & ((dv >= SLIP_V) | (dw >= SLIP_W)), 1.0, 0.0)


''' + a, 1)
b = open(B).read()
a2 = "    d = build(fr, sess[2], gt, cfg, p9)\n"
assert b.count(a2) == 1 and 'slip_label' not in b
b = b.replace(a2, a2 + '''    from learning.features import slip_label
    d['slip'] = slip_label(d.t.values, sess[1], gt)       # F10: from the CLEAN odometry
    d['r_odom'] = np.minimum(d.r_odom.values, 1.0 - d.slip.values)
''', 1)
b = b.replace("rows=len(d), rmse=rmse)", "rows=len(d), rmse=rmse, slip_pct=100 * float(d.slip.mean()))", 1)
a3 = "    print(f'wrote {len(df)} files to {FEAT_DIR}')"
assert b.count(a3) == 1
b = b.replace(a3, "    print('F10 slip label (% of rows, clean runs): ' + ', '.join(f\"{r['session']} {r['slip_pct']:.1f}\" for r in rows if r['name'] == r['session']))\n" + a3, 1)
open(F, 'w').write(f)
open(B, 'w').write(b)
print('F10 patch applied: features.slip_label + r_odom = min(fault label, 1 - slip)')
