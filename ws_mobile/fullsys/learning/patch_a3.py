#!/usr/bin/env python3
"""patch_a3.py -- A3 seed switch (learning/runcfg.py) + F14b-det variant of li_baseline. Default behaviour unchanged.
Backs up each file as *.bak_preA3. Usage (from ~/ws_mobile/fullsys): python3 learning/patch_a3.py"""
import shutil

P = {
    'learning/train.py': [
        ("MODEL_DIR = os.path.join(_P.WS, 'fullsys_logs', 'models')\nRES = os.path.join(_P.WS, 'results')\n"
         "WINDOW, HIDDEN, LAYERS, LR, BATCH, SEED, CLEAN_W = 50, 64, 1, 1e-3, 256, 42, 5.0\n",
         "from learning.runcfg import RES_SUB, SEED, SEED_SFX  # noqa: E402  (A3)\n"
         "MODEL_DIR = os.path.join(_P.WS, 'fullsys_logs', 'models' + SEED_SFX)\n"
         "RES = os.path.join(_P.WS, 'results', RES_SUB)\n"
         "WINDOW, HIDDEN, LAYERS, LR, BATCH, CLEAN_W = 50, 64, 1, 1e-3, 256, 5.0\n")],
    'learning/train_slip.py': [
        ("WINDOW, HIDDEN, LR, BATCH, EPOCHS, SEED, STRIDE = 50, 64, 1e-3, 256, 15, 42, 5\n",
         "from learning.runcfg import SEED  # noqa: E402  (A3)\n"
         "WINDOW, HIDDEN, LR, BATCH, EPOCHS, STRIDE = 50, 64, 1e-3, 256, 15, 5\n")],
    'learning/closed_loop.py': [
        ("MODEL_DIR = os.path.join(_P.WS, 'fullsys_logs', 'models')\nRES = os.path.join(_P.WS, 'results')\n",
         "from learning.runcfg import RES_SUB, SEED_SFX  # noqa: E402  (A3)\n"
         "MODEL_DIR = os.path.join(_P.WS, 'fullsys_logs', 'models' + SEED_SFX)\n"
         "RES = os.path.join(_P.WS, 'results', RES_SUB)\n")],
    'learning/li_baseline.py': [
        ("GS, GT, BETA, ALPHA, RATIO = 0.7, 0.6, 2.0, 2.5, 50.0\n",
         "GS, GT, BETA, ALPHA, RATIO = 0.7, 0.6, 2.0, 2.5, 50.0\n"
         "VARIANT = os.environ.get('FULLSYS_LI_VARIANT', 'li')   # 'li_det': our detector for ZUPT, classifier for NHC\n"
         "if VARIANT not in ('li', 'li_det'):\n"
         "    raise ValueError(VARIANT)\n"),
        ("        res = dict(base, method='li', share_ps=float((pr[:, 0] >= GS).mean()),\n",
         "        res = dict(base, method=VARIANT, share_ps=float((pr[:, 0] >= GS).mean()),\n"),
        ("        res.update(with_5c(s2, fr.odom, scans, cfg, p9, seg, ps=pr[:, 0], pt=pr[:, 2]))\n",
         "        ps_in = None if VARIANT == 'li_det' else pr[:, 0]\n"
         "        res.update(with_5c(s2, fr.odom, scans, cfg, p9, seg, ps=ps_in, pt=pr[:, 2]))\n"),
        ("    df.to_csv(os.path.join(RES, 'fullsys_li_baseline.csv'), index=False)\n",
         "    df.to_csv(os.path.join(RES, 'fullsys_li_baseline.csv' if VARIANT == 'li' else 'fullsys_li_det.csv'), index=False)\n"),
        ("    li = df[df.method == 'li'][['run', 'type', 'sensor', 'rmse', 'max_err', 'head_rmse_deg', 'diverged']]\n",
         "    li = df[df.method == VARIANT][['run', 'type', 'sensor', 'rmse', 'max_err', 'head_rmse_deg', 'diverged']]\n"),
        ("    allm = pd.concat([b[b.method.isin(['fixed', 'ours', 'hybrid'])], df[df.method == 'li']])\n",
         "    allm = pd.concat([b[b.method.isin(['fixed', 'ours', 'hybrid'])], df[df.method == VARIANT]])\n"),
        ("            print(f'{meth} vs li [{nm}]: {meth} wins {int((s.ref < s.rmse).sum())}/{len(s)}, '\n",
         "            print(f'{meth} vs {VARIANT} [{nm}]: {meth} wins {int((s.ref < s.rmse).sum())}/{len(s)}, '\n"),
        ("    print(f\"li: P_s>=0.7 share {df[df.method=='li'].share_ps.mean():.3f}, P_t>=0.6 share \"\n"
         "          f\"{df[df.method=='li'].share_pt.mean():.3f}\")\n",
         "    print(f\"{VARIANT}: P_s>=0.7 share {df[df.method==VARIANT].share_ps.mean():.3f}, P_t>=0.6 share \"\n"
         "          f\"{df[df.method==VARIANT].share_pt.mean():.3f}\")\n"),
    ],
}
texts = {}
for p, pairs in P.items():
    s = open(p).read()
    for a, b in pairs:
        if s.count(a) != 1:
            raise SystemExit(f'ABORT (nothing changed): pattern not found once in {p}: {a[:60]!r}')
        s = s.replace(a, b)
    texts[p] = s
for p, s in texts.items():
    shutil.copy2(p, p + '.bak_preA3')
    open(p, 'w').write(s)
    print('patched', p)
