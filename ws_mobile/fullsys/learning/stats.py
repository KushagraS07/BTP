#!/usr/bin/env python3
"""stats.py -- A2: session-level statistics for method comparisons (rules: FINDINGS A2).

Effect q = median over runs of RMSE_X/RMSE_Y - 1; 95 % CI by cluster bootstrap over sessions (10 000, seed 0);
exact two-sided Wilcoxon signed-rank over per-session d_s = median log(RMSE_X/RMSE_Y); Holm over the table.
Input: a long CSV with columns model (session), run, method, type, rmse (F14 / F14b format).
Usage: python3 learning/stats.py results/fullsys_baselines.csv --ref hybrid [--extra results/fullsys_li_baseline.csv]
"""
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))

N_BOOT, SEED = 10000, 0


def wilcoxon_exact(d):
    """Exact two-sided Wilcoxon signed-rank p-value (zeros dropped, average ranks for ties)."""
    d = np.asarray(d, float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 1.0
    ranks = pd.Series(np.abs(d)).rank().values
    w = ranks[d > 0].sum()
    tot = ranks.sum()
    stats = np.array([sum(r for r, s in zip(ranks, sg) if s) for sg in itertools.product([0, 1], repeat=n)])
    dev = abs(w - tot / 2)
    return float(np.mean(np.abs(stats - tot / 2) >= dev - 1e-12))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def compare(sub, x, y, rng):
    """sub: rows of one group; x vs y. Returns q, CI, sessions improved, n sessions, p."""
    w = sub.pivot_table(index=['model', 'run'], columns='method', values='rmse')[[x, y]].dropna()
    w['lr'] = np.log(w[x] / w[y])
    w['ratio'] = w[x] / w[y]
    sess = w.index.get_level_values('model').unique()
    by = {s: w.xs(s, level='model') for s in sess}
    q = float(np.median(w.ratio) - 1)
    boot = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = rng.choice(len(sess), len(sess), replace=True)
        boot[b] = np.median(np.concatenate([by[sess[i]].ratio.values for i in pick])) - 1
    ds = np.array([np.median(by[s].lr) for s in sess])
    return dict(q=q, lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5)),
                sess_better=int((ds < 0).sum()), n_sess=len(sess), runs=len(w), p=wilcoxon_exact(ds))


def table(df, ref, methods):
    rng = np.random.default_rng(SEED)
    rows = []
    for grp, sub in (('clean', df[df.type == 'clean']), ('fault', df[df.type != 'clean'])):
        for m in methods:
            r = compare(sub, ref, m, rng)
            rows.append(dict(group=grp, comparison=f'{ref} vs {m}', **r))
    out = pd.DataFrame(rows)
    out['p_holm'] = holm(out.p.values)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--ref', default='hybrid')
    ap.add_argument('--extra', nargs='*', default=[], help='more long CSVs (e.g. F14b li) to merge')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    for e in a.extra:
        x = pd.read_csv(e)
        df = pd.concat([df, x[x.method != 'fixed_check']], ignore_index=True)
    df = df[['model', 'run', 'method', 'type', 'rmse']]
    methods = [m for m in df.method.unique() if m != a.ref]
    out = table(df, a.ref, methods)
    pd.set_option('display.width', 200)
    fmt = out.assign(q=lambda d: (100 * d.q).round(1), lo=lambda d: (100 * d.lo).round(1),
                     hi=lambda d: (100 * d.hi).round(1), p=lambda d: d.p.round(4), p_holm=lambda d: d.p_holm.round(4))
    print(f'A2 statistics (development): {a.ref} vs each method; q = median change of {a.ref} (%), 95 % cluster-'
          f'bootstrap CI over sessions, sessions where {a.ref} better, exact Wilcoxon over sessions, Holm.')
    print(fmt[['group', 'comparison', 'q', 'lo', 'hi', 'sess_better', 'n_sess', 'runs', 'p', 'p_holm']]
          .to_string(index=False))
    if a.out:
        out.to_csv(a.out, index=False)


if __name__ == '__main__':
    main()
