#!/usr/bin/env python3
"""final_stats.py -- FR (FINDINGS FR): statistics for the confirmatory run.

Reads results/final/final_<set>.csv (whatever exists). Method of A2 (learning/stats.py): q = median over runs of
RMSE_hybrid / RMSE_X - 1, 95 % cluster bootstrap over sessions (10 000, seed 0), exact two-sided Wilcoxon over
per-session median log ratios.
PRIMARY family = main set, FAULT runs, hybrid vs fixed / chi2 / sage_husa / li (Holm over these 4).
  CONFIRMED  if q < 0 and CI upper < 0 and Holm p < 0.05;  else NOT CONFIRMED.
No-harm check: main set, clean runs, hybrid vs fixed.   Everything else: secondary (CI + unadjusted p).
Writes results/final/final_stats.csv.
Usage: python3 learning/final_stats.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from learning.stats import SEED, compare, holm  # noqa: E402
from learning.final_eval import ABLATIONS, OUT_DIR  # noqa: E402

PRIMARY_VS = ['fixed', 'chi2', 'sage_husa', 'li']
ALPHA = 0.05


def load(setname):
    p = os.path.join(OUT_DIR, f'final_{setname}.csv')
    return pd.read_csv(p) if os.path.exists(p) else None


def comp_rows(df, family, group, vs, rng, metric='rmse'):
    sub = df.copy()
    if metric != 'rmse':
        sub = sub.assign(rmse=sub[metric])
    rows = []
    for m in vs:
        if m not in set(sub.method):
            continue
        r = compare(sub[sub.method.isin(['hybrid', m])], 'hybrid', m, rng)
        rows.append(dict(family=family, group=group, metric=metric, comparison=f'hybrid vs {m}', **r))
    return rows


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    main_df = load('main')
    if main_df is None:
        sys.exit('ABORT: results/final/final_main.csv missing')
    fault = main_df[main_df.type != 'clean']
    clean = main_df[main_df.type == 'clean']
    prim = pd.DataFrame(comp_rows(fault, 'PRIMARY', 'main faults', PRIMARY_VS, rng))
    prim['p_holm'] = holm(prim.p.values)
    prim['verdict'] = np.where((prim.q < 0) & (prim.hi < 0) & (prim.p_holm < ALPHA), 'CONFIRMED', 'NOT CONFIRMED')
    rows += prim.to_dict('records')
    rows += comp_rows(clean, 'no-harm', 'main clean', ['fixed'], rng)
    rows += comp_rows(clean, 'secondary', 'main clean', PRIMARY_VS[1:], rng)
    rows += comp_rows(fault, 'ablation', 'main faults', ABLATIONS, rng)
    rows += comp_rows(clean, 'ablation', 'main clean', ABLATIONS, rng)
    for setname, groups in (('sigma', None), ('end', None), ('lidar', ('lidar', 'overlap')), ('unseen', None)):
        df = load(setname)
        if df is None:
            continue
        vs = [m for m in df.method.unique() if m != 'hybrid']
        if groups:
            for g in groups:
                rows += comp_rows(df[df.type == g], 'secondary', f'{setname}:{g}', vs, rng)
        else:
            rows += comp_rows(df, 'secondary', setname, vs, rng)
        if setname == 'end':
            rows += comp_rows(df, 'secondary', 'end (after fault)', vs, rng, metric='rmse_after')
        if setname == 'unseen':
            for t, g in df.groupby('type'):
                rows += comp_rows(g, 'secondary', f'unseen:{t}', PRIMARY_VS, rng)
    out = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(os.path.join(OUT_DIR, 'final_stats.csv'), index=False)
    pd.set_option('display.width', 220)
    pd.set_option('display.max_rows', 500)
    f = out.assign(q=(100 * out.q).round(1), lo=(100 * out.lo).round(1), hi=(100 * out.hi).round(1),
                   p=out.p.round(5))
    if 'p_holm' in f:
        f['p_holm'] = f.p_holm.round(5)
    cols = ['family', 'group', 'metric', 'comparison', 'q', 'lo', 'hi', 'sess_better', 'n_sess', 'runs', 'p']
    print('FINAL (confirmatory) statistics: q = median RMSE change of hybrid vs X (%), 95 % cluster-bootstrap CI over '
          'sessions, sessions where hybrid better, exact Wilcoxon over sessions.\n')
    print('PRIMARY family (Holm over 4):')
    print(f[f.family == 'PRIMARY'][cols + ['p_holm', 'verdict']].to_string(index=False))
    print('\nNo-harm (clean, hybrid vs fixed) and secondary analyses (unadjusted p, descriptive):')
    print(f[f.family != 'PRIMARY'][cols].to_string(index=False))
    print('\nMean RMSE by set and method (m):')
    for setname in ('main', 'sigma', 'end', 'lidar', 'unseen'):
        df = load(setname)
        if df is not None:
            print(f'[{setname}]')
            print(df.pivot_table(index='type', columns='method', values='rmse', aggfunc='mean').round(3).to_string())


if __name__ == '__main__':
    main()
