#!/usr/bin/env python3
"""a3_summary.py -- A3: seed robustness (FINDINGS A3). Seed 42 = results/fullsys_baselines.csv; other seeds
results/seed<k>/fullsys_baselines.csv (methods ours, hybrid, no_pslip). 'fixed' does not depend on the seed.
Usage: python3 learning/a3_summary.py --seeds 42 1 2"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from zupt import paths as _P  # noqa: E402

RES = os.path.join(_P.WS, 'results')
METHODS = ['ours', 'hybrid', 'no_pslip']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 1, 2])
    a = ap.parse_args()
    base = pd.read_csv(os.path.join(RES, 'fullsys_baselines.csv'))
    fixed = base[base.method == 'fixed'][['run', 'type', 'rmse']].rename(columns={'rmse': 'fixed'})
    rows = []
    for sd in a.seeds:
        p = os.path.join(RES, 'fullsys_baselines.csv') if sd == 42 else os.path.join(RES, f'seed{sd}', 'fullsys_baselines.csv')
        df = pd.read_csv(p)
        for m in METHODS:
            x = df[df.method == m][['run', 'rmse']].merge(fixed, on='run')
            for g, s in (('clean', x[x.type == 'clean']), ('fault', x[x.type != 'clean'])):
                rows.append(dict(seed=sd, method=m, group=g, n=len(s), mean_rmse=s.rmse.mean(),
                                 wins_vs_fixed=int((s.rmse < s.fixed).sum()),
                                 median_vs_fixed=100 * np.median(s.rmse / s.fixed - 1)))
        h = df[df.method == 'hybrid'][['run', 'rmse']].merge(df[df.method == 'ours'][['run', 'rmse']], on='run',
                                                              suffixes=('_h', '_o')).merge(fixed, on='run')
        for g, s in (('clean', h[h.type == 'clean']), ('fault', h[h.type != 'clean'])):
            rows.append(dict(seed=sd, method='hybrid_vs_ours', group=g, n=len(s),
                             mean_rmse=np.nan, wins_vs_fixed=int((s.rmse_h < s.rmse_o).sum()),
                             median_vs_fixed=100 * np.median(s.rmse_h / s.rmse_o - 1)))
    out = pd.DataFrame(rows)
    pd.set_option('display.width', 200)
    print('A3 seed robustness (development). wins_vs_fixed / median_vs_fixed: for hybrid_vs_ours = hybrid wins / '
          'median change vs ours.')
    print(out.round(3).to_string(index=False))
    print('\nspread over seeds (mean RMSE, m):')
    sp = out[out.method != 'hybrid_vs_ours'].groupby(['method', 'group']).mean_rmse.agg(['min', 'max', 'mean'])
    print(sp.round(3).to_string())
    fx = {g: fixed[(fixed.type == 'clean') == (g == 'clean')].fixed.mean() for g in ('clean', 'fault')}
    ok = True
    for sd in a.seeds:
        for g, need in (('clean', 7), ('fault', int(np.ceil(0.8 * 208)))):
            r = out[(out.seed == sd) & (out.method == 'hybrid') & (out.group == g)].iloc[0]
            ok &= r.wins_vs_fixed >= need
    for g in ('clean', 'fault'):
        r = sp.loc[('hybrid', g)]
        ok &= (r['max'] - r['min']) < 0.25 * (fx[g] - r['mean'])
    print(f'\nRobustness rule (A3): every seed hybrid beats fixed on >= 7/8 clean and >= 80 % fault runs, AND seed '
          f'range of hybrid mean RMSE < 25 % of the hybrid-fixed gap -> {"ROBUST" if ok else "NOT robust"}')


if __name__ == '__main__':
    main()
