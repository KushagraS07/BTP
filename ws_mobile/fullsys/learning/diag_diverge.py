#!/usr/bin/env python3
"""diag_diverge.py -- diagnostic (no method change): where do runs diverge (max position error > 2 m)?
Usage: python3 learning/diag_diverge.py ../results/fullsys_baselines.csv [more.csv] --methods ours hybrid fixed"""
import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv', nargs='+')
    ap.add_argument('--methods', nargs='+', default=['fixed', 'ours', 'hybrid'])
    a = ap.parse_args()
    pd.set_option('display.width', 200)
    for p in a.csv:
        df = pd.read_csv(p)
        df = df[df.method.isin(a.methods) & (df.type != 'clean')]
        print(f'\n=== {p}: share of fault runs diverged (max err > 2 m), by sensor/type/severity')
        print(df.pivot_table(index=['sensor', 'type', 'severity'], columns='method', values='diverged',
                             aggfunc='mean').round(2).to_string())
        print('\nby session (share diverged):')
        print(df.pivot_table(index='model', columns='method', values='diverged', aggfunc='mean').round(2).to_string())
        c = pd.read_csv(p)
        c = c[c.method.isin(a.methods) & (c.type == 'clean')]
        print('\nclean runs diverged:', c.groupby('method').diverged.sum().to_dict())


if __name__ == '__main__':
    main()
