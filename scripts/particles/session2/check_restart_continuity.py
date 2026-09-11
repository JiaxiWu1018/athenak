#!/usr/bin/env python3
"""Verify that a restart reproduces the state it restarted from.

Specification §5 asks that restart continuity be shown safe.  Only a real restart can
exercise it, and this session gets one at every milestone boundary: the segment for
milestone `k+1` restarts from the checkpoint written at `k P_1/2`.

Because `Driver::Finalize` writes a full output set at the end of a segment AND the
restarted segment writes its own output at the restart time, every ledger contains two
rows at exactly that time — one produced before the checkpoint was written and one
produced after it was read back.  Comparing that pair is the continuity test: a
bit-for-bit or near-roundoff match means the checkpoint captured and restored the complete
state, while a large difference means something was not in the checkpoint.

This is also why every reader in this session de-duplicates: the duplicate is expected.

Usage
    check_restart_continuity.py --rundir runs/prod_R6p5 --period 59.43569795258119
                               [--tol 1e-10]
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figs_session2 import read_hst    # noqa: E402


def duplicated_times(df, tcol='time'):
    """Times carrying more than one row."""
    c = df[tcol].value_counts()
    return sorted(float(t) for t in c[c > 1].index)


def compare(df, t, keys, tcol='time', label=''):
    rows = df[np.isclose(df[tcol], t)]
    if len(rows) < 2:
        return None
    a, b = rows.iloc[0], rows.iloc[-1]
    worst, worstk = 0.0, None
    for k in keys:
        if k not in rows.columns:
            continue
        va, vb = float(a[k]), float(b[k])
        if not (np.isfinite(va) and np.isfinite(vb)):
            continue
        d = abs(va - vb) / max(abs(va), abs(vb), 1e-300)
        if d > worst:
            worst, worstk = d, k
    return worst, worstk, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rundir', required=True)
    ap.add_argument('--period', type=float, required=True)
    ap.add_argument('--tol', type=float, default=1e-10)
    a = ap.parse_args()
    out = os.path.join(a.rundir, 'out')
    bad = 0
    checked = 0

    # --- the 20-column user history ------------------------------------------------
    hg = [p for p in glob.glob(os.path.join(out, '*.user.hst')) if '.z4c.' not in p]
    if hg:
        h = pd.read_csv(hg[0], comment='#', sep=r'\s+', header=None)
        names = None
        for line in open(hg[0]):
            if line.startswith('#') and '=' in line:
                names = [p.split('=', 1)[1] for p in line.replace('#', '').split()
                         if '=' in p]
        if names and len(names) == h.shape[1]:
            h.columns = names
        keys = [c for c in h.columns if c not in ('time', 'dt')]
        for t in duplicated_times(h):
            r = compare(h, t, keys)
            if r is None:
                continue
            worst, wk, n = r
            checked += 1
            ok = worst <= a.tol
            bad += 0 if ok else 1
            print('history   t = %-16.9g (t/P = %8.5f)  %d rows  worst relative '
                  'difference %.3e in %-12s  %s'
                  % (t, t / a.period, n, worst, wk, 'PASS' if ok else 'FAIL'))

    # --- the ADM-momentum ledger, keyed on (time, R) -------------------------------
    mg = glob.glob(os.path.join(out, '*.plummer_admmom.csv'))
    if mg:
        m = pd.read_csv(mg[0], comment='#')
        keys = ['Px_adm', 'Py_adm', 'Pz_adm', 'area_ratio', 'mean_trK',
                'Px_matter', 'Py_matter', 'Pz_matter', 'mom_l2']
        for t in duplicated_times(m):
            sub = m[np.isclose(m.time, t)]
            worst, wk, n = 0.0, None, 0
            for R, g in sub.groupby('R'):
                r = compare(g, t, keys)
                if r is None:
                    continue
                wv, k, nn = r
                n += nn
                if wv > worst:
                    worst, wk = wv, '%s @ R=%g' % (k, R)
            if n == 0:
                continue
            checked += 1
            ok = worst <= a.tol
            bad += 0 if ok else 1
            print('admmom    t = %-16.9g (t/P = %8.5f)  %d rows  worst relative '
                  'difference %.3e in %-18s  %s'
                  % (t, t / a.period, n, worst, wk, 'PASS' if ok else 'FAIL'))

    # --- the shell ledger, keyed on (time, bin) ------------------------------------
    sg = glob.glob(os.path.join(out, '*.plummer_shells.csv'))
    if sg:
        sh = pd.read_csv(sg[0], comment='#')
        keys = ['count', 'mass', 'm_r', 'm_vr', 'm_vr2', 'm_vt2', 'm_alphaW',
                'm_absL', 'd1', 'd2', 'd3']
        for t in duplicated_times(sh):
            sub = sh[np.isclose(sh.time, t)]
            worst, wk, n = 0.0, None, 0
            for ib, g in sub.groupby('bin'):
                r = compare(g, t, keys)
                if r is None:
                    continue
                wv, k, nn = r
                n += nn
                if wv > worst:
                    worst, wk = wv, '%s @ bin %s' % (k, ib)
            if n == 0:
                continue
            checked += 1
            ok = worst <= a.tol
            bad += 0 if ok else 1
            print('shells    t = %-16.9g (t/P = %8.5f)  %d rows  worst relative '
                  'difference %.3e in %-18s  %s'
                  % (t, t / a.period, n, worst, wk, 'PASS' if ok else 'FAIL'))

    if checked == 0:
        print('no duplicated output times found: no restart boundary in this run yet')
        return 0
    print('\n%d restart boundaries checked, %d FAIL (tolerance %.1e relative)'
          % (checked, bad, a.tol))
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
