#!/usr/bin/env python3
"""Print a run's last numerically healthy time at full precision.

Transcribing this time by hand is a trap.  R6p5's is 151.56835535065707; the
six-decimal 151.568355 that reads identically is 3.5e-07 SMALLER, which puts it below
the last healthy history row and silently drops that row — the row carrying the deepest
lapse, 0.1145, which is exactly the number a collapse is characterised by.  It also
dropped one frame from the particle-movie subset.

So no script or command line should carry the literal.  Use this instead:

    T=$(python3 analysis/valid_time.py --rundir runs/prod_R6p5 \
          --ref initial_data/reference_values_R6p5.json)
    ... --tmax "$T"

Exit status is 0 if the run is healthy throughout, 2 if it failed (the time printed is
the last healthy one either way), so a caller can branch on it.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figs_session2 import valid_window     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rundir', required=True)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()
    ref = json.load(open(a.ref))
    tgood, failed, tfail = valid_window(a.rundir, ref)
    if tgood is None:
        sys.exit('no history file under %s' % a.rundir)
    print(repr(float(tgood)))
    if a.verbose:
        P = ref['P_half']
        sys.stderr.write('last healthy t = %.17g (t/P = %.9f)\n' % (tgood, tgood / P))
        if failed:
            sys.stderr.write('run FAILED at t = %.17g (t/P = %.9f)\n'
                             % (tfail, tfail / P))
        else:
            sys.stderr.write('run healthy through its final row\n')
    return 2 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
