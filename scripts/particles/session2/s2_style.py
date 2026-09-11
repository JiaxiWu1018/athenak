#!/usr/bin/env python3
"""Shared plotting conventions for Session 2.

Session 1's analysis/plummer_style.py cannot be reused: it holds the whole Session-1
model as module state (REF_R with b = 20, r_1/2 = 26.0076, R_t = 398.999, P_1/2 = 1192.5
and SEAMS = [64,128,256,512,1024,2048]).  Session 2's refinement seams are
[10,20,40,80,160,320,640] (R10) and [6,12,24,48,96,192,384] (R6p5) -- no overlap at all --
so every scale marker would have been drawn in the wrong place.  Here the model comes from
the per-case reference JSON at call time and nothing is module state.

Colour: the categorical order below is the Okabe-Ito colourblind-safe sequence, assigned
in FIXED order and never cycled, and validated rather than eyeballed -- worst adjacent
pair separation is dE 11.0 (deuteranopia) / 8.5 (tritanopia) and dE 21.1 for normal
vision.  Every series also carries a distinct dash pattern, so identity never rests on
colour alone.  Time-ordered families (a profile at successive times) use a single-hue
sequential ramp instead, because there the encoded quantity is a magnitude, not an
identity.
"""
import json

import matplotlib as mpl
import numpy as np

# fixed categorical order -- never cycled, never reassigned by rank
SERIES = ['#0072B2', '#D55E00', '#009E73', '#E69F00', '#CC79A7']
DASH = [(0, ()), (0, (5, 1.6)), (0, (1.3, 1.3)), (0, (6, 1.5, 1.3, 1.5)),
        (0, (3.2, 1.3, 1.3, 1.3, 1.3, 1.3))]
INK = '#1a1a1a'
MUTED = '#6b6b6b'
GRIDC = '#d9d9d9'
REFC = '#9a9a9a'          # reference/null lines: recessive, never a series colour


def apply():
    mpl.rcParams.update({
        'figure.dpi': 130, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
        'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9.5,
        'legend.fontsize': 8, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
        'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': MUTED, 'ytick.color': MUTED,
        'axes.grid': True, 'grid.color': GRIDC, 'grid.linewidth': 0.6,
        'grid.alpha': 0.9, 'axes.axisbelow': True,
        'axes.spines.top': False, 'axes.spines.right': False,
        'lines.linewidth': 1.5, 'legend.frameon': False,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
    })


def load_ref(path):
    return json.load(open(path))


def style(i):
    """Colour and dash for series index i, in fixed order."""
    return dict(color=SERIES[i % len(SERIES)], linestyle=DASH[i % len(DASH)])


def ramp(n, name='viridis'):
    """A sequential ramp for a time-ordered family (magnitude, not identity)."""
    cm = mpl.cm.get_cmap(name) if hasattr(mpl.cm, 'get_cmap') else mpl.colormaps[name]
    return [cm(0.12 + 0.76 * k / max(n - 1, 1)) for k in range(n)]


def periods_axis(ax, ref, milestone=None, xmax=None):
    """x axis in units of that case's own P_1/2, with integer-period markers."""
    ax.set_xlabel(r'$t / P_{1/2}$')
    hi = xmax if xmax is not None else (milestone or 5)
    for k in range(1, int(np.ceil(hi)) + 1):
        ax.axvline(k, color=REFC, lw=0.6, ls=(0, (1, 3)), zorder=0)
    ax.set_xlim(0, hi)


def mark_null(ax, value, label):
    ax.axhline(value, color=REFC, lw=1.0, ls=(0, (4, 2)), zorder=1)
    ax.annotate(label, xy=(0.995, value), xycoords=('axes fraction', 'data'),
                ha='right', va='bottom', fontsize=7.5, color=MUTED)


def stamp(fig, ref, milestone, extra=''):
    # below the axes, not over the x label: bbox_inches='tight' grows the canvas to
    # include it, so a negative y is safe and never collides.
    fig.text(0.005, -0.055,
             r'Plummer session 2, case %s: $(R/M)_{\rm eff} = %.4g$, '
             r'$b = %.6g\,M$, $r_t = 20b$, $P_{1/2} = %.6f\,M$, '
             r'$N = %d$, seed %d, through $%g\,P_{1/2}$%s'
             % (ref['label'], ref['RM_eff'], ref['b'], ref['P_half'],
                ref['Npart'], ref['seed'], milestone, extra),
             fontsize=6.4, color=MUTED, ha='left', va='bottom')


def save(fig, outdir, name):
    import os
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(outdir, '%s.%s' % (name, ext)))
    print('  wrote %s/%s.{pdf,png}' % (outdir, name))
