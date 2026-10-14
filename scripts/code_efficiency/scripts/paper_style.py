#!/usr/bin/env python3
"""Shared publication style for the NRPIC method-paper figures."""

from __future__ import annotations

import shutil

import matplotlib as mpl


def configure() -> bool:
    """Apply the shared style and return whether external LaTeX is available."""
    use_tex = shutil.which("latex") is not None
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "text.usetex": use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.size": 9.5,
        "axes.labelsize": 10.0,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.0,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    return use_tex


def panel_label(ax, label: str, *, x: float = 0.02, y: float = 0.98,
                color: str = "black") -> None:
    text = ax.text2D if hasattr(ax, "text2D") else ax.text
    text(x, y, label, transform=ax.transAxes, ha="left", va="top",
         color=color, fontsize=10.0, fontweight="bold")

