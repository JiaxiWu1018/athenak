"""Shared matplotlib style for the Plummer campaign figures.

Categorical hues are taken in FIXED ORDER from the validated default palette
(dataviz skill, references/palette.md); slots are never cycled and never reassigned
by rank. Sequential magnitude uses a single hue, light to dark. Because three
light-mode slots fall below 3:1 contrast on a light surface, every figure with two
or more series carries BOTH a legend and direct labels, so identity is never
carried by colour alone. No figure uses two y-scales.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8880"
# categorical slots, fixed order (blue, orange, aqua, yellow, magenta, green, violet, red)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
REFLINE = "#52514e"          # the continuum reference is ink, not a series colour
BAND = "#d7d6d0"


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#e6e5df",
        "grid.linewidth": 0.7,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "text.color": INK,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "lines.markersize": 4.5,
        "figure.dpi": 140,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    })


def label_end(ax, x, y, text, color, va="center"):
    """Direct label at the right end of a series (identity never colour-alone)."""
    ax.annotate(text, xy=(x, y), xytext=(5, 0), textcoords="offset points",
                color=color, fontsize=9, va=va, ha="left",
                fontweight="semibold", clip_on=False)


REF_R = {
    "b": 20.0, "r_half": 26.007614229, "R_half": 25.209342804,
    "R_t": 398.999373433, "r_t": 400.0, "r_vcmax": 28.284271247,
    "P_half": 1192.496781,
}
# isotropic half-widths of the fixed refinement seams (production mesh)
SEAMS = [64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0]


def mark_scales(ax, seams=True, which=("b", "R_half", "R_t"), annotate=True):
    """Vertical guides for the physical scales and the refinement seams."""
    lbl = {"b": "b", "R_half": r"$R_{1/2}$", "R_t": r"$R_t$", "r_t": r"$r_t$",
           "r_half": r"$r_{1/2}$", "r_vcmax": r"$b\sqrt{2}$"}
    if seams:
        for s in SEAMS:
            ax.axvline(s, color="#c9c8c1", lw=0.7, ls="--", zorder=0)
    for k in which:
        ax.axvline(REF_R[k], color=MUTED, lw=0.9, ls=":", zorder=0)
        if annotate:
            ax.annotate(lbl.get(k, k), xy=(REF_R[k], 1.0),
                        xycoords=("data", "axes fraction"),
                        xytext=(2, -11), textcoords="offset points",
                        color=MUTED, fontsize=8, ha="left")
