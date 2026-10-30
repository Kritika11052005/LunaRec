"""Shared figure style for the LunaRec notebooks.

Defined once so the five notebooks read as one document rather than as five
sets of matplotlib defaults. Import at the top of each notebook:

    import sys; sys.path.insert(0, str(ROOT / "src"))
    from lunarec_style import *
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID = "#fcfcfb", "#e1e0d9"

# Categorical slots, in fixed order. Never cycled, never reassigned by rank:
# a given entity keeps its colour even when a filter changes how many series
# are on screen.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

# Sequential = ONE hue, light -> dark. Never a rainbow, which implies category
# boundaries where the data has continuous magnitude.
BLUES = LinearSegmentedColormap.from_list(
    "lunarec_blue",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "normal",
    "figure.dpi": 120, "savefig.dpi": 200, "savefig.facecolor": SURFACE,
})


def bare(ax):
    """Recessive frame: grid stays, spines go. The data carries the figure."""
    ax.grid(True, linewidth=0.6, alpha=0.7)
    for s in ax.spines.values():
        s.set_visible(False)
    return ax