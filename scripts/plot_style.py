"""Shared plotting style for compact, readable analysis figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager


COLUMN_WIDTH_IN = 3.35
COLUMN_FIGSIZE = (COLUMN_WIDTH_IN, 1.84)
DOUBLE_FIGSIZE = (7.05, 2.55)

BLUE = "#4C78A8"
RED = "#E45756"
ORANGE = "#F2A65A"
PURPLE = "#7A5195"
GREEN = "#59A14F"
TEAL = "#76B7B2"
LIGHT_BLUE = "#A0CBE8"
SAND = "#D9B48F"
GRAY = "#9D9D9D"
STACK_COLORS = [BLUE, RED, ORANGE, PURPLE, GREEN, SAND, LIGHT_BLUE, TEAL, GRAY]

_TIMES_FONT_FILES = (
    "/usr/share/fonts/TIMES.TTF",
    "/usr/share/fonts/TIMESBD.TTF",
    "/usr/share/fonts/TIMESI.TTF",
    "/usr/share/fonts/TIMESBI.TTF",
)


def apply_publication_style() -> None:
    """Apply a compact serif style suitable for multi-panel figures."""

    for font_file in _TIMES_FONT_FILES:
        if Path(font_file).is_file():
            font_manager.fontManager.addfont(font_file)

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Liberation Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.0,
            "axes.linewidth": 0.6,
            "axes.edgecolor": "#555555",
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.fontsize": 6.2,
            "legend.frameon": False,
            "lines.linewidth": 1.2,
            "grid.color": "#D7D7D7",
            "grid.linewidth": 0.45,
            "grid.alpha": 0.75,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def finish_axis(axis, *, grid_axis: str = "both") -> None:
    axis.grid(True, axis=grid_axis)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#666666")
