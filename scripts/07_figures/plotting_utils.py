"""
Shared plotting helpers -- RSE-style, publication-quality output.
Uses the paper's fixed palette (new for this analysis, distinct from the
viridis/inferno families used in the prior harmonization-audit figures).
"""
# Original script filename (sanitized): plotting_utils.py; public name: plotting_utils.py
import os as _os, sys as _sys
_p = _os.path.dirname(_os.path.abspath(__file__))
while _os.path.basename(_p) != 'scripts' and _os.path.dirname(_p) != _p:
    _p = _os.path.dirname(_p)
for _d, _dn, _fn in _os.walk(_p):  # sibling script folders are importable by module name
    if '__pycache__' not in _d and _d not in _sys.path:
        _sys.path.append(_d)
# Paths are configurable: set SIF_GPP_ROOT (working project/data root) and optionally SIF_GPP_DATA_ROOT
# (parent folder holding raw/third-party inputs). See docs/reproducibility.md.
PROJECT_ROOT = _os.environ.get("SIF_GPP_ROOT", ".")
DATA_ROOT = _os.environ.get("SIF_GPP_DATA_ROOT", PROJECT_ROOT)
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.font_manager as fm

SEQ_PALETTE = ["#16324F", "#235789", "#2C7A7B", "#3FA796", "#7CCBA2", "#B8DE6F", "#F6D55C"]
DIV_PALETTE = ["#3B4CC0", "#7B9FF9", "#DCE6F2", "#F7F7F7", "#F5C4A5", "#D6604D", "#8C2D24"]

PRODUCT_COLORS = {
    "GOSIF": "#1B4965",
    "GOME-2": "#5FA8A0",
    "TROPOSIF": "#C9A227",
    "OCO-2": "#D97D54",
    "OCO-3": "#8E5572",
}
PRODUCT_ORDER = ["GOSIF", "GOME-2", "TROPOSIF", "OCO-2", "OCO-3"]

SEQ_CMAP = LinearSegmentedColormap.from_list("paper_seq", SEQ_PALETTE, N=256)
DIV_CMAP = LinearSegmentedColormap.from_list("paper_div", DIV_PALETTE, N=256)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.7,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


def clean_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.6)
    return ax


def savefig(fig, out_dir, basename, formats=("png", "pdf", "svg", "tif"), tight=True):
    """tight=False for figures using cartopy GeoAxes with fixed aspect ratio --
    bbox_inches='tight' miscomputes the bounding box for aspect-locked geoaxes
    (uses the pre-aspect-correction extent), producing huge blank canvases.
    Use constrained_layout=True on the figure instead when tight=False."""
    os.makedirs(out_dir, exist_ok=True)
    bbox = "tight" if tight else None
    for fmt in formats:
        path = os.path.join(out_dir, f"{basename}.{fmt}")
        if fmt == "tif":
            fig.savefig(path, format="tiff", dpi=350, bbox_inches=bbox)
        else:
            fig.savefig(path, format=fmt, dpi=300 if fmt == "png" else None, bbox_inches=bbox)
    print(f"Saved {basename} -> {out_dir} ({', '.join(formats)})")
