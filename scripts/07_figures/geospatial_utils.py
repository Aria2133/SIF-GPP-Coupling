"""
Shared helpers for the geospatial-candidate figure exploration.
New diverging palette (distinct from the earlier cool-to-warm palette used in
Figure 3) and consistent basemap styling / missingness-marker vocabulary
across all five candidates.
"""
# Original script filename (sanitized): geospatial_utils.py; public name: geospatial_utils.py
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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature

NEW_DIV_PALETTE = ["#2C4C9C", "#6C8CD5", "#D9E3F0", "#F6E8E1", "#D98C6A", "#A63D40"]
NEW_DIV_CMAP = LinearSegmentedColormap.from_list("mechanistic_geo_div", NEW_DIV_PALETTE, N=256)

LAND_COLOR = "#EFE7D8"
OCEAN_COLOR = "#E7EEF1"
COAST_COLOR = "#A9A08C"
BORDER_COLOR = "#C9BFA8"
GRAY_NEUTRAL = "#9C9689"
GRAY_LIGHT = "#D8D3C7"

PRODUCT_COLORS = {
    "GOSIF": "#1B4965",
    "GOME-2": "#5FA8A0",
    "TROPOSIF": "#C9A227",
    "OCO-2": "#D97D54",
    "OCO-3": "#8E5572",
}
PRODUCT_ORDER = ["GOSIF", "GOME-2", "TROPOSIF", "OCO-2", "OCO-3"]

VMIN, VMAX = -0.6, 0.6

REASON_STYLE = {
    "ESTIMATE_OK": dict(marker="o", label="Reliable estimate (N\u226512, CI excludes 0)"),
    "CI_SPANS_ZERO": dict(marker="o", label="CI includes 0 (N\u226512)"),
    "BOOTSTRAP_NOT_RUN": dict(marker="o", label="No bootstrap run (N<12)"),
    "INSUFFICIENT_DESEASONALIZED_MONTHS": dict(marker="D", label="r undefined (single-year record)"),
    "N_BELOW_THRESHOLD": dict(marker="v", label="N<3 (no correlation possible)"),
}


def add_basemap(ax, extent=None, borders=False):
    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, facecolor=OCEAN_COLOR, zorder=0)
    ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR, zorder=0, edgecolor="none")
    ax.coastlines(linewidth=0.4, color=COAST_COLOR, zorder=1)
    if borders:
        ax.add_feature(cfeature.BORDERS, linewidth=0.25, edgecolor=BORDER_COLOR, zorder=1)
    ax.spines["geo"].set_visible(False) if hasattr(ax.spines, "__getitem__") and "geo" in ax.spines else None


def size_scale(n, k=55.0, ref=150.0, floor=12.0):
    return floor + k * np.sqrt(np.clip(n, 1, None) / ref)


def load_audit():
    ROOT = PROJECT_ROOT
    OUT_ROOT = os.path.join(ROOT, "SIF_EC_Mechanistic_FootprintAware_SIF_GPP_Paper")
    path = os.path.join(OUT_ROOT, "03_figures_main", "Geospatial_Candidates", "06_reports",
                         "Missing_Geospatial_Estimate_Audit.csv")
    return pd.read_csv(path)


def savefig_multi(fig, out_dir, basename, tight=True):
    os.makedirs(out_dir, exist_ok=True)
    bbox = "tight" if tight else None
    for fmt in ("png", "pdf", "svg", "tif"):
        path = os.path.join(out_dir, f"{basename}.{fmt}")
        if fmt == "tif":
            fig.savefig(path, format="tiff", dpi=350, bbox_inches=bbox)
        else:
            fig.savefig(path, format=fmt, dpi=300 if fmt == "png" else None, bbox_inches=bbox)
    print(f"Saved {basename} -> {out_dir}")


REGION_BOUNDS = [
    ("North America", -170, -50, 15, 75),
    ("South America", -90, -30, -58, 13),
    ("Europe", -15, 45, 34, 72),
    ("Africa", -20, 52, -36, 34),
    ("East & South Asia", 52, 150, -10, 55),
    ("Australia", 110, 155, -47, -9),
]


def assign_region(lat, lon):
    for name, lon_min, lon_max, lat_min, lat_max in REGION_BOUNDS:
        if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
            return name
    return "Other"
