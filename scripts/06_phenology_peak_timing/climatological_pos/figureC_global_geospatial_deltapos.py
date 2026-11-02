"""
FIGURE C -- Global geospatial ΔPOS map, one panel per product.

Marker color = DELTA_POS = POS_SIF - POS_GPP (circular), symmetric diverging colormap
centered on zero: negative (blue) = SIF peaks earlier, zero (white) = synchronous,
positive (red) = GPP peaks earlier. Marker size = number of distinct years of data
(temporal support) -- more years behind a tower's POS estimate = a bigger, more
trustworthy marker, never a continuous interpolated surface between sparse towers.

Basemap: real Natural Earth I shaded-relief raster (NE1_HR_LC_SR_W.tif), loaded via the
verbatim load_basemap()/add_basemap() helpers already used elsewhere in this project
(All_the_Visualizations_REVISED_4PRODUCT_FINAL_ARIDITY_JUNE_DIAGNOSTICS/10_scripts/
build_part1_aridity_context_maps.py) -- rasterio decimated read, cartopy EqualEarth
projection, ~32% white blend. No interpolation of sparse tower values onto a continuous
surface; only point markers at real tower locations, matching this project's standing
basemap-usage rule.
"""
# Original script filename (sanitized): figureC_global_geospatial_deltapos.py; public name: figureC_global_geospatial_deltapos.py
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
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import rasterio
from rasterio.enums import Resampling
import cartopy.crs as ccrs

warnings.filterwarnings("ignore")

import pos_lib as pl

OUT_DIR = (Path(PROJECT_ROOT) / 'All_the_Visualizations_REVISED/EXACT_REQUESTS_ONLY/PHENOLOGY_POS_REVISION/figureC_global_geospatial_deltapos.py').resolve().parent
BASEMAP_TIF = pl.BASEMAP_TIF
PRODUCTS = pl.PRODUCTS


def load_basemap(target_width_px: int = 4000, white_blend: float = 0.32):
    with rasterio.open(BASEMAP_TIF) as src:
        assert src.crs.to_epsg() == 4326, f"Unexpected basemap CRS: {src.crs}"
        scale = target_width_px / src.width
        out_width = target_width_px
        out_height = max(1, round(src.height * scale))
        arr = src.read(out_shape=(src.count, out_height, out_width), resampling=Resampling.average)
        bounds = src.bounds
    rgb = np.transpose(arr, (1, 2, 0)).astype(np.float32) / 255.0
    rgb = rgb * (1 - white_blend) + white_blend * 1.0
    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    return rgb, extent


def add_basemap(ax, rgb, extent):
    ax.imshow(rgb, origin="upper", extent=extent, transform=ccrs.PlateCarree(), interpolation="bilinear", zorder=0)
    ax.set_global()
    ax.spines["geo"].set_edgecolor("#888888")
    ax.spines["geo"].set_linewidth(0.5)


def main():
    tower_pos = pd.read_csv((Path(PROJECT_ROOT) / 'All_the_Visualizations_REVISED/EXACT_REQUESTS_ONLY/PHENOLOGY_POS_REVISION/figureC_global_geospatial_deltapos.py').resolve().parent / "tower_pos_table.csv")
    valid = tower_pos.dropna(subset=["DELTA_POS", "LATITUDE", "LONGITUDE"]).copy()
    valid["N_YEARS_SUPPORT"] = valid[["SIF_N_YEARS", "GPP_N_YEARS"]].min(axis=1)

    rgb, extent = load_basemap()
    norm = TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)
    cmap = plt.get_cmap("RdBu_r")

    fig = plt.figure(figsize=(15.5, 8.6), facecolor="white")
    for i, product in enumerate(PRODUCTS):
        ax = fig.add_subplot(2, 2, i + 1, projection=ccrs.EqualEarth())
        add_basemap(ax, rgb, extent)
        sub = valid[valid["SIF_PRODUCT"] == product]
        sizes = 18 + 9 * sub["N_YEARS_SUPPORT"].clip(upper=12)
        sc = ax.scatter(sub["LONGITUDE"], sub["LATITUDE"], c=sub["DELTA_POS"], cmap=cmap, norm=norm,
                        s=sizes, transform=ccrs.PlateCarree(), edgecolor="#333333", linewidth=0.5, zorder=5)
        ax.set_title(f"{product}  (n={len(sub)} towers)", fontsize=12, fontweight="bold")

    cax = fig.add_axes([0.30, 0.045, 0.40, 0.022])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal", extend="both")
    cb.set_label("ΔPOS = POS$_{SIF}$ − POS$_{GPP}$ (months)   "
                  "← SIF peaks earlier   |   synchronous   |   GPP peaks earlier →", fontsize=8.6)

    size_legend_years = [1, 5, 10]
    size_handles = [plt.scatter([], [], s=18 + 9 * v, color="#888888", edgecolor="#333333", linewidth=0.5,
                                label=f"{v} yr") for v in size_legend_years]
    fig.legend(handles=size_handles, loc="lower left", bbox_to_anchor=(0.02, 0.02), fontsize=8,
               frameon=False, title="Marker size: min(years of\nSIF, GPP) data", title_fontsize=7.6)

    fig.suptitle("Where does SIF peak earlier (or later) than GPP?", fontsize=16, fontweight="bold", y=0.99)
    fig.text(0.5, 0.955,
              "Each point = one EC tower's fitted Peak-Of-Season offset for that product. Real tower locations "
              "only -- no interpolation between sparse towers. Basemap: Natural Earth I shaded relief.",
              ha="center", fontsize=9.2, color="#333333")
    fig.subplots_adjust(top=0.90, bottom=0.10, hspace=0.15, wspace=0.05)

    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT_DIR / f"FigureC_Global_DeltaPOS_Map.{ext}", dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("Saved FigureC_Global_DeltaPOS_Map.{png,svg,pdf}")
    print(valid.groupby("SIF_PRODUCT")["DELTA_POS"].describe())


if __name__ == "__main__":
    main()
