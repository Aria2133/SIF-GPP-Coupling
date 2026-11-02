"""
FIGURE 3 (redesign) -- Global DeltaPOS map + biome-count bar chart, per SIF product.

Redesign of fig3_global_deltapos_map_classified.py's Figure3_Global_DeltaPOS_Map: same
underlying data (tower_pos_table.csv, from PHENOLOGY_POS_REVISION -- NOT recomputed),
same real Natural Earth basemap, same category DEFINITIONS the original figure already
used for its per-panel summary text (SIF earlier: DELTA_POS < -0.25 mo; Synchronous:
|DELTA_POS| <= 0.25 mo; GPP earlier: DELTA_POS > 0.25 mo) -- now applied as the map's
own marker coloring (replacing the original's 5-bin continuous-style shading) AND to a
new paired bar-chart panel per product showing biome x category tower counts.

Layout: one row per product (GOSIF, GOME-2, TROPOSIF, OCO-3); left = map, right =
biome count bar chart. Category colors are identical between the map and the bars, and
biome ordering is identical across all four bar panels (fixed, not per-panel-sorted).
"""
# Original script filename (sanitized): fig3b_global_deltapos_map_with_biome_counts.py; public name: fig11_global_deltapos_map.py
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
from matplotlib.colors import ListedColormap, BoundaryNorm
import rasterio
from rasterio.enums import Resampling
import cartopy.crs as ccrs

warnings.filterwarnings("ignore")

PHEN_DIR = Path(_os.path.join(PROJECT_ROOT, 'All_the_Visualizations_REVISED', 'EXACT_REQUESTS_ONLY', 'PHENOLOGY_POS_REVISION'))
OUT_DIR = (Path(PROJECT_ROOT) / 'SIF_GPP_PeakTiming_VISUAL_REDESIGN_FINAL/fig3b_global_deltapos_map_with_biome_counts.py').resolve().parent
BASEMAP_TIF = Path(_os.environ.get("NE_BASEMAP_TIF", "NE1_HR_LC_SR_W.tif"))
PRODUCTS = ["GOSIF", "GOME-2", "TROPOSIF", "OCO-3"]

# --- category definitions: IDENTICAL to the thresholds already used for the original
# figure's own per-panel summary text (pct_sif / pct_sync / pct_gpp) -- not new numbers.
SYNC_BAND = 0.25  # months
CATEGORIES = ["SIF earlier", "Synchronous", "GPP earlier"]
CATEGORY_COLORS = {"SIF earlier": "#2166ac", "Synchronous": "#f2f2ae", "GPP earlier": "#b2182b"}
CATEGORY_EDGECOLOR = "#333333"

IGBP_ORDER = ["ENF", "EBF", "DBF", "MF", "CSH", "OSH", "WSA", "SAV", "GRA", "CRO"]
IGBP_LABELS = {
    "ENF": "Evergreen Needleleaf Forest", "EBF": "Evergreen Broadleaf Forest",
    "DBF": "Deciduous Broadleaf Forest", "MF": "Mixed Forest", "CSH": "Closed Shrubland",
    "OSH": "Open Shrubland", "WSA": "Woody Savanna", "SAV": "Savanna",
    "GRA": "Grassland", "CRO": "Cropland",
}


def classify(delta_pos: float) -> str:
    if delta_pos < -SYNC_BAND:
        return "SIF earlier"
    if delta_pos > SYNC_BAND:
        return "GPP earlier"
    return "Synchronous"


def load_basemap(target_width_px: int = 3000, white_blend: float = 0.32):
    with rasterio.open(BASEMAP_TIF) as src:
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
    tower_pos = pd.read_csv(PHEN_DIR / "tower_pos_table.csv")
    valid = tower_pos.dropna(subset=["DELTA_POS", "LATITUDE", "LONGITUDE"]).copy()
    valid["N_YEARS_SUPPORT"] = valid[["SIF_N_YEARS", "GPP_N_YEARS"]].min(axis=1)
    valid["CATEGORY"] = valid["DELTA_POS"].apply(classify)

    biomes_present = [b for b in IGBP_ORDER if (valid["IGBP"] == b).any()]

    # biome x category x product counts (the actual aggregation used everywhere below,
    # also exported as a CSV so the bar-chart values are independently checkable)
    count_table = (valid.groupby(["SIF_PRODUCT", "IGBP", "CATEGORY"]).size()
                   .reset_index(name="N_TOWERS"))
    count_table.to_csv(OUT_DIR / "figure3_biome_category_counts.csv", index=False)

    rgb, extent = load_basemap()
    cmap = ListedColormap([CATEGORY_COLORS[c] for c in CATEGORIES])
    norm = BoundaryNorm(range(4), cmap.N)
    cat_to_code = {c: i for i, c in enumerate(CATEGORIES)}

    fig = plt.figure(figsize=(15.5, 17.0), facecolor="white")
    gs = fig.add_gridspec(4, 2, width_ratios=[1.55, 1], hspace=0.30, wspace=0.18)

    for row, product in enumerate(PRODUCTS):
        sub = valid[valid["SIF_PRODUCT"] == product].copy()

        # --- left: map ---
        ax_map = fig.add_subplot(gs[row, 0], projection=ccrs.EqualEarth())
        add_basemap(ax_map, rgb, extent)
        sub["CODE"] = sub["CATEGORY"].map(cat_to_code)
        sizes = 18 + 9 * sub["N_YEARS_SUPPORT"].clip(upper=12)
        ax_map.scatter(sub["LONGITUDE"], sub["LATITUDE"], c=sub["CODE"], cmap=cmap, norm=norm,
                       s=sizes, transform=ccrs.PlateCarree(), edgecolor=CATEGORY_EDGECOLOR, linewidth=0.5, zorder=5)
        ax_map.set_title(f"{product}   (N = {len(sub)} towers)", fontsize=13, fontweight="bold", pad=6, loc="left")

        # --- right: biome count bar chart (stacked, category-colored, fixed biome order) ---
        ax_bar = fig.add_subplot(gs[row, 1])
        y = np.arange(len(biomes_present))
        left = np.zeros(len(biomes_present))
        for cat in CATEGORIES:
            counts = np.array([
                len(sub[(sub["IGBP"] == b) & (sub["CATEGORY"] == cat)]) for b in biomes_present
            ])
            ax_bar.barh(y, counts, left=left, color=CATEGORY_COLORS[cat], edgecolor="white",
                       linewidth=0.6, height=0.68, label=cat)
            left += counts
        totals = left
        for yy, tot in zip(y, totals):
            if tot > 0:
                ax_bar.text(tot + max(totals.max(), 1) * 0.02, yy, f"{int(tot)}", va="center", fontsize=8, color="#333333")
        ax_bar.set_yticks(y)
        ax_bar.set_yticklabels(biomes_present, fontsize=9.3)
        ax_bar.invert_yaxis()
        ax_bar.set_xlim(0, max(totals.max(), 1) * 1.18)
        if row == len(PRODUCTS) - 1:
            ax_bar.set_xlabel("Number of towers", fontsize=9)
        ax_bar.set_title("Towers per biome, by timing category", fontsize=10.5, loc="left", color="#333333")
        ax_bar.spines[["top", "right"]].set_visible(False)
        ax_bar.tick_params(axis="x", labelsize=8)
        ax_bar.grid(axis="x", color="#eeeeee", lw=0.6, zorder=0)

    legend_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=CATEGORY_COLORS[c], edgecolor=CATEGORY_EDGECOLOR,
                                    linewidth=0.5, label=c) for c in CATEGORIES]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.005),
               fontsize=10.5, frameon=False, title="Timing category (map marker fill = bar color)", title_fontsize=9.5)

    size_legend_years = [1, 5, 10]
    size_handles = [plt.scatter([], [], s=18 + 9 * v, color="#888888", edgecolor=CATEGORY_EDGECOLOR, linewidth=0.5,
                                label=f"{v} yr") for v in size_legend_years]
    fig.legend(handles=size_handles, loc="lower left", bbox_to_anchor=(0.005, 0.005), fontsize=8,
               frameon=False, title="Map marker size: years of support", title_fontsize=8)

    fig.suptitle("Where does SIF peak earlier (or later) than GPP -- and does it depend on biome?",
                fontsize=18, fontweight="bold", y=0.998)
    fig.subplots_adjust(top=0.965, bottom=0.045, left=0.03, right=0.985)

    for ext in ("png", "svg", "pdf"):
        dpi = 600 if ext == "png" else 300
        fig.savefig(OUT_DIR / f"Figure3_Global_DeltaPOS_Map_WithBiomeCounts.{ext}", dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("Saved Figure3_Global_DeltaPOS_Map_WithBiomeCounts.{png,svg,pdf}")
    print(count_table.pivot_table(index=["SIF_PRODUCT", "IGBP"], columns="CATEGORY", values="N_TOWERS", fill_value=0))


if __name__ == "__main__":
    main()
