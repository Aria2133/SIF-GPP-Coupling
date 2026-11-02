"""
Part 1 -- new aridity / context geospatial figures for the SIF-GPP main-paper analysis
(4-product revision: GOSIF, GOME-2, TROPOSIF, OCO-3; OCO-2 excluded everywhere).

This script does NOT modify any existing production data, script, or figure. It only
reads existing CSVs/rasters and writes new files into:
  All_the_Visualizations_REVISED_4PRODUCT_FINAL_ARIDITY_JUNE_DIAGNOSTICS/08_geospatial_maps/

Mandatory basemap: <local path removed>
load_basemap()/add_basemap() below are copied verbatim (same logic, same defaults) from
All_the_Visualizations_REVISED_4PRODUCT_FINAL/06_scripts/build_new_geospatial_visualizations.py
per project instructions (reuse, don't rewrite).

Figures built here:
  1. Aridity_Global_Annual_With_Towers -- background = real Global Aridity Index v3.1
     annual raster (a genuine independent global gridded dataset, decimated read, NOT
     a tower-interpolated surface), tower markers colored by IGBP biome.
  2. Aridity_Class_Tower_Network -- real Earth-relief basemap, tower markers colored by
     the 5-class AI_CLASS (Hyper-Arid/Arid/Semi-Arid/Dry Sub-Humid/Humid) from the
     existing per-tower aridity extraction (tower_aridity_annual_and_monthly.csv).
  3. Temperature_Range_By_Tower -- real Earth-relief basemap, tower markers colored by
     continuous TAIR_range (deg C) from tower_temperature_range.csv.

Items 4 (June disagreement hotspot map) and 5 (low-coupling biome map) were checked and
SKIPPED: as of this run, 08_geospatial_maps/ has no JUNE_-prefixed file from a parallel
agent, and 05_aridity_and_biome_analysis/ has no biome-summary table yet either -- both
directories are empty. Building either from scratch here would require computing new
disagreement/low-coupling statistics not requested as part of this script's scope, so
they are left to whichever process owns 03_june_disagreement_analysis /
05_aridity_and_biome_analysis. Item 6 (extra biome-specific global maps) was also
skipped as lowest priority given time spent on the higher-priority items above and on
Part 2 (PC3 biome maps, separate script).

Author: project authors
Run date: 2026-09-06
"""
from __future__ import annotations
# Original script filename (sanitized): build_part1_aridity_context_maps.py; public name: fig02_aridity_context_map.py
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

import textwrap
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
import cartopy.crs as ccrs

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------

ROOT = Path(PROJECT_ROOT)

BASEMAP_TIF = Path(_os.environ.get("NE_BASEMAP_TIF", "NE1_HR_LC_SR_W.tif"))
AI_ANNUAL_TIF = ROOT / "Aridity Data'" / "extracted" / "Global-AI_ET0__annual_v3_1" / "ai_v31_yr.tif"

DIAG_ROOT = ROOT / "All_the_Visualizations_REVISED_4PRODUCT_FINAL_ARIDITY_JUNE_DIAGNOSTICS"
TOWER_ARIDITY_CSV = DIAG_ROOT / "01_inputs_inventory" / "tower_aridity_annual_and_monthly.csv"
TOWER_TEMPRANGE_CSV = DIAG_ROOT / "01_inputs_inventory" / "tower_temperature_range.csv"

FIGURE3_CSV = ROOT / "SIF_EC_SIF_GPP_Relationship_Analysis" / "05_figure_data" / "Figure3_Geospatial_Site_Level_Relationship.csv"
META_EXTRACT_CSV = ROOT / "Figure_1_Global_EC_Tower_BIF_Metadata_Extract.csv"
MASTER_ANALYSIS_READY_CSV = ROOT / "SIF_EC_SIF_GPP_Relationship_Analysis" / "01_analysis_ready" / "SIF_GPP_Analysis_Ready_Master.csv"

OUT_DIR = DIAG_ROOT / "08_geospatial_maps"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRODUCTS = ["GOSIF", "GOME-2", "TROPOSIF", "OCO-3"]  # main-paper products only

# IGBP color convention copied from Fig_G3_Biome_Context_Map's generating script
# (build_new_geospatial_visualizations.py) for cross-figure consistency.
IGBP_LABELS = {
    "CRO": "Cropland", "CSH": "Closed shrubland", "DBF": "Deciduous broadleaf forest",
    "EBF": "Evergreen broadleaf forest", "ENF": "Evergreen needleleaf forest", "GRA": "Grassland",
    "MF": "Mixed forest", "OSH": "Open shrubland", "SAV": "Savanna", "WSA": "Woody savanna",
    "WET": "Wetland", "DNF": "Deciduous needleleaf forest", "SNO": "Snow/ice", "Unknown": "Unknown / unassignable",
}
IGBP_COLORS = {
    "CRO": "#E6AB02", "CSH": "#A6761D", "DBF": "#66A61E", "EBF": "#1B9E77", "ENF": "#0B6E4F",
    "GRA": "#FFD92F", "MF": "#B2DF8A", "OSH": "#D95F02", "SAV": "#FC8D62", "WSA": "#FDBF6F",
    "WET": "#386CB0", "DNF": "#7570B3", "SNO": "#B3B3B3", "Unknown": "#999999",
}
IGBP_ORDER = ["ENF", "EBF", "DBF", "MF", "DNF", "CSH", "OSH", "WSA", "SAV", "GRA", "CRO", "WET", "SNO", "Unknown"]

AI_CLASS_ORDER = ["Hyper-Arid", "Arid", "Semi-Arid", "Dry Sub-Humid", "Humid"]
AI_CLASS_COLORS = {
    # dry (red/brown) -> wet (blue/green), colorblind-safe-ish sequential
    "Hyper-Arid": "#7F3B08", "Arid": "#E08214", "Semi-Arid": "#FEE0B6",
    "Dry Sub-Humid": "#8073AC", "Humid": "#1B7837",
}

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#333333", "axes.linewidth": 0.6})


# --------------------------------------------------------------------------------------
# Basemap (verbatim logic reused from build_new_geospatial_visualizations.py)
# --------------------------------------------------------------------------------------

def load_basemap(target_width_px: int = 6000, white_blend: float = 0.32):
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


def wrap_text(text: str, width: int = 150) -> str:
    lines = []
    for para in text.split("\n"):
        wrapped = textwrap.wrap(para, width=width) or [""]
        lines.extend(wrapped)
    return "\n".join(lines)


def save_figure(fig, stem: str):
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(OUT_DIR / f"{stem}.{ext}", dpi=600, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved: {stem}.png/.svg/.pdf")


# --------------------------------------------------------------------------------------
# Aridity Index raster (real independent global gridded dataset -- not tower-interpolated)
# --------------------------------------------------------------------------------------

def load_ai_annual_raster(target_width_px: int = 5000):
    """Decimated read of the Global Aridity Index v3.1 annual raster. Scale factor
    1/10000; nodata flags 0 and 65535 (per project data dictionary) are masked out
    BEFORE they can bias the averaged/downsampled cells."""
    with rasterio.open(AI_ANNUAL_TIF) as src:
        scale = target_width_px / src.width
        out_width = target_width_px
        out_height = max(1, round(src.height * scale))
        arr = src.read(1, out_shape=(out_height, out_width), resampling=Resampling.average)
        bounds = src.bounds
    arr = arr.astype(np.float64)
    # Cells that average out to (near-)exactly 0 or 65535 are blocks made up entirely
    # of nodata source pixels (ocean / flagged) -- mask them so they don't get colored.
    nodata_mask = (arr <= 0.5) | (arr >= 65534.5)
    ai = arr / 10000.0
    ai = np.where(nodata_mask, np.nan, ai)
    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    return ai, extent


# --------------------------------------------------------------------------------------
# Tower tables
# --------------------------------------------------------------------------------------

def build_igbp_lookup(site_ids) -> pd.Series:
    """Per-site IGBP: Figure3 table's mode IGBP, backfilled from the BIF metadata
    extract, then the analysis-ready master's IGBP_GPP_TABLE column. Unassigned
    sites are labeled 'Unknown' (not fabricated). Same precedence as
    build_new_geospatial_visualizations.py's build_igbp_lookup()."""
    fig3 = pd.read_csv(FIGURE3_CSV)
    fig3 = fig3[fig3["SIF_PRODUCT"].isin(PRODUCTS)]
    igbp_fig3 = (
        fig3.dropna(subset=["IGBP"]).groupby("SITE_ID")["IGBP"]
        .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan)
    )
    meta = pd.read_csv(META_EXTRACT_CSV)
    igbp_meta = meta.dropna(subset=["IGBP"]).drop_duplicates("SITE_ID").set_index("SITE_ID")["IGBP"]
    master = pd.read_csv(MASTER_ANALYSIS_READY_CSV, usecols=["SITE_ID", "IGBP_GPP_TABLE"], low_memory=False)
    igbp_master = master.dropna(subset=["IGBP_GPP_TABLE"]).drop_duplicates("SITE_ID").set_index("SITE_ID")["IGBP_GPP_TABLE"]

    out = igbp_fig3.reindex(site_ids)
    out = out.fillna(igbp_meta.reindex(site_ids))
    out = out.fillna(igbp_master.reindex(site_ids))
    out = out.replace({"": np.nan}).fillna("Unknown")
    return out


def build_tower_context_table() -> pd.DataFrame:
    """One row per unique tower (from the existing aridity extraction) with
    AI_ANNUAL, AI_CLASS, TAIR_range, and IGBP joined on -- no re-extraction."""
    aridity = pd.read_csv(TOWER_ARIDITY_CSV)
    temp = pd.read_csv(TOWER_TEMPRANGE_CSV)
    towers = aridity.merge(temp[["SITE_ID", "TAIR_range"]], on="SITE_ID", how="left")
    towers["IGBP"] = build_igbp_lookup(towers["SITE_ID"]).values
    return towers


# --------------------------------------------------------------------------------------
# Figure 1: Global annual aridity map with tower overlay
# --------------------------------------------------------------------------------------

def build_figure1_aridity_global(towers: pd.DataFrame):
    print("Building Figure 1: global annual aridity map with tower overlay ...")
    ai, ai_extent = load_ai_annual_raster(target_width_px=5000)

    fig = plt.figure(figsize=(13, 8.1))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.EqualEarth())
    ax.set_global()
    ax.spines["geo"].set_edgecolor("#888888")
    ax.spines["geo"].set_linewidth(0.5)

    # PowerNorm compresses the long wet tail (max AI ~6.5, but >95% of valid cells are
    # below ~2.3) so dry-to-subhumid gradations stay legible; values above vmax are
    # clipped only for *color assignment* via colorbar extend='max' -- raw data unaltered.
    vmax = 2.5
    norm = PowerNorm(gamma=0.55, vmin=0.0, vmax=vmax)
    im = ax.imshow(
        ai, origin="upper", extent=ai_extent, transform=ccrs.PlateCarree(),
        cmap="BrBG", norm=norm, interpolation="nearest", zorder=0,
    )

    proj = ccrs.PlateCarree()
    present_biomes = [b for b in IGBP_ORDER if (towers["IGBP"] == b).any()]
    for biome in present_biomes:
        sub = towers[towers["IGBP"] == biome]
        if biome == "Unknown":
            ax.scatter(
                sub["LONGITUDE"], sub["LATITUDE"], transform=proj,
                color=IGBP_COLORS[biome], marker="x", linewidth=1.6, s=75, alpha=0.95, zorder=6,
                label=f"{biome} - {IGBP_LABELS[biome]} (n={len(sub)})",
            )
        else:
            ax.scatter(
                sub["LONGITUDE"], sub["LATITUDE"], transform=proj,
                facecolor=IGBP_COLORS[biome], edgecolor="black", linewidth=0.55,
                marker="o", s=60, alpha=0.95, zorder=6,
                label=f"{biome} - {IGBP_LABELS[biome]} (n={len(sub)})",
            )

    # Band layout (figure-fraction y, bottom to top), generous gaps so nothing overlaps:
    # footer caption (0.015-0.06), colorbar bar (0.12-0.145), ticks+label ABOVE the bar
    # (toward the map, ~0.145-0.20) via ticks/label position "top" -- same trick used for
    # Fig_G4_Biome_Anomaly_Reliability_Map in this project's companion script, which keeps
    # the colorbar block and the footer caption from ever colliding.
    cbar_ax = fig.add_axes([0.30, 0.12, 0.40, 0.022])
    ticks = [0.03, 0.2, 0.5, 0.65, 1.0, 1.5, 2.0, 2.5]
    cb = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", extend="max", ticks=ticks)
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.xaxis.set_label_position("top")
    cb.ax.set_xticklabels([f"{t:g}" for t in ticks], fontsize=7.5)
    cb.set_label(
        "Global Aridity Index v3.1, annual (AI = P / ET0); dry \u2192 wet. Class edges marked: "
        "0.03 hyper-arid | 0.2 arid | 0.5 semi-arid | 0.65 dry sub-humid | humid above",
        fontsize=8.2,
    )

    leg = ax.legend(
        loc="lower left", fontsize=7.6, title="Tower IGBP biome (marker color/shape)",
        title_fontsize=8.2, frameon=True, facecolor="white", framealpha=0.9, ncol=1,
        bbox_to_anchor=(0.0, 0.0),
    )
    leg.get_frame().set_edgecolor("#999999")

    ax.set_title(
        f"Global annual aridity index (real gridded dataset) with eddy-covariance tower network overlay "
        f"({len(towers)} towers)",
        fontsize=12.3, fontweight="bold", loc="left", pad=10,
    )
    fig.text(
        0.5, 0.055,
        wrap_text(
            "Background: Global Aridity Index v3.1 annual raster (ai_v31_yr.tif, EPSG:4326, decimated ~5000px-wide read, "
            "Resampling.average; nodata flags 0/65535 masked) -- a genuine independent global gridded dataset, NOT a surface "
            "interpolated from these towers. Tower AI/biome values: tower_aridity_annual_and_monthly.csv and IGBP join "
            "(Figure3_Geospatial_Site_Level_Relationship.csv, backfilled from BIF metadata / analysis-ready master).",
            width=150,
        ),
        ha="center", va="top", fontsize=6.4, color="#333333",
    )
    fig.subplots_adjust(top=0.92, bottom=0.24)
    save_figure(fig, "Aridity_Global_Annual_With_Towers")


# --------------------------------------------------------------------------------------
# Figure 2: Aridity class by tower network (real Earth-relief basemap, not the AI raster)
# --------------------------------------------------------------------------------------

def build_figure2_aridity_class(towers: pd.DataFrame, basemap_rgb, basemap_extent):
    print("Building Figure 2: aridity class by tower network ...")
    fig = plt.figure(figsize=(13, 7.2))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.EqualEarth())
    add_basemap(ax, basemap_rgb, basemap_extent)

    proj = ccrs.PlateCarree()
    for cls in AI_CLASS_ORDER:
        sub = towers[towers["AI_CLASS"] == cls]
        if not len(sub):
            continue
        ax.scatter(
            sub["LONGITUDE"], sub["LATITUDE"], transform=proj,
            facecolor=AI_CLASS_COLORS[cls], edgecolor="black", linewidth=0.6,
            marker="o", s=95, alpha=0.95, zorder=6,
            label=f"{cls} (n={len(sub)})",
        )

    ax.set_title(
        f"Aridity class of each eddy-covariance tower ({len(towers)} towers; Global Aridity Index v3.1, annual)",
        fontsize=12.5, fontweight="bold", loc="left", pad=10,
    )
    leg = ax.legend(
        loc="lower left", fontsize=9, title="AI class (UNEP-style aridity classification)",
        title_fontsize=9.5, frameon=True, facecolor="white", framealpha=0.9,
        bbox_to_anchor=(0.0, 0.0),
    )
    leg.get_frame().set_edgecolor("#999999")

    class_counts = towers["AI_CLASS"].value_counts().reindex(AI_CLASS_ORDER).fillna(0).astype(int)
    summary = " | ".join(f"{c}: {n}" for c, n in class_counts.items())
    fig.text(
        0.5, 0.045,
        wrap_text(f"Tower counts by class -- {summary}", width=160),
        ha="center", fontsize=7.5, color="#333333",
    )
    fig.text(
        0.5, 0.015,
        wrap_text(
            "Basemap: NE1_HR_LC_SR_W.tif (Natural Earth I shaded relief w/ water), decimated read, blended 32% toward white. "
            "AI_CLASS: tower_aridity_annual_and_monthly.csv (per-tower extraction from Global Aridity Index v3.1, already computed; not re-extracted here).",
            width=170,
        ),
        ha="center", fontsize=6.4, color="#555555",
    )
    fig.subplots_adjust(top=0.90, bottom=0.13)
    save_figure(fig, "Aridity_Class_Tower_Network")


# --------------------------------------------------------------------------------------
# Figure 3: Temperature range by tower
# --------------------------------------------------------------------------------------

def build_figure3_temperature_range(towers: pd.DataFrame, basemap_rgb, basemap_extent):
    print("Building Figure 3: temperature range by tower ...")
    sub = towers.dropna(subset=["TAIR_range"]).copy()

    fig = plt.figure(figsize=(13, 8.0))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.EqualEarth())
    add_basemap(ax, basemap_rgb, basemap_extent)

    proj = ccrs.PlateCarree()
    vmin, vmax = float(sub["TAIR_range"].min()), float(sub["TAIR_range"].max())
    sc = ax.scatter(
        sub["LONGITUDE"], sub["LATITUDE"], transform=proj,
        c=sub["TAIR_range"], cmap="inferno_r", vmin=vmin, vmax=vmax,
        marker="o", s=95, edgecolor="black", linewidth=0.5, alpha=0.95, zorder=6,
    )

    n_missing = int(towers["TAIR_range"].isna().sum())
    ax.set_title(
        f"Air temperature range (max \u2212 min monthly mean) by tower ({len(sub)} of {len(towers)} towers with data)",
        fontsize=12.5, fontweight="bold", loc="left", pad=10,
    )
    # Same band layout as Figure 1: colorbar bar low, ticks+label pushed ABOVE the bar
    # (toward the map) so they never collide with the footer caption underneath.
    cbar_ax = fig.add_axes([0.30, 0.12, 0.40, 0.022])
    cb = fig.colorbar(sc, cax=cbar_ax, orientation="horizontal")
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.xaxis.set_label_position("top")
    cb.set_label("TAIR_range = max monthly mean air temperature \u2212 min monthly mean air temperature (\u00b0C)", fontsize=8.5)

    if n_missing:
        ax.text(
            0.99, 0.02, f"{n_missing} tower(s) with no TAIR_range value not shown.",
            transform=ax.transAxes, fontsize=7.3, color="#333333", ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#999999", linewidth=0.5),
        )
    fig.text(
        0.5, 0.055,
        wrap_text(
            "Basemap: NE1_HR_LC_SR_W.tif, decimated read, blended 32% toward white. TAIR_range: tower_temperature_range.csv "
            "(already computed per-tower extraction; not re-derived here). Tower points only, no interpolated surface.",
            width=150,
        ),
        ha="center", va="top", fontsize=6.4, color="#555555",
    )
    fig.subplots_adjust(top=0.92, bottom=0.24)
    save_figure(fig, "Temperature_Range_By_Tower")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    print("Loading tower context table (aridity + temp-range + IGBP join) ...")
    towers = build_tower_context_table()
    print(f"  {len(towers)} towers; AI_CLASS counts:\n{towers['AI_CLASS'].value_counts()}")
    print(f"  IGBP counts:\n{towers['IGBP'].value_counts()}")

    print("Loading Earth-relief basemap (decimated read of NE1_HR_LC_SR_W.tif) ...")
    basemap_rgb, basemap_extent = load_basemap(target_width_px=6000, white_blend=0.32)

    build_figure1_aridity_global(towers)
    build_figure2_aridity_class(towers, basemap_rgb, basemap_extent)
    build_figure3_temperature_range(towers, basemap_rgb, basemap_extent)

    print("\nSkipped (per task priority / availability check at run time):")
    print("  - JUNE disagreement hotspot map: no JUNE_*-prefixed file present in 08_geospatial_maps/, "
          "and no disagreement-statistics table was in this script's scope to compute from scratch.")
    print("  - Low-coupling biome map: 05_aridity_and_biome_analysis/ has no biome-summary table yet.")
    print("  - Extra biome-specific global maps (item 6): lowest priority, skipped for this run.")
    print("\nAll figures written to:", OUT_DIR)


if __name__ == "__main__":
    main()
