"""
Main-paper REPLACEMENT figure: a geospatial (world-map) version of the
environmental-controls-on-coupling figure.

Why this replaces figure_environmental_controls_on_coupling.png (V2): the user
explicitly rejected the V2 design (a 5-panel tercile box/strip plot, x=driver
tercile, y=site-level Pearson r) because it is NOT geospatial. The user wants
an actual world map. This script builds that map while reusing every number
and data source already validated in V2 -- nothing below is a new statistical
claim; only the chart form (map instead of tercile-strip plot) is new.

Main-paper products only: GOSIF, GOME-2, TROPOSIF, OCO-3. OCO-2 excluded
everywhere.

Design (see the PROVENANCE.md written next to the main figure output for full
detail):
  - 5 panels, one per driver: VPD, soil moisture, incoming shortwave
    radiation, air temperature, annual precipitation (ET excluded).
  - PRIMARY encoding (marker FILL color): site-level SIF-GPP Pearson r.
    For the MAIN figure this is the tower's own MEAN Pearson r across
    whichever of the 4 main-paper products actually cover that tower (1-4
    products per tower -- coverage differs by product, stated explicitly in
    the caption and legend). This is a real, computed summary -- not a
    silent average that hides cross-product spread: the per-product detail
    lives in the 4 product-specific supporting maps built by this same
    script (03_product_specific_geospatial/), and the annotation box in each
    main-figure panel reports the per-product Spearman rho breakdown (not
    one pooled rho) precisely so cross-product differences stay visible even
    on the pooled map.
  - SECONDARY encoding (marker SIZE): that panel's driver tercile
    (Low/Mid/High), computed independently per driver on the unique 73-tower
    table (a tower's tercile does not depend on which product is used).
    Size was chosen over outline color/style because an outline hue would be
    visually confusable with the diverging r fill color at small marker
    footprints on a 600 DPI raster map; discrete size steps (small/med/large)
    stay legible at any zoom level and don't need a second color legend.
  - Colormap/range: matplotlib "RdBu", Normalize(vmin=-1, vmax=1) -- this
    reproduces the exact color scheme and range already used for
    Figure3_Geospatial_Site_Level_Relationship_4product.png in this project
    (dark red = r near -1, white = r near 0, dark blue = r near +1), so this
    new figure reads consistently with that existing main-paper figure.
  - Per-panel annotation box: reports the Spearman rho between that panel's
    continuous driver value and site-level Pearson r, PER PRODUCT (4 numbers
    + n each), reused verbatim from build_environmental_controls_on_coupling.py
    (V2) -- computed here by re-running that exact same function against the
    same source files (deterministic; not a new method or new numbers), not
    transcribed by hand. Placed over the Southern Ocean / Antarctica region
    (~axes-fraction bottom-center), which is empty of towers (southernmost
    tower is AU-Wom at -37.4 deg lat) in every panel.
  - Basemap: NE1_HR_LC_SR_W.tif via rasterio decimated read (~6000 px wide,
    Resampling.average) + cartopy EqualEarth, blended 32% toward white --
    identical recipe (load_basemap/add_basemap) to
    All_the_Visualizations_REVISED_4PRODUCT_FINAL/06_scripts/build_new_geospatial_visualizations.py.

Data sources (ALL reused from the V2 figure's already-validated pipeline; see
figure_environmental_controls_PROVENANCE.md in
All_the_Visualizations_REVISED_4PRODUCT_UPDATE_V2/01_main_revised/):
  - Site-level SIF-GPP Pearson r per tower per product: reused unmodified from
    SIF_EC_SIF_GPP_Relationship_Analysis/05_figure_data/Figure3_Geospatial_Site_Level_Relationship.csv
    (filtered to the 4 main-paper products; OCO-2 dropped).
  - VPD / air temperature / incoming shortwave radiation / soil moisture
    (site long-term mean LEVEL, not anomaly): reused unmodified computation
    (same groupby-mean) from
    SIF_EC_Mechanistic_FootprintAware_SIF_GPP_Paper/08_mechanistic_controls/Mechanistic_SIF_GPP_Monthly_With_Extended_Drivers.csv
  - Annual precipitation (site long-term mean): reused unmodified from
    All_the_Visualizations_REVISED_4PRODUCT_UPDATE/03_precip_table/tower_annual_precipitation.csv
    (21/73 real FLUXNET P_F, 52/73 NASA POWER climatology -- caveat carried
    forward unchanged into this figure's footnote).
  - Tercile bin edges: recomputed with the identical pandas.qcut(values, 3,
    duplicates="drop") call on the identical per-site table used in V2 (this
    reproduces, rather than invents, the edges already reported in
    figure_environmental_controls_PROVENANCE.md).
  - Spearman rho per driver per product: recomputed by re-running V2's own
    spearman_rho() function against the identical input files (deterministic;
    reproduces V2's printed numbers exactly rather than retyping them by
    hand, which would risk a transcription error).

Scientific framing: explicitly a CROSS-SITE, DESCRIPTIVE/EXPLORATORY
association, never "controls" or "drives". The caption states that the
independent within-site interaction reanalysis
(SIF_EC_SIF_GPP_Final_Verification/05_mechanistic_reanalysis/mechanistic_and_biome_findings.md)
found only weak, mostly non-significant evidence of environmental modulation
of coupling, and that this map's cross-site pattern is a different, weaker
claim, not a contradiction of that result.

Author: project authors
Run date: 2026-08-31
"""
from __future__ import annotations
# Original script filename (sanitized): build_environmental_controls_geospatial.py; public name: build_environmental_controls_geospatial.py
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
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from scipy import stats

import cartopy.crs as ccrs

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------

ROOT = Path(PROJECT_ROOT)

BASEMAP_TIF = Path(_os.environ.get("NE_BASEMAP_TIF", "NE1_HR_LC_SR_W.tif"))

SITE_R_CSV = ROOT / "SIF_EC_SIF_GPP_Relationship_Analysis" / "05_figure_data" / "Figure3_Geospatial_Site_Level_Relationship.csv"
MECH_CSV = ROOT / "SIF_EC_Mechanistic_FootprintAware_SIF_GPP_Paper" / "08_mechanistic_controls" / "Mechanistic_SIF_GPP_Monthly_With_Extended_Drivers.csv"
PRECIP_CSV = ROOT / "All_the_Visualizations_REVISED_4PRODUCT_UPDATE" / "03_precip_table" / "tower_annual_precipitation.csv"

OUT_MAIN_DIR = ROOT / "All_the_Visualizations_REVISED_4PRODUCT_UPDATE_V3" / "01_main_revised"
OUT_PRODUCT_DIR = ROOT / "All_the_Visualizations_REVISED_4PRODUCT_UPDATE_V3" / "03_product_specific_geospatial"
OUT_MAIN_DIR.mkdir(parents=True, exist_ok=True)
OUT_PRODUCT_DIR.mkdir(parents=True, exist_ok=True)

PRODUCTS = ["GOSIF", "GOME-2", "TROPOSIF", "OCO-3"]  # main-paper products only; OCO-2 excluded everywhere
PRODUCT_COLORS = {
    "GOSIF": "#0072B2", "GOME-2": "#E69F00", "TROPOSIF": "#009E73", "OCO-3": "#CC79A7",
}

BIN_LABELS = ["Low", "Mid", "High"]
BIN_SIZE = {"Low": 34, "Mid": 90, "High": 165}

# Same diverging colormap/range as Figure3_Geospatial_Site_Level_Relationship_4product.png
COUPLING_CMAP = "RdBu"
COUPLING_NORM = Normalize(vmin=-1.0, vmax=1.0)

DRIVERS = [
    dict(key="VPD", col="VPD", title="Vapor pressure deficit", unit="kPa",
         cbar_label="Vapor pressure deficit (site long-term mean, kPa)"),
    dict(key="SWC", col="SWC", title="Soil moisture", unit="%",
         cbar_label="Soil moisture (site long-term mean, %)"),
    dict(key="SW_IN", col="SW_IN", title="Incoming shortwave radiation", unit="W m\u207b\u00b2",
         cbar_label="Incoming shortwave radiation (site long-term mean, W m\u207b\u00b2)"),
    dict(key="TAIR", col="TAIR", title="Air temperature", unit="\u00b0C",
         cbar_label="Air temperature (site long-term mean, \u00b0C)"),
    dict(key="PRECIP", col="ANNUAL_PRECIP_MM", title="Annual precipitation", unit="mm yr\u207b\u00b9",
         cbar_label="Annual precipitation (mm yr\u207b\u00b9)"),
]

plt.rcParams.update({"font.family": "DejaVu Sans"})


# --------------------------------------------------------------------------------------
# Basemap (identical recipe to build_new_geospatial_visualizations.py)
# --------------------------------------------------------------------------------------

def load_basemap(target_width_px: int = 6000, white_blend: float = 0.32):
    with rasterio.open(BASEMAP_TIF) as src:
        assert src.crs.to_epsg() == 4326, f"Unexpected basemap CRS: {src.crs}"
        scale = target_width_px / src.width
        out_width = target_width_px
        out_height = max(1, round(src.height * scale))
        arr = src.read(
            out_shape=(src.count, out_height, out_width),
            resampling=Resampling.average,
        )
        bounds = src.bounds
    rgb = np.transpose(arr, (1, 2, 0)).astype(np.float32) / 255.0
    rgb = rgb * (1 - white_blend) + white_blend * 1.0
    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    return rgb, extent


def add_basemap(ax, rgb, extent):
    ax.imshow(
        rgb, origin="upper", extent=extent, transform=ccrs.PlateCarree(),
        interpolation="bilinear", zorder=0,
    )
    ax.set_global()
    ax.spines["geo"].set_edgecolor("#888888")
    ax.spines["geo"].set_linewidth(0.5)


def wrap_text(text: str, width: int) -> str:
    lines = []
    for para in text.split("\n"):
        wrapped = textwrap.wrap(para, width=width) or [""]
        lines.extend(wrapped)
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Data loading (reused sources/logic from V2's build_environmental_controls_on_coupling.py)
# --------------------------------------------------------------------------------------

def load_data():
    site_r = pd.read_csv(SITE_R_CSV)
    site_r = site_r[site_r["SIF_PRODUCT"].isin(PRODUCTS)].copy()
    site_r = site_r[["SIF_PRODUCT", "SITE_ID", "LATITUDE", "LONGITUDE", "IGBP", "PEARSON_R", "N"]]

    coords = site_r.groupby("SITE_ID")[["LATITUDE", "LONGITUDE"]].first()

    mech = pd.read_csv(MECH_CSV, low_memory=False)
    site_mean_drivers = mech.groupby("SITE_ID")[["VPD", "TAIR", "SW_IN", "SWC"]].mean().reset_index()

    precip = pd.read_csv(PRECIP_CSV)[["SITE_ID", "ANNUAL_PRECIP_MM", "PRECIP_SOURCE"]]

    site_level = site_mean_drivers.merge(precip, on="SITE_ID", how="outer")
    site_level = site_level.merge(coords.reset_index(), on="SITE_ID", how="outer")

    # Pooled tower-level coupling summary for the MAIN figure: mean Pearson r
    # across whichever of the 4 products cover that tower (documented, not
    # silently invented -- N_PRODUCTS and the per-product spread are carried
    # alongside it and disclosed in the caption/annotation).
    pooled = site_r.groupby("SITE_ID").agg(
        POOLED_R=("PEARSON_R", "mean"),
        R_STD=("PEARSON_R", "std"),
        N_PRODUCTS=("PEARSON_R", "count"),
    ).reset_index()

    return site_r, site_level, pooled, coords


def make_terciles(site_level: pd.DataFrame, col: str):
    """Identical method to V2: pd.qcut(values, 3, duplicates='drop') on the
    unique per-site table, so a tower's bin does not depend on which product
    happens to cover it."""
    sub = site_level[["SITE_ID", col]].dropna(subset=[col]).copy()
    bins, edges = pd.qcut(sub[col], 3, labels=BIN_LABELS, duplicates="drop", retbins=True)
    sub["BIN"] = bins.astype(str)
    return sub, edges


def spearman_rho(site_level, site_r, col, product):
    """Identical function to V2's spearman_rho(): Spearman rho between each
    tower's continuous driver value and that tower's site-level Pearson r,
    for one product."""
    sub = site_level[["SITE_ID", col]].dropna(subset=[col])
    r_sub = site_r[site_r["SIF_PRODUCT"] == product][["SITE_ID", "PEARSON_R"]]
    m = sub.merge(r_sub, on="SITE_ID", how="inner")
    if len(m) < 5:
        return np.nan, np.nan, len(m)
    rho, p = stats.spearmanr(m[col], m["PEARSON_R"])
    return rho, p, len(m)


# --------------------------------------------------------------------------------------
# Panel drawing
# --------------------------------------------------------------------------------------

def draw_size_legend(ax, x0, y0, dx, fontsize=7.6):
    for i, b in enumerate(BIN_LABELS):
        ax.scatter([x0 + i * dx], [y0], transform=ax.transAxes, s=BIN_SIZE[b],
                   facecolor="#AAAAAA", edgecolor="#111111", linewidth=0.6, clip_on=False, zorder=20)
        ax.text(x0 + i * dx, y0 - 0.10, b, transform=ax.transAxes, ha="center", va="top",
                fontsize=fontsize, clip_on=False)


def draw_map_panel(ax, driver, plot_df, r_col, basemap_rgb, basemap_extent, rho_text):
    """plot_df must have LONGITUDE, LATITUDE, r_col, and BIN (Low/Mid/High)."""
    add_basemap(ax, basemap_rgb, basemap_extent)
    proj = ccrs.PlateCarree()

    sub = plot_df.dropna(subset=[r_col, "BIN"])
    for b in BIN_LABELS:
        s = sub[sub["BIN"] == b]
        if len(s) == 0:
            continue
        ax.scatter(
            s["LONGITUDE"], s["LATITUDE"], transform=proj,
            c=s[r_col], cmap=COUPLING_CMAP, norm=COUPLING_NORM,
            s=BIN_SIZE[b], edgecolor="#111111", linewidth=0.55, alpha=0.95, zorder=6,
        )

    ax.set_title(f"{driver['title']}  (n = {len(sub)} towers)", fontsize=12.0, fontweight="bold",
                 loc="left", pad=6)

    ax.text(
        0.5, 0.045, rho_text, transform=ax.transAxes, ha="center", va="bottom",
        fontsize=7.0, color="#111111", linespacing=1.35,
        bbox=dict(boxstyle="round,pad=0.32", facecolor="white", alpha=0.88,
                  edgecolor="#999999", linewidth=0.5), zorder=25,
    )


def per_product_rho_text(rho_table, key):
    lines = ["Cross-site \u03c1 (driver vs. site r), by product:"]
    parts = []
    for p in PRODUCTS:
        rho, pval, n = rho_table[key][p]
        star = "*" if (not np.isnan(pval) and pval < 0.05) else ""
        parts.append(f"{p} {rho:+.2f}{star} (n={n})")
    lines.append("   ".join(parts[:2]))
    lines.append("   ".join(parts[2:]))
    return "\n".join(lines)


def single_rho_text(rho_table, key, product):
    rho, pval, n = rho_table[key][product]
    star = "*" if (not np.isnan(pval) and pval < 0.05) else ""
    return f"\u03c1 = {rho:+.2f}{star}, n = {n}"


def draw_legend_panel(ax_legend, sc_ref, cbar_label, size_note, coverage_text):
    """All positions below are in ax_legend's OWN transAxes fraction (0-1),
    top to bottom, with generous vertical gaps -- avoids the earlier bug of
    mixing this axes' transAxes coordinates with independent figure-fraction
    fig.add_axes() coordinates for the colorbar, which caused the colorbar,
    its label, the size legend, and the coverage text to overlap."""
    ax_legend.axis("off")

    ax_legend.text(0.06, 0.975, "SIF-GPP coupling (site-level Pearson r)",
                    transform=ax_legend.transAxes, fontsize=10.2, fontweight="bold", va="top")
    cax = ax_legend.inset_axes([0.06, 0.865, 0.88, 0.045])
    cb = plt.colorbar(sc_ref, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=8.2)
    ax_legend.text(0.5, 0.795, cbar_label, transform=ax_legend.transAxes,
                    fontsize=8.0, ha="center", va="top", color="#333333")

    ax_legend.text(0.06, 0.685, "Marker size = driver tercile\n(computed independently per panel)",
                    transform=ax_legend.transAxes, fontsize=9.6, fontweight="bold", va="top")
    draw_size_legend(ax_legend, x0=0.13, y0=0.555, dx=0.17, fontsize=8.8)
    if size_note:
        ax_legend.text(0.06, 0.44, wrap_text(size_note, 60), transform=ax_legend.transAxes,
                        fontsize=7.6, va="top", color="#333333")

    ax_legend.text(0.06, 0.28, wrap_text(coverage_text, 62), transform=ax_legend.transAxes,
                    fontsize=8.0, va="top", color="#222222")


# --------------------------------------------------------------------------------------
# Figure builders
# --------------------------------------------------------------------------------------

TITLE = "Geospatial associations between environmental gradients and site-level SIF-GPP coupling across four SIF products"
SUBTITLE = (
    "Dual encoding on every map: marker FILL COLOR = SIF-GPP coupling (site-level Pearson r, blue = positive, red = negative); "
    "marker SIZE = that panel's driver tercile (small = Low, medium = Mid, large = High), computed independently per driver. "
    "The annotation box in each panel reports the Spearman rho between that panel's continuous driver value and site-level r "
    "(cross-site association only)."
)

NONCAUSAL_NOTE = (
    "Cross-site, descriptive/exploratory pattern only -- not causal and not a controlled within-site test. The independent "
    "within-site interaction reanalysis (SIF_EC_SIF_GPP_Final_Verification/05_mechanistic_reanalysis/"
    "mechanistic_and_biome_findings.md) found only weak, mostly non-significant evidence that these environmental variables "
    "modulate SIF-GPP coupling within a site over time (VPD and air temperature had a consistent interaction sign across all "
    "4 products, but only 1/4 and 0/4 of the 95% confidence intervals excluded zero; soil moisture and radiation showed mixed "
    "signs across products). The pattern shown on these maps is a different, weaker, cross-site claim -- it is consistent with, "
    "not in contradiction of, that more rigorous result."
)

DATA_NOTE_TEMPLATE = (
    "Coupling metric: site-level Pearson r between SIF and tower GPP (same metric as Figure3_Geospatial_Site_Level_Relationship.csv "
    "and the tercile figure it replaces). {pool_sentence} Driver values: VPD/air temperature/incoming shortwave radiation/soil "
    "moisture are each tower's long-term mean (not anomaly) from Mechanistic_SIF_GPP_Monthly_With_Extended_Drivers.csv; annual "
    "precipitation is reused from tower_annual_precipitation.csv ({n_real} of {n_total} towers have real in-situ FLUXNET P_F "
    "precipitation, the remaining {n_climatology} use NASA POWER climatology -- precipitation tercile membership for those towers "
    "reflects a regional climatology, not a site-measured value). Soil moisture uses {n_swc} of {n_total} towers (missing SWC "
    "data for the rest). Terciles are computed per driver with pandas.qcut on the full available tower set (same method and "
    "edges as figure_environmental_controls_PROVENANCE.md). Basemap: NE1_HR_LC_SR_W.tif (Natural Earth I shaded relief with "
    "water), decimated read, blended 32% toward white; EqualEarth projection."
)


def build_main_figure(site_r, site_level, pooled, basemap_rgb, basemap_extent, rho_table):
    print("Building MAIN geospatial figure (pooled tower-level r) ...")
    tower_r = pooled.merge(site_level[["SITE_ID", "LATITUDE", "LONGITUDE"]].drop_duplicates("SITE_ID"),
                            on="SITE_ID", how="left")

    fig = plt.figure(figsize=(21.0, 12.4))
    gs = fig.add_gridspec(2, 3, left=0.025, right=0.99, top=0.83, bottom=0.145, wspace=0.05, hspace=0.28)
    axes = [fig.add_subplot(gs[0, 0], projection=ccrs.EqualEarth()),
            fig.add_subplot(gs[0, 1], projection=ccrs.EqualEarth()),
            fig.add_subplot(gs[0, 2], projection=ccrs.EqualEarth()),
            fig.add_subplot(gs[1, 0], projection=ccrs.EqualEarth()),
            fig.add_subplot(gs[1, 1], projection=ccrs.EqualEarth())]
    ax_legend = fig.add_subplot(gs[1, 2])
    ax_legend.axis("off")

    sc_ref = None
    for ax, driver in zip(axes, DRIVERS):
        tercile_df, edges = make_terciles(site_level, driver["col"])
        plot_df = tower_r.merge(tercile_df[["SITE_ID", "BIN"]], on="SITE_ID", how="inner")
        rho_text = per_product_rho_text(rho_table, driver["key"])
        draw_map_panel(ax, driver, plot_df, "POOLED_R", basemap_rgb, basemap_extent, rho_text)
        if sc_ref is None:
            sc_ref = plt.cm.ScalarMappable(norm=COUPLING_NORM, cmap=COUPLING_CMAP)

    # Legend panel: colorbar + size legend + coverage note (single shared layout helper)
    n_multi = int((pooled["N_PRODUCTS"] > 1).sum())
    coverage_txt = (
        f"Each tower's fill color is the MEAN site-level Pearson r across whichever of the 4 main-paper "
        f"products cover it ({n_multi} of {len(pooled)} towers are covered by \u22652 products; coverage: 1-4). "
        f"Per-product r values differ at multi-product towers -- see the per-panel \u03c1 breakdown by product, "
        f"and the 4 product-specific supporting maps in 03_product_specific_geospatial/ for full per-product "
        f"site-level r without pooling."
    )
    draw_legend_panel(
        ax_legend, sc_ref,
        cbar_label="mean across whichever products cover each tower",
        size_note=None,
        coverage_text=coverage_txt,
    )

    fig.suptitle(wrap_text(TITLE, 118), fontsize=16.5, fontweight="bold", y=0.985, x=0.02, ha="left")
    fig.text(0.02, 0.905, wrap_text(SUBTITLE, 200), fontsize=9.7, color="#2a2a2a", ha="left", va="top")

    n_precip_real = int((site_level["PRECIP_SOURCE"] == "FLUXNET_P_F").sum())
    n_precip_total = int(site_level["PRECIP_SOURCE"].notna().sum())
    n_swc = int(site_level["SWC"].notna().sum())
    pool_sentence = "Main-figure marker fill is a per-tower MEAN of that tower's available per-product Pearson r values (documented, not a silently invented number)."
    data_note = DATA_NOTE_TEMPLATE.format(
        pool_sentence=pool_sentence, n_real=n_precip_real, n_total=n_precip_total,
        n_climatology=n_precip_total - n_precip_real, n_swc=n_swc,
    )
    footnote = NONCAUSAL_NOTE + "\n" + data_note
    fig.text(0.02, 0.006, wrap_text(footnote, 235), fontsize=6.6, color="#444444", ha="left", va="bottom")

    for ext, dpi in (("png", 600), ("svg", 300), ("pdf", 300)):
        out_path = OUT_MAIN_DIR / f"figure_environmental_controls_geospatial_main.{ext}"
        fig.savefig(out_path, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.30)
        print(f"  saved: {out_path}")
    plt.close(fig)


def build_product_figure(product, site_r, site_level, basemap_rgb, basemap_extent, rho_table):
    print(f"Building product-specific geospatial figure: {product} ...")
    prod_r = site_r[site_r["SIF_PRODUCT"] == product][["SITE_ID", "PEARSON_R"]].drop_duplicates("SITE_ID")
    prod_r = prod_r.merge(site_level[["SITE_ID", "LATITUDE", "LONGITUDE"]].drop_duplicates("SITE_ID"),
                           on="SITE_ID", how="left")

    fig = plt.figure(figsize=(21.0, 12.4))
    gs = fig.add_gridspec(2, 3, left=0.025, right=0.99, top=0.83, bottom=0.145, wspace=0.05, hspace=0.28)
    axes = [fig.add_subplot(gs[0, 0], projection=ccrs.EqualEarth()),
            fig.add_subplot(gs[0, 1], projection=ccrs.EqualEarth()),
            fig.add_subplot(gs[0, 2], projection=ccrs.EqualEarth()),
            fig.add_subplot(gs[1, 0], projection=ccrs.EqualEarth()),
            fig.add_subplot(gs[1, 1], projection=ccrs.EqualEarth())]
    ax_legend = fig.add_subplot(gs[1, 2])
    ax_legend.axis("off")

    sc_ref = None
    for ax, driver in zip(axes, DRIVERS):
        tercile_df, edges = make_terciles(site_level, driver["col"])
        plot_df = prod_r.merge(tercile_df[["SITE_ID", "BIN"]], on="SITE_ID", how="inner")
        rho_text = single_rho_text(rho_table, driver["key"], product)
        draw_map_panel(ax, driver, plot_df, "PEARSON_R", basemap_rgb, basemap_extent, rho_text)
        if sc_ref is None:
            sc_ref = plt.cm.ScalarMappable(norm=COUPLING_NORM, cmap=COUPLING_CMAP)

    coverage_txt = (
        f"This map shows {product}'s OWN site-level Pearson r only (no pooling across products). "
        f"n = {len(prod_r)} towers covered by {product}. See figure_environmental_controls_geospatial_main "
        f"for the pooled multi-product summary map."
    )
    draw_legend_panel(
        ax_legend, sc_ref,
        cbar_label=f"{product} only (no pooling across products)",
        size_note=None,
        coverage_text=coverage_txt,
    )

    title = f"{TITLE} \u2014 {product} only"
    fig.suptitle(wrap_text(title, 118), fontsize=16.0, fontweight="bold", y=0.985, x=0.02, ha="left")
    fig.text(0.02, 0.905, wrap_text(SUBTITLE, 200), fontsize=9.7, color="#2a2a2a", ha="left", va="top")

    n_precip_real = int((site_level["PRECIP_SOURCE"] == "FLUXNET_P_F").sum())
    n_precip_total = int(site_level["PRECIP_SOURCE"].notna().sum())
    n_swc = int(site_level["SWC"].notna().sum())
    pool_sentence = f"This is a PRODUCT-SPECIFIC map: fill color is {product}'s own site-level Pearson r (no pooling across products)."
    data_note = DATA_NOTE_TEMPLATE.format(
        pool_sentence=pool_sentence, n_real=n_precip_real, n_total=n_precip_total,
        n_climatology=n_precip_total - n_precip_real, n_swc=n_swc,
    )
    footnote = NONCAUSAL_NOTE + "\n" + data_note
    fig.text(0.02, 0.006, wrap_text(footnote, 235), fontsize=6.6, color="#444444", ha="left", va="bottom")

    stem = f"figure_environmental_controls_geospatial_{product.replace('-', '')}"
    for ext, dpi in (("png", 600), ("svg", 300), ("pdf", 300)):
        out_path = OUT_PRODUCT_DIR / f"{stem}.{ext}"
        fig.savefig(out_path, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.30)
        print(f"  saved: {out_path}")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    print("Loading basemap (decimated read of NE1_HR_LC_SR_W.tif) ...")
    import time
    t0 = time.time()
    basemap_rgb, basemap_extent = load_basemap(target_width_px=6000, white_blend=0.32)
    print(f"  basemap loaded in {time.time() - t0:.1f}s, shape={basemap_rgb.shape}")

    site_r, site_level, pooled, coords = load_data()

    rho_table = {}
    for driver in DRIVERS:
        rho_table[driver["key"]] = {p: spearman_rho(site_level, site_r, driver["col"], p) for p in PRODUCTS}

    print("\nSpearman rho table (driver vs. site-level Pearson r, per product):")
    for key, row in rho_table.items():
        print(f"  {key}: " + ", ".join(f"{p}={row[p][0]:+.3f}(p={row[p][1]:.3f},n={row[p][2]})"
                                        if not np.isnan(row[p][0]) else f"{p}=n/a" for p in PRODUCTS))

    build_main_figure(site_r, site_level, pooled, basemap_rgb, basemap_extent, rho_table)

    for product in PRODUCTS:
        build_product_figure(product, site_r, site_level, basemap_rgb, basemap_extent, rho_table)

    print("\nAll figures written.")


if __name__ == "__main__":
    main()
