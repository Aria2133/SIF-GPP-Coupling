"""
New geospatial visualizations for the SIF vs tower-GPP main-paper analysis.

Main-paper products only: GOSIF, GOME-2, TROPOSIF, OCO-3. OCO-2 is excluded
everywhere in this script (both from data filtering and from any legend/panel).

This script does NOT modify any existing production data, script, or figure.
It only reads existing CSVs/rasters and writes new files into:
  All_the_Visualizations_REVISED_4PRODUCT_FINAL/02_new_geospatial_visualizations/

Mandatory basemap: <local path removed>
(Natural Earth I shaded relief with water, 21600x10800, EPSG:4326). Read via
rasterio with a decimated out_shape (~6000 px wide, Resampling.average) and
plotted with cartopy ccrs.EqualEarth() via ax.imshow(transform=ccrs.PlateCarree()).
The raster is blended toward white (~32%) so colored tower markers stay dominant,
matching the approach already used for a companion figure in this project.

No sparse tower values are interpolated into a fabricated continuous global
surface anywhere in this script (see explicit project instructions). Real local
gridded driver rasters (VPD/LST/TWS "Grasslands"/"Savannas" TIFFs) were checked
and found to be valid only over grassland/savanna land-cover pixels (~4.6% of
global cells finite) -- i.e. NOT a full global field usable as a background for
towers of other biomes -- so they are NOT used as a background raster. Instead,
Figure 5 (environmental context) uses each tower's own mean driver anomaly value,
computed directly from real per-site-month EC-tower driver records (already
extracted at each tower's real location for the mechanistic reanalysis), plotted
as tower points only.

Author: project authors
Run date: 2026-08-27
"""

from __future__ import annotations
# Original script filename (sanitized): build_new_geospatial_visualizations.py; public name: fig06_site_level_coupling_maps.py
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
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
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

FIGURE3_CSV = ROOT / "SIF_EC_SIF_GPP_Relationship_Analysis" / "05_figure_data" / "Figure3_Geospatial_Site_Level_Relationship.csv"
BIOME_ANOMALY_CSV = ROOT / "SIF_EC_SIF_GPP_Final_Verification" / "05_mechanistic_reanalysis" / "biome_relationships_final_anomaly.csv"
HARMONIC_RESIDUALS_CSV = ROOT / "SIF_EC_SIF_GPP_Final_Verification" / "02_independent_reverification" / "harmonic_residuals_full.csv"
META_EXTRACT_CSV = ROOT / "Figure_1_Global_EC_Tower_BIF_Metadata_Extract.csv"
MASTER_ANALYSIS_READY_CSV = ROOT / "SIF_EC_SIF_GPP_Relationship_Analysis" / "01_analysis_ready" / "SIF_GPP_Analysis_Ready_Master.csv"
EXTENDED_DRIVERS_CSV = ROOT / "SIF_EC_Mechanistic_FootprintAware_SIF_GPP_Paper" / "08_mechanistic_controls" / "Mechanistic_SIF_GPP_Monthly_With_Extended_Drivers.csv"

OUT_DIR = ROOT / "All_the_Visualizations_REVISED_4PRODUCT_FINAL" / "02_new_geospatial_visualizations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRODUCTS = ["GOSIF", "GOME-2", "TROPOSIF", "OCO-3"]  # main-paper products only; OCO-2 excluded everywhere

# Okabe-Ito colorblind-safe categorical set, fixed assignment (never cycled/reused for another role)
PRODUCT_COLORS = {
    "GOSIF": "#0072B2",     # blue
    "GOME-2": "#E69F00",    # orange
    "TROPOSIF": "#009E73",  # bluish green
    "OCO-3": "#CC79A7",     # reddish purple
}

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

# Diverging blue-white-red, matching the style already used in this project's companion
# heatmaps (BIOME_R_CMAP in build_revised_4product_final_visualizations.py) for r-type values.
DIVERGING_R_CMAP = LinearSegmentedColormap.from_list(
    "biome_r_diverging_new", ["#284B8F", "#F7F7F7", "#B6463A"]
)

FONT = {"family": "DejaVu Sans"}
plt.rcParams.update({
    "font.family": FONT["family"],
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.6,
})


# --------------------------------------------------------------------------------------
# Basemap
# --------------------------------------------------------------------------------------

def load_basemap(target_width_px: int = 6000, white_blend: float = 0.32):
    """Read NE1_HR_LC_SR_W.tif decimated to ~target_width_px wide, respecting its
    CRS/geotransform, and blend toward white so colored markers stay dominant."""
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


# --------------------------------------------------------------------------------------
# Pie-slice scatter markers (for the multi-product tower network map)
# --------------------------------------------------------------------------------------

def _wedge_path(theta1: float, theta2: float) -> MplPath:
    """Unit-circle wedge path (centered at origin, radius 1) from theta1 to theta2
    (degrees), for use as a matplotlib scatter `marker=`. Independent of any data
    projection since scatter markers are defined in points-space."""
    if theta2 - theta1 >= 359.999:
        n = 60
        angles = np.linspace(0, 2 * np.pi, n)
        x = np.cos(angles)
        y = np.sin(angles)
        verts = np.column_stack([x, y])
        codes = [MplPath.MOVETO] + [MplPath.LINETO] * (n - 1) + [MplPath.CLOSEPOLY]
        verts = np.vstack([verts, verts[0]])
        return MplPath(verts, codes)
    n = max(2, int(np.ceil((theta2 - theta1) / 6)))
    angles = np.radians(np.linspace(theta1, theta2, n))
    arc_x = np.cos(angles)
    arc_y = np.sin(angles)
    verts = np.vstack([[0, 0], np.column_stack([arc_x, arc_y]), [0, 0]])
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 2) + [MplPath.CLOSEPOLY]
    return MplPath(verts, codes)


def scatter_pie(ax, x, y, flags: list[bool], colors: list[str], size: float, transform, zorder=6):
    """Draw one pie-slice marker per point at (x, y); `flags` says which of the
    `colors`-ordered categories are present at this point. Equal-angle slices are
    used for present categories only (this encodes *membership*, not any weighting)."""
    present = [c for c, f in zip(colors, flags) if f]
    k = len(present)
    if k == 0:
        return
    if k == 1:
        ax.scatter([x], [y], transform=transform, marker="o", s=size, facecolor=present[0],
                   edgecolor="white", linewidth=0.45, zorder=zorder)
        return
    step = 360.0 / k
    for i, c in enumerate(present):
        path = _wedge_path(i * step, (i + 1) * step)
        ax.scatter([x], [y], transform=transform, marker=path, s=size, facecolor=c,
                   edgecolor="white", linewidth=0.35, zorder=zorder)
    # thin outline ring on top so slice boundaries read cleanly
    ring = _wedge_path(0, 360)
    ax.scatter([x], [y], transform=transform, marker=ring, s=size, facecolor="none",
               edgecolor="white", linewidth=0.55, zorder=zorder + 1)


# --------------------------------------------------------------------------------------
# Data loading / harmonization
# --------------------------------------------------------------------------------------

def load_site_product_table() -> pd.DataFrame:
    df = pd.read_csv(FIGURE3_CSV)
    df = df[df["SIF_PRODUCT"].isin(PRODUCTS)].copy()
    assert not (df["SIF_PRODUCT"] == "OCO-2").any()
    return df


def build_igbp_lookup(site_ids: pd.Index) -> pd.Series:
    """Per-site IGBP, preferring (1) the Figure3 table's own non-null IGBP (mode
    across its product rows for that site), then (2) the BIF metadata extract,
    then (3) the analysis-ready master table's IGBP_GPP_TABLE column. Sites with
    no IGBP in any source are labeled 'Unknown' (not fabricated)."""
    fig3 = pd.read_csv(FIGURE3_CSV)
    fig3 = fig3[fig3["SIF_PRODUCT"].isin(PRODUCTS)]
    igbp_fig3 = (
        fig3.dropna(subset=["IGBP"])
        .groupby("SITE_ID")["IGBP"]
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


def build_tower_table() -> pd.DataFrame:
    """One row per unique SITE_ID with lat/lon, IGBP, and a boolean column per
    product indicating whether that tower has data in that product."""
    df = load_site_product_table()
    coords = df.groupby("SITE_ID")[["LATITUDE", "LONGITUDE"]].first()
    coverage = pd.crosstab(df["SITE_ID"], df["SIF_PRODUCT"]).astype(bool)
    for p in PRODUCTS:
        if p not in coverage.columns:
            coverage[p] = False
    coverage = coverage[PRODUCTS]
    towers = coords.join(coverage)
    towers["IGBP"] = build_igbp_lookup(towers.index)
    towers["N_PRODUCTS"] = towers[PRODUCTS].sum(axis=1)
    return towers.reset_index()


def legend_product_handles():
    return [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=PRODUCT_COLORS[p],
               markeredgecolor="white", markersize=9, label=p)
        for p in PRODUCTS
    ]


def save_figure(fig, stem: str):
    png = OUT_DIR / f"{stem}.png"
    svg = OUT_DIR / f"{stem}.svg"
    fig.savefig(png, dpi=600, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(svg, dpi=600, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved: {png.name}, {svg.name}")


def source_note(ax, text: str):
    ax.text(
        0.5, -0.06, wrap_text(text, 150), transform=ax.transAxes, ha="center", va="top",
        fontsize=6.3, color="#555555",
    )


def wrap_text(text: str, width: int = 150) -> str:
    """Hard-wrap long single-line text at a safe character width. Long unwrapped
    text handed to fig.suptitle/fig.text can blow up the bbox_inches='tight'
    canvas (the tight bbox includes the full unwrapped text extent), which then
    shrinks all other artists relative to the page -- so every multi-line title
    or footer in this script goes through this helper rather than relying on
    matplotlib's own (render-order-dependent) `wrap=True`."""
    lines = []
    for para in text.split("\n"):
        wrapped = textwrap.wrap(para, width=width) or [""]
        lines.extend(wrapped)
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Figure 1: Tower network map
# --------------------------------------------------------------------------------------

def build_figure1_tower_network(basemap_rgb, basemap_extent):
    print("Building Figure 1: tower network map ...")
    towers = build_tower_table()

    fig = plt.figure(figsize=(13, 7.2))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.EqualEarth())
    add_basemap(ax, basemap_rgb, basemap_extent)

    proj = ccrs.PlateCarree()
    for _, row in towers.iterrows():
        flags = [bool(row[p]) for p in PRODUCTS]
        colors = [PRODUCT_COLORS[p] for p in PRODUCTS]
        scatter_pie(ax, row["LONGITUDE"], row["LATITUDE"], flags, colors, size=95, transform=proj)

    ax.set_title(
        "Global eddy-covariance tower network used in the SIF-GPP main-paper analysis\n"
        f"({len(towers)} unique towers; pie slices show which of the 4 main-paper SIF products cover each tower)",
        fontsize=12.5, fontweight="bold", loc="left", pad=10,
    )
    handles = legend_product_handles()
    leg = ax.legend(
        handles=handles, title="SIF product (pie slice color)", loc="lower left",
        fontsize=9, title_fontsize=9.5, frameon=True, facecolor="white", framealpha=0.9,
        bbox_to_anchor=(0.0, 0.0),
    )
    leg.get_frame().set_edgecolor("#999999")

    n_multi = int((towers["N_PRODUCTS"] > 1).sum())
    ax.text(
        0.99, 0.02,
        f"{n_multi} of {len(towers)} towers are covered by more than one product (multi-slice markers);\n"
        f"OCO-2 excluded throughout.",
        transform=ax.transAxes, fontsize=7.3, color="#333333", ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#999999", linewidth=0.5),
    )
    source_note(
        ax,
        "Basemap: NE1_HR_LC_SR_W.tif (Natural Earth I shaded relief w/ water), decimated read, blended 32% toward white. "
        "Site data: Figure3_Geospatial_Site_Level_Relationship.csv (OCO-2 excluded). No gridded driver raster used.",
    )
    save_figure(fig, "Fig_G1_Tower_Network_Map")


# --------------------------------------------------------------------------------------
# Figure 2: Site-level anomaly-based SIF-GPP coupling map (complementary to Figure3)
# --------------------------------------------------------------------------------------

def compute_site_level_anomaly_coupling() -> pd.DataFrame:
    """Per SITE_ID x SIF_PRODUCT Pearson r between SIF_ANOM and GPP_ANOM, using the
    FINAL accepted anomaly method's own residual file (harmonic seasonal regression
    residuals), aggregated to site level rather than biome level. This is a genuine
    site-level table, not an invented one -- the same underlying anomalies feed the
    accepted biome_relationships_final_anomaly.csv."""
    h = pd.read_csv(HARMONIC_RESIDUALS_CSV, low_memory=False)
    h = h[h["SIF_PRODUCT"].isin(PRODUCTS)].copy()
    h = h.dropna(subset=["SIF_ANOM", "GPP_ANOM", "LATITUDE", "LONGITUDE"])

    rows = []
    for (site, product), g in h.groupby(["SITE_ID", "SIF_PRODUCT"]):
        n = len(g)
        if n >= 4 and g["SIF_ANOM"].nunique() > 1 and g["GPP_ANOM"].nunique() > 1:
            r = float(np.corrcoef(g["SIF_ANOM"], g["GPP_ANOM"])[0, 1])
        else:
            r = np.nan
        rows.append({
            "SITE_ID": site, "SIF_PRODUCT": product, "N_SITEMONTHS": n,
            "PEARSON_R_ANOMALY": r,
            "LATITUDE": g["LATITUDE"].iloc[0], "LONGITUDE": g["LONGITUDE"].iloc[0],
        })
    out = pd.DataFrame(rows)
    out["RELIABLE"] = out["N_SITEMONTHS"] >= 12  # same site-month threshold as the biome reliability rule
    return out


def build_figure2_site_anomaly_coupling(basemap_rgb, basemap_extent):
    print("Building Figure 2: site-level anomaly-coupling map (complementary to Figure3) ...")
    site_r = compute_site_level_anomaly_coupling()
    site_r = site_r.dropna(subset=["PEARSON_R_ANOMALY"])

    fig, axes = plt.subplots(
        2, 2, figsize=(14, 9), subplot_kw={"projection": ccrs.EqualEarth()},
    )
    axes = axes.ravel()
    panel_labels = ["a", "b", "c", "d"]
    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=1.0)

    for i, (ax, product) in enumerate(zip(axes, PRODUCTS)):
        add_basemap(ax, basemap_rgb, basemap_extent)
        sub = site_r[site_r["SIF_PRODUCT"] == product]
        reliable = sub[sub["RELIABLE"]]
        unreliable = sub[~sub["RELIABLE"]]

        if len(unreliable):
            ax.scatter(
                unreliable["LONGITUDE"], unreliable["LATITUDE"], transform=ccrs.PlateCarree(),
                c=unreliable["PEARSON_R_ANOMALY"], cmap=DIVERGING_R_CMAP, norm=norm,
                s=34, marker="o", edgecolor="#555555", linewidth=0.5, alpha=0.35, zorder=5,
            )
        sc = ax.scatter(
            reliable["LONGITUDE"], reliable["LATITUDE"], transform=ccrs.PlateCarree(),
            c=reliable["PEARSON_R_ANOMALY"], cmap=DIVERGING_R_CMAP, norm=norm,
            s=52, marker="o", edgecolor="white", linewidth=0.5, alpha=0.96, zorder=6,
        )
        ax.set_title(f"({panel_labels[i]}) {product}  (n={len(sub)} towers)", loc="left",
                     fontsize=11, fontweight="bold", pad=5)

    cbar_ax = fig.add_axes([0.30, 0.105, 0.40, 0.018])
    cb = fig.colorbar(sc, cax=cbar_ax, orientation="horizontal")
    cb.set_label("Site-level Pearson r (SIF anomaly vs GPP anomaly, harmonic-residual method)", fontsize=9)

    fig.suptitle(
        wrap_text(
            "Site-level SIF-GPP coupling from deseasonalized anomalies (complementary to the raw-value Figure 3 R² map)\n"
            "Faint/gray-edged points = fewer than 12 site-months at that tower (statistically underpowered, de-emphasized)",
            width=110,
        ),
        fontsize=12.2, fontweight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.025,
        wrap_text(
            "Basemap: NE1_HR_LC_SR_W.tif, decimated read, blended 32% toward white. "
            "Anomalies: harmonic seasonal regression residuals (harmonic_residuals_full.csv), the final accepted method. No interpolated surface.",
            width=155,
        ),
        ha="center", va="top", fontsize=6.6, color="#555555",
    )
    fig.subplots_adjust(top=0.90, bottom=0.16, hspace=0.12, wspace=0.05)
    save_figure(fig, "Fig_G2_SiteLevel_Anomaly_Coupling_Map")
    return site_r


# --------------------------------------------------------------------------------------
# Figure 3: Biome context map
# --------------------------------------------------------------------------------------

def build_figure3_biome_context(basemap_rgb, basemap_extent):
    print("Building Figure 3: biome context map ...")
    towers = build_tower_table()

    fig = plt.figure(figsize=(13, 7.2))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.EqualEarth())
    add_basemap(ax, basemap_rgb, basemap_extent)

    proj = ccrs.PlateCarree()
    present_biomes = [b for b in IGBP_ORDER if (towers["IGBP"] == b).any()]
    for biome in present_biomes:
        sub = towers[towers["IGBP"] == biome]
        if biome == "Unknown":
            # 'x' is an unfilled/stroke-only marker in matplotlib: it is drawn from
            # `color`/facecolor, NOT `edgecolor` (edgecolor is silently ignored for
            # unfilled marker styles) -- using edgecolor here would render invisible.
            ax.scatter(
                sub["LONGITUDE"], sub["LATITUDE"], transform=proj,
                color=IGBP_COLORS[biome], marker="x", linewidth=1.8, s=90, alpha=0.95, zorder=6,
                label=f"{biome} - {IGBP_LABELS[biome]} (n={len(sub)})",
            )
        else:
            ax.scatter(
                sub["LONGITUDE"], sub["LATITUDE"], transform=proj,
                facecolor=IGBP_COLORS[biome], edgecolor="white", linewidth=0.5,
                marker="o", s=70, alpha=0.95, zorder=6,
                label=f"{biome} - {IGBP_LABELS[biome]} (n={len(sub)})",
            )

    ax.set_title(
        f"IGBP biome class of each eddy-covariance tower ({len(towers)} unique towers, main-paper products only)",
        fontsize=12.5, fontweight="bold", loc="left", pad=10,
    )
    leg = ax.legend(
        loc="lower left", fontsize=8, title="IGBP biome (site property, not product-specific)",
        title_fontsize=8.8, frameon=True, facecolor="white", framealpha=0.9, ncol=1,
        bbox_to_anchor=(0.0, 0.0),
    )
    leg.get_frame().set_edgecolor("#999999")
    source_note(
        ax,
        "Basemap: NE1_HR_LC_SR_W.tif, decimated read, blended 32% toward white. IGBP source: Figure3_Geospatial_Site_Level_Relationship.csv "
        "(mode across products), backfilled from Figure_1_Global_EC_Tower_BIF_Metadata_Extract.csv then SIF_GPP_Analysis_Ready_Master.csv. "
        "'Unknown' = no IGBP found in any source (not fabricated).",
    )
    save_figure(fig, "Fig_G3_Biome_Context_Map")


# --------------------------------------------------------------------------------------
# Figure 4: Biome anomaly-coupling reliability map (faceted by product)
# --------------------------------------------------------------------------------------

def build_figure4_biome_reliability(basemap_rgb, basemap_extent):
    print("Building Figure 4: biome anomaly-coupling reliability map ...")
    towers = build_tower_table()
    biome_tbl = pd.read_csv(BIOME_ANOMALY_CSV)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), subplot_kw={"projection": ccrs.EqualEarth()})
    axes = axes.ravel()
    panel_labels = ["a", "b", "c", "d"]
    norm = TwoSlopeNorm(vmin=-0.2, vcenter=0.0, vmax=0.75)

    sc = None
    for i, (ax, product) in enumerate(zip(axes, PRODUCTS)):
        add_basemap(ax, basemap_rgb, basemap_extent)
        sub_towers = towers[towers[product]].copy()
        prod_biome = biome_tbl[biome_tbl["product"] == product].set_index("biome")

        sub_towers = sub_towers.merge(
            prod_biome[["pearson_r_anomaly", "reliability_flag", "N_site_months", "N_towers"]],
            left_on="IGBP", right_index=True, how="left",
        )
        no_cell = sub_towers[sub_towers["pearson_r_anomaly"].isna()]
        unreliable = sub_towers[(sub_towers["reliability_flag"] == False) & sub_towers["pearson_r_anomaly"].notna()]
        reliable = sub_towers[(sub_towers["reliability_flag"] == True) & sub_towers["pearson_r_anomaly"].notna()]

        if len(no_cell):
            ax.scatter(
                no_cell["LONGITUDE"], no_cell["LATITUDE"], transform=ccrs.PlateCarree(),
                facecolor="none", edgecolor="#888888", hatch="////", marker="s", s=42,
                linewidth=0.6, alpha=0.55, zorder=5,
            )
        if len(unreliable):
            ax.scatter(
                unreliable["LONGITUDE"], unreliable["LATITUDE"], transform=ccrs.PlateCarree(),
                c=unreliable["pearson_r_anomaly"], cmap=DIVERGING_R_CMAP, norm=norm,
                marker="o", s=36, edgecolor="#555555", linewidth=0.5, alpha=0.30, zorder=5,
            )
        if len(reliable):
            sc = ax.scatter(
                reliable["LONGITUDE"], reliable["LATITUDE"], transform=ccrs.PlateCarree(),
                c=reliable["pearson_r_anomaly"], cmap=DIVERGING_R_CMAP, norm=norm,
                marker="o", s=58, edgecolor="white", linewidth=0.55, alpha=0.97, zorder=6,
            )
        ax.set_title(f"({panel_labels[i]}) {product}  (n={len(sub_towers)} towers)", loc="left",
                     fontsize=11, fontweight="bold", pad=5)

    # Bottom band layout (figure-fraction y, bottom to top), with generous gaps so the
    # colorbar (bar + ticks + label) and the legend never overlap: caption (0.012-0.03),
    # legend (0.055-0.09), colorbar block (0.145-0.20, ticks/label placed ABOVE the bar
    # so they sit in the gap toward the maps, not toward the legend below).
    if sc is not None:
        cbar_ax = fig.add_axes([0.30, 0.145, 0.40, 0.018])
        cb = fig.colorbar(sc, cax=cbar_ax, orientation="horizontal")
        cb.ax.xaxis.set_ticks_position("top")
        cb.ax.xaxis.set_label_position("top")
        cb.set_label("Biome-level Pearson r (SIF anomaly vs GPP anomaly), final accepted method", fontsize=9)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#B6463A",
               markeredgecolor="white", markersize=9, alpha=0.97, label="Reliable (N\u226512 site-months & \u22653 towers)"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#B6463A",
               markeredgecolor="#555555", markersize=8, alpha=0.30, label="Unreliable (below threshold, faded)"),
        Line2D([0], [0], marker="s", linestyle="none", markerfacecolor="none",
               markeredgecolor="#888888", markersize=8, alpha=0.55, label="No product\u00d7biome cell at all (hatched)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=8.3,
               bbox_to_anchor=(0.5, 0.055), frameon=False)

    fig.suptitle(
        wrap_text(
            "Geographic reliability of biome-level SIF-GPP anomaly coupling, by product\n"
            "Full color = reliable biome\u00d7product cell; faded = below N threshold; hatched = no such cell in the reliability table",
            width=110,
        ),
        fontsize=12.2, fontweight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.018,
        wrap_text(
            "Basemap: NE1_HR_LC_SR_W.tif, decimated read, blended 32% toward white. "
            "Biome reliability: biome_relationships_final_anomaly.csv (26 of 34 product\u00d7biome cells reliable). No interpolated surface; towers only.",
            width=155,
        ),
        ha="center", fontsize=6.6, color="#555555",
    )
    fig.subplots_adjust(top=0.90, bottom=0.22, hspace=0.14, wspace=0.05)
    save_figure(fig, "Fig_G4_Biome_Anomaly_Reliability_Map")


# --------------------------------------------------------------------------------------
# Figure 5: Environmental context (tower points only -- no fabricated surface)
# --------------------------------------------------------------------------------------

DRIVER_DEFS = [
    ("VPD_ANOM", "VPD anomaly", "kPa"),
    ("TAIR_ANOM", "Air temperature anomaly", "\u00b0C"),
    ("SW_IN_ANOM", "Incoming shortwave radiation anomaly", "W m\u207b\u00b2"),
    ("SWC_ANOM", "Soil water content anomaly", "fraction"),
    ("ET_ANOM", "Evapotranspiration anomaly", "mm"),
]


def compute_site_mean_driver_anomalies() -> pd.DataFrame:
    d = pd.read_csv(EXTENDED_DRIVERS_CSV, low_memory=False)
    if "SIF_PRODUCT" in d.columns:
        d = d[d["SIF_PRODUCT"].isin(PRODUCTS)]
    lat_col = "LATITUDE" if "LATITUDE" in d.columns else "LAT"
    lon_col = "LONGITUDE" if "LONGITUDE" in d.columns else "LON"
    agg = {c: "mean" for c, _, _ in DRIVER_DEFS if c in d.columns}
    coords = d.groupby("SITE_ID")[[lat_col, lon_col]].first()
    means = d.groupby("SITE_ID").agg(agg)
    out = coords.join(means)
    out = out.rename(columns={lat_col: "LATITUDE", lon_col: "LONGITUDE"})
    return out.reset_index()


def build_figure5_environmental_context(basemap_rgb, basemap_extent):
    print("Building Figure 5: environmental context (tower points only, no interpolated surface) ...")
    site_drivers = compute_site_mean_driver_anomalies()
    available = [(c, label, unit) for c, label, unit in DRIVER_DEFS if c in site_drivers.columns]

    n = len(available)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    # EqualEarth maps have a ~2:1 (w:h) native aspect once ax.set_global() locks it;
    # size each cell close to that aspect (6in wide -> ~3.3in tall incl. title/colorbar)
    # so subplots_adjust below doesn't have to fight a mostly-empty tall cell.
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 3.3 * nrows),
                              subplot_kw={"projection": ccrs.EqualEarth()})
    axes = np.atleast_1d(axes).ravel()
    panel_labels = ["a", "b", "c", "d", "e", "f"]

    for i, (col, label, unit) in enumerate(available):
        ax = axes[i]
        add_basemap(ax, basemap_rgb, basemap_extent)
        sub = site_drivers.dropna(subset=[col])
        vmax = np.nanpercentile(np.abs(sub[col]), 95) if len(sub) else 1.0
        vmax = max(vmax, 1e-6)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        sc = ax.scatter(
            sub["LONGITUDE"], sub["LATITUDE"], transform=ccrs.PlateCarree(),
            c=sub[col], cmap=DIVERGING_R_CMAP, norm=norm,
            s=42, edgecolor="white", linewidth=0.4, alpha=0.95, zorder=6,
        )
        ax.set_title(f"({panel_labels[i]}) {label}  (n={len(sub)} towers)", loc="left",
                     fontsize=10.3, fontweight="bold", pad=5)
        cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.045, pad=0.04, shrink=0.85)
        cb.set_label(f"mean site anomaly ({unit})", fontsize=7.3)
        cb.ax.tick_params(labelsize=6.5)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    # NOTE: fig.suptitle/fig.text are given hard-wrapped (wrap_text) short lines rather
    # than one long unwrapped string. An earlier version of this figure passed one
    # ~330-character unwrapped suptitle line, which blew up the bbox_inches='tight'
    # canvas width (the tight bbox includes the full unwrapped text extent) and made
    # every map render tiny relative to the page -- confirmed by isolating the bug to
    # exactly this call. wrap_text() is the fix; keep using it for every multi-line
    # title/footer in this script.
    fig.suptitle(
        wrap_text(
            "Environmental context: each tower's own mean driver anomaly (points only, no interpolated field)",
            width=95,
        ),
        fontsize=11.6, fontweight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.945,
        wrap_text(
            "Shown for context; the independent mechanistic reanalysis found only weak, mostly non-significant "
            "SIF-GPP environmental-interaction evidence (VPD and temperature: consistent sign in 4/4 products, but "
            "only 1/4 and 0/4 CIs excluding zero respectively) -- this map is NOT a claim of a robust environmental "
            "control on coupling.",
            width=150,
        ),
        ha="center", va="top", fontsize=8.3, color="#333333",
    )
    fig.text(
        0.5, 0.01,
        wrap_text(
            "Basemap: NE1_HR_LC_SR_W.tif, decimated read, blended 32% toward white. Driver values: real per-site-month EC-tower "
            "records (Mechanistic_SIF_GPP_Monthly_With_Extended_Drivers.csv), averaged per tower. No gridded driver raster was used: "
            "the only locally available gridded VPD/LST/TWS TIFFs are valid over grassland/savanna land-cover pixels only "
            "(~4.6% global cell coverage) and would misrepresent towers of other biomes if used as a background field.",
            width=170,
        ),
        ha="center", fontsize=6.5, color="#555555",
    )
    fig.subplots_adjust(top=0.82, bottom=0.10, left=0.02, right=0.98, hspace=0.38, wspace=0.08)
    save_figure(fig, "Fig_G5_Environmental_Context_Map")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    print("Loading basemap (decimated read of NE1_HR_LC_SR_W.tif) ...")
    import time
    t0 = time.time()
    basemap_rgb, basemap_extent = load_basemap(target_width_px=6000, white_blend=0.32)
    print(f"  basemap loaded in {time.time() - t0:.1f}s, shape={basemap_rgb.shape}")

    build_figure1_tower_network(basemap_rgb, basemap_extent)
    build_figure2_site_anomaly_coupling(basemap_rgb, basemap_extent)
    build_figure3_biome_context(basemap_rgb, basemap_extent)
    build_figure4_biome_reliability(basemap_rgb, basemap_extent)
    build_figure5_environmental_context(basemap_rgb, basemap_extent)

    print("\nAll figures written to:", OUT_DIR)


if __name__ == "__main__":
    main()
