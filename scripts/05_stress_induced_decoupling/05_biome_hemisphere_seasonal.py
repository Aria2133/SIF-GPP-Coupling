"""
Phase 2 -- Biome, hemisphere, and seasonal-structure analysis of Stress-Induced
Decoupling (SID), built on the already-verified Phase 1 master table
(03_sid_monthly/monthly_SID_master.csv, 19775 site-month-product rows, 73
towers, 4 SIF products: GOSIF/GOME-2/TROPOSIF/OCO-3; OCO-2 already excluded).

Foundation is READ-ONLY: this script never recomputes SID_VPD/SID_SM/SID_TEMP
or any Phase-1 column; it only aggregates and visualizes them.

Reliability rule (applied everywhere, matches the project's established
statistical-audit standard): a product x stressor x biome (or x hemisphere)
cell needs N >= 12 site-months AND N >= 3 independent towers to be reported as
"reliable" -- unreliable cells are always shown de-emphasized, never hidden.

Denominator caveat (per 04_sid_validation/SID_initial_validation_summary.md,
item 4): the extreme top-decile SID values are somewhat inflated by
near-zero-denominator rows (SIF_norm+GPP_norm < 0.10, i.e. QC_FLAG_NOTES
contains 'LOW_DENOMINATOR_FLAG'). Every summary table therefore reports BOTH
the raw and the denominator-filtered (excluding those rows) statistics side
by side.

Hotspot definition: 05_hotspot_analysis/ (the parallel hotspot-figures agent's
output) did not exist yet when this script was first run, so this script uses
its own definition, noted here and in the output README: a site-month is a
"hotspot" if its SID_<stressor> value is >= the 90th percentile of SID_<stressor>
computed within that same product (pooling all biomes/hemispheres) -- i.e.
"top-10%-within-product". The denominator-filtered hotspot rate uses a
threshold recomputed on the denominator-filtered pool only, so it is not
itself contaminated by unstable near-zero-denominator extremes.

Outputs (all under SIF_EC_Stress_Induced_Decoupling_SID/):
  06_biome_hemisphere/biome_SID_summary.csv
  06_biome_hemisphere/hemisphere_SID_summary.csv
  06_biome_hemisphere/SID_Figure_Biome_Heatmap.{png,svg,pdf}
  06_biome_hemisphere/SID_Figure_Biome_Stressor_Comparison.{png,svg,pdf}
  06_biome_hemisphere/SID_Figure_Seasonal_NH.{png,svg,pdf}
  06_biome_hemisphere/SID_Figure_Seasonal_SH.{png,svg,pdf}
  06_biome_hemisphere/SID_Figure_Seasonal_Biome_<Biome>.{png,svg,pdf}  (qualifying biomes only)
  07_geospatial_figures/SID_Figure_Biome_Geographic_<Biome>.{png,svg,pdf}  (qualifying biomes only)

Author: project authors
Run date: 2026-09-01
"""
from __future__ import annotations
# Original script filename (sanitized): 05_biome_hemisphere_seasonal.py; public name: 05_biome_hemisphere_seasonal.py
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
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.lines import Line2D
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
SID_ROOT = ROOT / "SIF_EC_Stress_Induced_Decoupling_SID"
MASTER_CSV = SID_ROOT / "03_sid_monthly" / "monthly_SID_master.csv"
BASEMAP_TIF = Path(_os.environ.get("NE_BASEMAP_TIF", "NE1_HR_LC_SR_W.tif"))

OUT_BH = SID_ROOT / "06_biome_hemisphere"
OUT_GEO = SID_ROOT / "07_geospatial_figures"
OUT_BH.mkdir(parents=True, exist_ok=True)
OUT_GEO.mkdir(parents=True, exist_ok=True)

PRODUCTS = ["GOSIF", "GOME-2", "TROPOSIF", "OCO-3"]
STRESSORS = {"VPD": "SID_VPD", "SM": "SID_SM", "TEMP": "SID_TEMP"}
STRESS_ORDER = ["VPD", "SM", "TEMP"]

REL_MIN_N = 12
REL_MIN_TOWERS = 3
LOW_DENOM_FLAG = "LOW_DENOMINATOR_FLAG"

# Okabe-Ito colorblind-safe set, same fixed assignment used throughout this project
PRODUCT_COLORS = {
    "GOSIF": "#0072B2", "GOME-2": "#E69F00", "TROPOSIF": "#009E73", "OCO-3": "#CC79A7",
}
STRESSOR_COLORS = {"VPD": "#D55E00", "SM": "#0072B2", "TEMP": "#009E73"}

IGBP_LABELS = {
    "CRO": "Cropland", "CSH": "Closed shrubland", "DBF": "Deciduous broadleaf forest",
    "EBF": "Evergreen broadleaf forest", "ENF": "Evergreen needleleaf forest", "GRA": "Grassland",
    "MF": "Mixed forest", "OSH": "Open shrubland", "SAV": "Savanna", "WSA": "Woody savanna",
    "WET": "Wetland", "DNF": "Deciduous needleleaf forest", "SNO": "Snow/ice",
}
IGBP_ORDER = ["ENF", "EBF", "DBF", "MF", "DNF", "CSH", "OSH", "WSA", "SAV", "GRA", "CRO", "WET", "SNO"]

SID_CMAP = LinearSegmentedColormap.from_list(
    "sid_stress_sequential", ["#FFFFE0", "#FEC44F", "#D95F0E", "#7A0177"]
)
SID_CMAP.set_bad("#E8E8E8")

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

FONT = {"family": "DejaVu Sans"}
plt.rcParams.update({
    "font.family": FONT["family"],
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.6,
})


def wrap_text(text: str, width: int = 150) -> str:
    lines = []
    for para in text.split("\n"):
        wrapped = textwrap.wrap(para, width=width) or [""]
        lines.extend(wrapped)
    return "\n".join(lines)


def save_figure(fig, stem: str, out_dir: Path):
    png = out_dir / f"{stem}.png"
    svg = out_dir / f"{stem}.svg"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=350, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(svg, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved: {png}")


# --------------------------------------------------------------------------------------
# Basemap (reused pattern from All_the_Visualizations_REVISED_4PRODUCT_FINAL /
# 06_scripts/build_new_geospatial_visualizations.py -- load_basemap()/add_basemap())
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
    ax.imshow(rgb, origin="upper", extent=extent, transform=ccrs.PlateCarree(),
               interpolation="bilinear", zorder=0)
    ax.set_global()
    ax.spines["geo"].set_edgecolor("#888888")
    ax.spines["geo"].set_linewidth(0.5)


# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------

def load_master() -> pd.DataFrame:
    df = pd.read_csv(MASTER_CSV, low_memory=False)
    assert not (df["SIF_PRODUCT"] == "OCO-2").any()
    df["LOW_DENOM"] = df["QC_FLAG_NOTES"].fillna("").str.contains(LOW_DENOM_FLAG)
    df["HEMISPHERE"] = np.where(df["LATITUDE"] >= 0, "NH", "SH")
    return df


def reliable(n_sitemonths: int, n_towers: int) -> bool:
    return (n_sitemonths >= REL_MIN_N) and (n_towers >= REL_MIN_TOWERS)


def cell_stats(sub: pd.DataFrame, sid_col: str, threshold: float) -> dict:
    vals = sub[sid_col].dropna()
    n = len(vals)
    towers = sub.loc[vals.index, "SITE_ID"].nunique()
    if n == 0:
        return dict(N=0, N_towers=0, median=np.nan, Q1=np.nan, Q3=np.nan, IQR=np.nan,
                     p90=np.nan, hotspot_frac=np.nan, reliable=False)
    q1, q3, p90 = vals.quantile(0.25), vals.quantile(0.75), vals.quantile(0.90)
    hf = (vals >= threshold).mean() if pd.notna(threshold) else np.nan
    return dict(N=n, N_towers=towers, median=vals.median(), Q1=q1, Q3=q3, IQR=q3 - q1,
                p90=p90, hotspot_frac=hf, reliable=reliable(n, towers))


# --------------------------------------------------------------------------------------
# Step 1 / 2: biome_SID_summary.csv, hemisphere_SID_summary.csv
# --------------------------------------------------------------------------------------

def build_group_summary(df: pd.DataFrame, group_col: str, group_values: list[str], group_label_col: str | None,
                          out_name: str) -> pd.DataFrame:
    rows = []
    for stress_label, sid_col in STRESSORS.items():
        # hotspot thresholds per product (incl. pooled ALL), computed on the FULL
        # dataset (all biomes/hemispheres) for that product -- "top-10%-within-product"
        thr_raw, thr_filt = {}, {}
        for prod in PRODUCTS + ["ALL"]:
            pool = df if prod == "ALL" else df[df["SIF_PRODUCT"] == prod]
            raw_vals = pool[sid_col].dropna()
            thr_raw[prod] = raw_vals.quantile(0.90) if len(raw_vals) else np.nan
            filt_vals = pool.loc[~pool["LOW_DENOM"], sid_col].dropna()
            thr_filt[prod] = filt_vals.quantile(0.90) if len(filt_vals) else np.nan

        for prod in PRODUCTS + ["ALL"]:
            pool = df if prod == "ALL" else df[df["SIF_PRODUCT"] == prod]
            for gval in group_values:
                sub = pool[pool[group_col] == gval]
                raw_stats = cell_stats(sub, sid_col, thr_raw[prod])
                filt_sub = sub[~sub["LOW_DENOM"]]
                filt_stats = cell_stats(filt_sub, sid_col, thr_filt[prod])
                row = {
                    "product": prod, "stressor": stress_label, group_col: gval,
                    "N_sitemonths": raw_stats["N"], "N_towers": raw_stats["N_towers"],
                    "reliable": raw_stats["reliable"],
                    "median_SID": raw_stats["median"], "Q1_SID": raw_stats["Q1"], "Q3_SID": raw_stats["Q3"],
                    "IQR_SID": raw_stats["IQR"], "p90_SID": raw_stats["p90"],
                    "hotspot_frac": raw_stats["hotspot_frac"], "hotspot_threshold_SID": thr_raw[prod],
                    "N_sitemonths_denomfiltered": filt_stats["N"], "N_towers_denomfiltered": filt_stats["N_towers"],
                    "reliable_denomfiltered": filt_stats["reliable"],
                    "median_SID_denomfiltered": filt_stats["median"], "Q1_SID_denomfiltered": filt_stats["Q1"],
                    "Q3_SID_denomfiltered": filt_stats["Q3"], "IQR_SID_denomfiltered": filt_stats["IQR"],
                    "p90_SID_denomfiltered": filt_stats["p90"],
                    "hotspot_frac_denomfiltered": filt_stats["hotspot_frac"],
                    "hotspot_threshold_SID_denomfiltered": thr_filt[prod],
                }
                if group_label_col is not None:
                    row[group_label_col] = IGBP_LABELS.get(gval, gval)
                rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_BH / out_name, index=False)
    print(f"Saved: {OUT_BH / out_name}  shape={out.shape}")
    n_reliable = out[out["product"] != "ALL"]["reliable"].sum()
    n_total = (out["product"] != "ALL").sum()
    print(f"  reliable cells (product-level, excl. pooled ALL rows): {n_reliable}/{n_total}")
    return out


def main_summaries():
    df = load_master()
    print(f"Loaded master: {df.shape}, {df['SITE_ID'].nunique()} towers")
    n_no_igbp = df["IGBP"].isna().sum()
    print(f"Rows with no IGBP (excluded from biome table): {n_no_igbp} "
          f"({df.loc[df['IGBP'].isna(), 'SITE_ID'].nunique()} towers)")

    biomes_present = [b for b in IGBP_ORDER if (df["IGBP"] == b).any()]
    biome_summary = build_group_summary(df, "IGBP", biomes_present, "IGBP_label", "biome_SID_summary.csv")

    hemi_summary = build_group_summary(df, "HEMISPHERE", ["NH", "SH"], None, "hemisphere_SID_summary.csv")

    return df, biome_summary, hemi_summary, biomes_present


# --------------------------------------------------------------------------------------
# Step 1A: biome x product heatmap, faceted by stressor
# --------------------------------------------------------------------------------------

def build_biome_heatmap(biome_summary: pd.DataFrame, biomes_present: list[str]):
    print("Building Figure A: biome SID heatmap ...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 0.62 * len(biomes_present) + 3.2))

    for panel_idx, (ax, stressor) in enumerate(zip(axes, STRESS_ORDER)):
        sub = biome_summary[(biome_summary["stressor"] == stressor) & (biome_summary["product"] != "ALL")]
        val = sub.pivot(index="IGBP", columns="product", values="median_SID").reindex(
            index=biomes_present, columns=PRODUCTS)
        rel = sub.pivot(index="IGBP", columns="product", values="reliable").reindex(
            index=biomes_present, columns=PRODUCTS)
        n_tab = sub.pivot(index="IGBP", columns="product", values="N_sitemonths").reindex(
            index=biomes_present, columns=PRODUCTS)

        vmax = np.nanmax(val.values) if np.isfinite(np.nanmax(val.values)) else 1.0
        vmax = max(vmax, 1e-6)
        norm = Normalize(vmin=0, vmax=vmax)

        rgba = SID_CMAP(norm(val.values))
        has_data = n_tab.values > 0
        is_reliable = (rel.values == True) & has_data
        is_unreliable = (~(rel.values == True)) & has_data
        rgba[..., 3] = np.where(is_reliable, 1.0, np.where(is_unreliable, 0.35, 0.0))

        ax.imshow(rgba, aspect="auto")
        ax.set_facecolor("#EFEFEF")

        for i in range(val.shape[0]):
            for j in range(val.shape[1]):
                if not has_data[i, j]:
                    ax.text(j, i, "x", ha="center", va="center", fontsize=8, color="#999999")
                    continue
                txt_color = "black" if norm(val.values[i, j]) < 0.55 else "white"
                if is_unreliable[i, j]:
                    ax.add_patch(mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="none",
                                                      edgecolor="#666666", hatch="////", linewidth=0.4))
                    txt_color = "#444444"
                ax.text(j, i, f"{val.values[i, j]:.2f}", ha="center", va="center",
                         fontsize=7.3, color=txt_color)

        ax.set_xticks(range(len(PRODUCTS)))
        ax.set_xticklabels(PRODUCTS, rotation=30, ha="right", fontsize=8.5)
        ax.set_yticks(range(len(biomes_present)))
        if panel_idx == 0:
            ax.set_yticklabels([f"{b} ({IGBP_LABELS.get(b, b)})" for b in biomes_present], fontsize=8.2)
        else:
            ax.set_yticklabels([])
        ax.set_title(f"{stressor} stress", fontsize=12, fontweight="bold", loc="left")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(PRODUCTS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(biomes_present), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)

        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=SID_CMAP), ax=ax,
                           orientation="horizontal", fraction=0.045, pad=0.16)
        cb.set_label(f"median SID_{stressor}\n(this panel's own color scale)", fontsize=7.3)
        cb.ax.tick_params(labelsize=6.8)

    legend_handles = [
        mpatches.Patch(facecolor="#D95F0E", edgecolor="none", label="Reliable cell (full color)"),
        mpatches.Patch(facecolor="#D95F0E", edgecolor="#666666", hatch="////", alpha=0.35,
                        label=f"Unreliable (N<{REL_MIN_N} site-months or <{REL_MIN_TOWERS} towers, faded+hatched)"),
        Line2D([0], [0], marker="x", linestyle="none", color="#999999", markersize=8, label="No data for that cell"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.02), frameon=False)

    fig.suptitle(
        wrap_text(
            "Median Stress-Induced Decoupling (SID) by biome and SIF product, faceted by stressor\n"
            "Cross-site, descriptive summary of monthly site-level SID -- not a causal or trend claim. "
            "Each stressor panel uses its own color scale (VPD/SM and TEMP SID are on very different numeric ranges).",
            width=150,
        ),
        fontsize=12.3, fontweight="bold", y=1.04,
    )
    fig.text(0.5, -0.10,
              wrap_text(
                  "Reliability rule: N >= 12 site-months AND N >= 3 independent towers. Denominator caveat: top-decile SID "
                  "values are somewhat inflated by near-zero-denominator rows (SIF_norm+GPP_norm<0.10); see biome_SID_summary.csv "
                  "for denominator-filtered medians alongside these raw values. Source: 03_sid_monthly/monthly_SID_master.csv.",
                  width=175),
              ha="center", fontsize=6.8, color="#555555")
    fig.subplots_adjust(wspace=0.55)
    save_figure(fig, "SID_Figure_Biome_Heatmap", OUT_BH)


# --------------------------------------------------------------------------------------
# Step 1B: biome x stressor comparison (pooled across products, "ALL")
# --------------------------------------------------------------------------------------

def build_biome_stressor_comparison(biome_summary: pd.DataFrame, biomes_present: list[str]):
    print("Building Figure B: biome x stressor comparison ...")
    sub = biome_summary[biome_summary["product"] == "ALL"].copy()
    sub["IGBP"] = pd.Categorical(sub["IGBP"], categories=biomes_present, ordered=True)
    sub = sub.sort_values("IGBP")

    fig, ax = plt.subplots(figsize=(13, 6.2))
    n_biomes = len(biomes_present)
    n_stress = len(STRESS_ORDER)
    width = 0.8 / n_stress
    x = np.arange(n_biomes)

    for k, stressor in enumerate(STRESS_ORDER):
        s = sub[sub["stressor"] == stressor].set_index("IGBP").reindex(biomes_present)
        xpos = x + (k - (n_stress - 1) / 2) * width
        colors = [STRESSOR_COLORS[stressor] if rel else STRESSOR_COLORS[stressor] for rel in s["reliable"]]
        alphas = np.where(s["reliable"].values, 0.95, 0.35)
        yerr_lo = (s["median_SID"] - s["Q1_SID"]).clip(lower=0).values
        yerr_hi = (s["Q3_SID"] - s["median_SID"]).clip(lower=0).values
        for xi, yi, elo, ehi, a, rel_flag in zip(xpos, s["median_SID"].values, yerr_lo, yerr_hi,
                                                   alphas, s["reliable"].values):
            if pd.isna(yi):
                continue
            bar = ax.bar(xi, yi, width=width * 0.92, color=STRESSOR_COLORS[stressor], alpha=a,
                          edgecolor="white", linewidth=0.4,
                          hatch="////" if not rel_flag else None,
                          label=f"{stressor} stress" if xi == xpos[0] else None)
            ax.errorbar(xi, yi, yerr=[[elo], [ehi]], fmt="none", ecolor="#333333",
                         elinewidth=0.7, capsize=2, alpha=a)

    ax.set_xticks(x)
    ax.set_xticklabels(biomes_present, fontsize=10)
    ax.set_ylabel("Median SID (bars), IQR (error bars) -- pooled across all 4 SIF products", fontsize=9.5)
    ax.set_title(
        wrap_text("Stress-Induced Decoupling by biome: VPD vs soil-moisture vs temperature stress "
                   "(cross-site comparison, pooled across products)", width=110),
        fontsize=12.5, fontweight="bold", loc="left", pad=10,
    )

    handles = [mpatches.Patch(facecolor=STRESSOR_COLORS[s], label=f"{s} stress", alpha=0.95) for s in STRESS_ORDER]
    handles.append(mpatches.Patch(facecolor="#AAAAAA", edgecolor="#666666", hatch="////", alpha=0.35,
                                   label=f"Unreliable cell (N<{REL_MIN_N} site-months or <{REL_MIN_TOWERS} towers)"))
    ax.legend(handles=handles, loc="upper right", fontsize=8.3, frameon=True, framealpha=0.9)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    biome_code_key = "  |  ".join(f"{b}={IGBP_LABELS.get(b, b)}" for b in biomes_present)
    fig.text(0.5, -0.03,
              wrap_text(
                  "Bars pool all 4 SIF products for each biome x stressor cell (reliability re-evaluated on the pooled N). "
                  "Faded/hatched bars fall below the reliability threshold and should not be over-interpreted. "
                  "Descriptive cross-site comparison only, not a causal claim.", width=175),
              ha="center", fontsize=7.2, color="#555555")
    fig.text(0.5, -0.075, wrap_text(biome_code_key, width=175), ha="center", fontsize=7.0, color="#555555")
    save_figure(fig, "SID_Figure_Biome_Stressor_Comparison", OUT_BH)


# --------------------------------------------------------------------------------------
# Step 1C: biome-specific geographic hotspot maps (only for qualifying biomes)
# --------------------------------------------------------------------------------------

def qualifying_biomes_for_geo(biome_summary: pd.DataFrame, biomes_present: list[str]) -> dict[str, list[str]]:
    """Biome qualifies for a geographic figure if the pooled-ALL-product cell is
    reliable for at least one stressor; returns {biome: [reliable stressors]}."""
    pooled = biome_summary[biome_summary["product"] == "ALL"]
    out = {}
    for b in biomes_present:
        sub = pooled[pooled["IGBP"] == b]
        rel_stressors = sub[sub["reliable"]]["stressor"].tolist()
        if rel_stressors:
            out[b] = rel_stressors
    return out


def build_biome_geographic_maps(df: pd.DataFrame, biome_summary: pd.DataFrame, biomes_present: list[str],
                                  basemap_rgb, basemap_extent):
    qualifying = qualifying_biomes_for_geo(biome_summary, biomes_present)
    print(f"Biome geographic maps: {len(qualifying)}/{len(biomes_present)} biomes qualify "
          f"(pooled-ALL-product cell reliable for >=1 stressor): {list(qualifying.keys())}")
    skipped = [b for b in biomes_present if b not in qualifying]
    if skipped:
        pooled = biome_summary[biome_summary["product"] == "ALL"]
        for b in skipped:
            n_towers = pooled[(pooled["IGBP"] == b)]["N_towers"].max()
            print(f"  skipped {b}: max N_towers across stressors = {n_towers} (< {REL_MIN_TOWERS} needed)")

    # global per-stressor color scale: 90th pct of SITE-level median SID (pooled ALL
    # product, all qualifying biomes) so biome maps are comparable to one another
    site_level = (df.dropna(subset=["IGBP"])
                  .groupby(["SITE_ID", "IGBP"])[list(STRESSORS.values())].median().reset_index())
    vmax_by_stressor = {}
    for stressor, sid_col in STRESSORS.items():
        vals = site_level[sid_col].dropna()
        vmax_by_stressor[stressor] = max(vals.quantile(0.90), 1e-3) if len(vals) else 1.0

    coords = df.groupby("SITE_ID")[["LATITUDE", "LONGITUDE"]].first()

    # NOTE: an earlier version of this function used a regional PlateCarree crop
    # (ax.set_extent around each biome's tower bounding box). IGBP biomes in this
    # network are globally scattered (e.g. ENF towers span N. America to Europe;
    # OSH spans California/W. Africa/Australia), so a regional crop has a longitude
    # span far larger than its latitude span. Cartopy's GeoAxes keeps a true equal
    # -angle aspect ratio, so that crop rendered as a thin horizontal letterboxed
    # strip with huge blank canvas above/below (and orphaned the figure-fraction
    # colorbar/caption below the visible map) once bbox_inches='tight' captured the
    # true rendered content. Fixed by using the same global EqualEarth view as the
    # rest of this project's geospatial figures (no per-biome cropping needed).
    for biome, rel_stressors in qualifying.items():
        print(f"Building Figure C: biome geographic map for {biome} ...")
        sites_here = site_level[site_level["IGBP"] == biome].merge(coords, on="SITE_ID", how="left")

        fig, axes = plt.subplots(1, 3, figsize=(19, 4.2), subplot_kw={"projection": ccrs.EqualEarth()})
        sc_ref = None
        for ax, stressor in zip(axes, STRESS_ORDER):
            add_basemap(ax, basemap_rgb, basemap_extent)
            sid_col = STRESSORS[stressor]
            sub = sites_here.dropna(subset=[sid_col])
            n_towers_stress = sub["SITE_ID"].nunique()
            is_qualifying_panel = stressor in rel_stressors
            if len(sub):
                norm = Normalize(vmin=0, vmax=vmax_by_stressor[stressor])
                sc = ax.scatter(sub["LONGITUDE"], sub["LATITUDE"], transform=ccrs.PlateCarree(),
                                 c=sub[sid_col], cmap=SID_CMAP, norm=norm, s=110,
                                 edgecolor="white" if is_qualifying_panel else "#888888",
                                 linewidth=0.8, alpha=0.95 if is_qualifying_panel else 0.4, zorder=6)
                sc_ref = sc_ref or sc
            ax.set_title(f"{stressor} stress  (n={n_towers_stress} towers)"
                         + ("" if is_qualifying_panel else "  [below reliability threshold]"),
                         fontsize=10.5, fontweight="bold" if is_qualifying_panel else "normal",
                         loc="left", color="black" if is_qualifying_panel else "#888888")
            if not is_qualifying_panel:
                ax.text(0.5, 0.06, "Insufficient N for reliable reporting\n(shown faded for context only)",
                        transform=ax.transAxes, ha="center", va="bottom", fontsize=7.6, color="#B03A2E",
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="#B03A2E"))

        # Bottom band layout (figure-fraction y, bottom to top), generous gaps so the
        # colorbar and caption never collide: caption (0.01-0.055, below everything),
        # colorbar bar (0.115-0.135) with its ticks/label placed ABOVE the bar (toward
        # the maps) via set_ticks_position('top')/set_label_position('top') -- same
        # technique used in build_new_geospatial_visualizations.py Figure 4 to keep a
        # horizontal colorbar's label from colliding with text below it.
        if sc_ref is not None:
            cbar_ax = fig.add_axes([0.30, 0.115, 0.40, 0.02])
            cb = fig.colorbar(sc_ref, cax=cbar_ax, orientation="horizontal")
            cb.ax.xaxis.set_ticks_position("top")
            cb.ax.xaxis.set_label_position("top")
            cb.set_label("Site-level median SID (color scale fixed across all biome maps for this stressor)", fontsize=8)
            cb.ax.tick_params(labelsize=7)

        fig.suptitle(
            wrap_text(
                f"{biome} ({IGBP_LABELS.get(biome, biome)}) -- geographic pattern of Stress-Induced Decoupling "
                f"({sites_here['SITE_ID'].nunique()} towers). Cross-site, descriptive view; SID magnitude only, not a causal claim.",
                width=140),
            fontsize=12, fontweight="bold", y=1.08,
        )
        fig.text(0.5, 0.055,
                  wrap_text(
                      "Basemap: NE1_HR_LC_SR_W.tif, decimated read, blended 32% toward white. Points = per-tower median SID "
                      "(pooled across all 4 SIF products, all months). Panels marked [below reliability threshold] have "
                      f"< {REL_MIN_TOWERS} towers or < {REL_MIN_N} site-months pooled and are shown faded for context, not for interpretation.",
                      width=170),
                  ha="center", va="top", fontsize=6.8, color="#555555")
        fig.subplots_adjust(top=0.80, bottom=0.24, wspace=0.06)
        save_figure(fig, f"SID_Figure_Biome_Geographic_{biome}", OUT_GEO)

    return qualifying


# --------------------------------------------------------------------------------------
# Step 3: seasonal (monthly climatology) structure
# --------------------------------------------------------------------------------------

def monthly_stats(sub: pd.DataFrame, sid_col: str) -> pd.DataFrame:
    rows = []
    for m in range(1, 13):
        msub = sub[sub["MONTH"] == m].dropna(subset=[sid_col])
        n = len(msub)
        towers = msub["SITE_ID"].nunique()
        rows.append(dict(MONTH=m, N=n, N_towers=towers, median=msub[sid_col].median() if n else np.nan,
                          reliable=reliable(n, towers)))
    return pd.DataFrame(rows)


def build_seasonal_hemisphere_figure(df: pd.DataFrame, hemisphere: str, summer_months: list[int]):
    label = "Northern Hemisphere" if hemisphere == "NH" else "Southern Hemisphere"
    print(f"Building Figure: seasonal SID climatology, {label} ...")
    sub_hemi = df[df["HEMISPHERE"] == hemisphere]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    for ax, stressor in zip(axes, STRESS_ORDER):
        sid_col = STRESSORS[stressor]
        for m in summer_months:
            ax.axvspan(m - 0.5, m + 0.5, color="#FDECEC", zorder=0)

        for prod in PRODUCTS:
            msum = monthly_stats(sub_hemi[sub_hemi["SIF_PRODUCT"] == prod], sid_col)
            ax.plot(msum["MONTH"], msum["median"], color=PRODUCT_COLORS[prod], linewidth=1.1, alpha=0.55, zorder=3)
            rel_pts = msum[msum["reliable"]]
            unrel_pts = msum[~msum["reliable"]]
            ax.scatter(rel_pts["MONTH"], rel_pts["median"], color=PRODUCT_COLORS[prod], s=22, zorder=4)
            ax.scatter(unrel_pts["MONTH"], unrel_pts["median"], facecolor="none",
                       edgecolor=PRODUCT_COLORS[prod], s=22, linewidth=0.9, zorder=4)

        msum_all = monthly_stats(sub_hemi, sid_col)
        ax.plot(msum_all["MONTH"], msum_all["median"], color="black", linewidth=2.4, zorder=5, label="ALL (pooled)")
        rel_pts = msum_all[msum_all["reliable"]]
        unrel_pts = msum_all[~msum_all["reliable"]]
        ax.scatter(rel_pts["MONTH"], rel_pts["median"], color="black", s=42, zorder=6)
        ax.scatter(unrel_pts["MONTH"], unrel_pts["median"], facecolor="none", edgecolor="black",
                   s=42, linewidth=1.3, zorder=6)

        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(MONTH_NAMES, fontsize=8)
        ax.set_title(f"{stressor} stress", fontsize=11.5, fontweight="bold", loc="left")
        ax.set_ylabel(f"Median SID_{stressor}", fontsize=8.8)
        ax.grid(axis="y", color="#EEEEEE", linewidth=0.6, zorder=0)
        ax.set_ylim(bottom=0)

    handles = [Line2D([0], [0], color=PRODUCT_COLORS[p], marker="o", markersize=5, label=p) for p in PRODUCTS]
    handles.append(Line2D([0], [0], color="black", linewidth=2.4, marker="o", markersize=6, label="ALL (pooled)"))
    handles.append(Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none",
                           markeredgecolor="#555555", markersize=6, label=f"Unreliable month (N<{REL_MIN_N} or <{REL_MIN_TOWERS} towers, hollow marker)"))
    handles.append(mpatches.Patch(facecolor="#FDECEC", label="Local meteorological summer (approx.)"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.06), frameon=False)

    n_towers = sub_hemi["SITE_ID"].nunique()
    season_word = "Jun-Aug" if hemisphere == "NH" else "Dec-Feb"
    fig.suptitle(
        wrap_text(
            f"Calendar-month climatology of Stress-Induced Decoupling -- {label} ({n_towers} towers)\n"
            f"Shaded band = approximate local summer ({season_word}). Descriptive cross-site pattern, not a causal claim; "
            f"{'reversed' if hemisphere == 'SH' else 'standard'} seasonal cycle relative to the calendar year.",
            width=140),
        fontsize=12.3, fontweight="bold", y=1.08,
    )
    fig.text(0.5, -0.14,
              wrap_text(
                  "Hollow markers/points below the reliability threshold (N>=12 site-months AND N>=3 towers) are shown for "
                  "continuity of the line but should not be over-interpreted. Source: 03_sid_monthly/monthly_SID_master.csv.",
                  width=170),
              ha="center", fontsize=6.8, color="#555555")
    fig.subplots_adjust(wspace=0.28, top=0.80, bottom=0.22)
    save_figure(fig, f"SID_Figure_Seasonal_{hemisphere}", OUT_BH)
    return {stressor: monthly_stats(sub_hemi, STRESSORS[stressor]) for stressor in STRESS_ORDER}


def build_seasonal_biome_figures(df: pd.DataFrame, qualifying_geo: dict[str, list[str]]):
    for biome, rel_stressors in qualifying_geo.items():
        print(f"Building Figure: seasonal SID climatology, biome {biome} ...")
        sub_biome = df[df["IGBP"] == biome]
        # NH/SH mixed within a biome would blend opposite seasons; report the dominant
        # hemisphere for that biome and note if towers span both
        hemi_counts = sub_biome.groupby("HEMISPHERE")["SITE_ID"].nunique()
        dominant_hemi = hemi_counts.idxmax()
        mixed = len(hemi_counts) > 1
        summer_months = [6, 7, 8] if dominant_hemi == "NH" else [12, 1, 2]

        fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
        for ax, stressor in zip(axes, STRESS_ORDER):
            sid_col = STRESSORS[stressor]
            for m in summer_months:
                ax.axvspan(m - 0.5, m + 0.5, color="#FDECEC", zorder=0)
            msum = monthly_stats(sub_biome, sid_col)
            is_reliable_stressor = stressor in rel_stressors
            line_color = STRESSOR_COLORS[stressor]
            ax.plot(msum["MONTH"], msum["median"], color=line_color, linewidth=1.8,
                    alpha=0.9 if is_reliable_stressor else 0.4, zorder=3)
            rel_pts = msum[msum["reliable"]]
            unrel_pts = msum[~msum["reliable"]]
            ax.scatter(rel_pts["MONTH"], rel_pts["median"], color=line_color, s=34, zorder=4)
            ax.scatter(unrel_pts["MONTH"], unrel_pts["median"], facecolor="none", edgecolor=line_color,
                       s=34, linewidth=1.0, zorder=4)
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(MONTH_NAMES, fontsize=8)
            ax.set_title(f"{stressor} stress" + ("" if is_reliable_stressor else "  [biome-level below threshold]"),
                         fontsize=10.8, fontweight="bold", loc="left",
                         color="black" if is_reliable_stressor else "#888888")
            ax.set_ylabel(f"Median SID_{stressor}", fontsize=8.5)
            ax.grid(axis="y", color="#EEEEEE", linewidth=0.6, zorder=0)
            ax.set_ylim(bottom=0)

        handles = [Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none",
                          markeredgecolor="#555555", markersize=6,
                          label=f"Unreliable month (N<{REL_MIN_N} or <{REL_MIN_TOWERS} towers, hollow marker)"),
                   mpatches.Patch(facecolor="#FDECEC", label="Approx. local summer for dominant hemisphere")]
        fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.06), frameon=False)

        n_towers = sub_biome["SITE_ID"].nunique()
        mixed_note = (" Towers span both hemispheres; summer shading follows the majority-hemisphere convention only."
                      if mixed else "")
        fig.suptitle(
            wrap_text(
                f"Calendar-month climatology of Stress-Induced Decoupling -- {biome} ({IGBP_LABELS.get(biome, biome)}, "
                f"{n_towers} towers, pooled across products)\n"
                f"Shaded band = approx. local summer for the dominant hemisphere ({dominant_hemi}).{mixed_note}",
                width=145),
            fontsize=12, fontweight="bold", y=1.08,
        )
        fig.subplots_adjust(wspace=0.28, top=0.80, bottom=0.20)
        save_figure(fig, f"SID_Figure_Seasonal_Biome_{biome}", OUT_BH)


if __name__ == "__main__":
    df, biome_summary, hemi_summary, biomes_present = main_summaries()
    build_biome_heatmap(biome_summary, biomes_present)
    build_biome_stressor_comparison(biome_summary, biomes_present)

    print("Loading basemap (decimated read of NE1_HR_LC_SR_W.tif) ...")
    import time
    t0 = time.time()
    basemap_rgb, basemap_extent = load_basemap(target_width_px=6000, white_blend=0.32)
    print(f"  basemap loaded in {time.time() - t0:.1f}s")

    qualifying_geo = build_biome_geographic_maps(df, biome_summary, biomes_present, basemap_rgb, basemap_extent)

    build_seasonal_hemisphere_figure(df, "NH", summer_months=[6, 7, 8])
    build_seasonal_hemisphere_figure(df, "SH", summer_months=[12, 1, 2])
    build_seasonal_biome_figures(df, qualifying_geo)

    print("\nAll done.")
