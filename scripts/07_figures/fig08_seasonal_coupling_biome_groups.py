"""
reviewer Update: split the old combined Figure 4 (main seasonal-structure grid +
IGBP-stratified panel) into two standalone figures, and switch the "coupling"
statistic shown in the correlation row/curves from Pearson r to Spearman rho.

Outputs
-------
01_main_revised/Figure_Seasonal_Structure_Main_4product.{png,svg,pdf}
    Figure A: 4-column (GOSIF/GOME-2/TROPOSIF/OCO-3) x 3-row (SIF, GPP,
    Spearman rho) grid. Self-contained title/legend/caption.
01_main_revised/Figure_IGBP_Stratified_Seasonal_4product.{png,svg,pdf}
    Figure B: IGBP-stratified monthly Spearman rho (GOSIF, GOME-2 only;
    Forest/Grassland/Cropland). Self-contained title/legend/caption.
Figure4_split_PROVENANCE.md
    Short note on what was reused vs recomputed.

Data sources
------------
- SIF_EC_SIF_GPP_Relationship_Analysis/05_figure_data/Figure4_Temporal_Structure.csv
  This is the exact underlying data for the reference figure at
  All_the_Visualizations_REVISED_4PRODUCT_FINAL/03_recreated_figures_OCO2_removed_only/
  Figure4_Temporal_Seasonal_Structure_4product.png (verified: per-product N sums
  reproduce the "n=11,487 / 6,089 / 1,540 / 659 site-mo." annotations on that
  figure exactly, and PEARSON_R values reproduce the plotted r-curves exactly).
  It already carries a precomputed SPEARMAN_RHO / SPEARMAN_P column for the
  per-product-monthly panel (5 products; OCO-2 filtered out here) -> REUSED
  directly, no recomputation needed for Figure A.
  It does NOT carry Spearman for the igbp_stratified_monthly panel (Pearson
  only) -> Spearman recomputed for Figure B (see below).
- SIF_EC_SIF_GPP_Relationship_Analysis/01_analysis_ready/SIF_GPP_Analysis_Ready_Master.csv
  Site-month master table. Used only to recompute Spearman rho/p for the
  IGBP x product x month cells needed by Figure B. Methodology validated
  against the existing CSV before use: filtering to
  (SIF_PRODUCT, MONTH) with SIF_VALUE/GPP_VALUE dropna and computing
  scipy.stats.pearsonr reproduces the existing PEARSON_R/PEARSON_P values in
  Figure4_Temporal_Structure.csv exactly (both the per-product-monthly panel,
  e.g. GOSIF/Jan: r=0.9201529368331692, and the igbp-stratified panel, e.g.
  GOME-2/Cropland/Jan: r=0.1055242820936575) -> confirms the same filter/
  grouping is being replicated for the new Spearman numbers.

Colors / styling reused from the reference figure's own SVG (parsed with
lxml) rather than a different script's palette, since this is the direct
visual sibling of the figure being split:
  PRODUCT_COLORS: GOSIF #440154, GOME-2 #3E4989, TROPOSIF #26828E, OCO-3 #35B779
  IGBP_COLORS:    Forest #482878, Grassland #31688E, Cropland #6DCD59
"""

from __future__ import annotations
# Original script filename (sanitized): build_figure4_split_spearman_v2.py; public name: fig08_seasonal_coupling_biome_groups.py
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

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = (Path(PROJECT_ROOT) / 'All_the_Visualizations_REVISED_4PRODUCT_UPDATE_V2/07_scripts/build_figure4_split_spearman_v2.py').resolve().parents[2]
OUT_ROOT = ROOT / "All_the_Visualizations_REVISED_4PRODUCT_UPDATE_V2"
OUT_DIR = OUT_ROOT / "01_main_revised"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIG4_CSV = ROOT / "SIF_EC_SIF_GPP_Relationship_Analysis" / "05_figure_data" / "Figure4_Temporal_Structure.csv"
MASTER_CSV = ROOT / "SIF_EC_SIF_GPP_Relationship_Analysis" / "01_analysis_ready" / "SIF_GPP_Analysis_Ready_Master.csv"

# ---------------------------------------------------------------------------
# Style (reused verbatim from the reference figure's own SVG)
# ---------------------------------------------------------------------------
PRODUCTS = ["GOSIF", "GOME-2", "TROPOSIF", "OCO-3"]
PRODUCT_COLORS = {
    "GOSIF": "#440154",
    "GOME-2": "#3E4989",
    "TROPOSIF": "#26828E",
    "OCO-3": "#35B779",
}
IGBP_GROUP_ORDER = ["Forest", "Grassland", "Cropland"]
IGBP_COLORS = {
    "Forest": "#482878",
    "Grassland": "#31688E",
    "Cropland": "#6DCD59",
}
FOREST_CLASSES = {"ENF", "EBF", "DBF", "MF"}
IGBP_PRODUCTS = ["GOSIF", "GOME-2"]  # broadest coverage products only, matching reference figure
MIN_SITE_MONTHS = 15
MIN_SITES = 3

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

FONT = "Arial"
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [FONT, "DejaVu Sans", "Liberation Sans"],
        "axes.titlesize": 15,
        "axes.labelsize": 13.5,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "legend.fontsize": 12,
        "figure.dpi": 120,
        "savefig.dpi": 600,
        "axes.linewidth": 0.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def igbp_group(igbp: str) -> str | None:
    if igbp in FOREST_CLASSES:
        return "Forest"
    if igbp == "GRA":
        return "Grassland"
    if igbp == "CRO":
        return "Cropland"
    return None


# ---------------------------------------------------------------------------
# Load & prep: Figure A data (REUSED from existing figure-data CSV)
# ---------------------------------------------------------------------------
fig4 = pd.read_csv(FIG4_CSV)
per_prod = fig4[fig4["PANEL"] == "per_product_monthly"].copy()
numeric_cols = ["MONTH", "N", "N_SITES", "SIF_MEDIAN", "SIF_IQR", "GPP_MEDIAN", "GPP_IQR",
                 "PEARSON_R", "PEARSON_P", "SPEARMAN_RHO", "SPEARMAN_P"]
for c in numeric_cols:
    per_prod[c] = pd.to_numeric(per_prod[c], errors="coerce")
per_prod = per_prod[per_prod["SIF_PRODUCT"].isin(PRODUCTS)].sort_values(["SIF_PRODUCT", "MONTH"])

n_sitemonths_total = per_prod.groupby("SIF_PRODUCT")["N"].sum().to_dict()
n_sites_max_overall = per_prod["N_SITES"].max()  # 73 (GOSIF) -- same normalization reference used

# sanity check against the reference figure's printed n= annotations
_expected_n = {"GOSIF": 11487, "GOME-2": 6089, "TROPOSIF": 1540, "OCO-3": 659}
for p, n in _expected_n.items():
    assert n_sitemonths_total[p] == n, f"N mismatch for {p}: {n_sitemonths_total[p]} != {n}"

# ---------------------------------------------------------------------------
# Load & recompute: Figure B data (Spearman rho recomputed from master;
# median SIF/GPP/N/N_SITES validated to match existing Pearson panel exactly)
# ---------------------------------------------------------------------------
master = pd.read_csv(MASTER_CSV, low_memory=False)
master["IGBP_GROUP"] = master["IGBP_GPP_TABLE"].apply(igbp_group)
master = master.dropna(subset=["SIF_VALUE", "GPP_VALUE"])

igbp_rows = []
for prod in IGBP_PRODUCTS:
    for grp in IGBP_GROUP_ORDER:
        for month in range(1, 13):
            sub = master[
                (master["SIF_PRODUCT"] == prod)
                & (master["IGBP_GROUP"] == grp)
                & (master["MONTH"] == month)
            ]
            n = len(sub)
            n_sites = sub["SITE_ID"].nunique()
            if n < MIN_SITE_MONTHS or n_sites < MIN_SITES:
                continue
            rho, p_rho = stats.spearmanr(sub["SIF_VALUE"], sub["GPP_VALUE"])
            r_pear, p_pear = stats.pearsonr(sub["SIF_VALUE"], sub["GPP_VALUE"])
            igbp_rows.append(
                dict(SIF_PRODUCT=prod, IGBP_GROUP=grp, MONTH=month, N=n, N_SITES=n_sites,
                     SPEARMAN_RHO=rho, SPEARMAN_P=p_rho, PEARSON_R=r_pear, PEARSON_P=p_pear)
            )
igbp_df = pd.DataFrame(igbp_rows)

# Methodology cross-check against the existing (Pearson-only) igbp_stratified_monthly
# panel embedded lower in the legacy CSV (rows below the per_product_monthly table,
# with its own header) was done interactively during development -- e.g.
# GOME-2/Cropland/Jan: N=92, N_SITES=9, PEARSON_R=0.1055242820936575 reproduced
# exactly by this same filter+groupby. See Figure4_split_PROVENANCE.md.

n_sites_max_igbp = igbp_df["N_SITES"].max()  # 27 (GOSIF Forest)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def marker_size(n_sites, n_max, lo=28, hi=170):
    frac = np.sqrt(np.clip(n_sites / n_max, 0, 1))
    return lo + frac * (hi - lo)


def marker_alpha(n_sites, n_max, lo=0.45, hi=1.0):
    frac = np.sqrt(np.clip(n_sites / n_max, 0, 1))
    return lo + frac * (hi - lo)


# ===========================================================================
# FIGURE A -- main seasonal-structure grid (SIF, GPP, Spearman rho)
# ===========================================================================
fig_a = plt.figure(figsize=(20, 14.5))
gs = fig_a.add_gridspec(
    nrows=3, ncols=4, left=0.055, right=0.985, top=0.79, bottom=0.32,
    hspace=0.16, wspace=0.30, height_ratios=[1, 1, 1],
)

gpp_ymax = 26
months = np.arange(1, 13)

for j, product in enumerate(PRODUCTS):
    color = PRODUCT_COLORS[product]
    sub = per_prod[per_prod["SIF_PRODUCT"] == product].set_index("MONTH").reindex(months)

    # --- Row 0: SIF median +/- IQR/2 ---
    ax = fig_a.add_subplot(gs[0, j])
    lo = sub["SIF_MEDIAN"] - sub["SIF_IQR"] / 2
    hi = sub["SIF_MEDIAN"] + sub["SIF_IQR"] / 2
    ax.fill_between(months, lo, hi, color=color, alpha=0.18, linewidth=0)
    ax.plot(months, sub["SIF_MEDIAN"], "-o", color=color, lw=2.0, ms=5.5, mec="white", mew=0.6)
    ax.set_ylim(bottom=0)
    ax.set_xlim(0.5, 12.5)
    ax.set_xticks(months)
    ax.set_xticklabels([])
    if j == 0:
        ax.set_ylabel("SIF\n(median ± IQR)")
    # Header stack (all in axes-fraction units so vertical order/spacing is deterministic,
    # regardless of the axes' actual pixel height): line (topmost) > product name > n= subtitle.
    ax.plot([0.06, 0.94], [1.48, 1.48], transform=ax.transAxes, color=color, lw=2.4,
            clip_on=False, solid_capstyle="butt")
    ax.text(0.5, 1.30, product, transform=ax.transAxes, ha="center", va="bottom",
            color=color, fontsize=17, fontweight="bold")
    ax.text(0.5, 1.12, f"n={n_sitemonths_total[product]:,.0f} site-mo.", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=11.5, color="#333333")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", lw=0.7, zorder=0)

    # --- Row 1: GPP median +/- IQR/2 (shared y-axis across products) ---
    ax = fig_a.add_subplot(gs[1, j])
    lo = sub["GPP_MEDIAN"] - sub["GPP_IQR"] / 2
    hi = sub["GPP_MEDIAN"] + sub["GPP_IQR"] / 2
    ax.fill_between(months, lo, hi, color=color, alpha=0.18, linewidth=0)
    ax.plot(months, sub["GPP_MEDIAN"], "--D", color=color, lw=2.0, ms=5.5, mec="white", mew=0.6)
    ax.set_ylim(0, gpp_ymax)
    ax.set_xlim(0.5, 12.5)
    ax.set_xticks(months)
    ax.set_xticklabels([])
    if j == 0:
        ax.set_ylabel("GPP\n($\\mu$mol CO$_2$ m$^{-2}$ s$^{-1}$)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", lw=0.7, zorder=0)

    # --- Row 2: Spearman rho (SIF vs GPP), by month -- SWITCHED FROM PEARSON R ---
    ax = fig_a.add_subplot(gs[2, j])
    rho = sub["SPEARMAN_RHO"].to_numpy()
    p_rho = sub["SPEARMAN_P"].to_numpy()
    nsit = sub["N_SITES"].to_numpy()
    sizes = marker_size(nsit, n_sites_max_overall)
    alphas = marker_alpha(nsit, n_sites_max_overall)
    ax.plot(months, rho, "-", color=color, lw=1.8, zorder=2)
    sig = p_rho < 0.05
    if sig.any():
        ax.scatter(months[sig], rho[sig], s=sizes[sig], color=color, alpha=alphas[sig],
                   edgecolor="white", linewidth=0.6, zorder=3)
    if (~sig).any():
        ax.scatter(months[~sig], rho[~sig], s=sizes[~sig], facecolor="white", edgecolor=color,
                   linewidth=1.6, alpha=alphas[~sig], zorder=3)
    ax.axhline(0, color="#bbbbbb", lw=0.8, zorder=1)
    ax.set_ylim(-0.5, 1.0)
    ax.set_xlim(0.5, 12.5)
    ax.set_xticks(months)
    ax.set_xticklabels(MONTH_LABELS, rotation=90)
    if j == 0:
        ax.set_ylabel("Spearman $\\rho$\n(SIF vs GPP)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", lw=0.7, zorder=0)

fig_a.suptitle(
    "Seasonal structure of the SIF–GPP relationship across four SIF products",
    fontsize=21, fontweight="bold", x=0.055, ha="left", y=0.965,
)
fig_a.text(
    0.055, 0.925,
    "Coupling statistic is Spearman $\\rho$ (rank correlation), not Pearson r.",
    fontsize=12.5, style="italic", color="#444444", ha="left",
)

# --- Legend (own, self-contained) ---
legend_elems = [
    Line2D([0], [0], color="#555555", lw=2.0, marker="o", ms=6, mec="white", label="SIF (median, solid)"),
    Line2D([0], [0], color="#555555", lw=2.0, ls="--", marker="D", ms=6, mec="white", label="GPP (median, dashed)"),
    Patch(facecolor="#555555", alpha=0.18, label="IQR band"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#555555", markeredgecolor="white",
           ms=9, label="Spearman $p$ < 0.05 (filled)"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#555555",
           mew=1.6, ms=9, label="Spearman $p$ ≥ 0.05 (open)"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#999999", markeredgecolor="none",
           ms=5, label="marker size / alpha $\\propto \\sqrt{N_{sites}}$"),
]
fig_a.legend(handles=legend_elems, loc="lower center", ncol=6, frameon=False,
             bbox_to_anchor=(0.5, 0.225), fontsize=12.5, columnspacing=1.6, handletextpad=0.6)

caption_a = (
    "Each column is one SIF product (GOSIF, GOME-2, TROPOSIF, OCO-3). Top row: monthly median SIF with the interquartile range shaded; y-axis independent per\n"
    "column because raw SIF magnitude/units differ by product. Middle row: monthly median GPP ($\\mu$mol CO$_2$ m$^{-2}$ s$^{-1}$), IQR shaded; y-axis shared across all\n"
    "four columns (GPP units are common). Bottom row: monthly Spearman $\\rho$ between site-month SIF and GPP values, in the product's color. Filled markers = months\n"
    "with Spearman $p$ < 0.05; open = not significant. Marker size/opacity scale with $\\sqrt{N_{sites}/73}$ (73 = largest monthly site count, GOSIF)."
)
fig_a.text(0.055, 0.03, caption_a, fontsize=10.5, color="#222222", ha="left", va="bottom", linespacing=1.6)

for ext in ("png", "svg", "pdf"):
    out_path = OUT_DIR / f"Figure_Seasonal_Structure_Main_4product.{ext}"
    fig_a.savefig(out_path, dpi=600 if ext == "png" else None)
plt.close(fig_a)

# ===========================================================================
# FIGURE B -- IGBP-stratified seasonal Spearman rho (GOSIF, GOME-2 only)
# ===========================================================================
fig_b, ax = plt.subplots(figsize=(16, 12.5))
fig_b.subplots_adjust(left=0.08, right=0.97, top=0.80, bottom=0.45)

product_style = {
    "GOSIF": dict(ls="-", marker="o", label="GOSIF (solid, circle)"),
    "GOME-2": dict(ls="--", marker="s", label="GOME-2 (dashed, square)"),
}

for prod in IGBP_PRODUCTS:
    style = product_style[prod]
    for grp in IGBP_GROUP_ORDER:
        color = IGBP_COLORS[grp]
        sub = igbp_df[(igbp_df["SIF_PRODUCT"] == prod) & (igbp_df["IGBP_GROUP"] == grp)].sort_values("MONTH")
        if sub.empty:
            continue
        sub_full = sub.set_index("MONTH").reindex(months)
        rho = sub_full["SPEARMAN_RHO"].to_numpy()
        p_rho = sub_full["SPEARMAN_P"].to_numpy()
        nsit = sub_full["N_SITES"].to_numpy()
        sizes = marker_size(nsit, n_sites_max_igbp, lo=45, hi=260)
        ax.plot(months, rho, ls=style["ls"], color=color, lw=2.2, zorder=2)
        valid = ~np.isnan(rho)
        sig = valid & (p_rho < 0.05)
        nsig = valid & ~(p_rho < 0.05)
        if sig.any():
            ax.scatter(months[sig], rho[sig], s=sizes[sig], marker=style["marker"], color=color,
                       edgecolor="white", linewidth=0.7, zorder=3)
        if nsig.any():
            ax.scatter(months[nsig], rho[nsig], s=sizes[nsig], marker=style["marker"], facecolor="white",
                       edgecolor=color, linewidth=1.8, zorder=3)

ax.axhline(0, color="#bbbbbb", lw=0.9, zorder=1)
ax.set_ylim(-0.15, 1.0)
ax.set_xlim(0.5, 12.5)
ax.set_xticks(months)
ax.set_xticklabels(MONTH_LABELS, fontsize=13)
ax.set_ylabel("Spearman $\\rho$ (SIF vs GPP) by IGBP group", fontsize=14.5)
ax.set_xlabel("Month", fontsize=14.5)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e6e6e6", lw=0.8, zorder=0)

fig_b.suptitle(
    "IGBP-stratified seasonal SIF–GPP coupling (Spearman $\\rho$)",
    fontsize=21, fontweight="bold", x=0.08, ha="left", y=0.975,
)
fig_b.text(
    0.08, 0.925,
    "GOSIF, GOME-2 only — the two products with the broadest temporal and spatial coverage.\n"
    f"Reliability threshold: ≥{MIN_SITE_MONTHS} site-months and ≥{MIN_SITES} sites per (product × biome × month) cell; failing cells are omitted (gap in line).\n"
    "Coupling statistic is Spearman $\\rho$ (rank correlation), not Pearson r.",
    fontsize=12.5, style="italic", color="#444444", ha="left", va="top", linespacing=1.7,
)

# --- Legend: one combined, single legend box (avoids sub-legend bbox collisions) ---
color_handles = [Patch(facecolor=IGBP_COLORS[g], label=f"IGBP group: {g}") for g in IGBP_GROUP_ORDER]
style_handles = [
    Line2D([0], [0], color="#555555", lw=2.2, ls=product_style[p]["ls"], marker=product_style[p]["marker"],
           ms=9, mec="white", label=product_style[p]["label"])
    for p in IGBP_PRODUCTS
]
sig_handles = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#555555", markeredgecolor="white",
           ms=10, label="Spearman $p$ < 0.05 (filled)"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#555555",
           mew=1.8, ms=10, label="Spearman $p$ ≥ 0.05 (open)"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#999999", markeredgecolor="none",
           ms=6, label="marker size $\\propto \\sqrt{N_{sites}}$"),
]
all_handles = color_handles + style_handles + sig_handles
fig_b.legend(handles=all_handles, loc="lower center", bbox_to_anchor=(0.5, 0.30), frameon=False,
             fontsize=13, ncol=4, handletextpad=0.7, columnspacing=1.8, labelspacing=1.1)

caption_b = (
    "Monthly Spearman $\\rho$ between site-month SIF and GPP, stratified by IGBP land-cover group: Forest = ENF+EBF+DBF+MF combined,\n"
    "Grassland = GRA, Cropland = CRO (pooled site-months across all contributing sites, per product). Total product coverage (all biome\n"
    "groups combined, from the companion main-grid figure): GOSIF n=11,487 site-mo., GOME-2 n=6,089 site-mo. Land-cover identity visibly\n"
    "modulates both the strength and seasonal shape of the SIF-GPP coupling within each product."
)
fig_b.text(0.08, 0.22, caption_b, fontsize=11, color="#222222", ha="left", va="top", linespacing=1.7)

for ext in ("png", "svg", "pdf"):
    out_path = OUT_DIR / f"Figure_IGBP_Stratified_Seasonal_4product.{ext}"
    fig_b.savefig(out_path, dpi=600 if ext == "png" else None)
plt.close(fig_b)

print("Figure A saved to:", OUT_DIR / "Figure_Seasonal_Structure_Main_4product.png")
print("Figure B saved to:", OUT_DIR / "Figure_IGBP_Stratified_Seasonal_4product.png")
print("n_sitemonths_total:", n_sitemonths_total)
print("n_sites_max_overall:", n_sites_max_overall, "n_sites_max_igbp:", n_sites_max_igbp)
print("igbp_df rows:", len(igbp_df))
