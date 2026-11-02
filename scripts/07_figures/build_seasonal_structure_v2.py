"""
Build the REVISED (v2) seasonal-structure figures for Northern/Southern Hemisphere
sites, main-paper products only (GOSIF, GOME-2, TROPOSIF, OCO-3; OCO-2 excluded).

This redesigns the 4-col (product) x 3-row (SIF median+-IQR / GPP median+-IQR /
pooled monthly Spearman rho) figures at
    All_the_Visualizations_REVISED_4PRODUCT_UPDATE/03_biome_hemisphere_figures/
        Seasonal_Structure_NorthernHemisphere.png
        Seasonal_Structure_SouthernHemisphere.png
(read-only references; NOT modified by this script) into a 4-col x 2-row layout:

  Row A (was rows A+B): SIF and GPP shown TOGETHER per product panel on a dual
      y-axis -- LEFT axis = GPP (median +/- IQR), RIGHT axis = SIF (median +/-
      IQR) [axis sides intentionally reversed from the naive default, per
      design spec]. Both uncertainty bands are attempted (GPP = solid
      translucent fill; SIF = lighter hatched fill so the two remain
      distinguishable even where they overlap in on-screen vertical space).

  Row B (was row C): the original pooled/summary monthly Spearman rho line
      (reliability-flagged exactly as before: solid/dashed curve, filled/open
      marker) with an ADDED boxplot of the DISTRIBUTION of site-level
      (per-tower) Spearman rho values for that calendar month, so the reader
      sees both the pooled trend and the underlying across-tower spread.
      Months that fail the project reliability threshold (N>=12 site-months
      AND N>=3 towers, evaluated on the pooled master dataset, same
      convention as the original figure and biome_hemisphere_sample_size_report)
      get a hatched/faded box in addition to the existing open-marker /
      dashed-curve flagging on the pooled line -- never a silently-hidden or
      equal-confidence presentation.

Site-level rho computation (documented here and restated in the figure
caption): for each SIF product, each calendar month, and each EC tower, the
harmonic-residual anomaly table

    SIF_EC_SIF_GPP_Final_Verification/02_independent_reverification/
        harmonic_residuals_full.csv   (SITE_ID, SIF_PRODUCT, MONTH,
                                               YEAR, SIF_ANOM, GPP_ANOM)

is grouped by (SIF_PRODUCT, SITE_ID, MONTH) and a single Spearman rho is
computed BETWEEN THAT TOWER'S SIF_ANOM AND GPP_ANOM ACROSS YEARS (i.e. one
rho per tower per calendar month, using the tower's own interannual
variability in that month -- pooling all of a tower's Junes together, all its
Julys together, etc.). This is the interpretation that is readily comparable
to the pooled monthly-rho line: the pooled line already correlates SIF vs GPP
within one calendar month pooling across ALL towers and ALL years at once; the
box decomposes that same pooled quantity into its one-rho-per-tower building
blocks (computed on the anomaly series so that each tower's own within-month,
across-year covariation is isolated). A tower contributes to a given month's
box only if it has >=3 distinct years of data for that month (the practical
minimum for a Spearman rho to reflect more than a coin flip) and non-degenerate
(non-constant) anomaly series; below that it is neither fabricated nor hidden
-- for months with 1-2 qualifying towers those individual rho values are
plotted as open points instead of a box, and for months with 0 qualifying
towers nothing is drawn for that month's distribution (pooled line point is
still shown/flagged as before).

Output (this script only writes here; upstream folders are read-only):
    All_the_Visualizations_REVISED_4PRODUCT_FINAL_ARIDITY_JUNE_DIAGNOSTICS/
        04_revised_seasonal_figures/
            Seasonal_Structure_NorthernHemisphere_v2.{png,svg,pdf}
            Seasonal_Structure_SouthernHemisphere_v2.{png,svg,pdf}
            seasonal_structure_v2_sample_size_report.csv
"""

from __future__ import annotations
# Original script filename (sanitized): build_seasonal_structure_v2.py; public name: build_seasonal_structure_v2.py
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
from scipy.stats import spearmanr

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = (Path(PROJECT_ROOT) / 'All_the_Visualizations_REVISED_4PRODUCT_FINAL_ARIDITY_JUNE_DIAGNOSTICS/10_scripts/build_seasonal_structure_v2.py').resolve().parents[2]
MASTER_DATA_PATH = ROOT / "SIF_EC_SIF_GPP_Relationship_Analysis" / "01_analysis_ready" / "SIF_GPP_Analysis_Ready_Master.csv"
HARMONIC_DATA_PATH = ROOT / "SIF_EC_SIF_GPP_Final_Verification" / "02_independent_reverification" / "harmonic_residuals_full.csv"
OUT_DIR = ROOT / "All_the_Visualizations_REVISED_4PRODUCT_FINAL_ARIDITY_JUNE_DIAGNOSTICS" / "04_revised_seasonal_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Constants -- product set + colors matched to
# All_the_Visualizations_REVISED_4PRODUCT_FINAL/06_scripts/build_new_geospatial_visualizations.py
# (colorblind-safe Okabe-Ito style palette used across the FINAL/ARIDITY_JUNE_DIAGNOSTICS
# lineage of this project).
# --------------------------------------------------------------------------
PRODUCTS = ["GOSIF", "GOME-2", "TROPOSIF", "OCO-3"]
PRODUCT_COLORS = {
    "GOSIF": "#0072B2",     # blue
    "GOME-2": "#E69F00",    # orange
    "TROPOSIF": "#009E73",  # bluish green
    "OCO-3": "#CC79A7",     # reddish purple
}
PRODUCT_MARKERS = {"GOSIF": "o", "GOME-2": "s", "TROPOSIF": "^", "OCO-3": "D"}

N_MIN = 12         # site-months, pooled-cell reliability threshold (project standard)
SITES_MIN = 3      # independent EC towers, pooled-cell reliability threshold
MIN_YEARS_TOWER_RHO = 3   # min distinct years for a tower's own across-year rho (this figure only)
MIN_TOWERS_FOR_BOX = 3    # min qualifying towers to draw a box; below -> open points; 0 -> nothing

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

RSE_FONT = "Arial"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [RSE_FONT, "DejaVu Sans", "Liberation Sans"],
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.3,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7.2,
    "figure.dpi": 120,
    "savefig.dpi": 600,
    "axes.linewidth": 0.65,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


# --------------------------------------------------------------------------
# Data loading -- master (pooled SIF/GPP values + pooled reliability), reused
# directly from All_the_Visualizations_REVISED_4PRODUCT_UPDATE/07_scripts/
# build_biome_hemisphere_figures.py::load_data() / monthly_stats().
# --------------------------------------------------------------------------
def load_master_data() -> pd.DataFrame:
    df = pd.read_csv(MASTER_DATA_PATH, low_memory=False)
    df = df[df["SIF_PRODUCT"].isin(PRODUCTS)].copy()
    df = df.dropna(subset=["IGBP_GPP_TABLE", "LATITUDE", "SIF_VALUE", "GPP_VALUE", "MONTH"])
    df["IGBP_GPP_TABLE"] = df["IGBP_GPP_TABLE"].astype(str)
    df["MONTH"] = df["MONTH"].astype(int)

    site_lat = df.groupby("SITE_ID")["LATITUDE"].first()
    df["HEMI"] = df["SITE_ID"].map(lambda s: "NH" if site_lat[s] >= 0 else "SH")
    return df


def monthly_stats(sub: pd.DataFrame) -> pd.DataFrame:
    """Per (SIF_PRODUCT, MONTH): pooled Spearman rho(SIF,GPP), N, N_SITES, PASS,
    plus SIF/GPP median and IQR. Identical logic to the original build script."""
    rows = []
    for (product, month), g in sub.groupby(["SIF_PRODUCT", "MONTH"]):
        n = len(g)
        n_sites = g["SITE_ID"].nunique()
        if n >= 2 and g["SIF_VALUE"].nunique() > 1 and g["GPP_VALUE"].nunique() > 1:
            rho, pval = spearmanr(g["SIF_VALUE"], g["GPP_VALUE"])
        else:
            rho, pval = np.nan, np.nan
        rows.append({
            "SIF_PRODUCT": product, "MONTH": month, "N": n, "N_SITES": n_sites,
            "SPEARMAN_RHO": rho, "PVAL": pval,
            "PASS": (n >= N_MIN) and (n_sites >= SITES_MIN),
            "SIF_MEDIAN": g["SIF_VALUE"].median(),
            "SIF_Q25": g["SIF_VALUE"].quantile(0.25),
            "SIF_Q75": g["SIF_VALUE"].quantile(0.75),
            "GPP_MEDIAN": g["GPP_VALUE"].median(),
            "GPP_Q25": g["GPP_VALUE"].quantile(0.25),
            "GPP_Q75": g["GPP_VALUE"].quantile(0.75),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Data loading -- harmonic-residual anomaly table, for the NEW site-level
# (per-tower) monthly rho distribution.
# --------------------------------------------------------------------------
def load_harmonic_data() -> pd.DataFrame:
    h = pd.read_csv(HARMONIC_DATA_PATH, low_memory=False)
    h = h[h["SIF_PRODUCT"].isin(PRODUCTS)].copy()
    h = h.dropna(subset=["SIF_ANOM", "GPP_ANOM", "LATITUDE", "MONTH", "YEAR"])
    h["MONTH"] = h["MONTH"].astype(int)
    site_lat = h.groupby("SITE_ID")["LATITUDE"].first()
    h["HEMI"] = h["SITE_ID"].map(lambda s: "NH" if site_lat[s] >= 0 else "SH")
    return h


def per_tower_monthly_rhos(harm_sub: pd.DataFrame) -> list[float]:
    """harm_sub already filtered to one (product, month, hemisphere) slice.
    Returns one Spearman rho per qualifying tower (>=MIN_YEARS_TOWER_RHO
    distinct years, non-degenerate anomaly series), computed across that
    tower's years within this single calendar month."""
    rhos = []
    for _site, g in harm_sub.groupby("SITE_ID"):
        if g["YEAR"].nunique() < MIN_YEARS_TOWER_RHO:
            continue
        if g["SIF_ANOM"].nunique() <= 1 or g["GPP_ANOM"].nunique() <= 1:
            continue
        rho, _p = spearmanr(g["SIF_ANOM"], g["GPP_ANOM"])
        if not np.isnan(rho):
            rhos.append(rho)
    return rhos


# --------------------------------------------------------------------------
# Row-B (coupling) drawing: pooled rho line (reliability-flagged) + per-tower
# rho box/points overlay (reliability-flagged independently on the box style).
# --------------------------------------------------------------------------
def style_rho_axis(ax):
    ax.axhline(0, color="#888888", lw=0.55, zorder=1)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(0.5, 12.5)
    ax.set_xticks([1, 3, 5, 7, 9, 11])
    ax.set_xticklabels(["Jan", "Mar", "May", "Jul", "Sep", "Nov"])
    ax.grid(color="#e6e6e6", lw=0.35, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


def draw_pooled_rho_line(ax, ms: pd.DataFrame, product: str):
    sub = ms.set_index("MONTH").reindex(range(1, 13))
    if sub["SPEARMAN_RHO"].notna().sum() == 0:
        return
    color = PRODUCT_COLORS[product]
    marker = PRODUCT_MARKERS[product]
    months = np.arange(1, 13)
    y = sub["SPEARMAN_RHO"].values
    any_reliable = bool(sub["PASS"].astype("boolean").fillna(False).any())

    line_style = "-" if any_reliable else "--"
    line_alpha = 1.0 if any_reliable else 0.45
    ax.plot(months, y, line_style, lw=1.5, color=color, alpha=line_alpha, zorder=5)

    for m in months:
        yv = sub.loc[m, "SPEARMAN_RHO"]
        if pd.isna(yv):
            continue
        passed = bool(sub.loc[m, "PASS"])
        if passed:
            ax.plot(m, yv, marker=marker, ms=4.6, mfc=color, mec=color, mew=0.6, alpha=1.0, zorder=6)
        else:
            ax.plot(m, yv, marker=marker, ms=4.0, mfc="white", mec=color, mew=0.9, alpha=0.75, zorder=6)


def draw_site_level_box(ax, harm_all: pd.DataFrame, ms: pd.DataFrame, product: str, hemi: str):
    """Overlay, for each month, a box (n>=3 towers), open points (1-2 towers),
    or nothing (0 towers) showing the distribution of per-tower Spearman rho
    values. Box/point style is faded+hatched for months failing the pooled
    reliability threshold (ms.PASS), matching the pooled-line convention."""
    color = PRODUCT_COLORS[product]
    pass_by_month = ms.set_index("MONTH")["PASS"].reindex(range(1, 13)).astype("boolean").fillna(False)

    box_data, box_positions, box_reliable = [], [], []
    for m in range(1, 13):
        sub = harm_all[(harm_all["SIF_PRODUCT"] == product) & (harm_all["HEMI"] == hemi) & (harm_all["MONTH"] == m)]
        rhos = per_tower_monthly_rhos(sub)
        reliable = bool(pass_by_month.get(m, False))
        if len(rhos) >= MIN_TOWERS_FOR_BOX:
            box_data.append(rhos)
            box_positions.append(m)
            box_reliable.append(reliable)
        elif 1 <= len(rhos) < MIN_TOWERS_FOR_BOX:
            jitter = np.linspace(-0.12, 0.12, len(rhos)) if len(rhos) > 1 else [0.0]
            ax.scatter(
                np.array([m] * len(rhos)) + jitter, rhos,
                marker="x" if not reliable else "+", s=14, linewidths=0.9,
                color=color, alpha=0.55 if not reliable else 0.8, zorder=3,
            )

    if box_data:
        bp = ax.boxplot(
            box_data, positions=box_positions, widths=0.55, patch_artist=True,
            manage_ticks=False, showfliers=False, zorder=2,
            boxprops=dict(linewidth=0.7), whiskerprops=dict(linewidth=0.7, color="#555555"),
            capprops=dict(linewidth=0.7, color="#555555"),
            medianprops=dict(linewidth=1.1, color="#222222"),
        )
        for patch, reliable in zip(bp["boxes"], box_reliable):
            patch.set_facecolor(color)
            if reliable:
                patch.set_alpha(0.32)
                patch.set_edgecolor(color)
                patch.set_hatch(None)
            else:
                patch.set_alpha(0.16)
                patch.set_edgecolor(color)
                patch.set_hatch("///")


def reliability_legend_handles():
    return [
        Line2D([0], [0], color="#444444", lw=1.5, ls="-", label="Pooled monthly ρ — reliable curve"),
        Line2D([0], [0], color="#444444", lw=1.5, ls="--", alpha=0.5, label="Pooled monthly ρ — low-N curve"),
        Line2D([0], [0], marker="o", mfc="#444444", mec="#444444", color="none", ms=5, label="Pooled ρ, reliable month (filled)"),
        Line2D([0], [0], marker="o", mfc="white", mec="#444444", color="none", ms=5, label="Pooled ρ, low-N month (open marker)"),
        Patch(facecolor="#444444", alpha=0.32, edgecolor="#444444", label="Site-level ρ distribution, reliable month (box = towers' ρ)"),
        Patch(facecolor="#444444", alpha=0.16, edgecolor="#444444", hatch="///", label="Site-level ρ distribution, low-N month (hatched box)"),
    ]


def role_legend_handles():
    return [
        Line2D([0], [0], color="#444444", lw=1.6, ls="-", marker="o", ms=4.5, label="GPP median ± IQR (LEFT axis, filled band)"),
        Line2D([0], [0], color="#444444", lw=1.6, ls="--", marker="^", ms=4.5, mfc="none", label="SIF median ± IQR (RIGHT axis, hatched band)"),
    ]


def panel_letter(ax, letter):
    ax.text(-0.17, 1.16, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="left")


# --------------------------------------------------------------------------
# Row-A (combined SIF+GPP dual-axis) drawing
# --------------------------------------------------------------------------
def draw_combined_panel(ax_gpp, ms: pd.DataFrame, product: str):
    color = PRODUCT_COLORS[product]
    months = np.arange(1, 13)
    row = ms.set_index("MONTH").reindex(months)

    ax_sif = ax_gpp.twinx()
    # GPP visually on top (left axis is the primary-ordered variable here).
    ax_gpp.set_zorder(ax_sif.get_zorder() + 1)
    ax_gpp.patch.set_visible(False)

    # SIF (right axis) drawn first, underneath: dashed line, open marker,
    # lighter hatched band.
    ax_sif.fill_between(months, row["SIF_Q25"], row["SIF_Q75"], facecolor=color,
                          alpha=0.10, edgecolor=color, hatch="///", lw=0.0, zorder=1)
    ax_sif.plot(months, row["SIF_MEDIAN"], "--", color=color, lw=1.3, zorder=2)
    ax_sif.plot(months, row["SIF_MEDIAN"], "^", mfc="none", mec=color, mew=1.0, ms=3.6, zorder=2)

    # GPP (left axis): solid line, filled marker, solid translucent band.
    ax_gpp.fill_between(months, row["GPP_Q25"], row["GPP_Q75"], color=color, alpha=0.24, lw=0, zorder=3)
    ax_gpp.plot(months, row["GPP_MEDIAN"], "-o", color=color, lw=1.7, ms=3.6, zorder=4)

    for ax in (ax_gpp, ax_sif):
        ax.set_xlim(0.5, 12.5)
        ax.set_xticks([1, 3, 5, 7, 9, 11])
    ax_gpp.set_xticklabels([])
    ax_gpp.grid(color="#e6e6e6", lw=0.35, zorder=0)
    ax_gpp.spines["top"].set_visible(False)
    ax_sif.spines["top"].set_visible(False)
    return ax_sif


# --------------------------------------------------------------------------
# Figure assembly
# --------------------------------------------------------------------------
def build_hemisphere_figure(df_master: pd.DataFrame, harm: pd.DataFrame, hemi: str, label: str,
                             report_rows: list, sparse_note: str = None):
    sub_all = df_master[df_master["HEMI"].eq(hemi)]
    harm_hemi = harm[harm["HEMI"].eq(hemi)]

    fig, axes = plt.subplots(2, 4, figsize=(15.6, 9.3), facecolor="white",
                              gridspec_kw={"height_ratios": [1.28, 1.0], "hspace": 0.42, "wspace": 0.62})

    for c, product in enumerate(PRODUCTS):
        sub = sub_all[sub_all["SIF_PRODUCT"].eq(product)]
        ms = monthly_stats(sub).set_index("MONTH").reindex(range(1, 13)).reset_index()
        n_total = len(sub)
        n_sites = sub["SITE_ID"].nunique()
        months_pass = int(ms["PASS"].astype("boolean").fillna(False).sum())

        # per-tower box sample sizes for the report
        box_n_by_month = {}
        for m in range(1, 13):
            hsub = harm_hemi[(harm_hemi["SIF_PRODUCT"] == product) & (harm_hemi["MONTH"] == m)]
            box_n_by_month[m] = len(per_tower_monthly_rhos(hsub))

        flagged_months = [MONTH_LABELS[m - 1] for m in range(1, 13)
                           if not bool(ms.set_index("MONTH").loc[m, "PASS"])]
        report_rows.append({
            "HEMISPHERE": label, "SIF_PRODUCT": product,
            "N_SITE_MONTHS_TOTAL": n_total, "N_TOWERS_TOTAL": n_sites,
            "MONTHS_PASSING_OF_12": months_pass,
            "RELIABILITY_FLAGGED_MONTHS": ";".join(flagged_months),
            "MIN_TOWERS_WITH_BOX": min(box_n_by_month.values()) if box_n_by_month else 0,
            "MAX_TOWERS_WITH_BOX": max(box_n_by_month.values()) if box_n_by_month else 0,
            "TOWERS_WITH_BOX_BY_MONTH": ";".join(f"{MONTH_LABELS[m-1]}:{box_n_by_month[m]}" for m in range(1, 13)),
        })

        # Row A: combined SIF+GPP dual axis
        ax0 = axes[0, c]
        if n_total > 0:
            ax_sif = draw_combined_panel(ax0, ms, product)
        else:
            ax0.axis("off")
            ax_sif = None
        ax0.set_title(f"{product}\nn={n_total:,} site-mo. ({n_sites} towers)", fontsize=8.8, fontweight="bold", pad=6)
        if c == 0:
            ax0.set_ylabel("GPP (median ± IQR)\n[μmol CO$_2$ m$^{-2}$ s$^{-1}$]")
            panel_letter(ax0, "A")
        if c == len(PRODUCTS) - 1 and ax_sif is not None:
            ax_sif.set_ylabel("SIF (median ± IQR)", rotation=270, labelpad=13)

        # Row B: pooled rho line + site-level rho distribution
        ax1 = axes[1, c]
        draw_pooled_rho_line(ax1, ms, product)
        draw_site_level_box(ax1, harm_hemi, ms, product, hemi)
        style_rho_axis(ax1)
        if c == 0:
            ax1.set_ylabel("Spearman ρ\n(SIF vs GPP)")
            panel_letter(ax1, "B")
        ax1.set_xlabel("Month")

    handles_role = role_legend_handles()
    handles_rel = reliability_legend_handles()
    leg1 = fig.legend(handles=handles_role, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.095),
                       frameon=False, fontsize=7.6, title="Row A encoding (dual axis: LEFT = GPP, RIGHT = SIF)",
                       title_fontsize=7.8)
    fig.legend(handles=handles_rel, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.005),
               frameon=False, fontsize=7.0, title="Row B encoding (reliability threshold: N≥12 site-months AND N≥3 towers, per month)",
               title_fontsize=7.4)
    fig.add_artist(leg1)

    title = f"Seasonal SIF–GPP structure (v2) – {label} Hemisphere sites only"
    subtitle = ("Row A: SIF + GPP combined per product (dual axis).   "
                "Row B: pooled monthly Spearman ρ + distribution of per-tower ρ (site-level).")
    caption = ("Site-level ρ (box): Spearman ρ per EC tower between its own SIF and GPP harmonic-residual anomalies,\n"
               "across years, within one calendar month (≥3 years/tower; ≥3 towers required for a box, "
               "else individual tower ρ shown as open points, +/×).")
    if sparse_note:
        subtitle += f"\n{sparse_note}"

    fig.suptitle(title, fontsize=13.5, fontweight="bold", x=0.01, ha="left", y=1.06)
    fig.text(0.01, 1.01, subtitle, fontsize=8.8, ha="left", va="top")
    fig.text(0.01, 0.185, caption, fontsize=7.3, ha="left", va="top", style="italic", color="#333333")

    fig.subplots_adjust(left=0.055, right=0.945, top=0.855, bottom=0.335)
    save_all(fig, f"Seasonal_Structure_{label}Hemisphere_v2")


def save_all(fig, name: str):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"saved {name} (.png/.svg/.pdf)")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    df_master = load_master_data()
    harm = load_harmonic_data()
    report_rows: list = []

    # Determine sparse-hemisphere note dynamically (do not hardcode).
    def hemi_towers(hemi):
        return df_master[df_master["HEMI"].eq(hemi)]["SITE_ID"].nunique()

    nh_towers, sh_towers = hemi_towers("NH"), hemi_towers("SH")

    build_hemisphere_figure(df_master, harm, "NH", "Northern", report_rows)

    # Check whether OCO-3 (or any product) has zero reliable months in SH, to
    # keep the sparse note accurate rather than copy-pasted.
    sh_sub = df_master[df_master["HEMI"].eq("SH")]
    zero_reliable_products = []
    for product in PRODUCTS:
        ms = monthly_stats(sh_sub[sh_sub["SIF_PRODUCT"].eq(product)])
        if ms.empty or not ms["PASS"].astype("boolean").fillna(False).any():
            zero_reliable_products.append(product)
    sparse_bits = [f"Note: SH sample is far smaller than NH ({sh_towers} vs {nh_towers} towers)"]
    if zero_reliable_products:
        sparse_bits.append(
            f"{'/'.join(zero_reliable_products)} has zero months meeting the reliability threshold in the SH "
            f"and {'is' if len(zero_reliable_products)==1 else 'are'} shown low-N throughout."
        )
    sparse_note = "; ".join(sparse_bits)

    build_hemisphere_figure(df_master, harm, "SH", "Southern", report_rows, sparse_note=sparse_note)

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(OUT_DIR / "seasonal_structure_v2_sample_size_report.csv", index=False)
    print("wrote seasonal_structure_v2_sample_size_report.csv")
    print("Done.")


if __name__ == "__main__":
    main()
