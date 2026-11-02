"""
Revised biome x hemisphere seasonal-structure figures (v2) -- anomaly-based coupling.

Rebuilds, with the same general small-multiples design and reliability-flagging
language as the original reviewer-update figures
(All_the_Visualizations_REVISED_4PRODUCT_UPDATE/07_scripts/build_biome_hemisphere_figures.py),
but with one deliberate, documented methodological change:

  ORIGINAL figures plot Spearman rho computed on RAW SIF_VALUE vs GPP_VALUE,
  cross-sectionally, across sites, within each calendar month. That statistic
  is inflated by shared seasonality and cross-sectional biome-productivity
  confounds (wetter/greener towers have both higher SIF and higher GPP even
  with zero true local temporal coupling), and is unstable (regularly hits
  +/-1) whenever a month has only 2-3 contributing sites.

  THIS (v2) version instead plots Spearman rho computed on SIF_ANOM/GPP_ANOM
  -- the project's final accepted harmonic-residual deseasonalized anomalies
  (SIF_EC_SIF_GPP_Final_Verification/02_independent_reverification/
  harmonic_residuals_full.csv) -- which isolates genuine temporal
  (within-tower, within-month-across-years) coupling. Pooled-annual values
  reproduce SIF_EC_SIF_GPP_Final_Verification/05_mechanistic_reanalysis/
  biome_relationships_final_anomaly.csv exactly (verified in
  build_low_coupling_diagnostics.py's mistake-check, 0 mismatches / 34 cells).

Same reliability threshold as the original and as the project-wide anomaly
table: a (group, product, month) cell needs N >= 12 site-months AND
N >= 3 independent EC towers to be drawn "reliable" (filled marker, solid
line for the product if >=1 month clears it); cells/curves below threshold
are drawn open/faded/dashed, exactly as before.

Diagnostic footnotes (from 05_aridity_and_biome_analysis/Low_Biome_Coupling_Diagnostic_Report.md)
are added directly to each figure's caption where a specific, evidence-backed
explanation was found for a biome's low coupling.

Outputs: Seasonal_Structure_By_Biome_v2.{png,svg,pdf},
         Seasonal_Structure_Biome_within_Hemisphere_v2.{png,svg,pdf}
         (600 DPI), in 05_aridity_and_biome_analysis/.
"""
from __future__ import annotations
# Original script filename (sanitized): build_biome_hemisphere_figures_v2.py; public name: figS6_S7_seasonal_coupling_biome_hemisphere.py
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

ROOT = (Path(PROJECT_ROOT) / 'All_the_Visualizations_REVISED_4PRODUCT_FINAL_ARIDITY_JUNE_DIAGNOSTICS/10_scripts/build_biome_hemisphere_figures_v2.py').resolve().parents[2]
HARMONIC_PATH = ROOT / "SIF_EC_SIF_GPP_Final_Verification" / "02_independent_reverification" / "harmonic_residuals_full.csv"
OUT_DIR = ROOT / "All_the_Visualizations_REVISED_4PRODUCT_FINAL_ARIDITY_JUNE_DIAGNOSTICS" / "05_aridity_and_biome_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRODUCTS = ["GOSIF", "GOME-2", "TROPOSIF", "OCO-3"]
PRODUCT_COLORS = {"GOSIF": "#440154", "GOME-2": "#3e4989", "TROPOSIF": "#26828e", "OCO-3": "#35b779"}
PRODUCT_MARKERS = {"GOSIF": "o", "GOME-2": "s", "TROPOSIF": "^", "OCO-3": "D"}

N_MIN = 12
SITES_MIN = 3

BIOME_FULL_NAME = {
    "ENF": "Evergreen Needleleaf Forest", "EBF": "Evergreen Broadleaf Forest",
    "DBF": "Deciduous Broadleaf Forest", "MF": "Mixed Forest", "CSH": "Closed Shrubland",
    "OSH": "Open Shrubland", "WSA": "Woody Savanna", "SAV": "Savanna",
    "GRA": "Grassland", "CRO": "Cropland",
}
BIOME_ORDER_A = ["ENF", "EBF", "DBF", "MF", "OSH", "WSA", "SAV", "GRA", "CRO"]

BIOME_GROUP_MAP = {
    "ENF": "Forest", "EBF": "Forest", "DBF": "Forest", "MF": "Forest",
    "GRA": "Grassland", "SAV": "Grassland", "WSA": "Grassland", "OSH": "Grassland", "CSH": "Grassland",
    "CRO": "Cropland",
}

RSE_FONT = "Arial"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [RSE_FONT, "DejaVu Sans", "Liberation Sans"],
    "axes.titlesize": 9.5, "axes.labelsize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7.5, "figure.dpi": 120, "savefig.dpi": 600, "axes.linewidth": 0.65,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})


# --------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    df = pd.read_csv(HARMONIC_PATH, low_memory=False)
    df = df[df["SIF_PRODUCT"].isin(PRODUCTS)].copy()
    df = df.dropna(subset=["IGBP_GPP_TABLE", "LATITUDE", "SIF_ANOM", "GPP_ANOM", "MONTH"])
    df["IGBP_GPP_TABLE"] = df["IGBP_GPP_TABLE"].astype(str)
    df["MONTH"] = df["MONTH"].astype(int)

    site_lat = df.groupby("SITE_ID")["LATITUDE"].first()
    df["HEMI"] = df["SITE_ID"].map(lambda s: "NH" if site_lat[s] >= 0 else "SH")
    df["BIOME_GROUP"] = df["IGBP_GPP_TABLE"].map(BIOME_GROUP_MAP)
    return df


def monthly_stats(sub: pd.DataFrame) -> pd.DataFrame:
    """Per (SIF_PRODUCT, MONTH): Spearman rho(SIF_ANOM, GPP_ANOM), N, N_SITES, PASS."""
    rows = []
    for (product, month), g in sub.groupby(["SIF_PRODUCT", "MONTH"]):
        n = len(g)
        n_sites = g["SITE_ID"].nunique()
        if n >= 2 and g["SIF_ANOM"].nunique() > 1 and g["GPP_ANOM"].nunique() > 1:
            rho, pval = spearmanr(g["SIF_ANOM"], g["GPP_ANOM"])
        else:
            rho, pval = np.nan, np.nan
        rows.append({
            "SIF_PRODUCT": product, "MONTH": month, "N": n, "N_SITES": n_sites,
            "SPEARMAN_RHO": rho, "PVAL": pval,
            "PASS": (n >= N_MIN) and (n_sites >= SITES_MIN),
        })
    return pd.DataFrame(rows)


def annual_summary(sub: pd.DataFrame) -> pd.DataFrame:
    ms = monthly_stats(sub)
    out = []
    for product in PRODUCTS:
        g = sub[sub["SIF_PRODUCT"].eq(product)]
        mp = ms[ms["SIF_PRODUCT"].eq(product)]
        out.append({
            "SIF_PRODUCT": product, "N_TOTAL": len(g), "N_SITES_TOTAL": g["SITE_ID"].nunique(),
            "MONTHS_PASSING": int(mp["PASS"].sum()) if len(mp) else 0,
        })
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
def draw_rho_curve(ax, ms: pd.DataFrame, product: str, show_label: bool = True):
    sub = ms[ms["SIF_PRODUCT"].eq(product)].set_index("MONTH").reindex(range(1, 13))
    if sub["SPEARMAN_RHO"].notna().sum() == 0:
        return False

    color = PRODUCT_COLORS[product]
    marker = PRODUCT_MARKERS[product]
    months = np.arange(1, 13)
    y = sub["SPEARMAN_RHO"].values
    any_reliable = bool(sub["PASS"].astype("boolean").fillna(False).any())

    line_style = "-" if any_reliable else "--"
    line_alpha = 1.0 if any_reliable else 0.45
    ax.plot(months, y, line_style, lw=1.5, color=color, alpha=line_alpha,
             label=product if show_label else None, zorder=3)

    for m in months:
        yv = sub.loc[m, "SPEARMAN_RHO"]
        if pd.isna(yv):
            continue
        passed = bool(sub.loc[m, "PASS"])
        if passed:
            ax.plot(m, yv, marker=marker, ms=4.2, mfc=color, mec=color, mew=0.6, alpha=1.0, zorder=4)
        else:
            ax.plot(m, yv, marker=marker, ms=4.4, mfc="white", mec=color, mew=1.1, alpha=0.75, zorder=4)
    return True


def style_rho_axis(ax, title: str = None):
    ax.axhline(0, color="#888888", lw=0.55, zorder=1)
    ax.axhspan(-0.20, 0.20, color="#c0392b", alpha=0.06, zorder=0)  # low-coupling band, made visually distinct
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(0.5, 12.5)
    ax.set_xticks([1, 3, 5, 7, 9, 11])
    ax.set_xticklabels(["Jan", "Mar", "May", "Jul", "Sep", "Nov"])
    ax.grid(color="#e6e6e6", lw=0.35, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    if title:
        ax.set_title(title, loc="left", fontweight="bold", fontsize=8.8)


def reliability_legend_handles():
    return [
        Line2D([0], [0], color="#444444", lw=1.5, ls="-", label="Reliable curve (>=1 month meets N>=12 & towers>=3)"),
        Line2D([0], [0], color="#444444", lw=1.5, ls="--", alpha=0.5, label="Low-N curve (no month meets threshold)"),
        Line2D([0], [0], marker="o", mfc="#444444", mec="#444444", color="none", ms=5, label="Reliable month (filled)"),
        Line2D([0], [0], marker="o", mfc="white", mec="#444444", mew=1.3, color="none", ms=5.5, label="Low-N month (open, bold outline)"),
        Patch := plt.Rectangle((0, 0), 1, 1, fc="#c0392b", alpha=0.12, label="|rho| <= 0.20 zone (weak coupling)"),
    ]


def product_legend_handles(products=PRODUCTS):
    return [Line2D([0], [0], color=PRODUCT_COLORS[p], marker=PRODUCT_MARKERS[p], lw=1.6, ms=5, label=p)
            for p in products]


def panel_letter(ax, letter):
    ax.text(-0.14, 1.14, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="left")


# --------------------------------------------------------------------------
# Diagnostic footnote text (evidence-backed findings from
# Low_Biome_Coupling_Diagnostic_Report.md), keyed by biome IGBP code.
# --------------------------------------------------------------------------
BIOME_FOOTNOTES = {
    "ENF": "ENF: GOME-2 rho=0.14 & TROPOSIF rho=0.10 despite adequate N (Jan/Dec ~0) -- likely genuine evergreen-conifer winter NPQ decoupling (physiological), not an artifact.",
    "EBF": "EBF: GOME-2 rho=-0.02 vs GOSIF rho=0.30 on largely the same towers -- GOME-2's coarse (~2,000 km2) footprint dilutes signal in humid-tropical canopies; not aridity/hemisphere (towers mostly Humid-class).",
    "OSH": "OSH: GOME-2 rho=0.04 vs GOSIF rho=0.71 on the SAME 3 towers, every month -- a controlled product-footprint effect, compounded by (not caused by) the biome's inherent aridity.",
    "MF": "MF: GOME-2 rho=0.08 vs GOSIF rho=0.32 (same biome) -- likely GOME-2 footprint mixing across the ENF/DBF patch mosaic that defines 'mixed forest'; not aridity- or temp-range-linked.",
    "SAV": "SAV: GOME-2 rho=0.20 with towers 71% Semi-Arid/Arid (vs 37% project-wide) -- aridity-linked tree/grass asynchrony plausibly compounds GOME-2's coarse-footprint mixing.",
    "CRO": "CRO: GOME-2 rho=0.16 & OCO-3 rho=0.20 (2 months zero OCO-3 coverage) weaker than GOSIF/TROPOSIF -- management decoupling + product coverage gaps, not one bad site (per-tower rho dispersed).",
}


# --------------------------------------------------------------------------
def build_figure_A(df: pd.DataFrame, report_rows: list):
    biomes = BIOME_ORDER_A
    ncols, nrows = 3, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.6, 13.4), facecolor="white")
    axes_flat = axes.flatten()

    for i, biome in enumerate(biomes):
        ax = axes_flat[i]
        sub = df[df["IGBP_GPP_TABLE"].eq(biome)]
        ms = monthly_stats(sub)
        ann = annual_summary(sub)

        for product in PRODUCTS:
            draw_rho_curve(ax, ms, product)
            row = ann[ann["SIF_PRODUCT"].eq(product)].iloc[0]
            report_rows.append({
                "FIGURE": "A_By_Biome_v2", "GROUP": biome, "SIF_PRODUCT": product,
                "N_TOTAL": row["N_TOTAL"], "N_SITES_TOTAL": row["N_SITES_TOTAL"],
                "MONTHS_PASSING_OF_12": row["MONTHS_PASSING"],
                "STATUS": "no data" if row["N_TOTAL"] == 0 else
                          ("reliable (>=1 month)" if row["MONTHS_PASSING"] > 0 else "low-N (shown faded/dashed)"),
            })

        n_sites_max = ann["N_SITES_TOTAL"].max()
        panel_letter(ax, chr(65 + i))
        style_rho_axis(ax, title=f"{biome} - {BIOME_FULL_NAME[biome]}  (max n towers={n_sites_max})")
        if i % ncols == 0:
            ax.set_ylabel("Spearman ρ\n(SIF_ANOM vs GPP_ANOM)")
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("Month")
        if biome in BIOME_FOOTNOTES:
            ax.text(0.5, -0.32, "† see caption note below", transform=ax.transAxes, ha="center",
                    fontsize=6.3, color="#c0392b", style="italic")

    handles_p = product_legend_handles()
    handles_r = reliability_legend_handles()
    fig.legend(handles=handles_p, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.185),
               frameon=False, fontsize=8.5, title="SIF product")
    fig.legend(handles=handles_r, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.135),
               frameon=False, fontsize=7.2)

    fig.suptitle(
        "Seasonal SIF-GPP coupling strength by biome (IGBP class), all four products -- v2 (anomaly-based)\n"
        "Monthly Spearman ρ on harmonic-residual anomalies (SIF_ANOM vs GPP_ANOM), the project's final "
        "accepted deseasonalization; CSH (1 tower globally) omitted.",
        fontsize=12.3, fontweight="bold", x=0.02, ha="left", y=0.995)

    fig.text(0.02, 0.113, "† Diagnosed low-coupling cells (full evidence in "
             "Low_Biome_Coupling_Diagnostic_Report.md):", fontsize=6.6, color="#333333",
             family="sans-serif", fontweight="bold")
    y_cursor = 0.100
    for b in biomes:
        if b not in BIOME_FOOTNOTES:
            continue
        wrapped = _wrap(BIOME_FOOTNOTES[b], 172)
        n_lines = wrapped.count("\n") + 1
        fig.text(0.02, y_cursor, wrapped, fontsize=6.3, color="#333333", family="sans-serif", va="top")
        y_cursor -= n_lines * 0.0125 + 0.004

    fig.tight_layout(rect=[0, 0.20, 1, 0.925])
    save_all(fig, "Seasonal_Structure_By_Biome_v2")


def _wrap(text, width):
    import textwrap
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


# --------------------------------------------------------------------------
def build_figure_D(df: pd.DataFrame, report_rows: list):
    combos = [
        ("NH", "Forest"), ("NH", "Grassland"), ("NH", "Cropland"),
        ("SH", "Forest"), ("SH", "Grassland"),
    ]
    omitted = [("SH", "Cropland")]

    ncols, nrows = 3, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.6, 10.8), facecolor="white")
    axes_flat = axes.flatten()

    for i, (hemi, group) in enumerate(combos):
        ax = axes_flat[i]
        sub = df[(df["HEMI"].eq(hemi)) & (df["BIOME_GROUP"].eq(group))]
        ms = monthly_stats(sub)
        ann = annual_summary(sub)
        for product in PRODUCTS:
            draw_rho_curve(ax, ms, product)
            row = ann[ann["SIF_PRODUCT"].eq(product)].iloc[0]
            report_rows.append({
                "FIGURE": "D_Biome_within_Hemisphere_v2", "GROUP": f"{hemi}-{group}", "SIF_PRODUCT": product,
                "N_TOTAL": row["N_TOTAL"], "N_SITES_TOTAL": row["N_SITES_TOTAL"],
                "MONTHS_PASSING_OF_12": row["MONTHS_PASSING"],
                "STATUS": "no data" if row["N_TOTAL"] == 0 else
                          ("reliable (>=1 month)" if row["MONTHS_PASSING"] > 0 else "low-N (shown faded/dashed)"),
            })
        panel_letter(ax, chr(65 + i))
        n_sites_max = ann["N_SITES_TOTAL"].max()
        style_rho_axis(ax, title=f"{hemi} - {group}  (max n towers={n_sites_max})")
        if i % ncols == 0:
            ax.set_ylabel("Spearman ρ\n(SIF_ANOM vs GPP_ANOM)")
        ax.set_xlabel("Month")

    ax_note = axes_flat[5]
    ax_note.axis("off")
    for hemi, group in omitted:
        sub = df[(df["HEMI"].eq(hemi)) & (df["BIOME_GROUP"].eq(group))]
        n_total = len(sub); n_sites = sub["SITE_ID"].nunique()
        report_rows.append({
            "FIGURE": "D_Biome_within_Hemisphere_v2", "GROUP": f"{hemi}-{group}", "SIF_PRODUCT": "ALL",
            "N_TOTAL": n_total, "N_SITES_TOTAL": n_sites, "MONTHS_PASSING_OF_12": 0,
            "STATUS": "omitted -- no EC towers in this hemisphere/biome combination",
        })
    ax_note.text(0.5, 0.62, "SH - Cropland", ha="center", va="center", fontsize=9.5, fontweight="bold",
                 transform=ax_note.transAxes)
    ax_note.text(0.5, 0.42, "Omitted: no Southern Hemisphere\ncropland EC towers in the dataset\n(N = 0 site-months, 0 towers)",
                 ha="center", va="center", fontsize=8, color="#555555", transform=ax_note.transAxes)
    ax_note.text(0.5, 0.10,
                 "Note (v2): NH-Cropland's overall coupling is weaker for\n"
                 "GOME-2/OCO-3 than GOSIF/TROPOSIF -- product coverage/\n"
                 "footprint effect, see caption note below.",
                 ha="center", va="center", fontsize=6.6, color="#c0392b", style="italic", transform=ax_note.transAxes)

    handles_p = product_legend_handles()
    handles_r = reliability_legend_handles()
    fig.legend(handles=handles_p, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.155),
               frameon=False, fontsize=8.5, title="SIF product")
    fig.legend(handles=handles_r, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.108),
               frameon=False, fontsize=7.2)

    fig.suptitle(
        "Seasonal SIF-GPP coupling by biome group within hemisphere -- v2 (anomaly-based)\n"
        "Top row: Northern Hemisphere; bottom row: Southern Hemisphere. Monthly Spearman ρ on "
        "harmonic-residual anomalies, reliability-flagged.",
        fontsize=12.3, fontweight="bold", x=0.02, ha="left", y=0.985)

    footnote = ("† Diagnostic note: SH-Forest is dominated by EBF towers (see By-Biome figure, EBF footnote) -- "
                "GOME-2's weak SH-Forest coupling mirrors its weak EBF coupling, a product/footprint effect, "
                "not a Southern-Hemisphere-seasonality effect (GOSIF, same towers, stays positive all year). "
                "Full evidence in Low_Biome_Coupling_Diagnostic_Report.md.")
    fig.text(0.02, 0.070, _wrap(footnote, 172), fontsize=6.5, color="#333333", family="sans-serif", va="top")
    fig.tight_layout(rect=[0, 0.24, 1, 0.90])
    save_all(fig, "Seasonal_Structure_Biome_within_Hemisphere_v2")


# --------------------------------------------------------------------------
def save_all(fig, name: str):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {name} (.png/.svg/.pdf)")


def main():
    df = load_data()
    report_rows: list = []
    build_figure_A(df, report_rows)
    build_figure_D(df, report_rows)
    pd.DataFrame(report_rows).to_csv(OUT_DIR / "biome_hemisphere_v2_sample_size_report_data.csv", index=False)
    print("Done.")


if __name__ == "__main__":
    main()
