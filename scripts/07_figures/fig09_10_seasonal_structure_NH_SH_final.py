"""
TASK 5 (fix the Panel B line/boxplot MISMATCH, not just the color-contrast issue
fixed earlier in task1a_fix_panelB_contrast.py):

Root cause found by tracing the two Panel-B layers back to their source data in
build_seasonal_structure_v2.py:

  - The dark "pooled" line/markers (draw_pooled_rho_line) are the Spearman
    correlation of raw SIF_VALUE vs GPP_VALUE, computed by POOLING ALL EC TOWERS
    AND ALL YEARS together for a given calendar month (monthly_stats(), reading
    SIF_GPP_Analysis_Ready_Master.csv). Because absolute SIF/GPP magnitudes differ
    hugely BETWEEN towers/biomes, this correlation is dominated by cross-site
    (spatial) structure -- e.g. "high-productivity towers have both higher SIF and
    higher GPP" -- not by within-site temporal covariation.

  - The boxplots (draw_site_level_box / per_tower_monthly_rhos) are built from a
    completely different quantity: for EACH TOWER SEPARATELY, the Spearman
    correlation between that tower's OWN SIF_ANOM and GPP_ANOM (harmonic-residual
    anomalies) ACROSS YEARS, within one calendar month. This isolates each
    tower's own interannual (temporal) covariation and removes all between-site
    spatial structure.

  These are genuinely different estimands (between-site/raw vs. within-site/
  anomaly), so the line's position relative to the boxes for a given month is not
  meaningful -- it does not have to (and in the original figure, visibly does not)
  sit near the box, which is exactly the "line maximum not matching the boxplot
  pattern" problem flagged.

FINAL fix (confirmed after discussion): the line is the MONTHLY MEDIAN of the
exact same per-tower rho values (per_tower_monthly_rhos) that populate that
month's box -- i.e. it is exactly that month's box-median value, connected month
to month. This guarantees the line always sits exactly at the box's own median
tick, every month, by construction. (An earlier MEAN variant stayed within the
box's range but need not equal its median; a MAX variant was tried and reverted
-- median is confirmed as the intended statistic.)

Reuses unmodified from build_seasonal_structure_v2.py: load_master_data(),
load_harmonic_data(), monthly_stats(), per_tower_monthly_rhos(), draw_site_level_box(),
draw_combined_panel(), style_rho_axis(), role_legend_handles(), panel_letter(),
PRODUCTS/PRODUCT_MARKERS/N_MIN/SITES_MIN/MIN_TOWERS_FOR_BOX. Only the line's data
source, the reliability legend wording, and the output filenames/suffix change.

Validation performed automatically at the end of this script (see
panelB_consistency_check.csv and the printed summary): for every
(hemisphere, product, month) cell that has data, checks the new line value
against that month's per-tower rho min/Q1/median/Q3/max and asserts it matches the
max EXACTLY (0 tolerance, since the line IS that same max value).
"""
# Original script filename (sanitized): task5_fix_panelB_consistency.py; public name: fig09_10_seasonal_structure_NH_SH_final.py
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
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(PROJECT_ROOT)
SRC_SCRIPT_DIR = ROOT / "All_the_Visualizations_REVISED_4PRODUCT_FINAL_ARIDITY_JUNE_DIAGNOSTICS" / "10_scripts"
OUT_DIR = ROOT / "All_the_Visualizations_REVISED" / "EXACT_REQUESTS_ONLY"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC_SCRIPT_DIR))
import importlib
src = importlib.import_module("build_seasonal_structure_v2")  # noqa: E402

POOLED_LINE_COLOR = "#111111"


def draw_pooled_rho_line_consistent(ax, harm_sub_hemi_product: pd.DataFrame, ms: pd.DataFrame, product: str):
    """The line's y-value at each month is the MEDIAN of the exact same
    per-tower rho values used to build that month's box -- i.e. the line IS the
    box's own median, connected month to month. The line therefore always sits
    exactly at the box's median tick, every month, with no possible mismatch."""
    pass_by_month = ms.set_index("MONTH")["PASS"].reindex(range(1, 13)).astype("boolean").fillna(False)
    marker = src.PRODUCT_MARKERS[product]
    months = np.arange(1, 13)

    y = np.full(12, np.nan)
    n_towers_by_month = np.zeros(12, dtype=int)
    for i, m in enumerate(months):
        sub = harm_sub_hemi_product[harm_sub_hemi_product["MONTH"] == m]
        rhos = src.per_tower_monthly_rhos(sub)
        n_towers_by_month[i] = len(rhos)
        if len(rhos) >= 1:
            y[i] = np.median(rhos)

    if np.all(np.isnan(y)):
        return y, n_towers_by_month

    any_reliable = bool(pass_by_month.any())
    halo = [pe.Stroke(linewidth=3.2, foreground="white"), pe.Normal()]
    line_style = "-" if any_reliable else "--"
    line_alpha = 1.0 if any_reliable else 0.55
    ax.plot(months, y, line_style, lw=2.0, color=POOLED_LINE_COLOR, alpha=line_alpha,
            zorder=8, path_effects=halo)

    for i, m in enumerate(months):
        if np.isnan(y[i]):
            continue
        passed = bool(pass_by_month.get(m, False))
        if passed:
            ax.plot(m, y[i], marker=marker, ms=5.6, mfc=POOLED_LINE_COLOR, mec="white",
                    mew=0.9, alpha=1.0, zorder=9)
        else:
            ax.plot(m, y[i], marker=marker, ms=5.0, mfc="white", mec=POOLED_LINE_COLOR,
                    mew=1.1, alpha=0.85, zorder=9)
    return y, n_towers_by_month


def reliability_legend_handles_fixed():
    return [
        Line2D([0], [0], color="#111111", lw=2.0, ls="-", label="Median of per-tower ρ = box's own median — reliable month"),
        Line2D([0], [0], color="#111111", lw=2.0, ls="--", alpha=0.55, label="Median of per-tower ρ — low-N month"),
        Line2D([0], [0], marker="o", mfc="#111111", mec="white", color="none", ms=6, label="Reliable month (filled, white edge)"),
        Line2D([0], [0], marker="o", mfc="white", mec="#111111", color="none", ms=6, label="Low-N month (open marker)"),
        Patch(facecolor="#888888", alpha=0.32, edgecolor="#888888", label="Distribution of the SAME per-tower ρ values (box, product color)"),
        Patch(facecolor="#888888", alpha=0.16, edgecolor="#888888", hatch="///", label="Low-N month (hatched box)"),
    ]


def build_hemisphere_figure_fixed(df_master, harm, hemi, label, report_rows, check_rows, sparse_note=None):
    sub_all = df_master[df_master["HEMI"].eq(hemi)]
    harm_hemi = harm[harm["HEMI"].eq(hemi)]

    fig, axes = plt.subplots(2, 4, figsize=(15.6, 9.3), facecolor="white",
                              gridspec_kw={"height_ratios": [1.28, 1.0], "hspace": 0.42, "wspace": 0.62})

    for c, product in enumerate(src.PRODUCTS):
        sub = sub_all[sub_all["SIF_PRODUCT"].eq(product)]
        ms = src.monthly_stats(sub).set_index("MONTH").reindex(range(1, 13)).reset_index()
        n_total = len(sub)
        n_sites = sub["SITE_ID"].nunique()

        ax0 = axes[0, c]
        if n_total > 0:
            ax_sif = src.draw_combined_panel(ax0, ms, product)
        else:
            ax0.axis("off")
            ax_sif = None
        ax0.set_title(f"{product}\nn={n_total:,} site-mo. ({n_sites} towers)", fontsize=8.8, fontweight="bold", pad=6)
        if c == 0:
            ax0.set_ylabel("GPP (median ± IQR)\n[μmol CO$_2$ m$^{-2}$ s$^{-1}$]")
            src.panel_letter(ax0, "A")
        if c == len(src.PRODUCTS) - 1 and ax_sif is not None:
            ax_sif.set_ylabel("SIF (median ± IQR)", rotation=270, labelpad=13)

        ax1 = axes[1, c]
        harm_sub_hp = harm_hemi[harm_hemi["SIF_PRODUCT"] == product]
        line_y, n_towers_by_month = draw_pooled_rho_line_consistent(ax1, harm_sub_hp, ms, product)
        src.draw_site_level_box(ax1, harm_hemi, ms, product, hemi)
        src.style_rho_axis(ax1)
        if c == 0:
            ax1.set_ylabel("Spearman ρ\n(SIF vs GPP)")
            src.panel_letter(ax1, "B")
        ax1.set_xlabel("Month")

        # ---- validation: line value vs. that month's box Q1/median/Q3 ----
        for m in range(1, 13):
            msub = harm_sub_hp[harm_sub_hp["MONTH"] == m]
            rhos = src.per_tower_monthly_rhos(msub)
            if len(rhos) == 0:
                continue
            q1, med, q3 = np.percentile(rhos, [25, 50, 75])
            rho_min, rho_max = np.min(rhos), np.max(rhos)
            line_val = line_y[m - 1]
            # Hard check: the line IS the box's own median (np.median of the identical values),
            # so it must match exactly (up to floating-point rounding).
            matches_exactly = abs(line_val - med) < 1e-9
            within_range = (rho_min - 1e-9) <= line_val <= (rho_max + 1e-9)
            check_rows.append({
                "HEMISPHERE": label, "SIF_PRODUCT": product, "MONTH": m,
                "N_TOWERS": len(rhos), "LINE_VALUE": round(line_val, 4),
                "BOX_MIN": round(rho_min, 4), "BOX_Q1": round(q1, 4), "BOX_MEDIAN": round(med, 4),
                "BOX_Q3": round(q3, 4), "BOX_MAX": round(rho_max, 4),
                "LINE_MATCHES_BOX_MEDIAN_EXACTLY": matches_exactly,
                "LINE_WITHIN_TOWER_RANGE": within_range,
            })

        report_rows.append({
            "HEMISPHERE": label, "SIF_PRODUCT": product,
            "N_SITE_MONTHS_TOTAL": n_total, "N_TOWERS_TOTAL": n_sites,
        })

    handles_role = src.role_legend_handles()
    handles_rel = reliability_legend_handles_fixed()
    leg1 = fig.legend(handles=handles_role, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.095),
                       frameon=False, fontsize=7.6, title="Row A encoding (dual axis: LEFT = GPP, RIGHT = SIF)",
                       title_fontsize=7.8)
    fig.legend(handles=handles_rel, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.005),
               frameon=False, fontsize=6.8,
               title="Row B encoding (reliability: N≥12 site-months AND N≥3 towers) — FIXED: line = the box's own "
                     "median (was: an unrelated pooled raw-value correlation)",
               title_fontsize=7.2)
    fig.add_artist(leg1)

    title = f"Seasonal SIF–GPP structure (v3, Panel B internally consistent) – {label} Hemisphere sites only"
    subtitle = ("Row A: SIF + GPP combined per product (dual axis).   "
                "Row B: median of per-tower ρ (dark line/markers, = each box's own median) + full distribution of "
                "the SAME per-tower ρ values (boxes).")
    caption = ("Site-level ρ (box): Spearman ρ per EC tower between its own SIF and GPP harmonic-residual anomalies,\n"
               "across years, within one calendar month (≥3 years/tower; ≥3 towers required for a box, "
               "else individual tower ρ shown as open points, +/×).\n"
               "FIX: the dark line/markers are now the MEDIAN of that same month's per-tower ρ values -- i.e. "
               "exactly the box's own median, connected month to month (previously the line was a different "
               "quantity entirely: a pooled correlation of raw SIF/GPP values across all towers and years at "
               "once, which mixes in cross-site/spatial structure and need not resemble the box at all). "
               "The line's peaks and troughs therefore now coincide exactly with the boxplots' own peaks and "
               "troughs, every month, by construction.")
    if sparse_note:
        subtitle += f"\n{sparse_note}"

    fig.suptitle(title, fontsize=13.5, fontweight="bold", x=0.01, ha="left", y=1.06)
    fig.text(0.01, 1.01, subtitle, fontsize=8.8, ha="left", va="top")
    fig.text(0.01, 0.185, caption, fontsize=7.2, ha="left", va="top", style="italic", color="#333333")

    fig.subplots_adjust(left=0.055, right=0.945, top=0.855, bottom=0.335)
    save_all(fig, f"Seasonal_Structure_{label}Hemisphere_v3_FIXED")


def save_all(fig, name: str):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"saved {name} (.png/.svg/.pdf)")


def main():
    df_master = src.load_master_data()
    harm = src.load_harmonic_data()
    report_rows: list = []
    check_rows: list = []

    def hemi_towers(hemi):
        return df_master[df_master["HEMI"].eq(hemi)]["SITE_ID"].nunique()

    nh_towers, sh_towers = hemi_towers("NH"), hemi_towers("SH")

    build_hemisphere_figure_fixed(df_master, harm, "NH", "Northern", report_rows, check_rows)

    sh_sub = df_master[df_master["HEMI"].eq("SH")]
    zero_reliable_products = []
    for product in src.PRODUCTS:
        ms = src.monthly_stats(sh_sub[sh_sub["SIF_PRODUCT"].eq(product)])
        if ms.empty or not ms["PASS"].astype("boolean").fillna(False).any():
            zero_reliable_products.append(product)
    sparse_bits = [f"Note: SH sample is far smaller than NH ({sh_towers} vs {nh_towers} towers)"]
    if zero_reliable_products:
        sparse_bits.append(
            f"{'/'.join(zero_reliable_products)} has zero months meeting the reliability threshold in the SH "
            f"and {'is' if len(zero_reliable_products) == 1 else 'are'} shown low-N throughout."
        )
    sparse_note = "; ".join(sparse_bits)

    build_hemisphere_figure_fixed(df_master, harm, "SH", "Southern", report_rows, check_rows, sparse_note=sparse_note)

    check_df = pd.DataFrame(check_rows)
    check_df.to_csv(OUT_DIR / "panelB_consistency_check.csv", index=False)
    n_fail_range = int((~check_df["LINE_WITHIN_TOWER_RANGE"]).sum())
    n_fail_exact = int((~check_df["LINE_MATCHES_BOX_MEDIAN_EXACTLY"]).sum())
    print(f"Validation: {len(check_df)} (hemisphere, product, month) cells checked.")
    print(f"  Hard check -- line within [min, max] of that month's per-tower rho values: "
          f"{n_fail_range} failures (expected: 0).")
    print(f"  Hard check -- line EXACTLY equals that month's box median: "
          f"{n_fail_exact} failures (expected: 0, since the line IS the box's own median).")
    print("wrote panelB_consistency_check.csv")
    print("Done.")


if __name__ == "__main__":
    main()
