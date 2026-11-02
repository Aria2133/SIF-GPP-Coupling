"""
TASK 3 (reviewer exact-requests-only revision):
Rebuild SID_Figure_Seasonal_BiomeHemisphere_Overview with a SHARED y-axis range
(0 to 0.35) across every panel, replacing the original per-row/per-panel auto-scaled
axes. Nothing else about the figure changes. Reuses the original script's data-loading
and decision logic unmodified (SIF_EC_Stress_Induced_Decoupling_SID / the
ARIDITY_JUNE_DIAGNOSTICS lineage) -- only the plotting function's y-limit (and the
now-inaccurate "per-row scale" caption line, corrected to describe the new shared
scale) are changed. Original file is untouched; this writes only to the new revision
folder.
"""
# Original script filename (sanitized): task3_fix_overview_yaxis.py; public name: figS5_sid_seasonal_biome_hemisphere.py
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

ROOT = Path(PROJECT_ROOT)
SRC_SCRIPT_DIR = ROOT / "All_the_Visualizations_REVISED_4PRODUCT_FINAL_ARIDITY_JUNE_DIAGNOSTICS" / "10_scripts"
OUT_DIR = ROOT / "All_the_Visualizations_REVISED" / "EXACT_REQUESTS_ONLY"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC_SCRIPT_DIR))
import importlib
src = importlib.import_module("06_sid_biome_hemisphere_seasonal")  # noqa: E402

Y_MIN, Y_MAX = 0.0, 0.35


def build_overview_grid_fixed_yaxis(df, qualifying):
    print("Building overview grid with SHARED y-axis (0-0.35) across all panels ...")
    n_rows = len(qualifying)
    fig, axes = plt.subplots(n_rows, 3, figsize=(14, 2.05 * n_rows + 1.0), squeeze=False)

    for row_idx, (biome, hemi, rel_stressors, n_towers) in enumerate(qualifying):
        sub = df[(df["IGBP"] == biome) & (df["HEMISPHERE"] == hemi)]
        summer_months = [6, 7, 8] if hemi == "NH" else [12, 1, 2]
        for col_idx, stressor in enumerate(src.STRESS_ORDER):
            ax = axes[row_idx][col_idx]
            sid_col = src.STRESSORS[stressor]
            for m in summer_months:
                ax.axvspan(m - 0.5, m + 0.5, color="#FDECEC", zorder=0)
            msum = src.monthly_stats(sub, sid_col)
            is_rel = stressor in rel_stressors
            line_color = src.STRESSOR_COLORS[stressor]
            ax.plot(msum["MONTH"], msum["median"], color=line_color, linewidth=1.3,
                    alpha=0.9 if is_rel else 0.4, zorder=3)
            rel_pts = msum[msum["reliable"]]
            unrel_pts = msum[~msum["reliable"]]
            ax.scatter(rel_pts["MONTH"], rel_pts["median"], color=line_color, s=14, zorder=4)
            ax.scatter(unrel_pts["MONTH"], unrel_pts["median"], facecolor="none", edgecolor=line_color,
                       s=14, linewidth=0.8, zorder=4)
            ax.set_xticks(range(1, 13))
            if row_idx == n_rows - 1:
                ax.set_xticklabels(src.MONTH_NAMES, fontsize=6.3, rotation=45)
            else:
                ax.set_xticklabels([])
            ax.tick_params(axis="y", labelsize=6.3)
            ax.grid(axis="y", color="#EEEEEE", linewidth=0.5, zorder=0)
            ax.set_ylim(Y_MIN, Y_MAX)  # <-- TASK 3 FIX: shared range, was set_ylim(bottom=0) (auto top)
            if col_idx == 0:
                marg = " (marginal N)" if n_towers == src.REL_MIN_TOWERS else ""
                ax.set_ylabel(f"{biome}-{hemi}\n({n_towers} towers{marg})", fontsize=7.6, fontweight="bold")
            if row_idx == 0:
                ax.set_title(f"{stressor} stress", fontsize=9.5, fontweight="bold")

    handles = [Line2D([0], [0], color=src.STRESSOR_COLORS[s], linewidth=2, label=f"{s} stress") for s in src.STRESS_ORDER]
    handles.append(Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none",
                          markeredgecolor="#555555", markersize=6, label="Unreliable month (hollow)"))
    handles.append(mpatches.Patch(facecolor="#FDECEC", label="Approx. local summer (hemisphere-specific)"))
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.012), frameon=False)

    fig.suptitle(
        src.wrap_text(
            "Combined overview: calendar-month Stress-Induced Decoupling climatology, by biome x hemisphere "
            "(each row = one panel from the corresponding standalone SID_Figure_Seasonal_<Biome>_<Hemisphere> "
            f"figure; y-axis = median SID for that stressor, SHARED {Y_MIN:.2f}-{Y_MAX:.2f} scale on every panel "
            "for direct comparability -- revised from the original's per-row auto-scaled axes)",
            width=130),
        fontsize=12.5, fontweight="bold", y=1.012,
    )
    fig.text(0.5, -0.045,
              src.wrap_text(
                  "Descriptive cross-site pattern, not a causal claim. Reliability rule: N>=12 site-months AND "
                  "N>=3 towers per calendar month. Rows with exactly 3 towers are marginal -- see companion "
                  "SID_biome_hemisphere_support_summary.csv. Source: 03_sid_monthly/monthly_SID_master.csv. "
                  f"Y-axis fixed to {Y_MIN:.2f}-{Y_MAX:.2f} on every panel (task-requested revision).",
                  width=175),
              ha="center", fontsize=7.0, color="#555555")
    fig.subplots_adjust(wspace=0.32, hspace=0.35, top=0.955, bottom=0.075)
    src.save_figure(fig, "SID_Figure_Seasonal_BiomeHemisphere_Overview_REVISED", OUT_DIR)


def main():
    df = src.load_master()
    biomes_present = [b for b in src.IGBP_ORDER if (df["IGBP"] == b).any()]
    support = src.build_support_summary(df, biomes_present)
    decisions = src.decide_figures(support)
    built_combos = sorted([(b, h) for (b, h), d in decisions.items() if d["build"]])
    print(f"Reusing decision logic: {len(built_combos)} biome x hemisphere combos qualify (unchanged from original).")

    overview_rows = []
    for biome, hemi in built_combos:
        d = decisions[(biome, hemi)]
        sub = df[(df["IGBP"] == biome) & (df["HEMISPHERE"] == hemi)]
        overview_rows.append((biome, hemi, d["rel_stressors"], sub["SITE_ID"].nunique()))

    build_overview_grid_fixed_yaxis(df, overview_rows)


if __name__ == "__main__":
    main()
