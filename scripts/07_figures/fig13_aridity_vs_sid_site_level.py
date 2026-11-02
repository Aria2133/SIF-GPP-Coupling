"""
TASK 7 (supersedes task4/task6): go back to the ORIGINAL Diagnostic_Coupling_vs_Aridity_REVISED.png
logic exactly -- do not substitute a different variable/metric -- and simply average every
pixel/observation across all time steps for each site, so each panel shows one dot per site.

Original REVISED script (task2_build_coupling_vs_aridity_scatters.py) panels, UNCHANGED here:
  Panel A: y = SID (mean of SID_VPD, SID_SM, SID_TEMP), x = Aridity Index.
  Panel B: y = GPP anomaly, x = SIF anomaly, colored by aridity class.

The ONLY change from the original REVISED figure is the level of aggregation: every site-month
observation is averaged to ONE value per site before plotting, instead of plotting the full
site-month cloud. No new variable, no correlation/coupling metric, no redesign of what each axis
means -- exactly the same two variables on exactly the same two axes as the original figure.

Panel A: SID_SITE_MEAN (mean of SID_VPD/SID_SM/SID_TEMP, pooled across the 4 main-paper products
and all years/months for that tower) vs. AI_ANNUAL. Covers all 73 towers (SID master data and the
aridity table both have complete coverage).

Panel B: SIF_ANOM_SITE_MEAN vs. GPP_ANOM_SITE_MEAN (mean of that tower's own harmonic-residual
SIF_ANOM / GPP_ANOM values, pooled across the 4 main-paper products and all years/months),
colored by AI_CLASS -- exactly the original panel B's two variables and coloring, just averaged
per site instead of plotted site-month by site-month.

IMPORTANT, stated plainly: the harmonic-residual anomaly table
(harmonic_residuals_full.csv) that Panel B's variables come from has ONLY 71 of the 73
study towers -- two towers (US-ARc, US-ARb) have ZERO rows in that table anywhere in this project
(confirmed directly: dropping every NaN filter and searching by SITE_ID alone still returns 0
rows for both). This is a pre-existing gap in the anomaly-computation pipeline, not something
introduced here, and there is no honest way to compute a mean SIF/GPP anomaly for a site with zero
anomaly observations without fabricating data. Panel B therefore has 71 real dots, clearly stated
as such in the figure itself (not silently short of 73). Panel A is unaffected and has all 73.

Output: Diagnostic_Coupling_vs_Aridity_SITE_LEVEL_FINAL.{png,svg,pdf} (this is now the
authoritative site-level version; the file supersedes both task4 and task6's outputs).
"""
# Original script filename (sanitized): task7_site_level_aridity_ORIGINAL_LOGIC.py; public name: fig13_aridity_vs_sid_site_level.py
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(PROJECT_ROOT)
SID_MASTER = ROOT / "SIF_EC_Stress_Induced_Decoupling_SID" / "03_sid_monthly" / "monthly_SID_master.csv"
ARIDITY_TABLE = ROOT / "All_the_Visualizations_REVISED_4PRODUCT_FINAL_ARIDITY_JUNE_DIAGNOSTICS" / "01_inputs_inventory" / "tower_aridity_annual_and_monthly.csv"
HARMONIC = ROOT / "SIF_EC_SIF_GPP_Final_Verification" / "02_independent_reverification" / "harmonic_residuals_full.csv"
OUT_DIR = ROOT / "All_the_Visualizations_REVISED" / "EXACT_REQUESTS_ONLY"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AI_CLASS_ORDER = ["Hyper-Arid", "Arid", "Semi-Arid", "Dry Sub-Humid", "Humid"]
AI_CLASS_COLORS = {
    "Hyper-Arid": "#7f2704", "Arid": "#d94801", "Semi-Arid": "#fd8d3c",
    "Dry Sub-Humid": "#41ab5d", "Humid": "#08519c",
}


def main():
    sid = pd.read_csv(SID_MASTER, low_memory=False,
                       usecols=["SITE_ID", "YEAR", "MONTH", "SIF_PRODUCT", "SID_VPD", "SID_SM", "SID_TEMP"])
    aridity = pd.read_csv(ARIDITY_TABLE)
    harm = pd.read_csv(HARMONIC, low_memory=False, usecols=["SITE_ID", "SIF_PRODUCT", "YEAR", "MONTH", "SIF_ANOM", "GPP_ANOM"])

    site_meta = aridity[["SITE_ID", "LATITUDE", "AI_ANNUAL", "AI_CLASS"]].copy()
    site_meta["HEMI"] = np.where(site_meta["LATITUDE"] >= 0, "NH", "SH")

    # ---- Panel A: site-level mean SID vs site AI_ANNUAL (unchanged logic from the original figure) ----
    sid["SID_mean"] = sid[["SID_VPD", "SID_SM", "SID_TEMP"]].mean(axis=1)
    sid_valid = sid.dropna(subset=["SID_mean"])
    site_sid = sid_valid.groupby("SITE_ID", as_index=False).agg(
        SID_SITE_MEAN=("SID_mean", "mean"), N_SID_MONTHS=("SID_mean", "size"))
    panel_a = site_sid.merge(site_meta, on="SITE_ID", how="inner").dropna(subset=["AI_ANNUAL"])
    rho_a, p_a = stats.spearmanr(panel_a["AI_ANNUAL"], panel_a["SID_SITE_MEAN"])
    assert len(panel_a) == 73, f"Panel A has {len(panel_a)} points, expected 73"

    # ---- Panel B: site-level mean SIF_ANOM vs site-level mean GPP_ANOM, colored by AI_CLASS
    #      (exactly the original panel B's two variables, averaged per site) ----
    harm_valid = harm.dropna(subset=["SIF_ANOM", "GPP_ANOM"])
    site_anom = harm_valid.groupby("SITE_ID", as_index=False).agg(
        SIF_ANOM_SITE_MEAN=("SIF_ANOM", "mean"), GPP_ANOM_SITE_MEAN=("GPP_ANOM", "mean"),
        N_SITE_MONTHS=("SIF_ANOM", "size"))
    panel_b = site_anom.merge(site_meta, on="SITE_ID", how="inner").dropna(subset=["AI_CLASS"])

    all_73_sites = set(panel_a["SITE_ID"])
    missing_from_panel_b = sorted(all_73_sites - set(panel_b["SITE_ID"]))
    print(f"Panel A: {len(panel_a)} of 73 towers (SID + aridity data).")
    print(f"Panel B: {len(panel_b)} of 73 towers have SIF/GPP anomaly data in this project. "
          f"Missing (zero rows anywhere in the anomaly table): {missing_from_panel_b}")

    # ---- Plot: same 2-panel layout, colors, and axis meaning as the original REVISED figure ----
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.0, 5.3), facecolor="white")
    hemi_marker = {"NH": "o", "SH": "^"}

    for cls in AI_CLASS_ORDER:
        for hemi in ("NH", "SH"):
            sub = panel_a[(panel_a["AI_CLASS"] == cls) & (panel_a["HEMI"] == hemi)]
            if len(sub) == 0:
                continue
            axA.scatter(sub["AI_ANNUAL"], sub["SID_SITE_MEAN"], s=18 + 0.10 * sub["N_SID_MONTHS"],
                       color=AI_CLASS_COLORS[cls], marker=hemi_marker[hemi],
                       edgecolor="#333333", linewidth=0.6, alpha=0.85, zorder=3)
    axA.set_xlabel("Site Aridity Index (AI_ANNUAL; lower = drier)", fontsize=9.5)
    axA.set_ylabel("Site-mean SID\n(mean of SID_VPD, SID_SM, SID_TEMP)", fontsize=9.5)
    axA.set_title("(a) SID vs. aridity — site level", fontsize=11, loc="left")
    axA.text(0.97, 0.95, f"Spearman \u03c1 = {rho_a:.2f} (p={p_a:.1e})\nn = {len(panel_a)} sites",
              transform=axA.transAxes, ha="right", va="top", fontsize=9, color="#333333")
    axA.spines[["top", "right"]].set_visible(False)
    axA.axhline(0, color="#cccccc", lw=0.6, zorder=0)

    for cls in AI_CLASS_ORDER:
        sub = panel_b[panel_b["AI_CLASS"] == cls]
        if len(sub) == 0:
            continue
        axB.scatter(sub["SIF_ANOM_SITE_MEAN"], sub["GPP_ANOM_SITE_MEAN"], s=18 + 0.10 * sub["N_SITE_MONTHS"],
                   color=AI_CLASS_COLORS[cls], edgecolor="#333333", linewidth=0.6, alpha=0.85,
                   zorder=3, label=f"{cls} (n={len(sub)})")
    axB.axhline(0, color="#bbbbbb", lw=0.6, zorder=0)
    axB.axvline(0, color="#bbbbbb", lw=0.6, zorder=0)
    axB.set_xlabel("Site-mean SIF anomaly (harmonic-residual)", fontsize=9.5)
    axB.set_ylabel("Site-mean GPP anomaly (harmonic-residual)", fontsize=9.5)
    axB.set_title(f"(b) GPP anomaly vs. SIF anomaly, by aridity class — site level (n={len(panel_b)} of 73 towers)",
                 fontsize=10.5, loc="left")
    axB.legend(fontsize=8, loc="upper left", frameon=True, facecolor="white", framealpha=0.9, edgecolor="none")
    axB.spines[["top", "right"]].set_visible(False)

    class_handles = [plt.Line2D([0], [0], marker="o", color="none", mfc=AI_CLASS_COLORS[c], mec="#333333",
                                 ms=8, label=c) for c in AI_CLASS_ORDER]
    hemi_handles = [plt.Line2D([0], [0], marker=hemi_marker[h], color="none", mfc="#888888", mec="#333333",
                                ms=8, label=f"{h} site (panel a only)") for h in ("NH", "SH")]
    fig.legend(handles=hemi_handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08),
               frameon=False, fontsize=7.6)

    fig.suptitle("Coupling / disagreement vs. aridity — SITE LEVEL (each dot = one EC tower's mean)",
                 fontsize=12.5, y=1.03)
    fig.text(0.5, -0.16,
              f"Same two panels and variables as Diagnostic_Coupling_vs_Aridity_REVISED.png, now averaged to ONE "
              f"point per site (no other change to the figure's logic). (a): each tower's own mean SID "
              f"(Stress-Induced Decoupling, mean of SID_VPD/SID_SM/SID_TEMP, pooled across the 4 main-paper SIF "
              f"products and all years/months) against that tower's Aridity Index -- {len(panel_a)} of 73 towers. "
              f"(b): each tower's own mean SIF anomaly vs. mean GPP anomaly (harmonic-residual, pooled across "
              f"products/years), colored by aridity class -- {len(panel_b)} of 73 towers; "
              f"{', '.join(missing_from_panel_b) if missing_from_panel_b else 'none'} "
              f"{'are' if len(missing_from_panel_b) != 1 else 'is'} excluded because "
              f"{'they have' if len(missing_from_panel_b) != 1 else 'it has'} zero rows anywhere in this "
              f"project's SIF/GPP anomaly table (a pre-existing data gap, not a computation choice made here). "
              f"Marker size = N (site-months underlying that point).",
              ha="center", fontsize=7.3, color="#555555", wrap=True)
    fig.tight_layout()

    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT_DIR / f"Diagnostic_Coupling_vs_Aridity_SITE_LEVEL_FINAL.{ext}", dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    print(f"Panel A: Spearman rho(AI_ANNUAL, SID_site_mean) = {rho_a:.3f}, p={p_a:.2e}, n={len(panel_a)} sites")
    print("Saved Diagnostic_Coupling_vs_Aridity_SITE_LEVEL_FINAL.{png,svg,pdf}")

    panel_a.to_csv(OUT_DIR / "figure_data_FINAL_panelA.csv", index=False)
    panel_b.to_csv(OUT_DIR / "figure_data_FINAL_panelB.csv", index=False)


if __name__ == "__main__":
    main()
