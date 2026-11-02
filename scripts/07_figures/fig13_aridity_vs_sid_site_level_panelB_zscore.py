"""
TASK 9: fix ONLY the units shown on Panel B's axes. Per explicit instruction, this
figure (Diagnostic_Coupling_vs_Aridity_SITE_LEVEL_FINAL.png) is otherwise correct --
the only problem is that Panel B plots raw harmonic-residual site-mean anomalies,
which are ~1e-14 to 1e-17 in magnitude (confirmed directly from
figure_data_FINAL_panelB.csv: e.g. AT-Zoe SIF_ANOM_SITE_MEAN=3.57e-17) -- numerically
tiny, ugly scientific-notation axis units.

Fix: the SAME 71 site-level mean-anomaly values (same sites, same N, same colors, same
sizes, same Panel A, same everything else) are now expressed as a Z-SCORE ACROSS THE 71
SITES themselves (subtract the cross-site mean, divide by the cross-site standard
deviation of that same site-level statistic) -- NOT a per-observation or within-site
z-score (which would just reproduce the same ~0 degeneracy; an across-site z-score is
the only rescaling of an already-site-level quantity that is both non-degenerate and
leaves the relative pattern between sites completely unchanged, just on readable units).

Nothing else changed: same 71/73 coverage caveat, same two towers excluded for having
zero rows in the anomaly table, same aridity-class colors, same hemisphere markers,
same marker-size-by-N encoding, same Panel A.

Output: OVERWRITES Diagnostic_Coupling_vs_Aridity_SITE_LEVEL_FINAL.{png,svg,pdf} in
place (same filename) -- this is a units-only fix to an already-approved figure, not a
new analysis, so no new filename.
"""
# Original script filename (sanitized): task9_fix_panelB_units_zscore.py; public name: fig13_aridity_vs_sid_site_level_panelB_zscore.py
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
OUT_DIR = ROOT / "Figure_Fixes_Final_Deliverables" / "figures"
DATA_DIR = ROOT / "Figure_Fixes_Final_Deliverables" / "figure_data"
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

    # ---- Panel A: UNCHANGED ----
    sid["SID_mean"] = sid[["SID_VPD", "SID_SM", "SID_TEMP"]].mean(axis=1)
    sid_valid = sid.dropna(subset=["SID_mean"])
    site_sid = sid_valid.groupby("SITE_ID", as_index=False).agg(
        SID_SITE_MEAN=("SID_mean", "mean"), N_SID_MONTHS=("SID_mean", "size"))
    panel_a = site_sid.merge(site_meta, on="SITE_ID", how="inner").dropna(subset=["AI_ANNUAL"])
    rho_a, p_a = stats.spearmanr(panel_a["AI_ANNUAL"], panel_a["SID_SITE_MEAN"])
    assert len(panel_a) == 73, f"Panel A has {len(panel_a)} points, expected 73"

    # ---- Panel B: same site-level means, now rescaled as a cross-site z-score ----
    harm_valid = harm.dropna(subset=["SIF_ANOM", "GPP_ANOM"])
    site_anom = harm_valid.groupby("SITE_ID", as_index=False).agg(
        SIF_ANOM_SITE_MEAN=("SIF_ANOM", "mean"), GPP_ANOM_SITE_MEAN=("GPP_ANOM", "mean"),
        N_SITE_MONTHS=("SIF_ANOM", "size"))
    panel_b = site_anom.merge(site_meta, on="SITE_ID", how="inner").dropna(subset=["AI_CLASS"])

    # Cross-site z-score: subtract the mean ACROSS THE 71 SITE-LEVEL VALUES and divide by
    # their SD across sites -- this is NOT within-site/within-product (which is degenerate,
    # see docstring); it rescales the already-site-level statistic onto readable units
    # while leaving every site's RANK and RELATIVE spacing exactly as before.
    sif_mean_across, sif_sd_across = panel_b["SIF_ANOM_SITE_MEAN"].mean(), panel_b["SIF_ANOM_SITE_MEAN"].std(ddof=1)
    gpp_mean_across, gpp_sd_across = panel_b["GPP_ANOM_SITE_MEAN"].mean(), panel_b["GPP_ANOM_SITE_MEAN"].std(ddof=1)
    panel_b["SIF_ANOM_Z"] = (panel_b["SIF_ANOM_SITE_MEAN"] - sif_mean_across) / sif_sd_across
    panel_b["GPP_ANOM_Z"] = (panel_b["GPP_ANOM_SITE_MEAN"] - gpp_mean_across) / gpp_sd_across
    print(f"Cross-site rescaling: SIF site-means had cross-site mean={sif_mean_across:.3e}, SD={sif_sd_across:.3e}; "
          f"GPP site-means had cross-site mean={gpp_mean_across:.3e}, SD={gpp_sd_across:.3e}")

    all_73_sites = set(panel_a["SITE_ID"])
    missing_from_panel_b = sorted(all_73_sites - set(panel_b["SITE_ID"]))
    print(f"Panel A: {len(panel_a)} of 73 towers (SID + aridity data).")
    print(f"Panel B: {len(panel_b)} of 73 towers have SIF/GPP anomaly data in this project. "
          f"Missing (zero rows anywhere in the anomaly table): {missing_from_panel_b}")
    rho_b, p_b = stats.spearmanr(panel_b["SIF_ANOM_Z"], panel_b["GPP_ANOM_Z"])

    # ---- Plot: identical layout/colors/legend/Panel A to the original figure ----
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
    axA.set_title("(a) SID vs. aridity -- site level", fontsize=11, loc="left")
    axA.text(0.97, 0.95, f"Spearman rho = {rho_a:.2f} (p={p_a:.1e})\nn = {len(panel_a)} sites",
              transform=axA.transAxes, ha="right", va="top", fontsize=9, color="#333333")
    axA.spines[["top", "right"]].set_visible(False)
    axA.axhline(0, color="#cccccc", lw=0.6, zorder=0)

    for cls in AI_CLASS_ORDER:
        sub = panel_b[panel_b["AI_CLASS"] == cls]
        if len(sub) == 0:
            continue
        axB.scatter(sub["SIF_ANOM_Z"], sub["GPP_ANOM_Z"], s=18 + 0.10 * sub["N_SITE_MONTHS"],
                   color=AI_CLASS_COLORS[cls], edgecolor="#333333", linewidth=0.6, alpha=0.85,
                   zorder=3, label=f"{cls} (n={len(sub)})")
    axB.axhline(0, color="#bbbbbb", lw=0.6, zorder=0)
    axB.axvline(0, color="#bbbbbb", lw=0.6, zorder=0)
    axB.set_xlabel("SIF anomaly, z-score across sites", fontsize=9.5)
    axB.set_ylabel("GPP anomaly, z-score across sites", fontsize=9.5)
    axB.set_title(f"(b) GPP vs. SIF anomaly (z-score), by aridity class -- site level (n={len(panel_b)} of 73 towers)",
                 fontsize=10.3, loc="left")
    axB.text(0.97, 0.05, f"Spearman rho = {rho_b:.2f} (p={p_b:.1e})",
              transform=axB.transAxes, ha="right", va="bottom", fontsize=9, color="#333333",
              bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="#999999", linewidth=0.5))
    axB.legend(fontsize=8, loc="upper left", frameon=True, facecolor="white", framealpha=0.9, edgecolor="none")
    axB.spines[["top", "right"]].set_visible(False)

    hemi_handles = [plt.Line2D([0], [0], marker=hemi_marker[h], color="none", mfc="#888888", mec="#333333",
                                ms=8, label=f"{h} site (panel a only)") for h in ("NH", "SH")]
    fig.legend(handles=hemi_handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08),
               frameon=False, fontsize=7.6)

    fig.suptitle("Coupling / disagreement vs. aridity -- SITE LEVEL (each dot = one EC tower's mean)",
                 fontsize=12.5, y=1.03)
    fig.text(0.5, -0.17,
              f"Same two panels, same sites, same colors/sizes as before (no other change). (a): each tower's own "
              f"mean SID vs. Aridity Index -- {len(panel_a)} of 73 towers. (b): UNITS FIX ONLY -- the same "
              f"site-mean SIF/GPP harmonic-residual anomalies as before (these are genuinely tiny in their native "
              f"units, ~1e-14 to 1e-17, since a harmonic-residual series averages to ~0 over its own fit) are now "
              f"expressed as a z-score ACROSS the {len(panel_b)} sites themselves (subtract the cross-site mean, "
              f"divide by the cross-site SD of that same site-level statistic) -- every site's rank and relative "
              f"spacing is identical to before, only the axis units changed from the native tiny scale to a "
              f"readable z-score scale. {', '.join(missing_from_panel_b) if missing_from_panel_b else 'None'} "
              f"{'are' if len(missing_from_panel_b) != 1 else 'is'} excluded (zero rows anywhere in the anomaly "
              f"table, a pre-existing data gap). Marker size = N (site-months underlying that point).",
              ha="center", fontsize=7.2, color="#555555", wrap=True)
    fig.tight_layout()

    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT_DIR / f"Diagnostic_Coupling_vs_Aridity_SITE_LEVEL_FINAL.{ext}", dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    print(f"\nPanel A: Spearman rho(AI_ANNUAL, SID_site_mean) = {rho_a:.3f}, p={p_a:.2e}, n={len(panel_a)} sites")
    print(f"Panel B: Spearman rho(SIF_ANOM_Z, GPP_ANOM_Z) = {rho_b:.3f}, p={p_b:.2e}, n={len(panel_b)} sites")
    print("Saved (overwritten in place) Diagnostic_Coupling_vs_Aridity_SITE_LEVEL_FINAL.{png,svg,pdf}")

    panel_a.to_csv(DATA_DIR / "figure_data_FINAL_panelA.csv", index=False)
    panel_b.to_csv(DATA_DIR / "figure_data_FINAL_panelB.csv", index=False)


if __name__ == "__main__":
    main()
