"""
TASK 5 (reviewer exact-requests-only revision):
Revise Figure2_Aridity_Explains_Disagreement.png -- keep ONLY the per-biome forest plot
(old panel b); remove the old pooled scatter (old panel a) entirely; and switch the
decoupling metric from the old plain, non-stress-weighted "Decoupling" column to a
SID-based metric, stated clearly in the figure.

SID definition used (same convention as Task 2, for consistency across this revision
batch): SID_mean = mean(SID_VPD, SID_SM, SID_TEMP), the three stressor-specific
Stress-Induced-Decoupling values already computed and validated in
SIF_EC_Stress_Induced_Decoupling_SID, pooled across the 4 main-paper SIF products and
all years at the site-month level. This REPLACES the old plain "Decoupling" metric
(|SIFnorm-GPPnorm|/(SIFnorm+GPPnorm), unweighted by any stressor) used in the original
Figure2 -- stated explicitly in the figure title/caption per rule 4.

Reuses, unmodified, from the original build_disagreement_aridity_figures.py:
IGBP_LABELS, RELIABLE_N_SITEMONTHS, RELIABLE_N_TOWERS, biome_order() logic (driest-first
ordering by mean annual Aridity Index), and the tower-block bootstrap CI method for the
forest plot. Only the input metric (SID_mean vs. Decoupling) and the dropped pooled-
scatter panel differ. Data reused as-is, nothing recomputed upstream:
  - SIF_EC_Stress_Induced_Decoupling_SID/03_sid_monthly/monthly_SID_master.csv
  - All_the_Visualizations_REVISED_4PRODUCT_FINAL_ARIDITY_JUNE_DIAGNOSTICS/
    01_inputs_inventory/tower_aridity_annual_and_monthly.csv
"""
# Original script filename (sanitized): task5_build_sid_biome_forest.py; public name: figS8_aridity_sid_biome_forest.py
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
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(PROJECT_ROOT)
SID_MASTER = ROOT / "SIF_EC_Stress_Induced_Decoupling_SID" / "03_sid_monthly" / "monthly_SID_master.csv"
ARIDITY_TABLE = ROOT / "All_the_Visualizations_REVISED_4PRODUCT_FINAL_ARIDITY_JUNE_DIAGNOSTICS" / "01_inputs_inventory" / "tower_aridity_annual_and_monthly.csv"
SRC_SCRIPT_DIR = ROOT / "SIF_GPP_Severe_Disagreement_Aridity_Diagnosis"
OUT_DIR = ROOT / "All_the_Visualizations_REVISED" / "EXACT_REQUESTS_ONLY"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC_SCRIPT_DIR))
import importlib
src = importlib.import_module("build_disagreement_aridity_figures")  # noqa: E402

IGBP_LABELS = src.IGBP_LABELS
RELIABLE_N_SITEMONTHS = src.RELIABLE_N_SITEMONTHS
RELIABLE_N_TOWERS = src.RELIABLE_N_TOWERS


def load_sid_sitemonth() -> pd.DataFrame:
    sid = pd.read_csv(SID_MASTER, low_memory=False,
                       usecols=["SITE_ID", "YEAR", "MONTH", "SIF_PRODUCT", "IGBP",
                                "SID_VPD", "SID_SM", "SID_TEMP"])
    sid["SID_mean"] = sid[["SID_VPD", "SID_SM", "SID_TEMP"]].mean(axis=1)
    aridity = pd.read_csv(ARIDITY_TABLE)

    sm = sid.groupby(["SITE_ID", "YEAR", "MONTH", "IGBP"], as_index=False)["SID_mean"].mean()
    sm = sm.merge(aridity[["SITE_ID", "AI_ANNUAL"] + [f"AI_M{m:02d}" for m in range(1, 13)]],
                  on="SITE_ID", how="left")
    sm["AI_this_month"] = sm.apply(lambda r: r[f"AI_M{int(r['MONTH']):02d}"], axis=1)
    sm = sm.drop(columns=[f"AI_M{m:02d}" for m in range(1, 13)])
    return sm


def biome_order(sm: pd.DataFrame) -> list[str]:
    means = sm.groupby("IGBP")["AI_ANNUAL"].mean().sort_values()  # driest first
    return [b for b in means.index if b in IGBP_LABELS]


def build_forest_table(sm: pd.DataFrame, biomes: list[str]) -> pd.DataFrame:
    rows = []
    for b in biomes:
        sub = sm[sm["IGBP"] == b].dropna(subset=["AI_this_month", "SID_mean"])
        n_towers = sub["SITE_ID"].nunique()
        n = len(sub)
        if n_towers < RELIABLE_N_TOWERS or n < RELIABLE_N_SITEMONTHS:
            rows.append({"IGBP": b, "N": n, "N_towers": n_towers, "rho": np.nan,
                         "ci_low": np.nan, "ci_high": np.nan, "reliable": False})
            continue
        rho, p = stats.spearmanr(sub["AI_this_month"], sub["SID_mean"])
        towers = sub["SITE_ID"].unique()
        boots = []
        rng = np.random.default_rng(20260907)
        for _ in range(1000):
            pick = rng.choice(towers, size=len(towers), replace=True)
            bsub = sub[sub["SITE_ID"].isin(pick)]
            if bsub["AI_this_month"].nunique() < 2:
                continue
            r, _ = stats.spearmanr(bsub["AI_this_month"], bsub["SID_mean"])
            if np.isfinite(r):
                boots.append(r)
        ci_low, ci_high = (np.percentile(boots, [2.5, 97.5]) if len(boots) > 100 else (np.nan, np.nan))
        rows.append({"IGBP": b, "N": n, "N_towers": n_towers, "rho": rho,
                     "ci_low": ci_low, "ci_high": ci_high, "reliable": True, "p": p})
    return pd.DataFrame(rows)


def main():
    sm = load_sid_sitemonth()
    biomes = biome_order(sm)
    forest = build_forest_table(sm, biomes)
    forest.to_csv(OUT_DIR / "biome_aridity_SID_correlation.csv", index=False)

    plot_forest = forest.dropna(subset=["rho"]).sort_values("rho")
    y = np.arange(len(plot_forest))

    fig, ax = plt.subplots(figsize=(8.0, 5.2), facecolor="white")
    ax.errorbar(plot_forest["rho"], y,
                xerr=[plot_forest["rho"] - plot_forest["ci_low"], plot_forest["ci_high"] - plot_forest["rho"]],
                fmt="o", color="#8c2d04", ecolor="#8c2d04", elinewidth=1.4, capsize=3, ms=6)
    ax.axvline(0, color="#999999", lw=0.9, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{IGBP_LABELS[b]}  (n={int(nt)} towers)"
                         for b, nt in zip(plot_forest["IGBP"], plot_forest["N_towers"])], fontsize=9.5)
    ax.set_xlabel("Spearman \u03c1 (monthly Aridity Index vs. SID)", fontsize=10.5)
    ax.set_title("Per-biome association between aridity and Stress-Induced Decoupling (SID),\nwith tower-bootstrap 95% CI",
                 fontsize=12, loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    omitted = forest[~forest["reliable"]]["IGBP"].tolist()
    footer = ("SID = Stress-Induced Decoupling (mean of SID_VPD, SID_SM, SID_TEMP; reused unmodified from "
              "SIF_EC_Stress_Induced_Decoupling_SID), pooled across the 4 main-paper SIF products and all years. "
              "This REPLACES the earlier plain, non-stress-weighted 'Decoupling' metric used in the original "
              "Figure2 -- the old pooled-scatter panel (a) has been removed; only the per-biome forest plot "
              "(old panel b) is kept, now SID-based. Negative \u03c1 = drier months show more decoupling. "
              "Descriptive, cross-site association -- not a causal test.")
    if omitted:
        footer += (" Omitted for insufficient support (<{} towers or <{} site-months): ".format(
            RELIABLE_N_TOWERS, RELIABLE_N_SITEMONTHS) + ", ".join(IGBP_LABELS[b] for b in omitted) + ".")
    fig.text(0.5, -0.10, footer, ha="center", fontsize=7.8, color="#555555", wrap=True)

    fig.suptitle("Does aridity explain the severity of SIF-GPP decoupling? (SID-based)", fontsize=13, y=1.04)
    fig.tight_layout()

    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT_DIR / f"Figure2_Aridity_Explains_Disagreement_REVISED.{ext}", dpi=300,
                    facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("Saved Figure2_Aridity_Explains_Disagreement_REVISED.{png,svg,pdf}")
    print(forest[["IGBP", "N_towers", "N", "rho", "ci_low", "ci_high"]].to_string(index=False))


if __name__ == "__main__":
    main()
