"""
STEP 7 -- Hypotheses B (spring temperature activation), C (summer VPD / water
stress), and H (aridity), analyzed together via WITHIN-TOWER (fixed-effect) year-to-
year comparison, per section 24's explicit instruction: "use within-tower anomalies /
fixed effects so that differences in climate, biome, latitude, and instrumentation are
controlled naturally."

For each SITE_ID x YEAR x SIF_PRODUCT row with a valid GPP_POS, we define the
PRE/AROUND-PEAK WINDOW as the 3 calendar months ending at that year's GPP_POS
(rounded to the nearest month, circularly) and average each environmental anomaly
variable (TAIR_ANOM, VPD_ANOM, SWC_ANOM -- already deseasonalized site-specific
anomalies, reused from advanced_analysis_input_inventory.csv) and SID metric
(SID_VPD, SID_SM, SID_TEMP -- reused from monthly_SID_master.csv) over that window.
This captures "conditions before/during/around the seasonal peak," not an annual mean,
per the explicit instruction to focus on peak-proximate conditions.

WITHIN-TOWER CENTERING: both DELTA_POS and each environmental predictor are centered
on their own SITE_ID's multi-year mean before correlating, so the comparison is
"in years when THIS tower was relatively warmer/drier/more stressed than its own
average, was its SIF-GPP peak offset also different from its own average" -- exactly
the year-to-year, within-site question requested, immune to cross-tower confounds
(biome, latitude, climate) by construction.

Climatological aridity (Hypothesis H) enters as a MODERATOR: the within-tower
VPD/SWC-vs-DeltaPOS relationship is reported separately for Arid/Semi-Arid vs.
Humid/Sub-Humid tower classes (AI_CLASS, from tower_aridity_annual_and_monthly.csv),
combining the climatological aridity class with the year-specific stress anomaly, per
the explicit instruction to test both together rather than aridity class alone.
"""
# Original script filename (sanitized): step7_within_tower_environment.py; public name: step7_within_tower_environment.py
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
from scipy.stats import spearmanr, wilcoxon
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(PROJECT_ROOT)
ANALYSIS_ROOT = (Path(PROJECT_ROOT) / 'SIF_GPP_PEAK_TIMING_MECHANISM_FULL_ANALYSIS/14_scripts/step7_within_tower_environment.py').resolve().parents[1]
INVENTORY_CSV = ROOT / "SIF_EC_Advanced_Coupling_Analysis" / "01_input_audit" / "advanced_analysis_input_inventory.csv"
SID_CSV = ROOT / "SIF_EC_Stress_Induced_Decoupling_SID" / "03_sid_monthly" / "monthly_SID_master.csv"
ARIDITY_CSV = ROOT / "All_the_Visualizations_REVISED_4PRODUCT_FINAL_ARIDITY_JUNE_DIAGNOSTICS" / "01_inputs_inventory" / "tower_aridity_annual_and_monthly.csv"
MASTER_CSV = ANALYSIS_ROOT / "02_peak_timing" / "SiteYear_Product_PeakTiming_Master.csv"

FIG_DIR = ANALYSIS_ROOT / "11_figures"
OUT_TEMP = ANALYSIS_ROOT / "05_temperature_activation"
OUT_WATER = ANALYSIS_ROOT / "06_water_vpd_stress"
OUT_YEARTOYEAR = ANALYSIS_ROOT / "08_year_to_year_analysis"

EVERGREEN = ["ENF", "EBF"]


def circ_round_month(x: float) -> int:
    m = int(round(x)) % 12
    return 12 if m == 0 else m


def window_months(center: int, back: int = 2) -> list:
    return [((center - k - 1) % 12) + 1 for k in range(back + 1)]


def main():
    master = pd.read_csv(MASTER_CSV)
    inv = pd.read_csv(INVENTORY_CSV, low_memory=False)
    inv["MONTH"] = inv["MONTH"].astype(int)
    inv["YEAR"] = inv["YEAR"].astype(int)
    env = inv.groupby(["SITE_ID", "YEAR", "MONTH"])[["TAIR_ANOM", "VPD_ANOM", "SWC_ANOM"]].first().reset_index()

    sid = pd.read_csv(SID_CSV, low_memory=False, usecols=["SITE_ID", "YEAR", "MONTH", "SIF_PRODUCT", "SID_VPD", "SID_SM", "SID_TEMP"])
    aridity = pd.read_csv(ARIDITY_CSV, usecols=["SITE_ID", "AI_ANNUAL", "AI_CLASS"])

    valid = master.dropna(subset=["DELTA_POS", "GPP_POS"]).copy()
    valid["PEAK_MONTH_ROUND"] = valid["GPP_POS"].apply(circ_round_month)

    rows = []
    for _, r in valid.iterrows():
        wmonths = window_months(r["PEAK_MONTH_ROUND"])
        e = env[(env["SITE_ID"] == r["SITE_ID"]) & (env["YEAR"] == r["YEAR"]) & (env["MONTH"].isin(wmonths))]
        s = sid[(sid["SITE_ID"] == r["SITE_ID"]) & (sid["YEAR"] == r["YEAR"]) & (sid["SIF_PRODUCT"] == r["SIF_PRODUCT"]) & (sid["MONTH"].isin(wmonths))]
        rows.append({
            "SITE_ID": r["SITE_ID"], "YEAR": r["YEAR"], "SIF_PRODUCT": r["SIF_PRODUCT"], "IGBP": r["IGBP"],
            "DELTA_POS": r["DELTA_POS"], "DELTA_CLASS": r["DELTA_CLASS"],
            "PREPEAK_TAIR_ANOM": e["TAIR_ANOM"].mean() if len(e) else np.nan,
            "PREPEAK_VPD_ANOM": e["VPD_ANOM"].mean() if len(e) else np.nan,
            "PREPEAK_SWC_ANOM": e["SWC_ANOM"].mean() if len(e) else np.nan,
            "PREPEAK_SID_VPD": s["SID_VPD"].mean() if len(s) else np.nan,
            "PREPEAK_SID_SM": s["SID_SM"].mean() if len(s) else np.nan,
            "PREPEAK_SID_TEMP": s["SID_TEMP"].mean() if len(s) else np.nan,
        })
    joined = pd.DataFrame(rows).merge(aridity, on="SITE_ID", how="left")

    # within-tower centering
    predictors = ["DELTA_POS", "PREPEAK_TAIR_ANOM", "PREPEAK_VPD_ANOM", "PREPEAK_SWC_ANOM",
                  "PREPEAK_SID_VPD", "PREPEAK_SID_SM", "PREPEAK_SID_TEMP"]
    for col in predictors:
        joined[f"WT_{col}"] = joined[col] - joined.groupby("SITE_ID")[col].transform("mean")
    joined.to_csv(OUT_YEARTOYEAR / "within_tower_environment_deltapos.csv", index=False)

    # correlation summary: within-tower DELTA_POS vs each within-tower predictor,
    # for (a) all biomes, (b) evergreen only, (c) non-evergreen, and by aridity class
    env_predictors = ["PREPEAK_TAIR_ANOM", "PREPEAK_VPD_ANOM", "PREPEAK_SWC_ANOM",
                      "PREPEAK_SID_VPD", "PREPEAK_SID_SM", "PREPEAK_SID_TEMP"]
    groups = {
        "ALL_BIOMES": joined,
        "EVERGREEN_ONLY": joined[joined["IGBP"].isin(EVERGREEN)],
        "NON_EVERGREEN": joined[~joined["IGBP"].isin(EVERGREEN)],
        "ARID_OR_SEMIARID": joined[joined["AI_CLASS"].isin(["Arid", "Semi-Arid", "Hyper-Arid"])],
        "HUMID_OR_SUBHUMID": joined[joined["AI_CLASS"].isin(["Humid", "Dry Sub-Humid"])],
    }
    summary_rows = []
    for gname, g in groups.items():
        for pred in env_predictors:
            sub = g.dropna(subset=[f"WT_{pred}", "WT_DELTA_POS"])
            n_towers = sub["SITE_ID"].nunique()
            if len(sub) < 15 or n_towers < 5:
                summary_rows.append({"GROUP": gname, "PREDICTOR": pred, "N": len(sub), "N_TOWERS": n_towers,
                                     "RHO": np.nan, "P": np.nan, "RELIABLE": False})
                continue
            rho, p = spearmanr(sub[f"WT_{pred}"], sub["WT_DELTA_POS"])
            summary_rows.append({"GROUP": gname, "PREDICTOR": pred, "N": len(sub), "N_TOWERS": n_towers,
                                 "RHO": round(float(rho), 3), "P": round(float(p), 4), "RELIABLE": True})
    corr_summary = pd.DataFrame(summary_rows)
    corr_summary.to_csv(OUT_YEARTOYEAR / "within_tower_correlation_summary.csv", index=False)
    corr_summary[corr_summary["PREDICTOR"].isin(["PREPEAK_TAIR_ANOM"])].to_csv(OUT_TEMP / "hypothesisB_temperature_summary.csv", index=False)
    corr_summary[corr_summary["PREDICTOR"].str.contains("VPD|SWC|SM")].to_csv(OUT_WATER / "hypothesisC_H_water_summary.csv", index=False)
    print(corr_summary.to_string(index=False))

    # paired within-tower comparison: EARLIER_SIF years vs. other years (section 24)
    paired_rows = []
    for pred in env_predictors:
        tower_means_earlier, tower_means_other = [], []
        for site, g in joined.groupby("SITE_ID"):
            e = g[g["DELTA_CLASS"] == "EARLIER_SIF"][pred].dropna()
            o = g[g["DELTA_CLASS"] != "EARLIER_SIF"][pred].dropna()
            if len(e) >= 1 and len(o) >= 1:
                tower_means_earlier.append(e.mean())
                tower_means_other.append(o.mean())
        if len(tower_means_earlier) >= 5:
            diffs = np.array(tower_means_earlier) - np.array(tower_means_other)
            try:
                stat, p = wilcoxon(diffs)
            except ValueError:
                p = np.nan
            paired_rows.append({"PREDICTOR": pred, "N_TOWERS_WITH_BOTH": len(tower_means_earlier),
                                "MEAN_DIFF_EARLIER_SIF_MINUS_OTHER": round(float(diffs.mean()), 3), "WILCOXON_P": p})
    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(OUT_YEARTOYEAR / "paired_earlierSIF_vs_other_years.csv", index=False)
    print("\nPaired within-tower: EARLIER_SIF years vs. other years, mean predictor difference:")
    print(paired_df.to_string(index=False))

    # Figure: within-tower correlation coefficient plot (all-biomes + evergreen)
    fig, ax = plt.subplots(figsize=(8.5, 5.5), facecolor="white")
    plot_groups = ["ALL_BIOMES", "EVERGREEN_ONLY", "ARID_OR_SEMIARID"]
    colors = {"ALL_BIOMES": "#555555", "EVERGREEN_ONLY": "#1b7837", "ARID_OR_SEMIARID": "#b2182b"}
    y_base = np.arange(len(env_predictors))
    offsets = {"ALL_BIOMES": -0.22, "EVERGREEN_ONLY": 0.0, "ARID_OR_SEMIARID": 0.22}
    for gname in plot_groups:
        sub = corr_summary[(corr_summary["GROUP"] == gname) & (corr_summary["RELIABLE"])]
        sub = sub.set_index("PREDICTOR").reindex(env_predictors)
        ypos = y_base + offsets[gname]
        ax.scatter(sub["RHO"], ypos, color=colors[gname], s=70, label=gname.replace("_", " ").title(), zorder=3)
        for yy, rho, p in zip(ypos, sub["RHO"], sub["P"]):
            if not np.isnan(rho) and p < 0.05:
                ax.scatter([rho], [yy], facecolor="none", edgecolor=colors[gname], s=160, linewidth=1.8, zorder=4)
    ax.axvline(0, color="#999999", lw=0.9, ls="--")
    ax.set_yticks(y_base)
    ax.set_yticklabels(["Spring TAIR anomaly", "VPD anomaly", "SWC anomaly", "SID (VPD)", "SID (soil moisture)", "SID (temperature)"], fontsize=9.5)
    ax.set_xlabel("Within-tower Spearman \u03c1 (predictor anomaly vs. \u0394POS)\nlarge circle = p<0.05", fontsize=9.5)
    ax.set_title("Year-to-year: does a tower's own unusual conditions shift its SIF-GPP peak offset?",
                fontsize=12, fontweight="bold", loc="left")
    ax.legend(fontsize=8.5, loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "svg", "pdf"):
        dpi = 600 if ext == "png" else 300
        fig.savefig(FIG_DIR / f"Figure5_WithinTower_Environment_Correlates.{ext}", dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("Saved Figure5_WithinTower_Environment_Correlates")


if __name__ == "__main__":
    main()
