"""
STEP 10 -- Transparent statistical modeling of DELTA_POS.

Cluster-robust OLS (clustered by SITE_ID) is used throughout rather than MixedLM,
following this project's own established, previously-audited practice: the SEM audit
in this same project (SIF_EC_SEM_Environmental_Control_Audit) found MixedLM's
random-tower-intercept variance collapses to ~0 in effectively every equation once
within-site structure is otherwise accounted for (0 of 112 equations there trusted
MixedLM), with cluster-robust OLS as the documented, defensible fallback used
everywhere else in this project. The same reasoning applies here (many models already
include IGBP/product fixed effects), so cluster-robust OLS is used directly rather
than re-deriving the same collapse-then-fallback check for every model in this
analysis.

Predictor groups evaluated SEPARATELY first (per instruction), then combined only
after checking collinearity (VIF):
  A. canopy/light   -- SW_IN_GAP_TO_GPP (signed: SW_IN_POS - GPP_POS, circular)
  B. temperature    -- PREPEAK_TAIR_ANOM
  C. VPD/water      -- PREPEAK_VPD_ANOM, PREPEAK_SWC_ANOM, PREPEAK_SID_VPD, PREPEAK_SID_SM
  D. biome/phenology-- IGBP (categorical)
  E. methodological -- TOTAL_N_NATIVE_OBS (sampling density), SIF_PRODUCT (categorical)

All continuous predictors are standardized (z-scored) before fitting so coefficients
are directly comparable in magnitude. This is an ASSOCIATION analysis; no causal claim
is made anywhere in this script or its outputs.
"""
# Original script filename (sanitized): step10_statistical_modeling.py; public name: step10_statistical_modeling.py
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
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(PROJECT_ROOT)
ANALYSIS_ROOT = (Path(PROJECT_ROOT) / 'SIF_GPP_PEAK_TIMING_MECHANISM_FULL_ANALYSIS/14_scripts/step10_statistical_modeling.py').resolve().parents[1]
MASTER_CSV = ANALYSIS_ROOT / "02_peak_timing" / "SiteYear_Product_PeakTiming_Master.csv"
WT_CSV = ANALYSIS_ROOT / "08_year_to_year_analysis" / "within_tower_environment_deltapos.csv"
SUPPORT_CSV = ANALYSIS_ROOT / "09_methodological_artifacts"
OUT_DIR = ANALYSIS_ROOT / "10_mechanism_attribution"
FIG_DIR = ANALYSIS_ROOT / "11_figures"


def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=1)


def fit_cluster_robust(formula: str, data: pd.DataFrame):
    model = smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data["SITE_ID"]})
    return model


def main():
    master = pd.read_csv(MASTER_CSV).dropna(subset=["DELTA_POS", "SW_IN_POS", "GPP_POS"])
    master["SW_IN_GAP_TO_GPP"] = master.apply(lambda r: (r["SW_IN_POS"] - r["GPP_POS"] + 6) % 12 - 6, axis=1)
    wt = pd.read_csv(WT_CSV)[["SITE_ID", "YEAR", "SIF_PRODUCT", "PREPEAK_TAIR_ANOM", "PREPEAK_VPD_ANOM",
                              "PREPEAK_SWC_ANOM", "PREPEAK_SID_VPD", "PREPEAK_SID_SM"]]
    sampling = pd.read_csv(SUPPORT_CSV / "artifactA_satellite_sampling_vs_absdeltapos.csv")  # per-product summary only; need per-row -- recompute lightweight join instead
    harm = pd.read_csv(ROOT / "SIF_EC_SIF_GPP_Final_Verification" / "02_independent_reverification" / "harmonic_residuals_full.csv", low_memory=False)
    supp = harm.groupby(["SITE_ID", "YEAR", "SIF_PRODUCT"])["N_NATIVE_OBSERVATIONS"].sum().reset_index().rename(
        columns={"N_NATIVE_OBSERVATIONS": "TOTAL_N_NATIVE_OBS"})

    df = master.merge(wt, on=["SITE_ID", "YEAR", "SIF_PRODUCT"], how="left").merge(
        supp, on=["SITE_ID", "YEAR", "SIF_PRODUCT"], how="left")

    for col in ["SW_IN_GAP_TO_GPP", "PREPEAK_TAIR_ANOM", "PREPEAK_VPD_ANOM", "PREPEAK_SWC_ANOM",
                "PREPEAK_SID_VPD", "PREPEAK_SID_SM", "TOTAL_N_NATIVE_OBS"]:
        df[f"Z_{col}"] = zscore(df[col])

    model_specs = {
        "A_canopy_light": "DELTA_POS ~ Z_SW_IN_GAP_TO_GPP",
        "B_temperature": "DELTA_POS ~ Z_PREPEAK_TAIR_ANOM",
        "C_water_stress": "DELTA_POS ~ Z_PREPEAK_VPD_ANOM + Z_PREPEAK_SWC_ANOM + Z_PREPEAK_SID_VPD + Z_PREPEAK_SID_SM",
        "D_biome": "DELTA_POS ~ C(IGBP)",
        "E_methodological": "DELTA_POS ~ Z_TOTAL_N_NATIVE_OBS + C(SIF_PRODUCT)",
    }

    results_rows = []
    for name, formula in model_specs.items():
        needed = [w.strip() for w in formula.replace("DELTA_POS ~ ", "").split("+")]
        needed_cols = []
        for n in needed:
            if n.startswith("C("):
                needed_cols.append(n[2:-1])
            else:
                needed_cols.append(n)
        sub = df.dropna(subset=needed_cols + ["DELTA_POS"])
        if len(sub) < 30 or sub["SITE_ID"].nunique() < 5:
            continue
        m = fit_cluster_robust(formula, sub)
        for term in m.params.index:
            if term == "Intercept":
                continue
            results_rows.append({"MODEL": name, "TERM": term, "COEF": round(float(m.params[term]), 4),
                                 "SE": round(float(m.bse[term]), 4), "P": round(float(m.pvalues[term]), 4),
                                 "N": int(m.nobs), "N_TOWERS": sub["SITE_ID"].nunique(), "R2": round(float(m.rsquared), 4)})
    group_results = pd.DataFrame(results_rows)
    group_results.to_csv(OUT_DIR / "predictor_group_models.csv", index=False)
    print("=== Predictor-group models (cluster-robust by tower) ===")
    print(group_results.to_string(index=False))

    # --- Collinearity check (numeric predictors only) before combined model ---
    numeric_cols = ["Z_SW_IN_GAP_TO_GPP", "Z_PREPEAK_TAIR_ANOM", "Z_PREPEAK_VPD_ANOM",
                    "Z_PREPEAK_SWC_ANOM", "Z_PREPEAK_SID_SM", "Z_TOTAL_N_NATIVE_OBS"]
    combined_data = df.dropna(subset=numeric_cols + ["DELTA_POS", "IGBP", "SIF_PRODUCT"])
    X = combined_data[numeric_cols].to_numpy()
    vif = pd.DataFrame({"PREDICTOR": numeric_cols,
                        "VIF": [variance_inflation_factor(X, i) for i in range(X.shape[1])]})
    vif.to_csv(OUT_DIR / "combined_model_VIF.csv", index=False)
    print("\n=== VIF (combined model numeric predictors) ===")
    print(vif.to_string(index=False))
    high_vif = vif[vif["VIF"] > 5]["PREDICTOR"].tolist()
    if high_vif:
        print(f"WARNING: dropping high-VIF predictors from combined model: {high_vif}")
        numeric_cols = [c for c in numeric_cols if c not in high_vif]

    combined_formula = "DELTA_POS ~ " + " + ".join(numeric_cols) + " + C(IGBP) + C(SIF_PRODUCT)"
    combined_model = fit_cluster_robust(combined_formula, combined_data)
    combined_rows = []
    for term in combined_model.params.index:
        if term == "Intercept":
            continue
        combined_rows.append({"TERM": term, "COEF": round(float(combined_model.params[term]), 4),
                              "SE": round(float(combined_model.bse[term]), 4), "P": round(float(combined_model.pvalues[term]), 4)})
    combined_df = pd.DataFrame(combined_rows)
    combined_df.to_csv(OUT_DIR / "combined_model_coefficients.csv", index=False)
    with open(OUT_DIR / "combined_model_fit_stats.txt", "w") as f:
        f.write(f"N={int(combined_model.nobs)}, N_towers={combined_data['SITE_ID'].nunique()}, "
                f"R2={combined_model.rsquared:.4f}\nFormula: {combined_formula}\n")
    print(f"\n=== Combined model (N={int(combined_model.nobs)}, R2={combined_model.rsquared:.4f}) ===")
    print(combined_df.to_string(index=False))

    # --- Nonlinearity check: transparent binned-mean plot for the two strongest continuous predictors ---
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), facecolor="white")
    for ax, col, label in zip(axes, ["PREPEAK_VPD_ANOM", "PREPEAK_TAIR_ANOM"], ["VPD anomaly", "Spring TAIR anomaly"]):
        sub = df.dropna(subset=[col, "DELTA_POS"])
        sub = sub.copy()
        sub["BIN"] = pd.qcut(sub[col], 8, duplicates="drop")
        binned = sub.groupby("BIN", observed=True).agg(x=(col, "mean"), y=("DELTA_POS", "mean"),
                                                        yerr=("DELTA_POS", lambda s: s.std(ddof=1) / np.sqrt(len(s))))
        ax.errorbar(binned["x"], binned["y"], yerr=binned["yerr"], fmt="o-", color="#333333", capsize=3)
        ax.axhline(0, color="#bbbbbb", lw=0.8, ls="--")
        ax.set_xlabel(f"{label} (binned)", fontsize=9.5)
        ax.set_ylabel("Mean \u0394POS (months)", fontsize=9.5)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Transparent nonlinearity check: binned mean \u0394POS vs. predictor (8 quantile bins)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    for ext in ("png", "svg", "pdf"):
        dpi = 600 if ext == "png" else 300
        fig.savefig(FIG_DIR / f"Figure6_Nonlinearity_Check.{ext}", dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("Saved Figure6_Nonlinearity_Check")


if __name__ == "__main__":
    main()
