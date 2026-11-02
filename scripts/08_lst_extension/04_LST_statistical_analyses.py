"""
PART 6: A (partial Spearman), B (interaction/effect-modification), C (LST vs
decoupling), D (within-tower), E (LST vs TAIR). All associational, never causal.

Tower-block bootstrap (resample TOWERS with replacement, not individual
observations) used for every CI below, matching the project's existing
tower-clustered-uncertainty convention. 1000 resamples.
"""
# Original script filename (sanitized): 04_LST_statistical_analyses.py; public name: 04_LST_statistical_analyses.py
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
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = Path(PROJECT_ROOT)
HARMONIC_CSV = ROOT / "SIF_EC_SIF_GPP_Final_Verification" / "02_independent_reverification" / "harmonic_residuals_full.csv"
DRIVERS_CSV = ROOT / "SIF_EC_Mechanistic_FootprintAware_SIF_GPP_Paper" / "08_mechanistic_controls" / "Mechanistic_SIF_GPP_Monthly_With_Extended_Drivers.csv"
LST_ANOM_CSV = ROOT / "LST_SIF_GPP_EXTENSION_FINAL" / "02_processed_LST" / "tower_monthly_LST_anomaly_stress.csv"
SID_LST_CSV = ROOT / "LST_SIF_GPP_EXTENSION_FINAL" / "02_processed_LST" / "SID_LST_observation_level.csv"
TAB_DIR = ROOT / "LST_SIF_GPP_EXTENSION_FINAL" / "03_analysis_tables"
QC_DIR = ROOT / "LST_SIF_GPP_EXTENSION_FINAL" / "07_QC"

MAIN_PRODUCTS = ["GOSIF", "GOME-2", "TROPOSIF", "OCO-3"]
N_BOOT = 1000
RNG = np.random.default_rng(20261001)


def rank_resid(y, X):
    """Residuals of y's RANKS regressed on X's ranks (adds intercept). Returns np.array."""
    ranks_y = stats.rankdata(y)
    ranks_X = np.column_stack([stats.rankdata(col) for col in X.T]) if X.ndim > 1 else stats.rankdata(X).reshape(-1, 1)
    design = np.column_stack([np.ones(len(y)), ranks_X])
    coef, *_ = np.linalg.lstsq(design, ranks_y, rcond=None)
    return ranks_y - design @ coef


def partial_spearman(sif, gpp, controls_2d):
    """Partial Spearman rho between sif and gpp, controlling for one or more
    covariates (controls_2d: n x k array), via rank-regression residuals."""
    r_sif = rank_resid(sif, controls_2d)
    r_gpp = rank_resid(gpp, controls_2d)
    rho, p = stats.pearsonr(r_sif, r_gpp)  # Pearson on rank-residuals == partial Spearman
    return rho, p


def tower_block_bootstrap_partial(df, control_cols, n_boot=N_BOOT):
    """df must have SITE_ID, SIF_ANOM, GPP_ANOM, and control_cols (no NaNs).
    Returns (rho0, rho_partial, delta, ci_low_delta, ci_high_delta, boot_rho_partial_array)."""
    towers = df["SITE_ID"].unique()
    rho0_full, _ = stats.spearmanr(df["SIF_ANOM"], df["GPP_ANOM"])
    rho_partial_full, _ = partial_spearman(df["SIF_ANOM"].values, df["GPP_ANOM"].values,
                                            df[control_cols].values)
    boot_rho0, boot_rho_partial = [], []
    by_site = {s: sub for s, sub in df.groupby("SITE_ID")}
    for _ in range(n_boot):
        sample_towers = RNG.choice(towers, size=len(towers), replace=True)
        parts = [by_site[t] for t in sample_towers]
        boot_df = pd.concat(parts, ignore_index=True)
        if boot_df["SIF_ANOM"].nunique() < 3 or boot_df["GPP_ANOM"].nunique() < 3:
            continue
        r0, _ = stats.spearmanr(boot_df["SIF_ANOM"], boot_df["GPP_ANOM"])
        try:
            rp, _ = partial_spearman(boot_df["SIF_ANOM"].values, boot_df["GPP_ANOM"].values,
                                      boot_df[control_cols].values)
        except Exception:
            continue
        boot_rho0.append(r0)
        boot_rho_partial.append(rp)
    boot_delta = np.array(boot_rho0) - np.array(boot_rho_partial)
    ci_low, ci_high = np.percentile(boot_delta, [2.5, 97.5])
    return rho0_full, rho_partial_full, rho0_full - rho_partial_full, ci_low, ci_high, np.array(boot_rho_partial)


def main():
    harm = pd.read_csv(HARMONIC_CSV, low_memory=False,
                        usecols=["SITE_ID", "SIF_PRODUCT", "YEAR", "MONTH", "LATITUDE", "IGBP_GPP_TABLE",
                                 "SIF_ANOM", "GPP_ANOM"])
    harm = harm[harm["SIF_PRODUCT"].isin(MAIN_PRODUCTS)].dropna(subset=["SIF_ANOM", "GPP_ANOM"])
    harm["HEMI"] = np.where(harm["LATITUDE"] >= 0, "NH", "SH")

    drivers_anom = pd.read_csv(DRIVERS_CSV, low_memory=False,
                                usecols=["SITE_ID", "YEAR", "MONTH_y", "VPD_ANOM", "TAIR_ANOM", "SW_IN_ANOM", "SWC_ANOM"]
                                ).rename(columns={"MONTH_y": "MONTH"}).drop_duplicates(["SITE_ID", "YEAR", "MONTH"])

    lst_anom = pd.read_csv(LST_ANOM_CSV, usecols=["SITE_ID", "YEAR", "MONTH", "LST_degC", "LST_ANOM", "LST_stress"])

    df = harm.merge(drivers_anom, on=["SITE_ID", "YEAR", "MONTH"], how="left")
    df = df.merge(lst_anom, on=["SITE_ID", "YEAR", "MONTH"], how="left")

    # ================= A: PARTIAL SPEARMAN =================
    rowsA = []
    for product in MAIN_PRODUCTS:
        sub = df[(df["SIF_PRODUCT"] == product)].dropna(subset=["LST_ANOM"])
        n_towers = sub["SITE_ID"].nunique()
        if len(sub) < 15 or n_towers < 3:
            rowsA.append({"SIF_PRODUCT": product, "rho0": np.nan, "rho_LST": np.nan, "delta_rho_LST": np.nan,
                          "CI_low": np.nan, "CI_high": np.nan, "N_site_months": len(sub), "N_towers": n_towers,
                          "STATUS": "INSUFFICIENT_N"})
            continue
        rho0, rho_lst, delta, cl, ch, _ = tower_block_bootstrap_partial(sub, ["LST_ANOM"])
        rowsA.append({"SIF_PRODUCT": product, "rho0": rho0, "rho_LST": rho_lst, "delta_rho_LST": delta,
                      "CI_low": cl, "CI_high": ch, "N_site_months": len(sub), "N_towers": n_towers,
                      "STATUS": "OK"})
    partial_df = pd.DataFrame(rowsA)
    partial_df.to_csv(TAB_DIR / "LST_partial_spearman_by_product.csv", index=False)
    print("Partial Spearman (LST) by product:\n", partial_df.to_string(index=False))

    # joint model: VPD+SWC+TAIR+SW_IN vs +LST, POOLED across products (products stacked,
    # each site-month-product row is one observation; consistent with how the project's
    # existing joint environmental-control models pool across products)
    joint_rows = []
    sub_all = df.dropna(subset=["VPD_ANOM", "SWC_ANOM", "TAIR_ANOM", "SW_IN_ANOM"])
    base_controls = ["VPD_ANOM", "SWC_ANOM", "TAIR_ANOM", "SW_IN_ANOM"]
    rho0_j, rho_base, *_ = tower_block_bootstrap_partial(sub_all, base_controls, n_boot=200)[:3]
    sub_all_lst = sub_all.dropna(subset=["LST_ANOM"])
    rho0_jl, rho_joint_lst, delta_jl, cl_jl, ch_jl, _ = tower_block_bootstrap_partial(
        sub_all_lst, base_controls + ["LST_ANOM"], n_boot=200)
    rho0_base_samesample, rho_base_samesample, *_ = tower_block_bootstrap_partial(sub_all_lst, base_controls, n_boot=200)[:3]
    joint_rows.append({"MODEL": "VPD+SWC+TAIR+SW_IN (existing, same LST-covered sample)",
                        "rho0": rho0_base_samesample, "rho_partial": rho_base_samesample,
                        "N_site_months": len(sub_all_lst), "N_towers": sub_all_lst["SITE_ID"].nunique()})
    joint_rows.append({"MODEL": "VPD+SWC+TAIR+SW_IN+LST (extended)",
                        "rho0": rho0_jl, "rho_partial": rho_joint_lst,
                        "N_site_months": len(sub_all_lst), "N_towers": sub_all_lst["SITE_ID"].nunique()})
    pd.DataFrame(joint_rows).to_csv(TAB_DIR / "LST_joint_model_comparison.csv", index=False)
    print("\nJoint model comparison:\n", pd.DataFrame(joint_rows).to_string(index=False))

    # ================= B: INTERACTION / EFFECT MODIFICATION =================
    rowsB = []
    for product in MAIN_PRODUCTS:
        sub = df[df["SIF_PRODUCT"] == product].dropna(subset=["LST_ANOM"]).copy()
        n_towers = sub["SITE_ID"].nunique()
        if len(sub) < 20 or n_towers < 4:
            rowsB.append({"SIF_PRODUCT": product, "beta_SIF": np.nan, "beta_LST": np.nan, "beta_SIFxLST": np.nan,
                          "CI_low_interaction": np.nan, "CI_high_interaction": np.nan, "p_interaction": np.nan,
                          "N_site_months": len(sub), "N_towers": n_towers, "STATUS": "INSUFFICIENT_N"})
            continue
        # within-site standardization (z-score within SITE_ID)
        for col, zcol in [("SIF_ANOM", "SIF_z"), ("GPP_ANOM", "GPP_z"), ("LST_ANOM", "LST_z")]:
            sub[zcol] = sub.groupby("SITE_ID")[col].transform(lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else np.nan)
        sub = sub.dropna(subset=["SIF_z", "GPP_z", "LST_z"])
        sub["MONTH_C"] = sub["MONTH"].astype("category")
        sub["SITE_C"] = sub["SITE_ID"].astype("category")
        try:
            model = smf.ols("GPP_z ~ SIF_z * LST_z + MONTH_C + SITE_C", data=sub)
            res = model.fit(cov_type="cluster", cov_kwds={"groups": sub["SITE_ID"]})
            term = "SIF_z:LST_z"
            beta_int = res.params.get(term, np.nan)
            ci = res.conf_int().loc[term] if term in res.params.index else (np.nan, np.nan)
            p_int = res.pvalues.get(term, np.nan)
            rowsB.append({"SIF_PRODUCT": product, "beta_SIF": res.params.get("SIF_z", np.nan),
                          "beta_LST": res.params.get("LST_z", np.nan), "beta_SIFxLST": beta_int,
                          "CI_low_interaction": ci[0], "CI_high_interaction": ci[1], "p_interaction": p_int,
                          "N_site_months": len(sub), "N_towers": n_towers, "STATUS": "OK"})
        except Exception as e:
            rowsB.append({"SIF_PRODUCT": product, "beta_SIF": np.nan, "beta_LST": np.nan, "beta_SIFxLST": np.nan,
                          "CI_low_interaction": np.nan, "CI_high_interaction": np.nan, "p_interaction": np.nan,
                          "N_site_months": len(sub), "N_towers": n_towers, "STATUS": f"MODEL_FAILED: {e}"})
    interaction_df = pd.DataFrame(rowsB)
    interaction_df.to_csv(TAB_DIR / "LST_interaction_models_by_product.csv", index=False)
    print("\nInteraction models by product:\n", interaction_df.to_string(index=False))

    # ================= C: LST vs DECOUPLING =================
    sid_lst = pd.read_csv(SID_LST_CSV, low_memory=False)
    rowsC = []
    for product in MAIN_PRODUCTS:
        sub = sid_lst[(sid_lst["SIF_PRODUCT"] == product)].dropna(subset=["LST_ANOM", "Decoupling"])
        if len(sub) < 10:
            rowsC.append({"SIF_PRODUCT": product, "rho_LSTanom_vs_Decoupling": np.nan, "p_value": np.nan,
                         "rho_LSTstress_vs_Decoupling": np.nan, "p_value_stress": np.nan,
                         "N_site_months": len(sub), "N_towers": sub["SITE_ID"].nunique()})
            continue
        rho1, p1 = stats.spearmanr(sub["LST_ANOM"], sub["Decoupling"])
        sub2 = sub.dropna(subset=["LST_stress"])
        rho2, p2 = stats.spearmanr(sub2["LST_stress"], sub2["Decoupling"]) if len(sub2) >= 10 else (np.nan, np.nan)
        rowsC.append({"SIF_PRODUCT": product, "rho_LSTanom_vs_Decoupling": rho1, "p_value": p1,
                     "rho_LSTstress_vs_Decoupling": rho2, "p_value_stress": p2,
                     "N_site_months": len(sub), "N_towers": sub["SITE_ID"].nunique()})
    # pooled
    sub_pool = sid_lst.dropna(subset=["LST_ANOM", "Decoupling"])
    rho1p, p1p = stats.spearmanr(sub_pool["LST_ANOM"], sub_pool["Decoupling"])
    sub_pool2 = sub_pool.dropna(subset=["LST_stress"])
    rho2p, p2p = stats.spearmanr(sub_pool2["LST_stress"], sub_pool2["Decoupling"])
    rowsC.append({"SIF_PRODUCT": "ALL_POOLED", "rho_LSTanom_vs_Decoupling": rho1p, "p_value": p1p,
                 "rho_LSTstress_vs_Decoupling": rho2p, "p_value_stress": p2p,
                 "N_site_months": len(sub_pool), "N_towers": sub_pool["SITE_ID"].nunique()})
    decoupling_df = pd.DataFrame(rowsC)
    decoupling_df.to_csv(TAB_DIR / "LST_decoupling_associations.csv", index=False)
    print("\nLST vs Decoupling:\n", decoupling_df.to_string(index=False))

    # SID_LST biome + monthly summaries
    biome_summary = sid_lst.dropna(subset=["SID_LST"]).groupby("IGBP").agg(
        MEAN_SID_LST=("SID_LST", "mean"), MEDIAN_SID_LST=("SID_LST", "median"),
        N=("SID_LST", "size"), N_TOWERS=("SITE_ID", "nunique")).reset_index()
    biome_summary.to_csv(TAB_DIR / "SID_LST_biome_summary.csv", index=False)

    monthly_summary = sid_lst.dropna(subset=["SID_LST"]).groupby("MONTH").agg(
        MEAN_SID_LST=("SID_LST", "mean"), MEDIAN_SID_LST=("SID_LST", "median"),
        N=("SID_LST", "size"), N_TOWERS=("SITE_ID", "nunique")).reset_index()
    monthly_summary.to_csv(TAB_DIR / "SID_LST_monthly_summary.csv", index=False)
    print("\nSID_LST by biome:\n", biome_summary.to_string(index=False))
    print("\nSID_LST by month:\n", monthly_summary.to_string(index=False))

    # ================= D: WITHIN-TOWER ANALYSIS =================
    rowsD = []
    for (site_id, product), sub in sid_lst.dropna(subset=["LST_ANOM", "Decoupling"]).groupby(["SITE_ID", "SIF_PRODUCT"]):
        if len(sub) < 8 or sub["LST_ANOM"].nunique() < 3:
            continue
        rho, p = stats.spearmanr(sub["LST_ANOM"], sub["Decoupling"])
        rowsD.append({"SITE_ID": site_id, "SIF_PRODUCT": product, "IGBP": sub["IGBP"].iloc[0],
                     "rho_within_tower": rho, "p_value": p, "N": len(sub)})
    within_df = pd.DataFrame(rowsD)
    within_df.to_csv(QC_DIR / "LST_within_tower_coefficients.csv", index=False)

    def summarize_group(g):
        return pd.Series({"N_TOWER_PRODUCT_PAIRS": len(g), "MEDIAN_RHO": g["rho_within_tower"].median(),
                          "Q25": g["rho_within_tower"].quantile(.25), "Q75": g["rho_within_tower"].quantile(.75),
                          "PCT_POSITIVE": 100 * (g["rho_within_tower"] > 0).mean()})

    if len(within_df):
        by_product = within_df.groupby("SIF_PRODUCT").apply(summarize_group, include_groups=False).reset_index()
        by_biome = within_df.groupby("IGBP").apply(summarize_group, include_groups=False).reset_index()
        by_product.to_csv(QC_DIR / "LST_within_tower_summary_by_product.csv", index=False)
        by_biome.to_csv(QC_DIR / "LST_within_tower_summary_by_biome.csv", index=False)
        print("\nWithin-tower rho(LST_ANOM, Decoupling) summary by product:\n", by_product.to_string(index=False))
        print("\nWithin-tower rho(LST_ANOM, Decoupling) summary by biome:\n", by_biome.to_string(index=False))
    else:
        print("\nWithin-tower analysis: no tower-product pairs met the minimum-N threshold.")

    # ================= E: LST vs TAIR =================
    sub_e = df.dropna(subset=["LST_degC", "TAIR_ANOM", "LST_ANOM"])
    # pooled raw correlation needs raw TAIR (not anomaly) -- pulled separately below
    drivers_full = pd.read_csv(DRIVERS_CSV, low_memory=False,
                                usecols=["SITE_ID", "YEAR", "MONTH_y", "TAIR"]).rename(columns={"MONTH_y": "MONTH"}).drop_duplicates(["SITE_ID", "YEAR", "MONTH"])
    sub_e = sub_e.merge(drivers_full, on=["SITE_ID", "YEAR", "MONTH"], how="left")
    pooled_r, pooled_p = stats.pearsonr(sub_e["LST_degC"], sub_e["TAIR"])
    pooled_rho, pooled_rho_p = stats.spearmanr(sub_e["LST_degC"], sub_e["TAIR"])
    anom_r, anom_p = stats.pearsonr(sub_e["LST_ANOM"], sub_e["TAIR_ANOM"])

    within_tower_r = []
    for site_id, g in sub_e.groupby("SITE_ID"):
        if len(g) < 8 or g["LST_degC"].nunique() < 3:
            continue
        r, _ = stats.pearsonr(g["LST_degC"], g["TAIR"])
        within_tower_r.append(r)
    within_tower_mean_r = float(np.mean(within_tower_r)) if within_tower_r else np.nan

    rho_tair_only, p_tair_only = partial_spearman(
        sub_e["SIF_ANOM"].values, sub_e["GPP_ANOM"].values, sub_e[["TAIR_ANOM"]].values) if "SIF_ANOM" in sub_e.columns else (np.nan, np.nan)
    sub_e2 = sub_e.dropna(subset=["TAIR_ANOM", "LST_ANOM"])
    rho_tair_lst, p_tair_lst = partial_spearman(sub_e2["SIF_ANOM"].values, sub_e2["GPP_ANOM"].values,
                                                 sub_e2[["TAIR_ANOM", "LST_ANOM"]].values)
    rho0_e, _ = stats.spearmanr(sub_e2["SIF_ANOM"], sub_e2["GPP_ANOM"])

    tair_comp = pd.DataFrame([{
        "pooled_pearson_r_LST_vs_TAIR": pooled_r, "pooled_pearson_p": pooled_p,
        "pooled_spearman_rho_LST_vs_TAIR": pooled_rho,
        "anomaly_pearson_r_LSTanom_vs_TAIRanom": anom_r, "anomaly_pearson_p": anom_p,
        "mean_within_tower_pearson_r": within_tower_mean_r, "n_towers_within": len(within_tower_r),
        "rho0_SIF_GPP_anom_same_sample": rho0_e,
        "rho_partial_controlling_TAIR_only": rho_tair_only,
        "rho_partial_controlling_TAIR_plus_LST": rho_tair_lst,
        "additional_delta_from_adding_LST": rho_tair_only - rho_tair_lst,
        "N_site_months": len(sub_e2), "N_towers": sub_e2["SITE_ID"].nunique(),
    }])
    tair_comp.to_csv(TAB_DIR / "LST_vs_TAIR_comparison.csv", index=False)
    print("\nLST vs TAIR comparison:\n", tair_comp.T.to_string())

    print("\nPART 6 (A-E) DONE.")


if __name__ == "__main__":
    main()
