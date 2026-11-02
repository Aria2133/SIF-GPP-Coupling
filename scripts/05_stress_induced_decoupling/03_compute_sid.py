"""
Step 4 & 5 -- Compute SID (Stress-Induced Decoupling) per site-month-product,
its signed supporting variant, the denominator-threshold sensitivity analysis,
and assemble the final monthly master table.

Outputs:
  03_sid_monthly/monthly_SID_master.csv          (required deliverable)
  03_sid_monthly/SID_denominator_sensitivity.csv (required deliverable)
"""
# Original script filename (sanitized): 03_compute_sid.py; public name: 03_compute_sid.py
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
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(PROJECT_ROOT)
MASTER_PATH = ROOT / "SIF_EC_SIF_GPP_Relationship_Analysis" / "01_analysis_ready" / "SIF_GPP_Analysis_Ready_Master.csv"
DRIVERS_PATH = ROOT / "SIF_EC_Mechanistic_FootprintAware_SIF_GPP_Paper" / "08_mechanistic_controls" / "Mechanistic_SIF_GPP_Monthly_With_Extended_Drivers.csv"
NORM_DIR = ROOT / "SIF_EC_Stress_Induced_Decoupling_SID" / "02_normalization"
OUT_DIR = ROOT / "SIF_EC_Stress_Induced_Decoupling_SID" / "03_sid_monthly"
MAIN_PRODUCTS = ["GOSIF", "GOME-2", "TROPOSIF", "OCO-3"]
EPSILON = 1e-6
LOW_DENOM_FLAG_THRESHOLD = 0.10
DENOM_SENS_THRESHOLDS = [0.05, 0.10, 0.20]

STRESSORS = {
    "VPD": "VPD_stress_siterel",
    "SM": "SM_stress",
    "TEMP": "TEMP_stress",
}


def make_terciles(s):
    """Robust tercile labels even when the underlying values have many ties
    (e.g. TEMP_stress has a large mass of exact zeros). Ties are broken by
    row order via rank(method='first') before qcut so bin edges never collapse."""
    ranks = s.rank(method="first")
    n_unique_bins = min(3, ranks.nunique())
    if n_unique_bins < 3:
        return pd.Series(np.nan, index=s.index)
    return pd.qcut(ranks, 3, labels=["low", "mid", "high"])


def main():
    m = pd.read_csv(MASTER_PATH, low_memory=False)
    e = pd.read_csv(DRIVERS_PATH, low_memory=False)
    e = e[e["SIF_PRODUCT"].isin(MAIN_PRODUCTS)].copy()
    e["IGBP_BIOME"] = e["IGBP_GPP_TABLE"]

    norm = pd.read_csv(NORM_DIR / "normalized_sif_gpp_by_sitemonth.csv")
    stress = pd.read_csv(NORM_DIR / "stress_metrics_by_sitemonth.csv")

    key = ["SITE_ID", "YYYYMM", "SIF_PRODUCT"]
    assert e.duplicated(key).sum() == 0
    assert norm.duplicated(key).sum() == 0
    assert len(e) == len(norm)

    # base table = extended-drivers rows (1 row per real site x month x product record)
    base = e[["SITE_ID", "YYYYMM", "YEAR", "MONTH_x", "LATITUDE", "LONGITUDE", "IGBP_BIOME", "SIF_PRODUCT",
              "SIF_VALUE", "GPP_VALUE", "OBSERVED_OR_RECONSTRUCTED", "VALUE_STATUS", "QC_FLAG", "QC_SEVERITY",
              "SINGLE_YEAR_DEGENERATE"]].rename(columns={"MONTH_x": "MONTH", "IGBP_BIOME": "IGBP"})

    # merge PRIMARY_READY / QC_RETAINED_IN_PRIMARY from the analysis-ready master for full QC traceability
    m_qc = m[m["SIF_PRODUCT"].isin(MAIN_PRODUCTS)][key + ["PRIMARY_READY", "QC_RETAINED_IN_PRIMARY"]]
    base = base.merge(m_qc, on=key, how="left")

    df = base.merge(norm[key + ["SIF_norm_primary", "GPP_norm_primary", "SIF_norm_minmax", "GPP_norm_minmax",
                                 "SIF_norm_siterel", "GPP_norm_siterel"]], on=key, how="left")

    site_key = ["SITE_ID", "YYYYMM"]
    stress_cols = ["SITE_ID", "YYYYMM", "VPD_stress_siterel", "VPD_stress_global_sensitivity",
                   "SM_stress", "TEMP_stress", "TEMP_stress_symmetric_sensitivity"]
    n_before = len(df)
    df = df.merge(stress[stress_cols], on=site_key, how="left")
    assert len(df) == n_before, "row count changed during stress-metric merge -- unexpected fan-out"

    # rename primary normalized columns to the master-table spec names
    df["SIF_norm"] = df["SIF_norm_primary"]
    df["GPP_norm"] = df["GPP_norm_primary"]
    df["VPD_stress"] = df["VPD_stress_siterel"]
    df["SM_stress"] = df["SM_stress"]
    df["TEMP_stress"] = df["TEMP_stress"]

    denom = (df["SIF_norm"] + df["GPP_norm"]).clip(lower=EPSILON)
    df["SIF_plus_GPP_norm"] = df["SIF_norm"] + df["GPP_norm"]
    df["Decoupling"] = (df["SIF_norm"] - df["GPP_norm"]).abs() / denom
    df["Signed_Decoupling"] = df["GPP_norm"] - df["SIF_norm"]

    for label, col in STRESSORS.items():
        df[f"SID_{label}"] = df["Decoupling"] * df[col]
        df[f"Signed_SID_{label}"] = ((df["GPP_norm"] - df["SIF_norm"]) / denom) * df[col]

    # QC / flag column
    flags = []
    for _, r in df.iterrows():
        f = []
        if r["SIF_plus_GPP_norm"] < LOW_DENOM_FLAG_THRESHOLD:
            f.append("LOW_DENOMINATOR_FLAG")
        if pd.isna(r["SM_stress"]):
            f.append("MISSING_DRIVER_SM")
        if pd.isna(r["VPD_stress"]):
            f.append("MISSING_DRIVER_VPD")
        if pd.isna(r["TEMP_stress"]):
            f.append("MISSING_DRIVER_TEMP")
        if bool(r.get("SINGLE_YEAR_DEGENERATE", False)):
            f.append("SINGLE_YEAR_CLIMATOLOGY")
        if str(r.get("OBSERVED_OR_RECONSTRUCTED", "")) == "RECONSTRUCTED":
            f.append("OCO3_RECONSTRUCTED")
        flags.append("|".join(f) if f else "")
    df["QC_FLAG_NOTES"] = flags

    # sanity: no negative, inf, or absurd SID values should slip through unflagged
    sid_cols = [f"SID_{k}" for k in STRESSORS]
    for c in sid_cols:
        n_neg = (df[c] < 0).sum()
        n_inf = np.isinf(df[c]).sum()
        assert n_neg == 0, f"{c} has {n_neg} negative values"
        assert n_inf == 0, f"{c} has {n_inf} infinite values"
        n_gt1 = (df[c] > 1.0001).sum()
        print(f"{c}: max={df[c].max():.4f}, N>1={n_gt1} (expected 0 since Decoupling in [0,1] and stress in [0,1])")

    out_cols = ["SITE_ID", "YYYYMM", "YEAR", "MONTH", "SIF_PRODUCT", "LATITUDE", "LONGITUDE", "IGBP",
                "SIF_VALUE", "GPP_VALUE", "SIF_norm", "GPP_norm", "Decoupling",
                "VPD_stress", "SM_stress", "TEMP_stress",
                "SID_VPD", "SID_SM", "SID_TEMP",
                "Signed_Decoupling", "Signed_SID_VPD", "Signed_SID_SM", "Signed_SID_TEMP",
                "OBSERVED_OR_RECONSTRUCTED", "QC_FLAG_NOTES",
                # extra traceability / sensitivity-ready columns (not required by spec but useful downstream)
                "SIF_norm_minmax", "GPP_norm_minmax", "SIF_norm_siterel", "GPP_norm_siterel",
                "VPD_stress_global_sensitivity", "TEMP_stress_symmetric_sensitivity",
                "PRIMARY_READY", "QC_RETAINED_IN_PRIMARY", "SINGLE_YEAR_DEGENERATE"]
    master = df[out_cols].copy()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    master.to_csv(OUT_DIR / "monthly_SID_master.csv", index=False)
    print("\nSaved:", OUT_DIR / "monthly_SID_master.csv", master.shape)
    print("N towers:", master["SITE_ID"].nunique())
    print(master["SIF_PRODUCT"].value_counts())
    print("N rows with LOW_DENOMINATOR_FLAG:", master["QC_FLAG_NOTES"].str.contains("LOW_DENOMINATOR_FLAG").sum())

    # ------------------------------------------------------------------
    # Denominator-threshold sensitivity
    # ------------------------------------------------------------------
    denom_rows = []
    for thresh in DENOM_SENS_THRESHOLDS:
        for prod in MAIN_PRODUCTS + ["ALL"]:
            sub_full = df if prod == "ALL" else df[df["SIF_PRODUCT"] == prod]
            keep_mask = sub_full["SIF_plus_GPP_norm"] >= thresh
            n_excl = int((~keep_mask).sum())
            n_remain = int(keep_mask.sum())
            sub = sub_full[keep_mask]
            for label, stress_col in STRESSORS.items():
                sid_col = f"SID_{label}"
                s = sub[[sid_col, stress_col]].dropna()
                if len(s) < 10:
                    trend = "INSUFFICIENT_N"
                    med_lo = med_mid = med_hi = np.nan
                else:
                    terc = make_terciles(s[stress_col])
                    med_by_terc = s.groupby(terc, observed=True)[sid_col].median()
                    med_lo = med_by_terc.get("low", np.nan)
                    med_mid = med_by_terc.get("mid", np.nan)
                    med_hi = med_by_terc.get("high", np.nan)
                    if pd.notna(med_lo) and pd.notna(med_hi):
                        if med_hi > med_lo:
                            trend = "INCREASING"
                        elif med_hi < med_lo:
                            trend = "DECREASING"
                        else:
                            trend = "FLAT"
                    else:
                        trend = "UNDEFINED"
                denom_rows.append(dict(
                    denominator_threshold=thresh, product=prod, stressor=label,
                    N_excluded=n_excl, N_remaining=n_remain,
                    mean_SID=sub[sid_col].mean(), median_SID=sub[sid_col].median(),
                    p90_SID=sub[sid_col].quantile(0.90) if len(sub) else np.nan,
                    median_SID_low_tercile=med_lo, median_SID_mid_tercile=med_mid, median_SID_high_tercile=med_hi,
                    tercile_trend_direction=trend,
                ))
    denom_df = pd.DataFrame(denom_rows)
    denom_df.to_csv(OUT_DIR / "SID_denominator_sensitivity.csv", index=False)
    print("\nSaved:", OUT_DIR / "SID_denominator_sensitivity.csv", denom_df.shape)

    # quick look: fraction of extreme SID values at low denominator
    print("\nDenominator diagnostic (ALL products, unfiltered):")
    q90_vpd = df["SID_VPD"].quantile(0.90)
    extreme = df[df["SID_VPD"] >= q90_vpd]
    print(f"  Among top-10% SID_VPD values (N={len(extreme)}), "
          f"median SIF_norm+GPP_norm = {extreme['SIF_plus_GPP_norm'].median():.4f} "
          f"vs overall median = {df['SIF_plus_GPP_norm'].median():.4f}")
    print(f"  Fraction of top-10% SID_VPD rows with SIF_norm+GPP_norm < 0.10: "
          f"{(extreme['SIF_plus_GPP_norm'] < 0.10).mean():.2%}")


if __name__ == "__main__":
    main()
