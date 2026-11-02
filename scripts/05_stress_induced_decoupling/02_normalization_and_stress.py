"""
Step 2 & 3 -- SIF/GPP normalization (with sensitivity variants) and stress-metric
construction (VPD, soil moisture, temperature) for the SID foundational layer.

Outputs:
  02_normalization/SID_normalization_sensitivity.csv   (required deliverable)
  02_normalization/normalized_sif_gpp_by_sitemonth.csv (working table carried into Step 4)
  02_normalization/stress_metrics_by_sitemonth.csv
  02_normalization/stress_metrics_README.md
"""
# Original script filename (sanitized): 02_normalization_and_stress.py; public name: 02_normalization_and_stress.py
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
DRIVERS_PATH = ROOT / "SIF_EC_Mechanistic_FootprintAware_SIF_GPP_Paper" / "08_mechanistic_controls" / "Mechanistic_SIF_GPP_Monthly_With_Extended_Drivers.csv"
OUT_DIR = ROOT / "SIF_EC_Stress_Induced_Decoupling_SID" / "02_normalization"
MAIN_PRODUCTS = ["GOSIF", "GOME-2", "TROPOSIF", "OCO-3"]
SITE_REL_MIN_SITEMONTHS = 24  # minimum site-months per (SITE_ID, SIF_PRODUCT) to qualify for site-relative SIF/GPP normalization
QUANT_LEVELS = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


def robust_minmax(x, lo_pct, hi_pct):
    """clip[(x-P_lo)/(P_hi-P_lo), 0, 1] using percentiles lo_pct/hi_pct (0-100 scale) of x."""
    lo = np.nanpercentile(x, lo_pct)
    hi = np.nanpercentile(x, hi_pct)
    if hi == lo:
        norm = pd.Series(np.where(np.isnan(x), np.nan, 0.5), index=x.index if hasattr(x, "index") else None)
        return norm, lo, hi
    norm = (x - lo) / (hi - lo)
    norm = norm.clip(0, 1)
    return norm, lo, hi


def summarize_quantiles(x):
    x = pd.Series(x).dropna()
    out = {f"q{int(q*100):02d}": x.quantile(q) for q in QUANT_LEVELS}
    out["mean"] = x.mean()
    out["std"] = x.std()
    out["N"] = len(x)
    return out


def main():
    e = pd.read_csv(DRIVERS_PATH, low_memory=False)
    e = e[e["SIF_PRODUCT"].isin(MAIN_PRODUCTS)].copy()
    e["IGBP_BIOME"] = e["IGBP_GPP_TABLE"]  # documented choice: IGBP_GPP_TABLE used as biome field (raw IGBP 'Unknown' -> NaN already)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # STEP 2: SIF / GPP normalization, 3 methods, within each product separately
    # ------------------------------------------------------------------
    sens_rows = []
    norm_cols = {}  # (method, variable) -> pd.Series aligned to e.index

    for var, raw_col in [("SIF", "SIF_VALUE"), ("GPP", "GPP_VALUE")]:
        # ---- Method 1: PRIMARY robust percentile (P01/P99), pooled across sites within product ----
        col_primary = pd.Series(np.nan, index=e.index, dtype=float)
        for prod in MAIN_PRODUCTS:
            idx = e.index[e["SIF_PRODUCT"] == prod]
            x = e.loc[idx, raw_col]
            norm, lo, hi = robust_minmax(x, 1, 99)
            col_primary.loc[idx] = norm
            n_clip0 = int((norm == 0).sum())
            n_clip1 = int((norm == 1).sum())
            for biome, bidx in [("ALL", idx)] + [(b, idx[e.loc[idx, "IGBP_BIOME"] == b]) for b in sorted(e.loc[idx, "IGBP_BIOME"].dropna().unique())]:
                q = summarize_quantiles(norm.loc[bidx] if biome != "ALL" else norm)
                sens_rows.append(dict(
                    method="robust_percentile_P01_P99", product=prod, variable=var, biome=biome,
                    population_definition="all valid site-months for this product, pooled across sites",
                    low_pct=1, high_pct=99, low_value=lo, high_value=hi,
                    N_clipped_to_0=n_clip0 if biome == "ALL" else int((norm.loc[bidx] == 0).sum()),
                    N_clipped_to_1=n_clip1 if biome == "ALL" else int((norm.loc[bidx] == 1).sum()),
                    N_sites_qualifying=np.nan, N_sites_excluded=np.nan, **q,
                ))
        norm_cols[("primary_robust_p1_p99", var)] = col_primary

        # ---- Method 2: SENSITIVITY (a) ordinary min-max (P0/P100), pooled within product ----
        col_minmax = pd.Series(np.nan, index=e.index, dtype=float)
        for prod in MAIN_PRODUCTS:
            idx = e.index[e["SIF_PRODUCT"] == prod]
            x = e.loc[idx, raw_col]
            norm, lo, hi = robust_minmax(x, 0, 100)
            col_minmax.loc[idx] = norm
            for biome, bidx in [("ALL", idx)] + [(b, idx[e.loc[idx, "IGBP_BIOME"] == b]) for b in sorted(e.loc[idx, "IGBP_BIOME"].dropna().unique())]:
                q = summarize_quantiles(norm.loc[bidx] if biome != "ALL" else norm)
                sens_rows.append(dict(
                    method="sensitivity_minmax_P0_P100", product=prod, variable=var, biome=biome,
                    population_definition="all valid site-months for this product, pooled across sites",
                    low_pct=0, high_pct=100, low_value=lo, high_value=hi,
                    N_clipped_to_0=int((norm.loc[bidx] == 0).sum()) if biome != "ALL" else int((norm == 0).sum()),
                    N_clipped_to_1=int((norm.loc[bidx] == 1).sum()) if biome != "ALL" else int((norm == 1).sum()),
                    N_sites_qualifying=np.nan, N_sites_excluded=np.nan, **q,
                ))
        norm_cols[("sensitivity_minmax", var)] = col_minmax

        # ---- Method 3: SENSITIVITY (b) site-relative percentile, within (SITE_ID, product), sites with >=24 site-months ----
        col_siterel = pd.Series(np.nan, index=e.index, dtype=float)
        for prod in MAIN_PRODUCTS:
            idx_prod = e.index[e["SIF_PRODUCT"] == prod]
            counts = e.loc[idx_prod, "SITE_ID"].value_counts()
            qualifying_sites = counts[counts >= SITE_REL_MIN_SITEMONTHS].index.tolist()
            n_sites_qual = len(qualifying_sites)
            n_sites_excl = counts.shape[0] - n_sites_qual
            thresholds = []
            for site in qualifying_sites:
                sidx = idx_prod[e.loc[idx_prod, "SITE_ID"] == site]
                x = e.loc[sidx, raw_col]
                norm, lo, hi = robust_minmax(x, 1, 99)
                col_siterel.loc[sidx] = norm
                thresholds.append((lo, hi))
            idx_all_prod = idx_prod
            norm_full = col_siterel.loc[idx_all_prod]
            mean_lo = np.mean([t[0] for t in thresholds]) if thresholds else np.nan
            mean_hi = np.mean([t[1] for t in thresholds]) if thresholds else np.nan
            for biome, bidx in [("ALL", idx_all_prod)] + [(b, idx_all_prod[e.loc[idx_all_prod, "IGBP_BIOME"] == b]) for b in sorted(e.loc[idx_all_prod, "IGBP_BIOME"].dropna().unique())]:
                sub = col_siterel.loc[bidx]
                q = summarize_quantiles(sub)
                sens_rows.append(dict(
                    method="sensitivity_site_relative_percentile", product=prod, variable=var, biome=biome,
                    population_definition=f"per-(SITE_ID,product) P01/P99, only sites with >={SITE_REL_MIN_SITEMONTHS} site-months for this product (mean thresholds shown)",
                    low_pct=1, high_pct=99, low_value=mean_lo, high_value=mean_hi,
                    N_clipped_to_0=int((sub == 0).sum()), N_clipped_to_1=int((sub == 1).sum()),
                    N_sites_qualifying=n_sites_qual, N_sites_excluded=n_sites_excl, **q,
                ))
        norm_cols[("sensitivity_site_relative", var)] = col_siterel

    sens_df = pd.DataFrame(sens_rows)
    col_order = ["method", "product", "variable", "biome", "population_definition", "low_pct", "high_pct",
                 "low_value", "high_value", "N_clipped_to_0", "N_clipped_to_1", "N_sites_qualifying", "N_sites_excluded",
                 "N", "mean", "std"] + [f"q{int(q*100):02d}" for q in QUANT_LEVELS]
    sens_df = sens_df[col_order]
    sens_df.to_csv(OUT_DIR / "SID_normalization_sensitivity.csv", index=False)
    print("Saved:", OUT_DIR / "SID_normalization_sensitivity.csv", sens_df.shape)

    # working table with normalized columns (all 3 methods, both variables) for downstream steps
    work = e[["SITE_ID", "YYYYMM", "YEAR", "MONTH_x", "SIF_PRODUCT", "IGBP_BIOME"]].copy()
    work.rename(columns={"MONTH_x": "MONTH"}, inplace=True)
    work["SIF_norm_primary"] = norm_cols[("primary_robust_p1_p99", "SIF")]
    work["GPP_norm_primary"] = norm_cols[("primary_robust_p1_p99", "GPP")]
    work["SIF_norm_minmax"] = norm_cols[("sensitivity_minmax", "SIF")]
    work["GPP_norm_minmax"] = norm_cols[("sensitivity_minmax", "GPP")]
    work["SIF_norm_siterel"] = norm_cols[("sensitivity_site_relative", "SIF")]
    work["GPP_norm_siterel"] = norm_cols[("sensitivity_site_relative", "GPP")]
    work.to_csv(OUT_DIR / "normalized_sif_gpp_by_sitemonth.csv", index=False)
    print("Saved:", OUT_DIR / "normalized_sif_gpp_by_sitemonth.csv", work.shape)

    # ------------------------------------------------------------------
    # STEP 3: stress metrics, computed at deduplicated SITE_ID+YYYYMM level
    # (VPD/TAIR/SWC are tower-level meteorological drivers, verified identical
    # across SIF products for the same site-month; computing once per site-month
    # avoids inflating a site's temporal distribution with duplicate rows).
    # ------------------------------------------------------------------
    dedup = e.drop_duplicates(["SITE_ID", "YYYYMM"]).copy()
    dedup = dedup[["SITE_ID", "YYYYMM", "YEAR", "MONTH_x", "VPD", "VPD_CLIM", "VPD_ANOM", "VPD_AVAILABLE",
                    "TAIR", "TAIR_CLIM", "TAIR_ANOM", "TAIR_AVAILABLE", "SWC", "SWC_CLIM", "SWC_ANOM", "SWC_AVAILABLE",
                    "SINGLE_YEAR_DEGENERATE"]].rename(columns={"MONTH_x": "MONTH"})

    # ---- A. VPD stress ----
    # Primary: site-relative percentile rank of VPD_ANOM (site's own distribution), higher anomaly = higher stress.
    dedup["VPD_stress_siterel"] = dedup.groupby("SITE_ID")["VPD_ANOM"].rank(pct=True, na_option="keep")
    # Sensitivity: global robust-percentile (P01/P99) of raw VPD, pooled across all sites (not site-relative).
    vpd_lo, vpd_hi = np.nanpercentile(dedup["VPD"], 1), np.nanpercentile(dedup["VPD"], 99)
    dedup["VPD_stress_global_sensitivity"] = ((dedup["VPD"] - vpd_lo) / (vpd_hi - vpd_lo)).clip(0, 1)

    # ---- B. Soil-moisture stress ----
    # Primary: SM_stress = 1 - site-relative percentile rank of SWC (0=wet/low-stress site-month, 1=dry/high-stress).
    dedup["SWC_percentile_siterel"] = dedup.groupby("SITE_ID")["SWC"].rank(pct=True, na_option="keep")
    dedup["SM_stress"] = 1.0 - dedup["SWC_percentile_siterel"]
    dedup.loc[dedup["SWC"].isna(), "SM_stress"] = np.nan

    # ---- C. Temperature stress ----
    # Primary: positive TAIR_ANOM only (negative anomalies -> 0 stress), then global robust-percentile (P01/P99).
    dedup["TAIR_ANOM_pos"] = dedup["TAIR_ANOM"].clip(lower=0)
    t_lo, t_hi = np.nanpercentile(dedup["TAIR_ANOM_pos"], 1), np.nanpercentile(dedup["TAIR_ANOM_pos"], 99)
    if t_hi > t_lo:
        dedup["TEMP_stress"] = ((dedup["TAIR_ANOM_pos"] - t_lo) / (t_hi - t_lo)).clip(0, 1)
    else:
        dedup["TEMP_stress"] = np.where(dedup["TAIR_ANOM_pos"].isna(), np.nan, 0.0)
    # Sensitivity (lower priority, optional): symmetric anomaly magnitude, global robust-percentile of |TAIR_ANOM|
    dedup["TAIR_ANOM_abs"] = dedup["TAIR_ANOM"].abs()
    ta_lo, ta_hi = np.nanpercentile(dedup["TAIR_ANOM_abs"], 1), np.nanpercentile(dedup["TAIR_ANOM_abs"], 99)
    dedup["TEMP_stress_symmetric_sensitivity"] = ((dedup["TAIR_ANOM_abs"] - ta_lo) / (ta_hi - ta_lo)).clip(0, 1)

    dedup.to_csv(OUT_DIR / "stress_metrics_by_sitemonth.csv", index=False)
    print("Saved:", OUT_DIR / "stress_metrics_by_sitemonth.csv", dedup.shape)

    # sanity check counts
    print("\nVPD_stress_siterel non-null:", dedup["VPD_stress_siterel"].notna().sum(), "/", len(dedup))
    print("SM_stress non-null:", dedup["SM_stress"].notna().sum(), "/", len(dedup))
    print("TEMP_stress non-null:", dedup["TEMP_stress"].notna().sum(), "/", len(dedup))
    print("TAIR_ANOM_pos P01/P99:", t_lo, t_hi)
    print("VPD global P01/P99:", vpd_lo, vpd_hi)

    readme = f"""# Stress Metrics -- Formula Documentation (Step 3)

All stress metrics are computed on the **deduplicated site-month table**
(`stress_metrics_by_sitemonth.csv`, one row per SITE_ID x YYYYMM), because VPD,
TAIR and SWC are tower-level meteorological drivers that were verified to be
IDENTICAL across SIF products for the same site-month in the source
extended-drivers file (0 of {dedup.shape[0]}... site-months differ by product).
They are broadcast (left-joined on SITE_ID+YYYYMM) onto every product row in
the SID master table in Step 4/5.

## A. VPD stress (0 = lowest atmospheric stress, 1 = highest)

**Primary -- `VPD_stress_siterel`**: site-relative percentile rank of `VPD_ANOM`
(the deseasonalized VPD anomaly already computed in the extended-drivers file:
`VPD_ANOM = VPD - VPD_CLIM`, where `VPD_CLIM` is that site's own calendar-month
climatology). For each SITE_ID, `VPD_ANOM` values across all of that site's
available months are rank-transformed to the [0,1] interval with
`pandas.Series.rank(pct=True)` (average rank for ties). A value of 1 means that
site-month's VPD anomaly is the highest ever observed at that site; 0.5 is
site-typical for that calendar month. Using the anomaly (not raw VPD) means a
humid forest and an arid grassland are judged by deviation from their own
normal, not by absolute VPD level.

**Sensitivity -- `VPD_stress_global_sensitivity`**: robust-percentile (P01/P99)
0-1 normalization of RAW `VPD` pooled across all sites globally (not
site-relative): `clip[(VPD-P01)/(P99-P01), 0, 1]`, P01={vpd_lo:.4f},
P99={vpd_hi:.4f} kPa (deduplicated site-month population, N={dedup['VPD'].notna().sum()}).
This treats absolute VPD magnitude as the stress signal, ignoring site-to-site
climate differences -- kept purely as a documented sensitivity check, not used
in the primary SID computation.

## B. Soil-moisture stress (0 = wet/low-stress, 1 = dry/high-stress)

**Formula -- `SM_stress = 1 - percentile_rank_siterel(SWC)`**: for each
SITE_ID, `SWC` is rank-percentiled within that site's own time series
(`rank(pct=True)`), giving `SWC_percentile_siterel` where 1 = wettest month
ever observed at that site. `SM_stress` is then `1 - SWC_percentile_siterel`,
so the driest month at a site scores near 1 (high stress) and the wettest
scores near 0. Site-relative (not global pooled) because soil texture/porosity
differ enormously by biome and absolute volumetric SWC is not comparable
across sites. Rows where `SWC` is missing get `SM_stress = NaN` (flagged
`MISSING_DRIVER_SM` downstream) -- SWC has {int(dedup['SWC'].isna().sum())} of
{len(dedup)} site-months missing (see `01_input_audit/sid_input_inventory.csv`).

Sanity check performed in Step 6 script: an independently-known dry-season
site-month at an arid-biome tower is confirmed to receive a HIGH `SM_stress`
value (see validation log / README in 04_sid_validation).

## C. Temperature stress (0 = at/below normal, 1 = highest positive anomaly)

**Primary -- `TEMP_stress`**: positive part only of the deseasonalized
temperature anomaly, `TAIR_ANOM_pos = max(TAIR_ANOM, 0)` (negative anomalies,
i.e. cooler-than-normal months, are assigned 0 stress -- cold and warm are
NOT treated symmetrically). `TAIR_ANOM_pos` is then normalized with a global
robust-percentile 0-1 scale pooled across ALL site-months (not site-relative,
since the anomaly is already deseasonalized per-site so pooling globally is
appropriate for a physical quantity in degrees C):
`clip[(TAIR_ANOM_pos-P01)/(P99-P01), 0, 1]`, P01={t_lo:.4f} degC,
P99={t_hi:.4f} degC (N={dedup['TAIR_ANOM_pos'].notna().sum()}). Because a large
share of site-months have zero or negative anomaly, P01 is 0 or near 0 by
construction; this is expected and documented, not an error.

**Sensitivity (lower priority) -- `TEMP_stress_symmetric_sensitivity`**:
robust-percentile (P01/P99) of `|TAIR_ANOM|` (absolute anomaly magnitude,
treating hot and cold anomalies as equally stressful), pooled globally.
P01={ta_lo:.4f}, P99={ta_hi:.4f} degC. Provided only as an optional sensitivity
check; NOT used in the primary SID computation.

## Epsilon / denominator note
See `03_sid_monthly/` for the Decoupling denominator safeguard
(epsilon = 1e-6) and its dedicated sensitivity analysis
(`SID_denominator_sensitivity.csv`).
"""
    (OUT_DIR / "stress_metrics_README.md").write_text(readme, encoding="utf-8")
    print("Saved:", OUT_DIR / "stress_metrics_README.md")


if __name__ == "__main__":
    main()
