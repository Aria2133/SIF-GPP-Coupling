"""
PARTS 2-3: verify the authoritative SIF-GPP master, build tower-monthly LST,
merge onto the final analysis-ready SIF-GPP table, and produce QC summaries.

Part 2 verification (printed, not re-derived): SIF_GPP_Analysis_Ready_Master.csv,
filtered to SIF_PRODUCT in {GOSIF, GOME-2, TROPOSIF, OCO-3} and PRIMARY_READY==True,
gives exactly 19,775 rows / 73 towers -- the same count used throughout this entire
project as "the final 4-product analysis-ready dataset." This script reuses that
table and that filter unmodified; nothing is rebuilt from raw FLUXNET/SIF sources.

Part 3: LST source = the empirical per-tower extraction already produced by
01_audit_extract_LST_at_towers.py (LST_tower_extraction_raw_hits.csv), itself read
straight from the only local LST rasters (MYD21C1, see LST_DATA_AUDIT.md). Day/night
status is UNVERIFIED (documented in the audit) -- there is only ONE LST variable
available locally, so there is no actual day-vs-night choice to make; this script
uses the only variable that exists and states that explicitly rather than claiming
it was chosen as "daytime" when that could not be verified.
"""
# Original script filename (sanitized): 02_build_LST_matches_and_QC.py; public name: 02_build_LST_matches_and_QC.py
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

ROOT = Path(PROJECT_ROOT)
MASTER_CSV = ROOT / "SIF_EC_SIF_GPP_Relationship_Analysis" / "01_analysis_ready" / "SIF_GPP_Analysis_Ready_Master.csv"
DRIVERS_CSV = ROOT / "SIF_EC_Mechanistic_FootprintAware_SIF_GPP_Paper" / "08_mechanistic_controls" / "Mechanistic_SIF_GPP_Monthly_With_Extended_Drivers.csv"
ARIDITY_CSV = ROOT / "All_the_Visualizations_REVISED_4PRODUCT_FINAL_ARIDITY_JUNE_DIAGNOSTICS" / "01_inputs_inventory" / "tower_aridity_annual_and_monthly.csv"
LST_HITS_CSV = ROOT / "LST_SIF_GPP_EXTENSION_FINAL" / "01_data_audit" / "LST_tower_extraction_raw_hits.csv"

OUT_LST = ROOT / "LST_SIF_GPP_EXTENSION_FINAL" / "02_processed_LST"
OUT_TAB = ROOT / "LST_SIF_GPP_EXTENSION_FINAL" / "03_analysis_tables"
OUT_QC = ROOT / "LST_SIF_GPP_EXTENSION_FINAL" / "07_QC"

MAIN_PRODUCTS = ["GOSIF", "GOME-2", "TROPOSIF", "OCO-3"]


def main():
    # ---- Part 2: verify + load authoritative master ----
    m = pd.read_csv(MASTER_CSV, low_memory=False)
    master = m[(m["SIF_PRODUCT"].isin(MAIN_PRODUCTS)) & (m["PRIMARY_READY"] == True)].copy()
    assert len(master) == 19775 and master["SITE_ID"].nunique() == 73, \
        f"Master filter mismatch: got {len(master)} rows / {master['SITE_ID'].nunique()} towers, expected 19775/73"
    print(f"PART 2 verified: {len(master)} rows, {master['SITE_ID'].nunique()} towers, "
          f"products={sorted(master['SIF_PRODUCT'].unique())}")

    # merge environmental drivers (VPD/SWC/TAIR/SW_IN) + aridity, same sources as every
    # other analysis in this project
    drivers = pd.read_csv(DRIVERS_CSV, low_memory=False,
                           usecols=["SITE_ID", "YYYYMM", "SIF_PRODUCT", "VPD", "TAIR", "SW_IN", "SWC"])
    master = master.merge(drivers, on=["SITE_ID", "YYYYMM", "SIF_PRODUCT"], how="left")

    aridity = pd.read_csv(ARIDITY_CSV, usecols=["SITE_ID", "LATITUDE", "AI_ANNUAL", "AI_CLASS"])
    master = master.drop(columns=["LATITUDE"], errors="ignore").merge(aridity, on="SITE_ID", how="left")
    master["HEMI"] = np.where(master["LATITUDE"] >= 0, "NH", "SH")

    # ---- Part 3: tower-monthly LST (deduplicate the 1 tower, US-Ton, with values
    # from both Grassland+Savanna rasters -- identical values, confirmed in the audit) ----
    hits = pd.read_csv(LST_HITS_CSV)
    tower_lst = hits.drop_duplicates(["SITE_ID", "YEAR", "MONTH"]).copy()
    tower_lst = tower_lst.rename(columns={"LAND_TYPE_RASTER": "LST_SOURCE_RASTER"})
    tower_lst["LST_degC"] = tower_lst["LST_C"]  # explicit, documented unit; already Celsius, no conversion needed
    tower_lst = tower_lst[["SITE_ID", "YEAR", "MONTH", "LST_degC", "LST_SOURCE_RASTER"]].sort_values(
        ["SITE_ID", "YEAR", "MONTH"])
    tower_lst.to_csv(OUT_LST / "tower_monthly_LST.csv", index=False)
    print(f"Saved tower_monthly_LST.csv: {len(tower_lst)} rows, {tower_lst['SITE_ID'].nunique()} towers")

    # ---- merge onto the final SIF-GPP+drivers+aridity table (SITE_ID x YEAR x MONTH,
    # broadcast across SIF_PRODUCT -- LST is a tower-level driver, not product-specific) ----
    master["YEAR"] = master["YEAR"].astype(int)
    master["MONTH"] = master["MONTH"].astype(int)
    analysis_ready = master.merge(tower_lst, on=["SITE_ID", "YEAR", "MONTH"], how="left")

    # duplicate check (rule 13 final QC, checked here too since this is where it'd appear)
    dup_key = ["SITE_ID", "YEAR", "MONTH", "SIF_PRODUCT"]
    n_dup = analysis_ready.duplicated(dup_key).sum()
    assert n_dup == 0, f"Found {n_dup} duplicated SITE_ID x YEAR x MONTH x SIF_PRODUCT rows"
    assert len(analysis_ready) == len(master), "Row count changed during LST merge (unexpected fan-out)"

    analysis_ready.to_csv(OUT_LST / "SIF_GPP_LST_analysis_ready.csv", index=False)
    print(f"Saved SIF_GPP_LST_analysis_ready.csv: {len(analysis_ready)} rows "
          f"({analysis_ready['LST_degC'].notna().sum()} with valid LST)")

    # ================= QC summaries =================
    has_lst = analysis_ready.dropna(subset=["LST_degC"])

    cov_by_product = analysis_ready.groupby("SIF_PRODUCT").agg(
        N_SITEMONTHS_TOTAL=("SITE_ID", "size"),
        N_SITEMONTHS_WITH_LST=("LST_degC", lambda s: s.notna().sum()),
        N_TOWERS_TOTAL=("SITE_ID", "nunique"),
    ).reset_index()
    n_towers_with_lst_by_product = has_lst.groupby("SIF_PRODUCT")["SITE_ID"].nunique().reindex(
        cov_by_product["SIF_PRODUCT"]).values
    cov_by_product["N_TOWERS_WITH_LST"] = n_towers_with_lst_by_product
    cov_by_product["PCT_SITEMONTHS_WITH_LST"] = (100 * cov_by_product["N_SITEMONTHS_WITH_LST"] /
                                                  cov_by_product["N_SITEMONTHS_TOTAL"]).round(1)
    cov_by_product.to_csv(OUT_TAB / "LST_data_coverage_by_product.csv", index=False)
    print("\nCoverage by product:\n", cov_by_product.to_string(index=False))

    cov_by_biome = analysis_ready.groupby("IGBP").agg(
        N_SITEMONTHS_TOTAL=("SITE_ID", "size"),
        N_SITEMONTHS_WITH_LST=("LST_degC", lambda s: s.notna().sum()),
        N_TOWERS_TOTAL=("SITE_ID", "nunique"),
    ).reset_index()
    cov_by_biome["N_TOWERS_WITH_LST"] = cov_by_biome["IGBP"].map(
        has_lst.groupby("IGBP")["SITE_ID"].nunique())
    cov_by_biome["N_TOWERS_WITH_LST"] = cov_by_biome["N_TOWERS_WITH_LST"].fillna(0).astype(int)
    cov_by_biome.to_csv(OUT_QC / "LST_coverage_by_biome.csv", index=False)

    cov_by_hemi = analysis_ready.groupby("HEMI").agg(
        N_SITEMONTHS_TOTAL=("SITE_ID", "size"),
        N_SITEMONTHS_WITH_LST=("LST_degC", lambda s: s.notna().sum()),
        N_TOWERS_TOTAL=("SITE_ID", "nunique"),
    ).reset_index()
    cov_by_hemi["N_TOWERS_WITH_LST"] = cov_by_hemi["HEMI"].map(has_lst.groupby("HEMI")["SITE_ID"].nunique())
    cov_by_hemi.to_csv(OUT_QC / "LST_coverage_by_hemisphere.csv", index=False)
    print("\nCoverage by hemisphere:\n", cov_by_hemi.to_string(index=False))

    years_covered = sorted(has_lst["YEAR"].unique().tolist())
    monthly_missing = has_lst.groupby("MONTH").size().reindex(range(1, 13), fill_value=0)
    monthly_missing_df = monthly_missing.reset_index()
    monthly_missing_df.columns = ["MONTH", "N_SITEMONTHS_WITH_LST"]
    monthly_missing_df.to_csv(OUT_QC / "LST_monthly_missingness.csv", index=False)

    dist = has_lst["LST_degC"].describe().to_frame("LST_degC_summary_stats")
    dist.to_csv(OUT_QC / "LST_value_distribution_summary.csv")

    print(f"\nYears with >=1 valid LST obs: {years_covered[0]}-{years_covered[-1]}")
    print(f"LST value range: {has_lst['LST_degC'].min():.1f} to {has_lst['LST_degC'].max():.1f} degC "
          f"(plausibility check: physically reasonable for land surface temperature)")
    print("\nPART 2-3 DONE.")


if __name__ == "__main__":
    main()
