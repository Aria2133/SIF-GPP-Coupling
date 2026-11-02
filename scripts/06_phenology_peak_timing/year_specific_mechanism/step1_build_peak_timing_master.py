"""
STEP 1 -- Build the year-specific peak-timing master table.

For every valid SITE_ID x YEAR x SIF_PRODUCT, estimate POS_SIF and POS_GPP (and, at
the SITE_ID x YEAR level since they don't depend on SIF product, POS_SW_IN, POS_VPD,
POS_TAIR, POS_SWC) from that single year's monthly values, using the harmonic-fit +
delete-one-month-jackknife method in lib_year_specific_pos.py.

DATA SOURCES (existing, validated, NOT recomputed from raw satellite/footprint/EC
data -- see 01_data_inventory/Peak_Timing_Mechanism_Data_Inventory.md for the full
inventory this reuses):
  - SIF_EC_SIF_GPP_Relationship_Analysis/01_analysis_ready/SIF_GPP_Analysis_Ready_Master.csv
    (raw SIF_VALUE, GPP_VALUE per SITE_ID x YEAR x MONTH x SIF_PRODUCT)
  - SIF_EC_Advanced_Coupling_Analysis/01_input_audit/advanced_analysis_input_inventory.csv
    (raw VPD, TAIR, SW_IN, SWC per SITE_ID x YEAR x MONTH -- these do not depend on
    SIF product, so are deduplicated to one row per SITE_ID x YEAR x MONTH)

OUTPUT: 02_peak_timing/SiteYear_Product_PeakTiming_Master.csv
"""
# Original script filename (sanitized): step1_build_peak_timing_master.py; public name: step1_build_peak_timing_master.py
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

import numpy as np
import pandas as pd

SCRIPT_DIR = (Path(PROJECT_ROOT) / 'SIF_GPP_PEAK_TIMING_MECHANISM_FULL_ANALYSIS/14_scripts/step1_build_peak_timing_master.py').resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import lib_year_specific_pos as ly  # noqa: E402

ROOT = Path(PROJECT_ROOT)
MASTER_CSV = ROOT / "SIF_EC_SIF_GPP_Relationship_Analysis" / "01_analysis_ready" / "SIF_GPP_Analysis_Ready_Master.csv"
INVENTORY_CSV = ROOT / "SIF_EC_Advanced_Coupling_Analysis" / "01_input_audit" / "advanced_analysis_input_inventory.csv"
OUT_DIR = (Path(PROJECT_ROOT) / 'SIF_GPP_PEAK_TIMING_MECHANISM_FULL_ANALYSIS/14_scripts/step1_build_peak_timing_master.py').resolve().parents[1] / "02_peak_timing"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRODUCTS = ["GOSIF", "GOME-2", "TROPOSIF", "OCO-3"]
DRIVER_COLS = ["VPD", "TAIR", "SW_IN", "SWC"]


def load_sif_gpp() -> pd.DataFrame:
    df = pd.read_csv(MASTER_CSV, low_memory=False)
    df = df[df["SIF_PRODUCT"].isin(PRODUCTS)].copy()
    df = df.dropna(subset=["IGBP_GPP_TABLE", "LATITUDE", "SIF_VALUE", "GPP_VALUE", "MONTH", "YEAR"])
    df["MONTH"] = df["MONTH"].astype(int)
    df["YEAR"] = df["YEAR"].astype(int)
    site_lat = df.groupby("SITE_ID")["LATITUDE"].first()
    df["HEMI"] = df["SITE_ID"].map(lambda s: "NH" if site_lat[s] >= 0 else "SH")
    return df


def load_drivers() -> pd.DataFrame:
    inv = pd.read_csv(INVENTORY_CSV, low_memory=False)
    inv["MONTH"] = inv["MONTH"].astype(int)
    inv["YEAR"] = inv["YEAR"].astype(int)
    return inv.groupby(["SITE_ID", "YEAR", "MONTH"])[DRIVER_COLS].first().reset_index()


def main():
    df = load_sif_gpp()
    drivers = load_drivers()
    print(f"Loaded {len(df)} SIF-GPP rows, {df[['SITE_ID','SIF_PRODUCT']].drop_duplicates().shape[0]} tower-products")

    rows = []
    for (site, year, product), g in df.groupby(["SITE_ID", "YEAR", "SIF_PRODUCT"]):
        months = g["MONTH"].to_numpy()
        sif_vals = g["SIF_VALUE"].to_numpy(dtype=float)
        gpp_vals = g["GPP_VALUE"].to_numpy(dtype=float)

        sif_res = ly.estimate_pos_one_year(months, sif_vals)
        gpp_res = ly.estimate_pos_one_year(months, gpp_vals)

        delta = ly.pl.circular_diff(sif_res["POS"], gpp_res["POS"])
        if not np.isnan(sif_res["POS"]) and not np.isnan(gpp_res["POS"]):
            ci_low, ci_high = ly.jackknife_delta_pos(months, sif_vals, gpp_vals, sif_res["POS"], gpp_res["POS"])
        else:
            ci_low, ci_high = np.nan, np.nan
        delta_class = ly.classify_delta(delta, ci_low, ci_high)

        peak_shape = "AMBIGUOUS"
        sif_multi = "MULTIMODAL" in sif_res["FLAGS"]
        gpp_multi = "MULTIMODAL" in gpp_res["FLAGS"]
        if sif_res["FLAGS"] == "INSUFFICIENT_COVERAGE" or gpp_res["FLAGS"] == "INSUFFICIENT_COVERAGE":
            peak_shape = "INSUFFICIENT_DATA"
        elif sif_multi or gpp_multi:
            peak_shape = "MULTI_PEAK"
        else:
            peak_shape = "SINGLE_PEAK"

        rows.append({
            "SITE_ID": site, "YEAR": year, "SIF_PRODUCT": product,
            "IGBP": g["IGBP_GPP_TABLE"].iloc[0], "HEMI": g["HEMI"].iloc[0],
            "LATITUDE": g["LATITUDE"].iloc[0], "LONGITUDE": g["LONGITUDE"].iloc[0],
            "SIF_POS": sif_res["POS"], "SIF_POS_CI_LOW": sif_res["POS_CI_LOW"], "SIF_POS_CI_HIGH": sif_res["POS_CI_HIGH"],
            "SIF_N_MONTHS": sif_res["N_MONTHS"], "SIF_SEASONAL_P": sif_res["SEASONAL_P"], "SIF_FLAGS": sif_res["FLAGS"],
            "GPP_POS": gpp_res["POS"], "GPP_POS_CI_LOW": gpp_res["POS_CI_LOW"], "GPP_POS_CI_HIGH": gpp_res["POS_CI_HIGH"],
            "GPP_N_MONTHS": gpp_res["N_MONTHS"], "GPP_SEASONAL_P": gpp_res["SEASONAL_P"], "GPP_FLAGS": gpp_res["FLAGS"],
            "DELTA_POS": delta, "DELTA_POS_CI_LOW": ci_low, "DELTA_POS_CI_HIGH": ci_high,
            "DELTA_CLASS": delta_class, "PEAK_SHAPE": peak_shape,
        })
    master = pd.DataFrame(rows)

    # driver POS (SITE_ID x YEAR, independent of SIF_PRODUCT)
    driver_rows = []
    for (site, year), g in drivers.groupby(["SITE_ID", "YEAR"]):
        months = g["MONTH"].to_numpy()
        rec = {"SITE_ID": site, "YEAR": year}
        for col in DRIVER_COLS:
            vals = g[col].to_numpy(dtype=float)
            valid = ~np.isnan(vals)
            res = ly.estimate_pos_one_year(months[valid], vals[valid]) if valid.sum() >= ly.MIN_UNIQUE_MONTHS_YEAR else \
                dict(POS=np.nan, FLAGS="INSUFFICIENT_COVERAGE", N_MONTHS=int(valid.sum()))
            rec[f"{col}_POS"] = res["POS"]
            rec[f"{col}_FLAGS"] = res["FLAGS"]
        driver_rows.append(rec)
    driver_pos = pd.DataFrame(driver_rows)

    master = master.merge(driver_pos, on=["SITE_ID", "YEAR"], how="left")
    master.to_csv(OUT_DIR / "SiteYear_Product_PeakTiming_Master.csv", index=False)
    print(f"Saved SiteYear_Product_PeakTiming_Master.csv ({len(master)} rows)")
    print(master["DELTA_CLASS"].value_counts())
    print(master["PEAK_SHAPE"].value_counts())


if __name__ == "__main__":
    main()
