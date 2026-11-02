"""Merge monthly WD-bin histograms, conservative monthly GPP and site metadata by SITE_ID and YYYYMM into the monthly footprint input table.

Input: outputs of 01a and 02a, site metadata workbook. Output: merged monthly footprint-input CSV."""
# Original script filename (sanitized): Build Monthly WD Histogram and Conservative GPP Input for GEE.py; public name: 03a_build_monthly_wd_gpp_footprint_input.py
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
# ============================================================
# Code name: Build Monthly WD Histogram and Conservative GPP Input for GEE
#
# Purpose:
#   Prepare the final monthly CSV that should be uploaded to GEE.
#
# What this code does:
#   1. Reads monthly 30-minute WD-bin histogram outputs.
#   2. Reads conservative monthly tower GPP outputs.
#   3. Reads site metadata from Sites Information.xlsx.
#   4. Merges everything by SITE_ID and YYYYMM.
#   5. Exports one final GEE-ready CSV.
#
# Final GEE input:
#   ALL_SITES_monthly_WD_histogram_GPP_footprint_input_for_GEE.csv
#
# Each row = one site + one month + one WD bin.
#
# In GEE:
#   Each WD bin becomes one weighted wedge.
#   The monthly footprint probability is the sum of all weighted WD-bin wedges.
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
import re


# ============================================================
# 1. USER SETTINGS
# ============================================================

project_folder = Path(
    PROJECT_ROOT
)

base_folder = project_folder / "fluxnet"

wd_output_folder = (
    base_folder
    / "Hourly Data (for Wind Direction)"
    / "clean_30min_WD_outputs"
)

gpp_output_folder = (
    base_folder
    / "Hourly Data (for Wind Direction)"
    / "GPP_outputs_for_GEE_conservative"
)

site_info_file = project_folder / "Sites Information.xlsx"

wd_monthly_bins_file = wd_output_folder / "ALL_SITES_monthly_30min_WD_bins.csv"

gpp_monthly_file = gpp_output_folder / "ALL_SITES_monthly_GPP_conservative_for_GEE.csv"

output_folder = (
    base_folder
    / "Hourly Data (for Wind Direction)"
    / "final_monthly_GEE_inputs"
)

output_folder.mkdir(exist_ok=True)

output_file = output_folder / "ALL_SITES_monthly_WD_histogram_GPP_footprint_input_for_GEE.csv"

status_file = output_folder / "monthly_WD_GPP_merge_status.csv"


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def clean_column_names(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def standardize_site_info_columns(site_df):
    site_df = clean_column_names(site_df)

    rename_map = {
        "Site": "SITE_ID",
        "site": "SITE_ID",
        "Site ID": "SITE_ID",
        "Site_ID": "SITE_ID",
        "SITE": "SITE_ID",

        "Site name": "site_name",
        "Site Name": "site_name",
        "site name": "site_name",
        "SITE_NAME": "site_name",

        "Latitude": "lat",
        "Lat": "lat",
        "LAT": "lat",

        "Longitude": "lon",
        "Lon": "lon",
        "LON": "lon",

        "Canopy height m": "canopy_height_m",
        "Canopy height": "canopy_height_m",

        "EC measurement height m": "EC_measurement_height_m",
        "EC measurement height": "EC_measurement_height_m",

        "Height above canopy m": "height_above_canopy_m",
        "Height above canopy": "height_above_canopy_m",

        "Circular buffer radius m": "radius_m",
        "Circular buffer radius": "radius_m",

        "FLUXNET page": "FLUXNET_page",
        "Fluxnet page": "FLUXNET_page"
    }

    site_df = site_df.rename(columns={c: rename_map.get(c, c) for c in site_df.columns})

    return site_df


def parse_min_max_from_string(value):
    if pd.isna(value):
        return np.nan, np.nan

    text = str(value).replace("–", "-").replace("—", "-")
    nums = re.findall(r"-?\d+\.?\d*", text)

    if len(nums) == 0:
        return np.nan, np.nan

    nums = [float(x) for x in nums]

    if len(nums) == 1:
        return nums[0], nums[0]

    return min(nums), max(nums)


def ensure_numeric_min_max_columns(site_df):
    site_df = site_df.copy()

    expected_pairs = [
        ("canopy_height_m", "canopy_height_m_min", "canopy_height_m_max"),
        ("EC_measurement_height_m", "EC_measurement_height_m_min", "EC_measurement_height_m_max"),
        ("height_above_canopy_m", "height_above_canopy_m_min", "height_above_canopy_m_max"),
        ("radius_m", "radius_m_min", "radius_m_max")
    ]

    for source_col, min_col, max_col in expected_pairs:
        if min_col not in site_df.columns:
            site_df[min_col] = np.nan

        if max_col not in site_df.columns:
            site_df[max_col] = np.nan

        if source_col in site_df.columns:
            for idx, value in site_df[source_col].items():
                parsed_min, parsed_max = parse_min_max_from_string(value)

                if pd.isna(site_df.loc[idx, min_col]):
                    site_df.loc[idx, min_col] = parsed_min

                if pd.isna(site_df.loc[idx, max_col]):
                    site_df.loc[idx, max_col] = parsed_max

    numeric_cols = [
        "lat",
        "lon",
        "canopy_height_m_min",
        "canopy_height_m_max",
        "EC_measurement_height_m_min",
        "EC_measurement_height_m_max",
        "height_above_canopy_m_min",
        "height_above_canopy_m_max",
        "radius_m_min",
        "radius_m_max"
    ]

    for col in numeric_cols:
        if col in site_df.columns:
            site_df[col] = pd.to_numeric(site_df[col], errors="coerce")

    if "radius_m_min" in site_df.columns and "height_above_canopy_m_min" in site_df.columns:
        missing_radius_min = site_df["radius_m_min"].isna()
        site_df.loc[missing_radius_min, "radius_m_min"] = (
            site_df.loc[missing_radius_min, "height_above_canopy_m_min"] * 100
        )

    if "radius_m_max" in site_df.columns and "height_above_canopy_m_max" in site_df.columns:
        missing_radius_max = site_df["radius_m_max"].isna()
        site_df.loc[missing_radius_max, "radius_m_max"] = (
            site_df.loc[missing_radius_max, "height_above_canopy_m_max"] * 100
        )

    return site_df


def find_gpp_column(gpp_df):
    preferred = [
        "tower_GPP_monthly_mean_conservative_daytime",
        "tower_GPP_monthly_mean_of_daily_daytime",
        "tower_GPP_monthly_mean_daytime",
        "tower_GPP_monthly",
        "GPP_monthly"
    ]

    for col in preferred:
        if col in gpp_df.columns:
            return col

    possible = [
        c for c in gpp_df.columns
        if "GPP" in c
        and "monthly" in c
        and "quality" not in c.lower()
        and "flag" not in c.lower()
        and "units" not in c.lower()
    ]

    if possible:
        return possible[0]

    raise ValueError(
        "Could not find a monthly GPP column. Check the monthly GPP CSV column names."
    )


def check_required_file(path, label):
    if not isinstance(path, Path):
        raise TypeError(
            f"{label} is not a pathlib.Path object. Current value:\n{path}"
        )

    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found:\n{path}"
        )


# ============================================================
# 3. CHECK INPUT FILES
# ============================================================

check_required_file(wd_monthly_bins_file, "WD monthly bins file")
check_required_file(gpp_monthly_file, "GPP monthly file")
check_required_file(site_info_file, "Sites Information file")


# ============================================================
# 4. READ INPUT FILES
# ============================================================

wd_df = pd.read_csv(wd_monthly_bins_file, low_memory=False)
gpp_df = pd.read_csv(gpp_monthly_file, low_memory=False)
site_df = pd.read_excel(site_info_file)

wd_df = clean_column_names(wd_df)
gpp_df = clean_column_names(gpp_df)
site_df = standardize_site_info_columns(site_df)
site_df = ensure_numeric_min_max_columns(site_df)


# ============================================================
# 5. STANDARDIZE KEY COLUMNS
# ============================================================

if "SITE_ID" not in wd_df.columns:
    raise ValueError("WD monthly bins file does not contain SITE_ID.")

if "YYYYMM" not in wd_df.columns:
    raise ValueError("WD monthly bins file does not contain YYYYMM.")

if "SITE_ID" not in gpp_df.columns:
    raise ValueError("GPP monthly file does not contain SITE_ID.")

if "YYYYMM" not in gpp_df.columns:
    raise ValueError("GPP monthly file does not contain YYYYMM.")

if "SITE_ID" not in site_df.columns:
    raise ValueError("Sites Information file does not contain SITE_ID or Site column.")

wd_df["SITE_ID"] = wd_df["SITE_ID"].astype(str).str.strip()
gpp_df["SITE_ID"] = gpp_df["SITE_ID"].astype(str).str.strip()
site_df["SITE_ID"] = site_df["SITE_ID"].astype(str).str.strip()

wd_df["YYYYMM"] = pd.to_numeric(wd_df["YYYYMM"], errors="coerce").astype("Int64")
gpp_df["YYYYMM"] = pd.to_numeric(gpp_df["YYYYMM"], errors="coerce").astype("Int64")

wd_df = wd_df[wd_df["YYYYMM"].notna()].copy()
gpp_df = gpp_df[gpp_df["YYYYMM"].notna()].copy()

wd_df["YYYYMM"] = wd_df["YYYYMM"].astype(int)
gpp_df["YYYYMM"] = gpp_df["YYYYMM"].astype(int)


# ============================================================
# 6. STANDARDIZE WD BIN COLUMNS
# ============================================================

required_wd_cols = [
    "WD_bin_center_deg",
    "WD_bin_fraction",
    "WD_bin_count"
]

missing_wd_cols = [c for c in required_wd_cols if c not in wd_df.columns]

if missing_wd_cols:
    raise ValueError(
        "WD monthly bins file is missing these columns:\n"
        + ", ".join(missing_wd_cols)
    )

wd_df["WD_bin_center_deg"] = pd.to_numeric(wd_df["WD_bin_center_deg"], errors="coerce")
wd_df["WD_bin_fraction"] = pd.to_numeric(wd_df["WD_bin_fraction"], errors="coerce")
wd_df["WD_bin_count"] = pd.to_numeric(wd_df["WD_bin_count"], errors="coerce")

if "WD_bin_width_deg" not in wd_df.columns:
    wd_df["WD_bin_width_deg"] = 10

wd_df["WD_bin_width_deg"] = pd.to_numeric(wd_df["WD_bin_width_deg"], errors="coerce")

if "WD_bin_start_deg" not in wd_df.columns:
    wd_df["WD_bin_start_deg"] = wd_df["WD_bin_center_deg"] - (wd_df["WD_bin_width_deg"] / 2)

if "WD_bin_end_deg" not in wd_df.columns:
    wd_df["WD_bin_end_deg"] = wd_df["WD_bin_center_deg"] + (wd_df["WD_bin_width_deg"] / 2)

wd_df["WD_bin_start_deg"] = pd.to_numeric(wd_df["WD_bin_start_deg"], errors="coerce")
wd_df["WD_bin_end_deg"] = pd.to_numeric(wd_df["WD_bin_end_deg"], errors="coerce")

if "WD_bin_center_cardinal" not in wd_df.columns:
    wd_df["WD_bin_center_cardinal"] = ""


# ============================================================
# 7. PREPARE MONTHLY GPP TABLE
# ============================================================

gpp_col = find_gpp_column(gpp_df)

keep_gpp_cols = [
    "SITE_ID",
    "YYYYMM",
    gpp_col
]

optional_gpp_cols = [
    "year",
    "month",
    "primary_GPP_column",
    "GPP_monthly_quality_flag",
    "n_valid_daily_daytime_GPP_days",
    "n_valid_conservative_daytime_GPP_days",
    "tower_GPP_units_original_FLUXNET"
]

for col in optional_gpp_cols:
    if col in gpp_df.columns and col not in keep_gpp_cols:
        keep_gpp_cols.append(col)

gpp_keep = gpp_df[keep_gpp_cols].copy()
gpp_keep = gpp_keep.rename(columns={gpp_col: "tower_GPP_monthly"})


# ============================================================
# 8. PREPARE SITE METADATA TABLE
# ============================================================

required_site_cols = [
    "SITE_ID",
    "site_name",
    "lat",
    "lon",
    "IGBP",
    "radius_m_min",
    "radius_m_max"
]

missing_site_cols = [c for c in required_site_cols if c not in site_df.columns]

if missing_site_cols:
    raise ValueError(
        "Sites Information file is missing these required columns:\n"
        + ", ".join(missing_site_cols)
    )

site_keep_cols = [
    "SITE_ID",
    "site_name",
    "lat",
    "lon",
    "IGBP",
    "canopy_height_m",
    "canopy_height_m_min",
    "canopy_height_m_max",
    "EC_measurement_height_m",
    "EC_measurement_height_m_min",
    "EC_measurement_height_m_max",
    "height_above_canopy_m",
    "height_above_canopy_m_min",
    "height_above_canopy_m_max",
    "radius_m",
    "radius_m_min",
    "radius_m_max",
    "FLUXNET_page"
]

site_keep_cols = [c for c in site_keep_cols if c in site_df.columns]
site_keep = site_df[site_keep_cols].copy()


# ============================================================
# 9. MERGE WD BINS + GPP + SITE METADATA
# ============================================================

merged = wd_df.merge(
    gpp_keep,
    on=["SITE_ID", "YYYYMM"],
    how="left",
    indicator="GPP_merge_status"
)

merged = merged.merge(
    site_keep,
    on="SITE_ID",
    how="left",
    suffixes=("", "_site")
)

for base_col in ["site_name", "lat", "lon", "IGBP"]:
    site_col = base_col + "_site"

    if site_col in merged.columns:
        if base_col in merged.columns:
            merged[base_col] = merged[site_col].combine_first(merged[base_col])
        else:
            merged[base_col] = merged[site_col]

        merged = merged.drop(columns=[site_col])

if "year" not in merged.columns:
    merged["year"] = (merged["YYYYMM"] // 100).astype(int)

if "month" not in merged.columns:
    merged["month"] = (merged["YYYYMM"] % 100).astype(int)

merged["tower_GPP_monthly"] = pd.to_numeric(merged["tower_GPP_monthly"], errors="coerce")

merged["has_GPP"] = merged["tower_GPP_monthly"].notna().astype(int)

merged["valid_for_GEE_export"] = (
    merged["lat"].notna()
    & merged["lon"].notna()
    & merged["radius_m_max"].notna()
    & merged["WD_bin_center_deg"].notna()
    & merged["WD_bin_fraction"].notna()
    & merged["tower_GPP_monthly"].notna()
)

merged["valid_for_GEE_export"] = merged["valid_for_GEE_export"].astype(int)

merged["footprint_method"] = "monthly_probability_from_30min_WD_histogram"
merged["blind_buffer_method"] = "tower_centered_circular_buffer"


# ============================================================
# 10. ORDER COLUMNS
# ============================================================

priority_cols = [
    "SITE_ID",
    "site_name",
    "lat",
    "lon",
    "IGBP",
    "YYYYMM",
    "year",
    "month",

    "WD_bin_width_deg",
    "WD_bin_start_deg",
    "WD_bin_end_deg",
    "WD_bin_center_deg",
    "WD_bin_center_cardinal",
    "WD_bin_count",
    "WD_bin_fraction",
    "n_valid_monthly_daytime_WD_records",

    "monthly_circular_mean_WD_deg_diagnostic",
    "monthly_circular_mean_WD_cardinal_diagnostic",
    "monthly_WD_consistency_R_diagnostic",

    "tower_GPP_monthly",
    "primary_GPP_column",
    "GPP_monthly_quality_flag",
    "tower_GPP_units_original_FLUXNET",

    "canopy_height_m",
    "canopy_height_m_min",
    "canopy_height_m_max",
    "EC_measurement_height_m",
    "EC_measurement_height_m_min",
    "EC_measurement_height_m_max",
    "height_above_canopy_m",
    "height_above_canopy_m_min",
    "height_above_canopy_m_max",
    "radius_m",
    "radius_m_min",
    "radius_m_max",

    "mean_monthly_daytime_WS_F",
    "mean_monthly_daytime_USTAR",
    "mean_monthly_daytime_SW_IN_F",

    "daytime_filter_used",
    "quality_flag",
    "footprint_method",
    "blind_buffer_method",
    "GPP_merge_status",
    "has_GPP",
    "valid_for_GEE_export",
    "FLUXNET_page"
]

ordered_cols = [c for c in priority_cols if c in merged.columns]
remaining_cols = [c for c in merged.columns if c not in ordered_cols]

merged = merged[ordered_cols + remaining_cols]


# ============================================================
# 11. SAVE OUTPUTS
# ============================================================

merged.to_csv(output_file, index=False)

status = (
    merged
    .groupby(["SITE_ID", "YYYYMM"], as_index=False)
    .agg(
        n_WD_bins=("WD_bin_center_deg", "count"),
        WD_fraction_sum=("WD_bin_fraction", "sum"),
        has_GPP=("has_GPP", "max"),
        valid_for_GEE_export=("valid_for_GEE_export", "max"),
        tower_GPP_monthly=("tower_GPP_monthly", "first"),
        radius_m_max=("radius_m_max", "first"),
        GPP_merge_status=("GPP_merge_status", "first")
    )
)

status.to_csv(status_file, index=False)


# ============================================================
# 12. TERMINAL OUTPUT
# ============================================================

print("\nDONE.")

print("\nInput files used:")
print(f"WD monthly bins:\n{wd_monthly_bins_file}")
print(f"GPP monthly:\n{gpp_monthly_file}")
print(f"Sites information:\n{site_info_file}")

print(f"\nFinal GEE-ready file saved here:\n{output_file}")
print(f"\nMerge status file saved here:\n{status_file}")

print("\nFinal output rows:", len(merged))
print("Valid rows for GEE export:", int(merged["valid_for_GEE_export"].sum()))

print("\nNumber of unique sites:", merged["SITE_ID"].nunique())
print("Number of unique site-months:", merged[["SITE_ID", "YYYYMM"]].drop_duplicates().shape[0])

print("\nMonthly site-month preview:")
print(status.head(40).to_string(index=False))

print("\nGPP column used:")
print(gpp_col)

print("\nImportant:")
print("Upload this file to GEE:")
print(output_file)