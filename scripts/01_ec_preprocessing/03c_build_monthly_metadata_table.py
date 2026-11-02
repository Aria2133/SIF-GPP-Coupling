"""Read the additional sites' FLUXMET HH/HR files and build monthly tables combined with site metadata.

Input: FLUXMET HH/HR files and metadata workbook. Output: monthly combined tables and processing-status reports."""
# Original script filename (sanitized): Expanded FLUXNET Monthly Metadata CSV Builder for GEE.py; public name: 03c_build_monthly_metadata_table.py
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
# Code name: New Remaining FLUXNET Sites HH HR Monthly CSV Builder for GEE
#
# Purpose:
#   This is the updated version of the old monthly FLUXNET
#   metadata/GEE CSV builder.
#
#   It reads the new remaining FLUXNET sites from:
#   <local path removed>
#
#   The folder contains FLUXMET_HH and some FLUXMET_HR files.
#
# What this code does:
#   1. Reads all *_FLUXMET_HH_*.csv and *_FLUXMET_HR_*.csv files.
#   2. Extracts site IDs from filenames.
#   3. Converts TIMESTAMP_START to monthly YYYYMM.
#   4. Aggregates HH/HR variables to monthly values.
#   5. Reads metadata from:
#      New_FLUXNET_sites_metadata_for_KMZ_preparation.xlsx
#   6. Merges monthly flux/met data with site metadata.
#   7. Saves GEE-ready CSV files.
#
# Important:
#   This does NOT create KMZ.
#   This does NOT create wind-direction histograms.
#   This does NOT filter only GRA/SAV/WSA anymore.
#   It keeps ALL expanded biomes/sites in the folder.
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
import re


# ============================================================
# 1. SET YOUR FOLDER PATHS
# ============================================================

project_folder = Path(
    PROJECT_ROOT
)

folder = project_folder / "Remaining Fluxnet Sites" / "HH files"

metadata_path = project_folder / "New_FLUXNET_sites_metadata_for_KMZ_preparation.xlsx"

output_folder = folder / "monthly_GEE_csv_outputs_from_HH_HR"
output_folder.mkdir(parents=True, exist_ok=True)

if not folder.exists():
    raise FileNotFoundError(f"Folder not found:\n{folder}")

if not metadata_path.exists():
    raise FileNotFoundError(f"Metadata Excel file not found:\n{metadata_path}")

print("Working HH/HR folder:")
print(folder)

print("\nMetadata file:")
print(metadata_path)

print("\nOutput folder:")
print(output_folder)


# ============================================================
# 2. FIND NEW FLUXMET HH / HR FILES
# ============================================================

file_patterns = [
    "*_FLUXMET_HH_*.csv",
    "*_FLUXMET_HR_*.csv",
]

flux_files = []

for pattern in file_patterns:
    flux_files.extend(sorted(folder.glob(pattern)))

flux_files = sorted(set(flux_files), key=lambda p: p.name)

if len(flux_files) == 0:
    raise FileNotFoundError(
        "No FLUXMET HH/HR CSV files were found in this folder.\n\n"
        f"Folder checked:\n{folder}\n\n"
        "Expected patterns:\n"
        "*_FLUXMET_HH_*.csv\n"
        "*_FLUXMET_HR_*.csv"
    )

print(f"\nHH/HR files found: {len(flux_files)}")
for f in flux_files:
    print(f.name)


# ============================================================
# 3. COLUMNS TO KEEP FROM HH / HR FILES
# ============================================================

keep_cols = [
    "TIMESTAMP_START",
    "TIMESTAMP_END",

    "GPP_NT_VUT_REF",
    "GPP_NT_VUT_25",
    "GPP_NT_VUT_50",
    "GPP_NT_VUT_75",

    "GPP_DT_VUT_REF",
    "GPP_DT_VUT_25",
    "GPP_DT_VUT_50",
    "GPP_DT_VUT_75",

    "NEE_VUT_REF",
    "RECO_NT_VUT_REF",
    "RECO_DT_VUT_REF",

    "TA_F",
    "TA_F_QC",
    "VPD_F",
    "VPD_F_QC",
    "P_F",
    "P_F_QC",
    "SW_IN_F",
    "SW_IN_F_QC",
    "LW_IN_F",
    "LW_IN_F_QC",
    "WS_F",
    "WS_F_QC",
    "PA_F",
    "PA_F_QC",

    "LE_F_MDS",
    "LE_F_MDS_QC",
    "H_F_MDS",
    "H_F_MDS_QC",

    "SWC_F_MDS_1",
    "SWC_F_MDS_1_QC",
    "TS_F_MDS_1",
    "TS_F_MDS_1_QC",
]


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def extract_site_id(filename):
    """
    Extract site ID from new FLUXMET filenames.

    Examples:
        AMF_BR-Ma2_FLUXNET_FLUXMET_HH_1999-2016_v1.3_r1.csv
        AMF_US-Ne1_FLUXNET_FLUXMET_HR_2001-2024_v1.3_r1.csv
        CNF_CN-Dmn_FLUXNET_FLUXMET_HH_2012-2024_v1.3_r1.csv
        EUF_CH-Oe2_FLUXNET_FLUXMET_HH_2004-2024_v1.3_r1.csv
        ICOS_FR-Gri_FLUXNET_FLUXMET_HH_2004-2024_v1.3_r1.csv
        JP_JP-Tak_FLUXNET_FLUXMET_HH_1998-2021_v1.3_r1.csv
        TERN_AU-How_FLUXNET_FLUXMET_HH_2002-2025_v1.3_r1.csv
    """

    match = re.search(
        r"^(?:AMF|CNF|EUF|ICOS|JP|JPF|TERN)_([A-Z]{2}-[A-Za-z0-9]+)_FLUXNET_FLUXMET_(?:HH|HR)_",
        filename
    )

    if match:
        return match.group(1)

    return None


def parse_fluxnet_timestamp(series):
    """
    Parse FLUXNET TIMESTAMP_START values.

    Handles normal integer/string timestamps such as:
        200101010000
        200101010030

    Also handles possible scientific notation after Excel/Pandas reading.
    """

    s = series.astype(str).str.strip()

    def clean_one(x):
        try:
            if "E" in x.upper():
                return str(int(float(x)))
            if x.endswith(".0"):
                return x[:-2]
            return x
        except Exception:
            return np.nan

    s = s.apply(clean_one)
    s = s.str.replace(r"\D", "", regex=True)
    s = s.str.zfill(12)
    s = s.str.slice(0, 12)

    return pd.to_datetime(s, format="%Y%m%d%H%M", errors="coerce")


def monthly_sum_min_count(x):
    """
    Monthly sum that returns NaN if all values are NaN.
    Used mainly for precipitation.
    """
    return x.sum(min_count=1)


def clean_column_names(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def standardize_metadata_columns(site_df):
    """
    Read your new metadata Excel and standardize column names.
    """

    site_df = clean_column_names(site_df)

    rename_map = {
        "Site": "site_id",
        "site": "site_id",
        "SITE": "site_id",
        "SITE_ID": "site_id",
        "Site ID": "site_id",
        "Site_ID": "site_id",

        "Site name": "site_name",
        "Site Name": "site_name",
        "SITE_NAME": "site_name",

        "Lat": "lat",
        "Latitude": "lat",
        "LAT": "lat",

        "Lon": "lon",
        "Longitude": "lon",
        "LON": "lon",

        "IGBP": "igbp",
        "Igbp": "igbp",

        "Canopy height m": "canopy_height_m",
        "Canopy height": "canopy_height_m",

        "EC measurement height m": "EC_measurement_height_m",
        "EC measurement height": "EC_measurement_height_m",

        "Height above canopy m": "height_above_canopy_m",
        "Height above canopy": "height_above_canopy_m",

        "circle buffer radius m": "circle_buffer_radius_m",
        "Circle buffer radius m": "circle_buffer_radius_m",
        "Circular buffer radius m": "circle_buffer_radius_m",
        "circular buffer radius m": "circle_buffer_radius_m",

        "Dominant daytime WD": "Dominant_daytime_WD",
        "WD deg": "WD_deg",
        "WD consistency R": "WD_consistency_R",
        "Suggested Footprint": "Suggested_Footprint",
        "FLUXNET page": "FLUXNET_page",
    }

    site_df = site_df.rename(columns=rename_map)

    required_cols = ["site_id", "lat", "lon", "igbp"]
    missing_cols = [c for c in required_cols if c not in site_df.columns]

    if missing_cols:
        raise ValueError(
            "Metadata Excel is missing required columns after renaming:\n"
            f"{missing_cols}\n\n"
            "Check the headings in New_FLUXNET_sites_metadata_for_KMZ_preparation.xlsx"
        )

    site_df["site_id"] = site_df["site_id"].astype(str).str.strip()

    if "site_name" in site_df.columns:
        site_df["site_name"] = site_df["site_name"].astype(str).str.strip()

    site_df["igbp"] = site_df["igbp"].astype(str).str.strip()

    numeric_cols = [
        "lat",
        "lon",
        "canopy_height_m",
        "EC_measurement_height_m",
        "height_above_canopy_m",
        "circle_buffer_radius_m",
        "WD_deg",
        "WD_consistency_R",
    ]

    for col in numeric_cols:
        if col in site_df.columns:
            site_df[col] = pd.to_numeric(site_df[col], errors="coerce")

    # Add columns from old BIF workflow if they do not exist.
    # This keeps the final table structurally close to the old output.
    if "country" not in site_df.columns:
        site_df["country"] = pd.NA

    if "elevation" not in site_df.columns:
        site_df["elevation"] = pd.NA

    if "koppen_climate" not in site_df.columns:
        site_df["koppen_climate"] = pd.NA

    if "mean_annual_temperature" not in site_df.columns:
        site_df["mean_annual_temperature"] = pd.NA

    if "mean_annual_precipitation" not in site_df.columns:
        site_df["mean_annual_precipitation"] = pd.NA

    site_df = site_df.drop_duplicates(subset=["site_id"]).copy()

    return site_df


# ============================================================
# 5. READ METADATA FROM NEW EXCEL FILE
# ============================================================

metadata_raw = pd.read_excel(metadata_path)
metadata = standardize_metadata_columns(metadata_raw)

print("\nMetadata table:")
print(metadata.head())

print("\nMetadata sites:", metadata["site_id"].nunique())

print("\nMetadata IGBP counts:")
print(metadata["igbp"].value_counts(dropna=False))


# ============================================================
# 6. READ HH / HR FILES AND AGGREGATE TO MONTHLY
# ============================================================

all_monthly = []
status_rows = []

for file in flux_files:
    site_id = extract_site_id(file.name)

    if site_id is None:
        print(f"\nSkipping file because site ID could not be extracted: {file.name}")

        status_rows.append({
            "site_id": "",
            "file": file.name,
            "status": "SITE_ID_NOT_EXTRACTED",
            "n_input_rows": np.nan,
            "n_monthly_rows": np.nan,
        })

        continue

    print(f"\nProcessing {site_id}: {file.name}")

    try:
        df = pd.read_csv(
            file,
            usecols=lambda c: str(c).strip() in keep_cols,
            low_memory=False
        )
        df = clean_column_names(df)

        if "TIMESTAMP_START" not in df.columns:
            print(f"  WARNING: TIMESTAMP_START not found in {file.name}")

            status_rows.append({
                "site_id": site_id,
                "file": file.name,
                "status": "NO_TIMESTAMP_START",
                "n_input_rows": len(df),
                "n_monthly_rows": np.nan,
            })

            continue

        available_cols = [c for c in keep_cols if c in df.columns]
        df = df[available_cols].copy()

        df["site_id"] = site_id
        df["source_file"] = file.name

        df["datetime"] = parse_fluxnet_timestamp(df["TIMESTAMP_START"])

        df = df[df["datetime"].notna()].copy()

        df["YYYYMM"] = df["datetime"].dt.strftime("%Y%m").astype(int)
        df["year"] = df["datetime"].dt.year.astype(int)
        df["month"] = df["datetime"].dt.month.astype(int)

        # Convert all measurement columns to numeric and replace missing values.
        data_cols = [
            c for c in df.columns
            if c not in [
                "site_id",
                "source_file",
                "TIMESTAMP_START",
                "TIMESTAMP_END",
                "datetime",
                "YYYYMM",
                "year",
                "month",
            ]
        ]

        for col in data_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] == -9999, col] = np.nan

        # Monthly aggregation:
        #   Most variables: monthly mean
        #   P_F: monthly sum
        #   QC columns: monthly mean of QC flag
        agg_dict = {}

        for col in data_cols:
            if col == "P_F":
                agg_dict[col] = monthly_sum_min_count
            else:
                agg_dict[col] = "mean"

        monthly = (
            df
            .groupby(["site_id", "YYYYMM"], as_index=False)
            .agg(agg_dict)
        )

        monthly["source_file"] = file.name
        monthly["date"] = pd.to_datetime(monthly["YYYYMM"].astype(str), format="%Y%m", errors="coerce")
        monthly["TIMESTAMP"] = monthly["YYYYMM"].astype(str)
        monthly["year"] = monthly["date"].dt.year.astype("Int64")
        monthly["month"] = monthly["date"].dt.month.astype("Int64")

        record_counts = (
            df
            .groupby(["site_id", "YYYYMM"])
            .size()
            .reset_index(name="n_HH_HR_records_in_month")
        )

        monthly = monthly.merge(record_counts, on=["site_id", "YYYYMM"], how="left")

        all_monthly.append(monthly)

        status_rows.append({
            "site_id": site_id,
            "file": file.name,
            "status": "OK",
            "n_input_rows": len(df),
            "n_monthly_rows": len(monthly),
            "date_min": monthly["date"].min(),
            "date_max": monthly["date"].max(),
        })

        print(f"  Input rows after valid timestamp: {len(df)}")
        print(f"  Monthly rows: {len(monthly)}")
        print(f"  Date range: {monthly['date'].min()} to {monthly['date'].max()}")

    except Exception as e:
        print(f"  ERROR with {site_id}: {e}")

        status_rows.append({
            "site_id": site_id,
            "file": file.name,
            "status": f"ERROR: {e}",
            "n_input_rows": np.nan,
            "n_monthly_rows": np.nan,
        })


if len(all_monthly) == 0:
    raise RuntimeError(
        "No HH/HR files were successfully processed.\n"
        "Check that the files are local, not only OneDrive cloud placeholders, "
        "and that they contain TIMESTAMP_START."
    )

fluxnet_monthly = pd.concat(all_monthly, ignore_index=True)
status_df = pd.DataFrame(status_rows)


# ============================================================
# 7. RENAME COLUMNS TO MATCH OLD OUTPUT STYLE
# ============================================================

fluxnet_monthly = fluxnet_monthly.rename(columns={
    "GPP_NT_VUT_REF": "gpp_nt_ref",
    "GPP_NT_VUT_25": "gpp_nt_25",
    "GPP_NT_VUT_50": "gpp_nt_50",
    "GPP_NT_VUT_75": "gpp_nt_75",

    "GPP_DT_VUT_REF": "gpp_dt_ref",
    "GPP_DT_VUT_25": "gpp_dt_25",
    "GPP_DT_VUT_50": "gpp_dt_50",
    "GPP_DT_VUT_75": "gpp_dt_75",

    "NEE_VUT_REF": "nee_ref",
    "RECO_NT_VUT_REF": "reco_nt_ref",
    "RECO_DT_VUT_REF": "reco_dt_ref",

    "TA_F": "air_temperature",
    "TA_F_QC": "air_temperature_qc",
    "VPD_F": "vpd",
    "VPD_F_QC": "vpd_qc",
    "P_F": "precip",
    "P_F_QC": "precip_qc",
    "SW_IN_F": "shortwave_in",
    "SW_IN_F_QC": "shortwave_in_qc",
    "LW_IN_F": "longwave_in",
    "LW_IN_F_QC": "longwave_in_qc",
    "WS_F": "wind_speed",
    "WS_F_QC": "wind_speed_qc",
    "PA_F": "air_pressure",
    "PA_F_QC": "air_pressure_qc",

    "LE_F_MDS": "latent_heat",
    "LE_F_MDS_QC": "latent_heat_qc",
    "H_F_MDS": "sensible_heat",
    "H_F_MDS_QC": "sensible_heat_qc",

    "SWC_F_MDS_1": "soil_water_content_1",
    "SWC_F_MDS_1_QC": "soil_water_content_1_qc",
    "TS_F_MDS_1": "soil_temperature_1",
    "TS_F_MDS_1_QC": "soil_temperature_1_qc",
})

# Convert numeric columns after renaming.
non_numeric_cols = ["site_id", "source_file", "TIMESTAMP", "date"]

for col in fluxnet_monthly.columns:
    if col not in non_numeric_cols:
        fluxnet_monthly[col] = pd.to_numeric(fluxnet_monthly[col], errors="coerce")

print("\nCombined monthly FLUXNET table:")
print(fluxnet_monthly.head())

print("\nRows:", len(fluxnet_monthly))
print("Sites:", fluxnet_monthly["site_id"].nunique())


# ============================================================
# 8. MERGE MONTHLY FLUXNET DATA WITH NEW SITE METADATA
# ============================================================

final = fluxnet_monthly.merge(metadata, on="site_id", how="left")

first_cols = [
    "site_id",
    "date",
    "TIMESTAMP",
    "YYYYMM",
    "year",
    "month",

    "lat",
    "lon",
    "igbp",
    "country",
    "site_name",
    "elevation",
    "koppen_climate",

    "canopy_height_m",
    "EC_measurement_height_m",
    "height_above_canopy_m",
    "circle_buffer_radius_m",

    "Dominant_daytime_WD",
    "WD_deg",
    "WD_consistency_R",
    "Suggested_Footprint",
    "FLUXNET_page",

    "gpp_nt_ref",
    "gpp_dt_ref",
    "gpp_nt_25",
    "gpp_nt_50",
    "gpp_nt_75",
    "gpp_dt_25",
    "gpp_dt_50",
    "gpp_dt_75",

    "vpd",
    "air_temperature",
    "precip",
    "shortwave_in",
    "longwave_in",
    "wind_speed",
    "latent_heat",
    "sensible_heat",
    "soil_water_content_1",
    "soil_temperature_1",

    "nee_ref",
    "reco_nt_ref",
    "reco_dt_ref",

    "n_HH_HR_records_in_month",
    "source_file",
]

ordered_cols = [c for c in first_cols if c in final.columns]
remaining_cols = [c for c in final.columns if c not in ordered_cols]
final = final[ordered_cols + remaining_cols]

print("\nMerged table:")
print(final.head())

print("\nRows after merge:", len(final))
print("Sites after merge:", final["site_id"].nunique())

missing_meta = final[
    final["lat"].isna()
    | final["lon"].isna()
    | final["igbp"].isna()
]["site_id"].unique()

print("\nSites with missing lat/lon/IGBP metadata:")
print(missing_meta)


# ============================================================
# 9. CREATE SITE-ONLY TABLE FOR GEE POINT UPLOAD
# ============================================================

site_point_cols = [
    "site_id",
    "lat",
    "lon",
    "igbp",
    "country",
    "site_name",
    "elevation",
    "canopy_height_m",
    "EC_measurement_height_m",
    "height_above_canopy_m",
    "circle_buffer_radius_m",
    "Dominant_daytime_WD",
    "WD_deg",
    "WD_consistency_R",
    "Suggested_Footprint",
    "FLUXNET_page",
]

site_point_cols = [c for c in site_point_cols if c in final.columns]

site_points = (
    final[site_point_cols]
    .drop_duplicates(subset=["site_id"])
    .dropna(subset=["lat", "lon"])
    .copy()
)

print("\nSite points for GEE:")
print(site_points)

print("\nNumber of site points:", len(site_points))


# ============================================================
# 10. SAVE OUTPUTS
# ============================================================

out_monthly = output_folder / "new_sites_fluxnet_monthly_combined_from_HH_HR.csv"
out_metadata = output_folder / "new_sites_fluxnet_site_metadata_from_excel.csv"
out_all = output_folder / "new_sites_fluxnet_monthly_with_metadata_all_sites.csv"
out_points = output_folder / "new_sites_fluxnet_site_points_for_GEE.csv"
out_status = output_folder / "new_sites_fluxnet_HH_HR_processing_status.csv"
out_excel = output_folder / "new_sites_fluxnet_monthly_outputs_from_HH_HR.xlsx"

fluxnet_monthly.to_csv(out_monthly, index=False)
metadata.to_csv(out_metadata, index=False)
final.to_csv(out_all, index=False)
site_points.to_csv(out_points, index=False)
status_df.to_csv(out_status, index=False)

with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
    metadata.to_excel(writer, sheet_name="Site_Metadata", index=False)
    site_points.to_excel(writer, sheet_name="Site_Points_for_GEE", index=False)
    status_df.to_excel(writer, sheet_name="Processing_Status", index=False)

    if len(final) <= 1_000_000:
        final.to_excel(writer, sheet_name="Monthly_With_Metadata", index=False)
    else:
        print(
            "\nMonthly_With_Metadata is too large for Excel, "
            "so it was saved only as CSV."
        )

print("\nSaved files:")
print(out_monthly)
print(out_metadata)
print(out_all)
print(out_points)
print(out_status)
print(out_excel)


# ============================================================
# 11. BASIC QUALITY CHECKS
# ============================================================

print("\nQuality check:")
print("All processed sites:", sorted(final["site_id"].dropna().unique()))
print("Number of processed sites:", final["site_id"].nunique())

print("\nIGBP counts in processed site points:")
print(site_points["igbp"].value_counts(dropna=False))

print("\nMissing values in important columns:")
important_cols = [
    "gpp_nt_ref",
    "gpp_dt_ref",
    "vpd",
    "air_temperature",
    "precip",
    "shortwave_in",
    "lat",
    "lon",
    "igbp",
]

important_cols = [c for c in important_cols if c in final.columns]
print(final[important_cols].isna().sum())

print("\nDate range by site:")
date_summary = (
    final
    .groupby("site_id")["date"]
    .agg(["min", "max", "count"])
    .reset_index()
)

print(date_summary.to_string(index=False))

print("\nMain output for GEE/table use:")
print(out_all)

print("\nSite-point output:")
print(out_points)

print("\nDone.")
