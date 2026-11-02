"""Conservative daily/monthly GPP extraction (same filter as 02a) for the additional sites delivered as FLUXMET HH/HR files.

Input: FLUXNET FLUXMET HH/HR files. Output: daily and monthly conservative GPP tables."""
# Original script filename (sanitized): HR Sites.py; public name: 02b_extract_gpp_conservative_HR_sites.py
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
# CODE NAME:
# Conservative GPP Extractor for Remaining FLUXNET HH/HR Sites
#
# PURPOSE:
# Extract daily and monthly tower GPP values from the new FLUXMET HH/HR files.
# Uses the same conservative filter as the previous code:
# NEE_VUT_REF_QC == 0
#
# INPUT FOLDER:
# <local path removed>
#
# MAIN OUTPUT FOR GEE:
# ALL_SITES_monthly_GPP_conservative_for_GEE.csv
#
# RECOMMENDED GEE USE:
# Join this monthly GPP table with your monthly WD footprint-probability table
# using SITE_ID and YYYYMM.
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
import re


# ============================================================
# USER SETTINGS
# ============================================================

folder = Path(
    _os.path.join(PROJECT_ROOT, 'Remaining Fluxnet Sites', 'HH files')
)

metadata_path = Path(
    _os.path.join(PROJECT_ROOT, 'New_FLUXNET_sites_metadata_for_KMZ_preparation.xlsx')
)

output_folder = folder / "GPP_outputs_for_GEE_conservative"
output_folder.mkdir(exist_ok=True)

missing_value = -9999

# Main GPP preference for daytime GPP-SIF work.
# DT = daytime partitioning.
# NT is also exported as sensitivity.
primary_gpp_column = "GPP_DT_VUT_REF"

gpp_columns_to_export = [
    "GPP_DT_VUT_REF",
    "GPP_NT_VUT_REF",
    "GPP_DT_VUT_25",
    "GPP_DT_VUT_50",
    "GPP_DT_VUT_75",
    "GPP_NT_VUT_25",
    "GPP_NT_VUT_50",
    "GPP_NT_VUT_75"
]

min_conservative_daytime_records_per_day = 5
min_conservative_daytime_records_per_month = 100
min_valid_days_per_month = 10


# ============================================================
# HELPER FUNCTIONS
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

    patterns = [
        r"FLX_([A-Z]{2}-[A-Za-z0-9]+)_FLUXNET2015",
        r"(?:AMF|CNF|EUF|ICOS|JP|JPF|TERN)_([A-Z]{2}-[A-Za-z0-9]+)_FLUXNET",
    ]

    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return match.group(1)

    return None


def parse_fluxnet_timestamp(series):
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
    s = s.str.replace(r"\D", "", regex=True).str.zfill(12)
    s = s.str.slice(0, 12)

    return pd.to_datetime(s, format="%Y%m%d%H%M", errors="coerce")


def standardize_metadata_columns(site_df):
    site_df = site_df.copy()
    site_df.columns = [str(c).strip() for c in site_df.columns]

    rename_map = {
        "Site": "SITE_ID",
        "site": "SITE_ID",
        "SITE": "SITE_ID",
        "site_id": "SITE_ID",
        "Site ID": "SITE_ID",
        "Site_ID": "SITE_ID",

        "Site name": "site_name",
        "Site Name": "site_name",
        "SITE_NAME": "site_name",

        "Lat": "lat",
        "Latitude": "lat",
        "LAT": "lat",

        "Lon": "lon",
        "Longitude": "lon",
        "LON": "lon",

        "IGBP": "IGBP",
        "igbp": "IGBP",

        "FLUXNET page": "FLUXNET_page",
        "FLUXNET_page": "FLUXNET_page",
    }

    site_df = site_df.rename(columns=rename_map)

    if "SITE_ID" not in site_df.columns:
        raise ValueError("Metadata file must contain a Site or SITE_ID column.")

    site_df["SITE_ID"] = site_df["SITE_ID"].astype(str).str.strip()

    for col in ["lat", "lon"]:
        if col in site_df.columns:
            site_df[col] = pd.to_numeric(site_df[col], errors="coerce")
        else:
            site_df[col] = np.nan

    for col in ["site_name", "IGBP", "FLUXNET_page"]:
        if col not in site_df.columns:
            site_df[col] = np.nan

    site_df = site_df.drop_duplicates(subset=["SITE_ID"]).copy()

    return site_df


def load_site_metadata(metadata_path):
    if metadata_path.exists():
        metadata = pd.read_excel(metadata_path)
        metadata = standardize_metadata_columns(metadata)

        site_info = {}

        for _, row in metadata.iterrows():
            site_id = row["SITE_ID"]

            site_info[site_id] = [
                row.get("site_name", np.nan),
                row.get("lat", np.nan),
                row.get("lon", np.nan),
                row.get("IGBP", np.nan),
                row.get("FLUXNET_page", np.nan),
            ]

        print(f"Loaded metadata for {len(site_info)} sites from:")
        print(metadata_path)

        return site_info

    print("WARNING: Metadata Excel file was not found.")
    print("The code will still run, but site_name, lat, lon, IGBP, and FLUXNET_page will be blank.")
    return {}


site_info = load_site_metadata(metadata_path)


def add_site_metadata(df, site_id):
    site_name, lat, lon, igbp, page = site_info.get(
        site_id,
        [np.nan, np.nan, np.nan, np.nan, np.nan]
    )

    df.insert(0, "SITE_ID", site_id)
    df.insert(1, "site_name", site_name)
    df.insert(2, "lat", lat)
    df.insert(3, "lon", lon)
    df.insert(4, "IGBP", igbp)
    df["FLUXNET_page"] = page

    return df


def safe_numeric_column(df, col):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] == missing_value, col] = np.nan
    else:
        df[col] = np.nan
    return df


def prepare_hh_dataframe(file, site_id):
    gpp_input_cols = {
        "TIMESTAMP_START",
        "NEE_VUT_REF_QC",
        "NEE_VUT_REF",
        "NIGHT",
        "SW_IN_F",
        "FETCH_FILTER",
        *gpp_columns_to_export,
    }

    df = pd.read_csv(
        file,
        usecols=lambda c: str(c).strip() in gpp_input_cols,
        low_memory=False
    )
    df.columns = [c.strip() for c in df.columns]

    if "TIMESTAMP_START" not in df.columns:
        raise ValueError(f"{site_id}: TIMESTAMP_START column was not found")

    df["datetime"] = parse_fluxnet_timestamp(df["TIMESTAMP_START"])
    df = df[df["datetime"].notna()].copy()

    required_gpp_exists = primary_gpp_column in df.columns

    if not required_gpp_exists:
        available_gpp = [c for c in df.columns if c.startswith("GPP")]
        raise ValueError(
            f"{site_id}: primary GPP column {primary_gpp_column} was not found. "
            f"Available GPP columns: {available_gpp}"
        )

    # Numeric conversion for GPP columns
    available_gpp_columns = []

    for col in gpp_columns_to_export:
        if col in df.columns:
            df = safe_numeric_column(df, col)
            available_gpp_columns.append(col)

    # Required QC and supporting columns
    df = safe_numeric_column(df, "NEE_VUT_REF_QC")
    df = safe_numeric_column(df, "NEE_VUT_REF")
    df = safe_numeric_column(df, "NIGHT")
    df = safe_numeric_column(df, "SW_IN_F")
    df = safe_numeric_column(df, "FETCH_FILTER")

    # Date fields
    df["date"] = df["datetime"].dt.date
    df["YYYYMMDD"] = df["datetime"].dt.strftime("%Y%m%d").astype(int)
    df["YYYYMM"] = df["datetime"].dt.strftime("%Y%m").astype(int)
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day

    # Daytime definition
    if df["NIGHT"].notna().any():
        df["is_daytime"] = df["NIGHT"] == 0
        daytime_filter_used = "NIGHT_equals_0"
    elif df["SW_IN_F"].notna().any():
        df["is_daytime"] = df["SW_IN_F"] > 20
        daytime_filter_used = "SW_IN_F_gt_20"
    else:
        df["is_daytime"] = True
        daytime_filter_used = "no_daytime_filter_available_all_records_used"

    # Conservative GPP filter:
    # Keep GPP only when NEE_VUT_REF_QC == 0.
    # This is not perfect non-gap-filled GPP, but it is much closer to the project's recommendation
    # than using all gap-filled GPP records.
    if df["NEE_VUT_REF_QC"].notna().any():
        df["is_conservative_flux_record"] = df["NEE_VUT_REF_QC"] == 0
        conservative_filter_used = "NEE_VUT_REF_QC_equals_0"
    else:
        df["is_conservative_flux_record"] = True
        conservative_filter_used = "NEE_QC_not_available_all_records_used"

    # FETCH_FILTER availability
    fetch_filter_available = df["FETCH_FILTER"].notna().any()

    return (
        df,
        available_gpp_columns,
        daytime_filter_used,
        conservative_filter_used,
        fetch_filter_available
    )


def aggregate_daily(df, site_id, available_gpp_columns, daytime_filter_used, conservative_filter_used):
    rows = []

    for date_value, group in df.groupby("date"):
        daytime_group = group[group["is_daytime"]].copy()

        conservative_daytime_group = group[
            group["is_daytime"]
            & group["is_conservative_flux_record"]
        ].copy()

        row = {
            "date": date_value,
            "YYYYMMDD": int(group["YYYYMMDD"].iloc[0]),
            "YYYYMM": int(group["YYYYMM"].iloc[0]),
            "year": int(group["year"].iloc[0]),
            "month": int(group["month"].iloc[0]),
            "day": int(group["day"].iloc[0]),
            "primary_GPP_column": primary_gpp_column,
            "daytime_filter_used": daytime_filter_used,
            "conservative_flux_filter_used": conservative_filter_used,
            "n_total_HH_records": int(len(group)),
            "n_daytime_HH_records": int(len(daytime_group)),
            "n_conservative_daytime_HH_records": int(len(conservative_daytime_group))
        }

        # FETCH_FILTER daily summary
        if "FETCH_FILTER" in group.columns and group["FETCH_FILTER"].notna().any():
            row["FETCH_FILTER_available"] = True
            row["FETCH_FILTER_mean_daytime"] = round(daytime_group["FETCH_FILTER"].mean(), 6)
            row["FETCH_FILTER_fraction_accepted_daytime"] = round(
                (daytime_group["FETCH_FILTER"] == 1).mean(), 6
            )
        else:
            row["FETCH_FILTER_available"] = False
            row["FETCH_FILTER_mean_daytime"] = np.nan
            row["FETCH_FILTER_fraction_accepted_daytime"] = np.nan

        for col in available_gpp_columns:
            all_values = group[col].dropna()
            daytime_values = daytime_group[col].dropna()
            conservative_daytime_values = conservative_daytime_group[col].dropna()

            row[f"daily_mean_{col}_all_records"] = (
                round(all_values.mean(), 6) if len(all_values) > 0 else np.nan
            )

            row[f"daily_mean_{col}_daytime_all_QC"] = (
                round(daytime_values.mean(), 6) if len(daytime_values) > 0 else np.nan
            )

            row[f"daily_mean_{col}_daytime_conservative"] = (
                round(conservative_daytime_values.mean(), 6)
                if len(conservative_daytime_values) > 0
                else np.nan
            )

            row[f"n_valid_{col}_daytime_all_QC"] = int(len(daytime_values))
            row[f"n_valid_{col}_daytime_conservative"] = int(len(conservative_daytime_values))

        primary_values = conservative_daytime_group[primary_gpp_column].dropna()

        row["tower_GPP_daily_conservative"] = (
            round(primary_values.mean(), 6) if len(primary_values) > 0 else np.nan
        )

        row["tower_GPP_units_original_FLUXNET"] = "umol_CO2_m-2_s-1"

        if len(primary_values) < min_conservative_daytime_records_per_day:
            row["GPP_daily_quality_flag"] = "LOW_CONSERVATIVE_DAYTIME_RECORD_COUNT"
        else:
            row["GPP_daily_quality_flag"] = "OK"

        rows.append(row)

    out = pd.DataFrame(rows)
    out = add_site_metadata(out, site_id)

    return out


def aggregate_monthly(daily_df, site_id, available_gpp_columns):
    rows = []

    for yyyymm, group in daily_df.groupby("YYYYMM"):
        primary_daily = group["tower_GPP_daily_conservative"].dropna()

        row = {
            "YYYYMM": int(yyyymm),
            "year": int(group["year"].iloc[0]),
            "month": int(group["month"].iloc[0]),
            "primary_GPP_column": primary_gpp_column,
            "n_days_available": int(group["date"].nunique()),
            "n_valid_conservative_GPP_days": int(primary_daily.count()),
            "tower_GPP_monthly_conservative_mean_of_daily": (
                round(primary_daily.mean(), 6) if len(primary_daily) > 0 else np.nan
            ),
            "tower_GPP_monthly_conservative_median_of_daily": (
                round(primary_daily.median(), 6) if len(primary_daily) > 0 else np.nan
            ),
            "tower_GPP_units_original_FLUXNET": "umol_CO2_m-2_s-1"
        }

        # Monthly mean for each exported GPP column
        for col in available_gpp_columns:
            colname = f"daily_mean_{col}_daytime_conservative"

            if colname in group.columns:
                values = group[colname].dropna()

                row[f"monthly_mean_{col}_daytime_conservative"] = (
                    round(values.mean(), 6) if len(values) > 0 else np.nan
                )

                row[f"n_valid_days_{col}_daytime_conservative"] = int(values.count())

        # FETCH_FILTER monthly summary
        if "FETCH_FILTER_available" in group.columns and group["FETCH_FILTER_available"].any():
            row["FETCH_FILTER_available"] = True

            row["FETCH_FILTER_monthly_mean_daytime"] = round(
                group["FETCH_FILTER_mean_daytime"].mean(), 6
            )

            row["FETCH_FILTER_monthly_fraction_accepted_daytime"] = round(
                group["FETCH_FILTER_fraction_accepted_daytime"].mean(), 6
            )
        else:
            row["FETCH_FILTER_available"] = False
            row["FETCH_FILTER_monthly_mean_daytime"] = np.nan
            row["FETCH_FILTER_monthly_fraction_accepted_daytime"] = np.nan

        # Conservative quality flag
        total_conservative_hh = group["n_conservative_daytime_HH_records"].sum()

        if row["n_valid_conservative_GPP_days"] < min_valid_days_per_month:
            row["GPP_monthly_quality_flag"] = "LOW_VALID_DAYS"
        elif total_conservative_hh < min_conservative_daytime_records_per_month:
            row["GPP_monthly_quality_flag"] = "LOW_CONSERVATIVE_HH_RECORDS"
        else:
            row["GPP_monthly_quality_flag"] = "OK"

        rows.append(row)

    out = pd.DataFrame(rows)
    out = add_site_metadata(out, site_id)

    return out


# ============================================================
# MAIN PROCESSING
# ============================================================

if not folder.exists():
    raise FileNotFoundError(f"Folder not found: {folder}")

file_patterns = [
    "*_SUBSET_HH_*.csv",
    "*_SUBSET_HR_*.csv",
    "*_FLUXMET_HH_*.csv",
    "*_FLUXMET_HR_*.csv",
]

files = []

for pattern in file_patterns:
    files.extend(sorted(folder.glob(pattern)))

files = sorted(set(files), key=lambda p: p.name)

if not files:
    raise FileNotFoundError(
        f"No HH/HR CSV files found in: {folder}\n"
        "Expected files like *_FLUXMET_HH_*.csv or *_FLUXMET_HR_*.csv"
    )

print(f"Total HH/HR files found: {len(files)}")

all_daily = []
all_monthly = []
status_rows = []
processed_sites = []

for file in files:
    site_id = extract_site_id(file.name)

    if site_id is None:
        print(f"Skipping file because SITE_ID could not be extracted: {file.name}")

        status_rows.append({
            "SITE_ID": np.nan,
            "file": file.name,
            "status": "SITE_ID_NOT_EXTRACTED",
            "primary_GPP_column": primary_gpp_column,
            "available_GPP_columns": np.nan,
            "daytime_filter_used": np.nan,
            "conservative_flux_filter_used": np.nan,
            "FETCH_FILTER_available": np.nan,
            "n_HH_rows": np.nan,
            "n_daily_rows": np.nan,
            "n_monthly_rows": np.nan
        })

        continue

    print(f"\nProcessing {site_id}: {file.name}")

    try:
        (
            hh_df,
            available_gpp_columns,
            daytime_filter_used,
            conservative_filter_used,
            fetch_filter_available
        ) = prepare_hh_dataframe(file, site_id)

        daily_df = aggregate_daily(
            hh_df,
            site_id,
            available_gpp_columns,
            daytime_filter_used,
            conservative_filter_used
        )

        monthly_df = aggregate_monthly(
            daily_df,
            site_id,
            available_gpp_columns
        )

        site_daily_path = output_folder / f"{site_id}_daily_GPP_conservative_for_GEE.csv"
        site_monthly_path = output_folder / f"{site_id}_monthly_GPP_conservative_for_GEE.csv"

        daily_df.to_csv(site_daily_path, index=False)
        monthly_df.to_csv(site_monthly_path, index=False)

        all_daily.append(daily_df)
        all_monthly.append(monthly_df)
        processed_sites.append(site_id)

        status_rows.append({
            "SITE_ID": site_id,
            "file": file.name,
            "status": "OK",
            "primary_GPP_column": primary_gpp_column,
            "available_GPP_columns": "; ".join(available_gpp_columns),
            "daytime_filter_used": daytime_filter_used,
            "conservative_flux_filter_used": conservative_filter_used,
            "FETCH_FILTER_available": fetch_filter_available,
            "n_HH_rows": len(hh_df),
            "n_daily_rows": len(daily_df),
            "n_monthly_rows": len(monthly_df),
            "daily_GPP_file": str(site_daily_path),
            "monthly_GPP_file": str(site_monthly_path)
        })

        print(f"  HH/HR rows: {len(hh_df)}")
        print(f"  Daily rows: {len(daily_df)}")
        print(f"  Monthly rows: {len(monthly_df)}")
        print(f"  Saved monthly GPP: {site_monthly_path}")

    except Exception as e:
        status_rows.append({
            "SITE_ID": site_id,
            "file": file.name,
            "status": f"ERROR: {e}",
            "primary_GPP_column": primary_gpp_column,
            "available_GPP_columns": np.nan,
            "daytime_filter_used": np.nan,
            "conservative_flux_filter_used": np.nan,
            "FETCH_FILTER_available": np.nan,
            "n_HH_rows": np.nan,
            "n_daily_rows": np.nan,
            "n_monthly_rows": np.nan,
            "daily_GPP_file": "",
            "monthly_GPP_file": ""
        })

        print(f"ERROR with {site_id}: {e}")


# ============================================================
# SAVE OUTPUTS
# ============================================================

all_daily_df = pd.concat(all_daily, ignore_index=True) if all_daily else pd.DataFrame()
all_monthly_df = pd.concat(all_monthly, ignore_index=True) if all_monthly else pd.DataFrame()
status_df = pd.DataFrame(status_rows)

all_daily_csv = output_folder / "ALL_SITES_daily_GPP_conservative_for_GEE.csv"
all_monthly_csv = output_folder / "ALL_SITES_monthly_GPP_conservative_for_GEE.csv"
status_csv = output_folder / "GPP_conservative_processing_status.csv"
excel_path = output_folder / "ALL_SITES_GPP_conservative_daily_monthly_for_GEE.xlsx"

all_daily_df.to_csv(all_daily_csv, index=False)
all_monthly_df.to_csv(all_monthly_csv, index=False)
status_df.to_csv(status_csv, index=False)

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    print("Daily output is saved as CSV only to avoid a very large Excel workbook.")
    all_monthly_df.to_excel(writer, sheet_name="Monthly_GPP_Conservative", index=False)
    status_df.to_excel(writer, sheet_name="Processing_Status", index=False)


# ============================================================
# TERMINAL OUTPUT
# ============================================================

print("\nDONE.")
print(f"Output folder:\n{output_folder}")

print(f"\nSaved daily all-sites conservative GPP CSV:\n{all_daily_csv}")
print(f"Saved monthly all-sites conservative GPP CSV:\n{all_monthly_csv}")
print(f"Saved processing status CSV:\n{status_csv}")
print(f"Saved Excel workbook:\n{excel_path}")

print("\nProcessed site count:")
print(len(set(processed_sites)))

print("\nProcessed sites:")
print(sorted(set(processed_sites)))

print("\nProcessing status:")
if not status_df.empty:
    print(status_df.to_string(index=False))
else:
    print("No files were processed.")

print("\nMonthly conservative GPP preview:")
if not all_monthly_df.empty:
    preview_cols = [
        "SITE_ID",
        "YYYYMM",
        "year",
        "month",
        "primary_GPP_column",
        "n_valid_conservative_GPP_days",
        "tower_GPP_monthly_conservative_mean_of_daily",
        "FETCH_FILTER_available",
        "FETCH_FILTER_monthly_fraction_accepted_daytime",
        "GPP_monthly_quality_flag"
    ]

    existing_preview_cols = [c for c in preview_cols if c in all_monthly_df.columns]

    print(all_monthly_df[existing_preview_cols].head(50).to_string(index=False))
else:
    print("No monthly GPP output was created.")
