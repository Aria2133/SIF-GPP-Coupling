"""Extract daily and monthly tower GPP from FLUXNET HH files with a conservative filter (NEE_VUT_REF_QC == 0) to limit dependence on gap-filled periods.

Input: FLUXNET HH files. Output: daily and monthly conservative GPP tables."""
# Original script filename (sanitized): Extract_FLUXNET_HH_GPP_Daily_and_Monthly_for_GEE.py; public name: 02a_extract_gpp_daily_monthly_conservative.py
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
import re

# ============================================================
# CODE NAME:
# Conservative FLUXNET GPP Extractor for GEE
#
# PURPOSE:
# Extract daily and monthly tower GPP values from FLUXNET HH files.
# Uses a conservative filter based on NEE_VUT_REF_QC == 0 to reduce
# dependence on gap-filled/model-filled periods.
#
# MAIN OUTPUT FOR GEE:
# ALL_SITES_monthly_GPP_conservative_for_GEE.csv
#
# RECOMMENDED GEE USE:
# Join this monthly GPP table with your monthly footprint-probability table
# using SITE_ID and YYYYMM.
# ============================================================


# ============================================================
# USER SETTINGS
# ============================================================

folder = Path(
    _os.path.join(PROJECT_ROOT, 'fluxnet', 'Hourly Data (for Wind Direction)')
)

output_folder = folder / "GPP_outputs_for_GEE_conservative"
output_folder.mkdir(exist_ok=True)

target_sites = [
    "AU-ASM", "AU-Cpr", "AU-DaP", "AU-DaS", "AU-Dry", "AU-Emr", "AU-Gin",
    "AU-Rig", "AU-Stp", "CG-Tch", "CH-Oe1", "CN-Du2", "CZ-BK2", "SD-Dem",
    "SN-Dhr", "US-AR1", "US-AR2", "US-ARb", "US-ARc", "US-SRG", "US-Wkg"
]

site_info = {
    "AU-ASM": ["Alice Springs", -22.283, 133.249, "SAV", "https://fluxnet.org/sites/siteinfo/AU-ASM"],
    "AU-Cpr": ["Calperum", -34.0021, 140.5891, "SAV", "https://fluxnet.org/sites/siteinfo/AU-Cpr"],
    "AU-DaP": ["Daly River Savanna", -14.0633, 131.3181, "GRA", "https://fluxnet.org/sites/siteinfo/AU-DaP"],
    "AU-DaS": ["Daly River Cleared", -14.1593, 131.3881, "SAV", "https://fluxnet.org/sites/siteinfo/AU-DaS"],
    "AU-Dry": ["Dry River", -15.2588, 132.3706, "SAV", "https://fluxnet.org/sites/siteinfo/AU-Dry"],
    "AU-Emr": ["Emerald", -23.8587, 148.4746, "GRA", "https://fluxnet.org/sites/siteinfo/AU-Emr"],
    "AU-Gin": ["Gingin", -31.3764, 115.7138, "WSA", "https://fluxnet.org/sites/siteinfo/AU-Gin"],
    "AU-Rig": ["Riggs Creek", -36.6499, 145.5759, "GRA", "https://fluxnet.org/sites/siteinfo/AU-Rig"],
    "AU-Stp": ["Sturt Plains", -17.1507, 133.3502, "GRA", "https://fluxnet.org/sites/siteinfo/AU-Stp"],
    "CG-Tch": ["Tchizalamou", -4.2892, 11.6564, "SAV", "https://fluxnet.org/sites/siteinfo/CG-Tch"],
    "CH-Oe1": ["Oensingen grassland", 47.2858, 7.7319, "GRA", "https://fluxnet.org/sites/siteinfo/CH-Oe1"],
    "CN-Du2": ["Duolun grassland", 42.0467, 116.2836, "GRA", "https://fluxnet.org/sites/siteinfo/CN-Du2"],
    "CZ-BK2": ["Bily Kriz grassland", 49.4944, 18.5429, "GRA", "https://fluxnet.org/sites/siteinfo/CZ-BK2"],
    "SD-Dem": ["Demokeya", 13.2829, 30.4783, "SAV", "https://fluxnet.org/sites/siteinfo/SD-Dem"],
    "SN-Dhr": ["Dahra", 15.4028, -15.4322, "SAV", "https://fluxnet.org/sites/siteinfo/SN-Dhr"],
    "US-AR1": ["Woodward Switchgrass 1", 36.4267, -99.42, "GRA", "https://fluxnet.org/sites/siteinfo/US-AR1"],
    "US-AR2": ["Woodward Switchgrass 2", 36.6358, -99.5975, "GRA", "https://fluxnet.org/sites/siteinfo/US-AR2"],
    "US-ARb": ["Southern Great Plains burn site", 35.5497, -98.0402, "GRA", "https://fluxnet.org/sites/siteinfo/US-ARb"],
    "US-ARc": ["Southern Great Plains control site", 35.5465, -98.04, "GRA", "https://fluxnet.org/sites/siteinfo/US-ARc"],
    "US-SRG": ["Santa Rita Grassland", 31.7894, -110.8277, "GRA", "https://fluxnet.org/sites/siteinfo/US-SRG"],
    "US-Wkg": ["Walnut Gulch Kendall Grasslands", 31.7365, -109.9419, "GRA", "https://fluxnet.org/sites/siteinfo/US-Wkg"],
}

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
    match = re.search(r"FLX_([A-Z]{2}-[A-Za-z0-9]+)_", filename)
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

    return pd.to_datetime(s, format="%Y%m%d%H%M", errors="coerce")


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
    df = pd.read_csv(file, low_memory=False)
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
            group["is_daytime"] &
            group["is_conservative_flux_record"]
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

files = sorted(folder.glob("*_SUBSET_HH_*.csv"))

if not files:
    raise FileNotFoundError(f"No HH CSV files found in: {folder}")

all_daily = []
all_monthly = []
status_rows = []

for file in files:
    site_id = extract_site_id(file.name)

    if site_id not in target_sites:
        continue

    print(f"Processing {site_id}: {file.name}")

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
            "n_monthly_rows": len(monthly_df)
        })

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
            "n_monthly_rows": np.nan
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
    all_daily_df.to_excel(writer, sheet_name="Daily_GPP_Conservative", index=False)
    all_monthly_df.to_excel(writer, sheet_name="Monthly_GPP_Conservative", index=False)
    status_df.to_excel(writer, sheet_name="Processing_Status", index=False)


# ============================================================
# TERMINAL OUTPUT
# ============================================================

print("\nDONE.")
print(f"Saved daily all-sites conservative GPP CSV:\n{all_daily_csv}")
print(f"Saved monthly all-sites conservative GPP CSV:\n{all_monthly_csv}")
print(f"Saved processing status CSV:\n{status_csv}")
print(f"Saved Excel workbook:\n{excel_path}")

print("\nProcessing status:")
print(status_df.to_string(index=False))

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