"""Clean 30-minute FLUXNET wind-direction (WD) records (daytime, NIGHT == 0 when available) per site and build monthly WD histograms.

Input: FLUXNET HH files. Output: clean 30-minute WD tables and monthly WD-bin histograms used to construct monthly source areas."""
# Original script filename (sanitized): Wind Direction for Each EC Site.py; public name: 01a_wind_direction_monthly_histograms.py
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
# Code name: FLUXNET 30-Minute WD Cleaner and Monthly Histogram Builder
#
# Purpose:
#   1. Read all FLUXNET HH files for selected EC sites.
#   2. Extract valid 30-minute wind direction records.
#   3. Apply daytime filtering using NIGHT == 0 when available.
#   4. Save clean 30-minute WD records for each site.
#   5. Create monthly WD histograms from the 30-minute records.
#
# Why this code exists:
#   It is noted that long WD averages can be misleading.
#   So this code keeps the actual 30-minute WD distribution.
#
# Main outputs:
#   1. Per-site clean 30-minute WD CSV files
#   2. Combined all-sites clean 30-minute WD CSV
#   3. Per-site monthly WD-bin CSV files
#   4. Combined all-sites monthly WD-bin CSV
#   5. Excel workbook with monthly WD bins and processing status
#
# Important:
#   This code does NOT create footprint polygons.
#   It prepares the clean WD data needed for the next footprint code.
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
import re


# ============================================================
# 1. USER SETTINGS
# ============================================================

folder = Path(
    _os.path.join(PROJECT_ROOT, 'fluxnet', 'Hourly Data (for Wind Direction)')
)

output_folder = folder / "clean_30min_WD_outputs"
output_folder.mkdir(exist_ok=True)

target_sites = [
    "AU-ASM", "AU-Cpr", "AU-DaP", "AU-DaS", "AU-Dry", "AU-Emr", "AU-Gin",
    "AU-Rig", "AU-Stp", "CG-Tch", "CH-Oe1", "CN-Du2", "CZ-BK2", "SD-Dem",
    "SN-Dhr", "US-AR1", "US-AR2", "US-ARb", "US-ARc", "US-SRG", "US-Wkg"
]

missing_value = -9999

# Recommended because SIF is daytime/light-driven.
use_daytime_only = True

# If NIGHT is not available, use SW_IN_F as a backup daytime filter.
sw_in_daytime_threshold = 20

# WD histogram bin size.
# 10 degrees is detailed but not too noisy.
wd_bin_width_deg = 10

# Minimum valid daytime 30-minute WD records per month.
# This is used only as a quality flag.
min_valid_monthly_wd_records = 100

# Excel has a row limit. Keep full clean 30-min data in CSV.
# Excel will store monthly bins and processing status.
write_large_clean_wd_to_excel = False


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def extract_site_id(filename):
    """
    Extract site ID from FLUXNET filename.
    Example:
    FLX_AU-ASM_FLUXNET2015_SUBSET_HH_2010-2014_2-4.csv
    """
    match = re.search(r"FLX_([A-Z]{2}-[A-Za-z0-9]+)_", filename)
    if match:
        return match.group(1)
    return None


def parse_fluxnet_timestamp(series):
    """
    Parse FLUXNET TIMESTAMP_START values.

    Handles:
    201001010000
    201001010030
    2.01001E+11
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
    s = s.str.replace(r"\D", "", regex=True).str.zfill(12)

    return pd.to_datetime(s, format="%Y%m%d%H%M", errors="coerce")


def direction_to_cardinal(deg):
    """
    Convert WD degrees to compass direction.
    """
    if pd.isna(deg):
        return np.nan

    directions = [
        "N", "NNE", "NE", "ENE",
        "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW",
        "W", "WNW", "NW", "NNW"
    ]

    ix = int((deg + 11.25) / 22.5) % 16
    return directions[ix]


def circular_mean_degrees(degrees):
    """
    Diagnostic only.

    This calculates circular mean correctly by converting WD to sine/cosine,
    averaging the components, then converting back to degrees.
    """
    degrees = np.asarray(degrees, dtype=float)
    degrees = degrees[~np.isnan(degrees)]

    if len(degrees) == 0:
        return np.nan

    radians = np.deg2rad(degrees)

    sin_mean = np.nanmean(np.sin(radians))
    cos_mean = np.nanmean(np.cos(radians))

    mean_angle = np.rad2deg(np.arctan2(sin_mean, cos_mean))

    if mean_angle < 0:
        mean_angle += 360

    return mean_angle


def circular_resultant_length(degrees):
    """
    Diagnostic only.

    R close to 1 means WD is consistent.
    R close to 0 means WD is scattered or multi-directional.
    """
    degrees = np.asarray(degrees, dtype=float)
    degrees = degrees[~np.isnan(degrees)]

    if len(degrees) == 0:
        return np.nan

    radians = np.deg2rad(degrees)

    sin_mean = np.nanmean(np.sin(radians))
    cos_mean = np.nanmean(np.cos(radians))

    return np.sqrt(sin_mean**2 + cos_mean**2)


def prepare_clean_30min_wd(df, site_id):
    """
    Clean the 30-minute WD data for one site.
    """
    required_cols = ["TIMESTAMP_START", "WD"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{site_id}: missing required column {col}")

    work = df.copy()
    work.columns = [c.strip() for c in work.columns]

    work["datetime"] = parse_fluxnet_timestamp(work["TIMESTAMP_START"])
    work["WD"] = pd.to_numeric(work["WD"], errors="coerce")

    if "TIMESTAMP_END" in work.columns:
        work["datetime_end"] = parse_fluxnet_timestamp(work["TIMESTAMP_END"])
    else:
        work["datetime_end"] = pd.NaT

    if "NIGHT" in work.columns:
        work["NIGHT"] = pd.to_numeric(work["NIGHT"], errors="coerce")
    else:
        work["NIGHT"] = np.nan

    if "SW_IN_F" in work.columns:
        work["SW_IN_F"] = pd.to_numeric(work["SW_IN_F"], errors="coerce")
    else:
        work["SW_IN_F"] = np.nan

    if "WS_F" in work.columns:
        work["WS_F"] = pd.to_numeric(work["WS_F"], errors="coerce")
    else:
        work["WS_F"] = np.nan

    if "USTAR" in work.columns:
        work["USTAR"] = pd.to_numeric(work["USTAR"], errors="coerce")
    else:
        work["USTAR"] = np.nan

    # Replace FLUXNET missing values
    for col in ["WD", "NIGHT", "SW_IN_F", "WS_F", "USTAR"]:
        work.loc[work[col] == missing_value, col] = np.nan

    # Keep valid timestamp and valid WD
    work = work[
        work["datetime"].notna()
        & work["WD"].notna()
        & (work["WD"] >= 0)
        & (work["WD"] <= 360)
    ].copy()

    # Treat 360 as 0
    work.loc[work["WD"] == 360, "WD"] = 0

    # Daytime filter
    if use_daytime_only:
        if work["NIGHT"].notna().any():
            work["is_daytime"] = work["NIGHT"] == 0
            daytime_filter_used = "NIGHT_equals_0"
        elif work["SW_IN_F"].notna().any():
            work["is_daytime"] = work["SW_IN_F"] > sw_in_daytime_threshold
            daytime_filter_used = f"SW_IN_F_greater_than_{sw_in_daytime_threshold}"
        else:
            work["is_daytime"] = True
            daytime_filter_used = "no_daytime_filter_available_all_records_used"
    else:
        work["is_daytime"] = True
        daytime_filter_used = "all_records_used"

    work = work[work["is_daytime"]].copy()

    work["SITE_ID"] = site_id
    work["date"] = work["datetime"].dt.date.astype(str)
    work["YYYYMMDD"] = work["datetime"].dt.strftime("%Y%m%d").astype(int)
    work["YYYYMM"] = work["datetime"].dt.strftime("%Y%m").astype(int)
    work["year"] = work["datetime"].dt.year
    work["month"] = work["datetime"].dt.month
    work["day"] = work["datetime"].dt.day
    work["hour"] = work["datetime"].dt.hour
    work["minute"] = work["datetime"].dt.minute

    work["WD_cardinal"] = work["WD"].apply(direction_to_cardinal)

    work["daytime_filter_used"] = daytime_filter_used

    keep_cols = [
        "SITE_ID",
        "datetime",
        "datetime_end",
        "date",
        "YYYYMMDD",
        "YYYYMM",
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "WD",
        "WD_cardinal",
        "WS_F",
        "USTAR",
        "SW_IN_F",
        "NIGHT",
        "daytime_filter_used"
    ]

    return work[keep_cols]


def create_monthly_wd_bins(clean_df):
    """
    Convert clean 30-minute WD records into monthly WD-bin fractions.

    Each output row is:
    one site + one month + one WD bin
    """
    if clean_df.empty:
        return pd.DataFrame()

    work = clean_df.copy()

    work["WD_bin_start_deg"] = (
        np.floor(work["WD"] / wd_bin_width_deg) * wd_bin_width_deg
    ).astype(int)

    work["WD_bin_end_deg"] = work["WD_bin_start_deg"] + wd_bin_width_deg

    work["WD_bin_center_deg"] = (
        work["WD_bin_start_deg"] + wd_bin_width_deg / 2
    ) % 360

    rows = []

    for (site_id, yyyymm), group in work.groupby(["SITE_ID", "YYYYMM"]):
        n_total = len(group)

        mean_wd = circular_mean_degrees(group["WD"].values)
        r_value = circular_resultant_length(group["WD"].values)

        if n_total < min_valid_monthly_wd_records:
            quality_flag = "LOW_MONTHLY_DAYTIME_WD_RECORD_COUNT"
        else:
            quality_flag = "OK"

        bin_counts = (
            group
            .groupby(["WD_bin_start_deg", "WD_bin_end_deg", "WD_bin_center_deg"])
            .size()
            .reset_index(name="WD_bin_count")
        )

        bin_counts["WD_bin_fraction"] = bin_counts["WD_bin_count"] / n_total

        for _, bin_row in bin_counts.iterrows():
            rows.append({
                "SITE_ID": site_id,
                "YYYYMM": int(yyyymm),
                "year": int(group["year"].iloc[0]),
                "month": int(group["month"].iloc[0]),

                "WD_bin_width_deg": wd_bin_width_deg,
                "WD_bin_start_deg": int(bin_row["WD_bin_start_deg"]),
                "WD_bin_end_deg": int(bin_row["WD_bin_end_deg"]),
                "WD_bin_center_deg": float(bin_row["WD_bin_center_deg"]),
                "WD_bin_center_cardinal": direction_to_cardinal(float(bin_row["WD_bin_center_deg"])),

                "WD_bin_count": int(bin_row["WD_bin_count"]),
                "WD_bin_fraction": round(float(bin_row["WD_bin_fraction"]), 8),

                "n_valid_monthly_30min_WD_records": int(n_total),

                # Diagnostics only, not used as the main footprint direction
                "monthly_circular_mean_WD_deg_diagnostic": round(mean_wd, 2) if not pd.isna(mean_wd) else np.nan,
                "monthly_circular_mean_WD_cardinal_diagnostic": direction_to_cardinal(mean_wd),
                "monthly_WD_consistency_R_diagnostic": round(r_value, 3) if not pd.isna(r_value) else np.nan,

                "quality_flag": quality_flag,
                "footprint_method_next_step": "weighted_monthly_probability_from_30min_WD_bins"
            })

    return pd.DataFrame(rows)


# ============================================================
# 3. MAIN PROCESSING
# ============================================================

if not folder.exists():
    raise FileNotFoundError(f"Folder not found:\n{folder}")

files = sorted(folder.glob("*_SUBSET_HH_*.csv"))

if not files:
    raise FileNotFoundError(f"No HH CSV files found in:\n{folder}")

all_clean_wd = []
all_monthly_bins = []
status_rows = []

for file in files:
    site_id = extract_site_id(file.name)

    if site_id not in target_sites:
        continue

    print(f"Processing {site_id}: {file.name}")

    try:
        df = pd.read_csv(file, low_memory=False)
        df.columns = [c.strip() for c in df.columns]

        if "WD" not in df.columns:
            print(f"  WARNING: {site_id} has no WD column.")

            status_rows.append({
                "SITE_ID": site_id,
                "file": file.name,
                "status": "NO_WD_COLUMN",
                "n_input_rows": len(df),
                "n_clean_30min_WD_rows": np.nan,
                "n_monthly_bin_rows": np.nan,
                "n_months": np.nan
            })

            continue

        clean_site_df = prepare_clean_30min_wd(df, site_id)
        monthly_bin_site_df = create_monthly_wd_bins(clean_site_df)

        site_clean_path = output_folder / f"{site_id}_clean_30min_WD.csv"
        site_bins_path = output_folder / f"{site_id}_monthly_30min_WD_bins.csv"

        clean_site_df.to_csv(site_clean_path, index=False)
        monthly_bin_site_df.to_csv(site_bins_path, index=False)

        all_clean_wd.append(clean_site_df)
        all_monthly_bins.append(monthly_bin_site_df)

        status_rows.append({
            "SITE_ID": site_id,
            "file": file.name,
            "status": "OK",
            "n_input_rows": len(df),
            "n_clean_30min_WD_rows": len(clean_site_df),
            "n_monthly_bin_rows": len(monthly_bin_site_df),
            "n_months": monthly_bin_site_df["YYYYMM"].nunique() if not monthly_bin_site_df.empty else 0,
            "clean_30min_WD_file": str(site_clean_path),
            "monthly_bins_file": str(site_bins_path)
        })

        print(f"  Clean 30-min WD rows: {len(clean_site_df)}")
        print(f"  Monthly WD-bin rows: {len(monthly_bin_site_df)}")
        print(f"  Saved clean WD: {site_clean_path}")
        print(f"  Saved monthly bins: {site_bins_path}")

    except Exception as e:
        print(f"  ERROR with {site_id}: {e}")

        status_rows.append({
            "SITE_ID": site_id,
            "file": file.name,
            "status": f"ERROR: {e}",
            "n_input_rows": np.nan,
            "n_clean_30min_WD_rows": np.nan,
            "n_monthly_bin_rows": np.nan,
            "n_months": np.nan,
            "clean_30min_WD_file": "",
            "monthly_bins_file": ""
        })


# ============================================================
# 4. SAVE COMBINED OUTPUTS
# ============================================================

status_df = pd.DataFrame(status_rows)

if all_clean_wd:
    combined_clean_df = pd.concat(all_clean_wd, ignore_index=True)
else:
    combined_clean_df = pd.DataFrame()

if all_monthly_bins:
    combined_bins_df = pd.concat(all_monthly_bins, ignore_index=True)
else:
    combined_bins_df = pd.DataFrame()

combined_clean_path = output_folder / "ALL_SITES_clean_30min_WD.csv"
combined_bins_path = output_folder / "ALL_SITES_monthly_30min_WD_bins.csv"
status_path = output_folder / "processing_status_clean_30min_WD.csv"
excel_path = output_folder / "ALL_SITES_monthly_30min_WD_bins_and_status.xlsx"

combined_clean_df.to_csv(combined_clean_path, index=False)
combined_bins_df.to_csv(combined_bins_path, index=False)
status_df.to_csv(status_path, index=False)

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    combined_bins_df.to_excel(writer, sheet_name="Monthly_WD_Bins", index=False)
    status_df.to_excel(writer, sheet_name="Processing_Status", index=False)

    if write_large_clean_wd_to_excel:
        if len(combined_clean_df) <= 1_000_000:
            combined_clean_df.to_excel(writer, sheet_name="Clean_30min_WD", index=False)
        else:
            print(
                "Combined clean 30-min WD has more than 1,000,000 rows, "
                "so it was not written to Excel. Use the CSV instead."
            )


# ============================================================
# 5. TERMINAL OUTPUT
# ============================================================

print("\nDONE.")
print(f"Output folder:\n{output_folder}")

print(f"\nSaved combined clean 30-min WD CSV:\n{combined_clean_path}")

print(f"\nSaved combined monthly WD-bin CSV:\n{combined_bins_path}")

print(f"\nSaved processing status CSV:\n{status_path}")

print(f"\nSaved Excel workbook:\n{excel_path}")

print("\nProcessing status:")
if not status_df.empty:
    print(status_df.to_string(index=False))
else:
    print("No files were processed.")

print("\nMonthly WD-bin preview:")
if not combined_bins_df.empty:
    preview_cols = [
        "SITE_ID",
        "YYYYMM",
        "WD_bin_center_deg",
        "WD_bin_count",
        "WD_bin_fraction",
        "n_valid_monthly_30min_WD_records",
        "monthly_WD_consistency_R_diagnostic",
        "quality_flag"
    ]
    print(combined_bins_df[preview_cols].head(60).to_string(index=False))
else:
    print("No monthly WD-bin output was created.")