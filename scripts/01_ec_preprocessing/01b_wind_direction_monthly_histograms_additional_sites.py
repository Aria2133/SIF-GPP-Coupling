"""Same processing as 01a for additional sites delivered as FLUXMET HH/HR files (only the file-search pattern and site-ID extraction differ).

Input: FLUXNET FLUXMET HH/HR files. Output: clean WD tables and monthly WD histograms."""
# Original script filename (sanitized): Wind Direction for Each Site (new added sites).py; public name: 01b_wind_direction_monthly_histograms_additional_sites.py
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
# Code name: FLUXNET FLUXMET HH/HR WD Cleaner and Monthly Histogram Builder
#
# Purpose:
#   1. Read all FLUXNET FLUXMET HH/HR files for selected EC sites.
#   2. Extract valid wind direction records.
#   3. Apply daytime filtering using NIGHT == 0 when available.
#   4. Save clean WD records for each site.
#   5. Create monthly WD histograms from the high-frequency records.
#
# Important:
#   This code keeps the same processing logic as the previous code.
#   Only the file-search pattern and site-ID extraction were updated
#   so the code can read new files such as:
#       AMF_US-Ne1_FLUXNET_FLUXMET_HR_2001-2024_v1.3_r1.csv
#       ICOS_FR-Gri_FLUXNET_FLUXMET_HH_2004-2024_v1.3_r1.csv
#       TERN_AU-How_FLUXNET_FLUXMET_HH_2002-2025_v1.3_r1.csv
#
# Main outputs:
#   1. Per-site clean WD CSV files
#   2. Combined all-sites clean WD CSV
#   3. Per-site monthly WD-bin CSV files
#   4. Combined all-sites monthly WD-bin CSV
#   5. Excel workbook with monthly WD bins and processing status
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
import re


# ============================================================
# 1. USER SETTINGS
# ============================================================

folder = Path(
    _os.path.join(PROJECT_ROOT, 'Remaining Fluxnet Sites', 'HH files')
)

output_folder = folder / "clean_30min_WD_outputs"
output_folder.mkdir(exist_ok=True)

target_sites = [
    "BR-Ma2",
    "CA-Gro",
    "US-Akn",
    "US-Bar",
    "US-Ha1",
    "US-Ho2",
    "US-ICh",
    "US-ICt",
    "US-Jo1",
    "US-MMS",
    "US-Ne1",
    "US-Ne2",
    "US-Ne3",
    "US-NR1",
    "US-Rls",
    "US-Rms",
    "US-Ses",
    "US-SRM",
    "US-TKs",
    "US-Ton",
    "US-UiA",
    "US-UiB",
    "US-UiC",
    "US-WCr",
    "US-Whs",
    "US-Wjs",

    "CN-Ash",
    "CN-BeO",
    "CN-Din",
    "CN-Dmn",
    "CN-Dzo",
    "CN-Lch",
    "CN-Sdq",

    "AT-Zoe",
    "CH-Lae",
    "CH-Oe2",
    "ES-Agu",
    "ES-LJu",
    "ES-LM1",
    "ES-LM2",
    "IL-Yat",

    "BE-Bra",
    "BE-Maa",
    "BE-Vie",
    "CH-Dav",
    "DE-Hai",
    "DE-RuS",
    "DE-Tha",
    "DK-Sor",
    "FI-Hyy",
    "FR-Aur",
    "FR-Gri",
    "FR-Hes",
    "FR-Pue",
    "GF-Guy",
    "IT-BCi",
    "IT-Cp2",
    "IT-Ren",
    "NL-Loo",

    "JP-Tak",

    "AU-How",
    "AU-TTE",
    "AU-Tum",
    "AU-Whr",
    "AU-Wom",
]

missing_value = -9999

# Recommended because SIF is daytime/light-driven.
use_daytime_only = True

# If NIGHT is not available, use SW_IN_F as a backup daytime filter.
sw_in_daytime_threshold = 20

# WD histogram bin size.
# 10 degrees is detailed but not too noisy.
wd_bin_width_deg = 10

# Minimum valid daytime WD records per month.
# This is used only as a quality flag.
min_valid_monthly_wd_records = 100

# Excel has a row limit. Keep full clean WD data in CSV.
# Excel will store monthly bins and processing status.
write_large_clean_wd_to_excel = False


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def extract_site_id(filename):
    """
    Extract site ID from several FLUXNET filename styles.

    Old FLUXNET2015 example:
    FLX_AU-ASM_FLUXNET2015_SUBSET_HH_2010-2014_2-4.csv

    New FLUXMET examples:
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
    Clean the high-frequency WD data for one site.

    Output column names are kept the same as the previous code
    for compatibility with the downstream GEE merge workflow.
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
    Convert clean WD records into monthly WD-bin fractions.

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

file_patterns = [
    "*_SUBSET_HH_*.csv",
    "*_SUBSET_HR_*.csv",
    "*_FLUXMET_HH_*.csv",
    "*_FLUXMET_HR_*.csv",
]

files = []
for pattern in file_patterns:
    files.extend(sorted(folder.glob(pattern)))

# Remove duplicates while preserving sorted order
files = sorted(set(files), key=lambda x: x.name)

if not files:
    raise FileNotFoundError(
        "No HH/HR CSV files found in:\n"
        f"{folder}\n\n"
        "Expected filename patterns include:\n"
        "*_FLUXMET_HH_*.csv\n"
        "*_FLUXMET_HR_*.csv\n"
        "*_SUBSET_HH_*.csv\n"
        "*_SUBSET_HR_*.csv"
    )

print(f"Total candidate HH/HR files found: {len(files)}")

all_clean_wd = []
all_monthly_bins = []
status_rows = []
processed_sites = []

for file in files:
    site_id = extract_site_id(file.name)

    if site_id is None:
        print(f"Skipping file because site ID could not be extracted: {file.name}")
        continue

    if site_id not in target_sites:
        continue

    print(f"\nProcessing {site_id}: {file.name}")

    try:
        wd_input_cols = {
            "TIMESTAMP_START",
            "TIMESTAMP_END",
            "WD",
            "NIGHT",
            "SW_IN_F",
            "WS_F",
            "USTAR",
        }

        df = pd.read_csv(
            file,
            usecols=lambda c: str(c).strip() in wd_input_cols,
            low_memory=False
        )
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
        processed_sites.append(site_id)

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

        print(f"  Clean WD rows: {len(clean_site_df)}")
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
                "Combined clean WD has more than 1,000,000 rows, "
                "so it was not written to Excel. Use the CSV instead."
            )


# ============================================================
# 5. TERMINAL OUTPUT
# ============================================================

print("\nDONE.")
print(f"Output folder:\n{output_folder}")

print(f"\nSaved combined clean WD CSV:\n{combined_clean_path}")

print(f"\nSaved combined monthly WD-bin CSV:\n{combined_bins_path}")

print(f"\nSaved processing status CSV:\n{status_path}")

print(f"\nSaved Excel workbook:\n{excel_path}")

print("\nProcessing status:")
if not status_df.empty:
    print(status_df.to_string(index=False))
else:
    print("No files were processed.")

print("\nProcessed site count:")
print(len(set(processed_sites)))

print("\nProcessed sites:")
print(sorted(set(processed_sites)))

missing_sites = sorted(set(target_sites) - set(processed_sites))

print("\nTarget sites not processed:")
if missing_sites:
    print(missing_sites)
else:
    print("None. All target sites were processed.")

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
