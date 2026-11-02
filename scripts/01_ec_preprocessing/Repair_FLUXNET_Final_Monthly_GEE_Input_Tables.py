# Install required packages, if needed:
# pip install pandas numpy openpyxl xlrd

"""
Repair scientifically defensible gaps in final monthly FLUXNET footprint inputs.

Safety model
------------
* DRY_RUN=True by default.
* The two original final CSV files are never overwritten.
* No TIFF is created or modified.
* No default radius or fabricated GPP/coordinates are allowed.
* Dry-run performs the complete read-only candidate analysis, applies trusted
  candidates to an in-memory copy, and writes candidate/audit reports only.
* DRY_RUN=False additionally writes repaired copies to REPAIRED_OUTPUT_ROOT and
  may safely extract a valid BE-Lon HH/HR member. A corrupt ZIP is never used.

Scientific methods reproduced from the project's existing upstream scripts
----------------------------------------------------------------------------
* Daytime: NIGHT == 0; otherwise SW_IN_F > 20; otherwise all records.
* Conservative GPP: matching NEE partition QC == 0 when available.
* Daily GPP: mean of valid conservative daytime records.
* Monthly GPP: mean of daily means.
* GPP thresholds: >=5 records/day diagnostic, >=10 valid days/month and
  >=100 conservative daytime records/month for a repair-eligible monthly GPP.
* WD: daytime 30-minute/hourly WD histogram, 10-degree bins.
* Radius: trusted explicit radius first; otherwise
  height_above_canopy_m * 100, where missing height may be derived as
  EC_measurement_height_m - canopy_height_m when positive.
"""

from __future__ import annotations
# Original script filename (sanitized): Repair_FLUXNET_Final_Monthly_GEE_Input_Tables.py; public name: Repair_FLUXNET_Final_Monthly_GEE_Input_Tables.py
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

import math
import os
import re
import shutil
import struct
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable

import numpy as np
import pandas as pd

import Audit_FLUXNET_Footprint_Input_Completeness as completeness_audit


# =============================================================================
# EDITABLE SETTINGS AND PATHS
# =============================================================================

DRY_RUN = True

PROJECT_ROOT = Path(
    PROJECT_ROOT
)

ORIGINAL_HH_FOLDER = PROJECT_ROOT / "fluxnet" / "Hourly Data (for Wind Direction)"
REMAINING_HH_FOLDER = PROJECT_ROOT / "Remaining Fluxnet Sites" / "HH files"
REMAINING_ZIP_FOLDER = PROJECT_ROOT / "Remaining Fluxnet Sites"

INPUT_TABLE_PATHS = [
    ORIGINAL_HH_FOLDER
    / "final_monthly_GEE_inputs"
    / "ALL_SITES_monthly_WD_histogram_GPP_footprint_input_for_GEE.csv",
    REMAINING_HH_FOLDER
    / "final_monthly_GEE_inputs"
    / "ALL_SITES_monthly_WD_histogram_GPP_footprint_input_for_GEE.csv",
]

EXISTING_TIFF_OUTPUT_ROOT = PROJECT_ROOT / "Monthly_EC_Footprint_TIFFs_Python"
EXISTING_AUDIT_QC = EXISTING_TIFF_OUTPUT_ROOT / "_QC"

REPAIRED_OUTPUT_ROOT = PROJECT_ROOT / "repaired_final_monthly_GEE_inputs"
REPAIR_QC_OUTPUT = REPAIRED_OUTPUT_ROOT / "_QC"

BE_LON_ZIP = (
    REMAINING_ZIP_FOLDER
    / "ICOS_BE-Lon_FLUXNET_2004-2024_v1.3_r1.zip"
)
BE_LON_EXTRACT_FOLDER = REMAINING_HH_FOLDER / "BE-Lon_extracted"

# Nonstandard GPP-like fields are reported but are not automatically applied
# unless explicitly enabled. The requested standard priority fields are always
# eligible when they contain valid observations.
ALLOW_NONSTANDARD_GPP_FALLBACK = False

# Metadata-like tables larger than this remain discoverable but are not loaded.
MAX_METADATA_READ_SIZE_MB = 100.0


# =============================================================================
# SCIENTIFIC SETTINGS REPRODUCED FROM THE UPSTREAM PIPELINE
# =============================================================================

MISSING_VALUE = -9999.0
SW_IN_DAYTIME_THRESHOLD = 20.0
WD_BIN_WIDTH_DEG = 10.0
MIN_VALID_MONTHLY_WD_RECORDS = 1
MIN_CONSERVATIVE_DAYTIME_RECORDS_PER_DAY = 5
MIN_CONSERVATIVE_DAYTIME_RECORDS_PER_MONTH = 100
MIN_VALID_DAYS_PER_MONTH = 10

GPP_COLUMN_PRIORITY = [
    "GPP_DT_VUT_REF",
    "GPP_NT_VUT_REF",
    "GPP_DT_CUT_REF",
    "GPP_NT_CUT_REF",
    "GPP_DT",
    "GPP_NT",
    "GPP",
    "GPP_F",
    "GPP_PI_F",
]


# =============================================================================
# REPAIR SITE GROUPS
# =============================================================================

GPP_ONLY_SITES = [
    "AU-TTE",
    "AU-Tum",
    "CN-Din",
    "FR-Hes",
    "US-NR1",
    "US-WCr",
    "US-Whs",
]

RADIUS_ONLY_SITES = [
    "BE-Maa",
    "BR-Ma2",
    "CA-Gro",
    "ES-LM1",
    "ES-LM2",
    "FR-Aur",
    "IT-Cp2",
    "JP-Tak",
    "US-ICh",
    "US-ICt",
    "US-Jo1",
    "US-Rms",
    "US-Ses",
    "US-Wjs",
]

LATLON_RADIUS_SITES = [
    "AT-Zoe",
    "CN-Ash",
    "CN-BeO",
    "CN-Dmn",
    "CN-Dzo",
    "CN-Lch",
    "CN-Sdq",
    "US-Rls",
    "US-TKs",
]

RADIUS_GPP_SITES = [
    "ES-Agu",
    "US-Akn",
    "US-Ho2",
    "US-UiB",
    "US-UiC",
]

FULL_MISSING_SITES = ["US-UiA"]

BE_LON_SITE = "BE-Lon"

ALL_GPP_REPAIR_SITES = sorted(
    set(GPP_ONLY_SITES) | set(RADIUS_GPP_SITES) | set(FULL_MISSING_SITES)
)
ALL_METADATA_REPAIR_SITES = sorted(
    set(RADIUS_ONLY_SITES)
    | set(LATLON_RADIUS_SITES)
    | set(RADIUS_GPP_SITES)
    | set(FULL_MISSING_SITES)
    | {BE_LON_SITE}
)
ALL_REPAIR_SITES = sorted(
    set(ALL_GPP_REPAIR_SITES)
    | set(ALL_METADATA_REPAIR_SITES)
    | {BE_LON_SITE}
)


# =============================================================================
# REPORT SCHEMAS
# =============================================================================

GPP_REPORT_COLUMNS = [
    "SITE_ID",
    "raw_file_path",
    "candidate_GPP_columns",
    "selected_GPP_column",
    "n_months_total",
    "n_months_with_valid_GPP",
    "first_valid_month",
    "last_valid_month",
    "repair_status",
    "notes",
]

RADIUS_REPORT_COLUMNS = [
    "SITE_ID",
    "metadata_file_used",
    "lat_found",
    "lon_found",
    "canopy_height_m",
    "EC_measurement_height_m",
    "height_above_canopy_m",
    "radius_m",
    "radius_m_min",
    "radius_m_max",
    "repair_status",
    "notes",
]

LATLON_RADIUS_REPORT_COLUMNS = [
    "SITE_ID",
    "metadata_file_used",
    "lat",
    "lon",
    "IGBP",
    "canopy_height_m",
    "EC_measurement_height_m",
    "height_above_canopy_m",
    "radius_m",
    "radius_m_min",
    "radius_m_max",
    "repair_status",
    "notes",
]

BE_LON_REPORT_COLUMNS = [
    "SITE_ID",
    "zip_file_path",
    "zip_is_structurally_valid",
    "zip_member",
    "compressed_size_expected",
    "uncompressed_size_expected",
    "truncated_bytes_detected",
    "detected_columns",
    "timestamp_column",
    "WD_column",
    "GPP_candidate_columns",
    "selected_GPP_column",
    "extraction_performed",
    "extracted_file_path",
    "repair_status",
    "notes",
]


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class GPPRepairResult:
    """Monthly GPP candidate data and one site-level report record."""

    SITE_ID: str
    monthly: pd.DataFrame
    report: dict[str, Any]


@dataclass
class MetadataRepairResult:
    """Trusted metadata/radius fields and their provenance for one site."""

    SITE_ID: str
    metadata_file_used: str
    site_name: Any = np.nan
    lat: float | None = None
    lon: float | None = None
    IGBP: Any = np.nan
    canopy_height_m: float | None = None
    EC_measurement_height_m: float | None = None
    height_above_canopy_m: float | None = None
    radius_m: float | None = None
    radius_m_min: float | None = None
    radius_m_max: float | None = None
    repair_status: str = "NO_TRUSTED_REPAIR"
    notes: str = ""


@dataclass
class BELonZipInspection:
    """Structural and header information for the BE-Lon archive."""

    zip_is_structurally_valid: bool
    member: str
    compressed_size_expected: int | None
    uncompressed_size_expected: int | None
    truncated_bytes_detected: int | None
    columns: list[str]
    repair_status: str
    notes: str


# =============================================================================
# GENERAL HELPERS
# =============================================================================


def warn(message: str) -> None:
    """Print a visible repair warning."""
    print(f"WARNING: {message}", flush=True)


def native_path(path: Path) -> str:
    """Use the audit module's Windows extended-length path handling."""
    return completeness_audit.native_filesystem_path(path)


def path_exists(path: Path) -> bool:
    """Check existence with Windows long-path support."""
    return completeness_audit.path_exists(path)


def is_missing(value: Any) -> bool:
    """Delegate robust scalar missing-value detection."""
    return completeness_audit.is_missing(value)


def normalize_yyyymm(value: Any) -> str | None:
    """Normalize a site-month key."""
    return completeness_audit.normalize_yyyymm(value)


def clean_site_id(value: Any) -> str | None:
    """Normalize an exact FLUXNET SITE_ID."""
    return completeness_audit.clean_site_id(value)


def parse_fluxnet_timestamp(series: pd.Series) -> pd.Series:
    """Reproduce the upstream FLUXNET TIMESTAMP_START parser."""
    text = series.astype(str).str.strip()

    def clean_one(value: str) -> str | float:
        try:
            if "E" in value.upper():
                return str(int(float(value)))
            if value.endswith(".0"):
                return value[:-2]
            return value
        except Exception:
            return np.nan

    text = text.apply(clean_one)
    text = text.str.replace(r"\D", "", regex=True).str.zfill(12).str[:12]
    return pd.to_datetime(text, format="%Y%m%d%H%M", errors="coerce")


def numeric_series(series: pd.Series) -> pd.Series:
    """Convert numeric input and replace FLUXNET missing-value sentinels."""
    values = pd.to_numeric(series, errors="coerce")
    return values.mask(values == MISSING_VALUE)


def first_non_missing(series: pd.Series, default: Any = np.nan) -> Any:
    """Return the first meaningful scalar in a Series."""
    for value in series:
        if not is_missing(value):
            return value
    return default


def unique_join(values: Iterable[Any]) -> str:
    """Join unique, non-missing values deterministically."""
    return " | ".join(
        sorted({str(value).strip() for value in values if not is_missing(value)})
    )


def valid_lat_lon(lat: Any, lon: Any) -> bool:
    """Validate one latitude/longitude pair."""
    try:
        latitude = float(lat)
        longitude = float(lon)
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(
        math.isfinite(latitude)
        and math.isfinite(longitude)
        and -90.0 <= latitude <= 90.0
        and -180.0 <= longitude <= 180.0
    )


def positive_number(value: Any) -> float | None:
    """Return a positive finite number or None."""
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if math.isfinite(numeric) and numeric > 0.0 else None


def parse_positive_number_or_range(value: Any) -> tuple[float | None, float | None]:
    """Parse positive numbers/ranges without interpreting separators as signs."""
    if is_missing(value):
        return None, None
    text = (
        str(value)
        .strip()
        .replace(",", ".")
        .replace("–", "-")
        .replace("—", "-")
        .replace("â€“", "-")
        .replace("â€”", "-")
    )
    numbers = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", text)]
    numbers = [item for item in numbers if math.isfinite(item) and item > 0.0]
    if not numbers:
        return None, None
    return min(numbers), max(numbers)


def append_provenance(
    table: pd.DataFrame,
    mask: pd.Series,
    source: str,
    note: str,
) -> None:
    """Append repair source/note text to selected rows in-place."""
    if not bool(mask.any()):
        return
    for column, text in (("repair_source", source), ("repair_notes", note)):
        existing = table.loc[mask, column].fillna("").astype(str).str.strip()
        table.loc[mask, column] = np.where(
            existing.eq(""),
            text,
            existing + " | " + text,
        )


# =============================================================================
# FINAL TABLE LOADING
# =============================================================================


def load_final_input_tables() -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load both final CSVs, preserve source grouping, and align all columns."""
    tables: list[pd.DataFrame] = []
    original_columns: list[str] = []
    remaining_columns: list[str] = []

    for index, path in enumerate(INPUT_TABLE_PATHS):
        if not path_exists(path):
            raise FileNotFoundError(f"Final input CSV not found: {path}")
        table = completeness_audit.read_csv_robust(path)
        table.columns = [str(column).strip().lstrip("\ufeff") for column in table]
        source_group = "original" if index == 0 else "remaining"
        if source_group == "original":
            original_columns = list(table.columns)
        else:
            remaining_columns = list(table.columns)
        table["source_input_table"] = str(path)
        table["_source_group"] = source_group
        tables.append(table)

    merged = pd.concat(tables, ignore_index=True, sort=False)
    merged["SITE_ID"] = merged["SITE_ID"].astype(str).str.strip()
    merged["YYYYMM"] = merged["YYYYMM"].map(normalize_yyyymm)
    merged = merged.loc[merged["YYYYMM"].notna()].copy()
    merged["YYYYMM"] = merged["YYYYMM"].astype(str)

    duplicate_key = [
        "SITE_ID",
        "YYYYMM",
        "WD_bin_center_deg",
        "source_input_table",
    ]
    duplicates = merged.duplicated(duplicate_key, keep="first")
    if duplicates.any():
        warn(
            f"Removed {int(duplicates.sum()):,} duplicate rows using "
            f"{duplicate_key}."
        )
        merged = merged.loc[~duplicates].copy()

    for column, default in (
        ("repaired_lat_lon", False),
        ("repaired_radius", False),
        ("repaired_GPP", False),
        ("repair_source", ""),
        ("repair_notes", ""),
    ):
        if column not in merged:
            merged[column] = default
    return merged.reset_index(drop=True), original_columns, remaining_columns


# =============================================================================
# RAW HH/HR DISCOVERY AND GPP REPAIR
# =============================================================================


def load_raw_file_inventory() -> dict[str, Path]:
    """Map SITE_ID to detected raw HH/HR CSV, preferring remaining-site files."""
    report_path = EXISTING_AUDIT_QC / "all_detected_fluxnet_hh_files.csv"
    if report_path.exists():
        report = pd.read_csv(report_path, encoding="utf-8-sig")
    else:
        report = completeness_audit.detect_extracted_hh_files(
            [ORIGINAL_HH_FOLDER, REMAINING_HH_FOLDER]
        )
    inventory: dict[str, Path] = {}
    for row in report.to_dict("records"):
        site_id = clean_site_id(row.get("SITE_ID"))
        raw_path = Path(str(row.get("HH_file_path", "")))
        if site_id and path_exists(raw_path):
            inventory[site_id] = raw_path
    return inventory


def read_raw_header(path: Path) -> list[str]:
    """Read a raw CSV header without loading observations."""
    return list(pd.read_csv(native_path(path), nrows=0).columns)


def ordered_gpp_candidates(columns: Iterable[str]) -> tuple[list[str], list[str]]:
    """Return present priority columns followed by reported nonstandard fields."""
    columns = [str(column).strip() for column in columns]
    lookup = {column.upper(): column for column in columns}
    priority_present = [
        lookup[preferred.upper()]
        for preferred in GPP_COLUMN_PRIORITY
        if preferred.upper() in lookup
    ]
    all_gpp = [column for column in columns if "GPP" in column.upper()]
    other = [column for column in all_gpp if column not in priority_present]
    return priority_present, other


def matching_nee_qc_column(selected_gpp: str, columns: Iterable[str]) -> str | None:
    """Use the NEE QC family matching the selected VUT/CUT GPP partition."""
    lookup = {str(column).upper(): str(column) for column in columns}
    upper = selected_gpp.upper()
    if "_CUT_" in upper:
        return lookup.get("NEE_CUT_REF_QC")
    if "_VUT_" in upper:
        return lookup.get("NEE_VUT_REF_QC")
    for candidate in ("NEE_VUT_REF_QC", "NEE_CUT_REF_QC"):
        if candidate in lookup:
            return lookup[candidate]
    return None


def aggregate_monthly_gpp_from_raw(
    site_id: str,
    raw_path: Path,
) -> GPPRepairResult:
    """Repair monthly GPP using the documented conservative daily/monthly method."""
    header = read_raw_header(raw_path)
    priority_candidates, other_candidates = ordered_gpp_candidates(header)
    candidate_columns = priority_candidates + other_candidates

    base_report = {
        "SITE_ID": site_id,
        "raw_file_path": str(raw_path),
        "candidate_GPP_columns": " | ".join(candidate_columns),
        "selected_GPP_column": "",
        "n_months_total": 0,
        "n_months_with_valid_GPP": 0,
        "first_valid_month": "",
        "last_valid_month": "",
        "repair_status": "NO_USABLE_GPP",
        "notes": "",
    }
    if "TIMESTAMP_START" not in header:
        base_report["repair_status"] = "MISSING_TIMESTAMP_START"
        base_report["notes"] = "Raw file does not contain TIMESTAMP_START."
        return GPPRepairResult(site_id, pd.DataFrame(), base_report)
    if not candidate_columns:
        base_report["repair_status"] = "NO_GPP_COLUMNS"
        base_report["notes"] = "No columns containing GPP were found."
        return GPPRepairResult(site_id, pd.DataFrame(), base_report)

    useful_other = [
        column
        for column in other_candidates
        if not any(
            token in column.upper()
            for token in (
                "_SE",
                "USTAR",
                "_05",
                "_16",
                "_25",
                "_50",
                "_75",
                "_84",
                "_95",
            )
        )
    ]
    read_gpp_columns = priority_candidates + useful_other
    support_columns = [
        column
        for column in (
            "TIMESTAMP_START",
            "NIGHT",
            "SW_IN_F",
            "NEE_VUT_REF_QC",
            "NEE_CUT_REF_QC",
        )
        if column in header
    ]
    usecols = list(dict.fromkeys(support_columns + read_gpp_columns))
    raw = pd.read_csv(
        native_path(raw_path),
        usecols=usecols,
        low_memory=False,
    )
    raw.columns = [str(column).strip() for column in raw.columns]
    raw["datetime"] = parse_fluxnet_timestamp(raw["TIMESTAMP_START"])
    raw = raw.loc[raw["datetime"].notna()].copy()

    for column in set(read_gpp_columns) | {
        "NIGHT",
        "SW_IN_F",
        "NEE_VUT_REF_QC",
        "NEE_CUT_REF_QC",
    }:
        if column in raw:
            raw[column] = numeric_series(raw[column])

    selected_gpp: str | None = None
    selected_is_nonstandard = False
    for column in priority_candidates:
        if column in raw and raw[column].notna().any():
            selected_gpp = column
            break
    if selected_gpp is None:
        for column in useful_other:
            if column in raw and raw[column].notna().any():
                selected_gpp = column
                selected_is_nonstandard = True
                break

    if selected_gpp is None:
        base_report["notes"] = (
            "GPP columns exist, but all inspected standard/plausible values are "
            "-9999 or missing."
        )
        return GPPRepairResult(site_id, pd.DataFrame(), base_report)

    base_report["selected_GPP_column"] = selected_gpp
    if selected_is_nonstandard and not ALLOW_NONSTANDARD_GPP_FALLBACK:
        base_report["repair_status"] = "NONSTANDARD_GPP_REQUIRES_REVIEW"
        base_report["notes"] = (
            f"{selected_gpp} contains data but is outside the explicit priority "
            "list; enable ALLOW_NONSTANDARD_GPP_FALLBACK only after review."
        )
        return GPPRepairResult(site_id, pd.DataFrame(), base_report)

    if "NIGHT" in raw and raw["NIGHT"].notna().any():
        daytime = raw["NIGHT"].eq(0)
        daytime_method = "NIGHT_equals_0"
    elif "SW_IN_F" in raw and raw["SW_IN_F"].notna().any():
        daytime = raw["SW_IN_F"].gt(SW_IN_DAYTIME_THRESHOLD)
        daytime_method = f"SW_IN_F_gt_{SW_IN_DAYTIME_THRESHOLD:g}"
    else:
        daytime = pd.Series(True, index=raw.index)
        daytime_method = "no_daytime_filter_available_all_records_used"

    qc_column = matching_nee_qc_column(selected_gpp, raw.columns)
    if qc_column and raw[qc_column].notna().any():
        conservative = raw[qc_column].eq(0)
        conservative_method = f"{qc_column}_equals_0"
    else:
        conservative = pd.Series(True, index=raw.index)
        conservative_method = "matching_NEE_QC_unavailable_all_records_used"

    raw["date"] = raw["datetime"].dt.floor("D")
    raw["YYYYMM"] = raw["datetime"].dt.strftime("%Y%m")
    all_days = (
        raw[["date", "YYYYMM"]]
        .drop_duplicates()
        .groupby("YYYYMM")
        .size()
        .rename("n_days_available")
    )

    selected = raw.loc[daytime & conservative, ["date", "YYYYMM", selected_gpp]]
    conservative_counts = (
        selected.groupby(["YYYYMM", "date"])
        .size()
        .rename("n_conservative_daytime_records")
    )
    daily_gpp = (
        selected.groupby(["YYYYMM", "date"])[selected_gpp]
        .agg(["mean", "count"])
        .rename(columns={"mean": "daily_GPP", "count": "n_valid_GPP_records"})
    )
    daily = daily_gpp.join(conservative_counts, how="outer").reset_index()

    if daily.empty:
        base_report["notes"] = "No conservative daytime records remained."
        return GPPRepairResult(site_id, pd.DataFrame(), base_report)

    rows: list[dict[str, Any]] = []
    for yyyymm, group in daily.groupby("YYYYMM"):
        valid_daily = group["daily_GPP"].dropna()
        valid_day_count = int(valid_daily.count())
        total_conservative_records = int(
            group["n_conservative_daytime_records"].fillna(0).sum()
        )
        candidate_value = (
            round(float(valid_daily.mean()), 6)
            if valid_day_count > 0
            else np.nan
        )
        if valid_day_count < MIN_VALID_DAYS_PER_MONTH:
            quality = "LOW_VALID_DAYS"
        elif (
            total_conservative_records
            < MIN_CONSERVATIVE_DAYTIME_RECORDS_PER_MONTH
        ):
            quality = "LOW_CONSERVATIVE_HH_RECORDS"
        else:
            quality = "OK"

        eligible_value = candidate_value if quality == "OK" else np.nan
        rows.append(
            {
                "SITE_ID": site_id,
                "YYYYMM": str(yyyymm),
                "year": int(str(yyyymm)[:4]),
                "month": int(str(yyyymm)[4:]),
                "tower_GPP_monthly": eligible_value,
                "candidate_tower_GPP_monthly_before_quality_gate": (
                    candidate_value
                ),
                "primary_GPP_column": selected_gpp,
                "GPP_monthly_quality_flag": quality,
                "tower_GPP_units_original_FLUXNET": "umol_CO2_m-2_s-1",
                "n_days_available": int(all_days.get(yyyymm, 0)),
                "n_valid_conservative_GPP_days": valid_day_count,
                "n_conservative_daytime_HH_records": (
                    total_conservative_records
                ),
                "daytime_filter_used": daytime_method,
                "conservative_flux_filter_used": conservative_method,
            }
        )

    monthly = pd.DataFrame(rows)
    valid_months = monthly.loc[
        monthly["tower_GPP_monthly"].notna(),
        "YYYYMM",
    ].sort_values()
    base_report.update(
        {
            "selected_GPP_column": selected_gpp,
            "n_months_total": int(len(monthly)),
            "n_months_with_valid_GPP": int(valid_months.nunique()),
            "first_valid_month": (
                str(valid_months.iloc[0]) if not valid_months.empty else ""
            ),
            "last_valid_month": (
                str(valid_months.iloc[-1]) if not valid_months.empty else ""
            ),
            "repair_status": (
                "GPP_REPAIR_CANDIDATE_FOUND"
                if not valid_months.empty
                else "NO_MONTH_PASSES_GPP_QUALITY_THRESHOLDS"
            ),
            "notes": (
                f"Selected highest-priority column with real data; daytime="
                f"{daytime_method}; conservative_filter={conservative_method}; "
                "monthly values are applied only when quality flag is OK."
            ),
        }
    )
    return GPPRepairResult(site_id, monthly, base_report)


# =============================================================================
# TRUSTED METADATA AND RADIUS REPAIR
# =============================================================================


METADATA_ALIASES = {
    "SITE_ID": ["SITE_ID", "Site_ID", "Site ID", "Site", "site", "site_id", "SITE"],
    "site_name": ["site_name", "SITE_NAME", "Site name", "Site Name"],
    "lat": ["lat", "Lat", "LAT", "Latitude", "latitude"],
    "lon": ["lon", "Lon", "LON", "Longitude", "longitude"],
    "IGBP": ["IGBP", "igbp", "Igbp"],
    "canopy_height_m": [
        "Canopy height m",
        "Canopy height",
        "canopy_height_m",
    ],
    "EC_measurement_height_m": [
        "EC measurement height m",
        "EC measurement height",
        "EC_measurement_height_m",
    ],
    "height_above_canopy_m": [
        "Height above canopy m",
        "Height above canopy",
        "height_above_canopy_m",
    ],
    "radius_m": [
        "circle buffer radius m",
        "Circle buffer radius m",
        "Circular buffer radius m",
        "circular buffer radius m",
        "radius_m",
    ],
    "radius_m_min": ["radius_m_min", "Radius m min", "radius min"],
    "radius_m_max": ["radius_m_max", "Radius m max", "radius max"],
    "Metadata source": ["Metadata source", "metadata source"],
    "Metadata status": ["Metadata status", "metadata status"],
    "Notes": ["Notes", "notes"],
}


def normalized_column_name(value: Any) -> str:
    """Normalize a metadata header for alias matching."""
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def metadata_alias_lookup(columns: Iterable[Any]) -> dict[str, str]:
    """Map canonical metadata fields to actual columns."""
    actual_lookup = {
        normalized_column_name(column): str(column) for column in columns
    }
    result: dict[str, str] = {}
    for canonical, aliases in METADATA_ALIASES.items():
        for alias in aliases:
            actual = actual_lookup.get(normalized_column_name(alias))
            if actual is not None:
                result[canonical] = actual
                break
    return result


def metadata_path_priority(path: Path) -> int:
    """Rank explicitly maintained project metadata ahead of derived tables."""
    name = path.name.lower()
    if name in {
        "new_fluxnet_sites_metadata_for_kmz_preparation.xlsx",
        "sites information.xlsx",
    }:
        return 0
    if "site_metadata_from_excel" in name or "site metadata" in name:
        return 1
    if "metadata" in name:
        return 2
    return 3


def metadata_candidate_paths() -> list[Path]:
    """Discover requested metadata tables, using the audit inventory when present."""
    paths: set[Path] = {
        PROJECT_ROOT / "New_FLUXNET_sites_metadata_for_KMZ_preparation.xlsx",
        PROJECT_ROOT / "Sites Information.xlsx",
    }
    detected_report = EXISTING_AUDIT_QC / "detected_metadata_files.csv"
    if detected_report.exists():
        detected = pd.read_csv(detected_report, encoding="utf-8-sig")
        for value in detected.get("metadata_file_path", pd.Series(dtype=str)):
            if not is_missing(value):
                paths.add(Path(str(value)))
    else:
        report = completeness_audit.detect_metadata_files(
            PROJECT_ROOT,
            EXISTING_AUDIT_QC,
        )
        for value in report.get("metadata_file_path", pd.Series(dtype=str)):
            if not is_missing(value):
                paths.add(Path(str(value)))

    excluded = {
        os.path.normcase(os.path.abspath(os.fspath(path)))
        for path in INPUT_TABLE_PATHS
    }
    usable = [
        path
        for path in paths
        if path.suffix.lower() in {".csv", ".xlsx", ".xls"}
        and path_exists(path)
        and os.path.normcase(os.path.abspath(os.fspath(path))) not in excluded
        and os.path.getsize(native_path(path))
        <= MAX_METADATA_READ_SIZE_MB * 1024**2
    ]
    return sorted(
        usable,
        key=lambda path: (metadata_path_priority(path), str(path).lower()),
    )


def read_target_metadata_rows(
    path: Path,
    target_sites: set[str],
) -> list[dict[str, Any]]:
    """Read exact target SITE_ID rows from a metadata CSV or workbook."""
    records: list[dict[str, Any]] = []
    try:
        if path.suffix.lower() == ".csv":
            header = pd.read_csv(native_path(path), nrows=0)
            aliases = metadata_alias_lookup(header.columns)
            site_column = aliases.get("SITE_ID")
            if site_column is None:
                return records
            usecols = sorted(set(aliases.values()))
            for chunk in pd.read_csv(
                native_path(path),
                usecols=usecols,
                chunksize=100_000,
                low_memory=False,
            ):
                site_values = chunk[site_column].astype(str).str.strip()
                matched = chunk.loc[site_values.isin(target_sites)]
                for row in matched.to_dict("records"):
                    row["_metadata_sheet"] = ""
                    records.append(row)
        else:
            engine = "openpyxl" if path.suffix.lower() == ".xlsx" else "xlrd"
            workbook = pd.ExcelFile(native_path(path), engine=engine)
            for sheet in workbook.sheet_names:
                header = pd.read_excel(workbook, sheet_name=sheet, nrows=0)
                aliases = metadata_alias_lookup(header.columns)
                site_column = aliases.get("SITE_ID")
                if site_column is None:
                    continue
                usecols = sorted(set(aliases.values()))
                table = pd.read_excel(
                    workbook,
                    sheet_name=sheet,
                    usecols=usecols,
                )
                site_values = table[site_column].astype(str).str.strip()
                matched = table.loc[site_values.isin(target_sites)]
                for row in matched.to_dict("records"):
                    row["_metadata_sheet"] = str(sheet)
                    records.append(row)
    except Exception as error:
        warn(f"Could not inspect metadata file '{path}': {error}")
    return records


def canonical_metadata_candidate(
    path: Path,
    row: dict[str, Any],
) -> dict[str, Any] | None:
    """Standardize one exact metadata row and reproduce trusted radius derivation."""
    aliases = metadata_alias_lookup(row.keys())
    site_column = aliases.get("SITE_ID")
    if site_column is None:
        return None
    site_id = clean_site_id(row.get(site_column))
    if site_id is None:
        return None

    def value(canonical: str) -> Any:
        actual = aliases.get(canonical)
        return row.get(actual) if actual is not None else np.nan

    status = str(value("Metadata status")).strip()
    trusted = "needs manual verification" not in status.lower()
    lat = pd.to_numeric(pd.Series([value("lat")]), errors="coerce").iloc[0]
    lon = pd.to_numeric(pd.Series([value("lon")]), errors="coerce").iloc[0]
    if not valid_lat_lon(lat, lon):
        lat = np.nan
        lon = np.nan

    canopy_min, canopy_max = parse_positive_number_or_range(
        value("canopy_height_m")
    )
    ec_min, ec_max = parse_positive_number_or_range(
        value("EC_measurement_height_m")
    )
    height_min, height_max = parse_positive_number_or_range(
        value("height_above_canopy_m")
    )
    radius_min, radius_max = parse_positive_number_or_range(value("radius_m"))

    explicit_radius_min = positive_number(value("radius_m_min"))
    explicit_radius_max = positive_number(value("radius_m_max"))
    if explicit_radius_min is not None:
        radius_min = explicit_radius_min
    if explicit_radius_max is not None:
        radius_max = explicit_radius_max
    if radius_min is None and radius_max is not None:
        radius_min = radius_max
    if radius_max is None and radius_min is not None:
        radius_max = radius_min

    height_method = "trusted_explicit_height_above_canopy"
    if height_max is None and ec_max is not None and canopy_max is not None:
        calculated_height = ec_max - canopy_max
        if calculated_height > 0.0:
            height_min = calculated_height
            height_max = calculated_height
            height_method = "EC_measurement_height_minus_canopy_height"

    radius_method = "trusted_explicit_radius"
    if radius_max is None and height_max is not None and height_max > 0.0:
        # Exact formula used by the remaining-site final-merge workflow.
        calculated_radius = height_max * 100.0
        radius_min = calculated_radius
        radius_max = calculated_radius
        radius_method = f"{height_method}_times_100"

    source_value = value("Metadata source")
    source_note = (
        str(source_value).strip()
        if not is_missing(source_value)
        else str(path)
    )
    sheet = str(row.get("_metadata_sheet", "")).strip()
    file_label = str(path) + (f"::{sheet}" if sheet else "")

    return {
        "SITE_ID": site_id,
        "metadata_file": file_label,
        "metadata_priority": metadata_path_priority(path),
        "trusted": trusted,
        "site_name": value("site_name"),
        "lat": float(lat) if not pd.isna(lat) else np.nan,
        "lon": float(lon) if not pd.isna(lon) else np.nan,
        "IGBP": value("IGBP"),
        "canopy_height_m": canopy_max,
        "EC_measurement_height_m": ec_max,
        "height_above_canopy_m": height_max,
        "radius_m": radius_max,
        "radius_m_min": radius_min,
        "radius_m_max": radius_max,
        "radius_method": radius_method if radius_max is not None else "",
        "metadata_status": status,
        "metadata_source_note": source_note,
        "notes": value("Notes"),
    }


def build_metadata_registry(
    target_sites: Iterable[str],
) -> dict[str, MetadataRepairResult]:
    """Search all trusted metadata candidates for exact target SITE_ID matches."""
    target_set = set(target_sites)
    candidates: dict[str, list[dict[str, Any]]] = {
        site: [] for site in target_set
    }
    paths = metadata_candidate_paths()
    print(f"Metadata files inspected: {len(paths):,}", flush=True)
    for path in paths:
        rows = read_target_metadata_rows(path, target_set)
        for row in rows:
            candidate = canonical_metadata_candidate(path, row)
            if candidate and candidate["SITE_ID"] in candidates:
                candidates[candidate["SITE_ID"]].append(candidate)

    results: dict[str, MetadataRepairResult] = {}
    for site_id in sorted(target_set):
        site_candidates = sorted(
            candidates.get(site_id, []),
            key=lambda item: (
                not bool(item["trusted"]),
                int(item["metadata_priority"]),
                str(item["metadata_file"]).lower(),
            ),
        )
        trusted_candidates = [
            item for item in site_candidates if bool(item["trusted"])
        ]
        lat_candidate = next(
            (
                item
                for item in trusted_candidates
                if valid_lat_lon(item["lat"], item["lon"])
            ),
            None,
        )
        radius_candidate = next(
            (
                item
                for item in trusted_candidates
                if positive_number(item["radius_m_max"]) is not None
            ),
            None,
        )
        descriptive_candidate = lat_candidate or radius_candidate or (
            trusted_candidates[0] if trusted_candidates else None
        )

        sources = unique_join(
            item["metadata_file"]
            for item in (lat_candidate, radius_candidate)
            if item is not None
        )
        notes: list[str] = []
        if lat_candidate:
            notes.append(
                f"lat/lon from {lat_candidate['metadata_source_note']}"
            )
        if radius_candidate:
            notes.append(
                f"radius method={radius_candidate['radius_method']}; "
                f"source={radius_candidate['metadata_source_note']}"
            )
        if not trusted_candidates:
            notes.append("No exact trusted metadata row was found.")
        elif not radius_candidate:
            notes.append(
                "Exact metadata rows were found, but none contains or supports "
                "a defensible radius derivation."
            )

        if lat_candidate and radius_candidate:
            status = "TRUSTED_LAT_LON_AND_RADIUS_FOUND"
        elif radius_candidate:
            status = "TRUSTED_RADIUS_FOUND"
        elif lat_candidate:
            status = "TRUSTED_LAT_LON_FOUND_RADIUS_MISSING"
        else:
            status = "NO_TRUSTED_METADATA_REPAIR"

        results[site_id] = MetadataRepairResult(
            SITE_ID=site_id,
            metadata_file_used=sources,
            site_name=(
                descriptive_candidate["site_name"]
                if descriptive_candidate
                else np.nan
            ),
            lat=(
                float(lat_candidate["lat"]) if lat_candidate else None
            ),
            lon=(
                float(lat_candidate["lon"]) if lat_candidate else None
            ),
            IGBP=(
                descriptive_candidate["IGBP"]
                if descriptive_candidate
                else np.nan
            ),
            canopy_height_m=(
                radius_candidate["canopy_height_m"]
                if radius_candidate
                else None
            ),
            EC_measurement_height_m=(
                radius_candidate["EC_measurement_height_m"]
                if radius_candidate
                else None
            ),
            height_above_canopy_m=(
                radius_candidate["height_above_canopy_m"]
                if radius_candidate
                else None
            ),
            radius_m=(
                radius_candidate["radius_m"] if radius_candidate else None
            ),
            radius_m_min=(
                radius_candidate["radius_m_min"]
                if radius_candidate
                else None
            ),
            radius_m_max=(
                radius_candidate["radius_m_max"]
                if radius_candidate
                else None
            ),
            repair_status=status,
            notes="; ".join(notes),
        )
    return results


def metadata_result_to_radius_report(
    result: MetadataRepairResult,
) -> dict[str, Any]:
    """Convert a metadata result to the requested radius report schema."""
    return {
        "SITE_ID": result.SITE_ID,
        "metadata_file_used": result.metadata_file_used,
        "lat_found": result.lat,
        "lon_found": result.lon,
        "canopy_height_m": result.canopy_height_m,
        "EC_measurement_height_m": result.EC_measurement_height_m,
        "height_above_canopy_m": result.height_above_canopy_m,
        "radius_m": result.radius_m,
        "radius_m_min": result.radius_m_min,
        "radius_m_max": result.radius_m_max,
        "repair_status": result.repair_status,
        "notes": result.notes,
    }


def metadata_result_to_latlon_radius_report(
    result: MetadataRepairResult,
) -> dict[str, Any]:
    """Convert a metadata result to the requested lat/lon+radius schema."""
    record = metadata_result_to_radius_report(result)
    record.pop("lat_found")
    record.pop("lon_found")
    record.update(
        {
            "lat": result.lat,
            "lon": result.lon,
            "IGBP": result.IGBP,
        }
    )
    return {column: record.get(column) for column in LATLON_RADIUS_REPORT_COLUMNS}


# =============================================================================
# BE-LON ZIP VALIDATION, SAFE EXTRACTION, AND WD SUPPORT
# =============================================================================


def inspect_truncated_local_zip_header(path: Path) -> BELonZipInspection:
    """Recover the first member/header facts when a ZIP central directory is absent."""
    try:
        file_size = os.path.getsize(native_path(path))
        with open(native_path(path), "rb") as stream:
            local_header = stream.read(30)
            if len(local_header) != 30:
                raise ValueError("Local ZIP header is incomplete.")
            (
                signature,
                _version,
                _flags,
                compression_method,
                _time,
                _date,
                _crc,
                compressed_size,
                uncompressed_size,
                filename_length,
                extra_length,
            ) = struct.unpack("<IHHHHHIIIHH", local_header)
            if signature != 0x04034B50:
                raise ValueError("ZIP local-file signature was not found.")
            filename = stream.read(filename_length).decode(
                "utf-8",
                errors="replace",
            )
            stream.read(extra_length)
            data_offset = 30 + filename_length + extra_length
            truncated_bytes = max(
                0,
                data_offset + compressed_size - file_size,
            )

            columns: list[str] = []
            if compression_method == 8:
                decompressor = zlib.decompressobj(-15)
                output = b""
                while b"\n" not in output and len(output) < 1_000_000:
                    chunk = stream.read(64 * 1024)
                    if not chunk:
                        break
                    output += decompressor.decompress(chunk, 1_000_000)
                if output:
                    header_line = output.splitlines()[0].decode(
                        "utf-8-sig",
                        errors="replace",
                    )
                    columns = [item.strip() for item in header_line.split(",")]

        return BELonZipInspection(
            zip_is_structurally_valid=False,
            member=filename,
            compressed_size_expected=compressed_size,
            uncompressed_size_expected=uncompressed_size,
            truncated_bytes_detected=truncated_bytes,
            columns=columns,
            repair_status="CORRUPT_TRUNCATED_ZIP_REDOWNLOAD_REQUIRED",
            notes=(
                "ZIP central directory is missing/corrupt. The local member "
                f"header indicates {truncated_bytes:,} compressed bytes are "
                "missing. Partial data will not be used."
            ),
        )
    except Exception as error:
        return BELonZipInspection(
            zip_is_structurally_valid=False,
            member="",
            compressed_size_expected=None,
            uncompressed_size_expected=None,
            truncated_bytes_detected=None,
            columns=[],
            repair_status="UNREADABLE_ZIP",
            notes=f"{type(error).__name__}: {error}",
        )


def inspect_be_lon_zip(path: Path) -> BELonZipInspection:
    """Validate BE-Lon ZIP structure and identify its HH/HR member."""
    if not path_exists(path):
        return BELonZipInspection(
            False,
            "",
            None,
            None,
            None,
            [],
            "ZIP_NOT_FOUND",
            "BE-Lon ZIP file was not found.",
        )
    if not zipfile.is_zipfile(native_path(path)):
        return inspect_truncated_local_zip_header(path)

    try:
        with zipfile.ZipFile(native_path(path)) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                return BELonZipInspection(
                    False,
                    bad_member,
                    None,
                    None,
                    None,
                    [],
                    "ZIP_CRC_FAILURE",
                    f"CRC validation failed for member: {bad_member}",
                )
            members = [
                info
                for info in archive.infolist()
                if not info.is_dir()
                and info.filename.lower().endswith(".csv")
                and re.search(
                    r"(?:_HH_|_HR_|FLUXMET_HH|FLUXMET_HR)",
                    Path(info.filename).name,
                    re.IGNORECASE,
                )
            ]
            if not members:
                return BELonZipInspection(
                    True,
                    "",
                    None,
                    None,
                    0,
                    [],
                    "NO_HH_HR_MEMBER",
                    "No HH/HR FLUXMET CSV member was found in the valid ZIP.",
                )
            member = max(members, key=lambda info: info.file_size)
            with archive.open(member) as stream:
                header_line = stream.readline().decode(
                    "utf-8-sig",
                    errors="replace",
                )
            columns = [item.strip() for item in header_line.strip().split(",")]
            return BELonZipInspection(
                True,
                member.filename,
                member.compress_size,
                member.file_size,
                0,
                columns,
                "VALID_ZIP_HH_HR_MEMBER_FOUND",
                "ZIP structure and CRC validated.",
            )
    except Exception:
        return inspect_truncated_local_zip_header(path)


def safely_extract_be_lon_member(
    zip_path: Path,
    inspection: BELonZipInspection,
) -> Path | None:
    """Extract only a validated HH/HR member without traversal or overwrite."""
    if DRY_RUN or not inspection.zip_is_structurally_valid or not inspection.member:
        return None
    BE_LON_EXTRACT_FOLDER.mkdir(parents=True, exist_ok=True)
    target = BE_LON_EXTRACT_FOLDER / Path(inspection.member).name
    target_resolved = target.resolve()
    root_resolved = BE_LON_EXTRACT_FOLDER.resolve()
    if root_resolved not in target_resolved.parents:
        raise RuntimeError("Unsafe BE-Lon ZIP member path rejected.")
    if path_exists(target):
        return target
    with zipfile.ZipFile(native_path(zip_path)) as archive:
        with archive.open(inspection.member) as source, open(
            native_path(target),
            "xb",
        ) as destination:
            shutil.copyfileobj(source, destination)
    return target


def direction_to_cardinal(degrees: float) -> str:
    """Convert a bearing to the project's 16-point compass label."""
    directions = [
        "N",
        "NNE",
        "NE",
        "ENE",
        "E",
        "ESE",
        "SE",
        "SSE",
        "S",
        "SSW",
        "SW",
        "WSW",
        "W",
        "WNW",
        "NW",
        "NNW",
    ]
    return directions[int((degrees + 11.25) / 22.5) % 16]


def circular_mean_degrees(values: pd.Series) -> float:
    """Compute the diagnostic circular mean."""
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    if numeric.size == 0:
        return math.nan
    radians = np.deg2rad(numeric)
    angle = math.degrees(
        math.atan2(float(np.mean(np.sin(radians))), float(np.mean(np.cos(radians))))
    )
    return angle + 360.0 if angle < 0.0 else angle


def circular_resultant_length(values: pd.Series) -> float:
    """Compute the diagnostic circular resultant length."""
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    if numeric.size == 0:
        return math.nan
    radians = np.deg2rad(numeric)
    return float(
        math.sqrt(
            float(np.mean(np.sin(radians))) ** 2
            + float(np.mean(np.cos(radians))) ** 2
        )
    )


def build_monthly_wd_bins_from_raw(
    site_id: str,
    raw_path: Path,
) -> pd.DataFrame:
    """Reproduce the project's daytime 10-degree monthly WD histogram."""
    header = read_raw_header(raw_path)
    required = {"TIMESTAMP_START", "WD"}
    if not required.issubset(header):
        return pd.DataFrame()
    usecols = [
        column
        for column in (
            "TIMESTAMP_START",
            "WD",
            "NIGHT",
            "SW_IN_F",
            "WS_F",
            "USTAR",
        )
        if column in header
    ]
    raw = pd.read_csv(native_path(raw_path), usecols=usecols, low_memory=False)
    raw["datetime"] = parse_fluxnet_timestamp(raw["TIMESTAMP_START"])
    for column in ("WD", "NIGHT", "SW_IN_F", "WS_F", "USTAR"):
        if column not in raw:
            raw[column] = np.nan
        raw[column] = numeric_series(raw[column])
    raw = raw.loc[
        raw["datetime"].notna()
        & raw["WD"].between(0.0, 360.0, inclusive="both")
    ].copy()
    raw.loc[raw["WD"] == 360.0, "WD"] = 0.0
    if raw["NIGHT"].notna().any():
        daytime = raw["NIGHT"].eq(0)
        daytime_method = "NIGHT_equals_0"
    elif raw["SW_IN_F"].notna().any():
        daytime = raw["SW_IN_F"].gt(SW_IN_DAYTIME_THRESHOLD)
        daytime_method = f"SW_IN_F_greater_than_{SW_IN_DAYTIME_THRESHOLD:g}"
    else:
        daytime = pd.Series(True, index=raw.index)
        daytime_method = "no_daytime_filter_available_all_records_used"
    raw = raw.loc[daytime].copy()
    raw["YYYYMM"] = raw["datetime"].dt.strftime("%Y%m")
    raw["year"] = raw["datetime"].dt.year
    raw["month"] = raw["datetime"].dt.month
    raw["WD_bin_start_deg"] = (
        np.floor(raw["WD"] / WD_BIN_WIDTH_DEG) * WD_BIN_WIDTH_DEG
    ).astype(int)
    raw["WD_bin_end_deg"] = raw["WD_bin_start_deg"] + int(WD_BIN_WIDTH_DEG)
    raw["WD_bin_center_deg"] = (
        raw["WD_bin_start_deg"] + WD_BIN_WIDTH_DEG / 2.0
    ) % 360.0

    rows: list[dict[str, Any]] = []
    for yyyymm, group in raw.groupby("YYYYMM"):
        n_total = int(len(group))
        if n_total < MIN_VALID_MONTHLY_WD_RECORDS:
            continue
        mean_wd = circular_mean_degrees(group["WD"])
        resultant = circular_resultant_length(group["WD"])
        counts = (
            group.groupby(
                [
                    "WD_bin_start_deg",
                    "WD_bin_end_deg",
                    "WD_bin_center_deg",
                ]
            )
            .size()
            .reset_index(name="WD_bin_count")
        )
        for record in counts.to_dict("records"):
            rows.append(
                {
                    "SITE_ID": site_id,
                    "YYYYMM": str(yyyymm),
                    "year": int(str(yyyymm)[:4]),
                    "month": int(str(yyyymm)[4:]),
                    "WD_bin_width_deg": WD_BIN_WIDTH_DEG,
                    "WD_bin_start_deg": int(record["WD_bin_start_deg"]),
                    "WD_bin_end_deg": int(record["WD_bin_end_deg"]),
                    "WD_bin_center_deg": float(record["WD_bin_center_deg"]),
                    "WD_bin_center_cardinal": direction_to_cardinal(
                        float(record["WD_bin_center_deg"])
                    ),
                    "WD_bin_count": int(record["WD_bin_count"]),
                    "WD_bin_fraction": round(
                        float(record["WD_bin_count"] / n_total),
                        8,
                    ),
                    "n_valid_monthly_30min_WD_records": n_total,
                    "monthly_circular_mean_WD_deg_diagnostic": round(
                        mean_wd,
                        2,
                    ),
                    "monthly_circular_mean_WD_cardinal_diagnostic": (
                        direction_to_cardinal(mean_wd)
                    ),
                    "monthly_WD_consistency_R_diagnostic": round(
                        resultant,
                        3,
                    ),
                    "quality_flag": (
                        "OK"
                        if n_total >= 100
                        else "LOW_MONTHLY_DAYTIME_WD_RECORD_COUNT"
                    ),
                    "daytime_filter_used": daytime_method,
                    "mean_monthly_daytime_WS_F": group["WS_F"].mean(),
                    "mean_monthly_daytime_USTAR": group["USTAR"].mean(),
                    "mean_monthly_daytime_SW_IN_F": group["SW_IN_F"].mean(),
                    "footprint_method_next_step": (
                        "weighted_monthly_probability_from_30min_WD_bins"
                    ),
                }
            )
    return pd.DataFrame(rows)


def be_lon_report_record(
    inspection: BELonZipInspection,
    extracted_path: Path | None,
    selected_gpp: str = "",
) -> dict[str, Any]:
    """Build the BE-Lon status report record."""
    priority, other = ordered_gpp_candidates(inspection.columns)
    return {
        "SITE_ID": BE_LON_SITE,
        "zip_file_path": str(BE_LON_ZIP),
        "zip_is_structurally_valid": inspection.zip_is_structurally_valid,
        "zip_member": inspection.member,
        "compressed_size_expected": inspection.compressed_size_expected,
        "uncompressed_size_expected": inspection.uncompressed_size_expected,
        "truncated_bytes_detected": inspection.truncated_bytes_detected,
        "detected_columns": " | ".join(inspection.columns),
        "timestamp_column": (
            "TIMESTAMP_START"
            if "TIMESTAMP_START" in inspection.columns
            else ""
        ),
        "WD_column": "WD" if "WD" in inspection.columns else "",
        "GPP_candidate_columns": " | ".join(priority + other),
        "selected_GPP_column": selected_gpp,
        "extraction_performed": extracted_path is not None,
        "extracted_file_path": str(extracted_path or ""),
        "repair_status": inspection.repair_status,
        "notes": inspection.notes,
    }


def build_be_lon_final_rows(
    raw_path: Path,
    metadata: MetadataRepairResult,
) -> tuple[pd.DataFrame, GPPRepairResult]:
    """Build new BE-Lon final rows only from validated raw and trusted metadata."""
    wd = build_monthly_wd_bins_from_raw(BE_LON_SITE, raw_path)
    gpp = aggregate_monthly_gpp_from_raw(BE_LON_SITE, raw_path)
    if wd.empty:
        return pd.DataFrame(), gpp
    # WD already supplies year/month. Avoid merge suffixes so newly generated
    # BE-Lon rows retain the same final-table schema as every existing site.
    gpp_columns = [
        column
        for column in gpp.monthly.columns
        if column not in {"year", "month"}
    ]
    monthly = wd.merge(
        gpp.monthly[gpp_columns],
        on=["SITE_ID", "YYYYMM"],
        how="left",
    )
    for column, value in (
        ("site_name", metadata.site_name),
        ("lat", metadata.lat),
        ("lon", metadata.lon),
        ("IGBP", metadata.IGBP),
        ("canopy_height_m", metadata.canopy_height_m),
        ("EC_measurement_height_m", metadata.EC_measurement_height_m),
        ("height_above_canopy_m", metadata.height_above_canopy_m),
        ("radius_m", metadata.radius_m),
        ("radius_m_min", metadata.radius_m_min),
        ("radius_m_max", metadata.radius_m_max),
    ):
        monthly[column] = value
    monthly["GPP_source"] = "HH_conservative_repair"
    monthly["GPP_merge_status"] = "HH_CONSERVATIVE_GPP_REPAIR"
    monthly["footprint_method"] = "monthly_probability_from_30min_WD_histogram"
    monthly["blind_buffer_method"] = "tower_centered_circular_buffer"
    monthly["source_input_table"] = str(BE_LON_ZIP)
    monthly["_source_group"] = "remaining"
    monthly["repaired_lat_lon"] = True
    monthly["repaired_radius"] = True
    monthly["repaired_GPP"] = monthly["tower_GPP_monthly"].notna()
    monthly["repair_source"] = (
        f"BE-Lon validated raw HH/HR + {metadata.metadata_file_used}"
    )
    monthly["repair_notes"] = (
        "New site rows generated with the documented WD/GPP/radius methods."
    )
    return monthly, gpp


# =============================================================================
# APPLY TRUSTED REPAIRS TO AN IN-MEMORY COPY
# =============================================================================


def apply_metadata_repairs(
    table: pd.DataFrame,
    metadata_results: dict[str, MetadataRepairResult],
) -> None:
    """Fill only missing metadata/radius fields from trusted exact matches."""
    for site_id, result in metadata_results.items():
        site_mask = table["SITE_ID"].eq(site_id)
        if not site_mask.any():
            continue

        current_lat = pd.to_numeric(table.loc[site_mask, "lat"], errors="coerce")
        current_lon = pd.to_numeric(table.loc[site_mask, "lon"], errors="coerce")
        latlon_missing = ~(
            current_lat.between(-90.0, 90.0)
            & current_lon.between(-180.0, 180.0)
        )
        latlon_index = current_lat.index[latlon_missing]
        if (
            result.lat is not None
            and result.lon is not None
            and valid_lat_lon(result.lat, result.lon)
            and len(latlon_index) > 0
        ):
            mask = table.index.isin(latlon_index)
            table.loc[mask, "lat"] = result.lat
            table.loc[mask, "lon"] = result.lon
            table.loc[mask, "repaired_lat_lon"] = True
            append_provenance(
                table,
                pd.Series(mask, index=table.index),
                result.metadata_file_used,
                "trusted exact SITE_ID lat/lon repair",
            )

        for column, value in (
            ("site_name", result.site_name),
            ("IGBP", result.IGBP),
            ("canopy_height_m", result.canopy_height_m),
            ("EC_measurement_height_m", result.EC_measurement_height_m),
            ("height_above_canopy_m", result.height_above_canopy_m),
        ):
            if column not in table:
                table[column] = np.nan
            if is_missing(value):
                continue
            missing_mask = site_mask & table[column].map(is_missing)
            table.loc[missing_mask, column] = value

        if positive_number(result.radius_m_max) is not None:
            if "radius_m" not in table:
                table["radius_m"] = np.nan
            if "radius_m_min" not in table:
                table["radius_m_min"] = np.nan
            if "radius_m_max" not in table:
                table["radius_m_max"] = np.nan
            current_radius = pd.to_numeric(
                table["radius_m_max"],
                errors="coerce",
            )
            missing_radius = site_mask & (
                current_radius.isna() | current_radius.le(0.0)
            )
            table.loc[missing_radius, "radius_m"] = result.radius_m
            table.loc[missing_radius, "radius_m_min"] = result.radius_m_min
            table.loc[missing_radius, "radius_m_max"] = result.radius_m_max
            table.loc[missing_radius, "repaired_radius"] = True
            append_provenance(
                table,
                missing_radius,
                result.metadata_file_used,
                "trusted explicit/height-derived radius repair",
            )


def apply_gpp_repairs(
    table: pd.DataFrame,
    gpp_results: dict[str, GPPRepairResult],
) -> None:
    """Fill missing site-month GPP only where the repaired monthly quality is OK."""
    candidate_frames: list[pd.DataFrame] = []
    for result in gpp_results.values():
        if result.monthly.empty:
            continue
        valid = result.monthly.loc[
            result.monthly["tower_GPP_monthly"].notna(),
            [
                "SITE_ID",
                "YYYYMM",
                "tower_GPP_monthly",
                "primary_GPP_column",
                "GPP_monthly_quality_flag",
                "tower_GPP_units_original_FLUXNET",
            ],
        ].copy()
        if not valid.empty:
            candidate_frames.append(valid)
    if not candidate_frames:
        return

    candidates = pd.concat(candidate_frames, ignore_index=True)
    candidates["YYYYMM"] = candidates["YYYYMM"].astype(str)
    candidates = candidates.drop_duplicates(["SITE_ID", "YYYYMM"], keep="first")
    candidate_value_columns = [
        column
        for column in candidates.columns
        if column not in {"SITE_ID", "YYYYMM"}
    ]
    candidates = candidates.rename(
        columns={
            column: f"_repair_{column}"
            for column in candidate_value_columns
        }
    )

    # One vectorized site-month join replaces thousands of full-table scans.
    aligned = table[["SITE_ID", "YYYYMM"]].merge(
        candidates,
        on=["SITE_ID", "YYYYMM"],
        how="left",
        sort=False,
        validate="many_to_one",
    )
    if len(aligned) != len(table):
        raise RuntimeError("GPP repair join changed the final-table row count.")

    current_gpp = pd.to_numeric(table["tower_GPP_monthly"], errors="coerce")
    candidate_gpp = pd.to_numeric(
        aligned["_repair_tower_GPP_monthly"],
        errors="coerce",
    )
    repair_mask = (
        current_gpp.isna() | current_gpp.eq(MISSING_VALUE)
    ) & candidate_gpp.notna()
    if not repair_mask.any():
        return

    for column in (
        "tower_GPP_monthly",
        "primary_GPP_column",
        "GPP_monthly_quality_flag",
        "tower_GPP_units_original_FLUXNET",
    ):
        table.loc[repair_mask, column] = aligned.loc[
            repair_mask,
            f"_repair_{column}",
        ].to_numpy()
    if "GPP_source" not in table:
        table["GPP_source"] = pd.Series(
            pd.NA,
            index=table.index,
            dtype="object",
        )
    if "GPP_merge_status" not in table:
        table["GPP_merge_status"] = pd.Series(
            pd.NA,
            index=table.index,
            dtype="object",
        )
    table.loc[repair_mask, "GPP_source"] = "HH_conservative_repair"
    table.loc[
        repair_mask,
        "GPP_merge_status",
    ] = "HH_CONSERVATIVE_GPP_REPAIR"
    table.loc[repair_mask, "repaired_GPP"] = True

    for site_id, result in gpp_results.items():
        site_repair = repair_mask & table["SITE_ID"].eq(site_id)
        if not site_repair.any():
            continue
        append_provenance(
            table,
            site_repair,
            result.report["raw_file_path"],
            (
                "monthly GPP repaired with "
                f"{result.report['selected_GPP_column']} and quality OK"
            ),
        )


def recalculate_export_flags(table: pd.DataFrame) -> None:
    """Recalculate prerequisite flags without relaxing the scientific rules."""
    def numeric_column(name: str) -> pd.Series:
        if name not in table:
            return pd.Series(np.nan, index=table.index, dtype=float)
        return pd.to_numeric(table[name], errors="coerce")

    lat = numeric_column("lat")
    lon = numeric_column("lon")
    lat_lon_ok = (
        lat.between(-90.0, 90.0) & lon.between(-180.0, 180.0)
    )
    radius_max = numeric_column("radius_m_max")
    radius = numeric_column("radius_m")
    radius_ok = radius_max.gt(0.0) | radius.gt(0.0)
    gpp = numeric_column("tower_GPP_monthly")
    gpp_ok = gpp.notna() & gpp.ne(MISSING_VALUE)
    wd_center = numeric_column("WD_bin_center_deg")
    wd_fraction = numeric_column("WD_bin_fraction")
    wd_records = numeric_column("n_valid_monthly_30min_WD_records")
    wd_ok = (
        wd_center.notna()
        & wd_fraction.notna()
        & wd_fraction.ge(0.0)
        & wd_records.ge(MIN_VALID_MONTHLY_WD_RECORDS)
    )
    table["has_GPP"] = gpp_ok.astype(int)
    table["has_radius"] = radius_ok.astype(int)
    table["has_lat_lon"] = lat_lon_ok.astype(int)
    table["valid_for_GEE_export"] = (
        lat_lon_ok & radius_ok & gpp_ok & wd_ok
    ).astype(int)


# =============================================================================
# REPORT BUILDING AND REPAIRED AUDIT
# =============================================================================


def write_candidate_reports(
    gpp_results: dict[str, GPPRepairResult],
    metadata_results: dict[str, MetadataRepairResult],
    be_lon_record: dict[str, Any],
) -> dict[str, Path]:
    """Write all requested repair-candidate reports."""
    REPAIR_QC_OUTPUT.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    gpp_path = REPAIR_QC_OUTPUT / "_QC_GPP_repair_candidates.csv"
    pd.DataFrame(
        [gpp_results[site].report for site in GPP_ONLY_SITES],
        columns=GPP_REPORT_COLUMNS,
    ).to_csv(gpp_path, index=False, encoding="utf-8-sig")
    paths["gpp"] = gpp_path

    radius_path = REPAIR_QC_OUTPUT / "_QC_radius_repair_candidates.csv"
    pd.DataFrame(
        [
            metadata_result_to_radius_report(metadata_results[site])
            for site in RADIUS_ONLY_SITES
        ],
        columns=RADIUS_REPORT_COLUMNS,
    ).to_csv(radius_path, index=False, encoding="utf-8-sig")
    paths["radius"] = radius_path

    latlon_path = (
        REPAIR_QC_OUTPUT / "_QC_latlon_radius_repair_candidates.csv"
    )
    pd.DataFrame(
        [
            metadata_result_to_latlon_radius_report(metadata_results[site])
            for site in LATLON_RADIUS_SITES
        ],
        columns=LATLON_RADIUS_REPORT_COLUMNS,
    ).to_csv(latlon_path, index=False, encoding="utf-8-sig")
    paths["latlon_radius"] = latlon_path

    combined_rows: list[dict[str, Any]] = []
    for site in RADIUS_GPP_SITES:
        metadata_record = metadata_result_to_latlon_radius_report(
            metadata_results[site]
        )
        gpp_record = gpp_results[site].report
        combined_rows.append(
            {
                "SITE_ID": site,
                "raw_file_path": gpp_record["raw_file_path"],
                "selected_GPP_column": gpp_record["selected_GPP_column"],
                "n_months_with_valid_GPP": gpp_record[
                    "n_months_with_valid_GPP"
                ],
                "GPP_repair_status": gpp_record["repair_status"],
                "metadata_file_used": metadata_record["metadata_file_used"],
                "lat": metadata_record["lat"],
                "lon": metadata_record["lon"],
                "radius_m": metadata_record["radius_m"],
                "radius_m_min": metadata_record["radius_m_min"],
                "radius_m_max": metadata_record["radius_m_max"],
                "metadata_repair_status": metadata_record["repair_status"],
                "overall_repair_ready": bool(
                    gpp_record["n_months_with_valid_GPP"] > 0
                    and positive_number(metadata_record["radius_m_max"])
                    is not None
                ),
                "notes": (
                    gpp_record["notes"] + " | " + metadata_record["notes"]
                ),
            }
        )
    radius_gpp_path = (
        REPAIR_QC_OUTPUT / "_QC_radius_gpp_repair_candidates.csv"
    )
    pd.DataFrame(combined_rows).to_csv(
        radius_gpp_path,
        index=False,
        encoding="utf-8-sig",
    )
    paths["radius_gpp"] = radius_gpp_path

    full_rows: list[dict[str, Any]] = []
    for site in FULL_MISSING_SITES:
        metadata_record = metadata_result_to_latlon_radius_report(
            metadata_results[site]
        )
        gpp_record = gpp_results[site].report
        full_rows.append(
            {
                "SITE_ID": site,
                "raw_file_path": gpp_record["raw_file_path"],
                "selected_GPP_column": gpp_record["selected_GPP_column"],
                "n_months_with_valid_GPP": gpp_record[
                    "n_months_with_valid_GPP"
                ],
                "GPP_repair_status": gpp_record["repair_status"],
                "metadata_file_used": metadata_record["metadata_file_used"],
                "lat": metadata_record["lat"],
                "lon": metadata_record["lon"],
                "IGBP": metadata_record["IGBP"],
                "radius_m": metadata_record["radius_m"],
                "radius_m_max": metadata_record["radius_m_max"],
                "metadata_repair_status": metadata_record["repair_status"],
                "overall_repair_ready": bool(
                    gpp_record["n_months_with_valid_GPP"] > 0
                    and valid_lat_lon(
                        metadata_record["lat"],
                        metadata_record["lon"],
                    )
                    and positive_number(metadata_record["radius_m_max"])
                    is not None
                ),
                "notes": (
                    gpp_record["notes"] + " | " + metadata_record["notes"]
                ),
            }
        )
    full_path = REPAIR_QC_OUTPUT / "_QC_full_missing_repair_candidates.csv"
    pd.DataFrame(full_rows).to_csv(
        full_path,
        index=False,
        encoding="utf-8-sig",
    )
    paths["full_missing"] = full_path

    be_lon_path = REPAIR_QC_OUTPUT / "_QC_BE_Lon_repair_status.csv"
    pd.DataFrame(
        [be_lon_record],
        columns=BE_LON_REPORT_COLUMNS,
    ).to_csv(be_lon_path, index=False, encoding="utf-8-sig")
    paths["be_lon"] = be_lon_path
    return paths


def prepare_for_completeness_audit(table: pd.DataFrame) -> pd.DataFrame:
    """Add the normalized helper fields expected by the audit module."""
    prepared = table.copy()
    prepared["_SITE_ID_NORMALIZED"] = prepared["SITE_ID"].map(clean_site_id)
    prepared["_YYYYMM_NORMALIZED"] = prepared["YYYYMM"].map(normalize_yyyymm)
    prepared["_VALID_EXPORT_FLAG"] = prepared[
        "valid_for_GEE_export"
    ].map(completeness_audit.is_export_true)
    return prepared


def load_discovery_reports() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load or create ZIP and HH inventories for repaired completeness reports."""
    zip_path = EXISTING_AUDIT_QC / "all_detected_fluxnet_zip_files.csv"
    hh_path = EXISTING_AUDIT_QC / "all_detected_fluxnet_hh_files.csv"
    if zip_path.exists():
        zip_report = pd.read_csv(zip_path, encoding="utf-8-sig")
    else:
        zip_report = completeness_audit.detect_fluxnet_zip_files(PROJECT_ROOT)
    if hh_path.exists():
        hh_report = pd.read_csv(hh_path, encoding="utf-8-sig")
    else:
        hh_report = completeness_audit.detect_extracted_hh_files(
            [ORIGINAL_HH_FOLDER, REMAINING_HH_FOLDER]
        )
    return zip_report, hh_report


def write_repaired_audits(
    before_table: pd.DataFrame,
    repaired_table: pd.DataFrame,
) -> dict[str, Any]:
    """Run strict before/after audits using the repaired in-memory table."""
    REPAIR_QC_OUTPUT.mkdir(parents=True, exist_ok=True)
    before_prepared = prepare_for_completeness_audit(before_table)
    after_prepared = prepare_for_completeness_audit(repaired_table)
    before_month = completeness_audit.build_site_month_audit(
        before_prepared,
        EXISTING_TIFF_OUTPUT_ROOT,
    )
    after_month = completeness_audit.build_site_month_audit(
        after_prepared,
        EXISTING_TIFF_OUTPUT_ROOT,
    )

    zip_report, hh_report = load_discovery_reports()
    expected_real_zip_sites = {
        site
        for site in zip_report["SITE_ID"].fillna("").astype(str)
        if clean_site_id(site)
    }
    master = completeness_audit.build_master_site_audit(
        expected_sites=expected_real_zip_sites,
        zip_report=zip_report,
        hh_report=hh_report,
        final_table=after_prepared,
        site_month_audit=after_month,
        output_root=EXISTING_TIFF_OUTPUT_ROOT,
    )
    missing = master.loc[
        ~master["in_output_tiff_folders"]
    ].copy()
    repair_needed = completeness_audit.build_metadata_repair_report(
        master,
        after_month,
    )

    master_path = REPAIR_QC_OUTPUT / "repaired_site_completeness_audit.csv"
    month_path = (
        REPAIR_QC_OUTPUT / "repaired_site_month_exportability_audit.csv"
    )
    missing_path = (
        REPAIR_QC_OUTPUT
        / "repaired_missing_or_unexported_sites_diagnosis.csv"
    )
    repair_path = (
        REPAIR_QC_OUTPUT / "repaired_site_metadata_repair_needed.csv"
    )
    master.to_csv(master_path, index=False, encoding="utf-8-sig")
    after_month.to_csv(month_path, index=False, encoding="utf-8-sig")
    missing.to_csv(missing_path, index=False, encoding="utf-8-sig")
    repair_needed.to_csv(repair_path, index=False, encoding="utf-8-sig")

    before_exportable = before_month.loc[
        before_month["exportable_by_strict_rules"]
    ]
    after_exportable = after_month.loc[
        after_month["exportable_by_strict_rules"]
    ]
    before_sites = set(before_exportable["SITE_ID"].astype(str))
    after_sites = set(after_exportable["SITE_ID"].astype(str))
    newly_recovered = sorted(after_sites - before_sites)
    still_excluded = sorted(expected_real_zip_sites - after_sites)
    excluded_reasons = {
        str(row["SITE_ID"]): str(row["reason_site_not_exported"])
        for row in master.loc[
            master["SITE_ID"].astype(str).isin(still_excluded)
        ].to_dict("records")
    }
    return {
        "before_sites": len(before_sites),
        "before_months": int(len(before_exportable)),
        "after_sites": len(after_sites),
        "after_months": int(len(after_exportable)),
        "newly_recovered_sites": newly_recovered,
        "still_excluded_sites": still_excluded,
        "still_excluded_reasons": excluded_reasons,
        "report_paths": [
            master_path,
            month_path,
            missing_path,
            repair_path,
        ],
    }


def write_repaired_csvs(
    repaired_table: pd.DataFrame,
    original_columns: list[str],
    remaining_columns: list[str],
) -> list[Path]:
    """Write repaired copies only when DRY_RUN=False."""
    if DRY_RUN:
        return []
    REPAIRED_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    helper_columns = [
        "_source_group",
        "_SITE_ID_NORMALIZED",
        "_YYYYMM_NORMALIZED",
        "_VALID_EXPORT_FLAG",
    ]
    output = repaired_table.drop(columns=helper_columns, errors="ignore").copy()
    provenance = [
        "source_input_table",
        "repaired_lat_lon",
        "repaired_radius",
        "repaired_GPP",
        "repair_source",
        "repair_notes",
    ]

    merged_path = (
        REPAIRED_OUTPUT_ROOT
        / "repaired_ALL_SITES_monthly_WD_histogram_GPP_footprint_input_for_GEE.csv"
    )
    output.to_csv(merged_path, index=False, encoding="utf-8-sig")

    original_path = (
        REPAIRED_OUTPUT_ROOT / "repaired_original_sites_monthly_GEE_input.csv"
    )
    remaining_path = (
        REPAIRED_OUTPUT_ROOT / "repaired_remaining_sites_monthly_GEE_input.csv"
    )
    original_output = output.loc[
        repaired_table["_source_group"].eq("original")
    ].copy()
    remaining_output = output.loc[
        repaired_table["_source_group"].eq("remaining")
    ].copy()
    original_order = list(dict.fromkeys(original_columns + provenance))
    remaining_order = list(dict.fromkeys(remaining_columns + provenance))
    original_output = original_output.reindex(
        columns=original_order
        + [
            column
            for column in original_output.columns
            if column not in original_order
        ]
    )
    remaining_output = remaining_output.reindex(
        columns=remaining_order
        + [
            column
            for column in remaining_output.columns
            if column not in remaining_order
        ]
    )
    original_output.to_csv(original_path, index=False, encoding="utf-8-sig")
    remaining_output.to_csv(remaining_path, index=False, encoding="utf-8-sig")
    return [merged_path, original_path, remaining_path]


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Run repair discovery and an in-memory before/after audit."""
    REPAIR_QC_OUTPUT.mkdir(parents=True, exist_ok=True)
    print("========================================", flush=True)
    print("FLUXNET FINAL MONTHLY INPUT REPAIR", flush=True)
    print("========================================", flush=True)
    print(f"DRY_RUN={DRY_RUN}", flush=True)
    print(f"Repair output root: {REPAIRED_OUTPUT_ROOT}", flush=True)
    if DRY_RUN:
        print(
            "Dry-run safety: no ZIP extraction, repaired final CSV writing, "
            "or TIFF creation will occur.",
            flush=True,
        )

    print("\nLoading original final input tables...", flush=True)
    original_merged, original_columns, remaining_columns = (
        load_final_input_tables()
    )
    repaired = original_merged.copy()

    print("\nInspecting raw GPP candidates...", flush=True)
    raw_inventory = load_raw_file_inventory()
    gpp_results: dict[str, GPPRepairResult] = {}
    for index, site_id in enumerate(ALL_GPP_REPAIR_SITES, start=1):
        print(
            f"  GPP {index}/{len(ALL_GPP_REPAIR_SITES)}: {site_id}",
            flush=True,
        )
        raw_path = raw_inventory.get(site_id)
        if raw_path is None:
            report = {
                "SITE_ID": site_id,
                "raw_file_path": "",
                "candidate_GPP_columns": "",
                "selected_GPP_column": "",
                "n_months_total": 0,
                "n_months_with_valid_GPP": 0,
                "first_valid_month": "",
                "last_valid_month": "",
                "repair_status": "RAW_HH_HR_FILE_NOT_FOUND",
                "notes": "No extracted raw HH/HR file was detected.",
            }
            gpp_results[site_id] = GPPRepairResult(
                site_id,
                pd.DataFrame(),
                report,
            )
            continue
        gpp_results[site_id] = aggregate_monthly_gpp_from_raw(
            site_id,
            raw_path,
        )

    print("\nInspecting trusted metadata/radius candidates...", flush=True)
    metadata_results = build_metadata_registry(ALL_METADATA_REPAIR_SITES)

    print("\nInspecting BE-Lon ZIP...", flush=True)
    be_inspection = inspect_be_lon_zip(BE_LON_ZIP)
    extracted_be_path = safely_extract_be_lon_member(
        BE_LON_ZIP,
        be_inspection,
    )
    be_selected_gpp = ""
    if extracted_be_path is not None:
        be_rows, be_gpp = build_be_lon_final_rows(
            extracted_be_path,
            metadata_results[BE_LON_SITE],
        )
        be_selected_gpp = str(
            be_gpp.report.get("selected_GPP_column", "")
        )
        if not be_rows.empty:
            repaired = pd.concat(
                [repaired, be_rows],
                ignore_index=True,
                sort=False,
            )
    be_record = be_lon_report_record(
        be_inspection,
        extracted_be_path,
        selected_gpp=be_selected_gpp,
    )

    print("\nApplying trusted candidates to an in-memory copy...", flush=True)
    apply_metadata_repairs(repaired, metadata_results)
    apply_gpp_repairs(repaired, gpp_results)
    recalculate_export_flags(repaired)

    print("Writing repair candidate reports...", flush=True)
    candidate_paths = write_candidate_reports(
        gpp_results,
        metadata_results,
        be_record,
    )

    print("Running before/after strict exportability audit...", flush=True)
    audit_summary = write_repaired_audits(original_merged, repaired)
    repaired_csv_paths = write_repaired_csvs(
        repaired,
        original_columns,
        remaining_columns,
    )

    print("\n========================================", flush=True)
    print("REPAIR DRY RUN COMPLETE" if DRY_RUN else "REPAIR OUTPUT COMPLETE", flush=True)
    print("========================================", flush=True)
    print("Before:", flush=True)
    print(
        f"  exportable sites = {audit_summary['before_sites']:,}",
        flush=True,
    )
    print(
        f"  exportable months = {audit_summary['before_months']:,}",
        flush=True,
    )
    print("After in-memory trusted repairs:", flush=True)
    print(
        f"  exportable sites = {audit_summary['after_sites']:,}",
        flush=True,
    )
    print(
        f"  exportable months = {audit_summary['after_months']:,}",
        flush=True,
    )
    print(
        "  newly recovered sites = "
        + (
            ", ".join(audit_summary["newly_recovered_sites"])
            or "<none>"
        ),
        flush=True,
    )
    print("  still excluded sites and reasons:", flush=True)
    for site_id in audit_summary["still_excluded_sites"]:
        reason = audit_summary["still_excluded_reasons"].get(
            site_id,
            "No strictly exportable month after trusted repairs.",
        )
        print(f"    {site_id}: {reason}", flush=True)

    print("\nRepair candidate reports:", flush=True)
    for path in candidate_paths.values():
        print(f"  {path}", flush=True)
    print("Repaired audit reports:", flush=True)
    for path in audit_summary["report_paths"]:
        print(f"  {path}", flush=True)
    if DRY_RUN:
        print(
            "\nNo repaired final CSVs were written because DRY_RUN=True.",
            flush=True,
        )
        print(
            "Set DRY_RUN=False only after reviewing every candidate report.",
            flush=True,
        )
    else:
        print("Repaired final CSV copies:", flush=True)
        for path in repaired_csv_paths:
            print(f"  {path}", flush=True)


if __name__ == "__main__":
    main()
