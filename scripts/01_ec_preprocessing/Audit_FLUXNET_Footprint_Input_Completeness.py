# Install required packages, if needed:
# pip install pandas numpy openpyxl xlrd

"""
Audit the complete upstream chain for monthly FLUXNET footprint GeoTIFFs.

This script is read-only with respect to source data and existing TIFFs. It
does not extract ZIP files, repair metadata, alter valid/export flags, or
create/overwrite any GeoTIFF. It writes diagnostic CSV reports to OUTPUT_ROOT
/ "_QC".
"""

from __future__ import annotations
# Original script filename (sanitized): Audit_FLUXNET_Footprint_Input_Completeness.py; public name: Audit_FLUXNET_Footprint_Input_Completeness.py
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
from datetime import date, datetime
from numbers import Number
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


# =============================================================================
# EDITABLE PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(
    PROJECT_ROOT
)

ORIGINAL_HH_FOLDER = Path(
    _os.path.join(PROJECT_ROOT, 'fluxnet', 'Hourly Data (for Wind Direction)')
)

REMAINING_HH_FOLDER = Path(
    _os.path.join(PROJECT_ROOT, 'Remaining Fluxnet Sites', 'HH files')
)

REMAINING_ZIP_FOLDER = Path(
    _os.path.join(PROJECT_ROOT, 'Remaining Fluxnet Sites')
)

INPUT_TABLE_PATHS = [
    _os.path.join(PROJECT_ROOT, 'fluxnet', 'Hourly Data (for Wind Direction)', 'final_monthly_GEE_inputs', 'ALL_SITES_monthly_WD_histogram_GPP_footprint_input_for_GEE.csv'),
    _os.path.join(PROJECT_ROOT, 'Remaining Fluxnet Sites', 'HH files', 'final_monthly_GEE_inputs', 'ALL_SITES_monthly_WD_histogram_GPP_footprint_input_for_GEE.csv'),
]

OUTPUT_ROOT = Path(
    _os.path.join(PROJECT_ROOT, 'Monthly_EC_Footprint_TIFFs_Python')
)

QC_OUTPUT = OUTPUT_ROOT / "_QC"

# Large files with metadata-like names are still listed, but are not loaded as
# metadata workbooks/tables. Final input tables are loaded separately.
MAX_METADATA_READ_SIZE_MB = 100.0

# False performs a fresh recursive discovery. True reuses existing discovery
# CSVs in QC_OUTPUT when they are present, while rebuilding all master reports.
REUSE_EXISTING_DISCOVERY_REPORTS = False


# =============================================================================
# CONSTANTS
# =============================================================================

TRUE_STRINGS = {"1", "true", "yes"}
MISSING_GPP_SENTINEL = -9999.0
TIFF_FOLDER_PREFIX = "FLUXNET_monthly_histogram_footprint_GPP_TIFFs_"
TIFF_FILENAME_MIDDLE = "_monthly_30min_WD_histogram_footprint_GPP_"

FINAL_REQUIRED_COLUMNS = [
    "SITE_ID",
    "YYYYMM",
    "lat",
    "lon",
    "WD_bin_width_deg",
    "WD_bin_center_deg",
    "WD_bin_fraction",
    "tower_GPP_monthly",
    "GPP_monthly_quality_flag",
    "valid_for_GEE_export",
]

MASTER_SITE_COLUMNS = [
    "SITE_ID",
    "in_zip_files",
    "zip_file_paths",
    "in_extracted_HH_files",
    "HH_file_paths",
    "in_final_input_csv",
    "source_input_table_list",
    "in_output_tiff_folders",
    "output_folder_path",
    "n_rows_in_final_csv",
    "n_site_months_in_final_csv",
    "n_valid_rows",
    "n_valid_site_months",
    "n_tiffs_created",
    "first_YYYYMM_final_csv",
    "last_YYYYMM_final_csv",
    "first_YYYYMM_tiff",
    "last_YYYYMM_tiff",
    "has_any_lat_lon",
    "has_any_radius",
    "has_any_GPP",
    "has_any_valid_for_GEE_export_1",
    "all_months_invalid",
    "reason_site_not_exported",
    "recommended_action",
]

SITE_MONTH_COLUMNS = [
    "SITE_ID",
    "YYYYMM",
    "source_input_table",
    "n_WD_bins",
    "wd_fraction_sum",
    "has_lat_lon",
    "has_radius",
    "has_GPP",
    "valid_for_GEE_export_raw",
    "valid_for_GEE_export_interpreted",
    "radius_m",
    "radius_m_max",
    "tower_GPP_monthly",
    "GPP_monthly_quality_flag",
    "exportable_by_strict_rules",
    "expected_tiff_path",
    "tiff_exists",
    "export_blocking_reason",
]

SITE_ID_PATTERNS = [
    re.compile(
        r"(?:FLX|AMF|CNF|EUF|ICOS|JPF|TERN)_"
        r"(?P<site>[A-Z]{2}-[A-Za-z0-9]+)_FLUXNET",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?<![A-Za-z0-9])(?P<site>[A-Z]{2}-[A-Za-z0-9]{2,})(?![A-Za-z0-9])"
    ),
]


# =============================================================================
# FILESYSTEM AND SCALAR HELPERS
# =============================================================================


def warn(message: str) -> None:
    """Print a visible audit warning."""
    print(f"WARNING: {message}", flush=True)


def native_filesystem_path(path: Path) -> str:
    """Return an absolute path with Windows extended-length support."""
    absolute_path = os.path.abspath(os.fspath(path))
    if os.name != "nt" or absolute_path.startswith("\\\\?\\"):
        return absolute_path
    if absolute_path.startswith("\\\\"):
        return "\\\\?\\UNC\\" + absolute_path[2:]
    return "\\\\?\\" + absolute_path


def display_filesystem_path(path: str | Path) -> Path:
    """Remove a Windows extended-length prefix for readable report paths."""
    text = os.fspath(path)
    if text.startswith("\\\\?\\UNC\\"):
        text = "\\\\" + text[8:]
    elif text.startswith("\\\\?\\"):
        text = text[4:]
    return Path(text)


def path_exists(path: Path) -> bool:
    """Check existence with Windows long-path support."""
    return os.path.exists(native_filesystem_path(path))


def iter_files_recursively(root: Path) -> Iterable[Path]:
    """Yield files recursively with Windows long-path support."""
    if not path_exists(root):
        return
    for directory, _, filenames in os.walk(native_filesystem_path(root)):
        for filename in filenames:
            yield display_filesystem_path(os.path.join(directory, filename))


def list_files_in_folder(folder: Path, suffix: str | None = None) -> list[Path]:
    """List immediate files in one folder with long-path support."""
    if not path_exists(folder):
        return []
    results: list[Path] = []
    with os.scandir(native_filesystem_path(folder)) as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            if suffix is not None and not entry.name.lower().endswith(suffix.lower()):
                continue
            results.append(display_filesystem_path(entry.path))
    return results


def is_missing(value: Any) -> bool:
    """Return True for null scalars and blank strings."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if np.isscalar(missing) else False


def is_export_true(value: Any) -> bool:
    """Interpret the accepted valid_for_GEE_export representations."""
    if is_missing(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, Number):
        try:
            return float(value) == 1.0
        except (TypeError, ValueError, OverflowError):
            return False
    return str(value).strip().lower() in TRUE_STRINGS


def normalize_yyyymm(value: Any) -> str | None:
    """Normalize CSV/Excel YYYYMM values to validated six-digit strings."""
    if is_missing(value):
        return None
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.strftime("%Y%m")
    if isinstance(value, Number) and not isinstance(value, (bool, np.bool_)):
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            return None
        text = str(int(numeric))
    else:
        text = str(value).strip()
        match = re.fullmatch(r"(\d{6})\.0+", text)
        if match:
            text = match.group(1)
    if not re.fullmatch(r"\d{6}", text):
        return None
    month = int(text[4:])
    return text if 1 <= month <= 12 and int(text[:4]) >= 1 else None


def clean_site_id(value: Any) -> str | None:
    """Return a canonical-looking SITE_ID or None."""
    if is_missing(value):
        return None
    text = str(value).strip()
    match = re.fullmatch(r"[A-Z]{2}-[A-Za-z0-9]+", text)
    return text if match else None


def extract_site_id(text: str) -> tuple[str | None, str]:
    """Extract a FLUXNET SITE_ID from a filename, path, or cell value."""
    for pattern_number, pattern in enumerate(SITE_ID_PATTERNS, start=1):
        match = pattern.search(text)
        if match:
            raw_site = match.group("site")
            site = raw_site[:2].upper() + "-" + raw_site[3:]
            return site, f"filename_pattern_{pattern_number}"
    return None, ""


def safe_site_id(site_id: str) -> str:
    """Convert a FLUXNET SITE_ID to the output folder/file convention."""
    return site_id.replace("-", "_")


def unique_join(values: Iterable[Any]) -> str:
    """Join unique non-missing values into a deterministic report cell."""
    unique_values = sorted(
        {str(value).strip() for value in values if not is_missing(value)}
    )
    return " | ".join(unique_values)


def numeric_values(group: pd.DataFrame, column: str) -> np.ndarray:
    """Return numeric values, preserving NaNs for row-level pairing."""
    if column not in group.columns:
        return np.full(len(group), np.nan, dtype="float64")
    return pd.to_numeric(group[column], errors="coerce").to_numpy(dtype="float64")


def first_finite(group: pd.DataFrame, column: str) -> float | None:
    """Return the first finite value from a group column."""
    values = numeric_values(group, column)
    positions = np.flatnonzero(np.isfinite(values))
    return float(values[positions[0]]) if positions.size else None


def has_valid_lat_lon(group: pd.DataFrame) -> bool:
    """Check for at least one valid latitude/longitude pair."""
    latitudes = numeric_values(group, "lat")
    longitudes = numeric_values(group, "lon")
    valid = (
        np.isfinite(latitudes)
        & np.isfinite(longitudes)
        & (latitudes >= -90.0)
        & (latitudes <= 90.0)
        & (longitudes >= -180.0)
        & (longitudes <= 180.0)
    )
    return bool(np.any(valid))


def has_positive_radius(group: pd.DataFrame) -> bool:
    """Check radius_m_max first and radius_m second for any positive value."""
    for column in ("radius_m_max", "radius_m"):
        values = numeric_values(group, column)
        if bool(np.any(np.isfinite(values) & (values > 0.0))):
            return True
    return False


def resolve_radius(group: pd.DataFrame) -> float | None:
    """Resolve the first trusted positive radius from existing final-table data."""
    for column in ("radius_m_max", "radius_m"):
        values = numeric_values(group, column)
        positions = np.flatnonzero(np.isfinite(values) & (values > 0.0))
        if positions.size:
            return float(values[positions[0]])
    return None


def has_valid_gpp(group: pd.DataFrame) -> bool:
    """Check for at least one finite, non-sentinel tower GPP value."""
    values = numeric_values(group, "tower_GPP_monthly")
    valid = (
        np.isfinite(values)
        & (values != MISSING_GPP_SENTINEL)
        & (np.abs(values) <= np.finfo(np.float32).max)
    )
    return bool(np.any(valid))


def first_valid_gpp(group: pd.DataFrame) -> float | None:
    """Return the first finite, non-sentinel tower GPP value."""
    values = numeric_values(group, "tower_GPP_monthly")
    valid = (
        np.isfinite(values)
        & (values != MISSING_GPP_SENTINEL)
        & (np.abs(values) <= np.finfo(np.float32).max)
    )
    positions = np.flatnonzero(valid)
    return float(values[positions[0]]) if positions.size else None


def usable_wd_bin_mask(group: pd.DataFrame) -> np.ndarray:
    """Mirror the TIFF creator's minimum criteria for usable WD-bin rows."""
    centers = numeric_values(group, "WD_bin_center_deg")
    widths = numeric_values(group, "WD_bin_width_deg")
    fractions = numeric_values(group, "WD_bin_fraction")
    return (
        np.isfinite(centers)
        & np.isfinite(widths)
        & np.isfinite(fractions)
        & (fractions >= 0.0)
    )


# =============================================================================
# SOURCE FILE DISCOVERY
# =============================================================================


def detect_fluxnet_zip_files(project_root: Path) -> pd.DataFrame:
    """Find all FLUXNET-like ZIP files and extract SITE_IDs."""
    rows: list[dict[str, Any]] = []
    for path in iter_files_recursively(project_root):
        if path.suffix.lower() != ".zip":
            continue
        site_id, method = extract_site_id(path.name)
        if method == "filename_pattern_2" and "fluxnet" not in path.name.lower():
            # Avoid generic bundle names such as FLX_AA-Flx_BIF_ALL_*.zip,
            # where "AA-Flx" is not a tower SITE_ID.
            continue
        if site_id is None and "fluxnet" not in path.name.lower():
            continue
        rows.append(
            {
                "SITE_ID": site_id or "",
                "zip_file_path": str(path),
                "zip_filename": path.name,
                "file_size_bytes": os.path.getsize(native_filesystem_path(path)),
                "site_id_detection_method": method or "not_detected",
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "SITE_ID",
            "zip_file_path",
            "zip_filename",
            "file_size_bytes",
            "site_id_detection_method",
        ],
    )


def looks_like_raw_hh_file(path: Path) -> bool:
    """Distinguish raw FLUXNET HH/HR tables from derived monthly/QC CSVs."""
    name = path.name.lower()
    full_path = str(path).replace("\\", "/").lower()
    excluded_parts = (
        "/clean_30min_wd_outputs/",
        "/final_monthly_gee_inputs/",
        "/monthly_gee_csv_outputs",
        "/gpp_outputs",
        "/_qc/",
    )
    if any(part in full_path for part in excluded_parts):
        return False
    if "fluxnet" not in name:
        return False
    return bool(
        re.search(r"(?:_|-)(?:hh|hr)(?:_|-|\.)", name)
        or "fluxmet_hh" in name
        or "fluxmet_hr" in name
    )


def site_id_from_csv_contents(path: Path) -> tuple[str | None, str]:
    """Try to recover SITE_ID from the first rows of a CSV."""
    try:
        sample = read_csv_robust(path, nrows=10)
    except Exception:
        return None, ""
    candidate_columns = [
        column
        for column in sample.columns
        if re.sub(r"[^a-z0-9]", "", str(column).lower())
        in {"siteid", "sitecode", "fluxnetid", "site"}
    ]
    for column in candidate_columns:
        for value in sample[column]:
            site = clean_site_id(value)
            if site:
                return site, f"contents_column:{column}"
            site, _ = extract_site_id(str(value))
            if site:
                return site, f"contents_column:{column}"
    return None, ""


def detect_extracted_hh_files(roots: Iterable[Path]) -> pd.DataFrame:
    """Find raw half-hourly/hourly FLUXNET CSVs beneath both source roots."""
    rows: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for root in roots:
        if not path_exists(root):
            warn(f"HH source folder does not exist: {root}")
            continue
        for path in iter_files_recursively(root):
            if path.suffix.lower() != ".csv" or not looks_like_raw_hh_file(path):
                continue
            path_text = str(path)
            if path_text.lower() in seen_paths:
                continue
            seen_paths.add(path_text.lower())
            site_id, method = extract_site_id(path.name)
            if site_id is None:
                site_id, method = site_id_from_csv_contents(path)
            rows.append(
                {
                    "SITE_ID": site_id or "",
                    "HH_file_path": path_text,
                    "HH_filename": path.name,
                    "source_root": str(root),
                    "file_size_bytes": os.path.getsize(
                        native_filesystem_path(path)
                    ),
                    "site_id_detection_method": method or "not_detected",
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "SITE_ID",
            "HH_file_path",
            "HH_filename",
            "source_root",
            "file_size_bytes",
            "site_id_detection_method",
        ],
    )


def is_metadata_candidate(path: Path) -> bool:
    """Check the requested filename tokens and supported table extensions."""
    if path.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
        return False
    name = path.name.lower()
    return "site" in name or "metadata" in name or "sites information" in name


def read_csv_robust(
    path: Path,
    *,
    nrows: int | None = None,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    """Read a CSV with UTF-8/Windows fallback and long-path support."""
    native_path = native_filesystem_path(path)
    try:
        return pd.read_csv(
            native_path,
            low_memory=False,
            encoding="utf-8-sig",
            nrows=nrows,
            usecols=usecols,
        )
    except UnicodeDecodeError:
        return pd.read_csv(
            native_path,
            low_memory=False,
            encoding="cp1252",
            nrows=nrows,
            usecols=usecols,
        )


def metadata_site_column(columns: Iterable[Any]) -> str | None:
    """Choose a plausible site-ID column without guessing from site names."""
    normalized_targets = {"siteid", "sitecode", "fluxnetid", "site"}
    for column in columns:
        normalized = re.sub(r"[^a-z0-9]", "", str(column).lower())
        if normalized in normalized_targets:
            return str(column)
    return None


def extract_site_ids_from_metadata(path: Path) -> tuple[set[str], str, str]:
    """Read a plausible site-ID column from a metadata CSV/Excel file."""
    size_mb = os.path.getsize(native_filesystem_path(path)) / (1024.0**2)
    if size_mb > MAX_METADATA_READ_SIZE_MB:
        return set(), "", f"not_read_size_exceeds_{MAX_METADATA_READ_SIZE_MB:g}_MB"

    try:
        if path.suffix.lower() == ".csv":
            header = read_csv_robust(path, nrows=0)
            site_column = metadata_site_column(header.columns)
            if site_column is None:
                return set(), "", "readable_no_site_id_column"
            values = read_csv_robust(path, usecols=[site_column])[site_column]
            sheet_label = ""
        else:
            engine = "openpyxl" if path.suffix.lower() == ".xlsx" else "xlrd"
            excel_file = pd.ExcelFile(native_filesystem_path(path), engine=engine)
            values = None
            site_column = None
            sheet_label = ""
            for sheet_name in excel_file.sheet_names:
                header = pd.read_excel(
                    excel_file,
                    sheet_name=sheet_name,
                    nrows=0,
                )
                candidate = metadata_site_column(header.columns)
                if candidate is None:
                    continue
                values = pd.read_excel(
                    excel_file,
                    sheet_name=sheet_name,
                    usecols=[candidate],
                )[candidate]
                site_column = candidate
                sheet_label = str(sheet_name)
                break
            if values is None or site_column is None:
                return set(), "", "readable_no_site_id_column"

        detected: set[str] = set()
        for value in values:
            site = clean_site_id(value)
            if site:
                detected.add(site)
                continue
            site, _ = extract_site_id(str(value))
            if site:
                detected.add(site)
        location = site_column if not sheet_label else f"{sheet_label}:{site_column}"
        return detected, location, "read_success"
    except Exception as error:
        return set(), "", f"{type(error).__name__}: {error}"


def detect_metadata_files(project_root: Path, qc_output: Path) -> pd.DataFrame:
    """Find requested metadata-like files and extract readable site IDs."""
    rows: list[dict[str, Any]] = []
    qc_text = os.path.normcase(os.path.abspath(os.fspath(qc_output)))
    input_path_set = {
        os.path.normcase(os.path.abspath(os.fspath(Path(path))))
        for path in INPUT_TABLE_PATHS
    }

    for path in iter_files_recursively(project_root):
        absolute_text = os.path.normcase(os.path.abspath(os.fspath(path)))
        if absolute_text.startswith(qc_text + os.sep):
            continue
        if not is_metadata_candidate(path):
            continue

        if absolute_text in input_path_set:
            site_ids: set[str] = set()
            site_column = "SITE_ID"
            read_status = "final_input_loaded_separately"
        else:
            site_ids, site_column, read_status = extract_site_ids_from_metadata(path)

        rows.append(
            {
                "metadata_file_path": str(path),
                "metadata_filename": path.name,
                "extension": path.suffix.lower(),
                "file_size_bytes": os.path.getsize(native_filesystem_path(path)),
                "read_status": read_status,
                "site_id_column_or_sheet": site_column,
                "n_site_ids_detected": len(site_ids),
                "detected_site_ids": " | ".join(sorted(site_ids)),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "metadata_file_path",
            "metadata_filename",
            "extension",
            "file_size_bytes",
            "read_status",
            "site_id_column_or_sheet",
            "n_site_ids_detected",
            "detected_site_ids",
        ],
    )


# =============================================================================
# FINAL INPUT TABLE AND TIFF AUDIT
# =============================================================================


def load_table(path: Path) -> pd.DataFrame:
    """Load a final CSV/XLSX/XLS input table."""
    if not path_exists(path):
        raise FileNotFoundError(f"Final input table not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        table = read_csv_robust(path)
    elif suffix in {".xlsx", ".xls"}:
        engine = "openpyxl" if suffix == ".xlsx" else "xlrd"
        table = pd.read_excel(native_filesystem_path(path), engine=engine)
    else:
        raise ValueError(f"Unsupported final input format: {path.suffix}")
    table.columns = [str(column).strip().lstrip("\ufeff") for column in table.columns]
    missing = [column for column in FINAL_REQUIRED_COLUMNS if column not in table]
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {', '.join(missing)}"
        )
    return table


def load_merge_and_deduplicate_final_tables(
    paths: Iterable[str | Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load final tables, remove within-source duplicates, and flag cross-file ones."""
    tables: list[pd.DataFrame] = []
    input_summary_rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        table = load_table(path)
        source_path = str(path.resolve())
        table["source_input_table"] = source_path
        normalized_sites = table["SITE_ID"].map(clean_site_id)
        normalized_months = table["YYYYMM"].map(normalize_yyyymm)
        valid_keys = normalized_sites.notna() & normalized_months.notna()
        input_summary_rows.append(
            {
                "source_input_table": source_path,
                "n_rows": len(table),
                "n_sites": int(normalized_sites.dropna().nunique()),
                "n_site_months": int(
                    pd.DataFrame(
                        {
                            "SITE_ID": normalized_sites.loc[valid_keys],
                            "YYYYMM": normalized_months.loc[valid_keys],
                        }
                    )
                    .drop_duplicates()
                    .shape[0]
                ),
            }
        )
        tables.append(table)

    merged = pd.concat(tables, ignore_index=True, sort=False)
    within_source_key = [
        "SITE_ID",
        "YYYYMM",
        "WD_bin_center_deg",
        "source_input_table",
    ]
    exact_duplicate_mask = merged.duplicated(
        subset=within_source_key,
        keep="first",
    )
    removed_exact = merged.loc[exact_duplicate_mask].copy()
    deduplicated = merged.loc[~exact_duplicate_mask].copy().reset_index(drop=True)

    cross_file_key = ["SITE_ID", "YYYYMM", "WD_bin_center_deg"]
    cross_duplicate_mask = deduplicated.duplicated(
        subset=cross_file_key,
        keep=False,
    )
    cross_file_duplicates = deduplicated.loc[cross_duplicate_mask].copy()
    if not cross_file_duplicates.empty:
        cross_file_duplicates = cross_file_duplicates.sort_values(
            cross_file_key + ["source_input_table"]
        )

    deduplicated["_SITE_ID_NORMALIZED"] = deduplicated["SITE_ID"].map(clean_site_id)
    deduplicated["_YYYYMM_NORMALIZED"] = deduplicated["YYYYMM"].map(
        normalize_yyyymm
    )
    deduplicated["_VALID_EXPORT_FLAG"] = deduplicated[
        "valid_for_GEE_export"
    ].map(is_export_true)

    input_summary = pd.DataFrame(input_summary_rows)
    return deduplicated, removed_exact, cross_file_duplicates, input_summary


def expected_tiff_path(site_id: str, yyyymm: str, output_root: Path) -> Path:
    """Return the established site-folder/monthly-TIFF path."""
    safe_id = safe_site_id(site_id)
    return (
        output_root
        / f"{TIFF_FOLDER_PREFIX}{safe_id}"
        / f"{safe_id}{TIFF_FILENAME_MIDDLE}{yyyymm}.tif"
    )


def tiff_inventory(site_id: str, output_root: Path) -> dict[str, Any]:
    """Inventory one site's existing output folder without opening TIFF data."""
    folder = output_root / f"{TIFF_FOLDER_PREFIX}{safe_site_id(site_id)}"
    tiffs = list_files_in_folder(folder, ".tif")
    months: list[str] = []
    for path in tiffs:
        match = re.search(r"_(\d{6})\.tif$", path.name, re.IGNORECASE)
        if match and normalize_yyyymm(match.group(1)):
            months.append(match.group(1))
    months.sort()
    return {
        "in_output_tiff_folders": path_exists(folder) and bool(tiffs),
        "output_folder_path": str(folder),
        "n_tiffs_created": len(tiffs),
        "first_YYYYMM_tiff": months[0] if months else "",
        "last_YYYYMM_tiff": months[-1] if months else "",
    }


def format_raw_values(series: pd.Series) -> str:
    """Format unique raw values for a compact audit cell."""
    return unique_join(series)


def build_site_month_audit(
    final_table: pd.DataFrame,
    output_root: Path,
) -> pd.DataFrame:
    """Build strict site-month exportability diagnostics."""
    valid_keys = final_table[
        final_table["_SITE_ID_NORMALIZED"].notna()
        & final_table["_YYYYMM_NORMALIZED"].notna()
    ]
    rows: list[dict[str, Any]] = []

    grouped = valid_keys.groupby(
        ["_SITE_ID_NORMALIZED", "_YYYYMM_NORMALIZED"],
        sort=True,
        dropna=False,
    )
    for (site_id, yyyymm), group in grouped:
        flag_mask = group["_VALID_EXPORT_FLAG"].to_numpy(dtype=bool)
        flag_is_true = bool(np.any(flag_mask))
        strict_group = group.loc[flag_mask] if flag_is_true else group

        lat_lon_ok = has_valid_lat_lon(strict_group)
        radius_ok = has_positive_radius(strict_group)
        gpp_ok = has_valid_gpp(strict_group)
        usable_bins = usable_wd_bin_mask(strict_group)
        n_usable_bins = int(np.count_nonzero(usable_bins))
        fractions = numeric_values(strict_group, "WD_bin_fraction")
        fraction_sum = float(
            math.fsum(fractions[np.isfinite(fractions) & (fractions >= 0.0)])
        )

        blockers: list[str] = []
        if not flag_is_true:
            blockers.append("valid_for_GEE_export is false/invalid for all rows")
        if not lat_lon_ok:
            blockers.append("missing or invalid latitude/longitude")
        if not radius_ok:
            blockers.append("missing or non-positive radius_m_max/radius_m")
        if not gpp_ok:
            blockers.append("missing or invalid tower_GPP_monthly")
        if n_usable_bins == 0:
            blockers.append("no usable WD-bin rows")

        exportable = not blockers
        tif_path = expected_tiff_path(str(site_id), str(yyyymm), output_root)
        rows.append(
            {
                "SITE_ID": str(site_id),
                "YYYYMM": str(yyyymm),
                "source_input_table": unique_join(group["source_input_table"]),
                "n_WD_bins": n_usable_bins,
                "wd_fraction_sum": fraction_sum,
                "has_lat_lon": lat_lon_ok,
                "has_radius": radius_ok,
                "has_GPP": gpp_ok,
                "valid_for_GEE_export_raw": format_raw_values(
                    group["valid_for_GEE_export"]
                ),
                "valid_for_GEE_export_interpreted": flag_is_true,
                "radius_m": first_finite(strict_group, "radius_m"),
                "radius_m_max": first_finite(strict_group, "radius_m_max"),
                "tower_GPP_monthly": first_valid_gpp(strict_group),
                "GPP_monthly_quality_flag": (
                    format_raw_values(group["GPP_monthly_quality_flag"])
                ),
                "exportable_by_strict_rules": exportable,
                "expected_tiff_path": str(tif_path),
                "tiff_exists": path_exists(tif_path),
                "export_blocking_reason": "; ".join(blockers),
            }
        )
    return pd.DataFrame(rows, columns=SITE_MONTH_COLUMNS)


def paths_by_site(
    table: pd.DataFrame,
    site_column: str,
    path_column: str,
) -> dict[str, list[str]]:
    """Build a SITE_ID -> path list mapping from a discovery report."""
    mapping: dict[str, list[str]] = {}
    if table.empty:
        return mapping
    for site_id, group in table.loc[table[site_column] != ""].groupby(site_column):
        mapping[str(site_id)] = sorted(set(group[path_column].astype(str)))
    return mapping


def metadata_site_ids(metadata_report: pd.DataFrame) -> set[str]:
    """Unpack detected metadata SITE_ID lists."""
    detected: set[str] = set()
    if metadata_report.empty:
        return detected
    for cell in metadata_report["detected_site_ids"]:
        if is_missing(cell):
            continue
        for value in str(cell).split("|"):
            site = clean_site_id(value.strip())
            if site:
                detected.add(site)
    return detected


def select_expected_metadata_sites(
    metadata_report: pd.DataFrame,
    anchor_sites: set[str],
    minimum_overlap_fraction: float = 0.50,
) -> tuple[set[str], pd.DataFrame]:
    """
    Select metadata lists relevant to this project cohort.

    Every metadata file remains in detected_metadata_files.csv. A metadata
    table may add sites to the expected cohort only when at least half of its
    detected SITE_IDs overlap sites independently anchored by ZIP, HH/HR, or
    final-input evidence. This prevents a broad global reference catalog from
    turning unrelated FLUXNET sites into false project omissions.
    """
    report = metadata_report.copy()
    selected_sites: set[str] = set()
    used_flags: list[bool] = []
    overlap_counts: list[int] = []
    overlap_fractions: list[float] = []

    for cell in report.get("detected_site_ids", pd.Series(dtype=object)):
        detected: set[str] = set()
        if not is_missing(cell):
            for value in str(cell).split("|"):
                site = clean_site_id(value.strip())
                if site:
                    detected.add(site)
        overlap_count = len(detected & anchor_sites)
        overlap_fraction = (
            overlap_count / len(detected) if detected else 0.0
        )
        used = bool(
            detected
            and overlap_count > 0
            and overlap_fraction >= minimum_overlap_fraction
        )
        if used:
            selected_sites.update(detected)
        used_flags.append(used)
        overlap_counts.append(overlap_count)
        overlap_fractions.append(overlap_fraction)

    report["anchor_site_overlap_count"] = overlap_counts
    report["anchor_site_overlap_fraction"] = overlap_fractions
    report["used_for_expected_site_set"] = used_flags
    return selected_sites, report


def site_not_exported_diagnosis(
    *,
    in_zip: bool,
    in_hh: bool,
    in_final: bool,
    n_exportable_months: int,
    n_tiffs: int,
    has_lat_lon: bool,
    has_radius: bool,
    has_gpp: bool,
    has_valid_flag: bool,
) -> tuple[str, str]:
    """Return a clear site-level reason and recommended repair action."""
    if n_tiffs > 0:
        if n_exportable_months > n_tiffs:
            return (
                "Some strictly exportable site-months do not have TIFFs.",
                "Run TIFF validation, inspect failed/skipped months, then rebuild "
                "only this site if the final input rows are correct.",
            )
        return "", "No site-level repair required."

    if in_zip and not in_hh:
        return (
            "ZIP exists but no extracted raw HH/HR CSV was detected.",
            "Extract the trusted FLUXNET ZIP and run the upstream HH/HR, WD, GPP, "
            "metadata, and final-monthly-table preparation steps.",
        )
    if in_hh and not in_final:
        return (
            "Extracted HH/HR file exists but the site is absent from final input CSVs.",
            "Run the upstream monthly WD histogram, GPP merge, and final CSV "
            "generation for this site; do not create TIFFs before that.",
        )
    if not in_final:
        return (
            "Site was detected upstream/metadata but is absent from final input CSVs.",
            "Locate or extract its raw FLUXNET data and regenerate its final monthly "
            "input rows.",
        )

    missing_fields: list[str] = []
    if not has_lat_lon:
        missing_fields.append("latitude/longitude")
    if not has_radius:
        missing_fields.append("trusted radius")
    if not has_gpp:
        missing_fields.append("tower GPP")
    if missing_fields:
        return (
            "Final CSV exists but no month is exportable; missing site data: "
            + ", ".join(missing_fields)
            + ".",
            "Repair the missing fields from trusted source metadata/data, regenerate "
            "valid_for_GEE_export, and rebuild the final monthly CSV. Do not invent "
            "radius or GPP values.",
        )
    if not has_valid_flag:
        return (
            "Final CSV exists, required fields occur somewhere, but every row has "
            "valid_for_GEE_export false/invalid.",
            "Audit the upstream validity rule per month and correct it only where all "
            "scientific prerequisites are genuinely satisfied.",
        )
    if n_exportable_months == 0:
        return (
            "Final CSV exists but every month fails at least one strict export rule.",
            "Use MASTER_site_month_exportability_audit.csv to repair the exact "
            "month-level blockers from trusted data.",
        )
    return (
        "Strictly exportable months exist but no TIFF output folder was found.",
        "Validate the TIFF creator settings/logs and rebuild this site.",
    )


def build_master_site_audit(
    expected_sites: set[str],
    zip_report: pd.DataFrame,
    hh_report: pd.DataFrame,
    final_table: pd.DataFrame,
    site_month_audit: pd.DataFrame,
    output_root: Path,
) -> pd.DataFrame:
    """Build the required master site-level completeness report."""
    zip_paths = paths_by_site(zip_report, "SITE_ID", "zip_file_path")
    hh_paths = paths_by_site(hh_report, "SITE_ID", "HH_file_path")
    final_valid = final_table.loc[final_table["_SITE_ID_NORMALIZED"].notna()]
    final_groups = {
        str(site_id): group
        for site_id, group in final_valid.groupby("_SITE_ID_NORMALIZED")
    }
    month_groups = {
        str(site_id): group
        for site_id, group in site_month_audit.groupby("SITE_ID")
    }

    rows: list[dict[str, Any]] = []
    for site_id in sorted(expected_sites):
        final_group = final_groups.get(site_id)
        month_group = month_groups.get(site_id)
        in_final = final_group is not None

        if in_final:
            months = sorted(
                final_group["_YYYYMM_NORMALIZED"].dropna().unique().tolist()
            )
            n_rows = int(len(final_group))
            n_site_months = int(len(months))
            n_valid_rows = int(final_group["_VALID_EXPORT_FLAG"].sum())
            valid_months = final_group.loc[
                final_group["_VALID_EXPORT_FLAG"],
                "_YYYYMM_NORMALIZED",
            ].dropna()
            n_valid_months = int(valid_months.nunique())
            lat_lon_ok = has_valid_lat_lon(final_group)
            radius_ok = has_positive_radius(final_group)
            gpp_ok = has_valid_gpp(final_group)
            has_valid_flag = bool(final_group["_VALID_EXPORT_FLAG"].any())
            sources = unique_join(final_group["source_input_table"])
        else:
            months = []
            n_rows = 0
            n_site_months = 0
            n_valid_rows = 0
            n_valid_months = 0
            lat_lon_ok = False
            radius_ok = False
            gpp_ok = False
            has_valid_flag = False
            sources = ""

        n_exportable_months = (
            int(month_group["exportable_by_strict_rules"].sum())
            if month_group is not None
            else 0
        )
        inventory = tiff_inventory(site_id, output_root)
        reason, action = site_not_exported_diagnosis(
            in_zip=site_id in zip_paths,
            in_hh=site_id in hh_paths,
            in_final=in_final,
            n_exportable_months=n_exportable_months,
            n_tiffs=int(inventory["n_tiffs_created"]),
            has_lat_lon=lat_lon_ok,
            has_radius=radius_ok,
            has_gpp=gpp_ok,
            has_valid_flag=has_valid_flag,
        )

        rows.append(
            {
                "SITE_ID": site_id,
                "in_zip_files": site_id in zip_paths,
                "zip_file_paths": " | ".join(zip_paths.get(site_id, [])),
                "in_extracted_HH_files": site_id in hh_paths,
                "HH_file_paths": " | ".join(hh_paths.get(site_id, [])),
                "in_final_input_csv": in_final,
                "source_input_table_list": sources,
                "in_output_tiff_folders": inventory[
                    "in_output_tiff_folders"
                ],
                "output_folder_path": inventory["output_folder_path"],
                "n_rows_in_final_csv": n_rows,
                "n_site_months_in_final_csv": n_site_months,
                "n_valid_rows": n_valid_rows,
                "n_valid_site_months": n_valid_months,
                "n_tiffs_created": inventory["n_tiffs_created"],
                "first_YYYYMM_final_csv": months[0] if months else "",
                "last_YYYYMM_final_csv": months[-1] if months else "",
                "first_YYYYMM_tiff": inventory["first_YYYYMM_tiff"],
                "last_YYYYMM_tiff": inventory["last_YYYYMM_tiff"],
                "has_any_lat_lon": lat_lon_ok,
                "has_any_radius": radius_ok,
                "has_any_GPP": gpp_ok,
                "has_any_valid_for_GEE_export_1": has_valid_flag,
                "all_months_invalid": bool(in_final and n_valid_months == 0),
                "reason_site_not_exported": reason,
                "recommended_action": action,
            }
        )
    return pd.DataFrame(rows, columns=MASTER_SITE_COLUMNS)


def build_metadata_repair_report(
    master_report: pd.DataFrame,
    site_month_report: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate site/month metadata blockers without manufacturing values."""
    rows: list[dict[str, Any]] = []
    month_groups = {
        str(site_id): group
        for site_id, group in site_month_report.groupby("SITE_ID")
    }
    for record in master_report.to_dict("records"):
        site_id = str(record["SITE_ID"])
        month_group = month_groups.get(site_id)
        if month_group is None:
            missing_lat_months = 0
            missing_radius_months = 0
            missing_gpp_months = 0
        else:
            missing_lat_months = int((~month_group["has_lat_lon"]).sum())
            missing_radius_months = int((~month_group["has_radius"]).sum())
            missing_gpp_months = int((~month_group["has_GPP"]).sum())

        missing_hh = bool(record["in_zip_files"] and not record["in_extracted_HH_files"])
        missing_final = bool(
            record["in_extracted_HH_files"] and not record["in_final_input_csv"]
        )
        needs_repair = bool(
            missing_hh
            or missing_final
            or missing_lat_months
            or missing_radius_months
            or missing_gpp_months
            or (
                record["in_final_input_csv"]
                and not record["has_any_valid_for_GEE_export_1"]
            )
        )
        if not needs_repair:
            continue

        rows.append(
            {
                "SITE_ID": site_id,
                "missing_HH_extraction": missing_hh,
                "missing_final_CSV_generation": missing_final,
                "n_months_missing_lat_lon": missing_lat_months,
                "n_months_missing_radius": missing_radius_months,
                "n_months_missing_GPP": missing_gpp_months,
                "all_months_invalid": record["all_months_invalid"],
                "zip_file_paths": record["zip_file_paths"],
                "HH_file_paths": record["HH_file_paths"],
                "reason_site_not_exported": record["reason_site_not_exported"],
                "recommended_action": record["recommended_action"],
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# MAIN AUDIT
# =============================================================================


def main() -> None:
    """Run the read-only completeness audit and write all QC reports."""
    QC_OUTPUT.mkdir(parents=True, exist_ok=True)
    print("Starting FLUXNET footprint input completeness audit.", flush=True)
    print(f"Project root: {PROJECT_ROOT}", flush=True)
    print(f"QC output: {QC_OUTPUT}", flush=True)

    zip_path = QC_OUTPUT / "all_detected_fluxnet_zip_files.csv"
    hh_path = QC_OUTPUT / "all_detected_fluxnet_hh_files.csv"
    metadata_path = QC_OUTPUT / "detected_metadata_files.csv"

    can_reuse_discovery = bool(
        REUSE_EXISTING_DISCOVERY_REPORTS
        and zip_path.exists()
        and hh_path.exists()
        and metadata_path.exists()
    )
    if can_reuse_discovery:
        print("\nReusing existing ZIP/HH/metadata discovery reports.", flush=True)
        zip_report = pd.read_csv(zip_path, encoding="utf-8-sig")
        hh_report = pd.read_csv(hh_path, encoding="utf-8-sig")
        metadata_report = pd.read_csv(metadata_path, encoding="utf-8-sig")
    else:
        print("\nSearching for FLUXNET ZIP files...", flush=True)
        zip_report = detect_fluxnet_zip_files(PROJECT_ROOT)
        zip_report.to_csv(zip_path, index=False, encoding="utf-8-sig")

        print("Searching for extracted FLUXNET HH/HR CSV files...", flush=True)
        hh_report = detect_extracted_hh_files(
            [ORIGINAL_HH_FOLDER, REMAINING_HH_FOLDER]
        )
        hh_report.to_csv(hh_path, index=False, encoding="utf-8-sig")

        print("Searching for metadata-like CSV/Excel files...", flush=True)
        metadata_report = detect_metadata_files(PROJECT_ROOT, QC_OUTPUT)

    print("Loading and auditing final monthly input tables...", flush=True)
    (
        final_table,
        removed_exact,
        cross_file_duplicates,
        input_summary,
    ) = load_merge_and_deduplicate_final_tables(INPUT_TABLE_PATHS)
    input_summary.to_csv(
        QC_OUTPUT / "final_input_table_load_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    removed_exact.to_csv(
        QC_OUTPUT / "audit_removed_exact_duplicate_rows.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cross_file_duplicates.to_csv(
        QC_OUTPUT / "audit_cross_file_duplicate_rows.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Building site-month strict exportability audit...", flush=True)
    site_month_report = build_site_month_audit(final_table, OUTPUT_ROOT)
    site_month_path = QC_OUTPUT / "MASTER_site_month_exportability_audit.csv"
    site_month_report.to_csv(
        site_month_path,
        index=False,
        encoding="utf-8-sig",
    )

    zip_sites = {
        site for site in zip_report["SITE_ID"].astype(str) if clean_site_id(site)
    }
    hh_sites = {
        site for site in hh_report["SITE_ID"].astype(str) if clean_site_id(site)
    }
    final_sites = set(
        final_table["_SITE_ID_NORMALIZED"].dropna().astype(str).unique()
    )
    anchor_sites = zip_sites | hh_sites | final_sites
    metadata_sites, metadata_report = select_expected_metadata_sites(
        metadata_report,
        anchor_sites,
    )
    metadata_report.to_csv(
        metadata_path,
        index=False,
        encoding="utf-8-sig",
    )
    expected_sites = zip_sites | hh_sites | final_sites | metadata_sites

    print("Building master site-level audit...", flush=True)
    master_report = build_master_site_audit(
        expected_sites=expected_sites,
        zip_report=zip_report,
        hh_report=hh_report,
        final_table=final_table,
        site_month_audit=site_month_report,
        output_root=OUTPUT_ROOT,
    )
    master_path = QC_OUTPUT / "MASTER_site_completeness_audit.csv"
    master_report.to_csv(master_path, index=False, encoding="utf-8-sig")

    missing_report = master_report.loc[
        ~master_report["in_output_tiff_folders"]
    ].copy()
    missing_path = QC_OUTPUT / "missing_or_unexported_sites_diagnosis.csv"
    missing_report.to_csv(missing_path, index=False, encoding="utf-8-sig")

    repair_report = build_metadata_repair_report(
        master_report,
        site_month_report,
    )
    repair_path = QC_OUTPUT / "site_metadata_repair_needed.csv"
    repair_report.to_csv(repair_path, index=False, encoding="utf-8-sig")

    exportable_sites = set(
        site_month_report.loc[
            site_month_report["exportable_by_strict_rules"],
            "SITE_ID",
        ].astype(str)
    )
    output_sites = set(
        master_report.loc[
            master_report["in_output_tiff_folders"],
            "SITE_ID",
        ].astype(str)
    )
    missing_final_sites = expected_sites - final_sites
    nonexportable = site_month_report.loc[
        ~site_month_report["exportable_by_strict_rules"]
    ]
    radius_blocked_sites = set(
        nonexportable.loc[~nonexportable["has_radius"], "SITE_ID"].astype(str)
    )
    gpp_blocked_sites = set(
        nonexportable.loc[~nonexportable["has_GPP"], "SITE_ID"].astype(str)
    )
    lat_lon_blocked_sites = set(
        nonexportable.loc[~nonexportable["has_lat_lon"], "SITE_ID"].astype(str)
    )
    flag_only_mask = (
        ~nonexportable["valid_for_GEE_export_interpreted"]
        & nonexportable["has_radius"]
        & nonexportable["has_GPP"]
        & nonexportable["has_lat_lon"]
        & (nonexportable["n_WD_bins"] > 0)
    )
    flag_only_sites = set(
        nonexportable.loc[flag_only_mask, "SITE_ID"].astype(str)
    )

    print("\n========================================", flush=True)
    print("FLUXNET INPUT COMPLETENESS AUDIT COMPLETE", flush=True)
    print("========================================", flush=True)
    print(f"Total upstream sites detected: {len(expected_sites):,}", flush=True)
    print(f"Sites in ZIP files: {len(zip_sites):,}", flush=True)
    print(f"Sites in extracted HH/HR files: {len(hh_sites):,}", flush=True)
    print(f"Sites in final CSVs: {len(final_sites):,}", flush=True)
    print(
        f"Sites in cohort-relevant metadata lists: {len(metadata_sites):,}",
        flush=True,
    )
    print(
        f"Sites with strictly exportable site-months: {len(exportable_sites):,}",
        flush=True,
    )
    print(f"Sites with TIFF output folders: {len(output_sites):,}", flush=True)
    print(
        f"Sites missing entirely from final CSV: {len(missing_final_sites):,}",
        flush=True,
    )
    print(
        f"Sites with months blocked by missing radius: "
        f"{len(radius_blocked_sites):,}",
        flush=True,
    )
    print(
        f"Sites with months blocked by missing GPP: {len(gpp_blocked_sites):,}",
        flush=True,
    )
    print(
        f"Sites with months blocked by missing lat/lon: "
        f"{len(lat_lon_blocked_sites):,}",
        flush=True,
    )
    print(
        "Sites with months blocked only by valid_for_GEE_export=false: "
        f"{len(flag_only_sites):,}",
        flush=True,
    )
    print(
        "Site IDs without output folders: "
        + (", ".join(sorted(expected_sites - output_sites)) or "<none>"),
        flush=True,
    )
    print("\nRequired reports:", flush=True)
    for report_path in (
        zip_path,
        hh_path,
        metadata_path,
        master_path,
        site_month_path,
        missing_path,
        repair_path,
    ):
        print(f"  {report_path}", flush=True)


if __name__ == "__main__":
    main()
