# Install required packages, if needed:
# pip install pandas numpy rasterio shapely pyproj openpyxl xlrd tqdm matplotlib

"""
Create one monthly FLUXNET footprint/GPP GeoTIFF per site-month.

The monthly footprint is reconstructed from the 30-minute wind-direction
histogram in the final input table. Each wind-direction bin becomes an upwind
wedge, the wedge is weighted by its monthly fraction, and the weighted wedges
are summed and capped at 1.

This script creates footprint/GPP TIFFs only. It does not extract SIF.
"""

from __future__ import annotations
# Original script filename (sanitized): Create_Monthly_FLUXNET_Footprint_TIFFs_From_Final_CSV.py; public name: 01_Create_Monthly_FLUXNET_Footprint_TIFFs_From_Final_CSV.py
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
import traceback
from dataclasses import dataclass
from datetime import date, datetime
from numbers import Number
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS, Transformer
from rasterio.enums import MergeAlg
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import Polygon, mapping

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle as MatplotlibCircle
except ImportError:
    plt = None
    MatplotlibCircle = None

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


# =============================================================================
# EDITABLE PATHS AND PROCESSING CONTROLS
# =============================================================================

INPUT_TABLE_PATHS = [
    _os.path.join(PROJECT_ROOT, 'finalized_metadata_recovery', 'final_metadata_recovered_ALL_SITES_monthly_WD_histogram_GPP_footprint_input_for_GEE.csv'),
]

OUTPUT_ROOT = Path(
    _os.path.join(PROJECT_ROOT, 'Monthly_EC_Footprint_TIFFs_Python')
)

# None processes all sites. Example: ["US-Wkg", "AU-How"]
SELECTED_SITES = None

# Inclusive YYYYMM filters. Use None for no lower/upper limit.
START_YYYYMM = None
END_YYYYMM = None

SKIP_EXISTING = True

# True creates QC/preflight reports but does not create any TIFFs.
DRY_RUN_ONLY = False

# True skips input loading/creation and validates the existing TIFF collection.
VALIDATE_ONLY = True

# True permits existing TIFFs to be overwritten with the improved circle method.
RECREATE_EXISTING = False

# None rebuilds all eligible sites. Example: ["US-Wkg", "AU-ASM"]
SELECTED_SITES_FOR_REBUILD = None

# Save footprint/circle visual checks for the first QA_PNG_MAX newly written TIFFs.
CREATE_QA_PNGS = True
QA_PNG_MAX = 50

# True processes only rows whose valid_for_GEE_export value is accepted as true.
REQUIRE_VALID_FOR_GEE_EXPORT = True

# Stop after this many site-months have been considered. None means no limit.
MAX_SITE_MONTHS = None

# Raster and geometry settings.
EXPORT_MARGIN_M = 300.0
PIXEL_SIZE_M = 30.0
MINIMUM_WEDGE_WIDTH_DEG = 10.0
WEDGE_ARC_SEGMENTS = 12


# =============================================================================
# CONSTANTS
# =============================================================================

REQUIRED_COLUMNS = [
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
    "WD_bin_fraction",
    "tower_GPP_monthly",
    "primary_GPP_column",
    "GPP_monthly_quality_flag",
    "valid_for_GEE_export",
]

BAND_DESCRIPTIONS = [
    "footprint_probability",
    "tower_GPP",
    "blind_circular_buffer",
    "n_WD_bins",
    "n_valid_30min_WD_records",
    "footprint_probability_x_GPP",
]

SUMMARY_COLUMNS = [
    "SITE_ID",
    "YYYYMM",
    "output_tif",
    "status",
    "n_bins",
    "wd_fraction_sum",
    "radius_m_used",
    "tower_GPP_monthly",
    "n_valid_30min_WD_records",
    "footprint_probability_min",
    "footprint_probability_max",
    "footprint_probability_mean",
    "blind_buffer_pixel_count",
    "pixel_size_m",
    "expected_circle_area_m2",
    "raster_circle_area_m2",
    "circle_area_ratio",
    "circle_pixel_count",
    "circle_bbox_width_pixels",
    "circle_bbox_height_pixels",
    "circle_bbox_width_height_ratio",
    "circle_geometry_status",
    "error_message",
]

SKIPPED_COLUMNS = [
    "SITE_ID",
    "YYYYMM",
    "reason",
]

INPUT_COVERAGE_COLUMNS = [
    "source_input_table",
    "SITE_ID",
    "n_rows_total",
    "n_rows_valid",
    "n_site_months_total",
    "n_site_months_valid",
    "first_YYYYMM",
    "last_YYYYMM",
    "has_any_radius",
    "has_any_lat_lon",
    "has_any_GPP",
    "n_valid_for_GEE_export_1",
    "n_valid_for_GEE_export_0",
]

MERGED_VALID_SITE_COLUMNS = [
    "SITE_ID",
    "source_input_table_list",
    "n_valid_rows",
    "n_valid_site_months",
    "first_YYYYMM",
    "last_YYYYMM",
    "IGBP",
    "site_name",
]

DEDUPLICATION_KEY = ["SITE_ID", "YYYYMM", "WD_bin_center_deg"]

TRUE_STRINGS = {"1", "true", "yes"}
MISSING_RECORD_COUNT = -9999.0
MISSING_GPP_SENTINEL = -9999.0
CIRCLE_AREA_RATIO_MIN = 0.85
CIRCLE_AREA_RATIO_MAX = 1.15
CIRCLE_BBOX_RATIO_MIN = 0.95
CIRCLE_BBOX_RATIO_MAX = 1.05

EXISTING_TIFF_VALIDATION_COLUMNS = [
    "output_tif",
    "SITE_ID",
    "YYYYMM",
    "band_count",
    "has_six_bands",
    "band_descriptions_match",
    "band3_is_binary_0_1",
    "all_bands_float32",
    "crs_is_aeqd",
    "pixel_size_x_m",
    "pixel_size_y_m",
    "radius_m_used",
    "expected_circle_area_m2",
    "raster_circle_area_m2",
    "circle_area_ratio",
    "circle_pixel_count",
    "circle_bbox_width_pixels",
    "circle_bbox_height_pixels",
    "circle_bbox_width_height_ratio",
    "circle_geometry_status",
    "validation_status",
    "error_message",
]


class InvalidSiteMonthError(ValueError):
    """Raised when a site-month cannot satisfy the required validation checks."""


@dataclass
class PreparedSiteMonth:
    """Validated inputs needed to create one site-month raster."""

    site_id: str
    site_id_safe: str
    yyyymm: str
    site_name: str
    igbp: str
    year: Any
    month: Any
    latitude: float
    longitude: float
    radius_m: float
    tower_gpp: float
    primary_gpp_column: str
    gpp_quality_flag: str
    n_valid_wd_records: float
    bins: list[tuple[float, float, float]]
    wd_fraction_sum: float
    warnings: list[str]

    @property
    def n_bins(self) -> int:
        return len(self.bins)


@dataclass
class RasterResult:
    """QC statistics returned after a GeoTIFF is written."""

    footprint_min: float
    footprint_max: float
    footprint_mean: float
    blind_buffer_pixel_count: int
    pixel_size_m: float
    expected_circle_area_m2: float
    raster_circle_area_m2: float
    circle_area_ratio: float
    circle_pixel_count: int
    circle_bbox_width_pixels: int
    circle_bbox_height_pixels: int
    circle_bbox_width_height_ratio: float
    circle_geometry_status: str
    warnings: list[str]


@dataclass
class CircleQAMetrics:
    """Geometry metrics for a binary pixel-center distance circle."""

    pixel_size_m: float
    expected_circle_area_m2: float
    raster_circle_area_m2: float
    circle_area_ratio: float
    circle_pixel_count: int
    circle_bbox_width_pixels: int
    circle_bbox_height_pixels: int
    circle_bbox_width_height_ratio: float
    circle_geometry_status: str


def warn(message: str) -> None:
    """Print a clearly marked warning."""
    print(f"WARNING: {message}", flush=True)


def native_filesystem_path(path: Path) -> str:
    """
    Return an OS path that supports long absolute paths on Windows.

    Deep OneDrive output folders can exceed the legacy 260-character Windows
    path limit even when rasterio successfully writes the file.
    """
    absolute_path = os.path.abspath(os.fspath(path))
    if os.name != "nt" or absolute_path.startswith("\\\\?\\"):
        return absolute_path
    if absolute_path.startswith("\\\\"):
        return "\\\\?\\UNC\\" + absolute_path[2:]
    return "\\\\?\\" + absolute_path


def path_exists(path: Path) -> bool:
    """Check file existence with Windows long-path support."""
    return os.path.exists(native_filesystem_path(path))


def remove_file_if_present(path: Path) -> None:
    """Remove a file if present, with Windows long-path support."""
    native_path = native_filesystem_path(path)
    if os.path.exists(native_path):
        os.remove(native_path)


def is_missing(value: Any) -> bool:
    """Return True for pandas/NumPy missing scalar values and blank strings."""
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
    """
    Interpret the allowed valid_for_GEE_export representations.

    Accepted values are 1, "1", True, "TRUE", "true", "yes", and "YES".
    Everything else is invalid.
    """
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
    """Convert a YYYYMM value from CSV or Excel into a validated six-digit string."""
    if is_missing(value):
        return None

    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.strftime("%Y%m")

    if isinstance(value, Number) and not isinstance(value, (bool, np.bool_)):
        try:
            numeric_value = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(numeric_value) or not numeric_value.is_integer():
            return None
        text = str(int(numeric_value))
    else:
        text = str(value).strip()
        numeric_match = re.fullmatch(r"(\d{6})\.0+", text)
        if numeric_match:
            text = numeric_match.group(1)

    if not re.fullmatch(r"\d{6}", text):
        return None

    year = int(text[:4])
    month = int(text[4:])
    if year < 1 or not 1 <= month <= 12:
        return None
    return text


def clean_site_id(value: Any) -> str | None:
    """Return a stripped site ID or None if it is missing."""
    if is_missing(value):
        return None
    site_id = str(value).strip()
    return site_id if site_id else None


def safe_site_id(site_id: str) -> str:
    """Apply the requested FLUXNET output-name conversion."""
    return site_id.replace("-", "_")


def finite_numeric_values(series: pd.Series) -> np.ndarray:
    """Return finite numeric values from a pandas Series."""
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
    return values[np.isfinite(values)]


def first_finite_value(group: pd.DataFrame, column: str) -> float | None:
    """Return the first finite numeric value in a group column."""
    if column not in group.columns:
        return None
    values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype="float64")
    valid_positions = np.flatnonzero(np.isfinite(values))
    if valid_positions.size == 0:
        return None
    return float(values[valid_positions[0]])


def first_positive_value(group: pd.DataFrame, column: str) -> float | None:
    """Return the first finite value greater than zero in a group column."""
    if column not in group.columns:
        return None
    values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype="float64")
    valid_positions = np.flatnonzero(np.isfinite(values) & (values > 0.0))
    if valid_positions.size == 0:
        return None
    return float(values[valid_positions[0]])


def first_non_missing_value(series: pd.Series, default: Any = "") -> Any:
    """Return the first non-missing value without changing its meaning."""
    for value in series:
        if not is_missing(value):
            return value
    return default


def first_text_value(group: pd.DataFrame, column: str, default: str = "") -> str:
    """Return a stripped text representation of the first non-missing value."""
    if column not in group.columns:
        return default
    value = first_non_missing_value(group[column], default)
    return default if is_missing(value) else str(value).strip()


def tag_value(value: Any) -> str:
    """Convert metadata to a stable string accepted by rasterio tags."""
    if is_missing(value):
        return ""
    if isinstance(value, (np.integer, int)) and not isinstance(value, (bool, np.bool_)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            return ""
        if numeric_value.is_integer():
            return str(int(numeric_value))
        return format(numeric_value, ".15g")
    return str(value)


def add_inconsistency_warning(
    group: pd.DataFrame,
    column: str,
    site_id: str,
    yyyymm: str,
    warnings: list[str],
) -> None:
    """Warn when a site-month-level numeric field varies across WD-bin rows."""
    if column not in group.columns:
        return
    values = finite_numeric_values(group[column])
    if values.size <= 1:
        return
    reference = values[0]
    if not np.all(np.isclose(values, reference, rtol=1e-9, atol=1e-9)):
        warnings.append(
            f"{site_id} {yyyymm}: '{column}' varies across WD-bin rows; "
            f"the first valid value ({reference:g}) is used."
        )


def load_input_table(path: Path) -> pd.DataFrame:
    """Load one CSV, XLSX, or XLS input table based on its file extension."""
    if not path.exists():
        raise FileNotFoundError(f"Input table does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Input table path is not a file: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        try:
            table = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
        except UnicodeDecodeError:
            warn("UTF-8 CSV decoding failed; retrying with Windows cp1252 encoding.")
            table = pd.read_csv(path, low_memory=False, encoding="cp1252")
    elif suffix in {".xlsx", ".xls"}:
        engine = "openpyxl" if suffix == ".xlsx" else "xlrd"
        table = pd.read_excel(path, engine=engine)
    else:
        raise ValueError(
            f"Unsupported input format '{path.suffix}'. "
            "Use a .csv, .xlsx, or .xls file."
        )

    cleaned_columns = [str(column).strip().lstrip("\ufeff") for column in table.columns]
    duplicate_columns = sorted(
        {
            column
            for column in cleaned_columns
            if cleaned_columns.count(column) > 1
        }
    )
    if duplicate_columns:
        raise ValueError(
            "Duplicate columns remain after trimming names: "
            + ", ".join(duplicate_columns)
        )
    table.columns = cleaned_columns
    return table


def table_site_and_month_counts(table: pd.DataFrame) -> tuple[int, int]:
    """Count non-missing sites and valid SITE_ID + YYYYMM combinations."""
    if "SITE_ID" not in table.columns or "YYYYMM" not in table.columns:
        return 0, 0

    site_ids = table["SITE_ID"].map(clean_site_id)
    yyyymm_values = table["YYYYMM"].map(normalize_yyyymm)
    valid_keys = site_ids.notna() & yyyymm_values.notna()
    n_sites = int(site_ids.dropna().nunique())
    n_site_months = int(
        pd.DataFrame(
            {
                "SITE_ID": site_ids.loc[valid_keys],
                "YYYYMM": yyyymm_values.loc[valid_keys],
            }
        )
        .drop_duplicates()
        .shape[0]
    )
    return n_sites, n_site_months


def load_and_merge_input_tables(
    input_paths: Iterable[str | Path],
) -> pd.DataFrame:
    """
    Load one or more CSV/XLSX/XLS tables, add source provenance, and merge them.

    Every input is checked and validated separately so a missing column in one
    file cannot be hidden by the same column being present in another file.
    """
    if isinstance(input_paths, (str, Path)):
        paths = [Path(input_paths).expanduser()]
    else:
        paths = [Path(path).expanduser() for path in input_paths]

    if not paths:
        raise ValueError("INPUT_TABLE_PATHS must contain at least one input path.")

    missing_paths = [path for path in paths if not path.exists() or not path.is_file()]
    if missing_paths:
        formatted_paths = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(
            "The following input table(s) do not exist or are not files:\n"
            f"{formatted_paths}"
        )

    print(f"Number of input tables found: {len(paths):,}", flush=True)
    loaded_tables: list[pd.DataFrame] = []

    for table_number, path in enumerate(paths, start=1):
        resolved_path = path.resolve()
        print(
            f"\nLoading input table {table_number}/{len(paths)}: {resolved_path}",
            flush=True,
        )
        table = load_input_table(resolved_path)
        validate_input_columns(table)
        table["source_input_table"] = str(resolved_path)
        n_sites, n_site_months = table_site_and_month_counts(table)

        print(f"  Rows: {len(table):,}", flush=True)
        print(f"  Unique sites: {n_sites:,}", flush=True)
        print(f"  Unique site-months: {n_site_months:,}", flush=True)
        loaded_tables.append(table)

    merged = pd.concat(loaded_tables, ignore_index=True, sort=False)
    merged_sites, merged_site_months = table_site_and_month_counts(merged)
    print("\nInput tables merged.", flush=True)
    print(f"Merged row count: {len(merged):,}", flush=True)
    print(f"Merged unique sites: {merged_sites:,}", flush=True)
    print(f"Merged unique site-months: {merged_site_months:,}", flush=True)
    return merged


def validate_input_columns(table: pd.DataFrame) -> str | None:
    """
    Validate required columns and choose the available WD record-count field.

    The radius columns and WD record-count columns are optional because the
    requested workflow defines fallbacks for them.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(
            "Input table is missing required column(s): " + ", ".join(missing)
        )

    if "n_valid_monthly_30min_WD_records" in table.columns:
        return "n_valid_monthly_30min_WD_records"
    if "n_valid_monthly_daytime_WD_records" in table.columns:
        return "n_valid_monthly_daytime_WD_records"
    return None


def has_any_positive_radius(group: pd.DataFrame) -> bool:
    """Return True when either radius field contains a finite positive value."""
    for column in ("radius_m_max", "radius_m"):
        if column not in group.columns:
            continue
        values = pd.to_numeric(group[column], errors="coerce").to_numpy(
            dtype="float64"
        )
        if bool(np.any(np.isfinite(values) & (values > 0.0))):
            return True
    return False


def has_any_valid_lat_lon(group: pd.DataFrame) -> bool:
    """Return True when at least one row has a geographically valid lat/lon pair."""
    latitudes = pd.to_numeric(group["lat"], errors="coerce").to_numpy(
        dtype="float64"
    )
    longitudes = pd.to_numeric(group["lon"], errors="coerce").to_numpy(
        dtype="float64"
    )
    valid = (
        np.isfinite(latitudes)
        & np.isfinite(longitudes)
        & (latitudes >= -90.0)
        & (latitudes <= 90.0)
        & (longitudes >= -180.0)
        & (longitudes <= 180.0)
    )
    return bool(np.any(valid))


def has_any_valid_gpp(group: pd.DataFrame) -> bool:
    """Return True when at least one usable monthly tower GPP value is present."""
    values = pd.to_numeric(
        group["tower_GPP_monthly"], errors="coerce"
    ).to_numpy(dtype="float64")
    valid = (
        np.isfinite(values)
        & (values != MISSING_GPP_SENTINEL)
        & (np.abs(values) <= np.finfo(np.float32).max)
    )
    return bool(np.any(valid))


def build_input_coverage_report(merged_table: pd.DataFrame) -> pd.DataFrame:
    """Build the requested per-input-table, per-site preflight coverage report."""
    report_table = merged_table.copy()
    report_table["_REPORT_SITE_ID"] = report_table["SITE_ID"].map(clean_site_id)
    report_table["_REPORT_SITE_ID"] = report_table["_REPORT_SITE_ID"].fillna(
        "<missing>"
    )

    rows: list[dict[str, Any]] = []
    grouped = report_table.groupby(
        ["source_input_table", "_REPORT_SITE_ID"],
        sort=True,
        dropna=False,
    )
    for (source_path, site_id), group in grouped:
        valid_flags = group["valid_for_GEE_export"].map(is_export_true)
        months = group["YYYYMM"].map(normalize_yyyymm)
        total_months = sorted(months.dropna().unique().tolist())
        valid_months = sorted(months.loc[valid_flags].dropna().unique().tolist())

        rows.append(
            {
                "source_input_table": str(source_path),
                "SITE_ID": str(site_id),
                "n_rows_total": int(len(group)),
                "n_rows_valid": int(valid_flags.sum()),
                "n_site_months_total": int(len(total_months)),
                "n_site_months_valid": int(len(valid_months)),
                "first_YYYYMM": total_months[0] if total_months else "",
                "last_YYYYMM": total_months[-1] if total_months else "",
                "has_any_radius": has_any_positive_radius(group),
                "has_any_lat_lon": has_any_valid_lat_lon(group),
                "has_any_GPP": has_any_valid_gpp(group),
                "n_valid_for_GEE_export_1": int(valid_flags.sum()),
                "n_valid_for_GEE_export_0": int((~valid_flags).sum()),
            }
        )

    return pd.DataFrame(rows, columns=INPUT_COVERAGE_COLUMNS)


def build_merged_valid_site_list(deduplicated_table: pd.DataFrame) -> pd.DataFrame:
    """Build a site-level list from rows with a valid export flag."""
    valid_rows = deduplicated_table.loc[
        deduplicated_table["valid_for_GEE_export"].map(is_export_true)
    ].copy()
    valid_rows["_REPORT_SITE_ID"] = valid_rows["SITE_ID"].map(clean_site_id)
    valid_rows["_REPORT_YYYYMM"] = valid_rows["YYYYMM"].map(normalize_yyyymm)
    valid_rows = valid_rows.loc[valid_rows["_REPORT_SITE_ID"].notna()]

    rows: list[dict[str, Any]] = []
    for site_id, group in valid_rows.groupby(
        "_REPORT_SITE_ID",
        sort=True,
        dropna=False,
    ):
        months = sorted(group["_REPORT_YYYYMM"].dropna().unique().tolist())
        source_paths = sorted(
            {
                str(value)
                for value in group["source_input_table"]
                if not is_missing(value)
            }
        )
        rows.append(
            {
                "SITE_ID": str(site_id),
                "source_input_table_list": " | ".join(source_paths),
                "n_valid_rows": int(len(group)),
                "n_valid_site_months": int(len(months)),
                "first_YYYYMM": months[0] if months else "",
                "last_YYYYMM": months[-1] if months else "",
                "IGBP": first_text_value(group, "IGBP"),
                "site_name": first_text_value(group, "site_name"),
            }
        )

    return pd.DataFrame(rows, columns=MERGED_VALID_SITE_COLUMNS)


def deduplicate_merged_table(
    merged_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove repeated SITE_ID + YYYYMM + WD-bin-center rows after merging."""
    duplicate_mask = merged_table.duplicated(
        subset=DEDUPLICATION_KEY,
        keep="first",
    )
    removed_rows = merged_table.loc[duplicate_mask].copy()
    deduplicated = merged_table.loc[~duplicate_mask].copy().reset_index(drop=True)
    print(
        f"Duplicate rows removed using {DEDUPLICATION_KEY}: "
        f"{len(removed_rows):,}",
        flush=True,
    )
    return deduplicated, removed_rows


def write_preflight_reports(
    qc_folder: Path,
    merged_table_before_deduplication: pd.DataFrame,
    deduplicated_table: pd.DataFrame,
    removed_duplicate_rows: pd.DataFrame,
) -> tuple[Path, Path, Path]:
    """Write coverage, merged-site, and removed-duplicate reports."""
    qc_folder.mkdir(parents=True, exist_ok=True)
    coverage_path = qc_folder / "input_table_coverage_report.csv"
    merged_sites_path = qc_folder / "merged_valid_site_list.csv"
    duplicates_path = qc_folder / "removed_duplicate_rows.csv"

    build_input_coverage_report(merged_table_before_deduplication).to_csv(
        coverage_path,
        index=False,
        encoding="utf-8-sig",
    )
    build_merged_valid_site_list(deduplicated_table).to_csv(
        merged_sites_path,
        index=False,
        encoding="utf-8-sig",
    )
    removed_duplicate_rows.to_csv(
        duplicates_path,
        index=False,
        encoding="utf-8-sig",
    )
    return coverage_path, merged_sites_path, duplicates_path


def validate_settings() -> tuple[str | None, str | None]:
    """Validate editable controls and normalize optional month limits."""
    boolean_settings = {
        "DRY_RUN_ONLY": DRY_RUN_ONLY,
        "VALIDATE_ONLY": VALIDATE_ONLY,
        "RECREATE_EXISTING": RECREATE_EXISTING,
        "CREATE_QA_PNGS": CREATE_QA_PNGS,
        "REQUIRE_VALID_FOR_GEE_EXPORT": REQUIRE_VALID_FOR_GEE_EXPORT,
    }
    for setting_name, setting_value in boolean_settings.items():
        if not isinstance(setting_value, bool):
            raise ValueError(f"{setting_name} must be True or False.")
    if (
        isinstance(QA_PNG_MAX, bool)
        or not isinstance(QA_PNG_MAX, int)
        or QA_PNG_MAX < 0
    ):
        raise ValueError("QA_PNG_MAX must be a non-negative integer.")
    if PIXEL_SIZE_M <= 0:
        raise ValueError("PIXEL_SIZE_M must be greater than zero.")
    if EXPORT_MARGIN_M < 0:
        raise ValueError("EXPORT_MARGIN_M cannot be negative.")
    if MINIMUM_WEDGE_WIDTH_DEG <= 0:
        raise ValueError("MINIMUM_WEDGE_WIDTH_DEG must be greater than zero.")
    if not isinstance(WEDGE_ARC_SEGMENTS, int) or WEDGE_ARC_SEGMENTS < 1:
        raise ValueError("WEDGE_ARC_SEGMENTS must be a positive integer.")
    if MAX_SITE_MONTHS is not None:
        if (
            isinstance(MAX_SITE_MONTHS, bool)
            or not isinstance(MAX_SITE_MONTHS, int)
            or MAX_SITE_MONTHS < 0
        ):
            raise ValueError("MAX_SITE_MONTHS must be None or a non-negative integer.")

    start = normalize_yyyymm(START_YYYYMM) if START_YYYYMM is not None else None
    end = normalize_yyyymm(END_YYYYMM) if END_YYYYMM is not None else None
    if START_YYYYMM is not None and start is None:
        raise ValueError(f"Invalid START_YYYYMM: {START_YYYYMM!r}")
    if END_YYYYMM is not None and end is None:
        raise ValueError(f"Invalid END_YYYYMM: {END_YYYYMM!r}")
    if start is not None and end is not None and start > end:
        raise ValueError("START_YYYYMM cannot be later than END_YYYYMM.")
    return start, end


def resolve_radius(group: pd.DataFrame) -> float | None:
    """Use radius_m_max when valid, then fall back to radius_m."""
    radius = first_positive_value(group, "radius_m_max")
    if radius is not None:
        return radius
    return first_positive_value(group, "radius_m")


def resolve_tower_gpp(group: pd.DataFrame) -> float | None:
    """Return the first finite, non-sentinel monthly tower GPP value."""
    if "tower_GPP_monthly" not in group.columns:
        return None
    values = pd.to_numeric(
        group["tower_GPP_monthly"], errors="coerce"
    ).to_numpy(dtype="float64")
    valid = np.isfinite(values) & (values != MISSING_GPP_SENTINEL)
    positions = np.flatnonzero(valid)
    if positions.size == 0:
        return None
    value = float(values[positions[0]])
    if abs(value) > np.finfo(np.float32).max:
        return None
    return value


def resolve_record_count(
    group: pd.DataFrame,
    records_column: str | None,
) -> float:
    """Resolve the optional monthly valid WD-record count."""
    if records_column is None:
        return MISSING_RECORD_COUNT
    value = first_finite_value(group, records_column)
    if value is None or abs(value) > np.finfo(np.float32).max:
        return MISSING_RECORD_COUNT
    return value


def prepare_site_month(
    group: pd.DataFrame,
    site_id: str,
    yyyymm: str,
    records_column: str | None,
) -> PreparedSiteMonth:
    """Validate and prepare one grouped site-month."""
    warnings: list[str] = []

    latitude = first_finite_value(group, "lat")
    longitude = first_finite_value(group, "lon")
    if latitude is None or longitude is None:
        raise InvalidSiteMonthError("Missing or non-numeric latitude/longitude.")
    if not -90.0 <= latitude <= 90.0 or not -180.0 <= longitude <= 180.0:
        raise InvalidSiteMonthError(
            f"Invalid latitude/longitude: lat={latitude}, lon={longitude}."
        )

    radius_m = resolve_radius(group)
    if radius_m is None or not math.isfinite(radius_m) or radius_m <= 0.0:
        raise InvalidSiteMonthError(
            "No valid radius: radius_m_max and radius_m are missing or not > 0."
        )

    tower_gpp = resolve_tower_gpp(group)
    if tower_gpp is None:
        raise InvalidSiteMonthError(
            "tower_GPP_monthly is missing, non-finite, or invalid."
        )

    add_inconsistency_warning(group, "lat", site_id, yyyymm, warnings)
    add_inconsistency_warning(group, "lon", site_id, yyyymm, warnings)
    add_inconsistency_warning(
        group, "tower_GPP_monthly", site_id, yyyymm, warnings
    )
    if "radius_m_max" in group.columns:
        add_inconsistency_warning(
            group, "radius_m_max", site_id, yyyymm, warnings
        )
    if records_column is not None:
        add_inconsistency_warning(
            group, records_column, site_id, yyyymm, warnings
        )

    bins: list[tuple[float, float, float]] = []
    invalid_bin_count = 0
    widened_bin_count = 0

    bin_columns = [
        "WD_bin_center_deg",
        "WD_bin_width_deg",
        "WD_bin_fraction",
    ]
    numeric_bins = group[bin_columns].apply(pd.to_numeric, errors="coerce")

    for row in numeric_bins.itertuples(index=False, name=None):
        center_deg, width_deg, fraction = (float(value) for value in row)
        if (
            not math.isfinite(center_deg)
            or not math.isfinite(width_deg)
            or not math.isfinite(fraction)
            or fraction < 0.0
        ):
            invalid_bin_count += 1
            continue

        width_used = max(width_deg, MINIMUM_WEDGE_WIDTH_DEG)
        if width_used != width_deg:
            widened_bin_count += 1
        bins.append((center_deg, width_used, fraction))

    if invalid_bin_count:
        warnings.append(
            f"{site_id} {yyyymm}: ignored {invalid_bin_count} WD-bin row(s) "
            "with missing/non-finite geometry values or negative fractions."
        )
    if widened_bin_count:
        warnings.append(
            f"{site_id} {yyyymm}: enforced the {MINIMUM_WEDGE_WIDTH_DEG:g}° "
            f"minimum width for {widened_bin_count} WD bin(s)."
        )
    if not bins:
        raise InvalidSiteMonthError("No usable WD-bin rows remain for this site-month.")

    wd_fraction_sum = float(math.fsum(bin_data[2] for bin_data in bins))
    if not 0.99 <= wd_fraction_sum <= 1.01:
        warnings.append(
            f"{site_id} {yyyymm}: WD_bin_fraction sum is "
            f"{wd_fraction_sum:.6f}, outside 0.99-1.01; processing continues."
        )

    year_value = first_non_missing_value(group["year"], int(yyyymm[:4]))
    month_value = first_non_missing_value(group["month"], int(yyyymm[4:]))

    return PreparedSiteMonth(
        site_id=site_id,
        site_id_safe=safe_site_id(site_id),
        yyyymm=yyyymm,
        site_name=first_text_value(group, "site_name"),
        igbp=first_text_value(group, "IGBP"),
        year=year_value,
        month=month_value,
        latitude=latitude,
        longitude=longitude,
        radius_m=radius_m,
        tower_gpp=tower_gpp,
        primary_gpp_column=first_text_value(group, "primary_GPP_column"),
        gpp_quality_flag=first_text_value(group, "GPP_monthly_quality_flag"),
        n_valid_wd_records=resolve_record_count(group, records_column),
        bins=bins,
        wd_fraction_sum=wd_fraction_sum,
        warnings=warnings,
    )


def make_local_aeqd_crs(latitude: float, longitude: float) -> CRS:
    """Create the requested site-centered local Azimuthal Equidistant CRS."""
    proj4 = (
        f"+proj=aeqd +lat_0={latitude:.12f} +lon_0={longitude:.12f} "
        "+datum=WGS84 +units=m +no_defs"
    )
    return CRS.from_proj4(proj4)


def create_wedge_polygon(
    center_x: float,
    center_y: float,
    radius_m: float,
    center_bearing_deg: float,
    width_deg: float,
) -> Polygon:
    """
    Create an upwind wedge using bearings clockwise from north.

    WD_bin_center_deg is used directly. No 180-degree rotation is applied.
    """
    start_bearing = center_bearing_deg - width_deg / 2.0
    end_bearing = center_bearing_deg + width_deg / 2.0
    bearings = np.linspace(
        start_bearing,
        end_bearing,
        WEDGE_ARC_SEGMENTS + 1,
        dtype="float64",
    )
    radians = np.deg2rad(bearings)

    coordinates: list[tuple[float, float]] = [(center_x, center_y)]
    coordinates.extend(
        (
            center_x + radius_m * math.sin(angle),
            center_y + radius_m * math.cos(angle),
        )
        for angle in radians
    )
    coordinates.append((center_x, center_y))
    return Polygon(coordinates)


def validate_centered_north_up_grid(
    transform: Any,
    width: int,
    height: int,
    center_x: float,
    center_y: float,
) -> None:
    """Verify square pixels, north-up orientation, and symmetric site bounds."""
    tolerance = max(1e-7, PIXEL_SIZE_M * 1e-9)
    if width != height:
        raise RuntimeError(
            f"Output grid is not square: width={width}, height={height}."
        )
    if not math.isclose(transform.a, PIXEL_SIZE_M, abs_tol=tolerance):
        raise RuntimeError(
            f"Unexpected x pixel size {transform.a}; expected {PIXEL_SIZE_M}."
        )
    if not math.isclose(transform.e, -PIXEL_SIZE_M, abs_tol=tolerance):
        raise RuntimeError(
            f"Unexpected y pixel size {transform.e}; expected {-PIXEL_SIZE_M}."
        )
    if not math.isclose(transform.b, 0.0, abs_tol=tolerance) or not math.isclose(
        transform.d, 0.0, abs_tol=tolerance
    ):
        raise RuntimeError("Output transform is rotated instead of north-up.")

    west = transform.c
    east = transform.c + width * transform.a
    north = transform.f
    south = transform.f + height * transform.e
    if not math.isclose(
        center_x - west,
        east - center_x,
        abs_tol=tolerance,
    ) or not math.isclose(
        north - center_y,
        center_y - south,
        abs_tol=tolerance,
    ):
        raise RuntimeError("Output bounds are not symmetric around the site center.")


def create_distance_circle_mask(
    *,
    transform: Any,
    out_shape: tuple[int, int],
    center_x: float,
    center_y: float,
    radius_m: float,
) -> np.ndarray:
    """
    Create Band 3 from pixel-center distance in the local metric AEQD CRS.

    A pixel is inside the blind benchmark circle exactly when its center is no
    farther than radius_m from the tower center.
    """
    height, width = out_shape
    validate_centered_north_up_grid(
        transform=transform,
        width=width,
        height=height,
        center_x=center_x,
        center_y=center_y,
    )

    x_centers = transform.c + (
        np.arange(width, dtype="float64") + 0.5
    ) * transform.a
    y_centers = transform.f + (
        np.arange(height, dtype="float64") + 0.5
    ) * transform.e
    distance_squared = (
        (x_centers[np.newaxis, :] - center_x) ** 2
        + (y_centers[:, np.newaxis] - center_y) ** 2
    )
    return (distance_squared <= radius_m**2).astype("float32")


def calculate_circle_qa(
    circle_mask: np.ndarray,
    transform: Any,
    radius_m: float,
) -> CircleQAMetrics:
    """Calculate area and bounding-box QA metrics for Band 3."""
    circle_pixels = circle_mask > 0.5
    pixel_count = int(np.count_nonzero(circle_pixels))
    pixel_area_m2 = abs(
        transform.a * transform.e - transform.b * transform.d
    )
    expected_area_m2 = float(math.pi * radius_m**2)
    raster_area_m2 = float(pixel_count * pixel_area_m2)
    area_ratio = (
        float(raster_area_m2 / expected_area_m2)
        if expected_area_m2 > 0.0
        else math.nan
    )

    occupied_rows, occupied_columns = np.nonzero(circle_pixels)
    if pixel_count:
        bbox_height = int(occupied_rows.max() - occupied_rows.min() + 1)
        bbox_width = int(occupied_columns.max() - occupied_columns.min() + 1)
        bbox_ratio = (
            float(bbox_width / bbox_height) if bbox_height > 0 else math.nan
        )
    else:
        bbox_height = 0
        bbox_width = 0
        bbox_ratio = math.nan

    area_ok = (
        math.isfinite(area_ratio)
        and CIRCLE_AREA_RATIO_MIN <= area_ratio <= CIRCLE_AREA_RATIO_MAX
    )
    bbox_ok = (
        math.isfinite(bbox_ratio)
        and CIRCLE_BBOX_RATIO_MIN <= bbox_ratio <= CIRCLE_BBOX_RATIO_MAX
    )
    status = "OK" if area_ok and bbox_ok else "WARNING_CIRCLE_GEOMETRY"

    return CircleQAMetrics(
        pixel_size_m=float(abs(transform.a)),
        expected_circle_area_m2=expected_area_m2,
        raster_circle_area_m2=raster_area_m2,
        circle_area_ratio=area_ratio,
        circle_pixel_count=pixel_count,
        circle_bbox_width_pixels=bbox_width,
        circle_bbox_height_pixels=bbox_height,
        circle_bbox_width_height_ratio=bbox_ratio,
        circle_geometry_status=status,
    )


def create_raster_arrays(
    prepared: PreparedSiteMonth,
) -> tuple[np.ndarray, np.ndarray, Any, CRS, float, float]:
    """Build footprint and blind-buffer arrays in the site-centered AEQD CRS."""
    local_crs = make_local_aeqd_crs(prepared.latitude, prepared.longitude)
    transformer = Transformer.from_crs("EPSG:4326", local_crs, always_xy=True)
    center_x, center_y = transformer.transform(
        prepared.longitude,
        prepared.latitude,
    )
    if not math.isfinite(center_x) or not math.isfinite(center_y):
        raise RuntimeError("The site could not be transformed into its local AEQD CRS.")

    requested_half_extent = prepared.radius_m + EXPORT_MARGIN_M
    grid_pixels = max(
        1,
        int(math.ceil((2.0 * requested_half_extent) / PIXEL_SIZE_M)),
    )
    actual_half_extent = grid_pixels * PIXEL_SIZE_M / 2.0
    west = center_x - actual_half_extent
    north = center_y + actual_half_extent
    transform = from_origin(west, north, PIXEL_SIZE_M, PIXEL_SIZE_M)
    out_shape = (grid_pixels, grid_pixels)

    weighted_wedges: list[tuple[dict[str, Any], float]] = []
    for center_bearing, width_deg, fraction in prepared.bins:
        wedge = create_wedge_polygon(
            center_x=center_x,
            center_y=center_y,
            radius_m=prepared.radius_m,
            center_bearing_deg=center_bearing,
            width_deg=width_deg,
        )
        if wedge.is_empty or wedge.area <= 0.0:
            raise RuntimeError(
                "A WD wedge produced an empty or zero-area polygon "
                f"(center={center_bearing}, width={width_deg})."
            )
        weighted_wedges.append((mapping(wedge), fraction))

    footprint_probability = rasterize(
        weighted_wedges,
        out_shape=out_shape,
        transform=transform,
        fill=0.0,
        all_touched=False,
        merge_alg=MergeAlg.add,
        dtype="float32",
    )
    np.clip(footprint_probability, 0.0, 1.0, out=footprint_probability)

    blind_circular_buffer = create_distance_circle_mask(
        transform=transform,
        out_shape=out_shape,
        center_x=center_x,
        center_y=center_y,
        radius_m=prepared.radius_m,
    )

    return (
        footprint_probability,
        blind_circular_buffer,
        transform,
        local_crs,
        center_x,
        center_y,
    )


def output_path_for(prepared: PreparedSiteMonth, output_root: Path) -> Path:
    """Return the requested site-folder and TIFF filename."""
    site_folder = output_root / (
        "FLUXNET_monthly_histogram_footprint_GPP_TIFFs_"
        f"{prepared.site_id_safe}"
    )
    filename = (
        f"{prepared.site_id_safe}_monthly_30min_WD_histogram_"
        f"footprint_GPP_{prepared.yyyymm}.tif"
    )
    return site_folder / filename


def provisional_output_path(
    site_id: str,
    yyyymm: str,
    output_root: Path,
) -> Path:
    """Build an output path before the complete site-month is validated."""
    site_id_safe = safe_site_id(site_id)
    site_folder = output_root / (
        "FLUXNET_monthly_histogram_footprint_GPP_TIFFs_"
        f"{site_id_safe}"
    )
    filename = (
        f"{site_id_safe}_monthly_30min_WD_histogram_"
        f"footprint_GPP_{yyyymm}.tif"
    )
    return site_folder / filename


def create_visual_qa_png(
    *,
    prepared: PreparedSiteMonth,
    footprint: np.ndarray,
    circle_mask: np.ndarray,
    transform: Any,
    local_crs: CRS,
    center_x: float,
    center_y: float,
    output_png: Path,
) -> None:
    """Save a side-by-side footprint and distance-circle visual QA image."""
    if plt is None or MatplotlibCircle is None:
        raise RuntimeError(
            "matplotlib is not installed; run "
            "'pip install matplotlib' to create QA PNGs."
        )

    height, width = footprint.shape
    west = transform.c
    east = transform.c + width * transform.a
    north = transform.f
    south = transform.f + height * transform.e
    extent = (west, east, south, north)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    footprint_image = axes[0].imshow(
        footprint,
        origin="upper",
        extent=extent,
        cmap="viridis",
        vmin=0.0,
        vmax=max(1.0, float(np.max(footprint))),
        interpolation="nearest",
    )
    axes[0].set_title("Band 1: footprint_probability")
    figure.colorbar(footprint_image, ax=axes[0], fraction=0.046, pad=0.04)

    circle_image = axes[1].imshow(
        circle_mask,
        origin="upper",
        extent=extent,
        cmap="gray_r",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    axes[1].set_title("Band 3: blind_circular_buffer")
    figure.colorbar(circle_image, ax=axes[1], fraction=0.046, pad=0.04)

    for axis in axes:
        boundary = MatplotlibCircle(
            (center_x, center_y),
            prepared.radius_m,
            fill=False,
            edgecolor="red",
            linewidth=1.2,
            linestyle="--",
        )
        axis.add_patch(boundary)
        axis.plot(center_x, center_y, marker="+", color="red", markersize=8)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("Local AEQD x (m)")
        axis.set_ylabel("Local AEQD y (m)")

    figure.suptitle(
        f"{prepared.site_id} {prepared.yyyymm} | "
        f"radius={prepared.radius_m:g} m | pixel={PIXEL_SIZE_M:g} m | "
        f"AEQD centered at ({prepared.latitude:.6f}, {prepared.longitude:.6f})\n"
        f"CRS: {local_crs.to_string()}"
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        native_filesystem_path(output_png),
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def write_geotiff(
    prepared: PreparedSiteMonth,
    output_path: Path,
    qa_png_path: Path | None = None,
) -> RasterResult:
    """Create and atomically write one six-band monthly GeoTIFF."""
    (
        footprint,
        blind_buffer,
        transform,
        local_crs,
        center_x,
        center_y,
    ) = create_raster_arrays(prepared)

    footprint_min = float(np.min(footprint))
    footprint_max = float(np.max(footprint))
    footprint_mean = float(np.mean(footprint, dtype=np.float64))
    circle_qa = calculate_circle_qa(
        circle_mask=blind_buffer,
        transform=transform,
        radius_m=prepared.radius_m,
    )

    write_warnings: list[str] = []
    if footprint_max <= 0.0:
        write_warnings.append(
            f"{prepared.site_id} {prepared.yyyymm}: "
            "footprint_probability maximum is zero."
        )
    if circle_qa.circle_geometry_status == "WARNING_CIRCLE_GEOMETRY":
        write_warnings.append(
            f"{prepared.site_id} {prepared.yyyymm}: WARNING_CIRCLE_GEOMETRY "
            f"(area_ratio={circle_qa.circle_area_ratio:.6f}, "
            "bbox_width_height_ratio="
            f"{circle_qa.circle_bbox_width_height_ratio:.6f})."
        )

    height, width = footprint.shape
    profile: dict[str, Any] = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": len(BAND_DESCRIPTIONS),
        "dtype": "float32",
        "crs": local_crs,
        "transform": transform,
        "compress": "LZW",
    }
    if prepared.n_valid_wd_records == MISSING_RECORD_COUNT:
        profile["nodata"] = MISSING_RECORD_COUNT

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    remove_file_if_present(temporary_path)

    constant_buffer = np.empty((height, width), dtype="float32")
    try:
        with rasterio.open(temporary_path, "w", **profile) as dataset:
            dataset.write(footprint, 1)

            constant_buffer.fill(np.float32(prepared.tower_gpp))
            dataset.write(constant_buffer, 2)

            dataset.write(blind_buffer, 3)

            constant_buffer.fill(np.float32(prepared.n_bins))
            dataset.write(constant_buffer, 4)

            constant_buffer.fill(np.float32(prepared.n_valid_wd_records))
            dataset.write(constant_buffer, 5)

            footprint_x_gpp = footprint * np.float32(prepared.tower_gpp)
            dataset.write(footprint_x_gpp.astype("float32", copy=False), 6)

            for band_number, description in enumerate(BAND_DESCRIPTIONS, start=1):
                dataset.set_band_description(band_number, description)

            dataset.update_tags(
                SITE_ID=prepared.site_id,
                site_name=prepared.site_name,
                IGBP=prepared.igbp,
                YYYYMM=prepared.yyyymm,
                year=tag_value(prepared.year),
                month=tag_value(prepared.month),
                tower_GPP_monthly=tag_value(prepared.tower_gpp),
                primary_GPP_column=prepared.primary_gpp_column,
                GPP_monthly_quality_flag=prepared.gpp_quality_flag,
                radius_m_used=tag_value(prepared.radius_m),
                pixel_size_m=tag_value(circle_qa.pixel_size_m),
                footprint_method=(
                    "monthly_probability_from_30min_WD_histogram"
                ),
                comparison_method=(
                    "traditional_tower_centered_blind_circular_buffer"
                ),
                blind_circular_buffer_method=(
                    "pixel_center_distance_in_local_AEQD"
                ),
                created_by=(
                    "Python_local_reproduction_of_GEE_monthly_"
                    "footprint_exporter"
                ),
            )

        os.replace(
            native_filesystem_path(temporary_path),
            native_filesystem_path(output_path),
        )
    except Exception:
        if path_exists(temporary_path):
            try:
                remove_file_if_present(temporary_path)
            except OSError as cleanup_error:
                warn(
                    f"Could not remove incomplete temporary file "
                    f"'{temporary_path}': {cleanup_error}"
                )
        raise

    if qa_png_path is not None:
        try:
            create_visual_qa_png(
                prepared=prepared,
                footprint=footprint,
                circle_mask=blind_buffer,
                transform=transform,
                local_crs=local_crs,
                center_x=center_x,
                center_y=center_y,
                output_png=qa_png_path,
            )
        except Exception as error:
            write_warnings.append(
                f"{prepared.site_id} {prepared.yyyymm}: QA PNG creation failed: "
                f"{type(error).__name__}: {error}"
            )

    return RasterResult(
        footprint_min=footprint_min,
        footprint_max=footprint_max,
        footprint_mean=footprint_mean,
        blind_buffer_pixel_count=circle_qa.circle_pixel_count,
        pixel_size_m=circle_qa.pixel_size_m,
        expected_circle_area_m2=circle_qa.expected_circle_area_m2,
        raster_circle_area_m2=circle_qa.raster_circle_area_m2,
        circle_area_ratio=circle_qa.circle_area_ratio,
        circle_pixel_count=circle_qa.circle_pixel_count,
        circle_bbox_width_pixels=circle_qa.circle_bbox_width_pixels,
        circle_bbox_height_pixels=circle_qa.circle_bbox_height_pixels,
        circle_bbox_width_height_ratio=(
            circle_qa.circle_bbox_width_height_ratio
        ),
        circle_geometry_status=circle_qa.circle_geometry_status,
        warnings=write_warnings,
    )


def make_summary_record(
    *,
    site_id: str,
    yyyymm: str,
    output_tif: Path | str | None,
    status: str,
    n_bins: int | None = None,
    wd_fraction_sum: float | None = None,
    radius_m_used: float | None = None,
    tower_gpp: float | None = None,
    n_valid_records: float | None = None,
    footprint_min: float | None = None,
    footprint_max: float | None = None,
    footprint_mean: float | None = None,
    blind_pixel_count: int | None = None,
    pixel_size_m: float | None = None,
    expected_circle_area_m2: float | None = None,
    raster_circle_area_m2: float | None = None,
    circle_area_ratio: float | None = None,
    circle_pixel_count: int | None = None,
    circle_bbox_width_pixels: int | None = None,
    circle_bbox_height_pixels: int | None = None,
    circle_bbox_width_height_ratio: float | None = None,
    circle_geometry_status: str = "",
    error_message: str = "",
) -> dict[str, Any]:
    """Create one QC summary row with the exact requested schema."""
    return {
        "SITE_ID": site_id,
        "YYYYMM": yyyymm,
        "output_tif": "" if output_tif is None else str(output_tif),
        "status": status,
        "n_bins": n_bins,
        "wd_fraction_sum": wd_fraction_sum,
        "radius_m_used": radius_m_used,
        "tower_GPP_monthly": tower_gpp,
        "n_valid_30min_WD_records": n_valid_records,
        "footprint_probability_min": footprint_min,
        "footprint_probability_max": footprint_max,
        "footprint_probability_mean": footprint_mean,
        "blind_buffer_pixel_count": blind_pixel_count,
        "pixel_size_m": pixel_size_m,
        "expected_circle_area_m2": expected_circle_area_m2,
        "raster_circle_area_m2": raster_circle_area_m2,
        "circle_area_ratio": circle_area_ratio,
        "circle_pixel_count": circle_pixel_count,
        "circle_bbox_width_pixels": circle_bbox_width_pixels,
        "circle_bbox_height_pixels": circle_bbox_height_pixels,
        "circle_bbox_width_height_ratio": circle_bbox_width_height_ratio,
        "circle_geometry_status": circle_geometry_status,
        "error_message": error_message,
    }


def basic_existing_metrics(
    group: pd.DataFrame,
    records_column: str | None,
) -> dict[str, Any]:
    """Collect lightweight QC fields when an existing output is skipped."""
    fractions = finite_numeric_values(group["WD_bin_fraction"])
    nonnegative_fractions = fractions[fractions >= 0.0]
    tower_gpp = resolve_tower_gpp(group)
    radius_m = resolve_radius(group)
    return {
        "n_bins": int(len(group)),
        "wd_fraction_sum": (
            float(math.fsum(nonnegative_fractions))
            if nonnegative_fractions.size
            else None
        ),
        "radius_m_used": radius_m,
        "tower_gpp": tower_gpp,
        "n_valid_records": resolve_record_count(group, records_column),
    }


def invalid_export_flag_reason(group: pd.DataFrame) -> str:
    """Explain why an all-invalid export-flag site-month is not processed."""
    reasons = [
        "valid_for_GEE_export is false/invalid for every row and "
        "REQUIRE_VALID_FOR_GEE_EXPORT is True"
    ]

    if "has_radius" in group.columns:
        has_radius_values = pd.to_numeric(
            group["has_radius"], errors="coerce"
        ).to_numpy(dtype="float64")
        finite_values = has_radius_values[np.isfinite(has_radius_values)]
        if finite_values.size and not bool(np.any(finite_values == 1.0)):
            reasons.append("has_radius=0")

    if not has_any_positive_radius(group):
        reasons.append("radius_m_max and radius_m are missing or not > 0")

    return "; ".join(reasons) + "."


def write_qc_files(
    qc_folder: Path,
    summary_records: list[dict[str, Any]],
    skipped_records: list[dict[str, Any]],
    preserve_unprocessed_existing_rows: bool = False,
) -> tuple[Path, Path]:
    """Write both requested QC CSV files, including headers when empty."""
    qc_folder.mkdir(parents=True, exist_ok=True)
    summary_path = qc_folder / "footprint_tiff_creation_summary.csv"
    skipped_path = qc_folder / "skipped_site_months.csv"

    new_summary = pd.DataFrame(summary_records, columns=SUMMARY_COLUMNS)
    new_skipped = pd.DataFrame(skipped_records, columns=SKIPPED_COLUMNS)

    if (
        preserve_unprocessed_existing_rows
        and summary_path.exists()
        and not new_summary.empty
    ):
        existing_summary = pd.read_csv(
            summary_path,
            encoding="utf-8-sig",
            low_memory=False,
        )
        existing_summary = existing_summary.reindex(columns=SUMMARY_COLUMNS)

        def qc_keys(frame: pd.DataFrame) -> pd.Series:
            site_values = frame["SITE_ID"].fillna("").astype(str).str.strip()
            month_values = frame["YYYYMM"].map(
                lambda value: normalize_yyyymm(value)
                or ("" if is_missing(value) else str(value).strip())
            )
            return site_values + "|" + month_values

        processed_keys = set(qc_keys(new_summary))
        existing_summary = existing_summary.loc[
            ~qc_keys(existing_summary).isin(processed_keys)
        ]
        new_summary = pd.concat(
            [existing_summary, new_summary],
            ignore_index=True,
            sort=False,
        ).reindex(columns=SUMMARY_COLUMNS)

        if skipped_path.exists():
            existing_skipped = pd.read_csv(
                skipped_path,
                encoding="utf-8-sig",
                low_memory=False,
            ).reindex(columns=SKIPPED_COLUMNS)
            existing_skipped = existing_skipped.loc[
                ~qc_keys(existing_skipped).isin(processed_keys)
            ]
            new_skipped = pd.concat(
                [existing_skipped, new_skipped],
                ignore_index=True,
                sort=False,
            ).reindex(columns=SKIPPED_COLUMNS)

    new_summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )
    new_skipped.to_csv(
        skipped_path,
        index=False,
        encoding="utf-8-sig",
    )
    return summary_path, skipped_path


def iter_with_progress(
    groups: Iterable[tuple[tuple[str, str], pd.DataFrame]],
    total: int,
) -> Iterable[tuple[tuple[str, str], pd.DataFrame]]:
    """Use tqdm when installed; otherwise return the normal group iterator."""
    if tqdm is None:
        return groups
    return tqdm(
        groups,
        total=total,
        desc="Creating monthly footprint TIFFs",
        unit="site-month",
        dynamic_ncols=True,
    )


def display_filesystem_path(path: str | Path) -> Path:
    """Remove a Windows extended-length prefix for readable report paths."""
    text = os.fspath(path)
    if text.startswith("\\\\?\\UNC\\"):
        text = "\\\\" + text[8:]
    elif text.startswith("\\\\?\\"):
        text = text[4:]
    return Path(text)


def find_existing_tiffs(output_root: Path) -> list[Path]:
    """Find TIFFs recursively with Windows long-path support."""
    if not path_exists(output_root):
        return []
    tiffs: list[Path] = []
    for directory, _, filenames in os.walk(native_filesystem_path(output_root)):
        for filename in filenames:
            if filename.lower().endswith((".tif", ".tiff")):
                tiffs.append(
                    display_filesystem_path(os.path.join(directory, filename))
                )
    return sorted(tiffs, key=lambda path: str(path).lower())


def crs_is_local_aeqd(crs: Any) -> bool:
    """Check whether a raster CRS is an Azimuthal Equidistant projection."""
    if crs is None:
        return False
    try:
        if str(crs.to_dict().get("proj", "")).lower() == "aeqd":
            return True
    except Exception:
        pass
    crs_text = str(crs).lower()
    return "azimuthal equidistant" in crs_text or "+proj=aeqd" in crs_text


def validation_error_record(tif_path: Path, error: Exception) -> dict[str, Any]:
    """Create a complete validation row when a TIFF cannot be opened/read."""
    record = {column: None for column in EXISTING_TIFF_VALIDATION_COLUMNS}
    record.update(
        {
            "output_tif": str(tif_path),
            "has_six_bands": False,
            "band_descriptions_match": False,
            "band3_is_binary_0_1": False,
            "all_bands_float32": False,
            "crs_is_aeqd": False,
            "circle_geometry_status": "NOT_EVALUATED",
            "validation_status": "ERROR",
            "error_message": f"{type(error).__name__}: {error}",
        }
    )
    return record


def validate_existing_tiffs(
    output_root: str | Path,
) -> tuple[pd.DataFrame, Path]:
    """
    Validate existing six-band TIFFs without modifying them.

    The report checks band count/descriptions, Band 3 binary values and circle
    geometry, AEQD CRS, and float32 dtype.
    """
    root = Path(output_root).expanduser()
    qc_folder = root / "_QC"
    qc_folder.mkdir(parents=True, exist_ok=True)
    report_path = qc_folder / "existing_tiff_validation_report.csv"
    tiff_paths = find_existing_tiffs(root)
    rows: list[dict[str, Any]] = []

    iterator: Iterable[Path]
    if tqdm is None:
        iterator = tiff_paths
    else:
        iterator = tqdm(
            tiff_paths,
            desc="Validating existing TIFFs",
            unit="TIFF",
            dynamic_ncols=True,
        )

    for tif_path in iterator:
        try:
            with rasterio.open(native_filesystem_path(tif_path)) as dataset:
                tags = dataset.tags()
                site_id = tags.get("SITE_ID", "")
                yyyymm = tags.get("YYYYMM", "")
                band_count = int(dataset.count)
                has_six_bands = band_count == len(BAND_DESCRIPTIONS)
                descriptions_match = (
                    tuple(dataset.descriptions) == tuple(BAND_DESCRIPTIONS)
                )
                all_float32 = all(
                    dtype == "float32" for dtype in dataset.dtypes
                )
                aeqd_ok = crs_is_local_aeqd(dataset.crs)
                pixel_size_x = float(abs(dataset.transform.a))
                pixel_size_y = float(abs(dataset.transform.e))

                band3_binary = False
                circle_qa: CircleQAMetrics | None = None
                radius_value: float | None = None
                radius_tag = tags.get("radius_m_used", "")
                try:
                    parsed_radius = float(radius_tag)
                    if math.isfinite(parsed_radius) and parsed_radius > 0.0:
                        radius_value = parsed_radius
                except (TypeError, ValueError, OverflowError):
                    radius_value = None

                if band_count >= 3:
                    circle_band = dataset.read(3)
                    finite_values = circle_band[np.isfinite(circle_band)]
                    band3_binary = bool(
                        finite_values.size
                        and np.all(
                            (finite_values == 0.0) | (finite_values == 1.0)
                        )
                    )
                    if radius_value is not None:
                        circle_qa = calculate_circle_qa(
                            circle_mask=circle_band,
                            transform=dataset.transform,
                            radius_m=radius_value,
                        )

                problems: list[str] = []
                if not has_six_bands:
                    problems.append("MISSING_OR_EXTRA_BANDS")
                if not descriptions_match:
                    problems.append("BAND_DESCRIPTIONS_MISMATCH")
                if not band3_binary:
                    problems.append("BAND3_NOT_BINARY_0_1")
                if not all_float32:
                    problems.append("DTYPE_NOT_FLOAT32")
                if not aeqd_ok:
                    problems.append("CRS_NOT_AEQD")
                if radius_value is None:
                    problems.append("MISSING_RADIUS_TAG")
                elif (
                    circle_qa is not None
                    and circle_qa.circle_geometry_status
                    == "WARNING_CIRCLE_GEOMETRY"
                ):
                    problems.append("WARNING_CIRCLE_GEOMETRY")

                rows.append(
                    {
                        "output_tif": str(tif_path),
                        "SITE_ID": site_id,
                        "YYYYMM": yyyymm,
                        "band_count": band_count,
                        "has_six_bands": has_six_bands,
                        "band_descriptions_match": descriptions_match,
                        "band3_is_binary_0_1": band3_binary,
                        "all_bands_float32": all_float32,
                        "crs_is_aeqd": aeqd_ok,
                        "pixel_size_x_m": pixel_size_x,
                        "pixel_size_y_m": pixel_size_y,
                        "radius_m_used": radius_value,
                        "expected_circle_area_m2": (
                            circle_qa.expected_circle_area_m2
                            if circle_qa
                            else None
                        ),
                        "raster_circle_area_m2": (
                            circle_qa.raster_circle_area_m2
                            if circle_qa
                            else None
                        ),
                        "circle_area_ratio": (
                            circle_qa.circle_area_ratio if circle_qa else None
                        ),
                        "circle_pixel_count": (
                            circle_qa.circle_pixel_count if circle_qa else None
                        ),
                        "circle_bbox_width_pixels": (
                            circle_qa.circle_bbox_width_pixels
                            if circle_qa
                            else None
                        ),
                        "circle_bbox_height_pixels": (
                            circle_qa.circle_bbox_height_pixels
                            if circle_qa
                            else None
                        ),
                        "circle_bbox_width_height_ratio": (
                            circle_qa.circle_bbox_width_height_ratio
                            if circle_qa
                            else None
                        ),
                        "circle_geometry_status": (
                            circle_qa.circle_geometry_status
                            if circle_qa
                            else "NOT_EVALUATED_MISSING_RADIUS"
                        ),
                        "validation_status": "PASS" if not problems else "WARNING",
                        "error_message": "; ".join(problems),
                    }
                )
        except Exception as error:
            rows.append(validation_error_record(tif_path, error))

    report = pd.DataFrame(rows, columns=EXISTING_TIFF_VALIDATION_COLUMNS)
    report.to_csv(report_path, index=False, encoding="utf-8-sig")

    if report.empty:
        n_six_bands = 0
        n_missing_bands = 0
        n_nonbinary = 0
        n_abnormal_circle = 0
        n_non_aeqd = 0
        n_non_float32 = 0
    else:
        n_six_bands = int(report["has_six_bands"].fillna(False).sum())
        n_missing_bands = int(
            (~report["has_six_bands"].fillna(False).astype(bool)).sum()
        )
        n_nonbinary = int(
            (~report["band3_is_binary_0_1"].fillna(False).astype(bool)).sum()
        )
        n_abnormal_circle = int(
            (
                report["circle_geometry_status"]
                == "WARNING_CIRCLE_GEOMETRY"
            ).sum()
        )
        n_non_aeqd = int(
            (~report["crs_is_aeqd"].fillna(False).astype(bool)).sum()
        )
        n_non_float32 = int(
            (~report["all_bands_float32"].fillna(False).astype(bool)).sum()
        )

    print("\n========================================", flush=True)
    print("EXISTING TIFF VALIDATION COMPLETE", flush=True)
    print("========================================", flush=True)
    print(f"Number of TIFFs: {len(report):,}", flush=True)
    print(f"Number with six bands: {n_six_bands:,}", flush=True)
    print(f"Number missing/extra bands: {n_missing_bands:,}", flush=True)
    print(f"Number where Band 3 is not binary 0/1: {n_nonbinary:,}", flush=True)
    print(
        f"Number where Band 3 area ratio is abnormal: "
        f"{n_abnormal_circle:,}",
        flush=True,
    )
    print(f"Number where CRS is not AEQD: {n_non_aeqd:,}", flush=True)
    print(f"Number where dtype is not float32: {n_non_float32:,}", flush=True)
    print(f"Validation report: {report_path}", flush=True)
    return report, report_path


def main() -> None:
    """Run the complete monthly TIFF creation workflow."""
    start_yyyymm, end_yyyymm = validate_settings()
    output_root = Path(OUTPUT_ROOT).expanduser()
    qc_folder = output_root / "_QC"

    print(f"Output root: {output_root}", flush=True)
    if VALIDATE_ONLY:
        run_mode = "VALIDATE ONLY (no TIFF creation)"
    elif DRY_RUN_ONLY:
        run_mode = "DRY RUN (no TIFFs)"
    elif RECREATE_EXISTING:
        run_mode = "REBUILD/CREATE (existing TIFF overwrite enabled)"
    else:
        run_mode = "CREATE MISSING (existing TIFFs retained)"
    print(f"Run mode: {run_mode}", flush=True)
    print(
        f"REQUIRE_VALID_FOR_GEE_EXPORT={REQUIRE_VALID_FOR_GEE_EXPORT}",
        flush=True,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    qc_folder.mkdir(parents=True, exist_ok=True)

    if VALIDATE_ONLY:
        validate_existing_tiffs(output_root)
        return

    qa_png_enabled = CREATE_QA_PNGS and QA_PNG_MAX > 0
    if qa_png_enabled and (plt is None or MatplotlibCircle is None):
        warn(
            "CREATE_QA_PNGS=True but matplotlib is unavailable. TIFF creation "
            "will continue without QA PNGs."
        )
        qa_png_enabled = False
    qa_png_folder = qc_folder / "visual_QA_pngs"

    merged_before_deduplication = load_and_merge_input_tables(INPUT_TABLE_PATHS)
    records_column = validate_input_columns(merged_before_deduplication)
    table, removed_duplicate_rows = deduplicate_merged_table(
        merged_before_deduplication
    )

    coverage_path, merged_sites_path, duplicates_path = write_preflight_reports(
        qc_folder=qc_folder,
        merged_table_before_deduplication=merged_before_deduplication,
        deduplicated_table=table,
        removed_duplicate_rows=removed_duplicate_rows,
    )
    print("\nPreflight QC reports written:", flush=True)
    print(f"  Input coverage: {coverage_path}", flush=True)
    print(f"  Merged valid sites: {merged_sites_path}", flush=True)
    print(f"  Removed duplicates: {duplicates_path}", flush=True)

    table["_VALID_EXPORT_FLAG"] = table["valid_for_GEE_export"].map(is_export_true)
    table["_SITE_ID_NORMALIZED"] = table["SITE_ID"].map(clean_site_id)
    table["_YYYYMM_NORMALIZED"] = table["YYYYMM"].map(normalize_yyyymm)

    summary_records: list[dict[str, Any]] = []
    skipped_records: list[dict[str, Any]] = []

    active_selected_sites = (
        SELECTED_SITES_FOR_REBUILD
        if SELECTED_SITES_FOR_REBUILD is not None
        else SELECTED_SITES
    )
    if active_selected_sites is not None:
        selected_site_ids = {
            str(site).strip()
            for site in active_selected_sites
            if not is_missing(site) and str(site).strip()
        }
        table = table.loc[
            table["_SITE_ID_NORMALIZED"].isin(selected_site_ids)
        ].copy()
        print(
            f"Rows after active site-selection filter: {len(table):,}",
            flush=True,
        )
        print(
            "Selected sites: " + ", ".join(sorted(selected_site_ids)),
            flush=True,
        )

    invalid_keys = table.loc[
        table["_SITE_ID_NORMALIZED"].isna()
        | table["_YYYYMM_NORMALIZED"].isna()
    ]
    if not invalid_keys.empty:
        warn(
            f"{len(invalid_keys):,} row(s) have a missing SITE_ID "
            "or invalid YYYYMM and cannot be grouped."
        )
        invalid_key_groups: dict[tuple[str, str], int] = {}
        for _, row in invalid_keys.iterrows():
            site_text = (
                "<missing>"
                if is_missing(row.get("SITE_ID"))
                else str(row.get("SITE_ID")).strip()
            )
            month_text = (
                "<invalid>"
                if is_missing(row.get("YYYYMM"))
                else str(row.get("YYYYMM")).strip()
            )
            key = (site_text, month_text)
            invalid_key_groups[key] = invalid_key_groups.get(key, 0) + 1

        for (site_text, month_text), row_count in invalid_key_groups.items():
            reason = (
                f"Missing SITE_ID or invalid YYYYMM in {row_count} "
                "row(s)."
            )
            skipped_records.append(
                {"SITE_ID": site_text, "YYYYMM": month_text, "reason": reason}
            )
            summary_records.append(
                make_summary_record(
                    site_id=site_text,
                    yyyymm=month_text,
                    output_tif=None,
                    status="skipped_invalid",
                    n_bins=row_count,
                    error_message=reason,
                )
            )

    scoped_table = table.loc[
        table["_SITE_ID_NORMALIZED"].notna()
        & table["_YYYYMM_NORMALIZED"].notna()
    ].copy()

    if start_yyyymm is not None:
        scoped_table = scoped_table.loc[
            scoped_table["_YYYYMM_NORMALIZED"] >= start_yyyymm
        ].copy()
    if end_yyyymm is not None:
        scoped_table = scoped_table.loc[
            scoped_table["_YYYYMM_NORMALIZED"] <= end_yyyymm
        ].copy()
    if start_yyyymm is not None or end_yyyymm is not None:
        print(
            f"Rows after YYYYMM filter: {len(scoped_table):,}",
            flush=True,
        )

    grouped = scoped_table.groupby(
        ["_SITE_ID_NORMALIZED", "_YYYYMM_NORMALIZED"],
        sort=True,
        dropna=False,
    )
    n_site_months = int(grouped.ngroups)
    n_scoped_sites = int(scoped_table["_SITE_ID_NORMALIZED"].nunique())

    valid_rows = scoped_table.loc[scoped_table["_VALID_EXPORT_FLAG"]]
    n_valid_sites = int(valid_rows["_SITE_ID_NORMALIZED"].nunique())
    n_valid_site_months = int(
        valid_rows[
            ["_SITE_ID_NORMALIZED", "_YYYYMM_NORMALIZED"]
        ].drop_duplicates().shape[0]
    )

    valid_flag_by_site_month = grouped["_VALID_EXPORT_FLAG"].any()
    n_invalid_flag_site_months = int((~valid_flag_by_site_month).sum())

    print(f"\nScoped unique sites: {n_scoped_sites:,}", flush=True)
    print(f"Scoped site-months: {n_site_months:,}", flush=True)
    print(f"Valid unique sites: {n_valid_sites:,}", flush=True)
    print(f"Valid site-months: {n_valid_site_months:,}", flush=True)
    if REQUIRE_VALID_FOR_GEE_EXPORT:
        print(
            "Invalid/skipped site-months due to valid_for_GEE_export: "
            f"{n_invalid_flag_site_months:,}",
            flush=True,
        )
    else:
        print(
            "Invalid export-flag site-months allowed by current setting: "
            f"{n_invalid_flag_site_months:,}",
            flush=True,
        )

    if records_column is None:
        warn(
            "No valid-WD-record-count column was found. Band 5 will be "
            "-9999 for every newly written TIFF."
        )
    else:
        print(
            f"WD record-count source: {records_column}",
            flush=True,
        )

    attempted = len(summary_records)
    successfully_written = 0
    would_create = 0
    skipped_existing = 0
    recreated_existing = 0
    skipped_invalid = len(summary_records)
    failed = 0
    processed_group_count = 0
    qa_pngs_created = 0

    group_iterator = iter_with_progress(grouped, n_site_months)
    for (site_id, yyyymm), group in group_iterator:
        if (
            MAX_SITE_MONTHS is not None
            and processed_group_count >= MAX_SITE_MONTHS
        ):
            print(
                f"MAX_SITE_MONTHS={MAX_SITE_MONTHS} reached; stopping.",
                flush=True,
            )
            break

        processed_group_count += 1
        attempted += 1
        site_id = str(site_id)
        yyyymm = str(yyyymm)
        provisional_path = provisional_output_path(site_id, yyyymm, output_root)

        print("", flush=True)
        print(f"Processing SITE_ID={site_id}, YYYYMM={yyyymm}", flush=True)
        print(f"Output path: {provisional_path}", flush=True)

        if REQUIRE_VALID_FOR_GEE_EXPORT:
            group_to_process = group.loc[group["_VALID_EXPORT_FLAG"]].copy()
        else:
            group_to_process = group.copy()

        if group_to_process.empty:
            reason = invalid_export_flag_reason(group)
            warn(f"{site_id} {yyyymm}: skipped site-month: {reason}")
            skipped_invalid += 1
            skipped_records.append(
                {
                    "SITE_ID": site_id,
                    "YYYYMM": yyyymm,
                    "reason": reason,
                }
            )
            metrics = basic_existing_metrics(group, records_column)
            summary_records.append(
                make_summary_record(
                    site_id=site_id,
                    yyyymm=yyyymm,
                    output_tif=provisional_path,
                    status="skipped_invalid",
                    error_message=reason,
                    **metrics,
                )
            )
            continue

        try:
            prepared = prepare_site_month(
                group=group_to_process,
                site_id=site_id,
                yyyymm=yyyymm,
                records_column=records_column,
            )
            output_path = output_path_for(prepared, output_root)
            output_already_exists = path_exists(output_path)

            for warning_message in prepared.warnings:
                warn(warning_message)

            if (
                output_already_exists
                and not RECREATE_EXISTING
                and SKIP_EXISTING
            ):
                print("Status: skipped existing TIFF.", flush=True)
                skipped_existing += 1
                reason = "Output TIFF exists and SKIP_EXISTING is True."
                skipped_records.append(
                    {
                        "SITE_ID": site_id,
                        "YYYYMM": yyyymm,
                        "reason": reason,
                    }
                )
                summary_records.append(
                    make_summary_record(
                        site_id=prepared.site_id,
                        yyyymm=prepared.yyyymm,
                        output_tif=output_path,
                        status="skipped_existing",
                        n_bins=prepared.n_bins,
                        wd_fraction_sum=prepared.wd_fraction_sum,
                        radius_m_used=prepared.radius_m,
                        tower_gpp=prepared.tower_gpp,
                        n_valid_records=prepared.n_valid_wd_records,
                        error_message="; ".join(prepared.warnings),
                    )
                )
                continue

            if DRY_RUN_ONLY:
                would_create += 1
                dry_run_action = (
                    "recreate"
                    if output_already_exists and RECREATE_EXISTING
                    else "create"
                )
                status = (
                    f"dry_run_would_{dry_run_action}_with_warning"
                    if prepared.warnings
                    else f"dry_run_would_{dry_run_action}"
                )
                print(f"Status: {status}; no TIFF written.", flush=True)
                summary_records.append(
                    make_summary_record(
                        site_id=prepared.site_id,
                        yyyymm=prepared.yyyymm,
                        output_tif=output_path,
                        status=status,
                        n_bins=prepared.n_bins,
                        wd_fraction_sum=prepared.wd_fraction_sum,
                        radius_m_used=prepared.radius_m,
                        tower_gpp=prepared.tower_gpp,
                        n_valid_records=prepared.n_valid_wd_records,
                        error_message="; ".join(prepared.warnings),
                    )
                )
                continue

            qa_png_path: Path | None = None
            if qa_png_enabled and qa_pngs_created < QA_PNG_MAX:
                qa_png_path = qa_png_folder / (
                    f"{prepared.site_id_safe}_{prepared.yyyymm}_"
                    "footprint_and_circle_QA.png"
                )

            raster_result = write_geotiff(
                prepared,
                output_path,
                qa_png_path=qa_png_path,
            )
            if qa_png_path is not None and path_exists(qa_png_path):
                qa_pngs_created += 1

            all_warnings = prepared.warnings + raster_result.warnings
            for warning_message in raster_result.warnings:
                warn(warning_message)

            if output_already_exists and RECREATE_EXISTING:
                recreated_existing += 1
                status_base = "rewritten"
            else:
                status_base = "written"
            status = (
                f"{status_base}_with_warning" if all_warnings else status_base
            )
            successfully_written += 1
            print(f"Status: {status}.", flush=True)

            summary_records.append(
                make_summary_record(
                    site_id=prepared.site_id,
                    yyyymm=prepared.yyyymm,
                    output_tif=output_path,
                    status=status,
                    n_bins=prepared.n_bins,
                    wd_fraction_sum=prepared.wd_fraction_sum,
                    radius_m_used=prepared.radius_m,
                    tower_gpp=prepared.tower_gpp,
                    n_valid_records=prepared.n_valid_wd_records,
                    footprint_min=raster_result.footprint_min,
                    footprint_max=raster_result.footprint_max,
                    footprint_mean=raster_result.footprint_mean,
                    blind_pixel_count=raster_result.blind_buffer_pixel_count,
                    pixel_size_m=raster_result.pixel_size_m,
                    expected_circle_area_m2=(
                        raster_result.expected_circle_area_m2
                    ),
                    raster_circle_area_m2=raster_result.raster_circle_area_m2,
                    circle_area_ratio=raster_result.circle_area_ratio,
                    circle_pixel_count=raster_result.circle_pixel_count,
                    circle_bbox_width_pixels=(
                        raster_result.circle_bbox_width_pixels
                    ),
                    circle_bbox_height_pixels=(
                        raster_result.circle_bbox_height_pixels
                    ),
                    circle_bbox_width_height_ratio=(
                        raster_result.circle_bbox_width_height_ratio
                    ),
                    circle_geometry_status=(
                        raster_result.circle_geometry_status
                    ),
                    error_message="; ".join(all_warnings),
                )
            )

        except InvalidSiteMonthError as error:
            reason = str(error)
            warn(f"{site_id} {yyyymm}: skipped invalid site-month: {reason}")
            skipped_invalid += 1
            skipped_records.append(
                {"SITE_ID": site_id, "YYYYMM": yyyymm, "reason": reason}
            )
            summary_records.append(
                make_summary_record(
                    site_id=site_id,
                    yyyymm=yyyymm,
                    output_tif=provisional_path,
                    status="skipped_invalid",
                    n_bins=len(group_to_process),
                    error_message=reason,
                )
            )

        except Exception as error:
            failed += 1
            error_text = f"{type(error).__name__}: {error}"
            print(
                f"ERROR: {site_id} {yyyymm} failed: {error_text}",
                flush=True,
            )
            traceback.print_exc()
            summary_records.append(
                make_summary_record(
                    site_id=site_id,
                    yyyymm=yyyymm,
                    output_tif=provisional_path,
                    status="failed",
                    n_bins=len(group_to_process),
                    error_message=error_text,
                )
            )

    summary_path, skipped_path = write_qc_files(
        qc_folder,
        summary_records,
        skipped_records,
        preserve_unprocessed_existing_rows=bool(
            active_selected_sites is not None
            or START_YYYYMM is not None
            or END_YYYYMM is not None
            or MAX_SITE_MONTHS is not None
        ),
    )

    print("", flush=True)
    print("========================================", flush=True)
    if DRY_RUN_ONLY:
        print("MONTHLY FOOTPRINT TIFF DRY RUN COMPLETE", flush=True)
    else:
        print("MONTHLY FOOTPRINT TIFF CREATION COMPLETE", flush=True)
    print("========================================", flush=True)
    print(f"Total site-months attempted: {attempted:,}", flush=True)
    if DRY_RUN_ONLY:
        print(f"TIFFs that would be created: {would_create:,}", flush=True)
    print(f"Successfully written: {successfully_written:,}", flush=True)
    print(f"Existing TIFFs recreated: {recreated_existing:,}", flush=True)
    print(f"Skipped existing: {skipped_existing:,}", flush=True)
    print(f"Invalid/skipped site-months: {skipped_invalid:,}", flush=True)
    print(f"Failed: {failed:,}", flush=True)
    print(f"Output root: {output_root}", flush=True)
    print(f"QC summary: {summary_path}", flush=True)
    print(f"QC skipped list: {skipped_path}", flush=True)
    print(f"QC input coverage: {coverage_path}", flush=True)
    print(f"QC merged valid sites: {merged_sites_path}", flush=True)
    print(f"QC removed duplicates: {duplicates_path}", flush=True)
    if qa_png_enabled:
        print(f"Visual QA PNGs created: {qa_pngs_created:,}", flush=True)
        print(f"Visual QA PNG folder: {qa_png_folder}", flush=True)


if __name__ == "__main__":
    main()
