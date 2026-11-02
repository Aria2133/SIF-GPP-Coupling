# Install required packages, if needed:
# pip install pandas numpy openpyxl xlrd

"""
Finalize defensible FLUXNET metadata recovery without creating GeoTIFFs.

Safety
------
* DRY_RUN=True, WRITE_REPAIRED_CSV=False, and CREATE_TIFFS=False by default.
* Original final CSVs are never overwritten.
* No default, random, or unproven radius is permitted.
* Manual web research is candidate evidence, not automatic truth.
* FETCH fields are reported but never substituted for the project's radius method.
* This script contains no TIFF creation call.

project-confirmed method
--------------------
* Actual daytime half-hourly/hourly WD records are aggregated into monthly
  10-degree bins.
* WD_bin_center_deg is used directly as the upwind wedge bearing.
* WD_bin_fraction weights each wedge; monthly mean/circular-mean WD is
  diagnostic only and never defines footprint geometry.
* No 180-degree rotation is applied.
* Radius priority is trusted explicit radius, then
  (EC measurement height - canopy height) * 100, then trusted explicit
  height_above_canopy_m * 100.
* The circular buffer remains a comparison band only.
"""

from __future__ import annotations
# Original script filename (sanitized): Finalize_FLUXNET_Metadata_Recovery_And_Method_Compliance.py; public name: 04a_Finalize_FLUXNET_Metadata_Recovery_And_Method_Compliance.py
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

import io
import math
import os
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

import Audit_FLUXNET_Footprint_Input_Completeness as completeness_audit
import Repair_FLUXNET_Final_Monthly_GEE_Input_Tables as repair


# =============================================================================
# EDITABLE SETTINGS
# =============================================================================

DRY_RUN = False
WRITE_REPAIRED_CSV = True
CREATE_TIFFS = False
APPLY_MEDIUM_CONFIDENCE_REPAIRS = False

PROJECT_ROOT = Path(
    PROJECT_ROOT
)

FLUXNET_ROOT = PROJECT_ROOT / "fluxnet"
REMAINING_ROOT = PROJECT_ROOT / "Remaining Fluxnet Sites"

ORIGINAL_FINAL_CSV = (
    FLUXNET_ROOT
    / "Hourly Data (for Wind Direction)"
    / "final_monthly_GEE_inputs"
    / "ALL_SITES_monthly_WD_histogram_GPP_footprint_input_for_GEE.csv"
)
REMAINING_FINAL_CSV = (
    REMAINING_ROOT
    / "HH files"
    / "final_monthly_GEE_inputs"
    / "ALL_SITES_monthly_WD_histogram_GPP_footprint_input_for_GEE.csv"
)
INPUT_TABLE_PATHS = [ORIGINAL_FINAL_CSV, REMAINING_FINAL_CSV]

PREVIOUS_GPP_REPAIRED_CSV_CANDIDATES = [
    PROJECT_ROOT
    / "repaired_final_monthly_GEE_inputs"
    / "repaired_ALL_SITES_monthly_WD_histogram_GPP_footprint_input_for_GEE.csv",
]

MANUAL_METADATA_DRAFT = (
    PROJECT_ROOT / "manual_metadata_patch_web_research_draft.xlsx"
)

OUTPUT_ROOT = PROJECT_ROOT / "finalized_metadata_recovery"
QC_OUTPUT = OUTPUT_ROOT / "_QC"

TIFF_CREATION_SCRIPT = (
    (Path(PROJECT_ROOT) / 'Codes/pythonProject/Finalize_FLUXNET_Metadata_Recovery_And_Method_Compliance.py').resolve().parent
    / "Create_Monthly_FLUXNET_Footprint_TIFFs_From_Final_CSV.py"
)

MAX_LOCAL_METADATA_FILE_MB = 50.0
MISSING_VALUE = -9999.0
WD_BIN_WIDTH_REQUIRED_DEG = 10.0


# =============================================================================
# BLOCKED SITE COHORT
# =============================================================================

RADIUS_MISSING_ONLY = [
    "BE-Maa",
    "BR-Ma2",
    "CA-Gro",
    "ES-Agu",
    "ES-LM1",
    "ES-LM2",
    "FR-Aur",
    "IT-Cp2",
    "JP-Tak",
    "US-Akn",
    "US-Ho2",
    "US-ICh",
    "US-ICt",
    "US-Jo1",
    "US-Rms",
    "US-Ses",
    "US-UiB",
    "US-UiC",
    "US-Wjs",
]

LATLON_AND_RADIUS_MISSING = [
    "AT-Zoe",
    "CN-Ash",
    "CN-BeO",
    "CN-Dmn",
    "CN-Dzo",
    "CN-Lch",
    "CN-Sdq",
    "US-Rls",
    "US-TKs",
    "US-UiA",
]

BE_LON_SITE = "BE-Lon"
BLOCKED_METADATA_SITES = sorted(
    set(RADIUS_MISSING_ONLY) | set(LATLON_AND_RADIUS_MISSING)
)
ALL_TARGET_SITES = sorted(set(BLOCKED_METADATA_SITES) | {BE_LON_SITE})


# =============================================================================
# REPORT SCHEMAS
# =============================================================================

LOCAL_CANDIDATE_COLUMNS = [
    "SITE_ID",
    "source_type",
    "source_path",
    "file_inside_zip",
    "field_names_found",
    "lat_candidate",
    "lon_candidate",
    "canopy_height_m_candidate",
    "EC_measurement_height_m_candidate",
    "height_above_canopy_m_candidate",
    "explicit_radius_m_candidate",
    "FETCH_70",
    "FETCH_80",
    "FETCH_90",
    "FETCH_FILTER",
    "FETCH_MAX",
    "candidate_confidence",
    "candidate_reason",
    "usable_for_auto_repair",
    "why_not_usable",
]

RADIUS_DECISION_COLUMNS = [
    "SITE_ID",
    "problem_type",
    "selected_lat",
    "selected_lon",
    "selected_canopy_height_m",
    "selected_EC_measurement_height_m",
    "selected_height_above_canopy_m",
    "selected_radius_m",
    "selected_radius_m_min",
    "selected_radius_m_max",
    "radius_decision_priority",
    "radius_source",
    "latlon_source",
    "confidence",
    "auto_apply",
    "manual_review_required",
    "FETCH_70",
    "FETCH_80",
    "FETCH_90",
    "FETCH_FILTER",
    "FETCH_MAX",
    "decision_notes",
]

COMPLIANCE_COLUMNS = [
    "SITE_ID",
    "YYYYMM",
    "uses_30min_or_hourly_WD_records",
    "WD_bin_width_deg",
    "n_WD_bins",
    "WD_bin_fraction_sum",
    "uses_monthly_mean_WD_for_geometry",
    "uses_daily_mean_WD_for_geometry",
    "monthly_circular_mean_WD_diagnostic_only",
    "wedge_geometry_uses_WD_bin_center",
    "applies_180_degree_rotation",
    "radius_method",
    "radius_m_used",
    "radius_source",
    "GPP_column_used",
    "GPP_quality_flag",
    "GPP_QC_filter_used",
    "FETCH_FILTER_available",
    "FETCH_FILTER_used",
    "compliance_status",
    "compliance_notes",
]

PROVENANCE_COLUMNS = [
    "final_metadata_recovery_applied",
    "final_latlon_recovery_applied",
    "final_radius_recovery_applied",
    "final_metadata_source",
    "final_metadata_confidence",
    "final_radius_method",
    "final_recovery_notes",
]


# =============================================================================
# CANDIDATE MODEL AND GENERAL HELPERS
# =============================================================================


@dataclass
class MetadataCandidate:
    """One exact-SITE_ID metadata evidence record."""

    report: dict[str, Any]
    canopy_values: list[float] = field(default_factory=list)
    ec_height_values: list[float] = field(default_factory=list)
    height_above_values: list[float] = field(default_factory=list)
    explicit_radius_values: list[float] = field(default_factory=list)
    confidence: str = "LOW"
    source_priority: int = 99
    lat_auto_allowed: bool = False
    radius_auto_allowed: bool = False
    radius_derivation_allowed: bool = False
    radius_method_hint: str = ""


def native_path(path: Path) -> str:
    """Return a Windows long-path-safe filesystem string."""
    return completeness_audit.native_filesystem_path(path)


def path_exists(path: Path) -> bool:
    """Check a path with long-path support."""
    return completeness_audit.path_exists(path)


def warn(message: str) -> None:
    """Print a visible warning."""
    print(f"WARNING: {message}", flush=True)


def is_missing(value: Any) -> bool:
    """Use the audit module's robust missing-value test."""
    return completeness_audit.is_missing(value)


def true_value(value: Any) -> bool:
    """Interpret explicit approval-style true values only."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(math.isfinite(float(value)) and float(value) == 1.0)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def valid_lat_lon(lat: Any, lon: Any) -> bool:
    """Validate one coordinate pair."""
    return repair.valid_lat_lon(lat, lon)


def positive(value: Any) -> float | None:
    """Return a positive finite scalar or None."""
    return repair.positive_number(value)


def normalize_yyyymm(value: Any) -> str | None:
    """Normalize a YYYYMM key."""
    return completeness_audit.normalize_yyyymm(value)


def unique_text(values: Iterable[Any]) -> str:
    """Join unique nonempty values deterministically."""
    cleaned = {
        str(value).strip()
        for value in values
        if not is_missing(value) and str(value).strip()
    }
    return " | ".join(sorted(cleaned))


def first_meaningful(
    frame: pd.DataFrame,
    column: str,
    default: Any = "",
) -> Any:
    """Return the first nonmissing value from a frame column."""
    if column not in frame:
        return default
    return repair.first_non_missing(frame[column], default)


def numeric_tokens(value: Any) -> list[float]:
    """Parse finite numeric values/ranges while rejecting FLUXNET sentinels."""
    if is_missing(value):
        return []
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if math.isfinite(number) and number != MISSING_VALUE:
            return [number]
        return []
    text = (
        str(value)
        .strip()
        .replace(",", ".")
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
    )
    values: list[float] = []
    for token in re.findall(r"(?<!\d)-?\d+(?:\.\d+)?", text):
        try:
            number = float(token)
        except ValueError:
            continue
        if math.isfinite(number) and number != MISSING_VALUE:
            values.append(number)
    return values


def positive_values(values: Iterable[Any]) -> list[float]:
    """Collect unique positive numbers from values or ranges."""
    numbers: set[float] = set()
    for value in values:
        for number in numeric_tokens(value):
            if number > 0.0:
                numbers.add(float(number))
    return sorted(numbers)


def finite_values(values: Iterable[Any]) -> list[float]:
    """Collect unique finite non-sentinel values."""
    numbers: set[float] = set()
    for value in values:
        for number in numeric_tokens(value):
            if math.isfinite(number) and number != MISSING_VALUE:
                numbers.add(float(number))
    return sorted(numbers)


def report_value(values: list[float]) -> float | str:
    """Represent one value directly or a compact min/max range."""
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    return f"{min(values):.10g} | {max(values):.10g}"


def confidence_from_text(value: Any) -> str:
    """Normalize free-text confidence labels."""
    text = str(value).strip().upper()
    if "HIGH" in text:
        return "HIGH"
    if "MEDIUM" in text:
        return "MEDIUM"
    return "LOW"


def confidence_allows(level: str) -> bool:
    """Apply HIGH by default and MEDIUM only with the explicit setting."""
    return level == "HIGH" or (
        level == "MEDIUM" and APPLY_MEDIUM_CONFIDENCE_REPAIRS
    )


def exact_site_in_name(path_or_member: str, site_id: str) -> bool:
    """Match an exact FLUXNET SITE_ID token in a filename/path."""
    pattern = rf"(?<![A-Za-z0-9]){re.escape(site_id)}(?![A-Za-z0-9])"
    return re.search(pattern, path_or_member, flags=re.IGNORECASE) is not None


def make_report_record(
    site_id: str,
    source_type: str,
    source_path: str,
    file_inside_zip: str = "",
    fields: Iterable[str] = (),
    lat: Any = "",
    lon: Any = "",
    canopy_values: list[float] | None = None,
    ec_values: list[float] | None = None,
    height_values: list[float] | None = None,
    radius_values: list[float] | None = None,
    fetch: dict[str, Any] | None = None,
    confidence: str = "LOW",
    reason: str = "",
    usable: bool = False,
    why_not: str = "",
) -> dict[str, Any]:
    """Build exactly the requested local-candidate report schema."""
    canopy_values = canopy_values or []
    ec_values = ec_values or []
    height_values = height_values or []
    radius_values = radius_values or []
    fetch = fetch or {}
    return {
        "SITE_ID": site_id,
        "source_type": source_type,
        "source_path": source_path,
        "file_inside_zip": file_inside_zip,
        "field_names_found": unique_text(fields),
        "lat_candidate": lat,
        "lon_candidate": lon,
        "canopy_height_m_candidate": report_value(canopy_values),
        "EC_measurement_height_m_candidate": report_value(ec_values),
        "height_above_canopy_m_candidate": report_value(height_values),
        "explicit_radius_m_candidate": report_value(radius_values),
        "FETCH_70": fetch.get("FETCH_70", ""),
        "FETCH_80": fetch.get("FETCH_80", ""),
        "FETCH_90": fetch.get("FETCH_90", ""),
        "FETCH_FILTER": fetch.get("FETCH_FILTER", ""),
        "FETCH_MAX": fetch.get("FETCH_MAX", ""),
        "candidate_confidence": confidence,
        "candidate_reason": reason,
        "usable_for_auto_repair": bool(usable),
        "why_not_usable": why_not,
    }


# =============================================================================
# BASE TABLE: ORIGINAL INPUTS PLUS THE VERIFIED GPP REPAIR
# =============================================================================


def ensure_repair_provenance_columns(table: pd.DataFrame) -> None:
    """Ensure columns expected by the previous repair module exist."""
    defaults: dict[str, Any] = {
        "repaired_lat_lon": False,
        "repaired_radius": False,
        "repaired_GPP": False,
        "repair_source": "",
        "repair_notes": "",
    }
    for column, default in defaults.items():
        if column not in table:
            table[column] = default


def source_group_from_path(value: Any) -> str:
    """Recover original/remaining provenance from a source-table path."""
    text = str(value).lower()
    if "remaining fluxnet sites" in text:
        return "remaining"
    return "original"


def load_previous_repaired_csv(path: Path) -> pd.DataFrame:
    """Load a previous repaired merged copy without modifying it."""
    table = completeness_audit.read_csv_robust(path)
    table.columns = [str(column).strip().lstrip("\ufeff") for column in table]
    table["SITE_ID"] = table["SITE_ID"].astype(str).str.strip()
    table["YYYYMM"] = table["YYYYMM"].map(normalize_yyyymm)
    table = table.loc[table["YYYYMM"].notna()].copy()
    table["YYYYMM"] = table["YYYYMM"].astype(str)
    if "source_input_table" not in table:
        table["source_input_table"] = ""
    if "_source_group" not in table:
        table["_source_group"] = table["source_input_table"].map(
            source_group_from_path
        )
    ensure_repair_provenance_columns(table)
    repair.recalculate_export_flags(table)
    return table.reset_index(drop=True)


def reconstruct_gpp_repaired_table(
    original_merged: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, repair.GPPRepairResult]]:
    """Re-run only the verified GPP repair when no repaired merged copy exists."""
    print(
        "No previous repaired merged CSV found; rebuilding verified GPP repairs "
        "in memory.",
        flush=True,
    )
    raw_inventory = repair.load_raw_file_inventory()
    results: dict[str, repair.GPPRepairResult] = {}
    for index, site_id in enumerate(repair.ALL_GPP_REPAIR_SITES, start=1):
        print(
            f"  GPP repair {index}/{len(repair.ALL_GPP_REPAIR_SITES)}: "
            f"{site_id}",
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
                "notes": "No exact-site raw HH/HR file was found.",
            }
            results[site_id] = repair.GPPRepairResult(
                site_id,
                pd.DataFrame(),
                report,
            )
        else:
            results[site_id] = repair.aggregate_monthly_gpp_from_raw(
                site_id,
                raw_path,
            )
    repaired = original_merged.copy()
    repair.apply_gpp_repairs(repaired, results)
    repair.recalculate_export_flags(repaired)
    return repaired, results


def load_base_table() -> tuple[pd.DataFrame, list[str], list[str], str]:
    """Load the best available 57-site GPP-repaired base table."""
    original_merged, original_columns, remaining_columns = (
        repair.load_final_input_tables()
    )
    for candidate in PREVIOUS_GPP_REPAIRED_CSV_CANDIDATES:
        if path_exists(candidate):
            print(f"Using previous GPP-repaired merged CSV: {candidate}", flush=True)
            return (
                load_previous_repaired_csv(candidate),
                original_columns,
                remaining_columns,
                str(candidate),
            )
    repaired, _ = reconstruct_gpp_repaired_table(original_merged)
    return (
        repaired,
        original_columns,
        remaining_columns,
        "in_memory_verified_GPP_repair",
    )


# =============================================================================
# MANUAL WEB-RESEARCH DRAFT
# =============================================================================


def load_manual_draft_candidates() -> list[MetadataCandidate]:
    """Load manual Excel rows under explicit confidence/approval constraints."""
    if not path_exists(MANUAL_METADATA_DRAFT):
        warn(f"Manual metadata draft was not found: {MANUAL_METADATA_DRAFT}")
        return []
    workbook = pd.ExcelFile(native_path(MANUAL_METADATA_DRAFT), engine="openpyxl")
    selected_sheet: str | None = None
    for sheet in workbook.sheet_names:
        header = pd.read_excel(workbook, sheet_name=sheet, nrows=0)
        if "SITE_ID" in header.columns:
            selected_sheet = sheet
            break
    if selected_sheet is None:
        warn("No SITE_ID sheet was found in the manual metadata draft.")
        return []

    table = pd.read_excel(workbook, sheet_name=selected_sheet)
    table.columns = [str(column).strip() for column in table]
    for column in (
        "web_lat",
        "web_lon",
        "lat",
        "lon",
        "source_url",
        "metadata_source",
        "source_name",
        "confidence",
        "evidence_note",
        "radius_m_final_to_apply",
        "height_above_canopy_m_user",
        "canopy_height_candidate_m",
        "EC_height_candidate_m",
        "apply_lat_lon",
        "apply_radius",
        "safe_to_auto_apply",
        "notes",
    ):
        if column not in table:
            table[column] = np.nan

    candidates: list[MetadataCandidate] = []
    for row in table.to_dict("records"):
        site_id = completeness_audit.clean_site_id(row.get("SITE_ID"))
        if site_id not in ALL_TARGET_SITES:
            continue
        lat = row.get("web_lat")
        lon = row.get("web_lon")
        if not valid_lat_lon(lat, lon):
            lat = row.get("lat")
            lon = row.get("lon")
        if not valid_lat_lon(lat, lon):
            lat = ""
            lon = ""

        confidence = confidence_from_text(row.get("confidence"))
        safe = true_value(row.get("safe_to_auto_apply"))
        apply_lat = true_value(row.get("apply_lat_lon"))
        apply_radius = true_value(row.get("apply_radius"))
        radius_values = positive_values(
            [row.get("radius_m_final_to_apply")]
        )
        canopy_values = positive_values(
            [row.get("canopy_height_candidate_m")]
        )
        ec_values = positive_values([row.get("EC_height_candidate_m")])
        height_values = positive_values(
            [row.get("height_above_canopy_m_user")]
        )

        # Coordinates may be used when explicitly HIGH and marked for
        # application, or when the row is globally marked safe.
        lat_allowed = bool(
            valid_lat_lon(lat, lon)
            and (confidence == "HIGH" or safe)
            and (apply_lat or safe)
        )
        # Manual radius requires both a positive final value and explicit safe
        # approval. Height candidates alone never become an automatic radius.
        radius_allowed = bool(radius_values and safe and apply_radius)
        usable = lat_allowed or radius_allowed
        why_not: list[str] = []
        if valid_lat_lon(lat, lon) and not lat_allowed:
            why_not.append(
                "coordinates are not both HIGH/safe and approved for application"
            )
        if radius_values and not radius_allowed:
            why_not.append(
                "manual radius is not explicitly safe_to_auto_apply + apply_radius"
            )
        if not radius_values:
            why_not.append("no approved final radius is present")

        source = (
            str(row.get("source_url")).strip()
            if not is_missing(row.get("source_url"))
            else str(MANUAL_METADATA_DRAFT)
        )
        reason = unique_text(
            [
                row.get("evidence_note"),
                row.get("notes"),
                row.get("source_name"),
                row.get("metadata_source"),
            ]
        )
        fields = [
            name
            for name in (
                "web_lat" if valid_lat_lon(lat, lon) else "",
                "web_lon" if valid_lat_lon(lat, lon) else "",
                "canopy_height_candidate_m" if canopy_values else "",
                "EC_height_candidate_m" if ec_values else "",
                "height_above_canopy_m_user" if height_values else "",
                "radius_m_final_to_apply" if radius_values else "",
            )
            if name
        ]
        report = make_report_record(
            site_id=site_id,
            source_type="manual_web_research_draft",
            source_path=str(MANUAL_METADATA_DRAFT),
            file_inside_zip=selected_sheet,
            fields=fields,
            lat=lat,
            lon=lon,
            canopy_values=canopy_values,
            ec_values=ec_values,
            height_values=height_values,
            radius_values=radius_values,
            confidence=confidence,
            reason=reason,
            usable=usable,
            why_not="; ".join(why_not),
        )
        candidates.append(
            MetadataCandidate(
                report=report,
                canopy_values=canopy_values,
                ec_height_values=ec_values,
                height_above_values=height_values,
                explicit_radius_values=radius_values,
                confidence=confidence,
                source_priority=20,
                lat_auto_allowed=lat_allowed,
                radius_auto_allowed=radius_allowed,
                radius_derivation_allowed=False,
                radius_method_hint="manual_approved_explicit_radius",
            )
        )
    return candidates


# =============================================================================
# ZIP BIF/BADM MINING
# =============================================================================


BIF_CANOPY_FIELDS = {
    "HEIGHTC",
    "CANOPY_HEIGHT",
    "VEGETATION_HEIGHT",
    "VEG_HEIGHT",
}
BIF_HEIGHT_ABOVE_FIELDS = {
    "HEIGHT_ABOVE_CANOPY",
    "HEIGHT_ABOVE_CANOPY_M",
    "EC_HEIGHT_ABOVE_CANOPY",
}
BIF_EXPLICIT_RADIUS_FIELDS = {
    "RADIUS_M",
    "RADIUS_M_MAX",
    "FOOTPRINT_RADIUS",
    "FOOTPRINT_RADIUS_M",
    "CIRCLE_BUFFER_RADIUS_M",
}
BIF_FETCH_FIELDS = {
    "FETCH_70",
    "FETCH_80",
    "FETCH_90",
    "FETCH_FILTER",
    "FETCH_MAX",
}
EC_HEIGHT_VARIABLE_PRIORITY = [
    "NEE_CUT_REF",
    "NEE_VUT_REF",
    "USTAR",
    "FC",
    "FC_RAW",
    "H_F_MDS",
    "LE_F_MDS",
]
SECONDARY_HEIGHT_VARIABLES = ["WD", "WS", "WS_F"]


def long_bif_values(table: pd.DataFrame, fields: set[str]) -> list[float]:
    """Extract positive values for exact BIF VARIABLE names."""
    if not {"VARIABLE", "DATAVALUE"}.issubset(table.columns):
        return []
    variable = table["VARIABLE"].fillna("").astype(str).str.strip().str.upper()
    return positive_values(table.loc[variable.isin(fields), "DATAVALUE"])


def long_bif_finite_values(
    table: pd.DataFrame,
    fields: set[str],
) -> list[float]:
    """Extract finite signed values such as longitude."""
    if not {"VARIABLE", "DATAVALUE"}.issubset(table.columns):
        return []
    variable = table["VARIABLE"].fillna("").astype(str).str.strip().str.upper()
    return finite_values(table.loc[variable.isin(fields), "DATAVALUE"])


def extract_bifvar_height(
    table: pd.DataFrame,
) -> tuple[list[float], str, str]:
    """Find explicit EC-flux VAR_INFO_HEIGHT values by variable priority."""
    required = {"GROUP_ID", "VARIABLE", "DATAVALUE"}
    if not required.issubset(table.columns):
        return [], "", ""
    pivot = table.pivot_table(
        index=["SITE_ID", "GROUP_ID"],
        columns="VARIABLE",
        values="DATAVALUE",
        aggfunc="first",
    ).reset_index()
    if "VAR_INFO_VARNAME" not in pivot or "VAR_INFO_HEIGHT" not in pivot:
        return [], "", ""
    variable_names = (
        pivot["VAR_INFO_VARNAME"].fillna("").astype(str).str.strip().str.upper()
    )
    for variable in EC_HEIGHT_VARIABLE_PRIORITY:
        values = positive_values(
            pivot.loc[
                variable_names.eq(variable),
                "VAR_INFO_HEIGHT",
            ]
        )
        if values:
            return values, variable, "explicit_EC_flux_VAR_INFO_HEIGHT"
    for variable in SECONDARY_HEIGHT_VARIABLES:
        values = positive_values(
            pivot.loc[
                variable_names.eq(variable),
                "VAR_INFO_HEIGHT",
            ]
        )
        if values:
            return values, variable, "wind_instrument_VAR_INFO_HEIGHT"
    return [], "", ""


def read_zip_csv(archive: zipfile.ZipFile, member: str) -> pd.DataFrame:
    """Read one small metadata CSV member and validate its CRC on close."""
    with archive.open(member) as stream:
        data = stream.read()
    return pd.read_csv(io.BytesIO(data), dtype=str, low_memory=False)


def mine_one_site_zip(site_id: str, zip_path: Path) -> MetadataCandidate:
    """Mine exact-site BIF/BIFVARINFO evidence without extracting the archive."""
    if not zipfile.is_zipfile(native_path(zip_path)):
        report = make_report_record(
            site_id=site_id,
            source_type="corrupt_or_invalid_ZIP",
            source_path=str(zip_path),
            confidence="LOW",
            reason="ZIP structure is invalid; no metadata member was trusted.",
            usable=False,
            why_not="archive must be re-downloaded or repaired",
        )
        return MetadataCandidate(report=report)

    try:
        with zipfile.ZipFile(native_path(zip_path)) as archive:
            members = archive.namelist()
            bif_members = [
                member
                for member in members
                if re.search(r"_BIF_", Path(member).name, re.IGNORECASE)
                and "BIFVARINFO" not in Path(member).name.upper()
                and member.lower().endswith(".csv")
            ]
            var_members = [
                member
                for member in members
                if "BIFVARINFO_HH" in Path(member).name.upper()
                and member.lower().endswith(".csv")
            ]
            bif = pd.DataFrame()
            bif_member = ""
            if bif_members:
                bif_member = bif_members[0]
                bif = read_zip_csv(archive, bif_member)
            varinfo = pd.DataFrame()
            var_member = ""
            if var_members:
                var_member = var_members[0]
                varinfo = read_zip_csv(archive, var_member)
    except Exception as error:
        report = make_report_record(
            site_id=site_id,
            source_type="unreadable_ZIP_metadata",
            source_path=str(zip_path),
            confidence="LOW",
            reason=f"{type(error).__name__}: {error}",
            usable=False,
            why_not="metadata members could not be read and CRC-validated",
        )
        return MetadataCandidate(report=report)

    exact_bif = bif
    if "SITE_ID" in bif:
        exact_bif = bif.loc[
            bif["SITE_ID"].fillna("").astype(str).str.strip().eq(site_id)
        ].copy()
    exact_var = varinfo
    if "SITE_ID" in varinfo:
        exact_var = varinfo.loc[
            varinfo["SITE_ID"].fillna("").astype(str).str.strip().eq(site_id)
        ].copy()
    if exact_bif.empty and exact_var.empty:
        report = make_report_record(
            site_id=site_id,
            source_type="ZIP_metadata_no_exact_SITE_ID",
            source_path=str(zip_path),
            file_inside_zip=unique_text([bif_member, var_member]),
            confidence="LOW",
            reason="Metadata members exist but contain no exact SITE_ID row.",
            usable=False,
            why_not="exact SITE_ID requirement failed",
        )
        return MetadataCandidate(report=report)

    lat_values = long_bif_finite_values(exact_bif, {"LOCATION_LAT", "LAT"})
    lon_values = long_bif_finite_values(
        exact_bif,
        {"LOCATION_LONG", "LOCATION_LON", "LON"},
    )
    lat = lat_values[-1] if lat_values else ""
    lon = lon_values[-1] if lon_values else ""
    if not valid_lat_lon(lat, lon):
        lat = ""
        lon = ""

    canopy_values = long_bif_values(exact_bif, BIF_CANOPY_FIELDS)
    height_values = long_bif_values(exact_bif, BIF_HEIGHT_ABOVE_FIELDS)
    radius_values = long_bif_values(exact_bif, BIF_EXPLICIT_RADIUS_FIELDS)
    ec_values, ec_variable, ec_evidence = extract_bifvar_height(exact_var)

    fetch: dict[str, Any] = {}
    found_fields: list[str] = []
    if not exact_bif.empty and {"VARIABLE", "DATAVALUE"}.issubset(exact_bif):
        variables = (
            exact_bif["VARIABLE"].fillna("").astype(str).str.strip().str.upper()
        )
        for field_name in sorted(BIF_FETCH_FIELDS):
            values = finite_values(
                exact_bif.loc[variables.eq(field_name), "DATAVALUE"]
            )
            if values:
                fetch[field_name] = report_value(values)
                found_fields.append(field_name)
    if valid_lat_lon(lat, lon):
        found_fields.extend(["LOCATION_LAT", "LOCATION_LONG"])
    if canopy_values:
        found_fields.append("HEIGHTC/canopy_height")
    if height_values:
        found_fields.append("height_above_canopy")
    if radius_values:
        found_fields.append("explicit_radius")
    if ec_values:
        found_fields.append(f"VAR_INFO_HEIGHT[{ec_variable}]")

    has_primary_ec_height = ec_evidence == "explicit_EC_flux_VAR_INFO_HEIGHT"
    radius_derivable = bool(
        radius_values
        or height_values
        or (canopy_values and ec_values and has_primary_ec_height)
    )
    partial_secondary = bool(
        canopy_values and ec_values and not has_primary_ec_height
    )
    confidence = "HIGH" if (
        valid_lat_lon(lat, lon) or radius_derivable
    ) else ("MEDIUM" if partial_secondary else "LOW")
    reason_parts: list[str] = [
        "Exact SITE_ID in CRC-readable FLUXNET BIF/BIFVARINFO metadata."
    ]
    if ec_values:
        reason_parts.append(
            f"{ec_evidence} from {ec_variable}; values={ec_values}."
        )
    if canopy_values:
        reason_parts.append(
            f"Explicit HEIGHTC/canopy values; min={min(canopy_values):g}, "
            f"max={max(canopy_values):g} m."
        )
    if radius_values:
        reason_parts.append("Trusted explicit radius field is present.")
    why_not: list[str] = []
    if not radius_derivable:
        if not canopy_values:
            why_not.append("trusted canopy/vegetation height is missing")
        if not ec_values:
            why_not.append("trusted EC measurement height is missing")
        elif not has_primary_ec_height:
            why_not.append(
                "only a secondary wind-instrument height was found"
            )
        if not height_values and not radius_values:
            why_not.append("no explicit radius/height-above-canopy exists")

    report = make_report_record(
        site_id=site_id,
        source_type="FLUXNET_ZIP_BIF_and_BIFVARINFO_HH",
        source_path=str(zip_path),
        file_inside_zip=unique_text([bif_member, var_member]),
        fields=found_fields,
        lat=lat,
        lon=lon,
        canopy_values=canopy_values,
        ec_values=ec_values,
        height_values=height_values,
        radius_values=radius_values,
        fetch=fetch,
        confidence=confidence,
        reason=" ".join(reason_parts),
        usable=bool(valid_lat_lon(lat, lon) or radius_derivable),
        why_not="; ".join(why_not),
    )
    method_hint = ""
    if radius_values:
        method_hint = "trusted_explicit_radius"
    elif canopy_values and ec_values and has_primary_ec_height:
        method_hint = "EC_measurement_height_minus_canopy_height_times_100"
    elif height_values:
        method_hint = "explicit_height_above_canopy_times_100"
    return MetadataCandidate(
        report=report,
        canopy_values=canopy_values,
        ec_height_values=ec_values,
        height_above_values=height_values,
        explicit_radius_values=radius_values,
        confidence=confidence,
        source_priority=0,
        lat_auto_allowed=bool(valid_lat_lon(lat, lon)),
        radius_auto_allowed=bool(radius_values),
        radius_derivation_allowed=bool(
            (canopy_values and ec_values and has_primary_ec_height)
            or height_values
        ),
        radius_method_hint=method_hint,
    )


def discover_target_zip_paths() -> dict[str, list[Path]]:
    """Find ZIPs whose filenames contain an exact blocked SITE_ID token."""
    mapping: dict[str, list[Path]] = {site: [] for site in ALL_TARGET_SITES}
    seen: set[str] = set()
    for root in (FLUXNET_ROOT, REMAINING_ROOT):
        if not path_exists(root):
            continue
        for zip_path in root.rglob("*.zip"):
            normalized = os.path.normcase(os.path.abspath(os.fspath(zip_path)))
            if normalized in seen:
                continue
            seen.add(normalized)
            for site_id in ALL_TARGET_SITES:
                if exact_site_in_name(zip_path.name, site_id):
                    mapping[site_id].append(zip_path)
    return mapping


def mine_zip_candidates() -> list[MetadataCandidate]:
    """Mine BIF/BADM-style metadata inside all exact-site ZIPs."""
    mapping = discover_target_zip_paths()
    candidates: list[MetadataCandidate] = []
    for index, site_id in enumerate(ALL_TARGET_SITES, start=1):
        paths = mapping.get(site_id, [])
        print(
            f"  ZIP metadata {index}/{len(ALL_TARGET_SITES)}: "
            f"{site_id} ({len(paths)} archive(s))",
            flush=True,
        )
        if not paths:
            report = make_report_record(
                site_id=site_id,
                source_type="ZIP_not_found",
                source_path="",
                confidence="LOW",
                reason="No exact-site ZIP filename was detected.",
                usable=False,
                why_not="no local ZIP evidence",
            )
            candidates.append(MetadataCandidate(report=report))
            continue
        for zip_path in paths:
            candidates.append(mine_one_site_zip(site_id, zip_path))
    return candidates


# =============================================================================
# EXTRACTED/PROJECT METADATA TABLE MINING
# =============================================================================


def discover_local_metadata_tables() -> list[Path]:
    """Discover plausible metadata tables while excluding raw flux/time-series files."""
    paths: set[Path] = {
        PROJECT_ROOT / "New_FLUXNET_sites_metadata_for_KMZ_preparation.xlsx",
        PROJECT_ROOT / "Sites Information.xlsx",
    }
    name_pattern = re.compile(
        r"(BIF|BADM|CommonAnc|metadata|sites?[_ -]?information|site[_ -]?info)",
        re.IGNORECASE,
    )
    for root in (FLUXNET_ROOT, REMAINING_ROOT):
        if not path_exists(root):
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {
                ".csv",
                ".xlsx",
                ".xls",
                ".txt",
                ".xml",
                ".json",
            }:
                continue
            if not name_pattern.search(path.name):
                continue
            lower_path = str(path).lower()
            if (
                "monthly_with_metadata" in path.name.lower()
                or "\\_qc\\" in lower_path
                or "/_qc/" in lower_path
                or "finalized_metadata_recovery" in lower_path
                or "repaired_final_monthly_gee_inputs" in lower_path
            ):
                # These are derived outputs, not independent metadata evidence.
                continue
            try:
                if os.path.getsize(native_path(path)) > (
                    MAX_LOCAL_METADATA_FILE_MB * 1024**2
                ):
                    continue
            except OSError:
                continue
            paths.add(path)
    return sorted(
        [
            path
            for path in paths
            if path_exists(path)
            and os.path.normcase(os.path.abspath(os.fspath(path)))
            != os.path.normcase(
                os.path.abspath(os.fspath(MANUAL_METADATA_DRAFT))
            )
        ],
        key=lambda item: str(item).lower(),
    )


def trusted_metadata_filename(path: Path) -> bool:
    """Identify maintained project/BIF/BADM tables eligible for HIGH confidence."""
    name = path.name.lower()
    return bool(
        name
        in {
            "new_fluxnet_sites_metadata_for_kmz_preparation.xlsx",
            "sites information.xlsx",
        }
        or "bif" in name
        or "badm" in name
        or "commonanc" in name
    )


def mine_tabular_metadata_candidates() -> list[MetadataCandidate]:
    """Read exact target rows from discovered local metadata tables."""
    candidates: list[MetadataCandidate] = []
    seen_records: set[tuple[Any, ...]] = set()
    target_set = set(ALL_TARGET_SITES)
    paths = discover_local_metadata_tables()
    print(f"  Extracted/project metadata tables: {len(paths):,}", flush=True)
    for path in paths:
        if path.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
            # Unstructured text is reported only when it contains an exact site
            # and a relevant keyword; it remains LOW confidence.
            try:
                text = Path(native_path(path)).read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            except Exception:
                continue
            upper = text.upper()
            relevant = any(
                keyword in upper
                for keyword in (
                    "HEIGHTC",
                    "CANOPY HEIGHT",
                    "MEASUREMENT HEIGHT",
                    "FETCH_70",
                    "FETCH_80",
                    "FETCH_90",
                    "FETCH_MAX",
                )
            )
            if not relevant:
                continue
            for site_id in ALL_TARGET_SITES:
                if site_id.upper() not in upper:
                    continue
                report = make_report_record(
                    site_id=site_id,
                    source_type="unstructured_local_metadata_text",
                    source_path=str(path),
                    fields=["relevant_text_keyword"],
                    confidence="LOW",
                    reason="Exact SITE_ID and relevant text coexist, but fields "
                    "were not safely structured.",
                    usable=False,
                    why_not="manual interpretation required",
                )
                candidates.append(MetadataCandidate(report=report))
            continue

        rows = repair.read_target_metadata_rows(path, target_set)
        for row in rows:
            canonical = repair.canonical_metadata_candidate(path, row)
            if canonical is None:
                continue
            site_id = canonical["SITE_ID"]
            lat = canonical.get("lat")
            lon = canonical.get("lon")
            if not valid_lat_lon(lat, lon):
                lat = ""
                lon = ""
            canopy_values = positive_values(
                [canonical.get("canopy_height_m")]
            )
            ec_values = positive_values(
                [canonical.get("EC_measurement_height_m")]
            )
            height_values = positive_values(
                [canonical.get("height_above_canopy_m")]
            )
            radius_values = positive_values([canonical.get("radius_m_max")])
            signature = (
                site_id,
                str(path),
                str(row.get("_metadata_sheet", "")),
                lat,
                lon,
                tuple(canopy_values),
                tuple(ec_values),
                tuple(height_values),
                tuple(radius_values),
                str(canonical.get("radius_method", "")),
                str(canonical.get("metadata_status", "")),
            )
            if signature in seen_records:
                continue
            seen_records.add(signature)
            maintained = trusted_metadata_filename(path)
            row_trusted = bool(canonical.get("trusted"))
            confidence = "HIGH" if maintained and row_trusted else (
                "MEDIUM" if row_trusted else "LOW"
            )
            method = str(canonical.get("radius_method", ""))
            explicit_radius = method == "trusted_explicit_radius"
            radius_derivation = bool(
                method
                and not explicit_radius
                and (
                    height_values
                    or (canopy_values and ec_values)
                )
            )
            usable = bool(
                confidence_allows(confidence)
                and (
                    valid_lat_lon(lat, lon)
                    or radius_values
                    or radius_derivation
                )
            )
            fields = [
                field_name
                for field_name, values in (
                    ("lat/lon", [lat] if valid_lat_lon(lat, lon) else []),
                    ("canopy_height_m", canopy_values),
                    ("EC_measurement_height_m", ec_values),
                    ("height_above_canopy_m", height_values),
                    ("radius_m_max", radius_values),
                )
                if values
            ]
            why_not = ""
            if not usable:
                why_not = (
                    "source is not HIGH confidence or contains no complete "
                    "coordinate/radius evidence"
                )
            report = make_report_record(
                site_id=site_id,
                source_type="extracted_or_project_metadata_table",
                source_path=str(path),
                file_inside_zip=str(row.get("_metadata_sheet", "")),
                fields=fields,
                lat=lat,
                lon=lon,
                canopy_values=canopy_values,
                ec_values=ec_values,
                height_values=height_values,
                radius_values=radius_values,
                confidence=confidence,
                reason=(
                    f"Exact SITE_ID row; metadata status="
                    f"{canonical.get('metadata_status', '')}; "
                    f"radius_method={method or 'none'}."
                ),
                usable=usable,
                why_not=why_not,
            )
            candidates.append(
                MetadataCandidate(
                    report=report,
                    canopy_values=canopy_values,
                    ec_height_values=ec_values,
                    height_above_values=height_values,
                    explicit_radius_values=radius_values,
                    confidence=confidence,
                    source_priority=10,
                    lat_auto_allowed=bool(
                        valid_lat_lon(lat, lon)
                        and confidence_allows(confidence)
                    ),
                    radius_auto_allowed=bool(
                        explicit_radius
                        and radius_values
                        and confidence_allows(confidence)
                    ),
                    radius_derivation_allowed=bool(
                        radius_derivation
                        and confidence_allows(confidence)
                    ),
                    radius_method_hint=method,
                )
            )
    return candidates


def raw_fetch_header_candidates() -> tuple[
    list[MetadataCandidate],
    dict[str, set[str]],
]:
    """Report FETCH columns from raw HH/HR headers without treating them as radius."""
    inventory = repair.load_raw_file_inventory()
    candidates: list[MetadataCandidate] = []
    availability: dict[str, set[str]] = {}
    for site_id, raw_path in inventory.items():
        try:
            header = repair.read_raw_header(raw_path)
        except Exception as error:
            warn(f"Could not read raw header for {site_id}: {error}")
            continue
        fetch_columns = {
            str(column).strip().upper()
            for column in header
            if str(column).strip().upper() in BIF_FETCH_FIELDS
        }
        availability[site_id] = fetch_columns
        if site_id not in ALL_TARGET_SITES or not fetch_columns:
            continue
        fetch = {
            field_name: "COLUMN_AVAILABLE"
            for field_name in sorted(fetch_columns)
        }
        report = make_report_record(
            site_id=site_id,
            source_type="raw_HH_HR_header_FETCH_diagnostic",
            source_path=str(raw_path),
            fields=fetch_columns,
            fetch=fetch,
            confidence="HIGH",
            reason="Exact-site raw header contains FETCH diagnostics.",
            usable=False,
            why_not=(
                "FETCH is diagnostic only and is not auto-substituted for "
                "tower-height radius"
            ),
        )
        candidates.append(
            MetadataCandidate(
                report=report,
                confidence="HIGH",
                source_priority=30,
            )
        )
    return candidates, availability


# =============================================================================
# RADIUS DECISION TABLE
# =============================================================================


def current_site_values(
    table: pd.DataFrame,
    site_id: str,
) -> dict[str, Any]:
    """Read current site-level metadata without manufacturing values."""
    site = table.loc[table["SITE_ID"].eq(site_id)]
    if site.empty:
        return {
            "lat": None,
            "lon": None,
            "radius": None,
            "radius_min": None,
            "radius_max": None,
        }
    lat = first_meaningful(site, "lat", np.nan)
    lon = first_meaningful(site, "lon", np.nan)
    radius = positive(first_meaningful(site, "radius_m", np.nan))
    radius_min = positive(first_meaningful(site, "radius_m_min", np.nan))
    radius_max = positive(first_meaningful(site, "radius_m_max", np.nan))
    return {
        "lat": float(lat) if valid_lat_lon(lat, lon) else None,
        "lon": float(lon) if valid_lat_lon(lat, lon) else None,
        "radius": radius_max or radius,
        "radius_min": radius_min,
        "radius_max": radius_max or radius,
    }


def candidate_source(candidate: MetadataCandidate) -> str:
    """Format candidate provenance."""
    report = candidate.report
    source = str(report.get("source_path", "")).strip()
    member = str(report.get("file_inside_zip", "")).strip()
    return source + (f"::{member}" if member else "")


def aggregate_fetch(
    candidates: Iterable[MetadataCandidate],
) -> dict[str, str]:
    """Collect FETCH diagnostics without choosing them as radius."""
    output: dict[str, str] = {}
    for field_name in sorted(BIF_FETCH_FIELDS):
        values = [
            candidate.report.get(field_name)
            for candidate in candidates
            if not is_missing(candidate.report.get(field_name))
            and str(candidate.report.get(field_name)).strip()
        ]
        output[field_name] = unique_text(values)
    return output


def build_radius_decisions(
    base_table: pd.DataFrame,
    candidates: list[MetadataCandidate],
) -> pd.DataFrame:
    """Apply Priority A/B/C rules and keep FETCH as diagnostics only."""
    by_site: dict[str, list[MetadataCandidate]] = {
        site_id: [] for site_id in ALL_TARGET_SITES
    }
    for candidate in candidates:
        site_id = str(candidate.report.get("SITE_ID", "")).strip()
        if site_id in by_site:
            by_site[site_id].append(candidate)
    for site_id in by_site:
        by_site[site_id].sort(
            key=lambda item: (
                item.source_priority,
                candidate_source(item).lower(),
            )
        )

    records: list[dict[str, Any]] = []
    for site_id in ALL_TARGET_SITES:
        site_candidates = by_site.get(site_id, [])
        current = current_site_values(base_table, site_id)
        needs_lat = site_id in LATLON_AND_RADIUS_MISSING
        problem_type = (
            "BE-Lon_corrupt_ZIP_no_final_rows"
            if site_id == BE_LON_SITE
            else (
                "lat_lon_and_radius_missing"
                if needs_lat
                else "radius_missing_only"
            )
        )

        selected_lat = current["lat"]
        selected_lon = current["lon"]
        lat_source = (
            "existing_final_input_table"
            if valid_lat_lon(selected_lat, selected_lon)
            else ""
        )
        lat_confidence = "HIGH" if lat_source else "LOW"
        lat_apply = False
        if not valid_lat_lon(selected_lat, selected_lon):
            for candidate in site_candidates:
                lat = candidate.report.get("lat_candidate")
                lon = candidate.report.get("lon_candidate")
                if not (
                    valid_lat_lon(lat, lon)
                    and candidate.lat_auto_allowed
                    and confidence_allows(candidate.confidence)
                ):
                    continue
                selected_lat = float(lat)
                selected_lon = float(lon)
                lat_source = candidate_source(candidate)
                lat_confidence = candidate.confidence
                lat_apply = True
                break

        selected_canopy: float | None = None
        selected_ec: float | None = None
        selected_height: float | None = None
        selected_radius: float | None = current["radius"]
        selected_radius_min: float | None = current["radius_min"]
        selected_radius_max: float | None = current["radius_max"]
        radius_priority = "EXISTING" if selected_radius else ""
        radius_source = (
            "existing_final_input_table" if selected_radius else ""
        )
        radius_confidence = "HIGH" if selected_radius else "LOW"
        radius_apply = False
        notes: list[str] = []

        # Priority A: trusted explicit radius.
        if selected_radius is None:
            for candidate in site_candidates:
                values = candidate.explicit_radius_values
                if not (
                    values
                    and candidate.radius_auto_allowed
                    and confidence_allows(candidate.confidence)
                ):
                    continue
                selected_radius_min = min(values)
                selected_radius_max = max(values)
                selected_radius = selected_radius_max
                selected_height = (
                    selected_radius / 100.0
                    if candidate.radius_method_hint.endswith("times_100")
                    else None
                )
                selected_canopy = (
                    max(candidate.canopy_values)
                    if candidate.canopy_values
                    else None
                )
                selected_ec = (
                    max(candidate.ec_height_values)
                    if candidate.ec_height_values
                    else None
                )
                radius_priority = "A_TRUSTED_EXPLICIT_RADIUS"
                radius_source = candidate_source(candidate)
                radius_confidence = candidate.confidence
                radius_apply = True
                notes.append("Priority A trusted explicit radius selected.")
                break

        # Priority B: explicit EC-flux height minus explicit canopy height.
        if selected_radius is None:
            for candidate in site_candidates:
                if not (
                    candidate.canopy_values
                    and candidate.ec_height_values
                    and candidate.radius_derivation_allowed
                    and confidence_allows(candidate.confidence)
                ):
                    continue
                canopy_max = max(candidate.canopy_values)
                ec_min = min(candidate.ec_height_values)
                ec_max = max(candidate.ec_height_values)
                height_min = ec_min - canopy_max
                height_max = ec_max - canopy_max
                if height_max <= 0.0:
                    notes.append(
                        f"Rejected Priority B candidate from "
                        f"{candidate_source(candidate)}: EC height does not "
                        "exceed maximum canopy height."
                    )
                    continue
                derived_confidence = candidate.confidence
                if height_min <= 0.0:
                    derived_confidence = "MEDIUM"
                    notes.append(
                        "EC/canopy range includes a non-positive height above "
                        "canopy; candidate retained for manual review."
                    )
                selected_canopy = canopy_max
                selected_ec = ec_max
                selected_height = height_max
                selected_radius_min = (
                    height_min * 100.0 if height_min > 0.0 else np.nan
                )
                selected_radius_max = height_max * 100.0
                selected_radius = selected_radius_max
                radius_priority = "B_EC_HEIGHT_MINUS_CANOPY_TIMES_100"
                radius_source = candidate_source(candidate)
                radius_confidence = derived_confidence
                radius_apply = confidence_allows(derived_confidence)
                notes.append(
                    "Priority B used explicit BIF/BIFVARINFO values: "
                    f"EC_max={ec_max:g} m, canopy_max={canopy_max:g} m, "
                    f"radius_max={selected_radius_max:g} m."
                )
                break

        # Priority C: trusted explicit height above canopy.
        if selected_radius is None:
            for candidate in site_candidates:
                values = candidate.height_above_values
                if not (
                    values
                    and candidate.radius_derivation_allowed
                    and confidence_allows(candidate.confidence)
                ):
                    continue
                selected_height = max(values)
                selected_radius_min = min(values) * 100.0
                selected_radius_max = max(values) * 100.0
                selected_radius = selected_radius_max
                selected_canopy = (
                    max(candidate.canopy_values)
                    if candidate.canopy_values
                    else None
                )
                selected_ec = (
                    max(candidate.ec_height_values)
                    if candidate.ec_height_values
                    else None
                )
                radius_priority = "C_EXPLICIT_HEIGHT_ABOVE_CANOPY_TIMES_100"
                radius_source = candidate_source(candidate)
                radius_confidence = candidate.confidence
                radius_apply = True
                notes.append(
                    "Priority C trusted explicit height_above_canopy_m used."
                )
                break

        fetch = aggregate_fetch(site_candidates)
        if any(fetch.values()):
            notes.append(
                "FETCH fields were reported as diagnostics and were not used "
                "as radius."
            )
        if selected_radius is None:
            notes.append(
                "No Priority A/B/C radius evidence passed the confidence rules."
            )

        # BE-Lon cannot be recovered by metadata alone because it has no final
        # WD/GPP rows and its ZIP is corrupt.
        if site_id == BE_LON_SITE:
            lat_apply = False
            radius_apply = False
            notes.append(
                "BE-Lon requires a clean ZIP re-download and complete HH/HR "
                "processing; metadata alone is insufficient."
            )

        # Every target site is missing radius, so a coordinate-only recovery
        # must not make the overall decision appear HIGH/completed.
        selected_confidence = (
            radius_confidence if selected_radius is not None else "LOW"
        )
        if needs_lat and not valid_lat_lon(selected_lat, selected_lon):
            selected_confidence = "LOW"
        complete_metadata = bool(
            valid_lat_lon(selected_lat, selected_lon)
            and positive(selected_radius) is not None
        )
        # In the public decision table, auto_apply means the site can become
        # metadata-complete automatically. Safe coordinate-only improvements
        # are still applied in memory through _apply_latlon, but do not make a
        # radius-blocked site look fully recoverable.
        auto_apply = bool(
            site_id != BE_LON_SITE
            and complete_metadata
            and (current["radius"] is not None or radius_apply)
            and (
                not needs_lat
                or valid_lat_lon(current["lat"], current["lon"])
                or lat_apply
            )
        )
        manual_review = bool(
            site_id == BE_LON_SITE
            or not complete_metadata
            or (
                selected_radius is not None
                and not radius_apply
                and current["radius"] is None
            )
        )
        records.append(
            {
                "SITE_ID": site_id,
                "problem_type": problem_type,
                "selected_lat": selected_lat,
                "selected_lon": selected_lon,
                "selected_canopy_height_m": selected_canopy,
                "selected_EC_measurement_height_m": selected_ec,
                "selected_height_above_canopy_m": selected_height,
                "selected_radius_m": selected_radius,
                "selected_radius_m_min": selected_radius_min,
                "selected_radius_m_max": selected_radius_max,
                "radius_decision_priority": radius_priority,
                "radius_source": radius_source,
                "latlon_source": lat_source,
                "confidence": selected_confidence,
                "auto_apply": auto_apply,
                "manual_review_required": manual_review,
                "FETCH_70": fetch.get("FETCH_70", ""),
                "FETCH_80": fetch.get("FETCH_80", ""),
                "FETCH_90": fetch.get("FETCH_90", ""),
                "FETCH_FILTER": fetch.get("FETCH_FILTER", ""),
                "FETCH_MAX": fetch.get("FETCH_MAX", ""),
                "decision_notes": " ".join(notes),
                "_apply_latlon": lat_apply,
                "_apply_radius": radius_apply,
                "_metadata_complete_after_decision": complete_metadata,
            }
        )
    return pd.DataFrame(records)


# =============================================================================
# SAFE IN-MEMORY APPLICATION AND EXPORT FLAG REBUILD
# =============================================================================


def ensure_final_provenance_columns(table: pd.DataFrame) -> None:
    """Create final recovery provenance columns with stable dtypes."""
    defaults: dict[str, Any] = {
        "final_metadata_recovery_applied": False,
        "final_latlon_recovery_applied": False,
        "final_radius_recovery_applied": False,
        "final_metadata_source": "",
        "final_metadata_confidence": "",
        "final_radius_method": "",
        "final_recovery_notes": "",
    }
    for column, default in defaults.items():
        if column not in table:
            table[column] = default


def append_text_column(
    table: pd.DataFrame,
    mask: pd.Series,
    column: str,
    value: str,
) -> None:
    """Append unique provenance text to selected rows."""
    if not mask.any() or not value:
        return
    existing = table.loc[mask, column].fillna("").astype(str).str.strip()
    table.loc[mask, column] = np.where(
        existing.eq(""),
        value,
        existing + " | " + value,
    )


def apply_safe_metadata_decisions(
    base_table: pd.DataFrame,
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    """Apply independently approved coordinates/radii to an in-memory copy."""
    table = base_table.copy()
    ensure_final_provenance_columns(table)
    for column in (
        "canopy_height_m",
        "EC_measurement_height_m",
        "height_above_canopy_m",
        "radius_m",
        "radius_m_min",
        "radius_m_max",
    ):
        if column not in table:
            table[column] = np.nan

    for decision in decisions.to_dict("records"):
        site_id = str(decision["SITE_ID"])
        site_mask = table["SITE_ID"].eq(site_id)
        if not site_mask.any():
            continue
        applied = pd.Series(False, index=table.index)

        if bool(decision.get("_apply_latlon")):
            lat = decision.get("selected_lat")
            lon = decision.get("selected_lon")
            if not valid_lat_lon(lat, lon):
                raise RuntimeError(
                    f"Unsafe lat/lon decision reached application for {site_id}."
                )
            current_lat = pd.to_numeric(table["lat"], errors="coerce")
            current_lon = pd.to_numeric(table["lon"], errors="coerce")
            missing_latlon = site_mask & ~(
                current_lat.between(-90.0, 90.0)
                & current_lon.between(-180.0, 180.0)
            )
            table.loc[missing_latlon, "lat"] = float(lat)
            table.loc[missing_latlon, "lon"] = float(lon)
            table.loc[
                missing_latlon,
                "final_latlon_recovery_applied",
            ] = True
            applied |= missing_latlon

        if bool(decision.get("_apply_radius")):
            radius = positive(decision.get("selected_radius_m_max"))
            if radius is None:
                raise RuntimeError(
                    f"Unsafe radius decision reached application for {site_id}."
                )
            current_max = pd.to_numeric(
                table["radius_m_max"],
                errors="coerce",
            )
            current_radius = pd.to_numeric(
                table["radius_m"],
                errors="coerce",
            )
            missing_radius = site_mask & ~(
                current_max.gt(0.0) | current_radius.gt(0.0)
            )
            table.loc[missing_radius, "radius_m"] = radius
            table.loc[missing_radius, "radius_m_max"] = radius
            radius_min = positive(decision.get("selected_radius_m_min"))
            table.loc[missing_radius, "radius_m_min"] = (
                radius_min if radius_min is not None else radius
            )
            for column, key in (
                ("canopy_height_m", "selected_canopy_height_m"),
                (
                    "EC_measurement_height_m",
                    "selected_EC_measurement_height_m",
                ),
                (
                    "height_above_canopy_m",
                    "selected_height_above_canopy_m",
                ),
            ):
                value = positive(decision.get(key))
                if value is not None:
                    table.loc[missing_radius, column] = value
            table.loc[
                missing_radius,
                "final_radius_recovery_applied",
            ] = True
            table.loc[
                missing_radius,
                "final_radius_method",
            ] = str(decision.get("radius_decision_priority", ""))
            applied |= missing_radius

        if applied.any():
            table.loc[
                applied,
                "final_metadata_recovery_applied",
            ] = True
            append_text_column(
                table,
                applied,
                "final_metadata_source",
                unique_text(
                    [
                        decision.get("latlon_source"),
                        decision.get("radius_source"),
                    ]
                ),
            )
            table.loc[
                applied,
                "final_metadata_confidence",
            ] = str(decision.get("confidence", ""))
            append_text_column(
                table,
                applied,
                "final_recovery_notes",
                str(decision.get("decision_notes", "")),
            )

    # This recalculation requires valid coordinates, positive radius, valid
    # monthly GPP, actual usable WD bins, and at least one valid WD record.
    repair.recalculate_export_flags(table)
    return table


# =============================================================================
# SITE-MONTH AUDITS AND BEFORE/AFTER REPORTS
# =============================================================================


def prepare_for_audit(table: pd.DataFrame) -> pd.DataFrame:
    """Add the normalized fields required by the existing strict audit."""
    return repair.prepare_for_completeness_audit(table)


def site_month_audit(table: pd.DataFrame) -> pd.DataFrame:
    """Build strict site-month exportability without touching TIFFs."""
    return completeness_audit.build_site_month_audit(
        prepare_for_audit(table),
        repair.EXISTING_TIFF_OUTPUT_ROOT,
    )


def exportable_sets(
    audit: pd.DataFrame,
) -> tuple[set[str], set[tuple[str, str]]]:
    """Return strict exportable sites and site-month keys."""
    exportable = audit.loc[audit["exportable_by_strict_rules"]]
    sites = set(exportable["SITE_ID"].astype(str))
    keys = set(
        zip(
            exportable["SITE_ID"].astype(str),
            exportable["YYYYMM"].astype(str),
        )
    )
    return sites, keys


def blocked_reason_from_audit(
    audit: pd.DataFrame,
    site_id: str,
) -> str:
    """Summarize unique blockers for a site."""
    site = audit.loc[audit["SITE_ID"].astype(str).eq(site_id)]
    if site.empty:
        return "No final site-month rows; BE-Lon/raw processing may be missing."
    reasons: set[str] = set()
    for value in site["export_blocking_reason"]:
        if is_missing(value):
            continue
        for reason in str(value).split(";"):
            reason = reason.strip()
            if reason:
                reasons.add(reason)
    return "; ".join(sorted(reasons))


def build_before_after_reports(
    before_audit: pd.DataFrame,
    after_audit: pd.DataFrame,
    decisions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build summary, recovered, and still-blocked site reports."""
    before_sites, before_keys = exportable_sets(before_audit)
    after_sites, after_keys = exportable_sets(after_audit)
    recovered_sites = sorted(after_sites - before_sites)
    still_blocked = sorted(set(ALL_TARGET_SITES) - after_sites)
    manual_review = sorted(
        decisions.loc[
            decisions["manual_review_required"],
            "SITE_ID",
        ].astype(str)
    )

    summary = pd.DataFrame(
        [
            {
                "metric": "exportable_sites",
                "before": len(before_sites),
                "after": len(after_sites),
                "change": len(after_sites) - len(before_sites),
            },
            {
                "metric": "exportable_site_months",
                "before": len(before_keys),
                "after": len(after_keys),
                "change": len(after_keys) - len(before_keys),
            },
            {
                "metric": "newly_recovered_sites",
                "before": "",
                "after": " | ".join(recovered_sites),
                "change": len(recovered_sites),
            },
            {
                "metric": "still_blocked_target_sites",
                "before": "",
                "after": " | ".join(still_blocked),
                "change": len(still_blocked),
            },
            {
                "metric": "manual_review_sites",
                "before": "",
                "after": " | ".join(manual_review),
                "change": len(manual_review),
            },
            {
                "metric": "BE-Lon_status",
                "before": "corrupt/truncated ZIP",
                "after": "re-download required",
                "change": "",
            },
        ]
    )

    decision_index = decisions.set_index("SITE_ID", drop=False)
    recovered_rows: list[dict[str, Any]] = []
    for site_id in recovered_sites:
        decision = decision_index.loc[site_id]
        before_count = sum(site == site_id for site, _ in before_keys)
        after_count = sum(site == site_id for site, _ in after_keys)
        recovered_rows.append(
            {
                "SITE_ID": site_id,
                "n_exportable_months_before": before_count,
                "n_exportable_months_after": after_count,
                "n_months_recovered": after_count - before_count,
                "selected_lat": decision["selected_lat"],
                "selected_lon": decision["selected_lon"],
                "selected_radius_m": decision["selected_radius_m"],
                "radius_decision_priority": decision[
                    "radius_decision_priority"
                ],
                "radius_source": decision["radius_source"],
                "latlon_source": decision["latlon_source"],
                "confidence": decision["confidence"],
            }
        )
    recovered = pd.DataFrame(recovered_rows)

    blocked_rows: list[dict[str, Any]] = []
    for site_id in still_blocked:
        decision = decision_index.loc[site_id]
        blocked_rows.append(
            {
                "SITE_ID": site_id,
                "problem_type": decision["problem_type"],
                "selected_lat": decision["selected_lat"],
                "selected_lon": decision["selected_lon"],
                "selected_radius_m": decision["selected_radius_m"],
                "confidence": decision["confidence"],
                "manual_review_required": decision[
                    "manual_review_required"
                ],
                "blocking_reason": blocked_reason_from_audit(
                    after_audit,
                    site_id,
                ),
                "radius_source_or_candidate": decision["radius_source"],
                "decision_notes": decision["decision_notes"],
                "recommended_action": (
                    "Re-download and process clean BE-Lon FLUXNET ZIP."
                    if site_id == BE_LON_SITE
                    else "Find trusted explicit radius or exact EC + canopy "
                    "height metadata; do not use a default."
                ),
            }
        )
    blocked = pd.DataFrame(blocked_rows)
    result = {
        "before_sites": len(before_sites),
        "before_months": len(before_keys),
        "after_sites": len(after_sites),
        "after_months": len(after_keys),
        "recovered_sites": recovered_sites,
        "still_blocked_sites": still_blocked,
        "manual_review_sites": manual_review,
    }
    return summary, recovered, blocked, result


# =============================================================================
# METHOD-COMPLIANCE AUDIT
# =============================================================================


def inspect_tiff_geometry_source() -> tuple[bool, bool, list[str]]:
    """Statically verify the directional wedge conventions in the TIFF script."""
    notes: list[str] = []
    if not path_exists(TIFF_CREATION_SCRIPT):
        return False, True, ["TIFF creation source file was not found."]
    text = Path(native_path(TIFF_CREATION_SCRIPT)).read_text(
        encoding="utf-8",
        errors="replace",
    )
    uses_center = bool(
        "WD_bin_center_deg" in text
        and "center_bearing_deg=center_bearing" in text
        and "create_wedge_polygon" in text
    )
    explicit_no_rotation = (
        "No 180-degree rotation is applied" in text
        or "Do NOT add 180" in text
    )
    applies_rotation = not explicit_no_rotation
    if uses_center:
        notes.append("TIFF source passes WD_bin_center_deg directly to wedges.")
    else:
        notes.append("Could not verify direct WD_bin_center_deg wedge use.")
    if explicit_no_rotation:
        notes.append("TIFF source explicitly prohibits 180-degree rotation.")
    else:
        notes.append("Could not verify the no-180-degree-rotation rule.")
    return uses_center, applies_rotation, notes


def build_fetch_availability_for_all_sites() -> dict[str, set[str]]:
    """Read only raw headers to record FETCH diagnostics for all sites."""
    inventory = repair.load_raw_file_inventory()
    availability: dict[str, set[str]] = {}
    for site_id, raw_path in inventory.items():
        try:
            header = repair.read_raw_header(raw_path)
        except Exception:
            availability[site_id] = set()
            continue
        availability[site_id] = {
            str(column).strip().upper()
            for column in header
            if str(column).strip().upper() in BIF_FETCH_FIELDS
        }
    return availability


def infer_gpp_qc_filter(group: pd.DataFrame) -> tuple[bool, str]:
    """Infer conservative/QC-filter provenance from documented pipeline fields."""
    source = str(first_meaningful(group, "GPP_source", "")).lower()
    merge_status = str(
        first_meaningful(group, "GPP_merge_status", "")
    ).lower()
    primary = str(first_meaningful(group, "primary_GPP_column", ""))
    if "conservative" in source or "conservative" in merge_status:
        return True, "documented conservative HH/HR GPP source"
    if merge_status == "both" and primary.upper().startswith("GPP_"):
        return True, "original conservative GPP table joined (merge_status=both)"
    if "fallback" in source or "fallback" in merge_status:
        return False, "monthly metadata GPP fallback; conservative QC not proven"
    return False, "GPP QC-filter provenance is not explicit"


def infer_radius_method(group: pd.DataFrame, radius: float) -> tuple[str, str]:
    """Describe existing or newly recovered radius provenance."""
    final_method = str(
        first_meaningful(group, "final_radius_method", "")
    ).strip()
    final_source = str(
        first_meaningful(group, "final_metadata_source", "")
    ).strip()
    if final_method:
        return final_method, final_source
    height = positive(
        first_meaningful(group, "height_above_canopy_m", np.nan)
    )
    metadata_source = unique_text(
        [
            first_meaningful(group, "Metadata source", ""),
            first_meaningful(group, "repair_source", ""),
            first_meaningful(group, "source_input_table", ""),
        ]
    )
    if height is not None and math.isclose(
        radius,
        height * 100.0,
        rel_tol=1e-5,
        abs_tol=1e-5,
    ):
        return "height_above_canopy_m_times_100", metadata_source
    return "trusted_explicit_or_existing_project_radius", metadata_source


def build_method_compliance_audit(
    table: pd.DataFrame,
    after_audit: pd.DataFrame,
    fetch_availability: dict[str, set[str]],
) -> pd.DataFrame:
    """Audit every strictly exportable site-month against the project's method."""
    uses_center, applies_rotation, static_notes = inspect_tiff_geometry_source()
    exportable_keys = set(
        zip(
            after_audit.loc[
                after_audit["exportable_by_strict_rules"],
                "SITE_ID",
            ].astype(str),
            after_audit.loc[
                after_audit["exportable_by_strict_rules"],
                "YYYYMM",
            ].astype(str),
        )
    )
    rows: list[dict[str, Any]] = []
    grouped = table.groupby(["SITE_ID", "YYYYMM"], sort=True)
    for (site_id, yyyymm), group in grouped:
        key = (str(site_id), str(yyyymm))
        if key not in exportable_keys:
            continue
        centers = pd.to_numeric(
            group["WD_bin_center_deg"],
            errors="coerce",
        )
        fractions = pd.to_numeric(
            group["WD_bin_fraction"],
            errors="coerce",
        )
        widths = pd.to_numeric(
            group["WD_bin_width_deg"],
            errors="coerce",
        ).dropna()
        records = pd.to_numeric(
            group["n_valid_monthly_30min_WD_records"],
            errors="coerce",
        )
        usable = centers.notna() & fractions.notna() & fractions.ge(0.0)
        n_bins = int(usable.sum())
        fraction_sum = float(fractions.loc[usable].sum())
        unique_widths = sorted(set(float(value) for value in widths))
        width_value: float | str = (
            unique_widths[0]
            if len(unique_widths) == 1
            else " | ".join(f"{value:g}" for value in unique_widths)
        )
        wd_records_ok = bool(records.notna().any() and records.max() > 0)
        width_ok = bool(
            unique_widths
            and all(
                math.isclose(
                    value,
                    WD_BIN_WIDTH_REQUIRED_DEG,
                    abs_tol=1e-9,
                )
                for value in unique_widths
            )
        )
        fraction_ok = 0.99 <= fraction_sum <= 1.01
        circular_present = bool(
            "monthly_circular_mean_WD_deg_diagnostic" in group
            and pd.to_numeric(
                group["monthly_circular_mean_WD_deg_diagnostic"],
                errors="coerce",
            ).notna().any()
        )
        radius = positive(
            first_meaningful(group, "radius_m_max", np.nan)
        ) or positive(first_meaningful(group, "radius_m", np.nan))
        if radius is None:
            radius = math.nan
        radius_method, radius_source = infer_radius_method(group, radius)
        gpp_column = str(
            first_meaningful(group, "primary_GPP_column", "")
        )
        gpp_quality = str(
            first_meaningful(group, "GPP_monthly_quality_flag", "")
        )
        gpp_qc, gpp_qc_note = infer_gpp_qc_filter(group)
        fetch_columns = fetch_availability.get(str(site_id), set())
        fetch_filter_available = "FETCH_FILTER" in fetch_columns

        failures: list[str] = []
        limitations: list[str] = []
        if not wd_records_ok:
            failures.append("no valid actual WD record count")
        if not width_ok:
            failures.append("WD bin width is not consistently 10 degrees")
        if n_bins == 0:
            failures.append("no usable WD bins")
        if not fraction_ok:
            failures.append("WD_bin_fraction sum outside 0.99-1.01")
        if not uses_center:
            failures.append("wedge-center code compliance not verified")
        if applies_rotation:
            failures.append("no-180-degree-rotation rule not verified")
        if not math.isfinite(radius) or radius <= 0.0:
            failures.append("radius is missing/non-positive")
        if not gpp_qc:
            limitations.append(gpp_qc_note)
        if gpp_quality and gpp_quality.upper() != "OK":
            limitations.append(f"GPP quality flag={gpp_quality}")
        if not circular_present:
            limitations.append(
                "monthly circular-mean diagnostic is absent for this month"
            )
        if failures:
            status = "FAIL"
        elif limitations:
            status = "PASS_WITH_LIMITATION"
        else:
            status = "PASS"
        notes = static_notes + [
            "Monthly/daily mean WD is not used for geometry.",
            "FETCH_FILTER is not used for footprint radius or wedge geometry.",
            gpp_qc_note,
        ]
        notes.extend(f"FAIL: {value}" for value in failures)
        notes.extend(f"LIMITATION: {value}" for value in limitations)
        rows.append(
            {
                "SITE_ID": str(site_id),
                "YYYYMM": str(yyyymm),
                "uses_30min_or_hourly_WD_records": wd_records_ok,
                "WD_bin_width_deg": width_value,
                "n_WD_bins": n_bins,
                "WD_bin_fraction_sum": fraction_sum,
                "uses_monthly_mean_WD_for_geometry": False,
                "uses_daily_mean_WD_for_geometry": False,
                "monthly_circular_mean_WD_diagnostic_only": circular_present,
                "wedge_geometry_uses_WD_bin_center": uses_center,
                "applies_180_degree_rotation": applies_rotation,
                "radius_method": radius_method,
                "radius_m_used": radius,
                "radius_source": radius_source,
                "GPP_column_used": gpp_column,
                "GPP_quality_flag": gpp_quality,
                "GPP_QC_filter_used": gpp_qc,
                "FETCH_FILTER_available": fetch_filter_available,
                "FETCH_FILTER_used": False,
                "compliance_status": status,
                "compliance_notes": " ".join(notes),
            }
        )
    return pd.DataFrame(rows, columns=COMPLIANCE_COLUMNS)


# =============================================================================
# OPTIONAL REPAIRED CSV WRITING
# =============================================================================


def write_repaired_csvs(
    table: pd.DataFrame,
    original_columns: list[str],
    remaining_columns: list[str],
) -> list[Path]:
    """Write only new copies when explicitly enabled outside dry-run."""
    if DRY_RUN or not WRITE_REPAIRED_CSV:
        return []
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    helper_columns = [
        "_SITE_ID_NORMALIZED",
        "_YYYYMM_NORMALIZED",
        "_VALID_EXPORT_FLAG",
    ]
    output = table.drop(columns=helper_columns, errors="ignore").copy()
    paths = [
        OUTPUT_ROOT
        / (
            "final_metadata_recovered_ALL_SITES_monthly_WD_histogram_"
            "GPP_footprint_input_for_GEE.csv"
        ),
        OUTPUT_ROOT
        / "final_metadata_recovered_original_sites_monthly_GEE_input.csv",
        OUTPUT_ROOT
        / "final_metadata_recovered_remaining_sites_monthly_GEE_input.csv",
    ]
    output.drop(columns=["_source_group"], errors="ignore").to_csv(
        paths[0],
        index=False,
        encoding="utf-8-sig",
    )
    provenance = [
        "source_input_table",
        *PROVENANCE_COLUMNS,
    ]
    original_order = list(dict.fromkeys(original_columns + provenance))
    remaining_order = list(dict.fromkeys(remaining_columns + provenance))
    for group_name, path, preferred in (
        ("original", paths[1], original_order),
        ("remaining", paths[2], remaining_order),
    ):
        subset = output.loc[output["_source_group"].eq(group_name)].drop(
            columns=["_source_group"],
            errors="ignore",
        )
        order = preferred + [
            column for column in subset.columns if column not in preferred
        ]
        subset.reindex(columns=order).to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
        )
    return paths


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Run the final metadata recovery in dry-run mode by default."""
    if CREATE_TIFFS:
        raise RuntimeError(
            "CREATE_TIFFS must remain False. This script intentionally has no "
            "TIFF creation workflow."
        )
    if DRY_RUN and WRITE_REPAIRED_CSV:
        warn(
            "WRITE_REPAIRED_CSV=True is ignored while DRY_RUN=True. "
            "No repaired CSV will be written."
        )

    QC_OUTPUT.mkdir(parents=True, exist_ok=True)
    print("============================================================", flush=True)
    print("FLUXNET METADATA RECOVERY + METHOD COMPLIANCE", flush=True)
    print("============================================================", flush=True)
    print(f"DRY_RUN={DRY_RUN}", flush=True)
    print(f"WRITE_REPAIRED_CSV={WRITE_REPAIRED_CSV}", flush=True)
    print(f"CREATE_TIFFS={CREATE_TIFFS}", flush=True)
    print(
        f"APPLY_MEDIUM_CONFIDENCE_REPAIRS="
        f"{APPLY_MEDIUM_CONFIDENCE_REPAIRS}",
        flush=True,
    )
    print(f"Output root: {OUTPUT_ROOT}", flush=True)
    print(
        "Safety: no TIFF creation and no original-file overwrite are possible.",
        flush=True,
    )

    print("\n1. Loading GPP-repaired base table...", flush=True)
    (
        base_table,
        original_columns,
        remaining_columns,
        base_source,
    ) = load_base_table()
    ensure_final_provenance_columns(base_table)
    before_audit = site_month_audit(base_table)
    before_sites, before_keys = exportable_sets(before_audit)
    print(
        f"   Base source: {base_source}\n"
        f"   Exportable sites/months: {len(before_sites):,}/"
        f"{len(before_keys):,}",
        flush=True,
    )

    print("\n2. Loading manual web-research candidates...", flush=True)
    manual_candidates = load_manual_draft_candidates()
    print(f"   Manual candidate rows: {len(manual_candidates):,}", flush=True)

    print("\n3. Mining exact-site local ZIP BIF/BIFVARINFO metadata...", flush=True)
    zip_candidates = mine_zip_candidates()

    print("\n4. Mining extracted/project metadata tables...", flush=True)
    table_candidates = mine_tabular_metadata_candidates()

    print("\n5. Checking raw FETCH header diagnostics...", flush=True)
    fetch_candidates, fetch_availability = raw_fetch_header_candidates()

    all_candidates = (
        manual_candidates
        + zip_candidates
        + table_candidates
        + fetch_candidates
    )
    local_report = pd.DataFrame(
        [candidate.report for candidate in all_candidates],
        columns=LOCAL_CANDIDATE_COLUMNS,
    )
    local_report_path = QC_OUTPUT / "local_metadata_radius_candidates.csv"
    local_report.to_csv(
        local_report_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n6. Applying Priority A/B/C radius decisions...", flush=True)
    decisions_internal = build_radius_decisions(base_table, all_candidates)
    decision_report = decisions_internal.reindex(
        columns=RADIUS_DECISION_COLUMNS
    )
    decision_path = QC_OUTPUT / "radius_recovery_decision_table.csv"
    decision_report.to_csv(
        decision_path,
        index=False,
        encoding="utf-8-sig",
    )

    finalized = apply_safe_metadata_decisions(
        base_table,
        decisions_internal,
    )
    after_audit = site_month_audit(finalized)

    print("\n7. Building before/after and site-month reports...", flush=True)
    summary, recovered, blocked, result = build_before_after_reports(
        before_audit,
        after_audit,
        decisions_internal,
    )
    summary_path = (
        QC_OUTPUT / "final_metadata_recovery_before_after_summary.csv"
    )
    recovered_path = (
        QC_OUTPUT / "final_metadata_recovery_recovered_sites.csv"
    )
    blocked_path = (
        QC_OUTPUT / "final_metadata_recovery_still_blocked_sites.csv"
    )
    month_path = (
        QC_OUTPUT / "final_metadata_recovery_site_month_audit.csv"
    )
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    recovered.to_csv(recovered_path, index=False, encoding="utf-8-sig")
    blocked.to_csv(blocked_path, index=False, encoding="utf-8-sig")
    after_audit.to_csv(month_path, index=False, encoding="utf-8-sig")

    print("\n8. Building Method-compliance audit...", flush=True)
    compliance = build_method_compliance_audit(
        finalized,
        after_audit,
        fetch_availability,
    )
    compliance_path = QC_OUTPUT / "Method_compliance_audit.csv"
    compliance.to_csv(
        compliance_path,
        index=False,
        encoding="utf-8-sig",
    )

    repaired_paths = write_repaired_csvs(
        finalized,
        original_columns,
        remaining_columns,
    )

    print("\n============================================================", flush=True)
    print(
        "FINAL METADATA RECOVERY DRY RUN COMPLETE"
        if DRY_RUN
        else "FINAL METADATA RECOVERY COMPLETE",
        flush=True,
    )
    print("============================================================", flush=True)
    print("Before:", flush=True)
    print(
        f"  exportable sites = {result['before_sites']:,}\n"
        f"  exportable months = {result['before_months']:,}",
        flush=True,
    )
    print("After safe in-memory metadata recovery:", flush=True)
    print(
        f"  exportable sites = {result['after_sites']:,}\n"
        f"  exportable months = {result['after_months']:,}",
        flush=True,
    )
    print(
        "  newly recovered sites = "
        + (", ".join(result["recovered_sites"]) or "<none>"),
        flush=True,
    )
    print(
        "  still blocked sites = "
        + (", ".join(result["still_blocked_sites"]) or "<none>"),
        flush=True,
    )
    print(
        "  manual review sites = "
        + (", ".join(result["manual_review_sites"]) or "<none>"),
        flush=True,
    )
    print("  BE-Lon status = re-download required", flush=True)

    compliance_counts = compliance["compliance_status"].value_counts()
    print("Method compliance:", flush=True)
    for status in ("PASS", "PASS_WITH_LIMITATION", "FAIL"):
        print(
            f"  {status} = {int(compliance_counts.get(status, 0)):,}",
            flush=True,
        )

    print("\nQC reports:", flush=True)
    for path in (
        local_report_path,
        decision_path,
        compliance_path,
        summary_path,
        recovered_path,
        blocked_path,
        month_path,
    ):
        print(f"  {path}", flush=True)
    if repaired_paths:
        print("New repaired CSV copies:", flush=True)
        for path in repaired_paths:
            print(f"  {path}", flush=True)
    else:
        print(
            "No repaired final CSV was written. Set DRY_RUN=False and "
            "WRITE_REPAIRED_CSV=True only after reviewing QC.",
            flush=True,
        )
    print("No TIFF was created.", flush=True)


if __name__ == "__main__":
    main()
