#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Clean helper surface for the locked footprint-aware SIF TIFF pipeline.

This module contains only the helper functions used by
Create_FINAL_CLEAN_SIF_Companion_TIFFs_FOOTPRINT_AWARE.py:

* Windows-safe path utilities
* report writing
* GPP TIFF template inventory
* finalized GPP-reference CSV matching
* original SIF source discovery/date/schema inspection
* nearest-neighbour gridded SIF alignment
* GPP band readback and SIF/GPP summary math

It deliberately does not include any legacy swath point-output, point-centre
rasterization, sparse point raster, KMZ, or standalone circular-buffer workflow.
"""

from __future__ import annotations
# Original script filename (sanitized): SIF_TIFF_FOOTPRINT_AWARE_HELPERS.py; public name: SIF_TIFF_FOOTPRINT_AWARE_HELPERS.py
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
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import landscape, letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
except Exception:
    colors = None
    landscape = None
    letter = None
    getSampleStyleSheet = None
    Paragraph = None
    SimpleDocTemplate = None
    Spacer = None
    Table = None
    TableStyle = None


# These are overwritten by the active production script during configure_modules().
DRY_RUN = False
WRITE_OUTPUTS = True
OVERWRITE_EXISTING = False
PRODUCTS_TO_PROCESS: list[str] | None = None
SELECTED_SITES: list[str] | None = None
SELECTED_YYYYMM: list[str] | None = None
NODATA_VALUE = -9999.0
CREATE_MD_AND_PDF_REPORTS = True
CREATE_QUICKLOOKS = True

PROJECT_ROOT = Path(
    PROJECT_ROOT
)
GPP_TIFF_ROOT = PROJECT_ROOT / "Monthly_EC_Footprint_TIFFs_Python"
GPP_REFERENCE_CSV = (
    PROJECT_ROOT
    / "finalized_metadata_recovery"
    / "final_metadata_recovered_ALL_SITES_monthly_WD_histogram_GPP_footprint_input_for_GEE.csv"
)
SIF_ROOT = PROJECT_ROOT / "SIF Datasets"
OUTPUT_ROOT = PROJECT_ROOT / "SIF_TIFF_output"
COMPANION_ROOT = OUTPUT_ROOT
SUMMARY_ROOT = OUTPUT_ROOT / "reports"
QC_ROOT = OUTPUT_ROOT / "reports"
VALIDATION_ROOT = OUTPUT_ROOT / "reports"
QUICKLOOK_ROOT = OUTPUT_ROOT / "quicklooks"

PRODUCTS = ["GOSIF", "GOME-2", "gosat", "OCO-2", "OCO-3", "TROPOSIF"]
SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".nc", ".nc4", ".h5", ".hdf", ".he5"}
RESAMPLING_FOR_GRIDDED_SIF = "nearest"
SWATH_RASTERIZATION_METHOD = "footprint_polygon_to_GPP_grid_cells"
MULTIPLE_SWATH_POINTS_PER_CELL = "mean"

PRODUCT_INVALID_VALUES: dict[str, set[float]] = {"GOSIF": {32766.0, 32767.0}}

GPP_BAND_DESCRIPTIONS = (
    "footprint_probability",
    "tower_GPP",
    "blind_circular_buffer",
    "n_WD_bins",
    "n_valid_30min_WD_records",
    "footprint_probability_x_GPP",
)

SIF_BAND_DESCRIPTIONS = (
    "footprint_probability",
    "SIF_unmasked_aligned_to_GPP_grid",
    "blind_circular_buffer",
    "n_WD_bins",
    "n_valid_30min_WD_records",
    "footprint_probability_x_SIF",
    "SIF_masked_by_footprint_probability",
    "SIF_masked_by_blind_circular_buffer",
)

GPP_FILENAME_RE = re.compile(
    r"^(?P<site_safe>.+)_monthly_30min_WD_histogram_"
    r"footprint_GPP_(?P<yyyymm>\d{6})\.tif$",
    re.IGNORECASE,
)

PRODUCT_SCHEMAS: dict[str, dict[str, Any]] = {
    "GOME-2": {
        "sif": ["Daily_Averaged_SIF", "SIF_740", "SIF_Unadjusted"],
        "lat": ["Latitude"],
        "lon": ["Longitude"],
        "uncertainty": ["SIF_Uncertainty"],
        "quality": ["Quality_Flag"],
    },
    "gosat": {
        "sif": ["Daily_SIF_740nm", "SIF_740nm", "Science/SIF_757nm"],
        "lat": ["Latitude", "Geolocation/latitude"],
        "lon": ["Longitude", "Geolocation/longitude"],
        "uncertainty": ["SIF_Uncertainty_740nm", "Science/SIF_Uncertainty_757nm"],
        "quality": ["Quality_Flag", "Science/sounding_qual_flag"],
    },
    "GOSIF": {
        "sif": ["band_1"],
        "lat": [],
        "lon": [],
        "uncertainty": [],
        "quality": [],
    },
    "OCO-2": {
        "sif": ["Daily_SIF_740nm", "SIF_740nm", "Science/SIF_757nm"],
        "lat": ["Latitude", "Geolocation/latitude"],
        "lon": ["Longitude", "Geolocation/longitude"],
        "uncertainty": ["SIF_Uncertainty_740nm", "Science/SIF_Uncertainty_757nm"],
        "quality": ["SimplyGoodOrBadQualityFlag", "Quality_Flag"],
    },
    "OCO-3": {
        "sif": ["Daily_SIF_740nm", "SIF_740nm", "Science/SIF_757nm"],
        "lat": ["Latitude", "Geolocation/latitude"],
        "lon": ["Longitude", "Geolocation/longitude"],
        "uncertainty": ["SIF_Uncertainty_740nm", "Science/SIF_Uncertainty_757nm"],
        "quality": ["SimplyGoodOrBadQualityFlag", "Quality_Flag"],
    },
    "TROPOSIF": {
        "sif": [
            "PRODUCT/SIF_Corr_743",
            "PRODUCT/SIF_Corr_735",
            "PRODUCT/SIF_743",
            "PRODUCT/SIF_735",
        ],
        "lat": ["PRODUCT/latitude"],
        "lon": ["PRODUCT/longitude"],
        "uncertainty": ["PRODUCT/SIF_ERROR_743", "PRODUCT/SIF_ERROR_735"],
        "quality": [],
    },
}

SIF_INVENTORY_COLUMNS = [
    "product",
    "file_path",
    "filename",
    "extension",
    "detected_year",
    "detected_month",
    "detected_YYYYMM",
    "date_detection_method",
    "is_monthly",
    "is_daily_or_submonthly",
    "is_gridded",
    "is_swath_or_point",
    "detected_crs",
    "detected_sif_variable",
    "detected_lat_variable",
    "detected_lon_variable",
    "read_status",
    "notes",
]


def native_path(path: Path | str) -> str:
    value = os.path.abspath(os.fspath(path))
    if os.name == "nt" and not value.startswith("\\\\?\\"):
        if value.startswith("\\\\"):
            return "\\\\?\\UNC\\" + value.lstrip("\\")
        return "\\\\?\\" + value
    return value


def path_exists(path: Path | str) -> bool:
    return os.path.exists(native_path(path))


def ensure_directory(path: Path) -> None:
    os.makedirs(native_path(path), exist_ok=True)


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def site_to_safe(site_id: Any) -> str:
    return str(site_id).strip().replace("-", "_")


def safe_to_site(site_safe: Any) -> str:
    return str(site_safe).strip().replace("_", "-")


def valid_export_flag(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(np.isfinite(value) and float(value) == 1.0)
    return str(value).strip().lower() in {"1", "true", "yes"}


def normalize_yyyymm(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if re.fullmatch(r"\d{6}(?:\.0+)?", text):
        text = text[:6]
    else:
        digits = re.sub(r"\D", "", text)
        if len(digits) < 6:
            return None
        text = digits[:6]
    year, month = int(text[:4]), int(text[4:6])
    if not (1900 <= year <= 2100 and 1 <= month <= 12):
        return None
    return text


def iter_progress(
    values: Iterable[Any],
    *,
    total: int | None = None,
    description: str = "",
    unit: str = "item",
) -> Iterable[Any]:
    if tqdm is None:
        return values
    return tqdm(values, total=total, desc=description, unit=unit, dynamic_ncols=True)


def selected_site(site_id: str) -> bool:
    return SELECTED_SITES is None or site_id in set(SELECTED_SITES)


def selected_month(yyyymm: str) -> bool:
    return SELECTED_YYYYMM is None or yyyymm in set(SELECTED_YYYYMM)


def valid_data_mask(array: np.ndarray, nodata: Any = None) -> np.ndarray:
    nodata = NODATA_VALUE if nodata is None else nodata
    mask = np.isfinite(array)
    if nodata is not None:
        try:
            value = float(nodata)
            if math.isfinite(value):
                mask &= array != value
        except (TypeError, ValueError):
            pass
    return mask


def values_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(a.shape == b.shape and np.array_equal(a, b, equal_nan=True))


def compact_paths(paths: Iterable[str], max_chars: int = 28_000) -> str:
    values = [str(path) for path in paths]
    text = " | ".join(values)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 80] + f" ... [{len(values)} paths total]"


def transform_text(transform: Any) -> str:
    return ", ".join(f"{float(value):.15g}" for value in tuple(transform)[:6])


def bounds_text(bounds: Any) -> str:
    return ", ".join(
        f"{float(value):.15g}"
        for value in (bounds.left, bounds.bottom, bounds.right, bounds.top)
    )


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def decode_attr(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(-1)[0]
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def dataset_attr(dataset: h5py.Dataset, names: Iterable[str]) -> Any:
    for name in names:
        if name in dataset.attrs:
            return decode_attr(dataset.attrs[name])
    return None


def first_existing_hdf_path(handle: h5py.File, candidates: Iterable[str]) -> str:
    for candidate in candidates:
        normalized = candidate.lstrip("/")
        if normalized in handle and isinstance(handle[normalized], h5py.Dataset):
            return normalized
    return ""


def dataframe_markdown(frame: pd.DataFrame, max_rows: int = 40) -> str:
    if frame.empty:
        return "_No rows._\n"
    preview = frame.head(max_rows).copy()
    try:
        return preview.to_markdown(index=False) + "\n"
    except Exception:
        return "```\n" + preview.to_string(index=False) + "\n```\n"


def write_markdown(
    path: Path,
    title: str,
    summary_lines: list[str],
    frame: pd.DataFrame | None = None,
    max_rows: int = 40,
) -> None:
    ensure_directory(path.parent)
    with open(native_path(path), "w", encoding="utf-8") as handle:
        handle.write(f"# {title}\n\n")
        handle.write(f"Generated: {now_text()}\n\n")
        for line in summary_lines:
            handle.write(f"- {line}\n")
        if frame is not None:
            handle.write(f"\nRows in full CSV: {len(frame):,}\n\n")
            handle.write(
                "The CSV is the complete machine-readable report; "
                "this document shows a preview.\n\n"
            )
            handle.write(dataframe_markdown(frame, max_rows=max_rows))


def pdf_paragraph(text: str, style: Any) -> Any:
    escaped = (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )
    return Paragraph(escaped, style)


def write_pdf(
    path: Path,
    title: str,
    summary_lines: list[str],
    frame: pd.DataFrame | None = None,
    max_rows: int = 25,
) -> None:
    ensure_directory(path.parent)
    if SimpleDocTemplate is None:
        raise RuntimeError("reportlab is required because PDF reports were requested.")
    document = SimpleDocTemplate(
        native_path(path),
        pagesize=landscape(letter),
        rightMargin=24,
        leftMargin=24,
        topMargin=24,
        bottomMargin=24,
    )
    styles = getSampleStyleSheet()
    story = [
        pdf_paragraph(title, styles["Title"]),
        Spacer(1, 8),
        pdf_paragraph(f"Generated: {now_text()}", styles["Normal"]),
        Spacer(1, 8),
    ]
    for line in summary_lines:
        story.append(pdf_paragraph(f"• {line}", styles["Normal"]))
    if frame is not None:
        story.extend(
            [
                Spacer(1, 10),
                pdf_paragraph(f"Rows in full CSV: {len(frame):,}", styles["Heading2"]),
                pdf_paragraph(
                    "The CSV contains the complete report. This PDF contains a compact preview.",
                    styles["Normal"],
                ),
                Spacer(1, 8),
            ]
        )
        preview = frame.head(max_rows).copy()
        if not preview.empty:
            preview = preview.iloc[:, : min(8, len(preview.columns))]
            table_data = [list(preview.columns)]
            for _, row in preview.iterrows():
                table_data.append(
                    [
                        textwrap.shorten(str(value), width=38, placeholder="...")
                        for value in row.tolist()
                    ]
                )
            table = Table(table_data, repeatRows=1)
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 5.5),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )
            story.append(table)
    document.build(story)


def write_report_bundle(
    base: Path,
    title: str,
    frame: pd.DataFrame,
    summary_lines: list[str],
    max_rows: int = 40,
) -> tuple[Path, Path, Path]:
    csv_path = base.with_suffix(".csv")
    md_path = base.with_suffix(".md")
    pdf_path = base.with_suffix(".pdf")
    ensure_directory(csv_path.parent)
    frame.to_csv(native_path(csv_path), index=False, encoding="utf-8-sig")
    if CREATE_MD_AND_PDF_REPORTS:
        write_markdown(md_path, title, summary_lines, frame, max_rows=max_rows)
        write_pdf(pdf_path, title, summary_lines, frame, max_rows=min(max_rows, 25))
    return csv_path, md_path, pdf_path


def inventory_gpp_templates() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    paths = sorted(
        path
        for path in GPP_TIFF_ROOT.rglob("*.tif")
        if GPP_FILENAME_RE.match(path.name)
    )
    report_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for path in iter_progress(
        paths, total=len(paths), description="Inventorying GPP TIFF templates", unit="TIFF"
    ):
        match = GPP_FILENAME_RE.match(path.name)
        assert match is not None
        site_safe = match.group("site_safe")
        yyyymm = match.group("yyyymm")
        try:
            with rasterio.open(native_path(path)) as dataset:
                tags = dataset.tags()
                site_id = str(tags.get("SITE_ID", "")).strip() or safe_to_site(site_safe)
                descriptions = tuple(description or "" for description in dataset.descriptions)
                valid = bool(
                    dataset.count == 6
                    and descriptions == GPP_BAND_DESCRIPTIONS
                    and dataset.crs is not None
                    and dataset.width > 0
                    and dataset.height > 0
                )
                record = {
                    "SITE_ID": site_id,
                    "SITE_SAFE": site_safe,
                    "YYYYMM": yyyymm,
                    "year": int(yyyymm[:4]),
                    "month": int(yyyymm[4:6]),
                    "gpp_tif": str(path),
                    "crs": dataset.crs.to_string() if dataset.crs else "",
                    "transform": transform_text(dataset.transform),
                    "width": dataset.width,
                    "height": dataset.height,
                    "bounds": bounds_text(dataset.bounds),
                    "bounds_left": float(dataset.bounds.left),
                    "bounds_bottom": float(dataset.bounds.bottom),
                    "bounds_right": float(dataset.bounds.right),
                    "bounds_top": float(dataset.bounds.top),
                    "pixel_size_x": abs(float(dataset.transform.a)),
                    "pixel_size_y": abs(float(dataset.transform.e)),
                    "dtype": "|".join(dataset.dtypes),
                    "nodata": dataset.nodata,
                    "band_count": dataset.count,
                    "band_descriptions": "|".join(descriptions),
                    "valid_gpp_template": valid,
                    "_crs_obj": dataset.crs,
                    "_transform_obj": dataset.transform,
                    "_bounds_tuple": (
                        float(dataset.bounds.left),
                        float(dataset.bounds.bottom),
                        float(dataset.bounds.right),
                        float(dataset.bounds.top),
                    ),
                    "_band_descriptions": descriptions,
                }
                records.append(record)
                report_rows.append({key: value for key, value in record.items() if not key.startswith("_")})
        except Exception as error:
            report_rows.append(
                {
                    "SITE_ID": "",
                    "SITE_SAFE": site_safe,
                    "YYYYMM": yyyymm,
                    "gpp_tif": str(path),
                    "valid_gpp_template": False,
                    "error_message": f"{type(error).__name__}: {error}",
                }
            )
    return pd.DataFrame(report_rows), [record for record in records if record["valid_gpp_template"]]


def load_gpp_reference() -> tuple[pd.DataFrame, dict[tuple[str, str], dict[str, Any]]]:
    if not path_exists(GPP_REFERENCE_CSV):
        raise FileNotFoundError(f"GPP reference CSV not found: {GPP_REFERENCE_CSV}")
    header = pd.read_csv(native_path(GPP_REFERENCE_CSV), nrows=0).columns
    wanted = [
        name
        for name in (
            "SITE_ID",
            "YYYYMM",
            "year",
            "month",
            "IGBP",
            "tower_GPP_monthly",
            "valid_for_GEE_export",
        )
        if name in header
    ]
    frame = pd.read_csv(native_path(GPP_REFERENCE_CSV), usecols=wanted, low_memory=False)
    if "SITE_ID" not in frame:
        raise ValueError("GPP reference CSV is missing SITE_ID.")
    if "YYYYMM" in frame:
        frame["_YYYYMM"] = frame["YYYYMM"].map(normalize_yyyymm)
    elif {"year", "month"}.issubset(frame.columns):
        frame["_YYYYMM"] = [
            normalize_yyyymm(f"{int(y):04d}{int(m):02d}")
            if pd.notna(y) and pd.notna(m)
            else None
            for y, m in zip(frame["year"], frame["month"])
        ]
    else:
        raise ValueError("GPP reference CSV has no usable date columns.")
    frame["_SITE_ID"] = frame["SITE_ID"].astype(str).str.strip()
    valid = frame["_SITE_ID"].ne("") & frame["_YYYYMM"].notna()
    if "valid_for_GEE_export" in frame:
        valid &= frame["valid_for_GEE_export"].map(valid_export_flag)
    if "tower_GPP_monthly" in frame:
        gpp = pd.to_numeric(frame["tower_GPP_monthly"], errors="coerce")
        valid &= np.isfinite(gpp)
    valid_frame = frame.loc[valid].copy()
    rows = []
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for (site_id, yyyymm), group in valid_frame.groupby(["_SITE_ID", "_YYYYMM"], sort=True):
        igbp = ""
        if "IGBP" in group:
            values = group["IGBP"].dropna().astype(str)
            igbp = values.iloc[0] if len(values) else ""
        tower_gpp = np.nan
        if "tower_GPP_monthly" in group:
            values = pd.to_numeric(group["tower_GPP_monthly"], errors="coerce")
            values = values[np.isfinite(values)]
            tower_gpp = float(values.iloc[0]) if len(values) else np.nan
        record = {
            "SITE_ID": site_id,
            "SITE_SAFE": site_to_safe(site_id),
            "YYYYMM": yyyymm,
            "IGBP": igbp,
            "tower_GPP_monthly": tower_gpp,
            "valid_csv_rows": len(group),
        }
        rows.append(record)
        lookup[(site_id, yyyymm)] = record
    return pd.DataFrame(rows), lookup


def build_csv_template_check(
    csv_sites: pd.DataFrame, templates: list[dict[str, Any]]
) -> pd.DataFrame:
    csv_keys = set(zip(csv_sites["SITE_ID"], csv_sites["YYYYMM"]))
    template_keys = {(record["SITE_ID"], record["YYYYMM"]) for record in templates}
    all_keys = sorted(csv_keys | template_keys)
    rows = []
    for site_id, yyyymm in all_keys:
        in_csv = (site_id, yyyymm) in csv_keys
        in_gpp = (site_id, yyyymm) in template_keys
        rows.append(
            {
                "SITE_ID": site_id,
                "SITE_SAFE": site_to_safe(site_id),
                "YYYYMM": yyyymm,
                "valid_in_gpp_reference_csv": in_csv,
                "gpp_tif_found": in_gpp,
                "match_status": (
                    "MATCHED"
                    if in_csv and in_gpp
                    else ("CSV_VALID_BUT_NO_GPP_TIF" if in_csv else "GPP_TIF_NOT_IN_VALID_CSV")
                ),
            }
        )
    return pd.DataFrame(rows)


@dataclass
class DateDetection:
    year: int | None
    month: int | None
    yyyymm: str
    method: str
    is_monthly: bool
    is_submonthly: bool


def detect_sif_date(product: str, path: Path) -> DateDetection:
    name = path.name
    patterns: list[tuple[str, str, bool]] = []
    if product == "GOSIF":
        patterns.append((r"GOSIF_(\d{4})(?:[._-]?M?)(\d{2})(?!\d)", "GOSIF_YYYYMM_or_YYYY.MMM", True))
    elif product == "GOME-2":
        patterns.append((r"GOME-2[A-Z]?\.(\d{4})(\d{2})(\d{2})", "GOME2_YYYYMMDD", False))
    elif product in {"gosat", "OCO-2", "OCO-3"}:
        patterns.append((r"LtSIF_(\d{2})(\d{2})(\d{2})(?!\d)", "LtSIF_YYMMDD", False))
    elif product == "TROPOSIF":
        patterns.append((r"(\d{4})-(\d{2})-(\d{2})", "TROPOSIF_YYYY-MM-DD", False))
    for pattern, method, monthly in patterns:
        match = re.search(pattern, name, flags=re.IGNORECASE)
        if not match:
            continue
        groups = match.groups()
        if len(groups) == 2:
            year, month = int(groups[0]), int(groups[1])
        elif len(groups) == 3 and len(groups[0]) == 2:
            short_year = int(groups[0])
            year = 2000 + short_year if short_year <= 79 else 1900 + short_year
            month = int(groups[1])
        else:
            year, month = int(groups[0]), int(groups[1])
        if 1900 <= year <= 2100 and 1 <= month <= 12:
            return DateDetection(year, month, f"{year:04d}{month:02d}", method, monthly, not monthly)
    generic = re.search(r"(?<!\d)(19|20)(\d{2})[-_.]?(\d{2})[-_.]?(\d{2})(?!\d)", name)
    if generic:
        year = int(generic.group(1) + generic.group(2))
        month = int(generic.group(3))
        if 1 <= month <= 12:
            return DateDetection(year, month, f"{year:04d}{month:02d}", "generic_YYYYMMDD_filename", False, True)
    return DateDetection(None, None, "", "DATE_NOT_DETECTED", False, False)


def fallback_hdf_variable(handle: h5py.File, kind: str) -> str:
    candidates: list[tuple[int, str]] = []

    def visitor(name: str, obj: Any) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        lower = name.lower()
        score = 0
        if kind == "sif" and "sif" in lower:
            score = 10
            if "uncert" in lower or "error" in lower or "relative" in lower:
                score -= 8
            if "daily" in lower or "corr" in lower:
                score += 3
        elif kind == "lat" and (lower.endswith("latitude") or lower.endswith("/lat")):
            score = 10
        elif kind == "lon" and (lower.endswith("longitude") or lower.endswith("/lon")):
            score = 10
        if score > 0:
            candidates.append((score, name))

    handle.visititems(visitor)
    if not candidates:
        return ""
    return sorted(candidates, key=lambda item: (-item[0], item[1]))[0][1]


def inspect_hdf_sif_file(product: str, path: Path) -> dict[str, Any]:
    schema = PRODUCT_SCHEMAS[product]
    with h5py.File(native_path(path), "r") as handle:
        sif_path = first_existing_hdf_path(handle, schema["sif"]) or fallback_hdf_variable(handle, "sif")
        lat_path = first_existing_hdf_path(handle, schema["lat"]) or fallback_hdf_variable(handle, "lat")
        lon_path = first_existing_hdf_path(handle, schema["lon"]) or fallback_hdf_variable(handle, "lon")
        uncertainty_path = first_existing_hdf_path(handle, schema["uncertainty"])
        quality_path = first_existing_hdf_path(handle, schema["quality"])
        if not sif_path:
            raise ValueError("No SIF-like variable was detected.")
        if not lat_path or not lon_path:
            raise ValueError("Latitude/longitude variables were not detected.")
        sif_shape = handle[sif_path].shape
        lat_shape = handle[lat_path].shape
        lon_shape = handle[lon_path].shape
        is_point = bool(
            len(sif_shape) >= 1
            and len(lat_shape) >= 1
            and len(lon_shape) >= 1
            and sif_shape[0] == lat_shape[0] == lon_shape[0]
        )
        is_gridded = False
        if len(sif_shape) == 2:
            if len(lat_shape) == 1 and len(lon_shape) == 1:
                is_gridded = sif_shape in {
                    (lat_shape[0], lon_shape[0]),
                    (lon_shape[0], lat_shape[0]),
                }
            elif lat_shape == lon_shape == sif_shape:
                is_gridded = True
        if is_gridded:
            is_point = False
        return {
            "is_gridded": is_gridded,
            "is_swath_or_point": is_point,
            "detected_crs": "EPSG:4326",
            "detected_sif_variable": sif_path,
            "detected_lat_variable": lat_path,
            "detected_lon_variable": lon_path,
            "_uncertainty_variable": uncertainty_path,
            "_quality_variable": quality_path,
            "_sif_shape": sif_shape,
            "notes": f"SIF shape={sif_shape}; lat shape={lat_shape}; lon shape={lon_shape}.",
        }


def inspect_raster_sif_file(product: str, path: Path) -> dict[str, Any]:
    with rasterio.open(native_path(path)) as dataset:
        if dataset.crs is None:
            raise ValueError("Raster has no CRS.")
        if dataset.count < 1:
            raise ValueError("Raster has no bands.")
        notes = f"Raster shape={dataset.height}x{dataset.width}; dtype={dataset.dtypes[0]}."
        if product == "GOSIF":
            notes += " Product sentinel codes 32766/32767 are masked; other raw values are preserved."
        return {
            "is_gridded": True,
            "is_swath_or_point": False,
            "detected_crs": dataset.crs.to_string(),
            "detected_sif_variable": "band_1",
            "detected_lat_variable": "",
            "detected_lon_variable": "",
            "_uncertainty_variable": "",
            "_quality_variable": "",
            "_sif_bounds": (
                float(dataset.bounds.left),
                float(dataset.bounds.bottom),
                float(dataset.bounds.right),
                float(dataset.bounds.top),
            ),
            "_sif_transform": dataset.transform,
            "_sif_crs_obj": dataset.crs,
            "_sif_nodata": dataset.nodata,
            "notes": notes,
        }


def inspect_sif_file(product: str, path: Path) -> dict[str, Any]:
    date = detect_sif_date(product, path)
    record = {
        "product": product,
        "file_path": str(path),
        "filename": path.name,
        "extension": path.suffix.lower(),
        "detected_year": date.year if date.year is not None else "",
        "detected_month": date.month if date.month is not None else "",
        "detected_YYYYMM": date.yyyymm,
        "date_detection_method": date.method,
        "is_monthly": date.is_monthly,
        "is_daily_or_submonthly": date.is_submonthly,
        "is_gridded": False,
        "is_swath_or_point": False,
        "detected_crs": "",
        "detected_sif_variable": "",
        "detected_lat_variable": "",
        "detected_lon_variable": "",
        "read_status": "UNREADABLE",
        "notes": "",
        "_uncertainty_variable": "",
        "_quality_variable": "",
    }
    try:
        details = inspect_raster_sif_file(product, path) if path.suffix.lower() in {".tif", ".tiff"} else inspect_hdf_sif_file(product, path)
        record.update(details)
        if not date.yyyymm:
            record["read_status"] = "READABLE_DATE_NOT_DETECTED"
            record["notes"] += " Date could not be inferred."
        else:
            record["read_status"] = "OK"
    except Exception as error:
        record["notes"] = f"{type(error).__name__}: {error}"
    return record


def discover_sif_sources(products: list[str]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    paths: list[tuple[str, Path]] = []
    for product in products:
        folder = SIF_ROOT / product
        if not path_exists(folder):
            continue
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                paths.append((product, path))
    product_order = {product: index for index, product in enumerate(products)}
    paths.sort(key=lambda item: (product_order.get(item[0], 999), str(item[1]).lower()))
    records: list[dict[str, Any]] = []
    for product, path in iter_progress(paths, total=len(paths), description="Inventorying SIF files", unit="file"):
        records.append(inspect_sif_file(product, path))
    report = pd.DataFrame(records).reindex(columns=SIF_INVENTORY_COLUMNS)
    usable = [
        record
        for record in records
        if record["read_status"] == "OK" and record["detected_YYYYMM"]
    ]
    return report, usable


def filter_templates(
    templates: list[dict[str, Any]], csv_lookup: dict[tuple[str, str], dict[str, Any]]
) -> list[dict[str, Any]]:
    return [
        template
        for template in templates
        if (template["SITE_ID"], template["YYYYMM"]) in csv_lookup
        and selected_site(template["SITE_ID"])
        and selected_month(template["YYYYMM"])
    ]


def build_month_matching_report(
    templates: list[dict[str, Any]],
    products: list[str],
    usable_sources: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for source in usable_sources:
        grouped[(source["product"], source["detected_YYYYMM"])].append(source)
    rows = []
    matches = []
    for template in templates:
        for product in products:
            sources = grouped.get((product, template["YYYYMM"]), [])
            if sources:
                kinds = {"gridded_raster" if source["is_gridded"] else "point_or_swath" for source in sources}
                if len(kinds) != 1:
                    status = "MIXED_OR_AMBIGUOUS_SOURCE_FORMAT"
                else:
                    status = "MATCHED_LOCAL_SIF_MONTH"
                    matches.append(
                        {
                            "product": product,
                            "SITE_ID": template["SITE_ID"],
                            "SITE_SAFE": template["SITE_SAFE"],
                            "YYYYMM": template["YYYYMM"],
                            "gpp_tif": template["gpp_tif"],
                            "template": template,
                            "sif_records": sources,
                            "source_type": next(iter(kinds)),
                        }
                    )
            else:
                status = "NO_LOCAL_SIF_FOR_MONTH"
            rows.append(
                {
                    "product": product,
                    "SITE_ID": template["SITE_ID"],
                    "SITE_SAFE": template["SITE_SAFE"],
                    "YYYYMM": template["YYYYMM"],
                    "gpp_tif": template["gpp_tif"],
                    "n_sif_source_files_for_month": len(sources),
                    "sif_source_files": compact_paths(source["file_path"] for source in sources),
                    "match_status": status,
                    "notes": (
                        "Readable local SIF source month matched."
                        if sources
                        else "No readable local SIF file exists for this month."
                    ),
                }
            )
    return pd.DataFrame(rows), matches


def scaled_hdf_array(dataset: h5py.Dataset) -> tuple[np.ndarray, float]:
    raw = np.asarray(dataset[...])
    values = raw.astype(np.float64, copy=False)
    fill = safe_float(dataset_attr(dataset, ["_FillValue", "missing_value"]))
    invalid = ~np.isfinite(values)
    if fill is not None:
        invalid |= values == fill
    scale = safe_float(dataset_attr(dataset, ["scale_factor"]))
    offset = safe_float(dataset_attr(dataset, ["add_offset"]))
    scale = 1.0 if scale is None else scale
    offset = 0.0 if offset is None else offset
    values = values * scale + offset
    values[invalid] = NODATA_VALUE
    return values.astype(np.float32), NODATA_VALUE


def regular_grid_hdf_descriptor(record: dict[str, Any]) -> dict[str, Any]:
    path = Path(record["file_path"])
    with h5py.File(native_path(path), "r") as handle:
        sif, nodata = scaled_hdf_array(handle[record["detected_sif_variable"]])
        lat = np.asarray(handle[record["detected_lat_variable"]][...], dtype=np.float64).squeeze()
        lon = np.asarray(handle[record["detected_lon_variable"]][...], dtype=np.float64).squeeze()
    if lat.ndim != 1 or lon.ndim != 1 or sif.ndim != 2:
        raise ValueError("HDF grid is not a regular 1-D lon/lat raster.")
    if sif.shape == (lon.size, lat.size):
        sif = sif.T
    if sif.shape != (lat.size, lon.size):
        raise ValueError(f"SIF shape {sif.shape} does not match lat/lon axes {lat.size}/{lon.size}.")
    if lon[0] > lon[-1]:
        lon = lon[::-1]
        sif = sif[:, ::-1]
    if lat[0] < lat[-1]:
        lat = lat[::-1]
        sif = sif[::-1, :]
    dx_values = np.diff(lon)
    dy_values = np.abs(np.diff(lat))
    dx = float(np.median(dx_values))
    dy = float(np.median(dy_values))
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("Invalid regular-grid coordinate spacing.")
    if not np.allclose(dx_values, dx, rtol=1e-5, atol=1e-10):
        raise ValueError("Longitude spacing is not regular.")
    if not np.allclose(dy_values, dy, rtol=1e-5, atol=1e-10):
        raise ValueError("Latitude spacing is not regular.")
    transform = from_origin(float(lon[0] - dx / 2.0), float(lat[0] + dy / 2.0), dx, dy)
    bounds = rasterio.transform.array_bounds(sif.shape[0], sif.shape[1], transform)
    return {
        "array": sif,
        "crs": CRS.from_epsg(4326),
        "transform": transform,
        "bounds": tuple(bounds),
        "nodata": nodata,
    }


def align_one_gridded_source(source: dict[str, Any], template: dict[str, Any]) -> np.ndarray:
    if RESAMPLING_FOR_GRIDDED_SIF != "nearest":
        raise ValueError("Only nearest gridded resampling is permitted.")
    destination = np.full((template["height"], template["width"]), NODATA_VALUE, dtype=np.float32)
    path = Path(source["file_path"])
    if path.suffix.lower() in {".tif", ".tiff"}:
        with rasterio.open(native_path(path)) as dataset:
            reproject(
                source=rasterio.band(dataset, 1),
                destination=destination,
                src_transform=dataset.transform,
                src_crs=dataset.crs,
                src_nodata=dataset.nodata,
                dst_transform=template["_transform_obj"],
                dst_crs=template["_crs_obj"],
                dst_nodata=NODATA_VALUE,
                resampling=Resampling.nearest,
            )
    else:
        descriptor = regular_grid_hdf_descriptor(source)
        reproject(
            source=descriptor["array"],
            destination=destination,
            src_transform=descriptor["transform"],
            src_crs=descriptor["crs"],
            src_nodata=descriptor["nodata"],
            dst_transform=template["_transform_obj"],
            dst_crs=template["_crs_obj"],
            dst_nodata=NODATA_VALUE,
            resampling=Resampling.nearest,
        )
    invalid_values = PRODUCT_INVALID_VALUES.get(source["product"], set())
    if invalid_values:
        destination[np.isin(destination, list(invalid_values))] = NODATA_VALUE
    destination[~np.isfinite(destination)] = NODATA_VALUE
    return destination.astype(np.float32, copy=False)


def dataset_chunk_values(dataset: h5py.Dataset, start: int, stop: int, mode: str) -> np.ndarray:
    values = np.asarray(dataset[start:stop]).astype(np.float64, copy=False)
    fill = safe_float(dataset_attr(dataset, ["_FillValue", "missing_value"]))
    invalid = ~np.isfinite(values)
    if fill is not None:
        invalid |= values == fill
    scale = safe_float(dataset_attr(dataset, ["scale_factor"]))
    offset = safe_float(dataset_attr(dataset, ["add_offset"]))
    scale = 1.0 if scale is None else scale
    offset = 0.0 if offset is None else offset
    values = values * scale + offset
    values[invalid] = np.nan
    if values.ndim > 1:
        flattened = values.reshape(values.shape[0], -1)
        finite = np.isfinite(flattened)
        counts = finite.sum(axis=1)
        if mode == "quality":
            temporary = np.where(finite, flattened, -np.inf)
            values = temporary.max(axis=1)
            values[counts == 0] = np.nan
        else:
            sums = np.where(finite, flattened, 0.0).sum(axis=1, dtype=np.float64)
            values = np.full(counts.shape, np.nan, dtype=np.float64)
            values[counts > 0] = sums[counts > 0] / counts[counts > 0]
    return np.asarray(values).reshape(-1)


def read_gpp_arrays(gpp_path: Path) -> tuple[list[np.ndarray], dict[str, Any], dict[str, str]]:
    with rasterio.open(native_path(gpp_path)) as dataset:
        arrays = [dataset.read(index).astype(np.float32, copy=False) for index in range(1, 7)]
        profile = dataset.profile.copy()
        tags = dataset.tags()
    return arrays, profile, tags


def read_existing_band2(output_path: Path) -> np.ndarray:
    with rasterio.open(native_path(output_path)) as dataset:
        if dataset.count < 2:
            return np.full((dataset.height, dataset.width), NODATA_VALUE, dtype=np.float32)
        return dataset.read(2).astype(np.float32, copy=False)


def calculate_analysis_row(
    match: dict[str, Any],
    sif_array: np.ndarray,
    output_path: Path,
    csv_lookup: dict[tuple[str, str], dict[str, Any]],
    status: str,
    notes: str,
) -> dict[str, Any]:
    with rasterio.open(native_path(Path(match["gpp_tif"]))) as gpp:
        probability = gpp.read(1).astype(np.float32)
        tower = gpp.read(2).astype(np.float32)
        blind = gpp.read(3).astype(np.float32)
    valid = valid_data_mask(sif_array)
    in_footprint = valid & np.isfinite(probability) & (probability > 0)
    in_blind = valid & np.isfinite(blind) & (blind == 1)
    weight_sum = float(probability[in_footprint].sum(dtype=np.float64))
    weighted_mean = (
        float(
            (
                sif_array[in_footprint].astype(np.float64)
                * probability[in_footprint].astype(np.float64)
            ).sum(dtype=np.float64)
            / weight_sum
        )
        if in_footprint.any() and weight_sum > 0
        else np.nan
    )
    metadata = csv_lookup.get((match["SITE_ID"], match["YYYYMM"]), {})
    tower_valid = tower[np.isfinite(tower)]
    tower_gpp = float(tower_valid.flat[0]) if tower_valid.size else metadata.get("tower_GPP_monthly", np.nan)
    return {
        "SITE_ID": match["SITE_ID"],
        "SITE_SAFE": match["SITE_SAFE"],
        "YYYYMM": match["YYYYMM"],
        "year": int(match["YYYYMM"][:4]),
        "month": int(match["YYYYMM"][4:6]),
        "IGBP": metadata.get("IGBP", ""),
        "product": match["product"],
        "tower_GPP_monthly": tower_gpp,
        "matching_gpp_tif": match["gpp_tif"],
        "final_sif_companion_tif": str(output_path),
        "source_sif_files": compact_paths(source["file_path"] for source in match["sif_records"]),
        "source_sif_type": match["source_type"],
        "n_valid_sif_pixels_band2": int(valid.sum()),
        "n_valid_pixels_inside_footprint_probability_gt_0": int(in_footprint.sum()),
        "n_valid_pixels_inside_blind_buffer": int(in_blind.sum()),
        "SIF_mean_all_valid_band2": float(sif_array[valid].mean(dtype=np.float64)) if valid.any() else np.nan,
        "SIF_mean_inside_footprint_probability_gt_0": float(sif_array[in_footprint].mean(dtype=np.float64)) if in_footprint.any() else np.nan,
        "SIF_mean_inside_blind_buffer": float(sif_array[in_blind].mean(dtype=np.float64)) if in_blind.any() else np.nan,
        "SIF_mean_footprint_probability_weighted": weighted_mean,
        "status": status,
        "notes": notes,
    }
