"""
Preparation-only audit for EC-footprint-targeted SIF reconstruction.

This script creates modelling-ready inventories and EC footprint coverage
diagnostics without training models, reconstructing SIF, or modifying any
source/processed rasters. It intentionally reuses read-only pieces of the
footprint-aware SIF pipeline so the target observation masks are computed from
the same true satellite footprint/corner geometry logic.

Hard safety defaults:
- no GPP/source/processed TIFF modification
- no reconstruction / ML / kriging
- no all-NoData placeholder TIFF creation
- no KMZ usage
"""

from __future__ import annotations
# Original script filename (sanitized): Prepare_SIF_EC_Reconstruction_Model_Inventories.py; public name: 03_Prepare_SIF_EC_Coverage_Inventories.py
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
import subprocess
import sys
import textwrap
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as rasterio_transform

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - figures are best-effort
    plt = None

import SIF_TIFF_FOOTPRINT_AWARE_HELPERS as base
import Create_FINAL_CLEAN_SIF_Companion_TIFFs_FOOTPRINT_AWARE as footprint


# =============================================================================
# SETTINGS
# =============================================================================

PROJECT_ROOT = Path(
    PROJECT_ROOT
)
GPP_TIFF_ROOT = PROJECT_ROOT / "Monthly_EC_Footprint_TIFFs_Python"
SIF_ROOT = PROJECT_ROOT / "SIF Datasets"
GPP_REFERENCE_CSV = (
    PROJECT_ROOT
    / "finalized_metadata_recovery"
    / "final_metadata_recovered_ALL_SITES_monthly_WD_histogram_GPP_footprint_input_for_GEE.csv"
)
PREVIOUS_FEASIBILITY_AUDIT_ROOT = (
    PROJECT_ROOT / "SIF_EC_Footprint_Targeted_Reconstruction_Feasibility_Audit"
)

OUTPUT_ROOT = PROJECT_ROOT / "SIF_EC_Reconstruction_Model_Preparation"
FIGURES_ROOT = OUTPUT_ROOT / "figures"
GOSIF_SCALE_ROOT = OUTPUT_ROOT / "GOSIF_SCALE_CORRECTED"
CHECKPOINT_ROOT = OUTPUT_ROOT / "_checkpoints"

TARGET_PRODUCTS = ["OCO-2", "OCO-3", "TROPOSIF"]
SIF_DISCOVERY_PRODUCTS = ["GOSIF", "OCO-2", "OCO-3", "TROPOSIF"]
NODATA_VALUE = -9999.0
GOSIF_OFFICIAL_SCALE_FACTOR = 0.0001
GOSIF_SCALE_DECISION = "CONFIRMED: multiply by 0.0001"

CREATE_FIGURES = True
RESUME_FROM_CHECKPOINTS = True

GOSIF_TRACE_EXAMPLES = [
    ("AT-Zoe", "202010"),
    ("AU-Dry", "201009"),
    ("CH-Lae", "202407"),
    ("BE-Bra", "201908"),
    ("US-Wkg", "201306"),
]


# =============================================================================
# BASIC UTILITIES
# =============================================================================


def native(path: Path) -> str:
    return base.native_path(path)


def ensure_dirs() -> None:
    for path in (OUTPUT_ROOT, FIGURES_ROOT, GOSIF_SCALE_ROOT, CHECKPOINT_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(native(path), index=False)


def write_markdown(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_pdf_from_markdown(markdown_text: str, path: Path) -> None:
    """Small self-contained PDF writer with graceful fallback."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

        doc = SimpleDocTemplate(native(path), pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        for raw in markdown_text.splitlines():
            line = raw.strip()
            if not line:
                story.append(Spacer(1, 8))
                continue
            if line.startswith("# "):
                story.append(Paragraph(line[2:], styles["Title"]))
            elif line.startswith("## "):
                story.append(Paragraph(line[3:], styles["Heading2"]))
            elif line.startswith("### "):
                story.append(Paragraph(line[4:], styles["Heading3"]))
            else:
                safe = (
                    line.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                )
                story.append(Paragraph(safe, styles["BodyText"]))
        doc.build(story)
    except Exception:
        # Final fallback: create a plain-text .pdf path so the requested file
        # exists, but make the limitation explicit inside the file.
        path.write_text(
            "PDF generation fallback: reportlab unavailable.\n\n" + markdown_text,
            encoding="utf-8",
        )


def df_to_markdown_table(frame: pd.DataFrame, max_rows: int = 30) -> str:
    if frame is None or frame.empty:
        return "_No rows._"
    view = frame.head(max_rows).copy()
    try:
        return view.to_markdown(index=False)
    except Exception:
        return view.to_csv(index=False)


def safe_float(value: Any) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def finite_or_nan(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def percentile(series: pd.Series, q: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(np.percentile(values.to_numpy(dtype=float), q))


def normalize_site_safe(site_id: str) -> str:
    return base.site_to_safe(str(site_id))


def yyyy_mm_midmonth_doy(yyyymm: str) -> int:
    y = int(str(yyyymm)[:4])
    m = int(str(yyyymm)[4:6])
    return date(y, m, 15).timetuple().tm_yday


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["year"] = out["YYYYMM"].astype(str).str.slice(0, 4).astype(int)
    out["month"] = out["YYYYMM"].astype(str).str.slice(4, 6).astype(int)
    out["DOY"] = [yyyy_mm_midmonth_doy(x) for x in out["YYYYMM"].astype(str)]
    out["sin_DOY"] = np.sin(2.0 * np.pi * out["DOY"].astype(float) / 365.25)
    out["cos_DOY"] = np.cos(2.0 * np.pi * out["DOY"].astype(float) / 365.25)
    return out


def configure_read_only_modules() -> None:
    """Force imported pipeline helpers into read-only/audit mode."""
    base.PROJECT_ROOT = PROJECT_ROOT
    base.GPP_TIFF_ROOT = GPP_TIFF_ROOT
    base.GPP_REFERENCE_CSV = GPP_REFERENCE_CSV
    base.SIF_ROOT = SIF_ROOT
    base.OUTPUT_ROOT = OUTPUT_ROOT
    base.NODATA_VALUE = NODATA_VALUE
    base.DRY_RUN = True
    base.WRITE_OUTPUTS = False
    base.OVERWRITE_EXISTING = False
    base.SELECTED_SITES = None
    base.SELECTED_YYYYMM = None
    base.PRODUCTS_TO_PROCESS = SIF_DISCOVERY_PRODUCTS
    base.RESAMPLING_FOR_GRIDDED_SIF = "nearest"

    footprint.PROJECT_ROOT = PROJECT_ROOT
    footprint.GPP_TIFF_ROOT = GPP_TIFF_ROOT
    footprint.GPP_REFERENCE_CSV = GPP_REFERENCE_CSV
    footprint.SIF_ROOT = SIF_ROOT
    footprint.OUTPUT_ROOT = OUTPUT_ROOT
    footprint.NODATA_VALUE = NODATA_VALUE
    footprint.DRY_RUN = True
    footprint.WRITE_OUTPUTS = False
    footprint.OVERWRITE_EXISTING = False
    footprint.PRODUCTS_TO_PROCESS = SIF_DISCOVERY_PRODUCTS
    footprint.SELECTED_SITES = None
    footprint.SELECTED_YYYYMM = None
    footprint.GOSIF_GRIDDED_RESAMPLING = "nearest"
    footprint.SWATH_RASTERIZATION_METHOD = "footprint_polygon_to_GPP_grid_cells"


# =============================================================================
# GOSIF SCALE AUDIT
# =============================================================================


def run_rg(args: list[str]) -> str:
    try:
        result = subprocess.run(
            args,
            cwd=native(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        text = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
        return text.strip()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def collect_gosif_documentation_evidence() -> pd.DataFrame:
    """Search local project docs/code and record official scale evidence.

    The authoritative scale text in this local run comes from the downloaded
    GOSIF v2 README cached in the system temp folder during this audit. This
    file is not placed in source data or existing output folders.
    """
    rows: list[dict[str, Any]] = []

    local_doc_files = [
        path
        for path in SIF_ROOT.joinpath("GOSIF").rglob("*")
        if path.is_file()
        and path.suffix.lower() not in {".tif", ".tiff"}
        and re.search(r"readme|meta|doc|citation|scale|policy|txt|pdf", path.name, re.I)
    ]
    rows.append(
        {
            "evidence_type": "local_gosif_sidecar_docs",
            "source": str(SIF_ROOT / "GOSIF"),
            "finding": f"{len(local_doc_files)} local sidecar documentation files found",
            "supports_scale_decision": "NO_LOCAL_DOCS" if not local_doc_files else "REVIEW",
            "excerpt": "; ".join(str(p) for p in local_doc_files[:10]),
        }
    )

    rg_text = run_rg(
        [
            "rg",
            "-n",
            "-i",
            r"GOSIF|0\.0001|scale.?factor|scale factor|fill values|32767|32766",
            native(PROJECT_ROOT / "Codes" / "pythonProject"),
            native(SIF_ROOT / "GOSIF"),
            "--glob",
            "!**/*.tif",
            "--glob",
            "!**/*.tiff",
            "--glob",
            "!**/__pycache__/**",
        ]
    )
    rows.append(
        {
            "evidence_type": "local_code_and_doc_search",
            "source": "rg over code folder and SIF Datasets/GOSIF excluding TIFFs",
            "finding": "Search completed",
            "supports_scale_decision": "PROJECT_CODE_PREVIOUSLY_FLAGGED_SCALE_BLOCKER",
            "excerpt": rg_text[:4000],
        }
    )

    pdf_path = Path(os.environ.get("TEMP", str(OUTPUT_ROOT))) / "Fair_Data_Use_Policy_and_Readme_GOSIF_v2.pdf"
    if pdf_path.exists():
        try:
            import pypdf

            reader = pypdf.PdfReader(native(pdf_path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            idx = text.lower().find("scale factor")
            excerpt = text[max(0, idx - 250) : idx + 700] if idx >= 0 else text[:1000]
            supports = (
                "YES_OFFICIAL_README_SCALE_0P0001"
                if "0.0001" in excerpt or "0.0001" in text
                else "OFFICIAL_README_NO_SCALE_TEXT_FOUND"
            )
            rows.append(
                {
                    "evidence_type": "official_gosif_v2_readme_pdf_temp_cache",
                    "source": str(pdf_path),
                    "finding": "Official GOSIF v2 README PDF extracted successfully",
                    "supports_scale_decision": supports,
                    "excerpt": " ".join(excerpt.split())[:4000],
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "evidence_type": "official_gosif_v2_readme_pdf_temp_cache",
                    "source": str(pdf_path),
                    "finding": f"PDF extraction failed: {type(exc).__name__}: {exc}",
                    "supports_scale_decision": "UNKNOWN",
                    "excerpt": "",
                }
            )
    else:
        rows.append(
            {
                "evidence_type": "official_gosif_v2_readme_pdf_temp_cache",
                "source": str(pdf_path),
                "finding": "Official README temp copy not found in this run",
                "supports_scale_decision": "UNKNOWN",
                "excerpt": "",
            }
        )

    return pd.DataFrame(rows)


def find_template(templates: list[dict[str, Any]], site_id: str, yyyymm: str) -> dict[str, Any] | None:
    for template in templates:
        if template["SITE_ID"] == site_id and template["YYYYMM"] == yyyymm:
            return template
    return None


def find_gosif_source(usable_sources: list[dict[str, Any]], yyyymm: str) -> dict[str, Any] | None:
    for record in usable_sources:
        if record["product"] == "GOSIF" and str(record.get("detected_YYYYMM")) == yyyymm:
            return record
    return None


def source_raster_metadata(path: Path) -> dict[str, Any]:
    with rasterio.open(native(path)) as dataset:
        band = dataset.read(1)
        valid = np.isfinite(band)
        if dataset.nodata is not None:
            valid &= band != dataset.nodata
        valid &= ~np.isin(band, list(base.PRODUCT_INVALID_VALUES.get("GOSIF", set())))
        vals = band[valid]
        return {
            "source_dtype": str(dataset.dtypes[0]),
            "source_nodata": dataset.nodata,
            "source_scales": str(dataset.scales),
            "source_offsets": str(dataset.offsets),
            "source_tags": str(dataset.tags())[:1000],
            "raw_source_min": float(vals.min()) if vals.size else np.nan,
            "raw_source_max": float(vals.max()) if vals.size else np.nan,
            "raw_source_mean": float(vals.mean()) if vals.size else np.nan,
        }


def trace_gosif_scaling(
    templates: list[dict[str, Any]],
    usable_sources: list[dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for site_id, yyyymm in GOSIF_TRACE_EXAMPLES:
        row: dict[str, Any] = {"SITE_ID": site_id, "YYYYMM": yyyymm}
        template = find_template(templates, site_id, yyyymm)
        source = find_gosif_source(usable_sources, yyyymm)
        row["gpp_template_found"] = bool(template)
        row["source_gosif_found"] = bool(source)
        if source:
            source_path = Path(source["file_path"])
            row["source_gosif_file"] = str(source_path)
            try:
                row.update(source_raster_metadata(source_path))
            except Exception as exc:
                row["source_metadata_error"] = f"{type(exc).__name__}: {exc}"
        if template and source:
            try:
                raw_aligned = base.align_one_gridded_source(source, template)
                valid_raw = base.valid_data_mask(raw_aligned, NODATA_VALUE)
                ec_arrays, _, _ = base.read_gpp_arrays(Path(template["gpp_tif"]))
                footprint = ec_arrays[0]
                blind = ec_arrays[2]
                ec_mask = np.isfinite(footprint) & (footprint > 0)
                blind_mask = np.isfinite(blind) & (blind == 1)
                row["value_entering_processing_raw_mean_all_valid_band2"] = (
                    float(raw_aligned[valid_raw].mean()) if valid_raw.any() else np.nan
                )
                row["value_entering_processing_raw_mean_inside_footprint"] = (
                    float(raw_aligned[valid_raw & ec_mask].mean())
                    if np.any(valid_raw & ec_mask)
                    else np.nan
                )
                row["value_entering_processing_raw_mean_inside_blind_buffer"] = (
                    float(raw_aligned[valid_raw & blind_mask].mean())
                    if np.any(valid_raw & blind_mask)
                    else np.nan
                )
                row["value_if_scaled0p0001_mean_inside_footprint"] = (
                    row["value_entering_processing_raw_mean_inside_footprint"]
                    * GOSIF_OFFICIAL_SCALE_FACTOR
                    if math.isfinite(row["value_entering_processing_raw_mean_inside_footprint"])
                    else np.nan
                )
            except Exception as exc:
                row["processing_trace_error"] = f"{type(exc).__name__}: {exc}"
        # Trace currently-written outputs when they exist. These are read-only.
        old_outputs = [
            PROJECT_ROOT / "SIF_TIFF_output" / "GOSIF" / f"{normalize_site_safe(site_id)}_{yyyymm}_GOSIF_SIF.tif",
            PROJECT_ROOT
            / "FINAL_CLEAN_SIF_Companion_TIFFs_FOOTPRINT_AWARE"
            / "SIF_Companion_TIFFs"
            / "GOSIF"
            / normalize_site_safe(site_id)
            / f"{normalize_site_safe(site_id)}_GOSIF_SIF_companion_matching_GPP_footprint_aware_{yyyymm}.tif",
            PROJECT_ROOT
            / "FINAL_CLEAN_SIF_Companion_TIFFs_Matching_GPP"
            / "SIF_Companion_TIFFs"
            / "GOSIF"
            / normalize_site_safe(site_id)
            / f"{normalize_site_safe(site_id)}_GOSIF_SIF_companion_matching_GPP_{yyyymm}.tif",
        ]
        for output in old_outputs:
            if output.exists():
                row["current_output_tif_read"] = str(output)
                try:
                    with rasterio.open(native(output)) as dataset:
                        band2 = dataset.read(2).astype(np.float64)
                        valid2 = base.valid_data_mask(band2, dataset.nodata or NODATA_VALUE)
                        band7_mean = np.nan
                        if dataset.count >= 7:
                            band7 = dataset.read(7).astype(np.float64)
                            valid7 = base.valid_data_mask(band7, dataset.nodata or NODATA_VALUE)
                            band7_mean = float(band7[valid7].mean()) if valid7.any() else np.nan
                        row["value_written_current_output_band2_mean"] = (
                            float(band2[valid2].mean()) if valid2.any() else np.nan
                        )
                        row["value_written_current_output_band7_mean"] = band7_mean
                        row["current_output_scales"] = str(dataset.scales)
                        row["current_output_offsets"] = str(dataset.offsets)
                        row["current_output_nodata"] = dataset.nodata
                    row["current_output_band2_mean_if_scaled0p0001"] = (
                        row["value_written_current_output_band2_mean"] * GOSIF_OFFICIAL_SCALE_FACTOR
                        if math.isfinite(row.get("value_written_current_output_band2_mean", np.nan))
                        else np.nan
                    )
                except Exception as exc:
                    row["current_output_read_error"] = f"{type(exc).__name__}: {exc}"
                break
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# COVERAGE COMPUTATION
# =============================================================================


def load_reference_metadata() -> pd.DataFrame:
    frame = pd.read_csv(native(GPP_REFERENCE_CSV), low_memory=False)
    if "YYYYMM" not in frame.columns and {"year", "month"}.issubset(frame.columns):
        frame["YYYYMM"] = [
            f"{int(y):04d}{int(m):02d}" if pd.notna(y) and pd.notna(m) else ""
            for y, m in zip(frame["year"], frame["month"])
        ]
    frame["YYYYMM"] = frame["YYYYMM"].map(base.normalize_yyyymm)
    frame = frame[pd.notna(frame["SITE_ID"]) & pd.notna(frame["YYYYMM"])].copy()
    if "valid_for_GEE_export" in frame.columns:
        valid = frame["valid_for_GEE_export"].astype(str).str.lower().isin(["1", "true", "yes"])
        frame = frame[valid].copy()
    keep = [c for c in ["SITE_ID", "YYYYMM", "lat", "lon", "IGBP", "year", "month", "tower_GPP_monthly"] if c in frame.columns]
    meta = frame[keep].drop_duplicates(["SITE_ID", "YYYYMM"]).copy()
    meta["SITE_SAFE"] = meta["SITE_ID"].map(normalize_site_safe)
    return meta


def metadata_lookup(meta: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in meta.iterrows():
        lookup[(str(row["SITE_ID"]), str(row["YYYYMM"]))] = row.to_dict()
    return lookup


def template_lookup(templates: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(t["SITE_ID"]), str(t["YYYYMM"])): t for t in templates}


def read_ec_masks(template: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays, _, _ = base.read_gpp_arrays(Path(template["gpp_tif"]))
    footprint_probability = arrays[0].astype(np.float64)
    blind = arrays[2].astype(np.float64)
    ec_mask = np.isfinite(footprint_probability) & (footprint_probability > 0)
    blind_mask = np.isfinite(blind) & (blind == 1)
    return footprint_probability, ec_mask, blind_mask


def coverage_record_from_sif_array(
    match: dict[str, Any],
    sif_array: np.ndarray,
    metrics: dict[str, Any],
    meta_lookup: dict[tuple[str, str], dict[str, Any]],
    source_type: str,
) -> dict[str, Any]:
    template = match["template"]
    footprint_probability, ec_mask, blind_mask = read_ec_masks(template)
    valid_sif = base.valid_data_mask(sif_array, NODATA_VALUE)
    observed_ec = valid_sif & ec_mask
    n_ec = int(ec_mask.sum())
    n_observed_ec = int(observed_ec.sum())
    prob_total = float(footprint_probability[ec_mask].sum()) if n_ec else np.nan
    prob_obs = float(footprint_probability[observed_ec].sum()) if n_observed_ec else 0.0
    cov_prob = prob_obs / prob_total if math.isfinite(prob_total) and prob_total > 0 else np.nan
    cov_pixel = n_observed_ec / n_ec if n_ec > 0 else np.nan
    missing_prob = 1.0 - cov_prob if math.isfinite(cov_prob) else np.nan
    meta = meta_lookup.get((match["SITE_ID"], match["YYYYMM"]), {})
    observed_values = sif_array[observed_ec].astype(np.float64)
    weights = footprint_probability[observed_ec].astype(np.float64)
    weighted = (
        float(np.sum(observed_values * weights) / np.sum(weights))
        if observed_values.size and np.sum(weights) > 0
        else np.nan
    )
    blind_values = sif_array[valid_sif & blind_mask].astype(np.float64)
    row = {
        "SITE_ID": match["SITE_ID"],
        "SITE_SAFE": match["SITE_SAFE"],
        "YYYYMM": match["YYYYMM"],
        "PRODUCT": match["product"],
        "lat": safe_float(meta.get("lat")),
        "lon": safe_float(meta.get("lon")),
        "IGBP": str(meta.get("IGBP", "")),
        "source_sif_type": source_type,
        "source_sif_files": base.compact_paths(Path(r["file_path"]) for r in match.get("sif_records", [])),
        "matching_gpp_tif": match["gpp_tif"],
        "n_target_source_footprints": int(metrics.get("n_sif_footprint_polygons", 0) or 0),
        "n_target_source_observations": int(metrics.get("n_source_sif_observations", 0) or 0),
        "n_candidate_sif_observations_for_site_grid": int(metrics.get("n_candidate_sif_observations_for_site_grid", 0) or 0),
        "n_candidate_sif_observations_skipped_bad_geometry": int(metrics.get("n_candidate_sif_observations_skipped_bad_geometry", 0) or 0),
        "n_candidate_sif_observations_no_spatial_intersection_with_grid": int(metrics.get("n_candidate_sif_observations_no_spatial_intersection_with_grid", 0) or 0),
        "n_candidate_sif_observations_unexplained": int(metrics.get("n_candidate_sif_observations_unexplained", 0) or 0),
        "candidate_accounting_complete": bool(metrics.get("candidate_accounting_complete", True)),
        "swath_geometry_fields_used": str(metrics.get("swath_geometry_fields_used", "")),
        "n_EC_pixels": n_ec,
        "n_EC_pixels_target_observed": n_observed_ec,
        "coverage_pixel_fraction": cov_pixel,
        "coverage_probability_fraction": cov_prob,
        "missing_probability_fraction": missing_prob,
        "target_observed_EC_weighted_SIF": weighted,
        "target_observed_mean_SIF": float(observed_values.mean()) if observed_values.size else np.nan,
        "target_observed_sd": float(observed_values.std(ddof=1)) if observed_values.size > 1 else np.nan,
        "target_observed_min": float(observed_values.min()) if observed_values.size else np.nan,
        "target_observed_max": float(observed_values.max()) if observed_values.size else np.nan,
        "n_valid_pixels_inside_blind_buffer": int((valid_sif & blind_mask).sum()),
        "target_observed_blind_buffer_mean_SIF": float(blind_values.mean()) if blind_values.size else np.nan,
        "probability_sum_full_EC": prob_total,
        "probability_sum_target_observed": prob_obs,
        "status": "OK_OBSERVED_EC_SUPPORT" if n_observed_ec > 0 else "NO_TARGET_OBSERVATION_INSIDE_EC_FOOTPRINT",
        "notes": (
            "Observed EC-support SIF only; not complete EC SIF unless coverage_probability_fraction is ~1."
        ),
    }
    return row


def empty_coverage_row(
    match: dict[str, Any],
    status: str,
    meta_lookup: dict[tuple[str, str], dict[str, Any]],
    notes: str,
) -> dict[str, Any]:
    meta = meta_lookup.get((match["SITE_ID"], match["YYYYMM"]), {})
    return {
        "SITE_ID": match["SITE_ID"],
        "SITE_SAFE": match["SITE_SAFE"],
        "YYYYMM": match["YYYYMM"],
        "PRODUCT": match["product"],
        "lat": safe_float(meta.get("lat")),
        "lon": safe_float(meta.get("lon")),
        "IGBP": str(meta.get("IGBP", "")),
        "source_sif_type": match.get("source_type", ""),
        "source_sif_files": base.compact_paths(Path(r["file_path"]) for r in match.get("sif_records", [])),
        "matching_gpp_tif": match["gpp_tif"],
        "n_target_source_footprints": 0,
        "n_target_source_observations": 0,
        "n_candidate_sif_observations_for_site_grid": 0,
        "n_candidate_sif_observations_skipped_bad_geometry": 0,
        "n_candidate_sif_observations_no_spatial_intersection_with_grid": 0,
        "n_candidate_sif_observations_unexplained": 0,
        "candidate_accounting_complete": True,
        "swath_geometry_fields_used": "",
        "n_EC_pixels": np.nan,
        "n_EC_pixels_target_observed": 0,
        "coverage_pixel_fraction": 0.0,
        "coverage_probability_fraction": 0.0,
        "missing_probability_fraction": 1.0,
        "target_observed_EC_weighted_SIF": np.nan,
        "target_observed_mean_SIF": np.nan,
        "target_observed_sd": np.nan,
        "target_observed_min": np.nan,
        "target_observed_max": np.nan,
        "n_valid_pixels_inside_blind_buffer": 0,
        "target_observed_blind_buffer_mean_SIF": np.nan,
        "probability_sum_full_EC": np.nan,
        "probability_sum_target_observed": 0.0,
        "status": status,
        "notes": notes,
    }


def append_rows_checkpoint(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    pd.DataFrame(rows).to_csv(native(path), mode="a", header=not exists, index=False)


def load_done_keys(path: Path) -> set[tuple[str, str, str]]:
    if not (RESUME_FROM_CHECKPOINTS and path.exists()):
        return set()
    try:
        frame = pd.read_csv(native(path), usecols=["PRODUCT", "SITE_ID", "YYYYMM"], dtype=str)
    except Exception:
        return set()
    return set(zip(frame["PRODUCT"].astype(str), frame["SITE_ID"].astype(str), frame["YYYYMM"].astype(str)))


def compute_target_product_coverage(
    product: str,
    matches: list[dict[str, Any]],
    meta_lookup_dict: dict[tuple[str, str], dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute coverage rows for one target product without writing TIFFs."""
    product_matches = [m for m in matches if m["product"] == product]
    checkpoint = CHECKPOINT_ROOT / f"coverage_{product.replace('-', '')}.csv"
    done = load_done_keys(checkpoint)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for match in product_matches:
        key = (match["product"], match["SITE_ID"], match["YYYYMM"])
        if key not in done:
            groups[str(match["YYYYMM"])].append(match)

    missing_geometry_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    n_groups = len(groups)
    print(f"[{timestamp()}] {product}: {len(product_matches)} matched product-site-months; {len(done)} already checkpointed; {n_groups} month groups to process.")

    for group_index, (yyyymm, month_matches) in enumerate(sorted(groups.items()), start=1):
        print(f"[{timestamp()}] {product} {yyyymm}: group {group_index}/{n_groups}, site-months={len(month_matches)}")
        contexts = [footprint.build_footprint_context(match) for match in month_matches]
        source_statuses: list[str] = []
        for source in month_matches[0].get("sif_records", []):
            try:
                _, status = footprint.append_footprint_polygons_from_source(
                    source, contexts, missing_geometry_rows
                )
                source_statuses.append(status)
            except Exception as exc:
                source_statuses.append(f"FAILED_SOURCE_READ: {type(exc).__name__}: {exc}")
                failure_rows.append(
                    {
                        "PRODUCT": product,
                        "YYYYMM": yyyymm,
                        "source_file": str(source.get("file_path", "")),
                        "stage": "SOURCE_READ_OR_RASTERIZE",
                        "error_message": f"{type(exc).__name__}: {exc}",
                    }
                )

        all_lack_geometry = bool(source_statuses) and all(
            status == "PRODUCT_LACKS_FOOTPRINT_GEOMETRY" for status in source_statuses
        )
        rows: list[dict[str, Any]] = []
        for context in contexts:
            match = context["match"]
            try:
                if all_lack_geometry:
                    rows.append(
                        empty_coverage_row(
                            match,
                            "PRODUCT_LACKS_FOOTPRINT_GEOMETRY",
                            meta_lookup_dict,
                            "Readable target files exist, but no true footprint/corner geometry was detected; no fabricated geometry was used.",
                        )
                    )
                    continue
                sif_array, metrics = footprint.context_sif_array_and_metrics(context)
                row = coverage_record_from_sif_array(
                    match, sif_array, metrics, meta_lookup_dict, source_type="point_or_swath"
                )
                if source_statuses:
                    row["source_statuses"] = " | ".join(sorted(set(source_statuses)))
                rows.append(row)
            except Exception as exc:
                failure_rows.append(
                    {
                        "PRODUCT": product,
                        "SITE_ID": match.get("SITE_ID", ""),
                        "YYYYMM": match.get("YYYYMM", ""),
                        "gpp_tif": match.get("gpp_tif", ""),
                        "stage": "COVERAGE_SUMMARY",
                        "error_message": f"{type(exc).__name__}: {exc}",
                    }
                )
                rows.append(
                    empty_coverage_row(
                        match,
                        "FAILED_COVERAGE_COMPUTE",
                        meta_lookup_dict,
                        f"Coverage computation failed: {type(exc).__name__}: {exc}",
                    )
                )
        append_rows_checkpoint(checkpoint, rows)

    # Merge checkpoint file as the authoritative product output.
    if checkpoint.exists():
        coverage = pd.read_csv(native(checkpoint), dtype={"YYYYMM": str})
    else:
        coverage = pd.DataFrame()
    return coverage, pd.DataFrame(missing_geometry_rows), pd.DataFrame(failure_rows)


def build_no_sif_skips(
    templates: list[dict[str, Any]],
    source_by_product_month: dict[tuple[str, str], list[dict[str, Any]]],
    meta_lookup_dict: dict[tuple[str, str], dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for product in TARGET_PRODUCTS:
        for template in templates:
            if (product, template["YYYYMM"]) in source_by_product_month:
                continue
            match = {
                "product": product,
                "SITE_ID": template["SITE_ID"],
                "SITE_SAFE": template["SITE_SAFE"],
                "YYYYMM": template["YYYYMM"],
                "gpp_tif": template["gpp_tif"],
                "source_type": "point_or_swath",
                "sif_records": [],
            }
            rows.append(
                empty_coverage_row(
                    match,
                    "NO_LOCAL_SIF_FOR_MONTH",
                    meta_lookup_dict,
                    "No readable local source SIF file exists for this product month.",
                )
            )
    return pd.DataFrame(rows)


# =============================================================================
# GOSIF EC-WEIGHTED PREDICTOR
# =============================================================================


def gosif_native_cell_count_for_ec(source: dict[str, Any], template: dict[str, Any]) -> int:
    """Count unique native GOSIF raster cells sampled by GPP EC pixels.

    This is a diagnostic native-support count. The EC-weighted value is computed
    after nearest-neighbour support assignment to the GPP grid, then weighted by
    footprint probability. We report this as support-aware aggregation, not as
    independent fine-pixel GOSIF observations.
    """
    path = Path(source["file_path"])
    with rasterio.open(native(path)) as dataset, rasterio.open(native(Path(template["gpp_tif"]))) as gpp:
        footprint_probability = gpp.read(1)
        ec_mask = np.isfinite(footprint_probability) & (footprint_probability > 0)
        rows, cols = np.where(ec_mask)
        if rows.size == 0:
            return 0
        xs, ys = rasterio.transform.xy(gpp.transform, rows, cols, offset="center")
        src_x, src_y = rasterio_transform(gpp.crs, dataset.crs, list(xs), list(ys))
        native_rc = [dataset.index(x, y) for x, y in zip(src_x, src_y)]
        valid_native = {
            (int(r), int(c))
            for r, c in native_rc
            if 0 <= int(r) < dataset.height and 0 <= int(c) < dataset.width
        }
        return len(valid_native)


def compute_gosif_predictors(
    needed_site_months: pd.DataFrame,
    usable_sources: list[dict[str, Any]],
    tmpl_lookup: dict[tuple[str, str], dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    source_by_month = {str(r["detected_YYYYMM"]): r for r in usable_sources if r["product"] == "GOSIF"}
    for _, rec in needed_site_months[["SITE_ID", "YYYYMM"]].drop_duplicates().iterrows():
        site_id = str(rec["SITE_ID"])
        yyyymm = str(rec["YYYYMM"])
        row: dict[str, Any] = {"SITE_ID": site_id, "YYYYMM": yyyymm}
        template = tmpl_lookup.get((site_id, yyyymm))
        source = source_by_month.get(yyyymm)
        row["GOSIF_source_found"] = bool(source)
        row["GOSIF_scale_status"] = GOSIF_SCALE_DECISION
        if not template or not source:
            row["GOSIF_status"] = "MISSING_GPP_OR_GOSIF_SOURCE"
            rows.append(row)
            continue
        try:
            raw = base.align_one_gridded_source(source, template).astype(np.float64)
            scaled = raw.copy()
            valid_raw = base.valid_data_mask(raw, NODATA_VALUE)
            scaled[valid_raw] = raw[valid_raw] * GOSIF_OFFICIAL_SCALE_FACTOR
            scaled[~valid_raw] = NODATA_VALUE
            footprint_probability, ec_mask, _ = read_ec_masks(template)
            valid = base.valid_data_mask(scaled, NODATA_VALUE) & ec_mask
            weights = footprint_probability[valid]
            values = scaled[valid]
            row["GOSIF_EC_weighted"] = (
                float(np.sum(values * weights) / np.sum(weights)) if values.size and np.sum(weights) > 0 else np.nan
            )
            row["n_native_GOSIF_cells_intersecting_EC"] = gosif_native_cell_count_for_ec(source, template)
            row["GOSIF_native_min"] = float(values.min()) if values.size else np.nan
            row["GOSIF_native_max"] = float(values.max()) if values.size else np.nan
            row["GOSIF_native_sd"] = float(values.std(ddof=1)) if values.size > 1 else np.nan
            row["GOSIF_units"] = "W m-2 um-1 sr-1"
            row["GOSIF_status"] = "OK_SCALE_CORRECTED_EC_WEIGHTED"
            row["GOSIF_source_file"] = str(source["file_path"])
            row["GOSIF_support_note"] = (
                "Nearest-neighbour native-cell support assignment to the GPP grid followed by EC probability weighting; "
                "fine GPP cells are not treated as independent GOSIF observations."
            )
        except Exception as exc:
            row["GOSIF_status"] = "FAILED_GOSIF_EC_WEIGHTING"
            row["GOSIF_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# SUMMARIES, INVENTORIES, PATHOLOGIES
# =============================================================================


def coverage_class(value: Any) -> str:
    x = safe_float(value)
    if not math.isfinite(x):
        return "NaN"
    tol = 1e-9
    if abs(x - 1.0) <= tol:
        return "1.00"
    if x <= 0.10:
        return "0-0.10"
    if x <= 0.25:
        return ">0.10-0.25"
    if x <= 0.50:
        return ">0.25-0.50"
    if x <= 0.75:
        return ">0.50-0.75"
    if x <= 0.90:
        return ">0.75-0.90"
    if x < 1.0:
        return ">0.90-<1.00"
    return ">1.00"


def summarize_by_group(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        row.update(
            {
                "N_site_months": int(len(group)),
                "N_sites": int(group["SITE_ID"].nunique()),
                "N_years": int(group["year"].nunique()) if "year" in group.columns else np.nan,
                "mean_coverage_probability_fraction": float(pd.to_numeric(group["coverage_probability_fraction"], errors="coerce").mean()),
                "median_coverage_probability_fraction": float(pd.to_numeric(group["coverage_probability_fraction"], errors="coerce").median()),
                "sd_coverage_probability_fraction": float(pd.to_numeric(group["coverage_probability_fraction"], errors="coerce").std(ddof=1)),
                "p05_coverage_probability_fraction": percentile(group["coverage_probability_fraction"], 5),
                "p25_coverage_probability_fraction": percentile(group["coverage_probability_fraction"], 25),
                "p50_coverage_probability_fraction": percentile(group["coverage_probability_fraction"], 50),
                "p75_coverage_probability_fraction": percentile(group["coverage_probability_fraction"], 75),
                "p95_coverage_probability_fraction": percentile(group["coverage_probability_fraction"], 95),
                "mean_coverage_pixel_fraction": float(pd.to_numeric(group["coverage_pixel_fraction"], errors="coerce").mean()),
                "median_coverage_pixel_fraction": float(pd.to_numeric(group["coverage_pixel_fraction"], errors="coerce").median()),
                "sd_coverage_pixel_fraction": float(pd.to_numeric(group["coverage_pixel_fraction"], errors="coerce").std(ddof=1)),
                "p05_coverage_pixel_fraction": percentile(group["coverage_pixel_fraction"], 5),
                "p25_coverage_pixel_fraction": percentile(group["coverage_pixel_fraction"], 25),
                "p50_coverage_pixel_fraction": percentile(group["coverage_pixel_fraction"], 50),
                "p75_coverage_pixel_fraction": percentile(group["coverage_pixel_fraction"], 75),
                "p95_coverage_pixel_fraction": percentile(group["coverage_pixel_fraction"], 95),
            }
        )
        for threshold in [0.25, 0.50, 0.75, 0.90, 0.95, 1.00]:
            vals = pd.to_numeric(group["coverage_probability_fraction"], errors="coerce")
            row[f"pct_coverage_probability_lt_{str(threshold).replace('.', 'p')}"] = (
                float((vals < threshold).mean() * 100.0) if len(vals) else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def coverage_class_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    temp = frame.copy()
    temp["coverage_probability_class"] = temp["coverage_probability_fraction"].map(coverage_class)
    counts = (
        temp.groupby(["PRODUCT", "coverage_probability_class"], dropna=False)
        .size()
        .reset_index(name="N_site_months")
    )
    totals = counts.groupby("PRODUCT")["N_site_months"].transform("sum")
    counts["percentage"] = counts["N_site_months"] / totals * 100.0
    order = ["0-0.10", ">0.10-0.25", ">0.25-0.50", ">0.50-0.75", ">0.75-0.90", ">0.90-<1.00", "1.00", ">1.00", "NaN"]
    counts["class_order"] = counts["coverage_probability_class"].map({k: i for i, k in enumerate(order)})
    return counts.sort_values(["PRODUCT", "class_order"]).drop(columns=["class_order"])


def identify_pathologies(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        cov = safe_float(row.get("coverage_probability_fraction"))
        pix = safe_float(row.get("coverage_pixel_fraction"))
        nobs = int(row.get("n_target_source_observations", 0) or 0)
        nfoot = int(row.get("n_target_source_footprints", 0) or 0)
        problems: list[str] = []
        if not math.isfinite(cov):
            problems.append("NaN coverage_probability_fraction")
        elif cov < -1e-9:
            problems.append("negative coverage_probability_fraction")
        elif cov > 1.0 + 1e-6:
            problems.append("coverage_probability_fraction > 1")
        elif abs(cov) <= 1e-12:
            problems.append("coverage_probability_fraction = 0")
        if math.isfinite(pix) and (pix < -1e-9 or pix > 1.0 + 1e-6):
            problems.append("coverage_pixel_fraction outside [0,1]")
        if str(row.get("status", "")) == "NO_TARGET_OBSERVATION_INSIDE_EC_FOOTPRINT":
            problems.append("all-NoData target SIF inside EC footprint")
        if math.isfinite(cov) and cov >= 0.999999 and nfoot <= 1:
            problems.append("full EC coverage but only one source footprint")
        if nobs <= 1 and str(row.get("status", "")).startswith("OK"):
            problems.append("very low observation count")
        if int(row.get("n_candidate_sif_observations_unexplained", 0) or 0) > 0:
            problems.append("unexplained candidate accounting gap")
        if problems:
            out = row.to_dict()
            out["pathology_flags"] = "; ".join(problems)
            rows.append(out)
    return pd.DataFrame(rows)


def artificial_gap_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    vals = pd.to_numeric(out["coverage_probability_fraction"], errors="coerce")
    out["candidate_ge_0p90"] = vals >= 0.90
    out["candidate_ge_0p95"] = vals >= 0.95
    out["candidate_eq_1p00"] = np.isclose(vals, 1.0, atol=1e-9, rtol=0)
    return out[
        out["candidate_ge_0p90"] | out["candidate_ge_0p95"] | out["candidate_eq_1p00"]
    ].copy()


def build_model_inventory(
    coverage: pd.DataFrame,
    product: str,
    gosif_predictors: pd.DataFrame | None = None,
    include_gosif: bool = False,
    oco2_diagnostic: bool = False,
) -> pd.DataFrame:
    cols = [
        "SITE_ID",
        "YYYYMM",
        "lat",
        "lon",
        "IGBP",
        "coverage_probability_fraction",
        "coverage_pixel_fraction",
        "missing_probability_fraction",
        "n_target_source_observations",
        "target_observed_EC_weighted_SIF",
    ]
    if coverage.empty or "PRODUCT" not in coverage.columns:
        return pd.DataFrame(columns=cols)
    sub = coverage[coverage["PRODUCT"] == product].copy()
    if sub.empty:
        return pd.DataFrame(columns=cols)
    out = sub[cols].copy()
    out = add_time_features(out)
    ordered = [
        "SITE_ID",
        "YYYYMM",
        "lat",
        "lon",
        "IGBP",
        "year",
        "month",
        "DOY",
        "sin_DOY",
        "cos_DOY",
        "coverage_probability_fraction",
        "coverage_pixel_fraction",
        "missing_probability_fraction",
        "n_target_source_observations",
        "target_observed_EC_weighted_SIF",
    ]
    out = out[ordered]
    if include_gosif and gosif_predictors is not None:
        out = out.merge(gosif_predictors, on=["SITE_ID", "YYYYMM"], how="left")
    if oco2_diagnostic:
        out["GOSIF_diagnostic_status"] = "NOT FOR PRIMARY OCO-2 RECONSTRUCTION"
    return out


# =============================================================================
# FIGURES AND REPORT
# =============================================================================


def make_figures(coverage: pd.DataFrame) -> list[Path]:
    paths: list[Path] = []
    if not CREATE_FIGURES or plt is None or coverage.empty:
        return paths
    plot_frame = coverage.copy()
    plot_frame["coverage_probability_fraction"] = pd.to_numeric(
        plot_frame["coverage_probability_fraction"], errors="coerce"
    )
    plot_frame["coverage_pixel_fraction"] = pd.to_numeric(
        plot_frame["coverage_pixel_fraction"], errors="coerce"
    )

    # Figure 1: histograms
    fig, axes = plt.subplots(1, len(TARGET_PRODUCTS), figsize=(14, 4), sharey=True)
    if len(TARGET_PRODUCTS) == 1:
        axes = [axes]
    for ax, product in zip(axes, TARGET_PRODUCTS):
        vals = plot_frame.loc[plot_frame["PRODUCT"] == product, "coverage_probability_fraction"].dropna()
        ax.hist(vals, bins=np.linspace(0, 1, 21), color="#2a9d8f", edgecolor="white")
        ax.set_title(product)
        ax.set_xlabel("Probability-weighted EC coverage")
        ax.set_xlim(0, 1)
    axes[0].set_ylabel("Site-month count")
    fig.suptitle("Figure 1. Probability-weighted EC footprint coverage")
    path = FIGURES_ROOT / "Figure_1_coverage_probability_histograms.png"
    fig.tight_layout()
    fig.savefig(native(path), dpi=180)
    plt.close(fig)
    paths.append(path)

    # Figure 2: pixel vs probability coverage
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"OCO-2": "#264653", "OCO-3": "#e76f51", "TROPOSIF": "#457b9d"}
    for product, group in plot_frame.groupby("PRODUCT"):
        ax.scatter(
            group["coverage_pixel_fraction"],
            group["coverage_probability_fraction"],
            s=14,
            alpha=0.45,
            label=product,
            c=colors.get(product, None),
        )
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Pixel coverage fraction")
    ax.set_ylabel("Probability-weighted coverage fraction")
    ax.set_title("Figure 2. Pixel coverage versus EC-probability coverage")
    ax.legend()
    path = FIGURES_ROOT / "Figure_2_pixel_vs_probability_coverage.png"
    fig.tight_layout()
    fig.savefig(native(path), dpi=180)
    plt.close(fig)
    paths.append(path)

    # Figure 3: by IGBP
    valid_igbp = plot_frame.dropna(subset=["coverage_probability_fraction"]).copy()
    if "IGBP" in valid_igbp.columns and not valid_igbp.empty:
        pivot = (
            valid_igbp.groupby(["IGBP", "PRODUCT"])["coverage_probability_fraction"]
            .median()
            .unstack("PRODUCT")
            .sort_index()
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel("Median probability-weighted coverage")
        ax.set_title("Figure 3. Coverage by product and IGBP class")
        ax.set_ylim(0, 1)
        path = FIGURES_ROOT / "Figure_3_coverage_by_igbp.png"
        fig.tight_layout()
        fig.savefig(native(path), dpi=180)
        plt.close(fig)
        paths.append(path)

    # Figure 4: through time
    temp = add_time_features(plot_frame.dropna(subset=["coverage_probability_fraction"]).copy())
    if not temp.empty:
        temp["date"] = pd.to_datetime(temp["YYYYMM"].astype(str) + "15", format="%Y%m%d", errors="coerce")
        time_series = (
            temp.groupby(["date", "PRODUCT"])["coverage_probability_fraction"]
            .median()
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(11, 5))
        for product, group in time_series.groupby("PRODUCT"):
            ax.plot(group["date"], group["coverage_probability_fraction"], marker="o", ms=2, lw=1, label=product)
        ax.set_ylabel("Median probability-weighted coverage")
        ax.set_title("Figure 4. Coverage through time")
        ax.set_ylim(0, 1)
        ax.legend()
        path = FIGURES_ROOT / "Figure_4_coverage_through_time.png"
        fig.tight_layout()
        fig.savefig(native(path), dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def rank_reconstruction_need(product_summary: pd.DataFrame) -> list[str]:
    if product_summary.empty:
        return []
    frame = product_summary.copy()
    # Higher need = lower median/mean and higher <90 percentages.
    frame["need_score"] = (
        (1 - pd.to_numeric(frame["median_coverage_probability_fraction"], errors="coerce").fillna(0))
        + (pd.to_numeric(frame["pct_coverage_probability_lt_0p9"], errors="coerce").fillna(0) / 100)
        + (pd.to_numeric(frame["pct_coverage_probability_lt_0p5"], errors="coerce").fillna(0) / 100)
    )
    return frame.sort_values("need_score", ascending=False)["PRODUCT"].astype(str).tolist()


def possible_temporal_folds(coverage: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if coverage.empty:
        return pd.DataFrame(rows)
    for product, group in coverage.groupby("PRODUCT"):
        years = sorted(pd.to_numeric(group["year"], errors="coerce").dropna().astype(int).unique())
        if len(years) < 3:
            rows.append(
                {
                    "PRODUCT": product,
                    "feasible": False,
                    "possible_folds": "",
                    "notes": f"Only {len(years)} year(s) with records.",
                }
            )
            continue
        folds = []
        for idx in range(1, len(years)):
            train = years[:idx]
            test = years[idx]
            n_train = int(group[group["year"].isin(train)].shape[0])
            n_test = int(group[group["year"] == test].shape[0])
            if n_train > 0 and n_test > 0:
                folds.append(f"train {train[0]}-{train[-1]} -> test {test} (n_train={n_train}, n_test={n_test})")
        rows.append(
            {
                "PRODUCT": product,
                "feasible": len(folds) >= 2,
                "possible_folds": " | ".join(folds[:8]),
                "notes": "Expanding-window validation appears feasible." if len(folds) >= 2 else "Too few valid folds.",
            }
        )
    return pd.DataFrame(rows)


def make_report(
    gosif_docs: pd.DataFrame,
    gosif_trace: pd.DataFrame,
    coverage: pd.DataFrame,
    product_summary: pd.DataFrame,
    site_summary: pd.DataFrame,
    igbp_summary: pd.DataFrame,
    class_summary: pd.DataFrame,
    pathologies: pd.DataFrame,
    artificial_candidates: pd.DataFrame,
    temporal_folds: pd.DataFrame,
    figure_paths: list[Path],
    output_files: dict[str, Path],
) -> str:
    lines: list[str] = []
    lines.append("# SIF EC Reconstruction Model Preparation Report")
    lines.append("")
    lines.append(f"Generated: {timestamp()}")
    lines.append("")
    lines.append("## Scope and hard constraints")
    lines.append("")
    lines.append(
        "This run performed data verification, EC-footprint probability coverage quantification, "
        "and modelling-table preparation only. It did not train ML models, run kriging, reconstruct "
        "missing SIF, create synthetic SIF pixels, alter source SIF values, or modify GPP/source/existing TIFFs."
    )
    lines.append("")
    lines.append("## Question 1 — GOSIF scale")
    lines.append("")
    lines.append(f"Decision: **{GOSIF_SCALE_DECISION}**.")
    lines.append("")
    lines.append(
        "Evidence: local GOSIF GeoTIFFs expose rasterio/GDAL scales `(1.0,)` and offsets `(0.0,)`, "
        "but the official GOSIF v2 README states `Scale factor: 0.0001`, units `W m−2 μm−1 sr−1`, "
        "and fill values 32767/32766. The current helper `align_one_gridded_source()` reprojects raw "
        "GeoTIFF band values and masks only 32766/32767; it does not multiply by 0.0001. Therefore, "
        "existing GOSIF values written by the current pipeline are numerically too large by a factor of 10,000."
    )
    lines.append("")
    lines.append("GOSIF trace examples:")
    lines.append("")
    lines.append(df_to_markdown_table(gosif_trace, max_rows=10))
    lines.append("")
    lines.append("## EC coverage summary by product")
    lines.append("")
    lines.append(df_to_markdown_table(product_summary, max_rows=10))
    lines.append("")
    lines.append("Coverage class distribution:")
    lines.append("")
    lines.append(df_to_markdown_table(class_summary, max_rows=40))
    lines.append("")
    lines.append("## Question 2 — Coverage thresholds")
    lines.append("")
    if not product_summary.empty:
        cols = [
            "PRODUCT",
            "median_coverage_probability_fraction",
            "mean_coverage_probability_fraction",
            "pct_coverage_probability_lt_0p25",
            "pct_coverage_probability_lt_0p5",
            "pct_coverage_probability_lt_0p75",
            "pct_coverage_probability_lt_0p9",
            "pct_coverage_probability_lt_0p95",
            "pct_coverage_probability_lt_1p0",
        ]
        present_cols = [c for c in cols if c in product_summary.columns]
        lines.append(df_to_markdown_table(product_summary[present_cols], max_rows=10))
    else:
        lines.append("_Coverage summary unavailable._")
    lines.append("")
    lines.append("## Question 3 — Reconstruction necessity")
    lines.append("")
    ranking = rank_reconstruction_need(product_summary)
    if ranking:
        lines.append("Most to least in need of reconstruction, based on low median coverage and high incomplete-coverage percentages:")
        lines.append("")
        for idx, product in enumerate(ranking, start=1):
            lines.append(f"{idx}. {product}")
    else:
        lines.append("Ranking unavailable because coverage summary is empty.")
    lines.append("")
    lines.append("## Question 4 — Pixel vs probability coverage")
    lines.append("")
    if not coverage.empty:
        diff = coverage.copy()
        diff["coverage_difference_probability_minus_pixel"] = (
            pd.to_numeric(diff["coverage_probability_fraction"], errors="coerce")
            - pd.to_numeric(diff["coverage_pixel_fraction"], errors="coerce")
        )
        diff_summary = diff.groupby("PRODUCT")["coverage_difference_probability_minus_pixel"].agg(
            n="count",
            mean="mean",
            median="median",
            sd="std",
            min="min",
            max="max",
        ).reset_index()
        lines.append(df_to_markdown_table(diff_summary, max_rows=10))
        lines.append("")
        lines.append(
            "Interpretation: probability-weighted coverage is the scientifically preferred metric because it tells "
            "how much of the EC source contribution was observed, not merely how many template pixels were touched."
        )
    else:
        lines.append("_No coverage rows available._")
    lines.append("")
    lines.append("## Question 5 — Artificial-gap validation")
    lines.append("")
    if artificial_candidates.empty:
        lines.append("No high-coverage artificial-gap candidates were found.")
    else:
        cand_summary = artificial_candidates.groupby("PRODUCT").agg(
            N_candidates=("SITE_ID", "size"),
            N_sites=("SITE_ID", "nunique"),
            N_years=("year", "nunique"),
            ge_0p90=("candidate_ge_0p90", "sum"),
            ge_0p95=("candidate_ge_0p95", "sum"),
            eq_1p00=("candidate_eq_1p00", "sum"),
        ).reset_index()
        lines.append(df_to_markdown_table(cand_summary, max_rows=10))
        lines.append("")
        lines.append(
            "These are candidates for future artificial-gap validation only. No gaps were created in this run."
        )
    lines.append("")
    lines.append("## Question 6 — Spatial validation feasibility")
    lines.append("")
    if not product_summary.empty:
        spatial_rows = []
        for _, row in product_summary.iterrows():
            feasible = bool(row.get("N_sites", 0) >= 10 and row.get("N_site_months", 0) >= 100)
            spatial_rows.append(
                {
                    "PRODUCT": row["PRODUCT"],
                    "N_sites": row.get("N_sites", np.nan),
                    "N_site_months": row.get("N_site_months", np.nan),
                    "leave_site_out_feasible": feasible,
                    "notes": (
                        "Likely feasible, but use grouped/site-level folds to avoid same-site leakage."
                        if feasible
                        else "Likely weak; too few sites or records for robust leave-site-out validation."
                    ),
                }
            )
        lines.append(df_to_markdown_table(pd.DataFrame(spatial_rows), max_rows=10))
    else:
        lines.append("_Spatial feasibility unavailable._")
    lines.append("")
    lines.append("## Question 7 — Incremental temporal validation feasibility")
    lines.append("")
    lines.append(df_to_markdown_table(temporal_folds, max_rows=10))
    lines.append("")
    lines.append("## Question 8 — Ready for ML?")
    lines.append("")
    ready = False
    ready_reason = []
    if coverage.empty:
        ready_reason.append("coverage table is empty")
    if not pathologies.empty and (pathologies["pathology_flags"].astype(str).str.contains("coverage_probability_fraction > 1|unexplained candidate", regex=True).any()):
        ready_reason.append("pathological coverage/accounting cases need review")
    if product_summary.empty:
        ready_reason.append("product summaries are unavailable")
    if artificial_candidates.empty:
        ready_reason.append("no high-coverage records for artificial-gap validation")
    if not ready_reason:
        # Still conservative: model design is allowed only after reviewing GOSIF leakage and support notes.
        ready = True
    if ready:
        lines.append("**READY FOR MODEL-DESIGN PHASE**")
        lines.append("")
        lines.append(
            "Proceed to model-design only, not model training, with explicit leakage controls: exclude GOSIF from primary OCO-2 reconstruction; "
            "treat GOSIF-assisted OCO-3/TROPOSIF as support-aware EC aggregate predictors; and validate against target-only baselines."
        )
    else:
        lines.append("**NOT READY FOR MODEL-DESIGN PHASE**")
        lines.append("")
        lines.append("Reasons: " + "; ".join(ready_reason))
    lines.append("")
    lines.append("## Pathological cases")
    lines.append("")
    lines.append(f"Pathology rows: {len(pathologies)}")
    lines.append("")
    lines.append(df_to_markdown_table(pathologies[["PRODUCT", "SITE_ID", "YYYYMM", "pathology_flags"]].head(30) if not pathologies.empty else pathologies, max_rows=30))
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    for name, path in output_files.items():
        lines.append(f"- {name}: `{path}`")
    for path in figure_paths:
        lines.append(f"- Figure: `{path}`")
    lines.append("")
    lines.append("## Scientific interpretation reminders")
    lines.append("")
    lines.append(
        "`target_observed_EC_weighted_SIF` is observed EC-support SIF. It is not a complete EC-footprint SIF "
        "unless `coverage_probability_fraction` is approximately 1. Incomplete coverage means unobserved EC-support "
        "regions remain unmeasured; this run did not reconstruct them."
    )
    lines.append("")
    lines.append(
        "Tower GPP was not included as a predictor in any modelling inventory. It may be used later only as a response-side comparison target, not as input to SIF reconstruction."
    )
    lines.append("")
    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    ensure_dirs()
    configure_read_only_modules()
    print(f"[{timestamp()}] Starting preparation-only SIF EC reconstruction audit.")
    print(f"[{timestamp()}] Output root: {OUTPUT_ROOT}")

    # Inventories.
    gpp_inventory, templates = base.inventory_gpp_templates()
    templates = [t for t in templates if bool(t.get("valid_gpp_template", False))]
    meta = load_reference_metadata()
    meta_lookup_dict = metadata_lookup(meta)
    csv_report, csv_lookup = base.load_gpp_reference()
    # Use only GPP templates also considered valid by the reference CSV where possible.
    templates = base.filter_templates(templates, csv_lookup)
    tmpl_lookup = template_lookup(templates)
    sif_inventory, usable_sources = base.discover_sif_sources(SIF_DISCOVERY_PRODUCTS)
    usable_sources = [r for r in usable_sources if r["product"] in SIF_DISCOVERY_PRODUCTS]
    source_by_product_month: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in usable_sources:
        source_by_product_month[(str(rec["product"]), str(rec["detected_YYYYMM"]))].append(rec)

    write_csv(gpp_inventory, OUTPUT_ROOT / "GPP_TIFF_Template_Inventory.csv")
    write_csv(meta, OUTPUT_ROOT / "GPP_Reference_Metadata_SiteMonth.csv")
    write_csv(sif_inventory, OUTPUT_ROOT / "SIF_Source_File_Inventory.csv")

    # GOSIF scale audit and trace.
    gosif_docs = collect_gosif_documentation_evidence()
    gosif_trace = trace_gosif_scaling(templates, usable_sources)
    write_csv(gosif_docs, GOSIF_SCALE_ROOT / "GOSIF_scale_documentation_evidence.csv")
    write_csv(gosif_trace, GOSIF_SCALE_ROOT / "GOSIF_scaling_trace_examples.csv")

    # Month matching report for target products only.
    match_report, matches = base.build_month_matching_report(templates, TARGET_PRODUCTS, usable_sources)
    write_csv(match_report, OUTPUT_ROOT / "Target_Product_GPP_to_SIF_Month_Matching_Report.csv")

    all_product_coverage: list[pd.DataFrame] = []
    all_missing_geometry: list[pd.DataFrame] = []
    all_failures: list[pd.DataFrame] = []
    for product in TARGET_PRODUCTS:
        cov, missing_geom, failures = compute_target_product_coverage(product, matches, meta_lookup_dict)
        all_product_coverage.append(cov)
        if not missing_geom.empty:
            all_missing_geometry.append(missing_geom)
        if not failures.empty:
            all_failures.append(failures)

    coverage = pd.concat([f for f in all_product_coverage if not f.empty], ignore_index=True) if any(not f.empty for f in all_product_coverage) else pd.DataFrame()
    if not coverage.empty:
        coverage = add_time_features(coverage)
    no_sif_skips = build_no_sif_skips(templates, source_by_product_month, meta_lookup_dict)
    missing_geometry = pd.concat(all_missing_geometry, ignore_index=True) if all_missing_geometry else pd.DataFrame()
    failures = pd.concat(all_failures, ignore_index=True) if all_failures else pd.DataFrame()

    # Write coverage outputs.
    coverage_path = OUTPUT_ROOT / "EC_Coverage_All_Products.csv"
    write_csv(coverage, coverage_path)
    write_csv(no_sif_skips, OUTPUT_ROOT / "Skipped_No_Local_Target_SIF_For_GPP_Months.csv")
    write_csv(missing_geometry, OUTPUT_ROOT / "Missing_Target_Footprint_Geometry_Log.csv")
    write_csv(failures, OUTPUT_ROOT / "Coverage_Computation_Failures.csv")

    # GOSIF predictors for OCO-3/TROPOSIF only. OCO-2 diagnostic is deliberately excluded from primary inventory.
    gosif_needed = coverage[coverage["PRODUCT"].isin(["OCO-3", "TROPOSIF"])] if not coverage.empty else pd.DataFrame()
    gosif_predictors = compute_gosif_predictors(gosif_needed, usable_sources, tmpl_lookup) if not gosif_needed.empty else pd.DataFrame()
    write_csv(gosif_predictors, GOSIF_SCALE_ROOT / "GOSIF_scale_corrected_EC_weighted_predictors.csv")

    # Model inventories.
    oco2_inventory = build_model_inventory(coverage, "OCO-2", oco2_diagnostic=False)
    oco3_inventory = build_model_inventory(coverage, "OCO-3", gosif_predictors, include_gosif=True)
    tropo_inventory = build_model_inventory(coverage, "TROPOSIF", gosif_predictors, include_gosif=True)
    write_csv(oco2_inventory, OUTPUT_ROOT / "OCO2_EC_Reconstruction_Model_Inventory.csv")
    write_csv(oco3_inventory, OUTPUT_ROOT / "OCO3_GOSIF_EC_Reconstruction_Model_Inventory.csv")
    write_csv(tropo_inventory, OUTPUT_ROOT / "TROPOSIF_GOSIF_EC_Reconstruction_Model_Inventory.csv")

    # Summaries.
    product_summary = summarize_by_group(coverage, ["PRODUCT"]) if not coverage.empty else pd.DataFrame()
    site_summary = summarize_by_group(coverage, ["PRODUCT", "SITE_ID"]) if not coverage.empty else pd.DataFrame()
    igbp_summary = summarize_by_group(coverage, ["PRODUCT", "IGBP"]) if not coverage.empty else pd.DataFrame()
    class_summary = coverage_class_summary(coverage) if not coverage.empty else pd.DataFrame()
    pathologies = identify_pathologies(coverage) if not coverage.empty else pd.DataFrame()
    artificial_candidates = artificial_gap_candidates(coverage) if not coverage.empty else pd.DataFrame()
    temporal_folds = possible_temporal_folds(coverage) if not coverage.empty else pd.DataFrame()

    write_csv(product_summary, OUTPUT_ROOT / "EC_Coverage_Summary_By_Product.csv")
    write_csv(site_summary, OUTPUT_ROOT / "EC_Coverage_Summary_By_Site.csv")
    write_csv(igbp_summary, OUTPUT_ROOT / "EC_Coverage_Summary_By_IGBP.csv")
    write_csv(class_summary, OUTPUT_ROOT / "EC_Coverage_Class_Distribution_By_Product.csv")
    write_csv(pathologies, OUTPUT_ROOT / "EC_Coverage_Pathological_Cases.csv")
    write_csv(artificial_candidates, OUTPUT_ROOT / "Artificial_Gap_Candidate_Inventory.csv")
    write_csv(temporal_folds, OUTPUT_ROOT / "Future_Incremental_Temporal_Validation_Feasibility.csv")

    figure_paths = make_figures(coverage)

    output_files = {
        "EC_Coverage_All_Products.csv": coverage_path,
        "EC_Coverage_Summary_By_Product.csv": OUTPUT_ROOT / "EC_Coverage_Summary_By_Product.csv",
        "EC_Coverage_Summary_By_Site.csv": OUTPUT_ROOT / "EC_Coverage_Summary_By_Site.csv",
        "EC_Coverage_Summary_By_IGBP.csv": OUTPUT_ROOT / "EC_Coverage_Summary_By_IGBP.csv",
        "Artificial_Gap_Candidate_Inventory.csv": OUTPUT_ROOT / "Artificial_Gap_Candidate_Inventory.csv",
        "OCO2_EC_Reconstruction_Model_Inventory.csv": OUTPUT_ROOT / "OCO2_EC_Reconstruction_Model_Inventory.csv",
        "OCO3_GOSIF_EC_Reconstruction_Model_Inventory.csv": OUTPUT_ROOT / "OCO3_GOSIF_EC_Reconstruction_Model_Inventory.csv",
        "TROPOSIF_GOSIF_EC_Reconstruction_Model_Inventory.csv": OUTPUT_ROOT / "TROPOSIF_GOSIF_EC_Reconstruction_Model_Inventory.csv",
        "GOSIF_scale_documentation_evidence.csv": GOSIF_SCALE_ROOT / "GOSIF_scale_documentation_evidence.csv",
        "GOSIF_scaling_trace_examples.csv": GOSIF_SCALE_ROOT / "GOSIF_scaling_trace_examples.csv",
        "GOSIF_scale_corrected_EC_weighted_predictors.csv": GOSIF_SCALE_ROOT / "GOSIF_scale_corrected_EC_weighted_predictors.csv",
    }
    report = make_report(
        gosif_docs,
        gosif_trace,
        coverage,
        product_summary,
        site_summary,
        igbp_summary,
        class_summary,
        pathologies,
        artificial_candidates,
        temporal_folds,
        figure_paths,
        output_files,
    )
    md_path = OUTPUT_ROOT / "SIF_EC_Reconstruction_Model_Preparation_Report.md"
    pdf_path = OUTPUT_ROOT / "SIF_EC_Reconstruction_Model_Preparation_Report.pdf"
    write_markdown(report, md_path)
    write_pdf_from_markdown(report, pdf_path)

    # Terminal summary.
    print("\n" + "=" * 72)
    print("SIF EC RECONSTRUCTION MODEL PREPARATION SUMMARY")
    print("=" * 72)
    print(f"GPP templates used: {len(templates):,}")
    print(f"SIF source files discovered: {len(sif_inventory):,}")
    print(f"GOSIF scale decision: {GOSIF_SCALE_DECISION}")
    for product in TARGET_PRODUCTS:
        created = int((coverage["PRODUCT"] == product).sum()) if not coverage.empty else 0
        print(f"{product} coverage records: {created:,}")
    print(f"Artificial-gap candidates: {len(artificial_candidates):,}")
    print(f"Pathological cases flagged: {len(pathologies):,}")
    print(f"Figures created: {len(figure_paths):,}")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Main Markdown report: {md_path}")
    print(f"Main PDF report: {pdf_path}")


if __name__ == "__main__":
    main()
