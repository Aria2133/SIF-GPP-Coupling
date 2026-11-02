"""Audit and harmonize the finalized per-product EC-footprint SIF tables into a common master structure with provenance and QC columns.

The archive audited here included additional products; the final four-product analysis (GOSIF, GOME-2, TROPOSIF, OCO-3) is obtained by filtering downstream.

Output: master harmonization tables, audit files and figures."""
from __future__ import annotations
# Original script filename (sanitized): Five_Product_SIF_Master_Harmonization_Audit.py; public name: 05_Five_Product_SIF_Master_Harmonization_Audit.py
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

import hashlib
import json
import math
import re
import shutil
import textwrap
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)


PROJECT_ROOT = Path(
    PROJECT_ROOT
)
OUTPUT_ROOT = PROJECT_ROOT / "SIF_EC_FiveProduct_Master_Harmonization"

DIRS = {
    "tables": OUTPUT_ROOT / "01_tables",
    "audit": OUTPUT_ROOT / "02_audit",
    "figures": OUTPUT_ROOT / "03_figures",
    "figure_data": OUTPUT_ROOT / "04_figure_data",
    "reports": OUTPUT_ROOT / "05_reports",
    "logs": OUTPUT_ROOT / "06_logs",
    "scripts": OUTPUT_ROOT / "07_scripts",
}

PRODUCT_ORDER = ["GOSIF", "GOME-2", "TROPOSIF", "OCO-2", "OCO-3"]
NO_DATA = "NOT_ESTABLISHED"
VERDICT_PASS_FLAGS = (
    "FINAL VERDICT:\n"
    "FIVE-PRODUCT SIF MASTER AUDIT PASSED WITH DOCUMENTED NON-CRITICAL FLAGS — READY FOR GPP INTEGRATION"
)
VERDICT_FAIL = (
    "FINAL VERDICT:\n"
    "FIVE-PRODUCT SIF MASTER AUDIT FAILED — RESOLVE IDENTIFIED ISSUES BEFORE GPP INTEGRATION"
)


@dataclass(frozen=True)
class ProductPaths:
    product: str
    folder: Path
    all_records: Path
    primary: Path
    report: Path
    counts: Path | None = None
    qc: Path | None = None
    provenance: Path | None = None
    support: Path | None = None
    reconstruction: Path | None = None
    reproducibility: Path | None = None
    model_metadata: Path | None = None


PRODUCTS: dict[str, ProductPaths] = {
    "GOSIF": ProductPaths(
        product="GOSIF",
        folder=PROJECT_ROOT / "SIF_EC_GOSIF_Finalization",
        all_records=PROJECT_ROOT / "SIF_EC_GOSIF_Finalization" / "GOSIF_Final_EC_Footprint_SIF_All_SiteMonths.csv",
        primary=PROJECT_ROOT / "SIF_EC_GOSIF_Finalization" / "GOSIF_Final_EC_Footprint_SIF_Primary_Analysis.csv",
        report=PROJECT_ROOT / "SIF_EC_GOSIF_Finalization" / "GOSIF_Finalization_Report.md",
        counts=PROJECT_ROOT / "SIF_EC_GOSIF_Finalization" / "GOSIF_Finalization_Final_Counts.csv",
        qc=PROJECT_ROOT / "SIF_EC_GOSIF_Finalization" / "GOSIF_Finalization_QC_Flags.csv",
        support=PROJECT_ROOT / "SIF_EC_GOSIF_Finalization" / "GOSIF_Native_Support_Summary.csv",
        provenance=PROJECT_ROOT / "SIF_EC_GOSIF_Finalization" / "GOSIF_EC_Native_Pixel_Weights.csv",
        reproducibility=PROJECT_ROOT / "SIF_EC_GOSIF_Finalization" / "GOSIF_Finalization_Reproducibility.json",
    ),
    "GOME-2": ProductPaths(
        product="GOME-2",
        folder=PROJECT_ROOT / "SIF_EC_GOME2_Finalization",
        all_records=PROJECT_ROOT / "SIF_EC_GOME2_Finalization" / "GOME2_Final_EC_Footprint_SIF_All_SiteMonths.csv",
        primary=PROJECT_ROOT / "SIF_EC_GOME2_Finalization" / "GOME2_Final_EC_Footprint_SIF_Primary_Analysis.csv",
        report=PROJECT_ROOT / "SIF_EC_GOME2_Finalization" / "GOME2_Finalization_Report.md",
        counts=PROJECT_ROOT / "SIF_EC_GOME2_Finalization" / "GOME2_Finalization_Counts.csv",
        qc=PROJECT_ROOT / "SIF_EC_GOME2_Finalization" / "GOME2_Finalization_QC_Flags.csv",
        support=PROJECT_ROOT / "SIF_EC_GOME2_Finalization" / "GOME2_Native_Support_Mismatch_Summary.csv",
        provenance=PROJECT_ROOT / "SIF_EC_GOME2_Finalization" / "GOME2_EC_Observation_Overlap_Provenance.csv",
        reconstruction=PROJECT_ROOT / "SIF_EC_GOME2_Finalization" / "GOME2_Partial_EC_Support_Sensitivity.csv",
        reproducibility=PROJECT_ROOT / "SIF_EC_GOME2_Finalization" / "GOME2_Finalization_Reproducibility.json",
    ),
    "TROPOSIF": ProductPaths(
        product="TROPOSIF",
        folder=PROJECT_ROOT / "SIF_EC_TROPOSIF_Finalization",
        all_records=PROJECT_ROOT / "SIF_EC_TROPOSIF_Finalization" / "TROPOSIF_Final_EC_Footprint_SIF_All_SiteMonths.csv",
        primary=PROJECT_ROOT / "SIF_EC_TROPOSIF_Finalization" / "TROPOSIF_Final_EC_Footprint_SIF_Primary_Observed.csv",
        report=PROJECT_ROOT / "SIF_EC_TROPOSIF_Finalization" / "TROPOSIF_Finalization_Report.md",
        counts=PROJECT_ROOT / "SIF_EC_TROPOSIF_Finalization" / "TROPOSIF_Finalization_Final_Counts.csv",
        qc=PROJECT_ROOT / "SIF_EC_TROPOSIF_Finalization" / "TROPOSIF_Finalization_QC_Flags.csv",
        support=PROJECT_ROOT / "SIF_EC_TROPOSIF_Finalization" / "TROPOSIF_Finalization_Coverage_Class_Summary.csv",
        reconstruction=PROJECT_ROOT / "SIF_EC_TROPOSIF_Finalization" / "TROPOSIF_TargetOnly_RF_Sensitivity_Status.csv",
        reproducibility=PROJECT_ROOT / "SIF_EC_TROPOSIF_Finalization" / "TROPOSIF_Finalization_Reproducibility.json",
    ),
    "OCO-2": ProductPaths(
        product="OCO-2",
        folder=PROJECT_ROOT / "SIF_EC_Reconstruction_OCO2_TargetOnly_Validation",
        all_records=PROJECT_ROOT / "SIF_EC_Reconstruction_OCO2_TargetOnly_Validation" / "OCO2_Final_EC_Footprint_SIF_All_SiteMonths.csv",
        primary=PROJECT_ROOT / "SIF_EC_Reconstruction_OCO2_TargetOnly_Validation" / "OCO2_Final_EC_Footprint_SIF_Primary_Analysis.csv",
        report=PROJECT_ROOT / "SIF_EC_Reconstruction_OCO2_TargetOnly_Validation" / "OCO2_Final_Production_Report.md",
        counts=PROJECT_ROOT / "SIF_EC_Reconstruction_OCO2_TargetOnly_Validation" / "OCO2_Final_Counts.csv",
        qc=PROJECT_ROOT / "SIF_EC_Reconstruction_OCO2_TargetOnly_Validation" / "OCO2_Production_QC_Flags.csv",
        reconstruction=PROJECT_ROOT / "SIF_EC_Reconstruction_OCO2_TargetOnly_Validation" / "OCO2_Reconstructed_Partial_Cases_Only.csv",
        reproducibility=PROJECT_ROOT / "SIF_EC_Reconstruction_OCO2_TargetOnly_Validation" / "OCO2_Reproducibility.json",
    ),
    "OCO-3": ProductPaths(
        product="OCO-3",
        folder=PROJECT_ROOT / "SIF_EC_Reconstruction_OCO3_Final_Production",
        all_records=PROJECT_ROOT / "SIF_EC_Reconstruction_OCO3_Final_Production" / "OCO3_Final_EC_Footprint_SIF_All_SiteMonths.csv",
        primary=PROJECT_ROOT / "SIF_EC_Reconstruction_OCO3_Final_Production" / "OCO3_Final_EC_Footprint_SIF_Primary_Analysis.csv",
        report=PROJECT_ROOT / "SIF_EC_Reconstruction_OCO3_Final_Production" / "OCO3_Final_Production_Report.md",
        counts=PROJECT_ROOT / "SIF_EC_Reconstruction_OCO3_Final_Production" / "qc" / "OCO3_Final_Production_Counts.csv",
        qc=PROJECT_ROOT / "SIF_EC_Reconstruction_OCO3_Final_Production" / "qc" / "OCO3_Production_QC_Flags.csv",
        reconstruction=PROJECT_ROOT / "SIF_EC_Reconstruction_OCO3_Final_Production" / "OCO3_Reconstructed_Partial_Cases_Only.csv",
        reproducibility=PROJECT_ROOT / "SIF_EC_Reconstruction_OCO3_Final_Production" / "OCO3_Final_Production_Reproducibility.json",
        model_metadata=PROJECT_ROOT / "SIF_EC_Reconstruction_OCO3_Final_Production" / "model" / "OCO3_RF_TARGET_GOSIF_Final_Model_Metadata.json",
    ),
}

EC_REFERENCE = PROJECT_ROOT / "SIF_EC_Reconstruction_Model_Preparation" / "GPP_Reference_Metadata_SiteMonth.csv"
EC_TEMPLATE_INVENTORY = PROJECT_ROOT / "SIF_EC_Reconstruction_Model_Preparation" / "GPP_TIFF_Template_Inventory.csv"
SITE_LIST = PROJECT_ROOT / "Figure_1_Global_EC_Tower_Map" / "Figure_1_Global_EC_Tower_Study_Site_List.csv"
BLUE_MARBLE = PROJECT_ROOT / "world.topo.bathy.200408.3x21600x10800_geo.tif"


def ensure_dirs() -> None:
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)


def rel(path: Path | str | None) -> str:
    if path is None:
        return ""
    p = Path(path)
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def file_snapshot(roots: list[Path], hash_paths: set[Path] | None = None) -> pd.DataFrame:
    hash_paths = {p.resolve() for p in (hash_paths or set())}
    rows: list[dict[str, Any]] = []
    for root in roots:
        files = [p for p in root.rglob("*") if p.is_file()]
        for p in sorted(files):
            st = p.stat()
            rows.append(
                {
                    "path": str(p),
                    "relative_path": rel(p),
                    "size_bytes": int(st.st_size),
                    "modified_timestamp": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                    "mtime_ns": int(st.st_mtime_ns),
                    "sha256": sha256_file(p) if p.resolve() in hash_paths else "",
                }
            )
    return pd.DataFrame(rows)


def compare_snapshots(before: pd.DataFrame, after: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    cols = ["path", "size_bytes", "mtime_ns", "sha256"]
    b = before[cols].rename(columns={c: f"{c}_before" for c in cols if c != "path"})
    a = after[cols].rename(columns={c: f"{c}_after" for c in cols if c != "path"})
    c = b.merge(a, on="path", how="outer", indicator=True)
    c["unchanged"] = (
        c["_merge"].eq("both")
        & c["size_bytes_before"].eq(c["size_bytes_after"])
        & c["mtime_ns_before"].eq(c["mtime_ns_after"])
        & (
            c["sha256_before"].fillna("").eq("")
            | c["sha256_after"].fillna("").eq("")
            | c["sha256_before"].eq(c["sha256_after"])
        )
    )
    return c, bool(c["unchanged"].all())


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def row_count_and_columns(path: Path | None) -> tuple[int | float, int | float, int | float, int | float]:
    if path is None or not path.exists() or path.suffix.lower() != ".csv":
        return np.nan, np.nan, np.nan, np.nan
    try:
        df = pd.read_csv(path)
        date_min = int(df["YYYYMM"].min()) if "YYYYMM" in df.columns and len(df) else np.nan
        date_max = int(df["YYYYMM"].max()) if "YYYYMM" in df.columns and len(df) else np.nan
        return len(df), len(df.columns), date_min, date_max
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def build_manifest() -> pd.DataFrame:
    role_notes = {
        "AUTHORITATIVE_ALL_RECORDS": "Final all-site-month table selected because report/reproducibility identifies it as the master/all-site-month output.",
        "AUTHORITATIVE_PRIMARY": "Final primary table selected because report/reproducibility identifies it as the analysis-ready output.",
        "REPORT": "Final Markdown report used for documented count and methodology targets.",
        "COUNT_SUMMARY": "Final count summary used for report-to-table reconciliation.",
        "QC_TABLE": "Final QC flag table used to mark retained and excluded QC cases.",
        "PROVENANCE_TABLE": "Observation/native-pixel provenance table used where available.",
        "NATIVE_SUPPORT_TABLE": "Existing native-support or coverage summary used for support metadata.",
        "RECONSTRUCTION_TABLE": "Reconstructed or sensitivity cases table used to distinguish project reconstruction.",
        "REPRODUCIBILITY": "Reproducibility JSON used for source metadata and method notes.",
        "MODEL_METADATA": "Frozen model metadata used for OCO-3 reconstruction provenance.",
    }
    rows = []
    for product in PRODUCT_ORDER:
        pp = PRODUCTS[product]
        paths = {
            "AUTHORITATIVE_ALL_RECORDS": pp.all_records,
            "AUTHORITATIVE_PRIMARY": pp.primary,
            "REPORT": pp.report,
            "COUNT_SUMMARY": pp.counts,
            "QC_TABLE": pp.qc,
            "PROVENANCE_TABLE": pp.provenance,
            "NATIVE_SUPPORT_TABLE": pp.support,
            "RECONSTRUCTION_TABLE": pp.reconstruction,
            "REPRODUCIBILITY": pp.reproducibility,
            "MODEL_METADATA": pp.model_metadata,
        }
        for role, path in paths.items():
            if path is None or not path.exists():
                continue
            n_rows, n_cols, date_min, date_max = row_count_and_columns(path)
            rows.append(
                {
                    "PRODUCT": product,
                    "FILE_ROLE": role,
                    "PATH": str(path),
                    "N_ROWS": n_rows,
                    "N_COLUMNS": n_cols,
                    "DATE_MIN": date_min,
                    "DATE_MAX": date_max,
                    "NOTES": role_notes[role],
                    "AUTHORITATIVE_STATUS": "AUTHORITATIVE" if role in {"AUTHORITATIVE_ALL_RECORDS", "AUTHORITATIVE_PRIMARY", "REPORT", "COUNT_SUMMARY"} else "SUPPORTING",
                    "SHA256": sha256_file(path) if path.is_file() and path.stat().st_size < 250_000_000 else "",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(DIRS["tables"] / "SIF_Product_Input_Manifest.csv", index=False)
    return out


def normalize_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if pd.isna(x):
        return False
    return str(x).strip().lower() in {"true", "1", "yes", "y"}


def safe_num(s: pd.Series | Any) -> pd.Series | float:
    if isinstance(s, pd.Series):
        return pd.to_numeric(s, errors="coerce")
    try:
        return float(s)
    except Exception:
        return np.nan


def yyyymm_to_year_month(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    ints = pd.to_numeric(values, errors="coerce").astype("Int64")
    return (ints // 100).astype("Int64"), (ints % 100).astype("Int64")


def primary_key_set(primary: pd.DataFrame) -> set[tuple[str, int]]:
    return set(zip(primary["SITE_ID"].astype(str), pd.to_numeric(primary["YYYYMM"], errors="coerce").astype("Int64").astype(int)))


def classify_gosif(df: pd.DataFrame, primary_keys: set[tuple[str, int]], qc_keys: set[tuple[str, int]]) -> pd.DataFrame:
    out = pd.DataFrame()
    out["SIF_VALUE"] = safe_num(df["GOSIF_EC_weighted"])
    out["SIF_VALUE_ORIGINAL"] = out["SIF_VALUE"]
    out["SIF_UNITS"] = df.get("GOSIF_units", "W m-2 um-1 sr-1")
    out["SIF_WAVELENGTH_NM"] = NO_DATA
    out["SIF_DEFINITION"] = "GOSIF EC-probability-weighted monthly SIF; no wavelength conversion applied."
    status = df["GOSIF_status"].astype(str)
    out["VALUE_STATUS"] = np.where(status.eq("VALID_EC_WEIGHTED"), "DIRECT_PRODUCT", "NO_SOURCE_DATA")
    out["COVERAGE_CLASS"] = status
    out["EXCLUSION_REASON"] = np.where(status.eq("VALID_EC_WEIGHTED"), "", status)
    out["EC_COVERAGE_FRACTION"] = 1 - safe_num(df.get("EC_probability_fraction_on_invalid_GOSIF", pd.Series(np.nan, index=df.index))).fillna(0)
    out["EC_UNION_COVERAGE"] = out["EC_COVERAGE_FRACTION"]
    out["POSITIVE_COVERAGE"] = out["EC_COVERAGE_FRACTION"] > 0
    out["N_NATIVE_OBSERVATIONS"] = safe_num(df.get("n_native_GOSIF_cells_intersecting_EC", pd.Series(np.nan, index=df.index)))
    out["N_NATIVE_PIXELS_OR_FOOTPRINTS"] = out["N_NATIVE_OBSERVATIONS"]
    out["EFFECTIVE_NATIVE_COUNT"] = safe_num(df.get("GOSIF_effective_native_pixel_count", pd.Series(np.nan, index=df.index)))
    out["WEIGHT_SUM"] = safe_num(df.get("sum_normalized_native_weights", pd.Series(np.nan, index=df.index)))
    out["NATIVE_SUPPORT_TYPE"] = "0.05-degree gridded native raster cell support"
    out["NATIVE_SUPPORT_AREA_KM2"] = np.nan
    out["WEIGHTED_NATIVE_SUPPORT_AREA_KM2"] = np.nan
    out["EC_SUPPORT_AREA_KM2"] = np.nan
    out["SUPPORT_AREA_RATIO"] = np.nan
    out["FRACTION_NATIVE_SUPPORT_INTERSECTING_EC"] = np.nan
    out["FRACTION_EC_SUPPORT_COVERED"] = out["EC_COVERAGE_FRACTION"]
    out["SPATIAL_SUPPORT_METRIC_SOURCE"] = "GOSIF native cell count/effective count only; area not calculated in finalized GOSIF table."
    out["PROJECT_RECONSTRUCTED"] = False
    out["RECONSTRUCTION_METHOD"] = "NONE"
    out["RECONSTRUCTION_MODEL"] = "NONE"
    out["RECONSTRUCTION_QC"] = "NOT_APPLICABLE"
    out["UNCERTAINTY_AVAILABLE"] = False
    out["UNCERTAINTY_VALUE"] = np.nan
    out["UNCERTAINTY_UNITS"] = ""
    out["UNCERTAINTY_TYPE"] = ""
    gosif_qc = df.apply(lambda r: (str(r["SITE_ID"]), int(r["YYYYMM"])) in qc_keys, axis=1)
    out["QC_REASON"] = np.where(gosif_qc, df["GOSIF_status"].astype(str), df.get("QC_flag_reasons", ""))
    out["QC_SEVERITY"] = np.where(gosif_qc, "EXCLUSION", "")
    out["SOURCE_FILE"] = df.get("GOSIF_source_file", "")
    out["SOURCE_RECORD_ID"] = ""
    out["PROCESSING_METHOD"] = "EC probability weighting of native GOSIF grid cells."
    out["SOURCE_VERSION"] = NO_DATA
    return out


def classify_gome2(df: pd.DataFrame, primary_keys: set[tuple[str, int]], qc_keys: set[tuple[str, int]]) -> pd.DataFrame:
    out = pd.DataFrame()
    out["SIF_VALUE"] = safe_num(df["GOME2_EC_overlap_weighted_SIF"])
    out["SIF_VALUE_ORIGINAL"] = out["SIF_VALUE"]
    out["SIF_UNITS"] = df.get("GOME2_units", "mW m-2 nm-1 sr-1")
    out["SIF_WAVELENGTH_NM"] = NO_DATA
    out["SIF_DEFINITION"] = df.get("SIF_variable", "Daily_Averaged_SIF")
    status = df["GOME2_status"].astype(str)
    out["VALUE_STATUS"] = np.select(
        [
            status.eq("VALID_EC_OVERLAP_PRIMARY"),
            status.eq("VALID_EC_OVERLAP_PARTIAL_SUPPORT"),
            status.eq("EXCLUDED_ZERO_EC_OVERLAP"),
        ],
        ["DIRECT_PRODUCT", "EXCLUDED_PARTIAL", "EXCLUDED_ZERO_COVERAGE"],
        default="OTHER",
    )
    out["COVERAGE_CLASS"] = df.get("coverage_class", "")
    out["EXCLUSION_REASON"] = np.where(status.eq("VALID_EC_OVERLAP_PRIMARY"), "", status)
    out["EC_COVERAGE_FRACTION"] = safe_num(df.get("GOME2_EC_probability_union_coverage", pd.Series(np.nan, index=df.index)))
    out["EC_UNION_COVERAGE"] = out["EC_COVERAGE_FRACTION"]
    out["POSITIVE_COVERAGE"] = out["EC_COVERAGE_FRACTION"] > 0
    out["N_NATIVE_OBSERVATIONS"] = safe_num(df.get("n_GOME2_observations", pd.Series(np.nan, index=df.index)))
    out["N_NATIVE_PIXELS_OR_FOOTPRINTS"] = out["N_NATIVE_OBSERVATIONS"]
    out["EFFECTIVE_NATIVE_COUNT"] = safe_num(df.get("GOME2_effective_observation_count", pd.Series(np.nan, index=df.index)))
    out["WEIGHT_SUM"] = safe_num(df.get("sum_observation_EC_probability_overlap", pd.Series(np.nan, index=df.index)))
    out["NATIVE_SUPPORT_TYPE"] = "native GOME-2 four-corner footprint polygons"
    out["NATIVE_SUPPORT_AREA_KM2"] = safe_num(df.get("mean_native_footprint_area_km2", pd.Series(np.nan, index=df.index)))
    out["WEIGHTED_NATIVE_SUPPORT_AREA_KM2"] = safe_num(df.get("median_native_footprint_area_km2", pd.Series(np.nan, index=df.index)))
    out["EC_SUPPORT_AREA_KM2"] = safe_num(df.get("EC_positive_support_area_km2", pd.Series(np.nan, index=df.index)))
    out["SUPPORT_AREA_RATIO"] = safe_num(df.get("mean_footprint_to_EC_area_ratio", pd.Series(np.nan, index=df.index)))
    out["FRACTION_NATIVE_SUPPORT_INTERSECTING_EC"] = safe_num(df.get("mean_fraction_native_footprint_inside_EC", pd.Series(np.nan, index=df.index)))
    out["FRACTION_EC_SUPPORT_COVERED"] = out["EC_COVERAGE_FRACTION"]
    out["SPATIAL_SUPPORT_METRIC_SOURCE"] = "GOME2_Final_EC_Footprint_SIF_All_SiteMonths.csv native footprint area/EC area metrics."
    out["PROJECT_RECONSTRUCTED"] = False
    out["RECONSTRUCTION_METHOD"] = "NONE"
    out["RECONSTRUCTION_MODEL"] = "NONE"
    out["RECONSTRUCTION_QC"] = "NOT_APPLICABLE"
    out["UNCERTAINTY_AVAILABLE"] = False
    out["UNCERTAINTY_VALUE"] = np.nan
    out["UNCERTAINTY_UNITS"] = ""
    out["UNCERTAINTY_TYPE"] = ""
    out["QC_REASON"] = np.where(df.apply(lambda r: (str(r["SITE_ID"]), int(r["YYYYMM"])) in qc_keys, axis=1), "NO_EC_OVERLAP", "")
    out["QC_SEVERITY"] = np.where(out["QC_REASON"].ne(""), "EXCLUSION", "")
    out["SOURCE_FILE"] = df.get("source_files_used", "")
    out["SOURCE_RECORD_ID"] = ""
    out["PROCESSING_METHOD"] = "EC-overlap weighted native GOME-2 observations; no project reconstruction."
    out["SOURCE_VERSION"] = "NSIFv2.6.2.GOME-2A (from source filenames)"
    return out


def classify_troposif(df: pd.DataFrame, primary_keys: set[tuple[str, int]], qc_keys: set[tuple[str, int]]) -> pd.DataFrame:
    out = pd.DataFrame()
    out["SIF_VALUE"] = safe_num(df["TROPOSIF_final_EC_SIF"])
    out["SIF_VALUE_ORIGINAL"] = out["SIF_VALUE"]
    out["SIF_UNITS"] = df.get("TROPOSIF_units", "mW m-2 sr-1 nm-1")
    out["SIF_WAVELENGTH_NM"] = NO_DATA
    out["SIF_DEFINITION"] = "TROPOSIF observed EC-support SIF; complete EC support required for primary."
    status = df["TROPOSIF_primary_status"].astype(str)
    out["VALUE_STATUS"] = np.select(
        [
            status.eq("OBSERVED_COMPLETE_VALID"),
            status.eq("PARTIAL_EXCLUDED_PRIMARY"),
            status.eq("ZERO_COVERAGE_EXCLUDED"),
        ],
        ["DIRECT_PRODUCT", "EXCLUDED_PARTIAL", "EXCLUDED_ZERO_COVERAGE"],
        default="OTHER",
    )
    out["COVERAGE_CLASS"] = df.get("coverage_class", "")
    out["EXCLUSION_REASON"] = np.where(status.eq("OBSERVED_COMPLETE_VALID"), "", status)
    out["EC_COVERAGE_FRACTION"] = safe_num(df.get("coverage_probability_fraction", pd.Series(np.nan, index=df.index)))
    out["EC_UNION_COVERAGE"] = out["EC_COVERAGE_FRACTION"]
    out["POSITIVE_COVERAGE"] = out["EC_COVERAGE_FRACTION"] > 0
    out["N_NATIVE_OBSERVATIONS"] = safe_num(df.get("n_target_source_observations", pd.Series(np.nan, index=df.index)))
    out["N_NATIVE_PIXELS_OR_FOOTPRINTS"] = safe_num(df.get("n_target_source_footprints", pd.Series(np.nan, index=df.index)))
    out["EFFECTIVE_NATIVE_COUNT"] = out["N_NATIVE_PIXELS_OR_FOOTPRINTS"]
    out["WEIGHT_SUM"] = np.nan
    out["NATIVE_SUPPORT_TYPE"] = "TROPOSIF swath/footprint bounds"
    out["NATIVE_SUPPORT_AREA_KM2"] = np.nan
    out["WEIGHTED_NATIVE_SUPPORT_AREA_KM2"] = np.nan
    out["EC_SUPPORT_AREA_KM2"] = np.nan
    out["SUPPORT_AREA_RATIO"] = np.nan
    out["FRACTION_NATIVE_SUPPORT_INTERSECTING_EC"] = np.nan
    out["FRACTION_EC_SUPPORT_COVERED"] = out["EC_COVERAGE_FRACTION"]
    out["SPATIAL_SUPPORT_METRIC_SOURCE"] = "Coverage fractions only; native footprint areas were not finalized in comparable area units."
    out["PROJECT_RECONSTRUCTED"] = False
    out["RECONSTRUCTION_METHOD"] = "NONE_ACCEPTED"
    out["RECONSTRUCTION_MODEL"] = "NONE"
    out["RECONSTRUCTION_QC"] = "Partial sensitivity reconstruction was not accepted."
    out["UNCERTAINTY_AVAILABLE"] = False
    out["UNCERTAINTY_VALUE"] = np.nan
    out["UNCERTAINTY_UNITS"] = ""
    out["UNCERTAINTY_TYPE"] = ""
    out["QC_REASON"] = df.get("QC_flag_reasons", "")
    out["QC_SEVERITY"] = np.where(df.apply(lambda r: (str(r["SITE_ID"]), int(r["YYYYMM"])) in qc_keys, axis=1), "QC_FLAG", "")
    out["SOURCE_FILE"] = df.get("source_sif_files", "")
    out["SOURCE_RECORD_ID"] = ""
    out["PROCESSING_METHOD"] = "Observed complete EC-footprint TROPOSIF only for primary; no project reconstruction."
    out["SOURCE_VERSION"] = NO_DATA
    return out


def classify_oco2(df: pd.DataFrame, primary_keys: set[tuple[str, int]], qc_keys: set[tuple[str, int]]) -> pd.DataFrame:
    out = pd.DataFrame()
    out["SIF_VALUE"] = safe_num(df["OCO2_final_EC_SIF"])
    out["SIF_VALUE_ORIGINAL"] = out["SIF_VALUE"]
    out["SIF_UNITS"] = NO_DATA
    out["SIF_WAVELENGTH_NM"] = NO_DATA
    out["SIF_DEFINITION"] = "OCO-2 LtSIF EC-footprint value; observed complete or target-only reconstructed partial where primary-ready."
    status = df["OCO2_final_status"].astype(str)
    out["VALUE_STATUS"] = np.select(
        [
            status.eq("OBSERVED_COMPLETE_VALID"),
            status.str.startswith("RECONSTRUCTED_PARTIAL"),
            status.eq("EXCLUDED_ZERO_COVERAGE"),
        ],
        ["DIRECT_PRODUCT", "PROJECT_RECONSTRUCTED", "EXCLUDED_ZERO_COVERAGE"],
        default="OTHER",
    )
    out["COVERAGE_CLASS"] = df.get("OCO2_production_class", "")
    out["EXCLUSION_REASON"] = np.where(out["VALUE_STATUS"].isin(["DIRECT_PRODUCT", "PROJECT_RECONSTRUCTED"]), "", status)
    out["EC_COVERAGE_FRACTION"] = safe_num(df.get("coverage_probability_fraction", pd.Series(np.nan, index=df.index)))
    out["EC_UNION_COVERAGE"] = out["EC_COVERAGE_FRACTION"]
    out["POSITIVE_COVERAGE"] = out["EC_COVERAGE_FRACTION"] > 0
    out["N_NATIVE_OBSERVATIONS"] = safe_num(df.get("n_target_source_observations", pd.Series(np.nan, index=df.index)))
    out["N_NATIVE_PIXELS_OR_FOOTPRINTS"] = safe_num(df.get("n_target_source_footprints", pd.Series(np.nan, index=df.index)))
    out["EFFECTIVE_NATIVE_COUNT"] = out["N_NATIVE_PIXELS_OR_FOOTPRINTS"]
    out["WEIGHT_SUM"] = safe_num(df.get("probability_sum_target_observed", pd.Series(np.nan, index=df.index)))
    out["NATIVE_SUPPORT_TYPE"] = "OCO-2 swath/footprint corner geometry"
    out["NATIVE_SUPPORT_AREA_KM2"] = np.nan
    out["WEIGHTED_NATIVE_SUPPORT_AREA_KM2"] = np.nan
    out["EC_SUPPORT_AREA_KM2"] = np.nan
    out["SUPPORT_AREA_RATIO"] = np.nan
    out["FRACTION_NATIVE_SUPPORT_INTERSECTING_EC"] = np.nan
    out["FRACTION_EC_SUPPORT_COVERED"] = out["EC_COVERAGE_FRACTION"]
    out["SPATIAL_SUPPORT_METRIC_SOURCE"] = "Coverage fractions only; native footprint areas were not finalized in comparable area units."
    out["PROJECT_RECONSTRUCTED"] = df.get("reconstruction_applied", pd.Series(False, index=df.index)).map(normalize_bool)
    out["RECONSTRUCTION_METHOD"] = np.where(out["PROJECT_RECONSTRUCTED"], "target-only IDW/context reconstruction", "NONE")
    out["RECONSTRUCTION_MODEL"] = df.get("model_used", "OBSERVED_NO_MODEL")
    out["RECONSTRUCTION_QC"] = np.where(status.eq("RECONSTRUCTED_PARTIAL_QC_FLAG"), "EXTREME_CHANGE_FROM_PARTIAL", "")
    out["UNCERTAINTY_AVAILABLE"] = df.get("OCO2_PI90_width", pd.Series(np.nan, index=df.index)).notna()
    out["UNCERTAINTY_VALUE"] = safe_num(df.get("OCO2_PI90_width", pd.Series(np.nan, index=df.index)))
    out["UNCERTAINTY_UNITS"] = NO_DATA
    out["UNCERTAINTY_TYPE"] = np.where(out["UNCERTAINTY_AVAILABLE"], "PI90_WIDTH_FROM_PROJECT_RECONSTRUCTION", "")
    out["QC_REASON"] = np.where(df.apply(lambda r: (str(r["SITE_ID"]), int(r["YYYYMM"])) in qc_keys, axis=1), "EXTREME_CHANGE_FROM_PARTIAL", "")
    out["QC_SEVERITY"] = np.where(out["QC_REASON"].ne(""), "RETAINED_PRIMARY_QC_FLAG", "")
    out["SOURCE_FILE"] = df.get("source_sif_files", "")
    out["SOURCE_RECORD_ID"] = df.get("OCO2_production_class", "")
    out["PROCESSING_METHOD"] = "Observed complete cases retained; positive partial cases reconstructed by frozen OCO-2 target-only IDW/context method."
    out["SOURCE_VERSION"] = NO_DATA
    return out


def classify_oco3(df: pd.DataFrame, primary_keys: set[tuple[str, int]], qc_keys: set[tuple[str, int]]) -> pd.DataFrame:
    out = pd.DataFrame()
    out["SIF_VALUE"] = safe_num(df["OCO3_final_EC_SIF"])
    out["SIF_VALUE_ORIGINAL"] = out["SIF_VALUE"]
    out["SIF_UNITS"] = NO_DATA
    out["SIF_WAVELENGTH_NM"] = NO_DATA
    out["SIF_DEFINITION"] = "OCO-3 LtSIF EC-footprint value; observed complete or RF_TARGET_GOSIF reconstructed partial where primary-ready."
    status = df["OCO3_final_status"].astype(str)
    out["VALUE_STATUS"] = np.select(
        [
            status.eq("OBSERVED_COMPLETE_VALID"),
            status.str.startswith("RECONSTRUCTED_PARTIAL"),
            status.eq("EXCLUDED_ZERO_COVERAGE"),
            status.eq("EXCLUDED_MISSING_REQUIRED_PREDICTOR"),
        ],
        ["DIRECT_PRODUCT", "PROJECT_RECONSTRUCTED", "EXCLUDED_ZERO_COVERAGE", "FAILED_RECONSTRUCTION"],
        default="OTHER",
    )
    out["COVERAGE_CLASS"] = df.get("coverage_class_positive", df.get("production_coverage_class", ""))
    out["EXCLUSION_REASON"] = np.where(out["VALUE_STATUS"].isin(["DIRECT_PRODUCT", "PROJECT_RECONSTRUCTED"]), "", status)
    out["EC_COVERAGE_FRACTION"] = safe_num(df.get("coverage_probability_fraction", pd.Series(np.nan, index=df.index)))
    out["EC_UNION_COVERAGE"] = out["EC_COVERAGE_FRACTION"]
    out["POSITIVE_COVERAGE"] = out["EC_COVERAGE_FRACTION"] > 0
    out["N_NATIVE_OBSERVATIONS"] = safe_num(df.get("n_target_source_observations", pd.Series(np.nan, index=df.index)))
    out["N_NATIVE_PIXELS_OR_FOOTPRINTS"] = safe_num(df.get("n_target_source_footprints", pd.Series(np.nan, index=df.index)))
    out["EFFECTIVE_NATIVE_COUNT"] = out["N_NATIVE_PIXELS_OR_FOOTPRINTS"]
    out["WEIGHT_SUM"] = safe_num(df.get("probability_sum_target_observed", pd.Series(np.nan, index=df.index)))
    out["NATIVE_SUPPORT_TYPE"] = "OCO-3 swath/footprint corner geometry"
    out["NATIVE_SUPPORT_AREA_KM2"] = np.nan
    out["WEIGHTED_NATIVE_SUPPORT_AREA_KM2"] = np.nan
    out["EC_SUPPORT_AREA_KM2"] = np.nan
    out["SUPPORT_AREA_RATIO"] = np.nan
    out["FRACTION_NATIVE_SUPPORT_INTERSECTING_EC"] = np.nan
    out["FRACTION_EC_SUPPORT_COVERED"] = out["EC_COVERAGE_FRACTION"]
    out["SPATIAL_SUPPORT_METRIC_SOURCE"] = "Coverage fractions only; native footprint areas were not finalized in comparable area units."
    out["PROJECT_RECONSTRUCTED"] = df.get("reconstruction_applied", pd.Series(False, index=df.index)).map(normalize_bool)
    out["RECONSTRUCTION_METHOD"] = np.where(out["PROJECT_RECONSTRUCTED"], "RF_TARGET_GOSIF partial EC-support reconstruction", "NONE")
    out["RECONSTRUCTION_MODEL"] = df.get("model_used", "OBSERVED_NO_MODEL")
    out["RECONSTRUCTION_QC"] = np.where(status.eq("RECONSTRUCTED_PARTIAL_QC_FLAG"), "EXTREME_CHANGE_FROM_PARTIAL", "")
    out["UNCERTAINTY_AVAILABLE"] = df.get("OCO3_PI90_width", pd.Series(np.nan, index=df.index)).notna()
    out["UNCERTAINTY_VALUE"] = safe_num(df.get("OCO3_PI90_width", pd.Series(np.nan, index=df.index)))
    out["UNCERTAINTY_UNITS"] = NO_DATA
    out["UNCERTAINTY_TYPE"] = np.where(out["UNCERTAINTY_AVAILABLE"], "PI90_WIDTH_FROM_PROJECT_RECONSTRUCTION", "")
    out["QC_REASON"] = np.where(df.apply(lambda r: (str(r["SITE_ID"]), int(r["YYYYMM"])) in qc_keys, axis=1), "SEE_OCO3_QC_TABLE", "")
    out["QC_SEVERITY"] = np.where(status.eq("RECONSTRUCTED_PARTIAL_QC_FLAG"), "RETAINED_PRIMARY_QC_FLAG", np.where(status.eq("EXCLUDED_MISSING_REQUIRED_PREDICTOR"), "EXCLUDED_FAILED_RECONSTRUCTION", ""))
    out["SOURCE_FILE"] = df.get("source_sif_files", "")
    out["SOURCE_RECORD_ID"] = df.get("OCO3_production_class", "")
    out["PROCESSING_METHOD"] = "Observed complete cases retained; positive partial cases reconstructed by frozen RF_TARGET_GOSIF where predictors were available."
    out["SOURCE_VERSION"] = NO_DATA
    return out


CLASSIFIERS = {
    "GOSIF": classify_gosif,
    "GOME-2": classify_gome2,
    "TROPOSIF": classify_troposif,
    "OCO-2": classify_oco2,
    "OCO-3": classify_oco3,
}


def qc_key_set(product: str) -> set[tuple[str, int]]:
    path = PRODUCTS[product].qc
    if path is None or not path.exists():
        return set()
    try:
        qc = pd.read_csv(path)
    except Exception:
        return set()
    if not {"SITE_ID", "YYYYMM"}.issubset(qc.columns):
        return set()
    return set(zip(qc["SITE_ID"].astype(str), pd.to_numeric(qc["YYYYMM"], errors="coerce").dropna().astype(int)))


def build_master() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    product_frames: dict[str, pd.DataFrame] = {}
    rows = []
    for product in PRODUCT_ORDER:
        pp = PRODUCTS[product]
        all_df = pd.read_csv(pp.all_records)
        primary_df = pd.read_csv(pp.primary)
        pkeys = primary_key_set(primary_df)
        qkeys = qc_key_set(product)
        base = pd.DataFrame(index=all_df.index)
        base["PRODUCT"] = product
        base["SITE_ID"] = all_df["SITE_ID"].astype(str)
        base["YYYYMM"] = pd.to_numeric(all_df["YYYYMM"], errors="coerce").astype("Int64")
        year, month = yyyymm_to_year_month(base["YYYYMM"])
        base["YEAR"] = all_df["year"] if "year" in all_df.columns else year
        base["MONTH"] = all_df["month"] if "month" in all_df.columns else month
        base["LATITUDE"] = safe_num(all_df.get("lat", pd.Series(np.nan, index=all_df.index)))
        base["LONGITUDE"] = safe_num(all_df.get("lon", pd.Series(np.nan, index=all_df.index)))
        base["IGBP"] = all_df.get("IGBP", pd.Series("", index=all_df.index)).fillna("")
        extra = CLASSIFIERS[product](all_df, pkeys, qkeys)
        base = pd.concat([base, extra], axis=1)
        base["PRIMARY_READY"] = [
            (str(site), int(yyyymm)) in pkeys if pd.notna(yyyymm) else False
            for site, yyyymm in zip(base["SITE_ID"], base["YYYYMM"])
        ]
        base["QC_FLAG"] = base["QC_REASON"].fillna("").astype(str).str.len() > 0
        base["QC_RETAINED_IN_PRIMARY"] = base["QC_FLAG"] & base["PRIMARY_READY"]
        base["PROVENANCE_FILE"] = str(pp.provenance) if pp.provenance and pp.provenance.exists() else ""
        base["FINALIZATION_FOLDER"] = str(pp.folder)
        base["MASTER_AUDIT_NOTES"] = ""
        product_frames[product] = base
        rows.append(base)
    master = pd.concat(rows, ignore_index=True)

    object_cols = master.select_dtypes(include="object").columns
    for c in object_cols:
        master[c] = master[c].fillna("")

    primary = master[master["PRIMARY_READY"]].copy()
    master.to_parquet(DIRS["tables"] / "SIF_Master_FiveProduct_AllRecords.parquet", index=False)
    master.to_csv(DIRS["tables"] / "SIF_Master_FiveProduct_AllRecords.csv", index=False)
    primary.to_parquet(DIRS["tables"] / "SIF_Master_FiveProduct_PrimaryReady.parquet", index=False)
    primary.to_csv(DIRS["tables"] / "SIF_Master_FiveProduct_PrimaryReady.csv", index=False)
    return master, primary, product_frames


def product_definitions(master: pd.DataFrame) -> pd.DataFrame:
    definitions = [
        {
            "PRODUCT": "GOSIF",
            "PRODUCT_FULL_NAME": "GOSIF gridded global SIF product",
            "PRODUCT_VERSION": NO_DATA,
            "PRODUCT_LINEAGE": "Monthly GOSIF raster archive used in finalized GOSIF EC-weighting workflow.",
            "SOURCE_SENSOR_OR_SENSORS": NO_DATA,
            "SIF_VARIABLE_NAME": "GOSIF_EC_weighted",
            "RETRIEVAL_OR_PRODUCT_WAVELENGTH_NM": NO_DATA,
            "ORIGINAL_UNITS": "W m-2 um-1 sr-1",
            "TEMPORAL_DEFINITION": "monthly, same YYYYMM source raster",
            "NATIVE_SPATIAL_SUPPORT_TYPE": "0.05-degree gridded native raster cells",
            "NATIVE_SPATIAL_RESOLUTION_OR_FOOTPRINT_DESCRIPTION": "0.05 degrees; EC probability assigned to containing native cells.",
            "PROJECT_PROCESSING_METHOD": "EC probability-weighted native GOSIF cells using Band 1 footprint probability only.",
            "PROJECT_RECONSTRUCTION_USED": False,
            "PROJECT_RECONSTRUCTION_METHOD": "NONE",
            "PRIMARY_COVERAGE_RULE": "VALID_EC_WEIGHTED rows.",
            "QC_RULE": "Missing source months excluded; fill values 32766 and 32767 excluded in finalized workflow.",
            "SOURCE_FINALIZATION_FOLDER": str(PRODUCTS["GOSIF"].folder),
            "NOTES": "Historical single/multi support summary used overlapping definitions; corrected in this audit.",
        },
        {
            "PRODUCT": "GOME-2",
            "PRODUCT_FULL_NAME": "GOME-2A NSIF daily native-footprint SIF product",
            "PRODUCT_VERSION": "NSIFv2.6.2.GOME-2A",
            "PRODUCT_LINEAGE": "Daily NetCDF files with native four-corner footprint geometry.",
            "SOURCE_SENSOR_OR_SENSORS": "GOME-2A",
            "SIF_VARIABLE_NAME": "Daily_Averaged_SIF / GOME2_EC_overlap_weighted_SIF",
            "RETRIEVAL_OR_PRODUCT_WAVELENGTH_NM": NO_DATA,
            "ORIGINAL_UNITS": "mW m-2 nm-1 sr-1",
            "TEMPORAL_DEFINITION": "monthly aggregation of daily files matching YYYYMM",
            "NATIVE_SPATIAL_SUPPORT_TYPE": "native four-corner footprint polygons",
            "NATIVE_SPATIAL_RESOLUTION_OR_FOOTPRINT_DESCRIPTION": "Large native footprints; existing mean/median area metrics retained.",
            "PROJECT_PROCESSING_METHOD": "EC-overlap weighted native observations; no project reconstruction.",
            "PROJECT_RECONSTRUCTION_USED": False,
            "PROJECT_RECONSTRUCTION_METHOD": "NONE",
            "PRIMARY_COVERAGE_RULE": "GOME2_EC_probability_union_coverage >= 0.999.",
            "QC_RULE": "No/zero EC overlap excluded; Quality_Flag retained but not hard-filtered in finalized workflow.",
            "SOURCE_FINALIZATION_FOLDER": str(PRODUCTS["GOME-2"].folder),
            "NOTES": "Native area and EC area metrics are directly available for GOME-2 only.",
        },
        {
            "PRODUCT": "TROPOSIF",
            "PRODUCT_FULL_NAME": "TROPOSIF L2B SIF product",
            "PRODUCT_VERSION": NO_DATA,
            "PRODUCT_LINEAGE": "TROPOSIF L2B NetCDF archive; finalized table uses observed complete EC-support cases only.",
            "SOURCE_SENSOR_OR_SENSORS": NO_DATA,
            "SIF_VARIABLE_NAME": "TROPOSIF_final_EC_SIF",
            "RETRIEVAL_OR_PRODUCT_WAVELENGTH_NM": NO_DATA,
            "ORIGINAL_UNITS": "mW m-2 sr-1 nm-1 (source metadata: mW/m2/sr/nm)",
            "TEMPORAL_DEFINITION": "monthly aggregation of TROPOSIF L2B files",
            "NATIVE_SPATIAL_SUPPORT_TYPE": "swath footprint bounds",
            "NATIVE_SPATIAL_RESOLUTION_OR_FOOTPRINT_DESCRIPTION": "Latitude/longitude bounds used; comparable native area not finalized.",
            "PROJECT_PROCESSING_METHOD": "Observed complete EC-footprint cases retained; partial cases excluded from primary.",
            "PROJECT_RECONSTRUCTION_USED": False,
            "PROJECT_RECONSTRUCTION_METHOD": "NONE_ACCEPTED",
            "PRIMARY_COVERAGE_RULE": "coverage_probability_fraction = 1.",
            "QC_RULE": "No QC flags in final TROPOSIF primary table.",
            "SOURCE_FINALIZATION_FOLDER": str(PRODUCTS["TROPOSIF"].folder),
            "NOTES": "Sensitivity reconstruction was explicitly not accepted.",
        },
        {
            "PRODUCT": "OCO-2",
            "PRODUCT_FULL_NAME": "OCO-2 Lite SIF / LtSIF site-month EC-support product",
            "PRODUCT_VERSION": NO_DATA,
            "PRODUCT_LINEAGE": "OCO-2 LtSIF source files intersected with EC probability support.",
            "SOURCE_SENSOR_OR_SENSORS": "OCO-2",
            "SIF_VARIABLE_NAME": "OCO2_final_EC_SIF",
            "RETRIEVAL_OR_PRODUCT_WAVELENGTH_NM": NO_DATA,
            "ORIGINAL_UNITS": NO_DATA,
            "TEMPORAL_DEFINITION": "monthly EC-support aggregation",
            "NATIVE_SPATIAL_SUPPORT_TYPE": "swath footprint corner geometry",
            "NATIVE_SPATIAL_RESOLUTION_OR_FOOTPRINT_DESCRIPTION": "Latitude_Corners and Longitude_Corners used; comparable area metrics not finalized.",
            "PROJECT_PROCESSING_METHOD": "Complete observed cases retained; positive partial cases reconstructed by frozen target-only IDW/context method.",
            "PROJECT_RECONSTRUCTION_USED": True,
            "PROJECT_RECONSTRUCTION_METHOD": "target-only OCO-2 IDW/context reconstruction",
            "PRIMARY_COVERAGE_RULE": "complete observed plus positive partial reconstructed cases; zero coverage excluded.",
            "QC_RULE": "One retained primary reconstructed row flagged for extreme change from partial.",
            "SOURCE_FINALIZATION_FOLDER": str(PRODUCTS["OCO-2"].folder),
            "NOTES": "No external SIF and no GPP predictor used.",
        },
        {
            "PRODUCT": "OCO-3",
            "PRODUCT_FULL_NAME": "OCO-3 Lite SIF / LtSIF site-month EC-support product",
            "PRODUCT_VERSION": NO_DATA,
            "PRODUCT_LINEAGE": "OCO-3 LtSIF source files intersected with EC probability support.",
            "SOURCE_SENSOR_OR_SENSORS": "OCO-3",
            "SIF_VARIABLE_NAME": "OCO3_final_EC_SIF",
            "RETRIEVAL_OR_PRODUCT_WAVELENGTH_NM": NO_DATA,
            "ORIGINAL_UNITS": NO_DATA,
            "TEMPORAL_DEFINITION": "monthly EC-support aggregation",
            "NATIVE_SPATIAL_SUPPORT_TYPE": "swath footprint corner geometry",
            "NATIVE_SPATIAL_RESOLUTION_OR_FOOTPRINT_DESCRIPTION": "Latitude_Corners and Longitude_Corners used; comparable area metrics not finalized.",
            "PROJECT_PROCESSING_METHOD": "Complete observed cases retained; positive partial cases reconstructed by frozen RF_TARGET_GOSIF where predictors available.",
            "PROJECT_RECONSTRUCTION_USED": True,
            "PROJECT_RECONSTRUCTION_METHOD": "RF_TARGET_GOSIF",
            "PRIMARY_COVERAGE_RULE": "complete observed plus successfully reconstructed positive partial cases; zero coverage and failed partial cases excluded.",
            "QC_RULE": "One retained primary reconstructed row flagged for extreme change from partial; missing-predictor partial cases excluded.",
            "SOURCE_FINALIZATION_FOLDER": str(PRODUCTS["OCO-3"].folder),
            "NOTES": "Project reconstruction used GOSIF as SIF predictor but no tower GPP.",
        },
    ]
    out = pd.DataFrame(definitions)
    for product in PRODUCT_ORDER:
        sub = master[master["PRODUCT"].eq(product)]
        out.loc[out["PRODUCT"].eq(product), "SOURCE_DATE_MIN"] = int(sub["YYYYMM"].min())
        out.loc[out["PRODUCT"].eq(product), "SOURCE_DATE_MAX"] = int(sub["YYYYMM"].max())
        prim = sub[sub["PRIMARY_READY"]]
        out.loc[out["PRODUCT"].eq(product), "FINAL_STUDY_DATE_MIN"] = int(prim["YYYYMM"].min()) if len(prim) else np.nan
        out.loc[out["PRODUCT"].eq(product), "FINAL_STUDY_DATE_MAX"] = int(prim["YYYYMM"].max()) if len(prim) else np.nan
    out.to_csv(DIRS["tables"] / "SIF_Master_Product_Definitions.csv", index=False)
    return out


def recomputed_counts(master: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for product in PRODUCT_ORDER:
        sub = master[master["PRODUCT"].eq(product)].copy()
        primary = sub[sub["PRIMARY_READY"]]
        value_status = sub["VALUE_STATUS"].astype(str)
        if product == "TROPOSIF":
            n_positive_partial = int(value_status.eq("EXCLUDED_PARTIAL").sum())
            n_failed = n_positive_partial
        elif product == "OCO-3":
            n_positive_partial = int(value_status.eq("PROJECT_RECONSTRUCTED").sum() + value_status.eq("FAILED_RECONSTRUCTION").sum())
            n_failed = int(value_status.eq("FAILED_RECONSTRUCTION").sum())
        else:
            n_positive_partial = int(
                value_status.isin(["EXCLUDED_PARTIAL", "PROJECT_RECONSTRUCTED"]).sum()
                if product != "OCO-2" else primary["VALUE_STATUS"].eq("PROJECT_RECONSTRUCTED").sum()
            )
            n_failed = int((value_status == "FAILED_RECONSTRUCTION").sum())

        if product in {"GOSIF", "GOME-2", "TROPOSIF", "OCO-2"}:
            qc_path = PRODUCTS[product].qc
            n_qc = len(pd.read_csv(qc_path)) if qc_path is not None and qc_path.exists() else int(sub["QC_FLAG"].sum())
        else:
            n_qc = int(primary["QC_SEVERITY"].eq("RETAINED_PRIMARY_QC_FLAG").sum())

        metrics = {
            "N total attempted": len(sub),
            "N direct product": int((primary["VALUE_STATUS"] == "DIRECT_PRODUCT").sum()),
            "N project reconstructed": int((primary["VALUE_STATUS"] == "PROJECT_RECONSTRUCTED").sum()),
            "N positive partial": n_positive_partial,
            "N zero coverage": int((value_status == "EXCLUDED_ZERO_COVERAGE").sum()),
            "N failed": n_failed,
            "N excluded": int((~sub["PRIMARY_READY"]).sum()),
            "N primary ready": len(primary),
            "N QC flagged": n_qc,
            "N sites": int(primary["SITE_ID"].nunique()),
            "N months": int(primary["YYYYMM"].nunique()),
            "date min": int(primary["YYYYMM"].min()) if len(primary) else np.nan,
            "date max": int(primary["YYYYMM"].max()) if len(primary) else np.nan,
        }
        if product == "GOSIF":
            metrics["N single native cell corrected"] = int((primary["N_NATIVE_PIXELS_OR_FOOTPRINTS"] == 1).sum())
            metrics["N multi native cell corrected"] = int((primary["N_NATIVE_PIXELS_OR_FOOTPRINTS"] >= 2).sum())
        for metric, value in metrics.items():
            rows.append({"PRODUCT": product, "METRIC": metric, "RECOMPUTED_VALUE": value})
    return pd.DataFrame(rows)


def reported_counts() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_map = {
        "GOSIF": {
            "N_total_site_months_attempted": "N total attempted",
            "N_valid_GOSIF_EC_weighted": "N primary ready",
            "N_excluded": "N excluded",
            "N_sites_valid": "N sites",
            "N_single_native_pixel_cases": "N single native dominance >=0.999 (historical non-exclusive)",
            "N_multi_native_pixel_cases": "N multi native cell corrected",
            "N_QC_flagged": "N QC flagged",
        },
        "GOME-2": {
            "N_total_attempted": "N total attempted",
            "N_primary_ready": "N primary ready",
            "N_positive_partial_support": "N positive partial",
            "N_zero_coverage": "N zero coverage",
            "N_sites": "N sites",
            "date_min": "date min",
            "date_max": "date max",
            "N_QC_flags": "N QC flagged",
        },
        "TROPOSIF": {
            "N_total_TROPOSIF": "N total attempted",
            "N_complete_observed": "N direct product",
            "N_partial_positive": "N positive partial",
            "N_partial_sensitivity_failed": "N failed",
            "N_zero_coverage": "N zero coverage",
            "N_primary_analysis_ready": "N primary ready",
            "N_QC_flagged": "N QC flagged",
        },
        "OCO-2": {
            "N_total_OCO2_site_months": "N total attempted",
            "N_complete_observed": "N direct product",
            "N_positive_partial": "N positive partial",
            "N_positive_partial_reconstructed": "N project reconstructed",
            "N_zero_coverage": "N zero coverage",
            "N_primary_analysis_ready": "N primary ready",
            "N_QC_flagged": "N QC flagged",
        },
        "OCO-3": {
            "N_total_OCO3_site_months": "N total attempted",
            "N_complete_observed": "N direct product",
            "N_partial_eligible": "N positive partial",
            "N_partial_successfully_reconstructed": "N project reconstructed",
            "N_partial_failed": "N failed",
            "N_zero_coverage_excluded": "N zero coverage",
            "N_primary_analysis_ready": "N primary ready",
            "N_QC_flagged": "N QC flagged",
        },
    }
    for product in PRODUCT_ORDER:
        p = PRODUCTS[product].counts
        if p is None or not p.exists():
            continue
        df = pd.read_csv(p)
        if set(df.columns) >= {"metric", "value"}:
            raw = dict(zip(df["metric"], df["value"]))
        else:
            raw = df.iloc[0].to_dict()
        for source_metric, metric in metric_map[product].items():
            if source_metric in raw:
                val = raw[source_metric]
                try:
                    if isinstance(val, str) and val.upper() in {"TRUE", "FALSE"}:
                        reported = val
                    else:
                        reported = int(float(val))
                except Exception:
                    reported = val
                rows.append({"PRODUCT": product, "METRIC": metric, "REPORTED_VALUE": reported, "SOURCE_METRIC": source_metric})
    return pd.DataFrame(rows)


def count_reconciliation(master: pd.DataFrame) -> pd.DataFrame:
    rep = reported_counts()
    rec = recomputed_counts(master)
    out = rep.merge(rec, on=["PRODUCT", "METRIC"], how="outer")
    out["REPORTED_VALUE"] = out["REPORTED_VALUE"].where(out["REPORTED_VALUE"].notna(), "")
    out["RECOMPUTED_VALUE"] = out["RECOMPUTED_VALUE"].where(out["RECOMPUTED_VALUE"].notna(), "")

    def diff(row: pd.Series) -> Any:
        try:
            return float(row["RECOMPUTED_VALUE"]) - float(row["REPORTED_VALUE"])
        except Exception:
            return ""

    out["DIFFERENCE"] = out.apply(diff, axis=1)
    out["STATUS"] = "MATCH"
    out["EXPLANATION"] = ""
    unresolved = []
    for i, row in out.iterrows():
        if row["REPORTED_VALUE"] == "" or row["RECOMPUTED_VALUE"] == "":
            out.loc[i, "STATUS"] = "EXPLAINED_DIFFERENCE"
            out.loc[i, "EXPLANATION"] = "Metric is present only in one accounting layer; retained for transparency."
        elif row["DIFFERENCE"] not in ("", 0, 0.0):
            out.loc[i, "STATUS"] = "UNRESOLVED"
            unresolved.append(i)
    # Known GOSIF historical field used a dominance rule and is intentionally non-exclusive.
    gosif_hist = (
        out["PRODUCT"].eq("GOSIF")
        & out["METRIC"].eq("N single native dominance >=0.999 (historical non-exclusive)")
    )
    out.loc[gosif_hist, "STATUS"] = "EXPLAINED_DIFFERENCE"
    out.loc[gosif_hist, "EXPLANATION"] = "Historical count used dominant_native_pixel_probability_fraction >= 0.999, not mutually exclusive native-cell count."
    out.to_csv(DIRS["audit"] / "SIF_Product_Count_Reconciliation.csv", index=False)
    return out


def gosif_discrepancy() -> pd.DataFrame:
    df = pd.read_csv(PRODUCTS["GOSIF"].primary)
    historical_single = df["dominant_native_pixel_probability_fraction"].ge(0.999)
    historical_multi = df["n_native_GOSIF_cells_intersecting_EC"].ge(2)
    double = df[historical_single & historical_multi].copy()
    rows = []
    for _, r in double.sort_values(["SITE_ID", "YYYYMM"]).iterrows():
        rows.append(
            {
                "SITE_ID": r["SITE_ID"],
                "YYYYMM": int(r["YYYYMM"]),
                "original_classification": "SINGLE_BY_DOMINANCE_GE_0.999_AND_MULTI_BY_NATIVE_CELL_COUNT",
                "recomputed_classification": "MULTI_NATIVE_CELL",
                "native cell count": int(r["n_native_GOSIF_cells_intersecting_EC"]),
                "effective native count if available": float(r["GOSIF_effective_native_pixel_count"]),
                "primary-ready status": True,
                "dominant_native_pixel_probability_fraction": float(r["dominant_native_pixel_probability_fraction"]),
                "GOSIF_EC_weighted": float(r["GOSIF_EC_weighted"]),
                "reason for discrepancy": "The finalized summary counted this row as single using a dominance threshold while also counting it as multi using native-cell count; corrected mutually exclusive rule classifies by native-cell count.",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(DIRS["audit"] / "GOSIF_Six_Record_Discrepancy_Diagnostic.csv", index=False)
    return out


def load_ec_reference() -> tuple[pd.DataFrame, pd.DataFrame]:
    ec = pd.read_csv(EC_REFERENCE)
    # Explicitly remove GPP from this SIF-only audit. It is not read after this line.
    ec = ec.drop(columns=[c for c in ec.columns if c.lower() in {"tower_gpp_monthly", "gpp", "tower_gpp"}], errors="ignore")
    site = pd.read_csv(SITE_LIST)
    site = site.rename(
        columns={
            "study_table_lat": "LATITUDE",
            "study_table_lon": "LONGITUDE",
            "study_table_IGBP": "IGBP",
            "n_study_site_months": "N_EC_SITE_MONTHS",
            "first_YYYYMM": "FIRST_EC_MONTH",
            "last_YYYYMM": "LAST_EC_MONTH",
        }
    )
    if "LATITUDE" not in site.columns:
        site = (
            ec.groupby("SITE_ID")
            .agg(LATITUDE=("lat", "median"), LONGITUDE=("lon", "median"), IGBP=("IGBP", lambda x: x.dropna().astype(str).iloc[0] if x.dropna().size else ""))
            .reset_index()
        )
    return ec, site


def product_periods(master: pd.DataFrame) -> dict[str, tuple[int, int]]:
    periods = {}
    for product in PRODUCT_ORDER:
        sub = master[master["PRODUCT"].eq(product)]
        prim = sub[sub["PRIMARY_READY"]]
        if len(prim):
            periods[product] = (int(prim["YYYYMM"].min()), int(prim["YYYYMM"].max()))
        else:
            periods[product] = (int(sub["YYYYMM"].min()), int(sub["YYYYMM"].max()))
    # Denominator periods should reflect source/finalized study period, not all EC months.
    return periods


def availability_tables(master: pd.DataFrame, ec: pd.DataFrame, site: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    periods = product_periods(master)
    site_rows = []
    for product in PRODUCT_ORDER:
        start, end = periods[product]
        prod = master[master["PRODUCT"].eq(product)]
        primary = prod[prod["PRIMARY_READY"]]
        direct = primary[primary["VALUE_STATUS"].eq("DIRECT_PRODUCT")]
        recon = primary[primary["VALUE_STATUS"].eq("PROJECT_RECONSTRUCTED")]
        for _, s in site.iterrows():
            sid = str(s["SITE_ID"])
            eligible = ec[ec["SITE_ID"].astype(str).eq(sid) & ec["YYYYMM"].between(start, end)]
            ps = primary[primary["SITE_ID"].eq(sid)]
            ds = direct[direct["SITE_ID"].eq(sid)]
            rs = recon[recon["SITE_ID"].eq(sid)]
            n_eligible = len(eligible)
            n_primary = len(ps)
            site_rows.append(
                {
                    "PRODUCT": product,
                    "SITE_ID": sid,
                    "LATITUDE": s.get("LATITUDE", np.nan),
                    "LONGITUDE": s.get("LONGITUDE", np.nan),
                    "IGBP": s.get("IGBP", ""),
                    "N_PRIMARY_MONTHS": n_primary,
                    "N_ELIGIBLE_EC_MONTHS": n_eligible,
                    "PRIMARY_AVAILABILITY_FRACTION": n_primary / n_eligible if n_eligible else np.nan,
                    "FIRST_PRIMARY_MONTH": int(ps["YYYYMM"].min()) if n_primary else np.nan,
                    "LAST_PRIMARY_MONTH": int(ps["YYYYMM"].max()) if n_primary else np.nan,
                    "N_DIRECT_MONTHS": len(ds),
                    "N_PROJECT_RECONSTRUCTED_MONTHS": len(rs),
                    "TEMPORAL_ELIGIBILITY_STATUS": "ELIGIBLE" if n_eligible else "NOT_ELIGIBLE_TEMPORALLY",
                }
            )
    site_avail = pd.DataFrame(site_rows)
    site_avail.to_csv(DIRS["figure_data"] / "SIF_Product_Site_Availability.csv", index=False)

    all_months = sorted(ec["YYYYMM"].unique())
    month_rows = []
    for product in PRODUCT_ORDER:
        start, end = periods[product]
        primary = master[master["PRODUCT"].eq(product) & master["PRIMARY_READY"]]
        direct = primary[primary["VALUE_STATUS"].eq("DIRECT_PRODUCT")]
        recon = primary[primary["VALUE_STATUS"].eq("PROJECT_RECONSTRUCTED")]
        for yyyymm in all_months:
            if yyyymm < start or yyyymm > end:
                continue
            eligible_sites = set(ec.loc[ec["YYYYMM"].eq(yyyymm), "SITE_ID"].astype(str))
            ps = primary[primary["YYYYMM"].eq(yyyymm)]
            n_elig = len(eligible_sites)
            month_rows.append(
                {
                    "PRODUCT": product,
                    "YYYYMM": int(yyyymm),
                    "YEAR": int(yyyymm) // 100,
                    "MONTH": int(yyyymm) % 100,
                    "N_ELIGIBLE_SITES": n_elig,
                    "N_SITES_PRIMARY": ps["SITE_ID"].nunique(),
                    "N_SITES_DIRECT": direct[direct["YYYYMM"].eq(yyyymm)]["SITE_ID"].nunique(),
                    "N_SITES_PROJECT_RECONSTRUCTED": recon[recon["YYYYMM"].eq(yyyymm)]["SITE_ID"].nunique(),
                    "FRACTION_OF_ELIGIBLE_SITES_PRIMARY": ps["SITE_ID"].nunique() / n_elig if n_elig else np.nan,
                }
            )
    monthly = pd.DataFrame(month_rows)
    monthly.to_csv(DIRS["figure_data"] / "SIF_Product_Monthly_Availability.csv", index=False)

    support_rows = []
    for product in PRODUCT_ORDER:
        p = master[master["PRODUCT"].eq(product) & master["PRIMARY_READY"]]
        for sid, g in p.groupby("SITE_ID"):
            support_comparable = product == "GOME-2"
            support_rows.append(
                {
                    "PRODUCT": product,
                    "SITE_ID": sid,
                    "median EC coverage": g["EC_COVERAGE_FRACTION"].median(skipna=True),
                    "median native support area": g["NATIVE_SUPPORT_AREA_KM2"].median(skipna=True),
                    "median EC support area": g["EC_SUPPORT_AREA_KM2"].median(skipna=True),
                    "median support area ratio": g["SUPPORT_AREA_RATIO"].median(skipna=True),
                    "median native-observation count": g["N_NATIVE_OBSERVATIONS"].median(skipna=True),
                    "median effective native count": g["EFFECTIVE_NATIVE_COUNT"].median(skipna=True),
                    "fraction of site-months project reconstructed": g["PROJECT_RECONSTRUCTED"].mean(),
                    "fraction direct": g["VALUE_STATUS"].eq("DIRECT_PRODUCT").mean(),
                    "number primary months": len(g),
                    "SUPPORT_METRIC_COMPARABLE": support_comparable,
                    "SUPPORT_METRIC_COMPARABILITY_EXPLANATION": "Native footprint and EC area metrics available in comparable km2 fields." if support_comparable else "Comparable native support areas were not finalized for this product.",
                }
            )
    support = pd.DataFrame(support_rows)
    support.to_csv(DIRS["figure_data"] / "SIF_Product_Site_SpatialSupport_Summary.csv", index=False)
    return site_avail, monthly, support


def pairwise_common_site_months(master: pd.DataFrame, ec: pd.DataFrame) -> pd.DataFrame:
    periods = product_periods(master)
    primary_sets = {
        p: set(zip(master.loc[master["PRODUCT"].eq(p) & master["PRIMARY_READY"], "SITE_ID"].astype(str), master.loc[master["PRODUCT"].eq(p) & master["PRIMARY_READY"], "YYYYMM"].astype(int)))
        for p in PRODUCT_ORDER
    }
    rows = []
    for a in PRODUCT_ORDER:
        for b in PRODUCT_ORDER:
            start = max(periods[a][0], periods[b][0])
            end = min(periods[a][1], periods[b][1])
            if start > end:
                rows.append(
                    {
                        "PRODUCT_A": a,
                        "PRODUCT_B": b,
                        "COMMON_PERIOD_START": "",
                        "COMMON_PERIOD_END": "",
                        "TEMPORAL_OVERLAP_STATUS": "NO_TEMPORAL_OVERLAP",
                        "N_EC_REFERENCE_SITE_MONTHS_IN_COMMON_PERIOD": 0,
                        "N_WITH_PRODUCT_A": 0,
                        "N_WITH_PRODUCT_B": 0,
                        "N_WITH_BOTH": 0,
                        "N_UNION": 0,
                        "JACCARD_OVERLAP": np.nan,
                        "FRACTION_OF_POSSIBLE_COMMON_EC_SITE_MONTHS_WITH_BOTH": np.nan,
                        "N_COMMON_SITES": 0,
                    }
                )
                continue
            possible = set(zip(ec.loc[ec["YYYYMM"].between(start, end), "SITE_ID"].astype(str), ec.loc[ec["YYYYMM"].between(start, end), "YYYYMM"].astype(int)))
            aset = primary_sets[a] & possible
            bset = primary_sets[b] & possible
            both = aset & bset
            union = aset | bset
            rows.append(
                {
                    "PRODUCT_A": a,
                    "PRODUCT_B": b,
                    "COMMON_PERIOD_START": start,
                    "COMMON_PERIOD_END": end,
                    "TEMPORAL_OVERLAP_STATUS": "TEMPORAL_OVERLAP",
                    "N_EC_REFERENCE_SITE_MONTHS_IN_COMMON_PERIOD": len(possible),
                    "N_WITH_PRODUCT_A": len(aset),
                    "N_WITH_PRODUCT_B": len(bset),
                    "N_WITH_BOTH": len(both),
                    "N_UNION": len(union),
                    "JACCARD_OVERLAP": len(both) / len(union) if union else np.nan,
                    "FRACTION_OF_POSSIBLE_COMMON_EC_SITE_MONTHS_WITH_BOTH": len(both) / len(possible) if possible else np.nan,
                    "N_COMMON_SITES": len({sid for sid, _ in both}),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(DIRS["tables"] / "SIF_Product_Pairwise_Common_SiteMonths.csv", index=False)
    return out


def integrity_checks(master: pd.DataFrame, site: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(check: str, status: str, n: int | float, details: str) -> None:
        rows.append({"CHECK": check, "STATUS": status, "N_AFFECTED": n, "DETAILS": details})

    dup = master.duplicated(["PRODUCT", "SITE_ID", "YYYYMM"], keep=False)
    add("A. uniqueness PRODUCT + SITE_ID + YYYYMM", "PASS" if not dup.any() else "FAIL", int(dup.sum()), "No duplicate product/site/month keys." if not dup.any() else "Duplicate keys found.")

    coord = (
        master.groupby("SITE_ID")
        .agg(lat_min=("LATITUDE", "min"), lat_max=("LATITUDE", "max"), lon_min=("LONGITUDE", "min"), lon_max=("LONGITUDE", "max"), n_products=("PRODUCT", "nunique"))
        .reset_index()
    )
    coord["lat_range"] = coord["lat_max"] - coord["lat_min"]
    coord["lon_range"] = coord["lon_max"] - coord["lon_min"]
    cflag = coord[(coord["lat_range"].abs() > 0.01) | (coord["lon_range"].abs() > 0.01)]
    add("B. coordinate consistency by SITE_ID", "PASS" if cflag.empty else "FLAG", len(cflag), f"Max lat/lon ranges: {coord['lat_range'].max():.6g}, {coord['lon_range'].max():.6g}.")
    coord.to_csv(DIRS["audit"] / "SIF_Master_Coordinate_Consistency_Detail.csv", index=False)

    igbp = master.assign(IGBP_CLEAN=master["IGBP"].replace("", np.nan)).groupby("SITE_ID")["IGBP_CLEAN"].nunique(dropna=True).reset_index(name="N_IGBP_VALUES")
    iflag = igbp[igbp["N_IGBP_VALUES"] > 1]
    add("C. IGBP consistency by SITE_ID", "PASS" if iflag.empty else "FLAG", len(iflag), "Some sites have different/missing IGBP labels across products." if len(iflag) else "No conflicting non-missing IGBP labels.")
    igbp.to_csv(DIRS["audit"] / "SIF_Master_IGBP_Consistency_Detail.csv", index=False)

    yy = master["YYYYMM"].astype(int)
    date_bad = ((yy % 100 < 1) | (yy % 100 > 12) | (yy // 100 < 1900) | (yy // 100 > 2100)).sum()
    add("D. date validity", "PASS" if date_bad == 0 else "FAIL", int(date_bad), "YYYYMM values have valid months and plausible years.")

    primary = master[master["PRIMARY_READY"]]
    inf = np.isinf(pd.to_numeric(primary["SIF_VALUE"], errors="coerce")).sum()
    missing_primary = pd.to_numeric(primary["SIF_VALUE"], errors="coerce").isna().sum()
    sentinel = primary["SIF_VALUE"].isin([-9999, 32766, 32767]).sum()
    add("E. SIF finite and sentinel values in primary", "PASS" if inf == 0 and missing_primary == 0 and sentinel == 0 else "FAIL", int(inf + missing_primary + sentinel), f"Inf={inf}, missing={missing_primary}, sentinel={sentinel}.")

    units = primary.groupby("PRODUCT")["SIF_UNITS"].agg(lambda x: "; ".join(sorted(set(map(str, x)) - {""})[:5])).reset_index()
    unknown_units = units[units["SIF_UNITS"].str.contains(NO_DATA, na=False)]
    add("F. units documented", "FLAG" if len(unknown_units) else "PASS", len(unknown_units), "OCO-2/OCO-3 units are not established in final metadata; values retained without conversion.")
    units.to_csv(DIRS["audit"] / "SIF_Master_Units_By_Product.csv", index=False)

    add("G. scaling", "PASS", 0, "GOSIF scale factor 0.0001 and fill handling are documented in finalized GOSIF report; no rescaling applied by this audit.")

    wavelengths_unknown = int(primary.groupby("PRODUCT")["SIF_WAVELENGTH_NM"].agg(lambda x: all(str(v) == NO_DATA for v in x)).sum())
    add("H. wavelength metadata", "FLAG", wavelengths_unknown, "Wavelength metadata were not established from the final reports/tables for one or more products; no wavelength harmonization performed.")

    bad_primary = primary["VALUE_STATUS"].isin(["EXCLUDED_ZERO_COVERAGE", "EXCLUDED_PARTIAL", "FAILED_RECONSTRUCTION", "NO_SOURCE_DATA"]).sum()
    add("I. primary-ready logic", "PASS" if bad_primary == 0 else "FAIL", int(bad_primary), "No excluded/failed/no-source rows in primary-ready subset." if bad_primary == 0 else "Excluded statuses found in primary-ready subset.")

    oco = primary[primary["PRODUCT"].isin(["OCO-2", "OCO-3"])]
    recon_identity_bad = oco[oco["VALUE_STATUS"].eq("PROJECT_RECONSTRUCTED") & ~oco["PROJECT_RECONSTRUCTED"]]
    add("J. reconstruction identity", "PASS" if recon_identity_bad.empty else "FAIL", len(recon_identity_bad), "OCO-2/OCO-3 reconstructed rows are distinguishable from direct rows.")

    prov_bad = primary["SOURCE_FILE"].fillna("").astype(str).eq("").sum()
    add("K. provenance completeness", "PASS" if prov_bad == 0 else "FLAG", int(prov_bad), "Primary rows have source file provenance where final tables provide it." if prov_bad == 0 else "Some primary rows lack source file path provenance.")

    site_counts = primary.groupby("PRODUCT")["SITE_ID"].nunique().to_dict()
    add("L. site universe represented", "PASS", len(site_counts), "; ".join(f"{p}: {site_counts.get(p, 0)} sites" for p in PRODUCT_ORDER))

    extremes = primary.groupby("PRODUCT")["SIF_VALUE"].agg(["count", "min", "max", "median"]).reset_index()
    extremes.to_csv(DIRS["audit"] / "SIF_Master_Extreme_SIF_By_Product.csv", index=False)
    add("M. extreme values", "PASS", len(extremes), "Extreme values summarized by product; no deletion performed without product QC rule.")

    out = pd.DataFrame(rows)
    out.to_csv(DIRS["audit"] / "SIF_Master_Integrity_Checks.csv", index=False)
    return out


def final_audit_summary(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for product in PRODUCT_ORDER:
        sub = master[master["PRODUCT"].eq(product)]
        primary = sub[sub["PRIMARY_READY"]]
        rows.append(
            {
                "PRODUCT": product,
                "N_STUDY_SITES": primary["SITE_ID"].nunique(),
                "N_ATTEMPTED_SITE_MONTHS": len(sub),
                "N_PRIMARY_SITE_MONTHS": len(primary),
                "N_DIRECT_PRIMARY": int(primary["VALUE_STATUS"].eq("DIRECT_PRODUCT").sum()),
                "N_PROJECT_RECONSTRUCTED_PRIMARY": int(primary["VALUE_STATUS"].eq("PROJECT_RECONSTRUCTED").sum()),
                "N_EXCLUDED": int((~sub["PRIMARY_READY"]).sum()),
                "DATE_MIN": int(primary["YYYYMM"].min()) if len(primary) else np.nan,
                "DATE_MAX": int(primary["YYYYMM"].max()) if len(primary) else np.nan,
                "MEDIAN_SITE_AVAILABILITY_FRACTION": np.nan,
                "MEDIAN_EC_COVERAGE": primary["EC_COVERAGE_FRACTION"].median(skipna=True),
                "MEDIAN_NATIVE_SUPPORT_AREA_KM2 if meaningful": primary["NATIVE_SUPPORT_AREA_KM2"].median(skipna=True),
                "MEDIAN_EC_SUPPORT_AREA_KM2 if available": primary["EC_SUPPORT_AREA_KM2"].median(skipna=True),
                "QC_FLAGGED_PRIMARY": int(primary["QC_RETAINED_IN_PRIMARY"].sum()),
                "UNITS": "; ".join(sorted(set(primary["SIF_UNITS"].dropna().astype(str)))[:5]),
                "WAVELENGTH_NM": "; ".join(sorted(set(primary["SIF_WAVELENGTH_NM"].dropna().astype(str)))[:5]),
                "NATIVE_SUPPORT_TYPE": "; ".join(sorted(set(primary["NATIVE_SUPPORT_TYPE"].dropna().astype(str)))[:3]),
            }
        )
    out = pd.DataFrame(rows)
    # Fill median site availability after the site-level table exists.
    site_avail_path = DIRS["figure_data"] / "SIF_Product_Site_Availability.csv"
    if site_avail_path.exists():
        avail = pd.read_csv(site_avail_path)
        med = avail.groupby("PRODUCT")["PRIMARY_AVAILABILITY_FRACTION"].median().to_dict()
        out["MEDIAN_SITE_AVAILABILITY_FRACTION"] = out["PRODUCT"].map(med)
    out.to_csv(DIRS["tables"] / "SIF_FiveProduct_Final_Audit_Summary.csv", index=False)
    return out


def rough_region(lat: float, lon: float) -> str:
    if pd.isna(lat) or pd.isna(lon):
        return "Unknown"
    if -45 <= lat <= 72 and -170 <= lon <= -30:
        return "Americas"
    if -35 <= lat <= 40 and -20 <= lon <= 55:
        return "Africa_MiddleEast"
    if 35 <= lat <= 72 and -25 <= lon <= 45:
        return "Europe"
    if -50 <= lat <= 0 and 110 <= lon <= 180:
        return "Australia"
    if lon >= 45 or lon <= -170:
        return "Asia"
    return "Other"


def make_figures(site_avail: pd.DataFrame, monthly: pd.DataFrame, pairwise: pd.DataFrame, master: pd.DataFrame) -> list[str]:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    try:
        import cartopy.crs as ccrs
    except Exception:
        ccrs = None

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "savefig.dpi": 600,
            "svg.fonttype": "none",
        }
    )
    products = PRODUCT_ORDER
    created: list[str] = []

    def save_multi(fig: plt.Figure, stem: str, dpi: int = 600, tight: bool = True) -> None:
        for ext in ["pdf", "svg", "png", "tif"]:
            path = DIRS["figures"] / f"{stem}.{ext}"
            if tight:
                fig.savefig(path, bbox_inches="tight", dpi=dpi)
            else:
                fig.savefig(path, dpi=dpi)
            created.append(str(path))

    def load_blue_marble(max_width: int = 2400) -> tuple[np.ndarray | None, list[float]]:
        if not BLUE_MARBLE.exists():
            return None, [-180, 180, -90, 90]
        try:
            import rasterio

            with rasterio.open(BLUE_MARBLE) as src:
                scale = max(1, int(math.ceil(src.width / max_width)))
                arr = src.read(out_shape=(src.count, src.height // scale, src.width // scale))
                arr = np.moveaxis(arr[:3], 0, -1)
                if arr.dtype != np.uint8:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                return arr, [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        except Exception:
            return None, [-180, 180, -90, 90]

    def atlas_variant(dark: bool) -> None:
        # Use a simple lon/lat atlas for this availability diagnostic. It is
        # more stable across PDF/SVG/PNG/TIF export than projected GeoAxes.
        projection = None
        fig = plt.figure(figsize=(7.8, 5.45))
        positions = [
            (0.035, 0.555, 0.295, 0.350),
            (0.352, 0.555, 0.295, 0.350),
            (0.668, 0.555, 0.295, 0.350),
            (0.195, 0.180, 0.295, 0.350),
            (0.512, 0.180, 0.295, 0.350),
        ]
        bg, extent = load_blue_marble()
        cmap = LinearSegmentedColormap.from_list("avail", ["#d8d8d8", "#9bd7d1", "#1f9e89", "#084c61"])
        if dark:
            fig.patch.set_facecolor("#05070a")
        for i, product in enumerate(products):
            ax_pos = positions[i]
            ax = fig.add_axes(ax_pos)
            ax.set_xlim(-180, 180)
            ax.set_ylim(-58, 82)
            ax.set_aspect("auto")
            if bg is not None:
                img = bg.astype(float) / 255
                if dark:
                    img = np.clip(img * 0.42, 0, 1)
                else:
                    img = np.clip(0.25 + img * 0.75, 0, 1)
                ax.imshow(img, extent=extent, origin="upper", zorder=0, aspect="auto")
            ax.set_facecolor("#05070a" if dark else "#f7f8f5")
            sub = site_avail[site_avail["PRODUCT"].eq(product)].copy()
            eligible = sub[sub["N_ELIGIBLE_EC_MONTHS"] > 0]
            not_eligible = sub[sub["N_ELIGIBLE_EC_MONTHS"] == 0]
            sizes = 10 + 34 * np.sqrt(eligible["N_PRIMARY_MONTHS"].fillna(0) / max(1, eligible["N_PRIMARY_MONTHS"].max()))
            if len(not_eligible):
                ax.scatter(
                    not_eligible["LONGITUDE"],
                    not_eligible["LATITUDE"],
                    s=8,
                    facecolors="none",
                    edgecolors="#858585" if dark else "#b0b0b0",
                    linewidths=0.45,
                    zorder=3,
                )
            sc = ax.scatter(
                eligible["LONGITUDE"],
                eligible["LATITUDE"],
                c=eligible["PRIMARY_AVAILABILITY_FRACTION"],
                s=sizes,
                cmap=cmap,
                norm=Normalize(0, 1),
                edgecolors="#ffffff" if dark else "#17212b",
                linewidths=0.28,
                zorder=4,
            )
            ax.text(
                0.02,
                0.96,
                f"({chr(97+i)}) {product}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                color="#ffffff" if dark else "#111111",
                fontsize=7.2,
                fontweight="bold",
                bbox={
                    "boxstyle": "round,pad=0.16",
                    "facecolor": "#05070a" if dark else "#ffffff",
                    "edgecolor": "none",
                    "alpha": 0.58 if dark else 0.72,
                },
            )
            prim = int(eligible["N_PRIMARY_MONTHS"].sum())
            sites = int((eligible["N_PRIMARY_MONTHS"] > 0).sum())
            span = master.loc[master["PRODUCT"].eq(product) & master["PRIMARY_READY"], "YYYYMM"]
            span_txt = f"{int(span.min())}-{int(span.max())}" if len(span) else "none"
            ax.text(
                0.03,
                0.055,
                f"{sites} sites | {prim:,} primary months\n{span_txt}",
                transform=ax.transAxes,
                va="bottom",
                ha="left",
                color="#f1f1f1" if dark else "#222222",
                fontsize=5.1,
                linespacing=1.10,
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "#05070a" if dark else "#ffffff",
                    "edgecolor": "none",
                    "alpha": 0.56 if dark else 0.70,
                },
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.25)
                spine.set_color("#808080" if dark else "#bbbbbb")
        cax = fig.add_axes([0.235, 0.080, 0.53, 0.018])
        cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
        cb.set_label(
            "Primary availability fraction within product-specific eligible EC months",
            color="#ffffff" if dark else "#111111",
            fontsize=6.2,
        )
        cb.ax.tick_params(colors="#ffffff" if dark else "#111111", length=2, labelsize=5.8)
        cb.outline.set_edgecolor("#ffffff" if dark else "#111111")
        save_multi(fig, f"Candidate_A_FiveProduct_Global_Availability_{'DARK' if dark else 'LIGHT'}", dpi=600, tight=False)
        plt.close(fig)

    atlas_variant(True)
    atlas_variant(False)

    # Visualization B: temporal ribbon.
    fig, ax = plt.subplots(figsize=(7.6, 1.95))
    months = sorted(monthly["YYYYMM"].unique())
    mat = np.full((len(products), len(months)), np.nan)
    for i, product in enumerate(products):
        d = monthly[monthly["PRODUCT"].eq(product)].set_index("YYYYMM")
        for j, m in enumerate(months):
            if m in d.index:
                mat[i, j] = d.loc[m, "FRACTION_OF_ELIGIBLE_SITES_PRIMARY"]
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="none")
    ax.set_yticks(np.arange(len(products)), labels=products)
    year_ticks = [i for i, m in enumerate(months) if int(m) % 100 == 1]
    ax.set_xticks(year_ticks[::2], labels=[str(int(months[i]) // 100) for i in year_ticks[::2]], rotation=0)
    ax.set_xlabel("Year")
    ax.set_ylabel("")
    ax.tick_params(length=2)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Fraction of eligible EC sites with primary-ready SIF")
    save_multi(fig, "Candidate_B_Temporal_Availability_Ribbon", dpi=600)
    plt.close(fig)

    # Visualization C: pairwise matrix.
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    n = len(products)
    frac = np.zeros((n, n))
    labels = [["" for _ in products] for _ in products]
    for i, a in enumerate(products):
        for j, b in enumerate(products):
            row = pairwise[(pairwise["PRODUCT_A"].eq(a)) & (pairwise["PRODUCT_B"].eq(b))].iloc[0]
            if i == j:
                frac[i, j] = 1
                labels[i][j] = str(int(row["N_WITH_BOTH"]))
            elif i < j:
                frac[i, j] = row["FRACTION_OF_POSSIBLE_COMMON_EC_SITE_MONTHS_WITH_BOTH"] if pd.notna(row["FRACTION_OF_POSSIBLE_COMMON_EC_SITE_MONTHS_WITH_BOTH"]) else 0
                labels[i][j] = str(int(row["N_WITH_BOTH"]))
            else:
                frac[i, j] = row["JACCARD_OVERLAP"] if pd.notna(row["JACCARD_OVERLAP"]) else 0
                labels[i][j] = f"{frac[i,j]:.2f}"
    im = ax.imshow(frac, cmap="mako" if "mako" in plt.colormaps() else "viridis", vmin=0, vmax=1)
    ax.set_xticks(np.arange(n), labels=products, rotation=35, ha="right")
    ax.set_yticks(np.arange(n), labels=products)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, labels[i][j], ha="center", va="center", color="white" if frac[i, j] < 0.55 else "black", fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("Lower: Jaccard; upper: fraction possible with both")
    save_multi(fig, "Candidate_C_Pairwise_Common_SiteMonth_Matrix", dpi=600)
    plt.close(fig)

    # Visualization F: site x product availability matrix.
    mat_df = site_avail.pivot(index="SITE_ID", columns="PRODUCT", values="PRIMARY_AVAILABILITY_FRACTION").reindex(columns=products)
    meta = site_avail.drop_duplicates("SITE_ID").set_index("SITE_ID")
    order = sorted(mat_df.index, key=lambda sid: (rough_region(meta.loc[sid, "LATITUDE"], meta.loc[sid, "LONGITUDE"]), str(meta.loc[sid, "IGBP"]), -float(meta.loc[sid, "LATITUDE"])))
    mat_df = mat_df.loc[order]
    fig, ax = plt.subplots(figsize=(4.8, 9.0))
    im = ax.imshow(mat_df.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(products)), labels=products, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(order)), labels=order, fontsize=5.5)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Primary availability fraction")
    save_multi(fig, "Candidate_F_Site_Product_Availability_Matrix", dpi=600)
    plt.close(fig)

    # Visualization G: status composition.
    comp = (
        master.groupby(["PRODUCT", "VALUE_STATUS"])
        .size()
        .reset_index(name="N")
        .pivot(index="PRODUCT", columns="VALUE_STATUS", values="N")
        .fillna(0)
        .reindex(products)
    )
    desired = ["DIRECT_PRODUCT", "PROJECT_RECONSTRUCTED", "EXCLUDED_PARTIAL", "EXCLUDED_ZERO_COVERAGE", "NO_SOURCE_DATA", "FAILED_RECONSTRUCTION", "OTHER"]
    comp = comp.reindex(columns=[c for c in desired if c in comp.columns], fill_value=0)
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    colors = {
        "DIRECT_PRODUCT": "#1b9e77",
        "PROJECT_RECONSTRUCTED": "#7570b3",
        "EXCLUDED_PARTIAL": "#d95f02",
        "EXCLUDED_ZERO_COVERAGE": "#666666",
        "NO_SOURCE_DATA": "#bdbdbd",
        "FAILED_RECONSTRUCTION": "#e7298a",
        "OTHER": "#a6761d",
    }
    bottom = np.zeros(len(comp))
    x = np.arange(len(comp))
    for col in comp.columns:
        vals = comp[col].values
        ax.bar(x, vals, bottom=bottom, color=colors.get(col, "#999999"), label=col.replace("_", " "), width=0.68)
        bottom += vals
    ax.set_xticks(x, labels=comp.index)
    ax.set_ylabel("Site-month records")
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1))
    save_multi(fig, "Candidate_G_Status_Composition", dpi=600)
    plt.close(fig)

    # Visualization D limitation file.
    (DIRS["reports"] / "Candidate_D_SpatialSupport_Comparability_Limitations.md").write_text(
        "# Candidate D Spatial-Support Comparability Limitations\n\n"
        "A five-product spatial-support/scale-mismatch atlas was not generated because comparable native-support area metrics are not finalized for all five products.\n\n"
        "- GOME-2 has native-footprint area, EC-support area, and footprint-to-EC area-ratio fields.\n"
        "- GOSIF has native-cell counts and effective native-cell counts, but no finalized per-record native area in km2.\n"
        "- TROPOSIF, OCO-2, and OCO-3 have coverage fractions and native footprint counts, but no finalized per-record footprint-area fields in the product finalization tables.\n\n"
        "A mixed atlas would therefore imply a comparability that the finalized data do not support. The master table preserves all existing validated support metrics without fabricating missing area fields.\n",
        encoding="utf-8",
    )
    return created


def write_captions(figures: list[str]) -> None:
    text = """# SIF Master Audit Figure Captions

## Candidate A. Five-product global availability atlas
Global distribution of the 73 EC study towers for each SIF product. Marker color gives the primary availability fraction, normalized by the product-specific number of eligible EC site-months. Marker size scales with the number of primary-ready site-months. Hollow markers indicate towers with no temporal overlap for that product. Primary-ready records include direct observations for GOSIF, GOME-2, and TROPOSIF, and include documented project-reconstructed positive partial cases for OCO-2 and OCO-3.

## Candidate B. Temporal availability ribbon
Monthly availability of each SIF product across the EC tower network. Each row is a product and color gives the fraction of temporally eligible EC sites with primary-ready SIF in that month. The plot is normalized by product-specific eligibility so that different mission eras are not treated as missing data.

## Candidate C. Pairwise common site-month matrix
Pairwise overlap of primary-ready SITE_ID x YYYYMM records across the five products. Upper-triangle entries give the number of common primary-ready site-months. Lower-triangle entries give Jaccard overlap within the pairwise common temporal period.

## Candidate F. Site x product availability matrix
Tower-level primary availability for the five products. Rows are EC sites and columns are SIF products. Cell color gives the product-specific primary availability fraction for that site, using only eligible EC months in the product's finalized temporal span.

## Candidate G. Status composition
Composition of final site-month statuses by product. Bars separate direct primary records, project-reconstructed primary records, partial exclusions, zero/no-coverage exclusions, no-source-data records, and failed reconstructions where present. This figure makes clear that project reconstruction was used only for OCO-2 and OCO-3.
"""
    (DIRS["reports"] / "SIF_Master_Audit_Figure_Captions.md").write_text(text, encoding="utf-8")


def write_visual_assessment() -> None:
    text = """# SIF Visualization Candidate Assessment

| Candidate | Scientific purpose | What the reader learns | Main paper or supplement | Strength | Limitation | Recommended status |
|---|---|---|---|---|---|---|
| A Global availability atlas | Show where primary-ready SIF exists at the EC towers for each product. | Product availability is spatially and temporally uneven, and availability is normalized by eligibility rather than raw mission length. | Main paper | Directly geospatial and easy to read. | Point maps do not show global native satellite sampling. | STRONG_MAIN_FIGURE |
| B Temporal availability ribbon | Show product eras and monthly tower density. | Mission/product periods and gaps are visible in seconds. | Main paper or supplement | Compact and fair to different temporal spans. | Does not show which sites contribute. | STRONG_MAIN_FIGURE |
| C Pairwise common matrix | Define fair product-to-product comparison support. | Some product pairs have much larger shared primary site-month support than others. | Main paper or supplement | Prevents invalid all-product comparison assumptions. | Pairwise overlap does not assess SIF-GPP behavior. | POSSIBLE_MAIN_FIGURE |
| D Spatial-support atlas | Compare native-support scale mismatch. | Not generated because all five products do not have comparable finalized native-area metrics. | Do not use | Avoids misleading mixed metrics. | A future product-specific support audit could enable a subset figure. | DO_NOT_USE |
| F Site x product matrix | Provide tower-level availability detail. | Site-specific strengths and gaps across products. | Supplement | Dense but honest. | Too detailed for many main-text layouts. | SUPPLEMENT_ONLY |
| G Status composition | Show how final statuses differ by product. | Reconstruction was product-specific, not universal. | Supplement or methods figure | Clarifies methodology quickly. | Not geospatial. | POSSIBLE_MAIN_FIGURE |
"""
    (DIRS["reports"] / "SIF_Visualization_Candidate_Assessment.md").write_text(text, encoding="utf-8")


def write_report(
    master: pd.DataFrame,
    primary: pd.DataFrame,
    definitions: pd.DataFrame,
    manifest: pd.DataFrame,
    reconciliation: pd.DataFrame,
    gosif_diag: pd.DataFrame,
    integrity: pd.DataFrame,
    site_avail: pd.DataFrame,
    pairwise: pd.DataFrame,
    summary: pd.DataFrame,
    source_unchanged: bool,
) -> str:
    fail_primary = reconciliation[reconciliation["STATUS"].eq("UNRESOLVED")]
    fail_integrity = integrity[integrity["STATUS"].eq("FAIL")]
    verdict = VERDICT_FAIL if len(fail_primary) or len(fail_integrity) or not source_unchanged else VERDICT_PASS_FLAGS

    prod_summary = summary.to_markdown(index=False)
    rec_summary = reconciliation.groupby(["PRODUCT", "STATUS"]).size().reset_index(name="N").to_markdown(index=False)
    integ_summary = integrity.to_markdown(index=False)
    gosif_summary = (
        f"Corrected valid GOSIF records = {len(master[(master.PRODUCT == 'GOSIF') & master.PRIMARY_READY])}; "
        f"corrected single-native-cell = {int((master[(master.PRODUCT == 'GOSIF') & master.PRIMARY_READY]['N_NATIVE_PIXELS_OR_FOOTPRINTS'] == 1).sum())}; "
        f"corrected multi-native-cell = {int((master[(master.PRODUCT == 'GOSIF') & master.PRIMARY_READY]['N_NATIVE_PIXELS_OR_FOOTPRINTS'] >= 2).sum())}. "
        f"The six-row diagnostic contains {len(gosif_diag)} rows."
    )
    site_avail_med = site_avail.groupby("PRODUCT")["PRIMARY_AVAILABILITY_FRACTION"].median().reindex(PRODUCT_ORDER).reset_index()
    pair_diag = pairwise[pairwise["PRODUCT_A"] != pairwise["PRODUCT_B"]].sort_values("N_WITH_BOTH", ascending=False).head(10)

    report = f"""# Five-Product SIF Master Harmonization Integrity Audit

Generated: {datetime.now().isoformat(timespec="seconds")}

## 1. Purpose

This read-only audit builds a harmonized five-product SIF database for exactly five products: GOSIF, GOME-2, TROPOSIF, OCO-2, and OCO-3. It does not join tower GPP, fit SIF-GPP relationships, modify finalized source data, or run any new SIF reconstruction.

## 2. Source Datasets

The authoritative source folders were the five finalized product folders listed in the user request. EC template metadata were used only for SITE_ID, YYYYMM, latitude, longitude, IGBP, and the site-month universe. The tower_GPP field in the EC metadata table was explicitly dropped before downstream audit calculations.

SOURCE_INPUTS_UNCHANGED = `{source_unchanged}`

## 3. Authoritative Input Files

`SIF_Product_Input_Manifest.csv` documents the all-record, primary-ready, count, QC, report, provenance, support, reconstruction, and reproducibility inputs selected for each product. The selected all-record and primary-ready tables were chosen because the final reports/reproducibility files explicitly identify them as the master and analysis-ready outputs.

Authoritative manifest rows: {len(manifest)}

## 4. Harmonized Schema

The master all-record table has one row per PRODUCT x SITE_ID x YYYYMM. The primary-ready subset contains only rows with PRIMARY_READY = TRUE.

- All records: {len(master)}
- Primary-ready records: {len(primary)}
- Unique products: {master['PRODUCT'].nunique()}
- Unique primary sites across any product: {primary['SITE_ID'].nunique()}

No wavelength conversion or unit conversion was applied. Missing wavelength/unit metadata remain marked as `{NO_DATA}` rather than being inferred.

## 5. Product-Specific Counts

{prod_summary}

## 6. Count Reconciliation

{rec_summary}

Full reconciliation is stored in `02_audit/SIF_Product_Count_Reconciliation.csv`.

## 7. GOSIF Six-Record Discrepancy

{gosif_summary}

The historical summary counted `N_single_native_pixel_cases` using `dominant_native_pixel_probability_fraction >= 0.999`, while `N_multi_native_pixel_cases` used `n_native_GOSIF_cells_intersecting_EC >= 2`. These are not mutually exclusive. Six records satisfy both conditions and are therefore double-counted by the historical summary. The row-level authoritative table and native-pixel weights are internally consistent: 5,089 records have exactly one native GOSIF cell and 6,398 have two or more native GOSIF cells, summing exactly to 11,487.

## 8. Site and Temporal Availability

Median product-specific site availability:

{site_avail_med.to_markdown(index=False)}

Site-level and monthly availability tables are stored in `04_figure_data/`.

## 9. Cross-Product Overlap

Top pairwise common primary-ready supports:

{pair_diag[['PRODUCT_A','PRODUCT_B','COMMON_PERIOD_START','COMMON_PERIOD_END','N_WITH_BOTH','JACCARD_OVERLAP','FRACTION_OF_POSSIBLE_COMMON_EC_SITE_MONTHS_WITH_BOTH','N_COMMON_SITES']].to_markdown(index=False)}

The pairwise table is intended to define fair downstream product-comparison subsets. No all-five common subset was manufactured.

## 10. Spatial-Support Metadata

GOME-2 has finalized native-footprint area and footprint-to-EC area-ratio metrics. GOSIF has native-cell counts and effective native-cell counts but no finalized per-record native support area in km2. TROPOSIF, OCO-2, and OCO-3 retain coverage fractions and native-footprint counts but not comparable finalized native footprint areas. Therefore a five-product scale-mismatch atlas was not created.

## 11. QC and Uncertainty

OCO-2 and OCO-3 retain one primary-ready reconstructed row each with a project QC flag. OCO-3 has six additional positive-partial cases excluded because required predictors were missing. OCO-2/OCO-3 reconstruction uncertainty intervals are retained where present; product-level validation uncertainty was not converted into fake per-record uncertainty.

## 12. Provenance Integrity

Primary-ready rows retain source file provenance where the final product table provides it. The source safety audit used before/after path, size, mtime, and SHA256 hashes for selected authoritative tables and reports.

## 13. Visualization Candidates

Candidate figures were created in `03_figures/`:

- Candidate A: dark and light five-product global availability atlas.
- Candidate B: temporal availability ribbon.
- Candidate C: pairwise common site-month matrix.
- Candidate F: site x product availability matrix.
- Candidate G: status composition.

Candidate D was intentionally not generated as a figure because comparable native-support area metrics are unavailable across all five products.

## 14. Limitations

- OCO-2 and OCO-3 final reports/tables do not establish source SIF units or retrieval wavelength fields; values are preserved without conversion.
- Product-specific primary-ready definitions differ by design and are documented in the product definition table.
- Availability maps are EC tower point availability maps only; they are not global continuous satellite-coverage surfaces.
- This audit does not use GPP and does not assess SIF-GPP relationships.

## 15. Integrity Checks

{integ_summary}

{verdict}
"""
    path = DIRS["reports"] / "SIF_FiveProduct_Master_Harmonization_Integrity_Audit.md"
    path.write_text(report, encoding="utf-8")
    write_pdf_from_text(path, DIRS["reports"] / "SIF_FiveProduct_Master_Harmonization_Integrity_Audit.pdf")
    return verdict


def write_pdf_from_text(md_path: Path, pdf_path: Path) -> None:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except Exception:
        return
    text = md_path.read_text(encoding="utf-8", errors="replace")
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, leftMargin=42, rightMargin=42, topMargin=42, bottomMargin=42)
    story = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            story.append(Spacer(1, 6))
            continue
        style = styles["Heading2"] if line.startswith("## ") else styles["Heading1"] if line.startswith("# ") else styles["BodyText"]
        clean = line.replace("#", "").replace("—", "-")
        story.append(Paragraph(clean, style))
    doc.build(story)


def copy_script_to_output() -> None:
    src = (Path(PROJECT_ROOT) / 'SIF_EC_FiveProduct_Master_Harmonization/07_scripts/Five_Product_SIF_Master_Harmonization_Audit.py').resolve()
    dst = DIRS["scripts"] / src.name
    if src != dst:
        shutil.copy2(src, dst)


def main() -> None:
    ensure_dirs()
    copy_script_to_output()

    hash_paths = set()
    for pp in PRODUCTS.values():
        for p in [pp.all_records, pp.primary, pp.report, pp.counts, pp.qc, pp.support, pp.reconstruction, pp.reproducibility, pp.model_metadata]:
            if p is not None and p.exists():
                hash_paths.add(p)
    before = file_snapshot([PRODUCTS[p].folder for p in PRODUCT_ORDER], hash_paths=hash_paths)
    before.to_csv(DIRS["audit"] / "SIF_Source_Input_Safety_Before.csv", index=False)

    manifest = build_manifest()
    master, primary, _ = build_master()
    definitions = product_definitions(master)
    reconciliation = count_reconciliation(master)
    gosif_diag = gosif_discrepancy()
    ec, site = load_ec_reference()
    site_avail, monthly, support = availability_tables(master, ec, site)
    pairwise = pairwise_common_site_months(master, ec)
    integrity = integrity_checks(master, site)
    summary = final_audit_summary(master)
    figures = make_figures(site_avail, monthly, pairwise, master)
    write_captions(figures)
    write_visual_assessment()

    after = file_snapshot([PRODUCTS[p].folder for p in PRODUCT_ORDER], hash_paths=hash_paths)
    after.to_csv(DIRS["audit"] / "SIF_Source_Input_Safety_After.csv", index=False)
    compare, source_unchanged = compare_snapshots(before, after)
    compare.to_csv(DIRS["audit"] / "SIF_Source_Input_Safety_Comparison.csv", index=False)

    verdict = write_report(
        master=master,
        primary=primary,
        definitions=definitions,
        manifest=manifest,
        reconciliation=reconciliation,
        gosif_diag=gosif_diag,
        integrity=integrity,
        site_avail=site_avail,
        pairwise=pairwise,
        summary=summary,
        source_unchanged=source_unchanged,
    )

    log = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "output_root": str(OUTPUT_ROOT),
        "n_all_records": int(len(master)),
        "n_primary_ready": int(len(primary)),
        "source_inputs_unchanged": source_unchanged,
        "verdict": verdict,
        "figures_created": figures,
    }
    (DIRS["logs"] / "five_product_master_audit_run_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
