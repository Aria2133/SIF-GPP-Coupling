"""
Finalize GOSIF as native-support-aware EC-footprint weighted analytical values.

No ML, no reconstruction, no gap filling, no fine-resolution analytical GOSIF.
The fine EC/GPP grid is used only as a numerical integration lattice for
assigning EC footprint probability mass to native 0.05-degree GOSIF cells.
"""

from __future__ import annotations
# Original script filename (sanitized): Finalize_GOSIF_Native_Support_EC_Weighted_Dataset.py; public name: 04a_Finalize_GOSIF_Native_Support_EC_Weighted_Dataset.py
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

import json
import math
import platform
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import rasterio
from pyproj import Transformer


PROJECT_ROOT = Path(
    PROJECT_ROOT
)
OUTPUT_ROOT = PROJECT_ROOT / "SIF_EC_GOSIF_Finalization"
QC_ROOT = OUTPUT_ROOT / "qc"
FIG_ROOT = OUTPUT_ROOT / "figures"

GOSIF_ROOT = PROJECT_ROOT / "SIF Datasets" / "GOSIF"
GPP_ROOT = PROJECT_ROOT / "Monthly_EC_Footprint_TIFFs_Python"
PREP_ROOT = PROJECT_ROOT / "SIF_EC_Reconstruction_Model_Preparation"
GPP_TEMPLATE_INVENTORY = PREP_ROOT / "GPP_TIFF_Template_Inventory.csv"
GPP_REFERENCE_METADATA = PREP_ROOT / "GPP_Reference_Metadata_SiteMonth.csv"
PREVIOUS_SCALE_CORRECTED = PREP_ROOT / "GOSIF_SCALE_CORRECTED" / "GOSIF_scale_corrected_EC_weighted_predictors.csv"
SCALE_EVIDENCE = PREP_ROOT / "GOSIF_SCALE_CORRECTED" / "GOSIF_scale_documentation_evidence.csv"

SCALE_FACTOR = 0.0001
GOSIF_UNITS = "W m-2 um-1 sr-1"
FILL_VALUES = {32767, 32766}
WEIGHT_TOL = 1e-6
SINGLE_PIXEL_DOMINANCE_THRESHOLD = 0.999
MEANINGFUL_NATIVE_WEIGHT_THRESHOLD = 0.01
RANDOM_SEED = 42


def ensure_dirs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    QC_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)


def parse_gosif_yyyymm(path: Path) -> int | None:
    name = path.name
    m = re.match(r"^GOSIF_(\d{6})\.tif$", name, flags=re.I)
    if m:
        return int(m.group(1))
    m = re.match(r"^GOSIF_(\d{4})\.M(\d{2})\.tif$", name, flags=re.I)
    if m:
        return int(f"{m.group(1)}{m.group(2)}")
    return None


def discover_gosif_sources() -> dict[int, Path]:
    out: dict[int, Path] = {}
    for p in sorted(GOSIF_ROOT.glob("GOSIF_*.tif")):
        yyyymm = parse_gosif_yyyymm(p)
        if yyyymm is not None:
            out[yyyymm] = p
    return out


def file_snapshot(paths_or_roots: Iterable[Path]) -> pd.DataFrame:
    rows = []
    for base in paths_or_roots:
        base = Path(base)
        if not base.exists():
            rows.append(
                {
                    "path": str(base),
                    "exists": False,
                    "is_file": False,
                    "size_bytes": np.nan,
                    "mtime_ns": np.nan,
                    "audit_scope": "missing",
                }
            )
            continue
        files = [base] if base.is_file() else [p for p in base.rglob("*") if p.is_file()]
        for p in files:
            try:
                st = p.stat()
                rows.append(
                    {
                        "path": str(p),
                        "exists": True,
                        "is_file": p.is_file(),
                        "size_bytes": int(st.st_size),
                        "mtime_ns": int(st.st_mtime_ns),
                        "audit_scope": str(base),
                    }
                )
            except OSError as exc:
                rows.append(
                    {
                        "path": str(p),
                        "exists": False,
                        "is_file": False,
                        "size_bytes": np.nan,
                        "mtime_ns": np.nan,
                        "audit_scope": str(base),
                        "error": str(exc),
                    }
                )
    return pd.DataFrame(rows)


def compare_snapshots(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    b = before.rename(
        columns={
            "exists": "exists_before",
            "size_bytes": "size_bytes_before",
            "mtime_ns": "mtime_ns_before",
            "is_file": "is_file_before",
        }
    )
    a = after.rename(
        columns={
            "exists": "exists_after",
            "size_bytes": "size_bytes_after",
            "mtime_ns": "mtime_ns_after",
            "is_file": "is_file_after",
        }
    )
    merged = b.merge(
        a[["path", "exists_after", "is_file_after", "size_bytes_after", "mtime_ns_after"]],
        on="path",
        how="outer",
    )
    merged["unchanged_size_and_mtime"] = (
        (merged["exists_before"] == merged["exists_after"])
        & (merged["size_bytes_before"] == merged["size_bytes_after"])
        & (merged["mtime_ns_before"] == merged["mtime_ns_after"])
    )
    return merged


def valid_raw_mask(raw: np.ndarray, nodata: float | int | None = None) -> np.ndarray:
    mask = np.isfinite(raw)
    for fill in FILL_VALUES:
        mask &= raw != fill
    if nodata is not None and np.isfinite(nodata):
        mask &= raw != nodata
    return mask


def scale_raw(raw: np.ndarray | float) -> np.ndarray | float:
    return np.asarray(raw, dtype=np.float64) * SCALE_FACTOR


def load_attempt_inventory() -> pd.DataFrame:
    gpp = pd.read_csv(
        GPP_TEMPLATE_INVENTORY,
        usecols=[
            "SITE_ID",
            "SITE_SAFE",
            "YYYYMM",
            "year",
            "month",
            "gpp_tif",
            "valid_gpp_template",
        ],
    )
    gpp = gpp[gpp["valid_gpp_template"].astype(bool)].copy()
    meta = pd.read_csv(
        GPP_REFERENCE_METADATA,
        usecols=["SITE_ID", "YYYYMM", "lat", "lon", "IGBP", "year", "month", "SITE_SAFE"],
    )
    gpp["YYYYMM"] = gpp["YYYYMM"].astype(int)
    meta["YYYYMM"] = meta["YYYYMM"].astype(int)
    df = gpp.merge(
        meta[["SITE_ID", "YYYYMM", "lat", "lon", "IGBP"]],
        on=["SITE_ID", "YYYYMM"],
        how="left",
        validate="one_to_one",
    )
    return df.sort_values(["YYYYMM", "SITE_ID"]).reset_index(drop=True)


def source_convention_check(gosif_sources: dict[int, Path]) -> tuple[pd.DataFrame, dict]:
    evidence_rows = []
    if SCALE_EVIDENCE.exists():
        ev = pd.read_csv(SCALE_EVIDENCE)
        evidence_rows = ev.to_dict("records")
        scale_supported = ev["supports_scale_decision"].astype(str).str.contains("YES_OFFICIAL_README_SCALE_0P0001").any()
    else:
        scale_supported = False

    sample_months = []
    for yyyymm in [200101, 201306, 201908, 202010, 202407, 202412]:
        if yyyymm in gosif_sources:
            sample_months.append(yyyymm)
    sample_months.extend([min(gosif_sources), max(gosif_sources)])
    sample_months = sorted(set(sample_months))

    rows = []
    for yyyymm in sample_months:
        p = gosif_sources[yyyymm]
        with rasterio.open(p) as ds:
            arr = ds.read(1, masked=False)
            valid = valid_raw_mask(arr, ds.nodata)
            vals = arr[valid]
            rows.append(
                {
                    "YYYYMM": yyyymm,
                    "source_file": str(p),
                    "crs": str(ds.crs),
                    "transform": str(ds.transform),
                    "width": ds.width,
                    "height": ds.height,
                    "bounds": str(ds.bounds),
                    "dtype": ds.dtypes[0],
                    "rasterio_nodata": ds.nodata,
                    "rasterio_scale": ds.scales[0] if ds.scales else np.nan,
                    "rasterio_offset": ds.offsets[0] if ds.offsets else np.nan,
                    "raw_min_excluding_fill": float(np.min(vals)) if vals.size else np.nan,
                    "raw_median_excluding_fill": float(np.median(vals)) if vals.size else np.nan,
                    "raw_max_excluding_fill": float(np.max(vals)) if vals.size else np.nan,
                    "physical_min_scaled": float(np.min(vals) * SCALE_FACTOR) if vals.size else np.nan,
                    "physical_median_scaled": float(np.median(vals) * SCALE_FACTOR) if vals.size else np.nan,
                    "physical_max_scaled": float(np.max(vals) * SCALE_FACTOR) if vals.size else np.nan,
                    "fill_32767_count": int((arr == 32767).sum()),
                    "fill_32766_count": int((arr == 32766).sum()),
                    "local_official_scale_evidence_found": bool(scale_supported),
                    "decision": "CONFIRMED: multiply by 0.0001" if scale_supported else "UNRESOLVED",
                }
            )
    check = pd.DataFrame(rows)
    check.to_csv(QC_ROOT / "GOSIF_Source_Convention_Check.csv", index=False)
    if evidence_rows:
        pd.DataFrame(evidence_rows).to_csv(QC_ROOT / "GOSIF_Scale_Documentation_Evidence_Used.csv", index=False)
    if not scale_supported:
        raise RuntimeError("GOSIF scale factor unresolved or contradicted; stopping finalization.")

    # Sanity: all sampled files should be monthly global 0.05-degree EPSG:4326 int rasters.
    bad = check[
        (check["crs"] != "EPSG:4326")
        | (check["width"] != 7200)
        | (check["height"] != 3600)
        | (~np.isclose(check["rasterio_scale"].astype(float), 1.0))
    ]
    if not bad.empty:
        raise RuntimeError("Unexpected sampled GOSIF source convention; see GOSIF_Source_Convention_Check.csv")

    temporal = {
        "local_archive_temporal_resolution": "monthly",
        "filename_patterns": "GOSIF_YYYYMM.tif and GOSIF_YYYY.MMM.tif",
        "match_rule": "SITE_ID x YYYYMM uses same monthly GOSIF YYYYMM source raster",
        "n_source_months": len(gosif_sources),
        "first_source_month": min(gosif_sources),
        "last_source_month": max(gosif_sources),
    }
    return check, temporal


def source_raster_stats(raw_arr: np.ndarray, nodata: float | int | None) -> dict:
    valid = valid_raw_mask(raw_arr, nodata)
    vals = raw_arr[valid]
    return {
        "GOSIF_raw_source_min": float(vals.min()) if vals.size else np.nan,
        "GOSIF_raw_source_max": float(vals.max()) if vals.size else np.nan,
        "GOSIF_raw_source_median": float(np.median(vals)) if vals.size else np.nan,
        "GOSIF_source_valid_pixel_count": int(vals.size),
        "GOSIF_source_fill_32767_count": int((raw_arr == 32767).sum()),
        "GOSIF_source_fill_32766_count": int((raw_arr == 32766).sum()),
    }


def native_indices_from_lonlat(lon: np.ndarray, lat: np.ndarray, ds: rasterio.DatasetReader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = ds.transform
    if abs(t.b) > 1e-12 or abs(t.d) > 1e-12:
        inv = ~t
        col_f, row_f = inv * (lon, lat)
    else:
        col_f = (lon - t.c) / t.a
        row_f = (lat - t.f) / t.e
    cols = np.floor(col_f).astype(np.int64)
    rows = np.floor(row_f).astype(np.int64)
    inside = (rows >= 0) & (rows < ds.height) & (cols >= 0) & (cols < ds.width)
    return rows, cols, inside


def native_center(row: np.ndarray, col: np.ndarray, ds: rasterio.DatasetReader) -> tuple[np.ndarray, np.ndarray]:
    x, y = rasterio.transform.xy(ds.transform, row, col, offset="center")
    return np.asarray(y, dtype=float), np.asarray(x, dtype=float)


def get_grid_mapping(gpp_path: Path, gosif_ds: rasterio.DatasetReader, cache: dict) -> dict:
    with rasterio.open(gpp_path) as gpp:
        key = (
            str(gpp.crs),
            tuple(gpp.transform)[:6],
            gpp.width,
            gpp.height,
            str(gosif_ds.crs),
            tuple(gosif_ds.transform)[:6],
            gosif_ds.width,
            gosif_ds.height,
        )
        if key in cache:
            return cache[key]
        rr, cc = np.indices((gpp.height, gpp.width), dtype=np.int32)
        xs, ys = rasterio.transform.xy(gpp.transform, rr.ravel(), cc.ravel(), offset="center")
        transformer = Transformer.from_crs(gpp.crs, gosif_ds.crs, always_xy=True)
        lon, lat = transformer.transform(np.asarray(xs), np.asarray(ys))
        native_rows, native_cols, inside = native_indices_from_lonlat(np.asarray(lon), np.asarray(lat), gosif_ds)
        native_idx = native_rows * gosif_ds.width + native_cols
        native_idx[~inside] = -1
        mapping = {
            "shape": (gpp.height, gpp.width),
            "native_rows": native_rows.reshape(gpp.height, gpp.width),
            "native_cols": native_cols.reshape(gpp.height, gpp.width),
            "native_idx": native_idx.reshape(gpp.height, gpp.width),
            "inside_gosif": inside.reshape(gpp.height, gpp.width),
        }
        cache[key] = mapping
        return mapping


def aggregate_support(
    support_weights: np.ndarray,
    mapping: dict,
    raw_arr: np.ndarray,
    gosif_ds: rasterio.DatasetReader,
    support_name: str,
) -> tuple[dict, list[dict]]:
    finite = np.isfinite(support_weights) & (support_weights > 0) & mapping["inside_gosif"]
    total_support_mass = float(np.nansum(support_weights[finite])) if finite.any() else 0.0
    if not finite.any() or total_support_mass <= 0:
        return {
            f"{support_name}_status": "NO_VALID_SUPPORT",
            f"{support_name}_weighted": np.nan,
            f"{support_name}_support_mass_total": total_support_mass,
            f"{support_name}_support_mass_valid_gosif": 0.0,
            f"{support_name}_invalid_gosif_support_fraction": np.nan,
        }, []

    idx = mapping["native_idx"][finite].astype(np.int64)
    weights_in = support_weights[finite].astype(np.float64)
    unique_idx, inverse = np.unique(idx, return_inverse=True)
    masses = np.bincount(inverse, weights=weights_in, minlength=len(unique_idx)).astype(np.float64)
    rows = unique_idx // gosif_ds.width
    cols = unique_idx % gosif_ds.width
    raw = raw_arr[rows, cols].astype(np.float64)
    valid_raw = valid_raw_mask(raw, gosif_ds.nodata)
    valid_mass = masses[valid_raw]
    valid_raw_values = raw[valid_raw]
    valid_rows = rows[valid_raw]
    valid_cols = cols[valid_raw]

    support_mass_valid = float(valid_mass.sum())
    invalid_mass = float(masses[~valid_raw].sum())
    invalid_fraction = invalid_mass / total_support_mass if total_support_mass > 0 else np.nan

    if support_mass_valid <= 0 or valid_raw_values.size == 0:
        weight_rows = []
        for u, r, c, rv, mass in zip(unique_idx, rows, cols, raw, masses):
            lat_c, lon_c = native_center(np.array([r]), np.array([c]), gosif_ds)
            weight_rows.append(
                {
                    "native_pixel_id": int(u),
                    "native_row": int(r),
                    "native_col": int(c),
                    "native_pixel_center_lat": float(lat_c[0]),
                    "native_pixel_center_lon": float(lon_c[0]),
                    "GOSIF_raw": float(rv),
                    "GOSIF_scaled": np.nan,
                    "EC_probability_mass": float(mass),
                    "normalized_EC_weight": np.nan,
                    "support_name": support_name,
                    "native_value_valid": False,
                }
            )
        return {
            f"{support_name}_status": "ALL_GOSIF_NODATA",
            f"{support_name}_weighted": np.nan,
            f"{support_name}_support_mass_total": total_support_mass,
            f"{support_name}_support_mass_valid_gosif": 0.0,
            f"{support_name}_invalid_gosif_support_fraction": 1.0,
        }, weight_rows

    norm_weights = valid_mass / support_mass_valid
    scaled = valid_raw_values * SCALE_FACTOR
    weighted = float(np.sum(norm_weights * scaled))
    sum_w = float(norm_weights.sum())
    dominant = float(norm_weights.max())
    effective_n = float(1.0 / np.sum(norm_weights**2)) if norm_weights.size else np.nan
    meaningful_n = int((norm_weights >= MEANINGFUL_NATIVE_WEIGHT_THRESHOLD).sum())
    lat_c, lon_c = native_center(valid_rows, valid_cols, gosif_ds)

    weight_rows = []
    normalized_lookup = {int(u): 0.0 for u in unique_idx}
    for u, w in zip(unique_idx[valid_raw], norm_weights):
        normalized_lookup[int(u)] = float(w)
    for u, r, c, rv, mass, is_valid in zip(unique_idx, rows, cols, raw, masses, valid_raw):
        lat_one, lon_one = native_center(np.array([r]), np.array([c]), gosif_ds)
        weight_rows.append(
            {
                "native_pixel_id": int(u),
                "native_row": int(r),
                "native_col": int(c),
                "native_pixel_center_lat": float(lat_one[0]),
                "native_pixel_center_lon": float(lon_one[0]),
                "GOSIF_raw": float(rv),
                "GOSIF_scaled": float(rv * SCALE_FACTOR) if is_valid else np.nan,
                "EC_probability_mass": float(mass),
                "normalized_EC_weight": normalized_lookup[int(u)] if is_valid else 0.0,
                "support_name": support_name,
                "native_value_valid": bool(is_valid),
            }
        )

    value_min = float(np.min(scaled))
    value_max = float(np.max(scaled))
    outside_range = bool(weighted < value_min - WEIGHT_TOL or weighted > value_max + WEIGHT_TOL)
    return {
        f"{support_name}_status": "OK",
        f"{support_name}_weighted": weighted,
        f"{support_name}_support_mass_total": total_support_mass,
        f"{support_name}_support_mass_valid_gosif": support_mass_valid,
        f"{support_name}_invalid_gosif_support_fraction": invalid_fraction,
        f"{support_name}_sum_normalized_weights": sum_w,
        f"{support_name}_native_min": value_min,
        f"{support_name}_native_max": value_max,
        f"{support_name}_native_mean": float(np.mean(scaled)),
        f"{support_name}_native_sd": float(np.std(scaled, ddof=0)) if scaled.size > 1 else 0.0,
        f"{support_name}_n_native_cells": int(valid_raw_values.size),
        f"{support_name}_dominant_native_pixel_probability_fraction": dominant,
        f"{support_name}_effective_native_pixel_count": effective_n,
        f"{support_name}_meaningful_native_pixel_count": meaningful_n,
        f"{support_name}_weighted_outside_native_range": outside_range,
    }, weight_rows


def tower_pixel_value(lat: float, lon: float, raw_arr: np.ndarray, ds: rasterio.DatasetReader) -> tuple[float, int | None, int | None, str]:
    rows, cols, inside = native_indices_from_lonlat(np.asarray([lon]), np.asarray([lat]), ds)
    if not bool(inside[0]):
        return np.nan, None, None, "TOWER_OUTSIDE_GOSIF"
    r, c = int(rows[0]), int(cols[0])
    raw = float(raw_arr[r, c])
    if not bool(valid_raw_mask(np.asarray([raw]), ds.nodata)[0]):
        return np.nan, r, c, "TOWER_GOSIF_NODATA"
    return float(raw * SCALE_FACTOR), r, c, "OK"


def process_month_group(
    yyyymm: int,
    group: pd.DataFrame,
    source_path: Path | None,
    grid_cache: dict,
) -> tuple[list[dict], list[dict]]:
    rows_out: list[dict] = []
    weight_rows: list[dict] = []

    if source_path is None:
        for _, row in group.iterrows():
            rows_out.append(
                {
                    "SITE_ID": row["SITE_ID"],
                    "SITE_SAFE": row["SITE_SAFE"],
                    "YYYYMM": int(row["YYYYMM"]),
                    "lat": row["lat"],
                    "lon": row["lon"],
                    "IGBP": row["IGBP"],
                    "year": int(row["year"]),
                    "month": int(row["month"]),
                    "gpp_tif": row["gpp_tif"],
                    "GOSIF_status": "EXCLUDED_NO_GOSIF_SOURCE",
                    "GOSIF_source_file": "",
                    "GOSIF_scale_factor": SCALE_FACTOR,
                    "GOSIF_units": GOSIF_UNITS,
                }
            )
        return rows_out, weight_rows

    with rasterio.open(source_path) as ds:
        raw_arr = ds.read(1, masked=False)
        src_stats = source_raster_stats(raw_arr, ds.nodata)
        for _, row in group.iterrows():
            base = {
                "SITE_ID": row["SITE_ID"],
                "SITE_SAFE": row["SITE_SAFE"],
                "YYYYMM": int(row["YYYYMM"]),
                "lat": float(row["lat"]) if pd.notna(row["lat"]) else np.nan,
                "lon": float(row["lon"]) if pd.notna(row["lon"]) else np.nan,
                "IGBP": row["IGBP"],
                "year": int(row["year"]),
                "month": int(row["month"]),
                "gpp_tif": row["gpp_tif"],
                "GOSIF_source_file": str(source_path),
                "GOSIF_scale_factor": SCALE_FACTOR,
                "GOSIF_units": GOSIF_UNITS,
                "GOSIF_temporal_resolution": "monthly",
                "GOSIF_temporal_match_rule": "same YYYYMM monthly source raster",
                **src_stats,
            }
            try:
                gpp_path = Path(row["gpp_tif"])
                with rasterio.open(gpp_path) as gpp:
                    footprint = gpp.read(1, masked=False).astype(np.float64)
                    blind = gpp.read(3, masked=False).astype(np.float64) if gpp.count >= 3 else np.full_like(footprint, np.nan)
                mapping = get_grid_mapping(gpp_path, ds, grid_cache)

                ec_weights = np.where(np.isfinite(footprint) & (footprint > 0), footprint, 0.0)
                ec_stats, ec_weight_rows = aggregate_support(ec_weights, mapping, raw_arr, ds, "EC")
                blind_weights = np.where(np.isfinite(blind) & (blind > 0), blind, 0.0)
                blind_stats, _blind_weight_rows = aggregate_support(blind_weights, mapping, raw_arr, ds, "blind_buffer")
                tower_val, tower_row, tower_col, tower_status = tower_pixel_value(base["lat"], base["lon"], raw_arr, ds)

                status = "VALID_EC_WEIGHTED"
                qc_reasons = []
                if ec_stats["EC_status"] == "NO_VALID_SUPPORT":
                    status = "EXCLUDED_INVALID_EC_FOOTPRINT"
                    qc_reasons.append("no_valid_ec_probability")
                elif ec_stats["EC_status"] == "ALL_GOSIF_NODATA":
                    status = "EXCLUDED_ALL_GOSIF_NODATA"
                    qc_reasons.append("all_ec_support_on_fill_or_nodata_gosif")
                else:
                    if not np.isclose(ec_stats.get("EC_sum_normalized_weights", np.nan), 1.0, atol=WEIGHT_TOL):
                        status = "EXCLUDED_OTHER"
                        qc_reasons.append("weight_normalization_failure")
                    if ec_stats.get("EC_weighted_outside_native_range", False):
                        status = "EXCLUDED_OTHER"
                        qc_reasons.append("weighted_value_outside_contributing_native_range")
                    if ec_stats.get("EC_invalid_gosif_support_fraction", 0.0) > WEIGHT_TOL:
                        status = "EXCLUDED_INVALID_GOSIF"
                        qc_reasons.append("ec_probability_on_invalid_gosif_fill_or_nodata")

                single_pixel_consistency_ok = np.nan
                if int(ec_stats.get("EC_n_native_cells", 0)) == 1:
                    # The dominant native value is the value with max normalized weight.
                    valid_rows = [wr for wr in ec_weight_rows if wr["native_value_valid"]]
                    if valid_rows:
                        dom = max(valid_rows, key=lambda wr: wr["normalized_EC_weight"])
                        single_pixel_consistency_ok = bool(
                            abs(ec_stats["EC_weighted"] - dom["GOSIF_scaled"]) <= max(WEIGHT_TOL, 1e-9)
                        )
                        if not single_pixel_consistency_ok:
                            status = "EXCLUDED_OTHER"
                            qc_reasons.append("single_native_pixel_consistency_failure")

                for wr in ec_weight_rows:
                    wr.update(
                        {
                            "SITE_ID": base["SITE_ID"],
                            "YYYYMM": base["YYYYMM"],
                            "GOSIF_source_file": str(source_path),
                        }
                    )
                weight_rows.extend(ec_weight_rows)

                rows_out.append(
                    {
                        **base,
                        "GOSIF_status": status,
                        "QC_flag_reasons": ";".join(qc_reasons),
                        "GOSIF_EC_weighted": ec_stats.get("EC_weighted", np.nan),
                        "GOSIF_tower_native_pixel": tower_val,
                        "GOSIF_tower_native_row": tower_row,
                        "GOSIF_tower_native_col": tower_col,
                        "GOSIF_tower_native_pixel_status": tower_status,
                        "GOSIF_blind_buffer_weighted": blind_stats.get("blind_buffer_weighted", np.nan),
                        "n_native_GOSIF_cells_intersecting_EC": ec_stats.get("EC_n_native_cells", 0),
                        "n_meaningful_native_GOSIF_cells_ge_0p01": ec_stats.get("EC_meaningful_native_pixel_count", 0),
                        "GOSIF_native_min": ec_stats.get("EC_native_min", np.nan),
                        "GOSIF_native_max": ec_stats.get("EC_native_max", np.nan),
                        "GOSIF_native_mean": ec_stats.get("EC_native_mean", np.nan),
                        "GOSIF_native_sd": ec_stats.get("EC_native_sd", np.nan),
                        "dominant_native_pixel_probability_fraction": ec_stats.get(
                            "EC_dominant_native_pixel_probability_fraction", np.nan
                        ),
                        "GOSIF_effective_native_pixel_count": ec_stats.get("EC_effective_native_pixel_count", np.nan),
                        "EC_probability_sum_before_normalization": ec_stats.get("EC_support_mass_total", np.nan),
                        "EC_probability_sum_on_valid_GOSIF": ec_stats.get("EC_support_mass_valid_gosif", np.nan),
                        "EC_probability_fraction_on_invalid_GOSIF": ec_stats.get("EC_invalid_gosif_support_fraction", np.nan),
                        "n_valid_EC_probability_cells": int((ec_weights > 0).sum()),
                        "sum_normalized_native_weights": ec_stats.get("EC_sum_normalized_weights", np.nan),
                        "weighted_value_within_native_range": not bool(ec_stats.get("EC_weighted_outside_native_range", False)),
                        "single_native_pixel_consistency_ok": single_pixel_consistency_ok,
                        "blind_buffer_status": blind_stats.get("blind_buffer_status", ""),
                        "blind_buffer_n_native_cells": blind_stats.get("blind_buffer_n_native_cells", np.nan),
                        "blind_buffer_support_mass_total": blind_stats.get("blind_buffer_support_mass_total", np.nan),
                        "native_support_method": (
                            "EC/GPP grid cell centers assigned to containing native 0.05-degree GOSIF cell; "
                            "probability mass summed by native cell and normalized over valid native GOSIF cells"
                        ),
                    }
                )
            except Exception as exc:
                rows_out.append(
                    {
                        **base,
                        "GOSIF_status": "EXCLUDED_OTHER",
                        "QC_flag_reasons": f"projection_or_spatial_mapping_problem: {exc}",
                        "GOSIF_EC_weighted": np.nan,
                    }
                )
    return rows_out, weight_rows


def compare_previous(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not PREVIOUS_SCALE_CORRECTED.exists():
        return pd.DataFrame(), pd.DataFrame()
    old = pd.read_csv(PREVIOUS_SCALE_CORRECTED)
    old["YYYYMM"] = old["YYYYMM"].astype(int)
    comp = master[["SITE_ID", "YYYYMM", "GOSIF_EC_weighted", "GOSIF_status"]].merge(
        old[["SITE_ID", "YYYYMM", "GOSIF_EC_weighted", "GOSIF_status"]].rename(
            columns={
                "GOSIF_EC_weighted": "previous_scale_corrected_GOSIF_EC_weighted",
                "GOSIF_status": "previous_GOSIF_status",
            }
        ),
        on=["SITE_ID", "YYYYMM"],
        how="inner",
    )
    comp = comp[
        comp["GOSIF_EC_weighted"].notna()
        & comp["previous_scale_corrected_GOSIF_EC_weighted"].notna()
    ].copy()
    if comp.empty:
        return comp, pd.DataFrame()
    comp["new_minus_previous"] = comp["GOSIF_EC_weighted"] - comp["previous_scale_corrected_GOSIF_EC_weighted"]
    diff = comp["new_minus_previous"].to_numpy(float)
    summary = pd.DataFrame(
        [
            {
                "N_common_cases": int(len(comp)),
                "mean_difference_new_minus_previous": float(np.mean(diff)),
                "median_difference": float(np.median(diff)),
                "RMSE_difference": float(np.sqrt(np.mean(diff**2))),
                "P05_difference": float(np.percentile(diff, 5)),
                "P95_difference": float(np.percentile(diff, 95)),
                "max_absolute_difference": float(np.max(np.abs(diff))),
                "correlation": float(np.corrcoef(comp["GOSIF_EC_weighted"], comp["previous_scale_corrected_GOSIF_EC_weighted"])[0, 1]),
                "previous_method_note": (
                    "Comparison is against the previous scale-corrected GOSIF predictor table. "
                    "It is diagnostic only and does not use GPP."
                ),
            }
        ]
    )
    return comp, summary


def method_comparison(master: pd.DataFrame) -> pd.DataFrame:
    valid = master[master["GOSIF_status"].eq("VALID_EC_WEIGHTED")].copy()
    pairs = []
    comparisons = [
        ("tower_minus_EC", "GOSIF_tower_native_pixel", "GOSIF_EC_weighted"),
        ("blind_buffer_minus_EC", "GOSIF_blind_buffer_weighted", "GOSIF_EC_weighted"),
        ("tower_minus_blind_buffer", "GOSIF_tower_native_pixel", "GOSIF_blind_buffer_weighted"),
    ]
    for name, a, b in comparisons:
        sub = valid[[a, b]].dropna()
        if sub.empty:
            continue
        d = sub[a] - sub[b]
        pairs.append(
            {
                "comparison": name,
                "N": int(len(sub)),
                "mean_difference": float(d.mean()),
                "median_difference": float(d.median()),
                "RMSE_difference": float(np.sqrt(np.mean(np.square(d)))),
                "P05_difference": float(np.percentile(d, 5)),
                "P95_difference": float(np.percentile(d, 95)),
                "max_absolute_difference": float(np.max(np.abs(d))),
                "correlation": float(np.corrcoef(sub[a], sub[b])[0, 1]) if len(sub) > 1 else np.nan,
            }
        )
    return pd.DataFrame(pairs)


def support_summary(master: pd.DataFrame) -> pd.DataFrame:
    valid = master[master["GOSIF_status"].eq("VALID_EC_WEIGHTED")].copy()
    n = len(valid)
    if n == 0:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "N_valid": n,
                "N_single_native_pixel_cases": int((valid["dominant_native_pixel_probability_fraction"] >= SINGLE_PIXEL_DOMINANCE_THRESHOLD).sum()),
                "percent_single_native_pixel_cases": float((valid["dominant_native_pixel_probability_fraction"] >= SINGLE_PIXEL_DOMINANCE_THRESHOLD).mean() * 100),
                "N_multi_native_pixel_cases": int((valid["n_native_GOSIF_cells_intersecting_EC"] >= 2).sum()),
                "percent_multi_native_pixel_cases": float((valid["n_native_GOSIF_cells_intersecting_EC"] >= 2).mean() * 100),
                "N_cases_with_2plus_meaningful_pixels_weight_ge_0p01": int((valid["n_meaningful_native_GOSIF_cells_ge_0p01"] >= 2).sum()),
                "percent_cases_with_2plus_meaningful_pixels_weight_ge_0p01": float((valid["n_meaningful_native_GOSIF_cells_ge_0p01"] >= 2).mean() * 100),
                "N_cases_with_3plus_meaningful_pixels_weight_ge_0p01": int((valid["n_meaningful_native_GOSIF_cells_ge_0p01"] >= 3).sum()),
                "percent_cases_with_3plus_meaningful_pixels_weight_ge_0p01": float((valid["n_meaningful_native_GOSIF_cells_ge_0p01"] >= 3).mean() * 100),
                "meaningful_weight_threshold": MEANINGFUL_NATIVE_WEIGHT_THRESHOLD,
                "median_n_native_cells": float(valid["n_native_GOSIF_cells_intersecting_EC"].median()),
                "median_effective_native_pixel_count": float(valid["GOSIF_effective_native_pixel_count"].median()),
                "median_dominant_native_pixel_probability_fraction": float(valid["dominant_native_pixel_probability_fraction"].median()),
            }
        ]
    )


def make_scale_qc_examples(master: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    requested = [
        ("AT-Zoe", 202010),
        ("AU-Dry", 201009),
        ("CH-Lae", 202407),
        ("BE-Bra", 201908),
        ("US-Wkg", 201306),
    ]
    rows = []
    for site, yyyymm in requested:
        sub = master[(master["SITE_ID"].eq(site)) & (master["YYYYMM"].eq(yyyymm))]
        if sub.empty:
            site_rows = master[master["SITE_ID"].eq(site)].sort_values("YYYYMM")
            if site_rows.empty:
                rows.append({"requested_SITE_ID": site, "requested_YYYYMM": yyyymm, "status": "SITE_NOT_AVAILABLE"})
                continue
            rec = site_rows.iloc[0]
            status = f"REQUESTED_MONTH_NOT_AVAILABLE_USED_{int(rec['YYYYMM'])}"
        else:
            rec = sub.iloc[0]
            status = "REQUESTED_EXAMPLE_AVAILABLE"
        w = weights[(weights["SITE_ID"].eq(rec["SITE_ID"])) & (weights["YYYYMM"].eq(int(rec["YYYYMM"]))) & (weights["support_name"].eq("EC"))].copy()
        if not w.empty:
            dom = w.sort_values("normalized_EC_weight", ascending=False).iloc[0]
            raw_example = dom["GOSIF_raw"]
            scaled_example = dom["GOSIF_scaled"]
            weight_example = dom["normalized_EC_weight"]
        else:
            raw_example = np.nan
            scaled_example = np.nan
            weight_example = np.nan
        rows.append(
            {
                "requested_SITE_ID": site,
                "requested_YYYYMM": yyyymm,
                "SITE_ID": rec["SITE_ID"],
                "YYYYMM": int(rec["YYYYMM"]),
                "status": status,
                "dominant_native_GOSIF_raw": raw_example,
                "scale_factor": SCALE_FACTOR,
                "dominant_native_GOSIF_scaled": scaled_example,
                "dominant_native_weight": weight_example,
                "GOSIF_EC_weighted": rec.get("GOSIF_EC_weighted", np.nan),
                "GOSIF_status": rec.get("GOSIF_status", ""),
                "note": "raw values are multiplied by 0.0001 before entering EC weighting",
            }
        )
    return pd.DataFrame(rows)


def make_manual_checks(master: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    valid = master[
        (master["GOSIF_status"].eq("VALID_EC_WEIGHTED"))
        & (master["n_native_GOSIF_cells_intersecting_EC"] >= 2)
    ].sort_values(["GOSIF_effective_native_pixel_count", "n_native_GOSIF_cells_intersecting_EC"], ascending=False)
    rows = []
    for _, rec in valid.head(8).iterrows():
        w = weights[
            (weights["SITE_ID"].eq(rec["SITE_ID"]))
            & (weights["YYYYMM"].eq(int(rec["YYYYMM"])))
            & (weights["support_name"].eq("EC"))
            & (weights["native_value_valid"])
        ].sort_values("normalized_EC_weight", ascending=False)
        terms = []
        calc = 0.0
        for _, wr in w.iterrows():
            term = wr["GOSIF_scaled"] * wr["normalized_EC_weight"]
            calc += term
            terms.append(f"{wr['GOSIF_scaled']:.6g}×{wr['normalized_EC_weight']:.4f}")
        rows.append(
            {
                "SITE_ID": rec["SITE_ID"],
                "YYYYMM": int(rec["YYYYMM"]),
                "n_native_cells": int(rec["n_native_GOSIF_cells_intersecting_EC"]),
                "effective_native_pixel_count": rec["GOSIF_effective_native_pixel_count"],
                "calculation": " + ".join(terms),
                "manual_sum": calc,
                "GOSIF_EC_weighted": rec["GOSIF_EC_weighted"],
                "absolute_difference": abs(calc - rec["GOSIF_EC_weighted"]),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "_No rows._"
    return df.head(max_rows).to_markdown(index=False)


def write_pdf_from_text(md_path: Path, pdf_path: Path) -> None:
    text = md_path.read_text(encoding="utf-8", errors="replace")
    lines: list[str] = []
    for raw in text.splitlines():
        if len(raw) <= 120:
            lines.append(raw)
        else:
            for i in range(0, len(raw), 120):
                lines.append(raw[i : i + 120])
    with PdfPages(pdf_path) as pdf:
        for i in range(0, len(lines), 46):
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.06, 0.96, "\n".join(lines[i : i + 46]), va="top", ha="left", family="monospace", fontsize=8)
            pdf.savefig(fig)
            plt.close(fig)


def create_simple_summary(master: pd.DataFrame, old_summary: pd.DataFrame, manual: pd.DataFrame) -> None:
    valid = master[master["GOSIF_status"].eq("VALID_EC_WEIGHTED")].copy()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("GOSIF finalization: native-support EC-footprint weighting", fontsize=18, fontweight="bold")

    ax = axes[0, 0]
    ax.axis("off")
    ax.text(
        0.02,
        0.85,
        "A. What changed\n\nOLD:\ncoarse GOSIF → nearest-neighbour stretch\nto fine EC grid\n\nNEW:\nnative GOSIF + EC probability\n→ probability mass per native cell\n→ one EC-support GOSIF value",
        fontsize=12,
        va="top",
    )

    ax = axes[0, 1]
    ax.axis("off")
    single = valid[valid["dominant_native_pixel_probability_fraction"] >= SINGLE_PIXEL_DOMINANCE_THRESHOLD]
    if not single.empty:
        rec = single.iloc[0]
        txt = (
            "B. Single-native-pixel example\n\n"
            f"{rec['SITE_ID']} {int(rec['YYYYMM'])}\n"
            f"Dominant probability: {rec['dominant_native_pixel_probability_fraction']:.4f}\n"
            f"Tower native pixel: {rec['GOSIF_tower_native_pixel']:.4f}\n"
            f"EC-weighted GOSIF: {rec['GOSIF_EC_weighted']:.4f}"
        )
    else:
        txt = "B. Single-native-pixel example\n\nNo effectively single-pixel case found."
    ax.text(0.02, 0.85, txt, fontsize=12, va="top")

    ax = axes[0, 2]
    ax.axis("off")
    if not manual.empty:
        rec = manual.iloc[0]
        formula = rec["calculation"]
        if len(formula) > 220:
            formula = formula[:220] + " ..."
        txt = (
            "C. Multi-native-pixel example\n\n"
            f"{rec['SITE_ID']} {int(rec['YYYYMM'])}\n"
            f"{formula}\n=\n{rec['GOSIF_EC_weighted']:.5f}"
        )
    else:
        txt = "C. Multi-native-pixel example\n\nNo multi-pixel valid case found."
    ax.text(0.02, 0.85, txt, fontsize=11, va="top")

    ax = axes[1, 0]
    ax.hist(valid["n_native_GOSIF_cells_intersecting_EC"].dropna(), bins=np.arange(0.5, valid["n_native_GOSIF_cells_intersecting_EC"].max() + 1.5), color="#4c78a8")
    ax.set_title("D. Native GOSIF cells contributing to EC footprint")
    ax.set_xlabel("Native cells")
    ax.set_ylabel("Site-months")

    ax = axes[1, 1]
    comp_path = OUTPUT_ROOT / "GOSIF_Old_vs_New_Diagnostic_Common_Cases.csv"
    if comp_path.exists():
        comp = pd.read_csv(comp_path)
    else:
        comp = pd.DataFrame()
    if not comp.empty:
        ax.scatter(comp["previous_scale_corrected_GOSIF_EC_weighted"], comp["GOSIF_EC_weighted"], s=8, alpha=0.45)
        lims = [
            np.nanmin([comp["previous_scale_corrected_GOSIF_EC_weighted"].min(), comp["GOSIF_EC_weighted"].min()]),
            np.nanmax([comp["previous_scale_corrected_GOSIF_EC_weighted"].max(), comp["GOSIF_EC_weighted"].max()]),
        ]
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_xlabel("Previous scale-corrected value")
        ax.set_ylabel("New native-support value")
        ax.set_title("E. Previous vs new diagnostic")
    else:
        ax.axis("off")
        ax.text(0.02, 0.8, "E. Old vs new\n\nNo previous common values available.", fontsize=12)

    ax = axes[1, 2]
    ax.axis("off")
    ax.text(
        0.02,
        0.85,
        "F. Final status\n\nGOSIF FINALIZATION COMPLETE\n\n×0.0001 applied\nfill values excluded\nnative support preserved\nno GPP values used\nno reconstructed GOSIF TIFFs",
        fontsize=13,
        fontweight="bold",
        color="#1f4e79",
        va="top",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUTPUT_ROOT / "GOSIF_Finalization_Simple_Summary.png", dpi=200)
    fig.savefig(OUTPUT_ROOT / "GOSIF_Finalization_Simple_Summary.pdf")
    plt.close(fig)


def write_reports(
    master: pd.DataFrame,
    weights: pd.DataFrame,
    flags: pd.DataFrame,
    counts: dict,
    source_check: pd.DataFrame,
    temporal: dict,
    old_summary: pd.DataFrame,
    method_summary: pd.DataFrame,
    support: pd.DataFrame,
    scale_examples: pd.DataFrame,
    manual_checks: pd.DataFrame,
    source_inputs_unchanged: bool,
) -> None:
    valid = master[master["GOSIF_status"].eq("VALID_EC_WEIGHTED")].copy()
    old_diff = old_summary.iloc[0].to_dict() if not old_summary.empty else {}
    support_row = support.iloc[0].to_dict() if not support.empty else {}

    readme = f"""GOSIF Finalization — READ ME FIRST

1. Total site-months attempted: {counts['N_total_site_months_attempted']:,}.
2. Valid EC-weighted GOSIF site-months produced: {counts['N_valid_GOSIF_EC_weighted']:,}.
3. GOSIF scale factor used: {SCALE_FACTOR}.
4. Physical units: {GOSIF_UNITS}.
5. Did the old nearest-neighbour stretched raster method remain in final analysis? No.
6. New GOSIF EC value: native GOSIF cells are weighted by EC probability mass assigned to each native 0.05-degree cell.
7. Effectively one native pixel: {support_row.get('percent_single_native_pixel_cases', np.nan):.2f}% of valid cases.
8. Multiple native cells meaningfully matter (>=2 cells with weight >=0.01): {support_row.get('percent_cases_with_2plus_meaningful_pixels_weight_ge_0p01', np.nan):.2f}%.
9. Old-vs-new diagnostic RMSE difference: {old_diff.get('RMSE_difference', np.nan):.6g}; N common = {int(old_diff.get('N_common_cases', 0)) if old_diff else 0}.
10. Tower-pixel and blind-buffer comparator values created? Yes.
11. QC exclusions/flags: {counts['N_excluded']:,} excluded, {counts['N_QC_flagged']:,} flagged rows.
12. Were GPP values used? No; only GPP TIFF bands 1 and 3/spatial grid were read.
13. Were source files modified? {'No' if source_inputs_unchanged else 'Potential modification detected; inspect audit'}.
14. Is GOSIF ready for final SIF-GPP analysis? Yes, for valid primary records only.
"""
    (OUTPUT_ROOT / "GOSIF_Finalization_READ_ME_FIRST.txt").write_text(readme, encoding="utf-8")

    md = []
    md.append("# GOSIF Finalization Report\n")
    md.append(f"Run timestamp: `{datetime.now().isoformat(timespec='seconds')}`\n")
    md.append("Final verdict: **GOSIF FINALIZATION COMPLETE**\n")
    md.append("## Source convention and temporal matching\n")
    md.append(f"- Scale factor: `{SCALE_FACTOR}`\n")
    md.append(f"- Units: `{GOSIF_UNITS}`\n")
    md.append(f"- Fill values excluded: `{sorted(FILL_VALUES)}`\n")
    md.append(f"- Temporal resolution: `{temporal['local_archive_temporal_resolution']}`\n")
    md.append(f"- Temporal match rule: `{temporal['match_rule']}`\n")
    md.append(f"- Local GOSIF months: `{temporal['first_source_month']}` to `{temporal['last_source_month']}` (`{temporal['n_source_months']}` files)\n")
    md.append("Source convention sample:\n")
    md.append(markdown_table(source_check, 8) + "\n")
    md.append("## Native support EC weighting\n")
    md.append(
        "For each site-month, GPP TIFF Band 1 (`footprint_probability`) was used only as EC probability support. "
        "Each valid EC probability cell center was assigned to its containing native GOSIF 0.05-degree cell. "
        "Probability mass was summed by native GOSIF cell, normalized over valid native GOSIF cells, and applied to physically scaled native GOSIF values. "
        "Fine EC cells are a numerical integration lattice only, not independent GOSIF observations.\n"
    )
    md.append("## Counts\n")
    md.append(markdown_table(pd.DataFrame([counts])) + "\n")
    md.append("## Native support summary\n")
    md.append(markdown_table(support) + "\n")
    md.append("## Scale QC traces\n")
    md.append(markdown_table(scale_examples, 10) + "\n")
    md.append("## Multi-native-pixel manual checks\n")
    md.append(markdown_table(manual_checks, 10) + "\n")
    md.append("## Old/previous vs new diagnostic\n")
    md.append(markdown_table(old_summary) + "\n")
    md.append("## Tower / blind-buffer / EC comparator differences\n")
    md.append(markdown_table(method_summary) + "\n")
    md.append("## QC flags\n")
    md.append(f"QC flag rows: `{len(flags)}`\n")
    md.append(markdown_table(flags[["SITE_ID", "YYYYMM", "GOSIF_status", "QC_flag_reasons"]], 20) + "\n")
    md.append("## Input safety audit\n")
    md.append(f"SOURCE_INPUTS_UNCHANGED = `{source_inputs_unchanged}`\n")
    md.append("Raw GOSIF files, GPP TIFF root files, and previous preparation files were compared by size/mtime before and after.\n")
    md.append("## Required outputs\n")
    for name in [
        "GOSIF_Final_EC_Footprint_SIF_All_SiteMonths.csv",
        "GOSIF_Final_EC_Footprint_SIF_Primary_Analysis.csv",
        "GOSIF_EC_Native_Pixel_Weights.csv",
        "GOSIF_Finalization_QC_Flags.csv",
        "GOSIF_Finalization_READ_ME_FIRST.txt",
        "GOSIF_Finalization_Simple_Summary.png",
        "GOSIF_Finalization_Simple_Summary.pdf",
        "GOSIF_Finalization_Reproducibility.json",
    ]:
        md.append(f"- `{OUTPUT_ROOT / name}`\n")
    md.append("\n## Final verdict\n")
    md.append("**GOSIF FINALIZATION COMPLETE**\n")

    md_path = OUTPUT_ROOT / "GOSIF_Finalization_Report.md"
    pdf_path = OUTPUT_ROOT / "GOSIF_Finalization_Report.pdf"
    md_path.write_text("\n".join(md), encoding="utf-8")
    write_pdf_from_text(md_path, pdf_path)


def main() -> None:
    ensure_dirs()
    gosif_sources = discover_gosif_sources()

    safety_inputs = [
        GOSIF_ROOT,
        GPP_ROOT,
        GPP_TEMPLATE_INVENTORY,
        GPP_REFERENCE_METADATA,
        PREVIOUS_SCALE_CORRECTED,
        SCALE_EVIDENCE,
    ]
    before = file_snapshot(safety_inputs)

    source_check, temporal = source_convention_check(gosif_sources)
    attempts = load_attempt_inventory()

    print(f"GOSIF finalization: {len(attempts)} valid GPP site-month templates to attempt.")
    print(f"Local GOSIF monthly source rasters: {len(gosif_sources)} ({min(gosif_sources)}-{max(gosif_sources)}).")

    rows: list[dict] = []
    weight_rows: list[dict] = []
    grid_cache: dict = {}
    for i, (yyyymm, group) in enumerate(attempts.groupby("YYYYMM", sort=True), start=1):
        source = gosif_sources.get(int(yyyymm))
        month_rows, month_weights = process_month_group(int(yyyymm), group, source, grid_cache)
        rows.extend(month_rows)
        weight_rows.extend(month_weights)
        if i % 25 == 0 or i == 1:
            print(f"Processed month group {i}: {yyyymm}; cumulative rows={len(rows)} valid_weights={len(weight_rows)}")

    master = pd.DataFrame(rows).sort_values(["SITE_ID", "YYYYMM"]).reset_index(drop=True)
    weights = pd.DataFrame(weight_rows).sort_values(["SITE_ID", "YYYYMM", "support_name", "normalized_EC_weight"], ascending=[True, True, True, False]).reset_index(drop=True)

    flags = master[~master["GOSIF_status"].eq("VALID_EC_WEIGHTED")].copy()
    valid = master[master["GOSIF_status"].eq("VALID_EC_WEIGHTED")].copy()
    counts = {
        "N_total_site_months_attempted": int(len(master)),
        "N_valid_GOSIF_EC_weighted": int(len(valid)),
        "N_excluded": int(len(master) - len(valid)),
        "N_sites_valid": int(valid["SITE_ID"].nunique()),
        "N_single_native_pixel_cases": int((valid["dominant_native_pixel_probability_fraction"] >= SINGLE_PIXEL_DOMINANCE_THRESHOLD).sum()),
        "N_multi_native_pixel_cases": int((valid["n_native_GOSIF_cells_intersecting_EC"] >= 2).sum()),
        "N_QC_flagged": int(len(flags)),
    }
    if counts["N_valid_GOSIF_EC_weighted"] + counts["N_excluded"] != counts["N_total_site_months_attempted"]:
        raise RuntimeError("GOSIF count reconciliation failed.")

    primary = valid.copy()
    master.to_csv(OUTPUT_ROOT / "GOSIF_Final_EC_Footprint_SIF_All_SiteMonths.csv", index=False)
    primary.to_csv(OUTPUT_ROOT / "GOSIF_Final_EC_Footprint_SIF_Primary_Analysis.csv", index=False)
    weights.to_csv(OUTPUT_ROOT / "GOSIF_EC_Native_Pixel_Weights.csv", index=False)
    flags.to_csv(OUTPUT_ROOT / "GOSIF_Finalization_QC_Flags.csv", index=False)
    pd.DataFrame([counts]).to_csv(OUTPUT_ROOT / "GOSIF_Finalization_Final_Counts.csv", index=False)

    old_comp, old_summary = compare_previous(master)
    old_comp.to_csv(OUTPUT_ROOT / "GOSIF_Old_vs_New_Diagnostic_Common_Cases.csv", index=False)
    old_summary.to_csv(OUTPUT_ROOT / "GOSIF_Old_vs_New_Diagnostic_Summary.csv", index=False)
    method_summary = method_comparison(master)
    method_summary.to_csv(OUTPUT_ROOT / "GOSIF_Spatial_Extraction_Method_Comparison.csv", index=False)
    support = support_summary(master)
    support.to_csv(OUTPUT_ROOT / "GOSIF_Native_Support_Summary.csv", index=False)
    scale_examples = make_scale_qc_examples(master, weights)
    scale_examples.to_csv(QC_ROOT / "GOSIF_Scale_QC_Trace_Examples.csv", index=False)
    manual_checks = make_manual_checks(master, weights)
    manual_checks.to_csv(QC_ROOT / "GOSIF_Multi_Native_Pixel_Manual_Checks.csv", index=False)

    create_simple_summary(master, old_summary, manual_checks)

    after = file_snapshot(safety_inputs)
    audit = compare_snapshots(before, after)
    audit.to_csv(QC_ROOT / "GOSIF_Finalization_Source_Input_Safety_Audit.csv", index=False)
    source_inputs_unchanged = bool(audit["unchanged_size_and_mtime"].fillna(False).all())

    write_reports(
        master,
        weights,
        flags,
        counts,
        source_check,
        temporal,
        old_summary,
        method_summary,
        support,
        scale_examples,
        manual_checks,
        source_inputs_unchanged,
    )

    reproducibility = {
        "script_path": str((Path(PROJECT_ROOT) / 'Codes/pythonProject/Finalize_GOSIF_Native_Support_EC_Weighted_Dataset.py').resolve()),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "gosif_source_root": str(GOSIF_ROOT),
        "gpp_tiff_root": str(GPP_ROOT),
        "gpp_template_inventory": str(GPP_TEMPLATE_INVENTORY),
        "gpp_reference_metadata": str(GPP_REFERENCE_METADATA),
        "scale_factor": SCALE_FACTOR,
        "fill_values": sorted(FILL_VALUES),
        "units": GOSIF_UNITS,
        "native_crs": "EPSG:4326",
        "native_resolution_degrees": 0.05,
        "temporal_matching_logic": temporal,
        "ec_weighting_logic": (
            "GPP Band 1 probability cells >0 assigned by cell center to containing native GOSIF cell; "
            "probability mass summed by native cell and normalized over valid native GOSIF cells"
        ),
        "blind_buffer_logic": "GPP Band 3 positive support cells assigned to native GOSIF cells and area/count weighted",
        "tower_pixel_logic": "site lon/lat mapped directly to containing native GOSIF pixel; no resampling",
        "software_versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "rasterio": rasterio.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "record_counts": counts,
        "random_seed_for_qc_selection": RANDOM_SEED,
        "source_inputs_unchanged": source_inputs_unchanged,
    }
    (OUTPUT_ROOT / "GOSIF_Finalization_Reproducibility.json").write_text(json.dumps(reproducibility, indent=2), encoding="utf-8")

    print("GOSIF finalization complete")
    for k, v in counts.items():
        print(f"{k}: {v}")
    print(f"SOURCE_INPUTS_UNCHANGED: {source_inputs_unchanged}")
    print(f"Primary table: {OUTPUT_ROOT / 'GOSIF_Final_EC_Footprint_SIF_Primary_Analysis.csv'}")
    print(f"Master table: {OUTPUT_ROOT / 'GOSIF_Final_EC_Footprint_SIF_All_SiteMonths.csv'}")
    print(f"Weights table: {OUTPUT_ROOT / 'GOSIF_EC_Native_Pixel_Weights.csv'}")
    print(f"Report: {OUTPUT_ROOT / 'GOSIF_Finalization_Report.md'}")
    print("GOSIF FINALIZATION COMPLETE")


if __name__ == "__main__":
    main()
