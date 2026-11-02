#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create locked-spec 8-band footprint-aware SIF companion TIFFs matching GPP.

This script intentionally creates a new output root and does not modify any
previous GPP TIFFs, source SIF files, or earlier SIF outputs.

Key rule for swath products:
    SIF footprint polygon -> all overlapping GPP-grid cells

Point-to-cell rasterization is not used as the final method.  If a swath source
file lacks footprint/corner/bounds variables, the product/month/site is skipped
and reported instead of producing a misleading point-only TIFF.

Required packages:
    pip install pandas numpy rasterio pyproj shapely h5py matplotlib reportlab tqdm
"""

from __future__ import annotations
# Original script filename (sanitized): Create_FINAL_CLEAN_SIF_Companion_TIFFs_FOOTPRINT_AWARE.py; public name: Create_FINAL_CLEAN_SIF_Companion_TIFFs_FOOTPRINT_AWARE.py
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
import sys
import traceback
import hashlib
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS, Transformer
from rasterio.features import rasterize
from rasterio.warp import transform_bounds
from shapely.geometry import Polygon
from shapely.ops import transform as shapely_transform

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# Reuse only the clean helper surface required by this locked pipeline.
# The legacy helper scripts that contained point-centre swath workflows are
# intentionally not imported.
import SIF_TIFF_FOOTPRINT_AWARE_HELPERS as base

sif_core = base


# =============================================================================
# USER SETTINGS
# =============================================================================

DRY_RUN = False
WRITE_OUTPUTS = True
OVERWRITE_EXISTING = False

PRODUCTS_TO_PROCESS = [
    "GOSIF",
    "GOME-2",
    "gosat",
    "OCO-2",
    "OCO-3",
    "TROPOSIF",
]
SELECTED_SITES = None
SELECTED_YYYYMM = None

NODATA_VALUE = -9999.0

USE_GPP_TIFF_AS_ONLY_SPATIAL_TEMPLATE = True
USE_KMZ_FILES = False

CREATE_ONLY_FILES_WITH_ACTUAL_SIF_OBSERVATIONS = True
DO_NOT_CREATE_ALL_NODATA_PLACEHOLDER_TIFFS = True
DO_NOT_CREATE_STANDALONE_CIRCULAR_BUFFER_TIFFS = True

GOSIF_GRIDDED_RESAMPLING = "nearest"
SWATH_RASTERIZATION_METHOD = "footprint_polygon_to_GPP_grid_cells"
MULTIPLE_SIF_FOOTPRINTS_PER_CELL = "mean"

CREATE_MD_AND_PDF_REPORTS = True
CREATE_QUICKLOOKS = True
ENABLE_SOURCE_CHECKSUM_AUDIT = True
COUNT_ONLY = False

PROJECT_ROOT = Path(
    PROJECT_ROOT
)
GPP_TIFF_ROOT = PROJECT_ROOT / "Monthly_EC_Footprint_TIFFs_Python"
GPP_REFERENCE_CSV = (
    PROJECT_ROOT
    / "finalized_metadata_recovery"
    / (
        "final_metadata_recovered_ALL_SITES_monthly_WD_histogram_"
        "GPP_footprint_input_for_GEE.csv"
    )
)
SIF_ROOT = PROJECT_ROOT / "SIF Datasets"

OUTPUT_ROOT = PROJECT_ROOT / "SIF_TIFF_output"
COMPANION_ROOT = OUTPUT_ROOT
SUMMARY_ROOT = OUTPUT_ROOT / "reports"
QC_ROOT = OUTPUT_ROOT / "reports"
VALIDATION_ROOT = OUTPUT_ROOT / "reports"
QUICKLOOK_ROOT = OUTPUT_ROOT / "quicklooks"

PRODUCTS = ["GOSIF", "GOME-2", "gosat", "OCO-2", "OCO-3", "TROPOSIF"]

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

FOOTPRINT_GEOMETRY_TOKENS = (
    "bound",
    "bounds",
    "corner",
    "corners",
    "vertex",
    "vertices",
    "footprint",
)

EXPLICIT_FOOTPRINT_GEOMETRY_PAIRS = (
    ("Latitude_Corners", "Longitude_Corners"),
    ("latitude_corners", "longitude_corners"),
    ("lat_corners", "lon_corners"),
    ("Latitude_Bounds", "Longitude_Bounds"),
    ("latitude_bounds", "longitude_bounds"),
    ("lat_bounds", "lon_bounds"),
    ("latitude_vertices", "longitude_vertices"),
    ("lat_vertices", "lon_vertices"),
    ("vertex_latitude", "vertex_longitude"),
    ("footprint_latitude", "footprint_longitude"),
    ("footprint_corner_latitude", "footprint_corner_longitude"),
    ("corner_latitude", "corner_longitude"),
    ("Geolocation/footprint_latitude_vertices", "Geolocation/footprint_longitude_vertices"),
    ("Geolocation/footprint_latitude", "Geolocation/footprint_longitude"),
    ("Geolocation/latitude_bounds", "Geolocation/longitude_bounds"),
    ("Geolocation/lat_bounds", "Geolocation/lon_bounds"),
    ("PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds", "PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds"),
    ("PRODUCT/SUPPORT_DATA/GEOLOCATIONS/lat_bounds", "PRODUCT/SUPPORT_DATA/GEOLOCATIONS/lon_bounds"),
)


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_list(name: str, default: list[str] | None) -> list[str] | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


DRY_RUN = env_bool("FOOTPRINT_AWARE_SIF_DRY_RUN", DRY_RUN)
WRITE_OUTPUTS = env_bool("FOOTPRINT_AWARE_SIF_WRITE_OUTPUTS", WRITE_OUTPUTS)
OVERWRITE_EXISTING = env_bool(
    "FOOTPRINT_AWARE_SIF_OVERWRITE_EXISTING", OVERWRITE_EXISTING
)
PRODUCTS_TO_PROCESS = env_list(
    "FOOTPRINT_AWARE_SIF_PRODUCTS", PRODUCTS_TO_PROCESS
)
SELECTED_SITES = env_list("FOOTPRINT_AWARE_SIF_SELECTED_SITES", SELECTED_SITES)
SELECTED_YYYYMM = env_list(
    "FOOTPRINT_AWARE_SIF_SELECTED_YYYYMM", SELECTED_YYYYMM
)
ENABLE_SOURCE_CHECKSUM_AUDIT = env_bool(
    "FOOTPRINT_AWARE_SIF_ENABLE_SOURCE_CHECKSUM_AUDIT",
    ENABLE_SOURCE_CHECKSUM_AUDIT and not DRY_RUN,
)
COUNT_ONLY = env_bool("FOOTPRINT_AWARE_SIF_COUNT_ONLY", COUNT_ONLY)
RUN_LABEL = os.environ.get("FOOTPRINT_AWARE_SIF_RUN_LABEL", "footprint_aware").strip()


# =============================================================================
# CONFIGURATION
# =============================================================================


def configure_modules() -> None:
    """Point reused modules at the footprint-aware output root and settings."""
    for module in (base,):
        module.DRY_RUN = DRY_RUN
        module.WRITE_OUTPUTS = WRITE_OUTPUTS
        module.OVERWRITE_EXISTING = OVERWRITE_EXISTING
        module.PRODUCTS_TO_PROCESS = PRODUCTS_TO_PROCESS
        module.SELECTED_SITES = SELECTED_SITES
        module.SELECTED_YYYYMM = SELECTED_YYYYMM
        module.NODATA_VALUE = NODATA_VALUE
        module.USE_GPP_TIFF_AS_ONLY_SPATIAL_TEMPLATE = (
            USE_GPP_TIFF_AS_ONLY_SPATIAL_TEMPLATE
        )
        module.USE_KMZ_FILES = USE_KMZ_FILES
        module.CREATE_QUICKLOOKS = CREATE_QUICKLOOKS
        module.CREATE_MD_AND_PDF_REPORTS = CREATE_MD_AND_PDF_REPORTS
        module.PROJECT_ROOT = PROJECT_ROOT
        module.GPP_TIFF_ROOT = GPP_TIFF_ROOT
        module.GPP_REFERENCE_CSV = GPP_REFERENCE_CSV
        module.SIF_ROOT = SIF_ROOT
        module.OUTPUT_ROOT = OUTPUT_ROOT
        module.COMPANION_ROOT = COMPANION_ROOT
        module.SUMMARY_ROOT = SUMMARY_ROOT
        module.QC_ROOT = QC_ROOT
        module.VALIDATION_ROOT = VALIDATION_ROOT
        module.QUICKLOOK_ROOT = QUICKLOOK_ROOT
        module.PRODUCTS = PRODUCTS
        module.SIF_BAND_DESCRIPTIONS = SIF_BAND_DESCRIPTIONS

    base.RESAMPLING_FOR_GRIDDED_SIF = GOSIF_GRIDDED_RESAMPLING
    base.SWATH_RASTERIZATION_METHOD = SWATH_RASTERIZATION_METHOD
    base.MULTIPLE_SWATH_POINTS_PER_CELL = MULTIPLE_SIF_FOOTPRINTS_PER_CELL


def initialize_output_directories() -> None:
    for path in (
        COMPANION_ROOT,
        SUMMARY_ROOT,
        QC_ROOT,
        VALIDATION_ROOT,
        QUICKLOOK_ROOT,
    ):
        base.ensure_directory(path)


def normalize_product_name(product: str) -> str:
    """Normalize user/env product names to the canonical internal product name.

    The production env var is intentionally forgiving: hyphens, case, and common
    output-token spellings such as OCO2/GOME2/GOSAT all resolve to the canonical
    source product used internally.
    """
    token = re.sub(r"[^a-z0-9]+", "", str(product).strip().lower())
    lookup = {
        "gosif": "GOSIF",
        "gome2": "GOME-2",
        "gome02": "GOME-2",
        "gosat": "gosat",
        "oco2": "OCO-2",
        "oco02": "OCO-2",
        "oco3": "OCO-3",
        "oco03": "OCO-3",
        "troposif": "TROPOSIF",
    }
    return lookup.get(token, str(product).strip())


def selected_products() -> list[str]:
    products = PRODUCTS_TO_PROCESS if PRODUCTS_TO_PROCESS is not None else PRODUCTS
    normalized = [normalize_product_name(product) for product in products]
    invalid = [product for product in normalized if product not in PRODUCTS]
    if invalid:
        raise ValueError(f"Unknown product(s): {invalid}")
    return list(dict.fromkeys(normalized))


def iter_progress(items, total: int | None, description: str, unit: str):
    if tqdm is None:
        return items
    return tqdm(items, total=total, desc=description, unit=unit)


def valid_sif_mask(array: np.ndarray) -> np.ndarray:
    return base.valid_data_mask(array, NODATA_VALUE)


def output_product_token(product: str) -> str:
    """Return the locked-spec output product folder/file token."""
    return {
        "GOME-2": "GOME2",
        "OCO-2": "OCO2",
        "OCO-3": "OCO3",
        "gosat": "GOSAT",
    }.get(product, product)


def output_path_for(match: dict[str, Any]) -> Path:
    product = output_product_token(match["product"])
    site_safe = match["SITE_SAFE"]
    return COMPANION_ROOT / product / f"{site_safe}_{match['YYYYMM']}_{product}_SIF.tif"


def source_paths(match: dict[str, Any]) -> list[str]:
    return [str(source["file_path"]) for source in match.get("sif_records", [])]


# =============================================================================
# SIX-BAND WRITING, VALIDATION, AND ANALYSIS
# =============================================================================


def write_footprint_aware_companion(
    match: dict[str, Any],
    sif_array: np.ndarray,
    output_path: Path,
    geometry_metrics: dict[str, Any] | None = None,
) -> None:
    """Write one locked-schema 8-band footprint-aware companion TIFF.

    This function refuses all-NoData Band 2 arrays.
    """
    if DRY_RUN or not WRITE_OUTPUTS:
        return
    if not bool(valid_sif_mask(sif_array).any()):
        raise ValueError("Refusing to write an all-NoData SIF companion TIFF.")

    geometry_metrics = geometry_metrics or {}
    gpp_path = Path(match["gpp_tif"])
    gpp_bands, profile, gpp_tags = base.read_gpp_arrays(gpp_path)
    output_nodata = profile.get("nodata")
    output_nodata = float(output_nodata) if output_nodata is not None else NODATA_VALUE
    footprint_probability = gpp_bands[0]
    blind_circular_buffer = gpp_bands[2]
    sif = sif_array.astype(np.float32, copy=False)

    band6 = np.full(sif.shape, output_nodata, dtype=np.float32)
    valid = (
        valid_sif_mask(sif)
        & np.isfinite(footprint_probability)
        & (footprint_probability > 0)
    )
    band6[valid] = (
        footprint_probability[valid].astype(np.float32)
        * sif[valid].astype(np.float32)
    )

    band7 = np.full(sif.shape, output_nodata, dtype=np.float32)
    band7[valid] = sif[valid].astype(np.float32)

    band8 = np.full(sif.shape, output_nodata, dtype=np.float32)
    valid_blind = (
        valid_sif_mask(sif)
        & np.isfinite(blind_circular_buffer)
        & (blind_circular_buffer == 1)
    )
    band8[valid_blind] = sif[valid_blind].astype(np.float32)

    profile.update(
        driver="GTiff",
        count=8,
        dtype="float32",
        nodata=output_nodata,
        compress="LZW",
    )
    base.ensure_directory(output_path.parent)
    temporary = output_path.with_name(output_path.name + ".tmp.tif")
    if base.path_exists(temporary):
        os.remove(base.native_path(temporary))

    try:
        with rasterio.open(base.native_path(temporary), "w", **profile) as dst:
            dst.write(gpp_bands[0], 1)
            dst.write(sif, 2)
            dst.write(gpp_bands[2], 3)
            dst.write(gpp_bands[3], 4)
            dst.write(gpp_bands[4], 5)
            dst.write(band6, 6)
            dst.write(band7, 7)
            dst.write(band8, 8)
            for index, description in enumerate(SIF_BAND_DESCRIPTIONS, start=1):
                dst.set_band_description(index, description)

            tags = {
                key: str(value)
                for key, value in gpp_tags.items()
                if value is not None
            }
            tags.update(
                {
                    "SITE_ID": match["SITE_ID"],
                    "SITE_SAFE": match["SITE_SAFE"],
                    "YYYYMM": match["YYYYMM"],
                    "product": match["product"],
                    "matching_gpp_tif": str(gpp_path),
                    "source_sif_files": base.compact_paths(source_paths(match)),
                    "target_grid": "matching_existing_GPP_TIFF",
                    "band_structure": (
                        "eight_band_SIF_companion_matching_GPP_TIFF"
                    ),
                    "band_2_definition": "raw_unmasked_SIF_on_GPP_grid",
                    "band_7_definition": "SIF_where_footprint_probability_gt_0",
                    "band_8_definition": "SIF_where_blind_circular_buffer_eq_1",
                    "used_kmz": "false",
                    "created_standalone_circular_buffer": "false",
                    "sif_values_preserved": "true",
                    "not_interpolated": "true",
                    "not_filled": "true",
                    "gridded_resampling": "nearest_if_used",
                    "swath_rasterization": (
                        "footprint_polygon_to_GPP_grid_cells"
                        if match["source_type"] == "point_or_swath"
                        else "not_used_for_gridded_product"
                    ),
                    "multiple_sif_footprints_per_cell": "mean",
                    "source_sif_type": match["source_type"],
                    "contains_actual_sif_observations": "true",
                    "n_source_sif_observations": str(
                        geometry_metrics.get("n_source_sif_observations", "")
                    ),
                    "n_sif_footprint_polygons": str(
                        geometry_metrics.get("n_sif_footprint_polygons", "")
                    ),
                    "n_gpp_cells_filled_by_sif_footprints": str(
                        geometry_metrics.get(
                            "n_gpp_cells_filled_by_sif_footprints", ""
                        )
                    ),
                    "n_candidate_sif_observations_for_site_grid": str(
                        geometry_metrics.get(
                            "n_candidate_sif_observations_for_site_grid", ""
                        )
                    ),
                    "n_candidate_sif_observations_skipped_bad_geometry": str(
                        geometry_metrics.get(
                            "n_candidate_sif_observations_skipped_bad_geometry", ""
                        )
                    ),
                    "n_candidate_sif_observations_no_spatial_intersection_with_grid": str(
                        geometry_metrics.get(
                            "n_candidate_sif_observations_no_spatial_intersection_with_grid",
                            "",
                        )
                    ),
                    "n_candidate_sif_observations_unexplained": str(
                        geometry_metrics.get(
                            "n_candidate_sif_observations_unexplained", ""
                        )
                    ),
                    "candidate_accounting_complete": str(
                        geometry_metrics.get("candidate_accounting_complete", "")
                    ),
                    "swath_geometry_fields_used": str(
                        geometry_metrics.get("swath_geometry_fields_used", "")
                    ),
                    "mean_gpp_cells_filled_per_sif_observation": str(
                        geometry_metrics.get(
                            "mean_gpp_cells_filled_per_sif_observation", ""
                        )
                    ),
                    "rasterize_all_touched": "true",
                    "created_by": (
                        "Create_FINAL_CLEAN_SIF_Companion_TIFFs_"
                        "FOOTPRINT_AWARE.py"
                    ),
                }
            )
            dst.update_tags(**tags)
        os.replace(base.native_path(temporary), base.native_path(output_path))
    except Exception:
        if base.path_exists(temporary):
            os.remove(base.native_path(temporary))
        raise


def validate_footprint_aware_companion(
    match: dict[str, Any],
    output_path: Path,
    expected_sif: np.ndarray | None = None,
    geometry_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "product": match["product"],
        "SITE_ID": match["SITE_ID"],
        "SITE_SAFE": match["SITE_SAFE"],
        "YYYYMM": match["YYYYMM"],
        "gpp_tif": match["gpp_tif"],
        "sif_companion_tif": str(output_path),
        "output_exists": base.path_exists(output_path),
        "band_count_is_8": False,
        "crs_exact_match": False,
        "transform_exact_match": False,
        "width_exact_match": False,
        "height_exact_match": False,
        "bounds_exact_match": False,
        "pixel_size_exact_match": False,
        "dtype_float32": False,
        "nodata_matches_gpp_or_default": False,
        "band_descriptions_exact": False,
        "band1_equals_gpp_band1": False,
        "band3_equals_gpp_band3": False,
        "band4_equals_gpp_band4": False,
        "band5_equals_gpp_band5": False,
        "band2_equals_created_source_array": "",
        "band2_has_actual_sif": False,
        "band6_formula_correct": False,
        "band7_mask_rule_correct": False,
        "band8_mask_rule_correct": False,
        "gridded_nearest_only": match["source_type"] == "gridded_raster",
        "swath_footprint_polygon_only": match["source_type"] == "point_or_swath",
        "validation_status": "FAIL",
        "failure_reasons": "",
    }
    geometry_metrics = geometry_metrics or {}
    row["swath_geometry_mode"] = (
        "footprint_polygon_to_GPP_grid_cells"
        if match["source_type"] == "point_or_swath"
        else "gridded_nearest"
    )
    row["n_source_sif_observations"] = geometry_metrics.get(
        "n_source_sif_observations", ""
    )
    row["n_sif_footprint_polygons"] = geometry_metrics.get(
        "n_sif_footprint_polygons", ""
    )
    row["n_gpp_cells_filled_by_sif_footprints"] = geometry_metrics.get(
        "n_gpp_cells_filled_by_sif_footprints", ""
    )
    row["n_candidate_sif_observations_for_site_grid"] = geometry_metrics.get(
        "n_candidate_sif_observations_for_site_grid", ""
    )
    row["n_candidate_sif_observations_skipped_bad_geometry"] = geometry_metrics.get(
        "n_candidate_sif_observations_skipped_bad_geometry", ""
    )
    row[
        "n_candidate_sif_observations_no_spatial_intersection_with_grid"
    ] = geometry_metrics.get(
        "n_candidate_sif_observations_no_spatial_intersection_with_grid", ""
    )
    row["n_candidate_sif_observations_unexplained"] = geometry_metrics.get(
        "n_candidate_sif_observations_unexplained", ""
    )
    row["candidate_accounting_complete"] = geometry_metrics.get(
        "candidate_accounting_complete", ""
    )
    row["swath_geometry_fields_used"] = geometry_metrics.get(
        "swath_geometry_fields_used", ""
    )
    row["mean_gpp_cells_filled_per_sif_observation"] = geometry_metrics.get(
        "mean_gpp_cells_filled_per_sif_observation", ""
    )
    row["footprint_aware_rasterization_confirmed"] = (
        match["source_type"] == "gridded_raster"
        or row["swath_geometry_mode"] == "footprint_polygon_to_GPP_grid_cells"
    )

    failures = []
    try:
        if not base.path_exists(output_path):
            raise FileNotFoundError(output_path)
        with rasterio.open(base.native_path(Path(match["gpp_tif"]))) as gpp, rasterio.open(
            base.native_path(output_path)
        ) as dataset:
            row["band_count_is_8"] = dataset.count == 8
            row["crs_exact_match"] = dataset.crs == gpp.crs
            row["transform_exact_match"] = dataset.transform == gpp.transform
            row["width_exact_match"] = dataset.width == gpp.width
            row["height_exact_match"] = dataset.height == gpp.height
            row["bounds_exact_match"] = tuple(dataset.bounds) == tuple(gpp.bounds)
            row["pixel_size_exact_match"] = (
                abs(dataset.transform.a) == abs(gpp.transform.a)
                and abs(dataset.transform.e) == abs(gpp.transform.e)
            )
            row["dtype_float32"] = all(dtype == "float32" for dtype in dataset.dtypes)
            expected_nodata = gpp.nodata if gpp.nodata is not None else NODATA_VALUE
            row["nodata_matches_gpp_or_default"] = dataset.nodata == expected_nodata
            nodata = float(dataset.nodata if dataset.nodata is not None else NODATA_VALUE)
            row["band_descriptions_exact"] = (
                tuple(dataset.descriptions) == SIF_BAND_DESCRIPTIONS
            )

            out1 = dataset.read(1)
            out2 = dataset.read(2)
            out3 = dataset.read(3)
            out4 = dataset.read(4)
            out5 = dataset.read(5)
            out6 = dataset.read(6)
            out7 = dataset.read(7)
            out8 = dataset.read(8)

            row["band1_equals_gpp_band1"] = base.values_equal(out1, gpp.read(1))
            row["band3_equals_gpp_band3"] = base.values_equal(out3, gpp.read(3))
            row["band4_equals_gpp_band4"] = base.values_equal(out4, gpp.read(4))
            row["band5_equals_gpp_band5"] = base.values_equal(out5, gpp.read(5))
            row["band2_has_actual_sif"] = bool(valid_sif_mask(out2).any())
            if expected_sif is not None:
                row["band2_equals_created_source_array"] = base.values_equal(
                    out2, expected_sif
                )

            valid = valid_sif_mask(out2) & np.isfinite(out1) & (out1 > 0)
            expected6 = np.full(out2.shape, nodata, dtype=np.float32)
            expected6[valid] = out1[valid].astype(np.float32) * out2[valid].astype(
                np.float32
            )
            row["band6_formula_correct"] = base.values_equal(out6, expected6)

            expected7 = np.full(out2.shape, nodata, dtype=np.float32)
            expected7[valid] = out2[valid].astype(np.float32)
            row["band7_mask_rule_correct"] = base.values_equal(out7, expected7)

            valid_blind = valid_sif_mask(out2) & np.isfinite(out3) & (out3 == 1)
            expected8 = np.full(out2.shape, nodata, dtype=np.float32)
            expected8[valid_blind] = out2[valid_blind].astype(np.float32)
            row["band8_mask_rule_correct"] = base.values_equal(out8, expected8)

            with rasterio.open(base.native_path(output_path)) as dataset:
                tags = dataset.tags()
            if (
                match["source_type"] == "point_or_swath"
                and tags.get("swath_rasterization")
                != "footprint_polygon_to_GPP_grid_cells"
            ):
                failures.append("swath_rasterization tag is not footprint-aware")
            if tags.get("used_kmz") != "false":
                failures.append("used_kmz tag is not false")
            if tags.get("created_standalone_circular_buffer") != "false":
                failures.append("standalone circular buffer tag is not false")
    except Exception as error:
        failures.append(f"{type(error).__name__}: {error}")

    required = [
        "band_count_is_8",
        "crs_exact_match",
        "transform_exact_match",
        "width_exact_match",
        "height_exact_match",
        "bounds_exact_match",
        "pixel_size_exact_match",
        "dtype_float32",
        "nodata_matches_gpp_or_default",
        "band_descriptions_exact",
        "band1_equals_gpp_band1",
        "band3_equals_gpp_band3",
        "band4_equals_gpp_band4",
        "band5_equals_gpp_band5",
        "band2_has_actual_sif",
        "band6_formula_correct",
        "band7_mask_rule_correct",
        "band8_mask_rule_correct",
    ]
    if expected_sif is not None:
        required.append("band2_equals_created_source_array")
    failures.extend(field for field in required if not bool(row.get(field)))
    row["failure_reasons"] = "; ".join(dict.fromkeys(failures))
    row["validation_status"] = "PASS" if not failures else "FAIL"
    return row


def analysis_row(
    match: dict[str, Any],
    sif_array: np.ndarray,
    output_path: Path,
    csv_lookup: dict[tuple[str, str], dict[str, Any]],
    status: str,
    notes: str,
    geometry_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = base.calculate_analysis_row(
        match, sif_array, output_path, csv_lookup, status, notes
    )
    geometry_metrics = geometry_metrics or {}
    row["final_sif_companion_tif"] = str(output_path)
    row["swath_geometry_mode"] = (
        "footprint_polygon_to_GPP_grid_cells"
        if match["source_type"] == "point_or_swath"
        else "gridded_nearest"
    )
    row["n_source_sif_observations"] = geometry_metrics.get(
        "n_source_sif_observations", ""
    )
    row["n_sif_footprint_polygons"] = geometry_metrics.get(
        "n_sif_footprint_polygons", ""
    )
    row["n_candidate_sif_observations_for_site_grid"] = geometry_metrics.get(
        "n_candidate_sif_observations_for_site_grid", ""
    )
    row["n_candidate_sif_observations_skipped_bad_geometry"] = geometry_metrics.get(
        "n_candidate_sif_observations_skipped_bad_geometry", ""
    )
    row[
        "n_candidate_sif_observations_no_spatial_intersection_with_grid"
    ] = geometry_metrics.get(
        "n_candidate_sif_observations_no_spatial_intersection_with_grid", ""
    )
    row["n_candidate_sif_observations_unexplained"] = geometry_metrics.get(
        "n_candidate_sif_observations_unexplained", ""
    )
    row["candidate_accounting_complete"] = geometry_metrics.get(
        "candidate_accounting_complete", ""
    )
    row["swath_geometry_fields_used"] = geometry_metrics.get(
        "swath_geometry_fields_used", ""
    )
    row["mean_gpp_cells_filled_per_sif_observation"] = geometry_metrics.get(
        "mean_gpp_cells_filled_per_sif_observation", ""
    )
    return row


def skipped_row(
    match_or_row: dict[str, Any],
    reason: str,
    notes: str,
    source_files: str | None = None,
    error_message: str = "",
) -> dict[str, Any]:
    site_id = str(match_or_row.get("SITE_ID", ""))
    yyyymm = str(match_or_row.get("YYYYMM", ""))
    product = str(match_or_row.get("product", ""))
    source_text = (
        source_files
        if source_files is not None
        else base.compact_paths(source_paths(match_or_row))
    )
    return {
        "product": product,
        "SITE_ID": site_id,
        "SITE_SAFE": str(match_or_row.get("SITE_SAFE", base.site_to_safe(site_id))),
        "YYYYMM": yyyymm,
        "gpp_tif": str(match_or_row.get("gpp_tif", "")),
        "source_sif_files": source_text,
        "skip_reason": reason,
        "status": reason,
        "notes": notes,
        "error_message": error_message,
    }


def geometry_metrics_from_existing_tif(path: Path) -> dict[str, Any]:
    """Recover swath/gridded provenance metrics from an existing protected TIFF."""
    if not base.path_exists(path):
        return {}
    with rasterio.open(base.native_path(path)) as dataset:
        tags = dataset.tags()

    def numeric_tag(key: str) -> Any:
        value = tags.get(key, "")
        if value in ("", None, "None"):
            return ""
        try:
            number = float(value)
            return int(number) if number.is_integer() else number
        except Exception:
            return value

    return {
        "n_source_sif_observations": numeric_tag("n_source_sif_observations"),
        "n_sif_footprint_polygons": numeric_tag("n_sif_footprint_polygons"),
        "n_gpp_cells_filled_by_sif_footprints": numeric_tag(
            "n_gpp_cells_filled_by_sif_footprints"
        ),
        "n_candidate_sif_observations_for_site_grid": numeric_tag(
            "n_candidate_sif_observations_for_site_grid"
        ),
        "n_candidate_sif_observations_skipped_bad_geometry": numeric_tag(
            "n_candidate_sif_observations_skipped_bad_geometry"
        ),
        "n_candidate_sif_observations_no_spatial_intersection_with_grid": numeric_tag(
            "n_candidate_sif_observations_no_spatial_intersection_with_grid"
        ),
        "n_candidate_sif_observations_unexplained": numeric_tag(
            "n_candidate_sif_observations_unexplained"
        ),
        "candidate_accounting_complete": tags.get(
            "candidate_accounting_complete", ""
        ),
        "mean_gpp_cells_filled_per_sif_observation": numeric_tag(
            "mean_gpp_cells_filled_per_sif_observation"
        ),
        "swath_geometry_fields_used": tags.get("swath_geometry_fields_used", ""),
    }


def handle_array_for_match(
    match: dict[str, Any],
    sif_array: np.ndarray,
    csv_lookup: dict[tuple[str, str], dict[str, Any]],
    processing_notes: str,
    geometry_metrics: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    valid_count = int(valid_sif_mask(sif_array).sum())
    output_path = output_path_for(match)
    geometry_metrics = geometry_metrics or {}

    if valid_count == 0:
        reason = (
            "NO_ACTUAL_SIF_FOOTPRINT_OVERLAP_WITH_GPP_GRID"
            if match["source_type"] == "point_or_swath"
            else "SKIPPED_NO_ACTUAL_SIF_OBSERVATION_IN_GPP_GRID"
        )
        skip = skipped_row(
            match,
            reason,
            (
                "Readable local SIF month exists, but no valid SIF footprint/"
                "pixel overlaps this exact GPP grid. No all-NoData TIFF was "
                "created."
            ),
        )
        return None, None, skip

    if base.path_exists(output_path) and not OVERWRITE_EXISTING:
        validation = validate_footprint_aware_companion(
            match, output_path, geometry_metrics=geometry_metrics
        )
        if validation.get("validation_status") == "PASS":
            band2 = base.read_existing_band2(output_path)
            analysis = analysis_row(
                match,
                band2,
                output_path,
                csv_lookup,
                "SKIPPED_EXISTING",
                "Existing footprint-aware output was protected and validated.",
                geometry_metrics,
            )
            analysis["_action"] = "SKIPPED_EXISTING"
            return analysis, validation, None
        skip = skipped_row(
            match,
            "FAILED_EXISTING_OUTPUT_VALIDATION",
            "Existing output was protected and not overwritten.",
            error_message=str(validation.get("failure_reasons", "")),
        )
        return None, validation, skip

    if not DRY_RUN and WRITE_OUTPUTS:
        write_footprint_aware_companion(
            match, sif_array, output_path, geometry_metrics
        )
        action = "CREATED"
    else:
        action = "DRY_RUN_WOULD_CREATE"

    validation = (
        validate_footprint_aware_companion(
            match, output_path, expected_sif=sif_array, geometry_metrics=geometry_metrics
        )
        if base.path_exists(output_path)
        else {
            "product": match["product"],
            "SITE_ID": match["SITE_ID"],
            "YYYYMM": match["YYYYMM"],
            "gpp_tif": match["gpp_tif"],
            "sif_companion_tif": str(output_path),
            "output_exists": False,
            "validation_status": "NOT_RUN_DRY_RUN",
            "failure_reasons": "",
        }
    )
    analysis = analysis_row(
        match,
        sif_array,
        output_path,
        csv_lookup,
        "OK_HAS_VALID_SIF" if action == "CREATED" else action,
        f"{action}. {processing_notes}",
        geometry_metrics,
    )
    analysis["_action"] = action
    return analysis, validation, None


# =============================================================================
# SWATH FOOTPRINT GEOMETRY DETECTION AND RASTERIZATION
# =============================================================================


def all_dataset_paths(handle: h5py.File) -> list[str]:
    paths: list[str] = []

    def visitor(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            paths.append(name)

    handle.visititems(visitor)
    return paths


def lower_path_lookup(paths: Iterable[str]) -> dict[str, str]:
    return {path.lower(): path for path in paths}


def geometry_variable_candidates(paths: list[str]) -> tuple[list[str], list[str]]:
    lat_candidates: list[str] = []
    lon_candidates: list[str] = []
    for path in paths:
        lower = path.lower()
        has_geometry_token = any(token in lower for token in FOOTPRINT_GEOMETRY_TOKENS)
        if not has_geometry_token:
            continue
        if "lat" in lower or "latitude" in lower:
            lat_candidates.append(path)
        if "lon" in lower or "long" in lower or "longitude" in lower:
            lon_candidates.append(path)
    return lat_candidates, lon_candidates


def geometry_score(path: str) -> int:
    lower = path.lower()
    score = 0
    if "footprint" in lower:
        score += 40
    if "corner" in lower or "corners" in lower:
        score += 35
    if "vertices" in lower or "vertex" in lower:
        score += 35
    if "bounds" in lower or "bound" in lower:
        score += 30
    if "geolocation" in lower or "geolocations" in lower:
        score += 10
    if "support_data" in lower:
        score += 5
    return score


def compatible_vertex_shape(shape: tuple[int, ...], n_observations: int) -> bool:
    if len(shape) < 2:
        return False
    squeezed = tuple(dim for dim in shape if dim != 1)
    if len(squeezed) != 2:
        return False
    return bool(
        (squeezed[0] == n_observations and squeezed[1] >= 2)
        or (squeezed[1] == n_observations and squeezed[0] >= 2)
    )


def shape_compatible_pair(
    handle: h5py.File,
    lat_path: str,
    lon_path: str,
    n_observations: int,
) -> bool:
    try:
        lat_shape = tuple(handle[lat_path].shape)
        lon_shape = tuple(handle[lon_path].shape)
    except Exception:
        return False
    return (
        compatible_vertex_shape(lat_shape, n_observations)
        and compatible_vertex_shape(lon_shape, n_observations)
        and tuple(dim for dim in lat_shape if dim != 1)
        == tuple(dim for dim in lon_shape if dim != 1)
    )


def detect_footprint_geometry_pair(
    handle: h5py.File,
    source: dict[str, Any],
) -> tuple[str, str, str]:
    """Return lat/lon footprint-geometry dataset paths or blanks."""
    paths = all_dataset_paths(handle)
    lookup = lower_path_lookup(paths)
    n_observations = int(handle[source["detected_sif_variable"]].shape[0])

    for lat_candidate, lon_candidate in EXPLICIT_FOOTPRINT_GEOMETRY_PAIRS:
        lat_path = lookup.get(lat_candidate.lower())
        lon_path = lookup.get(lon_candidate.lower())
        if lat_path and lon_path and shape_compatible_pair(
            handle, lat_path, lon_path, n_observations
        ):
            return lat_path, lon_path, "explicit_candidate_pair"

    lat_candidates, lon_candidates = geometry_variable_candidates(paths)
    possible: list[tuple[int, str, str]] = []
    for lat_path in lat_candidates:
        for lon_path in lon_candidates:
            if not shape_compatible_pair(handle, lat_path, lon_path, n_observations):
                continue
            score = geometry_score(lat_path) + geometry_score(lon_path)
            lat_parent = lat_path.rsplit("/", 1)[0] if "/" in lat_path else ""
            lon_parent = lon_path.rsplit("/", 1)[0] if "/" in lon_path else ""
            if lat_parent == lon_parent:
                score += 20
            possible.append((score, lat_path, lon_path))
    if possible:
        _, lat_path, lon_path = sorted(
            possible, key=lambda item: (-item[0], item[1], item[2])
        )[0]
        return lat_path, lon_path, "recursive_geometry_variable_search"

    available = " | ".join(sorted(lat_candidates + lon_candidates)[:40])
    return "", "", available


def numeric_dataset_array(dataset: h5py.Dataset) -> np.ndarray:
    values = np.asarray(dataset[...]).astype(np.float64, copy=False)
    fill = sif_core.safe_float(sif_core.dataset_attr(dataset, ["_FillValue", "missing_value"]))
    invalid = ~np.isfinite(values)
    if fill is not None:
        invalid |= values == fill
    scale = sif_core.safe_float(sif_core.dataset_attr(dataset, ["scale_factor"]))
    offset = sif_core.safe_float(sif_core.dataset_attr(dataset, ["add_offset"]))
    scale = 1.0 if scale is None else scale
    offset = 0.0 if offset is None else offset
    values = values * scale + offset
    values[invalid] = np.nan
    return values


def vertices_as_observation_rows(
    array: np.ndarray,
    n_observations: int,
) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64).squeeze()
    if values.ndim != 2:
        raise ValueError(f"Footprint geometry array is not 2-D after squeeze: {values.shape}")
    if values.shape[0] == n_observations:
        rows = values
    elif values.shape[1] == n_observations:
        rows = values.T
    else:
        raise ValueError(
            f"Footprint geometry shape {values.shape} does not match {n_observations} observations."
        )
    if rows.shape[1] == 2:
        return rows
    if rows.shape[1] >= 3:
        return rows
    raise ValueError(f"Need at least two bounds or three polygon vertices, got {rows.shape[1]}.")


def bounds_rows_to_four_vertices(lat_rows: np.ndarray, lon_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert two-value lat/lon bounds into rectangle vertices if needed."""
    if lat_rows.shape[1] != 2 or lon_rows.shape[1] != 2:
        return lat_rows, lon_rows
    lat_min = np.nanmin(lat_rows, axis=1)
    lat_max = np.nanmax(lat_rows, axis=1)
    lon_min = np.nanmin(lon_rows, axis=1)
    lon_max = np.nanmax(lon_rows, axis=1)
    lat_vertices = np.column_stack([lat_min, lat_min, lat_max, lat_max])
    lon_vertices = np.column_stack([lon_min, lon_max, lon_max, lon_min])
    return lat_vertices, lon_vertices


def normalize_longitudes(lon: np.ndarray) -> np.ndarray:
    values = np.asarray(lon, dtype=np.float64)
    return ((values + 180.0) % 360.0) - 180.0


def make_polygon_from_vertices(lon_values: np.ndarray, lat_values: np.ndarray) -> Polygon | None:
    lon_values = normalize_longitudes(lon_values)
    lat_values = np.asarray(lat_values, dtype=np.float64)
    finite = np.isfinite(lon_values) & np.isfinite(lat_values)
    if finite.sum() < 3:
        return None
    coords = [(float(x), float(y)) for x, y in zip(lon_values[finite], lat_values[finite])]
    # Remove consecutive duplicates without changing the product-provided order.
    deduped: list[tuple[float, float]] = []
    for coord in coords:
        if not deduped or coord != deduped[-1]:
            deduped.append(coord)
    if len(set(deduped)) < 3:
        return None
    polygon = Polygon(deduped)
    if polygon.is_empty:
        return None
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty or not polygon.is_valid:
        return None
    return polygon


def template_geographic_bounds(template: dict[str, Any]) -> tuple[float, float, float, float]:
    return transform_bounds(
        template["_crs_obj"],
        CRS.from_epsg(4326),
        *template["_bounds_tuple"],
        densify_pts=21,
    )


def bounds_to_cell_keys(bounds: tuple[float, float, float, float]) -> set[int]:
    left, bottom, right, top = bounds
    left = max(-180.0, min(180.0, float(left)))
    right = max(-180.0, min(180.0, float(right)))
    bottom = max(-90.0, min(90.0, float(bottom)))
    top = max(-90.0, min(90.0, float(top)))
    lon_start = math.floor(left)
    lon_end = math.floor(np.nextafter(right, -np.inf))
    lat_start = math.floor(bottom)
    lat_end = math.floor(np.nextafter(top, -np.inf))
    keys: set[int] = set()
    for lat_cell in range(lat_start, lat_end + 1):
        for lon_cell in range(lon_start, lon_end + 1):
            keys.add(int((lat_cell + 90) * 361 + (lon_cell + 180)))
    return keys


def bbox_intersects(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> bool:
    return not (
        first[2] < second[0]
        or first[0] > second[2]
        or first[3] < second[1]
        or first[1] > second[3]
    )


def source_missing_geometry_report_row(
    source: dict[str, Any],
    detected_lat: str,
    detected_lon: str,
    details: str,
) -> dict[str, Any]:
    return {
        "product": source.get("product", ""),
        "file_path": source.get("file_path", ""),
        "YYYYMM": source.get("detected_YYYYMM", ""),
        "detected_lat_variable": detected_lat or source.get("detected_lat_variable", ""),
        "detected_lon_variable": detected_lon or source.get("detected_lon_variable", ""),
        "detected_sif_variable": source.get("detected_sif_variable", ""),
        "missing_footprint_geometry_reason": (
            "No usable footprint/corner/bounds/vertices latitude-longitude pair "
            "was found with one row per SIF observation."
        ),
        "available_coordinate_or_geometry_variables": details,
        "recommendation": (
            "Use a product file containing footprint corner/vertex/bounds "
            "geometry, or review product documentation before any non-point "
            "fallback is implemented."
        ),
    }


def build_footprint_context(match: dict[str, Any]) -> dict[str, Any]:
    template = match["template"]
    height = int(template["height"])
    width = int(template["width"])
    geographic_bounds = template_geographic_bounds(template)
    transformer = Transformer.from_crs(
        CRS.from_epsg(4326),
        template["_crs_obj"],
        always_xy=True,
    )
    return {
        "match": match,
        "sum": np.zeros((height, width), dtype=np.float64),
        "count": np.zeros((height, width), dtype=np.float32),
        "geographic_bounds": geographic_bounds,
        "geographic_cell_keys": bounds_to_cell_keys(geographic_bounds),
        "transformer": transformer,
        "n_source_sif_observations": 0,
        "n_sif_footprint_polygons": 0,
        "n_gpp_cells_filled_by_sif_footprints": 0,
        "n_candidate_sif_observations_for_site_grid": 0,
        "n_candidate_sif_observations_skipped_bad_geometry": 0,
        "n_candidate_sif_observations_no_spatial_intersection_with_grid": 0,
        "source_files_used": set(),
        "geometry_fields_used": set(),
    }


def context_sif_array_and_metrics(context: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    match = context["match"]
    template = match["template"]
    array = np.full(
        (template["height"], template["width"]),
        NODATA_VALUE,
        dtype=np.float32,
    )
    valid = context["count"] > 0
    array[valid] = (context["sum"][valid] / context["count"][valid]).astype(np.float32)
    polygons = int(context["n_sif_footprint_polygons"])
    cells_raw = int(context["n_gpp_cells_filled_by_sif_footprints"])
    metrics = {
        "n_source_sif_observations": int(context["n_source_sif_observations"]),
        "n_sif_footprint_polygons": polygons,
        "n_gpp_cells_filled_by_sif_footprints": cells_raw,
        "n_candidate_sif_observations_for_site_grid": int(
            context["n_candidate_sif_observations_for_site_grid"]
        ),
        "n_candidate_sif_observations_skipped_bad_geometry": int(
            context["n_candidate_sif_observations_skipped_bad_geometry"]
        ),
        "n_candidate_sif_observations_no_spatial_intersection_with_grid": int(
            context["n_candidate_sif_observations_no_spatial_intersection_with_grid"]
        ),
        "mean_gpp_cells_filled_per_sif_observation": (
            float(cells_raw / polygons) if polygons > 0 else 0.0
        ),
        "source_sif_files_used": base.compact_paths(context["source_files_used"]),
        "swath_geometry_fields_used": "; ".join(
            sorted(str(item) for item in context["geometry_fields_used"])
        ),
    }
    metrics["n_candidate_sif_observations_unexplained"] = max(
        int(metrics["n_candidate_sif_observations_for_site_grid"])
        - int(metrics["n_sif_footprint_polygons"])
        - int(metrics["n_candidate_sif_observations_skipped_bad_geometry"])
        - int(metrics["n_candidate_sif_observations_no_spatial_intersection_with_grid"]),
        0,
    )
    metrics["candidate_accounting_complete"] = (
        int(metrics["n_candidate_sif_observations_unexplained"]) == 0
    )
    return array, metrics


def append_footprint_polygons_from_source(
    source: dict[str, Any],
    contexts: list[dict[str, Any]],
    missing_geometry_rows: list[dict[str, Any]],
) -> tuple[int, str]:
    """Read one swath file and burn footprint polygons into each site grid.

    Returns
    -------
    tuple
        (number_of_contexts_updated, status)
    """
    if not contexts:
        return 0, "NO_CONTEXTS"

    product = str(source["product"])
    file_path = Path(source["file_path"])
    updated_contexts = 0

    context_keys: dict[int, list[dict[str, Any]]] = defaultdict(list)
    target_keys: set[int] = set()
    for context in contexts:
        for key in context["geographic_cell_keys"]:
            context_keys[key].append(context)
            target_keys.add(key)
    if not target_keys:
        return 0, "NO_TARGET_CELLS"

    with h5py.File(base.native_path(file_path), "r") as handle:
        lat_path = str(source.get("detected_lat_variable", ""))
        lon_path = str(source.get("detected_lon_variable", ""))
        sif_path = str(source.get("detected_sif_variable", ""))
        if not sif_path or sif_path not in handle:
            raise ValueError("Detected SIF variable is missing from file.")

        geom_lat_path, geom_lon_path, detection_method = detect_footprint_geometry_pair(
            handle, source
        )
        if not geom_lat_path or not geom_lon_path:
            missing_geometry_rows.append(
                source_missing_geometry_report_row(
                    source, lat_path, lon_path, detection_method
                )
            )
            return 0, "PRODUCT_LACKS_FOOTPRINT_GEOMETRY"

        n_observations = int(handle[sif_path].shape[0])
        sif_values = sif_core.dataset_chunk_values(
            handle[sif_path],
            0,
            n_observations,
            "mean",
        )
        lat_vertices = vertices_as_observation_rows(
            numeric_dataset_array(handle[geom_lat_path]),
            n_observations,
        )
        lon_vertices = vertices_as_observation_rows(
            numeric_dataset_array(handle[geom_lon_path]),
            n_observations,
        )
        lat_vertices, lon_vertices = bounds_rows_to_four_vertices(
            lat_vertices, lon_vertices
        )
        lon_vertices = normalize_longitudes(lon_vertices)

    valid_sif = np.isfinite(sif_values) & (sif_values > -9000)
    finite_vertices = np.isfinite(lat_vertices).all(axis=1) & np.isfinite(lon_vertices).all(axis=1)
    lon_min = np.full(lon_vertices.shape[0], np.nan, dtype=np.float64)
    lon_max = np.full(lon_vertices.shape[0], np.nan, dtype=np.float64)
    lat_min = np.full(lat_vertices.shape[0], np.nan, dtype=np.float64)
    lat_max = np.full(lat_vertices.shape[0], np.nan, dtype=np.float64)
    if bool(finite_vertices.any()):
        lon_min[finite_vertices] = np.min(lon_vertices[finite_vertices], axis=1)
        lon_max[finite_vertices] = np.max(lon_vertices[finite_vertices], axis=1)
        lat_min[finite_vertices] = np.min(lat_vertices[finite_vertices], axis=1)
        lat_max[finite_vertices] = np.max(lat_vertices[finite_vertices], axis=1)
    valid_bbox = (
        finite_vertices
        & (lat_max >= -90)
        & (lat_min <= 90)
        & (lon_max >= -180)
        & (lon_min <= 180)
    )
    candidates = np.flatnonzero(valid_sif & valid_bbox)
    if candidates.size == 0:
        return 0, "NO_VALID_SOURCE_FOOTPRINTS"

    # Coarse prefilter: footprint polygon bbox must share at least one 1-degree
    # cell with at least one selected GPP template.  This is only a speed filter;
    # actual output values still come from polygon-to-grid rasterization.
    possible_indices: list[int] = []
    for index in candidates:
        bbox = (
            float(lon_min[index]),
            float(lat_min[index]),
            float(lon_max[index]),
            float(lat_max[index]),
        )
        if bounds_to_cell_keys(bbox) & target_keys:
            possible_indices.append(int(index))
    if not possible_indices:
        return 0, "NO_BBOX_OVERLAP_WITH_SELECTED_SITES"

    for index in possible_indices:
        source_bbox = (
            float(lon_min[index]),
            float(lat_min[index]),
            float(lon_max[index]),
            float(lat_max[index]),
        )
        polygon = make_polygon_from_vertices(lon_vertices[index], lat_vertices[index])
        if polygon is None:
            continue
        value = float(sif_values[index])
        if not math.isfinite(value) or value <= -9000:
            continue

        possible_contexts: set[int] = set()
        for key in bounds_to_cell_keys(source_bbox):
            for context in context_keys.get(key, []):
                possible_contexts.add(id(context))
        context_by_id = {id(context): context for context in contexts}

        for context_id in possible_contexts:
            context = context_by_id[context_id]
            if not bbox_intersects(source_bbox, context["geographic_bounds"]):
                continue
            context["n_candidate_sif_observations_for_site_grid"] += 1
            context["geometry_fields_used"].add(
                f"{geom_lat_path} | {geom_lon_path}"
            )
            try:
                polygon_projected = shapely_transform(
                    context["transformer"].transform,
                    polygon,
                )
                if polygon_projected.is_empty:
                    context["n_candidate_sif_observations_skipped_bad_geometry"] += 1
                    continue
                if not polygon_projected.is_valid:
                    polygon_projected = polygon_projected.buffer(0)
                if polygon_projected.is_empty:
                    context["n_candidate_sif_observations_skipped_bad_geometry"] += 1
                    continue
                template = context["match"]["template"]
                mask = rasterize(
                    [(polygon_projected, 1)],
                    out_shape=(template["height"], template["width"]),
                    transform=template["_transform_obj"],
                    fill=0,
                    all_touched=True,
                    dtype="uint8",
                )
                filled = mask.astype(bool)
                n_filled = int(filled.sum())
                if n_filled == 0:
                    context[
                        "n_candidate_sif_observations_no_spatial_intersection_with_grid"
                    ] += 1
                    continue
                context["sum"][filled] += value
                context["count"][filled] += 1
                context["n_source_sif_observations"] += 1
                context["n_sif_footprint_polygons"] += 1
                context["n_gpp_cells_filled_by_sif_footprints"] += n_filled
                context["source_files_used"].add(str(file_path))
                updated_contexts += 1
            except Exception:
                # A single malformed source footprint should not block the
                # remaining observations in the file.
                context["n_candidate_sif_observations_skipped_bad_geometry"] += 1
                continue

    return updated_contexts, "OK"


# =============================================================================
# PROCESSING
# =============================================================================


def process_gridded_matches(
    matches: list[dict[str, Any]],
    csv_lookup: dict[tuple[str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    analysis_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for match in matches:
        grouped[(match["product"], match["YYYYMM"])].append(match)

    groups = sorted(grouped.items())
    for (_, _), month_matches in iter_progress(
        groups,
        total=len(groups),
        description="Processing footprint-aware gridded SIF months",
        unit="month",
    ):
        sources = month_matches[0]["sif_records"]
        if len(sources) != 1:
            reason = (
                "Multiple gridded files exist for one product-month; this "
                "workflow does not perform unrequested temporal averaging."
            )
            for match in month_matches:
                skipped_rows.append(skipped_row(match, "FAILED_SOURCE_READ", reason))
            continue
        source = sources[0]
        for match in month_matches:
            try:
                output_path = output_path_for(match)
                metrics = {
                    "n_source_sif_observations": "",
                    "n_sif_footprint_polygons": "",
                    "n_gpp_cells_filled_by_sif_footprints": "",
                    "mean_gpp_cells_filled_per_sif_observation": "",
                }
                if base.path_exists(output_path) and not OVERWRITE_EXISTING:
                    sif_array = base.read_existing_band2(output_path)
                    notes = "Existing output protected; read Band 2 for validation."
                else:
                    sif_array = base.align_one_gridded_source(source, match["template"])
                    notes = "Original gridded source aligned with nearest-neighbour only."
                analysis, validation, skip = handle_array_for_match(
                    match, sif_array, csv_lookup, notes, metrics
                )
                if analysis is not None:
                    analysis_rows.append(analysis)
                if validation is not None:
                    validation_rows.append(validation)
                if skip is not None:
                    skipped_rows.append(skip)
            except Exception as error:
                failure_rows.append(
                    {
                        "product": match["product"],
                        "SITE_ID": match["SITE_ID"],
                        "SITE_SAFE": match["SITE_SAFE"],
                        "YYYYMM": match["YYYYMM"],
                        "gpp_tif": match["gpp_tif"],
                        "stage": "GRIDDED_PROCESSING",
                        "error_message": (
                            f"{type(error).__name__}: {error}\n"
                            f"{traceback.format_exc()}"
                        ),
                    }
                )
                skipped_rows.append(
                    skipped_row(
                        match,
                        "FAILED_SOURCE_READ",
                        "Failed while aligning gridded SIF to the GPP grid.",
                        error_message=f"{type(error).__name__}: {error}",
                    )
                )
    return analysis_rows, validation_rows, skipped_rows, failure_rows


def process_footprint_aware_swath_matches(
    matches: list[dict[str, Any]],
    csv_lookup: dict[tuple[str, str], dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    analysis_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    missing_geometry_rows: list[dict[str, Any]] = []

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for match in matches:
        grouped[(match["product"], match["YYYYMM"])].append(match)

    groups = sorted(grouped.items())
    for (product, yyyymm), month_matches in iter_progress(
        groups,
        total=len(groups),
        description="Processing footprint-aware swath SIF months",
        unit="month",
    ):
        contexts: list[dict[str, Any]] = []
        existing_matches: list[dict[str, Any]] = []
        for match in month_matches:
            output_path = output_path_for(match)
            if base.path_exists(output_path) and not OVERWRITE_EXISTING:
                existing_matches.append(match)
                continue
            contexts.append(build_footprint_context(match))

        for match in existing_matches:
            try:
                output_path = output_path_for(match)
                sif_array = base.read_existing_band2(output_path)
                existing_metrics = geometry_metrics_from_existing_tif(output_path)
                analysis, validation, skip = handle_array_for_match(
                    match,
                    sif_array,
                    csv_lookup,
                    "Existing footprint-aware output protected; read Band 2 for validation.",
                    existing_metrics,
                )
                if analysis is not None:
                    analysis_rows.append(analysis)
                if validation is not None:
                    validation_rows.append(validation)
                if skip is not None:
                    skipped_rows.append(skip)
            except Exception as error:
                failure_rows.append(
                    {
                        "product": match["product"],
                        "SITE_ID": match["SITE_ID"],
                        "SITE_SAFE": match["SITE_SAFE"],
                        "YYYYMM": match["YYYYMM"],
                        "gpp_tif": match["gpp_tif"],
                        "stage": "EXISTING_OUTPUT_VALIDATION",
                        "error_message": (
                            f"{type(error).__name__}: {error}\n"
                            f"{traceback.format_exc()}"
                        ),
                    }
                )

        if not contexts:
            continue

        source_statuses: list[str] = []
        for source in month_matches[0]["sif_records"]:
            try:
                _, status = append_footprint_polygons_from_source(
                    source, contexts, missing_geometry_rows
                )
                source_statuses.append(status)
            except Exception as error:
                source_statuses.append(f"FAILED_SOURCE_READ: {type(error).__name__}: {error}")
                failure_rows.append(
                    {
                        "product": product,
                        "SITE_ID": "",
                        "SITE_SAFE": "",
                        "YYYYMM": yyyymm,
                        "gpp_tif": "",
                        "stage": "SWATH_SOURCE_READ",
                        "source_file": str(source.get("file_path", "")),
                        "error_message": (
                            f"{type(error).__name__}: {error}\n"
                            f"{traceback.format_exc()}"
                        ),
                    }
                )

        all_lack_geometry = bool(source_statuses) and all(
            status == "PRODUCT_LACKS_FOOTPRINT_GEOMETRY"
            for status in source_statuses
        )

        for context in contexts:
            match = context["match"]
            try:
                sif_array, metrics = context_sif_array_and_metrics(context)
                if all_lack_geometry:
                    skipped_rows.append(
                        skipped_row(
                            match,
                            "PRODUCT_LACKS_FOOTPRINT_GEOMETRY",
                            (
                                "Readable local SIF file(s) exist, but no usable "
                                "footprint/corner/bounds geometry was found. "
                                "Point-to-cell fallback is forbidden."
                            ),
                        )
                    )
                    continue
                notes = (
                    "Swath SIF observation footprint polygons were rasterized "
                    "to all touched GPP-grid cells; overlapping footprints "
                    "were averaged. Point-to-cell fallback was not used."
                )
                if source_statuses:
                    notes += " Source statuses: " + " | ".join(sorted(set(source_statuses)))
                analysis, validation, skip = handle_array_for_match(
                    match, sif_array, csv_lookup, notes, metrics
                )
                if analysis is not None:
                    analysis_rows.append(analysis)
                if validation is not None:
                    validation_rows.append(validation)
                if skip is not None:
                    skipped_rows.append(skip)
            except Exception as error:
                failure_rows.append(
                    {
                        "product": match["product"],
                        "SITE_ID": match["SITE_ID"],
                        "SITE_SAFE": match["SITE_SAFE"],
                        "YYYYMM": match["YYYYMM"],
                        "gpp_tif": match["gpp_tif"],
                        "stage": "SWATH_FOOTPRINT_PROCESSING",
                        "error_message": (
                            f"{type(error).__name__}: {error}\n"
                            f"{traceback.format_exc()}"
                        ),
                    }
                )
                skipped_rows.append(
                    skipped_row(
                        match,
                        "FAILED_SOURCE_READ",
                        "Failed while rasterizing SIF footprint polygons to the GPP grid.",
                        error_message=f"{type(error).__name__}: {error}",
                    )
                )

    return analysis_rows, validation_rows, skipped_rows, failure_rows, missing_geometry_rows


# =============================================================================
# QUICKLOOKS AND REPORTS
# =============================================================================


def quicklook_requests(analysis: pd.DataFrame) -> list[pd.Series]:
    if analysis.empty:
        return []
    observed = analysis[analysis["n_valid_sif_pixels_band2"] > 0].copy()
    if observed.empty:
        return []
    return [row for _, row in observed.sort_values(["product", "SITE_ID", "YYYYMM"]).iterrows()]


def create_quicklooks(analysis: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not CREATE_QUICKLOOKS or plt is None or DRY_RUN:
        return pd.DataFrame(rows)

    for row in quicklook_requests(analysis):
        output_tif = Path(row["final_sif_companion_tif"])
        if not base.path_exists(output_tif):
            continue
        try:
            with rasterio.open(base.native_path(output_tif)) as dataset:
                nodata = dataset.nodata if dataset.nodata is not None else NODATA_VALUE
                panels = [
                    (2, "SIF_unmasked_aligned_to_GPP_grid"),
                    (7, "SIF_masked_by_footprint_probability"),
                    (8, "SIF_masked_by_blind_circular_buffer"),
                ]
                created_pngs: list[str] = []
                for band_index, band_name in panels:
                    array = dataset.read(band_index).astype(float)
                    array[array == nodata] = np.nan
                    product_token = output_product_token(str(row["product"]))
                    png = QUICKLOOK_ROOT / (
                        f"{row['SITE_SAFE']}_{row['YYYYMM']}_{product_token}_{band_name}.png"
                    )
                    fig, axis = plt.subplots(1, 1, figsize=(7, 6))
                    image = axis.imshow(array, cmap="viridis")
                    axis.set_title(
                        f"{row['SITE_ID']} | {row['product']} | {row['YYYYMM']} | {band_name}"
                    )
                    axis.set_axis_off()
                    fig.colorbar(image, ax=axis, shrink=0.75)
                    fig.tight_layout()
                    base.ensure_directory(png.parent)
                    fig.savefig(base.native_path(png), dpi=160, bbox_inches="tight")
                    plt.close(fig)
                    created_pngs.append(str(png))
            for png in created_pngs:
                rows.append(
                    {
                        "SITE_ID": row["SITE_ID"],
                        "SITE_SAFE": row["SITE_SAFE"],
                        "product": row["product"],
                        "YYYYMM": row["YYYYMM"],
                        "quicklook_png": png,
                        "status": "CREATED",
                        "notes": "Single-band quicklook; no RGB composite.",
                    }
                )
        except Exception as error:
            rows.append(
                {
                    "SITE_ID": row["SITE_ID"],
                    "SITE_SAFE": row["SITE_SAFE"],
                    "product": row["product"],
                    "YYYYMM": row["YYYYMM"],
                    "quicklook_png": "",
                    "status": "FAILED",
                    "notes": f"{type(error).__name__}: {error}",
                }
            )
    return pd.DataFrame(rows)


def skipped_from_no_local(matching_report: pd.DataFrame) -> pd.DataFrame:
    if matching_report.empty:
        return pd.DataFrame()
    subset = matching_report[
        matching_report["match_status"] == "NO_LOCAL_SIF_FOR_MONTH"
    ].copy()
    if subset.empty:
        return pd.DataFrame()
    subset["skip_reason"] = "NO_LOCAL_SIF_FOR_MONTH"
    subset["status"] = "NO_LOCAL_SIF_FOR_MONTH"
    subset["notes"] = "No readable local SIF source file exists for this product/month."
    subset["error_message"] = ""
    return subset[
        [
            "product",
            "SITE_ID",
            "SITE_SAFE",
            "YYYYMM",
            "gpp_tif",
            "sif_source_files",
            "skip_reason",
            "status",
            "notes",
            "error_message",
        ]
    ].rename(columns={"sif_source_files": "source_sif_files"})


def empty_analysis_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "SITE_ID",
            "SITE_SAFE",
            "YYYYMM",
            "year",
            "month",
            "IGBP",
            "product",
            "tower_GPP_monthly",
            "matching_gpp_tif",
            "final_sif_companion_tif",
            "source_sif_files",
            "source_sif_type",
            "swath_geometry_mode",
            "n_source_sif_observations",
            "n_sif_footprint_polygons",
            "n_valid_sif_pixels_band2",
            "mean_gpp_cells_filled_per_sif_observation",
            "n_valid_pixels_inside_footprint_probability_gt_0",
            "n_valid_pixels_inside_blind_buffer",
            "SIF_mean_all_valid_band2",
            "SIF_mean_inside_footprint_probability_gt_0",
            "SIF_mean_inside_blind_buffer",
            "SIF_mean_footprint_probability_weighted",
            "status",
            "notes",
            "_action",
        ]
    )


def empty_validation_frame() -> pd.DataFrame:
    """Return an empty validation table with stable columns.

    Product-only resume runs can legitimately create zero output TIFFs
    (for example, when every candidate source lacks usable footprint
    overlap).  In that case validation_rows is empty, but the summary
    builder still needs the product and validation_status columns.
    """
    return pd.DataFrame(
        columns=[
            "product",
            "SITE_ID",
            "SITE_SAFE",
            "YYYYMM",
            "final_sif_companion_tif",
            "matching_gpp_tif",
            "validation_status",
            "failure_reasons",
            "band_count_is_8",
            "band_descriptions_exact",
            "crs_exact_match",
            "transform_exact_match",
            "width_exact_match",
            "height_exact_match",
            "bounds_exact_match",
            "pixel_size_exact_match",
            "swath_geometry_mode",
            "n_source_sif_observations",
            "n_sif_footprint_polygons",
            "n_gpp_cells_filled_by_sif_footprints",
            "mean_gpp_cells_filled_per_sif_observation",
            "footprint_aware_rasterization_confirmed",
        ]
    )


def build_processing_summary(
    templates_all: list[dict[str, Any]],
    templates_run: list[dict[str, Any]],
    source_inventory: pd.DataFrame,
    matching_report: pd.DataFrame,
    analysis: pd.DataFrame,
    validation: pd.DataFrame,
    skipped: pd.DataFrame,
    failures: pd.DataFrame,
    missing_geometry: pd.DataFrame,
    products: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for product in products:
        product_sources = source_inventory[source_inventory["product"] == product]
        product_matches = matching_report[matching_report["product"] == product]
        product_analysis = analysis[analysis["product"] == product]
        product_validation = validation[validation["product"] == product]
        product_skipped = skipped[skipped["product"] == product]
        product_missing_geometry = missing_geometry[
            missing_geometry["product"] == product
        ] if not missing_geometry.empty else pd.DataFrame()
        swath_rows = product_analysis[
            product_analysis["source_sif_type"] == "point_or_swath"
        ] if not product_analysis.empty else pd.DataFrame()
        mean_cells = (
            pd.to_numeric(
                swath_rows["mean_gpp_cells_filled_per_sif_observation"],
                errors="coerce",
            ).dropna().mean()
            if not swath_rows.empty
            else np.nan
        )
        rows.append(
            {
                "product": product,
                "gpp_templates_found": len(templates_all),
                "gpp_templates_selected_for_run": len(templates_run),
                "unique_sites_selected": len(
                    {record["SITE_ID"] for record in templates_run}
                ),
                "source_sif_files_found": len(product_sources),
                "readable_source_sif_files": int(
                    product_sources["read_status"].eq("OK").sum()
                )
                if not product_sources.empty
                else 0,
                "matched_product_site_months": int(
                    product_matches["match_status"].eq("MATCHED_LOCAL_SIF_MONTH").sum()
                )
                if not product_matches.empty
                else 0,
                "footprint_aware_sif_companion_tiffs_created": int(
                    product_analysis["_action"].eq("CREATED").sum()
                )
                if not product_analysis.empty
                else 0,
                "skipped_existing_valid_outputs": int(
                    product_analysis["_action"].eq("SKIPPED_EXISTING").sum()
                )
                if not product_analysis.empty
                else 0,
                "skipped_no_local_sif_month": int(
                    product_skipped["skip_reason"].eq("NO_LOCAL_SIF_FOR_MONTH").sum()
                )
                if not product_skipped.empty
                else 0,
                "skipped_missing_footprint_geometry": int(
                    product_skipped["skip_reason"].eq("PRODUCT_LACKS_FOOTPRINT_GEOMETRY").sum()
                )
                if not product_skipped.empty
                else 0,
                "skipped_no_footprint_overlap": int(
                    product_skipped["skip_reason"].eq("NO_ACTUAL_SIF_FOOTPRINT_OVERLAP_WITH_GPP_GRID").sum()
                )
                if not product_skipped.empty
                else 0,
                "source_files_missing_footprint_geometry": len(product_missing_geometry),
                "mean_gpp_cells_filled_per_swath_observation": mean_cells,
                "validation_PASS": int(
                    product_validation["validation_status"].eq("PASS").sum()
                )
                if not product_validation.empty
                else 0,
                "validation_FAIL": int(
                    product_validation["validation_status"].eq("FAIL").sum()
                )
                if not product_validation.empty
                else 0,
                "all_nodata_tiffs_created": 0,
                "standalone_circular_buffer_tiffs_created": 0,
                "swath_products_not_rasterized_as_single_points": (
                    "not_applicable_gridded"
                    if product == "GOSIF"
                    else "yes_footprint_polygons_or_skipped"
                ),
                "failures": int((failures["product"] == product).sum())
                if not failures.empty
                else 0,
                "output_product_folder": str(COMPANION_ROOT / output_product_token(product)),
            }
        )
    return pd.DataFrame(rows)


def count_tifs_outside_companion_root() -> int:
    if not base.path_exists(OUTPUT_ROOT):
        return 0
    allowed_product_roots = [
        (COMPANION_ROOT / output_product_token(product)).resolve()
        for product in PRODUCTS
    ]
    count = 0
    for path in OUTPUT_ROOT.rglob("*.tif"):
        resolved = path.resolve()
        if not any(
            resolved == root or root in resolved.parents
            for root in allowed_product_roots
        ):
            count += 1
    return count


def write_locked_spec_reports(
    csv_check: pd.DataFrame,
    validation: pd.DataFrame,
    skipped: pd.DataFrame,
    missing_geometry: pd.DataFrame,
    checksum_audit_text: str,
) -> None:
    """Write locked-spec report filenames required by the final SIF spec."""
    grid_fields = [
        "crs_exact_match",
        "transform_exact_match",
        "width_exact_match",
        "height_exact_match",
        "bounds_exact_match",
        "pixel_size_exact_match",
    ]
    grid_rows = validation.copy()
    if grid_rows.empty:
        grid_rows = pd.DataFrame(
            columns=["SITE_ID", "YYYYMM", "product", "pass_fail", "mismatch_detail"]
        )
    else:
        missing = [field for field in grid_fields if field not in grid_rows.columns]
        for field in missing:
            grid_rows[field] = False
        grid_rows["pass_fail"] = np.where(
            grid_rows[grid_fields].all(axis=1), "PASS", "FAIL"
        )
        grid_rows["mismatch_detail"] = grid_rows.apply(
            lambda row: "; ".join(
                field for field in grid_fields if not bool(row.get(field))
            ),
            axis=1,
        )
        grid_rows = grid_rows[
            ["SITE_ID", "YYYYMM", "product", "gpp_tif", "sif_companion_tif", "pass_fail", "mismatch_detail"]
        ]
    base.write_report_bundle(
        QC_ROOT / "grid_match_report",
        "Grid Match Report",
        grid_rows,
        ["Every output must match its paired GPP TIFF CRS, transform, shape, bounds, and pixel size."],
    )

    schema_rows = validation.copy()
    schema_fields = ["band_count_is_8", "band_descriptions_exact"]
    if schema_rows.empty:
        schema_rows = pd.DataFrame(
            columns=["SITE_ID", "YYYYMM", "product", "pass_fail", "mismatch_detail"]
        )
    else:
        for field in schema_fields:
            if field not in schema_rows.columns:
                schema_rows[field] = False
        schema_rows["pass_fail"] = np.where(
            schema_rows[schema_fields].all(axis=1), "PASS", "FAIL"
        )
        schema_rows["expected_band_descriptions"] = " | ".join(SIF_BAND_DESCRIPTIONS)
        schema_rows["mismatch_detail"] = schema_rows.apply(
            lambda row: "; ".join(
                field for field in schema_fields if not bool(row.get(field))
            ),
            axis=1,
        )
        schema_rows = schema_rows[
            [
                "SITE_ID",
                "YYYYMM",
                "product",
                "sif_companion_tif",
                "pass_fail",
                "expected_band_descriptions",
                "mismatch_detail",
            ]
        ]
    base.write_report_bundle(
        QC_ROOT / "band_schema_validation_report",
        "Band Schema Validation Report",
        schema_rows,
        ["Locked schema requires exactly 8 bands with exact GDAL band descriptions."],
    )

    missing_geometry_log = missing_geometry.copy()
    if missing_geometry_log.empty:
        missing_geometry_log = pd.DataFrame(
            columns=[
                "SITE_ID",
                "YYYYMM",
                "PRODUCT",
                "sounding_id_or_timestamp",
                "reason",
            ]
        )
    else:
        missing_geometry_log = missing_geometry_log.rename(
            columns={"product": "PRODUCT", "missing_footprint_geometry_reason": "reason"}
        )
        if "SITE_ID" not in missing_geometry_log.columns:
            missing_geometry_log["SITE_ID"] = ""
        if "sounding_id_or_timestamp" not in missing_geometry_log.columns:
            missing_geometry_log["sounding_id_or_timestamp"] = ""
    base.write_report_bundle(
        QC_ROOT / "missing_geometry_log",
        "Missing Geometry Log",
        missing_geometry_log,
        ["Swath source records/files without usable true footprint geometry are skipped, not approximated."],
    )

    skipped_no_gpp = csv_check[
        csv_check["match_status"].eq("CSV_VALID_BUT_NO_GPP_TIF")
    ].copy()
    base.write_report_bundle(
        QC_ROOT / "skipped_no_gpp",
        "Skipped No GPP",
        skipped_no_gpp,
        ["Valid GPP-reference CSV site-months with no paired GPP TIFF."],
    )

    skipped_no_sif = skipped[
        skipped["skip_reason"].eq("NO_LOCAL_SIF_FOR_MONTH")
    ].copy() if not skipped.empty else pd.DataFrame()
    base.write_report_bundle(
        QC_ROOT / "skipped_no_sif_data",
        "Skipped No SIF Data",
        skipped_no_sif,
        ["Site-month/product cases skipped because no local SIF source month exists."],
    )

    pipeline_files = [
        (Path(PROJECT_ROOT) / 'Codes/pythonProject/Create_FINAL_CLEAN_SIF_Companion_TIFFs_FOOTPRINT_AWARE.py'),
        Path(base.__file__),
        Path(sif_core.__file__),
    ]
    kmz_hits: list[str] = []
    for file_path in pipeline_files:
        text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), start=1):
            lowered = line.lower()
            is_actual_kmz_import = bool(
                re.search(r"^\s*(import|from)\s+(zipfile|fastkml)\b", line)
            )
            is_actual_kmz_path_or_call = bool(
                re.search(r"\.(kmz|kml)\b", lowered)
                and "audit" not in lowered
                and "kmz_usage" not in lowered
                and "kmz/kml" not in lowered
            )
            if is_actual_kmz_import or is_actual_kmz_path_or_call:
                kmz_hits.append(f"{file_path}:{lineno}: {line.strip()}")
    (QC_ROOT / "kmz_usage_audit.txt").write_text(
        "\n".join(
            [
                "KMZ usage audit for the locked SIF pipeline code path.",
                "Production rule: no KMZ/KML/AOI boxes may be used.",
                f"Read/write-style KMZ/KML hits found: {len(kmz_hits)}",
                "",
                *kmz_hits,
            ]
        ),
        encoding="utf-8",
    )

    interpolation_hits: list[str] = []
    for file_path in pipeline_files:
        text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip().strip(",").strip('"').strip("'")
            if stripped in {
                "scipy.interpolate",
                "griddata",
                "fillnodata",
                "Resampling.bilinear",
                "Resampling.cubic",
                "Resampling.average",
            }:
                continue
            if re.search(r"^\s*(import|from)\s+scipy\.interpolate\b", line) or re.search(
                r"\b(griddata|fillnodata)\s*\(", line
            ) or re.search(r"\bResampling\.(bilinear|cubic|average)\b", line):
                interpolation_hits.append(f"{file_path}:{lineno}: {line.strip()}")
    (QC_ROOT / "interpolation_audit.txt").write_text(
        "\n".join(
            [
                "Interpolation/gap-fill audit for the locked SIF pipeline code path.",
                "Allowed gridded method: rasterio.warp.reproject with Resampling.nearest only.",
                f"Forbidden interpolation/gap-fill hits found: {len(interpolation_hits)}",
                "",
                *interpolation_hits,
            ]
        ),
        encoding="utf-8",
    )

    (QC_ROOT / "source_modification_audit.txt").write_text(
        checksum_audit_text,
        encoding="utf-8",
    )

    successful_swath = pd.DataFrame()
    if not validation.empty and "swath_geometry_mode" in validation.columns:
        successful_swath = validation[
            validation["swath_geometry_mode"].eq("footprint_polygon_to_GPP_grid_cells")
        ].copy()
        keep = [
            column
            for column in [
                "product",
                "SITE_ID",
                "SITE_SAFE",
                "YYYYMM",
                "gpp_tif",
                "sif_companion_tif",
                "swath_geometry_fields_used",
                "n_candidate_sif_observations_for_site_grid",
                "n_candidate_sif_observations_skipped_bad_geometry",
                "n_candidate_sif_observations_no_spatial_intersection_with_grid",
                "n_candidate_sif_observations_unexplained",
                "candidate_accounting_complete",
                "n_source_sif_observations",
                "n_sif_footprint_polygons",
                "n_gpp_cells_filled_by_sif_footprints",
                "mean_gpp_cells_filled_per_sif_observation",
                "validation_status",
            ]
            if column in successful_swath.columns
        ]
        successful_swath = successful_swath[keep]
        successful_swath["geometry_status"] = "USED_TRUE_FOOTPRINT_GEOMETRY"

    missing_swath = (
        missing_geometry[
            [
                column
                for column in [
                    "product",
                    "file_path",
                    "YYYYMM",
                    "detected_lat_variable",
                    "detected_lon_variable",
                    "detected_sif_variable",
                    "available_coordinate_or_geometry_variables",
                ]
                if column in missing_geometry.columns
            ]
        ]
        .drop_duplicates()
        .copy()
        if not missing_geometry.empty
        else pd.DataFrame(columns=["product", "file_path", "YYYYMM"])
    )
    if not missing_swath.empty:
        missing_swath["geometry_status"] = "SKIPPED_MISSING_TRUE_FOOTPRINT_GEOMETRY"

    swath_report = pd.concat(
        [successful_swath, missing_swath], ignore_index=True, sort=False
    )
    if swath_report.empty:
        swath_report = pd.DataFrame(
            columns=["product", "YYYYMM", "geometry_status"]
        )
    swath_report["method"] = "true_footprint_polygon_to_GPP_grid_cells_or_skip_if_missing"
    swath_report["point_centroid_used"] = False
    swath_report["kmz_used"] = False
    swath_report["synthetic_box_or_circle_used"] = False
    base.write_report_bundle(
        QC_ROOT / "swath_geometry_method_report",
        "Swath Geometry Method Report",
        swath_report,
        ["Swath method is true footprint/corner polygon rasterization, never point-centroid/KMZ/synthetic box fallback."],
    )

    accounting_columns = [
        "product",
        "SITE_ID",
        "SITE_SAFE",
        "YYYYMM",
        "sif_companion_tif",
        "swath_geometry_fields_used",
        "n_candidate_sif_observations_for_site_grid",
        "n_sif_footprint_polygons",
        "n_candidate_sif_observations_skipped_bad_geometry",
        "n_candidate_sif_observations_no_spatial_intersection_with_grid",
        "n_candidate_sif_observations_unexplained",
        "candidate_accounting_complete",
    ]
    accounting_report = (
        validation[
            validation.get("swath_geometry_mode", pd.Series(dtype=str)).eq(
                "footprint_polygon_to_GPP_grid_cells"
            )
        ].copy()
        if not validation.empty and "swath_geometry_mode" in validation.columns
        else pd.DataFrame(columns=accounting_columns)
    )
    for column in accounting_columns:
        if column not in accounting_report.columns:
            accounting_report[column] = ""
    if not accounting_report.empty:
        accounting_report = accounting_report[accounting_columns]
        numeric_columns = [
            "n_candidate_sif_observations_for_site_grid",
            "n_sif_footprint_polygons",
            "n_candidate_sif_observations_skipped_bad_geometry",
            "n_candidate_sif_observations_no_spatial_intersection_with_grid",
            "n_candidate_sif_observations_unexplained",
        ]
        for column in numeric_columns:
            accounting_report[column] = pd.to_numeric(
                accounting_report[column], errors="coerce"
            ).fillna(0).astype(int)
        accounting_report["accounting_sum"] = (
            accounting_report["n_sif_footprint_polygons"]
            + accounting_report["n_candidate_sif_observations_skipped_bad_geometry"]
            + accounting_report[
                "n_candidate_sif_observations_no_spatial_intersection_with_grid"
            ]
            + accounting_report["n_candidate_sif_observations_unexplained"]
        )
        accounting_report["accounting_delta"] = (
            accounting_report["n_candidate_sif_observations_for_site_grid"]
            - accounting_report["accounting_sum"]
        )
        accounting_report["candidate_accounting_complete"] = (
            accounting_report["accounting_delta"].eq(0)
            & accounting_report["n_candidate_sif_observations_unexplained"].eq(0)
        )
        accounting_report["primary_exclusion_reason"] = np.where(
            accounting_report[
                "n_candidate_sif_observations_no_spatial_intersection_with_grid"
            ]
            > 0,
            (
                "candidate bbox overlapped the site search cell, but the true "
                "footprint polygon filled zero cells in the exact GPP raster grid"
            ),
            "",
        )
    base.write_report_bundle(
        QC_ROOT / "swath_candidate_accounting_report",
        "Swath Candidate Accounting Report",
        accounting_report,
        [
            "For every swath output, candidates must equal burned polygons plus explicit exclusion categories.",
            "The common legitimate exclusion is no_spatial_intersection_with_grid after true polygon-to-grid rasterization.",
        ],
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(base.native_path(path), "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def checksum_manifest(paths: Iterable[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted({Path(p) for p in paths}, key=lambda p: str(p).lower()):
        try:
            stat = path.stat()
            rows.append(
                {
                    "path": str(path),
                    "exists": True,
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "sha256": sha256_file(path),
                    "error_message": "",
                }
            )
        except Exception as error:
            rows.append(
                {
                    "path": str(path),
                    "exists": False,
                    "size_bytes": "",
                    "mtime_ns": "",
                    "sha256": "",
                    "error_message": f"{type(error).__name__}: {error}",
                }
            )
    return pd.DataFrame(rows)


def source_modification_audit_text(before: pd.DataFrame, after: pd.DataFrame) -> str:
    merged = before.merge(after, on="path", how="outer", suffixes=("_before", "_after"))
    checks = ["exists", "size_bytes", "mtime_ns", "sha256"]
    changed_rows = []
    for _, row in merged.iterrows():
        changed = [
            field
            for field in checks
            if str(row.get(f"{field}_before", "")) != str(row.get(f"{field}_after", ""))
        ]
        if changed:
            changed_rows.append((row["path"], ", ".join(changed)))
    lines = [
        "Source modification audit.",
        "Scope: paired GPP TIFFs selected for this run and raw SIF source files discovered for selected products.",
        f"Files checksummed before run: {len(before):,}",
        f"Files checksummed after run: {len(after):,}",
        f"Files changed: {len(changed_rows):,}",
        "Result: PASS" if not changed_rows else "Result: FAIL",
        "",
    ]
    if changed_rows:
        lines.append("Changed files:")
        lines.extend(f"{path} :: {fields}" for path, fields in changed_rows)
    else:
        lines.append("No GPP TIFF or raw SIF source checksum/size/mtime changes detected.")
    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================


def main() -> int:
    started = datetime.now()
    configure_modules()
    initialize_output_directories()
    products = selected_products()

    print("=" * 78)
    print("CREATE FINAL CLEAN FOOTPRINT-AWARE SIF COMPANION TIFFS")
    print("=" * 78)
    print(f"Run label: {RUN_LABEL}")
    print(f"Products: {products}")
    print(f"Selected sites: {SELECTED_SITES}")
    print(f"Selected YYYYMM: {SELECTED_YYYYMM}")
    print(f"DRY_RUN={DRY_RUN}; WRITE_OUTPUTS={WRITE_OUTPUTS}; OVERWRITE_EXISTING={OVERWRITE_EXISTING}")
    print(f"Output root: {OUTPUT_ROOT}")

    if USE_KMZ_FILES:
        raise ValueError("USE_KMZ_FILES must remain False.")
    if not USE_GPP_TIFF_AS_ONLY_SPATIAL_TEMPLATE:
        raise ValueError("The existing GPP TIFF must be the only spatial template.")
    if not CREATE_ONLY_FILES_WITH_ACTUAL_SIF_OBSERVATIONS:
        raise ValueError("Final outputs require actual SIF observations.")
    if not DO_NOT_CREATE_ALL_NODATA_PLACEHOLDER_TIFFS:
        raise ValueError("All-NoData placeholder TIFFs are forbidden.")
    if not DO_NOT_CREATE_STANDALONE_CIRCULAR_BUFFER_TIFFS:
        raise ValueError("Standalone circular-buffer TIFFs are forbidden.")
    if GOSIF_GRIDDED_RESAMPLING != "nearest":
        raise ValueError("Only nearest-neighbour GOSIF resampling is permitted.")
    if SWATH_RASTERIZATION_METHOD != "footprint_polygon_to_GPP_grid_cells":
        raise ValueError("Swath rasterization must be footprint polygon to GPP grid cells.")
    if MULTIPLE_SIF_FOOTPRINTS_PER_CELL != "mean":
        raise ValueError("Only mean of overlapping SIF footprints per cell is supported.")

    gpp_inventory, templates_all = base.inventory_gpp_templates()
    base.write_report_bundle(
        QC_ROOT / "gpp_tiff_template_inventory",
        "Footprint-Aware GPP TIFF Template Inventory",
        gpp_inventory,
        [
            f"GPP TIFF templates found: {len(gpp_inventory):,}",
            f"Valid six-band GPP templates: {len(templates_all):,}",
            f"Unique GPP sites: {len({r['SITE_ID'] for r in templates_all}):,}",
            "These existing GPP TIFFs are the only spatial templates.",
        ],
    )

    csv_sites, csv_lookup = base.load_gpp_reference()
    csv_check = base.build_csv_template_check(csv_sites, templates_all)
    base.write_report_bundle(
        QC_ROOT / "gpp_reference_csv_site_month_check",
        "Footprint-Aware GPP Reference CSV Site-Month Check",
        csv_check,
        [
            f"Valid unique CSV site-months: {len(csv_sites):,}",
            f"Unique CSV sites: {csv_sites['SITE_ID'].nunique():,}",
            f"CSV date range: {csv_sites['YYYYMM'].min()} to {csv_sites['YYYYMM'].max()}",
            "The CSV is used for site-month confirmation and metadata only, not spatial grids.",
        ],
    )

    templates_run = base.filter_templates(templates_all, csv_lookup)
    print(f"GPP templates selected for this run: {len(templates_run):,}")

    source_inventory, usable_sources = base.discover_sif_sources(products)
    base.write_report_bundle(
        QC_ROOT / "sif_source_file_inventory",
        "Footprint-Aware Original SIF Source File Inventory",
        source_inventory,
        [
            f"Original SIF root: {SIF_ROOT}",
            f"Products scanned: {', '.join(products)}",
            f"Science files discovered: {len(source_inventory):,}",
            f"Readable/date-parsed source files: {len(usable_sources):,}",
            "Previous derivative SIF folders are not used as final input sources.",
        ],
    )

    checksum_paths = [
        Path(record["gpp_tif"])
        for record in templates_run
        if record.get("gpp_tif")
    ] + [
        Path(record["file_path"])
        for record in usable_sources
        if record.get("file_path")
    ]
    checksum_before = (
        checksum_manifest(checksum_paths)
        if ENABLE_SOURCE_CHECKSUM_AUDIT
        else pd.DataFrame(columns=["path", "exists", "size_bytes", "mtime_ns", "sha256", "error_message"])
    )

    matching_report, matches = base.build_month_matching_report(
        templates_run, products, usable_sources
    )
    base.write_report_bundle(
        QC_ROOT / "gpp_tiff_to_sif_month_matching_report",
        "Footprint-Aware GPP TIFF to SIF Month Matching Report",
        matching_report,
        [
            f"Product-site-month rows checked: {len(matching_report):,}",
            f"Matched readable local SIF months: {len(matches):,}",
            f"No local SIF month: {int(matching_report['match_status'].eq('NO_LOCAL_SIF_FOR_MONTH').sum()):,}",
            "No TIFF is created where no local SIF source month exists.",
        ],
    )

    if COUNT_ONLY:
        product_counts = (
            matching_report.groupby("product", dropna=False)["match_status"]
            .value_counts()
            .unstack(fill_value=0)
            .reset_index()
        )
        attempted = int(len(matching_report))
        no_sif = int(
            matching_report["match_status"].eq("NO_LOCAL_SIF_FOR_MONTH").sum()
        )
        matched = int(
            matching_report["match_status"].eq("MATCHED_LOCAL_SIF_MONTH").sum()
        )
        no_gpp = int(csv_check["match_status"].eq("CSV_VALID_BUT_NO_GPP_TIF").sum())
        count_rows = pd.DataFrame(
            [
                {
                    "run_label": RUN_LABEL,
                    "products": ", ".join(products),
                    "total_sites_found": len({r["SITE_ID"] for r in templates_run}),
                    "total_gpp_templates_selected": len(templates_run),
                    "total_site_month_product_combinations_attempted": attempted,
                    "matched_local_sif_combinations": matched,
                    "skipped_up_front_no_sif_data": no_sif,
                    "csv_valid_site_months_with_no_gpp_tif": no_gpp,
                    "dry_run": DRY_RUN,
                    "count_only": COUNT_ONLY,
                }
            ]
        )
        base.write_report_bundle(
            QC_ROOT / "preproduction_full_count_report",
            "Pre-Production Full Count Report",
            count_rows,
            [
                "COUNT_ONLY mode stopped before any SIF array processing or TIFF writing.",
                "This report is the cheap before-production site/month/product count gate.",
            ],
        )
        base.write_report_bundle(
            QC_ROOT / "preproduction_full_count_by_product",
            "Pre-Production Full Count By Product",
            product_counts,
            ["Counts are grouped by normalized canonical product and temporal match status."],
        )
        print("COUNT_ONLY requested; stopping before SIF array processing.")
        print(f"Total sites found: {len({r['SITE_ID'] for r in templates_run}):,}")
        print(f"Total GPP templates selected: {len(templates_run):,}")
        print(f"Total site-month-product combinations attempted: {attempted:,}")
        print(f"Matched local SIF combinations: {matched:,}")
        print(f"Skipped up front for no local SIF data: {no_sif:,}")
        print(f"CSV valid site-months with no GPP TIFF: {no_gpp:,}")
        return 0

    gridded = [m for m in matches if m["source_type"] == "gridded_raster"]
    swath = [m for m in matches if m["source_type"] == "point_or_swath"]

    analysis_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    missing_geometry_rows: list[dict[str, Any]] = []

    a, v, s, f = process_gridded_matches(gridded, csv_lookup)
    analysis_rows.extend(a)
    validation_rows.extend(v)
    skipped_rows.extend(s)
    failure_rows.extend(f)

    a, v, s, f, mg = process_footprint_aware_swath_matches(swath, csv_lookup)
    analysis_rows.extend(a)
    validation_rows.extend(v)
    skipped_rows.extend(s)
    failure_rows.extend(f)
    missing_geometry_rows.extend(mg)

    analysis = pd.DataFrame(analysis_rows) if analysis_rows else empty_analysis_frame()
    validation = pd.DataFrame(validation_rows) if validation_rows else empty_validation_frame()
    processor_skipped = pd.DataFrame(skipped_rows)
    no_local_skipped = skipped_from_no_local(matching_report)
    skipped = pd.concat(
        [no_local_skipped, processor_skipped],
        ignore_index=True,
        sort=False,
    )
    failures = pd.DataFrame(failure_rows)
    missing_geometry = pd.DataFrame(missing_geometry_rows).drop_duplicates() if missing_geometry_rows else pd.DataFrame(
        columns=[
            "product",
            "file_path",
            "YYYYMM",
            "detected_lat_variable",
            "detected_lon_variable",
            "detected_sif_variable",
            "missing_footprint_geometry_reason",
            "available_coordinate_or_geometry_variables",
            "recommendation",
        ]
    )

    final_analysis = analysis[
        analysis["n_valid_sif_pixels_band2"].fillna(0).astype(int) > 0
    ].copy()
    final_analysis_no_action = final_analysis.drop(columns=["_action"], errors="ignore")
    base.write_report_bundle(
        SUMMARY_ROOT / "footprint_aware_sif_companion_analysis_table",
        "Footprint-Aware SIF Companion Analysis Table",
        final_analysis_no_action,
        [
            f"Footprint-aware analysis rows: {len(final_analysis):,}",
            "Rows are present only for outputs with actual SIF observations.",
            "Band 2 is not masked by footprint probability.",
            "Band 6 is footprint_probability x SIF where both are valid.",
            "Swath rows use footprint polygons, not sounding-center point cells.",
        ],
    )

    base.write_report_bundle(
        VALIDATION_ROOT / "footprint_aware_sif_companion_validation",
        "Footprint-Aware SIF Companion TIFF Validation",
        validation,
        [
            f"Footprint-aware outputs validated: {len(validation):,}",
            f"PASS: {int(validation.get('validation_status', pd.Series(dtype=str)).eq('PASS').sum()):,}",
            f"FAIL: {int(validation.get('validation_status', pd.Series(dtype=str)).eq('FAIL').sum()):,}",
            "Validation checks exact GPP grid match, copied GPP bands, and Band 6 math.",
            "Swath validation confirms footprint-aware rasterization tags and metrics.",
        ],
    )

    if skipped.empty:
        skipped = pd.DataFrame(
            columns=[
                "product",
                "SITE_ID",
                "SITE_SAFE",
                "YYYYMM",
                "gpp_tif",
                "source_sif_files",
                "skip_reason",
                "status",
                "notes",
                "error_message",
            ]
        )
    base.write_report_bundle(
        QC_ROOT / "skipped_footprint_aware_sif_cases",
        "Skipped Footprint-Aware SIF Cases",
        skipped,
        [
            f"Skipped cases reported: {len(skipped):,}",
            f"NO_LOCAL_SIF_FOR_MONTH: {int(skipped['skip_reason'].eq('NO_LOCAL_SIF_FOR_MONTH').sum()):,}",
            f"PRODUCT_LACKS_FOOTPRINT_GEOMETRY: {int(skipped['skip_reason'].eq('PRODUCT_LACKS_FOOTPRINT_GEOMETRY').sum()):,}",
            f"NO_ACTUAL_SIF_FOOTPRINT_OVERLAP_WITH_GPP_GRID: {int(skipped['skip_reason'].eq('NO_ACTUAL_SIF_FOOTPRINT_OVERLAP_WITH_GPP_GRID').sum()):,}",
            "Skipped/no-observation cases are reported here instead of written as all-NoData TIFFs.",
        ],
    )

    base.write_report_bundle(
        QC_ROOT / "products_or_files_missing_sif_footprint_geometry",
        "Products or Files Missing SIF Footprint Geometry",
        missing_geometry,
        [
            f"Source files lacking usable footprint geometry: {len(missing_geometry):,}",
            "Point-to-cell fallback is forbidden in this footprint-aware workflow.",
            "Files listed here need product documentation review or a source with footprint corners/bounds.",
        ],
    )

    if ENABLE_SOURCE_CHECKSUM_AUDIT:
        checksum_after = checksum_manifest(checksum_paths)
        checksum_audit_text = source_modification_audit_text(
            checksum_before, checksum_after
        )
    else:
        checksum_audit_text = "\n".join(
            [
                "Source modification audit.",
                "Result: NOT_RUN",
                "Reason: source checksum audit is disabled for this run.",
                "Set FOOTPRINT_AWARE_SIF_ENABLE_SOURCE_CHECKSUM_AUDIT=true for full production certification.",
                "The pipeline code opens GPP TIFFs and raw SIF source files read-only and writes outputs only under SIF_TIFF_output.",
            ]
        )
    write_locked_spec_reports(
        csv_check, validation, skipped, missing_geometry, checksum_audit_text
    )

    if not failures.empty:
        base.write_report_bundle(
            QC_ROOT / "processing_failures",
            "Footprint-Aware Processing Failures",
            failures,
            [f"Failures: {len(failures):,}"],
        )

    quicklooks = create_quicklooks(final_analysis)
    base.write_report_bundle(
        QUICKLOOK_ROOT / "quicklook_index",
        "Footprint-Aware SIF Companion Quicklook Index",
        quicklooks,
        [
            f"Quicklooks created: {int(quicklooks.get('status', pd.Series(dtype=str)).eq('CREATED').sum()):,}",
            f"Quicklook failures: {int(quicklooks.get('status', pd.Series(dtype=str)).eq('FAILED').sum()):,}",
            "Only observed/non-empty SIF companion TIFFs are used for quicklooks.",
        ],
    )

    processing_summary = build_processing_summary(
        templates_all,
        templates_run,
        source_inventory,
        matching_report,
        final_analysis,
        validation,
        skipped,
        failures,
        missing_geometry,
        products,
    )
    pass_count = int(
        validation.get("validation_status", pd.Series(dtype=str)).eq("PASS").sum()
    )
    fail_count = int(
        validation.get("validation_status", pd.Series(dtype=str)).eq("FAIL").sum()
    )
    created_count = int(
        final_analysis.get("_action", pd.Series(dtype=str)).eq("CREATED").sum()
    )
    all_nodata_outputs = int(
        (final_analysis["n_valid_sif_pixels_band2"].fillna(0).astype(int) == 0).sum()
    )
    standalone_tifs = count_tifs_outside_companion_root()
    if DRY_RUN or not WRITE_OUTPUTS:
        verdict = (
            "DRY_RUN_COMPLETE: no TIFF outputs were written; inspect reports before "
            "running production."
        )
    else:
        verdict = (
            "PASS: locked 8-band footprint-aware outputs contain actual SIF observations and pass strict validation."
            if fail_count == 0
            and failures.empty
            and all_nodata_outputs == 0
            and standalone_tifs == 0
            else "REVIEW: inspect validation, skipped, missing-geometry, and failure reports before proceeding."
        )
    summary_lines = [
        f"Run label: {RUN_LABEL}",
        f"Run started: {started.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Run finished: {base.now_text()}",
        f"GPP templates found: {len(templates_all):,}",
        f"GPP templates selected: {len(templates_run):,}",
        f"Unique selected sites: {len({r['SITE_ID'] for r in templates_run}):,}",
        f"Products processed: {', '.join(products)}",
        f"Matched readable local SIF product-site-months: {len(matches):,}",
        f"Footprint-aware SIF companion TIFFs created: {created_count:,}",
        f"Validation PASS: {pass_count:,}",
        f"Validation FAIL: {fail_count:,}",
        f"All-NoData SIF companion TIFFs created: {all_nodata_outputs:,}",
        f"Standalone circular-buffer TIFFs created: {standalone_tifs:,}",
        "KMZ files used: no",
        "Previous derivative SIF output folders used as final sources: no",
        "Every output contains Band 3 blind_circular_buffer copied from GPP Band 3.",
        "Every output uses the locked 8-band schema with Band 7 and Band 8 stored explicitly.",
        "GOSIF gridded resampling: nearest only",
        "Swath SIF rasterization: footprint polygon to all touched GPP-grid cells",
        "Point-to-cell swath fallback used: no",
        (
            "GIS display note: ArcGIS/QGIS default multiband RGB renderers are not "
            "scientifically meaningful for this TIFF. Use single-band stretched "
            "symbology by GDAL band description, especially Band 2, Band 7, or Band 8, "
            "or inspect the provided single-band quicklooks."
        ),
        f"Final verdict: {verdict}",
        f"Final output root: {OUTPUT_ROOT}",
    ]
    base.write_report_bundle(
        QC_ROOT / "run_summary",
        "SIF TIFF Pipeline Run Summary",
        processing_summary,
        summary_lines,
    )
    run_report_base = QC_ROOT / (
        "run_report_" + re.sub(r"[^A-Za-z0-9_.-]+", "_", RUN_LABEL)
    )
    base.write_markdown(
        run_report_base.with_suffix(".md"),
        f"Footprint-Aware SIF Companion Run Report - {RUN_LABEL}",
        summary_lines,
        processing_summary,
    )
    base.write_pdf(
        run_report_base.with_suffix(".pdf"),
        f"Footprint-Aware SIF Companion Run Report - {RUN_LABEL}",
        summary_lines,
        processing_summary,
    )

    print("\n" + "=" * 78)
    print("FOOTPRINT-AWARE SIF COMPANION RUN COMPLETE")
    print("=" * 78)
    print(f"GPP templates found: {len(templates_all):,}")
    print(f"Products processed: {', '.join(products)}")
    print("Per-product footprint-aware created/skipped:")
    for _, row in processing_summary.iterrows():
        print(
            f"  {row['product']}: "
            f"created={int(row['footprint_aware_sif_companion_tiffs_created']):,}; "
            f"no_local={int(row['skipped_no_local_sif_month']):,}; "
            f"missing_geometry={int(row['skipped_missing_footprint_geometry']):,}; "
            f"no_footprint_overlap={int(row['skipped_no_footprint_overlap']):,}; "
            f"validation_PASS={int(row['validation_PASS']):,}; "
            f"validation_FAIL={int(row['validation_FAIL']):,}"
        )
    print(f"Validation PASS count: {pass_count:,}")
    print(f"Validation FAIL count: {fail_count:,}")
    print(f"All-NoData TIFFs created: {all_nodata_outputs:,}")
    print(f"Standalone circular-buffer TIFFs created: {standalone_tifs:,}")
    print(f"Analysis table: {SUMMARY_ROOT / 'footprint_aware_sif_companion_analysis_table.csv'}")
    print(f"Main Markdown report: {QC_ROOT / 'run_summary.md'}")
    print(f"Main PDF report: {QC_ROOT / 'run_summary.pdf'}")
    print(f"Quicklook folder: {QUICKLOOK_ROOT}")
    print(f"Final verdict: {verdict}")
    return 0 if fail_count == 0 and all_nodata_outputs == 0 and standalone_tifs == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
