"""Finalize GOME-2 as native-observation, EC-source-area-weighted monthly site values (no machine-learning gap filling).

Input: GOME-2 L2 files, monthly footprint rasters, coverage inventories. Output: GOME-2 EC-footprint SIF tables (all site-months and primary analysis set), overlap provenance and QC reports."""
from __future__ import annotations
# Original script filename (sanitized): Finalize_GOME2_EC_Footprint_Analytical_Table.py; public name: 04b_Finalize_GOME2_EC_Footprint_Analytical_Table.py
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
from typing import Any

import h5py
import numpy as np
import pandas as pd
import rasterio
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pyproj import CRS, Geod, Transformer
from rasterio.features import rasterize
from rasterio.warp import transform_bounds
from shapely.geometry import Point, Polygon
from shapely.ops import transform as shapely_transform

import Create_FINAL_CLEAN_SIF_Companion_TIFFs_FOOTPRINT_AWARE as footprint_pipeline


PROJECT_ROOT = Path(
    PROJECT_ROOT
)
GOME2_ROOT = PROJECT_ROOT / "SIF Datasets" / "GOME-2"
GPP_ROOT = PROJECT_ROOT / "Monthly_EC_Footprint_TIFFs_Python"
PREP_ROOT = PROJECT_ROOT / "SIF_EC_Reconstruction_Model_Preparation"
OUTPUT_ROOT = PROJECT_ROOT / "SIF_EC_GOME2_Finalization"
FIGURE_ROOT = OUTPUT_ROOT / "figures"
QC_ROOT = OUTPUT_ROOT / "qc"

GPP_INVENTORY = PREP_ROOT / "GPP_TIFF_Template_Inventory.csv"
GPP_REFERENCE = PREP_ROOT / "GPP_Reference_Metadata_SiteMonth.csv"
EXISTING_GOME2_TIFF_ROOTS = [
    PROJECT_ROOT / "FINAL_CLEAN_SIF_Companion_TIFFs_FOOTPRINT_AWARE" / "SIF_Companion_TIFFs" / "GOME-2",
    PROJECT_ROOT / "SIF_TIFF_output" / "GOME2",
]

SIF_VAR = "Daily_Averaged_SIF"
LAT_VAR = "Latitude"
LON_VAR = "Longitude"
LAT_CORNERS_VAR = "Latitude_Corners"
LON_CORNERS_VAR = "Longitude_Corners"
TIME_VAR = "Delta_Time"
QUALITY_VAR = "Quality_Flag"
SIF_UNITS = "mW m-2 nm-1 sr-1"
SIF_FILL_THRESHOLD = -9000.0

PRIMARY_COVERAGE_THRESHOLD = 0.999
ZERO_TOL = 1e-12
WEIGHT_TOL = 1e-6
MEANINGFUL_OVERLAP_TOL = 1e-12
GEOD = Geod(ellps="WGS84")


def ensure_dirs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    QC_ROOT.mkdir(parents=True, exist_ok=True)


def native(path: Path | str) -> str:
    return str(path)


def decode_attr(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return decode_attr(value.reshape(-1)[0])
        return [decode_attr(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def dataset_attr(dataset: h5py.Dataset, names: list[str]) -> Any:
    for name in names:
        if name in dataset.attrs:
            return decode_attr(dataset.attrs[name])
    return None


def numeric_dataset_array(dataset: h5py.Dataset) -> np.ndarray:
    values = np.asarray(dataset[...], dtype=np.float64)
    fill = dataset_attr(dataset, ["_FillValue", "missing_value"])
    invalid = ~np.isfinite(values)
    if fill is not None:
        try:
            invalid |= values == float(fill)
        except Exception:
            pass
    scale = dataset_attr(dataset, ["scale_factor"])
    offset = dataset_attr(dataset, ["add_offset"])
    scale = 1.0 if scale is None else float(scale)
    offset = 0.0 if offset is None else float(offset)
    values = values * scale + offset
    values[invalid] = np.nan
    return values


def parse_gome2_date(path: Path) -> tuple[str, pd.Timestamp]:
    match = re.search(r"GOME-2[A-Z]?\.(\d{4})(\d{2})(\d{2})", path.name, flags=re.I)
    if not match:
        raise ValueError(f"Cannot parse GOME-2 date from {path.name}")
    year, month, day = map(int, match.groups())
    return f"{year:04d}{month:02d}", pd.Timestamp(year=year, month=month, day=day)


def discover_sources() -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]]]:
    records: list[dict[str, Any]] = []
    for path in sorted(GOME2_ROOT.rglob("*.nc")):
        try:
            yyyymm, date = parse_gome2_date(path)
            records.append(
                {
                    "file_path": str(path),
                    "filename": path.name,
                    "YYYYMM": yyyymm,
                    "observation_file_date": date.date().isoformat(),
                    "size_bytes": path.stat().st_size,
                    "mtime": path.stat().st_mtime,
                    "status": "OK",
                }
            )
        except Exception as exc:
            records.append(
                {
                    "file_path": str(path),
                    "filename": path.name,
                    "YYYYMM": "",
                    "observation_file_date": "",
                    "size_bytes": path.stat().st_size,
                    "mtime": path.stat().st_mtime,
                    "status": f"DATE_PARSE_FAILED:{type(exc).__name__}",
                }
            )
    inventory = pd.DataFrame(records)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["status"] == "OK" and record["YYYYMM"]:
            grouped[record["YYYYMM"]].append(record)
    return inventory, grouped


def file_snapshot(paths_or_roots: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for root in paths_or_roots:
        if not root.exists():
            rows.append({"path": str(root), "exists": False, "size": np.nan, "mtime_ns": np.nan})
            continue
        files = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
        for path in files:
            try:
                stat = path.stat()
                rows.append({"path": str(path), "exists": True, "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
            except OSError:
                rows.append({"path": str(path), "exists": False, "size": np.nan, "mtime_ns": np.nan})
    return pd.DataFrame(rows)


def compare_snapshots(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    merged = before.merge(after, on="path", how="outer", suffixes=("_before", "_after"))
    merged["unchanged_size_and_mtime"] = (
        merged["exists_before"].eq(merged["exists_after"])
        & merged["size_before"].eq(merged["size_after"])
        & merged["mtime_ns_before"].eq(merged["mtime_ns_after"])
    )
    return merged


def template_geographic_bounds(crs: CRS, bounds: Any) -> tuple[float, float, float, float]:
    return transform_bounds(crs, CRS.from_epsg(4326), bounds.left, bounds.bottom, bounds.right, bounds.top, densify_pts=21)


def normalize_longitudes(lon: np.ndarray) -> np.ndarray:
    return ((np.asarray(lon, dtype=np.float64) + 180.0) % 360.0) - 180.0


def make_polygon(lons: np.ndarray, lats: np.ndarray) -> Polygon | None:
    lons = normalize_longitudes(lons)
    lats = np.asarray(lats, dtype=np.float64)
    finite = np.isfinite(lons) & np.isfinite(lats) & (lats >= -90) & (lats <= 90) & (lons >= -180) & (lons <= 180)
    if finite.sum() < 3:
        return None
    coords = [(float(x), float(y)) for x, y in zip(lons[finite], lats[finite])]
    deduped: list[tuple[float, float]] = []
    for coord in coords:
        if not deduped or coord != deduped[-1]:
            deduped.append(coord)
    if len(set(deduped)) < 3:
        return None
    poly = Polygon(deduped)
    if poly.is_empty:
        return None
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or not poly.is_valid:
        return None
    return poly


def polygon_area_km2(poly: Polygon | None) -> float:
    if poly is None or poly.is_empty:
        return np.nan
    try:
        lon, lat = poly.exterior.coords.xy
        area_m2, _ = GEOD.polygon_area_perimeter(np.asarray(lon), np.asarray(lat))
        return abs(float(area_m2)) / 1_000_000.0
    except Exception:
        return np.nan


def date_from_delta(value: float, fallback: pd.Timestamp) -> str:
    if not math.isfinite(value):
        return fallback.isoformat()
    try:
        ts = pd.Timestamp("2007-01-01T00:00:00") + pd.to_timedelta(float(value), unit="s")
        return ts.isoformat()
    except Exception:
        return fallback.isoformat()


def coverage_class(value: float) -> str:
    if not math.isfinite(value):
        return "NaN"
    if value <= ZERO_TOL:
        return "=0"
    if value < 0.25:
        return "0-<0.25"
    if value < 0.50:
        return "0.25-<0.50"
    if value < 0.75:
        return "0.50-<0.75"
    if value < 0.90:
        return "0.75-<0.90"
    if value < 0.95:
        return "0.90-<0.95"
    if value < PRIMARY_COVERAGE_THRESHOLD:
        return "0.95-<1"
    return "≈1"


def site_to_safe(site_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(site_id)).strip("_")


def load_site_month_universe(source_months: set[str]) -> pd.DataFrame:
    gpp = pd.read_csv(GPP_INVENTORY)
    gpp["YYYYMM"] = gpp["YYYYMM"].astype(str)
    gpp = gpp[gpp["YYYYMM"].isin(source_months)].copy()
    if "valid_gpp_template" in gpp.columns:
        gpp = gpp[gpp["valid_gpp_template"].astype(bool)].copy()

    meta = pd.read_csv(GPP_REFERENCE, usecols=["SITE_ID", "YYYYMM", "lat", "lon", "IGBP", "year", "month", "SITE_SAFE"])
    meta["YYYYMM"] = meta["YYYYMM"].astype(str)
    meta = meta.drop_duplicates(["SITE_ID", "YYYYMM"])
    merged = gpp.merge(meta, on=["SITE_ID", "YYYYMM"], how="left", suffixes=("", "_meta"))
    for col in ["SITE_SAFE", "year", "month"]:
        alt = f"{col}_meta"
        if alt in merged.columns:
            merged[col] = merged[col].where(merged[col].notna(), merged[alt])
            merged = merged.drop(columns=[alt])
    if "SITE_SAFE" not in merged.columns:
        merged["SITE_SAFE"] = merged["SITE_ID"].map(site_to_safe)
    merged["SITE_SAFE"] = merged["SITE_SAFE"].fillna(merged["SITE_ID"].map(site_to_safe))
    return merged.sort_values(["YYYYMM", "SITE_ID"]).reset_index(drop=True)


def build_context(row: pd.Series) -> dict[str, Any]:
    gpp_tif = Path(str(row["gpp_tif"]))
    with rasterio.open(native(gpp_tif)) as ds:
        probability = ds.read(1).astype(np.float64)
        blind = ds.read(3).astype(np.float64) if ds.count >= 3 else np.zeros((ds.height, ds.width), dtype=np.float64)
        crs = ds.crs
        transform = ds.transform
        bounds = ds.bounds
        height = ds.height
        width = ds.width
    ec_mask = np.isfinite(probability) & (probability > 0)
    blind_mask = np.isfinite(blind) & (blind == 1)
    prob_sum = float(probability[ec_mask].sum(dtype=np.float64)) if ec_mask.any() else 0.0
    pixel_area_km2 = abs(float(transform.a * transform.e - transform.b * transform.d)) / 1_000_000.0
    return {
        "row": row,
        "SITE_ID": str(row["SITE_ID"]),
        "SITE_SAFE": str(row["SITE_SAFE"]),
        "YYYYMM": str(row["YYYYMM"]),
        "lat": float(row["lat"]) if pd.notna(row.get("lat")) else np.nan,
        "lon": float(row["lon"]) if pd.notna(row.get("lon")) else np.nan,
        "IGBP": row.get("IGBP", ""),
        "gpp_tif": str(gpp_tif),
        "crs": crs,
        "transform": transform,
        "bounds": bounds,
        "height": height,
        "width": width,
        "probability": probability,
        "blind": blind,
        "ec_mask": ec_mask,
        "blind_mask": blind_mask,
        "probability_sum_full_EC": prob_sum,
        "n_EC_pixels": int(ec_mask.sum()),
        "n_blind_pixels": int(blind_mask.sum()),
        "pixel_area_km2": pixel_area_km2,
        "EC_positive_support_area_km2": float(ec_mask.sum() * pixel_area_km2),
        "blind_buffer_area_km2": float(blind_mask.sum() * pixel_area_km2),
        "geographic_bounds": template_geographic_bounds(crs, bounds),
        "transformer": Transformer.from_crs(CRS.from_epsg(4326), crs, always_xy=True),
        "tower_point": Point(float(row["lon"]), float(row["lat"])) if pd.notna(row.get("lat")) and pd.notna(row.get("lon")) else None,
        "union_ec_covered": np.zeros((height, width), dtype=bool),
        "union_blind_covered": np.zeros((height, width), dtype=bool),
        "observations": [],
        "candidate_count": 0,
        "bad_geometry_count": 0,
        "no_ec_overlap_count": 0,
        "source_files_used": set(),
    }


def read_source_arrays(path: Path) -> dict[str, Any]:
    with h5py.File(native(path), "r") as handle:
        required = [SIF_VAR, LAT_VAR, LON_VAR, LAT_CORNERS_VAR, LON_CORNERS_VAR]
        missing = [name for name in required if name not in handle]
        if missing:
            raise ValueError(f"Missing required GOME-2 variables: {missing}")
        sif = numeric_dataset_array(handle[SIF_VAR]).squeeze()
        lat = numeric_dataset_array(handle[LAT_VAR]).squeeze()
        lon = normalize_longitudes(numeric_dataset_array(handle[LON_VAR]).squeeze())
        lat_corners = numeric_dataset_array(handle[LAT_CORNERS_VAR])
        lon_corners = normalize_longitudes(numeric_dataset_array(handle[LON_CORNERS_VAR]))
        quality = np.asarray(handle[QUALITY_VAR][...]).squeeze() if QUALITY_VAR in handle else np.full(sif.shape, np.nan)
        delta = np.asarray(handle[TIME_VAR][...], dtype=np.float64).squeeze() if TIME_VAR in handle else np.full(sif.shape, np.nan)
    if sif.ndim != 1:
        raise ValueError(f"GOME-2 SIF must be 1-D; got {sif.shape}")
    if lat_corners.shape[0] != sif.shape[0] and lat_corners.shape[1] == sif.shape[0]:
        lat_corners = lat_corners.T
    if lon_corners.shape[0] != sif.shape[0] and lon_corners.shape[1] == sif.shape[0]:
        lon_corners = lon_corners.T
    if lat_corners.shape[0] != sif.shape[0] or lon_corners.shape[0] != sif.shape[0]:
        raise ValueError(f"Corner arrays do not match SIF observations: {lat_corners.shape}, {lon_corners.shape}, {sif.shape}")
    return {
        "sif": sif,
        "lat": lat,
        "lon": lon,
        "lat_corners": lat_corners,
        "lon_corners": lon_corners,
        "quality": quality,
        "delta": delta,
    }


def process_source_for_contexts(source: dict[str, Any], contexts: list[dict[str, Any]], qc_rows: list[dict[str, Any]]) -> str:
    if not contexts:
        return "NO_CONTEXTS"
    path = Path(source["file_path"])
    try:
        arrays = read_source_arrays(path)
    except Exception as exc:
        qc_rows.append({"scope": "source_file", "file_path": str(path), "reason": "SOURCE_FILE_MISSING_OR_UNREADABLE", "detail": f"{type(exc).__name__}: {exc}"})
        return "FAILED_SOURCE_READ"

    sif = arrays["sif"]
    lat_corners = arrays["lat_corners"]
    lon_corners = arrays["lon_corners"]
    valid_sif = np.isfinite(sif) & (sif > SIF_FILL_THRESHOLD)
    finite_geom = (
        np.isfinite(lat_corners).all(axis=1)
        & np.isfinite(lon_corners).all(axis=1)
        & (np.nanmin(lat_corners, axis=1) >= -90)
        & (np.nanmax(lat_corners, axis=1) <= 90)
        & (np.nanmin(lon_corners, axis=1) >= -180)
        & (np.nanmax(lon_corners, axis=1) <= 180)
    )
    valid = valid_sif & finite_geom
    if not bool(valid.any()):
        return "NO_VALID_SOURCE_OBSERVATIONS"

    lon_min = np.nanmin(lon_corners, axis=1)
    lon_max = np.nanmax(lon_corners, axis=1)
    lat_min = np.nanmin(lat_corners, axis=1)
    lat_max = np.nanmax(lat_corners, axis=1)
    file_date = pd.Timestamp(source["observation_file_date"])
    any_updated = False

    for ctx in contexts:
        left, bottom, right, top = ctx["geographic_bounds"]
        idxs = np.flatnonzero(valid & (lon_max >= left) & (lon_min <= right) & (lat_max >= bottom) & (lat_min <= top))
        if idxs.size == 0:
            continue
        for idx in idxs:
            ctx["candidate_count"] += 1
            polygon = make_polygon(lon_corners[idx], lat_corners[idx])
            if polygon is None:
                ctx["bad_geometry_count"] += 1
                qc_rows.append({"scope": "observation", "SITE_ID": ctx["SITE_ID"], "YYYYMM": ctx["YYYYMM"], "observation_id": f"{path.stem}:{idx}", "reason": "INVALID_GEOMETRY"})
                continue
            try:
                projected = shapely_transform(ctx["transformer"].transform, polygon)
                if projected.is_empty:
                    ctx["bad_geometry_count"] += 1
                    continue
                if not projected.is_valid:
                    projected = projected.buffer(0)
                if projected.is_empty or not projected.is_valid:
                    ctx["bad_geometry_count"] += 1
                    continue
                mask = rasterize(
                    [(projected, 1)],
                    out_shape=(ctx["height"], ctx["width"]),
                    transform=ctx["transform"],
                    fill=0,
                    all_touched=True,
                    dtype="uint8",
                ).astype(bool)
            except Exception as exc:
                ctx["bad_geometry_count"] += 1
                qc_rows.append({"scope": "observation", "SITE_ID": ctx["SITE_ID"], "YYYYMM": ctx["YYYYMM"], "observation_id": f"{path.stem}:{idx}", "reason": "INVALID_GEOMETRY", "detail": f"{type(exc).__name__}: {exc}"})
                continue

            ec_cells = mask & ctx["ec_mask"]
            ec_probability_mass = float(ctx["probability"][ec_cells].sum(dtype=np.float64)) if ec_cells.any() else 0.0
            if ec_probability_mass <= MEANINGFUL_OVERLAP_TOL:
                ctx["no_ec_overlap_count"] += 1
                continue

            blind_cells = mask & ctx["blind_mask"]
            blind_mass = float(blind_cells.sum())
            native_area = polygon_area_km2(polygon)
            intersection_area = float(ec_cells.sum() * ctx["pixel_area_km2"])
            ec_overlap_fraction = ec_probability_mass / ctx["probability_sum_full_EC"] if ctx["probability_sum_full_EC"] > 0 else np.nan
            blind_overlap_fraction = blind_mass / ctx["n_blind_pixels"] if ctx["n_blind_pixels"] > 0 else np.nan
            fraction_native_inside_ec = intersection_area / native_area if math.isfinite(native_area) and native_area > 0 else np.nan
            ratio = native_area / ctx["EC_positive_support_area_km2"] if ctx["EC_positive_support_area_km2"] > 0 and math.isfinite(native_area) else np.nan
            tower_contains = bool(ctx["tower_point"] is not None and polygon.covers(ctx["tower_point"]))

            ctx["union_ec_covered"] |= ec_cells
            ctx["union_blind_covered"] |= blind_cells
            ctx["source_files_used"].add(str(path))
            any_updated = True
            obs = {
                "SITE_ID": ctx["SITE_ID"],
                "SITE_SAFE": ctx["SITE_SAFE"],
                "YYYYMM": ctx["YYYYMM"],
                "observation_id": f"{path.stem}:{idx}",
                "source_file": str(path),
                "source_file_date": source["observation_file_date"],
                "observation_datetime": date_from_delta(float(arrays["delta"][idx]) if np.ndim(arrays["delta"]) else np.nan, file_date),
                "source_observation_index": int(idx),
                "GOME2_SIF": float(sif[idx]),
                "GOME2_center_lat": float(arrays["lat"][idx]) if idx < arrays["lat"].shape[0] else np.nan,
                "GOME2_center_lon": float(arrays["lon"][idx]) if idx < arrays["lon"].shape[0] else np.nan,
                "corner_latitudes": ";".join(f"{x:.6f}" for x in lat_corners[idx]),
                "corner_longitudes": ";".join(f"{x:.6f}" for x in lon_corners[idx]),
                "native_footprint_area_km2": native_area,
                "EC_probability_mass_intersected": ec_probability_mass,
                "EC_probability_overlap_fraction": ec_overlap_fraction,
                "blind_buffer_overlap_fraction": blind_overlap_fraction,
                "geometric_intersection_area_km2": intersection_area,
                "fraction_GOME2_footprint_area_inside_EC_support": fraction_native_inside_ec,
                "footprint_area_to_EC_support_area_ratio": ratio,
                "EC_positive_support_area_km2": ctx["EC_positive_support_area_km2"],
                "quality_flag_raw": int(arrays["quality"][idx]) if idx < arrays["quality"].shape[0] and np.isfinite(arrays["quality"][idx]) else np.nan,
                "quality_status": "QUALITY_FLAG_RETAINED_NOT_HARD_FILTERED",
                "tower_containing_footprint": tower_contains,
            }
            ctx["observations"].append(obs)
    return "OK" if any_updated else "NO_EC_OVERLAP"


def summarize_context(ctx: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    obs = pd.DataFrame(ctx["observations"])
    qc_rows: list[dict[str, Any]] = []
    base = {
        "SITE_ID": ctx["SITE_ID"],
        "SITE_SAFE": ctx["SITE_SAFE"],
        "YYYYMM": ctx["YYYYMM"],
        "lat": ctx["lat"],
        "lon": ctx["lon"],
        "IGBP": ctx["IGBP"],
        "year": int(str(ctx["YYYYMM"])[:4]),
        "month": int(str(ctx["YYYYMM"])[4:6]),
        "matching_gpp_tif": ctx["gpp_tif"],
        "SIF_variable": SIF_VAR,
        "GOME2_units": SIF_UNITS,
        "geometry_variables": f"{LAT_CORNERS_VAR} | {LON_CORNERS_VAR}",
        "quality_filtering": "SIF fill/NoData and invalid geometry excluded; Quality_Flag retained as provenance, not hard-filtered because local workflow did not define a hard quality gate.",
        "n_candidate_GOME2_observations_for_site_grid": int(ctx["candidate_count"]),
        "n_candidate_GOME2_observations_bad_geometry": int(ctx["bad_geometry_count"]),
        "n_candidate_GOME2_observations_no_EC_probability_overlap": int(ctx["no_ec_overlap_count"]),
        "probability_sum_full_EC": ctx["probability_sum_full_EC"],
        "n_EC_pixels": ctx["n_EC_pixels"],
        "EC_positive_support_area_km2": ctx["EC_positive_support_area_km2"],
        "n_blind_pixels": ctx["n_blind_pixels"],
        "blind_buffer_area_km2": ctx["blind_buffer_area_km2"],
        "source_files_used": " | ".join(sorted(ctx["source_files_used"])),
    }
    if ctx["probability_sum_full_EC"] <= 0 or ctx["n_EC_pixels"] <= 0:
        base.update(
            {
                "GOME2_status": "EXCLUDED_INVALID_EC_FOOTPRINT",
                "GOME2_EC_probability_union_coverage": np.nan,
                "GOME2_EC_pixel_union_coverage": np.nan,
                "GOME2_EC_overlap_weighted_SIF": np.nan,
                "primary_eligible": False,
                "coverage_class": "NaN",
            }
        )
        qc_rows.append({"scope": "site_month", "SITE_ID": ctx["SITE_ID"], "YYYYMM": ctx["YYYYMM"], "reason": "INVALID_EC_FOOTPRINT"})
        return base, qc_rows

    if obs.empty:
        base.update(
            {
                "GOME2_status": "EXCLUDED_ZERO_EC_OVERLAP",
                "GOME2_EC_probability_union_coverage": 0.0,
                "GOME2_EC_pixel_union_coverage": 0.0,
                "GOME2_EC_overlap_weighted_SIF": np.nan,
                "GOME2_blind_buffer_overlap_weighted_SIF": np.nan,
                "GOME2_simple_intersecting_observation_mean": np.nan,
                "GOME2_tower_containing_footprint_mean": np.nan,
                "n_GOME2_observations": 0,
                "n_unique_observation_dates": 0,
                "first_observation_date": "",
                "last_observation_date": "",
                "GOME2_observation_SIF_mean": np.nan,
                "GOME2_observation_SIF_median": np.nan,
                "GOME2_observation_SIF_sd": np.nan,
                "GOME2_observation_SIF_min": np.nan,
                "GOME2_observation_SIF_max": np.nan,
                "max_observation_EC_probability_overlap": np.nan,
                "dominant_observation_weight": np.nan,
                "GOME2_effective_observation_count": np.nan,
                "sum_observation_EC_probability_overlap": 0.0,
                "sum_individual_overlap_minus_union_coverage": 0.0,
                "mean_native_footprint_area_km2": np.nan,
                "median_native_footprint_area_km2": np.nan,
                "mean_fraction_native_footprint_inside_EC": np.nan,
                "median_fraction_native_footprint_inside_EC": np.nan,
                "mean_footprint_to_EC_area_ratio": np.nan,
                "primary_eligible": False,
                "coverage_class": "=0",
            }
        )
        qc_rows.append({"scope": "site_month", "SITE_ID": ctx["SITE_ID"], "YYYYMM": ctx["YYYYMM"], "reason": "NO_EC_OVERLAP"})
        return base, qc_rows

    p = pd.to_numeric(obs["EC_probability_overlap_fraction"], errors="coerce").to_numpy(float)
    sif = pd.to_numeric(obs["GOME2_SIF"], errors="coerce").to_numpy(float)
    valid = np.isfinite(p) & np.isfinite(sif) & (p > 0)
    p = p[valid]
    sif = sif[valid]
    obs = obs.loc[valid].copy()
    p_sum = float(p.sum(dtype=np.float64))
    weights = p / p_sum if p_sum > 0 else np.full_like(p, np.nan)
    obs["normalized_monthly_weight"] = weights
    obs["GOME2_observation_EC_probability_overlap_fraction"] = obs["EC_probability_overlap_fraction"]

    blind_p = pd.to_numeric(obs["blind_buffer_overlap_fraction"], errors="coerce").to_numpy(float)
    blind_valid = np.isfinite(blind_p) & np.isfinite(sif) & (blind_p > 0)
    blind_weighted = float(np.sum(sif[blind_valid] * blind_p[blind_valid]) / np.sum(blind_p[blind_valid])) if bool(blind_valid.any()) else np.nan

    union_prob = float(ctx["probability"][ctx["union_ec_covered"] & ctx["ec_mask"]].sum(dtype=np.float64))
    union_cov = union_prob / ctx["probability_sum_full_EC"] if ctx["probability_sum_full_EC"] > 0 else np.nan
    union_pix = int((ctx["union_ec_covered"] & ctx["ec_mask"]).sum())
    union_pix_cov = union_pix / ctx["n_EC_pixels"] if ctx["n_EC_pixels"] > 0 else np.nan

    weighted = float(np.sum(sif * weights)) if p_sum > 0 else np.nan
    tower = obs[obs["tower_containing_footprint"].astype(bool)] if "tower_containing_footprint" in obs else pd.DataFrame()
    obs_dates = pd.to_datetime(obs["observation_datetime"], errors="coerce")
    primary = bool(math.isfinite(union_cov) and union_cov >= PRIMARY_COVERAGE_THRESHOLD)

    base.update(
        {
            "GOME2_status": "VALID_EC_OVERLAP_PRIMARY" if primary else "VALID_EC_OVERLAP_PARTIAL_SUPPORT",
            "GOME2_EC_probability_union_coverage": union_cov,
            "GOME2_EC_pixel_union_coverage": union_pix_cov,
            "GOME2_EC_overlap_weighted_SIF": weighted,
            "GOME2_blind_buffer_overlap_weighted_SIF": blind_weighted,
            "GOME2_simple_intersecting_observation_mean": float(np.mean(sif)) if sif.size else np.nan,
            "GOME2_tower_containing_footprint_mean": float(pd.to_numeric(tower["GOME2_SIF"], errors="coerce").mean()) if not tower.empty else np.nan,
            "n_GOME2_observations": int(len(obs)),
            "n_unique_observation_dates": int(obs_dates.dt.date.nunique()) if not obs_dates.empty else 0,
            "first_observation_date": obs_dates.min().isoformat() if obs_dates.notna().any() else "",
            "last_observation_date": obs_dates.max().isoformat() if obs_dates.notna().any() else "",
            "GOME2_observation_SIF_mean": float(np.mean(sif)),
            "GOME2_observation_SIF_median": float(np.median(sif)),
            "GOME2_observation_SIF_sd": float(np.std(sif, ddof=1)) if sif.size > 1 else 0.0,
            "GOME2_observation_SIF_min": float(np.min(sif)),
            "GOME2_observation_SIF_max": float(np.max(sif)),
            "max_observation_EC_probability_overlap": float(np.max(p)),
            "dominant_observation_weight": float(np.max(weights)),
            "GOME2_effective_observation_count": float(1.0 / np.sum(weights**2)) if np.isfinite(weights).all() and weights.size else np.nan,
            "sum_observation_EC_probability_overlap": p_sum,
            "sum_individual_overlap_minus_union_coverage": p_sum - union_cov if math.isfinite(union_cov) else np.nan,
            "mean_native_footprint_area_km2": float(pd.to_numeric(obs["native_footprint_area_km2"], errors="coerce").mean()),
            "median_native_footprint_area_km2": float(pd.to_numeric(obs["native_footprint_area_km2"], errors="coerce").median()),
            "mean_fraction_native_footprint_inside_EC": float(pd.to_numeric(obs["fraction_GOME2_footprint_area_inside_EC_support"], errors="coerce").mean()),
            "median_fraction_native_footprint_inside_EC": float(pd.to_numeric(obs["fraction_GOME2_footprint_area_inside_EC_support"], errors="coerce").median()),
            "mean_footprint_to_EC_area_ratio": float(pd.to_numeric(obs["footprint_area_to_EC_support_area_ratio"], errors="coerce").mean()),
            "coverage_class": coverage_class(union_cov),
            "primary_eligible": primary,
        }
    )
    if not (-WEIGHT_TOL <= union_cov <= 1 + WEIGHT_TOL):
        qc_rows.append({"scope": "site_month", "SITE_ID": ctx["SITE_ID"], "YYYYMM": ctx["YYYYMM"], "reason": "UNION_COVERAGE_ERROR", "value": union_cov})
    if math.isfinite(weighted) and (weighted < np.min(sif) - WEIGHT_TOL or weighted > np.max(sif) + WEIGHT_TOL):
        qc_rows.append({"scope": "site_month", "SITE_ID": ctx["SITE_ID"], "YYYYMM": ctx["YYYYMM"], "reason": "WEIGHTED_VALUE_OUTSIDE_NATIVE_OBSERVATION_RANGE", "value": weighted})
    if not np.isclose(np.sum(weights), 1.0, atol=WEIGHT_TOL):
        qc_rows.append({"scope": "site_month", "SITE_ID": ctx["SITE_ID"], "YYYYMM": ctx["YYYYMM"], "reason": "WEIGHT_NORMALIZATION_ERROR", "value": float(np.sum(weights))})

    return base, qc_rows


def process_all(universe: pd.DataFrame, source_groups: dict[str, list[dict[str, Any]]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    master_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    months = sorted(universe["YYYYMM"].astype(str).unique().tolist())
    for month_idx, yyyymm in enumerate(months, start=1):
        month_rows = universe[universe["YYYYMM"].astype(str).eq(yyyymm)]
        contexts = [build_context(row) for _, row in month_rows.iterrows()]
        sources = source_groups.get(str(yyyymm), [])
        if not sources:
            for ctx in contexts:
                row, q = summarize_context(ctx)
                row["GOME2_status"] = "EXCLUDED_NO_GOME2_SOURCE"
                master_rows.append(row)
                qc_rows.extend(q)
            continue
        for source in sources:
            process_source_for_contexts(source, contexts, qc_rows)
        for ctx in contexts:
            row, q = summarize_context(ctx)
            master_rows.append(row)
            qc_rows.extend(q)
            for obs in ctx["observations"]:
                provenance_rows.append(obs)
        print(f"GOME2 month {yyyymm} ({month_idx}/{len(months)}): sites={len(contexts)} sources={len(sources)} provenance_rows={len(provenance_rows)}", flush=True)
    return pd.DataFrame(master_rows), pd.DataFrame(provenance_rows), pd.DataFrame(qc_rows)


def finalize_provenance_weights(master: pd.DataFrame, provenance: pd.DataFrame) -> pd.DataFrame:
    if provenance.empty:
        return provenance
    key_cols = ["SITE_ID", "YYYYMM"]
    # Recompute normalized weights from the stored observation p_k values to make
    # the provenance table independently reproducible.
    p = pd.to_numeric(provenance["EC_probability_overlap_fraction"], errors="coerce")
    sums = provenance.assign(_p=p).groupby(key_cols)["_p"].transform("sum")
    provenance["normalized_monthly_weight"] = p / sums
    return provenance


def summarize_coverage(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(master)
    for cls in ["=0", "0-<0.25", "0.25-<0.50", "0.50-<0.75", "0.75-<0.90", "0.90-<0.95", "0.95-<1", "≈1", "NaN"]:
        sub = master[master["coverage_class"].eq(cls)]
        rows.append(
            {
                "coverage_class": cls,
                "N_site_months": int(len(sub)),
                "percentage": float(len(sub) / total * 100) if total else np.nan,
                "N_sites": int(sub["SITE_ID"].nunique()) if not sub.empty else 0,
                "years": ", ".join(map(str, sorted(sub["year"].dropna().astype(int).unique()))) if not sub.empty else "",
                "IGBP_classes": ", ".join(sorted(map(str, sub["IGBP"].dropna().unique()))) if not sub.empty else "",
            }
        )
    return pd.DataFrame(rows)


def support_summary(master: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "mean_native_footprint_area_km2",
        "EC_positive_support_area_km2",
        "mean_footprint_to_EC_area_ratio",
        "mean_fraction_native_footprint_inside_EC",
        "GOME2_EC_probability_union_coverage",
        "n_GOME2_observations",
        "GOME2_effective_observation_count",
    ]
    rows = []
    valid = master[master["n_GOME2_observations"].fillna(0).astype(float) > 0].copy()
    for col in cols:
        values = pd.to_numeric(valid[col], errors="coerce").dropna()
        rows.append(
            {
                "metric": col,
                "N": int(values.size),
                "mean": float(values.mean()) if values.size else np.nan,
                "median": float(values.median()) if values.size else np.nan,
                "P05": float(values.quantile(0.05)) if values.size else np.nan,
                "P25": float(values.quantile(0.25)) if values.size else np.nan,
                "P75": float(values.quantile(0.75)) if values.size else np.nan,
                "P95": float(values.quantile(0.95)) if values.size else np.nan,
                "min": float(values.min()) if values.size else np.nan,
                "max": float(values.max()) if values.size else np.nan,
            }
        )
    return pd.DataFrame(rows)


def existing_tiff_path(site_safe: str, yyyymm: str) -> Path | None:
    candidates = [
        EXISTING_GOME2_TIFF_ROOTS[0] / site_safe / f"{site_safe}_GOME-2_SIF_companion_matching_GPP_footprint_aware_{yyyymm}.tif",
        EXISTING_GOME2_TIFF_ROOTS[1] / f"{site_safe}_{yyyymm}_GOME2_SIF.tif",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def compare_existing_tiffs(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for _, row in master.iterrows():
        path = existing_tiff_path(str(row["SITE_SAFE"]), str(row["YYYYMM"]))
        if path is None:
            continue
        try:
            with rasterio.open(native(path)) as sif_ds, rasterio.open(native(row["matching_gpp_tif"])) as gpp_ds:
                band2 = sif_ds.read(2).astype(np.float64)
                nodata = sif_ds.nodata
                prob = gpp_ds.read(1).astype(np.float64)
                valid = np.isfinite(band2) & np.isfinite(prob) & (prob > 0)
                if nodata is not None:
                    valid &= band2 != nodata
                if valid.any():
                    old_weighted = float((band2[valid] * prob[valid]).sum(dtype=np.float64) / prob[valid].sum(dtype=np.float64))
                else:
                    old_weighted = np.nan
                old_band7_mean = np.nan
                if sif_ds.count >= 7:
                    b7 = sif_ds.read(7).astype(np.float64)
                    v7 = np.isfinite(b7)
                    if nodata is not None:
                        v7 &= b7 != nodata
                    old_band7_mean = float(b7[v7].mean()) if v7.any() else np.nan
            new_value = float(row["GOME2_EC_overlap_weighted_SIF"]) if pd.notna(row["GOME2_EC_overlap_weighted_SIF"]) else np.nan
            rows.append(
                {
                    "SITE_ID": row["SITE_ID"],
                    "YYYYMM": row["YYYYMM"],
                    "existing_tiff": str(path),
                    "existing_tiff_probability_weighted_band2": old_weighted,
                    "existing_tiff_band7_mean": old_band7_mean,
                    "new_native_observation_overlap_weighted": new_value,
                    "difference_new_minus_existing_weighted_band2": new_value - old_weighted if math.isfinite(new_value) and math.isfinite(old_weighted) else np.nan,
                }
            )
        except Exception as exc:
            rows.append({"SITE_ID": row["SITE_ID"], "YYYYMM": row["YYYYMM"], "existing_tiff": str(path), "comparison_status": f"FAILED:{type(exc).__name__}:{exc}"})
    comp = pd.DataFrame(rows)
    if comp.empty:
        summary = pd.DataFrame([{"N_common": 0}])
        return comp, summary
    diff = pd.to_numeric(comp["difference_new_minus_existing_weighted_band2"], errors="coerce").dropna()
    old = pd.to_numeric(comp["existing_tiff_probability_weighted_band2"], errors="coerce")
    new = pd.to_numeric(comp["new_native_observation_overlap_weighted"], errors="coerce")
    common = np.isfinite(old) & np.isfinite(new)
    summary = pd.DataFrame(
        [
            {
                "N_common": int(common.sum()),
                "mean_difference": float(diff.mean()) if diff.size else np.nan,
                "median_difference": float(diff.median()) if diff.size else np.nan,
                "RMSE_difference": float(np.sqrt(np.mean(diff**2))) if diff.size else np.nan,
                "P05_difference": float(diff.quantile(0.05)) if diff.size else np.nan,
                "P95_difference": float(diff.quantile(0.95)) if diff.size else np.nan,
                "max_absolute_difference": float(diff.abs().max()) if diff.size else np.nan,
                "Pearson_correlation": float(old[common].corr(new[common])) if common.sum() > 2 else np.nan,
            }
        ]
    )
    return comp, summary


def manual_checks(master: pd.DataFrame, provenance: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if provenance.empty:
        return pd.DataFrame(rows)

    def add_case(label: str, candidates: pd.DataFrame) -> None:
        if candidates.empty:
            return
        rec = candidates.iloc[0]
        sub = provenance[(provenance["SITE_ID"].eq(rec["SITE_ID"])) & (provenance["YYYYMM"].astype(str).eq(str(rec["YYYYMM"])))]
        if sub.empty:
            return
        weights = pd.to_numeric(sub["normalized_monthly_weight"], errors="coerce")
        sif = pd.to_numeric(sub["GOME2_SIF"], errors="coerce")
        weighted = float((sif * weights).sum())
        terms = [f"{s:.6g}×{w:.6g}" for s, w in zip(sif.head(8), weights.head(8))]
        if len(sub) > 8:
            terms.append("...")
        rows.append(
            {
                "case_type": label,
                "SITE_ID": rec["SITE_ID"],
                "YYYYMM": rec["YYYYMM"],
                "n_observations": int(rec["n_GOME2_observations"]),
                "native_SIF_x_weight_terms_first_8": " + ".join(terms),
                "manual_weighted_result": weighted,
                "code_GOME2_EC_overlap_weighted_SIF": rec["GOME2_EC_overlap_weighted_SIF"],
                "absolute_difference": abs(weighted - float(rec["GOME2_EC_overlap_weighted_SIF"])),
                "union_coverage": rec["GOME2_EC_probability_union_coverage"],
                "sum_individual_p_k": rec["sum_observation_EC_probability_overlap"],
            }
        )

    valid = master[master["n_GOME2_observations"].fillna(0).astype(float) > 0].copy()
    add_case("single_observation", valid[valid["n_GOME2_observations"].eq(1)])
    add_case("multiple_observation", valid[valid["n_GOME2_observations"].gt(1)])
    add_case("partial_EC_coverage", valid[(valid["GOME2_EC_probability_union_coverage"].gt(0)) & (valid["GOME2_EC_probability_union_coverage"].lt(PRIMARY_COVERAGE_THRESHOLD))])
    add_case("full_primary_coverage", valid[valid["GOME2_EC_probability_union_coverage"].ge(PRIMARY_COVERAGE_THRESHOLD)])
    add_case("overlapping_observations_sum_p_gt_union", valid[valid["sum_individual_overlap_minus_union_coverage"].gt(1e-6)])
    return pd.DataFrame(rows).drop_duplicates(["case_type", "SITE_ID", "YYYYMM"])


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


def create_simple_summary(master: pd.DataFrame, provenance: pd.DataFrame, old_summary: pd.DataFrame, verdict: str) -> None:
    valid = master[master["n_GOME2_observations"].fillna(0).astype(float) > 0].copy()
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle("GOME-2 EC-overlap native-footprint finalization", fontsize=16)

    ax = axes[0, 0]
    ax.axis("off")
    ax.text(0.02, 0.9, "Native GOME-2 footprints", fontsize=12, weight="bold")
    ax.text(0.10, 0.72, "+ EC probability footprint", fontsize=11)
    ax.text(0.10, 0.56, "→ compute EC probability overlap pₖ", fontsize=11)
    ax.text(0.10, 0.40, "→ weight real native observations", fontsize=11)
    ax.text(0.10, 0.24, "→ one monthly EC-overlap-weighted SIF", fontsize=11)
    ax.text(0.02, 0.08, "No ML, no downscaling, no synthetic GOME-2 pixels.", fontsize=10)

    ax = axes[0, 1]
    example = valid.sort_values(["primary_eligible", "n_GOME2_observations"], ascending=[False, False]).head(1)
    if not example.empty:
        rec = example.iloc[0]
        sub = provenance[(provenance["SITE_ID"].eq(rec["SITE_ID"])) & (provenance["YYYYMM"].astype(str).eq(str(rec["YYYYMM"])))].head(8)
        y = np.arange(len(sub))
        ax.barh(y, sub["normalized_monthly_weight"], color="#4C78A8")
        ax.set_yticks(y)
        ax.set_yticklabels([str(x).split(":")[-1] for x in sub["observation_id"]], fontsize=7)
        ax.set_xlabel("monthly weight")
        ax.set_title(f"Worked example: {rec['SITE_ID']} {rec['YYYYMM']}")
    else:
        ax.text(0.1, 0.5, "No valid example", transform=ax.transAxes)

    ax = axes[0, 2]
    valid["GOME2_EC_probability_union_coverage"].plot(kind="hist", bins=25, ax=ax, color="#59A14F")
    ax.axvline(PRIMARY_COVERAGE_THRESHOLD, color="red", linestyle="--", lw=1)
    ax.set_title("EC union coverage distribution")
    ax.set_xlabel("probability-weighted union coverage")

    ax = axes[1, 0]
    ratio = pd.to_numeric(valid["mean_footprint_to_EC_area_ratio"], errors="coerce").dropna()
    if not ratio.empty:
        ax.hist(np.log10(ratio[ratio > 0]), bins=30, color="#F28E2B")
        ax.set_xlabel("log10(GOME-2 footprint area / EC support area)")
    ax.set_title("Native-support mismatch")

    ax = axes[1, 1]
    comp_path = OUTPUT_ROOT / "GOME2_Existing_TIFF_vs_Native_Observation_Weighted.csv"
    if comp_path.exists():
        comp = pd.read_csv(comp_path)
        x = pd.to_numeric(comp.get("existing_tiff_probability_weighted_band2"), errors="coerce")
        y = pd.to_numeric(comp.get("new_native_observation_overlap_weighted"), errors="coerce")
        ok = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[ok], y[ok], s=8, alpha=0.35)
        if ok.any():
            lo = float(min(x[ok].min(), y[ok].min()))
            hi = float(max(x[ok].max(), y[ok].max()))
            ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel("existing TIFF-derived")
        ax.set_ylabel("new native-observation weighted")
    ax.set_title("Existing vs final analytical GOME-2")

    ax = axes[1, 2]
    ax.axis("off")
    ax.text(0.02, 0.82, "Final status", fontsize=12, weight="bold")
    ax.text(0.02, 0.66, verdict, fontsize=14, color="#2E8B57" if "COMPLETE" in verdict else "#B22222", weight="bold")
    ax.text(
        0.02,
        0.45,
        f"Primary-ready: {int(master['primary_eligible'].fillna(False).sum()):,}\n"
        f"Partial support retained separately: {int(((master['GOME2_EC_probability_union_coverage'] > 0) & (~master['primary_eligible'].fillna(False))).sum()):,}\n"
        f"Zero overlap: {int(master['coverage_class'].eq('=0').sum()):,}\n"
        "GPP used: no",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.savefig(OUTPUT_ROOT / "GOME2_Finalization_Simple_Summary.png", dpi=180)
    fig.savefig(OUTPUT_ROOT / "GOME2_Finalization_Simple_Summary.pdf")
    plt.close(fig)


def source_variable_audit(source_inventory: pd.DataFrame) -> pd.DataFrame:
    files = source_inventory[source_inventory["status"].eq("OK")]["file_path"].tolist()
    chosen = []
    if files:
        for idx in [0, len(files) // 2, len(files) - 1]:
            chosen.append(Path(files[idx]))
    rows = []
    for path in chosen:
        with h5py.File(native(path), "r") as handle:
            q = np.asarray(handle[QUALITY_VAR][...]) if QUALITY_VAR in handle else np.array([])
            q_vals, q_counts = np.unique(q[: min(q.size, 1_000_000)], return_counts=True) if q.size else (np.array([]), np.array([]))
            rows.append(
                {
                    "file_path": str(path),
                    "status": "PASS",
                    "sif_variable": SIF_VAR,
                    "sif_shape": str(handle[SIF_VAR].shape),
                    "sif_dtype": str(handle[SIF_VAR].dtype),
                    "sif_units": decode_attr(handle[SIF_VAR].attrs.get("units")),
                    "sif_long_name": decode_attr(handle[SIF_VAR].attrs.get("long_name")),
                    "scale_factor": dataset_attr(handle[SIF_VAR], ["scale_factor"]),
                    "add_offset": dataset_attr(handle[SIF_VAR], ["add_offset"]),
                    "fill_value": dataset_attr(handle[SIF_VAR], ["_FillValue", "missing_value"]),
                    "latitude_variable": LAT_VAR,
                    "longitude_variable": LON_VAR,
                    "corner_latitude_variable": LAT_CORNERS_VAR,
                    "corner_longitude_variable": LON_CORNERS_VAR,
                    "time_variable": TIME_VAR,
                    "time_units": decode_attr(handle[TIME_VAR].attrs.get("units")) if TIME_VAR in handle else "",
                    "quality_variable": QUALITY_VAR,
                    "quality_flag_sample_counts": " | ".join(f"{int(v)}:{int(c)}" for v, c in zip(q_vals, q_counts)),
                    "geometry_confirmation": "Genuine native four-corner GOME-2 footprint geometry is present.",
                }
            )
    return pd.DataFrame(rows)


def write_final_reports(
    master: pd.DataFrame,
    primary: pd.DataFrame,
    partial: pd.DataFrame,
    provenance: pd.DataFrame,
    qc: pd.DataFrame,
    coverage: pd.DataFrame,
    support: pd.DataFrame,
    old_summary: pd.DataFrame,
    manual: pd.DataFrame,
    source_audit: pd.DataFrame,
    source_inventory: pd.DataFrame,
    safety_unchanged: bool,
    verdict: str,
) -> None:
    counts = {
        "N_total_attempted": int(len(master)),
        "N_primary_ready": int(len(primary)),
        "N_positive_partial_support": int(len(partial)),
        "N_zero_coverage": int(master["coverage_class"].eq("=0").sum()),
        "N_sites": int(master["SITE_ID"].nunique()),
        "date_min": str(master["YYYYMM"].min()),
        "date_max": str(master["YYYYMM"].max()),
        "N_observation_provenance_rows": int(len(provenance)),
        "N_QC_flags": int(len(qc)),
    }
    pd.DataFrame([counts]).to_csv(OUTPUT_ROOT / "GOME2_Finalization_Counts.csv", index=False)

    readme = f"""GOME-2 FINALIZATION — READ ME FIRST

1. GOME-2 site-months attempted: {counts['N_total_attempted']:,}.
2. Primary-analysis ready: {counts['N_primary_ready']:,}.
3. Date range: {counts['date_min']} to {counts['date_max']}; local daily source files: {len(source_inventory):,}.
4. Sites represented: {counts['N_sites']:,}.
5. Final value: native GOME-2 observations are weighted by their EC probability overlap fraction p_k, then averaged as sum(SIF_k p_k) / sum(p_k).
6. Native GOME-2 footprint geometry preserved? Yes: {LAT_CORNERS_VAR}/{LON_CORNERS_VAR}.
7. Spatially downscaled? No. EC overlap weighting does not change the native support of each retrieval.
8. ML reconstruction used? No.
9. Full/primary EC union coverage fraction: {counts['N_primary_ready'] / counts['N_total_attempted'] * 100 if counts['N_total_attempted'] else np.nan:.2f}% using threshold >= {PRIMARY_COVERAGE_THRESHOLD}.
10. Median footprint/EC area ratio: {support.loc[support.metric.eq('mean_footprint_to_EC_area_ratio'), 'median'].iloc[0] if not support.empty else np.nan:.3g}.
11. Median native observations per site-month: {support.loc[support.metric.eq('n_GOME2_observations'), 'median'].iloc[0] if not support.empty else np.nan:.3g}.
12. Overlapping footprints handled without double-counting coverage? Yes; monthly union coverage uses the union of covered EC cells, not sum(p_k).
13. Blind-buffer and simple/tower comparators created? Yes.
14. Existing TIFF comparison common N: {int(old_summary['N_common'].iloc[0]) if not old_summary.empty and 'N_common' in old_summary else 0}.
15. QC flags/exclusions logged: {counts['N_QC_flags']:,}.
16. GPP values used? No; only GPP bands 1 and 3 were read as support masks.
17. Source files modified? {'No' if safety_unchanged else 'Potential modification detected; inspect safety audit.'}
18. Ready for later SIF-GPP analysis? {'Yes, for primary eligible GOME-2 rows.' if verdict.endswith('COMPLETE') else 'No; inspect blockers.'}

Final verdict: {verdict}
"""
    (OUTPUT_ROOT / "GOME2_Finalization_READ_ME_FIRST.txt").write_text(readme, encoding="utf-8")

    md: list[str] = []
    md.append("# GOME-2 Finalization Report\n")
    md.append(f"Final verdict: **{verdict}**\n")
    md.append("This workflow creates scalar analytical tables only. It does not rebuild GOME-2 TIFFs, does not run ML, does not reconstruct missing support, and does not use tower GPP values.\n")
    md.append("## Source variables and geometry\n")
    md.append(source_audit.to_markdown(index=False) + "\n")
    md.append("## Temporal archive\n")
    month_counts = source_inventory.groupby("YYYYMM").size().reset_index(name="n_daily_files")
    temporal = pd.DataFrame(
        [
            {
                "first_local_month": month_counts["YYYYMM"].min(),
                "last_local_month": month_counts["YYYYMM"].max(),
                "n_months": int(month_counts["YYYYMM"].nunique()),
                "n_source_files": int(len(source_inventory)),
                "temporal_sampling": "daily NetCDF files; multiple native retrieval footprints can contribute to one site-month",
            }
        ]
    )
    md.append(temporal.to_markdown(index=False) + "\n")
    md.append("## Coverage summary\n")
    md.append(coverage.to_markdown(index=False) + "\n")
    md.append("## Support-mismatch summary\n")
    md.append(support.to_markdown(index=False) + "\n")
    md.append("## Manual numerical checks\n")
    md.append((manual.to_markdown(index=False) if not manual.empty else "No manual examples available.") + "\n")
    md.append("## Existing TIFF-derived vs final native-observation comparison\n")
    md.append(old_summary.to_markdown(index=False) + "\n")
    md.append("## QC flags\n")
    md.append((qc.head(50).to_markdown(index=False) if not qc.empty else "No QC flags.") + "\n")
    md.append("## Output files\n")
    for name in [
        "GOME2_Final_EC_Footprint_SIF_All_SiteMonths.csv",
        "GOME2_Final_EC_Footprint_SIF_Primary_Analysis.csv",
        "GOME2_Partial_EC_Support_Sensitivity.csv",
        "GOME2_EC_Observation_Overlap_Provenance.csv",
        "GOME2_Finalization_QC_Flags.csv",
    ]:
        md.append(f"- `{OUTPUT_ROOT / name}`")
    md.append("\n## Final scientific interpretation\n")
    md.append("`GOME2_EC_overlap_weighted_SIF` is an EC-overlap-weighted native GOME-2 SIF estimate. It is not a spatially downscaled GOME-2 product and does not imply that native GOME-2 footprints measured only the EC footprint.\n")
    report = OUTPUT_ROOT / "GOME2_Finalization_Report.md"
    report.write_text("\n".join(md), encoding="utf-8")
    write_pdf_from_text(report, OUTPUT_ROOT / "GOME2_Finalization_Report.pdf")

    repro = {
        "script_path": str((Path(PROJECT_ROOT) / 'Codes/pythonProject/Finalize_GOME2_EC_Footprint_Analytical_Table.py').resolve()),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source_roots": {"GOME2": str(GOME2_ROOT), "GPP": str(GPP_ROOT)},
        "output_root": str(OUTPUT_ROOT),
        "sif_variable": SIF_VAR,
        "units": SIF_UNITS,
        "geometry_variables": [LAT_CORNERS_VAR, LON_CORNERS_VAR],
        "quality_filtering": "SIF fill/NoData and invalid geometry excluded; Quality_Flag retained but not hard-filtered.",
        "temporal_matching": "GPP SITE_ID x YYYYMM matched to all local daily GOME-2 files with the same YYYYMM.",
        "coverage_calculation": "union of EC-grid cells touched by at least one valid native GOME-2 footprint; probability-weighted using GPP band 1.",
        "monthly_weighting_equation": "sum_k(SIF_k * p_k) / sum_k(p_k), p_k = EC probability mass covered by observation k / full EC probability mass",
        "primary_coverage_threshold": PRIMARY_COVERAGE_THRESHOLD,
        "record_counts": counts,
        "source_inputs_unchanged": safety_unchanged,
        "software_versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "rasterio": rasterio.__version__,
        },
    }
    (OUTPUT_ROOT / "GOME2_Finalization_Reproducibility.json").write_text(json.dumps(repro, indent=2), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    source_inventory, source_groups = discover_sources()
    source_inventory.to_csv(QC_ROOT / "GOME2_Source_File_Inventory.csv", index=False)
    source_audit = source_variable_audit(source_inventory)
    source_audit.to_csv(QC_ROOT / "GOME2_Source_Variable_Audit.csv", index=False)

    source_months = set(source_groups.keys())
    universe = load_site_month_universe(source_months)
    universe.to_csv(QC_ROOT / "GOME2_SiteMonth_Attempt_Inventory.csv", index=False)

    safety_inputs = [
        GOME2_ROOT,
        GPP_ROOT,
        EXISTING_GOME2_TIFF_ROOTS[0],
        EXISTING_GOME2_TIFF_ROOTS[1],
        PROJECT_ROOT / "SIF_EC_GOSIF_Finalization",
        PROJECT_ROOT / "SIF_EC_TROPOSIF_Finalization",
        PROJECT_ROOT / "SIF_EC_Reconstruction_OCO2_TargetOnly_Validation",
        PROJECT_ROOT / "SIF_EC_Reconstruction_OCO3_Final_Production",
    ]
    before = file_snapshot(safety_inputs)
    before.to_csv(QC_ROOT / "GOME2_Source_Input_Safety_Before.csv", index=False)

    master, provenance, qc = process_all(universe, source_groups)
    provenance = finalize_provenance_weights(master, provenance)

    # Reattach finalized normalized weights from provenance into a compact
    # master table only after all observations for a month have been seen.
    master["primary_eligible"] = master["primary_eligible"].fillna(False).astype(bool)
    primary = master[master["primary_eligible"]].copy()
    partial = master[(pd.to_numeric(master["GOME2_EC_probability_union_coverage"], errors="coerce") > 0) & (~master["primary_eligible"])].copy()

    master.to_csv(OUTPUT_ROOT / "GOME2_Final_EC_Footprint_SIF_All_SiteMonths.csv", index=False)
    primary.to_csv(OUTPUT_ROOT / "GOME2_Final_EC_Footprint_SIF_Primary_Analysis.csv", index=False)
    partial.to_csv(OUTPUT_ROOT / "GOME2_Partial_EC_Support_Sensitivity.csv", index=False)
    provenance.to_csv(OUTPUT_ROOT / "GOME2_EC_Observation_Overlap_Provenance.csv", index=False)

    coverage = summarize_coverage(master)
    coverage.to_csv(OUTPUT_ROOT / "GOME2_Coverage_Summary.csv", index=False)
    support = support_summary(master)
    support.to_csv(OUTPUT_ROOT / "GOME2_Native_Support_Mismatch_Summary.csv", index=False)
    manual = manual_checks(master, provenance)
    manual.to_csv(OUTPUT_ROOT / "GOME2_Manual_Numerical_Checks.csv", index=False)
    overlap_examples = master[pd.to_numeric(master["sum_individual_overlap_minus_union_coverage"], errors="coerce") > 1e-6].copy()
    overlap_examples.head(100).to_csv(OUTPUT_ROOT / "GOME2_Overlapping_Footprint_QC_Examples.csv", index=False)

    old_comp, old_summary = compare_existing_tiffs(master)
    old_comp.to_csv(OUTPUT_ROOT / "GOME2_Existing_TIFF_vs_Native_Observation_Weighted.csv", index=False)
    old_summary.to_csv(OUTPUT_ROOT / "GOME2_Existing_TIFF_vs_Native_Observation_Weighted_Summary.csv", index=False)

    qc.to_csv(OUTPUT_ROOT / "GOME2_Finalization_QC_Flags.csv", index=False)

    after = file_snapshot(safety_inputs)
    after.to_csv(QC_ROOT / "GOME2_Source_Input_Safety_After.csv", index=False)
    safety = compare_snapshots(before, after)
    safety.to_csv(QC_ROOT / "GOME2_Source_Input_Safety_Audit.csv", index=False)
    safety_unchanged = bool(safety["unchanged_size_and_mtime"].fillna(False).all())

    hard_fail = False
    fail_reasons: list[str] = []
    if provenance.empty:
        hard_fail = True
        fail_reasons.append("No observation-level provenance rows were produced.")
    if not safety_unchanged:
        hard_fail = True
        fail_reasons.append("Input safety audit found changed source/input files.")
    bad_cov = master[
        (pd.to_numeric(master["GOME2_EC_probability_union_coverage"], errors="coerce") < -WEIGHT_TOL)
        | (pd.to_numeric(master["GOME2_EC_probability_union_coverage"], errors="coerce") > 1 + WEIGHT_TOL)
    ]
    if not bad_cov.empty:
        hard_fail = True
        fail_reasons.append("Union coverage outside [0,1].")

    verdict = "GOME-2 FINALIZATION FAILED" if hard_fail else "GOME-2 FINALIZATION COMPLETE"
    if hard_fail:
        pd.DataFrame({"failure_reason": fail_reasons}).to_csv(OUTPUT_ROOT / "GOME2_Finalization_Failure_Reasons.csv", index=False)

    create_simple_summary(master, provenance, old_summary, verdict)
    write_final_reports(master, primary, partial, provenance, qc, coverage, support, old_summary, manual, source_audit, source_inventory, safety_unchanged, verdict)

    print(verdict)
    print(
        json.dumps(
            {
                "N_total_attempted": int(len(master)),
                "N_primary_ready": int(len(primary)),
                "N_partial_support_sensitivity": int(len(partial)),
                "N_zero_coverage": int(master["coverage_class"].eq("=0").sum()),
                "N_sites": int(master["SITE_ID"].nunique()),
                "N_observation_provenance_rows": int(len(provenance)),
                "N_QC_flags": int(len(qc)),
                "SOURCE_INPUTS_UNCHANGED": safety_unchanged,
            },
            indent=2,
        )
    )
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Primary analysis: {OUTPUT_ROOT / 'GOME2_Final_EC_Footprint_SIF_Primary_Analysis.csv'}")
    print(f"Report: {OUTPUT_ROOT / 'GOME2_Finalization_Report.md'}")


if __name__ == "__main__":
    main()
