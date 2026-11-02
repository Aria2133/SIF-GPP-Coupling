"""
TROPOSIF QA sensitivity: independent re-computation of the EC-footprint
probability-weighted monthly SIF value directly from the 227 raw TROPOSIF L2B
netCDF files, run twice -- once UNFILTERED (mirrors the current production
rule) and once QA-FILTERED using the officially recommended TROPOSIF quality
gate approximated from the fields actually present in the raw files.

Why this script exists
-----------------------
An earlier audit (SIF_EC_SIF_GPP_Independent_Statistical_Audit\04_product_QA\
troposif_QA_audit.csv) flagged "QA threshold" and "Uncertainty filtering" as
UNCERTAIN because the analysis-ready master alone does not show whether/how a
quality gate was applied upstream of TROPOSIF_final_EC_SIF. This script
resolves that by rebuilding the EC-probability-weighted SIF value from raw
soundings under two explicit rules and comparing the resulting SIF-GPP
correlations.

Official TROPOSIF documentation (Guanter et al., ESSD 2021,
https://essd.copernicus.org/articles/13/5423/2021/):
  - SIF_Corr_743 (743 nm baseline retrieval) is the recommended field. This
    project's production pipeline already uses PRODUCT/SIF_Corr_743 as the
    first-priority candidate (see SIF_TIFF_FOOTPRINT_AWARE_HELPERS.py).
  - Recommended QA gate: qa_value = 1.0 (qa_value > 0.5), which folds in
    view zenith angle < 60 deg, sun zenith angle < 70 deg, and reduced
    chi-squared in [0.6, 2].
  - "A strict cloud filtering (e.g. cloud fraction lower than 0.2) should be
    applied if an accurate SIF mean value is needed."
  - Negative SIF retrievals should NOT be removed (removing them biases the
    mean upward). This script never drops negative SIF_Corr_743 values.
  - No recommendation to filter on SIF_ERROR magnitude.

LIMITATION (documented, not hidden): the raw TROPOSIF L2B files inspected for
this project (e.g. SIF Datasets\\TROPOSIF\\2018.05\\TROPOSIF_L2B_2018-05-01.nc)
contain NO literal `qa_value` or "reduced chi-squared" variable. The exact
official composite gate (qa_value > 0.5) can therefore not be reproduced
bit-for-bit from local data. Instead this script applies the CLOSEST
achievable approximation using fields that DO exist in the raw files:
    cloud_fraction_L2 < 0.2  AND  viewing_zenith_angle < 60  AND
    solar_zenith_angle < 70
This is the officially-recommended cloud gate plus the two zenith-angle
components of the qa_value>0.5 composite gate that are directly observable;
it omits only the reduced-chi-squared component, which has no corresponding
raw field in this file version.

Method (mirrors the production EC-footprint-weighting algorithm)
------------------------------------------------------------------
The production TROPOSIF EC-weighted SIF for a site-month is built by:
  Codes\\pythonProject\\Prepare_SIF_EC_Reconstruction_Model_Inventories.py
    -> compute_target_product_coverage() / coverage_record_from_sif_array()
  which calls into
  Codes\\pythonProject\\Create_FINAL_CLEAN_SIF_Companion_TIFFs_FOOTPRINT_AWARE.py
    -> build_footprint_context() / append_footprint_polygons_from_source()
       / context_sif_array_and_metrics()
using the EC-footprint-probability GPP TIFF (band 1) as the fixed target grid
(local Azimuthal-Equidistant projection, ~30 m cells) for each site-month:
  1. Each TROPOSIF sounding's 4-corner footprint polygon (from
     PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds / longitude_bounds) is
     reprojected onto the site's EC grid and rasterized (all_touched=True).
  2. Grid cells touched by more than one sounding's footprint in that month
     receive the MEAN of the overlapping soundings' SIF values
     (SWATH_RASTERIZATION_METHOD = "footprint_polygon_to_GPP_grid_cells",
     MULTIPLE_SIF_FOOTPRINTS_PER_CELL = "mean" in the production script).
  3. SIF_weighted = sum(cell_SIF_i * EC_probability_i) / sum(EC_probability_i)
     over grid cells with EC_probability > 0 AND a valid rasterized SIF value.

This script REUSES the exact geometry helper functions from the production
module (make_polygon_from_vertices, bounds_rows_to_four_vertices,
normalize_longitudes) via a read-only `import`, and reimplements the
rest directly, restricted to only the 1,540 TROPOSIF site-months that have
complete EC-footprint probability support (coverage_probability_fraction==1
-- i.e. exactly TROPOSIF's *primary* production population; see
SIF_EC_TROPOSIF_Finalization\\TROPOSIF_Finalization_Report.md). No existing
script, TIFF, or CSV is modified anywhere in this process -- only NEW files
are written into this output directory.

Outputs (all NEW files in this directory):
  troposif_ec_weighted_raw_recompute.csv   -- per-site-month recomputed values
  troposif_qa_sensitivity_unfiltered.csv   -- raw/current-anom/harmonic-anom r
  troposif_qa_sensitivity_qa_filtered.csv  -- same, QA-filtered population
  troposif_qa_sensitivity_summary.json     -- headline numbers + sanity check
"""
from __future__ import annotations
# Original script filename (sanitized): troposif_ec_weighted_qa_sensitivity.py; public name: troposif_ec_weighted_qa_sensitivity.py
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
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS, Transformer
from rasterio.features import rasterize
from rasterio.warp import transform_bounds
from scipy import stats
from shapely.ops import transform as shapely_transform

PROJECT_ROOT = Path(
    PROJECT_ROOT
)
CODE_ROOT = PROJECT_ROOT / "Codes" / "pythonProject"
sys.path.insert(0, str(CODE_ROOT))

# Read-only reuse of the production geometry helpers (no production file is
# modified by importing it; this only reads function definitions).
import Create_FINAL_CLEAN_SIF_Companion_TIFFs_FOOTPRINT_AWARE as footprint_pipeline  # noqa: E402

TROPOSIF_MASTER_CSV = PROJECT_ROOT / "SIF_EC_TROPOSIF_Finalization" / "TROPOSIF_Final_EC_Footprint_SIF_All_SiteMonths.csv"
ANALYSIS_READY_MASTER = (
    PROJECT_ROOT / "SIF_EC_SIF_GPP_Relationship_Analysis" / "01_analysis_ready" / "SIF_GPP_Analysis_Ready_Master.csv"
)
RAW_TROPOSIF_ROOT = PROJECT_ROOT / "SIF Datasets" / "TROPOSIF"
OUT_DIR = PROJECT_ROOT / "SIF_EC_SIF_GPP_Final_Verification" / "04_sampling_and_QA"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SIF_VAR = "PRODUCT/SIF_Corr_743"
LATB_VAR = "PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds"
LONB_VAR = "PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds"
CLOUD_VAR = "PRODUCT/SUPPORT_DATA/INPUT_DATA/cloud_fraction_L2"
VZA_VAR = "PRODUCT/SUPPORT_DATA/GEOLOCATIONS/viewing_zenith_angle"
SZA_VAR = "PRODUCT/SUPPORT_DATA/GEOLOCATIONS/solar_zenith_angle"

QA_CLOUD_MAX = 0.2
QA_VZA_MAX = 60.0
QA_SZA_MAX = 70.0

WGS84 = CRS.from_epsg(4326)


# =============================================================================
# STEP 1: target site-months (complete EC-support TROPOSIF primary population)
# =============================================================================


def load_targets() -> pd.DataFrame:
    master = pd.read_csv(TROPOSIF_MASTER_CSV, dtype={"YYYYMM": str})
    master["YYYYMM"] = master["YYYYMM"].astype(int)
    master["coverage_probability_fraction"] = pd.to_numeric(master["coverage_probability_fraction"], errors="coerce")
    prim = master[np.isclose(master["coverage_probability_fraction"], 1.0, atol=1e-9)].copy()
    keep = ["SITE_ID", "YYYYMM", "matching_gpp_tif", "TROPOSIF_final_EC_SIF", "n_target_source_observations"]
    prim = prim[keep].reset_index(drop=True)
    return prim


def month_folder_name(yyyymm: int) -> str:
    s = str(int(yyyymm))
    return f"{s[:4]}.{s[4:6]}"


# =============================================================================
# STEP 2: per-site EC grid (read-only from the existing GPP TIFF template)
# =============================================================================


def build_site_grid(site_id: str, yyyymm: int, gpp_tif: str, production_value: float, n_native_reported: float) -> dict:
    with rasterio.open(gpp_tif) as ds:
        band1 = ds.read(1).astype(np.float64)
        crs = ds.crs
        transform = ds.transform
        h, w = ds.height, ds.width
        bounds = ds.bounds
    geo_bounds = transform_bounds(crs, WGS84, *bounds, densify_pts=21)
    transformer = Transformer.from_crs(WGS84, crs, always_xy=True)
    ec_mask = np.isfinite(band1) & (band1 > 0)
    return dict(
        SITE_ID=site_id,
        YYYYMM=yyyymm,
        band1=band1,
        transform=transform,
        shape=(h, w),
        geo_bounds=geo_bounds,  # (left, bottom, right, top) in WGS84 lon/lat
        transformer=transformer,
        ec_mask=ec_mask,
        sum_unf=np.zeros((h, w), dtype=np.float64),
        count_unf=np.zeros((h, w), dtype=np.float64),
        sum_qa=np.zeros((h, w), dtype=np.float64),
        count_qa=np.zeros((h, w), dtype=np.float64),
        n_used_unf=0,
        n_used_qa=0,
        n_candidates_bbox=0,
        production_value=production_value,
        n_native_reported=n_native_reported,
    )


# =============================================================================
# STEP 3: process one raw daily TROPOSIF file against all active-month sites
# =============================================================================


def process_file(path: Path, site_grids: list[dict]) -> None:
    with h5py.File(str(path), "r") as h:
        sif = np.asarray(h[SIF_VAR][...], dtype=np.float64)
        latb = np.asarray(h[LATB_VAR][...], dtype=np.float64)
        lonb = np.asarray(h[LONB_VAR][...], dtype=np.float64)
        cf = np.asarray(h[CLOUD_VAR][...], dtype=np.float64)
        vza = np.asarray(h[VZA_VAR][...], dtype=np.float64)
        sza = np.asarray(h[SZA_VAR][...], dtype=np.float64)

    lonb = footprint_pipeline.normalize_longitudes(lonb)
    valid_sif = np.isfinite(sif)  # negatives are intentionally NOT removed
    finite_vert = np.isfinite(latb).all(axis=1) & np.isfinite(lonb).all(axis=1)
    base_candidates = valid_sif & finite_vert
    if not base_candidates.any():
        return

    lon_min = np.min(lonb, axis=1)
    lon_max = np.max(lonb, axis=1)
    lat_min = np.min(latb, axis=1)
    lat_max = np.max(latb, axis=1)

    qa_mask = (
        np.isfinite(cf) & np.isfinite(vza) & np.isfinite(sza)
        & (cf < QA_CLOUD_MAX) & (vza < QA_VZA_MAX) & (sza < QA_SZA_MAX)
    )

    for site in site_grids:
        glon_min, glat_min, glon_max, glat_max = site["geo_bounds"]
        bbox_ok = (lon_max >= glon_min) & (lon_min <= glon_max) & (lat_max >= glat_min) & (lat_min <= glat_max)
        cand_idx = np.flatnonzero(base_candidates & bbox_ok)
        if cand_idx.size == 0:
            continue
        site["n_candidates_bbox"] += int(cand_idx.size)
        for idx in cand_idx:
            polygon = footprint_pipeline.make_polygon_from_vertices(lonb[idx], latb[idx])
            if polygon is None:
                continue
            try:
                proj = shapely_transform(site["transformer"].transform, polygon)
                if proj.is_empty:
                    continue
                if not proj.is_valid:
                    proj = proj.buffer(0)
                if proj.is_empty:
                    continue
                mask = rasterize(
                    [(proj, 1)],
                    out_shape=site["shape"],
                    transform=site["transform"],
                    fill=0,
                    all_touched=True,
                    dtype="uint8",
                ).astype(bool)
            except Exception:
                continue
            if not mask.any():
                continue
            value = float(sif[idx])
            site["sum_unf"][mask] += value
            site["count_unf"][mask] += 1.0
            site["n_used_unf"] += 1
            if qa_mask[idx]:
                site["sum_qa"][mask] += value
                site["count_qa"][mask] += 1.0
                site["n_used_qa"] += 1


# =============================================================================
# STEP 4: finalize EC-probability-weighted mean per site-month
# =============================================================================


def weighted_from_accumulators(band1: np.ndarray, ec_mask: np.ndarray, sum_arr: np.ndarray, count_arr: np.ndarray):
    valid = count_arr > 0
    observed_ec = valid & ec_mask
    if not observed_ec.any():
        return np.nan, np.nan, 0
    pixel_mean = np.full(band1.shape, np.nan, dtype=np.float64)
    pixel_mean[valid] = sum_arr[valid] / count_arr[valid]
    w = band1[observed_ec]
    wsum = float(w.sum())
    full_prob = float(band1[ec_mask].sum())
    if wsum <= 0 or full_prob <= 0:
        return np.nan, np.nan, int(observed_ec.sum())
    val = float(np.sum(pixel_mean[observed_ec] * w) / wsum)
    cov = wsum / full_prob
    return val, cov, int(observed_ec.sum())


def finalize_site(site: dict) -> dict:
    unf_val, unf_cov, unf_npix = weighted_from_accumulators(site["band1"], site["ec_mask"], site["sum_unf"], site["count_unf"])
    qa_val, qa_cov, qa_npix = weighted_from_accumulators(site["band1"], site["ec_mask"], site["sum_qa"], site["count_qa"])
    return dict(
        SITE_ID=site["SITE_ID"],
        YYYYMM=site["YYYYMM"],
        SIF_unfiltered_recomputed=unf_val,
        coverage_recomputed_unfiltered=unf_cov,
        n_ec_pixels_observed_unfiltered=unf_npix,
        n_soundings_used_unfiltered=site["n_used_unf"],
        SIF_qa_filtered_recomputed=qa_val,
        coverage_recomputed_qa_filtered=qa_cov,
        n_ec_pixels_observed_qa_filtered=qa_npix,
        n_soundings_used_qa_filtered=site["n_used_qa"],
        n_candidates_bbox_prefilter=site["n_candidates_bbox"],
        production_TROPOSIF_final_EC_SIF=site["production_value"],
        n_native_reported_production=site["n_native_reported"],
    )


# =============================================================================
# STEP 5: statistics (mirrors gome2_quality_flag_sensitivity.py conventions)
# =============================================================================


def pearson_spearman_r2(x: np.ndarray, y: np.ndarray) -> dict:
    n = len(x)
    if n < 3 or np.std(x) == 0 or np.std(y) == 0:
        return dict(N=n, pearson_r=np.nan, spearman_rho=np.nan, r2=np.nan)
    r, _ = stats.pearsonr(x, y)
    rho, _ = stats.spearmanr(x, y)
    slope, intercept, rval, _, _ = stats.linregress(x, y)
    return dict(N=n, pearson_r=r, spearman_rho=rho, r2=rval ** 2)


def harmonic_r(g: pd.DataFrame, sif_col: str, min_n: int = 24, min_unique_months: int = 8) -> dict:
    recs = []
    for site, gs in g.groupby("SITE_ID"):
        if len(gs) < min_n or gs["MONTH"].nunique() < min_unique_months:
            continue
        t = 2 * np.pi * gs["MONTH"].to_numpy() / 12.0
        X = np.column_stack([np.ones(len(gs)), np.sin(t), np.cos(t), np.sin(2 * t), np.cos(2 * t)])
        gs = gs.copy()
        for col, out_col in ((sif_col, "SIF_ANOM"), ("GPP_VALUE", "GPP_ANOM")):
            y = gs[col].to_numpy(dtype=float)
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            gs[out_col] = y - X @ beta
        recs.append(gs)
    if not recs:
        return dict(N=0, N_towers=0, pearson_r=np.nan)
    gr = pd.concat(recs, ignore_index=True)
    s = pearson_spearman_r2(gr["SIF_ANOM"].to_numpy(), gr["GPP_ANOM"].to_numpy())
    return {"N_towers": gr["SITE_ID"].nunique(), **s}


def current_anomaly_r(g: pd.DataFrame, sif_col: str) -> dict:
    clim = g.groupby(["SITE_ID", "MONTH"]).agg(sif_clim=(sif_col, "mean"), gpp_clim=("GPP_VALUE", "mean"))
    gg = g.join(clim, on=["SITE_ID", "MONTH"])
    sif_anom = gg[sif_col] - gg["sif_clim"]
    gpp_anom = gg["GPP_VALUE"] - gg["gpp_clim"]
    return pearson_spearman_r2(sif_anom.to_numpy(), gpp_anom.to_numpy())


def run_stats_block(m: pd.DataFrame, sif_col: str, label: str) -> pd.DataFrame:
    raw = pearson_spearman_r2(m[sif_col].to_numpy(), m["GPP_VALUE"].to_numpy())
    cur = current_anomaly_r(m, sif_col)
    harm = harmonic_r(m, sif_col)
    print(f"\n=== {label} ===")
    print("  matched N site-months:", len(m), "| N towers:", m["SITE_ID"].nunique())
    print("  RAW           :", raw)
    print("  CURRENT ANOM  :", cur)
    print("  HARMONIC ANOM :", harm)
    return pd.DataFrame(
        [
            {"filter": label, "stage": "raw", "N_site_months": len(m), "N_towers": m["SITE_ID"].nunique(), **raw},
            {"filter": label, "stage": "current_anomaly", **cur},
            {"filter": label, "stage": "harmonic_anomaly", **harm},
        ]
    )


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    t0 = time.time()
    targets = load_targets()
    print(f"Target TROPOSIF primary (coverage=1) site-months: {len(targets)} across {targets['SITE_ID'].nunique()} towers, {targets['YYYYMM'].nunique()} unique months.")

    all_results: list[dict] = []
    for yyyymm, grp in targets.groupby("YYYYMM"):
        t_month = time.time()
        site_grids = [
            build_site_grid(r.SITE_ID, int(r.YYYYMM), r.matching_gpp_tif, r.TROPOSIF_final_EC_SIF, r.n_target_source_observations)
            for r in grp.itertuples()
        ]
        folder = RAW_TROPOSIF_ROOT / month_folder_name(yyyymm)
        files = sorted(folder.glob("*.nc"))
        for f in files:
            process_file(f, site_grids)
        for site in site_grids:
            all_results.append(finalize_site(site))
        print(
            f"[{time.strftime('%H:%M:%S')}] YYYYMM={yyyymm}: {len(site_grids)} site(s), {len(files)} raw file(s), "
            f"{time.time() - t_month:.1f}s"
        )

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUT_DIR / "troposif_ec_weighted_raw_recompute.csv", index=False)
    print(f"\nWrote raw recompute table: {OUT_DIR / 'troposif_ec_weighted_raw_recompute.csv'} ({len(results_df)} rows)")

    # -------------------------------------------------------------------
    # Sanity check: unfiltered recompute vs actual production SIF_VALUE
    # -------------------------------------------------------------------
    check = results_df.dropna(subset=["SIF_unfiltered_recomputed", "production_TROPOSIF_final_EC_SIF"]).copy()
    diff = (check["SIF_unfiltered_recomputed"] - check["production_TROPOSIF_final_EC_SIF"]).abs()
    corr = np.nan
    if len(check) >= 3 and check["SIF_unfiltered_recomputed"].std() > 0 and check["production_TROPOSIF_final_EC_SIF"].std() > 0:
        corr, _ = stats.pearsonr(check["SIF_unfiltered_recomputed"], check["production_TROPOSIF_final_EC_SIF"])
    sanity = dict(
        N_compared=int(len(check)),
        N_total_target_site_months=int(len(results_df)),
        median_abs_diff=float(diff.median()) if len(diff) else np.nan,
        mean_abs_diff=float(diff.mean()) if len(diff) else np.nan,
        max_abs_diff=float(diff.max()) if len(diff) else np.nan,
        pearson_r_vs_production=float(corr) if np.isfinite(corr) else None,
        production_sif_sd=float(check["production_TROPOSIF_final_EC_SIF"].std()),
    )
    print("\nSanity check (independent recompute, unfiltered, vs production TROPOSIF_final_EC_SIF):")
    print(" ", sanity)

    # -------------------------------------------------------------------
    # Merge with GPP and compute raw / current-anomaly / harmonic-anomaly r
    # -------------------------------------------------------------------
    analysis = pd.read_csv(ANALYSIS_READY_MASTER, low_memory=False)
    analysis = analysis[analysis["SIF_PRODUCT"].astype(str).str.upper() == "TROPOSIF"].copy()
    analysis["YYYYMM"] = pd.to_numeric(analysis["YYYYMM"], errors="coerce").astype("Int64")
    analysis["MONTH"] = pd.to_numeric(analysis["MONTH"], errors="coerce")
    analysis["GPP_VALUE"] = pd.to_numeric(analysis["GPP_VALUE"], errors="coerce")
    gpp_lookup = analysis[["SITE_ID", "YYYYMM", "MONTH", "GPP_VALUE"]].drop_duplicates(subset=["SITE_ID", "YYYYMM"])

    results_df["YYYYMM"] = results_df["YYYYMM"].astype(int)
    merged = results_df.merge(gpp_lookup, on=["SITE_ID", "YYYYMM"], how="left")

    unf = merged.dropna(subset=["SIF_unfiltered_recomputed", "GPP_VALUE", "MONTH"]).copy()
    qaf = merged.dropna(subset=["SIF_qa_filtered_recomputed", "GPP_VALUE", "MONTH"]).copy()

    unf_stats = run_stats_block(unf, "SIF_unfiltered_recomputed", "TROPOSIF_UNFILTERED_recomputed")
    qa_stats = run_stats_block(qaf, "SIF_qa_filtered_recomputed", "TROPOSIF_QA_FILTERED_cloud0.2_vza60_sza70")

    unf_stats.to_csv(OUT_DIR / "troposif_qa_sensitivity_unfiltered.csv", index=False)
    qa_stats.to_csv(OUT_DIR / "troposif_qa_sensitivity_qa_filtered.csv", index=False)

    # QA filter impact on soundings / coverage
    n_dropped_soundings = int(results_df["n_soundings_used_unfiltered"].sum() - results_df["n_soundings_used_qa_filtered"].sum())
    n_sitemonths_no_longer_complete = int(
        (results_df["coverage_recomputed_qa_filtered"].fillna(0) < 0.999).sum()
        - (results_df["coverage_recomputed_unfiltered"].fillna(0) < 0.999).sum()
    )

    summary = dict(
        run_timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        elapsed_seconds=round(time.time() - t0, 1),
        n_target_site_months=int(len(targets)),
        n_towers=int(targets["SITE_ID"].nunique()),
        n_unique_months=int(targets["YYYYMM"].nunique()),
        sanity_check_vs_production=sanity,
        n_soundings_total_unfiltered=int(results_df["n_soundings_used_unfiltered"].sum()),
        n_soundings_total_qa_filtered=int(results_df["n_soundings_used_qa_filtered"].sum()),
        n_soundings_dropped_by_qa_filter=n_dropped_soundings,
        n_sitemonths_dropping_below_near_complete_coverage_under_qa=n_sitemonths_no_longer_complete,
        qa_gate_used="cloud_fraction_L2 < 0.2 AND viewing_zenith_angle < 60 AND solar_zenith_angle < 70 (approximation of official qa_value>0.5; no literal qa_value/reduced-chi-squared field exists in these raw files)",
    )
    (OUT_DIR / "troposif_qa_sensitivity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSummary written:", OUT_DIR / "troposif_qa_sensitivity_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
