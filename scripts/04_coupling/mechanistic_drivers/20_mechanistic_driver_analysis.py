"""
Mechanistic driver analysis for footprint-aware SIF-GPP coupling.

This script is an add-on to the completed SIF-GPP coupling workflow. It does
not rebuild SIF extraction, GPP matching, or the site-level coupling estimates.
It reads the audited monthly anomaly table and existing site-level
deseasonalized relationships, adds tower-footprint environmental driver
summaries, and writes the requested mechanistic analysis products.
"""
from __future__ import annotations
# Original script filename (sanitized): 20_mechanistic_driver_analysis.py; public name: 20_mechanistic_driver_analysis.py
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
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import rasterio
from rasterio.enums import Resampling

import statsmodels.formula.api as smf

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import xgboost as xgb
from xgboost import XGBRegressor


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


ROOT = Path(
    PROJECT_ROOT
)
PAPER_ROOT = ROOT / "SIF_EC_Mechanistic_FootprintAware_SIF_GPP_Paper"
OUT_DIR = PAPER_ROOT / "08_mechanistic_controls"
WORKING_MONTHLY = PAPER_ROOT / "01_data" / "Mechanistic_SIF_GPP_Master_Working_Table.parquet"
SITE_REL = PAPER_ROOT / "02_tables" / "Site_Level_Deseasonalized_Relationships.csv"
F1 = (
    ROOT
    / "Remaining Fluxnet Sites"
    / "HH files"
    / "monthly_GEE_csv_outputs_from_HH_HR"
    / "new_sites_fluxnet_monthly_with_metadata_all_sites.csv"
)
F2 = ROOT / "fluxnet" / "Monthly Datasets" / "fluxnet_monthly_with_metadata_all_sites.csv"
TERRAIN_TIF = ROOT / "world.topo.bathy.200408.3x21600x10800_geo.tif"


PRODUCT_ORDER = ["GOSIF", "GOME-2", "TROPOSIF", "OCO-2", "OCO-3"]
PRODUCT_COLORS = {
    "GOSIF": "#123B73",      # deep blue
    "GOME-2": "#188C89",     # teal
    "TROPOSIF": "#C99A19",   # gold/ochre
    "OCO-2": "#D7652A",      # orange
    "OCO-3": "#7A3B8F",      # purple
}

DRIVERS = {
    "VPD": {
        "value": "VPD",
        "anom": "VPD_ANOM",
        "master": "vpd_anomaly",
        "mean": "vpd_mean",
        "label": "VPD",
        "long": "Vapor pressure deficit anomaly",
    },
    "TAIR": {
        "value": "TAIR",
        "anom": "TAIR_ANOM",
        "master": "air_temperature_anomaly",
        "mean": "air_temperature_mean",
        "label": "Temperature",
        "long": "Air temperature anomaly",
    },
    "SW_IN": {
        "value": "SW_IN",
        "anom": "SW_IN_ANOM",
        "master": "shortwave_radiation_anomaly",
        "mean": "shortwave_radiation_mean",
        "label": "Radiation",
        "long": "Shortwave radiation anomaly",
    },
    "SWC": {
        "value": "SWC",
        "anom": "SWC_ANOM",
        "master": "soil_moisture_anomaly",
        "mean": "soil_moisture_mean",
        "label": "Soil moisture",
        "long": "Soil water content anomaly",
    },
    "ET": {
        "value": "ET",
        "anom": "ET_ANOM",
        "master": "evapotranspiration_anomaly",
        "mean": "evapotranspiration_mean",
        "label": "ET",
        "long": "Evapotranspiration anomaly",
    },
}

IGBP_GROUPS = {
    "ENF": "Forest",
    "EBF": "Forest",
    "DBF": "Forest",
    "DNF": "Forest",
    "MF": "Forest",
    "CSH": "Shrubland",
    "OSH": "Shrubland",
    "WSA": "Savanna/Woodland",
    "SAV": "Savanna/Woodland",
    "GRA": "Grassland",
    "CRO": "Cropland",
    "WET": "Wetland",
}

IGBP_ORDER = [
    "ENF",
    "EBF",
    "DBF",
    "MF",
    "CSH",
    "OSH",
    "WSA",
    "SAV",
    "GRA",
    "CRO",
    "WET",
    "Unknown",
]

COUPLING_CMAP = LinearSegmentedColormap.from_list(
    "coupling_blue_gray_red", ["#2B5AA8", "#F2F2F2", "#8C1D18"], N=256
)
EFFECT_CMAP = LinearSegmentedColormap.from_list(
    "effect_matrix", ["#355C9A", "#F4F1EA", "#9A3E2F"], N=256
)


def ensure_outdir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_mean(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    return float(values.mean()) if values.notna().any() else np.nan


def safe_std(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    return float(values.std(ddof=1)) if values.notna().sum() > 1 else np.nan


def safe_abs_mean(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    return float(values.abs().mean()) if values.notna().any() else np.nan


def pearson_r(x: pd.Series, y: pd.Series, min_n: int = 3) -> tuple[float, float, int]:
    d = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(d)
    if n < min_n or d["x"].std(ddof=0) < 1e-12 or d["y"].std(ddof=0) < 1e-12:
        return np.nan, np.nan, n
    r, p = stats.pearsonr(d["x"], d["y"])
    return float(r), float(p), n


def fisher_weighted_mean_r(r: pd.Series, n: pd.Series | None = None) -> float:
    r = pd.to_numeric(r, errors="coerce").clip(-0.999999, 0.999999)
    mask = r.notna()
    if not mask.any():
        return np.nan
    z = np.arctanh(r[mask].to_numpy())
    if n is None:
        weights = np.ones_like(z)
    else:
        weights = pd.to_numeric(n, errors="coerce")[mask].fillna(3).to_numpy()
        weights = np.maximum(weights - 3.0, 1.0)
    return float(np.tanh(np.average(z, weights=weights)))


def bootstrap_mean_ci(
    df: pd.DataFrame,
    value_col: str,
    cluster_col: str = "tower_id",
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    d = df[[cluster_col, value_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if d.empty:
        return np.nan, np.nan, np.nan
    mean_obs = float(d[value_col].mean())
    clusters = d[cluster_col].unique()
    if len(clusters) < 2:
        return mean_obs, np.nan, np.nan
    rng = np.random.default_rng(seed)
    by_cluster = {c: g[value_col].to_numpy(float) for c, g in d.groupby(cluster_col)}
    boot = []
    for _ in range(n_boot):
        picks = rng.choice(clusters, size=len(clusters), replace=True)
        vals = np.concatenate([by_cluster[c] for c in picks])
        if len(vals):
            boot.append(float(np.mean(vals)))
    if not boot:
        return mean_obs, np.nan, np.nan
    return mean_obs, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def p_from_bootstrap(boot: np.ndarray) -> float:
    boot = boot[np.isfinite(boot)]
    if len(boot) == 0:
        return np.nan
    p = 2.0 * min(np.mean(boot <= 0), np.mean(boot >= 0))
    return float(max(p, 1.0 / len(boot)))


def load_driver_file(path: Path, source: str, priority: int) -> pd.DataFrame:
    wanted = {
        "site_id",
        "YYYYMM",
        "TIMESTAMP",
        "vpd",
        "air_temperature",
        "shortwave_in",
        "soil_water_content_1",
        "latent_heat",
    }
    df = pd.read_csv(path, usecols=lambda c: c in wanted, low_memory=False)
    df = df.rename(
        columns={
            "site_id": "SITE_ID",
            "vpd": "VPD",
            "air_temperature": "TAIR",
            "shortwave_in": "SW_IN",
            "soil_water_content_1": "SWC",
            "latent_heat": "LE",
        }
    )
    if "YYYYMM" not in df.columns:
        df["YYYYMM"] = df["TIMESTAMP"]
    df["YYYYMM"] = pd.to_numeric(df["YYYYMM"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["SITE_ID", "YYYYMM"]).copy()
    df["YYYYMM"] = df["YYYYMM"].astype(int)
    df["MONTH"] = (df["YYYYMM"] % 100).astype(int)
    df["DRIVER_SOURCE_TABLE"] = source
    df["DRIVER_SOURCE_PRIORITY"] = priority

    # Monthly mean ET-equivalent from latent heat flux.
    # 1 W m-2 = 86400 J day-1 m-2; latent heat of vaporization approx 2.45 MJ kg-1.
    # kg m-2 day-1 is numerically mm day-1.
    df["ET"] = pd.to_numeric(df["LE"], errors="coerce") * 86400.0 / 2.45e6
    df["ET_NEGATIVE_FLAG"] = df["ET"] < 0
    return df[
        [
            "SITE_ID",
            "YYYYMM",
            "MONTH",
            "VPD",
            "TAIR",
            "SW_IN",
            "SWC",
            "LE",
            "ET",
            "ET_NEGATIVE_FLAG",
            "DRIVER_SOURCE_TABLE",
            "DRIVER_SOURCE_PRIORITY",
        ]
    ].copy()


def build_extended_drivers(master_sites: np.ndarray) -> pd.DataFrame:
    d1 = load_driver_file(F1, "Remaining Fluxnet Sites HH/HR monthly tower table", 1)
    d2 = load_driver_file(F2, "FLUXNET monthly tower table", 2)
    drivers = pd.concat([d1, d2], ignore_index=True)
    drivers = drivers.sort_values(["SITE_ID", "YYYYMM", "DRIVER_SOURCE_PRIORITY"])
    drivers = drivers.drop_duplicates(["SITE_ID", "YYYYMM"], keep="first")
    drivers = drivers[drivers["SITE_ID"].isin(master_sites)].copy()

    for key, cfg in DRIVERS.items():
        value = cfg["value"]
        clim = drivers.groupby(["SITE_ID", "MONTH"])[value].transform("mean")
        drivers[f"{value}_CLIM"] = clim
        drivers[cfg["anom"]] = drivers[value] - clim
        drivers[f"{value}_AVAILABLE"] = drivers[value].notna()

    drivers.to_csv(OUT_DIR / "Environmental_Drivers_SiteMonth_Extended.csv", index=False)
    return drivers


def land_cover_group(igbp: object) -> str:
    if pd.isna(igbp):
        return "Unknown"
    return IGBP_GROUPS.get(str(igbp), "Other")


def build_monthly_with_drivers() -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly_base = pd.read_parquet(WORKING_MONTHLY)
    master_sites = monthly_base["SITE_ID"].dropna().unique()
    drivers = build_extended_drivers(master_sites)
    monthly = monthly_base.merge(drivers, on=["SITE_ID", "YYYYMM"], how="left", validate="many_to_one")
    monthly.to_parquet(OUT_DIR / "Mechanistic_SIF_GPP_Monthly_With_Extended_Drivers.parquet", index=False)
    monthly.to_csv(OUT_DIR / "Mechanistic_SIF_GPP_Monthly_With_Extended_Drivers.csv", index=False)
    return monthly, drivers


def site_product_driver_rows(monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (product, site), g in monthly.groupby(["SIF_PRODUCT", "SITE_ID"], sort=False):
        row = {
            "SIF_PRODUCT": product,
            "SITE_ID": site,
            "matched_months_min": int(g["YYYYMM"].min()) if g["YYYYMM"].notna().any() else np.nan,
            "matched_months_max": int(g["YYYYMM"].max()) if g["YYYYMM"].notna().any() else np.nan,
            "driver_matched_site_months": int(len(g)),
            "mean_ec_coverage_fraction": safe_mean(g["EC_COVERAGE_FRACTION"])
            if "EC_COVERAGE_FRACTION" in g
            else np.nan,
            "mean_native_observations": safe_mean(g["N_NATIVE_OBSERVATIONS"])
            if "N_NATIVE_OBSERVATIONS" in g
            else np.nan,
        }
        for key, cfg in DRIVERS.items():
            anom = cfg["anom"]
            value = cfg["value"]
            master_col = cfg["master"]
            mean_col = cfg["mean"]
            row[master_col] = safe_mean(g[anom])
            row[f"{master_col}_sd"] = safe_std(g[anom])
            row[f"{master_col}_abs_mean"] = safe_abs_mean(g[anom])
            row[mean_col] = safe_mean(g[value])
            row[f"{master_col}_coverage_fraction"] = float(g[anom].notna().mean())
        rows.append(row)
    return pd.DataFrame(rows)


def create_master(monthly: pd.DataFrame) -> pd.DataFrame:
    site_rel = pd.read_csv(SITE_REL, low_memory=False)
    site_rel = site_rel[
        (site_rel["TRANSFORM"] == "ANOM_ALL")
        & site_rel["PEARSON_R"].notna()
        & np.isfinite(site_rel["PEARSON_R"])
    ].copy()

    cov = site_product_driver_rows(monthly)
    master = site_rel.merge(cov, on=["SIF_PRODUCT", "SITE_ID"], how="left", validate="one_to_one")
    master["land_cover_group"] = master["IGBP"].map(land_cover_group)
    master["lai_or_vegetation_index_available"] = False
    master["lai_or_vegetation_index"] = np.nan
    master["lai_or_vegetation_index_source"] = "No LAI/NDVI/EVI/fPAR column found in the existing tower-month inputs."
    master["climate_source_dataset"] = (
        "FLUXNET2015/project monthly tower meteorology; ET derived from tower latent heat."
    )
    master["climate_spatial_representation"] = (
        "Tower-footprint/in-situ EC tower conditions; no surrounding-pixel climate extraction."
    )
    master["climate_temporal_resolution"] = "Monthly"
    master["climate_aggregation_method"] = (
        "Mean over months matched to each tower-product SIF-GPP record; anomalies are relative "
        "to each tower and calendar-month climatology."
    )

    rename = {
        "SITE_ID": "tower_id",
        "SIF_PRODUCT": "product",
        "LATITUDE": "latitude",
        "LONGITUDE": "longitude",
        "PEARSON_R": "deseasonalized_pearson_r",
        "R2": "site_level_r2",
        "N": "n_site_month_observations",
        "IGBP": "igbp_class",
        "PEARSON_P": "deseasonalized_pearson_p",
        "PEARSON_R_CI_LOW": "deseasonalized_pearson_r_ci_low",
        "PEARSON_R_CI_HIGH": "deseasonalized_pearson_r_ci_high",
        "R2_CI_LOW": "site_level_r2_ci_low",
        "R2_CI_HIGH": "site_level_r2_ci_high",
        "OLS_SLOPE": "site_level_slope",
        "CI_EXCLUDES_ZERO": "coupling_ci_excludes_zero",
    }
    master = master.rename(columns=rename)

    preferred = [
        "tower_id",
        "product",
        "latitude",
        "longitude",
        "deseasonalized_pearson_r",
        "site_level_r2",
        "n_site_month_observations",
        "deseasonalized_pearson_p",
        "deseasonalized_pearson_r_ci_low",
        "deseasonalized_pearson_r_ci_high",
        "site_level_r2_ci_low",
        "site_level_r2_ci_high",
        "site_level_slope",
        "coupling_ci_excludes_zero",
        "igbp_class",
        "land_cover_group",
        "vpd_anomaly",
        "air_temperature_anomaly",
        "shortwave_radiation_anomaly",
        "soil_moisture_anomaly",
        "evapotranspiration_anomaly",
        "vpd_mean",
        "air_temperature_mean",
        "shortwave_radiation_mean",
        "soil_moisture_mean",
        "evapotranspiration_mean",
        "vpd_anomaly_abs_mean",
        "air_temperature_anomaly_abs_mean",
        "shortwave_radiation_anomaly_abs_mean",
        "soil_moisture_anomaly_abs_mean",
        "evapotranspiration_anomaly_abs_mean",
        "vpd_anomaly_coverage_fraction",
        "air_temperature_anomaly_coverage_fraction",
        "shortwave_radiation_anomaly_coverage_fraction",
        "soil_moisture_anomaly_coverage_fraction",
        "evapotranspiration_anomaly_coverage_fraction",
        "lai_or_vegetation_index_available",
        "lai_or_vegetation_index",
        "lai_or_vegetation_index_source",
        "matched_months_min",
        "matched_months_max",
        "driver_matched_site_months",
        "mean_ec_coverage_fraction",
        "mean_native_observations",
        "climate_source_dataset",
        "climate_spatial_representation",
        "climate_temporal_resolution",
        "climate_aggregation_method",
    ]
    other = [c for c in master.columns if c not in preferred]
    master = master[[c for c in preferred if c in master.columns] + other]
    master.to_csv(OUT_DIR / "SIF_GPP_coupling_driver_master.csv", index=False)
    return master


def strat_site_rs(dbin: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for site, g in dbin.groupby("SITE_ID"):
        r, p, n = pearson_r(g["SIF_ANOM"], g["GPP_ANOM"], min_n=3)
        if np.isfinite(r):
            rows.append({"SITE_ID": site, "site_bin_r": r, "site_bin_p": p, "site_bin_n": n})
    return pd.DataFrame(rows)


def effect_bootstrap(low_rs: pd.DataFrame, high_rs: pd.DataFrame, seed: int = 42) -> tuple[float, float, float, float]:
    low = dict(zip(low_rs["SITE_ID"], low_rs["site_bin_r"]))
    high = dict(zip(high_rs["SITE_ID"], high_rs["site_bin_r"]))
    sites = np.array(sorted(set(low) | set(high)))
    if len(sites) < 2 or not low or not high:
        obs = np.nan
        if low and high:
            obs = float(np.mean(list(high.values())) - np.mean(list(low.values())))
        return obs, np.nan, np.nan, np.nan
    obs = float(np.mean(list(high.values())) - np.mean(list(low.values())))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(2000):
        picks = rng.choice(sites, size=len(sites), replace=True)
        lo_vals = [low[s] for s in picks if s in low]
        hi_vals = [high[s] for s in picks if s in high]
        if lo_vals and hi_vals:
            boot.append(float(np.mean(hi_vals) - np.mean(lo_vals)))
    boot = np.asarray(boot, dtype=float)
    if len(boot) == 0:
        return obs, np.nan, np.nan, np.nan
    return (
        obs,
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
        p_from_bootstrap(boot),
    )


def climate_stratification(monthly: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    valid_pairs = master[["tower_id", "product"]].drop_duplicates()
    m = monthly.merge(
        valid_pairs,
        left_on=["SITE_ID", "SIF_PRODUCT"],
        right_on=["tower_id", "product"],
        how="inner",
    )
    rows = []
    bin_labels = ["Low", "Medium", "High"]

    for product in PRODUCT_ORDER:
        mp = m[m["SIF_PRODUCT"] == product].copy()
        for key, cfg in DRIVERS.items():
            anom = cfg["anom"]
            d = mp.dropna(subset=["SIF_ANOM", "GPP_ANOM", anom]).copy()
            if len(d) < 30 or d["SITE_ID"].nunique() < 5:
                continue
            ranks = d[anom].rank(method="first")
            d["driver_bin"] = pd.qcut(ranks, 3, labels=bin_labels)
            site_rs_by_bin = {}
            for order, label in enumerate(bin_labels):
                dbin = d[d["driver_bin"] == label]
                site_rs = strat_site_rs(dbin)
                site_rs_by_bin[label] = site_rs
                tmp = site_rs.rename(columns={"SITE_ID": "tower_id"})
                mean_r, ci_low, ci_high = bootstrap_mean_ci(tmp, "site_bin_r", "tower_id")
                pooled_r, pooled_p, pooled_n = pearson_r(dbin["SIF_ANOM"], dbin["GPP_ANOM"], min_n=8)
                rows.append(
                    {
                        "analysis_type": "bin_mean",
                        "driver": cfg["label"],
                        "driver_anomaly_column": anom,
                        "product": product,
                        "bin": label,
                        "bin_order": order,
                        "bin_method": "Product-specific terciles of monthly tower driver anomaly",
                        "bin_driver_min": float(dbin[anom].min()),
                        "bin_driver_max": float(dbin[anom].max()),
                        "bin_driver_mean": float(dbin[anom].mean()),
                        "n_site_months": int(len(dbin)),
                        "n_towers": int(site_rs["SITE_ID"].nunique()) if not site_rs.empty else 0,
                        "mean_deseasonalized_pearson_r": mean_r,
                        "mean_r_ci_low": ci_low,
                        "mean_r_ci_high": ci_high,
                        "mean_r_se": float(site_rs["site_bin_r"].std(ddof=1) / math.sqrt(len(site_rs)))
                        if len(site_rs) > 1
                        else np.nan,
                        "pooled_pearson_r": pooled_r,
                        "pooled_pearson_p": pooled_p,
                        "pooled_n_site_months": pooled_n,
                        "effect_high_minus_low": np.nan,
                        "effect_ci_low": np.nan,
                        "effect_ci_high": np.nan,
                        "effect_p_value": np.nan,
                        "significant_05": np.nan,
                    }
                )

            obs, lo, hi, p = effect_bootstrap(site_rs_by_bin["Low"], site_rs_by_bin["High"])
            rows.append(
                {
                    "analysis_type": "high_minus_low_effect",
                    "driver": cfg["label"],
                    "driver_anomaly_column": anom,
                    "product": product,
                    "bin": "High-Low",
                    "bin_order": 99,
                    "bin_method": "Difference between High and Low anomaly tercile mean site-level coupling",
                    "bin_driver_min": np.nan,
                    "bin_driver_max": np.nan,
                    "bin_driver_mean": np.nan,
                    "n_site_months": int(len(d)),
                    "n_towers": int(d["SITE_ID"].nunique()),
                    "mean_deseasonalized_pearson_r": np.nan,
                    "mean_r_ci_low": np.nan,
                    "mean_r_ci_high": np.nan,
                    "mean_r_se": np.nan,
                    "pooled_pearson_r": np.nan,
                    "pooled_pearson_p": np.nan,
                    "pooled_n_site_months": int(len(d)),
                    "effect_high_minus_low": obs,
                    "effect_ci_low": lo,
                    "effect_ci_high": hi,
                    "effect_p_value": p,
                    "significant_05": bool(np.isfinite(p) and p < 0.05),
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "climate_stratification_results.csv", index=False)
    return out


def zscore_inplace(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    meta = {}
    for col in cols:
        mean = float(df[col].mean())
        sd = float(df[col].std(ddof=1))
        if not np.isfinite(sd) or sd < 1e-12:
            df[col + "_z"] = 0.0
            meta[col] = (mean, sd)
        else:
            df[col + "_z"] = (df[col] - mean) / sd
            meta[col] = (mean, sd)
    return df, meta


def mixed_model(master: pd.DataFrame) -> pd.DataFrame:
    driver_cols = [cfg["master"] for cfg in DRIVERS.values()]
    d = master[
        [
            "tower_id",
            "product",
            "land_cover_group",
            "deseasonalized_pearson_r",
            "n_site_month_observations",
        ]
        + driver_cols
    ].copy()
    d = d.replace([np.inf, -np.inf], np.nan)
    d["land_cover_group"] = d["land_cover_group"].fillna("Unknown")
    d = d.dropna(subset=["tower_id", "product", "land_cover_group", "deseasonalized_pearson_r"] + driver_cols)
    d["coupling_fisher_z"] = np.arctanh(d["deseasonalized_pearson_r"].clip(-0.999999, 0.999999))
    d, zmeta = zscore_inplace(d, driver_cols)

    formula = (
        "coupling_fisher_z ~ vpd_anomaly_z + air_temperature_anomaly_z + "
        "shortwave_radiation_anomaly_z + soil_moisture_anomaly_z + "
        "evapotranspiration_anomaly_z + C(land_cover_group) + C(product)"
    )
    n_groups = d["tower_id"].nunique()
    random_effect_appropriate = bool(n_groups >= 10 and len(d) / max(n_groups, 1) > 1.2)

    result_rows = []
    model_type = "MixedLM_random_intercept_tower" if random_effect_appropriate else "OLS_no_random_effect"
    status = "not_fit"
    res = None
    fit_exception = ""
    if random_effect_appropriate:
        for method in ["lbfgs", "powell", "cg"]:
            try:
                md = smf.mixedlm(formula, d, groups=d["tower_id"])
                res = md.fit(reml=False, method=method, maxiter=2000, disp=False)
                status = f"fit_ok_{method}" if getattr(res, "converged", True) else f"fit_not_converged_{method}"
                fit_exception = ""
                break
            except Exception as exc:
                fit_exception = str(exc)
                res = None
    if res is None:
        model_type = "OLS_cluster_robust_by_tower"
        try:
            res = smf.ols(formula, d).fit(cov_type="cluster", cov_kwds={"groups": d["tower_id"]})
            status = "fit_ok_fallback_ols"
        except Exception as exc:
            fit_exception = str(exc)
            res = None
            status = "fit_failed"

    if res is not None and model_type.startswith("MixedLM"):
        exog = res.model.exog
        fe_params = res.fe_params.to_numpy()
        fixed_pred = exog @ fe_params
        var_fixed = float(np.var(fixed_pred, ddof=1))
        var_random = float(res.cov_re.iloc[0, 0]) if getattr(res, "cov_re", pd.DataFrame()).size else 0.0
        var_resid = float(res.scale)
        denom = var_fixed + var_random + var_resid
        marginal_r2 = var_fixed / denom if denom > 0 else np.nan
        conditional_r2 = (var_fixed + var_random) / denom if denom > 0 else np.nan
        aic = float(res.aic) if hasattr(res, "aic") else np.nan
        bic = float(res.bic) if hasattr(res, "bic") else np.nan
    elif res is not None:
        marginal_r2 = float(getattr(res, "rsquared", np.nan))
        conditional_r2 = np.nan
        var_fixed = np.nan
        var_random = 0.0
        var_resid = float(np.var(res.resid, ddof=1))
        aic = float(res.aic)
        bic = float(res.bic)
    else:
        marginal_r2 = conditional_r2 = var_fixed = var_random = var_resid = aic = bic = np.nan

    if res is not None:
        ci = res.conf_int()
        for term in res.params.index:
            if term == "Group Var":
                continue
            lo, hi = ci.loc[term].tolist() if term in ci.index else (np.nan, np.nan)
            result_rows.append(
                {
                    "model_type": model_type,
                    "status": status,
                    "term": term,
                    "coefficient": float(res.params[term]),
                    "std_error": float(res.bse[term]) if term in res.bse.index else np.nan,
                    "ci_low": float(lo),
                    "ci_high": float(hi),
                    "p_value": float(res.pvalues[term]) if term in res.pvalues.index else np.nan,
                    "n_site_product_rows": int(len(d)),
                    "n_towers": int(n_groups),
                    "response": "Fisher z transform of site-level deseasonalized Pearson r",
                    "formula": formula,
                    "marginal_r2": marginal_r2,
                    "conditional_r2": conditional_r2,
                    "var_fixed": var_fixed,
                    "var_random_tower": var_random,
                    "var_residual": var_resid,
                    "aic": aic,
                    "bic": bic,
                    "fit_exception": fit_exception,
                }
            )
    else:
        result_rows.append(
            {
                "model_type": model_type,
                "status": status,
                "term": None,
                "coefficient": np.nan,
                "std_error": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "p_value": np.nan,
                "n_site_product_rows": int(len(d)),
                "n_towers": int(n_groups),
                "response": "Fisher z transform of site-level deseasonalized Pearson r",
                "formula": formula,
                "marginal_r2": np.nan,
                "conditional_r2": np.nan,
                "var_fixed": np.nan,
                "var_random_tower": np.nan,
                "var_residual": np.nan,
                "aic": np.nan,
                "bic": np.nan,
                "fit_exception": fit_exception,
            }
        )

    out = pd.DataFrame(result_rows)
    out.to_csv(OUT_DIR / "mixed_effects_model_results.csv", index=False)
    return out


def run_ml_shap(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = "deseasonalized_pearson_r"
    numeric = [
        "vpd_anomaly",
        "air_temperature_anomaly",
        "shortwave_radiation_anomaly",
        "soil_moisture_anomaly",
        "evapotranspiration_anomaly",
        "latitude",
    ]
    categorical = ["land_cover_group", "product"]
    d = master[["tower_id", target] + numeric + categorical].copy()
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=["tower_id", target])
    d[categorical] = d[categorical].fillna("Unknown")
    X = d[numeric + categorical]
    y = d[target].to_numpy(float)
    groups = d["tower_id"].to_numpy()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        verbose_feature_names_out=False,
    )
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=350,
        max_depth=2,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=1,
    )
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])

    n_splits = min(5, len(np.unique(groups)))
    cv_rows = []
    if n_splits >= 2:
        splitter = GroupKFold(n_splits=n_splits)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
            pipe.fit(X.iloc[train_idx], y[train_idx])
            pred = pipe.predict(X.iloc[test_idx])
            cv_rows.append(
                {
                    "fold": fold,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "r2": float(r2_score(y[test_idx], pred)),
                    "rmse": float(math.sqrt(mean_squared_error(y[test_idx], pred))),
                    "mae": float(mean_absolute_error(y[test_idx], pred)),
                }
            )
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(OUT_DIR / "ml_cross_validation_results.csv", index=False)
    cv_mean_r2 = float(cv["r2"].mean()) if not cv.empty else np.nan
    cv_sd_r2 = float(cv["r2"].std(ddof=1)) if len(cv) > 1 else np.nan
    cv_mean_rmse = float(cv["rmse"].mean()) if not cv.empty else np.nan
    cv_sd_rmse = float(cv["rmse"].std(ddof=1)) if len(cv) > 1 else np.nan
    cv_mean_mae = float(cv["mae"].mean()) if not cv.empty else np.nan

    pipe.fit(X, y)
    Xt = pipe.named_steps["preprocessor"].transform(X)
    feature_names = list(pipe.named_steps["preprocessor"].get_feature_names_out())
    booster = pipe.named_steps["model"].get_booster()
    contrib = booster.predict(xgb.DMatrix(Xt, feature_names=feature_names), pred_contribs=True)
    shap_values = contrib[:, :-1]

    def feature_group(name: str) -> str:
        mapping = {
            "vpd_anomaly": "VPD",
            "air_temperature_anomaly": "Temperature",
            "shortwave_radiation_anomaly": "Radiation",
            "soil_moisture_anomaly": "Soil moisture",
            "evapotranspiration_anomaly": "ET",
            "latitude": "Latitude",
        }
        if name in mapping:
            return mapping[name]
        if name.startswith("land_cover_group_"):
            return "Biome"
        if name.startswith("product_"):
            return "SIF product"
        return name

    encoded_rows = []
    for i, name in enumerate(feature_names):
        encoded_rows.append(
            {
                "row_type": "encoded_feature",
                "feature_group": feature_group(name),
                "encoded_feature": name,
                "mean_abs_shap": float(np.mean(np.abs(shap_values[:, i]))),
                "mean_shap": float(np.mean(shap_values[:, i])),
            }
        )
    encoded = pd.DataFrame(encoded_rows)
    encoded["normalized_importance"] = encoded["mean_abs_shap"] / encoded["mean_abs_shap"].sum()
    encoded["rank"] = encoded["mean_abs_shap"].rank(ascending=False, method="dense").astype(int)

    group_rows = []
    for group in encoded["feature_group"].unique():
        cols = [i for i, name in enumerate(feature_names) if feature_group(name) == group]
        grouped_contrib = shap_values[:, cols].sum(axis=1)
        group_rows.append(
            {
                "row_type": "feature_group",
                "feature_group": group,
                "encoded_feature": "ALL",
                "mean_abs_shap": float(np.mean(np.abs(grouped_contrib))),
                "mean_shap": float(np.mean(grouped_contrib)),
            }
        )
    grouped = pd.DataFrame(group_rows)
    grouped["normalized_importance"] = grouped["mean_abs_shap"] / grouped["mean_abs_shap"].sum()
    grouped["rank"] = grouped["mean_abs_shap"].rank(ascending=False, method="dense").astype(int)

    out = pd.concat([grouped, encoded], ignore_index=True)
    out["model"] = "XGBoostRegressor"
    out["target"] = "site-level deseasonalized Pearson r"
    out["n_site_product_rows"] = int(len(d))
    out["n_towers"] = int(len(np.unique(groups)))
    out["cv_grouping"] = "GroupKFold by tower_id"
    out["cv_mean_r2"] = cv_mean_r2
    out["cv_sd_r2"] = cv_sd_r2
    out["cv_mean_rmse"] = cv_mean_rmse
    out["cv_sd_rmse"] = cv_sd_rmse
    out["cv_mean_mae"] = cv_mean_mae
    out = out.sort_values(["row_type", "rank", "feature_group", "encoded_feature"])
    out.to_csv(OUT_DIR / "feature_importance_SHAP_results.csv", index=False)
    return out, cv


def read_terrain_background(max_width: int = 2400) -> tuple[np.ndarray | None, tuple[float, float, float, float] | None]:
    if not TERRAIN_TIF.exists():
        return None, None
    with rasterio.open(TERRAIN_TIF) as src:
        scale = max(src.width / max_width, 1.0)
        out_w = int(src.width / scale)
        out_h = int(src.height / scale)
        count = min(src.count, 3)
        arr = src.read(
            indexes=list(range(1, count + 1)),
            out_shape=(count, out_h, out_w),
            resampling=Resampling.bilinear,
        )
        img = np.moveaxis(arr, 0, -1)
        if img.dtype != np.uint8:
            img = img.astype(float)
            lo = np.nanpercentile(img, 1)
            hi = np.nanpercentile(img, 99)
            img = np.clip((img - lo) / max(hi - lo, 1e-9), 0, 1)
        else:
            img = img.astype(float) / 255.0
        extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
    return img, extent


def add_panel_label(ax, label: str, x: float = -0.03, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="bottom",
        ha="left",
        color="#202020",
    )


def significance_stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def marker_size(n: pd.Series | np.ndarray, nmax: float | None = None) -> np.ndarray:
    n = np.asarray(n, dtype=float)
    if nmax is None:
        nmax = np.nanmax(n) if np.isfinite(n).any() else 1.0
    return 35.0 + 135.0 * np.sqrt(np.clip(n, 1, None) / max(nmax, 1.0))


def map_site_summary(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for site, g in master.groupby("tower_id"):
        ci_width = (
            pd.to_numeric(g["deseasonalized_pearson_r_ci_high"], errors="coerce")
            - pd.to_numeric(g["deseasonalized_pearson_r_ci_low"], errors="coerce")
        )
        rows.append(
            {
                "tower_id": site,
                "latitude": float(g["latitude"].iloc[0]),
                "longitude": float(g["longitude"].iloc[0]),
                "r_weighted": fisher_weighted_mean_r(
                    g["deseasonalized_pearson_r"], g["n_site_month_observations"]
                ),
                "n_total": float(pd.to_numeric(g["n_site_month_observations"], errors="coerce").sum()),
                "uncertainty_width": float(ci_width.median()) if ci_width.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def draw_map_panel(fig: plt.Figure, ax, master: pd.DataFrame) -> None:
    pc = ccrs.PlateCarree()
    img, extent = read_terrain_background()
    ax.set_global()
    if img is not None and extent is not None:
        ax.imshow(img, origin="upper", extent=extent, transform=pc, zorder=0, alpha=0.92)
    else:
        ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#D8D0BD", zorder=0)
        ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="#DFE7EC", zorder=0)

    try:
        # Use geopandas to read the Natural Earth polygons used for cartographic context.
        import cartopy.io.shapereader as shpreader

        world_path = shpreader.natural_earth("110m", "cultural", "admin_0_countries")
        world = gpd.read_file(world_path)
        ax.add_geometries(
            world.geometry,
            crs=pc,
            facecolor="none",
            edgecolor="#4A4A4A",
            linewidth=0.25,
            alpha=0.75,
            zorder=2,
        )
    except Exception:
        ax.add_feature(cfeature.BORDERS.with_scale("110m"), edgecolor="#4A4A4A", linewidth=0.25, zorder=2)

    ax.coastlines(resolution="110m", linewidth=0.35, color="#3E3E3E", zorder=3)
    gl = ax.gridlines(draw_labels=False, linewidth=0.25, color="#8A8A8A", alpha=0.35)
    gl.xlocator = plt.FixedLocator(np.arange(-180, 181, 60))
    gl.ylocator = plt.FixedLocator(np.arange(-60, 91, 30))

    sites = map_site_summary(master)
    valid = sites.dropna(subset=["r_weighted", "latitude", "longitude"])
    ci = valid["uncertainty_width"].fillna(valid["uncertainty_width"].median())
    ci = ci.fillna(1.0)
    lw = np.clip(0.45 + 2.0 * (ci / max(float(ci.quantile(0.9)), 1e-6)), 0.45, 2.6)
    edge = np.where(ci <= ci.median(), "#151515", "#FFFFFF")
    sc = ax.scatter(
        valid["longitude"],
        valid["latitude"],
        c=valid["r_weighted"],
        s=marker_size(valid["n_total"]),
        cmap=COUPLING_CMAP,
        norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1),
        edgecolors=edge,
        linewidths=lw,
        transform=pc,
        alpha=0.94,
        zorder=5,
    )
    cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.045, pad=0.025)
    cb.set_label("Deseasonalized SIF-GPP coupling (weighted site mean Pearson r)")
    cb.set_ticks([-1, -0.5, 0, 0.5, 1])

    sizes = [25, 75, 150]
    handles = [
        ax.scatter([], [], s=marker_size(np.array([n]), valid["n_total"].max())[0], facecolor="#BBBBBB",
                   edgecolor="#222222", linewidth=0.8)
        for n in sizes
    ]
    leg1 = ax.legend(
        handles,
        [str(n) for n in sizes],
        title="Site-months",
        loc="lower left",
        frameon=True,
        framealpha=0.78,
        borderpad=0.35,
        handletextpad=0.8,
        fontsize=7,
        title_fontsize=7,
    )
    ax.add_artist(leg1)
    outline_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#BBBBBB", markeredgecolor="#151515",
               markeredgewidth=0.7, markersize=6, label="narrower CI"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#BBBBBB", markeredgecolor="#FFFFFF",
               markeredgewidth=1.8, markersize=6, label="wider CI"),
    ]
    ax.legend(
        handles=outline_handles,
        title="Outline",
        loc="lower right",
        frameon=True,
        framealpha=0.78,
        borderpad=0.35,
        fontsize=7,
        title_fontsize=7,
    )
    ax.set_title("Global footprint-aware coupling map", loc="left", fontsize=12, fontweight="bold", pad=7)
    add_panel_label(ax, "A", x=-0.055, y=1.02)


def draw_effect_matrix(ax, strat: pd.DataFrame) -> None:
    eff = strat[strat["analysis_type"] == "high_minus_low_effect"].copy()
    drivers = [cfg["label"] for cfg in DRIVERS.values()]
    mat = eff.pivot(index="driver", columns="product", values="effect_high_minus_low").reindex(
        index=drivers, columns=PRODUCT_ORDER
    )
    plo = eff.pivot(index="driver", columns="product", values="effect_ci_low").reindex(index=drivers, columns=PRODUCT_ORDER)
    phi = eff.pivot(index="driver", columns="product", values="effect_ci_high").reindex(index=drivers, columns=PRODUCT_ORDER)
    pp = eff.pivot(index="driver", columns="product", values="effect_p_value").reindex(index=drivers, columns=PRODUCT_ORDER)

    data = mat.to_numpy(dtype=float)
    max_abs = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 0.2
    im = ax.imshow(
        data,
        cmap=EFFECT_CMAP,
        norm=TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs),
        aspect="auto",
    )
    ax.set_xticks(np.arange(len(PRODUCT_ORDER)))
    ax.set_xticklabels(PRODUCT_ORDER, rotation=35, ha="right")
    for tick, prod in zip(ax.get_xticklabels(), PRODUCT_ORDER):
        tick.set_color(PRODUCT_COLORS[prod])
        tick.set_fontweight("bold")
    ax.set_yticks(np.arange(len(drivers)))
    ax.set_yticklabels(drivers)
    ax.set_title("Environmental control matrix", loc="left", fontsize=11, fontweight="bold")
    for i, drv in enumerate(drivers):
        for j, prod in enumerate(PRODUCT_ORDER):
            val = mat.loc[drv, prod]
            if pd.isna(val):
                ax.text(j, i, "NA", ha="center", va="center", fontsize=7, color="#555555")
                continue
            lo = plo.loc[drv, prod]
            hi = phi.loc[drv, prod]
            p = pp.loc[drv, prod]
            ci_txt = "" if pd.isna(lo) or pd.isna(hi) else f"\n[{lo:+.2f},{hi:+.2f}]"
            ax.text(
                j,
                i,
                f"{val:+.2f}{significance_stars(p)}{ci_txt}",
                ha="center",
                va="center",
                fontsize=6.7,
                color="#1B1B1B",
            )
    ax.set_xlabel("SIF product")
    ax.set_ylabel("Driver anomaly")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("High minus Low anomaly-bin mean r")
    add_panel_label(ax, "B")


def draw_biome_heatmap(ax, master: pd.DataFrame) -> None:
    g = (
        master.groupby(["igbp_class", "product"], dropna=False)
        .agg(mean_r=("deseasonalized_pearson_r", "mean"), n=("tower_id", "nunique"))
        .reset_index()
    )
    g["igbp_class"] = g["igbp_class"].fillna("Unknown")
    present = [i for i in IGBP_ORDER if i in set(g["igbp_class"])]
    for i in sorted(set(g["igbp_class"]) - set(present)):
        present.append(i)
    mat = g.pivot(index="igbp_class", columns="product", values="mean_r").reindex(index=present, columns=PRODUCT_ORDER)
    nmat = g.pivot(index="igbp_class", columns="product", values="n").reindex(index=present, columns=PRODUCT_ORDER)
    im = ax.imshow(
        mat.to_numpy(float),
        cmap=COUPLING_CMAP,
        norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1),
        aspect="auto",
    )
    ax.set_xticks(np.arange(len(PRODUCT_ORDER)))
    ax.set_xticklabels(PRODUCT_ORDER, rotation=35, ha="right")
    for tick, prod in zip(ax.get_xticklabels(), PRODUCT_ORDER):
        tick.set_color(PRODUCT_COLORS[prod])
        tick.set_fontweight("bold")
    ax.set_yticks(np.arange(len(present)))
    ax.set_yticklabels(present)
    ax.set_title("Biome controls on coupling", loc="left", fontsize=11, fontweight="bold")
    for i, igbp in enumerate(present):
        for j, prod in enumerate(PRODUCT_ORDER):
            val = mat.loc[igbp, prod]
            n = nmat.loc[igbp, prod]
            if pd.isna(val):
                ax.text(j, i, "NA", ha="center", va="center", fontsize=7, color="#555555")
            else:
                ax.text(j, i, f"{val:+.2f}\nn={int(n)}", ha="center", va="center", fontsize=6.9, color="#1F1F1F")
    ax.set_xlabel("SIF product")
    ax.set_ylabel("IGBP class")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("Mean site-level r")
    add_panel_label(ax, "C")


def draw_shap_panel(ax, shap_df: pd.DataFrame) -> None:
    groups = shap_df[shap_df["row_type"] == "feature_group"].copy()
    groups = groups.sort_values("mean_abs_shap", ascending=True)
    colors = {
        "VPD": "#9C4A3E",
        "ET": "#2D7F6F",
        "Soil moisture": "#4F8A50",
        "Radiation": "#C59B34",
        "Temperature": "#B05A32",
        "Biome": "#5B6F9E",
        "SIF product": "#6C4A7F",
        "Latitude": "#6C756B",
    }
    y = np.arange(len(groups))
    ax.barh(
        y,
        groups["mean_abs_shap"],
        color=[colors.get(g, "#777777") for g in groups["feature_group"]],
        edgecolor="#202020",
        linewidth=0.35,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(groups["feature_group"])
    ax.set_xlabel("Mean absolute SHAP contribution")
    ax.set_title("Machine-learning interpretation", loc="left", fontsize=11, fontweight="bold")
    ax.grid(axis="x", color="#D5D5D5", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not groups.empty:
        row = groups.iloc[0]
        r2 = row.get("cv_mean_r2", np.nan)
        rmse = row.get("cv_mean_rmse", np.nan)
        ax.text(
            0.98,
            0.05,
            f"Grouped CV by tower\nR2={r2:.2f}, RMSE={rmse:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.5,
            color="#333333",
        )
    add_panel_label(ax, "D")


def draw_concept_panel(ax) -> None:
    ax.set_axis_off()
    ax.set_title("Conceptual synthesis", loc="left", fontsize=11, fontweight="bold")
    add_panel_label(ax, "E")
    boxes = [
        (0.08, 0.70, 0.84, 0.18, "Climate stress\nVPD, heat, radiation anomalies", "#F0D4C4"),
        (0.08, 0.42, 0.84, 0.18, "Water and energy limitation\nsoil moisture and ET response", "#D3E3D0"),
        (0.08, 0.14, 0.84, 0.18, "Observed SIF-GPP coupling\nstrength and uncertainty", "#D9DCEB"),
    ]
    for x, y, w, h, text, color in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.018",
            facecolor=color,
            edgecolor="#333333",
            linewidth=0.8,
            transform=ax.transAxes,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.2, transform=ax.transAxes)
    for y0, y1 in [(0.70, 0.60), (0.42, 0.32)]:
        arrow = FancyArrowPatch(
            (0.50, y0),
            (0.50, y1),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.1,
            color="#333333",
            transform=ax.transAxes,
        )
        ax.add_patch(arrow)
    ax.text(
        0.04,
        0.06,
        "Biome traits set the sensitivity; satellite sampling and observation count set detectability.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        color="#333333",
    )


def create_figure(master: pd.DataFrame, strat: pd.DataFrame, shap_df: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 10,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "pdf.fonttype": 42,
            "savefig.dpi": 600,
        }
    )
    fig = plt.figure(figsize=(16.0, 13.2), constrained_layout=True)
    gs = fig.add_gridspec(3, 6, height_ratios=[1.18, 1.0, 0.9])
    ax_map = fig.add_subplot(gs[0, :], projection=ccrs.Robinson())
    ax_b = fig.add_subplot(gs[1, :3])
    ax_c = fig.add_subplot(gs[1, 3:])
    ax_d = fig.add_subplot(gs[2, :3])
    ax_e = fig.add_subplot(gs[2, 3:])

    draw_map_panel(fig, ax_map, master)
    draw_effect_matrix(ax_b, strat)
    draw_biome_heatmap(ax_c, master)
    draw_shap_panel(ax_d, shap_df)
    draw_concept_panel(ax_e)

    png = OUT_DIR / "Figure_mechanistic_controls_SIF_GPP.png"
    pdf = OUT_DIR / "Figure_mechanistic_controls_SIF_GPP.pdf"
    fig.savefig(png, dpi=600)
    fig.savefig(pdf)
    plt.close(fig)


def write_report(
    master: pd.DataFrame,
    strat: pd.DataFrame,
    mixed: pd.DataFrame,
    shap_df: pd.DataFrame,
    cv: pd.DataFrame,
    drivers: pd.DataFrame,
) -> None:
    n_rows = len(master)
    n_towers = master["tower_id"].nunique()
    product_counts = master.groupby("product")["tower_id"].nunique().reindex(PRODUCT_ORDER).fillna(0).astype(int)
    swc_cov = float(master["soil_moisture_anomaly"].notna().mean())
    et_cov = float(master["evapotranspiration_anomaly"].notna().mean())
    top_shap = shap_df[shap_df["row_type"] == "feature_group"].sort_values("mean_abs_shap", ascending=False).head(8)
    driver_terms = mixed[
        mixed["term"].isin(
            [
                "vpd_anomaly_z",
                "air_temperature_anomaly_z",
                "shortwave_radiation_anomaly_z",
                "soil_moisture_anomaly_z",
                "evapotranspiration_anomaly_z",
            ]
        )
    ].copy()

    effect = strat[strat["analysis_type"] == "high_minus_low_effect"].copy()
    effect = effect.sort_values("effect_high_minus_low", key=lambda s: s.abs(), ascending=False).head(10)

    report = []
    report.append("# Mechanistic Analysis Report\n")
    report.append("## 1. Scientific Question\n")
    report.append(
        "Within eddy-covariance tower footprints, what controls the strength of the relationship "
        "between satellite SIF anomalies and observed ecosystem GPP anomalies?\n"
    )
    report.append("## 2. Data Sources\n")
    report.append(
        "- SIF-GPP coupling input: `SIF_EC_SIF_GPP_Relationship_Analysis/01_analysis_ready/"
        "SIF_GPP_Analysis_Ready_Master.csv`, through the existing mechanistic working table. "
        "This preserves the completed footprint-aware SIF extraction, tower GPP matching, and "
        "deseasonalized SIF-GPP relationship calculations.\n"
    )
    report.append(
        "- Environmental drivers: monthly tower meteorology from `Remaining Fluxnet Sites/HH files/"
        "monthly_GEE_csv_outputs_from_HH_HR/new_sites_fluxnet_monthly_with_metadata_all_sites.csv` "
        "and `fluxnet/Monthly Datasets/fluxnet_monthly_with_metadata_all_sites.csv`.\n"
    )
    report.append(
        "- ET source: monthly tower latent heat flux converted to ET-equivalent mm day-1 using "
        "`ET = latent_heat * 86400 / 2.45e6`. This is an energy-flux ET proxy, not an independent "
        "water-balance ET product.\n"
    )
    report.append(
        "- Spatial representation: climate variables are tower/EC-footprint conditions from FLUXNET "
        "tower records, not values sampled from surrounding gridded pixels.\n"
    )
    report.append(
        "- Temporal representation: monthly, matched to the same tower-product months used for the "
        "SIF-GPP coupling analysis. Driver anomalies are relative to each tower and calendar-month "
        "climatology.\n"
    )
    report.append(
        "- Vegetation state: no LAI, NDVI, EVI, or fPAR column was found in the existing tower-month "
        "driver inputs, so this covariate is flagged unavailable rather than imputed.\n"
    )
    report.append("## 3. Methods\n")
    report.append(
        f"The master driver table contains {n_rows} tower-product rows across {n_towers} towers. "
        "Each row joins the existing site-level deseasonalized Pearson r and R2 to environmental "
        "driver summaries computed only over the matching monthly SIF-GPP record for that tower and product.\n"
    )
    report.append("Product coverage in the master table:\n")
    report.append(product_counts.to_frame("n_towers").to_markdown())
    report.append("\n")
    report.append(
        f"Soil moisture anomaly is available in {swc_cov:.1%} of tower-product rows; ET anomaly is "
        f"available in {et_cov:.1%}. Missing soil moisture is retained explicitly.\n"
    )
    report.append("Environmental stratification recomputed site-level SIF-GPP anomaly coupling within Low, Medium, and High product-specific terciles of each monthly tower driver anomaly. Uncertainty is a tower bootstrap of mean site-level r.\n")
    report.append("## 4. Statistical Models\n")
    report.append(
        "Mixed-effects model response is Fisher-z-transformed site-level deseasonalized Pearson r. "
        "Fixed effects are VPD, temperature, radiation, soil moisture, ET, land-cover group, and SIF product; "
        "tower is included as a random intercept when the model converges.\n"
    )
    if not mixed.empty:
        model_meta = mixed.iloc[0]
        report.append(
            f"Model status: `{model_meta['status']}`; model type: `{model_meta['model_type']}`; "
            f"N={int(model_meta['n_site_product_rows'])}, towers={int(model_meta['n_towers'])}; "
            f"marginal R2={model_meta['marginal_r2']:.3f}, conditional R2={model_meta['conditional_r2']:.3f}; "
            f"tower random-intercept variance={model_meta['var_random_tower']:.3g}.\n"
        )
    if not driver_terms.empty:
        report.append(driver_terms[["term", "coefficient", "ci_low", "ci_high", "p_value"]].to_markdown(index=False))
        report.append("\n")
    report.append(
        "Machine-learning interpretation uses XGBoost with GroupKFold cross-validation by tower_id, "
        "so rows from the same tower are not split across train and test folds. SHAP values come from "
        "XGBoost's built-in tree contribution method (`pred_contribs=True`).\n"
    )
    if not cv.empty:
        report.append(
            f"Grouped CV performance: mean R2={cv['r2'].mean():.3f} (SD {cv['r2'].std(ddof=1):.3f}), "
            f"mean RMSE={cv['rmse'].mean():.3f}, mean MAE={cv['mae'].mean():.3f}.\n"
        )
    report.append("Top SHAP feature groups:\n")
    report.append(top_shap[["rank", "feature_group", "mean_abs_shap", "normalized_importance"]].to_markdown(index=False))
    report.append("\n")
    report.append("## 5. Interpretation Of Results\n")
    report.append(
        "The new analysis shifts the paper from product ranking toward a mechanism: SIF-GPP coupling varies "
        "with water-atmosphere stress, energy availability, biome context, and the sampling structure of each "
        "satellite product. The environmental matrix reports how mean coupling changes between high and low "
        "monthly anomaly conditions, while the mixed model and SHAP analysis ask whether site-product coupling "
        "is predictable from those same drivers after accounting for biome and product identity.\n"
    )
    if not effect.empty:
        report.append("Largest High-Low stratification effects:\n")
        report.append(
            effect[
                ["driver", "product", "effect_high_minus_low", "effect_ci_low", "effect_ci_high", "effect_p_value"]
            ].to_markdown(index=False)
        )
        report.append("\n")
    report.append("## 6. Limitations\n")
    report.append(
        "- Product sampling is unequal. GOSIF, GOME-2, and TROPOSIF provide much denser time series than OCO-2 and OCO-3, and OCO rows include reconstruction flags from the completed workflow.\n"
    )
    report.append(
        "- OCO-2 remains sparse at many towers; site-level r and bin-level means with small N should be interpreted as weak evidence unless the uncertainty interval is narrow.\n"
    )
    report.append(
        "- Soil moisture is incomplete and is not imputed in the master table. Complete-case mixed models therefore represent the subset with valid SWC.\n"
    )
    report.append(
        "- ET is derived from latent heat as a monthly mean ET-equivalent flux. It is footprint-relevant but not a separate remote-sensing ET retrieval or hydrologic closure estimate.\n"
    )
    report.append(
        "- LAI/vegetation-index state was not available in the existing monthly inputs; adding MODIS LAI/fPAR/NDVI would strengthen the ecosystem-state mechanism but would be a new analysis step.\n"
    )
    report.append(
        "- These are observational associations. The mixed model and XGBoost SHAP summaries support mechanistic interpretation but do not prove causality.\n"
    )
    report.append("## 7. Figure Descriptions\n")
    report.append(
        "- Panel A: global tower map on a real terrain/topographic background. Marker color is weighted site mean deseasonalized Pearson r, size is observation count, and outline indicates uncertainty width.\n"
    )
    report.append(
        "- Panel B: driver effect matrix, defined as High minus Low anomaly-bin mean site-level coupling for each SIF product, with bootstrap confidence intervals and significance stars.\n"
    )
    report.append(
        "- Panel C: IGBP heatmap of mean site-level coupling by product, with tower sample size per cell.\n"
    )
    report.append(
        "- Panel D: XGBoost SHAP feature-group ranking for predicting site-level coupling strength.\n"
    )
    report.append(
        "- Panel E: synthesis schematic linking climate stress, ecosystem water/energy limitation, and changes in SIF-GPP coupling.\n"
    )

    (OUT_DIR / "Mechanistic_analysis_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    ensure_outdir()
    monthly, drivers = build_monthly_with_drivers()
    master = create_master(monthly)
    strat = climate_stratification(monthly, master)
    mixed = mixed_model(master)
    shap_df, cv = run_ml_shap(master)
    create_figure(master, strat, shap_df)
    write_report(master, strat, mixed, shap_df, cv, drivers)
    print("Mechanistic driver analysis complete.")
    print(f"Output directory: {OUT_DIR}")
    print(f"Master rows: {len(master)}, towers: {master['tower_id'].nunique()}")
    print(f"Stratification rows: {len(strat)}")
    if not mixed.empty:
        print(f"Mixed model status: {mixed.iloc[0]['status']}")
    if not cv.empty:
        print(f"XGBoost grouped CV mean R2: {cv['r2'].mean():.3f}")


if __name__ == "__main__":
    main()
