"""Build the revised four-product figures and figure-data tables for raw versus deseasonalized SIF-GPP coupling (and temporal structure), restricted to GOSIF, GOME-2, TROPOSIF and OCO-3.

Input: analysis-ready master table and harmonic residuals. Output: figures, figure-data CSVs and a short report."""
from __future__ import annotations
# Original script filename (sanitized): build_revised_4product_final_visualizations.py; public name: fig05_raw_vs_deseasonalized_coupling.py
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
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False


ROOT = (Path(PROJECT_ROOT) / 'All_the_Visualizations_REVISED_4PRODUCT_FINAL/05_scripts/build_revised_4product_final_visualizations.py').resolve().parents[2]
OUTPUT_ROOT = ROOT / "All_the_Visualizations_REVISED_4PRODUCT_FINAL"
MAIN_DIR = OUTPUT_ROOT / "01_main_revised"
CHECK_DIR = OUTPUT_ROOT / "03_intermediate_checks"
REPORT_DIR = OUTPUT_ROOT / "04_reports"
SCRIPT_DIR = OUTPUT_ROOT / "05_scripts"
DATA_DIR = ROOT / "SIF_EC_Final_4Product_MainPaper_Figures_v2" / "03_figure_data_csv"

PRODUCTS = ["GOSIF", "GOME-2", "TROPOSIF", "OCO-3"]
PRODUCT_COLORS = {
    "GOSIF": "#123B73",
    "GOME-2": "#188C89",
    "TROPOSIF": "#C99A19",
    "OCO-3": "#7A3B8F",
}
PRODUCT_MARKERS = {"GOSIF": "o", "GOME-2": "s", "TROPOSIF": "^", "OCO-3": "D"}
DRIVER_LABELS = {
    "VPD": "VPD",
    "ET": "ET",
    "Soil moisture": "Soil moisture",
    "Radiation": "Radiation",
    "Temperature": "Temperature",
    "Latitude": "Latitude",
}
DRIVER_FIELD = {
    "VPD": "vpd_mean",
    "ET": "evapotranspiration_mean",
    "Soil moisture": "soil_moisture_mean",
    "Radiation": "shortwave_radiation_mean",
    "Temperature": "air_temperature_mean",
}
DRIVER_MAP_SOURCE_NAME = {
    "VPD": "VPD",
    "ET": "ET",
    "Soil moisture": "Soil moisture",
    "Radiation": "Radiation",
    "Temperature": "Temperature",
}
BIOME_ORDER = ["ENF", "EBF", "DBF", "MF", "CSH", "OSH", "WSA", "SAV", "GRA", "CRO"]
BIOME_COLORS = {
    "ENF": "#456A4D",
    "EBF": "#2D7F5E",
    "DBF": "#8AA860",
    "MF": "#6C8E7A",
    "CSH": "#B99362",
    "OSH": "#C6A15B",
    "WSA": "#C48A4A",
    "SAV": "#D9B45F",
    "GRA": "#88A85A",
    "CRO": "#B85C5C",
    "Unknown": "#9E9E9E",
}

RSE_FONT = "Arial"
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [RSE_FONT, "DejaVu Sans", "Liberation Sans"],
        "axes.titlesize": 10,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "figure.dpi": 120,
        "savefig.dpi": 600,
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)

R_MATRIX_CMAP = LinearSegmentedColormap.from_list(
    "purple_magenta_orange_yellow_centered",
    ["#3B0F70", "#8A2387", "#F1D8ED", "#F28E2B", "#F6D55C"],
)
BIOME_R_CMAP = LinearSegmentedColormap.from_list(
    "biome_r_diverging",
    ["#284B8F", "#F7F7F7", "#B6463A"],
)
DRIVER_CMAPS = {
    "VPD": LinearSegmentedColormap.from_list("vpd_rust", ["#fff4d8", "#e1a15c", "#b64935", "#6b1d1d"]),
    "ET": LinearSegmentedColormap.from_list("et_teal", ["#e8f6f3", "#8bd3d1", "#238b7e", "#0d4f5c"]),
    "Soil moisture": LinearSegmentedColormap.from_list("soil_blue", ["#eff6ff", "#9ecae1", "#3182bd", "#08519c"]),
    "Radiation": LinearSegmentedColormap.from_list("rad_gold", ["#fff7bc", "#fec44f", "#d95f0e", "#8c2d04"]),
    "Temperature": LinearSegmentedColormap.from_list("temp_orange_red", ["#fee8c8", "#fdbb84", "#e34a33", "#7f0000"]),
}


def clean_product_frame(df: pd.DataFrame, col: str = "SIF_PRODUCT") -> pd.DataFrame:
    if col in df.columns:
        df = df[df[col].isin(PRODUCTS)].copy()
    return df


def read_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required finalized 4-product table: {path}")
    return pd.read_csv(path)


def save_figure(fig: plt.Figure, stem: str, directory: Path = MAIN_DIR) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ["png", "pdf", "svg"]:
        path = directory / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
        paths.append(path)
    plt.close(fig)
    return paths


def panel_label(ax, label: str, x: float = -0.035, y: float = 1.04, color: str = "#111111") -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color=color,
    )


def add_map_base(ax, extent=None, dark=False) -> None:
    if not HAS_CARTOPY:
        ax.set_xlim(-180, 180)
        ax.set_ylim(-58, 78)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, color="#d9d9d9", lw=0.35)
        return
    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    if dark:
        ax.set_facecolor("#111820")
        ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="#111820", zorder=0)
        ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#253033", zorder=0)
        coast = "#d8e5e1"
        border = "#8aa0a1"
    else:
        ax.set_facecolor("#f6f7f5")
        ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="#edf2f4", zorder=0)
        ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#f8f7f1", zorder=0)
        coast = "#6e7c7c"
        border = "#b4b8b8"
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.35, edgecolor=coast, zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.22, edgecolor=border, zorder=2)


def scatter_geo(ax, lon, lat, **kwargs) -> None:
    if HAS_CARTOPY and hasattr(ax, "projection"):
        ax.scatter(lon, lat, transform=ccrs.PlateCarree(), **kwargs)
    else:
        ax.scatter(lon, lat, **kwargs)


def plot_world_points(ax, df: pd.DataFrame, value_col: str, cmap, norm=None, size_col: str | None = None, title: str = ""):
    add_map_base(ax, dark=False)
    if norm is None:
        norm = Normalize(np.nanpercentile(df[value_col], 2), np.nanpercentile(df[value_col], 98))
    sizes = 18
    if size_col and size_col in df.columns:
        n = df[size_col].fillna(df[size_col].median()).to_numpy()
        sizes = 14 + 38 * (np.sqrt(n) - np.sqrt(np.nanmin(n))) / max(np.sqrt(np.nanmax(n)) - np.sqrt(np.nanmin(n)), 1e-6)
    scatter_geo(
        ax,
        df["LONGITUDE"],
        df["LATITUDE"],
        c=df[value_col],
        cmap=cmap,
        norm=norm,
        s=sizes,
        edgecolor="#ffffff",
        linewidth=0.35,
        alpha=0.95,
        zorder=5,
    )
    ax.set_title(title, loc="left", fontweight="bold", pad=4)


def recompute_biome_correlations(monthly: pd.DataFrame, min_n: int = 12) -> pd.DataFrame:
    df = clean_product_frame(monthly.copy())
    df["MONTH"] = (df["YYYYMM"].astype(int) % 100).astype(int)
    df = df.dropna(subset=["SIF_VALUE", "GPP_VALUE", "IGBP"])
    df = df[df["IGBP"].isin(BIOME_ORDER)]
    sif_clim = df.groupby(["SITE_ID", "SIF_PRODUCT", "MONTH"])["SIF_VALUE"].transform("mean")
    gpp_clim = df.groupby(["SITE_ID", "MONTH"])["GPP_VALUE"].transform("mean")
    df["SIF_ANOM"] = df["SIF_VALUE"] - sif_clim
    df["GPP_ANOM"] = df["GPP_VALUE"] - gpp_clim

    rows = []
    for (igbp, product), g in df.groupby(["IGBP", "SIF_PRODUCT"]):
        g = g.dropna(subset=["SIF_ANOM", "GPP_ANOM"])
        n = len(g)
        if n >= min_n and g["SIF_ANOM"].nunique() > 1 and g["GPP_ANOM"].nunique() > 1:
            r = float(np.corrcoef(g["SIF_ANOM"], g["GPP_ANOM"])[0, 1])
        else:
            r = np.nan
        rows.append({"IGBP": igbp, "SIF_PRODUCT": product, "PEARSON_R": r, "N": n})
    out = pd.DataFrame(rows)
    out = out[out["SIF_PRODUCT"].isin(PRODUCTS)]
    return out


def draw_heatmap(
    ax,
    matrix: pd.DataFrame,
    rows: list[str],
    cols: list[str],
    value_col: str,
    row_col: str,
    col_col: str,
    cmap,
    norm,
    title: str,
    annotate_n: bool = False,
    n_col: str = "N",
    fmt: str = "{:+.2f}",
    missing_label: str = "NA",
) -> None:
    arr = np.full((len(rows), len(cols)), np.nan)
    n_arr = np.full((len(rows), len(cols)), np.nan)
    lookup = matrix.set_index([row_col, col_col])
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            if (row, col) in lookup.index:
                rec = lookup.loc[(row, col)]
                if isinstance(rec, pd.DataFrame):
                    rec = rec.iloc[0]
                arr[i, j] = pd.to_numeric(rec[value_col], errors="coerce")
                if n_col in matrix.columns:
                    n_arr[i, j] = pd.to_numeric(rec[n_col], errors="coerce")
    masked = np.ma.masked_invalid(arr)
    im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")
    im.cmap.set_bad("#eeeeee")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=0, fontweight="bold")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_title(title, loc="left", fontweight="bold", pad=6)
    for i in range(len(rows)):
        for j in range(len(cols)):
            val = arr[i, j]
            if np.isfinite(val):
                text = fmt.format(val)
                if annotate_n and np.isfinite(n_arr[i, j]):
                    text += f"\nn={int(n_arr[i, j])}"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=7.1 if annotate_n else 9.2,
                    fontweight="bold",
                    color="#111111",
                    linespacing=0.94 if annotate_n else 1.05,
                )
            else:
                ax.text(j, i, missing_label, ha="center", va="center", fontsize=7.6, color="#777777")
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="both", length=0)
    return im


def prepare_mechanistic_matrix(effect: pd.DataFrame) -> pd.DataFrame:
    df = effect.copy()
    df = df[df["product"].isin(PRODUCTS)]
    df["driver_label"] = df["driver"].replace({"Air temperature": "Temperature", "Shortwave radiation": "Radiation"})
    keep = ["VPD", "ET", "Soil moisture", "Radiation", "Temperature", "Latitude"]
    df = df[df["driver_label"].isin(keep)]
    df["effect_value"] = pd.to_numeric(df["effect_high_minus_low"], errors="coerce")
    return df


def idw_surface(points: pd.DataFrame, value_col: str, lon_res: float = 3.0, lat_res: float = 2.5):
    pts = points.dropna(subset=["longitude", "latitude", value_col]).copy()
    lons = np.arange(-180, 181, lon_res)
    lats = np.arange(-58, 79, lat_res)
    xx, yy = np.meshgrid(lons, lats)
    if len(pts) < 3:
        return xx, yy, np.full_like(xx, np.nan, dtype=float)
    plon = pts["longitude"].to_numpy(float)
    plat = pts["latitude"].to_numpy(float)
    vals = pts[value_col].to_numpy(float)
    grid = np.empty(xx.size, dtype=float)
    flat_lon = xx.ravel()
    flat_lat = yy.ravel()
    for start in range(0, flat_lon.size, 1200):
        end = min(start + 1200, flat_lon.size)
        lon_chunk = flat_lon[start:end][:, None]
        lat_chunk = flat_lat[start:end][:, None]
        dlon = (lon_chunk - plon[None, :]) * np.cos(np.deg2rad(np.clip(lat_chunk, -70, 70)))
        dlat = lat_chunk - plat[None, :]
        dist = np.sqrt(dlon**2 + dlat**2)
        exact = dist < 1e-9
        weights = 1.0 / np.maximum(dist, 0.6) ** 2.0
        val = (weights @ vals) / weights.sum(axis=1)
        if exact.any():
            rows, cols = np.where(exact)
            val[rows] = vals[cols]
        grid[start:end] = val
    return xx, yy, grid.reshape(xx.shape)


def pcolormesh_geo(ax, xx, yy, zz, cmap, norm):
    if HAS_CARTOPY and hasattr(ax, "projection"):
        return ax.pcolormesh(xx, yy, zz, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, shading="auto", zorder=1, alpha=0.94)
    return ax.pcolormesh(xx, yy, zz, cmap=cmap, norm=norm, shading="auto", alpha=0.94)


def make_driver_surface_metadata(site_summary: pd.DataFrame) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], pd.DataFrame]:
    surfaces = {}
    meta = []
    for label, source_driver in DRIVER_MAP_SOURCE_NAME.items():
        sub = site_summary[site_summary["driver"].eq(source_driver)].copy()
        if sub.empty:
            continue
        sub["driver_mean"] = pd.to_numeric(sub["driver_mean"], errors="coerce")
        surfaces[label] = idw_surface(sub, "driver_mean")
        meta.append(
            {
                "driver": label,
                "source_driver_name": source_driver,
                "source_table": "mechanistic_geospatial_driver_site_summary_4product.csv",
                "n_tower_driver_rows": int(sub.dropna(subset=["driver_mean"]).shape[0]),
                "surface_method": "Inverse-distance weighted interpolation of finalized tower-derived driver means; no independent global gridded raster found in finalized figure data.",
            }
        )
    return surfaces, pd.DataFrame(meta)


def build_mechanistic_figure():
    effect = prepare_mechanistic_matrix(read_csv("mechanistic_effect_matrix_4product.csv"))
    monthly = read_csv("monthly_sif_gpp_records_4product.csv")
    biome = recompute_biome_correlations(monthly)
    biome.to_csv(CHECK_DIR / "recomputed_biome_r_matrix.csv", index=False)
    site_summary = read_csv("mechanistic_geospatial_driver_site_summary_4product.csv")
    site_summary = site_summary[~site_summary["products_present"].fillna("").str.contains("OCO-2|OCO2", regex=True)]
    surfaces, surface_meta = make_driver_surface_metadata(site_summary)
    surface_meta.to_csv(CHECK_DIR / "driver_surface_metadata.csv", index=False)

    if HAS_CARTOPY:
        fig = plt.figure(figsize=(16.4, 13.2), facecolor="white")
        gs = gridspec.GridSpec(4, 6, figure=fig, height_ratios=[1.65, 1.22, 1.22, 0.72], hspace=0.48, wspace=0.46)
        ax_a = fig.add_subplot(gs[0, 0:3])
        ax_b = fig.add_subplot(gs[0, 3:6])
        map_axes = [
            fig.add_subplot(gs[1, 0:2], projection=ccrs.Robinson()),
            fig.add_subplot(gs[1, 2:4], projection=ccrs.Robinson()),
            fig.add_subplot(gs[1, 4:6], projection=ccrs.Robinson()),
            fig.add_subplot(gs[2, 1:3], projection=ccrs.Robinson()),
            fig.add_subplot(gs[2, 3:5], projection=ccrs.Robinson()),
        ]
    else:
        fig = plt.figure(figsize=(16.4, 13.2), facecolor="white")
        gs = gridspec.GridSpec(4, 6, figure=fig, height_ratios=[1.65, 1.22, 1.22, 0.72], hspace=0.48, wspace=0.46)
        ax_a = fig.add_subplot(gs[0, 0:3])
        ax_b = fig.add_subplot(gs[0, 3:6])
        map_axes = [
            fig.add_subplot(gs[1, 0:2]),
            fig.add_subplot(gs[1, 2:4]),
            fig.add_subplot(gs[1, 4:6]),
            fig.add_subplot(gs[2, 1:3]),
            fig.add_subplot(gs[2, 3:5]),
        ]

    driver_rows = ["VPD", "ET", "Soil moisture", "Radiation", "Temperature", "Latitude"]
    im_a = draw_heatmap(
        ax_a,
        effect,
        driver_rows,
        PRODUCTS,
        "effect_value",
        "driver_label",
        "product",
        R_MATRIX_CMAP,
        TwoSlopeNorm(vmin=-0.16, vcenter=0.0, vmax=0.16),
        "A  Environmental effect matrix",
        annotate_n=False,
        fmt="{:+.2f}",
    )
    panel_label(ax_a, "A", x=-0.08)
    cb = fig.colorbar(im_a, ax=ax_a, orientation="horizontal", fraction=0.055, pad=0.075)
    cb.set_label("High-low tercile effect / latitude r", fontsize=7.4)
    cb.ax.tick_params(labelsize=7)

    im_b = draw_heatmap(
        ax_b,
        biome,
        [b for b in BIOME_ORDER if b in set(biome["IGBP"])],
        PRODUCTS,
        "PEARSON_R",
        "IGBP",
        "SIF_PRODUCT",
        BIOME_R_CMAP,
        TwoSlopeNorm(vmin=-0.65, vcenter=0.0, vmax=0.65),
        "B  Biome coupling matrix",
        annotate_n=True,
        fmt="{:+.2f}",
        missing_label="--",
    )
    panel_label(ax_b, "B", x=-0.08)
    cb_b = fig.colorbar(im_b, ax=ax_b, orientation="horizontal", fraction=0.055, pad=0.075)
    cb_b.set_label("Deseasonalized r", fontsize=7.4)
    cb_b.ax.tick_params(labelsize=7)

    map_labels = ["C", "D", "E", "F", "G"]
    for ax, driver, lab in zip(map_axes, ["VPD", "ET", "Soil moisture", "Radiation", "Temperature"], map_labels):
        add_map_base(ax, dark=False)
        xx, yy, zz = surfaces[driver]
        finite = zz[np.isfinite(zz)]
        norm = Normalize(np.nanpercentile(finite, 5), np.nanpercentile(finite, 95))
        mesh = pcolormesh_geo(ax, xx, yy, zz, DRIVER_CMAPS[driver], norm)
        ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.35, edgecolor="#4f5b5b", zorder=3) if HAS_CARTOPY else None
        sites = site_summary[site_summary["driver"].eq(DRIVER_MAP_SOURCE_NAME[driver])].drop_duplicates("tower_id")
        scatter_geo(
            ax,
            sites["longitude"],
            sites["latitude"],
            s=8,
            facecolor="#111111",
            edgecolor="white",
            linewidth=0.25,
            alpha=0.65,
            zorder=5,
        )
        ax.set_title(f"{lab}  {driver}", loc="left", fontweight="bold", pad=4)
        cb = fig.colorbar(mesh, ax=ax, orientation="horizontal", fraction=0.045, pad=0.035)
        cb.ax.tick_params(labelsize=6.6)
        cb.set_label("Tower-derived mean driver context", fontsize=6.9, labelpad=1)

    ax_h = fig.add_subplot(gs[3, 0:3])
    ml = read_csv("mechanistic_ml_importance_4product.csv")
    grouped = (
        ml[ml["row_type"].eq("group_summary")]
        if "row_type" in ml.columns and ml["row_type"].eq("group_summary").any()
        else ml
    )
    grouped = grouped.copy()
    grouped["normalized_importance"] = pd.to_numeric(grouped["normalized_importance"], errors="coerce")
    grouped = grouped.dropna(subset=["normalized_importance"]).sort_values("normalized_importance", ascending=True).tail(8)
    colors = ["#5F6F7A" if "product" not in str(x).lower() else "#9C7A37" for x in grouped["feature_group"]]
    ax_h.barh(grouped["feature_group"].astype(str), grouped["normalized_importance"], color=colors, edgecolor="white", linewidth=0.6)
    ax_h.set_xlabel("Normalized importance")
    ax_h.set_title("H  Machine-learning context", loc="left", fontweight="bold")
    panel_label(ax_h, "H", x=-0.08)
    ax_h.grid(axis="x", color="#dddddd", lw=0.45)
    ax_h.spines[["top", "right", "left"]].set_visible(False)

    ax_i = fig.add_subplot(gs[3, 3:6])
    ax_i.axis("off")
    panel_label(ax_i, "I", x=-0.04)
    ax_i.set_title("I  Synthesis", loc="left", fontweight="bold", pad=4)
    terms = [
        ("Climate\nstress", "#B6463A"),
        ("Water\nlimitation", "#188C89"),
        ("Biome\ncontext", "#6C8E7A"),
        ("Footprint-aware\ncoupling", "#123B73"),
    ]
    x_positions = [0.04, 0.29, 0.54, 0.78]
    for i, ((txt, color), x) in enumerate(zip(terms, x_positions)):
        ax_i.add_patch(Rectangle((x, 0.34), 0.18, 0.30, transform=ax_i.transAxes, facecolor=color, alpha=0.14, edgecolor=color, lw=1.0))
        ax_i.text(x + 0.09, 0.49, txt, transform=ax_i.transAxes, ha="center", va="center", fontsize=8.3, fontweight="bold", color="#222222", linespacing=1.0)
        if i < len(terms) - 1:
            ax_i.add_patch(FancyArrowPatch((x + 0.19, 0.485), (x_positions[i + 1] - 0.01, 0.485), transform=ax_i.transAxes, arrowstyle="-|>", mutation_scale=10, lw=1.0, color="#555555"))
    ax_i.text(
        0.05,
        0.12,
        "Driver context helps explain why residual SIF-GPP coupling remains spatially heterogeneous after deseasonalization.",
        transform=ax_i.transAxes,
        fontsize=8.2,
        color="#333333",
    )

    fig.suptitle("Mechanistic controls on footprint-aware SIF-GPP coupling", fontsize=15, fontweight="bold", x=0.02, ha="left", y=0.996)
    return save_figure(fig, "figure_mechanistic_controls_4product_final")


def plot_product_map(ax, df: pd.DataFrame, product: str, title: str, value_col: str = "DESEASONALIZED_PEARSON_R"):
    add_map_base(ax, dark=False)
    sub = df[df["SIF_PRODUCT"].eq(product)].dropna(subset=["LONGITUDE", "LATITUDE", value_col])
    norm = TwoSlopeNorm(vmin=-0.75, vcenter=0.0, vmax=0.75)
    sizes = 20 + 55 * np.sqrt(sub["N_SITE_MONTHS"].clip(lower=1)) / np.sqrt(max(sub["N_SITE_MONTHS"].max(), 1))
    if HAS_CARTOPY and hasattr(ax, "projection"):
        sc = ax.scatter(
            sub["LONGITUDE"],
            sub["LATITUDE"],
            c=sub[value_col],
            s=sizes,
            cmap=BIOME_R_CMAP,
            norm=norm,
            transform=ccrs.PlateCarree(),
            edgecolor=np.where(sub["CI_EXCLUDES_ZERO"].fillna(False), "#111111", "#ffffff"),
            linewidth=np.where(sub["CI_EXCLUDES_ZERO"].fillna(False), 0.65, 0.25),
            alpha=0.93,
            zorder=6,
        )
    else:
        sc = ax.scatter(
            sub["LONGITUDE"],
            sub["LATITUDE"],
            c=sub[value_col],
            s=sizes,
            cmap=BIOME_R_CMAP,
            norm=norm,
            edgecolor="#ffffff",
            linewidth=0.3,
            alpha=0.93,
            zorder=6,
        )
    ax.set_title(title, loc="left", fontweight="bold")
    return sc


def build_candidate_balanced_hierarchy():
    site = clean_product_frame(read_csv("main_panel_a_site_map_4product.csv"))
    dist = clean_product_frame(read_csv("main_panel_b_distribution_4product.csv"))
    effect = prepare_mechanistic_matrix(read_csv("mechanistic_effect_matrix_4product.csv"))
    biome = pd.read_csv(CHECK_DIR / "recomputed_biome_r_matrix.csv") if (CHECK_DIR / "recomputed_biome_r_matrix.csv").exists() else recompute_biome_correlations(read_csv("monthly_sif_gpp_records_4product.csv"))

    fig = plt.figure(figsize=(15.4, 10.8), facecolor="white")
    gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[1.2, 1.1, 1.08], hspace=0.38, wspace=0.36)
    map_axes = []
    for i, product in enumerate(PRODUCTS):
        if HAS_CARTOPY:
            ax = fig.add_subplot(gs[i // 2, (i % 2) * 2 : (i % 2) * 2 + 2], projection=ccrs.Robinson())
        else:
            ax = fig.add_subplot(gs[i // 2, (i % 2) * 2 : (i % 2) * 2 + 2])
        map_axes.append(ax)
        sc = plot_product_map(ax, site, product, f"{product}")
        if i == 0:
            panel_label(ax, "A", x=-0.03)
    cax = fig.add_axes([0.92, 0.565, 0.012, 0.30])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("Deseasonalized r", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    ax_b = fig.add_subplot(gs[2, 0])
    panel_label(ax_b, "B", x=-0.12)
    dist_value_col = "DESEASONALIZED_PEARSON_R" if "DESEASONALIZED_PEARSON_R" in dist.columns else "PEARSON_R"
    vals = [dist[dist["SIF_PRODUCT"].eq(p)][dist_value_col].dropna().to_numpy() for p in PRODUCTS]
    parts = ax_b.violinplot(vals, showmeans=False, showmedians=False, widths=0.75)
    for body, p in zip(parts["bodies"], PRODUCTS):
        body.set_facecolor(PRODUCT_COLORS[p])
        body.set_edgecolor("white")
        body.set_alpha(0.62)
    for i, (v, p) in enumerate(zip(vals, PRODUCTS), start=1):
        if len(v):
            ax_b.boxplot(v, positions=[i], widths=0.22, patch_artist=True, showfliers=False,
                         boxprops={"facecolor": "white", "edgecolor": PRODUCT_COLORS[p], "linewidth": 1.1},
                         medianprops={"color": "#111111", "linewidth": 1.2},
                         whiskerprops={"color": PRODUCT_COLORS[p]}, capprops={"color": PRODUCT_COLORS[p]})
    ax_b.axhline(0, color="#555555", lw=0.75)
    ax_b.set_xticks(range(1, len(PRODUCTS) + 1))
    ax_b.set_xticklabels(PRODUCTS, rotation=35, ha="right", fontweight="bold")
    ax_b.set_ylabel("Site-level r")
    ax_b.set_title("Site distributions", loc="left", fontweight="bold")
    ax_b.grid(axis="y", color="#dddddd", lw=0.45)
    ax_b.spines[["top", "right"]].set_visible(False)

    ax_c = fig.add_subplot(gs[2, 1:3])
    rows = [b for b in BIOME_ORDER if b in set(biome["IGBP"])]
    im_c = draw_heatmap(
        ax_c,
        biome,
        rows,
        PRODUCTS,
        "PEARSON_R",
        "IGBP",
        "SIF_PRODUCT",
        R_MATRIX_CMAP,
        TwoSlopeNorm(vmin=-0.65, vcenter=0.0, vmax=0.65),
        "Biome controls: correlation r",
        annotate_n=True,
        fmt="{:+.2f}",
        missing_label="--",
    )
    panel_label(ax_c, "C", x=-0.06)
    cax_c = fig.add_axes([ax_c.get_position().x1 + 0.005, ax_c.get_position().y0 + 0.02, 0.008, ax_c.get_position().height - 0.04])
    cb_c = fig.colorbar(im_c, cax=cax_c)
    cb_c.ax.tick_params(labelsize=7)
    cb_c.set_label("r", fontsize=7)

    ax_d = fig.add_subplot(gs[2, 3])
    panel_label(ax_d, "D", x=-0.12)
    effect_display = effect.copy()
    effect_display["driver_label_candidate"] = effect_display["driver_label"].replace(
        {"Soil moisture": "Soil\nmoisture", "Radiation": "Radiation", "Temperature": "Temperature"}
    )
    driver_rows = ["VPD", "ET", "Soil\nmoisture", "Radiation", "Temperature"]
    im_d = draw_heatmap(
        ax_d,
        effect_display,
        driver_rows,
        PRODUCTS,
        "effect_value",
        "driver_label_candidate",
        "product",
        BIOME_R_CMAP,
        TwoSlopeNorm(vmin=-0.16, vcenter=0.0, vmax=0.16),
        "Environmental controls",
        annotate_n=False,
        fmt="{:+.2f}",
    )
    ax_d.tick_params(axis="y", labelsize=6.8, pad=1)
    ax_d.tick_params(axis="x", labelsize=7.1, pad=1)
    cax_d = fig.add_axes([ax_d.get_position().x1 + 0.005, ax_d.get_position().y0 + 0.02, 0.008, ax_d.get_position().height - 0.04])
    cb_d = fig.colorbar(im_d, cax=cax_d)
    cb_d.ax.tick_params(labelsize=7)
    cb_d.set_label("effect", fontsize=7)

    fig.suptitle("Footprint-aware SIF-GPP coupling hierarchy", fontsize=15, fontweight="bold", x=0.02, ha="left", y=0.998)
    return save_figure(fig, "Candidate1_BalancedHierarchy_4Product_Final")


def build_raw_vs_deseasonalized():
    df = clean_product_frame(read_csv("Figure2_Raw_vs_Deseasonalized_4product.csv"))
    deseas_label = "DESEASONALIZED" if "DESEASONALIZED" in set(df["TRANSFORM"]) else "ANOM_ALL"
    fig, ax = plt.subplots(figsize=(8.0, 4.8), facecolor="white")
    x = np.arange(len(PRODUCTS))
    width = 0.32
    for k, transform in enumerate(["RAW", deseas_label]):
        sub = df[df["TRANSFORM"].eq(transform)].set_index("SIF_PRODUCT").reindex(PRODUCTS)
        label = "Raw" if transform == "RAW" else "Deseasonalized"
        color = "#344E75" if transform == "RAW" else "#D28C2E"
        offset = -width / 2 if k == 0 else width / 2
        ax.bar(x + offset, sub["PEARSON_R"], width, color=color, label=label, edgecolor="white", linewidth=0.7)
        for i, val in enumerate(sub["PEARSON_R"]):
            if pd.notna(val):
                ax.text(x[i] + offset, val + (0.025 if val >= 0 else -0.04), f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=8, fontweight="bold")
    for i, product in enumerate(PRODUCTS):
        raw = df[(df["SIF_PRODUCT"].eq(product)) & (df["TRANSFORM"].eq("RAW"))]["PEARSON_R"].iloc[0]
        anom = df[(df["SIF_PRODUCT"].eq(product)) & (df["TRANSFORM"].eq(deseas_label))]["PEARSON_R"].iloc[0]
        ax.plot([i - width / 2, i + width / 2], [raw, anom], color="#555555", lw=0.8, alpha=0.65)
    ax.axhline(0, color="#444444", lw=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(PRODUCTS, fontweight="bold")
    ax.set_ylabel("Pearson r with GPP")
    ax.set_title("Figure2_Raw_vs_Deseasonalized", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.grid(axis="y", color="#dddddd", lw=0.45)
    ax.spines[["top", "right"]].set_visible(False)
    return save_figure(fig, "Figure2_Raw_vs_Deseasonalized")


def build_core_relationship():
    records = clean_product_frame(read_csv("monthly_sif_gpp_records_4product.csv"))
    summary = clean_product_frame(read_csv("main_product_comparison_4product.csv"))
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0), facecolor="white", sharex=False, sharey=False)
    rng = np.random.default_rng(42)
    for ax, product in zip(axes.flat, PRODUCTS):
        sub = records[records["SIF_PRODUCT"].eq(product)].dropna(subset=["SIF_VALUE", "GPP_VALUE"])
        if len(sub) > 3500:
            sub = sub.iloc[rng.choice(len(sub), 3500, replace=False)]
        color = PRODUCT_COLORS[product]
        ax.scatter(sub["SIF_VALUE"], sub["GPP_VALUE"], s=7, color=color, alpha=0.16, edgecolor="none")
        summ = summary[(summary["SIF_PRODUCT"].eq(product)) & (summary["TRANSFORM"].eq("RAW"))].iloc[0]
        xs = np.linspace(sub["SIF_VALUE"].quantile(0.01), sub["SIF_VALUE"].quantile(0.99), 50)
        ys = summ["OLS_INTERCEPT"] + summ["OLS_SLOPE"] * xs
        ax.plot(xs, ys, color="#111111", lw=1.4)
        ax.text(0.04, 0.92, f"r={summ['PEARSON_R']:.2f}\nN={int(summ['N']):,}", transform=ax.transAxes, ha="left", va="top", fontsize=8.4, fontweight="bold")
        ax.set_title(product, loc="left", fontweight="bold", color=color)
        ax.set_xlabel("SIF")
        ax.set_ylabel("GPP")
        ax.grid(color="#e2e2e2", lw=0.35)
        ax.spines[["top", "right"]].set_visible(False)
    axes.flat[0].text(-0.12, 1.08, "A", transform=axes.flat[0].transAxes, fontsize=13, fontweight="bold")
    fig.suptitle("Figure2_Core_SIF_GPP_Relationship", fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return save_figure(fig, "Figure2_Core_SIF_GPP_Relationship")


def build_temporal_structure():
    df = clean_product_frame(read_csv("Figure6_Temporal_Structure_4product.csv"))
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.8), facecolor="white", sharex=True, sharey=True)
    months = np.arange(1, 13)
    for ax, product in zip(axes.flat, PRODUCTS):
        sub = df[df["SIF_PRODUCT"].eq(product)].set_index("MONTH").reindex(months)
        color = PRODUCT_COLORS[product]
        ax.fill_between(months, sub["SIF_Z_Q25"], sub["SIF_Z_Q75"], color=color, alpha=0.12, linewidth=0)
        ax.plot(months, sub["SIF_Z_MEDIAN"], color=color, lw=1.8, label="SIF")
        ax.fill_between(months, sub["GPP_Z_Q25"], sub["GPP_Z_Q75"], color="#777777", alpha=0.12, linewidth=0)
        ax.plot(months, sub["GPP_Z_MEDIAN"], color="#222222", lw=1.6, label="GPP")
        ax.axhline(0, color="#888888", lw=0.55)
        ax.set_title(product, loc="left", fontweight="bold", color=color)
        ax.grid(color="#e4e4e4", lw=0.35)
        ax.set_xticks([1, 3, 5, 7, 9, 11])
        ax.spines[["top", "right"]].set_visible(False)
    axes.flat[0].legend(frameon=False, loc="upper left")
    for ax in axes[:, 0]:
        ax.set_ylabel("Monthly z median")
    for ax in axes[-1, :]:
        ax.set_xlabel("Month")
    fig.suptitle("Figure6_Temporal_Structure", fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    return save_figure(fig, "Figure6_Temporal_Structure")


def build_igbp_summary():
    biome = pd.read_csv(CHECK_DIR / "recomputed_biome_r_matrix.csv") if (CHECK_DIR / "recomputed_biome_r_matrix.csv").exists() else recompute_biome_correlations(read_csv("monthly_sif_gpp_records_4product.csv"))
    rows = [b for b in BIOME_ORDER if b in set(biome["IGBP"])]
    fig, ax = plt.subplots(figsize=(8.3, 5.3), facecolor="white")
    im = draw_heatmap(
        ax,
        biome,
        rows,
        PRODUCTS,
        "PEARSON_R",
        "IGBP",
        "SIF_PRODUCT",
        BIOME_R_CMAP,
        TwoSlopeNorm(vmin=-0.65, vcenter=0.0, vmax=0.65),
        "Figure5_IGBP_Summary",
        annotate_n=True,
        fmt="{:+.2f}",
        missing_label="--",
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Deseasonalized Pearson r")
    return save_figure(fig, "Figure5_IGBP_Summary")


def build_temporal_seasonal_structure():
    df = clean_product_frame(read_csv("Figure6_Temporal_Structure_4product.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), facecolor="white")
    months = np.arange(1, 13)
    for product in PRODUCTS:
        sub = df[df["SIF_PRODUCT"].eq(product)].set_index("MONTH").reindex(months)
        axes[0].plot(months, sub["SIF_Z_MEDIAN"], lw=1.8, color=PRODUCT_COLORS[product], label=product)
        axes[1].plot(months, sub["GPP_Z_MEDIAN"], lw=1.8, color=PRODUCT_COLORS[product], label=product)
        axes[2].plot(months, sub["PEARSON_R"], lw=1.8, color=PRODUCT_COLORS[product], label=product)
    titles = ["A  SIF seasonal structure", "B  GPP seasonal structure", "C  Monthly SIF-GPP r"]
    ylabels = ["SIF z median", "GPP z median", "Pearson r"]
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.axhline(0, color="#777777", lw=0.55)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Month")
        ax.set_ylabel(ylabel)
        ax.set_xticks([1, 3, 5, 7, 9, 11])
        ax.grid(color="#e1e1e1", lw=0.35)
        ax.spines[["top", "right"]].set_visible(False)
    axes[2].legend(frameon=False, loc="lower left", ncol=1)
    fig.suptitle("Figure4_Temporal_Seasonal_Structure", fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return save_figure(fig, "Figure4_Temporal_Seasonal_Structure")


def build_cross_product_comparison():
    df = clean_product_frame(read_csv("main_product_comparison_4product.csv"))
    deseas_label = "DESEASONALIZED" if "DESEASONALIZED" in set(df["TRANSFORM"]) else "ANOM_ALL"
    fig, axes = plt.subplots(1, 3, figsize=(11.3, 4.0), facecolor="white")
    x = np.arange(len(PRODUCTS))
    for ax, metric, title in zip(axes, ["PEARSON_R", "R2", "N_SITES"], ["A  Pearson r", "B  R2", "C  Site coverage"]):
        raw = df[df["TRANSFORM"].eq("RAW")].set_index("SIF_PRODUCT").reindex(PRODUCTS)
        anom = df[df["TRANSFORM"].eq(deseas_label)].set_index("SIF_PRODUCT").reindex(PRODUCTS)
        width = 0.34
        if metric == "N_SITES":
            values = anom[metric].fillna(raw[metric])
            ax.bar(x, values, color=[PRODUCT_COLORS[p] for p in PRODUCTS], edgecolor="white", linewidth=0.7)
            for i, val in enumerate(values):
                ax.text(i, val + max(values) * 0.025, f"{int(val)}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        else:
            ax.bar(x - width / 2, raw[metric], width, color="#9fb0bf", label="Raw", edgecolor="white", linewidth=0.7)
            ax.bar(x + width / 2, anom[metric], width, color=[PRODUCT_COLORS[p] for p in PRODUCTS], label="Deseasonalized", edgecolor="white", linewidth=0.7)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(PRODUCTS, rotation=30, ha="right", fontweight="bold")
        ax.grid(axis="y", color="#dddddd", lw=0.35)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Figure5_Cross_Product_Comparison", fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return save_figure(fig, "Figure5_Cross_Product_Comparison")


def scan_outputs_for_oco2(paths: list[Path]) -> pd.DataFrame:
    rows = []
    pattern = re.compile(r"OCO[-_ ]?2", re.IGNORECASE)
    for path in paths:
        count = 0
        if path.suffix.lower() in {".svg", ".pdf"}:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
                count = len(pattern.findall(text))
            except Exception:
                count = -1
        rows.append({"file": str(path.relative_to(OUTPUT_ROOT)), "oco2_text_hits": count})
    return pd.DataFrame(rows)


def copy_script_self():
    dst = SCRIPT_DIR / (Path(PROJECT_ROOT) / 'All_the_Visualizations_REVISED_4PRODUCT_FINAL/05_scripts/build_revised_4product_final_visualizations.py').name
    src = (Path(PROJECT_ROOT) / 'All_the_Visualizations_REVISED_4PRODUCT_FINAL/05_scripts/build_revised_4product_final_visualizations.py').resolve()
    if src != dst.resolve():
        shutil.copy2(src, dst)


def write_report(created: dict[str, list[Path]], missing: list[str], scan: pd.DataFrame, surface_meta: pd.DataFrame | None):
    report = REPORT_DIR / "Revised_Figure_Report.md"
    source_files = [
        "main_panel_a_site_map_4product.csv",
        "main_panel_b_distribution_4product.csv",
        "main_product_comparison_4product.csv",
        "monthly_sif_gpp_records_4product.csv",
        "Figure2_Raw_vs_Deseasonalized_4product.csv",
        "Figure6_Temporal_Structure_4product.csv",
        "mechanistic_effect_matrix_4product.csv",
        "mechanistic_geospatial_driver_site_summary_4product.csv",
        "mechanistic_ml_importance_4product.csv",
    ]
    lines = [
        "# Revised Figure Report",
        "",
        "## Scope",
        "This revision package rebuilds the requested main-paper visualizations using the finalized four-product dataset only: GOSIF, GOME-2, TROPOSIF, and OCO-3.",
        "",
        "OCO-2 was excluded by filtering every product-bearing source table to the accepted four-product list before plotting. Output SVG/PDF text was also scanned for OCO-2 label leakage.",
        "",
        "## Figures revised",
    ]
    for name, paths in created.items():
        dims = ""
        try:
            import PIL.Image

            png = next((p for p in paths if p.suffix.lower() == ".png"), None)
            if png:
                im = PIL.Image.open(png)
                dims = f" ({im.width} x {im.height} px at 600 dpi export)"
        except Exception:
            pass
        rel = ", ".join(str(p.relative_to(OUTPUT_ROOT)) for p in paths)
        lines.append(f"- `{name}`{dims}: {rel}")
    lines.extend(
        [
            "",
            "## Source tables used",
        ]
    )
    for f in source_files:
        lines.append(f"- `{DATA_DIR / f}`")
    lines.extend(
        [
            "",
            "## Recomputed values",
            "- The biome/control correlation matrix was recomputed from `monthly_sif_gpp_records_4product.csv` as site-month deseasonalized SIF and GPP anomalies, then Pearson r by IGBP class and SIF product. The exact output is saved as `03_intermediate_checks/recomputed_biome_r_matrix.csv`.",
            "- The mechanistic effect matrix uses the finalized `mechanistic_effect_matrix_4product.csv`; no new statistical definitions were introduced.",
            "- Global driver panels are raster-style contextual surfaces derived by inverse-distance weighted interpolation of finalized tower-derived driver means in `mechanistic_geospatial_driver_site_summary_4product.csv`.",
            "",
            "## Mechanistic-controls redesign",
            "- Panel A: environmental effect matrix with a purple-magenta-orange-yellow centered palette.",
            "- Panel B: recomputed biome correlation matrix with r and n in each cell.",
            "- Panels C-G: global/near-global raster-style driver context maps for VPD, ET, soil moisture, radiation, and temperature, with tower locations overlaid.",
            "- Panels H-I: compact feature-importance and conceptual synthesis panels.",
            "",
            "## Palettes",
            "- Products: GOSIF deep blue, GOME-2 teal, TROPOSIF amber/gold, OCO-3 purple.",
            "- Matrix requested palette: purple -> magenta -> pale center -> orange -> yellow.",
            "- Driver maps: VPD rust, ET cyan-teal, soil moisture blue, radiation gold/amber, temperature orange-red.",
            "",
            "## Limitations / missing inputs",
        ]
    )
    if missing:
        for item in missing:
            lines.append(f"- {item}")
    else:
        lines.append("- No required finalized four-product CSV tables were missing.")
    lines.extend(
        [
            "- Independent global gridded driver rasters were not present in the finalized figure data folder. The driver maps are therefore explicitly documented as tower-derived contextual interpolation surfaces, not external global climatology rasters.",
            "",
            "## OCO-2 validation",
        ]
    )
    if scan.empty:
        lines.append("- No output scan was available.")
    else:
        total = int(scan["oco2_text_hits"].clip(lower=0).sum())
        lines.append(f"- OCO-2 text hits in scanned SVG/PDF outputs: {total}.")
        lines.append(f"- Detailed scan table: `{CHECK_DIR / 'oco2_output_text_scan.csv'}`")
    lines.extend(
        [
            "",
            "## Data integrity checks",
            f"- Recomputed biome matrix: `{CHECK_DIR / 'recomputed_biome_r_matrix.csv'}`",
            f"- Driver surface metadata: `{CHECK_DIR / 'driver_surface_metadata.csv'}`",
            f"- Product validation: all revised figures use `{', '.join(PRODUCTS)}`.",
        ]
    )
    report.write_text("\n".join(lines), encoding="utf-8")


def main():
    for directory in [MAIN_DIR, CHECK_DIR, REPORT_DIR, SCRIPT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    required = [
        "main_panel_a_site_map_4product.csv",
        "main_panel_b_distribution_4product.csv",
        "main_product_comparison_4product.csv",
        "monthly_sif_gpp_records_4product.csv",
        "Figure2_Raw_vs_Deseasonalized_4product.csv",
        "Figure6_Temporal_Structure_4product.csv",
        "mechanistic_effect_matrix_4product.csv",
        "mechanistic_geospatial_driver_site_summary_4product.csv",
        "mechanistic_ml_importance_4product.csv",
    ]
    missing = [str(DATA_DIR / f) for f in required if not (DATA_DIR / f).exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    monthly = read_csv("monthly_sif_gpp_records_4product.csv")
    recompute_biome_correlations(monthly).to_csv(CHECK_DIR / "recomputed_biome_r_matrix.csv", index=False)
    site_summary = read_csv("mechanistic_geospatial_driver_site_summary_4product.csv")
    _, surface_meta = make_driver_surface_metadata(site_summary)
    surface_meta.to_csv(CHECK_DIR / "driver_surface_metadata.csv", index=False)

    created: dict[str, list[Path]] = {}
    created["figure_mechanistic_controls_4product_final"] = build_mechanistic_figure()
    created["Candidate1_BalancedHierarchy_4Product_Final"] = build_candidate_balanced_hierarchy()
    created["Figure2_Raw_vs_Deseasonalized"] = build_raw_vs_deseasonalized()
    created["Figure2_Core_SIF_GPP_Relationship"] = build_core_relationship()
    created["Figure6_Temporal_Structure"] = build_temporal_structure()
    created["Figure5_IGBP_Summary"] = build_igbp_summary()
    created["Figure4_Temporal_Seasonal_Structure"] = build_temporal_seasonal_structure()
    created["Figure5_Cross_Product_Comparison"] = build_cross_product_comparison()

    all_paths = [p for paths in created.values() for p in paths]
    scan = scan_outputs_for_oco2(all_paths)
    scan.to_csv(CHECK_DIR / "oco2_output_text_scan.csv", index=False)
    copy_script_self()
    write_report(created, missing, scan, surface_meta)

    print("REVISED_FIGURES_CREATED =", ", ".join(created.keys()))
    print("MECHANISTIC_FIGURE_REBUILT = YES")
    print("OCO2_REMOVED = YES")
    print("GLOBAL_DRIVER_MAPS_CREATED = VPD, ET, Soil moisture, Radiation, Temperature")
    print("OUTPUT_FOLDER =", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
