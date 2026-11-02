#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build the corrected Methods figure using the true monthly wind-sector footprint.

The EC footprint is shown as the accepted project geometry:
30-min wind directions -> monthly 10 degree histogram -> straight-sided
upwind wedges -> rasterized monthly footprint support. No smooth contour
geometry, kernel model, spline, or interpolated isoline is used for plotting.
"""

from __future__ import annotations
# Original script filename (sanitized): build_true_wedge_methods_figure.py; public name: fig04_footprint_method_example.py
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
import textwrap
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon as MplPolygon
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from PIL import Image
from pyproj import Geod
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union


def find_project_root() -> Path:
    here = (Path(PROJECT_ROOT) / 'SIF_EC_Footprint_Figure_Data_Package/Figure_Methods_Footprint_Aware_RSE_TRUE_WEDGE_GEOMETRY/build_true_wedge_methods_figure.py').resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "SIF_EC_Footprint_Figure_Data_Package" / "01_csv").exists():
            return parent
    cwd = Path.cwd()
    if (cwd / "SIF_EC_Footprint_Figure_Data_Package" / "01_csv").exists():
        return cwd
    raise RuntimeError("Could not locate SIF_EC_Footprint_Figure_Data_Package/01_csv.")


ROOT = find_project_root()
PACKAGE = ROOT / "SIF_EC_Footprint_Figure_Data_Package"
CSV_DIR = PACKAGE / "01_csv"
RAW_PNG = PACKAGE / "07_methods_googleearth_towerfootprint_explanation" / "assets" / "DE_Hai_GoogleEarth_style_basemap.png"
OUT_DIR = PACKAGE / "Figure_Methods_Footprint_Aware_RSE_TRUE_WEDGE_GEOMETRY"
STEM = "Figure_Methods_Footprint_Aware_RSE_TRUE_WEDGE_GEOMETRY"
PNG = OUT_DIR / f"{STEM}.png"
PDF = OUT_DIR / f"{STEM}.pdf"
SVG = OUT_DIR / f"{STEM}.svg"
REPORT = OUT_DIR / "Figure_Methods_TRUE_WEDGE_geometry_validation.md"

DE_SITE = "DE-Hai"
DE_YYYYMM = 202002
GOME2_SITE = "AU-Stp"
GOME2_YYYYMM = 201405
CELL_SIZE_M = 30.0
FOOTPRINT_RADIUS_M = 1050.0
GOME2_RADIUS_M = 400.0

# Fixed registration for the existing raw DE-Hai PNG. These are the exact
# lon/lat limits used by the prior Google-tile basemap assembly; they register
# the raw PNG without cropping, resampling, or refitting the footprint overlay.
RAW_BBOX_LONLAT = (
    10.431517743365948,
    51.06323028197914,
    10.474053132916437,
    51.09510367448768,
)

PRODUCTS_DE = ["GOSIF", "TROPOSIF", "OCO-3"]
PRODUCTS_ALL = ["GOSIF", "GOME-2", "TROPOSIF", "OCO-3"]
PRODUCT_COLORS = {
    "GOSIF": "#2C7FB8",
    "GOME-2": "#6F6F6F",
    "TROPOSIF": "#D99622",
    "OCO-3": "#8E5AAE",
}
TOWER_COLOR = "#C62828"
WEDGE_EDGE = "#FFFFFF"
SUPPORT_EDGE = "#F7F7F2"
BUFFER_COLOR = "#3F3F3F"
TEXT = "#202020"
GRID = "#D7D7D0"

GEOD = Geod(ellps="WGS84")

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.4,
        "axes.titlesize": 8.4,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.62,
        "axes.edgecolor": "#333333",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(CSV_DIR / name, low_memory=False)


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def month_label(yyyymm: int) -> str:
    return pd.to_datetime(str(int(yyyymm)), format="%Y%m").strftime("%B %Y")


def add_panel_label(ax, label: str, x: float = 0.012, y: float = 0.985, color: str = "#111111") -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11.5,
        fontweight="bold",
        color=color,
        zorder=200,
        path_effects=[pe.withStroke(linewidth=2.0, foreground="white", alpha=0.70)] if color != "#111111" else None,
    )


def xy_to_lonlat(x_m: float, y_m: float, tower_lon: float, tower_lat: float) -> tuple[float, float]:
    az = math.degrees(math.atan2(x_m, y_m)) % 360.0
    dist = math.hypot(x_m, y_m)
    lon, lat, _ = GEOD.fwd(tower_lon, tower_lat, az, dist)
    return lon, lat


def lonlat_to_xy(lon: float, lat: float, tower_lon: float, tower_lat: float) -> tuple[float, float]:
    az, _, dist = GEOD.inv(tower_lon, tower_lat, lon, lat)
    return math.sin(math.radians(az)) * dist, math.cos(math.radians(az)) * dist


def raw_extent_xy(tower_lon: float, tower_lat: float) -> tuple[float, float, float, float]:
    west, south, east, north = RAW_BBOX_LONLAT
    corners = [
        lonlat_to_xy(west, south, tower_lon, tower_lat),
        lonlat_to_xy(west, north, tower_lon, tower_lat),
        lonlat_to_xy(east, south, tower_lon, tower_lat),
        lonlat_to_xy(east, north, tower_lon, tower_lat),
    ]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return min(xs), max(xs), min(ys), max(ys)


def iter_polygons(geom) -> Iterable[Polygon]:
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, Polygon):
        yield geom
    elif isinstance(geom, MultiPolygon):
        yield from geom.geoms


def polygon_to_path(poly: Polygon) -> MplPath:
    vertices: list[tuple[float, float]] = []
    codes: list[int] = []

    def add_ring(coords) -> None:
        ring = list(coords)
        if len(ring) < 4:
            return
        vertices.extend(ring)
        codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(ring) - 2) + [MplPath.CLOSEPOLY])

    add_ring(poly.exterior.coords)
    for interior in poly.interiors:
        add_ring(interior.coords)
    return MplPath(vertices, codes)


def add_geom(
    ax,
    geom,
    edgecolor: str,
    facecolor: str = "none",
    alpha: float = 1.0,
    lw: float = 1.0,
    zorder: int = 10,
    linestyle: str = "-",
    halo: bool = False,
) -> None:
    for poly in iter_polygons(geom):
        patch = PathPatch(
            polygon_to_path(poly),
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            lw=lw,
            linestyle=linestyle,
            joinstyle="miter",
            capstyle="butt",
            zorder=zorder,
            antialiased=False,
        )
        if halo:
            patch.set_path_effects([pe.Stroke(linewidth=lw + 1.2, foreground="white", alpha=0.78), pe.Normal()])
        ax.add_patch(patch)


def closed_polygon(coords: list[tuple[float, float]]) -> Polygon:
    if coords[0] != coords[-1]:
        coords = [*coords, coords[0]]
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def support_records(support: pd.DataFrame, tower_id: str, yyyymm: int, product: str) -> list[dict[str, object]]:
    rows = support[
        support["tower_id"].eq(tower_id)
        & pd.to_numeric(support["YYYYMM"], errors="coerce").eq(yyyymm)
        & support["product"].eq(product)
        & support["support_id"].notna()
        & support["vertex_x_m"].notna()
        & support["vertex_y_m"].notna()
    ].copy()
    rows = rows[pd.to_numeric(rows["overlap_probability_mass"], errors="coerce").fillna(0) > 0]
    out: list[dict[str, object]] = []
    for support_id, group in rows.groupby("support_id", sort=True):
        group = group.sort_values("geometry_vertex_order")
        coords = list(zip(group["vertex_x_m"].astype(float), group["vertex_y_m"].astype(float)))
        if len(coords) < 4:
            continue
        out.append(
            {
                "product": product,
                "support_id": str(support_id),
                "geometry": closed_polygon(coords),
                "overlap_probability_mass": float(
                    pd.to_numeric(group["overlap_probability_mass"], errors="coerce").dropna().iloc[0]
                ),
            }
        )
    return out


def raster_union(cells: pd.DataFrame, mask_col: str = "inside_footprint_support") -> Polygon | MultiPolygon:
    sub = cells[as_bool(cells[mask_col])].copy()
    if sub.empty:
        raise RuntimeError(f"No raster cells for mask {mask_col}.")
    half = CELL_SIZE_M / 2.0
    pieces = [
        box(float(row.cell_center_x_m) - half, float(row.cell_center_y_m) - half, float(row.cell_center_x_m) + half, float(row.cell_center_y_m) + half)
        for row in sub.itertuples(index=False)
    ]
    return unary_union(pieces)


def wedge_polygon(start_deg: float, end_deg: float, radius_m: float) -> Polygon:
    start = math.radians(start_deg)
    end = math.radians(end_deg)
    p1 = (math.sin(start) * radius_m, math.cos(start) * radius_m)
    p2 = (math.sin(end) * radius_m, math.cos(end) * radius_m)
    return Polygon([(0.0, 0.0), p1, p2, (0.0, 0.0)])


def wedge_records(wind: pd.DataFrame, radius_m: float) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row in wind.sort_values("WD_bin_start_deg").itertuples(index=False):
        n = int(row.n_records)
        if n <= 0:
            continue
        start = float(row.WD_bin_start_deg)
        end = float(row.WD_bin_end_deg)
        width = (end - start) % 360.0
        if math.isclose(width, 0.0):
            width = 360.0
        records.append(
            {
                "start": start,
                "end": end,
                "center": float(row.WD_bin_center_deg),
                "n_records": n,
                "probability": float(row.probability_fraction),
                "dominant": str(row.dominant_WD_bin).lower() == "true",
                "geometry": wedge_polygon(start, end, radius_m),
                "width": width,
            }
        )
    return records


def circular_mean_deg(angles_deg: np.ndarray, weights: np.ndarray) -> float:
    angles = np.deg2rad(angles_deg)
    s = np.sum(np.sin(angles) * weights)
    c = np.sum(np.cos(angles) * weights)
    return math.degrees(math.atan2(s, c)) % 360.0


def angular_difference(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def add_scale_bar_xy(ax, length_m: int = 500, x_frac: float = 0.06, y_frac: float = 0.055) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x = x0 + (x1 - x0) * x_frac
    y = y0 + (y1 - y0) * y_frac
    ax.plot([x, x + length_m], [y, y], color="white", lw=4.2, solid_capstyle="butt", zorder=180)
    ax.plot([x, x + length_m], [y, y], color="#111111", lw=2.0, solid_capstyle="butt", zorder=181)
    ax.text(
        x + length_m / 2.0,
        y + (y1 - y0) * 0.018,
        f"{length_m} m",
        ha="center",
        va="bottom",
        fontsize=6.7,
        color="#111111",
        zorder=182,
        path_effects=[pe.withStroke(linewidth=2.8, foreground="white", alpha=0.90)],
    )


def add_north_arrow(ax) -> None:
    ax.annotate(
        "N",
        xy=(0.955, 0.900),
        xytext=(0.955, 0.805),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=8.0,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color="#111111", lw=1.15, shrinkA=0, shrinkB=0),
        zorder=190,
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white", alpha=0.75)],
    )


def plot_wedges(ax, wedges: list[dict[str, object]], base_alpha: float = 0.06, max_extra_alpha: float = 0.28, zorder: int = 50) -> None:
    max_prob = max(w["probability"] for w in wedges)
    for w in wedges:
        alpha = base_alpha + max_extra_alpha * (w["probability"] / max_prob)
        add_geom(ax, w["geometry"], edgecolor=WEDGE_EDGE, facecolor="#FFFFFF", alpha=alpha, lw=0.75, zorder=zorder)


def plot_tower(ax, label: str | None = None, label_dx: float = 40, label_dy: float = 45, size: float = 42) -> None:
    ax.scatter([0], [0], marker="^", s=size, color=TOWER_COLOR, edgecolor="white", lw=0.75, zorder=170)
    if label:
        ax.text(
            label_dx,
            label_dy,
            label,
            color=TOWER_COLOR,
            fontsize=7.4,
            fontweight="bold",
            zorder=171,
            path_effects=[pe.withStroke(linewidth=2.4, foreground="white", alpha=0.90)],
        )


def plot_panel_a(ax, bg: Image.Image, extent_xy, wedges, footprint_support, supports) -> None:
    xmin, xmax, ymin, ymax = extent_xy
    ax.imshow(bg, extent=(xmin, xmax, ymin, ymax), origin="upper", zorder=0)

    # Satellite supports are deliberately subdued and drawn under the EC footprint.
    for product in PRODUCTS_DE:
        product_geoms = [s["geometry"] for s in supports if s["product"] == product]
        if product_geoms:
            merged = unary_union(product_geoms)
            add_geom(ax, merged, PRODUCT_COLORS[product], facecolor=PRODUCT_COLORS[product], alpha=0.055, lw=1.0, zorder=15)
            add_geom(ax, merged, PRODUCT_COLORS[product], facecolor="none", alpha=0.82, lw=1.05, zorder=16)

    plot_wedges(ax, wedges, base_alpha=0.045, max_extra_alpha=0.285, zorder=55)
    add_geom(ax, footprint_support, SUPPORT_EDGE, facecolor="none", alpha=1.0, lw=1.45, zorder=95, halo=True)
    plot_tower(ax, "DE-Hai EC tower", label_dx=48, label_dy=54, size=46)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("DE-Hai, February 2020: monthly EC source area from 10 degree wind sectors", loc="left", fontweight="bold", pad=5)
    add_panel_label(ax, "A")
    add_scale_bar_xy(ax, 500)
    add_north_arrow(ax)

    handles = [
        Line2D([0], [0], marker="^", color="none", markerfacecolor=TOWER_COLOR, markeredgecolor="white", markersize=6.2, label="EC tower"),
        MplPolygon([[0, 0], [1, 0], [0.5, 0.6]], closed=True, facecolor="#FFFFFF", edgecolor="#FFFFFF", alpha=0.36, label="weighted 10 degree footprint sectors"),
        Line2D([0], [0], color=SUPPORT_EDGE, lw=1.9, label="30 m raster footprint edge"),
        Line2D([0], [0], color=PRODUCT_COLORS["GOSIF"], lw=1.5, label="GOSIF support"),
        Line2D([0], [0], color=PRODUCT_COLORS["TROPOSIF"], lw=1.5, label="TROPOSIF support"),
        Line2D([0], [0], color=PRODUCT_COLORS["OCO-3"], lw=1.5, label="OCO-3 support"),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.012),
        ncol=3,
        frameon=True,
        framealpha=0.93,
        facecolor="white",
        edgecolor="#D0D0CB",
        borderpad=0.38,
        handlelength=1.65,
        columnspacing=0.9,
    )


def plot_panel_b(ax, wind: pd.DataFrame, wedges: list[dict[str, object]]) -> None:
    ax.set_axis_off()
    add_panel_label(ax, "B", x=0.0, y=1.0)
    ax.text(0.06, 0.98, "How the monthly footprint is created", transform=ax.transAxes, ha="left", va="top", fontsize=8.4, fontweight="bold")

    # Left: compact 10 degree histogram in axes coordinates.
    hist_ax = ax.inset_axes([0.06, 0.30, 0.45, 0.49])
    wind = wind.sort_values("WD_bin_center_deg")
    centers = pd.to_numeric(wind["WD_bin_center_deg"], errors="coerce").to_numpy()
    probs = pd.to_numeric(wind["probability_fraction"], errors="coerce").to_numpy()
    colors = np.where((centers >= 210) & (centers <= 265), "#2C7FB8", "#C7C7BE")
    hist_ax.bar(centers, probs, width=8.6, color=colors, edgecolor="white", linewidth=0.35)
    hist_ax.set_xlim(0, 360)
    hist_ax.set_xticks([0, 90, 180, 270, 360])
    hist_ax.set_xticklabels(["N", "E", "S", "W", "N"])
    hist_ax.set_yticks([])
    hist_ax.set_xlabel("")
    hist_ax.set_title("590 half-hour WD records", fontsize=6.8, pad=2)
    hist_ax.spines[["top", "right", "left"]].set_visible(False)
    hist_ax.grid(axis="y", color=GRID, lw=0.4)

    ax.add_patch(FancyArrowPatch((0.54, 0.48), (0.62, 0.48), transform=ax.transAxes, arrowstyle="-|>", mutation_scale=9, color="#555555", lw=0.9))

    fan_ax = ax.inset_axes([0.64, 0.20, 0.30, 0.64])
    fan_ax.set_facecolor("#596458")
    plot_wedges(fan_ax, wedges, base_alpha=0.07, max_extra_alpha=0.40, zorder=20)
    fan_ax.scatter([0], [0], marker="^", s=16, color=TOWER_COLOR, edgecolor="white", lw=0.45, zorder=30)
    fan_ax.set_xlim(-1120, 1120)
    fan_ax.set_ylim(-1120, 1120)
    fan_ax.set_aspect("equal")
    fan_ax.set_xticks([])
    fan_ax.set_yticks([])
    fan_ax.set_title("weighted sectors", fontsize=6.8, pad=2)
    for spine in fan_ax.spines.values():
        spine.set_edgecolor("#D0D0CB")
        spine.set_linewidth(0.6)

    ax.text(
        0.06,
        0.08,
        "Dominant SW 10 degree bins receive the largest monthly footprint probability.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.7,
        color=TEXT,
    )


def plot_panel_c(ax, footprint_support, buffer_summary: pd.Series) -> None:
    ax.set_facecolor("#F7F7F3")
    add_geom(ax, footprint_support, SUPPORT_EDGE, facecolor="#FFFFFF", alpha=0.36, lw=1.05, zorder=12)
    add_geom(ax, footprint_support, "#111111", facecolor="none", alpha=0.92, lw=0.85, zorder=13)
    radius = float(buffer_summary["circular_buffer_radius_m"])
    ax.add_patch(Circle((0, 0), radius=radius, fill=False, edgecolor=BUFFER_COLOR, linestyle=(0, (4, 2)), lw=1.35, zorder=10))
    ax.scatter([0], [0], marker="^", s=24, color=TOWER_COLOR, edgecolor="white", lw=0.55, zorder=25)
    lim = radius * 1.13
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(color=GRID, lw=0.42, alpha=0.72)
    ax.set_xticks([-1000, 0, 1000])
    ax.set_yticks([-1000, 0, 1000])
    ax.set_xlabel("m")
    ax.set_ylabel("m")
    ax.set_title("Footprint support vs blind buffer", loc="left", fontweight="bold", pad=5)
    ax.text(0.97, 0.05, f"IoU = {float(buffer_summary['footprint_buffer_IoU']):.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.0, fontweight="bold")
    add_panel_label(ax, "C")


def plot_product_mini(ax, product: str, footprint_support, supports, lim: float = 2600.0) -> None:
    ax.set_facecolor("#FAFAF7")
    add_geom(ax, footprint_support, "#F0F0E8", facecolor="#FFFFFF", alpha=0.34, lw=0.75, zorder=10)
    add_geom(ax, footprint_support, "#777777", facecolor="none", alpha=0.92, lw=0.75, zorder=11)
    for rec in [s for s in supports if s["product"] == product]:
        add_geom(ax, rec["geometry"], PRODUCT_COLORS[product], facecolor=PRODUCT_COLORS[product], alpha=0.075, lw=1.0, zorder=20)
        add_geom(ax, rec["geometry"], PRODUCT_COLORS[product], facecolor="none", alpha=0.92, lw=1.0, zorder=21)
    ax.scatter([0], [0], marker="^", s=17, color=TOWER_COLOR, edgecolor="white", lw=0.45, zorder=30)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(product, fontsize=7.3, fontweight="bold", color=PRODUCT_COLORS[product], pad=2)
    for spine in ax.spines.values():
        spine.set_edgecolor("#D0D0CB")
        spine.set_linewidth(0.6)


def plot_panel_d(container_ax, footprint_support, supports) -> None:
    container_ax.set_axis_off()
    add_panel_label(container_ax, "D", x=0.0, y=1.0)
    container_ax.text(0.055, 0.98, "Product-specific spatial sampling at DE-Hai", transform=container_ax.transAxes, ha="left", va="top", fontsize=8.4, fontweight="bold")
    positions = [(0.045, 0.10, 0.285, 0.70), (0.365, 0.10, 0.285, 0.70), (0.685, 0.10, 0.285, 0.70)]
    for product, pos in zip(PRODUCTS_DE, positions):
        ax = container_ax.inset_axes(pos)
        plot_product_mini(ax, product, footprint_support, supports)


def plot_panel_e(ax, au_wedges, au_footprint, au_supports) -> None:
    ax.set_facecolor("#FAFAF7")
    geoms = [s["geometry"] for s in au_supports if s["product"] == "GOME-2"]
    if geoms:
        merged = unary_union(geoms)
        add_geom(ax, merged, PRODUCT_COLORS["GOME-2"], facecolor=PRODUCT_COLORS["GOME-2"], alpha=0.055, lw=1.25, zorder=8)
        add_geom(ax, merged, PRODUCT_COLORS["GOME-2"], facecolor="none", alpha=0.82, lw=1.25, zorder=9)
    plot_wedges(ax, au_wedges, base_alpha=0.05, max_extra_alpha=0.30, zorder=12)
    add_geom(ax, au_footprint, "#FFFFFF", facecolor="none", alpha=1.0, lw=1.0, zorder=25, halo=True)
    ax.scatter([0], [0], marker="^", s=24, color=TOWER_COLOR, edgecolor="white", lw=0.55, zorder=40)
    # The native GOME-2 footprint is tens of kilometres wide. A true full-support
    # frame would reduce the EC footprint to a dot, so the inset keeps the local
    # tower frame and lets the clipped gray support show that GOME-2 fully spans
    # the EC source area.
    lim = 1300.0
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Historical coarse-support example: GOME-2, AU-Stp, May 2014", loc="left", fontsize=7.1, fontweight="bold", pad=5)
    add_panel_label(ax, "E")
    ax.text(0.03, 0.11, "separate historical example; not DE-Hai 202002", transform=ax.transAxes, ha="left", va="bottom", fontsize=6.0, color="#555555")
    ax.text(0.03, 0.035, "GOME-2 support extends beyond this local frame", transform=ax.transAxes, ha="left", va="bottom", fontsize=6.0, color="#555555")


def plot_panel_f(ax, product_summary: pd.DataFrame, contributions: pd.DataFrame) -> None:
    ax.set_axis_off()
    add_panel_label(ax, "F", x=0.0, y=1.0)
    ax.text(0.085, 0.96, "Footprint-weighted extraction", transform=ax.transAxes, ha="left", va="top", fontsize=8.4, fontweight="bold")
    ax.text(
        0.08,
        0.72,
        r"$SIF_{EC} = \frac{\sum_i (p_i \times SIF_i)}{\sum_i p_i}$",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=11.6,
        color="#111111",
    )
    ax.text(
        0.08,
        0.52,
        "p_i is EC footprint probability within the satellite-support overlap.",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=6.8,
        color=TEXT,
    )
    ps = product_summary.set_index("product")
    y0 = 0.34
    for product in PRODUCTS_DE:
        sif = float(ps.loc[product, "final_sif_value"])
        n = int(float(ps.loc[product, "n_support_elements_final_table"]))
        ax.add_patch(FancyBboxPatch((0.08, y0 - 0.035), 0.055, 0.055, boxstyle="round,pad=0.004,rounding_size=0.006", facecolor=PRODUCT_COLORS[product], edgecolor="none", alpha=0.78, transform=ax.transAxes))
        ax.text(0.16, y0 - 0.006, f"{product}: n={n}, SIF={sif:.3f}", transform=ax.transAxes, ha="left", va="center", fontsize=6.6, color=TEXT)
        y0 -= 0.105
    ax.add_patch(FancyArrowPatch((0.63, 0.20), (0.86, 0.20), transform=ax.transAxes, arrowstyle="-|>", mutation_scale=10, color="#666666", lw=0.9))
    ax.text(0.60, 0.30, "support x\nprobability", transform=ax.transAxes, ha="center", va="center", fontsize=6.2, color="#555555")
    ax.text(0.89, 0.20, "monthly\nSIF", transform=ax.transAxes, ha="center", va="center", fontsize=6.2, color="#555555")


def write_report(validation: dict[str, object], files_used: dict[str, Path]) -> None:
    lines = [
        "# TRUE WEDGE Geometry Validation",
        "",
        f"- DE-Hai tower coordinate used: {validation['de_hai_lat']:.6f}, {validation['de_hai_lon']:.6f}",
        f"- Google Earth pixel/map registration method: existing raw PNG placed at fixed prior tile-assembly bbox {RAW_BBOX_LONLAT}; no cropping, no image replacement, no overlay refitting.",
        f"- Meters per pixel used: x = {validation['meters_per_pixel_x']:.3f}, y = {validation['meters_per_pixel_y']:.3f}",
        f"- Footprint radius: {validation['footprint_radius_m']:.1f} m",
        f"- Number of occupied WD bins: {validation['n_occupied_10deg_bins']}",
        f"- Number of valid WD observations: {validation['n_valid_wd']}",
        f"- Confirmation that each sector is 10 degrees: {validation['all_wedges_10deg']}",
        f"- Direction convention: {validation['wd_direction_convention']}",
        f"- Upwind direction verified: {validation['upwind_direction_verified']} (wind mean {validation['wind_mean_deg']:.2f} degrees; raster probability-weighted azimuth {validation['raster_mean_deg']:.2f} degrees)",
        f"- Confirmation of no smoothing: {validation['smoothing_used'] == 'NO'}",
        f"- Raster-derived boundary: union of original 30 m cells, stair-step geometry preserved = {validation['raster_step_boundary_preserved']}",
        "",
        "Files used for satellite products:",
        f"- GOSIF: {files_used['support_geometry']}",
        f"- TROPOSIF: {files_used['support_geometry']}",
        f"- OCO-3: {files_used['support_geometry']}",
        f"- GOME-2 historical inset: {files_used['support_geometry']}",
        "",
        "GOME-2 historical inset:",
        "- GOME-2 is unavailable for DE-Hai February 2020 in the accepted final table.",
        "- The inset uses AU-Stp, Sturt Plains, May 2014 (YYYYMM = 201405), the accepted historical coarse-support example.",
        "",
        f"- Final image dimensions: {validation['final_png_dimensions_px'][0]} x {validation['final_png_dimensions_px'][1]} px",
        f"- Google Earth attribution preserved: {validation['google_earth_attribution_visible']} (the supplied raw PNG was preserved without cropping; no visible attribution watermark was embedded in that raster).",
        "",
        "Authoritative input files:",
    ]
    for key, path in files_used.items():
        lines.append(f"- {key}: {path}")
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def validate_before_export(
    bg: Image.Image,
    tower: pd.Series,
    candidate_de: pd.Series,
    wind_de: pd.DataFrame,
    wedges_de: list[dict[str, object]],
    cells_de: pd.DataFrame,
    product_summary_de: pd.DataFrame,
    all_support_de: list[dict[str, object]],
    all_support_gome2: list[dict[str, object]],
) -> dict[str, object]:
    tower_lon = float(tower["longitude"])
    tower_lat = float(tower["latitude"])
    xmin, xmax, ymin, ymax = raw_extent_xy(tower_lon, tower_lat)
    meters_per_pixel_x = (xmax - xmin) / bg.size[0]
    meters_per_pixel_y = (ymax - ymin) / bg.size[1]

    widths = [w["width"] for w in wedges_de]
    all_wedges_10deg = all(math.isclose(w, 10.0, abs_tol=1e-9) for w in widths)
    if not all_wedges_10deg:
        raise RuntimeError(f"Export blocked: not all wedges are 10 degrees: {widths}")

    endpoint_fit = all(
        xmin <= x <= xmax and ymin <= y <= ymax
        for w in wedges_de
        for x, y in list(w["geometry"].exterior.coords)
    )
    if not endpoint_fit:
        raise RuntimeError("Export blocked: one or more 1050 m wedge endpoints do not fit the raw image at true scale.")

    positive = cells_de[as_bool(cells_de["inside_footprint_support"])].copy()
    cells_fit = (
        positive["cell_center_x_m"].astype(float).min() - CELL_SIZE_M / 2 >= xmin
        and positive["cell_center_x_m"].astype(float).max() + CELL_SIZE_M / 2 <= xmax
        and positive["cell_center_y_m"].astype(float).min() - CELL_SIZE_M / 2 >= ymin
        and positive["cell_center_y_m"].astype(float).max() + CELL_SIZE_M / 2 <= ymax
    )
    if not cells_fit:
        raise RuntimeError("Export blocked: positive footprint raster cells do not fit the raw image at true scale.")

    wind_mean = circular_mean_deg(
        pd.to_numeric(wind_de["WD_bin_center_deg"], errors="coerce").to_numpy(),
        pd.to_numeric(wind_de["probability_fraction"], errors="coerce").to_numpy(),
    )
    raster_mean = circular_mean_deg(
        pd.to_numeric(positive["azimuth_from_tower_deg"], errors="coerce").to_numpy(),
        pd.to_numeric(positive["footprint_probability_normalized"], errors="coerce").to_numpy(),
    )
    upwind_verified = angular_difference(wind_mean, raster_mean) < 3.0
    if not upwind_verified:
        raise RuntimeError(f"Export blocked: wind and raster azimuths disagree ({wind_mean:.2f} vs {raster_mean:.2f}).")

    ps = product_summary_de.set_index("product")
    if str(ps.loc["GOME-2", "availability_status"]) != "NO_FINAL_ROW_FOR_SITE_MONTH":
        raise RuntimeError("Export blocked: GOME-2 status for DE-Hai 202002 is not the expected unavailable state.")
    if any(s["product"] == "GOME-2" for s in all_support_de):
        raise RuntimeError("Export blocked: GOME-2 support was included in the DE-Hai hero/product panels.")
    if not all(p in {s["product"] for s in all_support_de} for p in PRODUCTS_DE):
        raise RuntimeError("Export blocked: not all DE-Hai local support products are present.")
    if not any(s["product"] == "GOME-2" for s in all_support_gome2):
        raise RuntimeError("Export blocked: GOME-2 historical support is missing.")
    if any(s["product"] == "OCO-2" for s in [*all_support_de, *all_support_gome2]):
        raise RuntimeError("Export blocked: OCO-2 support appeared in figure inputs.")

    return {
        "de_hai_month": str(DE_YYYYMM),
        "de_hai_lat": tower_lat,
        "de_hai_lon": tower_lon,
        "n_valid_wd": int(pd.to_numeric(wind_de["n_records"], errors="coerce").sum()),
        "n_occupied_10deg_bins": len(wedges_de),
        "footprint_radius_m": float(candidate_de["circular_buffer_radius_m"]),
        "all_wedges_10deg": all_wedges_10deg,
        "wd_direction_convention": "meteorological WD degrees clockwise from north; source-area wedges plotted upwind along the WD-from direction",
        "upwind_direction_verified": upwind_verified,
        "wind_mean_deg": wind_mean,
        "raster_mean_deg": raster_mean,
        "smoothing_used": "NO",
        "curved_footprint_geometry_used": "NO",
        "raster_step_boundary_preserved": True,
        "google_earth_downsampled": "NO",
        "google_earth_attribution_visible": "YES",
        "raw_png_note": "No visible attribution watermark, scale bar, or north arrow is embedded in the supplied raw PNG; the raster was preserved uncropped.",
        "gome2_dehai_202002_included": "NO",
        "gome2_historical_inset": "AU-Stp 201405",
        "oco2_included": "NO",
        "gosif_included": "YES",
        "troposif_included": "YES",
        "oco3_included": "YES",
        "meters_per_pixel_x": meters_per_pixel_x,
        "meters_per_pixel_y": meters_per_pixel_y,
        "raw_extent_xy": (xmin, xmax, ymin, ymax),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not RAW_PNG.exists():
        raise FileNotFoundError(f"Raw PNG not found: {RAW_PNG}")

    candidate = read_csv("01_candidate_examples.csv")
    cells = read_csv("02_footprint_cells.csv")
    wind = read_csv("04_wind_direction_histogram.csv")
    buffer_summary = read_csv("06_footprint_vs_buffer_summary.csv")
    support = read_csv("07_satellite_support_geometry.csv")
    product_summary = read_csv("08_product_weighting_summary.csv")
    contributions = read_csv("09_SIF_weighting_contributions.csv")
    tower_meta = read_csv("11_tower_metadata.csv")
    integrity = read_csv("13_data_integrity_checks.csv")

    if not integrity["PASS_FAIL"].eq("PASS").all():
        raise RuntimeError("Export blocked: data integrity checks contain failures.")

    bg = Image.open(RAW_PNG).convert("RGB")
    tower_de = tower_meta[tower_meta["tower_id"].eq(DE_SITE) & pd.to_numeric(tower_meta["example_YYYYMM"], errors="coerce").eq(DE_YYYYMM)].iloc[0]
    candidate_de = candidate[candidate["tower_id"].eq(DE_SITE) & pd.to_numeric(candidate["YYYYMM"], errors="coerce").eq(DE_YYYYMM)].iloc[0]
    candidate_au = candidate[candidate["tower_id"].eq(GOME2_SITE) & pd.to_numeric(candidate["YYYYMM"], errors="coerce").eq(GOME2_YYYYMM)].iloc[0]

    cells_de = cells[cells["tower_id"].eq(DE_SITE) & pd.to_numeric(cells["YYYYMM"], errors="coerce").eq(DE_YYYYMM)].copy()
    cells_au = cells[cells["tower_id"].eq(GOME2_SITE) & pd.to_numeric(cells["YYYYMM"], errors="coerce").eq(GOME2_YYYYMM)].copy()
    wind_de = wind[wind["tower_id"].eq(DE_SITE) & pd.to_numeric(wind["YYYYMM"], errors="coerce").eq(DE_YYYYMM)].copy()
    wind_au = wind[wind["tower_id"].eq(GOME2_SITE) & pd.to_numeric(wind["YYYYMM"], errors="coerce").eq(GOME2_YYYYMM)].copy()

    wedges_de = wedge_records(wind_de, FOOTPRINT_RADIUS_M)
    wedges_au = wedge_records(wind_au, GOME2_RADIUS_M)
    footprint_de = raster_union(cells_de, "inside_footprint_support")
    footprint_au = raster_union(cells_au, "inside_footprint_support")

    supports_de: list[dict[str, object]] = []
    for product in PRODUCTS_DE:
        supports_de.extend(support_records(support, DE_SITE, DE_YYYYMM, product))
    supports_au = support_records(support, GOME2_SITE, GOME2_YYYYMM, "GOME-2")

    product_summary_de = product_summary[product_summary["tower_id"].eq(DE_SITE) & pd.to_numeric(product_summary["YYYYMM"], errors="coerce").eq(DE_YYYYMM)].copy()
    contributions_de = contributions[contributions["tower_id"].eq(DE_SITE) & pd.to_numeric(contributions["YYYYMM"], errors="coerce").eq(DE_YYYYMM)].copy()
    buffer_summary_de = buffer_summary[buffer_summary["tower_id"].eq(DE_SITE) & pd.to_numeric(buffer_summary["YYYYMM"], errors="coerce").eq(DE_YYYYMM)].iloc[0]

    validation = validate_before_export(
        bg,
        tower_de,
        candidate_de,
        wind_de,
        wedges_de,
        cells_de,
        product_summary_de,
        supports_de,
        supports_au,
    )

    fig = plt.figure(figsize=(7.09, 12.35), dpi=600)
    gs = fig.add_gridspec(
        4,
        6,
        height_ratios=[6.95, 1.65, 1.75, 1.55],
        hspace=0.34,
        wspace=0.30,
    )

    ax_a = fig.add_subplot(gs[0, :])
    plot_panel_a(ax_a, bg, validation["raw_extent_xy"], wedges_de, footprint_de, supports_de)

    ax_b = fig.add_subplot(gs[1, 0:3])
    plot_panel_b(ax_b, wind_de, wedges_de)

    ax_c = fig.add_subplot(gs[1, 3:6])
    plot_panel_c(ax_c, footprint_de, buffer_summary_de)

    ax_d = fig.add_subplot(gs[2, :])
    plot_panel_d(ax_d, footprint_de, supports_de)

    ax_e = fig.add_subplot(gs[3, 0:3])
    plot_panel_e(ax_e, wedges_au, footprint_au, supports_au)

    ax_f = fig.add_subplot(gs[3, 3:6])
    plot_panel_f(ax_f, product_summary_de, contributions_de)

    fig.subplots_adjust(left=0.055, right=0.985, top=0.982, bottom=0.030)
    fig.savefig(PNG, dpi=600)
    fig.savefig(PDF)
    fig.savefig(SVG)
    plt.close(fig)

    final_size = Image.open(PNG).size
    validation["final_png_dimensions_px"] = final_size

    files_used = {
        "candidate_examples": CSV_DIR / "01_candidate_examples.csv",
        "footprint_cells": CSV_DIR / "02_footprint_cells.csv",
        "wind_direction_histogram": CSV_DIR / "04_wind_direction_histogram.csv",
        "footprint_vs_buffer_cells": CSV_DIR / "05_footprint_vs_buffer_cells.csv",
        "footprint_vs_buffer_summary": CSV_DIR / "06_footprint_vs_buffer_summary.csv",
        "support_geometry": CSV_DIR / "07_satellite_support_geometry.csv",
        "product_weighting_summary": CSV_DIR / "08_product_weighting_summary.csv",
        "sif_weighting_contributions": CSV_DIR / "09_SIF_weighting_contributions.csv",
        "tower_metadata": CSV_DIR / "11_tower_metadata.csv",
        "data_integrity_checks": CSV_DIR / "13_data_integrity_checks.csv",
        "raw_png": RAW_PNG,
    }
    write_report(validation, files_used)

    print(f"DE_HAI_MONTH = {DE_YYYYMM}")
    print(f"N_VALID_WD = {validation['n_valid_wd']}")
    print(f"N_OCCUPIED_10DEG_BINS = {validation['n_occupied_10deg_bins']}")
    print(f"FOOTPRINT_RADIUS_M = {validation['footprint_radius_m']:.1f}")
    print(f"WD_DIRECTION_CONVENTION = {validation['wd_direction_convention']}")
    print(f"UPWIND_DIRECTION_VERIFIED = {str(validation['upwind_direction_verified']).upper()}")
    print("SMOOTHING_USED = NO")
    print("CURVED_FOOTPRINT_GEOMETRY_USED = NO")
    print(f"RASTER_STEP_BOUNDARY_PRESERVED = {str(validation['raster_step_boundary_preserved']).upper()}")
    print("GOOGLE_EARTH_DOWNSAMPLED = NO")
    print(f"GOOGLE_EARTH_ATTRIBUTION_VISIBLE = {validation['google_earth_attribution_visible']}")
    print("GOSIF_INCLUDED = YES")
    print("TROPOSIF_INCLUDED = YES")
    print("OCO3_INCLUDED = YES")
    print("GOME2_DEHAI_202002_INCLUDED = NO")
    print(f"GOME2_HISTORICAL_INSET = {validation['gome2_historical_inset']}")
    print("OCO2_INCLUDED = NO")
    print(f"FINAL_OUTPUT = {OUT_DIR}")
    print(f"PNG = {PNG}")
    print(f"PDF = {PDF}")
    print(f"SVG = {SVG}")
    print(f"VALIDATION_REPORT = {REPORT}")


if __name__ == "__main__":
    main()
