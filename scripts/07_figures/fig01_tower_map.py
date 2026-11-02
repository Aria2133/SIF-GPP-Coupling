"""Figure 1: global map of the eddy-covariance tower network with ecosystem classes and regional enlargements.

Input: cleaned tower metadata table; optional raster basemap. Output: map figure files."""
from __future__ import annotations
# Original script filename (sanitized): Figure_1_Global_EC_Tower_Map_final_highres.py; public name: fig01_tower_map.py
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
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform_bounds


PROJECT_ROOT = Path(
    PROJECT_ROOT
)
HANDOFF_CSV = PROJECT_ROOT / "Figure_1_Handoff" / "EC_Towers_Final.csv"
NASA_BASEMAP_TIF = PROJECT_ROOT / "world.topo.bathy.200408.3x21600x10800_geo.tif"
OUTPUT_DIR = PROJECT_ROOT / "Figure_1_Global_EC_Tower_Map"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# NOTE: this script originally wrote the Figure_1_Global_EC_Tower_Map_final.* files.
# Those are now a frozen, protected snapshot (label placement had known overlap/crossing
# bugs) and must never be overwritten. This script has since been corrected (collision-
# aware label placement -- see annotate_sites) and now writes the "_final_v2" outputs
# instead, produced only from a corrected/label_fix_preview render, never in-place over
# the original.
PNG_OUT = OUTPUT_DIR / "Figure_1_Global_EC_Tower_Map_final_v2.png"
PDF_OUT = OUTPUT_DIR / "Figure_1_Global_EC_Tower_Map_final_v2.pdf"
TIF_500_OUT = OUTPUT_DIR / "Figure_1_Global_EC_Tower_Map_final_v2.tif"
NOTES_OUT = OUTPUT_DIR / "Figure_1_Global_EC_Tower_Map_final_v2_export_notes.md"
MASTER_WIDTH_PX = 10000
MASTER_HEIGHT_PX = 5625
MASTER_DPI = 500

# Locked palette from the accepted Figure 1 handoff/style.
GROUP_COLORS = {
    "Forest": "#1b9e77",
    "Shrubland": "#d95f02",
    "Savanna / woody savanna": "#e6ab02",
    "Grassland": "#66a61e",
    "Cropland": "#7570b3",
    "Unknown": "#666666",
}
GROUP_ORDER = [
    "Forest",
    "Shrubland",
    "Savanna / woody savanna",
    "Grassland",
    "Cropland",
    "Unknown",
]

REGIONS = {
    "North America": (-130, -65, 20, 56),
    "Europe": (-12, 35, 30, 64),
    "Australia": (108, 154, -47, -8),
    "East Asia": (96, 146, 8, 52),
}


def load_sites() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(HANDOFF_CSV)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["IGBP_group"] = df["IGBP_group"].where(df["IGBP_group"].isin(GROUP_COLORS), "Unknown")
    valid = df[df["coordinate_status"].eq("VALID") & df["latitude"].notna() & df["longitude"].notna()].copy()
    missing = df[~df.index.isin(valid.index)].copy()
    return valid, missing


def cartopy_to_rasterio_crs(cartopy_crs: ccrs.Projection) -> CRS:
    return CRS.from_user_input(cartopy_crs.proj4_init)


def target_bounds_for_extent(
    extent: tuple[float, float, float, float],
    dst_crs: CRS,
) -> tuple[float, float, float, float]:
    left, right, bottom, top = extent
    return transform_bounds(
        "EPSG:4326",
        dst_crs,
        left,
        bottom,
        right,
        top,
        densify_pts=81,
    )


def darken_nasa_rgb(arr: np.ndarray) -> np.ndarray:
    """Apply the locked dark visual character to the true NASA RGB raster."""
    rgb = arr.astype(np.float32) / 255.0
    # Same broad visual character as the accepted figure: dark oceans, rich land,
    # crisp contrast, but no change to tower/label/legend styling.
    rgb = np.clip((rgb - 0.5) * 1.18 + 0.5, 0, 1)
    lum = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])[..., None]
    rgb = np.clip(lum + (rgb - lum) * 1.16, 0, 1)
    rgb = np.clip(rgb**0.94 * 0.72, 0, 1)
    blue_dominant = rgb[..., 2] > np.maximum(rgb[..., 0], rgb[..., 1])
    rgb[..., 2] = np.where(blue_dominant, np.clip(rgb[..., 2] * 1.10, 0, 1), rgb[..., 2])
    rgb[..., 1] = np.where(~blue_dominant, np.clip(rgb[..., 1] * 1.04, 0, 1), rgb[..., 1])
    return rgb


_RASTER_CACHE: dict[tuple, tuple[np.ndarray, tuple[float, float, float, float]]] = {}


def reproject_nasa_panel(
    extent: tuple[float, float, float, float],
    dst_cartopy_crs: ccrs.Projection,
    pixel_width: int,
    pixel_height: int,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Read the georeferenced NASA GeoTIFF and warp it to the panel CRS/size."""
    if not NASA_BASEMAP_TIF.exists():
        raise FileNotFoundError(f"NASA Blue Marble GeoTIFF not found: {NASA_BASEMAP_TIF}")
    # Avoid expensive repeated warps between PNG/PDF rendering phases.
    cache_key = (
        tuple(round(v, 6) for v in extent),
        dst_cartopy_crs.proj4_init,
        int(pixel_width),
        int(pixel_height),
    )
    if cache_key in _RASTER_CACHE:
        return _RASTER_CACHE[cache_key]

    dst_crs = cartopy_to_rasterio_crs(dst_cartopy_crs)
    dst_bounds_rio = target_bounds_for_extent(extent, dst_crs)  # left, bottom, right, top
    dst_transform = from_bounds(*dst_bounds_rio, int(pixel_width), int(pixel_height))
    dst = np.zeros((3, int(pixel_height), int(pixel_width)), dtype=np.uint8)

    with rasterio.open(NASA_BASEMAP_TIF) as src:
        for band in range(1, 4):
            reproject(
                source=rasterio.band(src, band),
                destination=dst[band - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata=0,
                resampling=Resampling.bilinear,
                num_threads=4,
            )

    rgb = np.moveaxis(dst, 0, -1)
    rgb = darken_nasa_rgb(rgb)
    dst_bounds_mpl = (dst_bounds_rio[0], dst_bounds_rio[2], dst_bounds_rio[1], dst_bounds_rio[3])
    _RASTER_CACHE[cache_key] = (rgb, dst_bounds_mpl)
    return rgb, dst_bounds_mpl


def add_basemap(ax, extent, dst_cartopy_crs, pixel_width, pixel_height, detail="110m"):
    ax.set_facecolor("#020a0f")
    try:
        rgb, dst_bounds = reproject_nasa_panel(extent, dst_cartopy_crs, pixel_width, pixel_height)
        ax.imshow(
            rgb,
            extent=dst_bounds,
            origin="upper",
            transform=dst_cartopy_crs,
            interpolation="none",
            zorder=0,
        )
    except Exception as exc:
        print(f"WARNING: NASA basemap failed for extent {extent}: {exc}")
        ax.add_feature(cfeature.OCEAN.with_scale(detail), facecolor="#03111a", zorder=0)
        ax.add_feature(cfeature.LAND.with_scale(detail), facecolor="#27351f", zorder=0)

    ax.add_feature(cfeature.COASTLINE.with_scale(detail), edgecolor="white", linewidth=0.35, alpha=0.42, zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale(detail), edgecolor="white", linewidth=0.25, alpha=0.22, zorder=2)
    ax.add_feature(cfeature.LAKES.with_scale(detail), edgecolor="white", facecolor="none", linewidth=0.18, alpha=0.22, zorder=2)


def add_grid(ax, extent, *, main=False):
    if main:
        xticks = np.arange(-150, 151, 30)
        yticks = np.arange(-60, 61, 30)
    else:
        lon_span = extent[1] - extent[0]
        lat_span = extent[3] - extent[2]
        xstep = 20 if lon_span > 50 else 10
        ystep = 10
        xticks = np.arange(math.ceil(extent[0] / xstep) * xstep, extent[1] + 0.01, xstep)
        yticks = np.arange(math.ceil(extent[2] / ystep) * ystep, extent[3] + 0.01, ystep)

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        xlocs=xticks,
        ylocs=yticks,
        linewidth=0.45,
        color="white",
        alpha=0.17,
        linestyle="--",
        zorder=3,
    )
    gl.bottom_labels = False
    gl.left_labels = True
    gl.right_labels = main
    # Keep the global coordinate labels, but leave the inter-panel legend band
    # free of regional longitude labels.
    gl.top_labels = main
    gl.xlabel_style = {"size": 7.5 if main else 6.8, "color": "white", "alpha": 0.85}
    gl.ylabel_style = {"size": 7.5 if main else 6.8, "color": "white", "alpha": 0.85}
    return gl


def scatter_sites(ax, sites: pd.DataFrame, size: float):
    pc = ccrs.PlateCarree()
    for group in GROUP_ORDER:
        sub = sites[sites["IGBP_group"].eq(group)]
        if sub.empty:
            continue
        ax.scatter(
            sub["longitude"],
            sub["latitude"],
            s=size,
            c=GROUP_COLORS[group],
            edgecolors="white",
            linewidths=1.05,
            alpha=0.98,
            transform=pc,
            zorder=10,
        )
        ax.scatter(
            sub["longitude"],
            sub["latitude"],
            s=size * 1.22,
            facecolors="none",
            edgecolors="black",
            linewidths=0.42,
            alpha=0.72,
            transform=pc,
            zorder=9,
        )


# Manual per-site label-offset overrides, in degrees. These were tuned by eye at a
# reference panel span of roughly _REFERENCE_SPAN_DEG (close to the North America /
# Europe / Australia / East Asia inset extents used below). See label_offsets_for_region,
# which rescales these for any panel whose extent differs from that reference -- most
# importantly the much wider main global panel, where an unscaled few-degree nudge is
# nearly invisible and previously left several labels sitting on top of their marker.
LABEL_OFFSETS = {
    # North America.
    "US-Ne1": (5.8, 2.2), "US-Ne2": (5.8, 1.0), "US-Ne3": (5.8, 3.4),
    "US-AR1": (-8.0, 1.8), "US-AR2": (-8.0, 0.4), "US-ARb": (5.0, -1.6), "US-ARc": (5.0, -3.0),
    "US-SRG": (-7.0, 2.2), "US-SRM": (-7.0, -1.0), "US-Whs": (6.0, -1.0), "US-Wkg": (6.0, -3.6),
    "US-NR1": (-9.0, 2.2), "US-Ton": (-8.0, -1.5), "US-WCr": (-6.5, 3.0), "CA-Gro": (5.0, 2.0),
    "US-Ha1": (5.4, -2.4), "US-Ho2": (5.4, 1.6), "US-Bar": (5.4, -0.2), "US-MMS": (5.0, -1.2),
    "US-UiC": (5.0, 0.4),
    # Europe.
    "FI-Hyy": (2.8, 2.1), "DK-Sor": (2.5, 1.5), "NL-Loo": (-4.0, 1.8),
    "BE-Bra": (-4.2, 1.6), "BE-Maa": (-4.2, 0.2), "BE-Vie": (-6.2, -1.4),
    "DE-Hai": (3.2, 1.3), "DE-Tha": (3.2, 0.2), "DE-RuS": (3.2, -0.8),
    "FR-Gri": (-4.7, 2.4), "FR-Hes": (-4.5, -1.2), "FR-Aur": (-4.5, -1.4), "FR-Pue": (-4.5, -2.7),
    "CH-Dav": (3.2, -2.6), "CH-Lae": (5.4, -0.3), "CH-Oe1": (2.0, -5.8), "CH-Oe2": (-5.0, -2.0),
    "IT-Ren": (-2.5, -5.5), "IT-BCi": (3.2, -2.8), "ES-Agu": (-4.8, 0.4), "ES-LJu": (-4.8, -2.4),
    "ES-LM1": (-4.8, 1.0), "ES-LM2": (-4.8, 2.2), "CZ-BK2": (3.0, 1.0), "AT-Zoe": (3.0, -0.4),
    "IL-Yat": (3.0, -1.0),
    # Australia.
    "AU-How": (3.5, 2.2), "AU-DaP": (3.5, 1.1), "AU-DaS": (3.5, 0.0), "AU-Dry": (3.5, -1.1),
    "AU-Stp": (3.5, -2.2), "AU-TTE": (4.5, 1.2), "AU-ASM": (-7.0, -1.0),
    "AU-Gin": (-5.5, 1.5), "AU-Cpr": (-5.5, -1.0), "AU-Emr": (4.8, 1.0), "AU-Tum": (4.8, 0.6),
    "AU-Rig": (4.2, -0.2), "AU-Whr": (4.2, -1.8), "AU-Wom": (4.2, -3.2),
    # East Asia.
    "CN-Dmn": (4.0, 1.4), "CN-Sdq": (4.0, -1.4), "CN-Ash": (-5.0, -1.7),
    "CN-Din": (4.2, -1.2), "CN-Du2": (4.2, 1.5), "JP-Tak": (4.0, 1.0),
    # Sparse global.
    "BR-Ma2": (7.0, -2.5), "GF-Guy": (7.0, 2.0), "CG-Tch": (7.0, -2.5), "SN-Dhr": (7.0, 2.0), "SD-Dem": (7.0, -1.0),
}

def label_offsets_for_region(region: str, labels: list[str], extent: tuple[float, float, float, float]):
    """Per-panel label offsets in degrees, tuned to the locked reference style.

    LABEL_OFFSETS values are used as-is on every panel (main global and regional): on the
    main panel their absolute degree size is small relative to the ~356 degree span, but
    that is fine -- the collision-aware ring search in annotate_sites (marker/label/line
    avoidance) supplies the actual displacement, and scaling these degree offsets up for
    the main panel was tried and rejected: for eastern-hemisphere sites near the map edge
    it can push the offset target past +/-180 degrees, which a linear PlateCarree
    transform wraps to the opposite side of the panel and produces a catastrophically
    long, wrong leader line. annotate_sites additionally clamps the offset target to the
    panel's own extent as a second safeguard against exactly that.
    """
    w = extent[1] - extent[0]
    h = extent[3] - extent[2]
    manual = dict(LABEL_OFFSETS)
    generic = [
        (0.035 * w, 0.050 * h),
        (-0.055 * w, 0.030 * h),
        (0.045 * w, -0.045 * h),
        (-0.060 * w, -0.035 * h),
        (0.060 * w, 0.000 * h),
        (-0.060 * w, 0.000 * h),
        (0.020 * w, 0.070 * h),
        (0.020 * w, -0.070 * h),
    ]
    return manual, generic


def _overlap_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Overlap area between display-coordinate boxes: left, right, bottom, top."""
    dx = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    dy = max(0.0, min(a[3], b[3]) - max(a[2], b[2]))
    return dx * dy


def _box_contains_point(box: tuple[float, float, float, float], px: float, py: float) -> bool:
    return box[0] <= px <= box[1] and box[2] <= py <= box[3]


def _segments_intersect(p1, p2, p3, p4) -> bool:
    """True if open line segments p1-p2 and p3-p4 cross (shared endpoints don't count)."""

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def _label_size_px(site_id: str, fontsize: float, dpi: float) -> tuple[float, float]:
    """Conservative label-box estimate in display pixels.

    The earlier degree-based estimate was too small in dense panels such as
    Europe.  This pixel-space estimate deliberately includes the rounded label
    box and a small visual clearance margin.
    """
    px_per_pt = dpi / 72.0
    width = max(44.0, len(site_id) * fontsize * px_per_pt * 0.78 + fontsize * px_per_pt * 1.40)
    height = fontsize * px_per_pt * 2.25
    return width, height


def annotate_sites(
    ax, sites: pd.DataFrame, extent, region: str, *, fontsize: float, main: bool = False, marker_size: float = 40.0
):
    """Place one leader-line label per site with collision-aware, outward-biased search.

    Must be called only after the containing figure has been drawn at least once
    (fig.canvas.draw()) -- cartopy GeoAxes adjust their true on-screen box to preserve
    aspect ratio during drawing, so ax.get_window_extent() is only accurate afterwards.
    Calling this before a draw silently clamps labels against the wrong (oversized,
    pre-aspect-correction) box, letting them spill past the actual panel frame.
    """
    pc = ccrs.PlateCarree()
    xmin, xmax, ymin, ymax = extent
    manual, generic = label_offsets_for_region(region, sites["SITE_ID"].astype(str).tolist(), extent)
    data_transform = pc._as_mpl_transform(ax)
    inv_data_transform = data_transform.inverted()
    axis_box = ax.get_window_extent()
    edge_pad = 12.0

    # All markers actually drawn in this panel are avoidance obstacles, whether or not
    # they end up labelled (e.g. the un-labelled majority on the main global panel), so
    # a label can never be placed on top of an unrelated site's marker.
    marker_radius_px = max(6.0, math.sqrt(marker_size / math.pi) * (ax.figure.dpi / 72.0) * 1.35)
    placed: list[tuple[float, float, float, float]] = []
    placed_lines: list[tuple[tuple[float, float], tuple[float, float]]] = []
    all_points_px: dict[str, np.ndarray] = {}
    for _, r in sites.iterrows():
        p = np.asarray(data_transform.transform((float(r["longitude"]), float(r["latitude"]))), dtype=float)
        all_points_px[str(r["SITE_ID"])] = p
        placed.append((p[0] - marker_radius_px, p[0] + marker_radius_px, p[1] - marker_radius_px, p[1] + marker_radius_px))

    # Main-panel label subset: sparse non-dense sites plus a few regional anchors.
    if main:
        keep = {
            "US-Ton", "US-Wkg", "US-NR1", "US-Ho2",
            "GF-Guy", "BR-Ma2", "SN-Dhr", "SD-Dem", "CG-Tch",
            "IL-Yat", "FI-Hyy", "DK-Sor", "ES-LM2", "ES-LJu",
            "CN-Dmn", "CN-Du2", "CN-Ash", "CN-Din", "JP-Tak",
            "AU-How", "AU-Dry", "AU-Emr", "AU-Gin", "AU-Wom",
        }
        label_sites = sites[sites["SITE_ID"].isin(keep)].copy()
    else:
        label_sites = sites

    # Local outward-from-cluster direction per labelled site: point minus the centroid
    # of nearby markers (within ~14% of the panel diagonal). Candidates that move further
    # into a cluster are penalised; candidates that move away from it are favoured.
    diag = math.hypot(axis_box.x1 - axis_box.x0, axis_box.y1 - axis_box.y0)
    neighbor_radius_px = 0.14 * diag
    outward_unit: dict[str, np.ndarray] = {}
    for sid, p in all_points_px.items():
        neighbors = [q for other_sid, q in all_points_px.items() if other_sid != sid and np.hypot(*(q - p)) <= neighbor_radius_px]
        if neighbors:
            centroid = np.mean(neighbors, axis=0)
            v = p - centroid
            norm = np.hypot(*v)
            outward_unit[sid] = v / norm if norm > 1e-6 else np.zeros(2)
        else:
            outward_unit[sid] = np.zeros(2)

    # Put north/west labels first; it helps dense Europe and North America.
    ordered = label_sites.sort_values(["latitude", "longitude"], ascending=[False, True]).reset_index(drop=True)
    for i, r in ordered.iterrows():
        sid = str(r["SITE_ID"])
        dx, dy = manual.get(sid, generic[i % len(generic)])
        lon = float(r["longitude"])
        lat = float(r["latitude"])
        point_px = all_points_px[sid]
        # Clamp the offset target to this panel's own extent so an offset can never push
        # a longitude past +/-180 deg and wrap to the opposite side of the map (see
        # label_offsets_for_region for why this matters).
        target_lon = min(max(lon + dx, xmin), xmax)
        target_lat = min(max(lat + dy, ymin), ymax)
        initial_px = np.asarray(data_transform.transform((target_lon, target_lat)), dtype=float)
        base_delta = initial_px - point_px

        label_w, label_h = _label_size_px(sid, fontsize, ax.figure.dpi)
        label_diag = math.hypot(label_w, label_h)
        out_vec = outward_unit.get(sid, np.zeros(2))

        # Ring search around the preferred manual/generic offset: dense clusters (Europe,
        # eastern North America, Australia's southeast, East Asia) need a wide, thorough
        # search with room for longer leader lines, not just a handful of nearby slots.
        candidate_shifts = [(0.0, 0.0)]
        for radius_mult in (0.9, 1.6, 2.3, 3.1, 4.0):
            radius = radius_mult * label_diag * 0.5
            for angle_deg in range(0, 360, 24):
                angle = math.radians(angle_deg)
                candidate_shifts.append((radius * math.cos(angle), radius * math.sin(angle)))

        best = None
        best_score = float("inf")
        for sx, sy in candidate_shifts:
            cx, cy = point_px + base_delta + np.array([sx, sy])
            cx = min(max(cx, axis_box.x0 + edge_pad + label_w / 2), axis_box.x1 - edge_pad - label_w / 2)
            cy = min(max(cy, axis_box.y0 + edge_pad + label_h / 2), axis_box.y1 - edge_pad - label_h / 2)
            box = (cx - label_w / 2, cx + label_w / 2, cy - label_h / 2, cy + label_h / 2)

            overlap = sum(_overlap_area(box, old) for old in placed)
            # A label must never cover its own marker even when clamped near an edge.
            if _box_contains_point(box, *point_px):
                overlap += label_w * label_h

            crossings = sum(
                1 for seg in placed_lines if _segments_intersect(tuple(point_px), (cx, cy), seg[0], seg[1])
            )

            disp = base_delta + np.array([sx, sy])
            disp_norm = np.hypot(*disp)
            outward_penalty = 0.0
            if disp_norm > 1e-6 and np.any(out_vec):
                alignment = float(np.dot(disp / disp_norm, out_vec))  # +1 outward, -1 inward
                outward_penalty = max(0.0, -alignment) * 9.0

            shift_penalty = 0.020 * math.hypot(sx, sy)
            leader_penalty = 0.006 * math.hypot(*(np.array([cx, cy]) - point_px))
            score = overlap * 1000.0 + crossings * 300.0 + outward_penalty + shift_penalty + leader_penalty
            if score < best_score:
                best_score = score
                best = (cx, cy, box)
            if overlap == 0 and crossings == 0 and outward_penalty == 0.0:
                break

        assert best is not None
        cx, cy, box = best
        # Add a little display-space breathing room between accepted label boxes.
        clearance = 5.0 if not main else 3.0
        placed.append((box[0] - clearance, box[1] + clearance, box[2] - clearance, box[3] + clearance))
        placed_lines.append((tuple(point_px), (cx, cy)))
        tx, ty = inv_data_transform.transform((cx, cy))

        ax.annotate(
            sid,
            xy=(r["longitude"], r["latitude"]),
            xycoords=data_transform,
            xytext=(tx, ty),
            textcoords=data_transform,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color="#1d1d1d",
            bbox=dict(boxstyle="round,pad=0.16", fc="#e9ecef", ec="#333333", lw=0.45, alpha=0.92),
            arrowprops=dict(arrowstyle="-", color="#e9ecef", lw=0.75, alpha=0.95),
            zorder=20,
        )


def set_panel_frame(ax):
    for spine in ax.spines.values():
        spine.set_edgecolor("#d8d8d8")
        spine.set_linewidth(0.9)


def region_subset(sites: pd.DataFrame, extent):
    return sites[
        sites["longitude"].between(extent[0], extent[1])
        & sites["latitude"].between(extent[2], extent[3])
    ].copy()


def build_figure():
    sites, missing = load_sites()

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "savefig.facecolor": "#02070b",
        "figure.facecolor": "#02070b",
    })
    pc = ccrs.PlateCarree()
    proj = pc
    fig = plt.figure(figsize=(20, 11.25), dpi=500, facecolor="#02070b")

    # Title and subtitle matching accepted figure character.
    fig.text(
        0.018, 0.975,
        "Figure 1. Global eddy-covariance tower network used for SIF–GPP analysis",
        color="white", fontsize=18.5, fontweight="bold", ha="left", va="top",
    )
    fig.text(
        0.018, 0.943,
        f"{len(sites)} of {len(sites) + len(missing)} EC towers mapped from accepted local metadata; "
        f"{len(missing)} study sites lacked accepted source coordinates.",
        color="white", fontsize=11.2, style="italic", ha="left", va="top", alpha=0.92,
    )

    # Main global panel.
    main_extent = (-178, 178, -62, 76)
    main_rect = [0.020, 0.420, 0.960, 0.485]
    ax = fig.add_axes(main_rect, projection=proj)
    ax.set_extent(main_extent, crs=pc)
    add_basemap(
        ax,
        main_extent,
        proj,
        pixel_width=int(MASTER_WIDTH_PX * main_rect[2]),
        pixel_height=int(MASTER_HEIGHT_PX * main_rect[3]),
        detail="110m",
    )
    add_grid(ax, main_extent, main=True)
    main_marker_size = 42
    scatter_sites(ax, sites, size=main_marker_size)
    set_panel_frame(ax)

    # Legend in the locked central style.
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=GROUP_COLORS[g],
               markeredgecolor="white", markeredgewidth=1.05, markersize=8.5, label=g)
        for g in GROUP_ORDER if g in set(sites["IGBP_group"])
    ]
    leg = fig.legend(
        handles=handles,
        title="Tower land-cover class (IGBP grouped)",
        loc="center",
        bbox_to_anchor=(0.5, 0.370),
        ncol=len(handles),
        fontsize=9.5,
        title_fontsize=10.8,
        frameon=True,
        facecolor="#050b0f",
        edgecolor="#a5a5a5",
        framealpha=0.92,
        borderpad=0.75,
        columnspacing=1.55,
        handletextpad=0.65,
        labelspacing=0.7,
    )
    for text in leg.get_texts():
        text.set_color("white")
    leg.get_title().set_color("white")
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_boxstyle("round,pad=0.45,rounding_size=0.18")

    # Bottom regional panels, same arrangement as accepted reference.
    panel_rects = {
        "North America": [0.020, 0.045, 0.230, 0.275],
        "Europe": [0.282, 0.045, 0.230, 0.275],
        "Australia": [0.532, 0.045, 0.210, 0.275],
        "East Asia": [0.770, 0.045, 0.210, 0.275],
    }
    regional_axes_info = []
    for name, rect in panel_rects.items():
        extent = REGIONS[name]
        rax = fig.add_axes(rect, projection=proj)
        rax.set_extent(extent, crs=pc)
        add_basemap(
            rax,
            extent,
            proj,
            pixel_width=int(MASTER_WIDTH_PX * rect[2]),
            pixel_height=int(MASTER_HEIGHT_PX * rect[3]),
            detail="50m",
        )
        add_grid(rax, extent, main=False)
        sub = region_subset(sites, extent)
        regional_marker_size = 36 if name in ["Australia", "East Asia"] else 34
        scatter_sites(rax, sub, size=regional_marker_size)
        set_panel_frame(rax)
        rax.text(
            0.02, 0.965, name,
            transform=rax.transAxes, ha="left", va="top",
            color="white", fontsize=13.0, fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2.6, foreground="black", alpha=0.85)],
            zorder=30,
        )
        regional_axes_info.append((rax, sub, extent, name, regional_marker_size))

    # Annotation must happen only after a real draw: cartopy GeoAxes resize themselves
    # to preserve aspect ratio during drawing, so ax.get_window_extent() inside
    # annotate_sites is only trustworthy once every basemap/scatter/frame element above
    # has been added and the figure has been rendered at least once.
    fig.canvas.draw()
    annotate_sites(ax, sites, main_extent, "Global", fontsize=6.9, main=True, marker_size=main_marker_size)
    for rax, sub, extent, name, regional_marker_size in regional_axes_info:
        annotate_sites(
            rax, sub, extent, name,
            fontsize=5.8 if name == "Europe" else 6.1,
            marker_size=regional_marker_size,
        )

    fig.text(
        0.018, 0.017,
        "Tower coordinates are from accepted local metadata only; CN-Dzo and CN-Lch are not mapped.",
        color="white", fontsize=9.4, ha="left", va="bottom", alpha=0.90,
    )
    fig.text(
        0.980, 0.017,
        "Projection: Equal Earth",
        color="white", fontsize=9.4, ha="right", va="bottom", alpha=0.90,
    )
    return fig, sites, missing


def main():
    fig, sites, missing = build_figure()

    # Native 10000 × 5625 px output: 20 × 11.25 in at 500 dpi.
    fig.savefig(PNG_OUT, dpi=500, facecolor="#02070b")
    fig.savefig(PDF_OUT, dpi=500, facecolor="#02070b")
    plt.close(fig)

    # RGB 500-dpi TIFF from the same native 10000 × 5625 render, not upsampled.
    with Image.open(PNG_OUT) as im:
        rgb = im.convert("RGB")
        rgb.save(TIF_500_OUT, dpi=(500, 500), compression="tiff_lzw")

    NOTES_OUT.write_text(
        f"""# Figure 1 high-resolution final export (v2 -- annotation-placement fix)

This is a label-placement-only correction of `Figure_1_Global_EC_Tower_Map_final.png`.
Basemap, projection, map layout/extents, title/subtitle, legend, marker colors/styles,
and figure dimensions are unchanged. Only site-label placement changed:

- `annotate_sites` now runs only after `fig.canvas.draw()`, because cartopy GeoAxes
  resize themselves to preserve aspect ratio during drawing -- calling
  `ax.get_window_extent()` beforehand (the original code) returned an oversized,
  incorrect box, letting edge labels (e.g. US-Ho2/US-Bar/US-Ha1) render outside the
  actual panel frame.
- Label placement is now collision-aware against every marker in the panel (not just
  other labels), so a label can no longer sit on top of an unrelated, unlabelled site's
  marker (this previously happened to JP-Tak, CN-Dmn, CN-Ash).
- A wide ring search (multiple radii and angles around the preferred offset) with an
  outward-from-local-cluster bias and a leader-line-crossing penalty replaces the old
  fixed shortlist of candidate offsets, so dense clusters (Europe, eastern North
  America, southeastern Australia) resolve without label-on-label or line-through-label
  collisions.
- A handful of `LABEL_OFFSETS` entries were manually retuned where the automatic search
  still produced an unnecessarily long or crossing leader line (e.g. `CH-Oe1`, `IT-Ren`,
  and a duplicate-offset conflict between `CH-Dav`/`IT-Ren` that made them fight for the
  same slot).
- Offset targets are clamped to each panel's own extent so an offset can never push a
  longitude past +/-180 deg and wrap to the opposite side of the map.

This script redraws the locked accepted Figure 1 style natively from:

`{HANDOFF_CSV}`

Basemap source:

`{NASA_BASEMAP_TIF}`

The NASA Blue Marble GeoTIFF is read with rasterio and reprojected separately
for the Equal Earth global panel and for each regional panel. It is not simply
stretched into the axes.

No previous PNG was upscaled. The master PNG canvas is 10000 × 5625 px,
created as 20 × 11.25 inches at 500 dpi.

Outputs:

- `{PNG_OUT.name}` — native 10000 × 5625 px PNG
- `{PDF_OUT.name}` — PDF export from the same redraw
- `{TIF_500_OUT.name}` — RGB 500 dpi TIFF with LZW compression

Data counts:

- Total study towers in handoff CSV: {len(sites) + len(missing)}
- Mapped towers: {len(sites)}
- Missing-coordinate towers: {len(missing)}
- Missing-coordinate SITE_IDs: {', '.join(missing['SITE_ID'].astype(str).tolist())}

Visual constraints:

- 16:9 aspect ratio preserved
- top global map plus bottom panels for North America, Europe, Australia, and East Asia
- locked dark satellite-style visual character preserved
- marker palette and label/legend style preserved
- NASA Blue Marble basemap replaces the earlier low-resolution stock relief raster
- no new EC tower coordinates were introduced
""",
        encoding="utf-8",
    )

    print(f"PNG: {PNG_OUT}")
    print(f"PDF: {PDF_OUT}")
    print(f"TIFF: {TIF_500_OUT}")
    print(f"mapped={len(sites)} missing={len(missing)}")


if __name__ == "__main__":
    main()
