"""
Shared helpers for the Advanced Coupling Analysis (Parts III spatial hotspots and IV
temporal hotspots). Basemap-loading recipe, product colors, save_figure, and the
project's reliability rule are copied/adapted from the already-successful companion
scripts in this project (All_the_Visualizations_REVISED_4PRODUCT_FINAL and
SIF_EC_Stress_Induced_Decoupling_SID) so visual style/conventions stay consistent
across the whole project.

IMPORTANT scope note: this module supports the CORRELATION-BASED coupling/decoupling
analysis (site-level Spearman rho of SIF_ANOM vs GPP_ANOM, and a within-tower
z-scored divergence metric for temporal hotspots). This is a related but DISTINCT
concept from SIF_EC_Stress_Induced_Decoupling_SID's stress-weighted SID ratio metric.
SID outputs may be referenced for context (see load_sid_context()) but are never
treated as equivalent to the coupling metrics computed here.

Author: project authors
Run date: 2026-09-01
"""

from __future__ import annotations
# Original script filename (sanitized): coupling_common.py; public name: coupling_common.py
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

import textwrap
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling

import cartopy.crs as ccrs

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------

ROOT = Path(PROJECT_ROOT)
ADV_ROOT = ROOT / "SIF_EC_Advanced_Coupling_Analysis"
SID_ROOT = ROOT / "SIF_EC_Stress_Induced_Decoupling_SID"

BASEMAP_TIF = Path(_os.environ.get("NE_BASEMAP_TIF", "NE1_HR_LC_SR_W.tif"))

HARMONIC_RESIDUALS_CSV = (
    ROOT / "SIF_EC_SIF_GPP_Final_Verification" / "02_independent_reverification"
    / "harmonic_residuals_full.csv"
)

SID_TOWER_SUMMARY_CSV = SID_ROOT / "05_hotspot_analysis" / "tower_SID_hotspot_summary.csv"
PARTIAL_CORR_CSV = ADV_ROOT / "02_partial_correlation" / "partial_correlation_site_summary.csv"

DIR_SPATIAL = ADV_ROOT / "04_spatial_hotspots"
DIR_TEMPORAL = ADV_ROOT / "05_temporal_hotspots"
DIR_GEOFIG = ADV_ROOT / "08_geospatial_figures"
for d in (DIR_SPATIAL, DIR_TEMPORAL, DIR_GEOFIG):
    d.mkdir(parents=True, exist_ok=True)

PRODUCTS = ["GOSIF", "GOME-2", "TROPOSIF", "OCO-3"]  # main-paper products only; OCO-2 excluded throughout

PRODUCT_COLORS = {
    "GOSIF": "#0072B2",     # blue
    "GOME-2": "#E69F00",    # orange
    "TROPOSIF": "#009E73",  # bluish green
    "OCO-3": "#CC79A7",     # reddish purple
}

DIVERGING_CMAP = plt.get_cmap("RdYlBu_r")   # coupling rho: red/warm = strong positive coupling, blue/cool = weak or negative
SEQ_CMAP = plt.get_cmap("YlOrRd")           # sequential: consensus strength / magnitude

FONT = {"family": "DejaVu Sans"}
plt.rcParams.update({
    "font.family": FONT["family"],
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.6,
})

CAVEAT_TEXT = (
    "Descriptive, associational pattern from real eddy-covariance tower observations across sites and/or time -- "
    "not proof of causality. Every marker is a real tower; no values are spatially interpolated between towers."
)

# Project-wide reliability rule (matches SIF_EC_Stress_Induced_Decoupling_SID and the
# broader statistical-audit standard used throughout this project): a cell needs
# N >= 12 site-months AND N >= 3 independent towers to be reported as "reliable".
# Unreliable cells are always shown de-emphasized (faded), never hidden.
REL_MIN_SITEMONTHS = 12
REL_MIN_TOWERS = 3


def reliable(n_sitemonths, n_towers) -> bool:
    return (n_sitemonths >= REL_MIN_SITEMONTHS) and (n_towers >= REL_MIN_TOWERS)


# --------------------------------------------------------------------------------------
# Basemap (mandatory recipe, reused verbatim from build_new_geospatial_visualizations.py
# via sid_geospatial_common.py)
# --------------------------------------------------------------------------------------

def load_basemap(target_width_px: int = 6000, white_blend: float = 0.32):
    with rasterio.open(BASEMAP_TIF) as src:
        assert src.crs.to_epsg() == 4326, f"Unexpected basemap CRS: {src.crs}"
        scale = target_width_px / src.width
        out_width = target_width_px
        out_height = max(1, round(src.height * scale))
        arr = src.read(
            out_shape=(src.count, out_height, out_width),
            resampling=Resampling.average,
        )
        bounds = src.bounds
    rgb = np.transpose(arr, (1, 2, 0)).astype(np.float32) / 255.0
    rgb = rgb * (1 - white_blend) + white_blend * 1.0
    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    return rgb, extent


def add_basemap(ax, rgb, extent, set_global: bool = True):
    ax.imshow(
        rgb, origin="upper", extent=extent, transform=ccrs.PlateCarree(),
        interpolation="bilinear", zorder=0,
    )
    if set_global:
        ax.set_global()
    ax.spines["geo"].set_edgecolor("#888888")
    ax.spines["geo"].set_linewidth(0.5)


def wrap_text(text: str, width: int = 150) -> str:
    lines = []
    for para in text.split("\n"):
        wrapped = textwrap.wrap(para, width=width) or [""]
        lines.extend(wrapped)
    return "\n".join(lines)


def save_figure(fig, out_dir: Path, stem: str, dpi: int = 600):
    png = out_dir / f"{stem}.png"
    svg = out_dir / f"{stem}.svg"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(svg, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved: {png.name}, {svg.name}, {pdf.name}")


def size_for_n(n, lo: float = 18.0, hi: float = 220.0, n_ref: float = None) -> np.ndarray:
    n = np.asarray(n, dtype=float)
    if n_ref is None:
        n_ref = np.nanmax(n) if np.nanmax(n) > 0 else 1.0
    frac = np.clip(n / n_ref, 0, 1)
    return lo + (hi - lo) * np.sqrt(frac)


# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------

def load_harmonic_residuals() -> pd.DataFrame:
    """The accepted per-site harmonic seasonal-regression SIF/GPP anomaly table --
    primary source for ALL Part III/IV coupling metrics. Read-only; nothing here is
    recomputed or modified from the Final Verification stream's output."""
    cols = ["SITE_ID", "YEAR", "MONTH", "YYYYMM", "LATITUDE", "LONGITUDE",
            "IGBP", "IGBP_GPP_TABLE", "SIF_PRODUCT", "SIF_ANOM", "GPP_ANOM"]
    df = pd.read_csv(HARMONIC_RESIDUALS_CSV, usecols=cols, low_memory=False)
    assert not (df["SIF_PRODUCT"] == "OCO-2").any(), "OCO-2 must be excluded (main-paper products only)"
    df = df[df["SIF_PRODUCT"].isin(PRODUCTS)].copy()
    df = df.dropna(subset=["SIF_ANOM", "GPP_ANOM"]).copy()
    # Biome: IGBP_GPP_TABLE is the curated/finalized biome label used for the GPP
    # lookup table (99.99% identical to raw IGBP where both are populated -- 0.959
    # non-null fill rate for both, verified equal on the overlap); prefer it, fall
    # back to raw IGBP where it is missing.
    df["BIOME"] = df["IGBP_GPP_TABLE"].fillna(df["IGBP"])
    df["HEMISPHERE"] = np.where(df["LATITUDE"] >= 0, "Northern", "Southern")
    # Fixed calendar-month meteorological season (NOT hemisphere-shifted) -- see
    # season interpretation helper below for the hemisphere-aware reading.
    season_map = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                  6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
    df["SEASON"] = df["MONTH"].map(season_map)
    return df


SEASON_INTERPRETATION = {
    "DJF": "DJF = astronomical/meteorological winter in the Northern Hemisphere, summer in the Southern Hemisphere.",
    "MAM": "MAM = spring in the Northern Hemisphere, autumn in the Southern Hemisphere.",
    "JJA": "JJA = summer in the Northern Hemisphere, winter in the Southern Hemisphere.",
    "SON": "SON = autumn in the Northern Hemisphere, spring in the Southern Hemisphere.",
}
SEASON_CAPTION = " ".join(SEASON_INTERPRETATION.values())


def load_sid_context():
    """Optional contextual read of the SID (Stress-Induced Decoupling) tower hotspot
    summary -- a DIFFERENT metric (stress-weighted divergence ratio) from a related
    but distinct prior analysis. Returned for context/cross-reference only; never
    merged into or treated as equivalent to this analysis's own coupling definitions."""
    if SID_TOWER_SUMMARY_CSV.exists():
        return pd.read_csv(SID_TOWER_SUMMARY_CSV)
    return None


def load_partial_correlation_context():
    """Optional third coupling definition (C: residual/partial correlation), built by
    a parallel stream of this same project (02_partial_correlation/). Skipped (not
    fabricated) if not yet available at run time."""
    if PARTIAL_CORR_CSV.exists():
        return pd.read_csv(PARTIAL_CORR_CSV)
    return None
