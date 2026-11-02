"""
Shared Peak-Of-Season (POS) estimation library for the phenology-based revision of
the SIF-vs-GPP peak-timing diagnostics (replaces Diagnostic1-4's lead/lag-
classification approach as the MAIN explanatory analysis; those figures are kept,
unmodified, as supporting/QC material -- see 00_README_POS_REVISION.md).

Literature precedent: Wu et al. (2022, RSE) compare SIF and GPP seasonal trajectories
and their peak timing directly, then examine APAR/canopy-development mechanisms,
rather than leading with a lead/lag classification. Phenology studies typically fit or
smooth the seasonal curve and extract Peak-Of-Season (POS) from the SMOOTHED curve,
not the raw maximum monthly observation (a single noisy month can be an outlier).

METHOD
------
Two independent smoothers are fit to every SITE_ID x SIF_PRODUCT series (or, for the
pooled hemisphere x product panels, to the pooled monthly climatology):

  1. PRIMARY: 1st+2nd order harmonic (Fourier) regression -- IDENTICAL fitting form to
     this project's own audited/accepted deseasonalization method (harmonic_anomaly()
     in independent_reverification.py): y ~ 1 + sin(t) + cos(t) + sin(2t) +
     cos(2t), t = 2*pi*MONTH/12, fit by OLS on ALL available raw site-months (not just
     the 12 monthly means) so more data are used where available. Chosen as primary
     because it is this project's own already-audited/accepted smoother.
  2. SENSITIVITY: a periodic (wrap-around) cubic spline through the 12 (or fewer)
     calendar-month climatological means. Compared against the harmonic POS as a
     smoother-choice sensitivity check (reported, not silently picked-the-best-of).

POS = the calendar-month location of the maximum of the fitted curve (continuous,
sub-month resolution, since it comes from a smooth curve fit to monthly data -- NOT
claimed as day-level precision; always reported with a bootstrap 95% CI so the reader
sees the real uncertainty rather than a bare point value).

Circular handling: all months are treated on a 12-month circle throughout (a Dec/Jan
POS does not create a spurious +/-11 month gap against a nearby June/July POS -- see
circular_diff()). This applies identically to Northern and Southern Hemisphere sites;
no special-casing is needed because "month" is already a genuine circular quantity for
every site regardless of which calendar months fall in its local summer.

FLAGS (reported, never silently forced):
  - INSUFFICIENT_COVERAGE: fewer than MIN_UNIQUE_MONTHS distinct calendar months, or
    fewer than MIN_YEARS distinct years, or fewer than MIN_N total site-months.
  - FLAT_SEASONAL_CYCLE: harmonic regression's seasonal terms are not jointly
    significant (F-test p >= 0.05) -- the "seasonal cycle" the peak is drawn from is
    statistically indistinguishable from flat.
  - MULTIMODAL: the fitted curve has more than one local maximum within 80% of the
    global peak height (fraction of the curve's own peak-to-trough range) -- a single
    POS may not represent the site's real phenology well.
  - SMOOTHER_DISAGREEMENT: |POS_harmonic - POS_spline| (circular) > 1.5 months.
  - LOW_YEAR_REPLICATION: fewer than 4 distinct years (bootstrap CI is coarse/discrete).
"""
from __future__ import annotations
# Original script filename (sanitized): pos_lib.py; public name: pos_lib.py
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

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import CubicSpline

ROOT = Path(PROJECT_ROOT)
MASTER_DATA_PATH = ROOT / "SIF_EC_SIF_GPP_Relationship_Analysis" / "01_analysis_ready" / "SIF_GPP_Analysis_Ready_Master.csv"
BASEMAP_TIF = Path(_os.environ.get("NE_BASEMAP_TIF", "NE1_HR_LC_SR_W.tif"))

PRODUCTS = ["GOSIF", "GOME-2", "TROPOSIF", "OCO-3"]
PRODUCT_COLORS = {
    "GOSIF": "#0072B2", "GOME-2": "#E69F00", "TROPOSIF": "#009E73", "OCO-3": "#CC79A7",
}
PRODUCT_MARKERS = {"GOSIF": "o", "GOME-2": "s", "TROPOSIF": "^", "OCO-3": "D"}
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

MIN_UNIQUE_MONTHS = 8     # calendar months represented, out of 12 -- matches this project's harmonic_anomaly() default
MIN_YEARS = 3             # distinct years -- minimum for the seasonal cycle to reflect more than one year's noise
MIN_N_SITEMONTHS = 12     # total site-months -- matches this project's standard reliability floor
MIN_TOWERS_POOLED = 3     # towers required to compute a pooled product x hemisphere POS
N_BOOT = 500
RNG_SEED = 20260912
GRID_N = 2400             # ~0.005-month resolution; NOT a claim of day-level precision, just a fine evaluation grid


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_master() -> pd.DataFrame:
    df = pd.read_csv(MASTER_DATA_PATH, low_memory=False)
    df = df[df["SIF_PRODUCT"].isin(PRODUCTS)].copy()
    df = df.dropna(subset=["IGBP_GPP_TABLE", "LATITUDE", "SIF_VALUE", "GPP_VALUE", "MONTH", "YEAR"])
    df["IGBP_GPP_TABLE"] = df["IGBP_GPP_TABLE"].astype(str)
    df["MONTH"] = df["MONTH"].astype(int)
    df["YEAR"] = df["YEAR"].astype(int)
    site_lat = df.groupby("SITE_ID")["LATITUDE"].first()
    df["HEMI"] = df["SITE_ID"].map(lambda s: "NH" if site_lat[s] >= 0 else "SH")
    return df


# --------------------------------------------------------------------------
# Circular arithmetic
# --------------------------------------------------------------------------
def circular_diff(a: float, b: float) -> float:
    """Shortest signed distance a-b on a 12-month circle, in [-6, +6]."""
    if a is None or b is None or (isinstance(a, float) and np.isnan(a)) or (isinstance(b, float) and np.isnan(b)):
        return np.nan
    return (a - b + 6) % 12 - 6


def circular_mean_month(months: np.ndarray) -> float:
    ang = 2 * np.pi * np.asarray(months) / 12.0
    m = np.degrees(np.arctan2(np.mean(np.sin(ang)), np.mean(np.cos(ang)))) / 30.0
    return m % 12 or 12.0


# --------------------------------------------------------------------------
# Harmonic (Fourier) fit -- primary smoother
# --------------------------------------------------------------------------
def _design(months: np.ndarray) -> np.ndarray:
    t = 2 * np.pi * np.asarray(months, dtype=float) / 12.0
    return np.column_stack([np.ones_like(t), np.sin(t), np.cos(t), np.sin(2 * t), np.cos(2 * t)])


def fit_harmonic(months: np.ndarray, values: np.ndarray):
    """Returns (beta, seasonal_F_pvalue) or (None, None) if unfittable."""
    X = _design(months)
    y = np.asarray(values, dtype=float)
    n = len(y)
    if n < 6 or np.linalg.matrix_rank(X) < 5:
        return None, None
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    ss_res_full = float(np.sum((y - fitted) ** 2))
    ss_res_null = float(np.sum((y - y.mean()) ** 2))
    df_full = n - 5
    if df_full <= 0 or ss_res_full <= 0:
        return beta, np.nan
    f_stat = ((ss_res_null - ss_res_full) / 4) / (ss_res_full / df_full)
    p = float(1 - stats.f.cdf(max(f_stat, 0), 4, df_full))
    return beta, p


def harmonic_curve_on_grid(beta: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return _design(grid) @ beta


# --------------------------------------------------------------------------
# Periodic cubic spline -- sensitivity smoother
# --------------------------------------------------------------------------
def fit_periodic_spline(month_means: pd.Series):
    """month_means: index = calendar month (1-12, need not be complete), values =
    climatological mean. Returns a callable curve(grid) or None if unfittable."""
    s = month_means.dropna().sort_index()
    if len(s) < 4:
        return None
    x = s.index.to_numpy(dtype=float)
    y = s.to_numpy(dtype=float)
    x_ext = np.append(x, x[0] + 12)
    y_ext = np.append(y, y[0])
    try:
        cs = CubicSpline(x_ext, y_ext, bc_type="periodic")
    except Exception:
        return None
    return cs


# --------------------------------------------------------------------------
# Peak finding + shape diagnostics on a fitted curve
# --------------------------------------------------------------------------
def find_pos_and_shape(grid: np.ndarray, curve: np.ndarray) -> dict:
    i_max = int(np.argmax(curve))
    pos = float(grid[i_max] % 12) or 12.0
    peak_val = curve[i_max]
    rng = curve.max() - curve.min()
    # circular local-maxima count (neighbors wrap)
    n = len(curve)
    is_localmax = (curve > np.roll(curve, 1)) & (curve > np.roll(curve, -1))
    local_maxima_vals = curve[is_localmax]
    n_comparable_peaks = int(np.sum(local_maxima_vals >= peak_val - 0.2 * rng)) if rng > 0 else 1
    return {"pos": pos, "peak_value": float(peak_val), "range": float(rng), "n_comparable_peaks": n_comparable_peaks}


# --------------------------------------------------------------------------
# Per-tower (SITE_ID x SIF_PRODUCT) POS estimation, one variable (SIF or GPP) at a time
# --------------------------------------------------------------------------
def estimate_pos_tower(g: pd.DataFrame, value_col: str, rng: np.random.Generator,
                        n_boot: int = N_BOOT) -> dict:
    months = g["MONTH"].to_numpy()
    years = g["YEAR"].to_numpy()
    values = g[value_col].to_numpy(dtype=float)
    n = len(values)
    n_unique_months = len(np.unique(months))
    unique_years = np.unique(years)
    n_years = len(unique_years)

    flags = []
    if n < MIN_N_SITEMONTHS or n_unique_months < MIN_UNIQUE_MONTHS or n_years < MIN_YEARS:
        flags.append("INSUFFICIENT_COVERAGE")
        return {"POS": np.nan, "POS_CI_LOW": np.nan, "POS_CI_HIGH": np.nan, "POS_SPLINE": np.nan,
                "N": n, "N_UNIQUE_MONTHS": n_unique_months, "N_YEARS": n_years,
                "SEASONAL_P": np.nan, "N_COMPARABLE_PEAKS": np.nan, "FLAGS": ";".join(flags)}

    beta, p_seasonal = fit_harmonic(months, values)
    grid = np.linspace(1e-6, 12, GRID_N)
    if beta is None:
        flags.append("INSUFFICIENT_COVERAGE")
        return {"POS": np.nan, "POS_CI_LOW": np.nan, "POS_CI_HIGH": np.nan, "POS_SPLINE": np.nan,
                "N": n, "N_UNIQUE_MONTHS": n_unique_months, "N_YEARS": n_years,
                "SEASONAL_P": np.nan, "N_COMPARABLE_PEAKS": np.nan, "FLAGS": ";".join(flags)}
    curve = harmonic_curve_on_grid(beta, grid)
    shape = find_pos_and_shape(grid, curve)
    pos = shape["pos"]

    if p_seasonal is not None and not np.isnan(p_seasonal) and p_seasonal >= 0.05:
        flags.append("FLAT_SEASONAL_CYCLE")
    if shape["n_comparable_peaks"] >= 2:
        flags.append("MULTIMODAL")
    if n_years < 4:
        flags.append("LOW_YEAR_REPLICATION")

    # spline sensitivity check (on monthly climatological means)
    month_means = g.groupby("MONTH")[value_col].mean()
    cs = fit_periodic_spline(month_means)
    pos_spline = np.nan
    if cs is not None:
        curve_s = cs(grid % 12 + (grid // 12) * 0)  # grid already in [0,12)
        pos_spline = float(grid[int(np.argmax(cs(grid)))] % 12) or 12.0
        if not np.isnan(pos_spline) and abs(circular_diff(pos, pos_spline)) > 1.5:
            flags.append("SMOOTHER_DISAGREEMENT")

    # year block-bootstrap CI
    boot_pos = []
    for _ in range(n_boot):
        pick_years = rng.choice(unique_years, size=n_years, replace=True)
        mask = np.isin(years, pick_years)
        if mask.sum() < 6 or len(np.unique(months[mask])) < 4:
            continue
        b_beta, _ = fit_harmonic(months[mask], values[mask])
        if b_beta is None:
            continue
        b_curve = harmonic_curve_on_grid(b_beta, grid)
        boot_pos.append(float(grid[int(np.argmax(b_curve))] % 12) or 12.0)
    if len(boot_pos) >= 50:
        # circular CI: recenter around point estimate before taking percentiles
        rel = np.array([circular_diff(bp, pos) for bp in boot_pos])
        lo, hi = np.percentile(rel, [2.5, 97.5])
        ci_low, ci_high = (pos + lo) % 12 or 12.0, (pos + hi) % 12 or 12.0
    else:
        ci_low, ci_high = np.nan, np.nan

    return {"POS": pos, "POS_CI_LOW": ci_low, "POS_CI_HIGH": ci_high, "POS_SPLINE": pos_spline,
            "N": n, "N_UNIQUE_MONTHS": n_unique_months, "N_YEARS": n_years,
            "SEASONAL_P": p_seasonal, "N_COMPARABLE_PEAKS": shape["n_comparable_peaks"],
            "FLAGS": ";".join(flags) if flags else "OK"}


def build_tower_pos_table(df_master: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for (site, product), g in df_master.groupby(["SITE_ID", "SIF_PRODUCT"]):
        base = {"SITE_ID": site, "SIF_PRODUCT": product, "HEMI": g["HEMI"].iloc[0],
                "IGBP": g["IGBP_GPP_TABLE"].iloc[0], "LATITUDE": g["LATITUDE"].iloc[0],
                "LONGITUDE": g["LONGITUDE"].iloc[0]}
        sif = estimate_pos_tower(g, "SIF_VALUE", rng)
        gpp = estimate_pos_tower(g, "GPP_VALUE", rng)
        row = dict(base)
        for k, v in sif.items():
            row[f"SIF_{k}"] = v
        for k, v in gpp.items():
            row[f"GPP_{k}"] = v
        row["DELTA_POS"] = circular_diff(row["SIF_POS"], row["GPP_POS"])
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Pooled (product x hemisphere) POS estimation -- for Figure A
# --------------------------------------------------------------------------
def estimate_pos_pooled(sub: pd.DataFrame, value_col: str, rng: np.random.Generator,
                         n_boot: int = N_BOOT) -> dict:
    """sub already filtered to one product x hemisphere. Pools the MEDIAN across
    towers for each calendar month (matches this project's monthly_stats() pooling
    convention), fits the harmonic curve to that 12-point pooled climatology, and
    bootstraps by resampling TOWERS (tower-block bootstrap, this project's standard
    convention for pooled-level uncertainty)."""
    towers = sub["SITE_ID"].unique()
    n_towers = len(towers)
    month_med = sub.groupby("MONTH")[value_col].median().reindex(range(1, 13))
    n_unique_months = month_med.notna().sum()

    flags = []
    if n_towers < MIN_TOWERS_POOLED or n_unique_months < MIN_UNIQUE_MONTHS:
        flags.append("INSUFFICIENT_COVERAGE")
        return {"POS": np.nan, "POS_CI_LOW": np.nan, "POS_CI_HIGH": np.nan,
                "N_TOWERS": n_towers, "N_UNIQUE_MONTHS": int(n_unique_months),
                "SEASONAL_P": np.nan, "curve_grid": None, "curve_values": None, "FLAGS": ";".join(flags)}

    present = month_med.dropna()
    beta, p_seasonal = fit_harmonic(present.index.to_numpy(), present.to_numpy())
    grid = np.linspace(1e-6, 12, GRID_N)
    if beta is None:
        flags.append("INSUFFICIENT_COVERAGE")
        return {"POS": np.nan, "POS_CI_LOW": np.nan, "POS_CI_HIGH": np.nan,
                "N_TOWERS": n_towers, "N_UNIQUE_MONTHS": int(n_unique_months),
                "SEASONAL_P": np.nan, "curve_grid": None, "curve_values": None, "FLAGS": ";".join(flags)}
    curve = harmonic_curve_on_grid(beta, grid)
    shape = find_pos_and_shape(grid, curve)
    pos = shape["pos"]
    if p_seasonal is not None and not np.isnan(p_seasonal) and p_seasonal >= 0.05:
        flags.append("FLAT_SEASONAL_CYCLE")
    if shape["n_comparable_peaks"] >= 2:
        flags.append("MULTIMODAL")

    boot_pos = []
    for _ in range(n_boot):
        pick = rng.choice(towers, size=n_towers, replace=True)
        bsub = sub[sub["SITE_ID"].isin(pick)]
        bmonth_med = bsub.groupby("MONTH")[value_col].median().dropna()
        if len(bmonth_med) < 6:
            continue
        b_beta, _ = fit_harmonic(bmonth_med.index.to_numpy(), bmonth_med.to_numpy())
        if b_beta is None:
            continue
        b_curve = harmonic_curve_on_grid(b_beta, grid)
        boot_pos.append(float(grid[int(np.argmax(b_curve))] % 12) or 12.0)
    if len(boot_pos) >= 50:
        rel = np.array([circular_diff(bp, pos) for bp in boot_pos])
        lo, hi = np.percentile(rel, [2.5, 97.5])
        ci_low, ci_high = (pos + lo) % 12 or 12.0, (pos + hi) % 12 or 12.0
    else:
        ci_low, ci_high = np.nan, np.nan

    return {"POS": pos, "POS_CI_LOW": ci_low, "POS_CI_HIGH": ci_high,
            "N_TOWERS": n_towers, "N_UNIQUE_MONTHS": int(n_unique_months),
            "SEASONAL_P": p_seasonal, "curve_grid": grid, "curve_values": curve,
            "FLAGS": ";".join(flags) if flags else "OK"}


def estimate_pos_from_climatology(month_series: pd.Series, min_unique_months: int = 6) -> dict:
    """Generic POS estimator for any per-calendar-month climatology series (index =
    month 1-12, need not be complete). Used for environmental-driver POS (Figure D),
    where a full year-level bootstrap is not needed for what is explicitly a
    supporting/secondary analysis -- point estimate + shape flags only."""
    s = month_series.dropna()
    if len(s) < min_unique_months:
        return {"POS": np.nan, "SEASONAL_P": np.nan, "FLAGS": "INSUFFICIENT_COVERAGE"}
    beta, p_seasonal = fit_harmonic(s.index.to_numpy(dtype=float), s.to_numpy(dtype=float))
    if beta is None:
        return {"POS": np.nan, "SEASONAL_P": np.nan, "FLAGS": "INSUFFICIENT_COVERAGE"}
    grid = np.linspace(1e-6, 12, GRID_N)
    curve = harmonic_curve_on_grid(beta, grid)
    shape = find_pos_and_shape(grid, curve)
    flags = []
    if p_seasonal is not None and not np.isnan(p_seasonal) and p_seasonal >= 0.05:
        flags.append("FLAT_SEASONAL_CYCLE")
    if shape["n_comparable_peaks"] >= 2:
        flags.append("MULTIMODAL")
    return {"POS": shape["pos"], "SEASONAL_P": p_seasonal, "FLAGS": ";".join(flags) if flags else "OK"}


def bootstrap_delta_pos_pooled(sub: pd.DataFrame, rng: np.random.Generator, n_boot: int = N_BOOT):
    """Tower-block bootstrap directly on DELTA_POS = POS_SIF - POS_GPP for one
    product x hemisphere pooled group, for the Figure-A annotation's own CI (distinct
    from each variable's own marginal CI above -- this is the CI on the DIFFERENCE)."""
    towers = sub["SITE_ID"].unique()
    n_towers = len(towers)
    grid = np.linspace(1e-6, 12, GRID_N)

    def pos_of(bsub, col):
        bmonth_med = bsub.groupby("MONTH")[col].median().dropna()
        if len(bmonth_med) < 6:
            return np.nan
        b_beta, _ = fit_harmonic(bmonth_med.index.to_numpy(), bmonth_med.to_numpy())
        if b_beta is None:
            return np.nan
        b_curve = harmonic_curve_on_grid(b_beta, grid)
        return float(grid[int(np.argmax(b_curve))] % 12) or 12.0

    point_sif = pos_of(sub, "SIF_VALUE")
    point_gpp = pos_of(sub, "GPP_VALUE")
    point_delta = circular_diff(point_sif, point_gpp)

    deltas = []
    for _ in range(n_boot):
        pick = rng.choice(towers, size=n_towers, replace=True)
        bsub = sub[sub["SITE_ID"].isin(pick)]
        ps, pg = pos_of(bsub, "SIF_VALUE"), pos_of(bsub, "GPP_VALUE")
        d = circular_diff(ps, pg)
        if not np.isnan(d):
            deltas.append(d)
    if len(deltas) >= 50:
        ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    else:
        ci_low, ci_high = np.nan, np.nan
    return point_delta, ci_low, ci_high
