"""
Shared library for YEAR-SPECIFIC Peak-Of-Season (POS) estimation.

This extends (does not replace) the climatological POS method already validated in
PHENOLOGY_POS_REVISION/pos_lib.py (1st+2nd order harmonic regression, the same
functional form as this project's accepted deseasonalization method). Here the
harmonic curve is fit to ONE calendar year's monthly values at a time (SITE_ID x YEAR
x SIF_PRODUCT for SIF/GPP; SITE_ID x YEAR for environmental drivers, which do not
depend on SIF product), because the mechanism question in this analysis is about
YEAR-TO-YEAR variation, which a climatological (all-years-pooled) POS cannot resolve.

KEY DIFFERENCE FROM THE CLIMATOLOGICAL METHOD (stated explicitly, not hidden):
A single year gives at most 12 monthly points for a 5-parameter harmonic fit, so:
  - the reliability floor is relaxed from 8 to >=6 distinct months present (a full
    year's data is harder to obtain in ONE year for irregular products; this is
    reported as MIN_UNIQUE_MONTHS_YEAR=6, distinct from the climatological method's
    MIN_UNIQUE_MONTHS=8 -- never conflated),
  - there are no repeated years to block-bootstrap over, so uncertainty on a
    single-year POS instead comes from a delete-one-month JACKKNIFE (refit the
    harmonic curve with each one of the available months removed in turn; the spread
    of the resulting POS estimates gives a jackknife standard error and 95% CI). This
    is a weaker uncertainty estimate than the multi-year bootstrap and is labeled as
    such everywhere it is used.
  - a LOW_DF flag is raised when residual degrees of freedom (n_months - 5) < 4,
    signaling a fit with little statistical power even though a point estimate can
    still be computed.

Circular handling, flat-seasonality/multimodal detection, and the underlying harmonic
fitting routine are reused directly from pos_lib.py (imported, not reimplemented) so
both the climatological and year-specific POS numbers in this project are produced by
the exact same core fitting code.
"""
from __future__ import annotations
# Original script filename (sanitized): lib_year_specific_pos.py; public name: lib_year_specific_pos.py
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

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PHEN_DIR = Path(_os.path.join(PROJECT_ROOT, 'All_the_Visualizations_REVISED', 'EXACT_REQUESTS_ONLY', 'PHENOLOGY_POS_REVISION'))
sys.path.insert(0, str(PHEN_DIR))
import pos_lib as pl  # noqa: E402  -- reused unmodified: fit_harmonic, harmonic_curve_on_grid,
                        # circular_diff, find_pos_and_shape, GRID_N

MIN_UNIQUE_MONTHS_YEAR = 6   # relaxed from pos_lib's climatological MIN_UNIQUE_MONTHS=8
LOW_DF_THRESHOLD = 4          # residual df = n_months - 5 (5 harmonic params)


def estimate_pos_one_year(months: np.ndarray, values: np.ndarray) -> dict:
    """Fit the harmonic curve to ONE year's monthly values and return POS + a
    delete-one-month jackknife 95% CI + shape/reliability flags. months/values must
    already be restricted to a single SITE_ID x YEAR (x SIF_PRODUCT, if applicable)."""
    n = len(values)
    n_unique_months = len(np.unique(months))
    flags = []

    if n_unique_months < MIN_UNIQUE_MONTHS_YEAR:
        return dict(POS=np.nan, POS_CI_LOW=np.nan, POS_CI_HIGH=np.nan, N_MONTHS=n_unique_months,
                    SEASONAL_P=np.nan, N_COMPARABLE_PEAKS=np.nan, FLAGS="INSUFFICIENT_COVERAGE")

    beta, p_seasonal = pl.fit_harmonic(months, values)
    if beta is None:
        return dict(POS=np.nan, POS_CI_LOW=np.nan, POS_CI_HIGH=np.nan, N_MONTHS=n_unique_months,
                    SEASONAL_P=np.nan, N_COMPARABLE_PEAKS=np.nan, FLAGS="INSUFFICIENT_COVERAGE")

    grid = np.linspace(1e-6, 12, pl.GRID_N)
    curve = pl.harmonic_curve_on_grid(beta, grid)
    shape = pl.find_pos_and_shape(grid, curve)
    pos = shape["pos"]

    if p_seasonal is not None and not np.isnan(p_seasonal) and p_seasonal >= 0.05:
        flags.append("FLAT_SEASONAL_CYCLE")
    if shape["n_comparable_peaks"] >= 2:
        flags.append("MULTIMODAL")
    if (n - 5) < LOW_DF_THRESHOLD:
        flags.append("LOW_DF")

    # delete-one-month jackknife CI
    uniq_months = np.unique(months)
    jack_pos = []
    for m in uniq_months:
        mask = months != m
        if len(np.unique(months[mask])) < 5:  # need >=5 distinct months left to refit 5 params meaningfully
            continue
        b, _ = pl.fit_harmonic(months[mask], values[mask])
        if b is None:
            continue
        c = pl.harmonic_curve_on_grid(b, grid)
        jack_pos.append(float(grid[int(np.argmax(c))] % 12) or 12.0)

    if len(jack_pos) >= 4:
        rel = np.array([pl.circular_diff(jp, pos) for jp in jack_pos])
        k = len(rel)
        jack_mean = rel.mean()
        jack_se = np.sqrt((k - 1) / k * np.sum((rel - jack_mean) ** 2))
        ci_low = (pos - 1.96 * jack_se) % 12 or 12.0
        ci_high = (pos + 1.96 * jack_se) % 12 or 12.0
    else:
        ci_low, ci_high = np.nan, np.nan

    return dict(POS=pos, POS_CI_LOW=ci_low, POS_CI_HIGH=ci_high, N_MONTHS=n_unique_months,
                SEASONAL_P=p_seasonal, N_COMPARABLE_PEAKS=shape["n_comparable_peaks"],
                FLAGS=";".join(flags) if flags else "OK")


def jackknife_delta_pos(months: np.ndarray, sif_vals: np.ndarray, gpp_vals: np.ndarray,
                         sif_pos: float, gpp_pos: float) -> tuple[float, float]:
    """Paired delete-one-month jackknife CI for DELTA_POS = SIF_POS - GPP_POS, refitting
    BOTH curves on the same reduced month set each time (so the same missing month
    affects both fits identically -- a fair paired jackknife)."""
    if np.isnan(sif_pos) or np.isnan(gpp_pos):
        return np.nan, np.nan
    grid = np.linspace(1e-6, 12, pl.GRID_N)
    point_delta = pl.circular_diff(sif_pos, gpp_pos)
    uniq_months = np.unique(months)
    jack_delta = []
    for m in uniq_months:
        mask = months != m
        if len(np.unique(months[mask])) < 5:
            continue
        b_sif, _ = pl.fit_harmonic(months[mask], sif_vals[mask])
        b_gpp, _ = pl.fit_harmonic(months[mask], gpp_vals[mask])
        if b_sif is None or b_gpp is None:
            continue
        p_sif = float(grid[int(np.argmax(pl.harmonic_curve_on_grid(b_sif, grid)))] % 12) or 12.0
        p_gpp = float(grid[int(np.argmax(pl.harmonic_curve_on_grid(b_gpp, grid)))] % 12) or 12.0
        jack_delta.append(pl.circular_diff(p_sif, p_gpp))
    if len(jack_delta) < 4:
        return np.nan, np.nan
    rel = np.array([pl.circular_diff(jd, point_delta) for jd in jack_delta])
    k = len(rel)
    jack_mean = rel.mean()
    jack_se = np.sqrt((k - 1) / k * np.sum((rel - jack_mean) ** 2))
    return point_delta - 1.96 * jack_se, point_delta + 1.96 * jack_se


def classify_delta(delta: float, ci_low: float, ci_high: float) -> str:
    """EARLIER_SIF / EARLIER_GPP only when the CI does not include zero -- otherwise
    SYNCHRONOUS_OR_UNCERTAIN (covers both genuinely-synchronous and simply-uncertain
    cases, since with year-specific single-season data the two are often
    indistinguishable, and the instruction is explicit: do not classify small
    differences as real if uncertainty includes zero)."""
    if np.isnan(delta):
        return "INSUFFICIENT_DATA"
    if np.isnan(ci_low) or np.isnan(ci_high):
        return "SYNCHRONOUS_OR_UNCERTAIN"
    if ci_low <= 0 <= ci_high:
        return "SYNCHRONOUS_OR_UNCERTAIN"
    return "EARLIER_SIF" if delta < 0 else "EARLIER_GPP"
