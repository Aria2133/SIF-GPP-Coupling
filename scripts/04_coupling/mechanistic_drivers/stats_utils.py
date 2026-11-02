"""
Shared statistics helpers for the mechanistic SIF-GPP paper analysis.

Conventions follow the previously audited SIF_EC_SIF_GPP_Relationship_Analysis
Phase 2 methods note:
  - Pearson r via scipy.stats.pearsonr, Spearman rho via scipy.stats.spearmanr.
  - R^2 for a single predictor + intercept == Pearson r^2 (verified identity
    in the prior independent audit to 1e-6); OLS slope/intercept via
    scipy.stats.linregress.
  - Robust fits use statsmodels RLM with Huber's T norm (MAD scale), chosen
    over Theil-Sen for the same computational-cost reasons documented in the
    prior methods note.
  - Bootstrap CIs are percentile intervals from 1000 resamples.
    * Product/pooled-level bootstrap resamples SITES with replacement
      (cluster/block bootstrap), keeping every row for each resampled site,
      because site-months within a site are not independent draws. This is a
      slightly more conservative choice than a prior audit that resampled
      cases directly; documented explicitly in the main report.
    * Site-level bootstrap resamples CASES (site-months) with replacement
      within that one site's own N rows, matching the prior audit.
"""
# Original script filename (sanitized): stats_utils.py; public name: stats_utils.py
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
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

RNG_SEED = 12345


def full_pair_stats(x, y, min_n=3):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    out = dict(N=n, PEARSON_R=np.nan, PEARSON_P=np.nan, SPEARMAN_RHO=np.nan,
               SPEARMAN_P=np.nan, OLS_SLOPE=np.nan, OLS_INTERCEPT=np.nan,
               R2=np.nan, RMSE=np.nan, MAE=np.nan, MEDIAN_X=np.nan, MEDIAN_Y=np.nan,
               IQR_X=np.nan, IQR_Y=np.nan)
    if n == 0:
        return out
    out["MEDIAN_X"] = float(np.median(x))
    out["MEDIAN_Y"] = float(np.median(y))
    out["IQR_X"] = float(stats.iqr(x)) if n > 1 else 0.0
    out["IQR_Y"] = float(stats.iqr(y)) if n > 1 else 0.0
    if n < min_n or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return out
    r, p = stats.pearsonr(x, y)
    rho, ps = stats.spearmanr(x, y)
    lr = stats.linregress(x, y)
    yhat = lr.slope * x + lr.intercept
    resid = y - yhat
    out.update(PEARSON_R=r, PEARSON_P=p, SPEARMAN_RHO=rho, SPEARMAN_P=ps,
               OLS_SLOPE=lr.slope, OLS_INTERCEPT=lr.intercept,
               R2=lr.rvalue ** 2, RMSE=float(np.sqrt(np.mean(resid ** 2))),
               MAE=float(np.mean(np.abs(resid))))
    return out


def rlm_fit(x, y, min_n=8):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < min_n or np.std(x) < 1e-12:
        return np.nan, np.nan
    X = sm.add_constant(x)
    try:
        res = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
        return float(res.params[1]), float(res.params[0])
    except Exception:
        return np.nan, np.nan


def _fast_pearson_r(x, y):
    """Vectorized Pearson r, no scipy p-value overhead -- for use inside bootstrap loops only."""
    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.sqrt((xm * xm).sum() * (ym * ym).sum())
    if denom < 1e-12:
        return np.nan
    return float((xm * ym).sum() / denom)


def cluster_bootstrap_ci(df, x_col, y_col, cluster_col, n_boot=1000, min_n=3, seed=RNG_SEED):
    """Resample cluster_col (e.g. SITE_ID) with replacement; recompute r and R2 each time.

    Fast path: pre-extract each cluster's (x, y) as numpy arrays once, then build each
    bootstrap replicate via np.concatenate (order of magnitude faster than pd.concat of
    DataFrames, with an identical resampling scheme).
    """
    rng = np.random.default_rng(seed)
    mask = df[x_col].notna() & df[y_col].notna()
    d = df.loc[mask, [cluster_col, x_col, y_col]]
    clusters = d[cluster_col].unique()
    if len(clusters) < 2:
        return (np.nan, np.nan), (np.nan, np.nan)
    x_by_c = {c: g[x_col].to_numpy(dtype=float) for c, g in d.groupby(cluster_col)}
    y_by_c = {c: g[y_col].to_numpy(dtype=float) for c, g in d.groupby(cluster_col)}
    r_boot = np.empty(n_boot)
    for i in range(n_boot):
        picks = rng.choice(clusters, size=len(clusters), replace=True)
        x_sample = np.concatenate([x_by_c[c] for c in picks])
        y_sample = np.concatenate([y_by_c[c] for c in picks])
        r_boot[i] = _fast_pearson_r(x_sample, y_sample) if len(x_sample) >= min_n else np.nan
    r_boot = r_boot[np.isfinite(r_boot)]
    if len(r_boot) == 0:
        return (np.nan, np.nan), (np.nan, np.nan)
    r2_boot = r_boot ** 2
    r_ci = (float(np.percentile(r_boot, 2.5)), float(np.percentile(r_boot, 97.5)))
    r2_ci = (float(np.percentile(r2_boot, 2.5)), float(np.percentile(r2_boot, 97.5)))
    return r_ci, r2_ci


def case_bootstrap_ci(x, y, n_boot=1000, min_n=3, seed=RNG_SEED):
    """Resample rows (site-months) with replacement within one group's own N rows."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < min_n:
        return (np.nan, np.nan), (np.nan, np.nan)
    r_boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r_boot[i] = _fast_pearson_r(x[idx], y[idx])
    r_boot = r_boot[np.isfinite(r_boot)]
    if len(r_boot) == 0:
        return (np.nan, np.nan), (np.nan, np.nan)
    r2_boot = r_boot ** 2
    r_ci = (float(np.percentile(r_boot, 2.5)), float(np.percentile(r_boot, 97.5)))
    r2_ci = (float(np.percentile(r2_boot, 2.5)), float(np.percentile(r2_boot, 97.5)))
    return r_ci, r2_ci


PRODUCT_COLORS = {
    "GOSIF": "#1B4965",
    "GOME-2": "#5FA8A0",
    "TROPOSIF": "#C9A227",
    "OCO-2": "#D97D54",
    "OCO-3": "#8E5572",
}

SEQ_PALETTE = ["#16324F", "#235789", "#2C7A7B", "#3FA796", "#7CCBA2", "#B8DE6F", "#F6D55C"]

DIVERGING_PALETTE = ["#3B4CC0", "#7B9FF9", "#DCE6F2", "#F7F7F7", "#F5C4A5", "#D6604D", "#8C2D24"]

PRODUCT_ORDER = ["GOSIF", "GOME-2", "TROPOSIF", "OCO-2", "OCO-3"]
