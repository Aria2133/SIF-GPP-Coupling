"""
Part I, Steps 3-5 -- Partial Spearman coupling, driver-specific attenuation, and
site-level partial correlations for the Advanced Coupling Analysis.

All partial-Spearman computations use the correct rank-based procedure:
  1. rank-transform every variable involved (outcome pair + all controlling
     covariates) with scipy.stats.rankdata;
  2. OLS-regress ranked SIF_ANOM on ranked covariates (+ intercept) -> residuals;
  3. OLS-regress ranked GPP_ANOM on ranked covariates (+ intercept) -> residuals;
  4. Pearson-correlate the two residual sets. This equals the partial Spearman
     correlation of SIF_ANOM and GPP_ANOM given the covariates.
This is never a raw-value partial Pearson correlation mislabeled as "partial
Spearman" -- ranks are taken first, in every step, including for the covariates.

Language throughout is explicitly associational ("partial correlation",
"descriptive/associational attenuation") -- never "causal effect" / "controls
for X causally".

Outputs (02_partial_correlation/):
  - partial_correlation_product_summary.csv   (Step 3, required)
  - driver_specific_attenuation.csv           (Step 4, required)
  - partial_correlation_site_summary.csv      (Step 5, required; also carries the
    site-level per-driver Delta_rho_D columns needed for Figures PC2/PC3)

Author: project authors
Run date: 2026-09-01
"""
from __future__ import annotations
# Original script filename (sanitized): 03_partial_correlation.py; public name: 03_partial_correlation.py
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
from scipy.stats import rankdata, spearmanr, pearsonr, norm

ROOT = Path(PROJECT_ROOT)
INVENTORY_CSV = ROOT / "SIF_EC_Advanced_Coupling_Analysis" / "01_input_audit" / "advanced_analysis_input_inventory.csv"
OUT_DIR = ROOT / "SIF_EC_Advanced_Coupling_Analysis" / "02_partial_correlation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRODUCTS = ["GOSIF", "GOME-2", "TROPOSIF", "OCO-3"]
DRIVERS = ["VPD", "SWC", "TAIR", "SW_IN"]
DRIVER_ANOM = {d: f"{d}_ANOM" for d in DRIVERS}

SITE_N_THRESHOLDS = [18, 24, 36]
SITE_N_PRIMARY = 24


# ----------------------------------------------------------------------------------
# Core partial-Spearman machinery
# ----------------------------------------------------------------------------------

def rank_ols_residuals(y: np.ndarray, X_cols: list[np.ndarray]) -> np.ndarray:
    """Rank-transform y and each column of X, OLS-regress ranked y on ranked X (+
    intercept), return residuals. If X_cols is empty, returns the (mean-centered)
    ranks of y themselves (degenerate case = ordinary Spearman)."""
    ry = rankdata(y).astype(float)
    n = len(y)
    if len(X_cols) == 0:
        return ry - ry.mean()
    rX = np.column_stack([rankdata(c).astype(float) for c in X_cols])
    design = np.column_stack([np.ones(n), rX])
    beta, *_ = np.linalg.lstsq(design, ry, rcond=None)
    resid = ry - design @ beta
    return resid


def partial_spearman(y: np.ndarray, x: np.ndarray, covariates: dict[str, np.ndarray]) -> tuple[float, float, int]:
    """Correct partial-Spearman: rank y, x, and every covariate; residualize ranked
    y and ranked x on ranked covariates via OLS; Pearson-correlate the residuals.
    Returns (rho, p, n)."""
    n = len(y)
    cov_arrays = list(covariates.values())
    resid_y = rank_ols_residuals(y, cov_arrays)
    resid_x = rank_ols_residuals(x, cov_arrays)
    if np.std(resid_y) == 0 or np.std(resid_x) == 0:
        return np.nan, np.nan, n
    rho, p = pearsonr(resid_x, resid_y)
    return float(rho), float(p), n


def fisher_ci(rho: float, n: int, n_covariates: int, alpha: float = 0.05) -> tuple[float, float]:
    """95% CI for a (partial) correlation via Fisher z-transform. SE uses the
    partial-correlation degrees-of-freedom adjustment: se = 1/sqrt(n - k - 3),
    where k = number of controlling variables (0 for an ordinary correlation)."""
    if not np.isfinite(rho) or n - n_covariates - 3 <= 0:
        return np.nan, np.nan
    z = np.arctanh(np.clip(rho, -0.999999, 0.999999))
    se = 1.0 / np.sqrt(n - n_covariates - 3)
    zcrit = norm.ppf(1 - alpha / 2)
    lo, hi = z - zcrit * se, z + zcrit * se
    return float(np.tanh(lo)), float(np.tanh(hi))


def complete_case(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.dropna(subset=cols).copy()


# ----------------------------------------------------------------------------------
# Step 3: product-level ordinary + partial Spearman coupling
# ----------------------------------------------------------------------------------

def step3_product_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    anom_cols = ["SIF_ANOM", "GPP_ANOM"] + [DRIVER_ANOM[d] for d in DRIVERS]
    raw_cols = ["SIF_VALUE_RAW", "GPP_VALUE_RAW"]

    for product in PRODUCTS:
        sub = df[df["SIF_PRODUCT"] == product]

        # --- RAW (non-deseasonalized) comparison: ordinary Spearman on raw values ---
        raw = complete_case(sub, raw_cols)
        if len(raw) >= 5:
            rho_raw, p_raw = spearmanr(raw["SIF_VALUE_RAW"], raw["GPP_VALUE_RAW"])
            n_raw = len(raw)
            towers_raw = raw["SITE_ID"].nunique()
            lo_raw, hi_raw = fisher_ci(rho_raw, n_raw, 0)
        else:
            rho_raw = p_raw = lo_raw = hi_raw = np.nan
            n_raw, towers_raw = len(raw), raw["SITE_ID"].nunique()
        rows.append({
            "SIF_PRODUCT": product, "METHOD": "RAW_VALUE_SPEARMAN_ORDINARY",
            "N_SITEMONTHS": n_raw, "N_TOWERS": towers_raw,
            "RHO": rho_raw, "P_VALUE": p_raw, "CI_LOW": lo_raw, "CI_HIGH": hi_raw,
            "N_COVARIATES": 0, "NOTE": "Non-deseasonalized SIF_VALUE vs GPP_VALUE, explicit raw-vs-anomaly comparison.",
        })

        # --- A. Ordinary Spearman coupling on anomalies ---
        cc = complete_case(sub, ["SIF_ANOM", "GPP_ANOM"])
        n_a = len(cc)
        towers_a = cc["SITE_ID"].nunique()
        if n_a >= 5:
            rho_a, p_a = spearmanr(cc["SIF_ANOM"], cc["GPP_ANOM"])
            lo_a, hi_a = fisher_ci(rho_a, n_a, 0)
        else:
            rho_a = p_a = lo_a = hi_a = np.nan
        rows.append({
            "SIF_PRODUCT": product, "METHOD": "ANOMALY_SPEARMAN_ORDINARY",
            "N_SITEMONTHS": n_a, "N_TOWERS": towers_a,
            "RHO": rho_a, "P_VALUE": p_a, "CI_LOW": lo_a, "CI_HIGH": hi_a,
            "N_COVARIATES": 0, "NOTE": "Harmonic-residual SIF_ANOM vs GPP_ANOM, no covariates.",
        })

        # --- B. Partial Spearman controlling for all 4 driver anomalies ---
        full_cc = complete_case(sub, anom_cols)
        n_p = len(full_cc)
        towers_p = full_cc["SITE_ID"].nunique()
        if n_p >= 10:
            covs = {d: full_cc[DRIVER_ANOM[d]].values for d in DRIVERS}
            rho_p, p_p, _ = partial_spearman(full_cc["GPP_ANOM"].values, full_cc["SIF_ANOM"].values, covs)
            lo_p, hi_p = fisher_ci(rho_p, n_p, len(DRIVERS))
            # ordinary rho on this SAME complete-case subset, for an apples-to-apples diff
            rho_a_samesub, _ = spearmanr(full_cc["SIF_ANOM"], full_cc["GPP_ANOM"])
        else:
            rho_p = p_p = lo_p = hi_p = rho_a_samesub = np.nan
        rows.append({
            "SIF_PRODUCT": product, "METHOD": "ANOMALY_SPEARMAN_PARTIAL_ALL4",
            "N_SITEMONTHS": n_p, "N_TOWERS": towers_p,
            "RHO": rho_p, "P_VALUE": p_p, "CI_LOW": lo_p, "CI_HIGH": hi_p,
            "N_COVARIATES": len(DRIVERS),
            "RHO_MINUS_ORDINARY_SAME_SUBSET": (rho_p - rho_a_samesub) if np.isfinite(rho_p) else np.nan,
            "NOTE": "Partial Spearman(SIF_ANOM, GPP_ANOM | VPD_ANOM, SWC_ANOM, TAIR_ANOM, SW_IN_ANOM); ordinary rho on the SAME complete-case subset given in RHO_MINUS_ORDINARY_SAME_SUBSET's baseline.",
        })

    out = pd.DataFrame(rows)
    return out


# ----------------------------------------------------------------------------------
# Step 4: driver-specific attenuation (product level)
# ----------------------------------------------------------------------------------

def step4_driver_attenuation(df: pd.DataFrame) -> pd.DataFrame:
    anom_cols = ["SIF_ANOM", "GPP_ANOM"] + [DRIVER_ANOM[d] for d in DRIVERS]
    rows = []
    for product in PRODUCTS:
        sub = df[df["SIF_PRODUCT"] == product]
        # Common complete-case sample across ALL variants for a fair, apples-to-apples comparison
        cc = complete_case(sub, anom_cols)
        n = len(cc)
        n_towers = cc["SITE_ID"].nunique()
        if n < 10:
            for D in DRIVERS:
                rows.append({
                    "SIF_PRODUCT": product, "DRIVER": D, "N_SITEMONTHS": n, "N_TOWERS": n_towers,
                    "RHO_UNADJUSTED": np.nan, "RHO_CONTROL_D": np.nan, "DELTA_RHO_D": np.nan,
                    "RHO_CONTROL_ALL": np.nan, "RHO_ALL_EXCEPT_D": np.nan, "DELTA_LOO_D": np.nan,
                })
            continue

        sif = cc["SIF_ANOM"].values
        gpp = cc["GPP_ANOM"].values
        rho_unadjusted, _ = spearmanr(sif, gpp)

        covs_all = {d: cc[DRIVER_ANOM[d]].values for d in DRIVERS}
        rho_control_all, _, _ = partial_spearman(gpp, sif, covs_all)

        for D in DRIVERS:
            covs_D = {D: cc[DRIVER_ANOM[D]].values}
            rho_control_D, _, _ = partial_spearman(gpp, sif, covs_D)
            delta_D = rho_unadjusted - rho_control_D

            others = {d: cc[DRIVER_ANOM[d]].values for d in DRIVERS if d != D}
            rho_all_except_D, _, _ = partial_spearman(gpp, sif, others)
            delta_loo = rho_all_except_D - rho_control_all

            rows.append({
                "SIF_PRODUCT": product, "DRIVER": D, "N_SITEMONTHS": n, "N_TOWERS": n_towers,
                "RHO_UNADJUSTED": rho_unadjusted, "RHO_CONTROL_D": rho_control_D,
                "DELTA_RHO_D": delta_D,
                "RHO_CONTROL_ALL": rho_control_all, "RHO_ALL_EXCEPT_D": rho_all_except_D,
                "DELTA_LOO_D": delta_loo,
            })
    out = pd.DataFrame(rows)
    out["NOTE"] = (
        "DELTA_RHO_D = RHO_UNADJUSTED - RHO_CONTROL_D: descriptive/associational attenuation of the SIF-GPP "
        "coupling when driver D alone is partialed out. Never a causal effect. DELTA_LOO_D = RHO_ALL_EXCEPT_D - "
        "RHO_CONTROL_ALL: leave-one-out check of whether adding D back into the 4-covariate control set materially "
        "changes the residual coupling (near-zero = D adds little beyond the other 3 drivers)."
    )
    return out


# ----------------------------------------------------------------------------------
# Step 5: site-level partial correlations (+ site-level driver attenuation for
# Figures PC1/PC2/PC3)
# ----------------------------------------------------------------------------------

def step5_site_summary(df: pd.DataFrame) -> pd.DataFrame:
    anom_cols = ["SIF_ANOM", "GPP_ANOM"] + [DRIVER_ANOM[d] for d in DRIVERS]
    rows = []
    for (site, product), g in df[df["SIF_PRODUCT"].isin(PRODUCTS)].groupby(["SITE_ID", "SIF_PRODUCT"]):
        cc = complete_case(g, anom_cols)
        n = len(cc)
        lat = g["LATITUDE"].iloc[0] if "LATITUDE" in g.columns else np.nan
        lon = g["LONGITUDE"].iloc[0] if "LONGITUDE" in g.columns else np.nan

        row = {
            "SITE_ID": site, "SIF_PRODUCT": product, "LATITUDE": lat, "LONGITUDE": lon,
            "N_SITEMONTHS": n,
            "MEETS_N18": n >= 18, "MEETS_N24_PRIMARY": n >= SITE_N_PRIMARY, "MEETS_N36": n >= 36,
        }

        if n < 18:
            # Too few months for a 4-covariate partial correlation -- flag, do not compute.
            row.update({
                "RHO_UNADJUSTED": np.nan, "RHO_CONTROL_ALL": np.nan, "P_CONTROL_ALL": np.nan,
                "CI_LOW": np.nan, "CI_HIGH": np.nan,
                "RHO_CONTROL_VPD": np.nan, "RHO_CONTROL_SWC": np.nan, "RHO_CONTROL_TAIR": np.nan, "RHO_CONTROL_SW_IN": np.nan,
                "DELTA_RHO_VPD": np.nan, "DELTA_RHO_SWC": np.nan, "DELTA_RHO_TAIR": np.nan, "DELTA_RHO_SW_IN": np.nan,
                "DOMINANT_DRIVER": None, "DOMINANT_DELTA_RHO": np.nan,
                "COMPUTED": False, "OMIT_REASON": "N<18: insufficient for a 4-covariate partial correlation",
            })
            rows.append(row)
            continue

        sif = cc["SIF_ANOM"].values
        gpp = cc["GPP_ANOM"].values
        rho_unadjusted, _ = spearmanr(sif, gpp)

        covs_all = {d: cc[DRIVER_ANOM[d]].values for d in DRIVERS}
        rho_control_all, p_control_all, _ = partial_spearman(gpp, sif, covs_all)
        ci_lo, ci_hi = fisher_ci(rho_control_all, n, len(DRIVERS))

        per_driver_control = {}
        per_driver_delta = {}
        for D in DRIVERS:
            rho_D, _, _ = partial_spearman(gpp, sif, {D: cc[DRIVER_ANOM[D]].values})
            per_driver_control[D] = rho_D
            per_driver_delta[D] = rho_unadjusted - rho_D if np.isfinite(rho_D) else np.nan

        finite_deltas = {d: v for d, v in per_driver_delta.items() if np.isfinite(v)}
        if finite_deltas:
            dom_driver = max(finite_deltas, key=lambda d: abs(finite_deltas[d]))
            dom_delta = finite_deltas[dom_driver]
        else:
            dom_driver, dom_delta = None, np.nan

        row.update({
            "RHO_UNADJUSTED": rho_unadjusted,
            "RHO_CONTROL_ALL": rho_control_all, "P_CONTROL_ALL": p_control_all,
            "CI_LOW": ci_lo, "CI_HIGH": ci_hi,
            "RHO_CONTROL_VPD": per_driver_control["VPD"], "RHO_CONTROL_SWC": per_driver_control["SWC"],
            "RHO_CONTROL_TAIR": per_driver_control["TAIR"], "RHO_CONTROL_SW_IN": per_driver_control["SW_IN"],
            "DELTA_RHO_VPD": per_driver_delta["VPD"], "DELTA_RHO_SWC": per_driver_delta["SWC"],
            "DELTA_RHO_TAIR": per_driver_delta["TAIR"], "DELTA_RHO_SW_IN": per_driver_delta["SW_IN"],
            "DOMINANT_DRIVER": dom_driver, "DOMINANT_DELTA_RHO": dom_delta,
            "COMPUTED": True, "OMIT_REASON": None,
        })
        rows.append(row)

    out = pd.DataFrame(rows)
    return out


def main():
    print("Loading joined inventory ...")
    df = pd.read_csv(INVENTORY_CSV, low_memory=False)
    df = df[df["SIF_PRODUCT"].isin(PRODUCTS)].copy()

    print("Step 3: product-level ordinary + partial Spearman coupling ...")
    step3 = step3_product_summary(df)
    p3 = OUT_DIR / "partial_correlation_product_summary.csv"
    step3.to_csv(p3, index=False)
    print(f"  Wrote {p3}  shape={step3.shape}")
    print(step3[["SIF_PRODUCT", "METHOD", "N_SITEMONTHS", "N_TOWERS", "RHO", "P_VALUE"]].to_string(index=False))

    print("\nStep 4: driver-specific attenuation ...")
    step4 = step4_driver_attenuation(df)
    p4 = OUT_DIR / "driver_specific_attenuation.csv"
    step4.to_csv(p4, index=False)
    print(f"  Wrote {p4}  shape={step4.shape}")
    print(step4[["SIF_PRODUCT", "DRIVER", "N_SITEMONTHS", "RHO_UNADJUSTED", "RHO_CONTROL_D", "DELTA_RHO_D", "DELTA_LOO_D"]].to_string(index=False))

    print("\nStep 5: site-level partial correlations ...")
    step5 = step5_site_summary(df)
    p5 = OUT_DIR / "partial_correlation_site_summary.csv"
    step5.to_csv(p5, index=False)
    n_computed = int(step5["COMPUTED"].sum())
    n_primary = int(step5["MEETS_N24_PRIMARY"].sum())
    print(f"  Wrote {p5}  shape={step5.shape}  (computed rows: {n_computed}, N>=24 primary: {n_primary})")


if __name__ == "__main__":
    main()
