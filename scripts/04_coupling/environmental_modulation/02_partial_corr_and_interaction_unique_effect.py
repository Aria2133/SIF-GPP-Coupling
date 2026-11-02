"""
Part A of the SEM Environmental Control Audit -- "How do you separate the effects of
VPD, Ta, etc. from the SIF-GPP relationship?"

This is NOT SEM. Two standard, independent techniques, both purely associational:
  1. Partial Spearman correlation (rank-then-residualize-then-Pearson-correlate), with a
     tower-block bootstrap 95% CI, quantifying how much of the raw SIF-GPP anomaly
     correlation is shared with each environmental driver ("coupling attenuation").
  2. A SIF x Driver interaction/effect-modification regression testing whether each
     driver changes the STRENGTH of the SIF-GPP relationship.
Neither implies, tests, or requires a causal pathway -- see
SIF_EC_SEM_Environmental_Control_Audit/01_input_audit/Old_SEM_Implementation_Audit.md
for why that is a separate (already-audited) question.

REUSE, verified before use (this script recomputes point estimates independently as a
verification step, then confirms they match the already-reviewed files to the printed
precision -- it does not blindly trust old numbers, nor does it duplicate effort building
things that already exist correctly):
  - Partial-Spearman machinery (rank_ols_residuals / partial_spearman below) is the SAME
    procedure implemented in
    SIF_EC_Advanced_Coupling_Analysis/12_scripts/03_partial_correlation.py
    (verified by reading that script: rank-transform ALL variables including covariates,
    OLS-residualize the ranks, Pearson-correlate the residuals -- this IS partial Spearman,
    not a mislabeled partial Pearson on raw values). rho0, rho_control_D, rho_ALL point
    estimates are recomputed here with that identical procedure directly from the
    inventory CSV and cross-checked against
    SIF_EC_Advanced_Coupling_Analysis/02_partial_correlation/{partial_correlation_product_summary,driver_specific_attenuation}.csv.
    NEW here: tower-block bootstrap 95% CIs (1000 reps), which those files do NOT already
    contain (they only had a Fisher-z analytic CI for rho_ALL, and no CI at all for the
    single-driver rho_control_D) -- this was the genuinely missing piece per Section 10 of
    the task.
  - Interaction/effect-modification models: REUSED, not refit, from
    SIF_EC_SIF_GPP_Final_Verification/05_mechanistic_reanalysis/
    environmental_interactions_product_specific.csv. That file already fits exactly
    `GPP_Z ~ SIF_Z + DRIVER_Z + SIF_Z:DRIVER_Z + C(MONTH)` per product x driver (SIF_Z,
    GPP_Z, DRIVER_Z = within-site z-scored SIF_ANOM/GPP_ANOM/driver-ANOM -- the SAME
    within-site-and-within-product standardization convention already established as
    "correct per the project's stated goal" for the SEM component models per the Old SEM
    audit, Section 3), with the exact MixedLM-random-intercept -> collapse-check
    (cov_re < 1e-6 floor, checked regardless of converged flag) -> warning-text check ->
    cluster-robust-OLS-fallback pattern required by this task. It already covers all
    4 products x 4 drivers (VPD, soil_moisture, temperature, radiation; a 5th driver, ET,
    exists in that file but is out of scope here and is dropped). NOTE ON STANDARDIZATION:
    that file's formula uses SIF_Z/GPP_Z (within-site standardized SIF_ANOM/GPP_ANOM), not
    literally raw SIF_ANOM/GPP_ANOM as in this task's illustrative formula
    `GPP_anom ~ SIF_anom + Driver_z + SIF_anom:Driver_z` -- flagged explicitly here as a
    real (but not incorrect) difference: within-site standardizing the response and
    predictor too, not just the driver, is the same convention this project's SEM used and
    is defensible (it prevents a few high-biomass towers from dominating a product-specific
    fit); it is NOT re-derived from scratch here to avoid duplicating already-reviewed work,
    but the distinction is carried into the Unique-Effect Table's methodology notes so no
    reader mistakes the interaction coefficient's units for raw GPP-anomaly units.

NEW in this script: Section 9 collinearity reported BY PRODUCT (not just pooled) with an
eigenvalue-based condition number added (the existing driver_collinearity.csv had Pearson/
Spearman/VIF pooled and within-site-anomaly-pooled and per-site, but no per-product
breakdown and no condition number) -- and the actual deliverable, the Unique-Effect Table
plus Figures 1-2 (Figure 3 is a documented reuse of the existing PC2 figure).

Main-paper products only: GOSIF, GOME-2, TROPOSIF, OCO-3 (OCO-2 excluded everywhere).
No upstream SIF/footprint/reconstruction pipeline is rerun.

Author: project authors
Run date: 2026-09-08
"""
from __future__ import annotations
# Original script filename (sanitized): 02_partial_corr_and_interaction_unique_effect.py; public name: 02_partial_corr_and_interaction_unique_effect.py
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

import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, pearsonr, norm
from statsmodels.stats.outliers_influence import variance_inflation_factor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path(PROJECT_ROOT)

INVENTORY_CSV = ROOT / "SIF_EC_Advanced_Coupling_Analysis" / "01_input_audit" / "advanced_analysis_input_inventory.csv"
EXISTING_PRODUCT_SUMMARY = ROOT / "SIF_EC_Advanced_Coupling_Analysis" / "02_partial_correlation" / "partial_correlation_product_summary.csv"
EXISTING_DRIVER_ATTEN = ROOT / "SIF_EC_Advanced_Coupling_Analysis" / "02_partial_correlation" / "driver_specific_attenuation.csv"
INTERACTION_CSV = ROOT / "SIF_EC_SIF_GPP_Final_Verification" / "05_mechanistic_reanalysis" / "environmental_interactions_product_specific.csv"
PC2_SRC_DIR = ROOT / "SIF_EC_Advanced_Coupling_Analysis" / "08_geospatial_figures"

AUDIT_ROOT = ROOT / "SIF_EC_SEM_Environmental_Control_Audit"
DIR_COLLIN = AUDIT_ROOT / "02_driver_collinearity"
DIR_PARTIAL = AUDIT_ROOT / "03_partial_correlation"
DIR_INTERACT = AUDIT_ROOT / "04_interaction_models"
DIR_GEO = AUDIT_ROOT / "07_geospatial_effects"
DIR_TABLES = AUDIT_ROOT / "08_tables"
DIR_FIGS = AUDIT_ROOT / "09_figures"
for d in [DIR_COLLIN, DIR_PARTIAL, DIR_INTERACT, DIR_GEO, DIR_TABLES, DIR_FIGS]:
    d.mkdir(parents=True, exist_ok=True)

PRODUCTS = ["GOSIF", "GOME-2", "TROPOSIF", "OCO-3"]
DRIVERS = ["VPD", "SWC", "TAIR", "SW_IN"]
DRIVER_ANOM = {d: f"{d}_ANOM" for d in DRIVERS}
DRIVER_LABEL = {
    "VPD": "VPD",
    "SWC": "Soil moisture (SWC)",
    "TAIR": "Air temperature (Ta)",
    "SW_IN": "Incoming shortwave radiation (radiation proxy for PAR)",
}
DRIVER_SHORT = {"VPD": "VPD", "SWC": "SM", "TAIR": "Ta", "SW_IN": "PAR-radiation"}
# Map this task's 4 drivers onto the driver names used in the reused interaction-model CSV
INTERACTION_DRIVER_KEY = {"VPD": "VPD", "SWC": "soil_moisture", "TAIR": "temperature", "SW_IN": "radiation"}

N_BOOT = 1000
BOOT_SEED = 20260908
PRODUCT_COLORS = {"GOSIF": "#0072B2", "GOME-2": "#E69F00", "TROPOSIF": "#009E73", "OCO-3": "#CC79A7"}


# ======================================================================================
# Partial-Spearman machinery (identical procedure to
# SIF_EC_Advanced_Coupling_Analysis/12_scripts/03_partial_correlation.py)
# ======================================================================================

def rank_ols_residuals(y: np.ndarray, X_cols: list[np.ndarray]) -> np.ndarray:
    ry = rankdata(y).astype(float)
    n = len(y)
    if len(X_cols) == 0:
        return ry - ry.mean()
    rX = np.column_stack([rankdata(c).astype(float) for c in X_cols])
    design = np.column_stack([np.ones(n), rX])
    beta, *_ = np.linalg.lstsq(design, ry, rcond=None)
    return ry - design @ beta


def partial_spearman(y: np.ndarray, x: np.ndarray, covariates: dict[str, np.ndarray]) -> tuple[float, int]:
    """Rank y, x, and every covariate; residualize ranked y and ranked x on ranked
    covariates via OLS; Pearson-correlate the residuals. Returns (rho, n)."""
    n = len(y)
    cov_arrays = list(covariates.values())
    resid_y = rank_ols_residuals(y, cov_arrays)
    resid_x = rank_ols_residuals(x, cov_arrays)
    if np.std(resid_y) == 0 or np.std(resid_x) == 0:
        return np.nan, n
    rho, _ = pearsonr(resid_x, resid_y)
    return float(rho), n


def tower_block_bootstrap_ci(cc: pd.DataFrame, covariates: list[str], n_boot: int = N_BOOT, seed: int = BOOT_SEED):
    """Tower-block bootstrap: resample TOWERS (not rows) with replacement, recompute the
    partial-Spearman rho on the resampled pool of site-months each time. Returns
    (ci_low, ci_high, n_valid_reps)."""
    towers = cc["SITE_ID"].unique()
    n_towers = len(towers)
    if n_towers < 3:
        return np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    by_tower = {t: g for t, g in cc.groupby("SITE_ID")}
    boot_rhos = np.full(n_boot, np.nan)
    for i in range(n_boot):
        sampled = rng.choice(towers, size=n_towers, replace=True)
        combo = pd.concat([by_tower[t] for t in sampled], ignore_index=True)
        sif = combo["SIF_ANOM"].values
        gpp = combo["GPP_ANOM"].values
        if len(covariates) == 0:
            if np.std(sif) == 0 or np.std(gpp) == 0:
                continue
            rho, _ = spearmanr(sif, gpp)
        else:
            covs = {c: combo[DRIVER_ANOM[c]].values for c in covariates}
            rho, _ = partial_spearman(gpp, sif, covs)
        boot_rhos[i] = rho
    valid = boot_rhos[np.isfinite(boot_rhos)]
    if len(valid) < 100:
        return np.nan, np.nan, len(valid)
    lo, hi = np.percentile(valid, [2.5, 97.5])
    return float(lo), float(hi), int(len(valid))


def complete_case(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.dropna(subset=cols).copy()


# ======================================================================================
# Section 10/11: Six partial-correlation quantities per product + bootstrap CIs +
# driver-specific attenuation
# ======================================================================================

def build_partial_correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    anom_cols = ["SIF_ANOM", "GPP_ANOM"] + [DRIVER_ANOM[d] for d in DRIVERS]
    rows = []
    for product in PRODUCTS:
        sub = df[df["SIF_PRODUCT"] == product]
        cc = complete_case(sub, anom_cols)  # common complete-case sample for ALL 6 quantities of this product
        n = len(cc)
        n_towers = cc["SITE_ID"].nunique()
        print(f"[partial-corr] {product}: N_sitemonths={n}, N_towers={n_towers}")

        # rho0: unadjusted anomaly Spearman
        rho0, _ = spearmanr(cc["SIF_ANOM"], cc["GPP_ANOM"]) if n >= 5 else (np.nan, np.nan)
        lo0, hi0, nb0 = tower_block_bootstrap_ci(cc, [])
        rows.append({"SIF_PRODUCT": product, "QUANTITY": "rho0_unadjusted", "DRIVERS_CONTROLLED": "none",
                     "RHO": rho0, "CI_LOW": lo0, "CI_HIGH": hi0, "N_BOOT_VALID": nb0,
                     "N_SITEMONTHS": n, "N_TOWERS": n_towers})

        # rho_D: controlling ONE driver at a time
        for D in DRIVERS:
            covs = {D: cc[DRIVER_ANOM[D]].values}
            rho_D, _ = partial_spearman(cc["GPP_ANOM"].values, cc["SIF_ANOM"].values, covs) if n >= 10 else (np.nan, n)
            lo_D, hi_D, nb_D = tower_block_bootstrap_ci(cc, [D])
            rows.append({"SIF_PRODUCT": product, "QUANTITY": f"rho_{D}", "DRIVERS_CONTROLLED": D,
                         "RHO": rho_D, "CI_LOW": lo_D, "CI_HIGH": hi_D, "N_BOOT_VALID": nb_D,
                         "N_SITEMONTHS": n, "N_TOWERS": n_towers})

        # rho_ALL: controlling all four jointly
        covs_all = {D: cc[DRIVER_ANOM[D]].values for D in DRIVERS}
        rho_all, _ = partial_spearman(cc["GPP_ANOM"].values, cc["SIF_ANOM"].values, covs_all) if n >= 10 else (np.nan, n)
        lo_all, hi_all, nb_all = tower_block_bootstrap_ci(cc, DRIVERS)
        rows.append({"SIF_PRODUCT": product, "QUANTITY": "rho_ALL", "DRIVERS_CONTROLLED": "VPD+SWC+TAIR+SW_IN",
                     "RHO": rho_all, "CI_LOW": lo_all, "CI_HIGH": hi_all, "N_BOOT_VALID": nb_all,
                     "N_SITEMONTHS": n, "N_TOWERS": n_towers})

    out = pd.DataFrame(rows)
    out["NOTE"] = (
        "rho values are partial Spearman correlations (rank-transform SIF_ANOM, GPP_ANOM, and every "
        "listed covariate; OLS-residualize the ranks; Pearson-correlate the residuals). CI_LOW/CI_HIGH "
        "are a tower-block bootstrap 95% CI (resample TOWERS with replacement, 1000 reps, recompute rho "
        "on the resampled site-month pool each time) -- NOT a Fisher-z analytic CI. Purely associational; "
        "never a causal effect or causal contribution."
    )
    return out


def build_driver_attenuation_table(partial_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for product in PRODUCTS:
        sub = partial_table[partial_table["SIF_PRODUCT"] == product].set_index("QUANTITY")
        rho0 = sub.loc["rho0_unadjusted", "RHO"]
        n_sm = sub.loc["rho0_unadjusted", "N_SITEMONTHS"]
        n_tw = sub.loc["rho0_unadjusted", "N_TOWERS"]
        for D in DRIVERS:
            rho_D = sub.loc[f"rho_{D}", "RHO"]
            lo_D, hi_D = sub.loc[f"rho_{D}", "CI_LOW"], sub.loc[f"rho_{D}", "CI_HIGH"]
            delta = rho0 - rho_D if np.isfinite(rho0) and np.isfinite(rho_D) else np.nan
            rows.append({
                "SIF_PRODUCT": product, "DRIVER": D, "DRIVER_LABEL": DRIVER_LABEL[D],
                "N_SITEMONTHS": n_sm, "N_TOWERS": n_tw,
                "RHO_UNADJUSTED": rho0, "RHO_CONTROL_D": rho_D,
                "RHO_CONTROL_D_CI_LOW": lo_D, "RHO_CONTROL_D_CI_HIGH": hi_D,
                "DELTA_RHO_D": delta,
                "NOTE": (
                    f"DELTA_RHO_D = rho0 - rho_control_{D}: coupling ATTENUATION after controlling for "
                    f"{D}, i.e. how much of the raw SIF-GPP anomaly correlation is statistically shared "
                    f"with {D}. This is NOT a causal effect and NOT a causal contribution."
                ),
            })
    return pd.DataFrame(rows)


# ======================================================================================
# Section 9: collinearity audit, BY PRODUCT and pooled, with condition number added
# ======================================================================================

def cond_number_from_corr(corr: pd.DataFrame) -> float:
    """Eigenvalue-based condition number of a correlation matrix: sqrt(max eig / min eig).
    Standard collinearity diagnostic reported alongside VIF."""
    vals = np.linalg.eigvalsh(corr.values)
    vals = vals[vals > 1e-12]
    if len(vals) == 0:
        return np.nan
    return float(np.sqrt(vals.max() / vals.min()))


def vif_and_cond(sub: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, float, int]:
    cc = sub[cols].dropna()
    n = len(cc)
    if n < len(cols) + 3:
        return pd.DataFrame({"VARIABLE": cols, "VIF": np.nan}), np.nan, n
    z = (cc - cc.mean()) / cc.std(ddof=0)
    X = np.column_stack([np.ones(n)] + [z[c].values for c in cols])
    vifs = [variance_inflation_factor(X, i + 1) for i in range(len(cols))]
    corr = z.corr(method="pearson")
    cond = cond_number_from_corr(corr)
    return pd.DataFrame({"VARIABLE": cols, "VIF": vifs}), cond, n


def build_collinearity_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Driver values (VPD/SWC/TAIR/SW_IN and their _ANOM columns) are identical across SIF
    # products for the same site-month (confirmed in
    # SIF_EC_Advanced_Coupling_Analysis/12_scripts/02_driver_collinearity.py's own
    # comment). For the POOLED block we therefore deduplicate by SITE_ID x YYYYMM ONLY (one
    # row per unique site-month, matching that script's convention) -- NOT per-product --
    # so a site-month used by all 4 products is not counted 4x and does not silently inflate
    # N or bias the pooled correlation toward whichever site-months happen to have the most
    # product coverage. Per-product blocks below correctly use each product's own subset.
    sm_pooled = df.drop_duplicates(subset=["SITE_ID", "YYYYMM"]).copy()
    sm = df.drop_duplicates(subset=["SITE_ID", "YYYYMM", "SIF_PRODUCT"]).copy()
    raw_cols = DRIVERS
    anom_cols = [DRIVER_ANOM[d] for d in DRIVERS]

    corr_rows, vif_rows = [], []

    def add_block(sub: pd.DataFrame, scope: str, version: str, cols: list[str], label_map: dict):
        cc = sub[cols].dropna()
        n = len(cc)
        if n < 5:
            return
        pear = cc.corr(method="pearson")
        spear = cc.corr(method="spearman")
        for i, a in enumerate(cols):
            for j, b in enumerate(cols):
                if j <= i:
                    continue
                corr_rows.append({"SCOPE": scope, "VERSION": version, "VAR_A": label_map[a], "VAR_B": label_map[b],
                                   "PEARSON_R": pear.loc[a, b], "SPEARMAN_RHO": spear.loc[a, b], "N": n})
        vtab, cond, nv = vif_and_cond(sub, cols)
        for _, r in vtab.iterrows():
            vif_rows.append({"SCOPE": scope, "VERSION": version, "VARIABLE": label_map[r["VARIABLE"]],
                              "VIF": r["VIF"], "CONDITION_NUMBER": cond, "N": nv,
                              "FLAG_VIF_GT_5": bool(np.isfinite(r["VIF"]) and r["VIF"] > 5),
                              "FLAG_CONDNUM_GT_30": bool(np.isfinite(cond) and cond > 30)})

    raw_label = {d: d for d in DRIVERS}
    anom_label = {DRIVER_ANOM[d]: f"{d}_ANOM (within-site standardized)" for d in DRIVERS}

    # Pooled (all products) -- deduplicated by site-month, see note above
    add_block(sm_pooled, "POOLED", "RAW", raw_cols, raw_label)
    add_block(sm_pooled, "POOLED", "WITHIN_SITE_STANDARDIZED", anom_cols, anom_label)
    # By product
    for product in PRODUCTS:
        psub = sm[sm["SIF_PRODUCT"] == product]
        add_block(psub, product, "RAW", raw_cols, raw_label)
        add_block(psub, product, "WITHIN_SITE_STANDARDIZED", anom_cols, anom_label)

    corr_df = pd.DataFrame(corr_rows)
    vif_df = pd.DataFrame(vif_rows)
    return corr_df, vif_df


def plot_collinearity_heatmap(df: pd.DataFrame):
    sm_pooled = df.drop_duplicates(subset=["SITE_ID", "YYYYMM"]).copy()
    sm = df.drop_duplicates(subset=["SITE_ID", "YYYYMM", "SIF_PRODUCT"]).copy()
    anom_cols = [DRIVER_ANOM[d] for d in DRIVERS]
    fig, axes = plt.subplots(1, len(PRODUCTS) + 1, figsize=(4.4 * (len(PRODUCTS) + 1), 4.6))
    scopes = [("POOLED (all products, deduplicated by site-month)", sm_pooled)] + [(p, sm[sm["SIF_PRODUCT"] == p]) for p in PRODUCTS]
    im = None
    for ax, (label, sub) in zip(axes, scopes):
        cc = sub[anom_cols].dropna()
        mat = cc.corr(method="spearman") if len(cc) >= 5 else pd.DataFrame(np.nan, index=anom_cols, columns=anom_cols)
        im = ax.imshow(mat.values, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(anom_cols)))
        ax.set_yticks(range(len(anom_cols)))
        ax.set_xticklabels([DRIVER_SHORT[d] for d in DRIVERS], rotation=30, ha="right", fontsize=9)
        ax.set_yticklabels([DRIVER_SHORT[d] for d in DRIVERS], fontsize=9)
        for i in range(len(anom_cols)):
            for j in range(len(anom_cols)):
                v = mat.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                             color="white" if abs(v) > 0.6 else "black")
        ax.set_title(f"{label}\n(N={len(cc)})", fontsize=9.5, fontweight="bold")
    fig.subplots_adjust(wspace=0.5, top=0.80, right=0.90, bottom=0.15)
    cbar_ax = fig.add_axes([0.92, 0.2, 0.012, 0.55])
    fig.colorbar(im, cax=cbar_ax, label="Spearman rho")
    fig.suptitle(
        "Driver collinearity (within-site standardized anomalies): VPD, soil moisture (SM), air "
        "temperature (Ta), incoming shortwave radiation (radiation proxy for PAR) -- pooled and by product",
        fontsize=11.5, fontweight="bold", y=0.98,
    )
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(DIR_COLLIN / f"Driver_Collinearity_Heatmap_By_Product.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


# ======================================================================================
# Interaction models: REUSE from SIF_EC_SIF_GPP_Final_Verification
# ======================================================================================

def load_interaction_models() -> pd.DataFrame:
    raw = pd.read_csv(INTERACTION_CSV)
    raw["product"] = raw["scope"].str.replace("product:", "", regex=False)
    rows = []
    for D in DRIVERS:
        key = INTERACTION_DRIVER_KEY[D]
        for product in PRODUCTS:
            r = raw[(raw["driver"] == key) & (raw["product"] == product)]
            if r.empty:
                rows.append({"SIF_PRODUCT": product, "DRIVER": D, "STATUS": "NOT_FOUND"})
                continue
            r = r.iloc[0]
            rows.append({
                "SIF_PRODUCT": product, "DRIVER": D,
                "STATUS": r.get("status", "OK") if pd.notna(r.get("status", np.nan)) else "OK",
                "N_OBSERVATIONS": r["N_observations"], "N_TOWERS": r["N_towers"],
                "MODEL_TYPE_TRUSTED": r["model_type_reported"],
                "TRUSTED_MIXEDLM": r["trusted_mixedlm"],
                "MIXEDLM_VAR_RANDOM_INTERCEPT": r["mixedlm_var_random_intercept"],
                "INTERACTION_COEF": r["interaction_coefficient"],
                "INTERACTION_SE": r["interaction_se"],
                "INTERACTION_CI_LOW": r["interaction_ci_low"],
                "INTERACTION_CI_HIGH": r["interaction_ci_high"],
                "INTERACTION_P_VALUE": r["interaction_p_value"],
            })
    out = pd.DataFrame(rows)
    out["NOTE"] = (
        "REUSED from SIF_EC_SIF_GPP_Final_Verification/05_mechanistic_reanalysis/"
        "environmental_interactions_product_specific.csv (not refit). Model: GPP_Z ~ SIF_Z + DRIVER_Z + "
        "SIF_Z:DRIVER_Z + C(MONTH), where SIF_Z/GPP_Z/DRIVER_Z are within-site z-scored SIF_ANOM/GPP_ANOM/"
        "driver-ANOM. MixedLM random intercept (groups=SITE_ID) attempted first; trusted only if converged, "
        "cov_re >= 1e-6, and no singular/boundary/Hessian warning text; otherwise falls back to OLS with "
        "site fixed effects and tower-cluster-robust SEs (MODEL_TYPE_TRUSTED records which was actually "
        "used). INTERACTION_COEF is the coefficient on SIF_Z:DRIVER_Z -- the primary quantity of interest "
        "for effect modification. Product-specific (not pooled) is primary, avoiding pseudoreplication "
        "from the same tower-month appearing under multiple products."
    )
    return out


# ======================================================================================
# The deliverable: Unique-Effect Table
# ======================================================================================

def interpret(row) -> str:
    D = row["DRIVER"]
    label = DRIVER_LABEL[D]
    delta = row["DELTA_RHO_D"]
    coef = row["INTERACTION_COEF"]
    p = row["INTERACTION_P_VALUE"]
    lo, hi = row["INTERACTION_CI_LOW"], row["INTERACTION_CI_HIGH"]

    if not np.isfinite(delta):
        atten_txt = "Attenuation could not be computed (insufficient data)."
    else:
        mag = abs(delta)
        size = "negligible" if mag < 0.01 else ("small" if mag < 0.03 else ("moderate" if mag < 0.06 else "notable"))
        direction = "reduces" if delta > 0 else ("increases" if delta < 0 else "does not change")
        atten_txt = f"Controlling for {label} {direction} the SIF-GPP correlation by {abs(delta):.3f} ({size})."

    if not np.isfinite(coef) or not np.isfinite(p):
        mod_txt = "the SIF x driver interaction could not be estimated reliably."
    else:
        sig = np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0) and p < 0.05
        if sig:
            dirn = "strengthens" if coef > 0 else "weakens"
            mod_txt = f"the SIF x {D} interaction is significant (p={p:.3g}) -- {label} {dirn} the SIF-GPP relationship."
        else:
            mod_txt = f"the SIF x {D} interaction is not significant (p={p:.3g}) -- no clear evidence {label} changes the strength of the relationship for this product."
    return f"{atten_txt} Separately, {mod_txt}"


def build_unique_effect_table(atten: pd.DataFrame, interact: pd.DataFrame) -> pd.DataFrame:
    merged = atten.merge(interact, on=["SIF_PRODUCT", "DRIVER"], how="left", suffixes=("", "_interact"))
    rows = []
    for _, r in merged.iterrows():
        ci_txt = (f"[{r['INTERACTION_CI_LOW']:.4f}, {r['INTERACTION_CI_HIGH']:.4f}]"
                  if np.isfinite(r.get("INTERACTION_CI_LOW", np.nan)) and np.isfinite(r.get("INTERACTION_CI_HIGH", np.nan))
                  else "NA")
        rows.append({
            "Product": r["SIF_PRODUCT"],
            "Driver": DRIVER_SHORT[r["DRIVER"]],
            "Driver_Full_Label": DRIVER_LABEL[r["DRIVER"]],
            "rho_unadjusted": round(r["RHO_UNADJUSTED"], 4) if np.isfinite(r["RHO_UNADJUSTED"]) else np.nan,
            "rho_after_controlling_driver": round(r["RHO_CONTROL_D"], 4) if np.isfinite(r["RHO_CONTROL_D"]) else np.nan,
            "rho_control_D_bootstrap_CI": (f"[{r['RHO_CONTROL_D_CI_LOW']:.4f}, {r['RHO_CONTROL_D_CI_HIGH']:.4f}]"
                                            if np.isfinite(r["RHO_CONTROL_D_CI_LOW"]) else "NA"),
            "Delta_rho": round(r["DELTA_RHO_D"], 4) if np.isfinite(r["DELTA_RHO_D"]) else np.nan,
            "SIF_x_Driver_interaction_coef": round(r["INTERACTION_COEF"], 4) if np.isfinite(r.get("INTERACTION_COEF", np.nan)) else np.nan,
            "interaction_SE": round(r["INTERACTION_SE"], 4) if np.isfinite(r.get("INTERACTION_SE", np.nan)) else np.nan,
            "interaction_95pct_CI": ci_txt,
            "interaction_p_value": r.get("INTERACTION_P_VALUE", np.nan),
            "interaction_model_type_trusted": r.get("MODEL_TYPE_TRUSTED", "NA"),
            "N_sitemonths_partial_corr": int(r["N_SITEMONTHS"]) if np.isfinite(r["N_SITEMONTHS"]) else np.nan,
            "N_towers_partial_corr": int(r["N_TOWERS"]) if np.isfinite(r["N_TOWERS"]) else np.nan,
            "N_observations_interaction_model": r.get("N_OBSERVATIONS", np.nan),
            "N_towers_interaction_model": r.get("N_TOWERS_interact", r.get("N_TOWERS", np.nan)),
            "Interpretation": interpret(r),
        })
    out = pd.DataFrame(rows)
    return out


# ======================================================================================
# Figures
# ======================================================================================

def save_fig(fig, stem: str, outdir: Path):
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(outdir / f"{stem}.{ext}", dpi=600, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved {stem}.png/.svg/.pdf")


def figure1_partial_correlation_summary(atten: pd.DataFrame):
    fig, axes = plt.subplots(1, len(PRODUCTS), figsize=(4.0 * len(PRODUCTS), 4.8), sharey=True)
    y_positions = {D: i for i, D in enumerate(DRIVERS)}
    for ax, product in zip(axes, PRODUCTS):
        sub = atten[atten["SIF_PRODUCT"] == product]
        for _, r in sub.iterrows():
            y = y_positions[r["DRIVER"]]
            delta = r["DELTA_RHO_D"]
            # CI on delta = rho0 - rho_control_D; approximate CI width from rho_control_D's CI
            # (rho0 is a constant within-product), i.e. Delta CI = rho0 - [hi, lo] of rho_control_D
            rho0 = r["RHO_UNADJUSTED"]
            lo_r, hi_r = r["RHO_CONTROL_D_CI_LOW"], r["RHO_CONTROL_D_CI_HIGH"]
            if np.isfinite(lo_r) and np.isfinite(hi_r):
                d_lo, d_hi = rho0 - hi_r, rho0 - lo_r
            else:
                d_lo, d_hi = np.nan, np.nan
            color = PRODUCT_COLORS[product]
            if np.isfinite(delta):
                ax.scatter([delta], [y], color=color, s=55, zorder=5, edgecolor="black", linewidth=0.5)
                if np.isfinite(d_lo) and np.isfinite(d_hi):
                    ax.plot([d_lo, d_hi], [y, y], color=color, linewidth=2, zorder=4, alpha=0.8)
        ax.axvline(0, color="#888888", linewidth=0.9, linestyle="--", zorder=1)
        ax.set_title(product, fontsize=11.5, fontweight="bold")
        ax.set_yticks(list(y_positions.values()))
        ax.set_yticklabels([DRIVER_SHORT[D] for D in DRIVERS], fontsize=10.5)
        ax.set_xlabel(r"$\Delta\rho$ (coupling attenuation)", fontsize=9.5)
        ax.set_ylim(-0.6, len(DRIVERS) - 0.4)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle(
        "Figure 1. Partial-correlation summary: coupling attenuation after controlling for each driver\n"
        r"$\Delta\rho$ = unadjusted SIF-GPP anomaly Spearman rho $-$ rho after controlling for that driver alone (tower-block bootstrap 95% CI). "
        "Descriptive/associational attenuation, NOT a causal effect.",
        fontsize=12, fontweight="bold", y=1.05,
    )
    fig.text(0.5, -0.04,
              "SM = soil moisture (SWC); Ta = air temperature; PAR-radiation = incoming shortwave radiation, a radiation proxy for PAR "
              "(true PAR/PPFD_IN available at only 36/73 towers). Positive Delta rho = coupling weakens once that driver is controlled for.",
              ha="center", fontsize=8, color="#555555")
    fig.subplots_adjust(wspace=0.15, top=0.82, bottom=0.18)
    save_fig(fig, "Figure1_Partial_Correlation_Summary", DIR_FIGS)


def figure2_interaction_effects(interact: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    labels, coefs, los, his, colors, sig_flags = [], [], [], [], [], []
    y = 0
    yticks, yticklabels = [], []
    for D in DRIVERS:
        for product in PRODUCTS:
            r = interact[(interact["DRIVER"] == D) & (interact["SIF_PRODUCT"] == product)]
            if r.empty:
                y += 1
                continue
            r = r.iloc[0]
            coef, lo, hi, p = r["INTERACTION_COEF"], r["INTERACTION_CI_LOW"], r["INTERACTION_CI_HIGH"], r["INTERACTION_P_VALUE"]
            labels.append(f"{DRIVER_SHORT[D]} x {product}")
            coefs.append(coef)
            los.append(lo)
            his.append(hi)
            sig = np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)
            sig_flags.append(sig)
            colors.append(PRODUCT_COLORS[product])
            yticks.append(y)
            yticklabels.append(f"{DRIVER_SHORT[D]} x {product}")
            y += 1
        y += 0.6  # gap between driver groups

    coefs = np.array(coefs, dtype=float)
    los = np.array(los, dtype=float)
    his = np.array(his, dtype=float)
    yticks = np.array(yticks, dtype=float)
    max_y = yticks.max() if len(yticks) else 1
    yticks_plot = max_y - yticks  # invert so first driver group is on top

    for yi, c, lo, hi, col, sig in zip(yticks_plot, coefs, los, his, colors, sig_flags):
        if not np.isfinite(c):
            continue
        ax.plot([lo, hi], [yi, yi], color=col, linewidth=2.2 if sig else 1.3, alpha=0.95 if sig else 0.55, zorder=4)
        ax.scatter([c], [yi], color=col, s=60 if sig else 38, zorder=5, edgecolor="black" if sig else "none", linewidth=0.6)

    ax.axvline(0, color="#333333", linewidth=1.0, linestyle="--", zorder=1)
    # IMPORTANT: yticks_plot and yticklabels are both in the SAME original build order
    # (index-matched to coefs/los/his/colors above) -- do NOT reverse one but not the
    # other. yticks_plot is already a decreasing sequence (max_y - yticks), which is what
    # puts the first-built group at the top of the axes; matplotlib pairs set_yticks(...)
    # and set_yticklabels(...) positionally by list index, not by sorted y-value, so
    # passing a reversed label list here (as an earlier version of this script did) silently
    # shifts every label relative to its marker/color -- caught by visual QC and fixed.
    ax.set_yticks(yticks_plot)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel(r"SIF $\times$ Driver interaction coefficient (95% CI)", fontsize=10.5)
    ax.grid(axis="x", alpha=0.25)

    legend_handles = [plt.Line2D([0], [0], marker="o", color=PRODUCT_COLORS[p], linestyle="none",
                                  markersize=8, label=p) for p in PRODUCTS]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9, frameon=True, title="Product", title_fontsize=9)

    fig.suptitle(
        "Figure 2. SIF x Driver interaction (effect-modification) coefficients, all 16 product x driver models\n"
        "Bold marker + thick line = 95% CI excludes zero (driver significantly modifies SIF-GPP strength). "
        "Model: GPP_Z ~ SIF_Z + DRIVER_Z + SIF_Z:DRIVER_Z + C(MONTH), product-specific, MixedLM-then-cluster-robust-OLS-fallback.",
        fontsize=11.3, fontweight="bold", y=1.0,
    )
    fig.subplots_adjust(top=0.90, left=0.28)
    save_fig(fig, "Figure2_Interaction_Effects", DIR_FIGS)


# ======================================================================================
# Verification step: cross-check recomputed rho0/rho_ALL against the existing (already
# reviewed) product-summary and driver-attenuation files
# ======================================================================================

def verify_against_existing(partial_table: pd.DataFrame):
    existing_summary = pd.read_csv(EXISTING_PRODUCT_SUMMARY)
    existing_atten = pd.read_csv(EXISTING_DRIVER_ATTEN)
    print("\n=== Verification: recomputed vs existing (already-reviewed) values ===")
    max_diff = 0.0
    for product in PRODUCTS:
        my_rho0 = partial_table.loc[(partial_table.SIF_PRODUCT == product) & (partial_table.QUANTITY == "rho0_unadjusted"), "RHO"].iloc[0]
        # NOTE: compare against driver_specific_attenuation.csv's RHO_UNADJUSTED, NOT
        # partial_correlation_product_summary.csv's ANOMALY_SPEARMAN_ORDINARY -- the latter
        # uses a different (larger, SIF/GPP-only) complete-case sample that does not require
        # all 4 driver anomalies to be present, so it is NOT the same quantity as our common
        # complete-case rho0 (which must share its sample with rho_control_D for Delta_rho to
        # be a valid same-sample attenuation, exactly as driver_specific_attenuation.csv itself
        # already did).
        old_rho0 = existing_atten.loc[existing_atten.SIF_PRODUCT == product, "RHO_UNADJUSTED"].iloc[0]
        my_rho_all = partial_table.loc[(partial_table.SIF_PRODUCT == product) & (partial_table.QUANTITY == "rho_ALL"), "RHO"].iloc[0]
        old_rho_all = existing_summary.loc[(existing_summary.SIF_PRODUCT == product) & (existing_summary.METHOD == "ANOMALY_SPEARMAN_PARTIAL_ALL4"), "RHO"].iloc[0]
        d1, d2 = abs(my_rho0 - old_rho0), abs(my_rho_all - old_rho_all)
        max_diff = max(max_diff, d1, d2)
        print(f"  {product}: rho0 mine={my_rho0:.6f} existing={old_rho0:.6f} (diff={d1:.2e}); "
              f"rho_ALL mine={my_rho_all:.6f} existing={old_rho_all:.6f} (diff={d2:.2e})")
        for D in DRIVERS:
            my_D = partial_table.loc[(partial_table.SIF_PRODUCT == product) & (partial_table.QUANTITY == f"rho_{D}"), "RHO"].iloc[0]
            old_D = existing_atten.loc[(existing_atten.SIF_PRODUCT == product) & (existing_atten.DRIVER == D), "RHO_CONTROL_D"].iloc[0]
            d3 = abs(my_D - old_D)
            max_diff = max(max_diff, d3)
    print(f"Max abs difference across all cross-checked rho values: {max_diff:.2e} "
          f"({'PASS -- matches existing reviewed values' if max_diff < 1e-8 else 'MISMATCH -- investigate'})")


# ======================================================================================
# Main
# ======================================================================================

def main():
    print("Loading inventory ...")
    df = pd.read_csv(INVENTORY_CSV, low_memory=False)
    df = df[df["SIF_PRODUCT"].isin(PRODUCTS)].copy()

    print("\n=== Section 9: Driver collinearity, by product and pooled (with condition number) ===")
    corr_df, vif_df = build_collinearity_tables(df)
    corr_df.to_csv(DIR_COLLIN / "Driver_Correlation_Matrix.csv", index=False)
    vif_df.to_csv(DIR_COLLIN / "VIF_Table.csv", index=False)
    print(f"  Wrote Driver_Correlation_Matrix.csv ({corr_df.shape}) and VIF_Table.csv ({vif_df.shape})")
    n_flag_vif = int(vif_df["FLAG_VIF_GT_5"].sum())
    n_flag_cond = int((vif_df.drop_duplicates(subset=['SCOPE','VERSION']))["FLAG_CONDNUM_GT_30"].sum())
    print(f"  VIF>5 flags: {n_flag_vif}/{len(vif_df)}; condition-number>30 flags (per scope/version): {n_flag_cond}")
    plot_collinearity_heatmap(df)

    print("\n=== Sections 10-11: Partial-Spearman quantities + tower-block bootstrap CIs ===")
    partial_table = build_partial_correlation_table(df)
    partial_table.to_csv(DIR_PARTIAL / "Partial_Correlation_Six_Quantities.csv", index=False)
    verify_against_existing(partial_table)

    atten = build_driver_attenuation_table(partial_table)
    atten.to_csv(DIR_PARTIAL / "Driver_Specific_Attenuation_With_Bootstrap_CI.csv", index=False)
    print(f"  Wrote Partial_Correlation_Six_Quantities.csv and Driver_Specific_Attenuation_With_Bootstrap_CI.csv")

    print("\n=== Sections 12-13: Interaction/effect-modification models (REUSED) ===")
    interact = load_interaction_models()
    interact.to_csv(DIR_INTERACT / "SIF_Driver_Interaction_Models_Reused.csv", index=False)
    print(f"  Wrote SIF_Driver_Interaction_Models_Reused.csv ({interact.shape})")
    print(interact[["SIF_PRODUCT", "DRIVER", "N_OBSERVATIONS", "N_TOWERS", "MODEL_TYPE_TRUSTED",
                     "INTERACTION_COEF", "INTERACTION_P_VALUE"]].to_string(index=False))

    print("\n=== The deliverable: Unique-Effect Table ===")
    uet = build_unique_effect_table(atten, interact)
    uet.to_csv(DIR_TABLES / "Unique_Effect_Table.csv", index=False)
    print(f"  Wrote {DIR_TABLES / 'Unique_Effect_Table.csv'} ({uet.shape})")
    print(uet[["Product", "Driver", "rho_unadjusted", "rho_after_controlling_driver", "Delta_rho",
               "SIF_x_Driver_interaction_coef", "interaction_p_value"]].to_string(index=False))

    print("\n=== Figures 1-2 ===")
    figure1_partial_correlation_summary(atten)
    figure2_interaction_effects(interact)

    print("\n=== Figure 3: reuse PC2 (geospatial driver attenuation) ===")
    for ext in ["png", "svg", "pdf"]:
        src = PC2_SRC_DIR / f"PC2_Driver_Attenuation_Geospatial.{ext}"
        dst = DIR_GEO / f"Figure3_Geospatial_Driver_Attenuation.{ext}"
        shutil.copyfile(src, dst)
    note = (
        "Figure3_Geospatial_Driver_Attenuation.* is an UNCHANGED copy of "
        "SIF_EC_Advanced_Coupling_Analysis/08_geospatial_figures/PC2_Driver_Attenuation_Geospatial.*\n\n"
        "Reused rather than rebuilt: independently reviewed against this task's Figure 3 spec and found to "
        "already implement it correctly -- 4 (driver) x 4 (product) grid, real basemap "
        "(NE1_HR_LC_SR_W.tif via rasterio+cartopy, tower points only, no interpolated surface), marker "
        "color = site-level Delta_rho_D (diverging colormap centered at 0), site-months N>=24 floor "
        "(the same primary reliability floor used by partial_correlation_site_summary.csv), and a "
        "per-panel 'n=' tower-count annotation as its reliability encoding (rather than continuously "
        "varying marker size by N, which Figure PC1/PC3 in the same folder do instead) -- a valid, "
        "legible, non-cluttered choice for a 16-panel grid, not a defect. Visually inspected: no "
        "overlapping text, legends, or colorbars; titles and row labels are all legible at 600 DPI.\n"
    )
    (DIR_GEO / "Figure3_REUSE_NOTE.txt").write_text(note, encoding="utf-8")
    print(f"  Copied PC2 -> Figure3_Geospatial_Driver_Attenuation.png/.svg/.pdf into {DIR_GEO}")

    print("\nDone.")


if __name__ == "__main__":
    main()
