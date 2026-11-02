"""
Part II -- Advanced Coupling Analysis: piecewise / component-model path analysis
(SEM-equivalent) linking environmental drivers -> SIF -> GPP (and reverse-direction
alternatives), fit separately per main-paper SIF product.

Main-paper products only: GOSIF, GOME-2, TROPOSIF, OCO-3. OCO-2 excluded everywhere.

WHY PIECEWISE COMPONENT MODELS INSTEAD OF A PACKAGED SEM FIT
--------------------------------------------------------------
`semopy` (2.3.11) is installed and importable in this environment. It was evaluated
for this task. Its only random-effects extension is `ModelEffects` /
`ModelGeneralizedEffects`, which requires the caller to supply an explicit N x N (or
block) "effects" covariance/relatedness matrix per random-effect grouping (the
design target is behavior-genetics / kinship-style structured covariance, not a
simple lme4-style grouped random intercept). Building and inverting such a matrix for
73 unevenly-sampled EC towers with 24-288 site-months each is not a natural fit, and
the plain `semopy.Model` assumes i.i.d. rows with no clustering at all -- exactly the
assumption that this project has already shown to be unsafe for this data (see
`SIF_EC_SIF_GPP_Final_Verification/05_mechanistic_reanalysis/
mechanistic_and_biome_findings.md`: every MixedLM fit on a within-site-standardized
outcome in that reanalysis had its random-intercept variance collapse to numerically
zero, because within-site standardization removes exactly the between-site mean
variance a random intercept would explain). We replicate that same check here
(explicitly, per-equation) and expect the same collapse; the documented fallback in
that precedent -- OLS with cluster-robust (by SITE_ID) standard errors -- is reused
here for consistency across this project's methodology.

So: the path model is built explicitly from chained ("piecewise") component
regressions, each fit as its own equation with cluster-robust inference, combined
into direct / indirect / total standardized path coefficients. This is the
piecewiseSEM logic (Shipley), implemented by hand in Python because no available
package here properly supports clustered/repeated-measures SEM for this design.

STANDARDIZATION
----------------
All environmental drivers (VPD, TAIR, SW_IN, SWC) and the SIF/GPP outcomes are used
in their already-deseasonalized ANOM form (site's own climatological-month anomaly /
harmonic-residual anomaly -- the project's final accepted deseasonalization method),
then additionally standardized WITHIN each SITE_ID x SIF_PRODUCT group: subtract that
site's own mean (over the modeled sample), divide by that site's own SD. This means
coefficients answer "when this site/product is more stressed than its own usual
conditions...", and -- as in the precedent above -- forces each site's own mean to
exactly zero for every variable, which is why the random-intercept variance collapses
in the MixedLM check (a mechanical consequence of the standardization, not a fitting
bug). Rows are dropped where a site x product group has fewer than MIN_OBS_PER_GROUP
site-months (its SD is not reliably estimable) or where any needed value is missing
(most missingness comes from SWC's ~79-93% coverage).

MODEL FAMILIES (see full reasoning in the written report)
------------------------------------------------------------
The task names four families. Read literally, "A (allow direct env->GPP paths too)"
and "C (env->SIF, env->GPP, SIF->GPP)" describe an identical regression structure, as
do "B (allow direct env->SIF paths too)" and "D (env->SIF, env->GPP, GPP->SIF)". To
make four genuinely distinct, nested, and separately informative models (rather than
two duplicated pairs), we treat A and B as the *parsimonious mediation-only* fits
(direct env terms fixed at zero in the outcome equation) and C and D as the
*saturated* fits that add those direct paths back in -- i.e. A is nested in C, and B
is nested in D. This lets us report a real, defensible test of "does the direct
path add anything beyond pure mediation" (a joint Wald test) instead of two pairs of
numerically identical fits.

  Model A (parsimonious mediation, SIF is the sole path to GPP):
    Eq1: SIF_Z ~ VPD_Z + TAIR_Z + SW_IN_Z + SWC_Z
    Eq2: GPP_Z ~ SIF_Z                                   (env excluded)
  Model B (parsimonious mediation, GPP is the sole path to SIF):
    Eq1: GPP_Z ~ VPD_Z + TAIR_Z + SW_IN_Z + SWC_Z
    Eq2: SIF_Z ~ GPP_Z                                   (env excluded)
  Model C (saturated / common-cause + directional, SIF -> GPP):
    Eq1: SIF_Z ~ VPD_Z + TAIR_Z + SW_IN_Z + SWC_Z
    Eq2: GPP_Z ~ SIF_Z + VPD_Z + TAIR_Z + SW_IN_Z + SWC_Z
  Model D (saturated / common-cause + directional, GPP -> SIF):
    Eq1: GPP_Z ~ VPD_Z + TAIR_Z + SW_IN_Z + SWC_Z
    Eq2: SIF_Z ~ GPP_Z + VPD_Z + TAIR_Z + SW_IN_Z + SWC_Z

No reciprocal SIF<->GPP path is ever fit (unidentified).

Author: project authors
Run date: 2026-09-01
"""

from __future__ import annotations
# Original script filename (sanitized): 03a_sem_component_models.py; public name: 03a_sem_component_models.py
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

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

ROOT = Path(PROJECT_ROOT)
ADV = ROOT / "SIF_EC_Advanced_Coupling_Analysis"
INPUT_CSV = ADV / "01_input_audit" / "advanced_analysis_input_inventory.csv"
OUT_DIR = ADV / "03_sem"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRODUCTS = ["GOSIF", "GOME-2", "TROPOSIF", "OCO-3"]  # main-paper products only; OCO-2 excluded everywhere
DRIVER_ANOM_COLS = ["VPD_ANOM", "TAIR_ANOM", "SW_IN_ANOM", "SWC_ANOM"]
DRIVER_Z = {"VPD_ANOM": "VPD_Z", "TAIR_ANOM": "TAIR_Z", "SW_IN_ANOM": "SW_IN_Z", "SWC_ANOM": "SWC_Z"}
OUTCOME_ANOM_COLS = ["SIF_ANOM", "GPP_ANOM"]
OUTCOME_Z = {"SIF_ANOM": "SIF_Z", "GPP_ANOM": "GPP_Z"}
ALL_ANOM_COLS = DRIVER_ANOM_COLS + OUTCOME_ANOM_COLS
ALL_Z = {**DRIVER_Z, **OUTCOME_Z}

MIN_OBS_PER_GROUP = 8       # minimum site-months (per SITE_ID x SIF_PRODUCT) to trust a within-site SD
RANDEFF_COLLAPSE_TOL = 1e-6  # variance below this is treated as numerically collapsed regardless of converged flag

RNG = np.random.default_rng(20260901)
N_BOOT = 1000

DRIVER_LABELS = {
    "VPD_Z": "VPD (vapor pressure deficit)",
    "TAIR_Z": "TAIR (air temperature)",
    "SW_IN_Z": "SW_IN (incoming shortwave radiation, radiation proxy)",
    "SWC_Z": "SWC (soil water content)",
}


# --------------------------------------------------------------------------------------
# Data prep
# --------------------------------------------------------------------------------------

def load_and_standardize() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV)
    assert set(df["SIF_PRODUCT"].unique()) <= set(PRODUCTS) | {"OCO-2"}, "Unexpected product in input"
    df = df[df["SIF_PRODUCT"].isin(PRODUCTS)].copy()
    assert "OCO-2" not in df["SIF_PRODUCT"].unique()

    df = df.dropna(subset=ALL_ANOM_COLS).copy()

    grp_n = df.groupby(["SITE_ID", "SIF_PRODUCT"])["SITE_ID"].transform("size")
    df = df[grp_n >= MIN_OBS_PER_GROUP].copy()

    g = df.groupby(["SITE_ID", "SIF_PRODUCT"])
    for col in ALL_ANOM_COLS:
        site_mean = g[col].transform("mean")
        site_sd = g[col].transform(lambda s: s.std(ddof=1))
        z = (df[col] - site_mean) / site_sd
        df[ALL_Z[col]] = z

    df = df.dropna(subset=list(ALL_Z.values())).copy()  # drops any zero-SD groups (constant series)
    return df


# --------------------------------------------------------------------------------------
# Component-equation fitting: MixedLM collapse check -> OLS cluster-robust fallback
# --------------------------------------------------------------------------------------

def fit_component_equation(data: pd.DataFrame, dv: str, ivs: list[str], site_col="SITE_ID"):
    """Fit one structural equation `dv ~ ivs` (no intercept needed conceptually since
    everything is centered, but we keep a formula intercept for correctness/diagnostics).
    Returns a dict with coefficients, SE, CI, p, fit stats, and which estimator was
    trusted (MixedLM vs OLS-cluster-robust fallback)."""
    formula = f"{dv} ~ {' + '.join(ivs)}"
    n = len(data)
    n_towers = data[site_col].nunique()

    # --- Step 1: MixedLM random-intercept check ---
    mixedlm_status = "NOT_ATTEMPTED"
    re_var = np.nan
    mixedlm_trusted = False
    try:
        md = smf.mixedlm(formula, data=data, groups=data[site_col])
        mdf = md.fit(reml=False)
        re_var = float(mdf.cov_re.iloc[0, 0])
        conv = bool(getattr(mdf, "converged", False))
        warn_text = " ".join(str(w.message) for w in getattr(mdf, "_fit_warnings", [])) if hasattr(mdf, "_fit_warnings") else ""
        if conv and re_var >= RANDEFF_COLLAPSE_TOL and np.isfinite(re_var):
            mixedlm_status = "CONVERGED_TRUSTED"
            mixedlm_trusted = True
        elif conv:
            mixedlm_status = "CONVERGED_BUT_VARIANCE_COLLAPSED"
        else:
            mixedlm_status = "NOT_CONVERGED"
    except Exception as e:  # noqa: BLE001
        mixedlm_status = f"ERROR:{type(e).__name__}"

    # --- Step 2: estimator selection ---
    if mixedlm_trusted:
        res = mdf
        model_used = "MixedLM_random_intercept"
        params = res.fe_params
        bse = res.bse_fe
        pvals = res.pvalues[params.index]
        conf = res.conf_int().loc[params.index]
        llf = res.llf
        k = len(params) + 2  # + re variance + residual variance
        aic = -2 * llf + 2 * k
        bic = -2 * llf + k * np.log(n)
        r2 = np.nan  # not directly comparable to OLS R2; left NA for mixed fits
    else:
        model_used = "OLS_cluster_robust_by_SITE_ID_fallback"
        ols = smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data[site_col]})
        params = ols.params
        bse = ols.bse
        pvals = ols.pvalues
        conf = ols.conf_int()
        conf.columns = [0, 1]
        aic = ols.aic
        bic = ols.bic
        r2 = ols.rsquared

    coef_rows = []
    for term in ivs:
        # statsmodels uses the raw column name as the param key for simple additive terms
        key = term
        coef_rows.append({
            "term": term,
            "coef": float(params[key]),
            "se": float(bse[key]),
            "ci_low": float(conf.loc[key, 0]),
            "ci_high": float(conf.loc[key, 1]),
            "p": float(pvals[key]),
        })

    return {
        "formula": formula,
        "n": n,
        "n_towers": n_towers,
        "model_used": model_used,
        "mixedlm_status": mixedlm_status,
        "mixedlm_re_var": re_var,
        "aic": float(aic),
        "bic": float(bic),
        "r2": float(r2) if r2 == r2 else np.nan,
        "coefs": coef_rows,
        "intercept": float(params.get("Intercept", np.nan)),
    }


def wald_joint_test(data: pd.DataFrame, dv: str, mediator: str, driver_terms: list[str], site_col="SITE_ID"):
    """Joint Wald test (cluster-robust) for whether the driver terms are jointly zero
    in `dv ~ mediator + drivers`, i.e. whether adding direct env paths (saturated model)
    improves on pure mediation. Returns (F/chi2 stat, p-value, df)."""
    formula = f"{dv} ~ {mediator} + {' + '.join(driver_terms)}"
    ols = smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data[site_col]})
    hyp = " , ".join(f"{t} = 0" for t in driver_terms)
    wt = ols.wald_test(hyp, scalar=True)
    return float(np.asarray(wt.statistic).ravel()[0]), float(np.asarray(wt.pvalue).ravel()[0]), int(wt.df_denom) if wt.df_denom is not None else None


# --------------------------------------------------------------------------------------
# Model family definitions
# --------------------------------------------------------------------------------------

DRIVERS_Z = ["VPD_Z", "TAIR_Z", "SW_IN_Z", "SWC_Z"]

MODEL_SPECS = {
    "A": dict(
        label="A: parsimonious mediation, env -> SIF -> GPP (no direct env->GPP path)",
        eq1=dict(dv="SIF_Z", ivs=DRIVERS_Z, name="drivers_to_SIF"),
        eq2=dict(dv="GPP_Z", ivs=["SIF_Z"], name="SIF_to_GPP"),
        mediator="SIF_Z", outcome="GPP_Z", nested_in="C",
    ),
    "C": dict(
        label="C: saturated, env -> SIF, env -> GPP (direct), SIF -> GPP",
        eq1=dict(dv="SIF_Z", ivs=DRIVERS_Z, name="drivers_to_SIF"),
        eq2=dict(dv="GPP_Z", ivs=["SIF_Z"] + DRIVERS_Z, name="SIF_and_drivers_to_GPP"),
        mediator="SIF_Z", outcome="GPP_Z", nests="A",
    ),
    "B": dict(
        label="B: parsimonious mediation, env -> GPP -> SIF (no direct env->SIF path)",
        eq1=dict(dv="GPP_Z", ivs=DRIVERS_Z, name="drivers_to_GPP"),
        eq2=dict(dv="SIF_Z", ivs=["GPP_Z"], name="GPP_to_SIF"),
        mediator="GPP_Z", outcome="SIF_Z", nested_in="D",
    ),
    "D": dict(
        label="D: saturated, env -> GPP, env -> SIF (direct), GPP -> SIF",
        eq1=dict(dv="GPP_Z", ivs=DRIVERS_Z, name="drivers_to_GPP"),
        eq2=dict(dv="SIF_Z", ivs=["GPP_Z"] + DRIVERS_Z, name="GPP_and_drivers_to_SIF"),
        mediator="GPP_Z", outcome="SIF_Z", nests="B",
    ),
}


def biological_plausibility_note(model_key: str, eq1_fit, eq2_fit) -> str:
    """Plain-language note based on sign/direction sanity, not on p-value hunting."""
    notes = []
    eq1_signs = {c["term"]: np.sign(c["coef"]) for c in eq1_fit["coefs"]}
    vpd_sign = eq1_signs.get("VPD_Z")
    swc_sign = eq1_signs.get("SWC_Z")
    if vpd_sign is not None and swc_sign is not None:
        if vpd_sign < 0 and swc_sign > 0:
            notes.append("plausible water-stress direction (higher VPD / lower SWC associated with lower "
                         f"{model_key and ('SIF' if 'SIF' in eq1_fit['formula'] else 'GPP')})")
        elif vpd_sign > 0 and swc_sign < 0:
            notes.append("water-stress signs reversed from the expected direction (higher VPD / lower SWC "
                         "associated with HIGHER response) -- treat with caution")
        else:
            notes.append("mixed/ambiguous water-stress signs (VPD and SWC not both pointing the expected way)")
    mediator_term = [c for c in eq2_fit["coefs"] if c["term"] in ("SIF_Z", "GPP_Z")]
    if mediator_term:
        mc = mediator_term[0]
        if mc["coef"] > 0:
            notes.append(f"mediator path positive ({mc['coef']:.3f}), consistent with shared photosynthetic "
                         "machinery driving both quantities in the same direction")
        else:
            notes.append(f"mediator path negative ({mc['coef']:.3f}) -- biologically implausible for two "
                         "quantities both tracking canopy photosynthetic activity; treat with caution")
    return "; ".join(notes)


def run_all():
    df = load_and_standardize()

    comparison_rows = []
    coef_rows = []
    fitted_cache = {}  # (product, model_key) -> dict(eq1=fit, eq2=fit, data=subset)

    for product in PRODUCTS:
        sub = df[df["SIF_PRODUCT"] == product].copy()
        if len(sub) == 0:
            continue
        for model_key, spec in MODEL_SPECS.items():
            eq1_spec, eq2_spec = spec["eq1"], spec["eq2"]
            eq1_fit = fit_component_equation(sub, eq1_spec["dv"], eq1_spec["ivs"])
            eq2_fit = fit_component_equation(sub, eq2_spec["dv"], eq2_spec["ivs"])

            fitted_cache[(product, model_key)] = dict(eq1=eq1_fit, eq2=eq2_fit, data=sub, spec=spec)

            for eq_name, fit in [(eq1_spec["name"], eq1_fit), (eq2_spec["name"], eq2_fit)]:
                for c in fit["coefs"]:
                    coef_rows.append({
                        "SIF_PRODUCT": product,
                        "MODEL": model_key,
                        "EQUATION": eq_name,
                        "FORMULA": fit["formula"],
                        "PREDICTOR": c["term"],
                        "PREDICTOR_LABEL": DRIVER_LABELS.get(c["term"], c["term"]),
                        "STD_COEF": c["coef"],
                        "SE": c["se"],
                        "CI_LOW": c["ci_low"],
                        "CI_HIGH": c["ci_high"],
                        "P_VALUE": c["p"],
                        "N": fit["n"],
                        "N_TOWERS": fit["n_towers"],
                        "ESTIMATOR_USED": fit["model_used"],
                        "MIXEDLM_STATUS": fit["mixedlm_status"],
                        "MIXEDLM_RANDINT_VAR": fit["mixedlm_re_var"],
                    })

            n = eq1_fit["n"]
            n_towers = eq1_fit["n_towers"]
            model_aic = eq1_fit["aic"] + eq2_fit["aic"]
            model_bic = eq1_fit["bic"] + eq2_fit["bic"]
            collapsed_both = (eq1_fit["mixedlm_status"] != "CONVERGED_TRUSTED") and (eq2_fit["mixedlm_status"] != "CONVERGED_TRUSTED")

            wald_stat = wald_p = wald_df = None
            if "nests" in spec:  # this is C or D: test its own direct-path terms against the nested A/B
                driver_terms = [t for t in eq2_spec["ivs"] if t != eq1_spec["dv"] and t in DRIVERS_Z]
                try:
                    wald_stat, wald_p, wald_df = wald_joint_test(sub, eq2_spec["dv"], eq1_spec["dv"], driver_terms)
                except Exception:
                    pass

            plaus = biological_plausibility_note(model_key, eq1_fit, eq2_fit)

            comparison_rows.append({
                "SIF_PRODUCT": product,
                "MODEL": model_key,
                "MODEL_LABEL": spec["label"],
                "N_OBS": n,
                "N_TOWERS": n_towers,
                "EQ1_ESTIMATOR": eq1_fit["model_used"],
                "EQ2_ESTIMATOR": eq2_fit["model_used"],
                "EQ1_MIXEDLM_STATUS": eq1_fit["mixedlm_status"],
                "EQ2_MIXEDLM_STATUS": eq2_fit["mixedlm_status"],
                "RANDOM_INTERCEPT_COLLAPSED_BOTH_EQS": collapsed_both,
                "EQ1_AIC": eq1_fit["aic"], "EQ1_BIC": eq1_fit["bic"], "EQ1_R2": eq1_fit["r2"],
                "EQ2_AIC": eq2_fit["aic"], "EQ2_BIC": eq2_fit["bic"], "EQ2_R2": eq2_fit["r2"],
                "MODEL_SUMMED_AIC": model_aic,
                "MODEL_SUMMED_BIC": model_bic,
                "CFI": "N/A (piecewise component-model approach; CFI/TLI/RMSEA/SRMR not applicable)",
                "TLI": "N/A (piecewise component-model approach; CFI/TLI/RMSEA/SRMR not applicable)",
                "RMSEA": "N/A (piecewise component-model approach; CFI/TLI/RMSEA/SRMR not applicable)",
                "SRMR": "N/A (piecewise component-model approach; CFI/TLI/RMSEA/SRMR not applicable)",
                "JOINT_WALD_STAT_DIRECT_PATHS_VS_NESTED": wald_stat,
                "JOINT_WALD_P_DIRECT_PATHS_VS_NESTED": wald_p,
                "JOINT_WALD_DF": wald_df,
                "NESTING_NOTE": (f"Nests parsimonious model {spec.get('nests')}: Wald test above is for "
                                 f"whether the added direct env->{eq2_spec['dv']} paths are jointly zero "
                                 f"(non-significant => prefer parsimonious {spec.get('nests')})") if "nests" in spec else
                                (f"Parsimonious model, nested in saturated model {spec.get('nested_in')}"),
                "BIOLOGICAL_PLAUSIBILITY_NOTE": plaus,
            })

    comp_df = pd.DataFrame(comparison_rows)
    coef_df = pd.DataFrame(coef_rows)
    comp_df.to_csv(OUT_DIR / "SEM_model_comparison.csv", index=False)
    coef_df.to_csv(OUT_DIR / "SEM_path_coefficients.csv", index=False)
    print(f"Wrote {OUT_DIR / 'SEM_model_comparison.csv'} ({len(comp_df)} rows)")
    print(f"Wrote {OUT_DIR / 'SEM_path_coefficients.csv'} ({len(coef_df)} rows)")

    return df, comp_df, coef_df, fitted_cache


if __name__ == "__main__":
    run_all()
