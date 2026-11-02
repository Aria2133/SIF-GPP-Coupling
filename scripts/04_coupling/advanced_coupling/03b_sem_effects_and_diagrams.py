"""
Part II -- Advanced Coupling Analysis: direct/indirect/total effect decomposition
(cluster/site bootstrap) for the best-supported path model per product, plus
FIGURE SEM1 (path diagrams).

BEST-SUPPORTED MODEL SELECTION (per product) -- reasoning, not just best-AIC
-----------------------------------------------------------------------------
From `SEM_model_comparison.csv` (built by 03a_sem_component_models.py):

1. The two saturated models (C: env->SIF->GPP + direct env->GPP; D: env->GPP->SIF +
   direct env->GPP) are mathematically guaranteed to have IDENTICAL total (summed)
   AIC/BIC for a given product -- confirmed numerically here. This is expected: C's
   two equations {SIF|env} + {GPP|SIF,env} and D's two equations {GPP|env} +
   {SIF|GPP,env} are two different chain-rule factorizations of the same saturated
   joint Gaussian model for (SIF,GPP)|env, so their total likelihood is identical.
   Total AIC therefore CANNOT be used to choose between C and D -- a different,
   genuine signal is needed.
2. The two parsimonious mediation-only models (A: SIF-mediation only; B:
   GPP-mediation only) DO differ in total AIC, because they omit different direct
   terms. The gap between A and B is driven almost entirely by which outcome the
   environmental drivers explain better on their own (Eq1 R^2 of SIF~env vs
   GPP~env) -- the two candidate mediators explain the *other* variable about
   equally well in the single-predictor Eq2 (expected: for standardized variables,
   the simple-regression slope of Y~X and X~Y both approximate the same
   correlation r). So the A-vs-B AIC gap is a genuine, interpretable signal for
   "which direction is better supported", and it is used to pick between the
   SIF-directional family (A/C) and the GPP-directional family (B/D) for a product.
3. A joint cluster-robust Wald test (already computed in 03a) checks whether the
   saturated model's added direct-path terms are jointly zero, i.e. whether
   parsimony (A or B) is rejected in favor of the corresponding saturated model
   (C or D). p<0.05 => prefer the saturated version of whichever direction won
   step 2; p>=0.05 => the direct paths add nothing measurable, so the parsimonious
   model is preferred on economy of explanation grounds.

Applying this to the actual 03a results (documented in the report, not re-derived
blindly here -- BEST_MODEL below is a fixed, reasoned choice per product):
  - GOSIF: A-vs-B AIC clearly favors the SIF-direction family (A better than B by
    ~123 AIC points); Wald test for C-vs-A direct paths p=7.6e-12 (highly
    significant) => saturated preferred => BEST = C.
    Plausibility note: GOSIF is itself a reconstructed/ML product whose own
    production algorithm uses meteorological/reflectance covariates, so drivers
    explaining more SIF variance than GPP variance for GOSIF specifically is
    expected from its construction, not necessarily a physiological signal --
    flagged explicitly in the written report.
  - GOME-2: A-vs-B AIC gap is negligible (~0.4 points, statistically
    indistinguishable) -- direction is NOT resolvable from these data. We default
    to the SIF->GPP framing (C) for consistency with the conventional SIF-as-proxy
    literature framing, but flag the near-tie explicitly; D's coefficients are
    reported in SEM_path_coefficients.csv for the reader to compare.
    Wald test C-vs-A p=1.7e-14, D-vs-B p=1.5e-26: saturated preferred either way.
  - TROPOSIF: A-vs-B AIC favors the GPP-direction family (B better than A by ~98
    points). Wald test D-vs-B p=0.029 (significant at 0.05) => saturated preferred
    => BEST = D.
  - OCO-3: A-vs-B AIC strongly favors the GPP-direction family (B better than A by
    ~58 points) but N is thin (308 obs, only 10 towers -- below the ~30-50-cluster
    rule of thumb for reliable cluster-robust asymptotics). Wald test D-vs-B
    p=0.139 (NOT significant) => direct paths not supported at this sample size =>
    BEST = B (parsimonious). OCO-3 results throughout should be read with caution
    given the very small number of clusters.

Net picture: the best-supported *model family* is NOT consistent across products
(GOSIF and GOME-2 lean SIF-directional; TROPOSIF and OCO-3 lean GPP-directional),
and is reported as such -- this is treated as a real finding about
product-dependent structure, not something to paper over by forcing one winner.
"Best-supported" language throughout means "best fit + most biologically plausible
+ most robust given N/clusters for that product", explicitly NOT "has the most
significant paths" (OCO-3's chosen model B is in fact the more parsimonious one
precisely because its extra paths did NOT reach significance).

Author: project authors
Run date: 2026-09-01
"""

from __future__ import annotations
# Original script filename (sanitized): 03b_sem_effects_and_diagrams.py; public name: 03b_sem_effects_and_diagrams.py
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
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

SCRIPT_DIR = (Path(PROJECT_ROOT) / 'SIF_EC_Advanced_Coupling_Analysis/12_scripts/03b_sem_effects_and_diagrams.py').resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from importlib import import_module
core = import_module("03a_sem_component_models")  # noqa: E402

ROOT = core.ROOT
ADV = core.ADV
OUT_DIR = core.OUT_DIR
DRIVERS_Z = core.DRIVERS_Z
MODEL_SPECS = core.MODEL_SPECS
DRIVER_LABELS = core.DRIVER_LABELS
RNG = core.RNG
N_BOOT = core.N_BOOT

BEST_MODEL = {"GOSIF": "C", "GOME-2": "C", "TROPOSIF": "D", "OCO-3": "B"}
BEST_MODEL_NOTE = {
    "GOSIF": "Saturated model C (env->SIF->GPP with direct env->GPP paths). SIF-direction family clearly "
             "favored over GPP-direction (AIC gap ~123 in favor of A over B); direct paths highly significant "
             "(Wald p=7.6e-12) so saturated C preferred over parsimonious A. Caveat: GOSIF is itself a "
             "reconstructed product whose own algorithm ingests meteorological covariates, so its drivers->SIF "
             "R^2 advantage may partly reflect its construction rather than a purely physiological signal.",
    "GOME-2": "Saturated model C (env->SIF->GPP with direct env->GPP paths), used as the conventional "
              "SIF-as-proxy framing. NOTE: the underlying A-vs-B AIC gap is negligible (~0.4 points) -- "
              "direction is statistically NOT resolvable for GOME-2; model D (GPP->SIF) is essentially tied "
              "and its coefficients are reported alongside for comparison. Direct paths highly significant "
              "in both directions (Wald p=1.7e-14 for C, p=1.5e-26 for D), so a saturated model is preferred "
              "over parsimonious mediation either way.",
    "TROPOSIF": "Saturated model D (env->GPP->SIF with direct env->SIF paths). GPP-direction family favored "
                "over SIF-direction (AIC gap ~98 in favor of B over A); direct paths significant at the 0.05 "
                "level (Wald p=0.029) so saturated D preferred over parsimonious B, though the margin is "
                "modest (unlike GOSIF/GOME-2's very small p-values) -- treat the marginal added value of the "
                "direct paths for TROPOSIF as a weaker signal than for GOSIF/GOME-2.",
    "OCO-3": "Parsimonious model B (env->GPP->SIF, no direct env->SIF paths). GPP-direction family strongly "
             "favored over SIF-direction (AIC gap ~58 in favor of B over A), but the saturated extension's "
             "direct paths are NOT significant (Wald p=0.139), so the simpler mediation-only model B is "
             "preferred on both fit and parsimony grounds. OCO-3 has only 10 towers (308 obs) -- below the "
             "conventional ~30-50-cluster threshold for trustworthy cluster-robust inference -- so ALL OCO-3 "
             "SEM results, including this model choice, should be read with substantial caution.",
}


def cluster_bootstrap_effects(data: pd.DataFrame, spec: dict, product: str, model_key: str, n_boot: int = N_BOOT):
    eq1_spec, eq2_spec = spec["eq1"], spec["eq2"]
    mediator = spec["mediator"]
    drivers_in_eq2 = [t for t in eq2_spec["ivs"] if t != mediator]  # [] for parsimonious A/B
    sites = data["SITE_ID"].unique()
    n_sites = len(sites)

    boot_a = {d: [] for d in DRIVERS_Z}          # driver -> mediator
    boot_c = {d: [] for d in DRIVERS_Z}          # driver -> outcome direct (0 if not in saturated eq)
    boot_indirect = {d: [] for d in DRIVERS_Z}
    boot_total = {d: [] for d in DRIVERS_Z}
    boot_b_mediator = []                          # mediator -> outcome

    site_groups = {s: data[data["SITE_ID"] == s] for s in sites}

    for _ in range(n_boot):
        sampled = RNG.choice(sites, size=n_sites, replace=True)
        boot_df = pd.concat([site_groups[s] for s in sampled], ignore_index=True)
        try:
            f1 = smf.ols(f"{eq1_spec['dv']} ~ {' + '.join(eq1_spec['ivs'])}", data=boot_df).fit()
            f2 = smf.ols(f"{eq2_spec['dv']} ~ {' + '.join(eq2_spec['ivs'])}", data=boot_df).fit()
        except Exception:
            continue
        b_med = f2.params.get(mediator, np.nan)
        boot_b_mediator.append(b_med)
        for d in DRIVERS_Z:
            a = f1.params.get(d, np.nan)
            c = f2.params.get(d, 0.0) if d in drivers_in_eq2 else 0.0
            boot_a[d].append(a)
            boot_c[d].append(c)
            boot_indirect[d].append(a * b_med)
            boot_total[d].append(c + a * b_med)

    # point estimates from the ORIGINAL (non-bootstrap) cluster-robust fits
    eq1_fit = core.fit_component_equation(data, eq1_spec["dv"], eq1_spec["ivs"])
    eq2_fit = core.fit_component_equation(data, eq2_spec["dv"], eq2_spec["ivs"])
    a_point = {c["term"]: c["coef"] for c in eq1_fit["coefs"]}
    c_point = {c["term"]: c["coef"] for c in eq2_fit["coefs"] if c["term"] in DRIVERS_Z}
    b_point = next(c["coef"] for c in eq2_fit["coefs"] if c["term"] == mediator)

    def summarize(arr, point):
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 50:
            return dict(se=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan, n_boot_valid=len(arr))
        se = float(np.std(arr, ddof=1))
        ci_low, ci_high = np.percentile(arr, [2.5, 97.5])
        prop_pos = float(np.mean(arr > 0))
        p = 2 * min(prop_pos, 1 - prop_pos)
        p = min(max(p, 1.0 / len(arr)), 1.0)
        return dict(se=se, ci_low=float(ci_low), ci_high=float(ci_high), p=p, n_boot_valid=len(arr))

    rows = []
    mediator_label = "SIF" if mediator == "SIF_Z" else "GPP"
    outcome_label = "GPP" if eq2_spec["dv"] == "GPP_Z" else "SIF"

    b_summary = summarize(boot_b_mediator, b_point)
    rows.append({
        "SIF_PRODUCT": product, "MODEL": model_key, "DRIVER": mediator_label, "DRIVER_LABEL": f"{mediator_label} (mediator)",
        "TARGET_OUTCOME": outcome_label, "EFFECT_TYPE": "mediator_to_outcome",
        "STD_COEF": b_point, **b_summary, "N": eq1_fit["n"], "N_TOWERS": eq1_fit["n_towers"],
    })

    for d in DRIVERS_Z:
        a_sum = summarize(boot_a[d], a_point[d])
        has_direct = d in drivers_in_eq2
        if has_direct:
            c_sum = summarize(boot_c[d], c_point.get(d, 0.0))
        else:
            # Parsimonious model: this direct path is fixed at 0 by model specification
            # (excluded, not estimated) -- report as structurally zero, not a "significant"
            # estimate. A constant-zero bootstrap distribution would otherwise produce a
            # spurious near-zero p-value artifact.
            c_sum = dict(se=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan, n_boot_valid=len(boot_c[d]))
        ind_sum = summarize(boot_indirect[d], a_point[d] * b_point)
        tot_sum = summarize(boot_total[d], c_point.get(d, 0.0) + a_point[d] * b_point)
        lbl = DRIVER_LABELS[d]
        rows.append({"SIF_PRODUCT": product, "MODEL": model_key, "DRIVER": d, "DRIVER_LABEL": lbl,
                      "TARGET_OUTCOME": mediator_label, "EFFECT_TYPE": "driver_to_mediator",
                      "STD_COEF": a_point[d], **a_sum, "N": eq1_fit["n"], "N_TOWERS": eq1_fit["n_towers"]})
        rows.append({"SIF_PRODUCT": product, "MODEL": model_key, "DRIVER": d, "DRIVER_LABEL": lbl,
                      "TARGET_OUTCOME": outcome_label,
                      "EFFECT_TYPE": "direct" if has_direct else "direct_fixed_at_zero_parsimonious_model",
                      "STD_COEF": c_point.get(d, 0.0), **c_sum, "N": eq2_fit["n"], "N_TOWERS": eq2_fit["n_towers"]})
        rows.append({"SIF_PRODUCT": product, "MODEL": model_key, "DRIVER": d, "DRIVER_LABEL": lbl,
                      "TARGET_OUTCOME": outcome_label, "EFFECT_TYPE": "indirect_via_" + mediator_label,
                      "STD_COEF": a_point[d] * b_point, **ind_sum, "N": eq2_fit["n"], "N_TOWERS": eq2_fit["n_towers"]})
        rows.append({"SIF_PRODUCT": product, "MODEL": model_key, "DRIVER": d, "DRIVER_LABEL": lbl,
                      "TARGET_OUTCOME": outcome_label, "EFFECT_TYPE": "total",
                      "STD_COEF": c_point.get(d, 0.0) + a_point[d] * b_point, **tot_sum,
                      "N": eq2_fit["n"], "N_TOWERS": eq2_fit["n_towers"]})
    return rows


def build_effects_table(fitted_cache):
    all_rows = []
    for product, model_key in BEST_MODEL.items():
        spec = MODEL_SPECS[model_key]
        data = fitted_cache[(product, model_key)]["data"]
        print(f"Bootstrapping {product} model {model_key} (N={len(data)}, towers={data['SITE_ID'].nunique()}) ...")
        rows = cluster_bootstrap_effects(data, spec, product, model_key)
        for r in rows:
            r["MODEL_CHOICE_RATIONALE"] = BEST_MODEL_NOTE[product]
        all_rows.extend(rows)
    out = pd.DataFrame(all_rows)
    out_path = OUT_DIR / "SEM_direct_indirect_effects.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out)} rows)")
    return out


# --------------------------------------------------------------------------------------
# FIGURE SEM1: path diagrams (one small panel per product, best-supported model)
# --------------------------------------------------------------------------------------

NODE_POS_MEDIATOR_A = {  # SIF/GPP as mediator layout: drivers (left) -> mediator (center) -> outcome (right)
    "VPD_Z": (0.0, 0.85), "TAIR_Z": (0.0, 0.55), "SW_IN_Z": (0.0, 0.25), "SWC_Z": (0.0, -0.05),
}


def draw_path_diagram(ax, product: str, model_key: str, coef_df: pd.DataFrame):
    spec = MODEL_SPECS[model_key]
    mediator = spec["mediator"]
    outcome = spec["outcome"]
    mediator_label = "SIF" if mediator == "SIF_Z" else "GPP"
    outcome_label = "GPP" if outcome == "GPP_Z" else "SIF"

    pos = dict(NODE_POS_MEDIATOR_A)
    pos["MED"] = (0.55, 0.4)
    pos["OUT"] = (1.1, 0.4)

    driver_disp = {"VPD_Z": "VPD", "TAIR_Z": "TAIR", "SW_IN_Z": "SW_IN\n(radiation proxy)", "SWC_Z": "SWC"}

    sub = coef_df[(coef_df.SIF_PRODUCT == product) & (coef_df.MODEL == model_key)]

    def edge_style(coef):
        color = "#0072B2" if coef >= 0 else "#D55E00"
        style = "-" if coef >= 0 else "--"
        lw = 0.8 + 6.5 * min(abs(coef), 1.0)
        return color, style, lw

    # driver -> mediator edges
    eq1_name = spec["eq1"]["name"]
    for d in DRIVERS_Z:
        row = sub[(sub.EQUATION == eq1_name) & (sub.PREDICTOR == d)]
        if row.empty:
            continue
        coef = row.iloc[0]["STD_COEF"]
        p = row.iloc[0]["P_VALUE"]
        color, style, lw = edge_style(coef)
        x0, y0 = pos[d]
        x1, y1 = pos["MED"]
        arr = FancyArrowPatch((x0 + 0.14, y0), (x1 - 0.16, y1), arrowstyle="-|>", mutation_scale=10,
                               color=color, linestyle=style, linewidth=lw, alpha=0.9,
                               connectionstyle="arc3,rad=0.08", zorder=2)
        ax.add_patch(arr)
        mx, my = (x0 + x1) / 2 + 0.05, (y0 + y1) / 2 + 0.03
        star = "*" if p < 0.05 else ""
        ax.text(mx, my, f"{coef:.2f}{star}", fontsize=6.3, color=color, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))

    # mediator -> outcome edge
    eq2_name = spec["eq2"]["name"]
    row = sub[(sub.EQUATION == eq2_name) & (sub.PREDICTOR == mediator)]
    if not row.empty:
        coef = row.iloc[0]["STD_COEF"]
        p = row.iloc[0]["P_VALUE"]
        color, style, lw = edge_style(coef)
        x0, y0 = pos["MED"]
        x1, y1 = pos["OUT"]
        arr = FancyArrowPatch((x0 + 0.16, y0), (x1 - 0.14, y1), arrowstyle="-|>", mutation_scale=12,
                               color=color, linestyle=style, linewidth=lw, alpha=0.95, zorder=3)
        ax.add_patch(arr)
        star = "*" if p < 0.05 else ""
        ax.text((x0 + x1) / 2, y0 + 0.08, f"{coef:.2f}{star}", fontsize=7.5, color=color, ha="center", fontweight="bold")

    # direct driver -> outcome edges (only present in saturated models C/D)
    for d in DRIVERS_Z:
        row = sub[(sub.EQUATION == eq2_name) & (sub.PREDICTOR == d)]
        if row.empty:
            continue
        coef = row.iloc[0]["STD_COEF"]
        p = row.iloc[0]["P_VALUE"]
        color, style, lw = edge_style(coef)
        x0, y0 = pos[d]
        x1, y1 = pos["OUT"]
        rad = 0.25 if d in ("VPD_Z", "TAIR_Z") else -0.25
        arr = FancyArrowPatch((x0 + 0.14, y0), (x1 - 0.16, y1), arrowstyle="-|>", mutation_scale=9,
                               color=color, linestyle=style, linewidth=max(lw * 0.6, 0.6), alpha=0.55,
                               connectionstyle=f"arc3,rad={rad}", zorder=1)
        ax.add_patch(arr)

    # nodes
    for d in DRIVERS_Z:
        x, y = pos[d]
        ax.add_patch(Circle((x, y), 0.1, fc="#F0F0F0", ec="#555555", lw=0.8, zorder=4))
        ax.text(x, y, driver_disp[d], fontsize=6.0, ha="center", va="center", zorder=5)
    for key, label, color in [("MED", mediator_label, "#E69F00"), ("OUT", outcome_label, "#009E73")]:
        x, y = pos[key]
        ax.add_patch(Circle((x, y), 0.14, fc=color, ec="#333333", lw=1.0, alpha=0.85, zorder=4))
        ax.text(x, y, label, fontsize=9.5, ha="center", va="center", fontweight="bold", color="white", zorder=5)

    ax.set_xlim(-0.2, 1.3)
    ax.set_ylim(-0.25, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{product}  --  Model {model_key}", fontsize=10.5, fontweight="bold", pad=6)


def build_figure_sem1(coef_df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5))
    for ax, product in zip(axes.flat, ["GOSIF", "GOME-2", "TROPOSIF", "OCO-3"]):
        draw_path_diagram(ax, product, BEST_MODEL[product], coef_df)

    legend_elems = [
        Line2D([0], [0], color="#0072B2", lw=3, linestyle="-", label="Positive path (solid, blue)"),
        Line2D([0], [0], color="#D55E00", lw=3, linestyle="--", label="Negative path (dashed, red)"),
        Line2D([0], [0], color="none", label="Line width = |standardized coefficient|"),
        Line2D([0], [0], color="none", label="* = p<0.05 (cluster-robust, by tower); thin curved arrows = direct env->outcome paths (saturated models only)"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("FIGURE SEM1. Best-supported hypothesized pathway model per SIF product\n"
                 "(piecewise cluster-robust component-model path analysis; standardized coefficients within site)",
                 fontsize=12, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])

    for ext in ("png", "svg", "pdf"):
        out = OUT_DIR / f"SEM1_Path_Diagrams.{ext}"
        fig.savefig(out, dpi=220 if ext == "png" else None, bbox_inches="tight")
        print(f"Wrote {out}")
    plt.close(fig)


def annotate_comparison_with_best_model(comp_df: pd.DataFrame) -> pd.DataFrame:
    comp_df = comp_df.copy()
    comp_df["IS_BEST_SUPPORTED_MODEL"] = comp_df.apply(
        lambda r: BEST_MODEL.get(r["SIF_PRODUCT"]) == r["MODEL"], axis=1)
    comp_df["BEST_SUPPORTED_MODEL_FOR_PRODUCT"] = comp_df["SIF_PRODUCT"].map(BEST_MODEL)
    comp_df["BEST_MODEL_CHOICE_RATIONALE"] = comp_df["SIF_PRODUCT"].map(BEST_MODEL_NOTE)
    out_path = OUT_DIR / "SEM_model_comparison.csv"
    comp_df.to_csv(out_path, index=False)
    print(f"Re-wrote {out_path} with best-supported-model annotation")
    return comp_df


if __name__ == "__main__":
    df, comp_df, coef_df, fitted_cache = core.run_all()
    comp_df = annotate_comparison_with_best_model(comp_df)
    effects_df = build_effects_table(fitted_cache)
    build_figure_sem1(coef_df)
