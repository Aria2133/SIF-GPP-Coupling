"""
TASK 4 (reviewer exact-requests-only revision):
Replace the per-product 5-panel Figure4_Corrected_SEM_Path_Diagrams with ONE new figure
containing ONLY two subfigures, pooling all 4 main-paper SIF products together:
  Subfigure 1: Variable -> SIF -> GPP  (env drivers -> SIF; SIF -> GPP)
  Subfigure 2: Variable -> GPP -> SIF  (env drivers -> GPP; GPP -> SIF)

Reuses the audited data-loading/standardization exactly as-is from the SEM audit's own
script (SIF_EC_SEM_Environmental_Control_Audit / the underlying
03a_sem_component_models.py machinery): deseasonalized SIF/GPP anomalies, each driver
and outcome standardized WITHIN SITE_ID x SIF_PRODUCT (so units/scale differences across
products don't bias the pooled result), then -- per this task's explicit instruction to
"combine all SIF datasets together" -- ALL 4 products' standardized rows are pooled into
one single sample for fitting (no per-product split, no product fixed effect, unlike the
audited per-product analysis). This is stated explicitly in the figure caption: pooling
repeated tower-months across products without a product term is a real simplification
made because this task explicitly asked for one combined result, not a replacement for
the product-specific, more rigorous analysis in the SEM audit.

Inference: cluster-robust (by SITE_ID) OLS, matching the project's established fallback
(within-site standardization collapses MixedLM's random intercept everywhere else in this
project; not re-tested here since pooling only increases N, it would not un-collapse it).
"""
# Original script filename (sanitized): task4_build_pooled_sem_figure.py; public name: fig14_pooled_pathway_analysis.py
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
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.lines import Line2D

ROOT = Path(PROJECT_ROOT)
SEM_SCRIPT_DIR = ROOT / "SIF_EC_Advanced_Coupling_Analysis" / "12_scripts"
OUT_DIR = ROOT / "All_the_Visualizations_REVISED" / "EXACT_REQUESTS_ONLY"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SEM_SCRIPT_DIR))
import importlib
core = importlib.import_module("03a_sem_component_models")  # noqa: E402

DRIVERS_Z = core.DRIVERS_Z  # ["VPD_Z","TAIR_Z","SW_IN_Z","SWC_Z"]
DRIVER_DISP = {"VPD_Z": "VPD", "TAIR_Z": "TAIR", "SW_IN_Z": "SW_IN\n(radiation proxy)", "SWC_Z": "SWC"}


def fit_mediation_model(data: pd.DataFrame, mediator_var: str, outcome_var: str):
    """Variable -> mediator -> outcome, parsimonious (no direct env->outcome path),
    cluster-robust by SITE_ID. Returns dict of driver coefficients (a-paths), the
    mediator->outcome coefficient (b-path), and fit stats."""
    f1 = smf.ols(f"{mediator_var} ~ {' + '.join(DRIVERS_Z)}", data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data["SITE_ID"]})
    f2 = smf.ols(f"{outcome_var} ~ {mediator_var}", data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data["SITE_ID"]})
    a_paths = {d: dict(coef=float(f1.params[d]), p=float(f1.pvalues[d])) for d in DRIVERS_Z}
    b_path = dict(coef=float(f2.params[mediator_var]), p=float(f2.pvalues[mediator_var]))
    return dict(a_paths=a_paths, b_path=b_path, n=int(len(data)), n_towers=int(data["SITE_ID"].nunique()),
                aic=float(f1.aic + f2.aic))


NODE_POS = {"VPD_Z": (0.0, 0.85), "TAIR_Z": (0.0, 0.55), "SW_IN_Z": (0.0, 0.25), "SWC_Z": (0.0, -0.05),
            "MED": (0.55, 0.4), "OUT": (1.1, 0.4)}


def edge_style(coef):
    color = "#0072B2" if coef >= 0 else "#D55E00"
    style = "-" if coef >= 0 else "--"
    lw = 0.8 + 6.5 * min(abs(coef), 1.0)
    return color, style, lw


def draw_diagram(ax, title, mediator_label, outcome_label, fit):
    for d in DRIVERS_Z:
        coef = fit["a_paths"][d]["coef"]
        p = fit["a_paths"][d]["p"]
        color, style, lw = edge_style(coef)
        x0, y0 = NODE_POS[d]
        x1, y1 = NODE_POS["MED"]
        arr = FancyArrowPatch((x0 + 0.14, y0), (x1 - 0.16, y1), arrowstyle="-|>", mutation_scale=10,
                               color=color, linestyle=style, linewidth=lw, alpha=0.9,
                               connectionstyle="arc3,rad=0.08", zorder=2)
        ax.add_patch(arr)
        mx, my = (x0 + x1) / 2 + 0.05, (y0 + y1) / 2 + 0.03
        star = "*" if p < 0.05 else ""
        ax.text(mx, my, f"{coef:.2f}{star}", fontsize=7.0, color=color, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8))

    coef = fit["b_path"]["coef"]
    p = fit["b_path"]["p"]
    color, style, lw = edge_style(coef)
    x0, y0 = NODE_POS["MED"]
    x1, y1 = NODE_POS["OUT"]
    arr = FancyArrowPatch((x0 + 0.16, y0), (x1 - 0.14, y1), arrowstyle="-|>", mutation_scale=13,
                           color=color, linestyle=style, linewidth=lw, alpha=0.95, zorder=3)
    ax.add_patch(arr)
    star = "*" if p < 0.05 else ""
    ax.text((x0 + x1) / 2, y0 + 0.09, f"{coef:.2f}{star}", fontsize=9.5, color=color, ha="center", fontweight="bold")

    for d in DRIVERS_Z:
        x, y = NODE_POS[d]
        ax.add_patch(Circle((x, y), 0.1, fc="#F0F0F0", ec="#555555", lw=0.8, zorder=4))
        ax.text(x, y, DRIVER_DISP[d], fontsize=6.4, ha="center", va="center", zorder=5)
    for key, label, color in [("MED", mediator_label, "#E69F00"), ("OUT", outcome_label, "#009E73")]:
        x, y = NODE_POS[key]
        ax.add_patch(Circle((x, y), 0.14, fc=color, ec="#333333", lw=1.0, alpha=0.88, zorder=4))
        ax.text(x, y, label, fontsize=10.5, ha="center", va="center", fontweight="bold", color="white", zorder=5)

    ax.set_xlim(-0.2, 1.3)
    ax.set_ylim(-0.25, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)


def main():
    df = core.load_and_standardize()  # already pools all 4 products; per-row site+product standardized
    n_total, towers_total = len(df), df["SITE_ID"].nunique()
    products_present = sorted(df["SIF_PRODUCT"].unique().tolist())
    print(f"Pooled sample (all products combined): N={n_total} site-months, {towers_total} towers, "
          f"products={products_present}")

    fit_sif_first = fit_mediation_model(df, "SIF_Z", "GPP_Z")   # Variable -> SIF -> GPP
    fit_gpp_first = fit_mediation_model(df, "GPP_Z", "SIF_Z")   # Variable -> GPP -> SIF

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 6.4), facecolor="white")
    better = "Subfigure 1" if fit_sif_first["aic"] < fit_gpp_first["aic"] else "Subfigure 2"
    draw_diagram(ax1, f"Subfigure 1\nVariable -> SIF -> GPP\n(total AIC = {fit_sif_first['aic']:,.0f})", "SIF", "GPP", fit_sif_first)
    draw_diagram(ax2, f"Subfigure 2\nVariable -> GPP -> SIF\n(total AIC = {fit_gpp_first['aic']:,.0f})", "GPP", "SIF", fit_gpp_first)
    fig.text(0.5, 0.895,
              f"Lower total AIC = {better} ({'Variable->SIF->GPP' if better=='Subfigure 1' else 'Variable->GPP->SIF'}) "
              f"-- these are non-saturated mediation models, so unlike the earlier product-specific SEM audit's "
              f"Models C/D, their AICs are not forced to tie.",
              ha="center", fontsize=8.5, color="#444444", style="italic")

    legend_elems = [
        Line2D([0], [0], color="#0072B2", lw=3, linestyle="-", label="Positive path (solid, blue)"),
        Line2D([0], [0], color="#D55E00", lw=3, linestyle="--", label="Negative path (dashed, red)"),
        Line2D([0], [0], color="none", label="Line width = |standardized coefficient|; * = p<0.05 (cluster-robust, by tower)"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=1, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"Pooled SEM (combined-all-products): comparison of the two hypothesized mediation directions\n"
        f"N={n_total:,} site-months, {towers_total} towers, products combined: {', '.join(products_present)} "
        f"(no product term -- this is a simplified, explicitly-requested combined-all-products view; "
        f"association only, not a causal claim)",
        fontsize=11.5, fontweight="bold", y=1.06,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.87])

    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT_DIR / f"Figure4_Pooled_SEM_TwoDirections.{ext}", dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("Saved Figure4_Pooled_SEM_TwoDirections.{png,svg,pdf}")

    summary = pd.DataFrame([
        {"direction": "Variable->SIF->GPP", "mediator_to_outcome_coef": fit_sif_first["b_path"]["coef"],
         "p": fit_sif_first["b_path"]["p"], "aic_total": fit_sif_first["aic"], "n": fit_sif_first["n"], "n_towers": fit_sif_first["n_towers"]},
        {"direction": "Variable->GPP->SIF", "mediator_to_outcome_coef": fit_gpp_first["b_path"]["coef"],
         "p": fit_gpp_first["b_path"]["p"], "aic_total": fit_gpp_first["aic"], "n": fit_gpp_first["n"], "n_towers": fit_gpp_first["n_towers"]},
    ])
    summary.to_csv(OUT_DIR / "Figure4_Pooled_SEM_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
