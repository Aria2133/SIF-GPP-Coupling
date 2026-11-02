"""STEP 12 -- Figure7 (combined-model standardized coefficients) and Figure8
(mechanism attribution breakdown, the capstone "why" figure)."""
# Original script filename (sanitized): step12_figure_combined_model_and_attribution.py; public name: step12_figure_combined_model_and_attribution.py
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = (Path(PROJECT_ROOT) / 'SIF_GPP_PEAK_TIMING_MECHANISM_FULL_ANALYSIS/14_scripts/step12_figure_combined_model_and_attribution.py').resolve().parents[1]
combined = pd.read_csv(ROOT / "10_mechanism_attribution" / "combined_model_coefficients.csv")
attribution = pd.read_csv(ROOT / "10_mechanism_attribution" / "siteyear_mechanism_attribution.csv")
FIG_DIR = ROOT / "11_figures"

LABELS = {
    "Z_SW_IN_GAP_TO_GPP": "Radiation-timing gap to GPP", "Z_PREPEAK_TAIR_ANOM": "Spring temperature anomaly",
    "Z_PREPEAK_VPD_ANOM": "VPD anomaly", "Z_PREPEAK_SWC_ANOM": "Soil-moisture anomaly",
    "Z_PREPEAK_SID_SM": "SID (soil moisture)", "Z_TOTAL_N_NATIVE_OBS": "Satellite sampling density",
}


def fig7():
    num = combined[combined["TERM"].isin(LABELS)].copy()
    num["LABEL"] = num["TERM"].map(LABELS)
    num = num.sort_values("COEF")
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor="white")
    colors = ["#b2182b" if p < 0.05 else "#999999" for p in num["P"]]
    y = np.arange(len(num))
    ax.errorbar(num["COEF"], y, xerr=1.96 * num["SE"], fmt="o", color="#333333", ecolor="#999999", capsize=3, zorder=2)
    ax.scatter(num["COEF"], y, s=90, color=colors, zorder=3)
    ax.axvline(0, color="#999999", lw=0.9, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(num["LABEL"], fontsize=10)
    ax.set_xlabel("Standardized coefficient on ΔPOS\n(cluster-robust by tower, controlling for biome + product)", fontsize=9.5)
    ax.set_title("Which factor best predicts the SIF-GPP peak-timing gap?\n(red = significant at p<0.05, controlling for all others)",
                fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "svg", "pdf"):
        dpi = 600 if ext == "png" else 300
        fig.savefig(FIG_DIR / f"Figure7_Combined_Model_Coefficients.{ext}", dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("Saved Figure7_Combined_Model_Coefficients")


def fig8():
    attributable = attribution[attribution["DELTA_CLASS"].isin(["EARLIER_SIF", "EARLIER_GPP"])].copy()

    def simplify(label):
        if label.startswith("MULTIPLE_MECHANISMS_CONSISTENT"):
            if "SPRING_TEMPERATURE_ACTIVATION" in label and "WATER_VPD_STRESS" in label and "CANOPY_LIGHT" not in label:
                return "Temperature + water stress"
            return "Light + other mechanism(s)" if "CANOPY_LIGHT" in label else "Multiple mechanisms"
        return {
            "CANOPY_LIGHT_APAR_TIMING_CONSISTENT": "Light/radiation timing only",
            "WATER_VPD_STRESS_CONSISTENT": "Water/VPD stress only",
            "SPRING_TEMPERATURE_ACTIVATION_CONSISTENT": "Spring temperature only",
            "PEAK_FIT_UNCERTAIN": "Peak-fit uncertain",
            "MULTI_PEAK_SYSTEM": "Multi-peak system",
            "UNEXPLAINED": "Unexplained",
        }.get(label, label)

    attributable["SIMPLE"] = attributable["MECHANISM_ATTRIBUTION"].apply(simplify)
    order = ["Light/radiation timing only", "Light + other mechanism(s)", "Temperature + water stress",
             "Water/VPD stress only", "Spring temperature only", "Peak-fit uncertain", "Multi-peak system", "Unexplained"]
    counts = attributable["SIMPLE"].value_counts().reindex(order, fill_value=0)
    counts = counts[counts > 0]

    fig, ax = plt.subplots(figsize=(8.5, 5.3), facecolor="white")
    colors_map = {
        "Light/radiation timing only": "#08519c", "Light + other mechanism(s)": "#4292c6",
        "Temperature + water stress": "#e08214", "Water/VPD stress only": "#8073ac",
        "Spring temperature only": "#d6604d", "Peak-fit uncertain": "#bbbbbb",
        "Multi-peak system": "#999999", "Unexplained": "#555555",
    }
    bars = ax.barh(counts.index[::-1], counts.values[::-1], color=[colors_map[c] for c in counts.index[::-1]])
    total = counts.sum()
    for b, v in zip(bars, counts.values[::-1]):
        ax.text(v + total * 0.01, b.get_y() + b.get_height() / 2, f"{v} ({100*v/total:.0f}%)", va="center", fontsize=9.5)
    ax.set_xlabel(f"Number of site-years with a real (non-synchronous) peak offset (n={total})", fontsize=9.8)
    ax.set_title("What best explains site-years where SIF and GPP genuinely peak at\ndifferent times?", fontsize=13, fontweight="bold", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, -0.02, "Evidence-based, sign-consistency classification (\"most consistent with,\" not \"caused by\"). See Final Report for detail.",
              ha="center", fontsize=8.3, color="#666666")
    fig.tight_layout()
    for ext in ("png", "svg", "pdf"):
        dpi = 600 if ext == "png" else 300
        fig.savefig(FIG_DIR / f"Figure8_Mechanism_Attribution_Summary.{ext}", dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("Saved Figure8_Mechanism_Attribution_Summary")


if __name__ == "__main__":
    fig7()
    fig8()
