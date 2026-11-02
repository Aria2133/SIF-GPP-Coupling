"""
SUPPORTING FIGURE/TABLE S2 -- leave-June-out sensitivity of the fitted POS (QC only).

Per section 9 of the phenology-POS revision request: keep this as a small supporting
check, not a main-paper figure, unless excluding June materially changes the inferred
peak offset. Re-run with the NEW POS methodology (harmonic fit, not the retired
peak-month-of-pooled-median-curve method used in the old Diagnostic3) for consistency
with Figures A-D.

Method: for each NH product (June is a Northern-Hemisphere summer month; Southern
Hemisphere sites are unaffected and not re-run here), refit the pooled harmonic POS
with all SITE_ID rows where MONTH==6 removed from the pooled sample, and compare the
resulting DELTA_POS (SIF-GPP) to the WITH-June value from Figure A
(figureA_pooled_pos_summary.csv, reused unmodified).
"""
# Original script filename (sanitized): figureS2_june_sensitivity.py; public name: pos_june_sensitivity.py
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
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str((Path(PROJECT_ROOT) / 'All_the_Visualizations_REVISED/EXACT_REQUESTS_ONLY/PHENOLOGY_POS_REVISION/supporting/figureS2_june_sensitivity.py').resolve().parents[1]))
import pos_lib as pl

OUT_DIR = (Path(PROJECT_ROOT) / 'All_the_Visualizations_REVISED/EXACT_REQUESTS_ONLY/PHENOLOGY_POS_REVISION/supporting/figureS2_june_sensitivity.py').resolve().parent
FIGUREA_DIR = (Path(PROJECT_ROOT) / 'All_the_Visualizations_REVISED/EXACT_REQUESTS_ONLY/PHENOLOGY_POS_REVISION/supporting/figureS2_june_sensitivity.py').resolve().parents[1]
JUNE = 6


def main():
    df = pl.load_master()
    rng = np.random.default_rng(pl.RNG_SEED)
    with_june = pd.read_csv(FIGUREA_DIR / "figureA_pooled_pos_summary.csv")
    with_june_nh = with_june[with_june["HEMISPHERE"] == "NH"].set_index("SIF_PRODUCT")

    rows = []
    for product in pl.PRODUCTS:
        sub_all = df[(df["HEMI"] == "NH") & (df["SIF_PRODUCT"] == product)]
        sub_no_june = sub_all[sub_all["MONTH"] != JUNE]

        delta_with = with_june_nh.loc[product, "DELTA_POS_SIF_MINUS_GPP"]
        delta_with_lo = with_june_nh.loc[product, "DELTA_CI_LOW"]
        delta_with_hi = with_june_nh.loc[product, "DELTA_CI_HIGH"]

        delta_wo, ci_lo_wo, ci_hi_wo = pl.bootstrap_delta_pos_pooled(sub_no_june, rng)

        material_change = bool(np.sign(delta_with) != np.sign(delta_wo) or abs(delta_with - delta_wo) > 1.0)
        rows.append({"SIF_PRODUCT": product, "DELTA_POS_WITH_JUNE": delta_with,
                     "CI_WITH_JUNE": f"[{delta_with_lo:+.1f}, {delta_with_hi:+.1f}]",
                     "DELTA_POS_WITHOUT_JUNE": delta_wo,
                     "CI_WITHOUT_JUNE": f"[{ci_lo_wo:+.1f}, {ci_hi_wo:+.1f}]",
                     "MATERIAL_CHANGE": material_change})

    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / "figureS2_june_sensitivity_table.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor="white")
    x = np.arange(len(result))
    ax.axhline(0, color="#999999", lw=0.8, ls="--")
    ax.scatter(x - 0.08, result["DELTA_POS_WITH_JUNE"], marker="o", s=70, color="#111111", label="with June", zorder=3)
    ax.scatter(x + 0.08, result["DELTA_POS_WITHOUT_JUNE"], marker="s", s=70, facecolor="white",
              edgecolor="#111111", linewidth=1.6, label="June excluded", zorder=3)
    for i in range(len(result)):
        ax.plot([x[i] - 0.08, x[i] + 0.08], [result["DELTA_POS_WITH_JUNE"][i], result["DELTA_POS_WITHOUT_JUNE"][i]],
               color="#999999", lw=1.0, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(result["SIF_PRODUCT"], fontsize=10)
    ax.set_ylabel("ΔPOS = POS$_{SIF}$ − POS$_{GPP}$ (months, NH pooled)", fontsize=9)
    ax.set_title("QC: does excluding June change the fitted peak-timing offset?", fontsize=11, loc="left")
    ax.legend(fontsize=8.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT_DIR / f"FigureS2_June_Sensitivity.{ext}", dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    print("Saved FigureS2_June_Sensitivity.{png,svg,pdf}")
    print(result.to_string(index=False))
    if result["MATERIAL_CHANGE"].any():
        print("\nNOTE: at least one product shows a material change when June is excluded -- "
              "see figureS2_june_sensitivity_table.csv for details.")
    else:
        print("\nNo product shows a material change -- confirms the fitted POS offset (harmonic-smoother based) "
              "is NOT a June-specific artifact.")


if __name__ == "__main__":
    main()
