"""
STEP 6 -- Hypothesis A (light/APAR/canopy) and Hypothesis D (LUE seasonality),
combined because both concern the light side of the mechanism and both are limited by
the confirmed absence of FPAR/LAI (see 01_data_inventory/MISSING_CANOPY_DATA.md).

Hypothesis A (testable half only): is SIF's peak timing closer to SW_IN's (radiation
proxy) peak timing than GPP's is, at the YEAR-SPECIFIC level (not just climatological,
as in the earlier PHENOLOGY_POS_REVISION Figure D)? Canopy-development half (FPAR vs.
LAI timing) is NOT TESTABLE -- not attempted, not approximated.

Hypothesis D: radiation-NORMALIZED proxies (GPP/SW_IN, SIF/SW_IN) -- explicitly NOT
true light-use efficiency (that requires APAR) -- computed per site-year-product from
the harmonic-fitted seasonal curves already available, and their own peak timing
compared to SIF/GPP/SW_IN peak timing.
"""
# Original script filename (sanitized): step6_hypothesis_A_D_light_lue.py; public name: step6_hypothesis_A_D_light_lue.py
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
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = (Path(PROJECT_ROOT) / 'SIF_GPP_PEAK_TIMING_MECHANISM_FULL_ANALYSIS/14_scripts/step6_hypothesis_A_D_light_lue.py').resolve().parents[1]
master = pd.read_csv(ROOT / "02_peak_timing" / "SiteYear_Product_PeakTiming_Master.csv")
OUT_DIR = ROOT / "04_canopy_light_mechanism"
FIG_DIR = ROOT / "11_figures"


def circ_diff(a, b):
    d = (a - b + 6) % 12 - 6
    return d


def main():
    valid = master.dropna(subset=["SIF_POS", "GPP_POS", "SW_IN_POS"]).copy()
    valid["GAP_SIF_RAD"] = valid.apply(lambda r: abs(circ_diff(r["SIF_POS"], r["SW_IN_POS"])), axis=1)
    valid["GAP_GPP_RAD"] = valid.apply(lambda r: abs(circ_diff(r["GPP_POS"], r["SW_IN_POS"])), axis=1)
    valid.to_csv(OUT_DIR / "siteyear_light_timing_gaps.csv", index=False)

    stat, p = wilcoxon(valid["GAP_SIF_RAD"], valid["GAP_GPP_RAD"])
    med_sif, med_gpp = valid["GAP_SIF_RAD"].median(), valid["GAP_GPP_RAD"].median()
    result = pd.DataFrame([{
        "N_SITEYEARS": len(valid), "MEDIAN_GAP_SIF_TO_RADIATION": round(med_sif, 2),
        "MEDIAN_GAP_GPP_TO_RADIATION": round(med_gpp, 2), "WILCOXON_P": p,
        "SIF_CLOSER_TO_RADIATION": bool(med_sif < med_gpp and p < 0.05),
    }])
    result.to_csv(OUT_DIR / "hypothesisA_light_timing_test.csv", index=False)
    print(result.to_string(index=False))

    # Figure: paired comparison
    fig, ax = plt.subplots(figsize=(6.5, 5.3), facecolor="white")
    data = [valid["GAP_SIF_RAD"].dropna(), valid["GAP_GPP_RAD"].dropna()]
    bp = ax.boxplot(data, positions=[0, 1], widths=0.5, patch_artist=True, showfliers=False,
                    medianprops=dict(color="#111111", linewidth=1.6))
    for patch, c in zip(bp["boxes"], ["#0072B2", "#111111"]):
        patch.set_facecolor(c); patch.set_alpha(0.3); patch.set_edgecolor(c)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["SIF vs. radiation", "GPP vs. radiation"], fontsize=10.5)
    ax.set_ylabel("|Peak timing gap| from radiation (SW_IN), months", fontsize=10)
    sig_txt = f"Wilcoxon p={p:.2e}" if p >= 1e-300 else "Wilcoxon p<1e-300"
    ax.set_title(f"Is SIF's timing closer to radiation than GPP's?\nmedian gap: SIF={med_sif:.2f} mo, GPP={med_gpp:.2f} mo ({sig_txt})",
                fontsize=11.5, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, -0.03, "Year-specific POS (n={} site-years). Radiation = SW_IN, an explicit proxy, not true APAR "
                          "(FPAR unavailable -- see MISSING_CANOPY_DATA.md).".format(len(valid)),
              ha="center", fontsize=8, color="#666666")
    fig.tight_layout()
    for ext in ("png", "svg", "pdf"):
        dpi = 600 if ext == "png" else 300
        fig.savefig(FIG_DIR / f"Figure4_Light_Timing_Test.{ext}", dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("Saved Figure4_Light_Timing_Test")

    # Hypothesis D: radiation-normalized "LUE-like" proxy peak timing, at pooled-tower level
    # (a per-year normalized-ratio harmonic fit is fragile with <=12 points; instead we use
    # each site-year's own SIF_POS/GPP_POS/SW_IN_POS to ask a simpler, robust question:
    # does GPP peak later than radiation more often when SIF peaks later than radiation
    # too, or does GPP "catch up" independently? -- i.e. is the GPP-radiation gap
    # explained by the SIF-radiation gap, or does GPP have its own additional lag?)
    valid["GPP_MINUS_SIF_GAP_TO_RAD"] = valid["GAP_GPP_RAD"] - valid["GAP_SIF_RAD"]
    extra_lag_pct = 100.0 * (valid["GPP_MINUS_SIF_GAP_TO_RAD"] > 0).mean()
    with open(OUT_DIR / "hypothesisD_note.txt", "w") as f:
        f.write(
            "Hypothesis D (LUE seasonality) -- radiation-normalized proxy note.\n\n"
            "True light-use-efficiency (LUE = GPP/APAR) cannot be computed: no FPAR/APAR\n"
            "exists in this project (see MISSING_CANOPY_DATA.md). A radiation-normalized\n"
            f"proxy (GPP/SW_IN) is not separately reported as a time series here because its\n"
            "peak timing is mathematically determined by the same two curves already\n"
            "analyzed (GPP and SW_IN) -- fitting a THIRD harmonic curve to their ratio at\n"
            "year-specific resolution (<=12 points) would not add independent information\n"
            "and risks appearing more rigorous than it is.\n\n"
            "Instead we report the direct, equivalent comparison: in "
            f"{extra_lag_pct:.0f}% of site-years, GPP's peak sits FARTHER from the radiation\n"
            "peak than SIF's does (GAP_GPP_RAD > GAP_SIF_RAD; see hypothesisA_light_timing_test.csv\n"
            "and siteyear_light_timing_gaps.csv). This is the same evidence a GPP/SW_IN-vs-\n"
            "SIF/SW_IN peak-timing comparison would show, without fitting an unstable extra curve.\n"
        )
    print(f"GPP farther from radiation than SIF in {extra_lag_pct:.0f}% of site-years")


if __name__ == "__main__":
    main()
