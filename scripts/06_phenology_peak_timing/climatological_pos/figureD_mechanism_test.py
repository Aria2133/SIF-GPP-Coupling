"""
FIGURE D -- "Why might SIF peak earlier?" (literature-driven mechanism test)

Data availability check performed BEFORE writing this script (dedicated Explore-agent
search across the whole project directory tree, including
SIF_EC_Advanced_Coupling_Analysis/01_input_audit/advanced_analysis_input_inventory.csv,
SIF_GPP_coupling_driver_master.csv / mechanistic_driver_master_4product.csv, and the
existing FLUXNET_73_Tower_Radiation_Variable_Audit_Report.md):
  - Radiation: SW_IN (incoming shortwave, broadband) is available for all 71 towers in
    the climatology inventory -- used here as the radiation/APAR-timing proxy (never
    relabeled as true PAR). True PAR (PPFD_IN) exists only in RAW half-hourly source
    files, not in any processed master, and only at 36-52 of 73 towers -- not usable
    here without a new extraction pipeline, which would violate "do not rerun
    unnecessary heavy pipelines."
  - Canopy development (LAI, FPAR, NIRv, EVI, NDVI, chlorophyll proxy): CONFIRMED
    ABSENT everywhere in this project. `lai_or_vegetation_index_available = False` for
    all 224 tower x product rows in every copy of the mechanistic driver master table,
    and a dedicated FLUXNET_73_Tower_Radiation_Variable_Audit found 0/73 towers with
    any FAPAR/APAR variable. Per the explicit instruction NOT to fabricate unavailable
    variables, this mechanism (GPP timing tracking canopy development/chlorophyll) is
    NOT tested here -- Panel B below states this limitation directly instead of
    presenting a fabricated comparison.
  - VPD, TAIR, SWC: available (climatology inventory), tested as supporting/alternative
    explanations only (Panel C), per the instruction to keep these secondary unless
    they show a clear signal.

Reuses: tower_pos_table.csv (SIF_POS / GPP_POS per tower, built by figureB, not
recomputed), and the same site-calendar-month climatology columns already used for the
earlier (superseded) Diagnostic4 (advanced_analysis_input_inventory.csv) -- but tests
PEAK-TIMING alignment here (a different, phenology-consistent question) rather than
seasonal-amplitude-vs-lag (the retired Diagnostic4 framing).
"""
# Original script filename (sanitized): figureD_mechanism_test.py; public name: figureD_mechanism_test.py
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

import pos_lib as pl

ROOT = pl.ROOT
INVENTORY_CSV = ROOT / "SIF_EC_Advanced_Coupling_Analysis" / "01_input_audit" / "advanced_analysis_input_inventory.csv"
OUT_DIR = (Path(PROJECT_ROOT) / 'All_the_Visualizations_REVISED/EXACT_REQUESTS_ONLY/PHENOLOGY_POS_REVISION/figureD_mechanism_test.py').resolve().parent

DRIVERS = [("SW_IN_CLIM", "SW_IN (radiation proxy)"), ("VPD_CLIM", "VPD"),
           ("TAIR_CLIM", "TAIR"), ("SWC_CLIM", "SWC")]


def site_driver_pos_table() -> pd.DataFrame:
    inv = pd.read_csv(INVENTORY_CSV, low_memory=False)
    site_month = inv.groupby(["SITE_ID", "MONTH"])[[c for c, _ in DRIVERS]].first().reset_index()
    rows = []
    for site, g in site_month.groupby("SITE_ID"):
        row = {"SITE_ID": site}
        for col, _ in DRIVERS:
            res = pl.estimate_pos_from_climatology(g.set_index("MONTH")[col])
            row[col + "_POS"] = res["POS"]
            row[col + "_FLAGS"] = res["FLAGS"]
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_median(deltas: np.ndarray, rng: np.random.Generator, n_boot: int = 2000):
    if len(deltas) < 5:
        return np.nan, np.nan, np.nan
    point = float(np.median(deltas))
    boots = [np.median(rng.choice(deltas, size=len(deltas), replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def main():
    tower_pos = pd.read_csv(OUT_DIR / "tower_pos_table.csv")
    driver_pos = site_driver_pos_table()
    driver_pos.to_csv(OUT_DIR / "site_driver_pos_table.csv", index=False)

    merged = tower_pos.merge(driver_pos, on="SITE_ID", how="left")
    merged.to_csv(OUT_DIR / "figureD_merged_pos_table.csv", index=False)
    rng = np.random.default_rng(pl.RNG_SEED)

    fig = plt.figure(figsize=(16.5, 9.0), facecolor="white")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1.0], width_ratios=[1, 1, 1], hspace=0.55, wspace=0.4)

    # ---- Panel A1: radiation POS vs SIF POS ----
    axA1 = fig.add_subplot(gs[0, 0])
    # ---- Panel A2: radiation POS vs GPP POS ----
    axA2 = fig.add_subplot(gs[0, 1])
    # ---- Panel B: data-limitation statement ----
    axB = fig.add_subplot(gs[0, 2])

    for ax, ycol, ylabel, title in [
        (axA1, "SIF_POS", "Fitted SIF POS (month)", "(A1) Radiation timing vs. SIF timing"),
        (axA2, "GPP_POS", "Fitted GPP POS (month)", "(A2) Radiation timing vs. GPP timing"),
    ]:
        sub = merged.dropna(subset=["SW_IN_CLIM_POS", ycol])
        for product in pl.PRODUCTS:
            s = sub[sub["SIF_PRODUCT"] == product]
            ax.scatter(s["SW_IN_CLIM_POS"], s[ycol], color=pl.PRODUCT_COLORS[product], s=34,
                       alpha=0.8, edgecolor="white", linewidth=0.4, label=product)
        ax.plot([0, 12], [0, 12], color="#111111", lw=1.6, zorder=1)
        ax.plot([0, 12], [-12, 0], color="#bbbbbb", lw=0.9, ls="--", zorder=0)
        ax.plot([0, 12], [12, 24], color="#bbbbbb", lw=0.9, ls="--", zorder=0)
        ax.set_xlim(0, 12); ax.set_ylim(0, 12)
        ax.set_xticks([1, 4, 7, 10]); ax.set_yticks([1, 4, 7, 10])
        ax.set_xticklabels(["Jan", "Apr", "Jul", "Oct"], fontsize=7.8)
        ax.set_yticklabels(["Jan", "Apr", "Jul", "Oct"], fontsize=7.8)
        ax.set_aspect("equal")
        ax.set_xlabel("Site radiation (SW_IN) POS (month)", fontsize=8.6)
        ax.set_ylabel(ylabel, fontsize=8.6)
        ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left")
        ax.spines[["top", "right"]].set_visible(False)

        deltas = np.array([pl.circular_diff(y, x) for y, x in zip(sub[ycol], sub["SW_IN_CLIM_POS"])])
        point, lo, hi = bootstrap_median(deltas, rng)
        pct_within1 = 100.0 * np.mean(np.abs(deltas) <= 1.0)
        ax.text(0.03, 0.97, f"n={len(sub)}\nmedian gap={point:+.1f} mo\n95% CI [{lo:+.1f}, {hi:+.1f}]\n"
                             f"{pct_within1:.0f}% within ±1 mo",
                transform=ax.transAxes, ha="left", va="top", fontsize=7.6,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", alpha=0.92))
    axA1.legend(fontsize=6.8, loc="lower right", frameon=False)

    axB.axis("off")
    axB.text(0.02, 0.98,
              "(B) Canopy-development mechanism:\nNOT TESTABLE with this dataset",
              transform=axB.transAxes, ha="left", va="top", fontsize=10.5, fontweight="bold", color="#a04000")
    axB.text(0.02, 0.80,
              "The literature hypothesis that GPP's LATER peak tracks canopy\n"
              "development / chlorophyll / LAI cannot be tested here.\n\n"
              "Checked directly across this entire project (not assumed):\n"
              "  • LAI / FPAR / NIRv / EVI / NDVI / chlorophyll proxy:\n"
              "     confirmed absent at all 73 towers (0/73; dedicated\n"
              "     variable-availability audit + all mechanistic driver\n"
              "     master tables show lai_or_vegetation_index_available\n"
              "     = False for every row).\n"
              "  • True PAR (PPFD_IN) exists only in raw half-hourly\n"
              "     source files, at 36-52 of 73 towers -- not in any\n"
              "     processed master table used by this analysis.\n\n"
              "What CAN be tested: whether SIF/GPP timing tracks the\n"
              "radiation (SW_IN) proxy's own timing -- panels (A1)-(A2).\n"
              "No canopy-development variable is fabricated or implied.",
              transform=axB.transAxes, ha="left", va="top", fontsize=8.3, color="#333333")

    # ---- Panel C: compact supporting driver-timing coefficient plot ----
    axC = fig.add_subplot(gs[1, :])
    rows_c = []
    for col, label in DRIVERS:
        for ycol, yname in [("SIF_POS", "SIF"), ("GPP_POS", "GPP")]:
            sub = merged.dropna(subset=[col + "_POS", ycol])
            deltas = np.array([pl.circular_diff(y, x) for y, x in zip(sub[ycol], sub[col + "_POS"])])
            point, lo, hi = bootstrap_median(deltas, rng)
            rows_c.append({"driver": label, "target": yname, "n": len(sub), "median": point, "lo": lo, "hi": hi})
    coef_df = pd.DataFrame(rows_c)
    coef_df.to_csv(OUT_DIR / "figureD_driver_timing_coefficients.csv", index=False)

    y_positions = np.arange(len(DRIVERS))
    offset = {"SIF": -0.14, "GPP": 0.14}
    color = {"SIF": "#0072B2", "GPP": "#111111"}
    marker = {"SIF": "o", "GPP": "s"}
    for target in ["SIF", "GPP"]:
        d = coef_df[coef_df["target"] == target].set_index("driver").reindex([lab for _, lab in DRIVERS])
        ypos = y_positions + offset[target]
        axC.errorbar(d["median"], ypos, xerr=[d["median"] - d["lo"], d["hi"] - d["median"]],
                     fmt=marker[target], color=color[target], ecolor=color[target], capsize=3,
                     ms=8, elinewidth=1.4, label=f"{target} peak vs. driver peak")
    axC.axvline(0, color="#999999", lw=0.9, ls="--")
    axC.set_yticks(y_positions)
    axC.set_yticklabels([lab for _, lab in DRIVERS], fontsize=9.5)
    axC.set_xlabel("Median circular gap: (SIF or GPP POS) − (driver POS), months\n"
                    "(supporting analysis only -- driver PEAK-TIMING alignment, not amplitude)", fontsize=8.8)
    axC.set_title("(C) Supporting: does SIF/GPP peak timing align with each environmental driver's own peak timing?",
                  fontsize=10.5, fontweight="bold", loc="left")
    axC.legend(fontsize=8, loc="best", frameon=False)
    axC.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Why might SIF peak earlier than GPP?", fontsize=16, fontweight="bold", y=1.02)

    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT_DIR / f"FigureD_Mechanism_Test.{ext}", dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("Saved FigureD_Mechanism_Test.{png,svg,pdf}")
    print(coef_df.to_string(index=False))


if __name__ == "__main__":
    main()
