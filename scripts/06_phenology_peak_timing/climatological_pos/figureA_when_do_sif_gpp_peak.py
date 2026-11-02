"""
FIGURE A -- "When do SIF and GPP peak?"
The simplest possible answer: normalized (0-1) pooled seasonal trajectories of SIF and
GPP, per product x hemisphere, with fitted Peak-Of-Season (POS) marked and the offset
annotated in one sentence. Must be readable in ~5 seconds; no dual y-axes, no
classification jargon.
"""
# Original script filename (sanitized): figureA_when_do_sif_gpp_peak.py; public name: figureA_when_do_sif_gpp_peak.py
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

OUT_DIR = (Path(PROJECT_ROOT) / 'All_the_Visualizations_REVISED/EXACT_REQUESTS_ONLY/PHENOLOGY_POS_REVISION/figureA_when_do_sif_gpp_peak.py').resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEMI_LABELS = {"NH": "Northern Hemisphere", "SH": "Southern Hemisphere"}


def normalize01(curve: np.ndarray, lo: float = None, hi: float = None):
    lo = curve.min() if lo is None else lo
    hi = curve.max() if hi is None else hi
    if hi - lo < 1e-12:
        return np.full_like(curve, 0.5)
    return (curve - lo) / (hi - lo)


def annotate_offset(ax, delta, ci_low, ci_high):
    # delta = POS_SIF - POS_GPP (circular). delta < 0 => SIF's peak month is numerically
    # earlier in the calendar than GPP's => "SIF earlier". delta > 0 => "GPP earlier".
    if np.isnan(delta):
        txt = "insufficient data for a peak-timing estimate"
    elif (not np.isnan(ci_low)) and ci_low <= 0 <= ci_high:
        txt = f"Approximately synchronous ({delta:+.1f} mo, 95% CI [{ci_low:+.1f}, {ci_high:+.1f}])"
    elif delta < 0:
        ci_txt = f", 95% CI [{ci_low:+.1f}, {ci_high:+.1f}]" if not np.isnan(ci_low) else ""
        txt = f"SIF earlier by {abs(delta):.1f} mo{ci_txt}"
    else:
        ci_txt = f", 95% CI [{ci_low:+.1f}, {ci_high:+.1f}]" if not np.isnan(ci_low) else ""
        txt = f"GPP earlier by {delta:.1f} mo{ci_txt}"
    ax.text(0.5, -0.30, txt, transform=ax.transAxes, ha="center", va="top", fontsize=8.6,
            fontweight="bold", color="#1a1a1a",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="#cccccc"))


def main():
    df = pl.load_master()
    rng = np.random.default_rng(pl.RNG_SEED)

    fig, axes = plt.subplots(2, 4, figsize=(16.5, 8.2), facecolor="white", sharey=True)
    summary_rows = []

    for r, hemi in enumerate(["NH", "SH"]):
        for c, product in enumerate(pl.PRODUCTS):
            ax = axes[r, c]
            sub = df[(df["HEMI"] == hemi) & (df["SIF_PRODUCT"] == product)]
            n_towers = sub["SITE_ID"].nunique()

            sif_res = pl.estimate_pos_pooled(sub, "SIF_VALUE", rng)
            gpp_res = pl.estimate_pos_pooled(sub, "GPP_VALUE", rng)

            if sif_res["curve_grid"] is None or gpp_res["curve_grid"] is None:
                ax.text(0.5, 0.5, f"insufficient coverage\n(n={n_towers} towers)", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9, color="#888888")
                ax.set_xticks([1, 3, 5, 7, 9, 11])
                ax.set_xticklabels(["Jan", "Mar", "May", "Jul", "Sep", "Nov"], fontsize=7.5)
                if c == 0:
                    ax.set_ylabel(f"{HEMI_LABELS[hemi]}\nNormalized value (0-1)", fontsize=8.8)
                if r == 0:
                    ax.set_title(product, fontsize=11, fontweight="bold")
                continue

            grid = sif_res["curve_grid"]
            sif_lo, sif_hi = sub["SIF_VALUE"].groupby(sub["MONTH"]).median().min(), sub["SIF_VALUE"].groupby(sub["MONTH"]).median().max()
            gpp_lo, gpp_hi = sub["GPP_VALUE"].groupby(sub["MONTH"]).median().min(), sub["GPP_VALUE"].groupby(sub["MONTH"]).median().max()
            sif_curve_n = normalize01(sif_res["curve_values"], sif_lo, sif_hi)
            gpp_curve_n = normalize01(gpp_res["curve_values"], gpp_lo, gpp_hi)

            ax.plot(grid, sif_curve_n, "-", color=pl.PRODUCT_COLORS[product], lw=2.4, label="SIF", zorder=4)
            ax.plot(grid, gpp_curve_n, "-", color="#111111", lw=2.0, label="GPP", zorder=5)

            for res, color, ls in [(sif_res, pl.PRODUCT_COLORS[product], "-"), (gpp_res, "#111111", "--")]:
                ax.axvline(res["POS"], color=color, lw=1.6, ls=ls, alpha=0.85, zorder=3)
                if not np.isnan(res["POS_CI_LOW"]):
                    lo_x, hi_x = res["POS_CI_LOW"], res["POS_CI_HIGH"]
                    if lo_x <= hi_x:
                        ax.axvspan(lo_x, hi_x, color=color, alpha=0.10, zorder=1)
                    else:  # CI wraps across Dec/Jan
                        ax.axvspan(lo_x, 12, color=color, alpha=0.10, zorder=1)
                        ax.axvspan(0, hi_x, color=color, alpha=0.10, zorder=1)

            delta, ci_low, ci_high = pl.bootstrap_delta_pos_pooled(sub, rng)
            annotate_offset(ax, delta, ci_low, ci_high)

            flag_bits = []
            if "FLAT_SEASONAL_CYCLE" in sif_res["FLAGS"]:
                flag_bits.append("SIF flat")
            if "FLAT_SEASONAL_CYCLE" in gpp_res["FLAGS"]:
                flag_bits.append("GPP flat")
            if "MULTIMODAL" in sif_res["FLAGS"]:
                flag_bits.append("SIF multimodal")
            if "MULTIMODAL" in gpp_res["FLAGS"]:
                flag_bits.append("GPP multimodal")
            title_extra = f" [{', '.join(flag_bits)}]" if flag_bits else ""

            ax.set_xlim(0, 12)
            ax.set_ylim(-0.08, 1.12)
            ax.set_xticks([1, 3, 5, 7, 9, 11])
            ax.set_xticklabels(["Jan", "Mar", "May", "Jul", "Sep", "Nov"], fontsize=7.5)
            ax.spines[["top", "right"]].set_visible(False)
            if r == 0:
                ax.set_title(f"{product}{title_extra}", fontsize=10.5, fontweight="bold" if not flag_bits else "normal",
                             color="#111111" if not flag_bits else "#a04000")
            if c == 0:
                ax.set_ylabel(f"{HEMI_LABELS[hemi]}\nNormalized value (0-1)", fontsize=8.8)
            ax.text(0.02, 0.96, f"n={n_towers} towers", transform=ax.transAxes, ha="left", va="top",
                    fontsize=7, color="#888888")

            summary_rows.append({
                "HEMISPHERE": hemi, "SIF_PRODUCT": product, "N_TOWERS": n_towers,
                "POS_SIF": sif_res["POS"], "POS_SIF_CI_LOW": sif_res["POS_CI_LOW"], "POS_SIF_CI_HIGH": sif_res["POS_CI_HIGH"],
                "POS_GPP": gpp_res["POS"], "POS_GPP_CI_LOW": gpp_res["POS_CI_LOW"], "POS_GPP_CI_HIGH": gpp_res["POS_CI_HIGH"],
                "DELTA_POS_SIF_MINUS_GPP": delta, "DELTA_CI_LOW": ci_low, "DELTA_CI_HIGH": ci_high,
                "SIF_FLAGS": sif_res["FLAGS"], "GPP_FLAGS": gpp_res["FLAGS"],
            })

    handles = [
        plt.Line2D([0], [0], color="#555555", lw=2.4, label="SIF (product color)"),
        plt.Line2D([0], [0], color="#111111", lw=2.0, label="GPP (dark)"),
        plt.Line2D([0], [0], color="#555555", lw=1.6, ls="-", label="SIF fitted POS (vertical line)"),
        plt.Line2D([0], [0], color="#111111", lw=1.6, ls="--", label="GPP fitted POS (vertical line)"),
        plt.Rectangle((0, 0), 1, 1, fc="#888888", alpha=0.15, label="POS bootstrap 95% CI (shaded)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=8.6)

    fig.suptitle("When do SIF and GPP peak?", fontsize=17, fontweight="bold", y=1.04)
    fig.text(0.5, 1.0,
              "Each panel: pooled seasonal SIF and GPP trajectories (both normalized 0-1), fitted Peak-Of-Season (POS) "
              "marked, offset annotated below each panel with a bootstrap 95% CI.",
              ha="center", fontsize=9.6, color="#333333")
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])

    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT_DIR / f"FigureA_When_Do_SIF_GPP_Peak.{ext}", dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "figureA_pooled_pos_summary.csv", index=False)
    print("Saved FigureA_When_Do_SIF_GPP_Peak.{png,svg,pdf}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
