"""
FIGURE B -- "Is it true at individual towers?"
Per-product scatter: x = fitted GPP POS, y = fitted SIF POS, bold 1:1 line, points
colored by biome, shaped by hemisphere. Points below the 1:1 line = SIF peaks earlier;
above = GPP peaks earlier.

Circular/phenological handling for the 1:1 comparison: rather than unwrapping each
point's coordinates (which would require choosing an arbitrary reference month), the
main diagonal is drawn THREE times -- y=x, y=x-12, y=x+12 -- so a tower whose POS pair
straddles the Dec/Jan boundary (common for Southern Hemisphere sites, whose peak season
falls there) still visibly lines up with one of the three parallel copies of the 1:1
relationship instead of appearing spuriously far from the diagonal. All SUMMARY
STATISTICS (median DeltaPOS, bootstrap CI, % SIF-earlier) are computed from
circular_diff() (shortest signed distance on the 12-month circle), never from the raw
plotted x-y difference, so they are correct regardless of where a point falls on the
page.
"""
# Original script filename (sanitized): figureB_tower_level_agreement.py; public name: figureB_tower_level_agreement.py
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

OUT_DIR = (Path(PROJECT_ROOT) / 'All_the_Visualizations_REVISED/EXACT_REQUESTS_ONLY/PHENOLOGY_POS_REVISION/figureB_tower_level_agreement.py').resolve().parent

IGBP_LABELS = {
    "ENF": "Evergreen needleleaf forest", "EBF": "Evergreen broadleaf forest",
    "DBF": "Deciduous broadleaf forest", "MF": "Mixed forest", "CSH": "Closed shrubland",
    "OSH": "Open shrubland", "WSA": "Woody savanna", "SAV": "Savanna",
    "GRA": "Grassland", "CRO": "Cropland", "WET": "Wetland",
}
BIOME_COLORS = dict(zip(IGBP_LABELS, plt.get_cmap("tab10").colors + plt.get_cmap("tab10").colors[:1]))
HEMI_MARKERS = {"NH": "o", "SH": "^"}


def bootstrap_median_delta(deltas: np.ndarray, rng: np.random.Generator, n_boot: int = 2000):
    if len(deltas) == 0:
        return np.nan, np.nan, np.nan
    point = float(np.median(deltas))
    boots = [np.median(rng.choice(deltas, size=len(deltas), replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def main():
    df = pl.load_master()
    tower_pos = pl.build_tower_pos_table(df)
    tower_pos.to_csv(OUT_DIR / "tower_pos_table.csv", index=False)

    valid = tower_pos.dropna(subset=["SIF_POS", "GPP_POS"]).copy()
    valid["IGBP_LABEL"] = valid["IGBP"].where(valid["IGBP"].isin(IGBP_LABELS), "Other")

    rng = np.random.default_rng(pl.RNG_SEED)
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 5.0), facecolor="white")
    stats_rows = []

    for ax, product in zip(axes, pl.PRODUCTS):
        sub = valid[valid["SIF_PRODUCT"] == product]
        for _, row in sub.iterrows():
            color = BIOME_COLORS.get(row["IGBP_LABEL"], "#999999")
            marker = HEMI_MARKERS.get(row["HEMI"], "o")
            ax.scatter(row["GPP_POS"], row["SIF_POS"], color=color, marker=marker, s=48,
                       alpha=0.85, edgecolor="white", linewidth=0.5, zorder=4)

        ax.plot([0, 12], [0, 12], color="#111111", lw=1.8, zorder=2, label="1:1 (synchronous)" if product == pl.PRODUCTS[0] else None)
        ax.plot([0, 12], [-12, 0], color="#999999", lw=1.0, ls="--", zorder=1)
        ax.plot([0, 12], [12, 24], color="#999999", lw=1.0, ls="--", zorder=1)
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 12)
        ax.set_xticks([1, 3, 5, 7, 9, 11])
        ax.set_yticks([1, 3, 5, 7, 9, 11])
        ax.set_xticklabels(["Jan", "Mar", "May", "Jul", "Sep", "Nov"], fontsize=7.5)
        ax.set_yticklabels(["Jan", "Mar", "May", "Jul", "Sep", "Nov"], fontsize=7.5)
        ax.set_aspect("equal")
        ax.set_xlabel("Fitted GPP POS (month)", fontsize=8.8)
        if product == pl.PRODUCTS[0]:
            ax.set_ylabel("Fitted SIF POS (month)", fontsize=8.8)
        ax.set_title(product, fontsize=11.5, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)

        deltas = sub["DELTA_POS"].dropna().to_numpy()
        n = len(deltas)
        point, ci_lo, ci_hi = bootstrap_median_delta(deltas, rng)
        pct_sif_earlier = 100.0 * np.mean(deltas < 0) if n else np.nan
        stats_rows.append({"SIF_PRODUCT": product, "N_TOWERS": n, "MEDIAN_DELTA_POS": point,
                           "CI_LOW": ci_lo, "CI_HIGH": ci_hi, "PCT_SIF_EARLIER": pct_sif_earlier})

        ax.text(0.03, 0.97,
                f"N={n} towers\nmedian ΔPOS={point:+.1f} mo\n95% CI [{ci_lo:+.1f}, {ci_hi:+.1f}]\n"
                f"{pct_sif_earlier:.0f}% SIF earlier",
                transform=ax.transAxes, ha="left", va="top", fontsize=7.8,
                bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#cccccc", alpha=0.92), zorder=6)

    biome_present = [b for b in IGBP_LABELS if b in valid["IGBP_LABEL"].unique()]
    biome_handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=BIOME_COLORS[b],
                                 markeredgecolor="white", markersize=7, label=IGBP_LABELS[b]) for b in biome_present]
    if "Other" in valid["IGBP_LABEL"].unique():
        biome_handles.append(plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#999999",
                                         markeredgecolor="white", markersize=7, label="Other/unclassified"))
    hemi_handles = [plt.Line2D([0], [0], marker=m, color="#333333", linestyle="none", markersize=7, label=h)
                    for h, m in HEMI_MARKERS.items()]
    leg1 = fig.legend(handles=biome_handles, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.07),
                       fontsize=7.6, frameon=False, title="Biome (color)", title_fontsize=8)
    fig.legend(handles=hemi_handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.16),
               fontsize=7.6, frameon=False, title="Hemisphere (marker shape)", title_fontsize=8)
    fig.add_artist(leg1)

    fig.suptitle("Is the SIF-before-GPP timing offset true at individual towers?", fontsize=15, fontweight="bold", y=1.06)
    fig.text(0.5, 1.0,
              "Each point = one tower. Below the black 1:1 line = SIF peaks earlier there; above = GPP peaks earlier. "
              "Gray dashed lines are the same 1:1 relationship shifted by one calendar year, so towers whose peak "
              "straddles the Dec/Jan boundary (mostly Southern Hemisphere) still read correctly.",
              ha="center", fontsize=9, color="#333333")
    fig.tight_layout(rect=[0, 0.1, 1, 0.94])

    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT_DIR / f"FigureB_Tower_Level_Agreement.{ext}", dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(OUT_DIR / "figureB_product_summary_stats.csv", index=False)
    print("Saved FigureB_Tower_Level_Agreement.{png,svg,pdf}")
    print(stats_df.to_string(index=False))
    print(f"\nTotal valid tower x product POS pairs: {len(valid)} (of {len(tower_pos)} attempted)")
    print("Flag counts (SIF):")
    print(valid["SIF_FLAGS"].value_counts())


if __name__ == "__main__":
    main()
