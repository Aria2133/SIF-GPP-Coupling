"""
RAW-only version of Figure5_IGBP_Summary_4product.png, per explicit request:
same data, same drawing logic, same colormap/thresholds as the original
two-panel (RAW + DESEASONALIZED) figure (11_figure5_igbp_summary.py) -- just
the RAW panel alone, not cropped from the PNG but redrawn from the same
source CSVs so it stays vector-clean at any size.

Source data (unmodified, reused): 02_tables/IGBP_Level_Raw_Relationships.csv.
Products: GOSIF, GOME-2, TROPOSIF, OCO-3 (OCO-2 excluded, matching the
_4product convention used everywhere else in this project, including the
sibling two-panel figure this is a subset of).
"""
# Original script filename (sanitized): 11b_figure5_igbp_summary_raw_only.py; public name: fig07_igbp_raw_coupling_heatmap.py
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
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotting_utils import SEQ_CMAP, savefig

ROOT = PROJECT_ROOT
OUT_ROOT = os.path.join(ROOT, "SIF_EC_Mechanistic_FootprintAware_SIF_GPP_Paper")
TAB = os.path.join(OUT_ROOT, "02_tables")
FIGDATA = os.path.join(OUT_ROOT, "05_figure_data")
OUT_DIR = os.path.join(ROOT, "All_the_Visualizations_REVISED_4PRODUCT_FINAL", "03_recreated_figures_OCO2_removed_only")

PRODUCTS_NO_OCO2 = ["GOSIF", "GOME-2", "TROPOSIF", "OCO-3"]  # OCO-2 excluded, matches the _4product figures
IGBP_ORDER = ["ENF", "EBF", "DBF", "MF", "CSH", "OSH", "WSA", "SAV", "GRA", "CRO"]
MIN_N_CELL = 3
MIN_N_RELIABLE_COLOR = 12  # below this, a "perfect" R^2 is a small-sample artifact, not signal


def build_grid(df):
    grid = pd.DataFrame(index=IGBP_ORDER, columns=PRODUCTS_NO_OCO2, dtype=float)
    n_grid = pd.DataFrame(index=IGBP_ORDER, columns=PRODUCTS_NO_OCO2, dtype=float)
    for _, row in df.iterrows():
        if row["IGBP"] in IGBP_ORDER and row["SIF_PRODUCT"] in PRODUCTS_NO_OCO2 and row["N"] >= MIN_N_CELL:
            grid.loc[row["IGBP"], row["SIF_PRODUCT"]] = row["R2"]
            n_grid.loc[row["IGBP"], row["SIF_PRODUCT"]] = row["N"]
    return grid, n_grid


def draw_heatmap(ax, grid, n_grid, title, vmax):
    reliable = n_grid.values >= MIN_N_RELIABLE_COLOR
    display_vals = np.where(reliable, grid.values, np.nan).astype(float)
    im = ax.imshow(display_vals, cmap=SEQ_CMAP, vmin=0, vmax=vmax, aspect="auto")
    low_n_mask = (~np.isnan(grid.values)) & (~reliable)
    for i, j in zip(*np.where(low_n_mask)):
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#E4E4E0",
                                    edgecolor="none", zorder=1))
    ax.set_xticks(range(len(PRODUCTS_NO_OCO2)))
    ax.set_xticklabels(PRODUCTS_NO_OCO2, rotation=30, ha="right")
    ax.set_yticks(range(len(IGBP_ORDER)))
    ax.set_yticklabels(IGBP_ORDER)
    if title:
        ax.set_title(title, fontsize=10.5, fontweight="bold")
    for i in range(len(IGBP_ORDER)):
        for j in range(len(PRODUCTS_NO_OCO2)):
            v = grid.values[i, j]
            n = n_grid.values[i, j]
            if np.isnan(v):
                ax.text(j, i, "\u2013", ha="center", va="center", fontsize=8, color="#AAAAAA", zorder=2)
                continue
            if n < MIN_N_RELIABLE_COLOR:
                ax.text(j, i, f"{v:.2f}\u2020\n(n={int(n)})", ha="center", va="center",
                         fontsize=7.4, color="#888880", linespacing=1.3, zorder=2)
                continue
            txt_color = "white" if v > vmax * 0.55 else "#1A1A1A"
            ax.text(j, i, f"{v:.2f}\n(n={int(n)})", ha="center", va="center",
                     fontsize=7.8, color=txt_color, linespacing=1.3, zorder=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(PRODUCTS_NO_OCO2), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(IGBP_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    return im


def main():
    raw = pd.read_csv(os.path.join(TAB, "IGBP_Level_Raw_Relationships.csv"))
    grid_raw, n_raw = build_grid(raw)

    fig, ax = plt.subplots(1, 1, figsize=(6.0, 5.2), constrained_layout=True)
    im1 = draw_heatmap(ax, grid_raw, n_raw, None, vmax=0.85)

    cb = fig.colorbar(im1, ax=ax, orientation="vertical", fraction=0.06, pad=0.03)
    cb.set_label("R\u00b2 (SIF vs GPP within biome)", fontsize=8.5)

    fig.suptitle("SIF\u2013GPP coupling strength by biome (IGBP) and product \u2014 RAW",
                  fontsize=12.5, fontweight="bold")
    fig.text(0.5, -0.03, f"\u2020 grayed cells: n<{MIN_N_RELIABLE_COLOR} site-months -- R\u00b2 not "
              "color-mapped (small-sample artifact risk, e.g. a near-perfect fit from 3-7 points)",
              ha="center", fontsize=7.5, color="#555555")

    savefig(fig, OUT_DIR, "Figure5_IGBP_Summary_4product_RAW_ONLY", formats=("png", "pdf", "svg"))
    plt.close(fig)

    long_out = []
    for igbp in IGBP_ORDER:
        for prod in PRODUCTS_NO_OCO2:
            long_out.append({"IGBP": igbp, "SIF_PRODUCT": prod, "TRANSFORM": "RAW",
                              "R2": grid_raw.loc[igbp, prod], "N": n_raw.loc[igbp, prod]})
    os.makedirs(FIGDATA, exist_ok=True)
    pd.DataFrame(long_out).to_csv(os.path.join(OUT_DIR, "Figure5_IGBP_Summary_4product_RAW_ONLY_data.csv"), index=False)
    print("Figure 5 RAW-only done.")


if __name__ == "__main__":
    main()
