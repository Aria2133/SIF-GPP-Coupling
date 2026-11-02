"""
Add LST as a 6th panel to figure_environmental_controls_geospatial_main.png AND to
each of the 4 product-specific maps in 03_product_specific_geospatial/, IN PLACE
(same filenames, overwritten), per explicit instruction: no other change to the
existing 5 panels' content -- same map style, same marker color/size logic, same
annotation text, same titles -- ONLY the layout changes: LST goes in the
bottom-right cell (where the legend used to sit), and the colorbar / size legend /
coverage text move to one horizontal strip BELOW the 2x3 grid of maps.

Reuses every plotting function from the verified original script
(build_environmental_controls_geospatial.py) unchanged, imported as a module --
draw_map_panel, make_terciles, spearman_rho, per_product_rho_text, load_data are
not reimplemented, so the 5 existing panels render identically to before.

The original Natural Earth raster basemap (NE1_HR_LC_SR_W.tif) was missing from its
original path when this script was first written; it has since been located at
<local path removed>
and build_environmental_controls_geospatial.py's own BASEMAP_TIF constant has been
updated to that path -- this script now uses the REAL raster basemap (src.load_basemap /
src.add_basemap, unmodified), restoring the exact original visual style. Nothing else
about the 5 existing panels changed.
"""
# Original script filename (sanitized): add_LST_panel_inplace.py; public name: fig12_figS1_S4_add_LST_panel.py
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

warnings.filterwarnings("ignore")

ROOT = Path(PROJECT_ROOT)
SRC_DIR = ROOT / "All_the_Visualizations_REVISED_4PRODUCT_UPDATE_V3" / "07_scripts"
OUT_MAIN_DIR = ROOT / "All_the_Visualizations_REVISED_4PRODUCT_UPDATE_V3" / "01_main_revised"
OUT_PRODUCT_DIR = ROOT / "All_the_Visualizations_REVISED_4PRODUCT_UPDATE_V3" / "03_product_specific_geospatial"
LST_DIR = ROOT / "LST_SIF_GPP_EXTENSION_FINAL" / "02_processed_LST"

sys.path.insert(0, str(SRC_DIR))
import importlib
src = importlib.import_module("build_environmental_controls_geospatial")  # noqa: E402


def draw_legend_strip(ax, sc_ref, cbar_label, coverage_text):
    """Horizontal legend strip (colorbar + size legend + coverage text side by side),
    replacing the original's vertical legend PANEL (which used to occupy the
    bottom-right grid cell -- that cell now holds the LST map instead)."""
    ax.axis("off")

    # colorbar (left third)
    cax = ax.inset_axes([0.0, 0.55, 0.28, 0.30])
    cb = plt.colorbar(sc_ref, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=7.6)
    ax.text(0.14, 0.95, "SIF-GPP coupling (site-level Pearson r)", transform=ax.transAxes,
            fontsize=9.2, fontweight="bold", ha="center", va="bottom")
    ax.text(0.14, 0.42, cbar_label, transform=ax.transAxes, fontsize=7.4, ha="center", va="top", color="#333333")

    # size legend (middle third)
    ax.text(0.47, 0.95, "Marker size = driver tercile", transform=ax.transAxes,
            fontsize=9.2, fontweight="bold", ha="center", va="bottom")
    src.draw_size_legend(ax, x0=0.40, y0=0.55, dx=0.07, fontsize=7.8)

    # coverage text (right third)
    ax.text(0.68, 0.95, src.wrap_text(coverage_text, 72), transform=ax.transAxes,
            fontsize=7.4, ha="left", va="top", color="#222222")


def build_6panel(site_r, site_level_lst, r_col, tower_or_prod_r, rho_table, all_drivers,
                  title, coverage_txt, cbar_label, out_dir, out_stem, basemap_rgb, basemap_extent,
                  single_product=None):
    fig = plt.figure(figsize=(19.0, 11.2))
    gs = fig.add_gridspec(3, 3, left=0.03, right=0.98, top=0.86, bottom=0.03,
                           height_ratios=[1.0, 1.0, 0.42], wspace=0.05, hspace=0.18)
    axes = [fig.add_subplot(gs[0, 0], projection=ccrs.EqualEarth()),
            fig.add_subplot(gs[0, 1], projection=ccrs.EqualEarth()),
            fig.add_subplot(gs[0, 2], projection=ccrs.EqualEarth()),
            fig.add_subplot(gs[1, 0], projection=ccrs.EqualEarth()),
            fig.add_subplot(gs[1, 1], projection=ccrs.EqualEarth()),
            fig.add_subplot(gs[1, 2], projection=ccrs.EqualEarth())]
    ax_legend = fig.add_subplot(gs[2, :])

    for ax, driver in zip(axes, all_drivers):
        tercile_df, edges = src.make_terciles(site_level_lst, driver["col"])
        plot_df = tower_or_prod_r.merge(tercile_df[["SITE_ID", "BIN"]], on="SITE_ID", how="inner")
        if single_product is None:
            rho_text = src.per_product_rho_text(rho_table, driver["key"])
        else:
            rho_text = src.single_rho_text(rho_table, driver["key"], single_product)
        src.draw_map_panel(ax, driver, plot_df, r_col, basemap_rgb, basemap_extent, rho_text)

    sc_ref = plt.cm.ScalarMappable(norm=src.COUPLING_NORM, cmap=src.COUPLING_CMAP)
    draw_legend_strip(ax_legend, sc_ref, cbar_label, coverage_txt)

    fig.suptitle(src.wrap_text(title, 130), fontsize=15.5, fontweight="bold", y=0.975, x=0.015, ha="left")
    fig.text(0.015, 0.905, src.wrap_text(src.SUBTITLE, 220), fontsize=8.8, color="#2a2a2a", ha="left", va="top")

    for ext, dpi in (("png", 600), ("pdf", 300), ("svg", 300)):
        out_path = out_dir / f"{out_stem}.{ext}"
        fig.savefig(out_path, dpi=dpi, facecolor="white", pad_inches=0.15)
        print(f"  saved: {out_path}")
    plt.close(fig)


def main():
    print("Loading real Natural Earth basemap ...")
    basemap_rgb, basemap_extent = src.load_basemap(target_width_px=6000, white_blend=0.32)

    site_r, site_level, pooled, coords = src.load_data()
    tower_lst = pd.read_csv(LST_DIR / "tower_monthly_LST.csv")
    lst_mean = tower_lst.groupby("SITE_ID", as_index=False)["LST_degC"].mean().rename(columns={"LST_degC": "LST"})
    site_level_lst = site_level.merge(lst_mean, on="SITE_ID", how="left")
    lst_driver = dict(key="LST", col="LST", title="Land surface temperature (LST)", unit="\u00b0C",
                       cbar_label="Land surface temperature (tower long-term mean, \u00b0C)")
    all_drivers = src.DRIVERS + [lst_driver]

    # ===== MAIN (pooled) figure, overwritten in place =====
    print("Rebuilding MAIN figure (6 panels, LST bottom-right, legend below) ...")
    rho_table = {}
    for driver in src.DRIVERS:
        rho_table[driver["key"]] = {p: src.spearman_rho(site_level, site_r, driver["col"], p) for p in src.PRODUCTS}
    rho_table["LST"] = {p: src.spearman_rho(site_level_lst, site_r, "LST", p) for p in src.PRODUCTS}

    tower_r = pooled.merge(site_level_lst[["SITE_ID", "LATITUDE", "LONGITUDE"]].drop_duplicates("SITE_ID"),
                            on="SITE_ID", how="left")
    n_multi = int((pooled["N_PRODUCTS"] > 1).sum())
    n_lst = int(site_level_lst["LST"].notna().sum())
    coverage_txt = (
        f"Each tower's fill color is the MEAN site-level Pearson r across whichever of the 4 main-paper "
        f"products cover it ({n_multi} of {len(pooled)} towers covered by \u22652 products). LST panel: marker "
        f"size reflects tower long-term mean LST (MYD21C1), available for {n_lst} of 73 towers "
        f"(see LST_SIF_GPP_EXTENSION_FINAL/01_data_audit/LST_DATA_AUDIT.md)."
    )
    build_6panel(site_r, site_level_lst, "POOLED_R", tower_r, rho_table, all_drivers,
                 src.TITLE, coverage_txt, "mean across whichever products cover each tower",
                 OUT_MAIN_DIR, "figure_environmental_controls_geospatial_main",
                 basemap_rgb, basemap_extent)

    # ===== 4 product-specific figures, each overwritten in place =====
    for product in src.PRODUCTS:
        print(f"Rebuilding {product} figure (6 panels, LST bottom-right, legend below) ...")
        prod_r = site_r[site_r["SIF_PRODUCT"] == product][["SITE_ID", "PEARSON_R"]].drop_duplicates("SITE_ID")
        prod_r = prod_r.merge(site_level_lst[["SITE_ID", "LATITUDE", "LONGITUDE"]].drop_duplicates("SITE_ID"),
                               on="SITE_ID", how="left")
        n_lst_prod = int(prod_r.merge(site_level_lst[["SITE_ID", "LST"]], on="SITE_ID", how="left")["LST"].notna().sum())
        coverage_txt_p = (
            f"This map shows {product}'s OWN site-level Pearson r only (no pooling across products). "
            f"LST panel: marker size reflects tower long-term mean LST (MYD21C1), available for {n_lst_prod} "
            f"of {len(prod_r)} {product}-covered towers (see LST_SIF_GPP_EXTENSION_FINAL/01_data_audit/"
            f"LST_DATA_AUDIT.md)."
        )
        title_p = f"{src.TITLE} \u2014 {product} only"
        stem = f"figure_environmental_controls_geospatial_{product.replace('-', '')}"
        build_6panel(site_r, site_level_lst, "PEARSON_R", prod_r, rho_table, all_drivers,
                     title_p, coverage_txt_p, f"{product} only (no pooling across products)",
                     OUT_PRODUCT_DIR, stem, basemap_rgb, basemap_extent, single_product=product)

    print("\nDONE -- all 5 figures overwritten in place with LST as the 6th (bottom-right) panel.")


if __name__ == "__main__":
    main()
