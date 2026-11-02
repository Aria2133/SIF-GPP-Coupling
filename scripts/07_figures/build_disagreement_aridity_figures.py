"""
Where and when does SIF-GPP disagreement become severe, and does aridity explain it?

Deliberately lean: two figures, both dense and information-rich, nothing decorative.
Reuses two already-validated tables (no new statistical machinery invented):

  - Decoupling: |SIF_norm - GPP_norm| / (SIF_norm + GPP_norm), 0 (perfect agreement in
    normalized terms) to 1 (severe disagreement), from the Stress-Induced-Decoupling
    work (SIF_EC_Stress_Induced_Decoupling_SID). This is an instance-level divergence
    magnitude, not a correlation coefficient -- a more direct measure of "severe
    disagreement" than pooled rho, which only tells you coupling is weak *somewhere*.
  - Global Aridity Index v3.1 (Zomer/Trabucco), already extracted and joined to all 73
    towers (annual AI, monthly AI climatology, AI class).

Pooled across the 4 SIF products and across years for this diagnostic (a single
severity signal per biome x month, not a product-by-product breakdown) -- this keeps
the figure count and complexity to a minimum, per the explicit request.
"""
from __future__ import annotations
# Original script filename (sanitized): build_disagreement_aridity_figures.py; public name: build_disagreement_aridity_figures.py
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(PROJECT_ROOT)
SID_MASTER = ROOT / "SIF_EC_Stress_Induced_Decoupling_SID" / "03_sid_monthly" / "monthly_SID_master.csv"
ARIDITY_TABLE = ROOT / "All_the_Visualizations_REVISED_4PRODUCT_FINAL_ARIDITY_JUNE_DIAGNOSTICS" / "01_inputs_inventory" / "tower_aridity_annual_and_monthly.csv"
OUT_DIR = ROOT / "SIF_GPP_Severe_Disagreement_Aridity_Diagnosis"
OUT_DIR.mkdir(exist_ok=True)

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
IGBP_LABELS = {
    "ENF": "Evergreen needleleaf forest", "EBF": "Evergreen broadleaf forest",
    "DBF": "Deciduous broadleaf forest", "MF": "Mixed forest", "CSH": "Closed shrubland",
    "OSH": "Open shrubland", "WSA": "Woody savanna", "SAV": "Savanna",
    "GRA": "Grassland", "CRO": "Cropland", "WET": "Wetland",
}
RELIABLE_N_SITEMONTHS = 12
RELIABLE_N_TOWERS = 3

# Muted, restrained palette -- sequential for severity, diverging avoided since this is
# a one-directional magnitude, not a signed quantity.
SEVERITY_CMAP = LinearSegmentedColormap.from_list("severity", ["#f7f7f2", "#d9a441", "#8c2d04"])
ARIDITY_CMAP = LinearSegmentedColormap.from_list("aridity", ["#8c510a", "#f5e8c8", "#01665e"])


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    sid = pd.read_csv(SID_MASTER, low_memory=False,
                       usecols=["SITE_ID", "YEAR", "MONTH", "SIF_PRODUCT", "IGBP", "Decoupling"])
    aridity = pd.read_csv(ARIDITY_TABLE)
    return sid, aridity


def build_sitemonth_table(sid: pd.DataFrame, aridity: pd.DataFrame) -> pd.DataFrame:
    # One Decoupling value per SITE_ID x YEAR x MONTH, averaged across whichever of the
    # 4 products cover that site-month (documented choice -- a single severity signal,
    # not a product breakdown, per the request to keep this minimal).
    sm = sid.groupby(["SITE_ID", "YEAR", "MONTH", "IGBP"], as_index=False)["Decoupling"].mean()
    sm = sm.merge(aridity[["SITE_ID", "AI_ANNUAL", "AI_CLASS"] + [f"AI_M{m:02d}" for m in range(1, 13)]],
                   on="SITE_ID", how="left")
    ai_month_cols = [f"AI_M{m:02d}" for m in range(1, 13)]
    sm["AI_this_month"] = sm.apply(lambda r: r[f"AI_M{int(r['MONTH']):02d}"], axis=1)
    sm = sm.drop(columns=ai_month_cols)
    return sm


def biome_month_summary(sm: pd.DataFrame) -> pd.DataFrame:
    g = sm.groupby(["IGBP", "MONTH"])
    out = g.agg(
        median_decoupling=("Decoupling", "median"),
        N_sitemonths=("Decoupling", "size"),
        N_towers=("SITE_ID", "nunique"),
        mean_AI_this_month=("AI_this_month", "mean"),
    ).reset_index()
    out["reliable"] = (out["N_sitemonths"] >= RELIABLE_N_SITEMONTHS) & (out["N_towers"] >= RELIABLE_N_TOWERS)
    return out


def biome_order(sm: pd.DataFrame) -> list[str]:
    means = sm.groupby("IGBP")["AI_ANNUAL"].mean().sort_values()  # driest first
    return [b for b in means.index if b in IGBP_LABELS]


# ---------------------------------------------------------------------------------
# Figure 1: biome x month severity heatmap, aligned against biome x month aridity
# ---------------------------------------------------------------------------------

def make_figure1(sm: pd.DataFrame):
    bm = biome_month_summary(sm)
    biomes = biome_order(sm)
    n_biomes = len(biomes)

    sev = np.full((n_biomes, 12), np.nan)
    ai = np.full((n_biomes, 12), np.nan)
    reliable = np.zeros((n_biomes, 12), dtype=bool)
    for i, b in enumerate(biomes):
        for m in range(1, 12 + 1):
            row = bm[(bm["IGBP"] == b) & (bm["MONTH"] == m)]
            if len(row):
                sev[i, m - 1] = row["median_decoupling"].values[0]
                ai[i, m - 1] = row["mean_AI_this_month"].values[0]
                reliable[i, m - 1] = row["reliable"].values[0]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9.5, 8.6), facecolor="white",
        gridspec_kw={"height_ratios": [n_biomes, n_biomes], "hspace": 0.55},
    )

    im1 = ax1.imshow(sev, cmap=SEVERITY_CMAP, vmin=0, vmax=np.nanpercentile(sev, 95), aspect="auto")
    for i in range(n_biomes):
        for j in range(12):
            if np.isfinite(sev[i, j]) and not reliable[i, j]:
                ax1.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                             hatch="////", edgecolor="#888888", linewidth=0))
    ax1.set_yticks(range(n_biomes))
    ax1.set_yticklabels([f"{IGBP_LABELS[b]}" for b in biomes], fontsize=8.5)
    ax1.set_xticks(range(12))
    ax1.set_xticklabels(MONTH_NAMES, fontsize=8.5)
    ax1.set_title("(a) SIF-GPP disagreement severity (median Decoupling, all products & years pooled)",
                   fontsize=10, loc="left")
    ax1.tick_params(length=0)

    im2 = ax2.imshow(ai, cmap=ARIDITY_CMAP, vmin=0, vmax=2.0, aspect="auto")
    ax2.set_yticks(range(n_biomes))
    ax2.set_yticklabels([f"{IGBP_LABELS[b]}" for b in biomes], fontsize=8.5)
    ax2.set_xticks(range(12))
    ax2.set_xticklabels(MONTH_NAMES, fontsize=8.5)
    ax2.set_title("(b) Mean monthly Aridity Index at the same towers (lower = drier)",
                   fontsize=10, loc="left")
    ax2.tick_params(length=0)

    # Two explicit, independently-positioned colorbar axes (figure-fraction coords) --
    # avoids relying on colorbar()'s `anchor=` behavior inside a shared placeholder axes,
    # which clipped text off the canvas in an earlier draft.
    cax1 = fig.add_axes([0.24, 0.075, 0.32, 0.018])
    cb1 = fig.colorbar(im1, cax=cax1, orientation="horizontal")
    cb1.set_label("Median Decoupling (0 = agreement, 1 = severe disagreement)", fontsize=7.5)
    cb1.ax.tick_params(labelsize=7.5)

    cax2 = fig.add_axes([0.63, 0.075, 0.32, 0.018])
    cb2 = fig.colorbar(im2, cax=cax2, orientation="horizontal")
    cb2.set_label("Aridity Index (<0.65 semi-arid or drier; >1.0 humid)", fontsize=7.5)
    cb2.ax.tick_params(labelsize=7.5)

    fig.suptitle("When and where does the SIF-GPP relationship disagree most, and is it dry when it does?",
                 fontsize=12, y=0.995)
    fig.text(0.5, 0.955, "Biomes ordered driest (top) to wettest (bottom) by mean annual Aridity Index. "
                          "Hatched cells: N<12 site-months or N<3 towers.",
              ha="center", fontsize=8, color="#555555")

    fig.subplots_adjust(left=0.24, right=0.97, top=0.88, bottom=0.20)

    out_stub = OUT_DIR / "Figure1_Disagreement_vs_Aridity_ByBiomeMonth"
    fig.savefig(out_stub.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(out_stub.with_suffix(".pdf"), facecolor="white")
    fig.savefig(out_stub.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    print(f"Saved {out_stub.name}.{{png,pdf,svg}}")
    return bm


# ---------------------------------------------------------------------------------
# Figure 2: quantitative test -- does monthly aridity predict disagreement magnitude?
# ---------------------------------------------------------------------------------

def make_figure2(sm: pd.DataFrame):
    biomes = biome_order(sm)
    rows = []
    for b in biomes:
        sub = sm[sm["IGBP"] == b].dropna(subset=["AI_this_month", "Decoupling"])
        n_towers = sub["SITE_ID"].nunique()
        n = len(sub)
        if n_towers < RELIABLE_N_TOWERS or n < RELIABLE_N_SITEMONTHS:
            rows.append({"IGBP": b, "N": n, "N_towers": n_towers, "rho": np.nan,
                         "ci_low": np.nan, "ci_high": np.nan, "reliable": False})
            continue
        rho, p = stats.spearmanr(sub["AI_this_month"], sub["Decoupling"])
        # tower-block bootstrap CI
        towers = sub["SITE_ID"].unique()
        boots = []
        rng = np.random.default_rng(20260907)
        for _ in range(1000):
            pick = rng.choice(towers, size=len(towers), replace=True)
            bsub = sub[sub["SITE_ID"].isin(pick)]
            if bsub["AI_this_month"].nunique() < 2:
                continue
            r, _ = stats.spearmanr(bsub["AI_this_month"], bsub["Decoupling"])
            if np.isfinite(r):
                boots.append(r)
        ci_low, ci_high = (np.percentile(boots, [2.5, 97.5]) if len(boots) > 100 else (np.nan, np.nan))
        rows.append({"IGBP": b, "N": n, "N_towers": n_towers, "rho": rho,
                     "ci_low": ci_low, "ci_high": ci_high, "reliable": True, "p": p})
    forest = pd.DataFrame(rows)
    forest.to_csv(OUT_DIR / "biome_aridity_disagreement_correlation.csv", index=False)

    pooled = sm.dropna(subset=["AI_this_month", "Decoupling"])
    pooled = pooled[pooled["IGBP"].isin(biomes)]
    rho_all, p_all = stats.spearmanr(pooled["AI_this_month"], pooled["Decoupling"])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 5.0), facecolor="white",
                                    gridspec_kw={"width_ratios": [1.15, 1]})

    biome_colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(biomes)))
    color_map = dict(zip(biomes, biome_colors))
    for b in biomes:
        sub = pooled[pooled["IGBP"] == b]
        axL.scatter(sub["AI_this_month"], sub["Decoupling"], s=8, alpha=0.35,
                    color=color_map[b], edgecolor="none", label=IGBP_LABELS[b])
    XCAP = 3.0
    frac_shown = (pooled["AI_this_month"] <= XCAP).mean()
    axL.set_xlim(0, XCAP)
    axL.set_xlabel("Aridity Index that month at that tower (lower = drier)", fontsize=9.5)
    axL.set_ylabel("Decoupling (site-month)", fontsize=9.5)
    axL.set_title("(a) Pooled: monthly aridity vs. disagreement magnitude", fontsize=10, loc="left")
    axL.text(0.98, 0.14, f"pooled Spearman \u03c1 = {rho_all:.2f}\nn = {len(pooled):,} site-months",
              transform=axL.transAxes, ha="right", va="bottom", fontsize=9, color="#333333")
    axL.text(0.98, 0.03, f"(x-axis capped at {XCAP:.0f}; {frac_shown:.0%} of points shown, "
                          f"all points used in \u03c1)",
              transform=axL.transAxes, ha="right", va="bottom", fontsize=6.8, color="#777777", style="italic")
    axL.spines[["top", "right"]].set_visible(False)
    axL.legend(fontsize=6.5, loc="upper left", ncol=1, markerscale=1.8,
               frameon=True, facecolor="white", framealpha=0.9, edgecolor="none")

    plot_forest = forest.dropna(subset=["rho"]).sort_values("rho")
    y = np.arange(len(plot_forest))
    axR.errorbar(plot_forest["rho"], y,
                 xerr=[plot_forest["rho"] - plot_forest["ci_low"], plot_forest["ci_high"] - plot_forest["rho"]],
                 fmt="o", color="#8c2d04", ecolor="#8c2d04", elinewidth=1.2, capsize=2.5, ms=5)
    axR.axvline(0, color="#999999", lw=0.8, ls="--")
    axR.set_yticks(y)
    axR.set_yticklabels([f"{IGBP_LABELS[b]}  (n={int(nt)} towers)"
                          for b, nt in zip(plot_forest["IGBP"], plot_forest["N_towers"])], fontsize=8.5)
    axR.set_xlabel("Spearman \u03c1 (monthly aridity vs. Decoupling)", fontsize=9.5)
    axR.set_title("(b) Per-biome association, with tower-bootstrap 95% CI", fontsize=10, loc="left")
    axR.spines[["top", "right"]].set_visible(False)

    omitted = forest[~forest["reliable"]]["IGBP"].tolist()
    footer = ("Negative \u03c1 = drier months show more disagreement. "
              "Decoupling reused unmodified from the Stress-Induced-Decoupling analysis "
              "(pooled across the 4 SIF products and all years). Descriptive, cross-site association -- not a causal test.")
    if omitted:
        footer += f" Omitted for insufficient support (<{RELIABLE_N_TOWERS} towers or <{RELIABLE_N_SITEMONTHS} site-months): " + ", ".join(IGBP_LABELS[b] for b in omitted) + "."
    fig.text(0.5, -0.06, footer, ha="center", fontsize=7.6, color="#555555", wrap=True)

    fig.suptitle("Does aridity explain the severity of SIF-GPP disagreement?", fontsize=12, y=1.03)
    fig.tight_layout()

    out_stub = OUT_DIR / "Figure2_Aridity_Explains_Disagreement"
    fig.savefig(out_stub.with_suffix(".png"), dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(out_stub.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    fig.savefig(out_stub.with_suffix(".svg"), facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_stub.name}.{{png,pdf,svg}}")
    print(f"Pooled Spearman rho (AI vs Decoupling) = {rho_all:.3f}, p = {p_all:.2e}, n = {len(pooled)}")
    return forest, rho_all


def main():
    sid, aridity = load_data()
    sm = build_sitemonth_table(sid, aridity)
    bm = make_figure1(sm)
    forest, rho_all = make_figure2(sm)

    print("\n=== SUMMARY ===")
    print(forest[["IGBP", "N_towers", "N", "rho", "ci_low", "ci_high"]].to_string(index=False))
    print(f"\nOutputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
