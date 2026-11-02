"""
Biome x Hemisphere seasonal SID figures -- extension of the existing Stress-Induced
Decoupling (SID) analysis under SIF_EC_Stress_Induced_Decoupling_SID/.

Context / what already exists (read-only inputs, never modified):
  - SIF_EC_Stress_Induced_Decoupling_SID/03_sid_monthly/monthly_SID_master.csv
    (SITE_ID, YYYYMM, YEAR, MONTH, SIF_PRODUCT, LATITUDE, LONGITUDE, IGBP,
    SID_VPD, SID_SM, SID_TEMP, plus QC_FLAG_NOTES/LOW_DENOMINATOR_FLAG). Main-paper
    products only: GOSIF, GOME-2, TROPOSIF, OCO-3 (OCO-2 already excluded upstream).
  - SIF_EC_Stress_Induced_Decoupling_SID/06_biome_hemisphere/05_biome_hemisphere_seasonal.py
    already built: (a) SID_Figure_Seasonal_NH.png / _SH.png -- monthly SID
    climatology pooling ALL biomes, split by hemisphere; (b) SID_Figure_Seasonal_Biome_
    {CRO,DBF,EBF,ENF,GRA,MF,OSH,SAV,WSA}.png -- monthly SID climatology per biome,
    POOLING both hemispheres together within each biome.
  - Confirmed by directory listing before writing this script: NO existing output
    anywhere under SIF_EC_Stress_Induced_Decoupling_SID/ crosses biome x hemisphere
    (i.e. NH-grassland vs SH-grassland shown distinctly). That is the genuinely new
    piece this script adds. Nothing under SIF_EC_Stress_Induced_Decoupling_SID/ is
    modified; all new outputs go to a separate, new directory tree.

This script is READ-ONLY with respect to SID itself: it never recomputes
SID_VPD/SID_SM/SID_TEMP, it only re-aggregates/re-visualizes the existing master
table, split jointly by IGBP biome AND hemisphere (LATITUDE>=0 => NH, else SH).

Reliability rule (same established project convention, reused unchanged):
a cell needs N >= 12 site-months AND N >= 3 independent towers to be "reliable".
Unreliable cells/months are always shown de-emphasized (hollow markers / faded
lines), never hidden -- and a whole biome x hemisphere combination is either
built (if it clears the bar for >=1 stressor, pooling across all 4 products) or
explicitly OMITTED with a stated reason; never fabricated.

Biome-grouping decisions made by this script (stated explicitly, not hidden):
  - GRA and CRO are used as single IGBP codes (no merging needed or attempted).
  - SAV and WSA are each checked INDIVIDUALLY per hemisphere (matching the
    existing pipeline's convention of never merging IGBP codes). No merged
    "Savanna" super-class is created: WSA_SH fails the reliability bar on its own
    (2 towers) and is *omitted* rather than propped up by pooling with SAV_SH,
    because the task's explicit pooling license was written for forest subtypes,
    not savanna subtypes, and manufacturing a merge to rescue a marginal cell
    would be exactly the kind of overinterpretation this task warns against.
  - Forest subtypes (ENF, EBF, DBF, MF) were CHECKED for the "pool into a single
    Forest group" fallback the task allows, but pooling turned out to be
    unnecessary in both hemispheres:
      * NH: ENF, EBF, DBF, MF are all individually reliable (>=3 towers, >=12
        site-months each) -- so each is reported as its own NH panel.
      * SH: ENF, DBF, MF have ZERO SH towers each (not merely low-N -- no data
        at all); only EBF has SH towers (4 towers, reliable). A pooled
        "Forest_SH" would therefore be numerically IDENTICAL to EBF_SH alone
        under a different label, so EBF_SH is reported directly and no
        redundant pooled "Forest" file is produced.
    This decision (pooling considered, found unnecessary, and why) is recorded
    both here and in the companion support-summary CSV.

Figure design/style is REUSED, not reinvented, from
build_seasonal_biome_figures() in 05_biome_hemisphere_seasonal.py: three panels
(VPD stress / SM stress / TEMP stress) of calendar-month median SID, pooled
across the 4 SIF products, shaded local-summer band, hollow markers for
below-threshold months. The only deliberate deviations: (1) 600 DPI PNG per the
task spec (vs 350 DPI in the original), (2) title/caption text reflects the
specific biome x hemisphere combination and states the pooling decision, (3)
output goes to the new 06_sid_figures/ directory, not 06_biome_hemisphere/.

Outputs (under All_the_Visualizations_REVISED_4PRODUCT_FINAL_ARIDITY_JUNE_DIAGNOSTICS/):
  06_sid_figures/SID_Figure_Seasonal_<Biome>_<Hemisphere>.{png,svg,pdf}   (qualifying combos only)
  06_sid_figures/SID_Figure_Seasonal_BiomeHemisphere_Overview.{png,svg,pdf}  (bonus combined grid)
  06_sid_figures/SID_biome_hemisphere_support_summary.csv

Author: project authors
Run date: 2026-09-06
"""
from __future__ import annotations
# Original script filename (sanitized): 06_sid_biome_hemisphere_seasonal.py; public name: 06_sid_biome_hemisphere_seasonal.py
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

import textwrap
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------

ROOT = Path(PROJECT_ROOT)
SID_ROOT = ROOT / "SIF_EC_Stress_Induced_Decoupling_SID"
MASTER_CSV = SID_ROOT / "03_sid_monthly" / "monthly_SID_master.csv"

OUT_DIR = ROOT / "All_the_Visualizations_REVISED_4PRODUCT_FINAL_ARIDITY_JUNE_DIAGNOSTICS" / "06_sid_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRODUCTS = ["GOSIF", "GOME-2", "TROPOSIF", "OCO-3"]
STRESSORS = {"VPD": "SID_VPD", "SM": "SID_SM", "TEMP": "SID_TEMP"}
STRESS_ORDER = ["VPD", "SM", "TEMP"]

REL_MIN_N = 12
REL_MIN_TOWERS = 3
LOW_DENOM_FLAG = "LOW_DENOMINATOR_FLAG"

STRESSOR_COLORS = {"VPD": "#D55E00", "SM": "#0072B2", "TEMP": "#009E73"}

IGBP_LABELS = {
    "CRO": "Cropland", "CSH": "Closed shrubland", "DBF": "Deciduous broadleaf forest",
    "EBF": "Evergreen broadleaf forest", "ENF": "Evergreen needleleaf forest", "GRA": "Grassland",
    "MF": "Mixed forest", "OSH": "Open shrubland", "SAV": "Savanna", "WSA": "Woody savanna",
    "WET": "Wetland", "DNF": "Deciduous needleleaf forest", "SNO": "Snow/ice",
}
IGBP_ORDER = ["ENF", "EBF", "DBF", "MF", "DNF", "CSH", "OSH", "WSA", "SAV", "GRA", "CRO", "WET", "SNO"]

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.6,
})

# Biomes the task requires attempting "at minimum" (grassland, savanna/woody
# savanna, cropland, forest group members) -- checked individually per the
# pooling decisions documented above.
REQUIRED_BIOMES = ["GRA", "SAV", "WSA", "CRO", "ENF", "EBF", "DBF", "MF"]


def wrap_text(text: str, width: int = 150) -> str:
    lines = []
    for para in text.split("\n"):
        wrapped = textwrap.wrap(para, width=width) or [""]
        lines.extend(wrapped)
    return "\n".join(lines)


def save_figure(fig, stem: str, out_dir: Path, dpi: int = 600):
    png = out_dir / f"{stem}.png"
    svg = out_dir / f"{stem}.svg"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(svg, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved: {png}")


# --------------------------------------------------------------------------------------
# Data loading (read-only reuse of Phase-1 output)
# --------------------------------------------------------------------------------------

def load_master() -> pd.DataFrame:
    df = pd.read_csv(MASTER_CSV, low_memory=False)
    assert not (df["SIF_PRODUCT"] == "OCO-2").any()
    df["LOW_DENOM"] = df["QC_FLAG_NOTES"].fillna("").str.contains(LOW_DENOM_FLAG)
    df["HEMISPHERE"] = np.where(df["LATITUDE"] >= 0, "NH", "SH")
    return df


def reliable(n_sitemonths: int, n_towers: int) -> bool:
    return (n_sitemonths >= REL_MIN_N) and (n_towers >= REL_MIN_TOWERS)


def cell_stats(sub: pd.DataFrame, sid_col: str) -> dict:
    vals = sub[sid_col].dropna()
    n = len(vals)
    towers = sub.loc[vals.index, "SITE_ID"].nunique()
    if n == 0:
        return dict(N=0, N_towers=0, median=np.nan, reliable=False)
    return dict(N=n, N_towers=towers, median=vals.median(), reliable=reliable(n, towers))


def monthly_stats(sub: pd.DataFrame, sid_col: str) -> pd.DataFrame:
    rows = []
    for m in range(1, 13):
        msub = sub[sub["MONTH"] == m].dropna(subset=[sid_col])
        n = len(msub)
        towers = msub["SITE_ID"].nunique()
        rows.append(dict(MONTH=m, N=n, N_towers=towers,
                          median=msub[sid_col].median() if n else np.nan,
                          reliable=reliable(n, towers)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Step 1: full biome x hemisphere x product support summary (all IGBP codes present,
# not just the required subset -- so the companion table documents every combination
# actually in the data, including ones intentionally not built into a figure).
# --------------------------------------------------------------------------------------

def build_support_summary(df: pd.DataFrame, biomes_present: list[str]) -> pd.DataFrame:
    rows = []
    for biome in biomes_present:
        for hemi in ["NH", "SH"]:
            sub_bh = df[(df["IGBP"] == biome) & (df["HEMISPHERE"] == hemi)]
            n_towers_bh = sub_bh["SITE_ID"].nunique()
            for prod in PRODUCTS + ["ALL"]:
                pool = sub_bh if prod == "ALL" else sub_bh[sub_bh["SIF_PRODUCT"] == prod]
                for stress_label, sid_col in STRESSORS.items():
                    st = cell_stats(pool, sid_col)
                    rows.append({
                        "IGBP": biome, "IGBP_label": IGBP_LABELS.get(biome, biome),
                        "HEMISPHERE": hemi, "product": prod, "stressor": stress_label,
                        "N_sitemonths": st["N"], "N_towers": st["N_towers"],
                        "reliable": st["reliable"], "median_SID": st["median"],
                    })
    out = pd.DataFrame(rows)
    return out


def decide_figures(support: pd.DataFrame) -> dict:
    """Return {(biome, hemi): {"build": bool, "reason": str, "rel_stressors": [...]}}
    Decision uses the pooled-ALL-product row per stressor (same rule the original
    pipeline used for qualifying_geo): a combo is built if IGBP is one of the
    biomes the task asks us to attempt, N_towers_combo-level (any stressor) is
    reliable, i.e. reliable==True for >=1 stressor at product=='ALL'."""
    decisions = {}
    pooled = support[support["product"] == "ALL"]
    combos = pooled[["IGBP", "HEMISPHERE"]].drop_duplicates().itertuples(index=False)
    for biome, hemi in combos:
        sub = pooled[(pooled["IGBP"] == biome) & (pooled["HEMISPHERE"] == hemi)]
        rel_stressors = sub[sub["reliable"]]["stressor"].tolist()
        n_towers = sub["N_towers"].max()
        n_sm = sub["N_sitemonths"].max()
        in_required_list = biome in REQUIRED_BIOMES
        if not in_required_list:
            reason = (f"{biome} is not in the required biome list (grassland/savanna/woody "
                      f"savanna/cropland/forest); not attempted for this task.")
            decisions[(biome, hemi)] = dict(build=False, reason=reason, rel_stressors=[])
            continue
        if n_towers == 0:
            reason = f"zero {hemi} towers for {biome} in the master table -- no data to plot."
            decisions[(biome, hemi)] = dict(build=False, reason=reason, rel_stressors=[])
        elif not rel_stressors:
            reason = (f"below reliability threshold for every stressor: max N_towers={n_towers}, "
                      f"max N_sitemonths={n_sm} (need N>=12 site-months AND N>=3 towers); "
                      f"omitted rather than shown as if reliable.")
            decisions[(biome, hemi)] = dict(build=False, reason=reason, rel_stressors=[])
        else:
            marginal = " (marginal: exactly at the N_towers=3 minimum)" if n_towers == 3 else ""
            reason = (f"reliable for {', '.join(rel_stressors)} stress (N_towers={n_towers}, "
                      f"N_sitemonths={n_sm}){marginal}; figure built.")
            decisions[(biome, hemi)] = dict(build=True, reason=reason, rel_stressors=rel_stressors)
    return decisions


# --------------------------------------------------------------------------------------
# Step 2: seasonal figure per biome x hemisphere (reuses the exact style/logic of
# build_seasonal_biome_figures() in 05_biome_hemisphere_seasonal.py)
# --------------------------------------------------------------------------------------

def build_biome_hemisphere_figure(df: pd.DataFrame, biome: str, hemi: str, rel_stressors: list[str],
                                    pooling_note: str = ""):
    label_hemi = "Northern Hemisphere" if hemi == "NH" else "Southern Hemisphere"
    print(f"Building Figure: seasonal SID climatology, {biome} x {hemi} ...")
    sub = df[(df["IGBP"] == biome) & (df["HEMISPHERE"] == hemi)]
    n_towers = sub["SITE_ID"].nunique()
    summer_months = [6, 7, 8] if hemi == "NH" else [12, 1, 2]
    season_word = "Jun-Aug" if hemi == "NH" else "Dec-Feb"

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for ax, stressor in zip(axes, STRESS_ORDER):
        sid_col = STRESSORS[stressor]
        for m in summer_months:
            ax.axvspan(m - 0.5, m + 0.5, color="#FDECEC", zorder=0)
        msum = monthly_stats(sub, sid_col)
        is_reliable_stressor = stressor in rel_stressors
        line_color = STRESSOR_COLORS[stressor]
        ax.plot(msum["MONTH"], msum["median"], color=line_color, linewidth=1.8,
                alpha=0.9 if is_reliable_stressor else 0.4, zorder=3)
        rel_pts = msum[msum["reliable"]]
        unrel_pts = msum[~msum["reliable"]]
        ax.scatter(rel_pts["MONTH"], rel_pts["median"], color=line_color, s=34, zorder=4)
        ax.scatter(unrel_pts["MONTH"], unrel_pts["median"], facecolor="none", edgecolor=line_color,
                   s=34, linewidth=1.0, zorder=4)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(MONTH_NAMES, fontsize=8)
        ax.set_title(f"{stressor} stress" + ("" if is_reliable_stressor else "  [below threshold]"),
                     fontsize=10.8, fontweight="bold", loc="left",
                     color="black" if is_reliable_stressor else "#888888")
        ax.set_ylabel(f"Median SID_{stressor}", fontsize=8.5)
        ax.grid(axis="y", color="#EEEEEE", linewidth=0.6, zorder=0)
        ax.set_ylim(bottom=0)

    handles = [Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none",
                      markeredgecolor="#555555", markersize=6,
                      label=f"Unreliable month (N<{REL_MIN_N} or <{REL_MIN_TOWERS} towers, hollow marker)"),
               mpatches.Patch(facecolor="#FDECEC", label=f"Approx. local summer ({season_word})")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.06), frameon=False)

    marginal_note = ""
    if n_towers == REL_MIN_TOWERS:
        marginal_note = (f" Note: {n_towers} towers is exactly at the reliability minimum "
                          f"(N_towers>=3) -- treat this panel as marginal, not strongly supported.")

    fig.suptitle(
        wrap_text(
            f"Calendar-month climatology of Stress-Induced Decoupling -- {biome} "
            f"({IGBP_LABELS.get(biome, biome)}), {label_hemi} ({n_towers} towers, pooled across 4 SIF products)\n"
            f"Shaded band = approximate local summer ({season_word}). Descriptive cross-site pattern, not a "
            f"causal claim.{marginal_note}{(' ' + pooling_note) if pooling_note else ''}",
            width=145),
        fontsize=12, fontweight="bold", y=1.10 if (marginal_note or pooling_note) else 1.06,
    )
    fig.text(0.5, -0.145 if (marginal_note or pooling_note) else -0.12,
              wrap_text(
                  "Reliability rule: N>=12 site-months AND N>=3 independent towers, applied per calendar month. "
                  "Hollow markers/faded lines fall below that threshold. Source: 03_sid_monthly/monthly_SID_master.csv "
                  "(read-only; SID values not recomputed here).",
                  width=170),
              ha="center", fontsize=6.8, color="#555555")
    fig.subplots_adjust(wspace=0.28, top=0.78, bottom=0.24)
    save_figure(fig, f"SID_Figure_Seasonal_{biome}_{hemi}", OUT_DIR)


# --------------------------------------------------------------------------------------
# Step 3: bonus combined overview grid (small multiples), so the ~10 qualifying
# combinations are also viewable as a single readable figure, not only as many
# separate per-combo files.
# --------------------------------------------------------------------------------------

def build_overview_grid(df: pd.DataFrame, qualifying: list[tuple]):
    print("Building bonus combined overview grid ...")
    n_rows = len(qualifying)
    fig, axes = plt.subplots(n_rows, 3, figsize=(14, 2.05 * n_rows + 1.0), squeeze=False)

    for row_idx, (biome, hemi, rel_stressors, n_towers) in enumerate(qualifying):
        sub = df[(df["IGBP"] == biome) & (df["HEMISPHERE"] == hemi)]
        summer_months = [6, 7, 8] if hemi == "NH" else [12, 1, 2]
        for col_idx, stressor in enumerate(STRESS_ORDER):
            ax = axes[row_idx][col_idx]
            sid_col = STRESSORS[stressor]
            for m in summer_months:
                ax.axvspan(m - 0.5, m + 0.5, color="#FDECEC", zorder=0)
            msum = monthly_stats(sub, sid_col)
            is_rel = stressor in rel_stressors
            line_color = STRESSOR_COLORS[stressor]
            ax.plot(msum["MONTH"], msum["median"], color=line_color, linewidth=1.3,
                    alpha=0.9 if is_rel else 0.4, zorder=3)
            rel_pts = msum[msum["reliable"]]
            unrel_pts = msum[~msum["reliable"]]
            ax.scatter(rel_pts["MONTH"], rel_pts["median"], color=line_color, s=14, zorder=4)
            ax.scatter(unrel_pts["MONTH"], unrel_pts["median"], facecolor="none", edgecolor=line_color,
                       s=14, linewidth=0.8, zorder=4)
            ax.set_xticks(range(1, 13))
            if row_idx == n_rows - 1:
                ax.set_xticklabels(MONTH_NAMES, fontsize=6.3, rotation=45)
            else:
                ax.set_xticklabels([])
            ax.tick_params(axis="y", labelsize=6.3)
            ax.grid(axis="y", color="#EEEEEE", linewidth=0.5, zorder=0)
            ax.set_ylim(bottom=0)
            if col_idx == 0:
                marg = " (marginal N)" if n_towers == REL_MIN_TOWERS else ""
                ax.set_ylabel(f"{biome}-{hemi}\n({n_towers} towers{marg})", fontsize=7.6, fontweight="bold")
            if row_idx == 0:
                ax.set_title(f"{stressor} stress", fontsize=9.5, fontweight="bold")

    handles = [Line2D([0], [0], color=STRESSOR_COLORS[s], linewidth=2, label=f"{s} stress") for s in STRESS_ORDER]
    handles.append(Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none",
                          markeredgecolor="#555555", markersize=6, label="Unreliable month (hollow)"))
    handles.append(mpatches.Patch(facecolor="#FDECEC", label="Approx. local summer (hemisphere-specific)"))
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.012), frameon=False)

    fig.suptitle(
        wrap_text(
            "Combined overview: calendar-month Stress-Induced Decoupling climatology, by biome x hemisphere "
            "(each row = one panel from the corresponding standalone SID_Figure_Seasonal_<Biome>_<Hemisphere> "
            "figure; y-axis = median SID for that stressor, per-row scale, not comparable across rows)",
            width=130),
        fontsize=12.5, fontweight="bold", y=1.012,
    )
    fig.text(0.5, -0.045,
              wrap_text(
                  "Descriptive cross-site pattern, not a causal claim. Reliability rule: N>=12 site-months AND "
                  "N>=3 towers per calendar month. Rows with exactly 3 towers are marginal -- see companion "
                  "SID_biome_hemisphere_support_summary.csv. Source: 03_sid_monthly/monthly_SID_master.csv.",
                  width=175),
              ha="center", fontsize=7.0, color="#555555")
    fig.subplots_adjust(wspace=0.32, hspace=0.35, top=0.955, bottom=0.075)
    save_figure(fig, "SID_Figure_Seasonal_BiomeHemisphere_Overview", OUT_DIR)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    df = load_master()
    print(f"Loaded master: {df.shape}, {df['SITE_ID'].nunique()} towers")

    biomes_present = [b for b in IGBP_ORDER if (df["IGBP"] == b).any()]
    print(f"IGBP codes present in data: {biomes_present}")

    support = build_support_summary(df, biomes_present)
    decisions = decide_figures(support)

    # attach figure_built / reason back onto the support table (repeated across
    # product/stressor rows for that biome x hemisphere combo, for easy filtering)
    support["figure_built"] = support.apply(lambda r: decisions[(r["IGBP"], r["HEMISPHERE"])]["build"], axis=1)
    support["decision_reason"] = support.apply(lambda r: decisions[(r["IGBP"], r["HEMISPHERE"])]["reason"], axis=1)
    support_path = OUT_DIR / "SID_biome_hemisphere_support_summary.csv"
    support.to_csv(support_path, index=False)
    print(f"Saved: {support_path}  shape={support.shape}")

    built_combos = [(b, h) for (b, h), d in decisions.items() if d["build"]]
    omitted_combos = [(b, h) for (b, h), d in decisions.items() if (not d["build"]) and b in REQUIRED_BIOMES]
    print(f"\nBuilding {len(built_combos)} biome x hemisphere figures: {built_combos}")
    print(f"Omitted (required-list) combos: {omitted_combos}")
    for b, h in omitted_combos:
        print(f"  omitted {b}-{h}: {decisions[(b, h)]['reason']}")

    pooling_notes = {
        ("WSA", "SH"): "",  # omitted, not built -- handled via decisions/reason only
    }

    overview_rows = []
    for biome, hemi in sorted(built_combos):
        d = decisions[(biome, hemi)]
        note = ""
        if biome in ("ENF", "DBF", "MF") and hemi == "SH":
            note = ""  # unreachable: these have zero SH towers, never in built_combos
        if biome == "EBF" and hemi == "SH":
            note = ("Forest-group pooling was considered but not applied: ENF/DBF/MF have zero SH "
                     "towers, so this EBF-only panel already represents the SH forest group in full.")
        build_biome_hemisphere_figure(df, biome, hemi, d["rel_stressors"], pooling_note=note)
        sub = df[(df["IGBP"] == biome) & (df["HEMISPHERE"] == hemi)]
        overview_rows.append((biome, hemi, d["rel_stressors"], sub["SITE_ID"].nunique()))

    build_overview_grid(df, overview_rows)

    print("\nAll done.")
