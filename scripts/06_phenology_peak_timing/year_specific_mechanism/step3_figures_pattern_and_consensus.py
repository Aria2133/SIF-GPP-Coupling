"""
STEP 3 -- Figures answering "is the early-SIF signal real, how big, and do products
agree": Figure_1_Frequency_By_Product.png and Figure_2_CrossProduct_Consensus.png.
"""
# Original script filename (sanitized): step3_figures_pattern_and_consensus.py; public name: step3_figures_pattern_and_consensus.py
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

ROOT = (Path(PROJECT_ROOT) / 'SIF_GPP_PEAK_TIMING_MECHANISM_FULL_ANALYSIS/14_scripts/step3_figures_pattern_and_consensus.py').resolve().parents[1]
master = pd.read_csv(ROOT / "02_peak_timing" / "SiteYear_Product_PeakTiming_Master.csv")
consensus = pd.read_csv(ROOT / "03_cross_product_consistency" / "cross_product_consensus_by_siteyear.csv")
FIG_DIR = ROOT / "11_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PRODUCTS = ["GOSIF", "GOME-2", "TROPOSIF", "OCO-3"]
PRODUCT_COLORS = {"GOSIF": "#0072B2", "GOME-2": "#E69F00", "TROPOSIF": "#009E73", "OCO-3": "#CC79A7"}
CLASS_ORDER = ["EARLIER_SIF", "SYNCHRONOUS_OR_UNCERTAIN", "EARLIER_GPP"]
CLASS_COLORS = {"EARLIER_SIF": "#2166ac", "SYNCHRONOUS_OR_UNCERTAIN": "#bbbbbb", "EARLIER_GPP": "#b2182b"}
CLASS_LABELS = {"EARLIER_SIF": "SIF earlier", "SYNCHRONOUS_OR_UNCERTAIN": "Synchronous / uncertain", "EARLIER_GPP": "GPP earlier"}


def fig1():
    valid = master[master["DELTA_CLASS"] != "INSUFFICIENT_DATA"]
    fig, ax = plt.subplots(figsize=(8.5, 5.3), facecolor="white")
    bottom = np.zeros(len(PRODUCTS))
    for cls in CLASS_ORDER:
        vals = []
        for p in PRODUCTS:
            sub = valid[valid["SIF_PRODUCT"] == p]
            vals.append(100.0 * (sub["DELTA_CLASS"] == cls).mean() if len(sub) else np.nan)
        ax.bar(PRODUCTS, vals, bottom=bottom, color=CLASS_COLORS[cls], label=CLASS_LABELS[cls], edgecolor="white")
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v >= 6:
                ax.text(i, b + v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=9.5,
                        color="white" if cls != "SYNCHRONOUS_OR_UNCERTAIN" else "#333333")
        bottom += np.array(vals)
    for i, p in enumerate(PRODUCTS):
        n = (valid["SIF_PRODUCT"] == p).sum()
        ax.text(i, 103, f"n={n}", ha="center", fontsize=8.5, color="#333333")
    ax.set_ylim(0, 112)
    ax.set_ylabel("% of site-years", fontsize=10.5)
    ax.set_title("Is the early-SIF signal real? Site-year peak-timing outcome by product", fontsize=13, fontweight="bold", loc="left")
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "svg", "pdf"):
        dpi = 600 if ext == "png" else 300
        fig.savefig(FIG_DIR / f"Figure1_Frequency_By_Product.{ext}", dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("Saved Figure1_Frequency_By_Product")


def fig2():
    multi = consensus[consensus["N_PRODUCTS"] >= 2]
    counts = multi["CONSENSUS"].value_counts().reindex(["HIGH", "MODERATE", "LOW"], fill_value=0)
    fig, ax = plt.subplots(figsize=(7.0, 5.0), facecolor="white")
    colors = ["#1a9850", "#fee08b", "#d73027"]
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white")
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 5, f"{v}\n({100*v/counts.sum():.0f}%)", ha="center", fontsize=10)
    ax.set_ylabel("Number of tower-years with ≥2 products", fontsize=10.5)
    ax.set_title("Do multiple SIF products agree on the same tower-year?", fontsize=13, fontweight="bold", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, -0.02, f"({(consensus['N_PRODUCTS']==1).sum()} additional tower-years have only 1 product available, not shown)",
              ha="center", fontsize=8.5, color="#666666")
    fig.tight_layout()
    for ext in ("png", "svg", "pdf"):
        dpi = 600 if ext == "png" else 300
        fig.savefig(FIG_DIR / f"Figure2_CrossProduct_Consensus.{ext}", dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("Saved Figure2_CrossProduct_Consensus")


if __name__ == "__main__":
    fig1()
    fig2()
