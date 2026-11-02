"""STEP 5 -- Figure3: biome forest plot of median DeltaPOS with bootstrap 95% CI."""
# Original script filename (sanitized): step5_figure_biome_forest.py; public name: step5_figure_biome_forest.py
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

ROOT = (Path(PROJECT_ROOT) / 'SIF_GPP_PEAK_TIMING_MECHANISM_FULL_ANALYSIS/14_scripts/step5_figure_biome_forest.py').resolve().parents[1]
summary = pd.read_csv(ROOT / "07_biome_analysis" / "Biome_Summary_Table.csv")
FIG_DIR = ROOT / "11_figures"

rel = summary[summary["RELIABLE"]].dropna(subset=["MEDIAN_DELTA_POS"]).sort_values("MEDIAN_DELTA_POS")

fig, ax = plt.subplots(figsize=(8.5, 5.5), facecolor="white")
y = np.arange(len(rel))
colors = ["#2166ac" if v < 0 else "#b2182b" for v in rel["MEDIAN_DELTA_POS"]]
ax.errorbar(rel["MEDIAN_DELTA_POS"], y, xerr=[rel["MEDIAN_DELTA_POS"] - rel["CI_LOW"], rel["CI_HIGH"] - rel["MEDIAN_DELTA_POS"]],
            fmt="none", ecolor="#555555", elinewidth=1.5, capsize=3, zorder=2)
ax.scatter(rel["MEDIAN_DELTA_POS"], y, s=90, color=colors, zorder=3, edgecolor="white", linewidth=0.8)
ax.axvline(0, color="#333333", lw=1.0, ls="--", zorder=1)
ax.set_yticks(y)
ax.set_yticklabels([f"{b}  (n={int(nt)} towers)" for b, nt in zip(rel["BIOME"], rel["N_TOWERS"])], fontsize=10)
ax.set_xlabel("Median ΔPOS = POS$_{SIF}$ − POS$_{GPP}$ (months)\n← SIF earlier            GPP earlier →", fontsize=10)
ax.set_title("Which biomes show SIF peaking earlier than GPP?", fontsize=13.5, fontweight="bold", loc="left")
ax.spines[["top", "right"]].set_visible(False)
fig.text(0.5, -0.03, "Bootstrap 95% CI (tower-block). Forests (bottom rows) tilt toward SIF-earlier; grasslands/savannas sit near zero.",
          ha="center", fontsize=8.5, color="#555555")
fig.tight_layout()

for ext in ("png", "svg", "pdf"):
    dpi = 600 if ext == "png" else 300
    fig.savefig(FIG_DIR / f"Figure3_Biome_Forest_Plot.{ext}", dpi=dpi, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("Saved Figure3_Biome_Forest_Plot")
