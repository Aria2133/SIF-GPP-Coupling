"""
STEP 4 -- Biome-level summary of DeltaPOS: typical offset, interannual variability,
cross-product agreement, and support. Feeds 07_biome_analysis and the biome-level
forest-plot figure (Figure3).
"""
# Original script filename (sanitized): step4_biome_summary.py; public name: step4_biome_summary.py
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

ROOT = (Path(PROJECT_ROOT) / 'SIF_GPP_PEAK_TIMING_MECHANISM_FULL_ANALYSIS/14_scripts/step4_biome_summary.py').resolve().parents[1]
master = pd.read_csv(ROOT / "02_peak_timing" / "SiteYear_Product_PeakTiming_Master.csv")
consensus = pd.read_csv(ROOT / "03_cross_product_consistency" / "cross_product_consensus_by_siteyear.csv")
OUT_DIR = ROOT / "07_biome_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IGBP_LABELS = {
    "ENF": "Evergreen Needleleaf Forest", "EBF": "Evergreen Broadleaf Forest",
    "DBF": "Deciduous Broadleaf Forest", "MF": "Mixed Forest", "CSH": "Closed Shrubland",
    "OSH": "Open Shrubland", "WSA": "Woody Savanna", "SAV": "Savanna",
    "GRA": "Grassland", "CRO": "Cropland",
}
RNG = np.random.default_rng(20260913)
MIN_TOWERS = 3
MIN_SITEYEARS = 12


def bootstrap_median(vals: np.ndarray, tower_ids: np.ndarray, n_boot: int = 1999):
    towers = np.unique(tower_ids)
    if len(towers) < MIN_TOWERS:
        return np.nan, np.nan
    boots = []
    for _ in range(n_boot):
        pick = set(RNG.choice(towers, size=len(towers), replace=True))
        mask = np.isin(tower_ids, list(pick))
        if mask.sum() < 3:
            continue
        boots.append(np.median(vals[mask]))
    if len(boots) < 100:
        return np.nan, np.nan
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    valid = master.dropna(subset=["DELTA_POS"]).copy()
    rows = []
    for biome in IGBP_LABELS:
        g = valid[valid["IGBP"] == biome]
        n_towers = g["SITE_ID"].nunique()
        n_siteyears = len(g)
        if n_towers < MIN_TOWERS or n_siteyears < MIN_SITEYEARS:
            rows.append({"IGBP": biome, "BIOME": IGBP_LABELS[biome], "N_TOWERS": n_towers,
                        "N_SITEYEARS": n_siteyears, "RELIABLE": False})
            continue
        med = float(g["DELTA_POS"].median())
        ci_lo, ci_hi = bootstrap_median(g["DELTA_POS"].to_numpy(), g["SITE_ID"].to_numpy())
        # interannual variability: mean within-tower SD (only towers with >=3 years) + overall SD
        within_sd = g.groupby("SITE_ID")["DELTA_POS"].apply(lambda s: s.std(ddof=1) if len(s) >= 3 else np.nan)
        pct_sif = 100.0 * (g["DELTA_CLASS"] == "EARLIER_SIF").mean()
        pct_sync = 100.0 * (g["DELTA_CLASS"] == "SYNCHRONOUS_OR_UNCERTAIN").mean()
        pct_gpp = 100.0 * (g["DELTA_CLASS"] == "EARLIER_GPP").mean()

        cons_biome = consensus[(consensus["IGBP"] == biome) & (consensus["N_PRODUCTS"] >= 2)]
        pct_high_mod = 100.0 * cons_biome["CONSENSUS"].isin(["HIGH", "MODERATE"]).mean() if len(cons_biome) else np.nan

        rows.append({
            "IGBP": biome, "BIOME": IGBP_LABELS[biome], "N_TOWERS": n_towers, "N_SITEYEARS": n_siteyears,
            "RELIABLE": True, "MEDIAN_DELTA_POS": round(med, 2), "CI_LOW": round(ci_lo, 2) if not np.isnan(ci_lo) else np.nan,
            "CI_HIGH": round(ci_hi, 2) if not np.isnan(ci_hi) else np.nan,
            "PCT_SIF_EARLIER": round(pct_sif, 1), "PCT_SYNCHRONOUS": round(pct_sync, 1), "PCT_GPP_EARLIER": round(pct_gpp, 1),
            "WITHIN_TOWER_SD_MEAN": round(float(within_sd.mean()), 2) if within_sd.notna().any() else np.nan,
            "N_MULTIPRODUCT_TOWERYEARS": len(cons_biome), "PCT_HIGH_OR_MODERATE_CONSENSUS": round(pct_high_mod, 1) if not np.isnan(pct_high_mod) else np.nan,
            "N_PRODUCTS_PRESENT": g["SIF_PRODUCT"].nunique(),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "Biome_Summary_Table.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
