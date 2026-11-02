"""
GOME-2 Quality_Flag QA sensitivity check.

Official documentation (ORNL DAAC user guide for the MetOp-A GOME-2 SIF product,
Joiner et al.) defines Quality_Flag as:
  0 = bad (fails QC)
  1 = good, but cloud fraction > 30%
  2 = good AND cloud fraction < 30%   <-- "For gridding data, use pixels with a
                                          quality flag value of 2" (recommended)

The production GOME-2 finalization script (Codes/pythonProject/
Finalize_GOME2_EC_Footprint_Analytical_Table.py) reads Quality_Flag per native
observation and stores it in GOME2_EC_Observation_Overlap_Provenance.csv, but
does NOT apply a hard filter on it when computing the EC-probability-weighted
monthly SIF value used in the production master.

This script recomputes the EC-probability-weighted monthly SIF using ONLY
quality_flag_raw == 2 (the officially recommended subset), from the *already
computed* per-observation provenance table (no raw netCDF re-read needed —
avoids re-touching 3,865 x ~14MB native GOME-2 files), and compares raw/
harmonic/current-anomaly correlations against the production (unfiltered)
numbers. This is a side computation only; production data is not modified.
"""
from __future__ import annotations
# Original script filename (sanitized): gome2_quality_flag_sensitivity.py; public name: gome2_quality_flag_sensitivity.py
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
from scipy import stats

PROJECT_ROOT = Path(
    PROJECT_ROOT
)
PROV_PATH = PROJECT_ROOT / "SIF_EC_GOME2_Finalization" / "GOME2_EC_Observation_Overlap_Provenance.csv"
MASTER_PATH = (
    PROJECT_ROOT
    / "SIF_EC_SIF_GPP_Relationship_Analysis"
    / "01_analysis_ready"
    / "SIF_GPP_Analysis_Ready_Master.csv"
)
OUT_DIR = PROJECT_ROOT / "SIF_EC_SIF_GPP_Final_Verification" / "04_sampling_and_QA"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def pearson_spearman_r2(x: np.ndarray, y: np.ndarray) -> dict:
    n = len(x)
    if n < 3 or np.std(x) == 0 or np.std(y) == 0:
        return dict(N=n, pearson_r=np.nan, spearman_rho=np.nan, r2=np.nan)
    r, _ = stats.pearsonr(x, y)
    rho, _ = stats.spearmanr(x, y)
    slope, intercept, rval, _, _ = stats.linregress(x, y)
    return dict(N=n, pearson_r=r, spearman_rho=rho, r2=rval**2)


def harmonic_r(g: pd.DataFrame, sif_col: str, min_n: int = 24, min_unique_months: int = 8) -> dict:
    recs = []
    for site, gs in g.groupby("SITE_ID"):
        if len(gs) < min_n or gs["MONTH"].nunique() < min_unique_months:
            continue
        t = 2 * np.pi * gs["MONTH"].to_numpy() / 12.0
        X = np.column_stack([np.ones(len(gs)), np.sin(t), np.cos(t), np.sin(2 * t), np.cos(2 * t)])
        gs = gs.copy()
        for col, out_col in ((sif_col, "SIF_ANOM"), ("GPP_VALUE", "GPP_ANOM")):
            y = gs[col].to_numpy(dtype=float)
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            gs[out_col] = y - X @ beta
        recs.append(gs)
    if not recs:
        return dict(N=0, N_towers=0, pearson_r=np.nan)
    gr = pd.concat(recs, ignore_index=True)
    s = pearson_spearman_r2(gr["SIF_ANOM"].to_numpy(), gr["GPP_ANOM"].to_numpy())
    return {"N_towers": gr["SITE_ID"].nunique(), **s}


def current_anomaly_r(g: pd.DataFrame, sif_col: str) -> dict:
    clim = g.groupby(["SITE_ID", "MONTH"]).agg(sif_clim=(sif_col, "mean"), gpp_clim=("GPP_VALUE", "mean"))
    gg = g.join(clim, on=["SITE_ID", "MONTH"])
    sif_anom = gg[sif_col] - gg["sif_clim"]
    gpp_anom = gg["GPP_VALUE"] - gg["gpp_clim"]
    return pearson_spearman_r2(sif_anom.to_numpy(), gpp_anom.to_numpy())


def main() -> None:
    prov = pd.read_csv(PROV_PATH, usecols=["SITE_ID", "YYYYMM", "GOME2_SIF", "EC_probability_mass_intersected", "quality_flag_raw"])
    print("Provenance rows:", len(prov), "| flag counts:", prov["quality_flag_raw"].value_counts(dropna=False).to_dict())

    def weighted_monthly(df: pd.DataFrame) -> pd.DataFrame:
        def wavg(g: pd.DataFrame) -> float:
            w = g["EC_probability_mass_intersected"].to_numpy(dtype=float)
            v = g["GOME2_SIF"].to_numpy(dtype=float)
            if w.sum() <= 0:
                return np.nan
            return float(np.sum(v * w) / np.sum(w))

        out = df.groupby(["SITE_ID", "YYYYMM"]).apply(wavg).rename("SIF_RECOMPUTED").reset_index()
        return out

    unfiltered = weighted_monthly(prov)
    qa2 = weighted_monthly(prov[prov["quality_flag_raw"] == 2])

    master = pd.read_csv(MASTER_PATH, low_memory=False)
    master = master[master["SIF_PRODUCT"] == "GOME-2"].copy()
    master["YYYYMM"] = pd.to_numeric(master["YYYYMM"], errors="coerce").astype("Int64")
    master["MONTH"] = pd.to_numeric(master["MONTH"], errors="coerce")
    master["GPP_VALUE"] = pd.to_numeric(master["GPP_VALUE"], errors="coerce")
    master["SIF_VALUE"] = pd.to_numeric(master["SIF_VALUE"], errors="coerce")

    for label, weighted in (("PRODUCTION_UNFILTERED_reproduced_from_provenance", unfiltered), ("QA_FLAG2_ONLY", qa2)):
        weighted["YYYYMM"] = pd.to_numeric(weighted["YYYYMM"], errors="coerce").astype("Int64")
        m = master.merge(weighted, on=["SITE_ID", "YYYYMM"], how="inner").dropna(subset=["SIF_RECOMPUTED", "GPP_VALUE", "MONTH"])
        raw = pearson_spearman_r2(m["SIF_RECOMPUTED"].to_numpy(), m["GPP_VALUE"].to_numpy())
        cur = current_anomaly_r(m, "SIF_RECOMPUTED")
        harm = harmonic_r(m, "SIF_RECOMPUTED")
        print(f"\n=== {label} ===")
        print("  matched N site-months:", len(m), "| N towers:", m["SITE_ID"].nunique())
        print("  RAW           :", raw)
        print("  CURRENT ANOM  :", cur)
        print("  HARMONIC ANOM :", harm)
        pd.DataFrame(
            [
                {"filter": label, "stage": "raw", **raw},
                {"filter": label, "stage": "current_anomaly", **cur},
                {"filter": label, "stage": "harmonic_anomaly", **harm},
            ]
        ).to_csv(OUT_DIR / f"gome2_qa_sensitivity_{'unfiltered' if 'UNFILTERED' in label else 'flag2only'}.csv", index=False)

    # also: sanity check that unfiltered-reconstructed matches the true production SIF_VALUE closely
    check = master.merge(unfiltered, on=["SITE_ID", "YYYYMM"], how="inner").dropna(subset=["SIF_VALUE", "SIF_RECOMPUTED"])
    diff = (check["SIF_VALUE"] - check["SIF_RECOMPUTED"]).abs()
    print("\nSanity check (reproduced-unfiltered vs production SIF_VALUE): N=", len(check),
          "median abs diff=", float(diff.median()), "max abs diff=", float(diff.max()))


if __name__ == "__main__":
    main()
