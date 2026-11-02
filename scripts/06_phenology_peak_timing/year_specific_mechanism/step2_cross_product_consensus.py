"""
STEP 2 -- Cross-product consensus.

For every SITE_ID x YEAR where >=2 SIF products have a valid (non-INSUFFICIENT_DATA)
DELTA_CLASS, compare their peak-timing verdicts. A biological mechanism is far more
convincing if independent SIF products agree on the same tower-year than if only one
product shows it.

CROSS_PRODUCT_CONSENSUS levels (based on how many of the available products agree on
the SAME DELTA_CLASS label for that tower-year):
  HIGH     : >=3 products available AND all agree
  MODERATE : 2 products available and agree, OR >=3 available with a clear majority
             (more than half agree on one label)
  LOW      : products disagree (no label reaches a majority)
  SINGLE_PRODUCT : only one product has valid data for that tower-year (no consensus
             possible; reported separately, not scored as agreement or disagreement)

Also reports, for the subset with >=2 products: sign agreement rate, magnitude
(Spearman rank) agreement of DELTA_POS across products, and pairwise product
consistency.
"""
# Original script filename (sanitized): step2_cross_product_consensus.py; public name: step2_cross_product_consensus.py
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
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = (Path(PROJECT_ROOT) / 'SIF_GPP_PEAK_TIMING_MECHANISM_FULL_ANALYSIS/14_scripts/step2_cross_product_consensus.py').resolve().parents[1]
master = pd.read_csv(ROOT / "02_peak_timing" / "SiteYear_Product_PeakTiming_Master.csv")
OUT_DIR = ROOT / "03_cross_product_consistency"
OUT_DIR.mkdir(parents=True, exist_ok=True)

valid = master[master["DELTA_CLASS"] != "INSUFFICIENT_DATA"].copy()


def consensus_row(g: pd.DataFrame) -> dict:
    n_products = g["SIF_PRODUCT"].nunique()
    labels = g["DELTA_CLASS"].tolist()
    if n_products < 2:
        return {"N_PRODUCTS": n_products, "CONSENSUS": "SINGLE_PRODUCT", "MAJORITY_LABEL": labels[0] if labels else None,
                "N_AGREEING": 1 if labels else 0}
    vc = pd.Series(labels).value_counts()
    top_label, top_n = vc.index[0], vc.iloc[0]
    if top_n == n_products:
        level = "HIGH" if n_products >= 3 else "MODERATE"
    elif top_n > n_products / 2:
        level = "MODERATE"
    else:
        level = "LOW"
    return {"N_PRODUCTS": n_products, "CONSENSUS": level, "MAJORITY_LABEL": top_label, "N_AGREEING": int(top_n)}


rows = []
for (site, year), g in valid.groupby(["SITE_ID", "YEAR"]):
    rec = {"SITE_ID": site, "YEAR": year, "IGBP": g["IGBP"].iloc[0], "HEMI": g["HEMI"].iloc[0]}
    rec.update(consensus_row(g))
    rec["PRODUCTS"] = ",".join(sorted(g["SIF_PRODUCT"].unique()))
    rows.append(rec)
consensus = pd.DataFrame(rows)
consensus.to_csv(OUT_DIR / "cross_product_consensus_by_siteyear.csv", index=False)

# Pairwise product agreement (sign of DELTA_POS, and rank correlation of DELTA_POS magnitude)
products = sorted(valid["SIF_PRODUCT"].unique())
pair_rows = []
for p1, p2 in combinations(products, 2):
    m1 = valid[valid["SIF_PRODUCT"] == p1][["SITE_ID", "YEAR", "DELTA_POS", "DELTA_CLASS"]].rename(
        columns={"DELTA_POS": "D1", "DELTA_CLASS": "C1"})
    m2 = valid[valid["SIF_PRODUCT"] == p2][["SITE_ID", "YEAR", "DELTA_POS", "DELTA_CLASS"]].rename(
        columns={"DELTA_POS": "D2", "DELTA_CLASS": "C2"})
    merged = m1.merge(m2, on=["SITE_ID", "YEAR"])
    if len(merged) < 5:
        pair_rows.append({"PRODUCT_1": p1, "PRODUCT_2": p2, "N_OVERLAP": len(merged),
                          "SIGN_AGREEMENT_PCT": np.nan, "RANK_CORR": np.nan})
        continue
    sign_agree = np.mean(np.sign(merged["D1"]) == np.sign(merged["D2"])) * 100
    rho, _ = spearmanr(merged["D1"], merged["D2"])
    pair_rows.append({"PRODUCT_1": p1, "PRODUCT_2": p2, "N_OVERLAP": len(merged),
                      "SIGN_AGREEMENT_PCT": round(sign_agree, 1), "RANK_CORR": round(float(rho), 3)})
pairwise = pd.DataFrame(pair_rows)
pairwise.to_csv(OUT_DIR / "pairwise_product_agreement.csv", index=False)

print("Consensus level counts (multi-product tower-years only):")
print(consensus[consensus["N_PRODUCTS"] >= 2]["CONSENSUS"].value_counts())
print(f"\nSingle-product tower-years: {(consensus['N_PRODUCTS'] == 1).sum()} of {len(consensus)}")
print("\nPairwise agreement:")
print(pairwise.to_string(index=False))
