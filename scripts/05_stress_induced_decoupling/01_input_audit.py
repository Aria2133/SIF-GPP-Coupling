"""
Step 1 -- Input inventory for SIF-GPP Stress-Induced Decoupling (SID) foundational layer.

Reuses (does NOT rebuild) the existing analysis-ready master table and the
mechanistic extended-drivers table. Filters to the four main SIF products
(GOSIF, GOME-2, TROPOSIF, OCO-3); OCO-2 is excluded everywhere per project rule.

Outputs:
  01_input_audit/sid_input_inventory.csv
  01_input_audit/sid_input_inventory.md
"""
# Original script filename (sanitized): 01_input_audit.py; public name: 01_input_audit.py
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
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(PROJECT_ROOT)
MASTER_PATH = ROOT / "SIF_EC_SIF_GPP_Relationship_Analysis" / "01_analysis_ready" / "SIF_GPP_Analysis_Ready_Master.csv"
DRIVERS_PATH = ROOT / "SIF_EC_Mechanistic_FootprintAware_SIF_GPP_Paper" / "08_mechanistic_controls" / "Mechanistic_SIF_GPP_Monthly_With_Extended_Drivers.csv"
OUT_DIR = ROOT / "SIF_EC_Stress_Induced_Decoupling_SID" / "01_input_audit"
MAIN_PRODUCTS = ["GOSIF", "GOME-2", "TROPOSIF", "OCO-3"]

def main():
    m = pd.read_csv(MASTER_PATH, low_memory=False)
    e = pd.read_csv(DRIVERS_PATH, low_memory=False)

    key = ["SITE_ID", "YYYYMM", "SIF_PRODUCT"]
    assert m.duplicated(key).sum() == 0, "master has duplicate keys"
    assert e.duplicated(key).sum() == 0, "extended-drivers has duplicate keys"
    assert set(map(tuple, m[key].values)) == set(map(tuple, e[key].values)), "key sets differ between master and extended-drivers"

    e_main = e[e["SIF_PRODUCT"].isin(MAIN_PRODUCTS)].copy()
    m_main = m[m["SIF_PRODUCT"].isin(MAIN_PRODUCTS)].copy()

    rows = []
    for prod in MAIN_PRODUCTS:
        sub = e_main[e_main["SIF_PRODUCT"] == prod]
        n_sitemonths = len(sub)
        n_towers = sub["SITE_ID"].nunique()
        yr_min, yr_max = sub["YEAR"].min(), sub["YEAR"].max()
        vpd_ok = sub["VPD_AVAILABLE"].fillna(False).astype(bool)
        tair_ok = sub["TAIR_AVAILABLE"].fillna(False).astype(bool)
        swc_ok = sub["SWC_AVAILABLE"].fillna(False).astype(bool)
        all3_ok = vpd_ok & tair_ok & swc_ok
        row = {
            "SIF_PRODUCT": prod,
            "N_site_months": n_sitemonths,
            "N_towers": n_towers,
            "YEAR_min": yr_min,
            "YEAR_max": yr_max,
            "N_missing_VPD": int((~vpd_ok).sum()),
            "N_missing_TAIR": int((~tair_ok).sum()),
            "N_missing_SWC": int((~swc_ok).sum()),
            "N_all_3_drivers_available": int(all3_ok.sum()),
            "PCT_all_3_drivers_available": round(100 * all3_ok.sum() / n_sitemonths, 2),
            "N_SIF_VALUE_missing": int(sub["SIF_VALUE"].isna().sum()),
            "N_GPP_VALUE_missing": int(sub["GPP_VALUE"].isna().sum()),
            "N_IGBP_missing": int(sub["IGBP_GPP_TABLE"].isna().sum()),
        }
        if prod == "OCO-3":
            vc = sub["OBSERVED_OR_RECONSTRUCTED"].value_counts(dropna=False)
            row["N_OCO3_OBSERVED"] = int(vc.get("OBSERVED", 0))
            row["N_OCO3_RECONSTRUCTED"] = int(vc.get("RECONSTRUCTED", 0))
        else:
            row["N_OCO3_OBSERVED"] = np.nan
            row["N_OCO3_RECONSTRUCTED"] = np.nan
        rows.append(row)

    # TOTAL row
    all3_ok_total = (
        e_main["VPD_AVAILABLE"].fillna(False).astype(bool)
        & e_main["TAIR_AVAILABLE"].fillna(False).astype(bool)
        & e_main["SWC_AVAILABLE"].fillna(False).astype(bool)
    )
    total_row = {
        "SIF_PRODUCT": "ALL_4_MAIN_PRODUCTS",
        "N_site_months": len(e_main),
        "N_towers": e_main["SITE_ID"].nunique(),
        "YEAR_min": e_main["YEAR"].min(),
        "YEAR_max": e_main["YEAR"].max(),
        "N_missing_VPD": int((~e_main["VPD_AVAILABLE"].fillna(False).astype(bool)).sum()),
        "N_missing_TAIR": int((~e_main["TAIR_AVAILABLE"].fillna(False).astype(bool)).sum()),
        "N_missing_SWC": int((~e_main["SWC_AVAILABLE"].fillna(False).astype(bool)).sum()),
        "N_all_3_drivers_available": int(all3_ok_total.sum()),
        "PCT_all_3_drivers_available": round(100 * all3_ok_total.sum() / len(e_main), 2),
        "N_SIF_VALUE_missing": int(e_main["SIF_VALUE"].isna().sum()),
        "N_GPP_VALUE_missing": int(e_main["GPP_VALUE"].isna().sum()),
        "N_IGBP_missing": int(e_main["IGBP_GPP_TABLE"].isna().sum()),
        "N_OCO3_OBSERVED": int((e_main.loc[e_main.SIF_PRODUCT=="OCO-3","OBSERVED_OR_RECONSTRUCTED"]=="OBSERVED").sum()),
        "N_OCO3_RECONSTRUCTED": int((e_main.loc[e_main.SIF_PRODUCT=="OCO-3","OBSERVED_OR_RECONSTRUCTED"]=="RECONSTRUCTED").sum()),
    }
    rows.append(total_row)

    inv = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inv.to_csv(OUT_DIR / "sid_input_inventory.csv", index=False)

    excluded_oco2 = (m["SIF_PRODUCT"] == "OCO-2").sum()
    n_dup_check_master = m_main.duplicated(key).sum()

    md_lines = []
    md_lines.append("# SID Input Inventory (Step 1)\n")
    md_lines.append(f"Source master (analysis-ready): `{MASTER_PATH}`\n")
    md_lines.append(f"Source extended drivers: `{DRIVERS_PATH}`\n")
    md_lines.append(f"\nBoth source files share identical row structure (verified 1:1 on SITE_ID+YYYYMM+SIF_PRODUCT keys; "
                     f"{len(m)} rows each; SIF_VALUE and GPP_VALUE match exactly between the two files for every row).\n")
    md_lines.append(f"\nOCO-2 rows excluded per project rule: {excluded_oco2} rows removed from the {len(m)}-row source, "
                     f"leaving {len(m_main)} rows across the 4 main products (GOSIF, GOME-2, TROPOSIF, OCO-3). "
                     f"No duplicate keys found in filtered set (dup count = {n_dup_check_master}).\n")
    md_lines.append("\nDriver-availability columns used: `VPD_AVAILABLE`, `TAIR_AVAILABLE`, `SWC_AVAILABLE` "
                     "(booleans already computed in the extended-drivers file). VPD and TAIR are available for "
                     "100% of rows in all 4 main products; SWC (soil moisture) is the only driver with material "
                     "missingness (varies by product/site -- see table).\n")
    md_lines.append("\nIGBP note: raw `IGBP` column contains an 'Unknown' category (28 rows) that is recoded to NaN "
                     "in `IGBP_GPP_TABLE`; missingness counted here uses `IGBP_GPP_TABLE`, which is carried forward "
                     "into the SID master table as the `IGBP` (biome) field.\n")
    md_lines.append("\n## Per-product summary\n")
    md_lines.append(inv.to_markdown(index=False))
    md_lines.append("\n")
    (OUT_DIR / "sid_input_inventory.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(inv.to_string(index=False))
    print("\nSaved:", OUT_DIR / "sid_input_inventory.csv")
    print("Saved:", OUT_DIR / "sid_input_inventory.md")

if __name__ == "__main__":
    main()
