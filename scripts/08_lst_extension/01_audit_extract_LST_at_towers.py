"""
PART 1 AUDIT (empirical step): extract MYD21C1 LST at all 73 study-tower
coordinates, from both LST Grasslands/ and LST Savannas/, to determine
empirically (not by assumption) which towers have ANY valid LST and how many
valid site-months each has. This is the only LST data found anywhere in the
project (see LST_DATA_AUDIT.md for the full file-search audit).

Does NOT modify any original file -- reads only.
"""
# Original script filename (sanitized): 01_audit_extract_LST_at_towers.py; public name: 01_audit_extract_LST_at_towers.py
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
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

ROOT = Path(PROJECT_ROOT)
LST_DIRS = {"Grasslands": ROOT / "LST Grasslands", "Savannas": ROOT / "LST Savannas"}
TOWER_CSV = ROOT / "All_the_Visualizations_REVISED_4PRODUCT_FINAL_ARIDITY_JUNE_DIAGNOSTICS" / "01_inputs_inventory" / "tower_aridity_annual_and_monthly.csv"
OUT_DIR = ROOT / "LST_SIF_GPP_EXTENSION_FINAL" / "01_data_audit"

FNAME_RE = re.compile(r"MYD21C1_Global_(Grasslands|Savannas)_LST_(\d{4})_(\d{2})\.tif$")


def main():
    towers = pd.read_csv(TOWER_CSV, usecols=["SITE_ID", "LATITUDE", "LONGITUDE"])
    coords = list(zip(towers["LONGITUDE"], towers["LATITUDE"]))
    n_towers = len(towers)
    print(f"Loaded {n_towers} study towers.")

    records = []
    file_list = []
    for land_type, d in LST_DIRS.items():
        files = sorted(d.glob("*.tif"))
        print(f"{land_type}: {len(files)} files")
        for f in files:
            m = FNAME_RE.match(f.name)
            if not m:
                print(f"  WARNING: filename did not match expected pattern: {f.name}")
                continue
            _, year, month = m.groups()
            file_list.append({"land_type": land_type, "file": f.name, "year": int(year), "month": int(month)})
            with rasterio.open(f) as src:
                vals = list(src.sample(coords, indexes=1))
            for (site_id,), v in zip(towers[["SITE_ID"]].values, vals):
                val = float(v[0])
                if np.isnan(val):
                    continue
                records.append({"SITE_ID": site_id, "YEAR": int(year), "MONTH": int(month),
                                 "LAND_TYPE_RASTER": land_type, "LST_C": val})

    file_df = pd.DataFrame(file_list)
    file_df.to_csv(OUT_DIR / "LST_raster_file_list.csv", index=False)
    print(f"\nTotal raster files indexed: {len(file_df)}")
    print(f"Year range: {file_df['year'].min()}-{file_df['year'].max()}")

    hits = pd.DataFrame(records)
    hits.to_csv(OUT_DIR / "LST_tower_extraction_raw_hits.csv", index=False)
    print(f"\nTotal non-NaN (SITE_ID x YEAR x MONTH x raster) extractions: {len(hits)}")

    if len(hits) == 0:
        print("NO TOWERS HAVE ANY VALID LST VALUE IN EITHER RASTER SET.")
        return

    # if a tower happens to get a valid pixel from BOTH rasters for the same month
    # (shouldn't happen if masks are disjoint, but check empirically, don't assume)
    dup = hits.duplicated(["SITE_ID", "YEAR", "MONTH"], keep=False)
    n_dup = int(dup.sum())
    print(f"SITE_ID x YEAR x MONTH rows with a valid value from BOTH Grassland and Savanna rasters: {n_dup}")

    per_tower = hits.groupby("SITE_ID").agg(
        N_VALID_SITEMONTHS=("LST_C", "size"),
        MIN_LST_C=("LST_C", "min"), MAX_LST_C=("LST_C", "max"), MEAN_LST_C=("LST_C", "mean"),
        MIN_YEAR=("YEAR", "min"), MAX_YEAR=("YEAR", "max"),
        RASTER_SOURCE=("LAND_TYPE_RASTER", lambda s: "/".join(sorted(set(s)))),
    ).reset_index()
    per_tower = towers.merge(per_tower, on="SITE_ID", how="left")
    per_tower["HAS_VALID_LST"] = per_tower["N_VALID_SITEMONTHS"].notna()
    per_tower.to_csv(OUT_DIR / "LST_coverage_by_tower.csv", index=False)

    n_with = int(per_tower["HAS_VALID_LST"].sum())
    print(f"\nTowers with >=1 valid LST site-month: {n_with} of {n_towers}")
    print(per_tower[per_tower["HAS_VALID_LST"]][["SITE_ID", "N_VALID_SITEMONTHS", "MIN_YEAR", "MAX_YEAR",
                                                   "MIN_LST_C", "MAX_LST_C"]].to_string(index=False))
    print("\nTowers WITHOUT any valid LST:")
    print(per_tower[~per_tower["HAS_VALID_LST"]]["SITE_ID"].tolist())

    # monthly missingness per covered tower
    monthly_missing_rows = []
    covered_towers = per_tower[per_tower["HAS_VALID_LST"]]["SITE_ID"].tolist()
    for site in covered_towers:
        sub = hits[hits["SITE_ID"] == site]
        years_present = sorted(sub["YEAR"].unique())
        full_months_possible = (max(years_present) - min(years_present) + 1) * 12
        monthly_missing_rows.append({
            "SITE_ID": site, "N_VALID": len(sub), "N_YEARS_SPAN": max(years_present) - min(years_present) + 1,
            "N_MONTHS_POSSIBLE_IN_SPAN": full_months_possible,
            "PCT_COMPLETE_WITHIN_SPAN": round(100 * len(sub) / full_months_possible, 1),
        })
    pd.DataFrame(monthly_missing_rows).to_csv(OUT_DIR / "LST_monthly_completeness_by_tower.csv", index=False)

    print("\nDONE.")


if __name__ == "__main__":
    main()
