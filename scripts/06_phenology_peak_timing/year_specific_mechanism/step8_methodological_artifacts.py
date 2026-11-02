"""
STEP 8 -- Methodological artifacts A-C: satellite temporal sampling, reconstruction
(OCO-3), and footprint/overlap mismatch (GOME-2, OCO-3 only -- these are the only
products with real, non-placeholder overlap metrics; see 01_data_inventory).

Tests whether |DELTA_POS| is larger under poor sampling / reconstruction / low
footprint-overlap conditions -- if so, part of the apparent SIF-earlier pattern could
be methodological rather than biological.
"""
# Original script filename (sanitized): step8_methodological_artifacts.py; public name: step8_methodological_artifacts.py
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
from scipy.stats import spearmanr

ROOT = Path(PROJECT_ROOT)
ANALYSIS_ROOT = (Path(PROJECT_ROOT) / 'SIF_GPP_PEAK_TIMING_MECHANISM_FULL_ANALYSIS/14_scripts/step8_methodological_artifacts.py').resolve().parents[1]
HARMONIC_CSV = ROOT / "SIF_EC_SIF_GPP_Final_Verification" / "02_independent_reverification" / "harmonic_residuals_full.csv"
MASTER_CSV = ANALYSIS_ROOT / "02_peak_timing" / "SiteYear_Product_PeakTiming_Master.csv"
OUT_DIR = ANALYSIS_ROOT / "09_methodological_artifacts"

PRODUCTS = ["GOSIF", "GOME-2", "TROPOSIF", "OCO-3"]


def main():
    master = pd.read_csv(MASTER_CSV)
    valid = master.dropna(subset=["DELTA_POS"]).copy()
    valid["ABS_DELTA_POS"] = valid["DELTA_POS"].abs()

    harm = pd.read_csv(HARMONIC_CSV, low_memory=False)
    harm = harm[harm["SIF_PRODUCT"].isin(PRODUCTS)]

    # --- A: satellite sampling density (annual N_NATIVE_OBSERVATIONS total, N months covered) ---
    supp = harm.groupby(["SITE_ID", "YEAR", "SIF_PRODUCT"]).agg(
        TOTAL_N_NATIVE_OBS=("N_NATIVE_OBSERVATIONS", "sum"),
        MEAN_EFFECTIVE_COUNT=("EFFECTIVE_NATIVE_COUNT", "mean"),
        N_MONTHS_WITH_DATA=("MONTH", "nunique"),
    ).reset_index()
    merged = valid.merge(supp, on=["SITE_ID", "YEAR", "SIF_PRODUCT"], how="left")

    rows = []
    for product in PRODUCTS:
        sub = merged[(merged["SIF_PRODUCT"] == product)].dropna(subset=["TOTAL_N_NATIVE_OBS", "ABS_DELTA_POS"])
        if len(sub) < 10:
            rows.append({"SIF_PRODUCT": product, "N": len(sub), "RHO_SAMPLING_VS_ABSDELTA": np.nan, "P": np.nan})
            continue
        rho, p = spearmanr(sub["TOTAL_N_NATIVE_OBS"], sub["ABS_DELTA_POS"])
        rows.append({"SIF_PRODUCT": product, "N": len(sub), "RHO_SAMPLING_VS_ABSDELTA": round(float(rho), 3), "P": round(float(p), 4)})
    sampling_result = pd.DataFrame(rows)
    sampling_result.to_csv(OUT_DIR / "artifactA_satellite_sampling_vs_absdeltapos.csv", index=False)
    print("Artifact A -- satellite sampling density vs |DeltaPOS| (negative rho = sparser sampling -> bigger offset):")
    print(sampling_result.to_string(index=False))

    # --- B: OCO-3 reconstruction ---
    oco3 = harm[harm["SIF_PRODUCT"] == "OCO-3"]
    recon_frac = oco3.groupby(["SITE_ID", "YEAR"]).agg(
        N_TOTAL=("OBSERVED_OR_RECONSTRUCTED", "size"),
        N_RECONSTRUCTED=("OBSERVED_OR_RECONSTRUCTED", lambda s: (s == "RECONSTRUCTED").sum()),
    ).reset_index()
    recon_frac["RECON_FRACTION"] = recon_frac["N_RECONSTRUCTED"] / recon_frac["N_TOTAL"]
    oco3_merged = valid[valid["SIF_PRODUCT"] == "OCO-3"].merge(recon_frac, on=["SITE_ID", "YEAR"], how="left")
    oco3_merged.to_csv(OUT_DIR / "artifactB_oco3_reconstruction_by_siteyear.csv", index=False)
    if oco3_merged["RECON_FRACTION"].notna().sum() >= 10 and oco3_merged["RECON_FRACTION"].nunique() > 1:
        rho_b, p_b = spearmanr(oco3_merged["RECON_FRACTION"].fillna(0), oco3_merged["ABS_DELTA_POS"])
        print(f"\nArtifact B -- OCO-3 reconstruction fraction vs |DeltaPOS|: rho={rho_b:.3f}, p={p_b:.4f}, n={len(oco3_merged)}")
    else:
        rho_b, p_b = np.nan, np.nan
        print("\nArtifact B -- insufficient variation in OCO-3 reconstruction fraction to test.")
    with open(OUT_DIR / "artifactB_summary.txt", "w") as f:
        f.write(f"OCO-3 reconstruction fraction vs |DeltaPOS|: rho={rho_b}, p={p_b}, n={len(oco3_merged)}\n"
                f"GOSIF/GOME-2/TROPOSIF are 100% OBSERVED in the analysis-ready master (no reconstruction to test).\n")

    # --- C: footprint/overlap mismatch (GOME-2: SUPPORT_AREA_RATIO; OCO-3: FRACTION_EC_SUPPORT_COVERED) ---
    fp_rows = []
    for product, col in [("GOME-2", "SUPPORT_AREA_RATIO"), ("OCO-3", "FRACTION_EC_SUPPORT_COVERED")]:
        h = harm[harm["SIF_PRODUCT"] == product]
        if col not in h.columns or h[col].nunique() <= 1:
            fp_rows.append({"SIF_PRODUCT": product, "METRIC": col, "N": 0, "RHO": np.nan, "P": np.nan, "NOTE": "constant/placeholder -- not testable"})
            continue
        agg = h.groupby(["SITE_ID", "YEAR"])[col].mean().reset_index()
        m = valid[valid["SIF_PRODUCT"] == product].merge(agg, on=["SITE_ID", "YEAR"], how="left").dropna(subset=[col, "ABS_DELTA_POS"])
        if len(m) < 10:
            fp_rows.append({"SIF_PRODUCT": product, "METRIC": col, "N": len(m), "RHO": np.nan, "P": np.nan, "NOTE": "insufficient N"})
            continue
        rho, p = spearmanr(m[col], m["ABS_DELTA_POS"])
        fp_rows.append({"SIF_PRODUCT": product, "METRIC": col, "N": len(m), "RHO": round(float(rho), 3), "P": round(float(p), 4), "NOTE": "OK"})
    footprint_result = pd.DataFrame(fp_rows)
    footprint_result.to_csv(OUT_DIR / "artifactC_footprint_overlap_vs_absdeltapos.csv", index=False)
    print("\nArtifact C -- footprint/overlap coverage vs |DeltaPOS| (GOSIF/TROPOSIF not testable, see data inventory):")
    print(footprint_result.to_string(index=False))

    # --- D: viewing geometry -- confirmed untestable, documented, not fabricated ---
    with open(OUT_DIR / "artifactD_viewing_geometry_NOTE.txt", "w") as f:
        f.write("Viewing/solar geometry (VZA, SZA, relative azimuth) is CONFIRMED ABSENT from the\n"
                "actual 5-product pipeline used in this project (see 01_data_inventory). This\n"
                "mechanism CANNOT be tested with this dataset. Not fabricated, not approximated.\n")
    print("\nArtifact D -- viewing geometry: confirmed untestable (see artifactD_viewing_geometry_NOTE.txt)")


if __name__ == "__main__":
    main()
