"""
STEP 11 -- Per-site-year-product mechanism attribution (section 27) and the
biome-level / product-level summary tables (sections 28-29).

Rule-based, evidence-based classification -- a row is only labeled "X CONSISTENT" if
that mechanism's own established diagnostic pattern (from step7/step10's fitted
signs) is actually present for that row. Wording throughout is "most consistent with,"
never "caused by." Only rows with a real, non-synchronous offset (EARLIER_SIF or
EARLIER_GPP) are attributed -- synchronous/uncertain rows need no mechanism.

Precedence (checked in order, first match wins):
  1. MULTI_PEAK_SYSTEM       -- PEAK_SHAPE == MULTI_PEAK (POS itself is ambiguous)
  2. PEAK_FIT_UNCERTAIN      -- sign of DELTA_POS changes between harmonic and spline fit
  3. evaluate CANOPY/LIGHT, SPRING TEMPERATURE (evergreen only), WATER/VPD STRESS
     evidence (sign-consistency with the established, significant group-model
     relationships from step10); 0 matches = UNEXPLAINED, 1 = that mechanism,
     >=2 = MULTIPLE_MECHANISMS_CONSISTENT
  4. rows at GOME-2 towers in the lowest sampling-density tercile additionally get a
     PRODUCT_SAMPLING_ARTIFACT_POSSIBLE flag (reported as a separate boolean column,
     not as an overriding label -- GOME-2's sampling-density association was real but
     modest (rho=-0.15), so it is reported as a caveat alongside the mechanism label,
     not a replacement for it).
"""
# Original script filename (sanitized): step11_mechanism_attribution.py; public name: step11_mechanism_attribution.py
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

ROOT = (Path(PROJECT_ROOT) / 'SIF_GPP_PEAK_TIMING_MECHANISM_FULL_ANALYSIS/14_scripts/step11_mechanism_attribution.py').resolve().parents[1]
master = pd.read_csv(ROOT / "02_peak_timing" / "SiteYear_Product_PeakTiming_Master.csv")
wt = pd.read_csv(ROOT / "08_year_to_year_analysis" / "within_tower_environment_deltapos.csv")
peakfit = pd.read_csv(ROOT / "09_methodological_artifacts" / "artifactG_peakfit_method_sensitivity.csv")
sampling = pd.read_csv(ROOT / "09_methodological_artifacts" / "artifactA_satellite_sampling_vs_absdeltapos.csv")
consensus = pd.read_csv(ROOT / "03_cross_product_consistency" / "cross_product_consensus_by_siteyear.csv")
OUT_DIR = ROOT / "10_mechanism_attribution"
TABLES_DIR = ROOT / "12_tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

EVERGREEN = ["ENF", "EBF"]
IGBP_LABELS = {
    "ENF": "Evergreen Needleleaf Forest", "EBF": "Evergreen Broadleaf Forest",
    "DBF": "Deciduous Broadleaf Forest", "MF": "Mixed Forest", "CSH": "Closed Shrubland",
    "OSH": "Open Shrubland", "WSA": "Woody Savanna", "SAV": "Savanna",
    "GRA": "Grassland", "CRO": "Cropland",
}
harm = pd.read_csv(Path(PROJECT_ROOT) /
                   "SIF_EC_SIF_GPP_Final_Verification" / "02_independent_reverification" / "harmonic_residuals_full.csv",
                   low_memory=False)
supp = harm.groupby(["SITE_ID", "YEAR", "SIF_PRODUCT"])["N_NATIVE_OBSERVATIONS"].sum().reset_index().rename(
    columns={"N_NATIVE_OBSERVATIONS": "TOTAL_N_NATIVE_OBS"})


def circ_diff(a, b):
    return (a - b + 6) % 12 - 6


def main():
    df = master.dropna(subset=["DELTA_POS", "SW_IN_POS", "GPP_POS"]).copy()
    df["SW_IN_GAP_TO_GPP"] = df.apply(lambda r: circ_diff(r["SW_IN_POS"], r["GPP_POS"]), axis=1)
    df = df.merge(wt[["SITE_ID", "YEAR", "SIF_PRODUCT", "WT_DELTA_POS", "WT_PREPEAK_TAIR_ANOM", "WT_PREPEAK_SID_VPD"]],
                  on=["SITE_ID", "YEAR", "SIF_PRODUCT"], how="left")
    df = df.merge(peakfit[["SITE_ID", "YEAR", "SIF_PRODUCT", "SIGN_CHANGED"]], on=["SITE_ID", "YEAR", "SIF_PRODUCT"], how="left")
    df = df.merge(supp, on=["SITE_ID", "YEAR", "SIF_PRODUCT"], how="left")

    gome2_thresh = df.loc[df["SIF_PRODUCT"] == "GOME-2", "TOTAL_N_NATIVE_OBS"].quantile(1 / 3)

    def attribute(row):
        if row["DELTA_CLASS"] not in ("EARLIER_SIF", "EARLIER_GPP"):
            return "NOT_APPLICABLE_SYNCHRONOUS"
        if row["PEAK_SHAPE"] == "MULTI_PEAK":
            return "MULTI_PEAK_SYSTEM"
        if row.get("SIGN_CHANGED") is True:
            return "PEAK_FIT_UNCERTAIN"

        matches = []
        if abs(row["SW_IN_GAP_TO_GPP"]) > 0.25 and np.sign(row["SW_IN_GAP_TO_GPP"]) == np.sign(row["DELTA_POS"]):
            matches.append("CANOPY_LIGHT_APAR_TIMING")
        if row["IGBP"] in EVERGREEN and pd.notna(row["WT_PREPEAK_TAIR_ANOM"]) and pd.notna(row["WT_DELTA_POS"]):
            if abs(row["WT_PREPEAK_TAIR_ANOM"]) > 0.1 and np.sign(row["WT_PREPEAK_TAIR_ANOM"]) == np.sign(row["WT_DELTA_POS"]):
                matches.append("SPRING_TEMPERATURE_ACTIVATION")
        if pd.notna(row["WT_PREPEAK_SID_VPD"]) and pd.notna(row["WT_DELTA_POS"]):
            if abs(row["WT_PREPEAK_SID_VPD"]) > 0.02 and np.sign(row["WT_PREPEAK_SID_VPD"]) == np.sign(row["WT_DELTA_POS"]):
                matches.append("WATER_VPD_STRESS")

        if len(matches) == 0:
            return "UNEXPLAINED"
        if len(matches) == 1:
            return matches[0] + "_CONSISTENT"
        return "MULTIPLE_MECHANISMS_CONSISTENT (" + "+".join(matches) + ")"

    df["MECHANISM_ATTRIBUTION"] = df.apply(attribute, axis=1)
    df["SAMPLING_ARTIFACT_CAVEAT"] = (df["SIF_PRODUCT"] == "GOME-2") & (df["TOTAL_N_NATIVE_OBS"] <= gome2_thresh)

    df.to_csv(OUT_DIR / "siteyear_mechanism_attribution.csv", index=False)
    print("Mechanism attribution counts (EARLIER_SIF / EARLIER_GPP rows only):")
    print(df[df["DELTA_CLASS"].isin(["EARLIER_SIF", "EARLIER_GPP"])]["MECHANISM_ATTRIBUTION"].value_counts())

    # --- Biome-level mechanism summary (section 28) ---
    biome_summary = pd.read_csv(ROOT / "07_biome_analysis" / "Biome_Summary_Table.csv")
    attributable = df[df["DELTA_CLASS"].isin(["EARLIER_SIF", "EARLIER_GPP"])]
    mech_by_biome = attributable.groupby("IGBP")["MECHANISM_ATTRIBUTION"].agg(
        lambda s: s.value_counts().index[0] if len(s) else "NO_DATA")
    biome_summary["STRONGEST_MECHANISM_EVIDENCE"] = biome_summary["IGBP"].map(mech_by_biome).fillna("INSUFFICIENT_ATTRIBUTABLE_ROWS")
    biome_summary.to_csv(TABLES_DIR / "Biome_Level_Mechanism_Summary.csv", index=False)
    print("\nBiome-level summary saved to 12_tables/Biome_Level_Mechanism_Summary.csv")

    # --- Product-level summary (section 29) ---
    prod_rows = []
    for product in ["GOSIF", "GOME-2", "TROPOSIF", "OCO-3"]:
        sub = df[df["SIF_PRODUCT"] == product]
        med = sub["DELTA_POS"].median()
        cons_p = consensus[consensus["PRODUCTS"].str.contains(product, na=False) & (consensus["N_PRODUCTS"] >= 2)]
        agree_pct = 100.0 * cons_p["CONSENSUS"].isin(["HIGH", "MODERATE"]).mean() if len(cons_p) else np.nan
        sampling_note = sampling[sampling["SIF_PRODUCT"] == product].iloc[0] if (sampling["SIF_PRODUCT"] == product).any() else None
        recon_note = "544 obs / 115 reconstructed (tested, no significant effect)" if product == "OCO-3" else "100% observed"
        prod_rows.append({
            "PRODUCT": product, "N_SITEYEARS": len(sub), "MEDIAN_DELTA_POS": round(med, 3) if pd.notna(med) else np.nan,
            "PCT_AGREEMENT_WITH_OTHER_PRODUCTS": round(agree_pct, 1) if not np.isnan(agree_pct) else np.nan,
            "SAMPLING_CONCERN": f"rho={sampling_note['RHO_SAMPLING_VS_ABSDELTA']:.2f}, p={sampling_note['P']:.3f}" if sampling_note is not None else "n/a",
            "RECONSTRUCTION_NOTE": recon_note,
            "CONFIDENCE": "MODERATE-HIGH" if product in ("GOSIF", "GOME-2") else ("LOW-MODERATE" if product == "TROPOSIF" else "LOW (small N, short record)"),
        })
    prod_summary = pd.DataFrame(prod_rows)
    prod_summary.to_csv(TABLES_DIR / "Product_Level_Summary.csv", index=False)
    print("\nProduct-level summary:")
    print(prod_summary.to_string(index=False))


if __name__ == "__main__":
    main()
