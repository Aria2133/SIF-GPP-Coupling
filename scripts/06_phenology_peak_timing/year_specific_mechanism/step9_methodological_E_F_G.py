"""
STEP 9 -- Methodological artifacts E (GOSIF model dependence), F (GPP partitioning
DT vs. NT), G (peak-fitting/curve-method sensitivity).
"""
# Original script filename (sanitized): step9_methodological_E_F_G.py; public name: step9_methodological_E_F_G.py
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
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = (Path(PROJECT_ROOT) / 'SIF_GPP_PEAK_TIMING_MECHANISM_FULL_ANALYSIS/14_scripts/step9_methodological_E_F_G.py').resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import lib_year_specific_pos as ly  # noqa: E402

ROOT = Path(PROJECT_ROOT)
ANALYSIS_ROOT = (Path(PROJECT_ROOT) / 'SIF_GPP_PEAK_TIMING_MECHANISM_FULL_ANALYSIS/14_scripts/step9_methodological_E_F_G.py').resolve().parents[1]
MASTER_CSV = ANALYSIS_ROOT / "02_peak_timing" / "SiteYear_Product_PeakTiming_Master.csv"
SIF_GPP_MASTER = ROOT / "SIF_EC_SIF_GPP_Relationship_Analysis" / "01_analysis_ready" / "SIF_GPP_Analysis_Ready_Master.csv"
MONTHLY_FLUXNET = ROOT / "fluxnet" / "Monthly Datasets" / "fluxnet_monthly_combined.csv"
OUT_DIR = ANALYSIS_ROOT / "09_methodological_artifacts"


def artifact_E_gosif_dependence():
    master = pd.read_csv(MASTER_CSV)
    valid = master.dropna(subset=["DELTA_POS"])
    with_gosif = valid
    without_gosif = valid[valid["SIF_PRODUCT"] != "GOSIF"]

    def summarize(df, label):
        n = len(df)
        pct_sif = 100.0 * (df["DELTA_CLASS"] == "EARLIER_SIF").mean()
        pct_gpp = 100.0 * (df["DELTA_CLASS"] == "EARLIER_GPP").mean()
        med = df["DELTA_POS"].median()
        return {"SUBSET": label, "N_SITEYEARS": n, "PCT_SIF_EARLIER": round(pct_sif, 1),
                "PCT_GPP_EARLIER": round(pct_gpp, 1), "MEDIAN_DELTA_POS": round(med, 3)}

    result = pd.DataFrame([summarize(with_gosif, "WITH_GOSIF (all products)"),
                          summarize(without_gosif, "WITHOUT_GOSIF (GOME-2+TROPOSIF+OCO-3 only)")])
    result.to_csv(OUT_DIR / "artifactE_gosif_dependence.csv", index=False)
    print("Artifact E -- conclusion with vs. without GOSIF:")
    print(result.to_string(index=False))
    direction_same = np.sign(with_gosif["DELTA_POS"].median()) == np.sign(without_gosif["DELTA_POS"].median())
    print(f"Same-direction median DeltaPOS with/without GOSIF: {direction_same}")


def artifact_F_gpp_partitioning():
    if not MONTHLY_FLUXNET.exists():
        print("Artifact F -- fluxnet_monthly_combined.csv not found; skipping.")
        return
    fx = pd.read_csv(MONTHLY_FLUXNET, low_memory=False)
    fx["date"] = pd.to_datetime(fx["date"])
    fx["YEAR"] = fx["date"].dt.year
    fx["MONTH"] = fx["date"].dt.month
    fx = fx.dropna(subset=["gpp_dt_ref", "gpp_nt_ref"])

    rows = []
    for (site, year), g in fx.groupby(["site_id", "YEAR"]):
        months = g["MONTH"].to_numpy()
        dt_res = ly.estimate_pos_one_year(months, g["gpp_dt_ref"].to_numpy(dtype=float))
        nt_res = ly.estimate_pos_one_year(months, g["gpp_nt_ref"].to_numpy(dtype=float))
        if np.isnan(dt_res["POS"]) or np.isnan(nt_res["POS"]):
            continue
        gap = ly.pl.circular_diff(nt_res["POS"], dt_res["POS"])
        rows.append({"SITE_ID": site, "YEAR": year, "POS_GPP_DT": dt_res["POS"], "POS_GPP_NT": nt_res["POS"],
                     "GAP_NT_MINUS_DT": gap})
    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / "artifactF_gpp_partitioning_sensitivity.csv", index=False)
    if len(result):
        med_gap = result["GAP_NT_MINUS_DT"].abs().median()
        pct_within1 = 100.0 * (result["GAP_NT_MINUS_DT"].abs() <= 1.0).mean()
        print(f"\nArtifact F -- GPP DT vs. NT partitioning (n={len(result)} site-years, "
              f"{result['SITE_ID'].nunique()} towers, PARTIAL coverage: 21/73 towers, 2002-2014 only):")
        print(f"  median |POS_GPP_NT - POS_GPP_DT| = {med_gap:.2f} months; {pct_within1:.0f}% agree within 1 month")
        with open(OUT_DIR / "artifactF_summary.txt", "w") as f:
            f.write(f"GPP partitioning (DT vs NT) sensitivity: n={len(result)} site-years, "
                    f"{result['SITE_ID'].nunique()} towers (PARTIAL coverage -- only 21/73 towers have "
                    f"monthly-aggregated GPP_NT_VUT_REF available, 2002-2014 only).\n"
                    f"Median |gap| = {med_gap:.2f} months. {pct_within1:.0f}% of site-years agree within 1 month.\n"
                    f"This is a bounded, partial-coverage sensitivity check, not a full-network result.\n")
    else:
        print("Artifact F -- no valid site-years for DT/NT comparison.")


def artifact_G_peakfit_sensitivity():
    """Compare year-specific POS from the primary harmonic fit vs. a periodic cubic
    spline fit (reusing pos_lib's fit_periodic_spline, applied per-year rather than
    per-tower-climatology) -- flags site-years where the SIGN of DELTA_POS changes."""
    df = pd.read_csv(SIF_GPP_MASTER, low_memory=False)
    df = df[df["SIF_PRODUCT"].isin(["GOSIF", "GOME-2", "TROPOSIF", "OCO-3"])].dropna(
        subset=["SIF_VALUE", "GPP_VALUE", "MONTH", "YEAR"])
    df["MONTH"] = df["MONTH"].astype(int)
    df["YEAR"] = df["YEAR"].astype(int)

    master = pd.read_csv(MASTER_CSV)
    master_valid = master.dropna(subset=["DELTA_POS"])

    rows = []
    for (site, year, product), g in df.groupby(["SITE_ID", "YEAR", "SIF_PRODUCT"]):
        row = master_valid[(master_valid["SITE_ID"] == site) & (master_valid["YEAR"] == year) & (master_valid["SIF_PRODUCT"] == product)]
        if len(row) == 0 or np.isnan(row.iloc[0]["DELTA_POS"]):
            continue
        month_mean_sif = g.groupby("MONTH")["SIF_VALUE"].mean()
        month_mean_gpp = g.groupby("MONTH")["GPP_VALUE"].mean()
        cs_sif = ly.pl.fit_periodic_spline(month_mean_sif)
        cs_gpp = ly.pl.fit_periodic_spline(month_mean_gpp)
        if cs_sif is None or cs_gpp is None:
            continue
        grid = np.linspace(1e-6, 12, ly.pl.GRID_N)
        pos_sif_spline = float(grid[int(np.argmax(cs_sif(grid)))] % 12) or 12.0
        pos_gpp_spline = float(grid[int(np.argmax(cs_gpp(grid)))] % 12) or 12.0
        delta_spline = ly.pl.circular_diff(pos_sif_spline, pos_gpp_spline)
        delta_harmonic = row.iloc[0]["DELTA_POS"]
        rows.append({"SITE_ID": site, "YEAR": year, "SIF_PRODUCT": product,
                     "DELTA_POS_HARMONIC": delta_harmonic, "DELTA_POS_SPLINE": delta_spline,
                     "SIGN_CHANGED": bool(np.sign(delta_harmonic) != np.sign(delta_spline))})
    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / "artifactG_peakfit_method_sensitivity.csv", index=False)
    pct_sign_changed = 100.0 * result["SIGN_CHANGED"].mean() if len(result) else np.nan
    print(f"\nArtifact G -- peak-fitting method sensitivity (harmonic vs. periodic spline), n={len(result)}:")
    print(f"  sign of DeltaPOS changes with fitting method in {pct_sign_changed:.1f}% of site-years")


if __name__ == "__main__":
    artifact_E_gosif_dependence()
    artifact_F_gpp_partitioning()
    artifact_G_peakfit_sensitivity()
