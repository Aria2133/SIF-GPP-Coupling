"""
PART 4: LST seasonal anomaly (per-site harmonic regression, same functional form
as the project's existing SIF/GPP harmonic deseasonalization) + positive thermal
stress, scaled using the EXACT SAME philosophy as the existing TEMP_stress
(verified below by reading 02_normalization_and_stress.py before writing this):

  TEMP_stress = clip[(max(TAIR_ANOM, 0) - P01) / (P99 - P01), 0, 1]
  computed GLOBALLY (pooled across ALL site-months, NOT site-relative).

So LST_stress uses the identical recipe:
  LST_STRESS_RAW = max(LST_ANOM, 0)
  LST_stress     = clip[(LST_STRESS_RAW - P01) / (P99 - P01), 0, 1], pooled globally.
This makes SID_LST directly comparable in scale/philosophy to SID_TEMP.

PART 5: SID_LST = Decoupling * LST_stress. `Decoupling` is NOT recomputed --
reused unmodified from SIF_EC_Stress_Induced_Decoupling_SID/03_sid_monthly/
monthly_SID_master.csv (the existing final implementation), joined on
SITE_ID x YYYYMM x SIF_PRODUCT. SID_VPD/SID_SM/SID_TEMP are untouched and not
read/written by this script at all.

Minimum-data rule for the harmonic fit: a site needs >=12 valid LST site-months
to get a harmonic anomaly (5 free parameters in the model; 12 is this project's
standing reliability threshold elsewhere, applied here for consistency). Towers
below this are EXCLUDED from LST_ANOM/LST_stress/SID_LST (documented, not
silently dropped) -- their raw LST_degC values remain in
SIF_GPP_LST_analysis_ready.csv regardless.
"""
# Original script filename (sanitized): 03_LST_anomaly_stress_and_SID_LST.py; public name: 03_LST_anomaly_stress_and_SID_LST.py
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

ROOT = Path(PROJECT_ROOT)
LST_DIR = ROOT / "LST_SIF_GPP_EXTENSION_FINAL" / "02_processed_LST"
TAB_DIR = ROOT / "LST_SIF_GPP_EXTENSION_FINAL" / "03_analysis_tables"
QC_DIR = ROOT / "LST_SIF_GPP_EXTENSION_FINAL" / "07_QC"
SID_MASTER_CSV = ROOT / "SIF_EC_Stress_Induced_Decoupling_SID" / "03_sid_monthly" / "monthly_SID_master.csv"

N_MIN_HARMONIC = 12


def fit_harmonic(sub: pd.DataFrame, value_col: str):
    """Per-site 1st+2nd order Fourier regression on calendar month (not year-aware --
    identical functional form to the project's existing SIF/GPP harmonic deseasonalization).
    Returns residuals aligned to sub's index, or all-NaN if below N_MIN_HARMONIC."""
    n = sub[value_col].notna().sum()
    if n < N_MIN_HARMONIC:
        return pd.Series(np.nan, index=sub.index), False
    t = 2 * np.pi * sub["MONTH"].values / 12.0
    X = np.column_stack([np.ones(len(sub)), np.sin(t), np.cos(t), np.sin(2 * t), np.cos(2 * t)])
    y = sub[value_col].values
    mask = ~np.isnan(y)
    coef, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
    fitted = X @ coef
    resid = np.full(len(sub), np.nan)
    resid[mask] = y[mask] - fitted[mask]
    return pd.Series(resid, index=sub.index), True


def main():
    tower_lst = pd.read_csv(LST_DIR / "tower_monthly_LST.csv")

    anom_rows = []
    fit_status = []
    for site_id, sub in tower_lst.groupby("SITE_ID"):
        sub = sub.sort_values(["YEAR", "MONTH"]).reset_index(drop=True)
        resid, fitted_ok = fit_harmonic(sub, "LST_degC")
        sub["LST_ANOM"] = resid.values
        anom_rows.append(sub)
        fit_status.append({"SITE_ID": site_id, "N_VALID_MONTHS": int(sub["LST_degC"].notna().sum()),
                            "HARMONIC_FITTED": fitted_ok})
    tower_lst_anom = pd.concat(anom_rows, ignore_index=True)
    fit_status_df = pd.DataFrame(fit_status)
    n_fitted = int(fit_status_df["HARMONIC_FITTED"].sum())
    print(f"Harmonic LST anomaly fitted for {n_fitted} of {len(fit_status_df)} covered towers "
          f"(threshold: >={N_MIN_HARMONIC} valid months); "
          f"excluded: {fit_status_df[~fit_status_df['HARMONIC_FITTED']]['SITE_ID'].tolist()}")

    # ---- positive thermal stress, GLOBAL P01/P99 scaling (same philosophy as TEMP_stress) ----
    tower_lst_anom["LST_STRESS_RAW"] = tower_lst_anom["LST_ANOM"].clip(lower=0)
    valid_stress = tower_lst_anom["LST_STRESS_RAW"].dropna()
    p01, p99 = np.nanpercentile(valid_stress, 1), np.nanpercentile(valid_stress, 99)
    print(f"LST_STRESS_RAW global P01={p01:.4f}, P99={p99:.4f} degC (N={len(valid_stress)})")
    tower_lst_anom["LST_stress"] = ((tower_lst_anom["LST_STRESS_RAW"] - p01) / (p99 - p01)).clip(0, 1)
    tower_lst_anom.loc[tower_lst_anom["LST_ANOM"].isna(), "LST_stress"] = np.nan

    tower_lst_anom.to_csv(LST_DIR / "tower_monthly_LST_anomaly_stress.csv", index=False)
    fit_status_df.to_csv(QC_DIR / "LST_harmonic_fit_status_by_tower.csv", index=False)
    print(f"Saved tower_monthly_LST_anomaly_stress.csv: {len(tower_lst_anom)} rows")

    # ================= PART 5: SID_LST =================
    sid = pd.read_csv(SID_MASTER_CSV, low_memory=False,
                       usecols=["SITE_ID", "YYYYMM", "YEAR", "MONTH", "SIF_PRODUCT", "Decoupling", "IGBP"])
    merged = sid.merge(
        tower_lst_anom[["SITE_ID", "YEAR", "MONTH", "LST_degC", "LST_ANOM", "LST_STRESS_RAW", "LST_stress"]],
        on=["SITE_ID", "YEAR", "MONTH"], how="left")
    merged["SID_LST"] = merged["Decoupling"] * merged["LST_stress"]

    # sanity checks (mirrors the existing SID_* sanity checks in 03_compute_sid.py)
    valid_sid = merged["SID_LST"].dropna()
    assert (valid_sid >= -1e-9).all(), "SID_LST has negative values"
    assert not np.isinf(valid_sid).any(), "SID_LST has infinite values"
    print(f"SID_LST: n_valid={len(valid_sid)}, min={valid_sid.min():.4f}, max={valid_sid.max():.4f} "
          f"(expected max ~1.0 since Decoupling in [0,1] and LST_stress in [0,1])")

    merged.to_csv(LST_DIR / "SID_LST_observation_level.csv", index=False)
    print(f"Saved SID_LST_observation_level.csv: {len(merged)} rows, {merged['SID_LST'].notna().sum()} with valid SID_LST")

    print("\nPART 4-5 DONE.")


if __name__ == "__main__":
    main()
