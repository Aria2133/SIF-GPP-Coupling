"""
GOME-2 temporal-support-mismatch sensitivity check.

Motivation
----------
The production SIF-GPP master pairs a monthly satellite-EC-weighted SIF value
with a monthly tower GPP value that is a full-month daytime aggregate -- it is
NOT restricted to the specific calendar dates on which a GOME-2 sounding
actually contributed to that site-month's EC-weighted SIF. An earlier audit
(SIF_EC_SIF_GPP_Independent_Statistical_Audit) concluded that a real
overpass-day-restricted GPP sensitivity check was not feasible from the
analysis-ready master alone.

This script tests whether it IS feasible for GOME-2 specifically, using two
locally-available ingredients that ARE fine-grained enough:
  1. SIF_EC_GOME2_Finalization\\GOME2_EC_Observation_Overlap_Provenance.csv,
     which records `observation_datetime` (exact date/time) for every native
     GOME-2 sounding that contributed to every site-month's EC-weighted SIF,
     together with `normalized_monthly_weight` (its contribution weight).
  2. Raw half-hourly FLUXNET2015 tower files
     (fluxnet\\Hourly Data (for Wind Direction)\\FLX_<SITE>_..._HH_*.csv),
     which contain GPP_DT_VUT_REF (half-hourly GPP, daytime-partitioning
     method) and TIMESTAMP_START/END.

For a limited sample of towers (the ones with the most GOME-2 site-months
AMONG the towers that actually have a local raw FLUXNET HH file -- only 19 of
GOME-2's 68 towers have one locally; see "Coverage limitation" below), this
script computes, per site-month:
  (a) FULL-MONTH daytime GPP  = mean(GPP_DT_VUT_REF) over all half-hours in
      that calendar month with SW_IN_POT > 0 (potential incoming shortwave
      radiation, a purely geometric/deterministic daytime indicator, so this
      does not depend on any gap-filling or QC flag choice).
  (b) SATELLITE-DAY-RESTRICTED daytime GPP = same, but restricted further to
      only the calendar dates on which a GOME-2 sounding with positive
      contribution weight is recorded for that site-month in the provenance
      table.
and compares the raw and site-calendar-month-climatology-anomaly Pearson r
between GOME-2 SIF and each of the two GPP definitions.

Coverage limitation (documented, not hidden)
---------------------------------------------
Only 19 of the 68 GOME-2 towers in the production master have a local raw
FLUXNET2015 HH file under `fluxnet\\Hourly Data (for Wind Direction)\\`
(checked by globbing FLX_*_HH_*.csv across the whole project: 21 files exist
in total, covering 21 distinct sites, 19 of which are GOME-2 towers). None of
the GOME-2 towers with the largest number of site-months (US-MMS, CH-Dav,
CH-Oe2, CH-Lae, US-Bar, US-Ha1, US-Ne1, IT-Ren, US-Ne2, US-Ton, US-SRM,
ES-LJu, FR-Pue, JP-Tak, IT-BCi -- all 127-132 site-months) have a local HH
file. This script therefore uses the largest-by-site-month-count subset of
the 19 towers that DO have local HH data, per the task's explicit allowance
to use "however many you can" when full coverage is not available.

No timezone reconciliation is attempted between the GOME-2 `observation_datetime`
(satellite UTC) and the FLUXNET `TIMESTAMP_START` (site local standard time
per the FLUXNET2015 convention): calendar dates are compared as given. This is
a second documented approximation -- for towers far from the Greenwich
meridian, an overpass shortly before/after local midnight UTC could match to
the "wrong" local calendar date. Given GOME-2's ~09:30 local solar overpass
time, this edge effect is expected to be rare (the overpass is mid-morning
local time almost everywhere) but is not empirically ruled out here.

This is a side computation only; no production data is modified.
"""
from __future__ import annotations
# Original script filename (sanitized): gome2_satellite_day_restricted_gpp_sensitivity.py; public name: gome2_satellite_day_restricted_gpp_sensitivity.py
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

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(
    PROJECT_ROOT
)
PROV_PATH = PROJECT_ROOT / "SIF_EC_GOME2_Finalization" / "GOME2_EC_Observation_Overlap_Provenance.csv"
MASTER_PATH = (
    PROJECT_ROOT / "SIF_EC_SIF_GPP_Relationship_Analysis" / "01_analysis_ready" / "SIF_GPP_Analysis_Ready_Master.csv"
)
FLUXNET_HH_ROOT = PROJECT_ROOT / "fluxnet" / "Hourly Data (for Wind Direction)"
OUT_DIR = PROJECT_ROOT / "SIF_EC_SIF_GPP_Final_Verification" / "04_sampling_and_QA"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_TOWERS = 15


def pearson_spearman_r2(x: np.ndarray, y: np.ndarray) -> dict:
    n = len(x)
    if n < 3 or np.std(x) == 0 or np.std(y) == 0:
        return dict(N=n, pearson_r=np.nan, spearman_rho=np.nan, r2=np.nan)
    r, _ = stats.pearsonr(x, y)
    rho, _ = stats.spearmanr(x, y)
    slope, intercept, rval, _, _ = stats.linregress(x, y)
    return dict(N=n, pearson_r=r, spearman_rho=rho, r2=rval ** 2)


def current_anomaly_r(g: pd.DataFrame, sif_col: str, gpp_col: str) -> dict:
    clim = g.groupby(["SITE_ID", "MONTH"]).agg(sif_clim=(sif_col, "mean"), gpp_clim=(gpp_col, "mean"))
    gg = g.join(clim, on=["SITE_ID", "MONTH"])
    sif_anom = gg[sif_col] - gg["sif_clim"]
    gpp_anom = gg[gpp_col] - gg["gpp_clim"]
    return pearson_spearman_r2(sif_anom.to_numpy(), gpp_anom.to_numpy())


def find_hh_file(site_id: str) -> Path | None:
    matches = sorted(FLUXNET_HH_ROOT.glob(f"FLX_{site_id}_FLUXNET2015_SUBSET_HH_*.csv"))
    return matches[0] if matches else None


def load_site_hh(site_id: str) -> pd.DataFrame | None:
    path = find_hh_file(site_id)
    if path is None:
        return None
    usecols = ["TIMESTAMP_START", "SW_IN_POT", "GPP_DT_VUT_REF"]
    df = pd.read_csv(path, usecols=usecols)
    df["TIMESTAMP_START"] = df["TIMESTAMP_START"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["dt"] = pd.to_datetime(df["TIMESTAMP_START"], format="%Y%m%d%H%M", errors="coerce")
    df = df.dropna(subset=["dt"])
    df["date"] = df["dt"].dt.date
    df["YYYYMM"] = df["dt"].dt.year * 100 + df["dt"].dt.month
    df["GPP_DT_VUT_REF"] = pd.to_numeric(df["GPP_DT_VUT_REF"], errors="coerce")
    df.loc[df["GPP_DT_VUT_REF"] <= -9990, "GPP_DT_VUT_REF"] = np.nan
    df["is_daytime"] = pd.to_numeric(df["SW_IN_POT"], errors="coerce") > 0
    return df


def main() -> None:
    prov = pd.read_csv(
        PROV_PATH,
        usecols=["SITE_ID", "YYYYMM", "observation_datetime", "normalized_monthly_weight"],
    )
    prov["contributes"] = pd.to_numeric(prov["normalized_monthly_weight"], errors="coerce").fillna(0) > 0
    prov["obs_date"] = pd.to_datetime(prov["observation_datetime"], errors="coerce").dt.date
    contributing = prov[prov["contributes"] & prov["obs_date"].notna()].copy()

    master = pd.read_csv(MASTER_PATH, low_memory=False)
    master = master[master["SIF_PRODUCT"] == "GOME-2"].copy()
    master["YYYYMM"] = pd.to_numeric(master["YYYYMM"], errors="coerce").astype("Int64")
    master["MONTH"] = pd.to_numeric(master["MONTH"], errors="coerce")
    master["GPP_VALUE"] = pd.to_numeric(master["GPP_VALUE"], errors="coerce")
    master["SIF_VALUE"] = pd.to_numeric(master["SIF_VALUE"], errors="coerce")

    all_hh_sites = sorted({p.name.split("_")[1] for p in FLUXNET_HH_ROOT.glob("FLX_*_FLUXNET2015_SUBSET_HH_*.csv")})
    gome2_sites_with_hh = sorted(set(master["SITE_ID"].unique()) & set(all_hh_sites))
    counts = master[master["SITE_ID"].isin(gome2_sites_with_hh)].groupby("SITE_ID")["YYYYMM"].nunique().sort_values(ascending=False)
    chosen_sites = counts.head(MAX_TOWERS).index.tolist()

    print(f"GOME-2 towers with a local FLUXNET HH file: {len(gome2_sites_with_hh)} of {master['SITE_ID'].nunique()} total GOME-2 towers.")
    print(f"Using top {len(chosen_sites)} of those by GOME-2 site-month count:")
    for s in chosen_sites:
        print(f"  {s}: {int(counts[s])} GOME-2 site-months")

    rows = []
    site_status = []
    for site_id in chosen_sites:
        hh = load_site_hh(site_id)
        if hh is None or hh.empty:
            site_status.append({"SITE_ID": site_id, "status": "NO_HH_DATA_LOADED", "n_sitemonths_completed": 0})
            continue
        site_master = master[master["SITE_ID"] == site_id]
        site_prov = contributing[contributing["SITE_ID"] == site_id]
        n_done = 0
        for _, mrow in site_master.iterrows():
            yyyymm = int(mrow["YYYYMM"])
            month_hh = hh[hh["YYYYMM"] == yyyymm]
            if month_hh.empty:
                continue  # tower has no half-hourly coverage for this calendar month
            full_daytime = month_hh[month_hh["is_daytime"]]["GPP_DT_VUT_REF"].dropna()
            if full_daytime.empty:
                continue
            sat_dates = set(site_prov.loc[site_prov["YYYYMM"] == yyyymm, "obs_date"])
            if not sat_dates:
                continue  # no provenance-recorded contributing observation for this site-month
            sat_daytime = month_hh[month_hh["is_daytime"] & month_hh["date"].isin(sat_dates)]["GPP_DT_VUT_REF"].dropna()
            if sat_daytime.empty:
                continue
            rows.append(
                {
                    "SITE_ID": site_id,
                    "YYYYMM": yyyymm,
                    "MONTH": int(yyyymm % 100),
                    "SIF_VALUE": mrow["SIF_VALUE"],
                    "GPP_full_month_daytime": float(full_daytime.mean()),
                    "n_halfhours_full_month_daytime": int(full_daytime.shape[0]),
                    "GPP_satellite_day_restricted_daytime": float(sat_daytime.mean()),
                    "n_halfhours_satellite_day_restricted": int(sat_daytime.shape[0]),
                    "n_satellite_contributing_dates": len(sat_dates),
                }
            )
            n_done += 1
        site_status.append({"SITE_ID": site_id, "status": "OK", "n_sitemonths_completed": n_done})

    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / "gome2_satellite_day_restricted_gpp_recompute.csv", index=False)
    status_df = pd.DataFrame(site_status)
    status_df.to_csv(OUT_DIR / "gome2_satellite_day_restricted_gpp_site_status.csv", index=False)

    print(f"\nCompleted site-months: {len(result)} across {result['SITE_ID'].nunique() if not result.empty else 0} towers.")
    print(status_df.to_string(index=False))

    summary = {
        "n_gome2_towers_total": int(master["SITE_ID"].nunique()),
        "n_gome2_towers_with_local_HH_file": len(gome2_sites_with_hh),
        "towers_attempted": chosen_sites,
        "n_sitemonths_completed": int(len(result)),
        "n_towers_completed": int(result["SITE_ID"].nunique()) if not result.empty else 0,
    }

    if len(result) >= 10:
        d = result.dropna(subset=["SIF_VALUE", "GPP_full_month_daytime", "GPP_satellite_day_restricted_daytime"]).copy()
        raw_full = pearson_spearman_r2(d["SIF_VALUE"].to_numpy(), d["GPP_full_month_daytime"].to_numpy())
        raw_sat = pearson_spearman_r2(d["SIF_VALUE"].to_numpy(), d["GPP_satellite_day_restricted_daytime"].to_numpy())
        anom_full = current_anomaly_r(d, "SIF_VALUE", "GPP_full_month_daytime")
        anom_sat = current_anomaly_r(d, "SIF_VALUE", "GPP_satellite_day_restricted_daytime")

        diff = (d["GPP_full_month_daytime"] - d["GPP_satellite_day_restricted_daytime"]).abs()
        rel_diff = diff / d["GPP_full_month_daytime"].abs().clip(lower=1e-6)

        summary.update(
            {
                "N_matched_site_months": int(len(d)),
                "N_towers_matched": int(d["SITE_ID"].nunique()),
                "raw_r_full_month_daytime_GPP": raw_full,
                "raw_r_satellite_day_restricted_GPP": raw_sat,
                "current_anomaly_r_full_month_daytime_GPP": anom_full,
                "current_anomaly_r_satellite_day_restricted_GPP": anom_sat,
                "GPP_median_abs_diff_full_vs_satellite_day": float(diff.median()),
                "GPP_median_relative_diff_full_vs_satellite_day": float(rel_diff.median()),
                "GPP_mean_relative_diff_full_vs_satellite_day": float(rel_diff.mean()),
            }
        )
        print("\n=== FULL-MONTH daytime GPP ===")
        print("  RAW r:", raw_full)
        print("  CURRENT ANOMALY r:", anom_full)
        print("\n=== SATELLITE-DAY-RESTRICTED daytime GPP ===")
        print("  RAW r:", raw_sat)
        print("  CURRENT ANOMALY r:", anom_sat)
        print(f"\nGPP median |full - satellite-day| = {diff.median():.4f}; median relative diff = {rel_diff.median() * 100:.1f}%")

        pd.DataFrame(
            [
                {"gpp_definition": "full_month_daytime", "stage": "raw", **raw_full},
                {"gpp_definition": "full_month_daytime", "stage": "current_anomaly", **anom_full},
                {"gpp_definition": "satellite_day_restricted_daytime", "stage": "raw", **raw_sat},
                {"gpp_definition": "satellite_day_restricted_daytime", "stage": "current_anomaly", **anom_sat},
            ]
        ).to_csv(OUT_DIR / "gome2_satellite_day_restricted_gpp_correlations.csv", index=False)
    else:
        summary["note"] = "Fewer than 10 matched site-months; correlation stats not computed (documented as a limitation)."
        print("\nFewer than 10 matched site-months -- skipping correlation computation; documenting as a limitation.")

    (OUT_DIR / "gome2_satellite_day_restricted_gpp_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("\nSummary written:", OUT_DIR / "gome2_satellite_day_restricted_gpp_summary.json")


if __name__ == "__main__":
    main()
