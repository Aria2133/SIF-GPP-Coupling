"""
Independent re-verification of the earlier audit SIF-GPP statistical audit.

This is FRESH code (not a re-run of the earlier audit audit script) against the same
analysis-ready master table, written to independently check whether the
headline numbers in the earlier audit audit report are reproducible. Main-paper
products only: GOSIF, GOME-2, TROPOSIF, OCO-3. OCO-2 is dropped up front.
"""
from __future__ import annotations
# Original script filename (sanitized): independent_reverification.py; public name: 01_harmonic_deseasonalization_and_core_statistics.py
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

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(
    PROJECT_ROOT
)
MASTER_PATH = (
    PROJECT_ROOT
    / "SIF_EC_SIF_GPP_Relationship_Analysis"
    / "01_analysis_ready"
    / "SIF_GPP_Analysis_Ready_Master.csv"
)
OUT_DIR = PROJECT_ROOT / "SIF_EC_SIF_GPP_Final_Verification" / "02_independent_reverification"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRODUCTS = ["GOSIF", "GOME-2", "TROPOSIF", "OCO-3"]
RNG = np.random.default_rng(20260827)


def pearson_spearman_r2(x: np.ndarray, y: np.ndarray) -> dict:
    n = len(x)
    if n < 3 or np.std(x) == 0 or np.std(y) == 0:
        return dict(N=n, pearson_r=np.nan, spearman_rho=np.nan, r2=np.nan, slope=np.nan, intercept=np.nan)
    r, _ = stats.pearsonr(x, y)
    rho, _ = stats.spearmanr(x, y)
    slope, intercept, rval, _, _ = stats.linregress(x, y)
    return dict(N=n, pearson_r=r, spearman_rho=rho, r2=rval**2, slope=slope, intercept=intercept)


def load_master() -> pd.DataFrame:
    df = pd.read_csv(MASTER_PATH, low_memory=False)
    df = df[df["SIF_PRODUCT"].isin(PRODUCTS)].copy()
    df["SIF_VALUE"] = pd.to_numeric(df["SIF_VALUE"], errors="coerce")
    df["GPP_VALUE"] = pd.to_numeric(df["GPP_VALUE"], errors="coerce")
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
    df["MONTH"] = pd.to_numeric(df["MONTH"], errors="coerce")
    df = df.dropna(subset=["SIF_VALUE", "GPP_VALUE", "YEAR", "MONTH", "SITE_ID"])
    return df


def raw_reproduction(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for p in PRODUCTS:
        g = df[df["SIF_PRODUCT"] == p]
        s = pearson_spearman_r2(g["SIF_VALUE"].to_numpy(), g["GPP_VALUE"].to_numpy())
        rows.append({"product": p, "N_towers": g["SITE_ID"].nunique(), **s})
    return pd.DataFrame(rows)


def climatology_anomaly(df: pd.DataFrame, min_years: int | None = None) -> pd.DataFrame:
    """Method A: site x calendar-month climatology anomaly (independent re-implementation)."""
    rows = []
    for p in PRODUCTS:
        g = df[df["SIF_PRODUCT"] == p].copy()
        clim = g.groupby(["SITE_ID", "MONTH"]).agg(
            sif_clim=("SIF_VALUE", "mean"),
            gpp_clim=("GPP_VALUE", "mean"),
            n_years=("YEAR", "nunique"),
        )
        g = g.join(clim, on=["SITE_ID", "MONTH"])
        if min_years is not None:
            g = g[g["n_years"] >= min_years]
        g["SIF_ANOM"] = g["SIF_VALUE"] - g["sif_clim"]
        g["GPP_ANOM"] = g["GPP_VALUE"] - g["gpp_clim"]
        s = pearson_spearman_r2(g["SIF_ANOM"].to_numpy(), g["GPP_ANOM"].to_numpy())
        rows.append({"product": p, "min_years_filter": min_years or 1, "N_towers": g["SITE_ID"].nunique(), **s})
    return pd.DataFrame(rows)


def harmonic_anomaly(df: pd.DataFrame, min_n: int = 24, min_unique_months: int = 8) -> pd.DataFrame:
    """Method C: per-site first+second harmonic seasonal regression residuals."""
    rows = []
    resid_frames = []
    for p in PRODUCTS:
        g = df[df["SIF_PRODUCT"] == p].copy()
        recs = []
        for site, gs in g.groupby("SITE_ID"):
            if len(gs) < min_n or gs["MONTH"].nunique() < min_unique_months:
                continue
            t = 2 * np.pi * gs["MONTH"].to_numpy() / 12.0
            X = np.column_stack([np.ones(len(gs)), np.sin(t), np.cos(t), np.sin(2 * t), np.cos(2 * t)])
            for col, out_col in (("SIF_VALUE", "SIF_ANOM"), ("GPP_VALUE", "GPP_ANOM")):
                y = gs[col].to_numpy(dtype=float)
                beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                gs[out_col] = y - X @ beta
            recs.append(gs)
        if not recs:
            rows.append({"product": p, "N_towers": 0, "N": 0, "pearson_r": np.nan})
            continue
        gr = pd.concat(recs, ignore_index=True)
        resid_frames.append(gr.assign(SIF_PRODUCT=p))
        s = pearson_spearman_r2(gr["SIF_ANOM"].to_numpy(), gr["GPP_ANOM"].to_numpy())
        rows.append({"product": p, "N_towers": gr["SITE_ID"].nunique(), **s})
    return pd.DataFrame(rows), (pd.concat(resid_frames, ignore_index=True) if resid_frames else pd.DataFrame())


def minimum_temporal_support(df: pd.DataFrame, harmonic_resid: pd.DataFrame) -> pd.DataFrame:
    """Sensitivity: restrict to towers with >= K distinct calendar months of data."""
    rows = []
    for p in PRODUCTS:
        g = harmonic_resid[harmonic_resid["SIF_PRODUCT"] == p]
        months_per_site = g.groupby("SITE_ID")["MONTH"].nunique()
        for k in (6, 8, 10, 12):
            keep_sites = months_per_site[months_per_site >= k].index
            gk = g[g["SITE_ID"].isin(keep_sites)]
            s = pearson_spearman_r2(gk["SIF_ANOM"].to_numpy(), gk["GPP_ANOM"].to_numpy())
            rows.append({"product": p, "min_unique_months": k, "N_towers": gk["SITE_ID"].nunique(), **s})
    return pd.DataFrame(rows)


def oco3_observed_reconstructed(df: pd.DataFrame) -> pd.DataFrame:
    g = df[df["SIF_PRODUCT"] == "OCO-3"].copy()
    status_col = "OBSERVED_OR_RECONSTRUCTED"
    rows = []
    for label, sub in (
        ("combined", g),
        ("observed_only", g[g[status_col].astype(str).str.upper().str.contains("OBSERV", na=False)]),
        ("reconstructed_only", g[g[status_col].astype(str).str.upper().str.contains("RECON", na=False)]),
    ):
        s = pearson_spearman_r2(sub["SIF_VALUE"].to_numpy(), sub["GPP_VALUE"].to_numpy())
        rows.append({"population": label, "N_towers": sub["SITE_ID"].nunique(), **s})
    return pd.DataFrame(rows)


def tower_bootstrap_ci(sif: np.ndarray, gpp: np.ndarray, sites: np.ndarray, n_boot: int = 2000) -> tuple[float, float]:
    uniq = np.unique(sites)
    if len(uniq) < 3:
        return (np.nan, np.nan)
    boots = []
    for _ in range(n_boot):
        pick = RNG.choice(uniq, size=len(uniq), replace=True)
        mask = np.isin(sites, pick)
        x, y = sif[mask], gpp[mask]
        if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
            continue
        r, _ = stats.pearsonr(x, y)
        boots.append(r)
    if len(boots) < 100:
        return (np.nan, np.nan)
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def biome_reliability(df: pd.DataFrame, harmonic_resid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    biome_col = "IGBP" if "IGBP" in df.columns else "IGBP_GPP_TABLE"
    for p in PRODUCTS:
        g = harmonic_resid[harmonic_resid["SIF_PRODUCT"] == p]
        for biome, gb in g.groupby(biome_col):
            n_site_months = len(gb)
            n_towers = gb["SITE_ID"].nunique()
            s = pearson_spearman_r2(gb["SIF_ANOM"].to_numpy(), gb["GPP_ANOM"].to_numpy())
            ci_lo, ci_hi = tower_bootstrap_ci(
                gb["SIF_ANOM"].to_numpy(), gb["GPP_ANOM"].to_numpy(), gb["SITE_ID"].to_numpy()
            )
            reliable = (n_site_months >= 12) and (n_towers >= 3)
            rows.append(
                {
                    "product": p,
                    "biome": biome,
                    "N_site_months": n_site_months,
                    "N_towers": n_towers,
                    "pearson_r": s["pearson_r"],
                    "spearman_rho": s["spearman_rho"],
                    "boot_ci_low": ci_lo,
                    "boot_ci_high": ci_hi,
                    "reliable": reliable,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    df = load_master()
    print("Loaded master:", df.shape, "products:", df["SIF_PRODUCT"].value_counts().to_dict())

    raw = raw_reproduction(df)
    raw.to_csv(OUT_DIR / "raw_reproduction.csv", index=False)
    print("\n=== RAW ===\n", raw[["product", "N", "N_towers", "pearson_r", "spearman_rho", "r2"]])

    anomA = climatology_anomaly(df, min_years=None)
    anomA.to_csv(OUT_DIR / "current_climatology_anomaly.csv", index=False)
    print("\n=== CURRENT SITE-CALENDAR-MONTH ANOMALY (Method A) ===\n", anomA[["product", "N", "pearson_r", "r2"]])

    sens_rows = []
    for k in (3, 4, 5):
        a = climatology_anomaly(df, min_years=k)
        a["min_years_filter"] = k
        sens_rows.append(a)
    sens = pd.concat(sens_rows, ignore_index=True)
    sens.to_csv(OUT_DIR / "climatology_min_years_sensitivity.csv", index=False)
    print("\n=== N>=3/4/5 CLIMATOLOGY SENSITIVITY ===\n", sens[["product", "min_years_filter", "N", "pearson_r"]])

    harm, harm_resid = harmonic_anomaly(df)
    harm.to_csv(OUT_DIR / "harmonic_residual_anomaly.csv", index=False)
    harm_resid.to_csv(OUT_DIR / "harmonic_residuals_full.csv", index=False)
    print("\n=== HARMONIC RESIDUAL ANOMALY (Method C) ===\n", harm[["product", "N", "pearson_r", "r2"]])

    mts = minimum_temporal_support(df, harm_resid)
    mts.to_csv(OUT_DIR / "minimum_temporal_support_sensitivity.csv", index=False)
    print("\n=== MINIMUM TEMPORAL SUPPORT SENSITIVITY (on harmonic anomalies) ===\n", mts)

    oco3 = oco3_observed_reconstructed(df)
    oco3.to_csv(OUT_DIR / "oco3_observed_vs_reconstructed.csv", index=False)
    print("\n=== OCO-3 OBSERVED VS RECONSTRUCTED (raw) ===\n", oco3)

    biome = biome_reliability(df, harm_resid)
    biome.to_csv(OUT_DIR / "biome_reliability.csv", index=False)
    print("\n=== BIOME RELIABILITY (n reliable cells) ===", int(biome["reliable"].sum()), "/", len(biome))

    print("\nAll outputs written to", OUT_DIR)


if __name__ == "__main__":
    main()
