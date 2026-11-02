"""
Part I, Step 1 -- Input audit for the Advanced Coupling Analysis.

Joins:
  - Primary anomaly source: harmonic_residuals_full.csv (per-site harmonic
    seasonal-regression residuals; the FINAL accepted deseasonalization method,
    already computed -- not recomputed here). One row per SITE_ID x YYYYMM x
    SIF_PRODUCT for the 4 main-paper products (GOSIF, GOME-2, TROPOSIF, OCO-3).
    OCO-2 is already absent from this file (confirmed).
  - Raw comparison source: SIF_GPP_Analysis_Ready_Master.csv (SIF_VALUE, GPP_VALUE,
    pre-deseasonalization), filtered to the 4 main-paper products (OCO-2 dropped).
  - Driver source: Mechanistic_SIF_GPP_Monthly_With_Extended_Drivers.csv. Driver
    values (VPD, TAIR, SW_IN, SWC and their _CLIM/_ANOM variants) are IDENTICAL
    across SIF products for the same SITE_ID x YYYYMM (confirmed by direct check:
    nunique==1 for every site-month across product rows), so this table is
    collapsed to one row per SITE_ID x YYYYMM before joining -- avoids redundant
    per-product duplication of driver data.

Outputs:
  - 01_input_audit/advanced_analysis_input_inventory.csv
  - 01_input_audit/advanced_analysis_input_inventory.md

Author: project authors
Run date: 2026-09-01
"""
from __future__ import annotations
# Original script filename (sanitized): 01_build_input_inventory.py; public name: 01_build_input_inventory.py
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

HARMONIC_CSV = ROOT / "SIF_EC_SIF_GPP_Final_Verification" / "02_independent_reverification" / "harmonic_residuals_full.csv"
MASTER_CSV = ROOT / "SIF_EC_SIF_GPP_Relationship_Analysis" / "01_analysis_ready" / "SIF_GPP_Analysis_Ready_Master.csv"
DRIVERS_CSV = ROOT / "SIF_EC_Mechanistic_FootprintAware_SIF_GPP_Paper" / "08_mechanistic_controls" / "Mechanistic_SIF_GPP_Monthly_With_Extended_Drivers.csv"

OUT_DIR = ROOT / "SIF_EC_Advanced_Coupling_Analysis" / "01_input_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRODUCTS = ["GOSIF", "GOME-2", "TROPOSIF", "OCO-3"]  # main-paper products only; OCO-2 excluded everywhere

DRIVER_COLS = ["VPD", "TAIR", "SW_IN", "SWC"]
DRIVER_ANOM_COLS = [f"{c}_ANOM" for c in DRIVER_COLS]
DRIVER_CLIM_COLS = [f"{c}_CLIM" for c in DRIVER_COLS]


def load_driver_lookup() -> pd.DataFrame:
    """One row per SITE_ID x YYYYMM (driver values confirmed identical across
    SIF_PRODUCT rows for the same site-month -- collapsed here, not duplicated)."""
    d = pd.read_csv(DRIVERS_CSV, low_memory=False)
    keep = ["SITE_ID", "YYYYMM"] + DRIVER_COLS + DRIVER_CLIM_COLS + DRIVER_ANOM_COLS
    keep = [c for c in keep if c in d.columns]
    d = d[keep].drop_duplicates(subset=["SITE_ID", "YYYYMM"], keep="first")
    return d


def main():
    print("Loading harmonic-anomaly table (primary anomaly source) ...")
    h = pd.read_csv(HARMONIC_CSV, low_memory=False)
    h = h[h["SIF_PRODUCT"].isin(PRODUCTS)].copy()
    assert not (h["SIF_PRODUCT"] == "OCO-2").any(), "OCO-2 must be excluded"

    h_keep = [
        "SITE_ID", "YYYYMM", "YEAR", "MONTH", "LATITUDE", "LONGITUDE", "IGBP",
        "IGBP_GPP_TABLE", "SIF_PRODUCT", "SIF_VALUE", "GPP_VALUE",
        "OBSERVED_OR_RECONSTRUCTED", "SIF_ANOM", "GPP_ANOM",
    ]
    h_keep = [c for c in h_keep if c in h.columns]
    h = h[h_keep].rename(columns={"SIF_VALUE": "SIF_VALUE_HARMONIC_SRC", "GPP_VALUE": "GPP_VALUE_HARMONIC_SRC"})

    print("Loading raw master table (non-deseasonalized comparison source) ...")
    m = pd.read_csv(MASTER_CSV, low_memory=False)
    m = m[m["SIF_PRODUCT"].isin(PRODUCTS)].copy()
    assert not (m["SIF_PRODUCT"] == "OCO-2").any(), "OCO-2 must be excluded"
    m_keep = ["SITE_ID", "YYYYMM", "SIF_PRODUCT", "SIF_VALUE", "GPP_VALUE", "OBSERVED_OR_RECONSTRUCTED"]
    m_keep = [c for c in m_keep if c in m.columns]
    m = m[m_keep].rename(columns={
        "SIF_VALUE": "SIF_VALUE_RAW", "GPP_VALUE": "GPP_VALUE_RAW",
        "OBSERVED_OR_RECONSTRUCTED": "OBSERVED_OR_RECONSTRUCTED_RAWMASTER",
    })

    print("Loading + collapsing driver table ...")
    d = load_driver_lookup()

    print("Joining ...")
    out = h.merge(m, on=["SITE_ID", "YYYYMM", "SIF_PRODUCT"], how="left")
    out = out.merge(d, on=["SITE_ID", "YYYYMM"], how="left")

    # Provenance: prefer the harmonic-file's own OBSERVED_OR_RECONSTRUCTED (it is the
    # provenance flag attached to the exact SIF_VALUE/GPP_VALUE that fed the anomaly
    # computation); carry the raw-master's copy alongside for cross-check.
    out["PROVENANCE_MATCH"] = out["OBSERVED_OR_RECONSTRUCTED"] == out["OBSERVED_OR_RECONSTRUCTED_RAWMASTER"]

    out["HEMISPHERE"] = np.where(out["LATITUDE"] >= 0, "Northern", "Southern")

    csv_path = OUT_DIR / "advanced_analysis_input_inventory.csv"
    out.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}  shape={out.shape}")

    # ---------------- Companion report ----------------
    n_sitemonths = len(out)
    n_sitemonths_unique = out[["SITE_ID", "YYYYMM"]].drop_duplicates().shape[0]
    n_towers = out["SITE_ID"].nunique()
    yyyymm_min, yyyymm_max = int(out["YYYYMM"].min()), int(out["YYYYMM"].max())

    # per-site hemisphere assignment (mode of per-row hemisphere, should be constant per site)
    site_hemi = out.groupby("SITE_ID")["HEMISPHERE"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan)
    hemi_counts = site_hemi.value_counts()

    site_biome = out.groupby("SITE_ID")["IGBP"].agg(lambda s: s.dropna().mode().iloc[0] if len(s.dropna().mode()) else "Unknown")
    biome_counts = site_biome.fillna("Unknown").value_counts()

    completeness_vars = ["SIF_ANOM", "GPP_ANOM", "VPD", "SWC", "TAIR", "SW_IN",
                          "VPD_ANOM", "SWC_ANOM", "TAIR_ANOM", "SW_IN_ANOM"]
    completeness_vars = [c for c in completeness_vars if c in out.columns]

    comp_by_product = out.groupby("SIF_PRODUCT")[completeness_vars].apply(
        lambda g: g.notna().mean() * 100.0
    )

    obs_recon = out["OBSERVED_OR_RECONSTRUCTED"].value_counts(dropna=False)
    obs_recon_by_product = pd.crosstab(out["SIF_PRODUCT"], out["OBSERVED_OR_RECONSTRUCTED"])

    n_provenance_mismatch = int((~out["PROVENANCE_MATCH"].fillna(True)).sum())

    lines = []
    lines.append("# Advanced Coupling Analysis -- Input Inventory (Part I, Step 1)\n")
    lines.append("Main-paper products only: GOSIF, GOME-2, TROPOSIF, OCO-3. OCO-2 excluded "
                  "everywhere (asserted programmatically in the build script).\n")
    lines.append("## Join structure\n")
    lines.append(
        "- Primary anomaly source (SIF_ANOM, GPP_ANOM): `harmonic_residuals_full.csv` "
        "(per-site harmonic seasonal-regression residuals -- the final accepted deseasonalization "
        "method, reused as-is, not recomputed).\n"
        "- Raw (non-deseasonalized) comparison columns (SIF_VALUE_RAW, GPP_VALUE_RAW): "
        "`SIF_GPP_Analysis_Ready_Master.csv`, joined on SITE_ID+YYYYMM+SIF_PRODUCT.\n"
        "- Driver columns (VPD, TAIR, SW_IN, SWC and _CLIM/_ANOM variants): "
        "`Mechanistic_SIF_GPP_Monthly_With_Extended_Drivers.csv`, collapsed to one row per "
        "SITE_ID+YYYYMM before joining (driver values were confirmed identical across SIF "
        "products for the same site-month, so a per-product join would only duplicate rows, "
        "not add information).\n"
    )
    lines.append("## Headline counts\n")
    lines.append(f"- **N site-month x product rows**: {n_sitemonths}\n")
    lines.append(f"- **N unique site-months** (SITE_ID x YYYYMM, across all products): {n_sitemonths_unique}\n")
    lines.append(f"- **N towers**: {n_towers}\n")
    lines.append(f"- **Time span**: {yyyymm_min} to {yyyymm_max} (YYYYMM)\n")
    lines.append(f"- **Products present**: {', '.join(sorted(out['SIF_PRODUCT'].unique()))}\n")

    lines.append("\n## Rows per product\n")
    lines.append("| Product | N site-months |\n|---|---|\n")
    for p, n in out["SIF_PRODUCT"].value_counts().reindex(PRODUCTS).items():
        lines.append(f"| {p} | {n} |\n")

    lines.append("\n## Hemisphere counts (per tower, LATITUDE>=0 = Northern)\n")
    lines.append("| Hemisphere | N towers |\n|---|---|\n")
    for h_, n in hemi_counts.items():
        lines.append(f"| {h_} | {n} |\n")

    lines.append("\n## Biome (IGBP, modal per tower) counts\n")
    lines.append("| IGBP | N towers |\n|---|---|\n")
    for b, n in biome_counts.items():
        lines.append(f"| {b} | {n} |\n")

    lines.append("\n## OBSERVED vs RECONSTRUCTED provenance (from harmonic-residuals source; "
                  "OCO-3 is the product where RECONSTRUCTED rows are expected)\n")
    lines.append("| Status | N rows |\n|---|---|\n")
    for s, n in obs_recon.items():
        lines.append(f"| {s} | {n} |\n")
    lines.append("\nBy product:\n\n")
    lines.append(obs_recon_by_product.to_markdown() + "\n")
    lines.append(f"\nProvenance cross-check (harmonic-file OBSERVED_OR_RECONSTRUCTED vs raw-master's "
                  f"copy for the same row): {n_provenance_mismatch} mismatches out of {n_sitemonths} rows "
                  f"(mismatches expected only where a row failed to left-join to the raw master, e.g. NaN "
                  f"vs a real flag).\n")

    lines.append("\n## Variable completeness (% non-null), by product\n\n")
    lines.append(comp_by_product.round(1).to_markdown() + "\n")

    lines.append("\n## Notes\n")
    lines.append(
        "- SW_IN is incoming shortwave radiation (radiation proxy for PAR; true PAR/PPFD_IN "
        "available at only 36/73 towers -- see `PAR_availability_audit.md` in this folder, "
        "reused, not re-derived).\n"
        "- SIF_ANOM/GPP_ANOM are harmonic-seasonal-regression residuals (final accepted method). "
        "SIF_VALUE_RAW/GPP_VALUE_RAW are the pre-deseasonalization values, kept for the explicit "
        "raw-vs-anomaly comparison requested for Part I.\n"
        "- Driver _ANOM columns in this inventory are each site's own climatological-month anomaly "
        "(value minus that site's mean for that calendar month), as computed upstream in the "
        "mechanistic-controls table; these are the 'within-site' anomalies used as the primary "
        "collinearity/partial-correlation covariates in Steps 2-5.\n"
    )

    md_path = OUT_DIR / "advanced_analysis_input_inventory.md"
    md_path.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
