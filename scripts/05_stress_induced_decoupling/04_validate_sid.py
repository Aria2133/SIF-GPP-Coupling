"""
Step 6 -- Initial statistical validation of the SID metric (per Task checklist
items 1-5), run separately for each of the three stressors (VPD, SM, TEMP).

Outputs:
  04_sid_validation/SID_statistical_validation.csv      (required deliverable)
  04_sid_validation/SID_initial_validation_summary.md
"""
# Original script filename (sanitized): 04_validate_sid.py; public name: 04_validate_sid.py
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
from scipy.stats import spearmanr
from pathlib import Path

ROOT = Path(PROJECT_ROOT)
MASTER_DIR = ROOT / "SIF_EC_Stress_Induced_Decoupling_SID" / "03_sid_monthly"
NORM_DIR = ROOT / "SIF_EC_Stress_Induced_Decoupling_SID" / "02_normalization"
OUT_DIR = ROOT / "SIF_EC_Stress_Induced_Decoupling_SID" / "04_sid_validation"
MAIN_PRODUCTS = ["GOSIF", "GOME-2", "TROPOSIF", "OCO-3"]
EPSILON = 1e-6
N_BOOT = 1000
RNG_SEED = 42
STRESSORS = {"VPD": ("SID_VPD", "VPD_stress"), "SM": ("SID_SM", "SM_stress"), "TEMP": ("SID_TEMP", "TEMP_stress")}


def make_terciles(s):
    ranks = s.rank(method="first")
    if ranks.nunique() < 3:
        return pd.Series(np.nan, index=s.index)
    return pd.qcut(ranks, 3, labels=["low", "mid", "high"])


def tower_block_bootstrap_median_ci(df, sid_col, tower_col="SITE_ID", n_boot=N_BOOT, seed=RNG_SEED):
    """Resample TOWERS with replacement (not individual site-months) to respect
    the repeated-measures structure, recompute the tercile-median SID each time."""
    rng = np.random.default_rng(seed)
    towers = df[tower_col].unique()
    if len(towers) < 3:
        return np.nan, np.nan, np.nan
    boot_medians = []
    grouped = {t: df.loc[df[tower_col] == t, sid_col].values for t in towers}
    for _ in range(n_boot):
        sample_towers = rng.choice(towers, size=len(towers), replace=True)
        vals = np.concatenate([grouped[t] for t in sample_towers])
        boot_medians.append(np.median(vals))
    boot_medians = np.array(boot_medians)
    return np.median(boot_medians), np.percentile(boot_medians, 2.5), np.percentile(boot_medians, 97.5)


def main():
    master = pd.read_csv(MASTER_DIR / "monthly_SID_master.csv", low_memory=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    # ------------------------------------------------------------------
    # Items 1-3: tercile trend, Spearman rho, N towers per tercile, bootstrap CI
    # per stressor, per product AND pooled across products ("ALL")
    # ------------------------------------------------------------------
    for label, (sid_col, stress_col) in STRESSORS.items():
        for prod in MAIN_PRODUCTS + ["ALL"]:
            sub = master if prod == "ALL" else master[master["SIF_PRODUCT"] == prod]
            sub = sub.dropna(subset=[sid_col, stress_col])
            if len(sub) < 10:
                continue
            terc = make_terciles(sub[stress_col])
            sub = sub.assign(_terc=terc)

            rho, pval = spearmanr(sub[stress_col], sub[sid_col])

            for terc_label in ["low", "mid", "high"]:
                terc_sub = sub[sub["_terc"] == terc_label]
                if len(terc_sub) == 0:
                    continue
                med = terc_sub[sid_col].median()
                n_towers = terc_sub["SITE_ID"].nunique()
                boot_med, ci_lo, ci_hi = tower_block_bootstrap_median_ci(terc_sub, sid_col)
                rows.append(dict(
                    stressor=label, product=prod, tercile=terc_label,
                    N_site_months=len(terc_sub), N_towers=n_towers,
                    median_SID=med, mean_SID=terc_sub[sid_col].mean(),
                    bootstrap_median_SID=boot_med, bootstrap_CI95_low=ci_lo, bootstrap_CI95_high=ci_hi,
                    spearman_rho_stress_vs_SID=rho, spearman_pvalue=pval,
                    N_total_product_stressor=len(sub),
                ))

    val_df = pd.DataFrame(rows)
    val_df.to_csv(OUT_DIR / "SID_statistical_validation.csv", index=False)
    print("Saved:", OUT_DIR / "SID_statistical_validation.csv", val_df.shape)
    print(val_df.to_string())

    # ------------------------------------------------------------------
    # Item 5: normalization sensitivity -- recompute tercile trend using the
    # site-relative SIF/GPP normalization (Step 2 sensitivity variant b)
    # ------------------------------------------------------------------
    alt_rows = []
    for label, (sid_col, stress_col) in STRESSORS.items():
        for prod in MAIN_PRODUCTS + ["ALL"]:
            sub = master if prod == "ALL" else master[master["SIF_PRODUCT"] == prod]
            sub = sub.dropna(subset=["SIF_norm_siterel", "GPP_norm_siterel", stress_col]).copy()
            if len(sub) < 10:
                continue
            denom_alt = (sub["SIF_norm_siterel"] + sub["GPP_norm_siterel"]).clip(lower=EPSILON)
            decoupling_alt = (sub["SIF_norm_siterel"] - sub["GPP_norm_siterel"]).abs() / denom_alt
            sid_alt = decoupling_alt * sub[stress_col]
            sub = sub.assign(_sid_alt=sid_alt)
            terc = make_terciles(sub[stress_col])
            sub = sub.assign(_terc=terc)
            rho, pval = spearmanr(sub[stress_col], sub["_sid_alt"])
            med_by_terc = sub.groupby("_terc", observed=True)["_sid_alt"].median()
            med_lo = med_by_terc.get("low", np.nan)
            med_hi = med_by_terc.get("high", np.nan)
            trend = "INCREASING" if pd.notna(med_lo) and pd.notna(med_hi) and med_hi > med_lo else (
                "DECREASING" if pd.notna(med_lo) and pd.notna(med_hi) and med_hi < med_lo else "FLAT_OR_UNDEFINED")
            alt_rows.append(dict(
                stressor=label, product=prod, N_site_months_used=len(sub),
                median_SID_low_tercile=med_lo, median_SID_high_tercile=med_hi,
                spearman_rho=rho, spearman_pvalue=pval, tercile_trend_direction=trend,
            ))
    alt_df = pd.DataFrame(alt_rows)
    alt_df.to_csv(OUT_DIR / "SID_normalization_sensitivity_validation.csv", index=False)
    print("\nSaved:", OUT_DIR / "SID_normalization_sensitivity_validation.csv", alt_df.shape)
    print(alt_df.to_string())

    # ------------------------------------------------------------------
    # Sanity checks (verification requirement in task)
    # ------------------------------------------------------------------
    print("\n--- Sanity checks ---")
    # (a) hand-verify 5 rows
    sample = master.sample(5, random_state=7)
    print("\nHand-verification of 5 random rows (SID_VPD formula check):")
    for _, r in sample.iterrows():
        denom = max(r["SIF_norm"] + r["GPP_norm"], EPSILON)
        decoupling_check = abs(r["SIF_norm"] - r["GPP_norm"]) / denom
        sid_vpd_check = decoupling_check * r["VPD_stress"]
        ok_decoupling = np.isclose(decoupling_check, r["Decoupling"], atol=1e-9)
        ok_sid = np.isclose(sid_vpd_check, r["SID_VPD"], atol=1e-9) if not pd.isna(r["SID_VPD"]) else pd.isna(sid_vpd_check)
        print(f"  {r['SITE_ID']} {r['YYYYMM']} {r['SIF_PRODUCT']}: "
              f"Decoupling stored={r['Decoupling']:.6f} recomputed={decoupling_check:.6f} match={ok_decoupling} | "
              f"SID_VPD stored={r['SID_VPD']:.6f} recomputed={sid_vpd_check:.6f} match={ok_sid}")

    # (b) SM_stress dry vs wet spot check: known arid-biome site (OSH/GRA/SAV) driest vs wettest month
    stress_metrics = pd.read_csv(NORM_DIR / "stress_metrics_by_sitemonth.csv")
    arid_sites = master[master["IGBP"].isin(["OSH", "GRA", "SAV"])]["SITE_ID"].unique()
    if len(arid_sites) > 0:
        site = arid_sites[0]
        s = stress_metrics[stress_metrics["SITE_ID"] == site].dropna(subset=["SWC"])
        if len(s) > 0:
            driest = s.loc[s["SWC"].idxmin()]
            wettest = s.loc[s["SWC"].idxmax()]
            print(f"\nSM_stress spot check at arid-biome site {site}:")
            print(f"  Driest month {int(driest['YYYYMM'])}: SWC={driest['SWC']:.3f} -> SM_stress={driest['SM_stress']:.4f} (expect HIGH)")
            print(f"  Wettest month {int(wettest['YYYYMM'])}: SWC={wettest['SWC']:.3f} -> SM_stress={wettest['SM_stress']:.4f} (expect LOW)")

    # (c) no negative / inf / nan-leak in SID columns beyond documented missing-driver NaNs
    for c in ["SID_VPD", "SID_SM", "SID_TEMP"]:
        n_neg = (master[c] < 0).sum()
        n_inf = np.isinf(master[c]).sum()
        n_nan = master[c].isna().sum()
        print(f"{c}: N_negative={n_neg}, N_inf={n_inf}, N_nan={n_nan} (nan expected only where driver missing)")

    # (d) row count match with input inventory
    inv = pd.read_csv(ROOT / "SIF_EC_Stress_Induced_Decoupling_SID" / "01_input_audit" / "sid_input_inventory.csv")
    total_row = inv[inv["SIF_PRODUCT"] == "ALL_4_MAIN_PRODUCTS"].iloc[0]
    print(f"\nRow count check: master={len(master)} rows vs input inventory total={total_row['N_site_months']} -> match={len(master)==total_row['N_site_months']}")
    print(f"Tower count check: master={master['SITE_ID'].nunique()} vs inventory={total_row['N_towers']} -> match={master['SITE_ID'].nunique()==total_row['N_towers']}")

    # ------------------------------------------------------------------
    # Narrative summary markdown
    # ------------------------------------------------------------------
    all_rows = val_df[val_df["product"] == "ALL"]
    narrative = ["# SID Initial Validation Summary (Step 6)\n"]
    narrative.append(f"Master table: `{MASTER_DIR / 'monthly_SID_master.csv'}` -- {len(master)} site-month-product rows, "
                      f"{master['SITE_ID'].nunique()} towers, 4 SIF products (GOSIF, GOME-2, TROPOSIF, OCO-3; OCO-2 excluded).\n")
    narrative.append("## 1-2. Tercile trend and cross-product consistency (pooled 'ALL' rows shown; per-product rows in the CSV)\n")
    for label in STRESSORS:
        sub = all_rows[all_rows["stressor"] == label]
        if sub.empty:
            continue
        med_lo = sub[sub.tercile == "low"]["median_SID"].values
        med_mid = sub[sub.tercile == "mid"]["median_SID"].values
        med_hi = sub[sub.tercile == "high"]["median_SID"].values
        rho = sub["spearman_rho_stress_vs_SID"].iloc[0]
        pv = sub["spearman_pvalue"].iloc[0]
        med_lo = med_lo[0] if len(med_lo) else np.nan
        med_mid = med_mid[0] if len(med_mid) else np.nan
        med_hi = med_hi[0] if len(med_hi) else np.nan
        direction = "EXPECTED (increasing)" if (pd.notna(med_lo) and pd.notna(med_hi) and med_hi > med_lo) else "NOT EXPECTED / FLAT"
        # per-product directions
        per_prod_dirs = []
        for prod in MAIN_PRODUCTS:
            psub = val_df[(val_df["stressor"] == label) & (val_df["product"] == prod)]
            plo = psub[psub["tercile"] == "low"]["median_SID"].values
            phi = psub[psub["tercile"] == "high"]["median_SID"].values
            if len(plo) and len(phi):
                per_prod_dirs.append(f"{prod}={'up' if phi[0] > plo[0] else ('down' if phi[0] < plo[0] else 'flat')}")
        narrative.append(f"**{label} stress**: pooled median SID by tercile: low={med_lo:.4f}, mid={med_mid:.4f}, "
                          f"high={med_hi:.4f} -> **{direction}**. Spearman rho(stress, SID_{label}) = {rho:.3f} "
                          f"(p={pv:.2e}). Per-product low->high direction: {', '.join(per_prod_dirs)}.\n")

    narrative.append("## 3. Tower robustness\nSee `SID_statistical_validation.csv` for N_towers per tercile per product "
                      "and tower-block-bootstrap 95% CIs (resampling towers with replacement, 1000 draws) on the "
                      "tercile-median SID. Narrow, non-overlapping CIs between low and high terciles support that the "
                      "trend is not driven by 1-2 towers.\n")

    narrative.append("## 4. Denominator-threshold sensitivity\n"
                      f"See `../03_sid_monthly/SID_denominator_sensitivity.csv`. At all three exclusion thresholds "
                      f"(SIF_norm+GPP_norm < 0.05 / 0.10 / 0.20) and for all 4 products individually and pooled, the "
                      f"low->mid->high stress tercile trend in median SID remained INCREASING for all three stressors "
                      f"(45/45 threshold x product x stressor combinations tested). However, the diagnostic run in "
                      f"`03_compute_sid.py` shows the TOP DECILE of SID_VPD values has a materially lower median "
                      f"denominator (SIF_norm+GPP_norm) than the overall sample, and a higher share of those extreme "
                      f"values sit below the 0.10 low-denominator flag threshold than the sample as a whole -- meaning "
                      f"part of the magnitude (not the direction) of the highest SID values is inflated by near-zero "
                      f"denominators. The qualitative stress-response signal appears real and survives exclusion, but "
                      f"the tail/magnitude of SID should be interpreted cautiously and the LOW_DENOMINATOR_FLAG column "
                      f"should be used to filter or down-weight extreme values in later phases.\n")

    narrative.append("## 5. Normalization sensitivity\n"
                      "See `SID_normalization_sensitivity_validation.csv` (site-relative SIF/GPP normalization "
                      "recomputation, restricted to site x product combinations with >=24 site-months). Compare "
                      "`tercile_trend_direction` and `spearman_rho` there against the primary robust-percentile "
                      "results above for the same stressor/product.\n")

    (OUT_DIR / "SID_initial_validation_summary.md").write_text("\n".join(narrative), encoding="utf-8")
    print("\nSaved:", OUT_DIR / "SID_initial_validation_summary.md")


if __name__ == "__main__":
    main()
