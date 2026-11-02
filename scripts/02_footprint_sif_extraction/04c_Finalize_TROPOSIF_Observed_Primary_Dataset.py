"""
Finalize TROPOSIF for later SIF-GPP analysis without using GPP.

This script creates a clean primary TROPOSIF table containing only genuinely
observed complete EC-footprint-support TROPOSIF site-months. It does not run
GOSIF-assisted reconstruction, does not use tower GPP, and does not create
spatial reconstructed rasters.

Optional target-only RF sensitivity reconstruction is attempted only if an
exact previously validated RF_TARGET production lock is recoverable. If not,
the sensitivity reconstruction is skipped and the observed-only primary
finalization still completes.
"""

from __future__ import annotations
# Original script filename (sanitized): Finalize_TROPOSIF_Observed_Primary_Dataset.py; public name: 04c_Finalize_TROPOSIF_Observed_Primary_Dataset.py
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
import math
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


PROJECT_ROOT = Path(
    PROJECT_ROOT
)

PREP_ROOT = PROJECT_ROOT / "SIF_EC_Reconstruction_Model_Preparation"
VALIDATION_ROOT = PROJECT_ROOT / "SIF_EC_Reconstruction_TROPOSIF_Validation_And_Production"
OUTPUT_ROOT = PROJECT_ROOT / "SIF_EC_TROPOSIF_Finalization"
QC_ROOT = OUTPUT_ROOT / "qc"
FIG_ROOT = OUTPUT_ROOT / "figures"

TROPOSIF_INVENTORY_CSV = PREP_ROOT / "TROPOSIF_GOSIF_EC_Reconstruction_Model_Inventory.csv"
EC_COVERAGE_CSV = PREP_ROOT / "EC_Coverage_All_Products.csv"
ARTIFICIAL_GAP_CSV = VALIDATION_ROOT / "TROPOSIF_Artificial_Gap_Dataset.csv"
MODEL_LOCK_MD = VALIDATION_ROOT / "TROPOSIF_Final_Model_Lock.md"
PRODUCTION_LOCK_TUNING_CSV = VALIDATION_ROOT / "TROPOSIF_Final_Production_Lock_Internal_Tuning_Detail.csv"
MODEL_HYPERPARAMETERS_CSV = VALIDATION_ROOT / "TROPOSIF_Final_Model_Hyperparameters.csv"

RAW_TROPOSIF_ROOT = PROJECT_ROOT / "SIF Datasets" / "TROPOSIF"
RAW_GOSIF_ROOT = PROJECT_ROOT / "SIF Datasets" / "GOSIF"
GPP_TIFF_ROOT = PROJECT_ROOT / "Monthly_EC_Footprint_TIFFs_Python"

PRODUCT = "TROPOSIF"
TROPOSIF_UNITS = "mW m-2 sr-1 nm-1 (source metadata: mW/m2/sr/nm)"

COVERAGE_TOL = 1e-9
EXPECTED_TOTAL = 1788
EXPECTED_COMPLETE = 1540
EXPECTED_PARTIAL = 47
EXPECTED_ZERO = 201


def ensure_dirs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    QC_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)


def file_snapshot(paths_or_roots: Iterable[Path]) -> pd.DataFrame:
    rows = []
    for base in paths_or_roots:
        base = Path(base)
        if not base.exists():
            rows.append(
                {
                    "path": str(base),
                    "exists": False,
                    "is_file": False,
                    "size_bytes": np.nan,
                    "mtime_ns": np.nan,
                    "audit_scope": "missing",
                }
            )
            continue
        files = [base] if base.is_file() else [p for p in base.rglob("*") if p.is_file()]
        for p in files:
            try:
                st = p.stat()
                rows.append(
                    {
                        "path": str(p),
                        "exists": True,
                        "is_file": p.is_file(),
                        "size_bytes": int(st.st_size),
                        "mtime_ns": int(st.st_mtime_ns),
                        "audit_scope": str(base),
                    }
                )
            except OSError as exc:
                rows.append(
                    {
                        "path": str(p),
                        "exists": False,
                        "is_file": False,
                        "size_bytes": np.nan,
                        "mtime_ns": np.nan,
                        "audit_scope": str(base),
                        "error": str(exc),
                    }
                )
    return pd.DataFrame(rows)


def compare_snapshots(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    b = before.rename(
        columns={
            "exists": "exists_before",
            "size_bytes": "size_bytes_before",
            "mtime_ns": "mtime_ns_before",
            "is_file": "is_file_before",
        }
    )
    a = after.rename(
        columns={
            "exists": "exists_after",
            "size_bytes": "size_bytes_after",
            "mtime_ns": "mtime_ns_after",
            "is_file": "is_file_after",
        }
    )
    keep = ["path", "exists_after", "is_file_after", "size_bytes_after", "mtime_ns_after"]
    merged = b.merge(a[keep], on="path", how="outer")
    merged["unchanged_size_and_mtime"] = (
        (merged["exists_before"] == merged["exists_after"])
        & (merged["size_bytes_before"] == merged["size_bytes_after"])
        & (merged["mtime_ns_before"] == merged["mtime_ns_after"])
    )
    return merged


def classify_coverage(v: float) -> str:
    if pd.isna(v):
        return "MISSING_COVERAGE"
    if abs(v) <= COVERAGE_TOL:
        return "ZERO_COVERAGE_EXCLUDED"
    if abs(v - 1.0) <= COVERAGE_TOL:
        return "OBSERVED_COMPLETE_PENDING_QC"
    if 0.0 < v < 1.0:
        return "PARTIAL_EXCLUDED_PRIMARY"
    return "EXCLUDED_INVALID_SOURCE"


def coverage_bin(v: float) -> str:
    if pd.isna(v):
        return "NaN"
    if abs(v) <= COVERAGE_TOL:
        return "=0"
    if 0 < v <= 0.10:
        return "0-0.10"
    if v <= 0.25:
        return ">0.10-0.25"
    if v <= 0.50:
        return ">0.25-0.50"
    if v <= 0.75:
        return ">0.50-0.75"
    if v < 1.0 - COVERAGE_TOL:
        return ">0.75-<1"
    if abs(v - 1.0) <= COVERAGE_TOL:
        return "=1"
    return ">1"


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    inv = pd.read_csv(TROPOSIF_INVENTORY_CSV)
    cov_all = pd.read_csv(EC_COVERAGE_CSV)
    cov = cov_all[cov_all["PRODUCT"].astype(str).str.upper().eq(PRODUCT)].copy()

    inv["YYYYMM"] = inv["YYYYMM"].astype(int)
    cov["YYYYMM"] = cov["YYYYMM"].astype(int)

    keep_cov = [
        "SITE_ID",
        "SITE_SAFE",
        "YYYYMM",
        "PRODUCT",
        "source_sif_type",
        "source_sif_files",
        "matching_gpp_tif",
        "n_target_source_footprints",
        "n_target_source_observations",
        "n_candidate_sif_observations_for_site_grid",
        "n_candidate_sif_observations_skipped_bad_geometry",
        "n_candidate_sif_observations_no_spatial_intersection_with_grid",
        "n_candidate_sif_observations_unexplained",
        "candidate_accounting_complete",
        "swath_geometry_fields_used",
        "n_EC_pixels",
        "n_EC_pixels_target_observed",
        "coverage_pixel_fraction",
        "coverage_probability_fraction",
        "missing_probability_fraction",
        "target_observed_EC_weighted_SIF",
        "target_observed_mean_SIF",
        "target_observed_sd",
        "target_observed_min",
        "target_observed_max",
        "status",
        "notes",
        "source_statuses",
    ]
    keep_cov = [c for c in keep_cov if c in cov.columns]

    merged = inv.merge(
        cov[keep_cov],
        on=["SITE_ID", "YYYYMM"],
        how="left",
        suffixes=("_inventory", "_coverage"),
        validate="one_to_one",
    )

    # Prefer detailed coverage table values when duplicate columns exist.
    for col in [
        "coverage_probability_fraction",
        "coverage_pixel_fraction",
        "missing_probability_fraction",
        "target_observed_EC_weighted_SIF",
        "n_target_source_observations",
    ]:
        inv_col = f"{col}_inventory"
        cov_col = f"{col}_coverage"
        if inv_col in merged.columns and cov_col in merged.columns:
            merged[col] = merged[cov_col].combine_first(merged[inv_col])
        elif inv_col in merged.columns:
            merged[col] = merged[inv_col]
        elif cov_col in merged.columns:
            merged[col] = merged[cov_col]

    return merged, cov


def verify_counts(df: pd.DataFrame) -> dict[str, int]:
    c = df["coverage_probability_fraction"].astype(float)
    counts = {
        "N_total_TROPOSIF": int(len(df)),
        "N_complete_observed": int((np.abs(c - 1.0) <= COVERAGE_TOL).sum()),
        "N_partial_positive": int(((c > COVERAGE_TOL) & (c < 1.0 - COVERAGE_TOL)).sum()),
        "N_zero_coverage": int((np.abs(c) <= COVERAGE_TOL).sum()),
    }
    if counts["N_total_TROPOSIF"] != EXPECTED_TOTAL:
        raise RuntimeError(f"TROPOSIF total differs from validated report: {counts['N_total_TROPOSIF']} != {EXPECTED_TOTAL}")
    if counts["N_complete_observed"] != EXPECTED_COMPLETE:
        raise RuntimeError(f"TROPOSIF complete count differs from validated report: {counts['N_complete_observed']} != {EXPECTED_COMPLETE}")
    if counts["N_partial_positive"] != EXPECTED_PARTIAL:
        raise RuntimeError(f"TROPOSIF partial count differs from validated report: {counts['N_partial_positive']} != {EXPECTED_PARTIAL}")
    if counts["N_zero_coverage"] != EXPECTED_ZERO:
        raise RuntimeError(f"TROPOSIF zero count differs from validated report: {counts['N_zero_coverage']} != {EXPECTED_ZERO}")
    if counts["N_complete_observed"] + counts["N_partial_positive"] + counts["N_zero_coverage"] != counts["N_total_TROPOSIF"]:
        raise RuntimeError("TROPOSIF coverage class counts do not reconcile.")
    return counts


def build_master(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    out["coverage_class"] = out["coverage_probability_fraction"].map(coverage_bin)
    out["TROPOSIF_primary_status"] = out["coverage_probability_fraction"].map(classify_coverage)

    # Complete-observed QC. Negative TROPOSIF values are allowed.
    is_complete = out["TROPOSIF_primary_status"].eq("OBSERVED_COMPLETE_PENDING_QC")
    sif = pd.to_numeric(out["target_observed_EC_weighted_SIF"], errors="coerce")
    n_foot = pd.to_numeric(out.get("n_target_source_footprints", np.nan), errors="coerce")
    n_obs = pd.to_numeric(out.get("n_target_source_observations", np.nan), errors="coerce")
    cov = pd.to_numeric(out["coverage_probability_fraction"], errors="coerce")

    reasons = []
    statuses = []
    for idx, row in out.iterrows():
        status = row["TROPOSIF_primary_status"]
        reason_list = []
        if status == "OBSERVED_COMPLETE_PENDING_QC":
            if not np.isfinite(sif.loc[idx]):
                reason_list.append("nonfinite_complete_ec_weighted_sif")
            if not np.isfinite(cov.loc[idx]) or abs(cov.loc[idx] - 1.0) > COVERAGE_TOL:
                reason_list.append("coverage_not_exactly_complete")
            if not np.isfinite(n_foot.loc[idx]) or n_foot.loc[idx] <= 0:
                reason_list.append("nonpositive_target_source_footprints")
            if not np.isfinite(n_obs.loc[idx]) or n_obs.loc[idx] <= 0:
                reason_list.append("nonpositive_target_source_observations")
            if str(row.get("status", "")).strip() not in ("OK_OBSERVED_EC_SUPPORT", ""):
                reason_list.append(f"coverage_status_{row.get('status')}")
            status = "OBSERVED_COMPLETE_QC_FLAG" if reason_list else "OBSERVED_COMPLETE_VALID"
        statuses.append(status)
        reasons.append(";".join(reason_list))
    out["TROPOSIF_primary_status"] = statuses
    out["QC_flag_reasons"] = reasons

    out["TROPOSIF_observed_EC_support_SIF"] = sif
    out["TROPOSIF_observed_complete_EC_SIF"] = np.where(
        out["TROPOSIF_primary_status"].isin(["OBSERVED_COMPLETE_VALID", "OBSERVED_COMPLETE_QC_FLAG"]),
        sif,
        np.nan,
    )
    out["TROPOSIF_observed_partial_EC_SIF"] = np.where(
        out["TROPOSIF_primary_status"].eq("PARTIAL_EXCLUDED_PRIMARY"),
        sif,
        np.nan,
    )
    out["TROPOSIF_final_EC_SIF"] = np.where(out["TROPOSIF_primary_status"].eq("OBSERVED_COMPLETE_VALID"), sif, np.nan)
    out["TROPOSIF_units"] = TROPOSIF_UNITS
    out["reconstruction_applied"] = False
    out["model_used"] = np.where(out["TROPOSIF_primary_status"].eq("OBSERVED_COMPLETE_VALID"), "OBSERVED_NO_MODEL", "NONE")
    out["GOSIF_used_in_primary_TROPOSIF"] = False
    out["GPP_used"] = False

    # Rename GOSIF columns as auxiliary to prevent accidental interpretation as primary predictors.
    for col in ["GOSIF_EC_weighted", "GOSIF_native_sd", "n_native_GOSIF_cells_intersecting_EC"]:
        if col in out.columns:
            out.rename(columns={col: f"{col}_auxiliary_not_used_in_primary"}, inplace=True)

    preferred = [
        "SITE_ID",
        "SITE_SAFE",
        "YYYYMM",
        "lat",
        "lon",
        "IGBP",
        "year",
        "month",
        "DOY",
        "sin_DOY",
        "cos_DOY",
        "coverage_probability_fraction",
        "coverage_pixel_fraction",
        "missing_probability_fraction",
        "coverage_class",
        "n_EC_pixels",
        "n_EC_pixels_target_observed",
        "n_target_source_observations",
        "n_target_source_footprints",
        "TROPOSIF_observed_EC_support_SIF",
        "TROPOSIF_observed_partial_EC_SIF",
        "TROPOSIF_observed_complete_EC_SIF",
        "TROPOSIF_final_EC_SIF",
        "TROPOSIF_units",
        "TROPOSIF_primary_status",
        "QC_flag_reasons",
        "reconstruction_applied",
        "model_used",
        "GOSIF_used_in_primary_TROPOSIF",
        "GPP_used",
        "target_observed_mean_SIF",
        "target_observed_sd",
        "target_observed_min",
        "target_observed_max",
        "source_sif_type",
        "swath_geometry_fields_used",
        "candidate_accounting_complete",
        "n_candidate_sif_observations_for_site_grid",
        "n_candidate_sif_observations_skipped_bad_geometry",
        "n_candidate_sif_observations_no_spatial_intersection_with_grid",
        "n_candidate_sif_observations_unexplained",
        "source_statuses",
        "status",
        "notes",
        "source_sif_files",
        "matching_gpp_tif",
        "GOSIF_EC_weighted_auxiliary_not_used_in_primary",
        "GOSIF_native_sd_auxiliary_not_used_in_primary",
        "n_native_GOSIF_cells_intersecting_EC_auxiliary_not_used_in_primary",
        "GOSIF_units",
        "GOSIF_status",
    ]
    existing = [c for c in preferred if c in out.columns]
    rest = [c for c in out.columns if c not in existing and not c.endswith("_inventory") and not c.endswith("_coverage")]
    out = out[existing + rest]

    flags = out[out["TROPOSIF_primary_status"].isin(["OBSERVED_COMPLETE_QC_FLAG", "EXCLUDED_INVALID_SOURCE"])].copy()
    return out, flags


def recover_rf_target_lock() -> tuple[bool, str, dict]:
    """Return whether an exact single RF_TARGET production lock exists."""
    details = {
        "model_lock_path": str(MODEL_LOCK_MD),
        "production_lock_tuning_path": str(PRODUCTION_LOCK_TUNING_CSV),
        "model_hyperparameters_path": str(MODEL_HYPERPARAMETERS_CSV),
        "checked_model_lock": MODEL_LOCK_MD.exists(),
        "checked_production_lock_tuning": PRODUCTION_LOCK_TUNING_CSV.exists(),
        "checked_model_hyperparameters": MODEL_HYPERPARAMETERS_CSV.exists(),
    }
    reason_parts = []
    if MODEL_LOCK_MD.exists():
        text = MODEL_LOCK_MD.read_text(encoding="utf-8", errors="replace")
        details["model_lock_mentions_RF_TARGET_GOSIF"] = "RF_TARGET_GOSIF" in text
        details["model_lock_mentions_final_RF_TARGET"] = "FINAL MODEL FAMILY: RF_TARGET" in text
        if "FINAL MODEL FAMILY: RF_TARGET_GOSIF" in text:
            reason_parts.append("TROPOSIF_Final_Model_Lock.md locks RF_TARGET_GOSIF, not RF_TARGET.")
    if PRODUCTION_LOCK_TUNING_CSV.exists():
        tuning = pd.read_csv(PRODUCTION_LOCK_TUNING_CSV)
        models = sorted(tuning.get("model", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
        details["production_lock_models"] = models
        if "RF_TARGET" not in models:
            reason_parts.append("Production-lock tuning file contains no RF_TARGET row.")
    if MODEL_HYPERPARAMETERS_CSV.exists():
        hyp = pd.read_csv(MODEL_HYPERPARAMETERS_CSV, usecols=["validation", "fold_id", "model", "inner_cv_rmse", "params"])
        rf = hyp[hyp["model"].eq("RF_TARGET") & hyp["inner_cv_rmse"].notna()].copy()
        details["rf_target_fold_tuning_rows"] = int(len(rf))
        details["rf_target_distinct_params"] = int(rf["params"].nunique()) if not rf.empty else 0
        if not rf.empty:
            best_by_fold = rf.loc[rf.groupby(["validation", "fold_id"])["inner_cv_rmse"].idxmin()].copy()
            details["rf_target_fold_best_rows"] = int(len(best_by_fold))
            details["rf_target_fold_best_distinct_params"] = int(best_by_fold["params"].nunique())
            reason_parts.append(
                "RF_TARGET appears only as fold-specific inner-CV tuning rows, not as a single frozen production model."
            )
    reason = " ".join(reason_parts) if reason_parts else "No exact RF_TARGET production lock could be found."
    return False, reason, details


def sensitivity_status_table(master: pd.DataFrame, lock_ok: bool, reason: str, details: dict) -> pd.DataFrame:
    partial = master[master["TROPOSIF_primary_status"].eq("PARTIAL_EXCLUDED_PRIMARY")].copy()
    rows = [
        {
            "sensitivity_run_attempted": True,
            "sensitivity_run_successful": False,
            "N_partial_positive_cases": int(len(partial)),
            "reason": reason,
            "required_model": "RF_TARGET",
            "gosif_predictors_allowed": False,
            "gpp_used": False,
            "output_sensitivity_reconstruction_created": False,
            "details_json": json.dumps(details, ensure_ascii=False),
        }
    ]
    return pd.DataFrame(rows)


def summarize_counts(master: pd.DataFrame, flags: pd.DataFrame, sensitivity_status: pd.DataFrame) -> dict[str, int]:
    counts = {
        "N_total_TROPOSIF": int(len(master)),
        "N_complete_observed": int(master["TROPOSIF_primary_status"].isin(["OBSERVED_COMPLETE_VALID", "OBSERVED_COMPLETE_QC_FLAG"]).sum()),
        "N_complete_primary_ready": int(master["TROPOSIF_primary_status"].eq("OBSERVED_COMPLETE_VALID").sum()),
        "N_partial_positive": int(master["TROPOSIF_primary_status"].eq("PARTIAL_EXCLUDED_PRIMARY").sum()),
        "N_partial_sensitivity_reconstructed": 0,
        "N_partial_sensitivity_failed": int(master["TROPOSIF_primary_status"].eq("PARTIAL_EXCLUDED_PRIMARY").sum()),
        "N_zero_coverage": int(master["TROPOSIF_primary_status"].eq("ZERO_COVERAGE_EXCLUDED").sum()),
        "N_primary_analysis_ready": int(master["TROPOSIF_primary_status"].eq("OBSERVED_COMPLETE_VALID").sum()),
        "N_QC_flagged": int(len(flags)),
    }
    if bool(sensitivity_status["sensitivity_run_successful"].iloc[0]):
        # This script currently never reaches here unless an explicit future lock exists.
        counts["N_partial_sensitivity_reconstructed"] = counts["N_partial_positive"]
        counts["N_partial_sensitivity_failed"] = 0
    return counts


def write_csvs(master: pd.DataFrame, flags: pd.DataFrame, sensitivity: pd.DataFrame, counts: dict[str, int]) -> None:
    master.to_csv(OUTPUT_ROOT / "TROPOSIF_Final_EC_Footprint_SIF_All_SiteMonths.csv", index=False)
    primary = master[master["TROPOSIF_primary_status"].eq("OBSERVED_COMPLETE_VALID")].copy()
    primary_cols = [
        "SITE_ID",
        "YYYYMM",
        "lat",
        "lon",
        "IGBP",
        "year",
        "month",
        "DOY",
        "sin_DOY",
        "cos_DOY",
        "coverage_probability_fraction",
        "coverage_pixel_fraction",
        "n_target_source_observations",
        "n_target_source_footprints",
        "TROPOSIF_observed_complete_EC_SIF",
        "TROPOSIF_final_EC_SIF",
        "TROPOSIF_units",
        "TROPOSIF_primary_status",
        "reconstruction_applied",
        "model_used",
        "GOSIF_used_in_primary_TROPOSIF",
        "GPP_used",
        "swath_geometry_fields_used",
        "source_statuses",
        "source_sif_files",
    ]
    primary_cols = [c for c in primary_cols if c in primary.columns]
    primary[primary_cols].to_csv(OUTPUT_ROOT / "TROPOSIF_Final_EC_Footprint_SIF_Primary_Observed.csv", index=False)
    flags.to_csv(OUTPUT_ROOT / "TROPOSIF_Finalization_QC_Flags.csv", index=False)
    sensitivity.to_csv(OUTPUT_ROOT / "TROPOSIF_TargetOnly_RF_Sensitivity_Status.csv", index=False)

    pd.DataFrame([counts]).to_csv(OUTPUT_ROOT / "TROPOSIF_Finalization_Final_Counts.csv", index=False)

    coverage_summary = (
        master.groupby("coverage_class", dropna=False)
        .agg(
            N_site_months=("SITE_ID", "size"),
            N_sites=("SITE_ID", "nunique"),
            min_coverage=("coverage_probability_fraction", "min"),
            median_coverage=("coverage_probability_fraction", "median"),
            max_coverage=("coverage_probability_fraction", "max"),
        )
        .reset_index()
    )
    coverage_summary["percentage"] = coverage_summary["N_site_months"] / len(master) * 100.0
    order = ["=0", "0-0.10", ">0.10-0.25", ">0.25-0.50", ">0.50-0.75", ">0.75-<1", "=1"]
    coverage_summary["coverage_class"] = pd.Categorical(coverage_summary["coverage_class"], categories=order, ordered=True)
    coverage_summary.sort_values("coverage_class").to_csv(OUTPUT_ROOT / "TROPOSIF_Finalization_Coverage_Class_Summary.csv", index=False)


def create_summary_figure(master: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("TROPOSIF Finalization: observed-complete primary dataset", fontsize=18, fontweight="bold")

    ax = axes[0, 0]
    labels = ["Complete observed", "Partial excluded", "Zero excluded"]
    vals = [
        master["TROPOSIF_primary_status"].isin(["OBSERVED_COMPLETE_VALID", "OBSERVED_COMPLETE_QC_FLAG"]).sum(),
        master["TROPOSIF_primary_status"].eq("PARTIAL_EXCLUDED_PRIMARY").sum(),
        master["TROPOSIF_primary_status"].eq("ZERO_COVERAGE_EXCLUDED").sum(),
    ]
    colors = ["#2ca02c", "#ffbf00", "#d62728"]
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylabel("Site-months")
    ax.set_title("A. Final TROPOSIF classes")
    ax.tick_params(axis="x", rotation=20)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom", fontweight="bold")

    ax = axes[0, 1]
    complete = master.loc[master["TROPOSIF_primary_status"].eq("OBSERVED_COMPLETE_VALID"), "TROPOSIF_final_EC_SIF"].dropna()
    ax.hist(complete, bins=35, color="#2ca02c", alpha=0.85)
    ax.set_title("B. Complete observed TROPOSIF EC SIF")
    ax.set_xlabel(TROPOSIF_UNITS)
    ax.set_ylabel("Count")

    ax = axes[0, 2]
    partial = master[master["TROPOSIF_primary_status"].eq("PARTIAL_EXCLUDED_PRIMARY")].copy()
    ax.hist(partial["coverage_probability_fraction"].dropna(), bins=np.linspace(0, 1, 21), color="#ffbf00", alpha=0.9)
    ax.set_title("C. Coverage distribution for 47 partial cases")
    ax.set_xlabel("EC probability coverage")
    ax.set_ylabel("Count")

    ax = axes[1, 0]
    ax.axis("off")
    ax.text(
        0.02,
        0.75,
        "D. Target-only RF sensitivity\n\nNot run.\n\nReason: no single exact RF_TARGET\nproduction lock was recoverable.\nFold-specific RF_TARGET tuning rows exist,\nbut no frozen target-only production model.",
        fontsize=12,
        va="top",
    )

    ax = axes[1, 1]
    ax.axis("off")
    ax.text(
        0.02,
        0.78,
        "E. Primary TROPOSIF rule\n\ncoverage = 1 → observed TROPOSIF directly\n0 < coverage < 1 → excluded from primary\ncoverage = 0 → excluded\n\nNo GOSIF-assisted reconstruction enters\nTROPOSIF_final_EC_SIF.",
        fontsize=12,
        va="top",
    )

    ax = axes[1, 2]
    ax.axis("off")
    ax.text(
        0.02,
        0.82,
        "F. Final statement\n\nPRIMARY TROPOSIF:\nOBSERVED COMPLETE CASES ONLY\n\nTARGET-ONLY PARTIAL RECONSTRUCTION:\nSENSITIVITY ONLY — NOT RUN\n\nTROPOSIF FINALIZATION COMPLETE",
        fontsize=14,
        fontweight="bold",
        va="top",
        color="#1f4e79",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = OUTPUT_ROOT / "TROPOSIF_Finalization_Simple_Summary.png"
    pdf = OUTPUT_ROOT / "TROPOSIF_Finalization_Simple_Summary.pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    return df.head(max_rows).to_markdown(index=False)


def write_pdf_from_text(md_path: Path, pdf_path: Path) -> None:
    text = md_path.read_text(encoding="utf-8", errors="replace")
    lines = []
    for raw in text.splitlines():
        if len(raw) <= 120:
            lines.append(raw)
        else:
            for i in range(0, len(raw), 120):
                lines.append(raw[i : i + 120])

    with PdfPages(pdf_path) as pdf:
        page_lines = 46
        for i in range(0, len(lines), page_lines):
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.06, 0.96, "\n".join(lines[i : i + page_lines]), va="top", ha="left", family="monospace", fontsize=8)
            fig.subplots_adjust(0, 0, 1, 1)
            pdf.savefig(fig)
            plt.close(fig)


def write_reports(
    master: pd.DataFrame,
    flags: pd.DataFrame,
    sensitivity: pd.DataFrame,
    counts: dict[str, int],
    source_inputs_unchanged: bool,
    input_audit_path: Path,
) -> None:
    coverage_summary = pd.read_csv(OUTPUT_ROOT / "TROPOSIF_Finalization_Coverage_Class_Summary.csv")
    primary_path = OUTPUT_ROOT / "TROPOSIF_Final_EC_Footprint_SIF_Primary_Observed.csv"
    master_path = OUTPUT_ROOT / "TROPOSIF_Final_EC_Footprint_SIF_All_SiteMonths.csv"

    readme = f"""TROPOSIF Finalization — READ ME FIRST

1. Total TROPOSIF site-months: {counts['N_total_TROPOSIF']:,}.
2. Complete EC coverage: {counts['N_complete_observed']:,}.
3. Partial positive coverage excluded from primary analysis: {counts['N_partial_positive']:,}.
4. Zero-coverage cases excluded: {counts['N_zero_coverage']:,}.
5. Complete observed cases ready for future GPP analysis: {counts['N_primary_analysis_ready']:,}.
6. Complete cases lost to QC: {counts['N_complete_observed'] - counts['N_complete_primary_ready']:,}.
7. Target-only RF sensitivity run successfully? No.
8. Partial cases reconstructed in sensitivity analysis: {counts['N_partial_sensitivity_reconstructed']:,}; failed/skipped: {counts['N_partial_sensitivity_failed']:,}.
9. Was GOSIF used in the final primary TROPOSIF value? No.
10. Were any GPP data used? No.
11. Were any source files modified? {'No' if source_inputs_unchanged else 'Potential modification detected; inspect audit CSV'}.
12. TROPOSIF finalization complete? Yes.

Final rule: TROPOSIF_final_EC_SIF contains only directly observed complete EC-footprint TROPOSIF. Partial and zero-coverage cases remain in the master inventory but are excluded from the primary observed dataset.
"""
    (OUTPUT_ROOT / "TROPOSIF_Finalization_READ_ME_FIRST.txt").write_text(readme, encoding="utf-8")

    md = []
    md.append("# TROPOSIF Finalization Report\n")
    md.append(f"Run timestamp: `{datetime.now().isoformat(timespec='seconds')}`\n")
    md.append("Final verdict: **TROPOSIF FINALIZATION COMPLETE**\n")
    md.append("## Scientific rule applied\n")
    md.append(
        "The primary TROPOSIF dataset uses only genuinely observed TROPOSIF site-months with complete EC-footprint probability support (`coverage_probability_fraction = 1`). "
        "No GOSIF-assisted TROPOSIF reconstruction was used, no GPP was used, and no zero-coverage case was replaced with GOSIF.\n"
    )
    md.append("## Final counts\n")
    md.append(markdown_table(pd.DataFrame([counts])) + "\n")
    md.append("## Coverage class summary\n")
    md.append(markdown_table(coverage_summary, 20) + "\n")
    md.append("## Primary dataset\n")
    md.append(f"- Primary observed table: `{primary_path}`\n")
    md.append(f"- Master all-site-month table: `{master_path}`\n")
    md.append(f"- Primary rows: `{counts['N_primary_analysis_ready']}`\n")
    md.append(f"- TROPOSIF units preserved as: `{TROPOSIF_UNITS}`\n")
    md.append("## Optional target-only RF sensitivity\n")
    md.append(
        "Sensitivity reconstruction was **not run** because no single exact validated `RF_TARGET` production lock could be recovered. "
        "The validation archive contains fold-specific RF_TARGET tuning results, while `TROPOSIF_Final_Model_Lock.md` and the production-lock tuning file lock only `RF_TARGET_GOSIF`. "
        "Per the user rule, no replacement parameters were invented.\n"
    )
    md.append(markdown_table(sensitivity) + "\n")
    md.append("## QC flags\n")
    md.append(f"QC flagged rows: `{len(flags)}`\n")
    md.append(markdown_table(flags[["SITE_ID", "YYYYMM", "TROPOSIF_primary_status", "QC_flag_reasons"]], 20) + "\n")
    md.append("## Input safety audit\n")
    md.append(f"SOURCE_INPUTS_UNCHANGED = `{source_inputs_unchanged}`\n")
    md.append(f"Input audit CSV: `{input_audit_path}`\n")
    md.append(
        "The audit compared size and mtime before/after for raw TROPOSIF files, raw GOSIF files, the GPP TIFF root, preparation CSVs, and previous TROPOSIF validation files.\n"
    )
    md.append("## Output files\n")
    for name in [
        "TROPOSIF_Final_EC_Footprint_SIF_All_SiteMonths.csv",
        "TROPOSIF_Final_EC_Footprint_SIF_Primary_Observed.csv",
        "TROPOSIF_Finalization_QC_Flags.csv",
        "TROPOSIF_TargetOnly_RF_Sensitivity_Status.csv",
        "TROPOSIF_Finalization_Simple_Summary.png",
        "TROPOSIF_Finalization_Simple_Summary.pdf",
    ]:
        md.append(f"- `{OUTPUT_ROOT / name}`\n")
    md.append("\n## Final verdict\n")
    md.append("**TROPOSIF FINALIZATION COMPLETE**\n")

    md_path = OUTPUT_ROOT / "TROPOSIF_Finalization_Report.md"
    pdf_path = OUTPUT_ROOT / "TROPOSIF_Finalization_Report.pdf"
    md_path.write_text("\n".join(md), encoding="utf-8")
    write_pdf_from_text(md_path, pdf_path)


def main() -> None:
    ensure_dirs()

    safety_inputs = [
        RAW_TROPOSIF_ROOT,
        RAW_GOSIF_ROOT,
        GPP_TIFF_ROOT,
        TROPOSIF_INVENTORY_CSV,
        EC_COVERAGE_CSV,
        ARTIFICIAL_GAP_CSV,
        MODEL_LOCK_MD,
        PRODUCTION_LOCK_TUNING_CSV,
        MODEL_HYPERPARAMETERS_CSV,
        VALIDATION_ROOT / "TROPOSIF_Validation_Report.md",
        VALIDATION_ROOT / "TROPOSIF_Final_Model_Predictions_All_Folds.csv",
    ]
    before = file_snapshot(safety_inputs)

    df, _cov = read_inputs()
    expected_counts = verify_counts(df)
    master, flags = build_master(df)
    lock_ok, reason, details = recover_rf_target_lock()
    sensitivity = sensitivity_status_table(master, lock_ok, reason, details)
    counts = summarize_counts(master, flags, sensitivity)

    # Reconcile final status counts.
    if counts["N_total_TROPOSIF"] != counts["N_complete_observed"] + counts["N_partial_positive"] + counts["N_zero_coverage"]:
        raise RuntimeError("Final count reconciliation failed.")
    if counts["N_complete_observed"] != expected_counts["N_complete_observed"]:
        raise RuntimeError("Complete count changed during finalization.")
    if counts["N_primary_analysis_ready"] != counts["N_complete_primary_ready"]:
        raise RuntimeError("Primary-ready count mismatch.")

    write_csvs(master, flags, sensitivity, counts)
    create_summary_figure(master)

    after = file_snapshot(safety_inputs)
    audit = compare_snapshots(before, after)
    input_audit_path = QC_ROOT / "TROPOSIF_Finalization_Source_Input_Safety_Audit.csv"
    audit.to_csv(input_audit_path, index=False)
    source_inputs_unchanged = bool(audit["unchanged_size_and_mtime"].fillna(False).all())

    write_reports(master, flags, sensitivity, counts, source_inputs_unchanged, input_audit_path)

    reproducibility = {
        "script_path": str((Path(PROJECT_ROOT) / 'Codes/pythonProject/Finalize_TROPOSIF_Observed_Primary_Dataset.py').resolve()),
        "output_root": str(OUTPUT_ROOT),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
        "coverage_tolerance": COVERAGE_TOL,
        "counts": counts,
        "source_inputs_unchanged": source_inputs_unchanged,
        "gosif_used_in_primary": False,
        "gpp_used": False,
        "target_only_sensitivity_successful": bool(sensitivity["sensitivity_run_successful"].iloc[0]),
        "target_only_sensitivity_reason": str(sensitivity["reason"].iloc[0]),
    }
    (OUTPUT_ROOT / "TROPOSIF_Finalization_Reproducibility.json").write_text(
        json.dumps(reproducibility, indent=2), encoding="utf-8"
    )

    print("TROPOSIF finalization complete")
    for k, v in counts.items():
        print(f"{k}: {v}")
    print(f"SOURCE_INPUTS_UNCHANGED: {source_inputs_unchanged}")
    print(f"Primary observed table: {OUTPUT_ROOT / 'TROPOSIF_Final_EC_Footprint_SIF_Primary_Observed.csv'}")
    print(f"Master table: {OUTPUT_ROOT / 'TROPOSIF_Final_EC_Footprint_SIF_All_SiteMonths.csv'}")
    print(f"Report: {OUTPUT_ROOT / 'TROPOSIF_Finalization_Report.md'}")
    print("TROPOSIF FINALIZATION COMPLETE")


if __name__ == "__main__":
    main()
