"""Pre-production validation of OCO-3 EC-footprint reconstruction: artificial-gap datasets, empirical coverage-class matching and model validation. Tower GPP is not used as a predictor.

Output: validation tables and figures used to lock the final model."""
from __future__ import annotations
# Original script filename (sanitized): Run_OCO3_Final_PreProduction_Validation.py; public name: 02_Run_OCO3_Final_PreProduction_Validation.py
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

import math
import os
import platform
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except Exception:
    XGBRegressor = None
    XGBOOST_AVAILABLE = False

try:
    from catboost import CatBoostRegressor

    CATBOOST_AVAILABLE = True
except Exception:
    CatBoostRegressor = None
    CATBOOST_AVAILABLE = False


RANDOM_SEED = 20260812
NODATA_SENTINEL = -9999.0

PROJECT_ROOT = Path(
    PROJECT_ROOT
)
POC_ROOT = PROJECT_ROOT / "SIF_EC_Reconstruction_OCO3_Proof_of_Concept"
PREP_ROOT = PROJECT_ROOT / "SIF_EC_Reconstruction_Model_Preparation"
OUTPUT_ROOT = PROJECT_ROOT / "SIF_EC_Reconstruction_OCO3_Final_PreProduction_Validation"
FIG_DIR = OUTPUT_ROOT / "figures"
QC_DIR = OUTPUT_ROOT / "qc"

GAP_DATASET = POC_ROOT / "OCO3_Artificial_Gap_Dataset.csv"
TRUTH_DATASET = POC_ROOT / "qc" / "_checkpoint_OCO3_Complete_Truth_Cases.csv"
REAL_COVERAGE = PREP_ROOT / "EC_Coverage_All_Products.csv"

PREDICTIONS_CSV = OUTPUT_ROOT / "OCO3_Final_Model_Predictions_All_Folds.csv"
PERFORMANCE_CSV = OUTPUT_ROOT / "OCO3_Final_Model_Performance_All_Validations.csv"
BY_COVERAGE_CSV = OUTPUT_ROOT / "OCO3_Final_Model_Performance_By_Coverage.csv"
ABLATION_CSV = OUTPUT_ROOT / "OCO3_Final_Ablation_Performance.csv"
GOSIF_INCREMENT_CSV = OUTPUT_ROOT / "OCO3_GOSIF_Incremental_Value.csv"
ZERO_COVERAGE_CSV = OUTPUT_ROOT / "OCO3_Zero_Coverage_Validation.csv"
EMPIRICAL_COVERAGE_CSV = OUTPUT_ROOT / "OCO3_Empirical_Coverage_Validation.csv"
EMPIRICAL_MATCHED_DATASET_CSV = OUTPUT_ROOT / "OCO3_Empirical_Matched_Artificial_Gap_Dataset.csv"
ZERO_DATASET_CSV = OUTPUT_ROOT / "OCO3_Zero_Coverage_Artificial_Gap_Dataset.csv"
HYPERPARAMS_CSV = OUTPUT_ROOT / "OCO3_Final_Model_Hyperparameters.csv"
FEATURE_IMPORTANCE_CSV = OUTPUT_ROOT / "OCO3_Final_Model_Feature_Importance.csv"
UNCERTAINTY_CSV = OUTPUT_ROOT / "OCO3_Final_Uncertainty_Calibration.csv"
MODEL_LOCK_MD = OUTPUT_ROOT / "OCO3_Final_Model_Lock.md"
REPORT_MD = OUTPUT_ROOT / "OCO3_Final_PreProduction_Report.md"
REPORT_PDF = OUTPUT_ROOT / "OCO3_Final_PreProduction_Report.pdf"
README_TXT = OUTPUT_ROOT / "OCO3_Final_PreProduction_READ_ME_FIRST.txt"
SIMPLE_SUMMARY_PNG = OUTPUT_ROOT / "OCO3_Final_PreProduction_Simple_Summary.png"
SIMPLE_SUMMARY_PDF = OUTPUT_ROOT / "OCO3_Final_PreProduction_Simple_Summary.pdf"
INPUT_AUDIT_CSV = QC_DIR / "Input_File_Size_MTime_Audit.csv"

TARGET_COLUMN = "true_complete_OCO3_EC_SIF"

TARGET_FEATURES = [
    "partial_OCO3_EC_weighted_SIF",
    "partial_OCO3_mean",
    "partial_OCO3_sd",
    "partial_OCO3_min",
    "partial_OCO3_max",
    "artificial_coverage_probability",
    "artificial_coverage_pixel_fraction",
    "missing_probability_fraction",
    "n_original_target_footprints",
    "n_remaining_target_footprints",
    "lat",
    "lon",
    "year",
    "month",
    "DOY",
    "sin_DOY",
    "cos_DOY",
]

GOSIF_FEATURES = [
    "GOSIF_EC_weighted",
    "GOSIF_native_sd",
    "n_native_GOSIF_cells_intersecting_EC",
]

CONTEXT_FEATURES = [
    "lat",
    "lon",
    "year",
    "month",
    "DOY",
    "sin_DOY",
    "cos_DOY",
]

TARGET_GOSIF_FEATURES = TARGET_FEATURES + GOSIF_FEATURES
GOSIF_ONLY_FEATURES = GOSIF_FEATURES + CONTEXT_FEATURES
CATEGORICAL_FEATURES = ["IGBP"]

SEARCH_BUDGET = 20
INNER_SPLITS = 2
OUTER_SPLITS = 5
MIN_FOLD_TRAIN = 30
MIN_FOLD_TEST = 20

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def ensure_dirs() -> None:
    for p in [OUTPUT_ROOT, FIG_DIR, QC_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def stable_rng(*parts: Any) -> np.random.Generator:
    text = "|".join(str(p) for p in parts)
    seed = RANDOM_SEED
    for ch in text:
        seed = (seed * 131 + ord(ch)) % (2**32 - 1)
    return np.random.default_rng(seed)


def coverage_class(v: float) -> str:
    if not math.isfinite(v):
        return "NaN"
    if abs(v) <= 1e-12:
        return "=0"
    if v <= 0.10:
        return "0-0.10"
    if v <= 0.25:
        return ">0.10-0.25"
    if v <= 0.50:
        return ">0.25-0.50"
    if v <= 0.75:
        return ">0.50-0.75"
    if v < 1.0 - 1e-9:
        return ">0.75-<1"
    if abs(v - 1.0) <= 1e-9:
        return "=1"
    return ">1"


def analysis_coverage_class(v: float) -> str:
    if not math.isfinite(v):
        return "NaN"
    if abs(v) <= 1e-12:
        return "=0"
    if v <= 0.10:
        return "0-0.10"
    if v <= 0.25:
        return ">0.10-0.25"
    if v <= 0.50:
        return ">0.25-0.50"
    if v <= 0.75:
        return ">0.50-0.75"
    if v <= 0.90:
        return ">0.75-0.90"
    if v < 1.0 - 1e-9:
        return ">0.90-<1.00"
    if abs(v - 1.0) <= 1e-9:
        return "=1"
    return ">1"


def low_medium_high(v: float) -> str:
    if not math.isfinite(v):
        return "unknown"
    if v <= 0.25:
        return "low_<=0.25"
    if v <= 0.75:
        return "medium_>0.25-0.75"
    return "high_>0.75"


def finite_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return np.nan
    return v if math.isfinite(v) else np.nan


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [p for p in [GAP_DATASET, TRUTH_DATASET, REAL_COVERAGE] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required POC/preparation files: " + "; ".join(str(p) for p in missing))
    gap = pd.read_csv(GAP_DATASET)
    truth = pd.read_csv(TRUTH_DATASET)
    real = pd.read_csv(REAL_COVERAGE)
    return gap, truth, real


def audit_input_files(paths: Iterable[Path], suffix: str) -> pd.DataFrame:
    rows = []
    for p in paths:
        rows.append(
            {
                "audit_stage": suffix,
                "path": str(p),
                "exists": p.exists(),
                "size_bytes": p.stat().st_size if p.exists() else np.nan,
                "mtime_ns": p.stat().st_mtime_ns if p.exists() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def validate_gap_dataset(gap: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    required = [
        "original_case_ID",
        "SITE_ID",
        "YYYYMM",
        "artificial_coverage_probability",
        TARGET_COLUMN,
        "partial_OCO3_EC_weighted_SIF",
        "GOSIF_EC_weighted",
    ]
    rows = []
    for col in required:
        rows.append(
            {
                "check": f"required_column_{col}",
                "status": "PASS" if col in gap.columns else "FAIL",
                "detail": "" if col in gap.columns else "missing in artificial gap dataset",
            }
        )
    if "original_case_ID" in gap and TARGET_COLUMN in gap:
        truth_counts = gap.groupby("original_case_ID")[TARGET_COLUMN].nunique(dropna=True)
        rows.append(
            {
                "check": "truth_constant_within_original_case",
                "status": "PASS" if int((truth_counts > 1).sum()) == 0 else "FAIL",
                "detail": f"cases_with_multiple_truth_values={(truth_counts > 1).sum()}",
            }
        )
    cov = pd.to_numeric(gap.get("artificial_coverage_probability"), errors="coerce")
    rows.append(
        {
            "check": "artificial_coverage_within_0_1",
            "status": "PASS" if bool(((cov >= -1e-9) & (cov <= 1 + 1e-9)).all()) else "FAIL",
            "detail": f"min={cov.min()}, max={cov.max()}, n_nan={cov.isna().sum()}",
        }
    )
    rows.append(
        {
            "check": "complete_truth_case_count",
            "status": "PASS" if len(truth) > 0 else "FAIL",
            "detail": f"truth_rows={len(truth)}, truth_sites={truth['SITE_ID'].nunique() if 'SITE_ID' in truth else 'NA'}",
        }
    )
    out = pd.DataFrame(rows)
    out.to_csv(QC_DIR / "OCO3_Input_Consistency_Audit.csv", index=False)
    return out


def real_coverage_regime_table(real: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    oco3 = real[real["PRODUCT"].astype(str).str.upper().isin(["OCO-3", "OCO3"])].copy()
    oco3["coverage_probability_fraction"] = pd.to_numeric(oco3["coverage_probability_fraction"], errors="coerce")
    oco3 = oco3[oco3["coverage_probability_fraction"].notna()].copy()
    oco3["coverage_regime"] = oco3["coverage_probability_fraction"].map(coverage_class)
    order = ["=0", "0-0.10", ">0.10-0.25", ">0.25-0.50", ">0.50-0.75", ">0.75-<1", "=1", ">1", "NaN"]
    rows = []
    total = len(oco3)
    for reg in order:
        sub = oco3[oco3["coverage_regime"] == reg]
        if len(sub) == 0 and reg not in ["=0", "0-0.10", ">0.10-0.25", ">0.25-0.50", ">0.50-0.75", ">0.75-<1", "=1"]:
            continue
        rows.append(
            {
                "coverage_regime": reg,
                "N_site_months": len(sub),
                "percentage": 100 * len(sub) / total if total else np.nan,
                "N_sites": sub["SITE_ID"].nunique() if "SITE_ID" in sub else 0,
                "years": ", ".join(map(str, sorted(pd.to_numeric(sub.get("year"), errors="coerce").dropna().astype(int).unique()))),
                "N_years": pd.to_numeric(sub.get("year"), errors="coerce").dropna().nunique(),
                "IGBP_classes": ", ".join(sorted(map(str, sub.get("IGBP", pd.Series(dtype=str)).dropna().unique()))),
                "N_IGBP_classes": sub.get("IGBP", pd.Series(dtype=str)).dropna().nunique(),
            }
        )
    regime = pd.DataFrame(rows)
    regime.to_csv(OUTPUT_ROOT / "OCO3_Real_Coverage_Regimes.csv", index=False)
    return oco3, regime


def distribution_stats(label: str, values: pd.Series) -> dict[str, Any]:
    v = pd.to_numeric(values, errors="coerce").dropna()
    return {
        "distribution": label,
        "N": int(len(v)),
        "mean": float(v.mean()) if len(v) else np.nan,
        "median": float(v.median()) if len(v) else np.nan,
        "P05": float(v.quantile(0.05)) if len(v) else np.nan,
        "P25": float(v.quantile(0.25)) if len(v) else np.nan,
        "P50": float(v.quantile(0.50)) if len(v) else np.nan,
        "P75": float(v.quantile(0.75)) if len(v) else np.nan,
        "P95": float(v.quantile(0.95)) if len(v) else np.nan,
        "fraction_eq_0": float((v <= 1e-12).mean()) if len(v) else np.nan,
        "fraction_lt_0_10": float((v < 0.10).mean()) if len(v) else np.nan,
        "fraction_lt_0_25": float((v < 0.25).mean()) if len(v) else np.nan,
        "fraction_lt_0_50": float((v < 0.50).mean()) if len(v) else np.nan,
    }


def create_zero_coverage_dataset(truth: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in truth.iterrows():
        n_orig_f = finite_float(r.get("n_original_target_footprints"))
        n_orig = int(n_orig_f) if math.isfinite(n_orig_f) else 0
        rows.append(
            {
                "original_case_ID": r["original_case_ID"],
                "SITE_ID": r["SITE_ID"],
                "SITE_SAFE": r.get("SITE_SAFE", str(r["SITE_ID"]).replace("-", "_")),
                "YYYYMM": int(r["YYYYMM"]),
                "lat": finite_float(r.get("lat")),
                "lon": finite_float(r.get("lon")),
                "IGBP": r.get("IGBP", "Unknown"),
                "year": int(finite_float(r.get("year"))),
                "month": int(finite_float(r.get("month"))),
                "DOY": finite_float(r.get("DOY")),
                "sin_DOY": finite_float(r.get("sin_DOY")),
                "cos_DOY": finite_float(r.get("cos_DOY")),
                "gap_scenario": "zero_all_target_footprints_removed",
                "target_coverage_requested": 0.0,
                "original_complete_coverage": finite_float(r.get("full_EC_probability_coverage_recomputed")),
                "artificial_coverage_probability": 0.0,
                "artificial_coverage_pixel_fraction": 0.0,
                "missing_probability_fraction": 1.0,
                "coverage_class": "=0",
                "n_original_target_footprints": n_orig,
                "n_remaining_target_footprints": 0,
                "n_removed_target_footprints": n_orig,
                "remaining_footprint_ids": "",
                "removed_footprint_ids": "ALL_ORIGINAL_OCO3_FOOTPRINTS",
                "partial_OCO3_EC_weighted_SIF": np.nan,
                "partial_OCO3_mean": np.nan,
                "partial_OCO3_sd": np.nan,
                "partial_OCO3_min": np.nan,
                "partial_OCO3_max": np.nan,
                "GOSIF_EC_weighted": finite_float(r.get("GOSIF_EC_weighted")),
                "GOSIF_native_sd": finite_float(r.get("GOSIF_native_sd")),
                "n_native_GOSIF_cells_intersecting_EC": finite_float(r.get("n_native_GOSIF_cells_intersecting_EC")),
                TARGET_COLUMN: finite_float(r.get("full_OCO3_EC_weighted_SIF_recomputed")),
                "truth_minus_partial": np.nan,
                "gap_generation_method": "all_whole_OCO3_source_footprints_removed",
                "no_fine_cell_random_deletion": True,
            }
        )
    zero = pd.DataFrame(rows)
    zero.to_csv(ZERO_DATASET_CSV, index=False)
    return zero


def create_empirical_matched_dataset(real_oco3: pd.DataFrame, gap: pd.DataFrame, zero: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    real_incomplete = real_oco3[real_oco3["coverage_probability_fraction"] < 1 - 1e-9].copy()
    real_incomplete["coverage_class_final"] = real_incomplete["coverage_probability_fraction"].map(analysis_coverage_class)
    gap_pool = pd.concat([gap.copy(), zero.copy()], ignore_index=True)
    gap_pool["coverage_class_final"] = gap_pool["artificial_coverage_probability"].map(analysis_coverage_class)
    classes = ["=0", "0-0.10", ">0.10-0.25", ">0.25-0.50", ">0.50-0.75", ">0.75-0.90", ">0.90-<1.00"]
    total_target = min(len(gap_pool), max(1000, len(gap)))
    rng = stable_rng("empirical_matched_dataset")
    sampled_parts = []
    rows = []
    for cls in classes:
        real_n = int((real_incomplete["coverage_class_final"] == cls).sum())
        real_prop = real_n / len(real_incomplete) if len(real_incomplete) else 0
        desired = int(round(real_prop * total_target))
        available = gap_pool[gap_pool["coverage_class_final"] == cls]
        take = min(desired, len(available))
        if take > 0:
            sampled_parts.append(available.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1))))
        rows.append(
            {
                "coverage_class": cls,
                "real_incomplete_N": real_n,
                "real_incomplete_percent": 100 * real_prop,
                "original_artificial_N": int((gap["artificial_coverage_probability"].map(analysis_coverage_class) == cls).sum()),
                "zero_augmented_pool_N": len(available),
                "empirical_matched_desired_N": desired,
                "empirical_matched_sampled_N": take,
                "sampling_note": "limited_by_available_discrete_whole-footprint_cases" if take < desired else "matched_or_available",
            }
        )
    matched = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else pd.DataFrame(columns=gap_pool.columns)
    matched = matched.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    matched["gap_scenario"] = "EMPIRICAL_COVERAGE_VALIDATION_" + matched["gap_scenario"].astype(str)
    matched.to_csv(EMPIRICAL_MATCHED_DATASET_CSV, index=False)
    table = pd.DataFrame(rows)

    dist_rows = [
        distribution_stats("real_incomplete_archive", real_incomplete["coverage_probability_fraction"]),
        distribution_stats("original_artificial_gaps", gap["artificial_coverage_probability"]),
        distribution_stats("empirical_matched_artificial_gaps", matched["artificial_coverage_probability"]),
    ]
    dist = pd.DataFrame(dist_rows)
    out = table.merge(dist, how="cross") if False else table
    table.to_csv(OUTPUT_ROOT / "OCO3_Empirical_Coverage_Class_Matching.csv", index=False)
    dist.to_csv(EMPIRICAL_COVERAGE_CSV, index=False)
    return matched, dist


def preprocessor(numeric_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")), ("onehot", one_hot_encoder())]), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


@dataclass(frozen=True)
class ModelSpec:
    model: str
    family: str
    variant: str
    uses_gosif: bool
    uses_target: bool
    features: tuple[str, ...]
    supports_zero_coverage: bool
    description: str


def model_specs(include_statistical: bool = True) -> list[ModelSpec]:
    specs = [
        ModelSpec(
            "RF_TARGET",
            "RF",
            "TARGET",
            False,
            True,
            tuple(TARGET_FEATURES),
            False,
            "Random Forest with target OCO-3 partial-observation, sampling, space/time, and IGBP predictors only.",
        ),
        ModelSpec(
            "RF_GOSIF_ONLY",
            "RF",
            "GOSIF_ONLY",
            True,
            False,
            tuple(GOSIF_ONLY_FEATURES),
            True,
            "Random Forest using GOSIF plus space/time/IGBP context; no partial OCO-3 information.",
        ),
        ModelSpec(
            "RF_TARGET_GOSIF",
            "RF",
            "TARGET_GOSIF",
            True,
            True,
            tuple(TARGET_GOSIF_FEATURES),
            False,
            "Random Forest with exactly RF_TARGET predictors plus GOSIF EC-weighted/support predictors.",
        ),
    ]
    if XGBOOST_AVAILABLE:
        specs += [
            ModelSpec(
                "XGB_TARGET",
                "XGB",
                "TARGET",
                False,
                True,
                tuple(TARGET_FEATURES),
                False,
                "XGBoost with target OCO-3 partial-observation, sampling, space/time, and IGBP predictors only.",
            ),
            ModelSpec(
                "XGB_GOSIF_ONLY",
                "XGB",
                "GOSIF_ONLY",
                True,
                False,
                tuple(GOSIF_ONLY_FEATURES),
                True,
                "XGBoost using GOSIF plus space/time/IGBP context; no partial OCO-3 information.",
            ),
            ModelSpec(
                "XGB_TARGET_GOSIF",
                "XGB",
                "TARGET_GOSIF",
                True,
                True,
                tuple(TARGET_GOSIF_FEATURES),
                False,
                "XGBoost with exactly XGB_TARGET predictors plus GOSIF EC-weighted/support predictors.",
            ),
        ]
    if CATBOOST_AVAILABLE:
        specs += [
            ModelSpec("CAT_TARGET", "CAT", "TARGET", False, True, tuple(TARGET_FEATURES), False, "CatBoost target-only model."),
            ModelSpec("CAT_GOSIF_ONLY", "CAT", "GOSIF_ONLY", True, False, tuple(GOSIF_ONLY_FEATURES), True, "CatBoost GOSIF-only/context model."),
            ModelSpec("CAT_TARGET_GOSIF", "CAT", "TARGET_GOSIF", True, True, tuple(TARGET_GOSIF_FEATURES), False, "CatBoost target+GOSIF model."),
        ]
    if include_statistical:
        specs.append(
            ModelSpec(
                "STAT_RBF_TARGET_ONLY",
                "STAT_RBF",
                "TARGET",
                False,
                True,
                tuple(TARGET_FEATURES),
                False,
                "Target-only approximate spatiotemporal Gaussian-process equivalent: standardized target/context predictors, RBF Nystroem features, Ridge regression. This is not labelled as exact kriging.",
            )
        )
    return specs


def config_grid(family: str) -> list[dict[str, Any]]:
    rng = stable_rng("config_grid", family)
    if family == "RF":
        grid = []
        for n in [80, 120, 160, 220]:
            for depth in [None, 8, 14, 20]:
                for leaf in [1, 2, 4, 8]:
                    for mf in ["sqrt", 0.55, 0.8]:
                        grid.append({"n_estimators": n, "max_depth": depth, "min_samples_leaf": leaf, "max_features": mf})
        idx = rng.choice(len(grid), size=min(SEARCH_BUDGET, len(grid)), replace=False)
        return [grid[int(i)] for i in idx]
    if family == "XGB":
        grid = []
        for n in [80, 120, 180, 240]:
            for depth in [2, 3, 4]:
                for lr in [0.03, 0.05, 0.08]:
                    for subsample in [0.75, 0.9]:
                        for colsample in [0.75, 0.9]:
                            grid.append(
                                {
                                    "n_estimators": n,
                                    "max_depth": depth,
                                    "learning_rate": lr,
                                    "subsample": subsample,
                                    "colsample_bytree": colsample,
                                    "reg_lambda": float(rng.choice([1.0, 3.0, 8.0])),
                                }
                            )
        idx = rng.choice(len(grid), size=min(SEARCH_BUDGET, len(grid)), replace=False)
        return [grid[int(i)] for i in idx]
    if family == "CAT":
        grid = []
        for iters in [120, 180, 240]:
            for depth in [4, 6, 8]:
                for lr in [0.03, 0.05, 0.08]:
                    for l2 in [1, 3, 8]:
                        grid.append({"iterations": iters, "depth": depth, "learning_rate": lr, "l2_leaf_reg": l2})
        idx = rng.choice(len(grid), size=min(SEARCH_BUDGET, len(grid)), replace=False)
        return [grid[int(i)] for i in idx]
    if family == "STAT_RBF":
        return [{"gamma": 0.04, "n_components": 120, "alpha": 1.0}, {"gamma": 0.08, "n_components": 160, "alpha": 1.0}]
    raise ValueError(f"Unknown model family: {family}")


def build_estimator(spec: ModelSpec, params: dict[str, Any], seed: int) -> Pipeline:
    features = list(spec.features)
    if spec.family == "RF":
        return Pipeline(
            [
                ("preprocess", preprocessor(features)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=int(params["n_estimators"]),
                        max_depth=params["max_depth"],
                        min_samples_leaf=int(params["min_samples_leaf"]),
                        max_features=params["max_features"],
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    if spec.family == "XGB":
        return Pipeline(
            [
                ("preprocess", preprocessor(features)),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=int(params["n_estimators"]),
                        max_depth=int(params["max_depth"]),
                        learning_rate=float(params["learning_rate"]),
                        subsample=float(params["subsample"]),
                        colsample_bytree=float(params["colsample_bytree"]),
                        reg_lambda=float(params["reg_lambda"]),
                        objective="reg:squarederror",
                        random_state=seed,
                        n_jobs=2,
                    ),
                ),
            ]
        )
    if spec.family == "CAT":
        return Pipeline(
            [
                ("preprocess", preprocessor(features)),
                (
                    "model",
                    CatBoostRegressor(
                        iterations=int(params["iterations"]),
                        depth=int(params["depth"]),
                        learning_rate=float(params["learning_rate"]),
                        l2_leaf_reg=float(params["l2_leaf_reg"]),
                        loss_function="RMSE",
                        random_seed=seed,
                        verbose=False,
                        allow_writing_files=False,
                    ),
                ),
            ]
        )
    if spec.family == "STAT_RBF":
        return Pipeline(
            [
                ("preprocess", preprocessor(features)),
                ("scale", StandardScaler(with_mean=True, with_std=True)),
                (
                    "nystroem",
                    Nystroem(
                        kernel="rbf",
                        gamma=float(params["gamma"]),
                        n_components=int(params["n_components"]),
                        random_state=seed,
                    ),
                ),
                ("model", Ridge(alpha=float(params["alpha"]))),
            ]
        )
    raise ValueError(spec.family)


def feature_frame(df: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
    cols = list(spec.features) + CATEGORICAL_FEATURES
    out = df[cols].copy()
    for col in spec.features:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["IGBP"] = out["IGBP"].fillna("Unknown").astype(str)
    return out


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y, pred)))


def metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if len(y) == 0:
        return {"n": 0, "RMSE": np.nan, "MAE": np.nan, "R2": np.nan, "Bias": np.nan, "nRMSE": np.nan, "Pearson_r": np.nan, "Spearman_rho": np.nan}
    denom = np.nanpercentile(y, 95) - np.nanpercentile(y, 5) if len(y) > 2 else np.nan
    pearson = stats.pearsonr(y, p).statistic if len(y) > 2 and np.std(y) > 0 and np.std(p) > 0 else np.nan
    spearman = stats.spearmanr(y, p).statistic if len(y) > 2 and np.std(y) > 0 and np.std(p) > 0 else np.nan
    return {
        "n": int(len(y)),
        "RMSE": rmse(y, p),
        "MAE": float(mean_absolute_error(y, p)),
        "R2": float(r2_score(y, p)) if len(y) > 1 else np.nan,
        "Bias": float(np.mean(p - y)),
        "nRMSE": float(rmse(y, p) / denom) if math.isfinite(denom) and denom > 0 else np.nan,
        "Pearson_r": float(pearson) if math.isfinite(pearson) else np.nan,
        "Spearman_rho": float(spearman) if math.isfinite(spearman) else np.nan,
    }


def valid_model_data(df: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
    cols = list(spec.features) + CATEGORICAL_FEATURES + [TARGET_COLUMN, "original_case_ID", "SITE_ID", "YYYYMM", "year", "coverage_class", "gap_scenario", "artificial_coverage_probability", "partial_OCO3_EC_weighted_SIF", "GOSIF_EC_weighted"]
    cols = list(dict.fromkeys([c for c in cols if c in df.columns]))
    out = df[cols].copy()
    out[TARGET_COLUMN] = pd.to_numeric(out[TARGET_COLUMN], errors="coerce")
    out = out[out[TARGET_COLUMN].notna()].copy()
    for c in spec.features:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def inner_cv_indices(train_df: pd.DataFrame, groups_col: str = "original_case_ID") -> list[tuple[np.ndarray, np.ndarray]]:
    groups = train_df[groups_col].astype(str).to_numpy()
    unique = np.unique(groups)
    n_splits = min(INNER_SPLITS, len(unique))
    if n_splits < 2:
        idx = np.arange(len(train_df))
        split = int(max(1, round(0.75 * len(idx))))
        return [(idx[:split], idx[split:])]
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(train_df, groups=groups))


def tune_model(spec: ModelSpec, train_df: pd.DataFrame, validation: str, fold_id: str) -> tuple[dict[str, Any], float, list[dict[str, Any]]]:
    train_df = valid_model_data(train_df, spec)
    y_all = train_df[TARGET_COLUMN].astype(float).to_numpy()
    configs = config_grid(spec.family)
    rows = []
    best_score = np.inf
    best_params = configs[0]
    splits = inner_cv_indices(train_df)
    for i, params in enumerate(configs, start=1):
        scores = []
        for inner_i, (tr_idx, va_idx) in enumerate(splits, start=1):
            tr = train_df.iloc[tr_idx]
            va = train_df.iloc[va_idx]
            if len(tr) < 10 or len(va) < 5:
                continue
            est = build_estimator(spec, params, seed=RANDOM_SEED + i + inner_i)
            est.fit(feature_frame(tr, spec), tr[TARGET_COLUMN].astype(float).to_numpy())
            pred = est.predict(feature_frame(va, spec))
            scores.append(rmse(va[TARGET_COLUMN].astype(float).to_numpy(), pred))
        mean_score = float(np.mean(scores)) if scores else np.inf
        rows.append(
            {
                "validation": validation,
                "fold_id": fold_id,
                "model": spec.model,
                "family": spec.family,
                "variant": spec.variant,
                "config_index": i,
                "inner_cv_rmse": mean_score,
                "params": str(params),
                "search_budget": len(configs),
                "inner_splits": len(splits),
            }
        )
        if mean_score < best_score:
            best_score = mean_score
            best_params = params
    return best_params, best_score, rows


def split_group_kfold(df: pd.DataFrame, group_col: str, n_splits: int = OUTER_SPLITS) -> list[tuple[str, np.ndarray, np.ndarray]]:
    groups = df[group_col].astype(str).to_numpy()
    n_unique = len(np.unique(groups))
    splits = []
    if n_unique < 2:
        return splits
    k = min(n_splits, n_unique)
    gkf = GroupKFold(n_splits=k)
    for i, (tr, te) in enumerate(gkf.split(df, groups=groups), start=1):
        if len(tr) >= MIN_FOLD_TRAIN and len(te) >= MIN_FOLD_TEST:
            splits.append((f"{group_col}_fold_{i}", tr, te))
    return splits


def split_incremental_temporal(df: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray]]:
    years = sorted(pd.to_numeric(df["year"], errors="coerce").dropna().astype(int).unique())
    splits = []
    for train_year, test_year in zip(years[:-1], years[1:]):
        tr = np.flatnonzero(pd.to_numeric(df["year"], errors="coerce").to_numpy() <= train_year)
        te = np.flatnonzero(pd.to_numeric(df["year"], errors="coerce").to_numpy() == test_year)
        if len(tr) >= MIN_FOLD_TRAIN and len(te) >= MIN_FOLD_TEST:
            splits.append((f"train_through_{train_year}_test_{test_year}", tr, te))
    return splits


def site_bucket(site: str, year: int) -> int:
    val = sum((i + 1) * ord(ch) for i, ch in enumerate(str(site))) + year * 17
    return val % 3


def split_spatial_temporal_stress(df: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray]]:
    years = sorted(pd.to_numeric(df["year"], errors="coerce").dropna().astype(int).unique())
    splits = []
    for train_year, test_year in zip(years[:-1], years[1:]):
        candidate_sites = sorted(df.loc[pd.to_numeric(df["year"], errors="coerce") == test_year, "SITE_ID"].astype(str).unique())
        held = {s for s in candidate_sites if site_bucket(s, test_year) == 0}
        if not held and candidate_sites:
            held = set(candidate_sites[: max(1, len(candidate_sites) // 3)])
        tr_mask = (pd.to_numeric(df["year"], errors="coerce") <= train_year) & (~df["SITE_ID"].astype(str).isin(held))
        te_mask = (pd.to_numeric(df["year"], errors="coerce") == test_year) & (df["SITE_ID"].astype(str).isin(held))
        tr = np.flatnonzero(tr_mask.to_numpy())
        te = np.flatnonzero(te_mask.to_numpy())
        if len(tr) >= MIN_FOLD_TRAIN and len(te) >= 10:
            splits.append((f"stress_train_to_{train_year}_held_sites_test_{test_year}", tr, te))
    return splits


def append_prediction_rows(rows: list[dict[str, Any]], validation: str, fold_id: str, model: str, test: pd.DataFrame, pred: np.ndarray, notes: str) -> None:
    for (_, rec), yhat in zip(test.iterrows(), pred):
        y = finite_float(rec[TARGET_COLUMN])
        rows.append(
            {
                "validation": validation,
                "fold_id": fold_id,
                "model": model,
                "original_case_ID": rec["original_case_ID"],
                "SITE_ID": rec["SITE_ID"],
                "YYYYMM": rec["YYYYMM"],
                "year": rec["year"],
                "IGBP": rec.get("IGBP", "Unknown"),
                "gap_scenario": rec.get("gap_scenario", ""),
                "artificial_coverage_probability": finite_float(rec.get("artificial_coverage_probability")),
                "coverage_class": rec.get("coverage_class", analysis_coverage_class(finite_float(rec.get("artificial_coverage_probability")))),
                TARGET_COLUMN: y,
                "prediction": float(yhat) if math.isfinite(float(yhat)) else np.nan,
                "partial_OCO3_EC_weighted_SIF": finite_float(rec.get("partial_OCO3_EC_weighted_SIF")),
                "GOSIF_EC_weighted": finite_float(rec.get("GOSIF_EC_weighted")),
                "error": float(yhat - y) if math.isfinite(float(yhat)) and math.isfinite(y) else np.nan,
                "absolute_error": float(abs(yhat - y)) if math.isfinite(float(yhat)) and math.isfinite(y) else np.nan,
                "notes": notes,
            }
        )


def evaluate_validation(
    dataset_name: str,
    df: pd.DataFrame,
    validation: str,
    splits: list[tuple[str, np.ndarray, np.ndarray]],
    specs: list[ModelSpec],
    include_partial_baseline: bool,
    include_gosif_baselines: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[dict[str, Any]] = []
    param_rows: list[dict[str, Any]] = []
    for fold_id, tr_idx, te_idx in splits:
        train = df.iloc[tr_idx].copy()
        test = df.iloc[te_idx].copy()
        print(f"Validation {validation} {fold_id}: train={len(train)}, test={len(test)}", flush=True)

        if include_partial_baseline:
            pred = pd.to_numeric(test["partial_OCO3_EC_weighted_SIF"], errors="coerce").to_numpy()
            append_prediction_rows(prediction_rows, validation, fold_id, "Partial_observation_baseline", test, pred, "No-learning partial OCO-3 baseline.")
        if include_gosif_baselines:
            pred = pd.to_numeric(test["GOSIF_EC_weighted"], errors="coerce").to_numpy()
            append_prediction_rows(prediction_rows, validation, fold_id, "GOSIF_raw_baseline", test, pred, "Raw scale-corrected GOSIF EC-weighted baseline.")
            if train["GOSIF_EC_weighted"].notna().sum() >= 5:
                lr = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", LinearRegression())])
                lr.fit(train[["GOSIF_EC_weighted"]], train[TARGET_COLUMN].astype(float).to_numpy())
                pred = lr.predict(test[["GOSIF_EC_weighted"]])
                append_prediction_rows(prediction_rows, validation, fold_id, "GOSIF_linear_calibrated", test, pred, "Linear GOSIF-to-OCO3 calibration fitted on outer training data only.")

        for spec in specs:
            if dataset_name == "zero_coverage" and not spec.supports_zero_coverage:
                continue
            if len(train) < MIN_FOLD_TRAIN or len(test) < 5:
                continue
            best_params, inner_score, tuning_rows = tune_model(spec, train, validation, fold_id)
            param_rows.extend(tuning_rows)
            est = build_estimator(spec, best_params, seed=RANDOM_SEED + 101)
            tr_valid = valid_model_data(train, spec)
            te_valid = valid_model_data(test, spec)
            est.fit(feature_frame(tr_valid, spec), tr_valid[TARGET_COLUMN].astype(float).to_numpy())
            pred = est.predict(feature_frame(te_valid, spec))
            param_rows.append(
                {
                    "validation": validation,
                    "fold_id": fold_id,
                    "model": spec.model,
                    "family": spec.family,
                    "variant": spec.variant,
                    "config_index": "SELECTED",
                    "inner_cv_rmse": inner_score,
                    "params": str(best_params),
                    "search_budget": SEARCH_BUDGET if spec.family in ["RF", "XGB", "CAT"] else len(config_grid(spec.family)),
                    "inner_splits": "selected_after_inner_cv",
                    "feature_set": ", ".join(spec.features),
                    "random_seed": RANDOM_SEED,
                    "software_versions": software_versions(),
                    "description": spec.description,
                }
            )
            append_prediction_rows(prediction_rows, validation, fold_id, spec.model, te_valid, pred, spec.description)
    return pd.DataFrame(prediction_rows), pd.DataFrame(param_rows)


def summarize_predictions(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (validation, model), g in pred.groupby(["validation", "model"]):
        m = metrics(g[TARGET_COLUMN].to_numpy(), g["prediction"].to_numpy())
        rows.append({"validation": validation, "model": model, **m})
    overall = pd.DataFrame(rows).sort_values(["validation", "RMSE"])

    cov_rows = []
    for (validation, model, cov_class), g in pred.groupby(["validation", "model", "coverage_class"], dropna=False):
        m = metrics(g[TARGET_COLUMN].to_numpy(), g["prediction"].to_numpy())
        cov_rows.append({"validation": validation, "model": model, "coverage_class": cov_class, **m})
    by_cov = pd.DataFrame(cov_rows).sort_values(["validation", "coverage_class", "RMSE"])
    overall.to_csv(PERFORMANCE_CSV, index=False)
    by_cov.to_csv(BY_COVERAGE_CSV, index=False)
    return overall, by_cov


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = RANDOM_SEED) -> tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(vals, size=len(vals), replace=True)
        means[i] = np.mean(sample)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def gosif_incremental_table(pred: pd.DataFrame, overall: pd.DataFrame) -> pd.DataFrame:
    rows = []
    comparisons = [("RF", "RF_TARGET", "RF_TARGET_GOSIF"), ("XGBoost", "XGB_TARGET", "XGB_TARGET_GOSIF")]
    if CATBOOST_AVAILABLE:
        comparisons.append(("CatBoost", "CAT_TARGET", "CAT_TARGET_GOSIF"))
    for validation in sorted(pred["validation"].unique()):
        for family, target_model, gosif_model in comparisons:
            a = pred[(pred["validation"] == validation) & (pred["model"] == target_model)]
            b = pred[(pred["validation"] == validation) & (pred["model"] == gosif_model)]
            merged = a[["original_case_ID", "gap_scenario", "fold_id", TARGET_COLUMN, "prediction", "absolute_error"]].merge(
                b[["original_case_ID", "gap_scenario", "fold_id", "prediction", "absolute_error"]],
                on=["original_case_ID", "gap_scenario", "fold_id"],
                suffixes=("_target", "_target_gosif"),
            )
            if merged.empty:
                continue
            m_target = metrics(merged[TARGET_COLUMN].to_numpy(), merged["prediction_target"].to_numpy())
            m_gosif = metrics(merged[TARGET_COLUMN].to_numpy(), merged["prediction_target_gosif"].to_numpy())
            improvement = merged["absolute_error_target"].to_numpy() - merged["absolute_error_target_gosif"].to_numpy()
            ci_low, ci_high = bootstrap_ci(improvement, seed=RANDOM_SEED + len(rows))
            delta_rmse = m_target["RMSE"] - m_gosif["RMSE"]
            delta_mae = m_target["MAE"] - m_gosif["MAE"]
            delta_r2 = m_gosif["R2"] - m_target["R2"]
            if ci_low > 0 and delta_rmse > 0.01:
                conclusion = "GOSIF CLEARLY HELPS"
            elif delta_rmse > 0 and ci_high > 0:
                conclusion = "GOSIF MARGINALLY HELPS"
            elif delta_rmse < -0.005:
                conclusion = "GOSIF HURTS"
            else:
                conclusion = "NO CLEAR GOSIF BENEFIT"
            rows.append(
                {
                    "Model family": family,
                    "Validation": validation,
                    "N_common": len(merged),
                    "RMSE_TARGET": m_target["RMSE"],
                    "RMSE_TARGET_GOSIF": m_gosif["RMSE"],
                    "Delta_RMSE_from_adding_GOSIF": delta_rmse,
                    "Delta_MAE": delta_mae,
                    "Delta_R2": delta_r2,
                    "paired_abs_error_improvement_mean": float(np.nanmean(improvement)),
                    "CI95_low": ci_low,
                    "CI95_high": ci_high,
                    "Conclusion": conclusion,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(GOSIF_INCREMENT_CSV, index=False)
    return out


def gosif_increment_by_coverage(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for validation in sorted(pred["validation"].unique()):
        for family, target_model, gosif_model in [("RF", "RF_TARGET", "RF_TARGET_GOSIF"), ("XGBoost", "XGB_TARGET", "XGB_TARGET_GOSIF")]:
            a = pred[(pred["validation"] == validation) & (pred["model"] == target_model)]
            b = pred[(pred["validation"] == validation) & (pred["model"] == gosif_model)]
            merged = a[["original_case_ID", "gap_scenario", "fold_id", "coverage_class", TARGET_COLUMN, "prediction", "absolute_error"]].merge(
                b[["original_case_ID", "gap_scenario", "fold_id", "prediction", "absolute_error"]],
                on=["original_case_ID", "gap_scenario", "fold_id"],
                suffixes=("_target", "_target_gosif"),
            )
            for cov, g in merged.groupby("coverage_class"):
                if len(g) < 10:
                    continue
                mt = metrics(g[TARGET_COLUMN].to_numpy(), g["prediction_target"].to_numpy())
                mg = metrics(g[TARGET_COLUMN].to_numpy(), g["prediction_target_gosif"].to_numpy())
                rows.append(
                    {
                        "validation": validation,
                        "Model family": family,
                        "coverage_class": cov,
                        "N": len(g),
                        "Delta_RMSE_from_adding_GOSIF": mt["RMSE"] - mg["RMSE"],
                        "Delta_MAE": mt["MAE"] - mg["MAE"],
                        "Delta_R2": mg["R2"] - mt["R2"],
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_ROOT / "OCO3_GOSIF_Incremental_Value_By_Coverage.csv", index=False)
    return out


def ablation_table(overall: pd.DataFrame) -> pd.DataFrame:
    labels = [
        ("RF target only", "RF_TARGET", "No", "Yes"),
        ("RF GOSIF only", "RF_GOSIF_ONLY", "Yes", "No"),
        ("RF target + GOSIF", "RF_TARGET_GOSIF", "Yes", "Yes"),
        ("XGB target only", "XGB_TARGET", "No", "Yes"),
        ("XGB GOSIF only", "XGB_GOSIF_ONLY", "Yes", "No"),
        ("XGB target + GOSIF", "XGB_TARGET_GOSIF", "Yes", "Yes"),
        ("CatBoost target only", "CAT_TARGET", "No", "Yes"),
        ("CatBoost GOSIF only", "CAT_GOSIF_ONLY", "Yes", "No"),
        ("CatBoost target + GOSIF", "CAT_TARGET_GOSIF", "Yes", "Yes"),
        ("Statistical RBF target only", "STAT_RBF_TARGET_ONLY", "No", "Yes"),
        ("Partial-observation baseline", "Partial_observation_baseline", "No", "Yes"),
        ("Raw GOSIF baseline", "GOSIF_raw_baseline", "Yes", "No"),
        ("Linear calibrated GOSIF", "GOSIF_linear_calibrated", "Yes", "No"),
    ]
    rows = []
    for label, model, uses_gosif, uses_target in labels:
        row = {"Model": label, "model_token": model, "GOSIF?": uses_gosif, "Target SIF?": uses_target}
        for validation, col in [
            ("grouped_artificial_gap", "Grouped RMSE"),
            ("spatial_site_held_out", "Spatial RMSE"),
            ("incremental_temporal", "Temporal RMSE"),
            ("spatial_temporal_stress", "Stress RMSE"),
            ("empirical_coverage_validation", "Empirical-matched RMSE"),
        ]:
            sub = overall[(overall["validation"] == validation) & (overall["model"] == model)]
            row[col] = float(sub["RMSE"].iloc[0]) if len(sub) else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(ABLATION_CSV, index=False)
    return out


def target_identity_table(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (validation, model), g in pred.groupby(["validation", "model"]):
        if model in ["Partial_observation_baseline", "GOSIF_raw_baseline", "GOSIF_linear_calibrated"]:
            continue
        p = pd.to_numeric(g["prediction"], errors="coerce")
        y = pd.to_numeric(g[TARGET_COLUMN], errors="coerce")
        partial = pd.to_numeric(g["partial_OCO3_EC_weighted_SIF"], errors="coerce")
        gosif = pd.to_numeric(g["GOSIF_EC_weighted"], errors="coerce")
        corr_true = p.corr(y)
        corr_partial = p.corr(partial)
        corr_gosif = p.corr(gosif)
        rows.append(
            {
                "validation": validation,
                "model": model,
                "n": len(g),
                "corr_prediction_true_OCO3": corr_true,
                "corr_prediction_partial_OCO3": corr_partial,
                "corr_prediction_GOSIF": corr_gosif,
                "TARGET_IDENTITY_DILUTION": "YES" if (pd.notna(corr_gosif) and pd.notna(corr_true) and corr_gosif > corr_true + 0.05 and "GOSIF_ONLY" not in model) else "NO",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_ROOT / "OCO3_Target_Sensor_Identity_Audit.csv", index=False)
    return out


def choose_final_model(overall: pd.DataFrame, inc: pd.DataFrame) -> tuple[str, str]:
    candidate_models = ["RF_TARGET_GOSIF", "RF_TARGET", "XGB_TARGET_GOSIF", "XGB_TARGET"]
    vals = ["grouped_artificial_gap", "spatial_site_held_out", "incremental_temporal", "spatial_temporal_stress", "empirical_coverage_validation"]
    scores = []
    for model in candidate_models:
        sub = overall[overall["model"] == model]
        if sub.empty:
            continue
        rmse_vals = []
        for v in vals:
            x = sub[sub["validation"] == v]
            if len(x):
                rmse_vals.append(float(x["RMSE"].iloc[0]))
        if len(rmse_vals) >= 3:
            scores.append({"model": model, "mean_validation_RMSE": float(np.mean(rmse_vals)), "max_validation_RMSE": float(np.max(rmse_vals))})
    score_df = pd.DataFrame(scores).sort_values(["mean_validation_RMSE", "max_validation_RMSE"])
    if score_df.empty:
        return "RF_TARGET", "Fallback: no candidate comparison table was available."
    best = score_df.iloc[0]["model"]
    # Prefer RF over XGB if practically tied, for stability and interpretability.
    rf_g = score_df[score_df["model"] == "RF_TARGET_GOSIF"]
    if len(rf_g):
        best_mean = float(score_df.iloc[0]["mean_validation_RMSE"])
        rf_mean = float(rf_g["mean_validation_RMSE"].iloc[0])
        if rf_mean <= best_mean + 0.01:
            best = "RF_TARGET_GOSIF"
    reason = f"Selected by cross-validation stability/simplicity: {score_df.to_dict(orient='records')}"
    return str(best), reason


def tune_final_production_params(df: pd.DataFrame, final_spec: ModelSpec) -> tuple[dict[str, Any], float, pd.DataFrame]:
    best_params, score, rows = tune_model(final_spec, df, "final_all_artificial_gap_internal_cv", "production_lock")
    out = pd.DataFrame(rows)
    return best_params, score, out


def conceptual_feature_groups(spec: ModelSpec) -> dict[str, list[str]]:
    return {
        "partial target SIF": [c for c in ["partial_OCO3_EC_weighted_SIF", "partial_OCO3_mean", "partial_OCO3_sd", "partial_OCO3_min", "partial_OCO3_max"] if c in spec.features],
        "coverage/sampling": [c for c in ["artificial_coverage_probability", "artificial_coverage_pixel_fraction", "missing_probability_fraction", "n_original_target_footprints", "n_remaining_target_footprints"] if c in spec.features],
        "GOSIF": [c for c in GOSIF_FEATURES if c in spec.features],
        "space": [c for c in ["lat", "lon"] if c in spec.features],
        "time": [c for c in ["year", "month", "DOY", "sin_DOY", "cos_DOY"] if c in spec.features],
        "ecosystem/IGBP": ["IGBP"],
    }


def feature_importance_grouped(df: pd.DataFrame, spec: ModelSpec, params: dict[str, Any]) -> pd.DataFrame:
    rows = []
    splits = split_group_kfold(df, "original_case_ID", 5)
    groups = conceptual_feature_groups(spec)
    for fold_id, tr_idx, te_idx in splits:
        train = valid_model_data(df.iloc[tr_idx].copy(), spec)
        test = valid_model_data(df.iloc[te_idx].copy(), spec)
        est = build_estimator(spec, params, seed=RANDOM_SEED + 303)
        est.fit(feature_frame(train, spec), train[TARGET_COLUMN].astype(float).to_numpy())
        base_pred = est.predict(feature_frame(test, spec))
        base_mae = mean_absolute_error(test[TARGET_COLUMN].astype(float).to_numpy(), base_pred)
        rng = stable_rng("permutation", fold_id)
        for group_name, cols in groups.items():
            cols = [c for c in cols if c in test.columns]
            if not cols:
                continue
            perm = test.copy()
            order = rng.permutation(len(perm))
            for col in cols:
                perm[col] = perm[col].to_numpy()[order]
            pred = est.predict(feature_frame(perm, spec))
            mae = mean_absolute_error(test[TARGET_COLUMN].astype(float).to_numpy(), pred)
            rows.append(
                {
                    "model": spec.model,
                    "fold_id": fold_id,
                    "feature_group": group_name,
                    "columns_permuted": ", ".join(cols),
                    "baseline_MAE": base_mae,
                    "permuted_MAE": mae,
                    "MAE_increase": mae - base_mae,
                }
            )
    out = pd.DataFrame(rows)
    summary = out.groupby(["model", "feature_group"], as_index=False).agg(
        mean_MAE_increase=("MAE_increase", "mean"),
        median_MAE_increase=("MAE_increase", "median"),
        n_folds=("MAE_increase", "count"),
        columns_permuted=("columns_permuted", "first"),
    )
    summary.to_csv(FEATURE_IMPORTANCE_CSV, index=False)
    return summary


def conformal_uncertainty(df: pd.DataFrame, validation_splits: dict[str, list[tuple[str, np.ndarray, np.ndarray]]], spec: ModelSpec, params_by_validation_fold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for validation, splits in validation_splits.items():
        for fold_id, tr_idx, te_idx in splits:
            outer_train = valid_model_data(df.iloc[tr_idx].copy(), spec)
            test = valid_model_data(df.iloc[te_idx].copy(), spec)
            if len(outer_train) < 50 or len(test) < 10:
                continue
            groups = outer_train["original_case_ID"].astype(str)
            unique_groups = groups.nunique()
            if unique_groups >= 4:
                gss = GroupShuffleSplit(n_splits=1, test_size=0.22, random_state=RANDOM_SEED)
                fit_idx, calib_idx = next(gss.split(outer_train, groups=groups))
                fit_df = outer_train.iloc[fit_idx].copy()
                calib_df = outer_train.iloc[calib_idx].copy()
            else:
                split = int(round(0.78 * len(outer_train)))
                fit_df = outer_train.iloc[:split].copy()
                calib_df = outer_train.iloc[split:].copy()
            if len(calib_df) < 10:
                continue
            params_row = params_by_validation_fold[
                (params_by_validation_fold["validation"] == validation)
                & (params_by_validation_fold["fold_id"] == fold_id)
                & (params_by_validation_fold["model"] == spec.model)
                & (params_by_validation_fold["config_index"].astype(str) == "SELECTED")
            ]
            params = parse_params(params_row["params"].iloc[0]) if len(params_row) else config_grid(spec.family)[0]
            est = build_estimator(spec, params, seed=RANDOM_SEED + 404)
            est.fit(feature_frame(fit_df, spec), fit_df[TARGET_COLUMN].astype(float).to_numpy())
            calib_pred = est.predict(feature_frame(calib_df, spec))
            calib_abs = np.abs(calib_pred - calib_df[TARGET_COLUMN].astype(float).to_numpy())
            pred = est.predict(feature_frame(test, spec))
            y = test[TARGET_COLUMN].astype(float).to_numpy()
            for nominal in [0.90, 0.95]:
                q = conformal_quantile(calib_abs, nominal)
                lo = pred - q
                hi = pred + q
                covered = (y >= lo) & (y <= hi)
                for group_label, sub_idx in [("all", np.arange(len(test)))] + [
                    (lab, np.flatnonzero(test["artificial_coverage_probability"].map(low_medium_high).to_numpy() == lab))
                    for lab in ["low_<=0.25", "medium_>0.25-0.75", "high_>0.75"]
                ]:
                    if len(sub_idx) == 0:
                        continue
                    rows.append(
                        {
                            "validation": validation,
                            "fold_id": fold_id,
                            "model": spec.model,
                            "nominal_interval": nominal,
                            "coverage_group": group_label,
                            "n_test": int(len(sub_idx)),
                            "empirical_coverage_probability": float(np.mean(covered[sub_idx])),
                            "mean_interval_width": float(np.mean((hi - lo)[sub_idx])),
                            "median_interval_width": float(np.median((hi - lo)[sub_idx])),
                            "calibration_n": int(len(calib_abs)),
                            "calibration_quantile_abs_error": float(q),
                            "method": "split/grouped conformal absolute-residual interval; calibration uses only outer-training data",
                        }
                    )
    out = pd.DataFrame(rows)
    out.to_csv(UNCERTAINTY_CSV, index=False)
    return out


def conformal_quantile(abs_residuals: np.ndarray, nominal: float) -> float:
    vals = np.asarray(abs_residuals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan
    alpha = 1 - nominal
    q_level = min(1.0, math.ceil((len(vals) + 1) * (1 - alpha)) / len(vals))
    return float(np.quantile(vals, q_level, method="higher"))


def parse_params(text: Any) -> dict[str, Any]:
    import ast

    try:
        return ast.literal_eval(str(text))
    except Exception:
        return {}


def software_versions() -> str:
    parts = [
        f"python={platform.python_version()}",
        f"pandas={pd.__version__}",
        f"numpy={np.__version__}",
        f"sklearn={__import__('sklearn').__version__}",
    ]
    if XGBOOST_AVAILABLE:
        import xgboost

        parts.append(f"xgboost={xgboost.__version__}")
    parts.append("catboost=" + (__import__("catboost").__version__ if CATBOOST_AVAILABLE else "NOT_AVAILABLE"))
    return "; ".join(parts)


def zero_coverage_interpretation(zero_perf: pd.DataFrame) -> str:
    if zero_perf.empty:
        return "ZERO-COVERAGE CASES SHOULD BE EXCLUDED FROM PRIMARY RECONSTRUCTION"
    best = zero_perf.sort_values("RMSE").iloc[0]
    gosif = zero_perf[zero_perf["model"] == "GOSIF_raw_baseline"]
    if len(gosif) and float(best["RMSE"]) < float(gosif["RMSE"].iloc[0]) * 0.90 and float(best["R2"]) > 0.45:
        return "ZERO-COVERAGE CASES MAY BE INCLUDED ONLY AS EXTERNAL-ESTIMATE SENSITIVITY ANALYSIS"
    return "ZERO-COVERAGE CASES SHOULD BE EXCLUDED FROM PRIMARY RECONSTRUCTION"


def eligibility_rule(by_cov: pd.DataFrame, final_model: str, zero_policy: str) -> str:
    sub = by_cov[(by_cov["validation"] == "grouped_artificial_gap") & (by_cov["model"] == final_model)].copy()
    low = sub[sub["coverage_class"].isin(["0-0.10", ">0.10-0.25"])]
    if zero_policy.startswith("ZERO-COVERAGE CASES SHOULD"):
        return "Primary reconstruction eligibility: OCO-3 EC coverage must be > 0. No additional minimum threshold above zero is justified by the artificial-gap tests unless uncertainty intervals are excessively wide for a specific case."
    if not low.empty and low["RMSE"].max() < 0.22:
        return "Primary reconstruction eligibility: coverage >= 0 may be acceptable, but zero-coverage outputs must be labelled external/context-based."
    return "Primary reconstruction eligibility: require OCO-3 EC coverage > 0; review very-low-coverage cases with uncertainty flags."


def make_figures(
    ablation: pd.DataFrame,
    by_cov: pd.DataFrame,
    zero_perf: pd.DataFrame,
    uncertainty: pd.DataFrame,
    empirical_dist: pd.DataFrame,
    final_model: str,
    readiness: str,
) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    # Simple required visual summary.
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    axes = axes.ravel()
    rf_models = ["RF target only", "RF GOSIF only", "RF target + GOSIF"]
    rf = ablation[ablation["Model"].isin(rf_models)].set_index("Model")
    if not rf.empty:
        axes[0].bar(rf.index, rf["Grouped RMSE"], color=["#6baed6", "#9ecae1", "#2171b5"])
        axes[0].set_title("A. Same Random Forest family: grouped artificial gaps")
        axes[0].set_ylabel("RMSE lower is better")
        axes[0].tick_params(axis="x", rotation=20)

        x = np.arange(len(rf.index))
        w = 0.35
        axes[1].bar(x - w / 2, rf["Spatial RMSE"], width=w, label="Unseen sites", color="#74c476")
        axes[1].bar(x + w / 2, rf["Temporal RMSE"], width=w, label="Future years", color="#fd8d3c")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(rf.index, rotation=20)
        axes[1].set_title("B. Spatial and temporal validation")
        axes[1].set_ylabel("RMSE")
        axes[1].legend()

    cov_sub = by_cov[(by_cov["validation"] == "grouped_artificial_gap") & (by_cov["model"].isin([final_model, "Partial_observation_baseline"]))]
    order = ["0-0.10", ">0.10-0.25", ">0.25-0.50", ">0.50-0.75", ">0.75-0.90", ">0.90-<1.00", ">0.75-<1"]
    cov_sub = cov_sub[cov_sub["coverage_class"].isin(order)].copy()
    if not cov_sub.empty:
        pivot = cov_sub.pivot_table(index="coverage_class", columns="model", values="RMSE", aggfunc="first")
        pivot = pivot.reindex([c for c in order if c in pivot.index])
        pivot.plot(kind="bar", ax=axes[2], color=["#636363", "#3182bd"])
        axes[2].set_title("C. Performance versus OCO-3 EC coverage")
        axes[2].set_ylabel("RMSE")
        axes[2].tick_params(axis="x", rotation=35)

    if not zero_perf.empty:
        zp = zero_perf.sort_values("RMSE").head(5)
        axes[3].bar(zp["model"], zp["RMSE"], color="#756bb1")
        axes[3].set_title("D. Zero OCO-3 coverage: external/context estimate only")
        axes[3].set_ylabel("RMSE")
        axes[3].tick_params(axis="x", rotation=25)

    if not uncertainty.empty:
        u = uncertainty[(uncertainty["coverage_group"] == "all") & (uncertainty["validation"].isin(["grouped_artificial_gap", "spatial_site_held_out", "incremental_temporal"]))]
        if not u.empty:
            labels = u["validation"] + " " + (u["nominal_interval"] * 100).astype(int).astype(str) + "%"
            axes[4].bar(labels, u["empirical_coverage_probability"], color="#31a354")
            axes[4].axhline(0.90, color="black", linestyle="--", linewidth=1)
            axes[4].axhline(0.95, color="black", linestyle=":", linewidth=1)
            axes[4].set_ylim(0, 1.05)
            axes[4].set_title("E. Conformal interval calibration")
            axes[4].set_ylabel("Empirical interval coverage")
            axes[4].tick_params(axis="x", rotation=35)

    axes[5].axis("off")
    color = "#31a354" if readiness.startswith("READY") else "#de2d26"
    axes[5].text(0.05, 0.75, "F. Final pre-production decision", fontsize=16, weight="bold")
    axes[5].text(0.05, 0.55, readiness, fontsize=20, weight="bold", color=color)
    axes[5].text(0.05, 0.35, f"Locked model candidate: {final_model}", fontsize=14)
    axes[5].text(0.05, 0.20, "No real incomplete archive reconstruction was run.", fontsize=12)
    fig.tight_layout()
    fig.savefig(SIMPLE_SUMMARY_PNG, dpi=180)
    fig.savefig(SIMPLE_SUMMARY_PDF)
    plt.close(fig)

    # Distribution comparison figure.
    fig, ax = plt.subplots(figsize=(10, 6))
    if EMPIRICAL_MATCHED_DATASET_CSV.exists():
        matched = pd.read_csv(EMPIRICAL_MATCHED_DATASET_CSV)
        gap = pd.read_csv(GAP_DATASET)
        ax.hist(gap["artificial_coverage_probability"], bins=np.linspace(0, 1, 31), alpha=0.5, label="Original artificial gaps")
        ax.hist(matched["artificial_coverage_probability"], bins=np.linspace(0, 1, 31), alpha=0.5, label="Empirical-matched + zero")
        ax.set_xlabel("OCO-3 EC probability coverage")
        ax.set_ylabel("Count")
        ax.legend()
        ax.set_title("Artificial-gap coverage realism check")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "coverage_distribution_artificial_vs_empirical.png", dpi=180)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df is None or df.empty:
        return "_No rows._"
    return df.head(max_rows).to_markdown(index=False)


def write_pdf_from_markdown(md_path: Path, pdf_path: Path) -> None:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

        styles = getSampleStyleSheet()
        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        story = []
        for line in md_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("# "):
                story.append(Paragraph(line[2:], styles["Title"]))
                story.append(Spacer(1, 8))
            elif line.startswith("## "):
                story.append(Paragraph(line[3:], styles["Heading2"]))
            elif line.startswith("### "):
                story.append(Paragraph(line[4:], styles["Heading3"]))
            elif line.strip() == "":
                story.append(Spacer(1, 6))
            else:
                escaped = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                story.append(Paragraph(escaped, styles["BodyText"]))
        doc.build(story)
    except Exception:
        pdf_path.write_text("PDF generation failed; see Markdown report.\n", encoding="utf-8")


def write_reports(
    readiness: str,
    final_model: str,
    final_reason: str,
    final_params: dict[str, Any],
    production_inner_score: float,
    zero_policy: str,
    eligibility: str,
    regime: pd.DataFrame,
    empirical_dist: pd.DataFrame,
    ablation: pd.DataFrame,
    inc: pd.DataFrame,
    zero_perf: pd.DataFrame,
    by_cov: pd.DataFrame,
    identity: pd.DataFrame,
    uncertainty: pd.DataFrame,
    feature_importance: pd.DataFrame,
    input_audit_after: pd.DataFrame,
) -> None:
    grouped_best = ablation.sort_values("Grouped RMSE").head(1)
    rf_inc = inc[(inc["Model family"] == "RF") & (inc["Validation"] == "grouped_artificial_gap")]
    xgb_inc = inc[(inc["Model family"] == "XGBoost") & (inc["Validation"] == "grouped_artificial_gap")]
    cat_answer = "CatBoost was not available; a safe install attempt timed out and no substitute model was fabricated."
    if CATBOOST_AVAILABLE:
        cat_answer = "CatBoost was available and included in the same-model ablation."
    md = []
    md.append("# OCO-3 Final Pre-Production Validation\n")
    md.append(f"Final status: **{readiness}**\n")
    md.append("This validation did not reconstruct the real incomplete OCO-3 archive, did not write SIF rasters, and did not use tower GPP as a predictor.\n")
    md.append("## Data reused\n")
    md.append("- Complete OCO-3 + GOSIF truth cases from POC: 509.\n")
    md.append("- Existing artificial-gap rows reused: 2,407.\n")
    md.append("- Mandatory zero-coverage rows generated from complete truth cases: 509.\n")
    md.append("- Empirical-matched validation rows sampled from whole-footprint gap rows plus zero rows.\n")
    md.append("## Real OCO-3 coverage regimes\n")
    md.append(markdown_table(regime, 20) + "\n")
    md.append("## Empirical coverage realism\n")
    md.append(markdown_table(empirical_dist, 10) + "\n")
    md.append("## Clean same-model ablation\n")
    md.append(markdown_table(ablation, 20) + "\n")
    md.append("## GOSIF incremental value\n")
    md.append(markdown_table(inc, 40) + "\n")
    md.append("## Zero-coverage validation\n")
    md.append(markdown_table(zero_perf, 20) + "\n")
    md.append(f"Zero-coverage policy: **{zero_policy}**\n")
    md.append("## Coverage-specific performance\n")
    cov = by_cov[(by_cov["validation"] == "grouped_artificial_gap") & (by_cov["model"].isin([final_model, "Partial_observation_baseline", "GOSIF_raw_baseline"]))]
    md.append(markdown_table(cov.sort_values(["coverage_class", "RMSE"]), 40) + "\n")
    md.append("## Target identity audit\n")
    md.append(markdown_table(identity[identity["validation"].isin(["grouped_artificial_gap", "spatial_site_held_out", "incremental_temporal"])], 40) + "\n")
    md.append("## Feature importance for final model\n")
    md.append(markdown_table(feature_importance.sort_values("mean_MAE_increase", ascending=False), 20) + "\n")
    md.append("## Conformal uncertainty calibration\n")
    md.append(markdown_table(uncertainty, 40) + "\n")
    md.append("## Statistical baseline audit\n")
    md.append("The previous `Approx_GP_target_only` style baseline is retained here as `STAT_RBF_TARGET_ONLY`: standardized target/context predictors are mapped through RBF Nystroem features and fitted with Ridge regression. This is a computationally reasonable Gaussian-process-equivalent kernel baseline, but it is **not** exact kriging and is not labelled as such. It is scientifically defensible as a target-only smooth spatiotemporal baseline, but it should not be presented as formal geostatistical kriging.\n")
    md.append("## Model lock\n")
    md.append(f"- FINAL MODEL FAMILY: `{final_model}`\n")
    md.append(f"- FINAL HYPERPARAMETERS: `{final_params}`\n")
    md.append(f"- Internal-CV RMSE for production-lock tuning: `{production_inner_score:.6f}`\n")
    md.append(f"- Selection reason: {final_reason}\n")
    md.append(f"- FINAL ELIGIBILITY RULE: {eligibility}\n")
    md.append(f"- ZERO-COVERAGE POLICY: {zero_policy}\n")
    md.append("- FINAL UNCERTAINTY METHOD: split/grouped conformal absolute-residual intervals at 90% and 95%, calibrated only within outer-training data.\n")
    md.append("## Final questions\n")
    q1 = rf_inc["Conclusion"].iloc[0] if len(rf_inc) else "not evaluated"
    q2 = xgb_inc["Conclusion"].iloc[0] if len(xgb_inc) else "not evaluated"
    md.append(f"1. Does GOSIF improve Random Forest when architecture/folds/non-GOSIF predictors are held constant? **{q1}**.\n")
    md.append(f"2. Does GOSIF improve XGBoost under the same controlled comparison? **{q2}**.\n")
    md.append(f"3. Does CatBoost provide additional value? **{cat_answer}**\n")
    md.append("4. Does the model still outperform the partial baseline under real-coverage-matched validation? See `Empirical-matched RMSE` in the ablation table.\n")
    md.append("5. Low-coverage performance is reported in the coverage-specific table; it remains explicitly separated.\n")
    md.append(f"6. At zero coverage: **{zero_policy}**\n")
    md.append(f"7. Minimum coverage rule: {eligibility}\n")
    md.append("8. Target-sensor identity: see audit table; target+GOSIF is flagged if prediction behaves more like GOSIF than true OCO-3.\n")
    md.append("9. Statistical baseline: defensible approximate RBF GP-equivalent, not exact kriging.\n")
    md.append("10. Uncertainty: conformal intervals are evaluated; use table above for empirical calibration.\n")
    md.append(f"11. Frozen model and features are in `{MODEL_LOCK_MD.name}`.\n")
    md.append(f"12. Final decision: **{readiness}**\n")
    md.append("## Input/source safety audit\n")
    unchanged = bool(input_audit_after["unchanged_size_and_mtime"].all()) if "unchanged_size_and_mtime" in input_audit_after else False
    md.append(f"Input CSV audit unchanged by size/mtime: `{unchanged}`. Raw SIF and GPP TIFF files were not opened for writing by this script.\n")
    REPORT_MD.write_text("\n".join(md), encoding="utf-8")
    write_pdf_from_markdown(REPORT_MD, REPORT_PDF)

    lock = [
        "# OCO-3 Final Model Lock",
        "",
        f"FINAL STATUS: {readiness}",
        "",
        f"FINAL MODEL FAMILY: {final_model}",
        f"FINAL FEATURE SET: {', '.join(model_spec_by_name(final_model).features)} plus categorical IGBP",
        f"FINAL HYPERPARAMETERS: {final_params}",
        f"FINAL TRAINING PROCEDURE: train on all eligible artificial-gap validation data with grouped internal CV using original_case_ID; use the locked modest search space and selected hyperparameters above.",
        "FINAL VALIDATION DESIGN: grouped original_case_ID, spatial SITE_ID-held-out, expanding-window temporal, spatial-temporal stress, empirical-coverage-matched validation, and zero-coverage external-estimate validation.",
        "FINAL UNCERTAINTY METHOD: split/grouped conformal absolute-residual intervals at 90% and 95%.",
        f"FINAL ELIGIBILITY RULE: {eligibility}",
        f"ZERO-COVERAGE POLICY: {zero_policy}",
        "GPP predictor usage: FORBIDDEN; tower GPP must not enter reconstruction.",
        "Production note: do not create fine-resolution reconstructed SIF rasters; predict one EC-footprint representative SIF value per eligible site-month.",
    ]
    MODEL_LOCK_MD.write_text("\n".join(lock), encoding="utf-8")

    readme = [
        "OCO-3 Final Pre-Production Validation — READ ME FIRST",
        "",
        f"1. Did GOSIF genuinely improve the SAME Random Forest model? {q1}.",
        f"2. Did it also improve XGBoost/CatBoost? XGBoost: {q2}. CatBoost: {cat_answer}",
        f"3. Which final model should we use? {final_model}.",
        f"4. Did tuning materially change the result? Tuned results are in {ABLATION_CSV.name}; selected production-lock params are {final_params}.",
        "5. Does the method still work under the real OCO-3 coverage distribution? See Empirical-matched RMSE in the ablation table and OCO3_Empirical_Coverage_Validation.csv.",
        f"6. Can we estimate months with zero OCO-3 coverage? {zero_policy}.",
        f"7. What minimum coverage should we require? {eligibility}",
        "8. Is uncertainty calibrated? Split/grouped conformal 90% and 95% intervals were evaluated; see OCO3_Final_Uncertainty_Calibration.csv.",
        "9. Does the model preserve OCO-3 identity? See OCO3_Target_Sensor_Identity_Audit.csv.",
        f"10. Are we ready for full real OCO-3 reconstruction? {readiness}.",
    ]
    README_TXT.write_text("\n".join(readme), encoding="utf-8")


def model_spec_by_name(name: str) -> ModelSpec:
    for s in model_specs():
        if s.model == name:
            return s
    raise KeyError(name)


def main() -> None:
    ensure_dirs()
    print("Starting OCO-3 final pre-production validation.", flush=True)
    input_paths = [GAP_DATASET, TRUTH_DATASET, REAL_COVERAGE]
    audit_before = audit_input_files(input_paths, "before")

    gap, truth, real = load_inputs()
    consistency = validate_gap_dataset(gap, truth)
    if (consistency["status"] == "FAIL").any():
        raise RuntimeError("Input consistency audit failed; see qc/OCO3_Input_Consistency_Audit.csv")

    real_oco3, regime = real_coverage_regime_table(real)
    zero = create_zero_coverage_dataset(truth)
    empirical, empirical_dist = create_empirical_matched_dataset(real_oco3, gap, zero)
    print(f"Datasets: main_gaps={len(gap)}, zero={len(zero)}, empirical_matched={len(empirical)}", flush=True)

    specs_main = model_specs(include_statistical=True)
    specs_zero = [s for s in model_specs(include_statistical=False) if s.supports_zero_coverage]

    all_predictions = []
    all_params = []
    validation_frames: dict[str, pd.DataFrame] = {
        "grouped_artificial_gap": gap,
        "spatial_site_held_out": gap,
        "incremental_temporal": gap,
        "spatial_temporal_stress": gap,
        "empirical_coverage_validation": empirical,
    }
    validation_splits = {
        "grouped_artificial_gap": split_group_kfold(gap, "original_case_ID", OUTER_SPLITS),
        "spatial_site_held_out": split_group_kfold(gap, "SITE_ID", OUTER_SPLITS),
        "incremental_temporal": split_incremental_temporal(gap),
        "spatial_temporal_stress": split_spatial_temporal_stress(gap),
        "empirical_coverage_validation": split_group_kfold(empirical, "original_case_ID", OUTER_SPLITS),
    }
    for validation, frame in validation_frames.items():
        preds, params = evaluate_validation("main", frame, validation, validation_splits[validation], specs_main, include_partial_baseline=True)
        all_predictions.append(preds)
        all_params.append(params)
        pd.concat(all_predictions, ignore_index=True).to_csv(PREDICTIONS_CSV, index=False)
        pd.concat(all_params, ignore_index=True).to_csv(HYPERPARAMS_CSV, index=False)

    zero_splits = split_group_kfold(zero, "original_case_ID", OUTER_SPLITS)
    zero_preds, zero_params = evaluate_validation("zero_coverage", zero, "zero_coverage_validation", zero_splits, specs_zero, include_partial_baseline=False)
    all_predictions.append(zero_preds)
    all_params.append(zero_params)
    predictions = pd.concat(all_predictions, ignore_index=True)
    params = pd.concat(all_params, ignore_index=True)
    predictions.to_csv(PREDICTIONS_CSV, index=False)
    params.to_csv(HYPERPARAMS_CSV, index=False)

    overall, by_cov = summarize_predictions(predictions)
    ablation = ablation_table(overall)
    inc = gosif_incremental_table(predictions, overall)
    inc_cov = gosif_increment_by_coverage(predictions)
    identity = target_identity_table(predictions)
    zero_perf = overall[overall["validation"] == "zero_coverage_validation"].sort_values("RMSE").copy()
    zero_perf["Interpretation"] = np.where(
        zero_perf["model"].str.contains("GOSIF_ONLY|GOSIF", regex=True),
        "External/context-based OCO-3 estimate; no target OCO-3 observation exists in the EC footprint.",
        "Not applicable to zero target coverage.",
    )
    zero_perf.to_csv(ZERO_COVERAGE_CSV, index=False)

    final_model, final_reason = choose_final_model(overall, inc)
    final_spec = model_spec_by_name(final_model)
    final_params, final_score, final_tune_rows = tune_final_production_params(gap, final_spec)
    final_tune_rows.to_csv(OUTPUT_ROOT / "OCO3_Final_Production_Lock_Internal_Tuning_Detail.csv", index=False)
    feature_importance = feature_importance_grouped(gap, final_spec, final_params)

    # Use validation splits that belong to the main gap data for calibrated uncertainty.
    uncertainty_splits = {k: v for k, v in validation_splits.items() if k != "empirical_coverage_validation"}
    uncertainty = conformal_uncertainty(gap, uncertainty_splits, final_spec, params)

    zero_policy = zero_coverage_interpretation(zero_perf)
    eligibility = eligibility_rule(by_cov, final_model, zero_policy)

    # Readiness rule: final model must beat partial baseline in grouped, spatial, temporal, empirical; uncertainty file must exist; no identity dilution.
    required_vals = ["grouped_artificial_gap", "spatial_site_held_out", "incremental_temporal", "empirical_coverage_validation"]
    ready = True
    reasons = []
    for val in required_vals:
        fm = overall[(overall["validation"] == val) & (overall["model"] == final_model)]
        pb = overall[(overall["validation"] == val) & (overall["model"] == "Partial_observation_baseline")]
        if fm.empty or pb.empty or float(fm["RMSE"].iloc[0]) >= float(pb["RMSE"].iloc[0]):
            ready = False
            reasons.append(f"{final_model} does not beat partial baseline for {val}")
    id_rows = identity[(identity["model"] == final_model) & (identity["TARGET_IDENTITY_DILUTION"] == "YES")]
    if len(id_rows):
        ready = False
        reasons.append("target identity dilution was flagged")
    if uncertainty.empty:
        ready = False
        reasons.append("uncertainty calibration could not be evaluated")
    readiness = "READY FOR FULL OCO-3 RECONSTRUCTION" if ready else "NOT READY FOR FULL OCO-3 RECONSTRUCTION"
    if reasons:
        (OUTPUT_ROOT / "OCO3_Final_Readiness_Blockers.txt").write_text("\n".join(reasons), encoding="utf-8")

    input_after = audit_input_files(input_paths, "after")
    merged_audit = audit_before.merge(input_after, on="path", suffixes=("_before", "_after"))
    merged_audit["unchanged_size_and_mtime"] = (
        (merged_audit["size_bytes_before"] == merged_audit["size_bytes_after"])
        & (merged_audit["mtime_ns_before"] == merged_audit["mtime_ns_after"])
        & (merged_audit["exists_before"] == merged_audit["exists_after"])
    )
    merged_audit.to_csv(INPUT_AUDIT_CSV, index=False)

    make_figures(ablation, by_cov, zero_perf, uncertainty, empirical_dist, final_model, readiness)
    write_reports(
        readiness,
        final_model,
        final_reason,
        final_params,
        final_score,
        zero_policy,
        eligibility,
        regime,
        empirical_dist,
        ablation,
        inc,
        zero_perf,
        by_cov,
        identity,
        uncertainty,
        feature_importance,
        merged_audit,
    )

    print("=" * 60, flush=True)
    print("OCO-3 final pre-production validation complete", flush=True)
    print(f"Readiness: {readiness}", flush=True)
    print(f"Final model: {final_model}", flush=True)
    print(f"Final params: {final_params}", flush=True)
    print(f"Main report: {REPORT_MD}", flush=True)
    print(f"PDF report: {REPORT_PDF}", flush=True)
    print(f"Simple summary: {SIMPLE_SUMMARY_PNG}", flush=True)
    print("No real incomplete OCO-3 reconstruction was performed.", flush=True)


if __name__ == "__main__":
    main()
