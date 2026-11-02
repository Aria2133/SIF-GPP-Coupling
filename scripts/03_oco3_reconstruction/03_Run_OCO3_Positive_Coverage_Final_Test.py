"""Final test of the locked OCO-3 reconstruction model on positive partial-coverage cases (grouped, site-held-out and temporal validation, bootstrap intervals). Tower GPP is not used.

Output: model-lock record and performance tables across validation designs."""
from __future__ import annotations
# Original script filename (sanitized): Run_OCO3_Positive_Coverage_Final_Test.py; public name: 03_Run_OCO3_Positive_Coverage_Final_Test.py
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
import platform
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


RANDOM_SEED = 20260812
TARGET_COLUMN = "true_complete_OCO3_EC_SIF"
UNCHANGED_TOLERANCE = 0.001
OUTER_SPLITS = 5
N_BOOTSTRAP = 5000

PROJECT_ROOT = Path(
    PROJECT_ROOT
)
POC_ROOT = PROJECT_ROOT / "SIF_EC_Reconstruction_OCO3_Proof_of_Concept"
PREP_ROOT = PROJECT_ROOT / "SIF_EC_Reconstruction_Model_Preparation"
PREPROD_ROOT = PROJECT_ROOT / "SIF_EC_Reconstruction_OCO3_Final_PreProduction_Validation"
OUTPUT_ROOT = PROJECT_ROOT / "SIF_EC_Reconstruction_OCO3_Positive_Coverage_Final_Test"
FIG_DIR = OUTPUT_ROOT / "figures"
QC_DIR = OUTPUT_ROOT / "qc"

ARTIFICIAL_GAPS_CSV = POC_ROOT / "OCO3_Artificial_Gap_Dataset.csv"
REAL_COVERAGE_CSV = PREP_ROOT / "EC_Coverage_All_Products.csv"
PREVIOUS_PERFORMANCE_CSV = PREPROD_ROOT / "OCO3_Final_Model_Performance_All_Validations.csv"
MODEL_LOCK_MD = PREPROD_ROOT / "OCO3_Final_Model_Lock.md"

REAL_DISTRIBUTION_CSV = OUTPUT_ROOT / "OCO3_Positive_Coverage_Real_Distribution.csv"
PREDICTIONS_CSV = OUTPUT_ROOT / "OCO3_Positive_Coverage_Empirical_Matched_Predictions.csv"
PERFORMANCE_CSV = OUTPUT_ROOT / "OCO3_Positive_Coverage_Model_Performance.csv"
BY_COVERAGE_CSV = OUTPUT_ROOT / "OCO3_Positive_Coverage_Performance_By_Coverage_Class.csv"
PAIRED_CSV = OUTPUT_ROOT / "OCO3_Positive_Coverage_Paired_Error_Comparison.csv"
THRESHOLD_CSV = OUTPUT_ROOT / "OCO3_Positive_Coverage_Threshold_Sensitivity.csv"
UNCERTAINTY_CSV = OUTPUT_ROOT / "OCO3_Positive_Coverage_Uncertainty_Calibration.csv"
MATCHED_DATASET_CSV = OUTPUT_ROOT / "OCO3_Positive_Coverage_Empirical_Matched_Artificial_Gap_Dataset.csv"
MATCHING_AUDIT_CSV = OUTPUT_ROOT / "OCO3_Positive_Coverage_Empirical_Matching_Audit.csv"
PRODUCTION_CLASS_CSV = OUTPUT_ROOT / "OCO3_Positive_Coverage_Production_Class_Table.csv"
REPORT_MD = OUTPUT_ROOT / "OCO3_Positive_Coverage_Final_Test_Report.md"
REPORT_PDF = OUTPUT_ROOT / "OCO3_Positive_Coverage_Final_Test_Report.pdf"
README_TXT = OUTPUT_ROOT / "OCO3_Positive_Coverage_READ_ME_FIRST.txt"
SUMMARY_PNG = OUTPUT_ROOT / "OCO3_Positive_Coverage_Final_Summary.png"
SUMMARY_PDF = OUTPUT_ROOT / "OCO3_Positive_Coverage_Final_Summary.pdf"
INPUT_AUDIT_CSV = QC_DIR / "Input_File_Size_MTime_Audit.csv"

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

TARGET_GOSIF_FEATURES = TARGET_FEATURES + GOSIF_FEATURES
CATEGORICAL_FEATURES = ["IGBP"]

LOCKED_RF_PARAMS = {
    "n_estimators": 80,
    "max_depth": 8,
    "min_samples_leaf": 4,
    "max_features": "sqrt",
}

POSITIVE_CLASSES = ["0-0.10", ">0.10-0.25", ">0.25-0.50", ">0.50-0.75", ">0.75-<1"]


@dataclass(frozen=True)
class ModelSpec:
    model: str
    features: tuple[str, ...]


MODEL_SPECS = {
    "RF_TARGET": ModelSpec("RF_TARGET", tuple(TARGET_FEATURES)),
    "RF_TARGET_GOSIF": ModelSpec("RF_TARGET_GOSIF", tuple(TARGET_GOSIF_FEATURES)),
}


def ensure_dirs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    QC_DIR.mkdir(parents=True, exist_ok=True)


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def preprocessor(numeric_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                        ("onehot", one_hot_encoder()),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_locked_rf(features: list[str], seed: int) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", preprocessor(features)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=LOCKED_RF_PARAMS["n_estimators"],
                    max_depth=LOCKED_RF_PARAMS["max_depth"],
                    min_samples_leaf=LOCKED_RF_PARAMS["min_samples_leaf"],
                    max_features=LOCKED_RF_PARAMS["max_features"],
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def stable_rng(*parts: Any) -> np.random.Generator:
    text = "|".join(str(p) for p in parts)
    seed = RANDOM_SEED
    for ch in text:
        seed = (seed * 131 + ord(ch)) % (2**32 - 1)
    return np.random.default_rng(seed)


def finite_float(v: Any) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else np.nan
    except Exception:
        return np.nan


def positive_coverage_class(v: Any) -> str:
    x = finite_float(v)
    if not math.isfinite(x):
        return "NaN"
    if x <= 0 or x >= 1:
        return "outside_positive_incomplete"
    if x <= 0.10:
        return "0-0.10"
    if x <= 0.25:
        return ">0.10-0.25"
    if x <= 0.50:
        return ">0.25-0.50"
    if x <= 0.75:
        return ">0.50-0.75"
    return ">0.75-<1"


def audit_inputs(stage: str) -> pd.DataFrame:
    paths = [ARTIFICIAL_GAPS_CSV, REAL_COVERAGE_CSV, PREVIOUS_PERFORMANCE_CSV, MODEL_LOCK_MD]
    rows = []
    for path in paths:
        rows.append(
            {
                "audit_stage": stage,
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else np.nan,
                "mtime_ns": path.stat().st_mtime_ns if path.exists() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, seed_label: str, n_boot: int = N_BOOTSTRAP) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return (np.nan, np.nan)
    rng = stable_rng("bootstrap", seed_label)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    m = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[m]
    yp = yp[m]
    if len(yt) == 0:
        return {
            "N": 0,
            "R2": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "Bias": np.nan,
            "nRMSE": np.nan,
            "Pearson_r": np.nan,
            "Spearman_rho": np.nan,
        }
    rmse = math.sqrt(mean_squared_error(yt, yp))
    mae = mean_absolute_error(yt, yp)
    denom = float(np.nanmax(yt) - np.nanmin(yt))
    pear = stats.pearsonr(yt, yp).statistic if len(yt) > 2 and np.nanstd(yp) > 0 and np.nanstd(yt) > 0 else np.nan
    spear = stats.spearmanr(yt, yp).statistic if len(yt) > 2 and np.nanstd(yp) > 0 and np.nanstd(yt) > 0 else np.nan
    return {
        "N": int(len(yt)),
        "R2": float(r2_score(yt, yp)) if len(yt) > 1 else np.nan,
        "RMSE": float(rmse),
        "MAE": float(mae),
        "Bias": float(np.mean(yp - yt)),
        "nRMSE": float(rmse / denom) if denom > 0 else np.nan,
        "Pearson_r": float(pear) if math.isfinite(pear) else np.nan,
        "Spearman_rho": float(spear) if math.isfinite(spear) else np.nan,
    }


def load_and_check_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [p for p in [ARTIFICIAL_GAPS_CSV, REAL_COVERAGE_CSV, PREVIOUS_PERFORMANCE_CSV, MODEL_LOCK_MD] if not p.exists()]
    if missing:
        raise FileNotFoundError("Required input file(s) missing: " + "; ".join(map(str, missing)))
    gap = pd.read_csv(ARTIFICIAL_GAPS_CSV)
    real = pd.read_csv(REAL_COVERAGE_CSV)
    previous_perf = pd.read_csv(PREVIOUS_PERFORMANCE_CSV)

    required_gap = set(TARGET_GOSIF_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN, "original_case_ID", "SITE_ID", "YYYYMM"])
    missing_cols = sorted(required_gap.difference(gap.columns))
    if missing_cols:
        raise ValueError(f"Artificial gap dataset is missing required frozen feature columns: {missing_cols}")
    if "coverage_probability_fraction" not in real.columns:
        raise ValueError("Real coverage inventory lacks coverage_probability_fraction.")
    return gap, real, previous_perf


def build_real_positive_distribution(real: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    oco3 = real[real["PRODUCT"].astype(str).str.upper().isin(["OCO-3", "OCO3"])].copy()
    oco3["coverage_probability_fraction"] = pd.to_numeric(oco3["coverage_probability_fraction"], errors="coerce")
    oco3 = oco3[oco3["coverage_probability_fraction"].notna()].copy()
    tol = 1e-9
    oco3["production_class"] = np.select(
        [
            oco3["coverage_probability_fraction"] <= tol,
            (oco3["coverage_probability_fraction"] > tol) & (oco3["coverage_probability_fraction"] < 1 - tol),
            np.isclose(oco3["coverage_probability_fraction"], 1.0, atol=tol),
        ],
        ["Class C - No OCO3 support", "Class B - Partial", "Class A - Complete"],
        default="Pathological/outside 0-1",
    )
    class_action = {
        "Class A - Complete": "USE OBSERVED EC-WEIGHTED OCO-3 DIRECTLY",
        "Class B - Partial": "DECIDE FROM POSITIVE-COVERAGE FINAL TEST",
        "Class C - No OCO3 support": "EXCLUDE FROM PRIMARY OCO-3 RECONSTRUCTION/ANALYSIS",
        "Pathological/outside 0-1": "REVIEW",
    }
    production_rows = []
    for cls, sub in oco3.groupby("production_class", dropna=False):
        production_rows.append(
            {
                "production_class": cls,
                "N_site_months": len(sub),
                "percentage_of_all_oco3_site_months": 100 * len(sub) / len(oco3) if len(oco3) else np.nan,
                "N_sites": sub["SITE_ID"].nunique(),
                "action": class_action.get(cls, "REVIEW"),
            }
        )
    production_table = pd.DataFrame(production_rows).sort_values("production_class")
    production_table.to_csv(PRODUCTION_CLASS_CSV, index=False)

    positive = oco3[(oco3["coverage_probability_fraction"] > tol) & (oco3["coverage_probability_fraction"] < 1 - tol)].copy()
    positive["coverage_class"] = positive["coverage_probability_fraction"].map(positive_coverage_class)
    total_pos = len(positive)
    rows = []
    for cls in POSITIVE_CLASSES:
        sub = positive[positive["coverage_class"] == cls]
        rows.append(
            {
                "Coverage class": cls,
                "Real N": len(sub),
                "Percentage of positive-incomplete cases": 100 * len(sub) / total_pos if total_pos else np.nan,
                "N_sites": sub["SITE_ID"].nunique(),
                "years": ", ".join(map(str, sorted(pd.to_numeric(sub["year"], errors="coerce").dropna().astype(int).unique()))),
                "IGBP_classes": ", ".join(sorted(map(str, sub.get("IGBP", pd.Series(dtype=str)).dropna().unique()))),
            }
        )
    dist = pd.DataFrame(rows)
    dist.to_csv(REAL_DISTRIBUTION_CSV, index=False)
    return oco3, positive, dist


def build_positive_empirical_matched_dataset(gap: pd.DataFrame, real_dist: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pool = gap.copy()
    pool["artificial_coverage_probability"] = pd.to_numeric(pool["artificial_coverage_probability"], errors="coerce")
    pool = pool[(pool["artificial_coverage_probability"] > 1e-9) & (pool["artificial_coverage_probability"] < 1 - 1e-9)].copy()
    pool["coverage_class"] = pool["artificial_coverage_probability"].map(positive_coverage_class)
    pool = pool[pool["coverage_class"].isin(POSITIVE_CLASSES)].copy()

    real_counts = real_dist.set_index("Coverage class")["Real N"].to_dict()
    positive_real_total = sum(int(real_counts.get(cls, 0)) for cls in POSITIVE_CLASSES)
    available_counts = {cls: int((pool["coverage_class"] == cls).sum()) for cls in POSITIVE_CLASSES}

    ratios = []
    for cls in POSITIVE_CLASSES:
        rn = int(real_counts.get(cls, 0))
        if rn > 0:
            ratios.append(available_counts.get(cls, 0) / rn)
    scale = math.floor(min(ratios)) if ratios else 0
    if scale < 1:
        scale = 1

    rng = stable_rng("positive_coverage_empirical_matched")
    sampled = []
    rows = []
    for cls in POSITIVE_CLASSES:
        real_n = int(real_counts.get(cls, 0))
        desired = int(real_n * scale)
        available = pool[pool["coverage_class"] == cls]
        take = min(desired, len(available))
        if take > 0:
            sampled.append(available.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1)), replace=False))
        rows.append(
            {
                "Coverage class": cls,
                "real_positive_incomplete_N": real_n,
                "real_positive_incomplete_percent": 100 * real_n / positive_real_total if positive_real_total else np.nan,
                "artificial_available_N": len(available),
                "matching_scale_factor": scale,
                "desired_sample_N": desired,
                "sampled_N": take,
                "sampled_percent": np.nan,
                "sampling_note": "matched_scaled_distribution" if take == desired else "limited_by_available_artificial_samples",
            }
        )

    matched = pd.concat(sampled, ignore_index=True) if sampled else pd.DataFrame(columns=pool.columns)
    matched = matched.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    matched["positive_gap_sample_ID"] = [f"POS_GAP_{i:06d}" for i in range(1, len(matched) + 1)]
    matched["validation_scenario"] = "POSITIVE_COVERAGE_EMPIRICAL_MATCHED"
    matched.to_csv(MATCHED_DATASET_CSV, index=False)

    audit = pd.DataFrame(rows)
    if len(matched):
        sampled_counts = matched["coverage_class"].value_counts().to_dict()
        audit["sampled_percent"] = audit["Coverage class"].map(lambda c: 100 * sampled_counts.get(c, 0) / len(matched))
    audit.to_csv(MATCHING_AUDIT_CSV, index=False)
    return matched, audit


def valid_for_model(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    cols = list(
        dict.fromkeys(
            features
            + CATEGORICAL_FEATURES
            + [
                TARGET_COLUMN,
                "original_case_ID",
                "positive_gap_sample_ID",
                "SITE_ID",
                "YYYYMM",
                "coverage_class",
                "artificial_coverage_probability",
                "partial_OCO3_EC_weighted_SIF",
            ]
        )
    )
    out = df[cols].copy()
    out[TARGET_COLUMN] = pd.to_numeric(out[TARGET_COLUMN], errors="coerce")
    out = out[out[TARGET_COLUMN].notna()].copy()
    return out


def make_grouped_splits(df: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray]]:
    groups = df["original_case_ID"].astype(str).to_numpy()
    n_groups = pd.Series(groups).nunique()
    n_splits = min(OUTER_SPLITS, n_groups)
    if n_splits < 2:
        raise ValueError("Not enough original_case_ID groups for grouped validation.")
    gkf = GroupKFold(n_splits=n_splits)
    splits = []
    X_dummy = np.zeros(len(df))
    for i, (train_idx, test_idx) in enumerate(gkf.split(X_dummy, groups=groups), start=1):
        splits.append((f"original_case_ID_fold_{i}", train_idx, test_idx))
    return splits


def run_grouped_validation(matched: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, np.ndarray, np.ndarray]]]:
    model_features_union = sorted(set(TARGET_GOSIF_FEATURES))
    model_df = valid_for_model(matched, model_features_union).reset_index(drop=True)
    splits = make_grouped_splits(model_df)
    pred_rows: list[dict[str, Any]] = []

    for fold_id, train_idx, test_idx in splits:
        train = model_df.iloc[train_idx].copy()
        test = model_df.iloc[test_idx].copy()
        y_test = pd.to_numeric(test[TARGET_COLUMN], errors="coerce").to_numpy()

        partial_pred = pd.to_numeric(test["partial_OCO3_EC_weighted_SIF"], errors="coerce").to_numpy()
        for row_i, pred in zip(test.index, partial_pred):
            rec = model_df.loc[row_i]
            pred_rows.append(prediction_record(rec, fold_id, "Partial_observation_baseline", pred))

        # Contextual benchmark only; calibration is fit on training data in each fold.
        lin_train = train[["GOSIF_EC_weighted", TARGET_COLUMN]].dropna()
        if len(lin_train) >= 10 and lin_train["GOSIF_EC_weighted"].nunique() > 1:
            lr = LinearRegression()
            lr.fit(lin_train[["GOSIF_EC_weighted"]], lin_train[TARGET_COLUMN])
            lin_pred = lr.predict(test[["GOSIF_EC_weighted"]])
            for row_i, pred in zip(test.index, lin_pred):
                rec = model_df.loc[row_i]
                pred_rows.append(prediction_record(rec, fold_id, "GOSIF_linear_calibrated", pred))

        for model_name, spec in MODEL_SPECS.items():
            features = list(spec.features)
            estimator = build_locked_rf(features, seed=RANDOM_SEED + len(pred_rows) % 100000)
            estimator.fit(train[features + CATEGORICAL_FEATURES], train[TARGET_COLUMN])
            preds = estimator.predict(test[features + CATEGORICAL_FEATURES])
            for row_i, pred in zip(test.index, preds):
                rec = model_df.loc[row_i]
                pred_rows.append(prediction_record(rec, fold_id, model_name, pred))

    predictions = pd.DataFrame(pred_rows)
    predictions.to_csv(PREDICTIONS_CSV, index=False)
    return predictions, splits


def prediction_record(rec: pd.Series, fold_id: str, model: str, pred: float) -> dict[str, Any]:
    true = finite_float(rec[TARGET_COLUMN])
    partial = finite_float(rec.get("partial_OCO3_EC_weighted_SIF"))
    gosif = finite_float(rec.get("GOSIF_EC_weighted"))
    return {
        "validation": "positive_coverage_empirical_matched",
        "fold_id": fold_id,
        "model": model,
        "positive_gap_sample_ID": rec.get("positive_gap_sample_ID", ""),
        "original_case_ID": rec["original_case_ID"],
        "SITE_ID": rec["SITE_ID"],
        "YYYYMM": rec["YYYYMM"],
        "coverage_class": rec["coverage_class"],
        "artificial_coverage_probability": finite_float(rec["artificial_coverage_probability"]),
        "y_true": true,
        "y_pred": finite_float(pred),
        "error": finite_float(pred) - true,
        "abs_error": abs(finite_float(pred) - true) if math.isfinite(finite_float(pred)) and math.isfinite(true) else np.nan,
        "partial_OCO3_EC_weighted_SIF": partial,
        "GOSIF_EC_weighted": gosif,
    }


def summarize_performance(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, sub in predictions.groupby("model"):
        m = metrics(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy())
        rows.append({"validation": "positive_coverage_empirical_matched", "model": model, **m})
    out = pd.DataFrame(rows).sort_values("RMSE")
    out.to_csv(PERFORMANCE_CSV, index=False)
    return out


def summarize_by_coverage(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cls in POSITIVE_CLASSES:
        for model, sub in predictions[predictions["coverage_class"] == cls].groupby("model"):
            m = metrics(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy())
            rows.append({"coverage_class": cls, "model": model, **m})
    out = pd.DataFrame(rows)
    out.to_csv(BY_COVERAGE_CSV, index=False)
    return out


def paired_comparisons(predictions: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("RF_TARGET_GOSIF_vs_Partial_observation_baseline", "Partial_observation_baseline", "RF_TARGET_GOSIF"),
        ("RF_TARGET_GOSIF_vs_RF_TARGET", "RF_TARGET", "RF_TARGET_GOSIF"),
    ]
    rows = []
    for label, base_model, candidate_model in comparisons:
        base = predictions[predictions["model"] == base_model].copy()
        cand = predictions[predictions["model"] == candidate_model].copy()
        key = ["fold_id", "positive_gap_sample_ID", "original_case_ID", "SITE_ID", "YYYYMM", "coverage_class", "artificial_coverage_probability", "y_true"]
        merged = base[key + ["y_pred", "abs_error"]].merge(
            cand[key + ["y_pred", "abs_error"]],
            on=key,
            suffixes=("_baseline", "_candidate"),
        )
        merged["paired_improvement"] = merged["abs_error_baseline"] - merged["abs_error_candidate"]
        merged["changed_class"] = np.select(
            [
                merged["paired_improvement"] > UNCHANGED_TOLERANCE,
                merged["paired_improvement"] < -UNCHANGED_TOLERANCE,
            ],
            ["improved", "worsened"],
            default="approximately_unchanged",
        )
        ci = bootstrap_ci(merged["paired_improvement"].to_numpy(), label)
        m_base = metrics(merged["y_true"].to_numpy(), merged["y_pred_baseline"].to_numpy())
        m_cand = metrics(merged["y_true"].to_numpy(), merged["y_pred_candidate"].to_numpy())
        rows.append(
            {
                "comparison": label,
                "baseline_model": base_model,
                "candidate_model": candidate_model,
                "N_common": len(merged),
                "baseline_RMSE": m_base["RMSE"],
                "candidate_RMSE": m_cand["RMSE"],
                "Delta_RMSE_positive_means_candidate_better": m_base["RMSE"] - m_cand["RMSE"],
                "baseline_MAE": m_base["MAE"],
                "candidate_MAE": m_cand["MAE"],
                "Delta_MAE_positive_means_candidate_better": m_base["MAE"] - m_cand["MAE"],
                "baseline_R2": m_base["R2"],
                "candidate_R2": m_cand["R2"],
                "Delta_R2_candidate_minus_baseline": m_cand["R2"] - m_base["R2"],
                "mean_paired_improvement": float(merged["paired_improvement"].mean()),
                "median_paired_improvement": float(merged["paired_improvement"].median()),
                "bootstrap_CI95_low": ci[0],
                "bootstrap_CI95_high": ci[1],
                "percent_improved": 100 * (merged["changed_class"] == "improved").mean(),
                "percent_worsened": 100 * (merged["changed_class"] == "worsened").mean(),
                "percent_approximately_unchanged": 100 * (merged["changed_class"] == "approximately_unchanged").mean(),
                "unchanged_tolerance": UNCHANGED_TOLERANCE,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(PAIRED_CSV, index=False)
    return out


def threshold_sensitivity(predictions: pd.DataFrame, positive_real: pd.DataFrame) -> pd.DataFrame:
    thresholds = [
        ("coverage > 0", 0.0, False),
        ("coverage >= 0.10", 0.10, True),
        ("coverage >= 0.25", 0.25, True),
        ("coverage >= 0.50", 0.50, True),
    ]
    total_real_pos = len(positive_real)
    rows = []
    for label, threshold, inclusive in thresholds:
        real_cov = positive_real["coverage_probability_fraction"]
        if threshold == 0.0:
            real_keep = positive_real[real_cov > 0]
            pred_keep = predictions[predictions["artificial_coverage_probability"] > 0].copy()
        elif inclusive:
            real_keep = positive_real[real_cov >= threshold - 1e-12]
            pred_keep = predictions[predictions["artificial_coverage_probability"] >= threshold - 1e-12].copy()
        else:
            real_keep = positive_real[real_cov > threshold]
            pred_keep = predictions[predictions["artificial_coverage_probability"] > threshold].copy()

        partial = pred_keep[pred_keep["model"] == "Partial_observation_baseline"]
        rf = pred_keep[pred_keep["model"] == "RF_TARGET_GOSIF"]
        base_m = metrics(partial["y_true"].to_numpy(), partial["y_pred"].to_numpy())
        rf_m = metrics(rf["y_true"].to_numpy(), rf["y_pred"].to_numpy())
        key = ["fold_id", "positive_gap_sample_ID", "original_case_ID", "SITE_ID", "YYYYMM", "coverage_class", "artificial_coverage_probability", "y_true"]
        merged = partial[key + ["y_pred", "abs_error"]].merge(
            rf[key + ["y_pred", "abs_error"]],
            on=key,
            suffixes=("_partial", "_rf_target_gosif"),
        )
        improvement = merged["abs_error_partial"] - merged["abs_error_rf_target_gosif"]
        rows.append(
            {
                "threshold": label,
                "N_retained_real_production_cases": len(real_keep),
                "percent_positive_incomplete_cases_retained": 100 * len(real_keep) / total_real_pos if total_real_pos else np.nan,
                "validation_N": int(rf_m["N"]),
                "RF_TARGET_GOSIF_RMSE": rf_m["RMSE"],
                "RF_TARGET_GOSIF_MAE": rf_m["MAE"],
                "Partial_baseline_RMSE": base_m["RMSE"],
                "Partial_baseline_MAE": base_m["MAE"],
                "Delta_RMSE_vs_partial_positive_means_reconstruction_better": base_m["RMSE"] - rf_m["RMSE"],
                "Delta_MAE_vs_partial_positive_means_reconstruction_better": base_m["MAE"] - rf_m["MAE"],
                "mean_paired_improvement_vs_partial": float(improvement.mean()) if len(improvement) else np.nan,
                "median_paired_improvement_vs_partial": float(improvement.median()) if len(improvement) else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(THRESHOLD_CSV, index=False)
    return out


def conformal_uncertainty(matched: pd.DataFrame, splits: list[tuple[str, np.ndarray, np.ndarray]]) -> pd.DataFrame:
    features = TARGET_GOSIF_FEATURES
    model_df = valid_for_model(matched, features).reset_index(drop=True)
    rows = []
    for fold_id, train_idx, test_idx in splits:
        train = model_df.iloc[train_idx].copy()
        test = model_df.iloc[test_idx].copy()
        groups = train["original_case_ID"].astype(str)
        if groups.nunique() < 3 or len(train) < 50:
            continue
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_SEED + len(rows))
        model_train_idx, calib_idx = next(gss.split(train, groups=groups))
        model_train = train.iloc[model_train_idx]
        calib = train.iloc[calib_idx]
        estimator = build_locked_rf(features, seed=RANDOM_SEED + len(rows))
        estimator.fit(model_train[features + CATEGORICAL_FEATURES], model_train[TARGET_COLUMN])
        calib_pred = estimator.predict(calib[features + CATEGORICAL_FEATURES])
        calib_abs = np.abs(calib_pred - calib[TARGET_COLUMN].to_numpy(dtype=float))
        test_pred = estimator.predict(test[features + CATEGORICAL_FEATURES])
        test_true = test[TARGET_COLUMN].to_numpy(dtype=float)
        test_cov = pd.to_numeric(test["artificial_coverage_probability"], errors="coerce")
        test_groups = {
            "all": np.ones(len(test), dtype=bool),
            "<=0.25": (test_cov <= 0.25).to_numpy(),
            ">0.25-0.75": ((test_cov > 0.25) & (test_cov <= 0.75)).to_numpy(),
            ">0.75": (test_cov > 0.75).to_numpy(),
        }
        for nominal in [0.90, 0.95]:
            q = conformal_quantile(calib_abs, nominal)
            lo = test_pred - q
            hi = test_pred + q
            inside = (test_true >= lo) & (test_true <= hi)
            for group_name, mask in test_groups.items():
                if mask.sum() == 0:
                    continue
                rows.append(
                    {
                        "validation": "positive_coverage_empirical_matched",
                        "fold_id": fold_id,
                        "model": "RF_TARGET_GOSIF",
                        "nominal_interval": nominal,
                        "coverage_group": group_name,
                        "n_test": int(mask.sum()),
                        "empirical_coverage_probability": float(inside[mask].mean()),
                        "mean_interval_width": float(np.mean((hi - lo)[mask])),
                        "median_interval_width": float(np.median((hi - lo)[mask])),
                        "calibration_n": int(len(calib_abs)),
                        "calibration_quantile_abs_error": float(q),
                        "method": "split/grouped conformal absolute-residual interval; calibration uses only outer-training data",
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(UNCERTAINTY_CSV, index=False)
    return out


def conformal_quantile(abs_residuals: np.ndarray, nominal: float) -> float:
    vals = np.sort(np.asarray(abs_residuals, dtype=float)[np.isfinite(abs_residuals)])
    if len(vals) == 0:
        return np.nan
    rank = int(math.ceil((len(vals) + 1) * nominal))
    rank = min(max(rank, 1), len(vals))
    return float(vals[rank - 1])


def compact_by_coverage_table(by_cov: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cls in POSITIVE_CLASSES:
        sub = by_cov[by_cov["coverage_class"] == cls]
        def val(model: str, col: str) -> float:
            s = sub[sub["model"] == model]
            return float(s[col].iloc[0]) if len(s) else np.nan
        rmses = {
            "Partial": val("Partial_observation_baseline", "RMSE"),
            "RF target": val("RF_TARGET", "RMSE"),
            "RF target+GOSIF": val("RF_TARGET_GOSIF", "RMSE"),
        }
        maes = {
            "Partial": val("Partial_observation_baseline", "MAE"),
            "RF target": val("RF_TARGET", "MAE"),
            "RF target+GOSIF": val("RF_TARGET_GOSIF", "MAE"),
        }
        valid_rmses = {k: v for k, v in rmses.items() if math.isfinite(v)}
        rows.append(
            {
                "Coverage": cls,
                "N": int(val("RF_TARGET_GOSIF", "N")) if math.isfinite(val("RF_TARGET_GOSIF", "N")) else 0,
                "Partial RMSE": rmses["Partial"],
                "RF target RMSE": rmses["RF target"],
                "RF target+GOSIF RMSE": rmses["RF target+GOSIF"],
                "Partial MAE": maes["Partial"],
                "RF target MAE": maes["RF target"],
                "RF target+GOSIF MAE": maes["RF target+GOSIF"],
                "Best": min(valid_rmses, key=valid_rmses.get) if valid_rmses else "NA",
            }
        )
    return pd.DataFrame(rows)


def decide(perf: pd.DataFrame, paired: pd.DataFrame, by_cov: pd.DataFrame, previous_perf: pd.DataFrame) -> tuple[str, list[str]]:
    reasons = []
    def metric(model: str, col: str) -> float:
        s = perf[perf["model"] == model]
        return float(s[col].iloc[0]) if len(s) else np.nan
    partial_rmse = metric("Partial_observation_baseline", "RMSE")
    partial_mae = metric("Partial_observation_baseline", "MAE")
    rf_rmse = metric("RF_TARGET_GOSIF", "RMSE")
    rf_mae = metric("RF_TARGET_GOSIF", "MAE")
    if not (rf_rmse < partial_rmse):
        reasons.append("RF_TARGET_GOSIF does not have lower positive-coverage empirical RMSE than the partial baseline.")
    if not (rf_mae < partial_mae):
        reasons.append("RF_TARGET_GOSIF does not have lower positive-coverage empirical MAE than the partial baseline.")
    comp = paired[paired["comparison"] == "RF_TARGET_GOSIF_vs_Partial_observation_baseline"]
    if len(comp):
        row = comp.iloc[0]
        if not (row["mean_paired_improvement"] > 0 and row["bootstrap_CI95_high"] > 0 and row["bootstrap_CI95_low"] >= -0.005):
            reasons.append("Paired improvement over partial baseline is weak or compatible with noise.")
    else:
        reasons.append("Missing paired RF_TARGET_GOSIF vs partial comparison.")

    low = by_cov[(by_cov["coverage_class"] == "0-0.10") & (by_cov["model"] == "RF_TARGET_GOSIF")]
    low_partial = by_cov[(by_cov["coverage_class"] == "0-0.10") & (by_cov["model"] == "Partial_observation_baseline")]
    if len(low) and len(low_partial):
        if not (float(low["RMSE"].iloc[0]) < float(low_partial["RMSE"].iloc[0]) * 1.25):
            reasons.append("Lowest positive coverage class shows catastrophic degradation relative to partial baseline.")

    prev_checks = [
        ("spatial_site_held_out", "RF_TARGET_GOSIF", "Partial_observation_baseline"),
        ("incremental_temporal", "RF_TARGET_GOSIF", "Partial_observation_baseline"),
    ]
    for validation, model, baseline in prev_checks:
        m = previous_perf[(previous_perf["validation"] == validation) & (previous_perf["model"] == model)]
        b = previous_perf[(previous_perf["validation"] == validation) & (previous_perf["model"] == baseline)]
        if len(m) and len(b):
            if not (float(m["RMSE"].iloc[0]) < float(b["RMSE"].iloc[0])):
                reasons.append(f"Previous {validation} check contradicts readiness: RF_TARGET_GOSIF did not beat partial baseline.")
        else:
            reasons.append(f"Previous {validation} check could not be verified from the locked validation report.")
    verdict = "READY FOR POSITIVE-COVERAGE OCO-3 RECONSTRUCTION" if not reasons else "NOT READY FOR POSITIVE-COVERAGE OCO-3 RECONSTRUCTION"
    return verdict, reasons


def create_figures(
    real_dist: pd.DataFrame,
    perf: pd.DataFrame,
    by_cov_compact: pd.DataFrame,
    predictions: pd.DataFrame,
    threshold: pd.DataFrame,
    verdict: str,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle("OCO-3 Positive-Coverage Final Eligibility Test", fontsize=18, fontweight="bold")

    ax = axes[0, 0]
    ax.bar(real_dist["Coverage class"], real_dist["Real N"], color="#4C78A8")
    ax.set_title("A. Real positive-incomplete coverage distribution")
    ax.set_ylabel("Real site-months")
    ax.tick_params(axis="x", rotation=30)

    ax = axes[0, 1]
    order = ["Partial_observation_baseline", "RF_TARGET", "RF_TARGET_GOSIF"]
    labels = ["Partial", "RF target", "RF target+GOSIF"]
    vals = [float(perf[perf["model"] == m]["RMSE"].iloc[0]) for m in order]
    colors = ["#F58518", "#54A24B", "#4C78A8"]
    ax.bar(labels, vals, color=colors)
    ax.set_title("B. Overall RMSE")
    ax.set_ylabel("RMSE")

    ax = axes[1, 0]
    x = np.arange(len(by_cov_compact))
    width = 0.25
    ax.bar(x - width, by_cov_compact["Partial RMSE"], width, label="Partial", color="#F58518")
    ax.bar(x, by_cov_compact["RF target RMSE"], width, label="RF target", color="#54A24B")
    ax.bar(x + width, by_cov_compact["RF target+GOSIF RMSE"], width, label="RF target+GOSIF", color="#4C78A8")
    ax.set_title("C. RMSE by positive-coverage class")
    ax.set_ylabel("RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(by_cov_compact["Coverage"], rotation=30)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    base = predictions[predictions["model"] == "Partial_observation_baseline"]
    cand = predictions[predictions["model"] == "RF_TARGET_GOSIF"]
    key = ["fold_id", "positive_gap_sample_ID", "original_case_ID", "SITE_ID", "YYYYMM", "coverage_class", "artificial_coverage_probability", "y_true"]
    merged = base[key + ["abs_error"]].merge(cand[key + ["abs_error"]], on=key, suffixes=("_partial", "_rf"))
    improvement = merged["abs_error_partial"] - merged["abs_error_rf"]
    ax.hist(improvement, bins=35, color="#72B7B2", edgecolor="white")
    ax.axvline(0, color="black", linestyle="--")
    ax.set_title("D. Paired improvement distribution\npositive = reconstruction helped")
    ax.set_xlabel("|error partial| - |error RF+GOSIF|")
    ax.set_ylabel("Samples")

    ax = axes[2, 0]
    ax.plot(threshold["threshold"], threshold["RF_TARGET_GOSIF_RMSE"], marker="o", label="RF+GOSIF RMSE", color="#4C78A8")
    ax.plot(threshold["threshold"], threshold["Partial_baseline_RMSE"], marker="o", label="Partial RMSE", color="#F58518")
    ax.set_title("E. Potential minimum coverage thresholds")
    ax.set_ylabel("RMSE")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=8)

    ax = axes[2, 1]
    ax.axis("off")
    status = "READY" if verdict.startswith("READY") else "NOT READY"
    color = "#2CA02C" if status == "READY" else "#D62728"
    ax.text(0.5, 0.72, "F. Final decision", ha="center", va="center", fontsize=16, fontweight="bold")
    ax.text(0.5, 0.48, status, ha="center", va="center", fontsize=28, color=color, fontweight="bold")
    ax.text(0.5, 0.24, "Zero coverage excluded; complete cases use observed OCO-3 directly.", ha="center", va="center", fontsize=10, wrap=True)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(SUMMARY_PNG, dpi=180)
    fig.savefig(SUMMARY_PDF)
    plt.close(fig)


def write_simple_pdf(text: str, path: Path) -> None:
    from matplotlib.backends.backend_pdf import PdfPages

    wrapped_lines: list[str] = []
    for line in text.splitlines():
        if not line:
            wrapped_lines.append("")
        elif line.startswith("|"):
            wrapped_lines.extend(textwrap.wrap(line, width=115, replace_whitespace=False) or [""])
        else:
            wrapped_lines.extend(textwrap.wrap(line, width=105, replace_whitespace=False) or [""])
    with PdfPages(path) as pdf:
        for i in range(0, len(wrapped_lines), 48):
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.06, 0.96, "\n".join(wrapped_lines[i : i + 48]), va="top", ha="left", family="monospace", fontsize=8.5)
            fig.patch.set_facecolor("white")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```"


def write_report(
    verdict: str,
    blockers: list[str],
    real_dist: pd.DataFrame,
    production_class: pd.DataFrame,
    matching_audit: pd.DataFrame,
    perf: pd.DataFrame,
    by_cov_compact: pd.DataFrame,
    paired: pd.DataFrame,
    threshold: pd.DataFrame,
    uncertainty: pd.DataFrame,
    previous_perf: pd.DataFrame,
    input_audit: pd.DataFrame,
) -> None:
    partial = perf[perf["model"] == "Partial_observation_baseline"].iloc[0]
    rf_target = perf[perf["model"] == "RF_TARGET"].iloc[0]
    rf_g = perf[perf["model"] == "RF_TARGET_GOSIF"].iloc[0]
    paired_main = paired[paired["comparison"] == "RF_TARGET_GOSIF_vs_Partial_observation_baseline"].iloc[0]
    paired_gosif = paired[paired["comparison"] == "RF_TARGET_GOSIF_vs_RF_TARGET"].iloc[0]

    prev_subset = previous_perf[
        previous_perf["validation"].isin(["spatial_site_held_out", "incremental_temporal", "spatial_temporal_stress"])
        & previous_perf["model"].isin(["Partial_observation_baseline", "RF_TARGET_GOSIF"])
    ][["validation", "model", "n", "RMSE", "MAE", "R2"]].sort_values(["validation", "RMSE"])

    unc_summary = (
        uncertainty[uncertainty["coverage_group"] == "all"]
        .groupby("nominal_interval", as_index=False)
        .agg(
            folds=("fold_id", "nunique"),
            total_n=("n_test", "sum"),
            mean_empirical_coverage=("empirical_coverage_probability", "mean"),
            min_empirical_coverage=("empirical_coverage_probability", "min"),
            max_empirical_coverage=("empirical_coverage_probability", "max"),
            mean_interval_width=("mean_interval_width", "mean"),
            median_interval_width=("median_interval_width", "median"),
        )
    )

    text = f"""# OCO-3 Positive-Coverage Final Eligibility Test

Final verdict:

```text
{verdict}
```

This was a narrow, frozen-model eligibility test for the scientifically meaningful primary reconstruction regime:

```text
0 < OCO-3 EC footprint probability coverage < 1
```

Zero-coverage cases were excluded from every primary metric. Complete cases were used only as validation truth sources, not as production reconstruction cases.

## Frozen model

- Model tested for production: `RF_TARGET_GOSIF`
- Hyperparameters: `{LOCKED_RF_PARAMS}`
- Feature set: the exact frozen target + GOSIF feature set from `OCO3_Final_Model_Lock.md`.
- No retuning, no new predictors, no new algorithms, no GPP predictor use.

## Real OCO-3 production classes

{md_table(production_class)}

## Real positive-incomplete coverage distribution

{md_table(real_dist)}

## Positive-coverage empirical matching audit

{md_table(matching_audit)}

## Primary model performance

{md_table(perf[["model", "N", "R2", "RMSE", "MAE", "Bias", "nRMSE", "Pearson_r", "Spearman_rho"]])}

Primary RMSE values:

- Partial observation baseline: {partial.RMSE:.6f}
- RF target only: {rf_target.RMSE:.6f}
- RF target + GOSIF: {rf_g.RMSE:.6f}

Primary MAE values:

- Partial observation baseline: {partial.MAE:.6f}
- RF target only: {rf_target.MAE:.6f}
- RF target + GOSIF: {rf_g.MAE:.6f}

## Paired error comparisons

{md_table(paired)}

Main paired comparison:

- Mean paired improvement over partial baseline: {paired_main.mean_paired_improvement:.6f}
- Median paired improvement: {paired_main.median_paired_improvement:.6f}
- Bootstrap 95% CI: [{paired_main.bootstrap_CI95_low:.6f}, {paired_main.bootstrap_CI95_high:.6f}]
- Improved cases: {paired_main.percent_improved:.2f}%
- Worsened cases: {paired_main.percent_worsened:.2f}%
- Approximately unchanged: {paired_main.percent_approximately_unchanged:.2f}% using tolerance {UNCHANGED_TOLERANCE}

GOSIF incremental value over RF target-only:

- ΔRMSE: {paired_gosif.Delta_RMSE_positive_means_candidate_better:.6f}
- ΔMAE: {paired_gosif.Delta_MAE_positive_means_candidate_better:.6f}
- ΔR²: {paired_gosif.Delta_R2_candidate_minus_baseline:.6f}
- Paired mean absolute-error improvement: {paired_gosif.mean_paired_improvement:.6f}
- Bootstrap 95% CI: [{paired_gosif.bootstrap_CI95_low:.6f}, {paired_gosif.bootstrap_CI95_high:.6f}]

## Performance by real positive coverage class

{md_table(by_cov_compact)}

## Threshold sensitivity

{md_table(threshold)}

## Positive-coverage conformal uncertainty calibration

{md_table(unc_summary)}

Full uncertainty details are in `{UNCERTAINTY_CSV.name}`.

## Previous spatial/temporal checks retained

The previous locked validation is not rerun here, but it remains accepted and was checked for contradiction:

{md_table(prev_subset)}

## Safety/input audit

Only CSV validation artifacts were read. No GPP TIFFs, raw SIF files, or previous validation datasets were modified by this script.

{md_table(input_audit)}

## Decision rule evaluation

1. RF_TARGET_GOSIF lower RMSE than partial baseline: `{bool(rf_g.RMSE < partial.RMSE)}`
2. RF_TARGET_GOSIF lower MAE than partial baseline: `{bool(rf_g.MAE < partial.MAE)}`
3. Positive paired improvement with bootstrap support: `{bool(paired_main.mean_paired_improvement > 0 and paired_main.bootstrap_CI95_low >= -0.005 and paired_main.bootstrap_CI95_high > 0)}`
4. No catastrophic lowest-positive-coverage failure: checked in by-coverage table.
5. Previous spatial/temporal validation remains accepted: checked.

Blockers:

{chr(10).join('- ' + b for b in blockers) if blockers else '- None'}

## Final production classification

```text
coverage = 1:
use observed OCO-3 directly

0 < coverage < 1:
{"reconstruct" if verdict.startswith("READY") else "do not reconstruct"}

coverage = 0:
exclude from primary OCO-3 reconstruction/analysis
```

## Final required verdict

```text
{verdict}
```
"""
    REPORT_MD.write_text(text, encoding="utf-8")
    write_simple_pdf(text, REPORT_PDF)

    readme = f"""OCO-3 Positive-Coverage Final Test — READ ME FIRST

1. REAL OCO-3 site-months with 0<coverage<1: {int(real_dist['Real N'].sum())}.
2. Real coverage distribution: {', '.join(f"{r['Coverage class']}={int(r['Real N'])}" for _, r in real_dist.iterrows())}.
3. Matching validation partial OCO-3 RMSE: {partial.RMSE:.6f}.
4. RF target-only RMSE: {rf_target.RMSE:.6f}.
5. RF target+GOSIF RMSE: {rf_g.RMSE:.6f}.
6. Reconstruction improvement over partial: ΔRMSE={partial.RMSE-rf_g.RMSE:.6f}, ΔMAE={partial.MAE-rf_g.MAE:.6f}.
7. Paired support: mean improvement={paired_main.mean_paired_improvement:.6f}, 95% CI=[{paired_main.bootstrap_CI95_low:.6f}, {paired_main.bootstrap_CI95_high:.6f}], improved={paired_main.percent_improved:.2f}%, worsened={paired_main.percent_worsened:.2f}%.
8. Below 10% coverage: see by-coverage table; this class has {int(by_cov_compact.loc[by_cov_compact['Coverage']=='0-0.10','N'].iloc[0])} validation samples.
9. Minimum coverage threshold above zero: see threshold table; recommendation follows the final verdict and observed low-coverage performance.
10. Real OCO-3 partial cases eligible for production under 0<coverage<1: {int(real_dist['Real N'].sum())}.
11. Final decision: {verdict}.
"""
    README_TXT.write_text(readme, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    before = audit_inputs("before")

    gap, real, previous_perf = load_and_check_inputs()
    _, positive_real, real_dist = build_real_positive_distribution(real)
    production_class = pd.read_csv(PRODUCTION_CLASS_CSV)
    matched, matching_audit = build_positive_empirical_matched_dataset(gap, real_dist)

    predictions, splits = run_grouped_validation(matched)
    perf = summarize_performance(predictions)
    by_cov = summarize_by_coverage(predictions)
    by_cov_compact = compact_by_coverage_table(by_cov)
    paired = paired_comparisons(predictions)
    threshold = threshold_sensitivity(predictions, positive_real)
    uncertainty = conformal_uncertainty(matched, splits)

    verdict, blockers = decide(perf, paired, by_cov, previous_perf)

    create_figures(real_dist, perf, by_cov_compact, predictions, threshold, verdict)

    after = audit_inputs("after")
    input_audit = before.merge(after, on="path", suffixes=("_before", "_after"))
    input_audit["unchanged_size_and_mtime"] = (
        (input_audit["size_bytes_before"] == input_audit["size_bytes_after"])
        & (input_audit["mtime_ns_before"] == input_audit["mtime_ns_after"])
    )
    input_audit.to_csv(INPUT_AUDIT_CSV, index=False)

    write_report(
        verdict,
        blockers,
        real_dist,
        production_class,
        matching_audit,
        perf,
        by_cov_compact,
        paired,
        threshold,
        uncertainty,
        previous_perf,
        input_audit,
    )

    print("=" * 60)
    print("OCO-3 positive-coverage final eligibility test complete")
    print(f"Verdict: {verdict}")
    print(f"Real positive-incomplete cases: {int(real_dist['Real N'].sum())}")
    print(f"Validation samples: {len(matched)}")
    print(f"Report: {REPORT_MD}")
    print(f"PDF: {REPORT_PDF}")
    print(f"Summary figure: {SUMMARY_PNG}")
    print("No real incomplete OCO-3 reconstruction was performed.")


if __name__ == "__main__":
    main()
