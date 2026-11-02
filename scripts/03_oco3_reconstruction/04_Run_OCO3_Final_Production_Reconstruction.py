"""Apply the frozen OCO-3 EC-footprint reconstruction to positive partial-coverage site-months. Zero-coverage months are excluded and never reconstructed; no GPP predictor is used; no rasters are created.

Output: OCO-3 EC-footprint SIF tables (all site-months and primary analysis set), model-lock record and counts."""
from __future__ import annotations
# Original script filename (sanitized): Run_OCO3_Final_Production_Reconstruction.py; public name: 04_Run_OCO3_Final_Production_Reconstruction.py
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

import hashlib
import json
import math
import platform
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


RANDOM_SEED = 20260812
TARGET_COLUMN = "true_complete_OCO3_EC_SIF"
COVERAGE_TOLERANCE = 1e-9

PROJECT_ROOT = Path(
    PROJECT_ROOT
)
OUTPUT_ROOT = PROJECT_ROOT / "SIF_EC_Reconstruction_OCO3_Final_Production"
FIG_DIR = OUTPUT_ROOT / "figures"
QC_DIR = OUTPUT_ROOT / "qc"
MODEL_DIR = OUTPUT_ROOT / "model"

PREP_ROOT = PROJECT_ROOT / "SIF_EC_Reconstruction_Model_Preparation"
POC_ROOT = PROJECT_ROOT / "SIF_EC_Reconstruction_OCO3_Proof_of_Concept"
PREPROD_ROOT = PROJECT_ROOT / "SIF_EC_Reconstruction_OCO3_Final_PreProduction_Validation"
POSITIVE_TEST_ROOT = PROJECT_ROOT / "SIF_EC_Reconstruction_OCO3_Positive_Coverage_Final_Test"
GPP_TIFF_ROOT = PROJECT_ROOT / "Monthly_EC_Footprint_TIFFs_Python"
SIF_ROOT = PROJECT_ROOT / "SIF Datasets"
RAW_OCO3_ROOT = SIF_ROOT / "OCO-3"
RAW_GOSIF_ROOT = SIF_ROOT / "GOSIF"

ARTIFICIAL_GAPS_CSV = POC_ROOT / "OCO3_Artificial_Gap_Dataset.csv"
REAL_COVERAGE_CSV = PREP_ROOT / "EC_Coverage_All_Products.csv"
OCO3_GOSIF_INVENTORY_CSV = PREP_ROOT / "OCO3_GOSIF_EC_Reconstruction_Model_Inventory.csv"
MODEL_LOCK_MD = PREPROD_ROOT / "OCO3_Final_Model_Lock.md"
POSITIVE_TEST_REPORT_MD = POSITIVE_TEST_ROOT / "OCO3_Positive_Coverage_Final_Test_Report.md"
POSITIVE_TEST_UNCERTAINTY_CSV = POSITIVE_TEST_ROOT / "OCO3_Positive_Coverage_Uncertainty_Calibration.csv"

MODEL_JOBLIB = MODEL_DIR / "OCO3_RF_TARGET_GOSIF_Final_Model.joblib"
MODEL_METADATA_JSON = MODEL_DIR / "OCO3_RF_TARGET_GOSIF_Final_Model_Metadata.json"
MASTER_CSV = OUTPUT_ROOT / "OCO3_Final_EC_Footprint_SIF_All_SiteMonths.csv"
PRIMARY_CSV = OUTPUT_ROOT / "OCO3_Final_EC_Footprint_SIF_Primary_Analysis.csv"
RECON_ONLY_CSV = OUTPUT_ROOT / "OCO3_Reconstructed_Partial_Cases_Only.csv"
QC_FLAGS_CSV = QC_DIR / "OCO3_Production_QC_Flags.csv"
COVERAGE_QC_CSV = QC_DIR / "OCO3_Production_Reconstructed_By_Coverage_Class.csv"
DISTRIBUTION_QC_CSV = QC_DIR / "OCO3_Production_Complete_vs_Reconstructed_Distribution.csv"
GOSIF_DEPENDENCE_CSV = QC_DIR / "OCO3_Production_GOSIF_Dependence.csv"
COUNTS_CSV = QC_DIR / "OCO3_Final_Production_Counts.csv"
SAFETY_AUDIT_CSV = QC_DIR / "OCO3_Final_Production_Input_Safety_Audit.csv"
REPORT_MD = OUTPUT_ROOT / "OCO3_Final_Production_Report.md"
REPORT_PDF = OUTPUT_ROOT / "OCO3_Final_Production_Report.pdf"
README_TXT = OUTPUT_ROOT / "OCO3_Final_Production_READ_ME_FIRST.txt"
REPRO_JSON = OUTPUT_ROOT / "OCO3_Final_Production_Reproducibility.json"
SUMMARY_PNG = OUTPUT_ROOT / "OCO3_Final_Production_Simple_Summary.png"
SUMMARY_PDF = OUTPUT_ROOT / "OCO3_Final_Production_Simple_Summary.pdf"

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

FEATURES = TARGET_FEATURES + GOSIF_FEATURES
CATEGORICAL_FEATURES = ["IGBP"]
LOCKED_RF_PARAMS = {
    "n_estimators": 80,
    "max_depth": 8,
    "min_samples_leaf": 4,
    "max_features": "sqrt",
}

PRODUCTION_FEATURE_MAPPING = {
    "partial_OCO3_EC_weighted_SIF": "target_observed_EC_weighted_SIF",
    "partial_OCO3_mean": "target_observed_mean_SIF",
    "partial_OCO3_sd": "target_observed_sd",
    "partial_OCO3_min": "target_observed_min",
    "partial_OCO3_max": "target_observed_max",
    "artificial_coverage_probability": "coverage_probability_fraction",
    "artificial_coverage_pixel_fraction": "coverage_pixel_fraction",
    "n_original_target_footprints": "n_target_source_footprints",
    "n_remaining_target_footprints": "n_target_source_footprints",
}

POSITIVE_CLASSES = ["0-0.10", ">0.10-0.25", ">0.25-0.50", ">0.50-0.75", ">0.75-<1"]


def ensure_dirs() -> None:
    for path in [OUTPUT_ROOT, FIG_DIR, QC_DIR, MODEL_DIR]:
        path.mkdir(parents=True, exist_ok=True)


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


def build_locked_rf(seed: int = RANDOM_SEED) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", preprocessor(FEATURES)),
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


def finite_float(v: Any) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else np.nan
    except Exception:
        return np.nan


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def directory_fingerprint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "kind": "directory",
            "file_count": 0,
            "total_size_bytes": 0,
            "max_mtime_ns": np.nan,
            "fingerprint": None,
        }
    file_count = 0
    total_size = 0
    max_mtime = 0
    h = hashlib.sha256()
    for file in sorted([p for p in path.rglob("*") if p.is_file()], key=lambda p: str(p).lower()):
        try:
            rel = str(file.relative_to(path)).replace("\\", "/")
            stat = file.stat()
        except OSError:
            continue
        file_count += 1
        total_size += stat.st_size
        max_mtime = max(max_mtime, stat.st_mtime_ns)
        h.update(rel.encode("utf-8", errors="ignore"))
        h.update(str(stat.st_size).encode())
        h.update(str(stat.st_mtime_ns).encode())
    return {
        "path": str(path),
        "exists": True,
        "kind": "directory",
        "file_count": file_count,
        "total_size_bytes": total_size,
        "max_mtime_ns": max_mtime,
        "fingerprint": h.hexdigest(),
    }


def file_fingerprint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "kind": "file",
            "file_count": np.nan,
            "total_size_bytes": np.nan,
            "max_mtime_ns": np.nan,
            "fingerprint": None,
            "sha256": None,
        }
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "kind": "file",
        "file_count": 1,
        "total_size_bytes": stat.st_size,
        "max_mtime_ns": stat.st_mtime_ns,
        "fingerprint": f"{stat.st_size}:{stat.st_mtime_ns}",
        "sha256": sha256_file(path),
    }


def audit_inputs(stage: str) -> pd.DataFrame:
    items = [
        ("GPP_TIFF_ROOT", GPP_TIFF_ROOT, "directory"),
        ("RAW_OCO3_ROOT", RAW_OCO3_ROOT, "directory"),
        ("RAW_GOSIF_ROOT", RAW_GOSIF_ROOT, "directory"),
        ("ARTIFICIAL_GAPS_CSV", ARTIFICIAL_GAPS_CSV, "file"),
        ("REAL_COVERAGE_CSV", REAL_COVERAGE_CSV, "file"),
        ("OCO3_GOSIF_INVENTORY_CSV", OCO3_GOSIF_INVENTORY_CSV, "file"),
        ("MODEL_LOCK_MD", MODEL_LOCK_MD, "file"),
        ("POSITIVE_TEST_REPORT_MD", POSITIVE_TEST_REPORT_MD, "file"),
        ("POSITIVE_TEST_UNCERTAINTY_CSV", POSITIVE_TEST_UNCERTAINTY_CSV, "file"),
    ]
    rows = []
    for label, path, kind in items:
        rec = directory_fingerprint(path) if kind == "directory" else file_fingerprint(path)
        rec["label"] = label
        rec["audit_stage"] = stage
        rows.append(rec)
    return pd.DataFrame(rows)


def require_inputs() -> None:
    missing = [
        p
        for p in [
            ARTIFICIAL_GAPS_CSV,
            REAL_COVERAGE_CSV,
            OCO3_GOSIF_INVENTORY_CSV,
            MODEL_LOCK_MD,
            POSITIVE_TEST_REPORT_MD,
            POSITIVE_TEST_UNCERTAINTY_CSV,
        ]
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing required validated input artifacts: " + "; ".join(map(str, missing)))
    lock_text = MODEL_LOCK_MD.read_text(encoding="utf-8", errors="replace")
    required_phrases = ["RF_TARGET_GOSIF", "n_estimators", "max_depth", "min_samples_leaf", "max_features", "sqrt"]
    absent = [p for p in required_phrases if p not in lock_text]
    if absent:
        raise RuntimeError("STOP PRODUCTION: model lock does not unambiguously contain required frozen RF specification: " + ", ".join(absent))
    positive_text = POSITIVE_TEST_REPORT_MD.read_text(encoding="utf-8", errors="replace")
    if "READY FOR POSITIVE-COVERAGE OCO-3 RECONSTRUCTION" not in positive_text:
        raise RuntimeError("STOP PRODUCTION: positive-coverage final test is not READY in the validated report.")


def coverage_class(v: Any) -> str:
    x = finite_float(v)
    if not math.isfinite(x):
        return "NaN"
    if x <= 0.10:
        return "0-0.10"
    if x <= 0.25:
        return ">0.10-0.25"
    if x <= 0.50:
        return ">0.25-0.50"
    if x <= 0.75:
        return ">0.50-0.75"
    if x < 1.0:
        return ">0.75-<1"
    return "=1"


def classify_production(v: Any) -> str:
    x = finite_float(v)
    if not math.isfinite(x):
        return "EXCLUDED_INVALID_SOURCE"
    if x <= COVERAGE_TOLERANCE:
        return "ZERO_COVERAGE_EXCLUDED"
    if x >= 1.0 - COVERAGE_TOLERANCE:
        return "COMPLETE_OBSERVED"
    return "PARTIAL_RECONSTRUCTION"


def load_training_data() -> pd.DataFrame:
    gap = pd.read_csv(ARTIFICIAL_GAPS_CSV)
    gap["artificial_coverage_probability"] = pd.to_numeric(gap["artificial_coverage_probability"], errors="coerce")
    train = gap[(gap["artificial_coverage_probability"] > COVERAGE_TOLERANCE) & (gap["artificial_coverage_probability"] < 1.0 - COVERAGE_TOLERANCE)].copy()
    required = FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN, "original_case_ID"]
    missing_cols = [c for c in required if c not in train.columns]
    if missing_cols:
        raise RuntimeError("STOP PRODUCTION: training data lacks frozen columns: " + ", ".join(missing_cols))
    train[TARGET_COLUMN] = pd.to_numeric(train[TARGET_COLUMN], errors="coerce")
    train = train[train[TARGET_COLUMN].notna()].copy()
    if len(train) == 0:
        raise RuntimeError("STOP PRODUCTION: no positive-coverage artificial training samples are available.")
    return train


def train_final_model(train: pd.DataFrame) -> Pipeline:
    model = build_locked_rf(seed=RANDOM_SEED)
    model.fit(train[FEATURES + CATEGORICAL_FEATURES], train[TARGET_COLUMN])
    return model


def load_production_inventory() -> pd.DataFrame:
    cov = pd.read_csv(REAL_COVERAGE_CSV)
    gosif = pd.read_csv(OCO3_GOSIF_INVENTORY_CSV)
    oco3 = cov[cov["PRODUCT"].astype(str).str.upper().isin(["OCO-3", "OCO3"])].copy()
    for col in ["SITE_ID", "YYYYMM"]:
        if col not in oco3.columns or col not in gosif.columns:
            raise RuntimeError(f"STOP PRODUCTION: required merge key {col} missing.")
    oco3["YYYYMM"] = pd.to_numeric(oco3["YYYYMM"], errors="coerce").astype("Int64")
    gosif["YYYYMM"] = pd.to_numeric(gosif["YYYYMM"], errors="coerce").astype("Int64")
    keep_gosif = [
        "SITE_ID",
        "YYYYMM",
        "GOSIF_EC_weighted",
        "GOSIF_native_sd",
        "n_native_GOSIF_cells_intersecting_EC",
        "GOSIF_units",
        "GOSIF_status",
        "GOSIF_scale_status",
        "GOSIF_source_file",
    ]
    merged = oco3.merge(gosif[[c for c in keep_gosif if c in gosif.columns]], on=["SITE_ID", "YYYYMM"], how="left", suffixes=("", "_gosif_inventory"))
    merged["OCO3_production_class"] = merged["coverage_probability_fraction"].map(classify_production)
    merged["production_coverage_class"] = merged["coverage_probability_fraction"].map(coverage_class)
    return merged


def make_production_features(inv: pd.DataFrame) -> pd.DataFrame:
    prod = inv.copy()
    for frozen_name, source_name in PRODUCTION_FEATURE_MAPPING.items():
        prod[frozen_name] = pd.to_numeric(prod.get(source_name), errors="coerce")
    prod["missing_probability_fraction"] = pd.to_numeric(prod["missing_probability_fraction"], errors="coerce")
    prod["lat"] = pd.to_numeric(prod["lat"], errors="coerce")
    prod["lon"] = pd.to_numeric(prod["lon"], errors="coerce")
    prod["year"] = pd.to_numeric(prod["year"], errors="coerce")
    prod["month"] = pd.to_numeric(prod["month"], errors="coerce")
    prod["DOY"] = pd.to_numeric(prod["DOY"], errors="coerce")
    prod["sin_DOY"] = pd.to_numeric(prod["sin_DOY"], errors="coerce")
    prod["cos_DOY"] = pd.to_numeric(prod["cos_DOY"], errors="coerce")
    prod["GOSIF_EC_weighted"] = pd.to_numeric(prod["GOSIF_EC_weighted"], errors="coerce")
    prod["GOSIF_native_sd"] = pd.to_numeric(prod["GOSIF_native_sd"], errors="coerce")
    prod["n_native_GOSIF_cells_intersecting_EC"] = pd.to_numeric(prod["n_native_GOSIF_cells_intersecting_EC"], errors="coerce")
    if "IGBP" not in prod.columns:
        prod["IGBP"] = "Unknown"
    prod["IGBP"] = prod["IGBP"].fillna("Unknown").astype(str)
    return prod


def compute_conformal_quantiles(train: pd.DataFrame) -> dict[str, float]:
    groups = train["original_case_ID"].astype(str)
    if groups.nunique() < 3:
        raise RuntimeError("STOP PRODUCTION: insufficient groups for conformal calibration split.")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_SEED)
    model_idx, calib_idx = next(gss.split(train, groups=groups))
    model_train = train.iloc[model_idx]
    calib = train.iloc[calib_idx]
    model = build_locked_rf(seed=RANDOM_SEED + 90)
    model.fit(model_train[FEATURES + CATEGORICAL_FEATURES], model_train[TARGET_COLUMN])
    preds = model.predict(calib[FEATURES + CATEGORICAL_FEATURES])
    residuals = np.abs(preds - calib[TARGET_COLUMN].to_numpy(dtype=float))
    return {
        "q90": conformal_quantile(residuals, 0.90),
        "q95": conformal_quantile(residuals, 0.95),
        "calibration_n": int(len(residuals)),
        "calibration_abs_error_mean": float(np.mean(residuals)),
        "calibration_abs_error_median": float(np.median(residuals)),
    }


def conformal_quantile(abs_residuals: np.ndarray, nominal: float) -> float:
    vals = np.sort(np.asarray(abs_residuals, dtype=float)[np.isfinite(abs_residuals)])
    if len(vals) == 0:
        return np.nan
    rank = int(math.ceil((len(vals) + 1) * nominal))
    rank = min(max(rank, 1), len(vals))
    return float(vals[rank - 1])


def apply_production(model: Pipeline, prod: pd.DataFrame, train: pd.DataFrame, conformal: dict[str, float]) -> pd.DataFrame:
    out = prod.copy()
    out["OCO3_observed_partial_EC_SIF"] = pd.to_numeric(out["target_observed_EC_weighted_SIF"], errors="coerce")
    out["OCO3_observed_complete_EC_SIF"] = np.where(
        out["OCO3_production_class"].eq("COMPLETE_OBSERVED"),
        out["OCO3_observed_partial_EC_SIF"],
        np.nan,
    )
    out["OCO3_reconstructed_complete_EC_SIF"] = np.nan
    out["OCO3_final_EC_SIF"] = np.nan
    out["reconstruction_applied"] = False
    out["model_used"] = "NONE"

    partial_mask = out["OCO3_production_class"].eq("PARTIAL_RECONSTRUCTION")
    required_numeric = FEATURES
    predictor_missing = partial_mask & out[required_numeric].isna().any(axis=1)
    invalid_partial = partial_mask & ~np.isfinite(out["OCO3_observed_partial_EC_SIF"].astype(float))
    valid_partial = partial_mask & ~predictor_missing & ~invalid_partial

    if valid_partial.any():
        preds = model.predict(out.loc[valid_partial, FEATURES + CATEGORICAL_FEATURES])
        out.loc[valid_partial, "OCO3_reconstructed_complete_EC_SIF"] = preds
        out.loc[valid_partial, "OCO3_final_EC_SIF"] = preds
        out.loc[valid_partial, "reconstruction_applied"] = True
        out.loc[valid_partial, "model_used"] = "RF_TARGET_GOSIF"

    complete_mask = out["OCO3_production_class"].eq("COMPLETE_OBSERVED")
    out.loc[complete_mask, "OCO3_final_EC_SIF"] = out.loc[complete_mask, "OCO3_observed_complete_EC_SIF"]
    out.loc[complete_mask, "model_used"] = "OBSERVED_NO_MODEL"

    zero_mask = out["OCO3_production_class"].eq("ZERO_COVERAGE_EXCLUDED")
    out.loc[zero_mask, "model_used"] = "NONE"

    for col in ["OCO3_PI90_lower", "OCO3_PI90_upper", "OCO3_PI95_lower", "OCO3_PI95_upper", "OCO3_PI90_width", "OCO3_PI95_width"]:
        out[col] = np.nan
    out.loc[valid_partial, "OCO3_PI90_lower"] = out.loc[valid_partial, "OCO3_final_EC_SIF"] - conformal["q90"]
    out.loc[valid_partial, "OCO3_PI90_upper"] = out.loc[valid_partial, "OCO3_final_EC_SIF"] + conformal["q90"]
    out.loc[valid_partial, "OCO3_PI95_lower"] = out.loc[valid_partial, "OCO3_final_EC_SIF"] - conformal["q95"]
    out.loc[valid_partial, "OCO3_PI95_upper"] = out.loc[valid_partial, "OCO3_final_EC_SIF"] + conformal["q95"]
    out.loc[valid_partial, "OCO3_PI90_width"] = 2 * conformal["q90"]
    out.loc[valid_partial, "OCO3_PI95_width"] = 2 * conformal["q95"]

    y_train = pd.to_numeric(train[TARGET_COLUMN], errors="coerce").dropna()
    train_stats = {
        "training_truth_min": float(y_train.min()),
        "training_truth_max": float(y_train.max()),
        "training_truth_P01": float(y_train.quantile(0.01)),
        "training_truth_P99": float(y_train.quantile(0.99)),
    }
    for k, v in train_stats.items():
        out[k] = v
    out["prediction_outside_training_range"] = valid_partial & (
        (out["OCO3_final_EC_SIF"] < train_stats["training_truth_min"]) | (out["OCO3_final_EC_SIF"] > train_stats["training_truth_max"])
    )
    out["prediction_outside_training_P01_P99"] = valid_partial & (
        (out["OCO3_final_EC_SIF"] < train_stats["training_truth_P01"]) | (out["OCO3_final_EC_SIF"] > train_stats["training_truth_P99"])
    )
    out["delta_reconstruction"] = out["OCO3_final_EC_SIF"] - out["OCO3_observed_partial_EC_SIF"]
    out["absolute_delta_reconstruction"] = out["delta_reconstruction"].abs()

    # Defensible extreme-change rule: P95 of absolute correction observed in the validated artificial-gap training set.
    abs_training_delta = (pd.to_numeric(train[TARGET_COLUMN], errors="coerce") - pd.to_numeric(train["partial_OCO3_EC_weighted_SIF"], errors="coerce")).abs().dropna()
    extreme_threshold = float(abs_training_delta.quantile(0.95))
    out["extreme_change_threshold_abs_delta"] = extreme_threshold
    out["extreme_change_from_partial"] = valid_partial & (out["absolute_delta_reconstruction"] > extreme_threshold)

    status = []
    for _, row in out.iterrows():
        cls = row["OCO3_production_class"]
        if cls == "ZERO_COVERAGE_EXCLUDED":
            status.append("EXCLUDED_ZERO_COVERAGE")
        elif cls == "EXCLUDED_INVALID_SOURCE":
            status.append("EXCLUDED_INVALID_SOURCE")
        elif cls == "COMPLETE_OBSERVED":
            if math.isfinite(finite_float(row["OCO3_final_EC_SIF"])):
                status.append("OBSERVED_COMPLETE_VALID")
            else:
                status.append("EXCLUDED_INVALID_SOURCE")
        elif cls == "PARTIAL_RECONSTRUCTION":
            if bool(predictor_missing.loc[row.name]) or bool(invalid_partial.loc[row.name]):
                status.append("EXCLUDED_MISSING_REQUIRED_PREDICTOR")
            elif bool(row["prediction_outside_training_range"]) or bool(row["prediction_outside_training_P01_P99"]) or bool(row["extreme_change_from_partial"]):
                status.append("RECONSTRUCTED_PARTIAL_QC_FLAG")
            elif math.isfinite(finite_float(row["OCO3_final_EC_SIF"])):
                status.append("RECONSTRUCTED_PARTIAL_VALID")
            else:
                status.append("EXCLUDED_INVALID_SOURCE")
        else:
            status.append("EXCLUDED_INVALID_SOURCE")
    out["OCO3_final_status"] = status
    out["SOURCE_INPUTS_UNCHANGED"] = np.nan
    out["coverage_class_positive"] = out["coverage_probability_fraction"].map(coverage_class)
    return out


def write_model_metadata(train: pd.DataFrame, conformal: dict[str, float]) -> dict[str, Any]:
    import sklearn

    metadata = {
        "model_family": "RandomForestRegressor",
        "model_token": "RF_TARGET_GOSIF",
        "hyperparameters": LOCKED_RF_PARAMS,
        "features": FEATURES,
        "feature_order": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "preprocessing": {
            "numeric": "SimpleImputer(strategy='median')",
            "categorical": "SimpleImputer(strategy='constant', fill_value='Unknown') + OneHotEncoder(handle_unknown='ignore')",
        },
        "training_sample_count": int(len(train)),
        "training_original_case_count": int(train["original_case_ID"].nunique()),
        "training_coverage_regime": "positive artificial gaps only: 0 < artificial_coverage_probability < 1",
        "random_seed": RANDOM_SEED,
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_training_dataset_path": str(ARTIFICIAL_GAPS_CSV),
        "model_lock_source_path": str(MODEL_LOCK_MD),
        "positive_coverage_test_report_path": str(POSITIVE_TEST_REPORT_MD),
        "conformal_uncertainty": conformal,
        "gpp_predictor_used": False,
        "real_zero_coverage_reconstructed": False,
        "fine_resolution_rasters_created": False,
    }
    MODEL_METADATA_JSON.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def select_master_columns(out: pd.DataFrame) -> pd.DataFrame:
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
        "coverage_class_positive",
        "OCO3_production_class",
        "OCO3_final_status",
        "OCO3_observed_partial_EC_SIF",
        "OCO3_observed_complete_EC_SIF",
        "GOSIF_EC_weighted",
        "GOSIF_native_sd",
        "n_native_GOSIF_cells_intersecting_EC",
        "OCO3_reconstructed_complete_EC_SIF",
        "OCO3_final_EC_SIF",
        "reconstruction_applied",
        "model_used",
        "OCO3_PI90_lower",
        "OCO3_PI90_upper",
        "OCO3_PI90_width",
        "OCO3_PI95_lower",
        "OCO3_PI95_upper",
        "OCO3_PI95_width",
        "n_target_source_observations",
        "n_target_source_footprints",
        "partial_OCO3_mean",
        "partial_OCO3_sd",
        "partial_OCO3_min",
        "partial_OCO3_max",
        "delta_reconstruction",
        "absolute_delta_reconstruction",
        "prediction_outside_training_range",
        "prediction_outside_training_P01_P99",
        "extreme_change_from_partial",
        "training_truth_min",
        "training_truth_max",
        "training_truth_P01",
        "training_truth_P99",
        "GOSIF_status",
        "GOSIF_scale_status",
        "GOSIF_source_file",
        "source_sif_files",
        "matching_gpp_tif",
        "status",
        "notes",
    ]
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    return out[cols]


def create_qc_tables(out: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    flag_rows = []
    partial = out[out["OCO3_production_class"].eq("PARTIAL_RECONSTRUCTION")].copy()
    flag_specs = {
        "OUTSIDE_TRAINING_RANGE": "prediction_outside_training_range",
        "OUTSIDE_TRAINING_P01_P99": "prediction_outside_training_P01_P99",
        "EXTREME_CHANGE_FROM_PARTIAL": "extreme_change_from_partial",
    }
    for reason, col in flag_specs.items():
        sub = partial[partial[col].fillna(False).astype(bool)].copy()
        for _, r in sub.iterrows():
            flag_rows.append(
                {
                    "SITE_ID": r["SITE_ID"],
                    "YYYYMM": r["YYYYMM"],
                    "coverage_probability_fraction": r["coverage_probability_fraction"],
                    "OCO3_observed_partial_EC_SIF": r["OCO3_observed_partial_EC_SIF"],
                    "OCO3_reconstructed_complete_EC_SIF": r["OCO3_reconstructed_complete_EC_SIF"],
                    "delta_reconstruction": r["delta_reconstruction"],
                    "absolute_delta_reconstruction": r["absolute_delta_reconstruction"],
                    "OCO3_final_status": r["OCO3_final_status"],
                    "flag_reason": reason,
                }
            )
    for status in ["EXCLUDED_MISSING_REQUIRED_PREDICTOR", "EXCLUDED_INVALID_SOURCE"]:
        sub = out[out["OCO3_final_status"].eq(status)]
        for _, r in sub.iterrows():
            flag_rows.append(
                {
                    "SITE_ID": r["SITE_ID"],
                    "YYYYMM": r["YYYYMM"],
                    "coverage_probability_fraction": r["coverage_probability_fraction"],
                    "OCO3_observed_partial_EC_SIF": r["OCO3_observed_partial_EC_SIF"],
                    "OCO3_reconstructed_complete_EC_SIF": r["OCO3_reconstructed_complete_EC_SIF"],
                    "delta_reconstruction": r["delta_reconstruction"],
                    "absolute_delta_reconstruction": r["absolute_delta_reconstruction"],
                    "OCO3_final_status": r["OCO3_final_status"],
                    "flag_reason": status,
                }
            )
    flags = pd.DataFrame(flag_rows)
    flags.to_csv(QC_FLAGS_CSV, index=False)

    cov_rows = []
    for cls in POSITIVE_CLASSES:
        sub = partial[partial["coverage_class_positive"].eq(cls)]
        cov_rows.append(
            {
                "coverage_class": cls,
                "N_real_reconstructed_cases": len(sub),
                "mean_partial_OCO3": float(sub["OCO3_observed_partial_EC_SIF"].mean()) if len(sub) else np.nan,
                "mean_reconstructed_OCO3": float(sub["OCO3_final_EC_SIF"].mean()) if len(sub) else np.nan,
                "mean_absolute_correction": float(sub["absolute_delta_reconstruction"].mean()) if len(sub) else np.nan,
                "mean_PI90_width": float(sub["OCO3_PI90_width"].mean()) if len(sub) else np.nan,
                "mean_PI95_width": float(sub["OCO3_PI95_width"].mean()) if len(sub) else np.nan,
            }
        )
    by_cov = pd.DataFrame(cov_rows)
    by_cov.to_csv(COVERAGE_QC_CSV, index=False)

    dist_rows = []
    for cls, sub in [
        ("COMPLETE_OBSERVED", out[out["OCO3_production_class"].eq("COMPLETE_OBSERVED")]),
        ("PARTIAL_RECONSTRUCTION", out[out["OCO3_production_class"].eq("PARTIAL_RECONSTRUCTION")]),
    ]:
        vals = pd.to_numeric(sub["OCO3_final_EC_SIF"], errors="coerce").dropna()
        dist_rows.append(
            {
                "production_class": cls,
                "N": len(vals),
                "mean": float(vals.mean()) if len(vals) else np.nan,
                "median": float(vals.median()) if len(vals) else np.nan,
                "SD": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                "P05": float(vals.quantile(0.05)) if len(vals) else np.nan,
                "P25": float(vals.quantile(0.25)) if len(vals) else np.nan,
                "P75": float(vals.quantile(0.75)) if len(vals) else np.nan,
                "P95": float(vals.quantile(0.95)) if len(vals) else np.nan,
                "min": float(vals.min()) if len(vals) else np.nan,
                "max": float(vals.max()) if len(vals) else np.nan,
            }
        )
    dist = pd.DataFrame(dist_rows)
    dist.to_csv(DISTRIBUTION_QC_CSV, index=False)

    recon = partial[pd.to_numeric(partial["OCO3_final_EC_SIF"], errors="coerce").notna()].copy()
    corr_partial = safe_corr(recon["OCO3_final_EC_SIF"], recon["OCO3_observed_partial_EC_SIF"])
    corr_gosif = safe_corr(recon["OCO3_final_EC_SIF"], recon["GOSIF_EC_weighted"])
    dep = pd.DataFrame(
        [
            {
                "N_reconstructed_cases": len(recon),
                "corr_reconstructed_OCO3_partial_OCO3": corr_partial,
                "corr_reconstructed_OCO3_GOSIF": corr_gosif,
                "diagnostic_note": "Production diagnostic only; true complete OCO-3 is unknown for real partial cases.",
            }
        ]
    )
    dep.to_csv(GOSIF_DEPENDENCE_CSV, index=False)
    return flags, by_cov, dist, dep


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce")
    mask = aa.notna() & bb.notna()
    if mask.sum() < 3 or aa[mask].std() == 0 or bb[mask].std() == 0:
        return np.nan
    return float(stats.pearsonr(aa[mask], bb[mask]).statistic)


def make_counts(out: pd.DataFrame, flags: pd.DataFrame, source_inputs_unchanged: bool) -> pd.DataFrame:
    n_total = len(out)
    n_complete = int(out["OCO3_production_class"].eq("COMPLETE_OBSERVED").sum())
    n_partial = int(out["OCO3_production_class"].eq("PARTIAL_RECONSTRUCTION").sum())
    n_recon_success = int(out["OCO3_final_status"].isin(["RECONSTRUCTED_PARTIAL_VALID", "RECONSTRUCTED_PARTIAL_QC_FLAG"]).sum())
    n_partial_failed = int(out["OCO3_final_status"].isin(["EXCLUDED_MISSING_REQUIRED_PREDICTOR", "EXCLUDED_INVALID_SOURCE"]).sum() & out["OCO3_production_class"].eq("PARTIAL_RECONSTRUCTION").any())
    # recompute accurately
    n_partial_failed = int(((out["OCO3_production_class"].eq("PARTIAL_RECONSTRUCTION")) & (out["OCO3_final_status"].isin(["EXCLUDED_MISSING_REQUIRED_PREDICTOR", "EXCLUDED_INVALID_SOURCE"]))).sum())
    n_zero = int(out["OCO3_production_class"].eq("ZERO_COVERAGE_EXCLUDED").sum())
    n_primary = int(out["OCO3_production_class"].isin(["COMPLETE_OBSERVED", "PARTIAL_RECONSTRUCTION"]).sum() - n_partial_failed)
    n_qc = int(out["OCO3_final_status"].eq("RECONSTRUCTED_PARTIAL_QC_FLAG").sum())
    reconcile = n_total == n_complete + n_partial + n_zero + int(out["OCO3_production_class"].eq("EXCLUDED_INVALID_SOURCE").sum())
    complete_verdict = (
        "OCO-3 PRODUCTION RECONSTRUCTION COMPLETE"
        if (
            n_partial == n_recon_success + n_partial_failed
            and n_zero > 0
            and source_inputs_unchanged
            and reconcile
        )
        else "OCO-3 PRODUCTION RECONSTRUCTION FAILED"
    )
    counts = pd.DataFrame(
        [
            {"metric": "N_total_OCO3_site_months", "value": n_total},
            {"metric": "N_complete_observed", "value": n_complete},
            {"metric": "N_partial_eligible", "value": n_partial},
            {"metric": "N_partial_successfully_reconstructed", "value": n_recon_success},
            {"metric": "N_partial_failed", "value": n_partial_failed},
            {"metric": "N_zero_coverage_excluded", "value": n_zero},
            {"metric": "N_primary_analysis_ready", "value": n_primary},
            {"metric": "N_QC_flagged", "value": n_qc},
            {"metric": "SOURCE_INPUTS_UNCHANGED", "value": str(source_inputs_unchanged).upper()},
            {"metric": "counts_reconcile", "value": str(reconcile).upper()},
            {"metric": "final_verdict", "value": complete_verdict},
        ]
    )
    counts.to_csv(COUNTS_CSV, index=False)
    return counts


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    d = df.head(max_rows) if max_rows is not None else df
    try:
        return d.to_markdown(index=False)
    except Exception:
        return "```\n" + d.to_string(index=False) + "\n```"


def write_pdf_from_markdown(md: str, path: Path) -> None:
    from matplotlib.backends.backend_pdf import PdfPages

    lines: list[str] = []
    for line in md.splitlines():
        if not line:
            lines.append("")
        else:
            lines.extend(textwrap.wrap(line, width=108, replace_whitespace=False) or [""])
    with PdfPages(path) as pdf:
        for i in range(0, len(lines), 50):
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.06, 0.96, "\n".join(lines[i : i + 50]), va="top", ha="left", family="monospace", fontsize=8.5)
            fig.patch.set_facecolor("white")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def create_simple_summary(out: pd.DataFrame, by_cov: pd.DataFrame, counts: pd.DataFrame, verdict: str) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle("OCO-3 Final EC-Footprint Production Reconstruction", fontsize=18, fontweight="bold")
    classes = {
        "Complete observed": int(out["OCO3_production_class"].eq("COMPLETE_OBSERVED").sum()),
        "Partial reconstructed": int(out["OCO3_production_class"].eq("PARTIAL_RECONSTRUCTION").sum()),
        "Zero excluded": int(out["OCO3_production_class"].eq("ZERO_COVERAGE_EXCLUDED").sum()),
    }
    ax = axes[0, 0]
    ax.bar(classes.keys(), classes.values(), color=["#4C78A8", "#54A24B", "#F58518"])
    ax.set_title("A. Production classes")
    ax.set_ylabel("Site-months")
    ax.tick_params(axis="x", rotation=20)

    ax = axes[0, 1]
    ax.bar(by_cov["coverage_class"], by_cov["N_real_reconstructed_cases"], color="#54A24B")
    ax.set_title("B. Reconstructed partial cases by coverage")
    ax.set_ylabel("Cases")
    ax.tick_params(axis="x", rotation=30)

    recon = out[out["OCO3_production_class"].eq("PARTIAL_RECONSTRUCTION")].copy()
    ax = axes[1, 0]
    ax.scatter(recon["OCO3_observed_partial_EC_SIF"], recon["OCO3_final_EC_SIF"], s=32, alpha=0.75, color="#4C78A8")
    both = pd.concat([recon["OCO3_observed_partial_EC_SIF"], recon["OCO3_final_EC_SIF"]]).dropna()
    if len(both):
        lo, hi = both.min(), both.max()
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
    ax.set_title("C. Partial vs reconstructed SIF")
    ax.set_xlabel("Partial OCO-3 EC SIF")
    ax.set_ylabel("Reconstructed complete EC SIF")

    ax = axes[1, 1]
    ax.scatter(recon["coverage_probability_fraction"], recon["delta_reconstruction"], s=32, alpha=0.75, color="#E45756")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("D. Reconstruction correction by coverage")
    ax.set_xlabel("EC coverage")
    ax.set_ylabel("Reconstructed - partial")

    ax = axes[2, 0]
    ax.scatter(recon["coverage_probability_fraction"], recon["OCO3_PI90_width"], s=30, alpha=0.7, label="90%", color="#72B7B2")
    ax.scatter(recon["coverage_probability_fraction"], recon["OCO3_PI95_width"], s=30, alpha=0.5, label="95%", color="#B279A2")
    ax.set_title("E. Prediction interval width by coverage")
    ax.set_xlabel("EC coverage")
    ax.set_ylabel("Interval width")
    ax.legend()

    ax = axes[2, 1]
    ax.axis("off")
    ax.text(0.5, 0.72, "F. Production status", ha="center", va="center", fontsize=16, fontweight="bold")
    status = "PRODUCTION COMPLETED" if verdict.endswith("COMPLETE") else "PRODUCTION FAILED QC"
    color = "#2CA02C" if verdict.endswith("COMPLETE") else "#D62728"
    ax.text(0.5, 0.48, status, ha="center", va="center", fontsize=24, color=color, fontweight="bold")
    ax.text(0.5, 0.25, "No GPP predictor used. No rasters created. Zero-coverage cases excluded.", ha="center", va="center", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(SUMMARY_PNG, dpi=180)
    fig.savefig(SUMMARY_PDF)
    plt.close(fig)


def write_report_and_readme(
    out: pd.DataFrame,
    primary: pd.DataFrame,
    recon: pd.DataFrame,
    flags: pd.DataFrame,
    by_cov: pd.DataFrame,
    dist: pd.DataFrame,
    dep: pd.DataFrame,
    counts: pd.DataFrame,
    metadata: dict[str, Any],
    conformal: dict[str, float],
    safety_audit: pd.DataFrame,
) -> str:
    verdict = str(counts[counts["metric"].eq("final_verdict")]["value"].iloc[0])
    n_total = int(counts[counts["metric"].eq("N_total_OCO3_site_months")]["value"].iloc[0])
    n_complete = int(counts[counts["metric"].eq("N_complete_observed")]["value"].iloc[0])
    n_partial = int(counts[counts["metric"].eq("N_partial_eligible")]["value"].iloc[0])
    n_recon = int(counts[counts["metric"].eq("N_partial_successfully_reconstructed")]["value"].iloc[0])
    n_failed = int(counts[counts["metric"].eq("N_partial_failed")]["value"].iloc[0])
    n_zero = int(counts[counts["metric"].eq("N_zero_coverage_excluded")]["value"].iloc[0])
    n_primary = int(counts[counts["metric"].eq("N_primary_analysis_ready")]["value"].iloc[0])
    n_qc = int(counts[counts["metric"].eq("N_QC_flagged")]["value"].iloc[0])
    source_unchanged = str(counts[counts["metric"].eq("SOURCE_INPUTS_UNCHANGED")]["value"].iloc[0])
    recon_delta_mean = float(recon["delta_reconstruction"].mean()) if len(recon) else np.nan
    recon_delta_median = float(recon["delta_reconstruction"].median()) if len(recon) else np.nan
    pi90_mean = float(recon["OCO3_PI90_width"].mean()) if len(recon) else np.nan
    pi95_mean = float(recon["OCO3_PI95_width"].mean()) if len(recon) else np.nan

    readme = f"""OCO-3 Final Production Reconstruction — READ ME FIRST

1. Total real OCO-3 site-months: {n_total}.
2. Complete observed: {n_complete}.
3. Partially observed and reconstructed: {n_recon} of {n_partial} eligible.
4. Zero coverage excluded: {n_zero}.
5. Every eligible partial case received a prediction: {'YES' if n_failed == 0 and n_recon == n_partial else 'NO'}.
6. Reconstruction cases with QC flags: {n_qc}.
7. Predictions outside training range: {int(recon['prediction_outside_training_range'].sum()) if len(recon) else 0}; outside training P01-P99: {int(recon['prediction_outside_training_P01_P99'].sum()) if len(recon) else 0}.
8. Mean/median reconstruction correction: {recon_delta_mean:.6f} / {recon_delta_median:.6f}.
9. Typical interval widths: mean PI90={pi90_mean:.6f}, mean PI95={pi95_mean:.6f}.
10. OCO-3 site-months ready for later GPP analysis: {n_primary}.
11. Input/source files modified: {'NO' if source_unchanged == 'TRUE' else 'YES/REVIEW'}.
12. Production status: {verdict}.
"""
    README_TXT.write_text(readme, encoding="utf-8")

    md = f"""# OCO-3 Final Production Reconstruction Report

Final verdict:

```text
{verdict}
```

This run executed the frozen OCO-3 EC-footprint reconstruction methodology. It did not use tower GPP, did not reconstruct zero-coverage OCO-3 cases, and did not create any fine-resolution reconstructed SIF rasters.

## Frozen model

- Model: `RF_TARGET_GOSIF`
- Hyperparameters: `{LOCKED_RF_PARAMS}`
- Feature order: `{FEATURES}`
- Categorical handling: `IGBP` imputed to `Unknown` then one-hot encoded.
- Numeric missing handling: median imputation inside the sklearn pipeline.
- Random seed: `{RANDOM_SEED}`
- Model file: `{MODEL_JOBLIB}`
- Metadata file: `{MODEL_METADATA_JSON}`

## Training data

- Source: `{ARTIFICIAL_GAPS_CSV}`
- Training samples: `{metadata['training_sample_count']}`
- Training original cases: `{metadata['training_original_case_count']}`
- Training regime: positive artificial gaps only, `0 < artificial_coverage_probability < 1`.
- Target: `{TARGET_COLUMN}`.
- GPP predictor used: `False`.

## Production classification

Tolerance around 0 and 1: `{COVERAGE_TOLERANCE}`.

{md_table(counts)}

```text
coverage = 1:
use observed OCO-3 directly

0 < coverage < 1:
reconstruct using frozen RF_TARGET_GOSIF

coverage = 0:
exclude from primary OCO-3 reconstruction/analysis
```

## Output tables

- Master table: `{MASTER_CSV}`
- Primary analysis table: `{PRIMARY_CSV}`
- Reconstructed-only table: `{RECON_ONLY_CSV}`
- QC flags: `{QC_FLAGS_CSV}`

## QC flags

Flagged cases are not removed unless a predefined hard requirement fails.

{md_table(flags, max_rows=40) if len(flags) else "No QC flags."}

## Reconstructed cases by coverage

{md_table(by_cov)}

## Complete vs reconstructed final SIF distributions

{md_table(dist)}

## GOSIF dependence diagnostic

{md_table(dep)}

## Conformal uncertainty

Frozen method: split/grouped conformal absolute-residual interval.

- Calibration N: `{conformal['calibration_n']}`
- q90: `{conformal['q90']:.6f}` → PI90 width `{2 * conformal['q90']:.6f}`
- q95: `{conformal['q95']:.6f}` → PI95 width `{2 * conformal['q95']:.6f}`

## Safety audit

`SOURCE_INPUTS_UNCHANGED = {source_unchanged}`

{md_table(safety_audit)}

## Final counts

```text
N_total_OCO3_site_months = {n_total}
N_complete_observed = {n_complete}
N_partial_eligible = {n_partial}
N_partial_successfully_reconstructed = {n_recon}
N_partial_failed = {n_failed}
N_zero_coverage_excluded = {n_zero}
N_primary_analysis_ready = {n_primary}
N_QC_flagged = {n_qc}
```

## Final verdict

```text
{verdict}
```

After this OCO-3 production run, stop. Do not run TROPOSIF, OCO-2, or final SIF-GPP analysis until these outputs are inspected.
"""
    REPORT_MD.write_text(md, encoding="utf-8")
    write_pdf_from_markdown(md, REPORT_PDF)
    return verdict


def write_reproducibility(metadata: dict[str, Any], counts: pd.DataFrame, source_inputs_unchanged: bool) -> None:
    record_counts = {str(r["metric"]): r["value"] for _, r in counts.iterrows()}
    repro = {
        "script_path": str((Path(PROJECT_ROOT) / 'Codes/pythonProject/Run_OCO3_Final_Production_Reconstruction.py').resolve()),
        "model_file": str(MODEL_JOBLIB),
        "model_metadata": metadata,
        "input_paths": {
            "artificial_gaps_csv": str(ARTIFICIAL_GAPS_CSV),
            "real_coverage_csv": str(REAL_COVERAGE_CSV),
            "oco3_gosif_inventory_csv": str(OCO3_GOSIF_INVENTORY_CSV),
            "model_lock_md": str(MODEL_LOCK_MD),
            "positive_test_report_md": str(POSITIVE_TEST_REPORT_MD),
        },
        "output_paths": {
            "master_csv": str(MASTER_CSV),
            "primary_csv": str(PRIMARY_CSV),
            "reconstructed_only_csv": str(RECON_ONLY_CSV),
            "qc_flags_csv": str(QC_FLAGS_CSV),
            "report_md": str(REPORT_MD),
            "report_pdf": str(REPORT_PDF),
        },
        "feature_names_order": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "hyperparameters": LOCKED_RF_PARAMS,
        "random_seed": RANDOM_SEED,
        "software_versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "record_counts": record_counts,
        "production_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_inputs_unchanged": source_inputs_unchanged,
        "sha256": {
            "model_joblib": sha256_file(MODEL_JOBLIB),
            "model_metadata_json": sha256_file(MODEL_METADATA_JSON),
            "master_csv": sha256_file(MASTER_CSV),
            "primary_csv": sha256_file(PRIMARY_CSV),
            "reconstructed_only_csv": sha256_file(RECON_ONLY_CSV),
        },
    }
    REPRO_JSON.write_text(json.dumps(repro, indent=2), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    require_inputs()
    before = audit_inputs("before")

    train = load_training_data()
    model = train_final_model(train)
    conformal = compute_conformal_quantiles(train)
    joblib.dump(model, MODEL_JOBLIB)
    metadata = write_model_metadata(train, conformal)

    inv = load_production_inventory()
    prod_features = make_production_features(inv)
    out = apply_production(model, prod_features, train, conformal)

    master = select_master_columns(out)
    master.to_csv(MASTER_CSV, index=False)
    primary = master[master["OCO3_production_class"].isin(["COMPLETE_OBSERVED", "PARTIAL_RECONSTRUCTION"]) & ~master["OCO3_final_status"].isin(["EXCLUDED_MISSING_REQUIRED_PREDICTOR", "EXCLUDED_INVALID_SOURCE"])].copy()
    primary.to_csv(PRIMARY_CSV, index=False)
    recon = master[master["OCO3_production_class"].eq("PARTIAL_RECONSTRUCTION")].copy()
    recon.to_csv(RECON_ONLY_CSV, index=False)

    flags, by_cov, dist, dep = create_qc_tables(master)

    after = audit_inputs("after")
    safety = before.merge(after, on=["label", "path", "kind"], suffixes=("_before", "_after"))
    safety["unchanged_size_mtime_fingerprint"] = (
        (safety["file_count_before"] == safety["file_count_after"])
        & (safety["total_size_bytes_before"] == safety["total_size_bytes_after"])
        & (safety["max_mtime_ns_before"] == safety["max_mtime_ns_after"])
        & (safety["fingerprint_before"] == safety["fingerprint_after"])
    )
    source_inputs_unchanged = bool(safety["unchanged_size_mtime_fingerprint"].all())
    master["SOURCE_INPUTS_UNCHANGED"] = source_inputs_unchanged
    master.to_csv(MASTER_CSV, index=False)
    primary = master[master["OCO3_production_class"].isin(["COMPLETE_OBSERVED", "PARTIAL_RECONSTRUCTION"]) & ~master["OCO3_final_status"].isin(["EXCLUDED_MISSING_REQUIRED_PREDICTOR", "EXCLUDED_INVALID_SOURCE"])].copy()
    primary.to_csv(PRIMARY_CSV, index=False)
    recon = master[master["OCO3_production_class"].eq("PARTIAL_RECONSTRUCTION")].copy()
    recon.to_csv(RECON_ONLY_CSV, index=False)

    counts = make_counts(master, flags, source_inputs_unchanged)
    safety.to_csv(SAFETY_AUDIT_CSV, index=False)
    verdict = write_report_and_readme(master, primary, recon, flags, by_cov, dist, dep, counts, metadata, conformal, safety)
    create_simple_summary(master, by_cov, counts, verdict)
    write_reproducibility(metadata, counts, source_inputs_unchanged)

    print("=" * 60)
    print("OCO-3 final production reconstruction complete")
    print(f"Verdict: {verdict}")
    for _, row in counts.iterrows():
        print(f"{row['metric']}: {row['value']}")
    print(f"Master table: {MASTER_CSV}")
    print(f"Primary table: {PRIMARY_CSV}")
    print(f"Report: {REPORT_MD}")
    print(f"PDF: {REPORT_PDF}")
    print("No GPP predictor was used; no SIF/GPP source files were modified; no rasters were created.")


if __name__ == "__main__":
    main()
