"""Build the monthly footprint input table for the additional sites: WD-bin histograms, HH-derived conservative GPP first (metadata GPP only as fallback), no invented radii or other missing metadata.

Input: outputs of 01b and 02b, metadata. Output: merged monthly footprint-input CSV."""
# Original script filename (sanitized): Final Merge for Remaining FLUXNET Sites GEE Footprint Input.py; public name: 03b_merge_remaining_sites_footprint_input.py
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
# ============================================================
# Code name: Final Merge for Remaining FLUXNET Sites GEE Footprint Input
#
# Purpose:
#   Build the final GEE-ready monthly footprint input for the
#   remaining/new FLUXNET sites.
#
# Scientific process intentionally preserved:
#   1. Use the existing daytime monthly WD-bin histogram file.
#   2. Use HH-derived conservative monthly GPP first.
#   3. Use monthly metadata GPP only as a fallback for missing GPP.
#   4. Do not invent footprint radius or other missing metadata.
# ============================================================

import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

project_folder = Path(
    PROJECT_ROOT
)

hh_folder = project_folder / "Remaining Fluxnet Sites" / "HH files"

wd_output_folder = hh_folder / "clean_30min_WD_outputs"
gpp_output_folder = hh_folder / "GPP_outputs_for_GEE_conservative"
monthly_metadata_folder = hh_folder / "monthly_GEE_csv_outputs_from_HH_HR"

wd_bins_path = wd_output_folder / "ALL_SITES_monthly_30min_WD_bins.csv"
clean_wd_path = wd_output_folder / "ALL_SITES_clean_30min_WD.csv"
gpp_monthly_path = gpp_output_folder / "ALL_SITES_monthly_GPP_conservative_for_GEE.csv"
metadata_path = project_folder / "New_FLUXNET_sites_metadata_for_KMZ_preparation.xlsx"
monthly_metadata_path = monthly_metadata_folder / "new_sites_fluxnet_monthly_with_metadata_all_sites.csv"

output_folder = hh_folder / "final_monthly_GEE_inputs"
output_folder.mkdir(parents=True, exist_ok=True)

final_output_path = output_folder / "ALL_SITES_monthly_WD_histogram_GPP_footprint_input_for_GEE.csv"
status_output_path = output_folder / "monthly_WD_GPP_merge_status.csv"
invalid_sites_report_path = output_folder / "invalid_sites_report.csv"
site_level_summary_path = output_folder / "site_level_summary.csv"


# ============================================================
# 2. SETTINGS
# ============================================================

footprint_method_label = "monthly_probability_from_30min_WD_histogram"
blind_buffer_method_label = "tower_centered_circular_buffer"
default_gpp_units = "umol_CO2_m-2_s-1"


# ============================================================
# 3. HELPERS
# ============================================================

def check_required_file(path, label):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found:\n{path}")


def clean_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def normalize_column_name(name):
    text = str(name).strip().lower()
    text = re.sub(r"[_\s]+", " ", text)
    return text


def rename_with_aliases(df, alias_map):
    df = clean_columns(df)
    normalized_to_actual = {normalize_column_name(c): c for c in df.columns}
    rename_map = {}

    for canonical, aliases in alias_map.items():
        for alias in aliases:
            actual = normalized_to_actual.get(normalize_column_name(alias))
            if actual is not None:
                rename_map[actual] = canonical
                break

    return df.rename(columns=rename_map)


def text_or_na(value):
    if pd.isna(value):
        return pd.NA

    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null", "<na>"}:
        return pd.NA

    return text


def clean_text_column(series):
    return series.map(text_or_na)


def parse_number_or_range(value):
    """
    Parse positive numeric values and ranges such as:
    5.17, 1700-1850, 1700 to 1850, 1.5-3.0, 1.5 to 3.0.
    Returns min_value, max_value.
    """
    if pd.isna(value):
        return np.nan, np.nan

    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null", "<na>"}:
        return np.nan, np.nan

    text = (
        text.replace(",", ".")
        .replace("~", " ")
        .replace(">", " ")
        .replace("<", " ")
        .replace("=", " ")
    )

    numbers = re.findall(r"\d+(?:\.\d+)?", text)

    if not numbers:
        return np.nan, np.nan

    values = [float(x) for x in numbers]
    return min(values), max(values)


def coerce_key_columns(df, site_col="SITE_ID"):
    df = df.copy()
    df[site_col] = df[site_col].astype(str).str.strip()
    df["YYYYMM"] = pd.to_numeric(df["YYYYMM"], errors="coerce").astype("Int64")
    df = df[df[site_col].notna() & df["YYYYMM"].notna()].copy()
    df["YYYYMM"] = df["YYYYMM"].astype(int)
    return df


def first_non_missing(series):
    for value in series:
        if not pd.isna(value):
            if isinstance(value, str):
                clean = value.strip()
                if clean and clean.lower() not in {"nan", "none", "null", "<na>"}:
                    return value
            else:
                return value
    return pd.NA


def fill_missing_from(df, target_col, fallback_col):
    if target_col not in df.columns:
        df[target_col] = pd.NA

    if fallback_col not in df.columns:
        return df

    use_fallback = df[target_col].isna() & df[fallback_col].notna()
    if not use_fallback.any():
        return df

    df.loc[use_fallback, target_col] = df.loc[use_fallback, fallback_col]
    return df


def build_invalid_reason_from_flags(row):
    reason_cols = ["missing_GPP", "missing_radius", "missing_lat_lon", "missing_WD_bins"]
    reasons = [col for col in reason_cols if int(row.get(col, 0)) == 1]
    if reasons:
        return "; ".join(reasons)
    if int(row.get("valid_for_GEE_export", 0)) == 1 or int(row.get("valid_months", 0)) > 0:
        return "valid"
    return "invalid_other"


def make_radius_notes_truthful(df):
    df = df.copy()
    if "Notes" not in df.columns:
        df["Notes"] = pd.NA

    radius = pd.to_numeric(df["radius_m_max"], errors="coerce")
    missing_radius = radius.isna() | (radius <= 0)
    df.loc[missing_radius, "Notes"] = (
        "Radius is still missing; manual verification is needed."
    )
    return df


# ============================================================
# 4. METADATA
# ============================================================

metadata_aliases = {
    "SITE_ID": ["SITE_ID", "Site_ID", "Site ID", "Site", "site", "site_id", "SITE"],
    "site_name": ["site_name", "SITE_NAME", "Site name", "Site Name"],
    "lat": ["lat", "Lat", "LAT", "Latitude", "latitude"],
    "lon": ["lon", "Lon", "LON", "Longitude", "longitude"],
    "IGBP": ["IGBP", "igbp", "Igbp"],
    "canopy_height_m": ["Canopy height m", "Canopy height", "canopy_height_m"],
    "EC_measurement_height_m": [
        "EC measurement height m",
        "EC measurement height",
        "EC_measurement_height_m",
    ],
    "height_above_canopy_m": [
        "Height above canopy m",
        "Height above canopy",
        "height_above_canopy_m",
    ],
    "radius_m": [
        "circle buffer radius m",
        "Circle buffer radius m",
        "Circular buffer radius m",
        "circular buffer radius m",
        "radius_m",
    ],
    "FLUXNET_page": ["FLUXNET page", "FLUXNET_page", "Fluxnet page"],
    "Metadata source": ["Metadata source", "metadata source"],
    "Metadata status": ["Metadata status", "metadata status"],
    "Notes": ["Notes", "notes"],
}


def standardize_site_metadata(metadata_raw):
    metadata = rename_with_aliases(metadata_raw, metadata_aliases)

    required_cols = ["SITE_ID"]
    missing_required = [c for c in required_cols if c not in metadata.columns]
    if missing_required:
        raise ValueError(f"Metadata file is missing required columns: {missing_required}")

    for col in [
        "site_name",
        "IGBP",
        "FLUXNET_page",
        "Metadata source",
        "Metadata status",
        "Notes",
    ]:
        if col not in metadata.columns:
            metadata[col] = pd.NA
        metadata[col] = clean_text_column(metadata[col])

    metadata["SITE_ID"] = metadata["SITE_ID"].astype(str).str.strip()

    for col in ["lat", "lon"]:
        if col not in metadata.columns:
            metadata[col] = np.nan
        metadata[col] = pd.to_numeric(metadata[col], errors="coerce")

    for col in [
        "canopy_height_m",
        "EC_measurement_height_m",
        "height_above_canopy_m",
        "radius_m",
    ]:
        if col not in metadata.columns:
            metadata[col] = np.nan

        parsed = metadata[col].apply(parse_number_or_range)
        metadata[f"{col}_min"] = parsed.map(lambda x: x[0])
        metadata[f"{col}_max"] = parsed.map(lambda x: x[1])
        metadata[col] = metadata[f"{col}_max"]

    metadata["height_calculated_from_canopy_and_EC"] = 0
    missing_height = metadata["height_above_canopy_m"].isna()
    can_calculate_height = (
        metadata["EC_measurement_height_m"].notna()
        & metadata["canopy_height_m"].notna()
    )
    calculated_height = metadata["EC_measurement_height_m"] - metadata["canopy_height_m"]
    use_calculated_height = missing_height & can_calculate_height & (calculated_height > 0)

    metadata.loc[use_calculated_height, "height_above_canopy_m"] = calculated_height.loc[
        use_calculated_height
    ]
    metadata.loc[use_calculated_height, "height_above_canopy_m_min"] = calculated_height.loc[
        use_calculated_height
    ]
    metadata.loc[use_calculated_height, "height_above_canopy_m_max"] = calculated_height.loc[
        use_calculated_height
    ]
    metadata.loc[use_calculated_height, "height_calculated_from_canopy_and_EC"] = 1

    metadata["radius_calculated_from_height"] = 0
    missing_radius = metadata["radius_m_max"].isna()
    can_calculate_radius = (
        metadata["height_above_canopy_m"].notna()
        & (metadata["height_above_canopy_m"] > 0)
    )
    calculated_radius = metadata["height_above_canopy_m"] * 100
    use_calculated_radius = missing_radius & can_calculate_radius

    metadata.loc[use_calculated_radius, "radius_m"] = calculated_radius.loc[use_calculated_radius]
    metadata.loc[use_calculated_radius, "radius_m_min"] = calculated_radius.loc[
        use_calculated_radius
    ]
    metadata.loc[use_calculated_radius, "radius_m_max"] = calculated_radius.loc[
        use_calculated_radius
    ]
    metadata.loc[use_calculated_radius, "radius_calculated_from_height"] = 1

    keep_cols = [
        "SITE_ID",
        "site_name",
        "lat",
        "lon",
        "IGBP",
        "canopy_height_m",
        "EC_measurement_height_m",
        "height_above_canopy_m",
        "radius_m",
        "radius_m_min",
        "radius_m_max",
        "FLUXNET_page",
        "Metadata source",
        "Metadata status",
        "Notes",
        "height_calculated_from_canopy_and_EC",
        "radius_calculated_from_height",
    ]

    metadata = metadata[keep_cols].drop_duplicates(subset=["SITE_ID"]).copy()
    return metadata


def load_monthly_site_metadata(monthly_metadata_path):
    monthly = pd.read_csv(monthly_metadata_path, low_memory=False)
    monthly = clean_columns(monthly)

    monthly = monthly.rename(
        columns={
            "site_id": "SITE_ID",
            "igbp": "IGBP",
            "FLUXNET_page": "FLUXNET_page",
        }
    )

    for col in ["SITE_ID", "site_name", "IGBP", "FLUXNET_page", "Metadata source", "Metadata status", "Notes"]:
        if col not in monthly.columns:
            monthly[col] = pd.NA

    for col in ["lat", "lon"]:
        if col not in monthly.columns:
            monthly[col] = np.nan
        monthly[col] = pd.to_numeric(monthly[col], errors="coerce")

    monthly["SITE_ID"] = monthly["SITE_ID"].astype(str).str.strip()

    site_metadata = (
        monthly.groupby("SITE_ID", as_index=False)
        .agg(
            site_name=("site_name", first_non_missing),
            lat=("lat", first_non_missing),
            lon=("lon", first_non_missing),
            IGBP=("IGBP", first_non_missing),
            FLUXNET_page=("FLUXNET_page", first_non_missing),
            **{
                "Metadata source": ("Metadata source", first_non_missing),
                "Metadata status": ("Metadata status", first_non_missing),
                "Notes": ("Notes", first_non_missing),
            },
        )
    )

    return site_metadata


def apply_monthly_metadata_fallback(metadata, monthly_site_metadata):
    fallback_cols = [
        "site_name",
        "lat",
        "lon",
        "IGBP",
        "FLUXNET_page",
        "Metadata source",
        "Metadata status",
        "Notes",
    ]

    merged = metadata.merge(
        monthly_site_metadata,
        on="SITE_ID",
        how="outer",
        suffixes=("", "_monthly_fallback"),
    )

    for col in fallback_cols:
        merged = fill_missing_from(merged, col, f"{col}_monthly_fallback")

    drop_cols = [c for c in merged.columns if c.endswith("_monthly_fallback")]
    merged = merged.drop(columns=drop_cols)

    for col in ["height_calculated_from_canopy_and_EC", "radius_calculated_from_height"]:
        if col not in merged.columns:
            merged[col] = 0
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0).astype(int)

    for col in ["lat", "lon", "radius_m", "radius_m_min", "radius_m_max"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    return merged


# ============================================================
# 5. GPP
# ============================================================

def prepare_conservative_gpp(gpp_raw):
    gpp = clean_columns(gpp_raw)

    if "SITE_ID" not in gpp.columns or "YYYYMM" not in gpp.columns:
        raise ValueError("Conservative GPP file must contain SITE_ID and YYYYMM.")

    gpp = coerce_key_columns(gpp)

    if "tower_GPP_monthly_conservative_mean_of_daily" in gpp.columns:
        gpp["tower_GPP_monthly"] = gpp["tower_GPP_monthly_conservative_mean_of_daily"]
    elif "tower_GPP_monthly" not in gpp.columns:
        raise ValueError(
            "Could not find conservative monthly GPP column. Expected "
            "tower_GPP_monthly_conservative_mean_of_daily or tower_GPP_monthly."
        )

    gpp["tower_GPP_monthly"] = pd.to_numeric(gpp["tower_GPP_monthly"], errors="coerce")

    if "primary_GPP_column" not in gpp.columns:
        gpp["primary_GPP_column"] = "GPP_DT_VUT_REF"

    if "GPP_monthly_quality_flag" not in gpp.columns:
        gpp["GPP_monthly_quality_flag"] = "UNKNOWN"

    if "tower_GPP_units_original_FLUXNET" not in gpp.columns:
        gpp["tower_GPP_units_original_FLUXNET"] = default_gpp_units

    keep_cols = [
        "SITE_ID",
        "YYYYMM",
        "tower_GPP_monthly",
        "primary_GPP_column",
        "GPP_monthly_quality_flag",
        "tower_GPP_units_original_FLUXNET",
    ]

    return gpp[keep_cols].copy()


def prepare_monthly_gpp_fallback(monthly_metadata_path):
    header = pd.read_csv(monthly_metadata_path, nrows=0).columns
    header_map = {normalize_column_name(c): c for c in header}

    site_col = header_map.get("site id") or header_map.get("site_id") or header_map.get("site")
    yyyymm_col = header_map.get("yyyymm")

    if site_col is None or yyyymm_col is None:
        raise ValueError("Monthly metadata file must contain site_id and YYYYMM columns.")

    candidate_cols = []
    for col in header:
        col_norm = normalize_column_name(col)
        if "gpp" not in col_norm:
            continue
        if any(token in col_norm for token in ["qc", "flag", "valid", "source"]):
            continue
        candidate_cols.append(col)

    preferred_cols = []
    for preferred in ["gpp_dt_ref", "gpp_nt_ref"]:
        actual = header_map.get(normalize_column_name(preferred))
        if actual is not None and actual not in preferred_cols:
            preferred_cols.append(actual)

    for col in candidate_cols:
        if col not in preferred_cols:
            preferred_cols.append(col)

    usecols = [site_col, yyyymm_col] + preferred_cols
    monthly_gpp = pd.read_csv(monthly_metadata_path, usecols=usecols, low_memory=False)
    monthly_gpp = clean_columns(monthly_gpp)
    monthly_gpp = monthly_gpp.rename(columns={site_col: "SITE_ID", yyyymm_col: "YYYYMM"})
    monthly_gpp = coerce_key_columns(monthly_gpp)

    for col in preferred_cols:
        actual_col = col
        if actual_col in monthly_gpp.columns:
            monthly_gpp[actual_col] = pd.to_numeric(monthly_gpp[actual_col], errors="coerce")

    dt_col = header_map.get("gpp dt ref")
    nt_col = header_map.get("gpp nt ref")
    other_cols = [c for c in preferred_cols if c not in {dt_col, nt_col}]

    def select_fallback(row):
        if dt_col in monthly_gpp.columns and not pd.isna(row[dt_col]):
            return row[dt_col], "monthly_metadata_gpp_dt_ref_fallback", "gpp_dt_ref"

        if nt_col in monthly_gpp.columns and not pd.isna(row[nt_col]):
            return row[nt_col], "monthly_metadata_gpp_nt_ref_fallback", "gpp_nt_ref"

        for col in other_cols:
            if col in monthly_gpp.columns and not pd.isna(row[col]):
                return row[col], f"monthly_metadata_{col}_fallback", col

        return np.nan, "missing_GPP", pd.NA

    fallback = monthly_gpp.apply(select_fallback, axis=1, result_type="expand")
    monthly_gpp["fallback_GPP_monthly"] = fallback[0]
    monthly_gpp["fallback_GPP_source"] = fallback[1]
    monthly_gpp["fallback_primary_GPP_column"] = fallback[2]

    return monthly_gpp[
        [
            "SITE_ID",
            "YYYYMM",
            "fallback_GPP_monthly",
            "fallback_GPP_source",
            "fallback_primary_GPP_column",
        ]
    ].copy()


# ============================================================
# 6. WD SUPPORT METRICS
# ============================================================

def aggregate_clean_wd_support(clean_wd_path):
    expected_output = pd.DataFrame(
        columns=[
            "SITE_ID",
            "YYYYMM",
            "mean_monthly_daytime_WS_F",
            "mean_monthly_daytime_USTAR",
            "mean_monthly_daytime_SW_IN_F",
            "daytime_filter_used",
        ]
    )

    if not clean_wd_path.exists():
        print("\nWARNING: Clean WD file not found, monthly WS/USTAR/SW_IN fields will be blank:")
        print(clean_wd_path)
        return expected_output

    usecols = {
        "SITE_ID",
        "YYYYMM",
        "WS_F",
        "USTAR",
        "SW_IN_F",
        "daytime_filter_used",
    }

    numeric_parts = []
    filter_parts = []

    print("\nAggregating monthly daytime WS/USTAR/SW_IN from clean WD file:")
    print(clean_wd_path)

    for chunk in pd.read_csv(
        clean_wd_path,
        usecols=lambda c: str(c).strip() in usecols,
        chunksize=500_000,
        low_memory=False,
    ):
        chunk = clean_columns(chunk)
        if "SITE_ID" not in chunk.columns or "YYYYMM" not in chunk.columns:
            continue

        chunk["SITE_ID"] = chunk["SITE_ID"].astype(str).str.strip()
        chunk["YYYYMM"] = pd.to_numeric(chunk["YYYYMM"], errors="coerce")
        chunk = chunk[chunk["YYYYMM"].notna()].copy()
        chunk["YYYYMM"] = chunk["YYYYMM"].astype(int)

        for col in ["WS_F", "USTAR", "SW_IN_F"]:
            if col not in chunk.columns:
                chunk[col] = np.nan
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        grouped_numeric = (
            chunk.groupby(["SITE_ID", "YYYYMM"], as_index=False)
            .agg(
                WS_F_sum=("WS_F", "sum"),
                WS_F_count=("WS_F", "count"),
                USTAR_sum=("USTAR", "sum"),
                USTAR_count=("USTAR", "count"),
                SW_IN_F_sum=("SW_IN_F", "sum"),
                SW_IN_F_count=("SW_IN_F", "count"),
            )
        )
        numeric_parts.append(grouped_numeric)

        if "daytime_filter_used" in chunk.columns:
            grouped_filter = (
                chunk.groupby(["SITE_ID", "YYYYMM"], as_index=False)
                .agg(daytime_filter_used=("daytime_filter_used", first_non_missing))
            )
            filter_parts.append(grouped_filter)

    if not numeric_parts:
        return expected_output

    numeric = pd.concat(numeric_parts, ignore_index=True)
    numeric = (
        numeric.groupby(["SITE_ID", "YYYYMM"], as_index=False)
        .sum(numeric_only=True)
    )

    numeric["mean_monthly_daytime_WS_F"] = numeric["WS_F_sum"] / numeric["WS_F_count"]
    numeric["mean_monthly_daytime_USTAR"] = numeric["USTAR_sum"] / numeric["USTAR_count"]
    numeric["mean_monthly_daytime_SW_IN_F"] = (
        numeric["SW_IN_F_sum"] / numeric["SW_IN_F_count"]
    )

    support = numeric[
        [
            "SITE_ID",
            "YYYYMM",
            "mean_monthly_daytime_WS_F",
            "mean_monthly_daytime_USTAR",
            "mean_monthly_daytime_SW_IN_F",
        ]
    ].copy()

    if filter_parts:
        filters = pd.concat(filter_parts, ignore_index=True)
        filters = (
            filters.groupby(["SITE_ID", "YYYYMM"], as_index=False)
            .agg(daytime_filter_used=("daytime_filter_used", first_non_missing))
        )
        support = support.merge(filters, on=["SITE_ID", "YYYYMM"], how="left")
    else:
        support["daytime_filter_used"] = pd.NA

    return support


# ============================================================
# 7. READ INPUTS
# ============================================================

for required_path, label in [
    (wd_bins_path, "WD monthly bins file"),
    (gpp_monthly_path, "Conservative monthly GPP file"),
    (metadata_path, "New site metadata Excel file"),
    (monthly_metadata_path, "Monthly metadata fallback file"),
]:
    check_required_file(required_path, label)

print("Input files:")
print(f"WD monthly bins:\n{wd_bins_path}")
print(f"\nConservative monthly GPP:\n{gpp_monthly_path}")
print(f"\nMetadata Excel:\n{metadata_path}")
print(f"\nMonthly metadata fallback:\n{monthly_metadata_path}")
print(f"\nOutput folder:\n{output_folder}")

wd = pd.read_csv(wd_bins_path, low_memory=False)
wd = clean_columns(wd)

required_wd_cols = [
    "SITE_ID",
    "YYYYMM",
    "WD_bin_center_deg",
    "WD_bin_fraction",
]
missing_wd_cols = [c for c in required_wd_cols if c not in wd.columns]
if missing_wd_cols:
    raise ValueError(f"WD monthly bin file is missing required columns: {missing_wd_cols}")

wd = coerce_key_columns(wd)
wd["year"] = pd.to_numeric(wd.get("year"), errors="coerce")
wd["month"] = pd.to_numeric(wd.get("month"), errors="coerce")
missing_year = wd["year"].isna()
missing_month = wd["month"].isna()
wd.loc[missing_year, "year"] = wd.loc[missing_year, "YYYYMM"] // 100
wd.loc[missing_month, "month"] = wd.loc[missing_month, "YYYYMM"] % 100
wd["year"] = wd["year"].astype(int)
wd["month"] = wd["month"].astype(int)

for col in [
    "WD_bin_width_deg",
    "WD_bin_start_deg",
    "WD_bin_end_deg",
    "WD_bin_center_deg",
    "WD_bin_count",
    "WD_bin_fraction",
    "n_valid_monthly_30min_WD_records",
    "monthly_circular_mean_WD_deg_diagnostic",
    "monthly_WD_consistency_R_diagnostic",
]:
    if col in wd.columns:
        wd[col] = pd.to_numeric(wd[col], errors="coerce")

conservative_gpp = prepare_conservative_gpp(pd.read_csv(gpp_monthly_path, low_memory=False))
fallback_gpp = prepare_monthly_gpp_fallback(monthly_metadata_path)

metadata = standardize_site_metadata(pd.read_excel(metadata_path))
monthly_site_metadata = load_monthly_site_metadata(monthly_metadata_path)
metadata = apply_monthly_metadata_fallback(metadata, monthly_site_metadata)
metadata = make_radius_notes_truthful(metadata)

wd_support = aggregate_clean_wd_support(clean_wd_path)


# ============================================================
# 8. MERGE
# ============================================================

merged = wd.merge(
    conservative_gpp,
    on=["SITE_ID", "YYYYMM"],
    how="left",
)

merged = merged.merge(
    fallback_gpp,
    on=["SITE_ID", "YYYYMM"],
    how="left",
)

merged["GPP_source"] = np.where(
    merged["tower_GPP_monthly"].notna(),
    "HH_conservative",
    "missing_GPP",
)

needs_gpp_fallback = merged["tower_GPP_monthly"].isna() & merged["fallback_GPP_monthly"].notna()
merged.loc[needs_gpp_fallback, "tower_GPP_monthly"] = merged.loc[
    needs_gpp_fallback, "fallback_GPP_monthly"
]
merged.loc[needs_gpp_fallback, "GPP_source"] = merged.loc[
    needs_gpp_fallback, "fallback_GPP_source"
]
merged.loc[needs_gpp_fallback, "primary_GPP_column"] = merged.loc[
    needs_gpp_fallback, "fallback_primary_GPP_column"
]
merged.loc[needs_gpp_fallback, "GPP_monthly_quality_flag"] = "MONTHLY_METADATA_FALLBACK"

merged["tower_GPP_units_original_FLUXNET"] = merged[
    "tower_GPP_units_original_FLUXNET"
].fillna(default_gpp_units)

merged["GPP_merge_status"] = np.select(
    [
        merged["GPP_source"].eq("HH_conservative"),
        merged["GPP_source"].str.contains("fallback", na=False),
    ],
    [
        "HH_conservative_GPP_FOUND",
        "MONTHLY_METADATA_GPP_FALLBACK",
    ],
    default="NO_GPP_FOR_SITE_MONTH",
)

merged = merged.drop(
    columns=[
        "fallback_GPP_monthly",
        "fallback_GPP_source",
        "fallback_primary_GPP_column",
    ],
    errors="ignore",
)

merged = merged.merge(metadata, on="SITE_ID", how="left")
merged = merged.merge(wd_support, on=["SITE_ID", "YYYYMM"], how="left")


# ============================================================
# 9. FLAGS AND FINAL COLUMNS
# ============================================================

merged["tower_GPP_monthly"] = pd.to_numeric(merged["tower_GPP_monthly"], errors="coerce")
merged["has_GPP"] = merged["tower_GPP_monthly"].notna().astype(int)
merged["has_radius"] = (
    merged["radius_m_max"].notna()
    & (pd.to_numeric(merged["radius_m_max"], errors="coerce") > 0)
).astype(int)
merged["has_lat_lon"] = (merged["lat"].notna() & merged["lon"].notna()).astype(int)

merged["valid_for_GEE_export"] = (
    merged["lat"].notna()
    & merged["lon"].notna()
    & merged["radius_m_max"].notna()
    & (pd.to_numeric(merged["radius_m_max"], errors="coerce") > 0)
    & merged["WD_bin_center_deg"].notna()
    & merged["WD_bin_fraction"].notna()
    & merged["tower_GPP_monthly"].notna()
).astype(int)

merged["footprint_method"] = footprint_method_label
merged["blind_buffer_method"] = blind_buffer_method_label

for col in [
    "WD_bin_width_deg",
    "WD_bin_start_deg",
    "WD_bin_end_deg",
    "WD_bin_center_deg",
    "WD_bin_count",
    "WD_bin_fraction",
    "n_valid_monthly_30min_WD_records",
    "monthly_circular_mean_WD_deg_diagnostic",
    "monthly_WD_consistency_R_diagnostic",
    "tower_GPP_monthly",
    "canopy_height_m",
    "EC_measurement_height_m",
    "height_above_canopy_m",
    "radius_m",
    "radius_m_min",
    "radius_m_max",
    "mean_monthly_daytime_WS_F",
    "mean_monthly_daytime_USTAR",
    "mean_monthly_daytime_SW_IN_F",
]:
    if col not in merged.columns:
        merged[col] = np.nan

for col in [
    "WD_bin_center_cardinal",
    "monthly_circular_mean_WD_cardinal_diagnostic",
    "primary_GPP_column",
    "GPP_monthly_quality_flag",
    "tower_GPP_units_original_FLUXNET",
    "daytime_filter_used",
    "quality_flag",
    "FLUXNET_page",
    "Metadata source",
    "Metadata status",
    "Notes",
]:
    if col not in merged.columns:
        merged[col] = pd.NA

merged = make_radius_notes_truthful(merged)

final_cols = [
    "SITE_ID",
    "site_name",
    "lat",
    "lon",
    "IGBP",
    "YYYYMM",
    "year",
    "month",
    "WD_bin_width_deg",
    "WD_bin_start_deg",
    "WD_bin_end_deg",
    "WD_bin_center_deg",
    "WD_bin_center_cardinal",
    "WD_bin_count",
    "WD_bin_fraction",
    "n_valid_monthly_30min_WD_records",
    "monthly_circular_mean_WD_deg_diagnostic",
    "monthly_circular_mean_WD_cardinal_diagnostic",
    "monthly_WD_consistency_R_diagnostic",
    "tower_GPP_monthly",
    "primary_GPP_column",
    "GPP_source",
    "GPP_monthly_quality_flag",
    "tower_GPP_units_original_FLUXNET",
    "canopy_height_m",
    "EC_measurement_height_m",
    "height_above_canopy_m",
    "radius_m",
    "radius_m_min",
    "radius_m_max",
    "mean_monthly_daytime_WS_F",
    "mean_monthly_daytime_USTAR",
    "mean_monthly_daytime_SW_IN_F",
    "daytime_filter_used",
    "quality_flag",
    "footprint_method",
    "blind_buffer_method",
    "GPP_merge_status",
    "has_GPP",
    "has_radius",
    "has_lat_lon",
    "valid_for_GEE_export",
    "FLUXNET_page",
    "Metadata source",
    "Metadata status",
    "Notes",
]

merged = merged[final_cols].copy()


# ============================================================
# 10. QC REPORTS
# ============================================================

status = (
    merged.groupby(["SITE_ID", "YYYYMM"], as_index=False)
    .agg(
        n_WD_bins=("WD_bin_center_deg", lambda x: int(x.notna().sum())),
        WD_fraction_sum=("WD_bin_fraction", lambda x: x.sum(min_count=1)),
        has_GPP=("has_GPP", "max"),
        has_radius=("has_radius", "max"),
        has_lat_lon=("has_lat_lon", "max"),
        valid_for_GEE_export=("valid_for_GEE_export", "max"),
        tower_GPP_monthly=("tower_GPP_monthly", first_non_missing),
        GPP_source=("GPP_source", first_non_missing),
        radius_m_max=("radius_m_max", first_non_missing),
    )
)

for col in ["has_GPP", "has_radius", "has_lat_lon", "valid_for_GEE_export"]:
    status[col] = pd.to_numeric(status[col], errors="coerce").fillna(0).astype(int)

status["WD_fraction_sum"] = pd.to_numeric(status["WD_fraction_sum"], errors="coerce")
status["radius_m_max"] = pd.to_numeric(status["radius_m_max"], errors="coerce")
status["tower_GPP_monthly"] = pd.to_numeric(status["tower_GPP_monthly"], errors="coerce")
status["has_WD_bins"] = (
    status["n_WD_bins"].gt(0) & status["WD_fraction_sum"].notna()
).astype(int)
status["missing_GPP"] = status["has_GPP"].eq(0).astype(int)
status["missing_radius"] = status["has_radius"].eq(0).astype(int)
status["missing_lat_lon"] = status["has_lat_lon"].eq(0).astype(int)
status["missing_WD_bins"] = status["has_WD_bins"].eq(0).astype(int)
status["invalid_reason"] = status.apply(build_invalid_reason_from_flags, axis=1)

wd_fraction_bad = status[
    (status["WD_fraction_sum"] < 0.99)
    | (status["WD_fraction_sum"] > 1.01)
].copy()

status = status[
    [
        "SITE_ID",
        "YYYYMM",
        "n_WD_bins",
        "WD_fraction_sum",
        "has_GPP",
        "has_radius",
        "has_lat_lon",
        "valid_for_GEE_export",
        "tower_GPP_monthly",
        "GPP_source",
        "radius_m_max",
        "invalid_reason",
    ]
].copy()

target_sites = sorted(metadata["SITE_ID"].dropna().astype(str).str.strip().unique())
site_summary = pd.DataFrame({"SITE_ID": target_sites})

site_row_summary = (
    merged.groupby("SITE_ID", as_index=False)
    .agg(
        final_rows=("SITE_ID", "size"),
        has_any_GPP=("has_GPP", "max"),
        has_any_radius=("has_radius", "max"),
        has_any_lat_lon=("has_lat_lon", "max"),
        has_WD_bins=("WD_bin_center_deg", lambda x: int(x.notna().any())),
    )
)

site_month_summary = (
    status.groupby("SITE_ID", as_index=False)
    .agg(
        site_months=("YYYYMM", "nunique"),
        valid_months=("valid_for_GEE_export", "sum"),
    )
)
site_month_summary["invalid_months"] = (
    site_month_summary["site_months"] - site_month_summary["valid_months"]
)

site_summary = site_summary.merge(site_row_summary, on="SITE_ID", how="left")
site_summary = site_summary.merge(site_month_summary, on="SITE_ID", how="left")

for col in [
    "final_rows",
    "site_months",
    "valid_months",
    "invalid_months",
    "has_any_GPP",
    "has_any_radius",
    "has_any_lat_lon",
    "has_WD_bins",
]:
    site_summary[col] = pd.to_numeric(site_summary[col], errors="coerce").fillna(0).astype(int)

metadata_summary = metadata[
    [
        "SITE_ID",
        "site_name",
        "lat",
        "lon",
        "IGBP",
        "radius_m_max",
        "radius_calculated_from_height",
        "height_calculated_from_canopy_and_EC",
        "Metadata source",
        "Metadata status",
        "Notes",
    ]
].copy()

site_summary = site_summary.merge(metadata_summary, on="SITE_ID", how="left")

fallback_gpp_sites = (
    merged.loc[merged["GPP_source"].str.contains("fallback", na=False), "SITE_ID"]
    .drop_duplicates()
    .sort_values()
    .tolist()
)

site_summary["recovered_by_fallback_GPP"] = site_summary["SITE_ID"].isin(
    fallback_gpp_sites
).astype(int)

site_summary["missing_GPP"] = site_summary["has_any_GPP"].eq(0).astype(int)
site_summary["missing_radius"] = site_summary["has_any_radius"].eq(0).astype(int)
site_summary["missing_lat_lon"] = site_summary["has_any_lat_lon"].eq(0).astype(int)
site_summary["missing_WD_bins"] = site_summary["has_WD_bins"].eq(0).astype(int)
site_summary["invalid_reason"] = site_summary.apply(build_invalid_reason_from_flags, axis=1)

invalid_report_cols = [
    "SITE_ID",
    "final_rows",
    "site_months",
    "valid_months",
    "invalid_months",
    "has_any_GPP",
    "has_any_radius",
    "has_any_lat_lon",
    "has_WD_bins",
    "missing_GPP",
    "missing_radius",
    "missing_lat_lon",
    "missing_WD_bins",
    "invalid_reason",
]
invalid_sites_report = site_summary[invalid_report_cols].copy()
zero_valid_sites = site_summary[site_summary["valid_months"] == 0].copy()

duplicate_key_cols = ["SITE_ID", "YYYYMM", "WD_bin_center_deg"]
duplicate_site_month_bin_rows = int(merged.duplicated(subset=duplicate_key_cols).sum())


# ============================================================
# 11. SAVE OUTPUTS
# ============================================================

merged.to_csv(final_output_path, index=False)
status.to_csv(status_output_path, index=False)
invalid_sites_report.to_csv(invalid_sites_report_path, index=False)
site_summary.to_csv(site_level_summary_path, index=False)


# ============================================================
# 12. TERMINAL OUTPUT
# ============================================================

valid_site_months = int(status["valid_for_GEE_export"].sum())
invalid_site_months = int((status["valid_for_GEE_export"] == 0).sum())
total_target_sites = len(target_sites)
sites_in_final_csv = int(merged["SITE_ID"].nunique())
sites_with_no_wd_rows = int(site_summary["missing_WD_bins"].sum())
valid_sites = int(site_summary["valid_months"].gt(0).sum())
zero_valid_site_count = int(site_summary["valid_months"].eq(0).sum())
wd_fraction_min = status["WD_fraction_sum"].min()
wd_fraction_max = status["WD_fraction_sum"].max()
wd_fraction_bad_count = len(wd_fraction_bad)

print("\nDONE.")
print("\nFinal GEE upload CSV:")
print(final_output_path)
print("\nStatus CSV:")
print(status_output_path)
print("\nInvalid sites report:")
print(invalid_sites_report_path)
print("\nSite-level summary:")
print(site_level_summary_path)

print("\nQC summary:")
print("Total target sites:", total_target_sites)
print("Sites in final CSV:", sites_in_final_csv)
print("Sites with no WD rows:", sites_with_no_wd_rows)
print("Valid sites:", valid_sites)
print("Zero-valid sites:", zero_valid_site_count)
print("Total site-months:", len(status))
print("Valid site-months:", valid_site_months)
print("Invalid site-months:", invalid_site_months)
print("WD_fraction_sum min:", wd_fraction_min)
print("WD_fraction_sum max:", wd_fraction_max)
print("Site-months outside 0.99 to 1.01:", wd_fraction_bad_count)
print(
    "Duplicate SITE_ID + YYYYMM + WD_bin_center_deg rows:",
    duplicate_site_month_bin_rows,
)
print("Final rows:", len(merged))
print("Valid rows for GEE export:", int(merged["valid_for_GEE_export"].sum()))

print("\nWD fraction checks:")
if wd_fraction_bad.empty:
    print("All SITE_ID + YYYYMM WD_fraction_sum values are within 0.99 to 1.01.")
else:
    print("Site-months outside 0.99 to 1.01:")
    print(wd_fraction_bad[["SITE_ID", "YYYYMM", "WD_fraction_sum"]].to_string(index=False))

print("\nSites recovered by fallback GPP:")
if fallback_gpp_sites:
    print(fallback_gpp_sites)
else:
    print("None.")

radius_recovered_sites = (
    metadata.loc[metadata["radius_calculated_from_height"] == 1, "SITE_ID"]
    .drop_duplicates()
    .sort_values()
    .tolist()
)

print("\nSites recovered by fallback radius calculation:")
if radius_recovered_sites:
    print(radius_recovered_sites)
else:
    print("None.")

print("\nSites with zero valid months and reasons:")
if zero_valid_sites.empty:
    print("None.")
else:
    print(
        zero_valid_sites[
            [
                "SITE_ID",
                "site_name",
                "valid_months",
                "missing_GPP",
                "missing_radius",
                "missing_lat_lon",
                "missing_WD_bins",
                "invalid_reason",
            ]
        ].to_string(index=False)
    )
