# Archive: internal QA scripts (not used in the manuscript)

These scripts were run as internal checks. The current manuscript draft makes no statement that depends on them, so they are kept for transparency only and are not part of the reproducibility mapping.

| Script | What it checks |
|---|---|
| `internal_qa_not_in_manuscript/gome2_quality_flag_sensitivity.py` | Sensitivity of GOME-2 results to quality-flag treatment |
| `internal_qa_not_in_manuscript/gome2_satellite_day_restricted_gpp_sensitivity.py` | GOME-2 comparison using GPP restricted to satellite-overpass days |
| `internal_qa_not_in_manuscript/troposif_ec_weighted_qa_sensitivity.py` | TROPOSIF source-area-weighted QA sensitivity |
| `internal_qa_not_in_manuscript/pos_june_sensitivity.py` | Leave-June-out check of peak-timing patterns (superseded diagnostic) |

The manuscript's peak-fitting sensitivity (harmonic vs periodic-spline fits) is supported by `scripts/06_phenology_peak_timing/year_specific_mechanism/step9_methodological_E_F_G.py`, not by these scripts.
