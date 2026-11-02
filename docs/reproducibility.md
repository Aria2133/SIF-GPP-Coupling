# ♻️ Reproducibility

This page states what can and cannot be reproduced from this repository **as verified against the original project files**.

> **Master-table provenance.** `SIF_GPP_Analysis_Ready_Master.csv` is the common analysis-ready input used by most downstream analyses. The exact upstream script that generated this final master table **could not be recovered**. The repository therefore supports **analysis-level reproducibility** (from the master table onward) but **not full raw-data-to-final-table reproducibility**. The full raw-data workflow is **not** one-click reproducible: it depends on large third-party downloads, on local folder layouts created by earlier steps, and on a few intermediate products whose generating script could not be located (see [Known gaps](#known-gaps)).

## Software environment

* Python 3.13.4 (CPython, Windows 11) was used for all scripts. No R code or notebooks were found in the project.
* Package versions recorded on the authors' machine: numpy 2.2–2.3, pandas 2.3, scipy 1.16, matplotlib 3.10, statsmodels 0.14, scikit-learn 1.7, cartopy 0.25, rasterio 1.4, shapely 2.1, pyproj 3.7, geopandas 1.1. See [`requirements.txt`](../requirements.txt) and [`environment.yml`](../environment.yml). These are a record, not a validated lock file.
* `xgboost` and `catboost` are used by the model-comparison / reconstruction scripts (`catboost` is imported in a try/except in several of them).

## Path configuration

The original scripts contained hard-coded absolute Windows paths. In this repository they were replaced mechanically by environment variables (no computation was changed):

| Variable | Meaning | Default |
|---|---|---|
| `SIF_GPP_ROOT` | Working root under which intermediate/output folders are read and written | current directory |
| `SIF_GPP_DATA_ROOT` | Parent folder of raw third-party downloads | `SIF_GPP_ROOT` |
| `NE_BASEMAP_TIF` | Optional Natural Earth raster for map backgrounds | `NE1_HR_LC_SR_W.tif` |
| `GOME2_SAMPLE_NC` | Sample GOME-2 file used by one figure script | file name only |

See [`config/paths.example.env`](../config/paths.example.env). Scripts still expect the **folder layout of the original project** below `SIF_GPP_ROOT` (e.g. `SIF_EC_SIF_GPP_Relationship_Analysis/01_analysis_ready/`); folder names that contained personal or tool names were sanitized, so a few folder names differ from the original workspace. Scripts also import each other by module name; each script adds the sibling folders under `scripts/` to `sys.path` at start-up.

> Run from a local (non-cloud-synced) drive. Several scripts write very long paths and note this limitation themselves.

## Order of scripts

Within a folder, run in numeric/alphabetical order. Across folders:

1. `01_ec_preprocessing/` — wind-direction histograms → conservative GPP extraction → merged monthly input table → metadata recovery / repair / completeness audit.
2. `02_footprint_sif_extraction/` — monthly footprint rasters → footprint-aware SIF companion rasters → coverage inventories → per-product finalization (GOSIF, GOME-2, TROPOSIF) → master harmonization (the script's archive audit covered an additional product; the final analysis uses only GOSIF, GOME-2, TROPOSIF and OCO-3).
3. `03_oco3_reconstruction/` — proof of concept → pre-production validation → positive-coverage final test → final production. (`troposif_tested_not_retained/` documents the tested, rejected TROPOSIF reconstruction.)
4. *(Not in repository — see gaps)* construction of the analysis-ready site-month master table (`SIF_GPP_Analysis_Ready_Master.csv`).
5. `04_coupling/01_harmonic_deseasonalization_and_core_statistics.py` → `04_coupling/mechanistic_drivers/` → `04_coupling/advanced_coupling/` → `04_coupling/environmental_modulation/`.
6. `05_stress_induced_decoupling/` (01 → 05).
7. `06_phenology_peak_timing/climatological_pos/` and `06_phenology_peak_timing/year_specific_mechanism/` (`step1` → `step12`).
8. `08_lst_extension/` (01 → 04), needed only for the LST panels.
9. `07_figures/`.
10. `archive/` holds internal QA scripts that are not part of the manuscript workflow (not required).

## Which analyses need which data

| Analysis | Needs |
|---|---|
| EC preprocessing | FLUXNET-format half-hourly files (GPP, VPD, TA, SW_IN_F, SWC_F_MDS_1, WD, P_F) and site metadata |
| Footprint matching / SIF extraction | Monthly wind-direction tables, SIF products (GOSIF, GOME-2 L2, TROPOSIF, OCO-3 Lite) |
| OCO-3 reconstruction | OCO-3 Lite files, footprint coverage tables, GOSIF context (no tower GPP) |
| Coupling / anomalies / SID / POS | Analysis-ready master table + monthly driver table |
| Aridity | Global Aridity Index v3.1 rasters at tower locations |
| Precipitation context | Tower `P_F`; NASA POWER for the remaining towers |
| LST panels | MODIS MYD21C1 monthly GeoTIFFs at towers |

## Manuscript output → script

Confidence labels: **Confirmed** = the script writes a file whose name matches the final figure file in the author's figure folder; **Probable** = best match by content/name, but several revisions of the script exist, so please verify; **Manual** = assembled outside code.

| Manuscript output | Script | Main inputs | Confidence |
|---|---|---|---|
| Fig. 1 Tower network map | `scripts/07_figures/fig01_tower_map.py` | Tower metadata table | Confirmed |
| Fig. 2 Aridity context map | `scripts/07_figures/fig02_aridity_context_map.py` | Aridity Index raster, tower table | Confirmed |
| Fig. 3 Workflow | — | — | Manual (slide drawing) |
| Fig. 4 Footprint-matching example (DE-Hai) | `scripts/07_figures/fig04_footprint_method_example.py` | Monthly footprint rasters, wind histograms, SIF samples | Confirmed |
| Fig. 5 Raw (and deseasonalized) coupling | `scripts/07_figures/fig05_raw_vs_deseasonalized_coupling.py` | Analysis-ready master, harmonic residuals | Confirmed |
| Fig. 6 Site-level coupling maps | `scripts/07_figures/fig06_site_level_coupling_maps.py` | Analysis-ready master | Confirmed |
| Fig. 7 Biome heatmap (R²) | `scripts/07_figures/fig07_igbp_raw_coupling_heatmap.py` | Analysis-ready master | Confirmed |
| Fig. 8 Seasonal coupling, biome groups | `scripts/07_figures/fig08_seasonal_coupling_biome_groups.py` | Analysis-ready master | Probable |
| Figs. 9–10 NH / SH seasonal structure | `scripts/07_figures/build_seasonal_structure_v2.py`, `fig09_10_seasonal_structure_NH_SH_final.py` | Analysis-ready master | Confirmed (final panel fix) |
| Fig. 11 ΔPOS map + biome counts | `scripts/07_figures/fig11_global_deltapos_map.py` (uses outputs of `06_phenology_peak_timing/climatological_pos/`) | Climatological POS table | Confirmed |
| Fig. 12 Environmental context maps (incl. LST) and Figs. S1–S4 per product | `scripts/07_figures/build_environmental_controls_geospatial.py` + `fig12_figS1_S4_add_LST_panel.py` | Site-level coupling, tower drivers, LST table (`08_lst_extension`) | Confirmed |
| Fig. 13 Aridity vs SID (site level) | `scripts/07_figures/fig13_aridity_vs_sid_site_level.py` (+ `..._panelB_zscore.py` for the Panel B unit fix) | SID table, aridity | Probable (the two-panel diagnostic figure is the one with this file name; confirm which panel is Fig. 13) |
| Fig. 14 Pooled pathway analysis | `scripts/07_figures/fig14_pooled_pathway_analysis.py` (uses `04_coupling/advanced_coupling/03a_*`) | Monthly driver table | Confirmed |
| Fig. 15 Conceptual diagram | — | — | Manual |
| Fig. S5 SID seasonal, biome × hemisphere | `scripts/07_figures/figS5_sid_seasonal_biome_hemisphere.py` (+ `06_sid_biome_hemisphere_seasonal.py`) | SID monthly master | Probable |
| Figs. S6–S7 Seasonal coupling, biome groups / IGBP classes | `scripts/07_figures/figS6_S7_seasonal_coupling_biome_hemisphere.py` | Analysis-ready master | Probable |
| Fig. S8 Aridity–SID biome forest plot | `scripts/07_figures/figS8_aridity_sid_biome_forest.py` | SID table, aridity | Confirmed |
| Table 1 | — | — | Manual |

Analysis results (non-figure):

| Result | Script |
|---|---|
| Harmonic deseasonalization, raw statistics, minimum-support and OCO-3 observed vs reconstructed checks | `04_coupling/01_harmonic_deseasonalization_and_core_statistics.py` |
| Partial Spearman with tower-block bootstrap (1,000 resamples); per-tower-standardized interaction models with site effects and tower-clustered SE | `04_coupling/environmental_modulation/02_partial_corr_and_interaction_unique_effect.py` (primary), `04_coupling/advanced_coupling/03_partial_correlation.py`, `04_coupling/environmental_modulation/audit_environmental_interactions_and_biome.py` (independent check) |
| SID construction and validation | `05_stress_induced_decoupling/02_*`, `03_*`, `04_*` |
| Climatological POS (year-block bootstrap, 500) and year-specific POS (delete-one-month jackknife) | `06_phenology_peak_timing/climatological_pos/pos_lib.py`, `year_specific_mechanism/lib_year_specific_pos.py` |
| Radiation-offset test: median abs(POS_SIF − POS_SW_IN) = 0.79 months, median abs(POS_GPP − POS_SW_IN) = 1.34 months, n = 1,629 site-years, paired Wilcoxon p = 7.5e-71 | `06_phenology_peak_timing/year_specific_mechanism/step6_hypothesis_A_D_light_lue.py` — **Confirmed**: it computes the circular absolute offsets to the `SW_IN` peak timing, the medians and `scipy.stats.wilcoxon`, and writes `hypothesisA_light_timing_test.csv`, which contains exactly 1629, 0.79, 1.34 and 7.547820485498006e-71. Input: `SiteYear_Product_PeakTiming_Master.csv` from `step1_build_peak_timing_master.py`. |
| OCO-3 reconstruction (544 observed + 115 reconstructed site-months) | `03_oco3_reconstruction/04_Run_OCO3_Final_Production_Reconstruction.py` |

Original-to-public script names are listed in [`script_name_mapping.csv`](script_name_mapping.csv).

## Known gaps

* **Analysis-ready master table builder.** `SIF_GPP_Analysis_Ready_Master.csv` (all products, including OCO-2 rows that later scripts drop) is the input to nearly every analysis, but the script that assembled it was **not found** in the project. Its contents are described by the `SIF_EC_SIF_GPP_Relationship_Analysis` reports in the original workspace. The closest repository code is `02_footprint_sif_extraction/05_Five_Product_SIF_Master_Harmonization_Audit.py` and the per-product finalization scripts.
* **Source of per-site metadata** (measurement/canopy heights, source-area radii) is handled by the metadata-recovery scripts, but the original manual metadata spreadsheets are not part of this repository.
* **"GEE" in script names.** Several preprocessing scripts (and some output file names) refer to "GEE"-ready inputs, a legacy of an earlier Google Earth Engine plan. No Earth Engine code was found in the project and the manuscript does not describe an Earth Engine step; footprint rasterization here is done locally in Python (`02_footprint_sif_extraction/01_*`).
* **Product versions** are only partly recoverable; see [data_availability.md](data_availability.md#product-versions-and-provenance-verified-from-the-project-archive). GOSIF's version is not recoverable from the archive.
* **Internal QA scripts** (GOME-2 quality flag, satellite-day-restricted GPP, TROPOSIF QA, June leave-out) are in `scripts/archive/` and are not used in the manuscript. The manuscript's peak-fitting sensitivity (sign of ΔPOS under harmonic vs periodic-spline fits, 22.4% of site-years) comes from `06_phenology_peak_timing/year_specific_mechanism/step9_methodological_E_F_G.py`.
* Scripts were run interactively over time; several outputs depend on the state of folders created by earlier steps and by superseded runs. A clean-slate rerun has **not** been performed.

## Computational notes

* Footprint rasterization on a 30 m grid for each site-month and the SIF companion rasters create one raster per site-month and product and are expected to be the most storage- and time-intensive steps (run-time and disk use were not benchmarked for this documentation).
* Bootstrap analyses (up to 1,000 resamples) and the OCO-3 model comparison are the main compute-heavy statistical steps.
* Bootstrap seeds are defined inside the scripts; identical inputs are required for identical results.
