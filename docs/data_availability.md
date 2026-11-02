# 📦 Data availability

> All input datasets used in this study are publicly available from official data repositories, subject to the access, registration, citation, and licensing conditions of the respective data providers. Raw third-party datasets are therefore not redistributed in this repository.

This repository contains **code and documentation only**. To rerun the workflow you must obtain each dataset directly from its provider, accept the provider's terms, and cite it as the provider requires. Nothing here relicenses any third-party data (see [LICENSE](../LICENSE)).

## Datasets used

| Dataset | Variable / role | Provider | Official access | Notes |
|---|---|---|---|---|
| **FLUXNET** (FLUXNET2015 / ONEFlux-processed records) | Half-hourly `GPP_DT_VUT_REF`, `VPD`, `TA`, `SW_IN_F`, `SWC_F_MDS_1`, `WD`, `P_F`; site metadata incl. IGBP class | FLUXNET community / data hubs | Data policy: <https://fluxnet.org/data/data-policy/> · Dataset page: <https://fluxnet.org/data/> · Reference: Pastorello et al. (2020), <https://doi.org/10.1038/s41597-020-0534-3> | Acceptance of the data policy is required. The CC-BY-4.0 tier requires citing Pastorello et al. (2020) and listing each site by FLUXNET ID / per-site DOI. The FLUXNET2015 page notes newer releases are distributed through the FLUXNET Shuttle; the exact release used per site is not recorded in this repository. |
| **AmeriFlux** | Additional site records (where used) | AmeriFlux Management Project, Lawrence Berkeley National Laboratory | <https://ameriflux.lbl.gov/> | Registration may be required. Each site has its own DOI and citation; follow the AmeriFlux data-use policy. Which towers came from which network is not recorded in the scripts. |
| **ICOS** | Additional site records (where used) | ICOS Carbon Portal | <https://data.icos-cp.eu/> | Licence and citation are given per data object on the portal. |
| **OzFlux** | Additional site records (where used) | OzFlux / TERN | <https://ozflux.org.au/> | Follow the OzFlux/TERN data-use and citation terms for each site. |
| **GOSIF** | Satellite SIF, 0.05°, 8-day (monthly composites also exist); local archive covered Jan 2001–Dec 2024 | Global Ecology Group, University of New Hampshire | <https://globalecology.unh.edu/data/GOSIF.html> · Reference: Li & Xiao (2019), <https://doi.org/10.3390/rs11050517> | Read the provider's Fair Data Use Policy before use. Several GOSIF versions (v2, v3) exist; **the version used is not recorded in the project files** (`SOURCE_VERSION` = `NOT_ESTABLISHED`), so confirm it against the manuscript. |
| **GOME-2 SIF** (MetOp-A, Level 2 daily, **Version 2**) | `Daily_Averaged_SIF`; Feb 2007–Jan 2018 used | NASA ORNL DAAC (SIF-ESDR L2) | <https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=2292> · DOI <https://doi.org/10.3334/ORNLDAAC/2292> | NASA Earthdata login required to download. Source filenames in the project follow `NSIFv2.6.2.GOME-2A.*`. |
| **TROPOSIF** (Sentinel-5P TROPOMI) | `SIF_Corr_743`; May 2018–Dec 2021 used | ESA TROPOSIF project / S5P-PAL | Project portal: <https://s5p-troposif.noveltis.fr/> · S5P-PAL portal: <https://data-portal.s5p-pal.com/products/troposif.html> · Reference: Guanter et al. (2021), <https://doi.org/10.5194/essd-13-5423-2021> | The provider documents release v2.1 (May 2018–Apr 2021; dataset DOI `10.5270/esa-s5p_innovation-sif-20180501_20210320-v2.1-202104`). Because the analysis used data through Dec 2021, **confirm the exact release you use**; it is not recorded in the project files. |
| **OCO-3 SIF Lite** (Level 2) | `Daily_SIF_740nm`; Aug 2019–Dec 2025 used | NASA GES DISC (OCO-3 L2 Lite SIF) | <https://disc.gsfc.nasa.gov/> (search "OCO3_L2_Lite_SIF") · Reference: Doughty et al. (2022), <https://doi.org/10.5194/essd-14-1513-2022> | NASA Earthdata login required. The processing version used is **not recorded in the project files**; check the collection landing page for the DOI of the version you download. |
| **Global Aridity Index and PET Database, v3.1** | Annual and monthly Aridity Index (AI = P/PET), 30 arc-sec, 1970–2000 climatology; AI scale factor 10,000 | Zomer, Xu & Trabucco; hosted on figshare | <https://doi.org/10.6084/m9.figshare.7504448> (record "Global Aridity Index and PET Database: Version 3.1") · Reference: Zomer et al. (2022), <https://doi.org/10.1038/s41597-022-01493-1> | CC BY 4.0 per community catalogue listings; confirm on the figshare record. v3.1 supersedes v3.0. |
| **NASA POWER** | Climatological precipitation for the 52 towers without usable tower `P_F` | NASA Langley Research Center POWER Project | <https://power.larc.nasa.gov/> · Reference: Zhang et al. (2008), <https://doi.org/10.1007/978-3-540-75997-3_532> | Public API; follow NASA POWER citation guidance. Precipitation is used only in the environmental-context analysis. |
| **MODIS/Aqua MYD21C1 v061** (LST) | Monthly land surface temperature (0.05°) for the LST panels (Fig. 12, Figs. S1–S4) | NASA LP DAAC | <https://doi.org/10.5067/MODIS/MYD21C1.061> | LP DAAC states no restrictions on use or redistribution. **The manuscript Methods do not currently describe the LST source**; add it before submission. Day/night composition and QA filtering of the LST files were not verified in the project (see `scripts/08_lst_extension`). |
| **Natural Earth** (NE1 with shaded relief) | Optional map background for figure scripts | Natural Earth | <https://www.naturalearthdata.com/> | Public domain. Cosmetic only; set `NE_BASEMAP_TIF`. |
| **Tower meteorology** | VPD, air temperature, `SW_IN_F`, `SWC_F_MDS_1`, wind direction | From the EC records above | — | Not a separate dataset. `SW_IN_F` is incoming shortwave radiation and is used as a **radiation proxy**, not PAR/APAR. |

## What the repository does *not* contain

* No raw or processed FLUXNET / AmeriFlux / ICOS / OzFlux tower files.
* No SIF NetCDF/HDF/GeoTIFF files, no footprint rasters, no LST rasters.
* No credentials or Earthdata tokens. Provide your own through each provider's standard mechanism and never commit them (`.gitignore` excludes common token and `.env` files).

## Derived data

Intermediate and final analysis tables (for example the analysis-ready site-month master table) are generated by the scripts and are not committed. Whether a derived table can be shared depends on the redistribution terms of the underlying tower and satellite data, so any release of derived tables should be checked against those terms first.
