# 📦 Data availability

> All input datasets used in this study are publicly available from official data repositories, subject to the access, registration, citation, and licensing conditions of the respective data providers. Raw third-party datasets are therefore not redistributed in this repository.

This repository contains **code and documentation only**. To rerun the workflow you must obtain each dataset directly from its provider, accept the provider's terms, and cite it as the provider requires. Nothing here relicenses any third-party data (see [LICENSE](../LICENSE)).

## Datasets used

| Dataset | Variable / role | Provider | Official access | Notes |
|---|---|---|---|---|
| **FLUXNET** (FLUXNET2015 and the newer FLUXNET data-portal release) | Half-hourly/hourly `GPP_DT_VUT_REF`, `VPD`, `TA`, `SW_IN_F`, `SWC_F_MDS_1`, `WD`, `P_F`; site metadata incl. IGBP class | FLUXNET community / data hubs | Data policy: <https://fluxnet.org/data/data-policy/> · Data portal: <https://data.fluxnet.org/> · FLUXNET2015 dataset: <https://fluxnet.org/data/> · Reference: Pastorello et al. (2020), <https://doi.org/10.1038/s41597-020-0534-3> | Acceptance of the data policy is required. The CC-BY-4.0 tier requires citing Pastorello et al. (2020) and listing each site by FLUXNET ID / per-site DOI. The project archive contains both FLUXNET2015-named files and files named `*_FLUXNET_FLUXMET_HH/HR_*_v1.3_r1.csv` (see the version section below). |
| **AmeriFlux** | Additional site records (where used) | AmeriFlux Management Project, Lawrence Berkeley National Laboratory | <https://ameriflux.lbl.gov/> | Registration may be required. Each site has its own DOI and citation; follow the AmeriFlux data-use policy. File-name prefixes in the archive include `AMF`, `ICOS`, `EUF`, `CNF`, `JPF` and `TERN`; the mapping of every tower to its network is not recorded in the scripts. |
| **ICOS** | Additional site records (where used) | ICOS Carbon Portal | <https://data.icos-cp.eu/> | Licence and citation are given per data object on the portal. |
| **OzFlux** | Additional site records (where used) | OzFlux / TERN | <https://ozflux.org.au/> | Follow the OzFlux/TERN data-use and citation terms for each site. |
| **GOSIF** | Satellite SIF, 0.05°, 8-day (monthly composites also exist); local archive covered Jan 2001–Dec 2024 | Global Ecology Group, University of New Hampshire | <https://globalecology.unh.edu/data/GOSIF.html> · Reference: Li & Xiao (2019), <https://doi.org/10.3390/rs11050517> | Read the provider's Fair Data Use Policy before use. Several GOSIF versions (v2, v3) exist; **the version used is not recoverable from the project archive** (local files are monthly 0.05° GeoTIFFs renamed `GOSIF_YYYYMM.tif`, with no version metadata). |
| **GOME-2 SIF** (MetOp-A, Level 2 daily, **Version 2**) | `Daily_Averaged_SIF`; Feb 2007–Jan 2018 used | NASA ORNL DAAC (SIF-ESDR L2) | <https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=2292> · DOI <https://doi.org/10.3334/ORNLDAAC/2292> | NASA Earthdata login required to download. Source filenames in the project follow `NSIFv2.6.2.GOME-2A.*`. |
| **TROPOSIF** (Sentinel-5P TROPOMI) | `SIF_Corr_743`; May 2018–Dec 2021 used | ESA TROPOSIF project / S5P-PAL | Project portal: <https://s5p-troposif.noveltis.fr/> · S5P-PAL portal: <https://data-portal.s5p-pal.com/products/troposif.html> · Reference: Guanter et al. (2021), <https://doi.org/10.5194/essd-13-5423-2021> | The project's download note points to the SRON open-access `v2.1/l2b` directory, and the local files are daily `TROPOSIF_L2B_YYYY-MM-DD.nc`; the file attributes themselves do not state a version. Release v2.1 is documented by the provider with DOI `10.5270/esa-s5p_innovation-sif-20180501_20210320-v2.1-202104`. |
| **OCO-3 SIF Lite** (Level 2) | `Daily_SIF_740nm`; Aug 2019–Dec 2025 used | NASA GES DISC (OCO-3 L2 Lite SIF) | <https://disc.gsfc.nasa.gov/> (search "OCO3_L2_Lite_SIF") · Reference: Doughty et al. (2022), <https://doi.org/10.5194/essd-14-1513-2022> | NASA Earthdata login required. Local daily files carry build tags `B11074Ar`, `B11076Ar` and `B11077Ar` (see the version section below). |
| **Global Aridity Index and PET Database, v3.1** | Annual and monthly Aridity Index (AI = P/PET), 30 arc-sec, 1970–2000 climatology; AI scale factor 10,000 | Zomer, Xu & Trabucco; hosted on figshare | <https://doi.org/10.6084/m9.figshare.7504448> (record "Global Aridity Index and PET Database: Version 3.1") · Reference: Zomer et al. (2022), <https://doi.org/10.1038/s41597-022-01493-1> | CC BY 4.0 per community catalogue listings; confirm on the figshare record. v3.1 supersedes v3.0. |
| **NASA POWER** | Climatological precipitation for the 52 towers without usable tower `P_F` | NASA Langley Research Center POWER Project | <https://power.larc.nasa.gov/> · Reference: Zhang et al. (2008), <https://doi.org/10.1007/978-3-540-75997-3_532> | Public API; follow NASA POWER citation guidance. Precipitation is used only in the environmental-context analysis. |
| **MODIS/Aqua MYD21C1** (LST) | Monthly land surface temperature (°C, 0.05°, about 5.6 km) used for the LST panels (Fig. 12, Figs. S1–S4) | NASA LP DAAC | Product page: <https://doi.org/10.5067/MODIS/MYD21C1.061> | LP DAAC states no restrictions on use or redistribution. Product identity is verified from local file names and band descriptions; collection version, day/night selection, QA treatment and monthly aggregation are **unverified** (see the LST section below). The manuscript Methods do not currently describe the LST source. |
| **Natural Earth** (NE1 with shaded relief) | Optional map background for figure scripts | Natural Earth | <https://www.naturalearthdata.com/> | Public domain. Cosmetic only; set `NE_BASEMAP_TIF`. |
| **Tower meteorology** | VPD, air temperature, `SW_IN_F`, `SWC_F_MDS_1`, wind direction | From the EC records above | — | Not a separate dataset. `SW_IN_F` is incoming shortwave radiation and is used as a **radiation proxy**, not PAR/APAR. |

## Product versions and provenance (verified from the project archive)

Versions are stated only where they can be recovered from file names, file attributes or project notes. Otherwise the item is marked not recoverable.

| Item | What the archive shows | Status |
|---|---|---|
| **GOSIF** | Monthly GeoTIFFs `GOSIF_YYYYMM.tif` (0.05°, int16, Jan 2001 onward; 288 files); no version string in file names or tags. | **Version not recoverable from the project archive.** |
| **GOME-2 SIF** | Daily files `NSIFv2.6.2.GOME-2A.YYYYMMDD_all.nc`; the master table records "NSIFv2.6.2.GOME-2A (from source filenames)". The ORNL DAAC record is "L2 Daily SIF from MetOp-A GOME-2, 2007–2018, V2" (DOI 10.3334/ORNLDAAC/2292). | Recoverable (file-name version `NSIFv2.6.2`; ORNL DAAC V2 per manuscript). |
| **TROPOSIF** | Daily files `TROPOSIF_L2B_YYYY-MM-DD.nc` (May 2018–Dec 2021); the project's link note refers to the SRON `.../sif/v2.1/l2b/` directory; file attributes (`title`, `date_created` 2021-09-14 for the sampled file) do not state a version. | **v2.1 per project download note** (not stated inside the files). |
| **OCO-3 SIF Lite** | Daily files `oco3_LtSIF_YYMMDD_B11xxxAr_*.nc4`: build `B11074Ar` (1,578 files), `B11077Ar` (420), `B11076Ar` (2). | Processing build tags recorded; the product-version label (**V11r**) is **inferred from the `B11…r` build tag** and not stated in the files. |
| **FLUXNET tower files** | 21 files named `FLX_*_FLUXNET2015_FLUXMET_HH_*`; 61 files named `*_FLUXNET_FLUXMET_HH/HR_*_v1.3_r1.csv` with network prefixes `AMF`, `ICOS`, `EUF`, `CNF`, `JPF`, `TERN`; a few other files without a release tag. | Release tags `FLUXNET2015` and `v1.3_r1` recorded in file names. |

### LST provenance (Fig. 12 and Figs. S1–S4)

* **Verified:** the only LST data in the project are 547 monthly GeoTIFFs in two folders, `MYD21C1_Global_Grasslands_LST_{YYYY}_{MM}.tif` (270 files) and `MYD21C1_Global_Savannas_LST_{YYYY}_{MM}.tif` (277 files, including 7 duplicate re-downloads), July 2002 to December 2024. Each file has one band with description `LST_Celsius_Grasslands` or `LST_Celsius_Savannas` (°C), 0.0503° (about 5.6 km), EPSG:4326. The product is MODIS/Aqua MYD21C1.
* **Coverage consequence:** because the files are the grassland and savanna extracts, LST is available only at towers whose pixels fall in those masks (36 of 73 towers in the project's LST extension). The LST panels in Fig. 12 therefore do not cover all towers.
* **Not verified (do not assume):** the product collection/version (v061 is the version linked above, but file names do not state it); whether the day, night or a combined LST was exported; any QA or cloud screening; and how daily values were aggregated to monthly means. These steps happened outside the project (probably in an export that was not preserved), and the local code only reads the finished GeoTIFFs. State these limitations wherever LST results are reported.

## What the repository does *not* contain

* No raw or processed FLUXNET / AmeriFlux / ICOS / OzFlux tower files.
* No SIF NetCDF/HDF/GeoTIFF files, no footprint rasters, no LST rasters.
* No credentials or Earthdata tokens. Provide your own through each provider's standard mechanism and never commit them (`.gitignore` excludes common token and `.env` files).

## Derived data

Intermediate and final analysis tables (for example the analysis-ready site-month master table) are generated by the scripts and are not committed. Whether a derived table can be shared depends on the redistribution terms of the underlying tower and satellite data, so any release of derived tables should be checked against those terms first.
