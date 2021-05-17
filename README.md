# Measurement Report:An assessment of the impact of a nationwide lockdown on air pollution – a remote sensing perspective over India

This repository contains code used in the study "An assessment of the impact of a nationwide lockdown on air pollution – a remote sensing perspective over India"

Brief descripton of the codes are given below.

  - AOD.py: fuctions to read AOD data from MOD08 and MYD08 datasets for given region and generate time averaged maps, difference maps and metrics
  - Analyse_AOD_data.py: This python script is a sample code on how to use functions defined in AOD.py to generate time average maps, difference maps, metrix using AOD data
  - NO2.py: fuctions to read NO2 data from OMI/TROPOMI datasets for given region and generate time averaged maps, difference maps and metrics
  - Analyse_NO2_data.py: This python script is a sample code on how to use functions defined in AOD.py to generate time average maps, difference maps, metrix using NO2 data
  -  CO.py: fuctions to read CO data from TROPOMI datasets for given region and generate time averaged maps, difference maps and metrics
  - Analyse_CO_data.py: This python script is a sample code on how to use functions defined in AOD.py to generate time average maps, difference maps, metrix using CO data

### Dataset details
| S.No | Parameter | Data source                                 | Resolution  | Website |
|------|-----------|---------------------------------------------|-------------|---------|
| 1    | AOD       | MOD08_D3 from Terra and MYD08_D3 from Aqua  | 1deg x 1deg | https://ladsweb.modaps.eosdis.nasa.gov/ |
| 2    | NO2       | Aura/OMI, Sentinel-5P/TROPOMI   | 0.25º×0.25º 3.5×7 km2 (year, 2019) & 3.5×5.5 km2 (year,2020) | https://earthdata.nasa.gov/ |
| 3    | CO        | Sentinel-5P/TROPOMI                         |  7×7 km2 (year, 2019) & 5.5×7km2(year, 2020)           | https://earthdata.nasa.gov/ |
