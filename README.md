# Phenological Response in Protected and Urban Areas under Climate Change

This repository contains the code and data used in the analysis for the paper:

> **"Protected areas mitigate phenological shifts relative to urban areas under global climate change"**  

## 🗂 Repository Overview

This project investigates the differences in vegetation phenology (start and end of growing season, SOS and EOS) between protected areas (PAs) and urban areas (UAs) globally from 2001 to 2022.

### 📁 Directory Structure

- `data/`: Input data including SOS/EOS raster files
- `scripts/`: Python scripts for data processing and analysis
- `gee_scripts/`: Google Earth Engine scripts for climate variable extraction
- `outputs/`: Processed data and results (optional)
- `plots.py`: Custom script to generate figures used in the paper

---

## 📄 Data Availability

All datasets used in this study are publicly available:

- **MODIS NDVI (MOD13C1 V006)**  
  Provided by NASA LP DAAC  
- **TerraClimate climate data**  
  University of California, Merced  
  https://www.climatologylab.org/terraclimate.html  
- **Protected areas (WDPA)**  
  https://www.protectedplanet.net/en/thematic-areas/wdpa?tab=WDPA  
- **Urban area boundaries (UAs dataset)**  
  https://data-starcloud.pcl.ac.cn/iearthdata/

---

## ⚙️ Main Scripts

All scripts assume SOS/EOS raster files and shapefiles are stored in the `data/` directory.

### 1. `1-Phenological_slope_pvalue.py`
- Input: 2001–2022 SOS or EOS raster files + PA/UA shapefiles  
- Output: Per-pixel **phenology slope**, **p-value**, and **multi-year mean**

### 2. `2-Calculate_phenological_mean.py`
- Input: Same as above  
- Output: Annual **mean SOS/EOS** for PA and UA (2001–2022)

### 3. `3-Advance_delayed_trend.py`
- Input: SOS/EOS raster files + PA/UA shapefiles + slope/p-value TIFFs from script 1  
- Output: For PA and UA:
  - % of pixels showing **advanced/delayed** trends (with and without significance)
  - Annual means and linear trends for each group

### 4. `4-GEE_script_PA.txt`  
### 5. `5-GEE_script_UA.txt`
- GEE JavaScript files for extracting climate variable statistics from TerraClimate
- Output: For each PA/UA: mean or sum of meteorological variables in the 90 days before phenological events

### 6. `6-Rescale_climate_data.py`
- Rescales the climate data extracted via GEE back to correct units (TerraClimate data is scaled)

### 7. `7-Resample_PA_data.py`
- Downsamples the PA climate dataset to match the sample size of UA data for SHAP analysis

---

## 📊 Plotting Script

### `8-plots.py`
This script generates Figures 2–4 and Extended Data Fig. 1 of the paper.  
Note: Not directly executable – run specific sections for each figure.

- **Fig. 2a** (Line 12): Requires output from `2-Calculate_phenological_mean.py`
- **Fig. 2b** (Line 95): Requires output from `3-Advance_delayed_trend.py`
- **Fig. 3** (Line 195): SHAP analysis results using climate + phenology data
- **Fig. 4 / Extended Data Fig. 1** (Line 325): SHAP dependence plots

---

