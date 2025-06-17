'''

Author: Fan Liangxian

output : The mean and slope of all pixels showing an advance trend and a delay trend in protected areas and urban areas
# The mean and slope of all pixels showing a significant advance trend  and a significant delay trend in protected areas and urban areas
# All advance trends (slope < 0) Significant advance trends (slope < 0 and p < 0.05)
# All delay trends (slope > 0) Significant delay trends (slope > 0 and p < 0.05)


'''

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import box
from scipy.stats import linregress

sos_folder = "SOS_b2_north"
eos_folder = "EOS_b2_north"
sos_slope_path = "SOS_slope.tif"
eos_slope_path = "EOS_slope.tif"
sos_pval_path = "SOS_pvalue.tif"
eos_pval_path = "EOS_pvalue.tif"
pa_path = "PA.shp"
ua_path = "UA.shp"

prefix_sos = "sos"
prefix_eos = "eos"
suffix = "_b2.tif"
years = list(range(2001, 2023))

gdf_pa = gpd.read_file(pa_path)
gdf_ua = gpd.read_file(ua_path)

with rasterio.open(os.path.join(sos_folder, f"{prefix_sos}2001{suffix}")) as src:
    raster_crs = src.crs

gdf_pa = gdf_pa.to_crs(raster_crs)
gdf_ua = gdf_ua.to_crs(raster_crs)

region_defs = [("All_North", box(-180, 23.44, 180, 90))]

# === 读取栅格数据 ===
def read_raster(path):
    with rasterio.open(path) as src:
        return src.read(1), src.transform, src.crs

slope_sos, _, _ = read_raster(sos_slope_path)
slope_eos, _, _ = read_raster(eos_slope_path)
pval_sos, _, _ = read_raster(sos_pval_path)
pval_eos, _, _ = read_raster(eos_pval_path)

def extract_stack(folder, prefix, shapes, years):
    stack = []
    for year in years:
        tif_path = os.path.join(folder, f"{prefix}{year}{suffix}")
        with rasterio.open(tif_path) as src:
            out_image, _ = mask(src, shapes, crop=False, nodata=np.nan)
            stack.append(out_image[0])
    return np.stack(stack, axis=0)

def compute_trend(y, x):
    if np.sum(~np.isnan(y)) >= 5:
        slope, _, _, _, _ = linregress(x, y)
        return slope
    return np.nan

records = []
for pa_label, gdf in zip(["UA", "PA"], [gdf_ua, gdf_pa]):
    for region_name, region_geom in region_defs:
        gdf_region = gpd.GeoDataFrame(geometry=[region_geom], crs=raster_crs)
        clipped = gpd.overlay(gdf, gdf_region, how="intersection")
        if clipped.empty:
            continue

        shapes = [f["geometry"] for f in clipped.__geo_interface__["features"]]
        sos_stack = extract_stack(sos_folder, prefix_sos, shapes, years)
        eos_stack = extract_stack(eos_folder, prefix_eos, shapes, years)
        region_mask = ~np.isnan(sos_stack[0])
        x = np.array(years)

        trend_masks = {
            "SOS_Early": (slope_sos < 0),
            "SOS_Late": (slope_sos > 0),
            "SOS_Early_Sig": (slope_sos < 0) & (pval_sos < 0.05),
            "SOS_Late_Sig": (slope_sos > 0) & (pval_sos < 0.05),
            "EOS_Early": (slope_eos < 0),
            "EOS_Late": (slope_eos > 0),
            "EOS_Early_Sig": (slope_eos < 0) & (pval_eos < 0.05),
            "EOS_Late_Sig": (slope_eos > 0) & (pval_eos < 0.05),
        }

        total_pixels = np.sum(region_mask)

        for trend_label, trend_mask in trend_masks.items():
            mask_final = region_mask & trend_mask
            if np.sum(mask_final) == 0:
                continue

            sos_series = [np.nanmean(sos_stack[i][mask_final]) for i in range(len(years))]
            eos_series = [np.nanmean(eos_stack[i][mask_final]) for i in range(len(years))]
            sos_slope_val = compute_trend(sos_series, x)
            eos_slope_val = compute_trend(eos_series, x)
            pixel_pct = np.sum(mask_final) / total_pixels * 100

            for i, y in enumerate(years):
                records.append({
                    "PAUA": pa_label,
                    "Region": region_name,
                    "Trend": trend_label,
                    "Year": y,
                    "SOS": sos_series[i],
                    "EOS": eos_series[i],
                    "slope_SOS": sos_slope_val,
                    "slope_EOS": eos_slope_val,
                    "pixel_pct": pixel_pct
                })

df_final = pd.DataFrame(records)
import ace_tools as tools; tools.display_dataframe_to_user(name="Phenology Trend Statistics", dataframe=df_final)
