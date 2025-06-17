'''

Author: Fan Liangxian

Statistics: phenology slope, P value and mean of all pixels

'''

#output files: EOS_slope.tif; EOS_pvalue.tif; EOS_mean.tif; SOS_slope.tif; SOS_pvalue.tif; SOS_mean.tif
import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.transform import from_origin
from scipy.stats import linregress
import geopandas as gpd

years = list(range(2001, 2023))

sos_folder = "SOS_b2_north"
sos_prefix = "sos"
sos_suffix = "_b2.tif"
sos_output_csv = "SOS_stats_by_region.csv"
sos_trend_folder = "SOS_trend"
os.makedirs(sos_trend_folder, exist_ok=True)

eos_folder = "EOS_b2_north"
eos_prefix = "eos"
eos_suffix = "_b2.tif"
eos_trend_folder = "EOS_trend"
os.makedirs(eos_trend_folder, exist_ok=True)

pa_shapefile = "PA.shp"
ua_shapefile = "UA.shp"

def write_tif(output_name, array, profile, folder):
    out_path = os.path.join(folder, output_name)
    profile.update(dtype=rasterio.float32, count=1, compress='lzw')
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(array, 1)
    print(f”Done：{output_name}")

# ==========Function: Pixel-by-pixel trend analysis ==========
def compute_pixelwise_trend(input_folder, prefix, suffix, output_folder):
    stack = []
    for year in years:
        fname = f"{prefix}{year}{suffix}"
        fpath = os.path.join(input_folder, fname)
        with rasterio.open(fpath) as src:
            arr = src.read(1).astype(np.float32)
            profile = src.profile
            stack.append(arr)

    stack = np.stack(stack, axis=0)
    print(f" {prefix.upper()} Done：”, stack.shape)

    rows, cols = stack.shape[1:]
    slope = np.full((rows, cols), np.nan, dtype=np.float32)
    pval = np.full_like(slope, np.nan)
    mean = np.full_like(slope, np.nan)

    x = np.array(years)
    for i in range(rows):
        for j in range(cols):
            y = stack[:, i, j]
            if np.count_nonzero(~np.isnan(y)) < 5:
                continue
            reg = linregress(x, y)
            slope[i, j] = reg.slope
            pval[i, j] = reg.pvalue
            mean[i, j] = np.nanmean(y)

    write_tif(f"{prefix.upper()}_slope.tif", slope, profile, output_folder)
    write_tif(f"{prefix.upper()}_pvalue.tif", pval, profile, output_folder)
    write_tif(f"{prefix.upper()}_mean.tif", mean, profile, output_folder)


def analyze_region(masked_stack, label, years):
    means = np.nanmean(masked_stack, axis=(1, 2))
    slope, intercept, r, p, stderr = linregress(years, means)
    overall_mean = np.nanmean(means)
    df = pd.DataFrame({
        "RegionType": [label] * len(years),
        "Year": years,
        "SOS_Mean": means,
        "Slope": [slope] * len(years),
        "PValue": [p] * len(years),
        "MultiYearMean": [overall_mean] * len(years)
    })
    return df

def mask_stack(folder, prefix, suffix, shapes):
    masked = []
    for i, year in enumerate(years):
        path = os.path.join(folder, f"{prefix}{year}{suffix}")
        with rasterio.open(path) as src:
            out_image, _ = mask(src, shapes, crop=False, nodata=np.nan)
            masked.append(out_image[0])
    return np.stack(masked, axis=0)


# Step 1: calculate pixel trend（SOS + EOS）
compute_pixelwise_trend(sos_folder, sos_prefix, sos_suffix, sos_trend_folder)
compute_pixelwise_trend(eos_folder, eos_prefix, eos_suffix, eos_trend_folder)

# Step 2: regional Analysis (SOS only)
with rasterio.open(os.path.join(sos_folder, f"{sos_prefix}2001{sos_suffix}")) as src:
    meta = src.meta.copy()

gdf_pa = gpd.read_file(pa_shapefile).to_crs(meta['crs'])
gdf_ua = gpd.read_file(ua_shapefile).to_crs(meta['crs'])

shapes_pa = [feature["geometry"] for feature in gdf_pa.__geo_interface__["features"]]
shapes_ua = [feature["geometry"] for feature in gdf_ua.__geo_interface__["features"]]

masked_pa = mask_stack(sos_folder, sos_prefix, sos_suffix, shapes_pa)
masked_ua = mask_stack(sos_folder, sos_prefix, sos_suffix, shapes_ua)

df_pa = analyze_region(masked_pa, “PA”, years)RegionTyrRrrrrrYearYyyyyyyyyyYyyyyyyy
df_ua = analyze_region(masked_ua, “UA”, years)
df_all = pd.concat([df_pa, df_ua], ignore_index=True)
df_all.to_csv(sos_output_csv, index=False)

import ace_tools as tools; tools.display_dataframe_to_user(name="Combined SOS EOS Analysis", dataframe=df_all)
