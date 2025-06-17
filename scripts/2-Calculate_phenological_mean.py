'''

Author: Fan Liangxian

Calculate the phenological mean of all pixels of PA and UA

'''

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask

years = range(2001, 2023)

sos_folder = "SOS_b2_north"
eos_folder = "EOS_b2_north"
sos_prefix = "sos"
eos_prefix = "eos"
suffix = "_b2.tif"

pa_shp = "/Volumes/FLX的小可爱/pheno_climate/shapefiles/PA.shp"
ua_shp = "/Volumes/FLX的小可爱/pheno_climate/shapefiles/UA.shp"

with rasterio.open(os.path.join(sos_folder, f"sos2001{suffix}")) as src:
    raster_crs = src.crs

gdf_pa = gpd.read_file(pa_shp).to_crs(raster_crs)
gdf_ua = gpd.read_file(ua_shp).to_crs(raster_crs)

shapes_pa = [f["geometry"] for f in gdf_pa.__geo_interface__["features"]]
shapes_ua = [f["geometry"] for f in gdf_ua.__geo_interface__["features"]]

def extract_mean_by_mask(folder, prefix, shapes, region_label):
    values = []
    for year in years:
        path = os.path.join(folder, f"{prefix}{year}{suffix}")
        with rasterio.open(path) as src:
            out_image, _ = mask(src, shapes, crop=False, nodata=np.nan)
            mean_val = np.nanmean(out_image[0])
        values.append({"Year": year, "Region": region_label, "Mean": mean_val})
    return pd.DataFrame(values)

df_sos_pa = extract_mean_by_mask(sos_folder, sos_prefix, shapes_pa, "PA")
df_sos_ua = extract_mean_by_mask(sos_folder, sos_prefix, shapes_ua, "UA")
df_sos = pd.concat([df_sos_pa, df_sos_ua], ignore_index=True)
df_sos["Phenology"] = "SOS"

df_eos_pa = extract_mean_by_mask(eos_folder, eos_prefix, shapes_pa, "PA")
df_eos_ua = extract_mean_by_mask(eos_folder, eos_prefix, shapes_ua, "UA")
df_eos = pd.concat([df_eos_pa, df_eos_ua], ignore_index=True)
df_eos["Phenology"] = "EOS"

# === 合并并保存 ===
df_final = pd.concat([df_sos, df_eos], ignore_index=True)
df_pivot = df_final.pivot_table(index=["Year", "Region"], columns="Phenology", values="Mean").reset_index()
df_pivot.to_csv("PA_UA_SOS_EOS_Mean_By_Year.csv", index=False)

print("PA and UA annual mean SOS/EOS")
