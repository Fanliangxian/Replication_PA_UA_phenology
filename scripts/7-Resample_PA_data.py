'''

Author: Fan Liangxian

Resample the PA data (climate data & phenology data for every PA):
The data volume of PA is much larger than that of UA, so we first resample the PA data to make the data volumes of the two data similar.

'''

import pandas as pd
 
pa_path = "PA_SOS_merged_with_vap_by_year_geo.csv"
pa_df = pd.read_csv(pa_path)
 
target_sample_size = 243049
 
pa_sampled = pa_df.sample(n=target_sample_size, random_state=42)
 
output_path = "PA_SOS_TerraClimate_2001_2022_resample.csv"
pa_sampled.to_csv(output_path, index=False)
