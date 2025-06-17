'''

Author: Fan Liangxian

Rescale the climate data:
Since the Terra Climate data in GEE is scaled, we need to rescale the data before using it.

'''

import pandas as pd
 
file_path = "UA_EOS_TerraClimate_2001_2022.csv" #or PA_SOS
df = pd.read_csv(file_path)
 
# scale
scale_map = {
    'tmmx_C': 1,
    'tmmn_C': 1,
    'srad_mean': 0.1,
    'vpd_mean': 0.01,
    'aet_mean': 0.1,
    'pet_mean': 0.1,
    'def_mean': 0.1,
    'soil_mean': 0.1,
    'swe_mean': 1,
    'ro_mean': 1,       
    'pdsi_mean': 0.01,
    'vap_mean': 0.001,
    'pr_sum': 1     
}
 

for col, scale in scale_map.items():
    if col in df.columns:
        df[col] = df[col] * scale
 
print(df.head(100))
 
output_path = file_path.replace(".csv", "_rescaled.csv")
df.to_csv(output_path, index=False)
