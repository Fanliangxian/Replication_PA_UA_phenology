'''

Author: Fan Liangxian

Plots

'''

#%% 2001-2022 Line chart of phenological data
# mean phenology value of all pixels of PAs and UAs
# obtained from '2-Calculate_phenological_mean.py'.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
 
file_path = "" #The results are obtained from the script2 that calculates the mean SOS and EOS for PA and UA pixels.
df_sos = pd.read_excel(file_path, sheet_name="SOS")
df_eos = pd.read_excel(file_path, sheet_name="EOS")
 
# Remove non-numeric rows (such as "slope" and "average")
df_sos = df_sos[pd.to_numeric(df_sos['Year'], errors='coerce').notna()]
df_eos = df_eos[pd.to_numeric(df_eos['Year'], errors='coerce').notna()]
 
df_sos['Year'] = df_sos['Year'].astype(int)
df_eos['Year'] = df_eos['Year'].astype(int)
years = df_sos['Year']
 
fig, (ax1, ax2) = plt.subplots(
    2, 1, sharex=True, figsize=(8, 5), gridspec_kw={'height_ratios': [1, 1]}
)
 
ax1.set_ylim(260, 300)
ax2.set_ylim(90, 130)
ax1.set_yticks(np.arange(270, 301, 10))
ax2.set_yticks(np.arange(90, 121, 10))
 
trend_texts_colored = []
 
def plot_with_trend(ax, x, y, label, color):
    ax.plot(x, y, color=color, linewidth=2.5)
    ax.scatter(x, y, color=color, edgecolors='black', s=60, label=label, marker='o')
    slope, intercept, r_value, _, _ = linregress(x, y)
    x_array = np.array(x)
    y_pred = intercept + slope * x_array
    ax.plot(x_array, y_pred, linestyle='--', color=color, linewidth=1.5)
    intercept_2001 = intercept + slope * 2001
    formula = f'{label}:\ny = {slope:.2f}x + {intercept_2001:.1f}'
    trend_texts_colored.append((formula, color))
 
# SOS 
plot_with_trend(ax2, years, df_sos['PA_SOS_AllPixels'], 'PA-SOS', '#90E0EF')
plot_with_trend(ax2, years, df_sos['UA_SOS_AllPixels'], 'UA-SOS', 'deepskyblue')
 
# EOS 
plot_with_trend(ax1, years, df_eos['PA_EOS_AllPixels'], 'PA-EOS', '#FF8FAB')
plot_with_trend(ax1, years, df_eos['UA_EOS_AllPixels'], 'UA-EOS', 'crimson')
 
ax1.tick_params(axis='both', labelsize=14)
ax2.tick_params(axis='both', labelsize=14)
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.tick_params(labeltop=False)
ax2.xaxis.tick_bottom()
ax1.grid(False)
ax2.grid(False)
 
d = .01  
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kwargs)
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
kwargs.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
 
fig.text(0.5, -0.02, 'Year', ha='center', fontsize=17)
fig.text(-0.02, 0.5, 'DOY', va='center', rotation='vertical', fontsize=17)
 
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
 
plt.tight_layout()
plt.subplots_adjust(hspace=0.05, top=0.78)
plt.savefig("SOS_EOS_2001-2022trend.jpg", dpi=700, bbox_inches="tight")
plt.show()

#%% Histogram showing pixels with early/late trends
# The secondary axis represents the slope’s absolute value,
# while the main axis represents the proportion of pixels that are advanced or delayed.

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
 
file_path = “”#The results are obtained from the script3 that calculates the mean and slope of pixels with an early or delayed trend.
df = pd.read_excel(file_path, sheet_name="Sheet4")
 
label_order = ["PA-SOS", "PA-EOS", "UA-SOS", "UA-EOS"]
df_sorted = df.set_index("PAUA").loc[label_order].reset_index()
 
labels = df_sorted["PAUA"].tolist()
early_vals = df_sorted["Early_pixel_pct"].values #The proportion of pixels show an advance trend. The results are obtained from the script3 that calculates the mean and slope of pixels with an early or delayed trend.
late_vals = df_sorted[“Late_pixel_pct"].values #The proportion of pixels show a delayed trend.
early_sig_vals = df_sorted["Early_Sig_pixel_pct"].values #The proportion of pixels show a significant advance trend.
late_sig_vals = df_sorted[“Late_Sig_pixel_pct"].values  #The proportion of pixels show a significant delayed trend.
 
slope_early = np.abs(df_sorted["Early_Slope”].values) # Slope of pixels showing an advance trend. The results are obtained from the script3.
slope_late = np.abs(df_sorted[“Late_Slope"].values) # Slope of pixels showing a delayed trend
slope_early_sig = np.abs(df_sorted["Early_Sig_Slope"].values) # Slope of pixels showing a significant advance trend
slope_late_sig = np.abs(df_sorted[“Late_Sig_Slope"].values) # Slope of pixels showing a significant delayed trend
 

color_early = "#90e0ef"
color_early_sig = "#0077b6"
color_late = "#ffccd5"
color_late_sig = "#d00000"
 
fig, axes = plt.subplots(4, 1, figsize=(2.5, 7), dpi=700, sharex=True)
axes = axes.flatten()
 
bar_width = 0.35
bar_gap = 0.5
bar_pos = [0, bar_gap]
xlim_min = -0.3
xlim_max = bar_gap + 0.3
 

y_max = max(max(early_vals), max(late_vals)) * 1.3
slope_max = max(slope_early.max(), slope_late.max(), slope_early_sig.max(), slope_late_sig.max()) * 1.3
 
for i in range(4):
    ax = axes[i]
    label = labels[i]
 
    heights = [early_vals[i], late_vals[i]]
    sig_heights = [early_sig_vals[i], late_sig_vals[i]]
    base_colors = [color_early, color_late]
    sig_colors = [color_early_sig, color_late_sig]
 
    ax.bar(bar_pos, heights, width=bar_width, color=base_colors, edgecolor='black')
    for j in range(2):
        if sig_heights[j] > 0:
            ax.bar(bar_pos[j], sig_heights[j], width=bar_width, color=sig_colors[j])
 

    ax.text(bar_pos[0], heights[0] + 1, f"{heights[0]:.2f}", ha='center', fontsize=10, fontweight='bold')
    ax.text(bar_pos[0], sig_heights[0] + 1, f"{sig_heights[0]:.2f}", ha='center', fontsize=10, fontweight='bold')
    ax.text(bar_pos[1], heights[1] + 1, f"{heights[1]:.2f}", ha='center', fontsize=10, fontweight='bold')
    ax.text(bar_pos[1], sig_heights[1] + 1, f"{sig_heights[1]:.2f}", ha='center', fontsize=10, fontweight='bold')
 
    ax.set_ylim(0, y_max)
    ax.set_yticks(np.arange(0, y_max + 1, 20))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(xlim_min, xlim_max)
    ax.set_title(label, fontsize=12, pad=8)
    ax.tick_params(axis='both', direction='out')
    ax.grid(False)
 
 
   # The secondary axis represents the slope’s absolute value)
    ax2 = ax.twinx()
    ax2.set_ylim(0, 1.8)
    ax2.set_yticks(np.arange(0, 1.8 + 0.01, 0.4))
    ax2.tick_params(axis='y', labelsize=10, direction='out')
 
    # 画散点
    #ax2.plot(bar_pos[0], slope_early[i], 'ko', color=color_early, markersize=4, markeredgecolor='black', markeredgewidth=1)
    #ax2.plot(bar_pos[1], slope_late[i], 'ko', color=color_late, markersize=4, markeredgecolor='black', markeredgewidth=1)
    #ax2.plot(bar_pos[0], slope_early_sig[i], marker='X', color=color_early_sig, markersize=4)
    #ax2.plot(bar_pos[1], slope_late_sig[i], marker='X', color=color_late_sig, markersize=4)
    ax2.plot(bar_pos[0], slope_early[i], 'ko', markersize=4, markeredgecolor='black', markeredgewidth=1)
    ax2.plot(bar_pos[1], slope_late[i], 'ko', markersize=4, markeredgecolor='black', markeredgewidth=1)
    ax2.plot(bar_pos[0], slope_early_sig[i], color='black', marker='X', markersize=4)
    ax2.plot(bar_pos[1], slope_late_sig[i], color='black', marker='X', markersize=4)
 
    if i == 3:
        ax.set_xticks(bar_pos)
        ax.set_xticklabels(['Advance', 'Delay'], fontsize=12)
    else:
        ax.set_xticks(bar_pos)
        ax.set_xticklabels(['', ''])
 
    ax.set_ylabel("Proportion (%)", fontsize=12)
    ax2.set_ylabel("|Slope|", fontsize=12)
 
plt.tight_layout(h_pad=1.0)
plt.savefig("barplot.jpg", dpi=700)
plt.show()

#%% SHAPbar_beeswarm
# SHAP summary plots (bee swarm + bar) for the SOS and EOS in PAs and UAs.

import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib  
 
 
# Rename the fields
rename_dict = {
    'pet_mean': 'PET', 'aet_mean': 'AET', 'def_mean': 'DEF', 'pr_sum': 'P',
    'soil_mean': 'TSM', 'vpd_mean': 'VPD', 'vap_mean': 'VAP', 'tmmx_C': 'Tmax',
    'tmmn_C': 'Tmin', 'srad_mean': 'SRAD', 'swe_mean': 'SWE', 'pdsi_mean': 'PDSI',
    'ro_mean': 'RO' 
}
 
df = pd.read_csv("UA_SOS_TerraClimate_2001_2022_rescaled.csv")
target = "SOS"
features = list(rename_dict.keys())
df = df[features + [target]].dropna()
 
X = df[features]
X_renamed = X.rename(columns=rename_dict)
y = df[target]
 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=features).rename(columns=rename_dict)
 
# Split training set and test set
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)
 
# Training the model
model = xgb.XGBRegressor(n_estimators=2000, max_depth=6, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)
 
y_pred = model.predict(X_test)
print("R² on test set:", r2_score(y_test, y_pred))
 
# calculate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_scaled_df)
 
# Calculate and print the standard deviation of the SHAP value for each feature
print("\nSHAP std:”)
shap_std = np.std(shap_values, axis=0)
for feature, std in zip(X_scaled_df.columns, shap_std):
    print(f"{feature}: {std:.4f}")
 
mean_shap = np.abs(shap_values).mean(axis=0)
feature_names = X_scaled_df.columns
sorted_idx = np.argsort(mean_shap)  # 从小到大
sorted_features = feature_names[sorted_idx]
sorted_mean_shap = mean_shap[sorted_idx]

plt.rcParams.update({
    'font.size': 22,
    'axes.labelsize': 22,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'legend.fontsize': 20  
})
 

plt.figure(figsize=(8, 8), dpi=700)
 
# Sort by mean_shap in ascending order (smallest to largest)
sorted_idx = np.argsort(mean_shap)  
sorted_features = feature_names[sorted_idx]
sorted_mean_shap = mean_shap[sorted_idx]
 

shap.summary_plot(
    shap_values[:, sorted_idx], 
    X_scaled_df.iloc[:, sorted_idx],  
    feature_names=sorted_features,  
    plot_type="dot",
    show=False,
)
 
fig = plt.gcf()
for text in fig.findobj(matplotlib.text.Text):
    if text.get_text() in ['Feature value', 'Low', 'High']:
        text.set_fontsize(20)
 
ax1 = plt.gca()
ax1.set_xlabel("SHAP value (Bee Swarm)", fontsize=22, color='black')
ax1.set_ylabel("Features", fontsize=22, color='black')
ax1.tick_params(axis='both', which='major', labelsize=20, colors='black')
 
ax1.set_xticks(np.arange(-150, 151, 50))
 
max_val = np.abs(shap_values).max()
ax1.set_xlim(-max_val * 1.1, max_val * 1.1)
 
ax2 = ax1.twiny()
ax2.barh(
    y=np.arange(len(sorted_features)),
    width=sorted_mean_shap,
    height=0.6,
    color='b',
    alpha=0.3,
)
ax2.set_xlim(0, sorted_mean_shap.max() * 1.1)
ax2.set_xlabel("Mean |SHAP value| (UA-SOS)", fontsize=22, color='black')
ax2.xaxis.set_label_position('top')
ax2.xaxis.tick_top()
ax2.tick_params(axis='x', labelsize=20, colors='black')
ax2.set_yticks([])  
 
plt.yticks(np.arange(len(sorted_features)), sorted_features, fontsize=20)
 
ax1.set_position([0.3, 0.1, 0.65, 0.8])
ax2.set_position([0.3, 0.1, 0.65, 0.8])
 
plt.tight_layout()
plt.savefig("UA_SOS_beeswarm.jpg", dpi=700, bbox_inches='tight')
plt.show()

#%% Dependence plot of climate drivers

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib.colorbar import Colorbar  
from matplotlib.ticker import MaxNLocator
 
# Rename the fields
rename_dict = {
    'pet_mean': 'PET', 'aet_mean': 'AET', 'def_mean': 'DEF', 'pr_sum': 'P',
    'soil_mean': 'TSM', 'vpd_mean': 'VPD', 'vap_mean': 'VAP', 'tmmx_C': 'Tmax',
    'tmmn_C': 'Tmin', 'srad_mean': 'SRAD', 'swe_mean': 'SWE', 'pdsi_mean': 'PDSI',
    'ro_mean': 'RO' 
}
 
df = pd.read_csv(“UA_SOS_TerraClimate_2001_2022_rescaled.csv")
 
# Create Output Folder
output_dir = "UA_SOS_dependence_plots"
os.makedirs(output_dir, exist_ok=True)
 
# Setting dependent variables and features
target = "SOS" #change it to EOS, if needed
features = list(rename_dict.keys())
 
df = df[features + [target]].dropna()
 
X = df[features].copy()
y = df[target].copy()
 
X_renamed = X.rename(columns=rename_dict)
features_renamed = list(X_renamed.columns)
 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=features)
#X_scaled_df_renamed = X_scaled_df.rename(columns=rename_dict)
X_scaled_df_renamed  = X.rename(columns=rename_dict)  
 
# Split training set and test set
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df_renamed, y, test_size=0.2, random_state=42)
 
# Training the model
model = xgb.XGBRegressor(n_estimators=2000, max_depth=6, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)
 
explainer = shap.Explainer(model)
shap_values = explainer(X_scaled_df_renamed

# Draw a dependence plot for each feature
for feat in features_renamed:
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
 
    shap.dependence_plot(
        feat,
        shap_values.values,
        X_scaled_df_renamed,
        feature_names=features_renamed,
        ax=ax,
        show=False
    )
 
    ax.set_xlabel(ax.get_xlabel(), fontsize=14)
    ax.set_ylabel("SHAP Value", fontsize=14)
    ax.tick_params(axis='both', labelsize=13)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_title("")
 
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
 
    for cb_ax in fig.axes:
        if cb_ax is not ax:
            cb_ax.tick_params(labelsize=11)
            cb_ax.set_ylabel(cb_ax.get_ylabel(), fontsize=13)
 
    plt.tight_layout(pad=0.1)
 
    plt.savefig(f"{output_dir}/{feat}_dependence_plot.jpg", dpi=700)
    plt.close()

