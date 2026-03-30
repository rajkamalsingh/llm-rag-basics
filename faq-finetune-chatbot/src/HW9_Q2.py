import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR
from statsmodels.graphics.tsaplots import plot_acf
import warnings
warnings.filterwarnings("ignore")
# ===========================================================
# LOAD DATA
# ===========================================================
co2_path = "/content/drive/MyDrive/data_643/co2-mm-mlo.csv"
temp_path = "/content/drive/MyDrive/data_643/monthly.csv"
# Load CO2
co2 = pd.read_csv(co2_path, index_col=0)
print("Raw CO2 head:\n", co2.head())
# Parse date safely, drop duplicates
co2.index = pd.to_datetime(co2.index, errors='coerce')
co2 = co2[~co2.index.duplicated(keep='first')]
co2 = co2[['Average']].rename(columns={'Average':'co2'})
co2 = co2.dropna()
print("\nParsed CO2:\n", co2.head())
# Load Temperature
temp = pd.read_csv(temp_path)
print("\nRaw Temp head:\n", temp.head())
# Detect date column automatically
date_col = temp.columns[temp.iloc[0].astype(str).str.contains('-')][0]
temp[date_col] = pd.to_datetime(temp[date_col], errors='coerce')
temp = temp[~temp[date_col].duplicated(keep='first')]
temp = temp.set_index(date_col)
temp = temp[['Mean']].rename(columns={'Mean':'temp'})
temp = temp.dropna()
print("\nParsed Temp:\n", temp.head())
# ===========================================================
# DIFFERENCING
# ===========================================================
co2_diff1 = co2.diff().dropna()
co2_diff12 = co2.diff(12).dropna()
temp_diff1 = temp.diff().dropna()
temp_diff12 = temp.diff(12).dropna()
# ===========================================================
# ADF TEST FUNCTION
# ===========================================================
def adf_test(series, name):
    print(f"\nADF Test for {name}")
    result = adfuller(series)
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    for k, v in result[4].items():
        print(f"Critical Value ({k}): {v:.4f}")
    print("=> Stationary" if result[1] < 0.05 else "=> NOT Stationary")

# Run ADF tests
adf_test(co2['co2'], "CO2 Original")
adf_test(co2_diff1['co2'], "CO2 1st Difference")
adf_test(co2_diff12['co2'], "CO2 12th Difference")
adf_test(temp['temp'], "Temperature Original")
adf_test(temp_diff1['temp'], "Temperature 1st Difference")
adf_test(temp_diff12['temp'], "Temperature 12th Difference")
# ===========================================================
# PLOT ADF
# ===========================================================
def plot_adf(series, name):
    result = adfuller(series)
    stat = result[0]
    crit = result[4]
    plt.figure(figsize=(10,5))
    plt.axvline(stat, color='red', label=f"ADF Statistic = {stat:.3f}")
    for level, value in crit.items():
        plt.axvline(value, linestyle='--', label=f"{level} critical = {value:.3f}")
    plt.axvline(0, color='black', linewidth=1)
    plt.title(f"ADF Test Visualization for {name}")
    plt.xlabel("Test Statistic")
    plt.legend()
    plt.grid(True)
    plt.show()
# Plot first differences
df_diff = pd.concat([co2_diff1.rename(columns={'co2':'CO2_diff1'}),
temp_diff1.rename(columns={'temp':'Temp_diff1'})],
axis=1, join='inner').dropna()
plot_adf(df_diff['CO2_diff1'], "Δ CO₂ (1st diff)")
plot_adf(df_diff['Temp_diff1'], "Δ Temp (1st diff)")
# ===========================================================
# MERGE FOR VAR/GRANGER (use 12th diff)
# ===========================================================
merged = pd.concat([co2_diff12, temp_diff12], axis=1, join='inner').dropna()
merged.columns = ['co2','temp']
print("\nMerged shape:", merged.shape)
print(merged.head())
# ===========================================================
# ACF PLOTS
# ===========================================================
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.title("ACF - CO2 (12th Diff)")
plot_acf(merged['co2'], ax=plt.gca())
plt.subplot(1,2,2)
plt.title("ACF - Temp (12th Diff)")
plot_acf(merged['temp'], ax=plt.gca())
plt.tight_layout()
plt.show()

===========================================================
# GRANGER CAUSALITY TESTS
# ===========================================================
n_obs = merged.shape[0]
maxlag_allowed = n_obs//3 - 1
safe_maxlag = min(12, maxlag_allowed) # use <=12 for monthly data
print(f"\nNumber of observations: {n_obs}")
print(f"Maximum allowable lag: {maxlag_allowed}")
print(f"Using safe maxlag: {safe_maxlag}")
print("\nCO2 → Temp")
grangercausalitytests(merged[['temp','co2']], maxlag=safe_maxlag, verbose=True)
print("\nTemp → CO2")
grangercausalitytests(merged[['co2','temp']], maxlag=safe_maxlag, verbose=True)
# ===========================================================
# VAR(1) MODEL
# ===========================================================
model = VAR(merged)
results = model.fit(1)
print("\nVAR(1) Summary:\n")
print(results.summary())

# ===========================================================
# VAR FORECASTS
# ===========================================================
# Forecast next 24 months
forecast_steps = 24
forecast = results.forecast(merged.values[-results.k_ar:], steps=forecast_steps)
forecast_index = pd.date_range(start=merged.index[-1] + pd.DateOffset(months=1),
forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=merged.column)
print("\nForecasted CO2 and Temperature (next 24 months):\n", forecast_df.head())
# ===========================================================
# PLOT FORECASTS
# ===========================================================
plt.figure(figsize=(14,6))
plt.plot(merged['co2'], label='CO2 (Observed)')
plt.plot(forecast_df['co2'], label='CO2 (Forecast)', linestyle='--')
plt.title('CO2 Forecast (VAR(1))')
plt.xlabel('Date')
plt.ylabel('CO2 (ppm)')
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(14,6))
plt.plot(merged['temp'], label='Temperature (Observed)')
plt.plot(forecast_df['temp'], label='Temperature (Forecast)', linestyle='--')
plt.title('Temperature Forecast (VAR(1))')
plt.xlabel('Date')
plt.ylabel('Temperature Anomaly (°C)')
plt.legend()
plt.grid(True)
plt.show()