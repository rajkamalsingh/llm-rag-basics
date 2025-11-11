import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.datasets import sunspots
from statsmodels.tsa.seasonal import STL

# Load the sunspots dataset
data = sunspots.load_pandas().data
data.index = pd.Index(pd.period_range('1700', '2008', freq='A'))
series = data['SUNACTIVITY']

# STL decomposition with m = 11 (non-robust)
stl_nonrobust = STL(series, period=11, robust=False).fit()

# STL decomposition with m = 11 (robust)
stl_robust = STL(series, period=11, robust=True).fit()

# Convert PeriodIndex to datetime for plotting
series = series.to_timestamp()

# --- Plot manually to compare ---
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
fig.suptitle("STL Decomposition of Sunspot Activity (m=11)", fontsize=14)

# Non-Robust STL
axes[0, 0].plot(series.index, series.values, color='black')
axes[0, 0].set_title("Original (Non-Robust)")
axes[0, 1].plot(series.index, stl_nonrobust.trend, color='blue')
axes[0, 1].set_title("Trend (Non-Robust)")
axes[0, 2].plot(series.index, stl_nonrobust.seasonal, color='green')
axes[0, 2].set_title("Seasonal (Non-Robust)")

# Robust STL
axes[1, 0].plot(series.index, series.values, color='black')
axes[1, 0].set_title("Original (Robust)")
axes[1, 1].plot(series.index, stl_robust.trend, color='blue')
axes[1, 1].set_title("Trend (Robust)")
axes[1, 2].plot(series.index, stl_robust.seasonal, color='green')
axes[1, 2].set_title("Seasonal (Robust)")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
