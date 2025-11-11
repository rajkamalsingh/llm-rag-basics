import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.datasets import sunspots
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# --- Load sunspots series (annual) and make a datetime index ---
data = sunspots.load_pandas().data
years = pd.date_range(start=str(int(data['YEAR'].min())), periods=len(data), freq='A')
series = pd.Series(data['SUNACTIVITY'].values, index=years)
series.name = "sunspots"

# --- Fit Holt-Winters (additive trend, additive seasonal) with seasonal_periods=11 ---
hw_model = ExponentialSmoothing(series,
                                trend='add',
                                seasonal='add',
                                seasonal_periods=11,
                                initialization_method="estimated")
hw_res = hw_model.fit(optimized=True)

# --- Forecast through 2050 ---
last_year = series.index[-1].year
horizon_year = 2050
steps = horizon_year - last_year
fc_index = pd.date_range(start=series.index[-1] + pd.offsets.DateOffset(years=1),
                         periods=steps, freq='A')

hw_forecast = hw_res.forecast(steps)

# --- Plot observed data, fitted values (in-sample), and forecast ---
plt.figure(figsize=(12,5))
plt.plot(series.index, series.values, label='Observed', color='black', linewidth=0.8)
plt.plot(series.index, hw_res.fittedvalues, label='Holt-Winters (fitted)', color='tab:blue', alpha=0.9)
plt.plot(fc_index, hw_forecast.values, label='Holt-Winters Forecast (to 2050)', color='tab:orange')

plt.title('Holt–Winters (additive trend + additive seasonality, m=11) — Sunspots')
plt.xlabel('Year')
plt.ylabel('Sunspot activity')
plt.legend()
plt.tight_layout()
plt.show()

# --- Quick model printout (coeffs) ---
print(hw_res.params)
print("\nAIC/BIC are not available directly for ExponentialSmoothing in statsmodels.\n")
