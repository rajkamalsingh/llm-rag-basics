# Required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.datasets import sunspots
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.stl import STLForecast   # statsmodels >= 0.13

# 1) Load data and compute robust STL (period m = 11)
data = sunspots.load_pandas().data
# make a datetime index for plotting convenience (annual)
years = pd.date_range(start=str(int(data['YEAR'].min())), periods=len(data), freq='A')
series = pd.Series(data['SUNACTIVITY'].values, index=years)

# robust STL with seasonal period 11
stl_robust = STL(series, period=11, robust=True).fit()
trend = stl_robust.trend.dropna()   # drop any leading/trailing NaNs

# 2) ACF and PACF of the robust trend to pick p and q
plt.figure(figsize=(12,4))
ax1 = plt.subplot(1,2,1)
plot_acf(trend, lags=40, ax=ax1, title="ACF of Robust STL Trend")
ax2 = plt.subplot(1,2,2)
plot_pacf(trend, lags=40, ax=ax2, method='ywm', title="PACF of Robust STL Trend")
plt.tight_layout()
plt.show()

# ---- Guidance for choosing p and q ----
# Inspect the ACF and PACF plots produced above.
# Typical heuristics:
#  - If PACF cuts off after lag p and ACF tails off -> AR(p)
#  - If ACF cuts off after lag q and PACF tails off -> MA(q)
#  - If both tail off -> ARMA(p,q)
#
# For the sunspot trend (yearly, ~11-year cycle) you will likely see PACF cut off
# after lag 2 , and ACF decaying -> this suggests AR(2) is a reasonable starting point.
# We'll use AR(2).
p = 2
d = 0   # trend series is (approximately) stationary in levels after STL trend extraction
q = 0   # set q = 0 for AR(2). If PACF/ACF suggest MA part, set q>0.

# 3) Use STLForecast wrapping an ARIMA model for the seasonal series
# Forecast horizon: until year 2050
last_year = trend.index[-1].year
horizon_year = 2050
steps = horizon_year - last_year
print(f"Last year in data: {last_year}. Forecasting {steps} years into future to {horizon_year}.")

# Build STLForecast; seasonal=11 uses the same seasonal period as decomposition
stl_forecaster = STLForecast(series,
                             model=ARIMA,
                             model_kwargs={"order": (p, d, q)},
                             period=11)

stl_res = stl_forecaster.fit()
# get forecast (predicted mean) and conf_int
forecast_res = stl_res.forecast(steps=steps)
# Statsmodels STLForecast results offers get_prediction with return_conf_int if available:
# The .forecast(steps) returns the forecasted mean. To obtain intervals:
pred = stl_res.get_prediction(steps=steps)
pred_mean = pred.predicted_mean
pred_ci = pred.conf_int()

# 4) Plot observed series, fitted values (from underlying ARIMA), and forecasts with confidence intervals.
fig, ax = plt.subplots(figsize=(12,5))
ax.plot(series.index, series.values, label='Observed', color='black', linewidth=0.8)

# Get fitted values from the underlying ARIMA model inside STLForecast
fitted = stl_res.model_result.fittedvalues
ax.plot(fitted.index, fitted.values, label='STL+ARIMA fitted', color='tab:blue', alpha=0.9)

# Forecast mean and confidence intervals
# Get only the forecast horizon (last `steps` points)
pred_mean_future = pred_mean[-steps:]
pred_ci_future = pred_ci[-steps:]

# Create matching datetime index
fc_index = pd.date_range(start=series.index[-1] + pd.offsets.DateOffset(years=1), periods=steps, freq='A')

# Plot forecasts
ax.plot(fc_index, pred_mean_future.values, label='Forecast (mean)', color='tab:orange')
ax.fill_between(fc_index,
                pred_ci_future.iloc[:, 0].values,
                pred_ci_future.iloc[:, 1].values,
                color='tab:orange', alpha=0.25, label='95% CI')


ax.set_title(f"STLForecast with ARIMA({p},{d},{q}) - Forecast through {horizon_year}")
ax.set_xlabel("Year")
ax.set_ylabel("Sunspot activity")
ax.legend()
plt.tight_layout()
plt.show()

# 5) Quick diagnostics: print model summary and comment
print(stl_res.model_result.summary())
