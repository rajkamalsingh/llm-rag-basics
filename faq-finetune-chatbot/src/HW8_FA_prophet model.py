import numpy as np
import matplotlib.pyplot as plt

theta = -0.5
phi = -0.5
sigma2 = 1.0

# frequency in cycles/sample (0..0.5)
f = np.linspace(0, 0.5, 500)

# spectral densities (per radian normalization 1/(2π))
f_MA = (sigma2/(2*np.pi)) * (1 + theta**2 - 2*theta*np.cos(2*np.pi*f))
f_AR = (sigma2/(2*np.pi)) / (1 + phi**2 - 2*phi*np.cos(2*np.pi*f))

plt.figure(figsize=(10,6))
plt.plot(f, f_MA, label='MA(1) θ=-0.5')
plt.plot(f, f_AR, label='AR(1) φ=-0.5')
plt.xlabel('Frequency (cycles per sample)')
plt.ylabel('Spectral density')
plt.title('Spectral densities (consistent conventions)')
plt.legend()
plt.grid(True)
plt.show()


# prophet_mortgage_forecast.py
# Run in an environment with prophet installed. Example:
# pip install prophet pandas matplotlib scikit-learn

import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# -----------------------------
# Load data from FRED
# -----------------------------
url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
df = pd.read_csv(url)

# Rename to Prophet-compatible column names
df = df.rename(columns={'observation_date': 'ds', 'MORTGAGE30US': 'y'})

# Convert to datetime and numeric
df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
df['y'] = pd.to_numeric(df['y'], errors='coerce')
df = df.dropna().sort_values('ds').reset_index(drop=True)

print(f"Data loaded: {len(df)} rows from {df['ds'].min()} to {df['ds'].max()}")

# -----------------------------
# Split into train/test sets
# -----------------------------
split_date = pd.to_datetime("2020-10-29")
train = df[df['ds'] <= split_date]
test = df[df['ds'] > split_date]

print(f"Training samples: {len(train)}, Test samples: {len(test)}")

# -----------------------------
# Prophet model setup
# -----------------------------
m = Prophet(
    changepoint_prior_scale=0.05,  # 10 change points by default
    yearly_seasonality=False,      # we'll add custom ones
    weekly_seasonality=False,
    daily_seasonality=False
)

# Add seasonality components
m.add_seasonality(name='yearly', period=365.25, fourier_order=10)
m.add_seasonality(name='two_year', period=365.25*2, fourier_order=10)
m.add_seasonality(name='five_year', period=365.25*5, fourier_order=10)

# Fit model
m.fit(train)

# -----------------------------
# Forecast through test period
# -----------------------------
future = m.make_future_dataframe(periods=len(test), freq='W')
forecast = m.predict(future)

# -----------------------------
# Plots
# -----------------------------
# 1. Prophet built-in plots
fig1 = m.plot(forecast)
plt.title("Prophet Forecast – 30-Year Mortgage Rate")
plt.xlabel("Date")
plt.ylabel("Rate (%)")

fig2 = m.plot_components(forecast)

# 2. Plot predictions vs. observed
plt.figure(figsize=(10,6))
plt.plot(train['ds'], train['y'], label='Training Data', color='blue')
plt.plot(test['ds'], test['y'], label='Test Data (Actual)', color='black')
plt.plot(forecast['ds'], forecast['yhat'], label='Prophet Forecast', color='orange')
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='orange', alpha=0.3)
plt.axvline(split_date, color='red', linestyle='--', label='Train/Test Split')
plt.legend()
plt.title("Prophet Predicted vs Actual Mortgage Rates")
plt.xlabel("Date")
plt.ylabel("Rate (%)")
plt.show()

