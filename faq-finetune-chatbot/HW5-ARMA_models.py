import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.datasets import sunspots
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox


# Load dataset
data = sunspots.load_pandas().data
data.head()
series = data['SUNACTIVITY']

plt.figure(figsize=(12, 5))
plt.plot(data['YEAR'], data['SUNACTIVITY'], color='steelblue', linewidth=1.2)
plt.title('Yearly Sunspot Activity (1700–2008)', fontsize=14)
plt.xlabel('Year')
plt.ylabel('Sunspot Count')
plt.grid(True, alpha=0.3)
plt.show()

fig, ax = plt.subplots(2, 1, figsize=(10,8))

plot_acf(series, lags=40, ax=ax[0])
ax[0].set_title('Autocorrelation Function (ACF)')

plot_pacf(series, lags=40, ax=ax[1], method='ywm')
ax[1].set_title('Partial Autocorrelation Function (PACF)')

plt.tight_layout()
plt.show()


# Define model configurations
models = {
    'AR(1)': (1,0,0),
    'AR(2)': (2,0,0),
    'MA(1)': (0,0,1),
    'MA(2)': (0,0,2),
    'ARMA(1,1)': (1,0,1),
    'ARMA(2,2)': (2,0,2),
    'ARMA(1,2)': (1,0,2),
    'ARMA(2,1)': (2,0,1)
}

results = []

# Fit each model and store AIC/BIC
for name, order in models.items():
    model = ARIMA(series, order=order)
    fitted = model.fit()
    results.append({
        'Model': name,
        'AIC': fitted.aic,
        'BIC': fitted.bic
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results).sort_values(by='BIC')
print(results_df)

def compute_roots_from_ar_coefs(ar_coefs):
    # ar_coefs: list/array [phi1, phi2, ...]
    # polynomial 1 - phi1 z - phi2 z^2 - ...
    # numpy.roots expects highest-first, so build [ -phi_n, ..., -phi1, 1 ]? easier:
    # For polynomial in z: c0 + c1 z + c2 z^2 ... we create coef array [c2, c1, c0]
    # Here c0 = 1, c1 = -phi1, c2 = -phi2, ...
    # numpy.roots expects highest degree first:
    coeffs = [1]  # we'll reorder below
    # build in increasing power then reverse
    coeffs_inc = [1] + [-p for p in ar_coefs]   # [1, -phi1, -phi2, ...]
    coeffs = coeffs_inc[::-1]  # reverse for numpy.roots
    roots = np.roots(coeffs)
    return roots

def interpret_roots(roots):
    for i,r in enumerate(roots,1):
        modulus = np.abs(r)
        angle = np.angle(r)
        print(f"Root {i}: {r:.4g}, modulus={modulus:.4f}", end="; ")
        if modulus > 1:
            print("outside unit circle (stationary mode); ", end="")
        elif modulus < 1:
            print("inside unit circle (non-stationary/explosive); ", end="")
        else:
            print("on unit circle; ", end="")
        if abs(r.imag) < 1e-8:
            print("real root")
        else:
            period = 2*np.pi / abs(angle) if abs(angle)>1e-12 else np.inf
            print(f"complex conjugate component -> implied period ≈ {period:.2f} lags")

models_for_coeff = {
    'AR(2)': (2,0,0),
    'ARMA(2,2)': (2,0,2),
    'ARMA(2,1)': (2,0,1)
}
for name, order in models_for_coeff.items():
    model = ARIMA(series, order=order)
    fitted = model.fit()
    phi1 = fitted.params.get('ar.L1')
    phi2 = fitted.params.get('ar.L2')
    #print(phi1, phi2)
    roots = compute_roots_from_ar_coefs([phi1, phi2])
    interpret_roots(roots)


# Fit best model (as per BIC)
best_model = ARIMA(series, order=(2, 0, 0)).fit()  #  AR(2) is best

# Display summary
print(best_model.summary())
print("\n")
# Diagnostic plots
best_model.plot_diagnostics(figsize=(10, 8))
plt.show()

# Ljung-Box test for autocorrelation
ljung_box_result = acorr_ljungbox(best_model.resid, lags=[10], return_df=True)
print(ljung_box_result)

