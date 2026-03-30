# ar1_stationarity_sim.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)

def simulate_ar1(phi, sigma, n, x0=0.0):
    x = np.empty(n)
    x[0] = x0 + np.random.normal(0, sigma)
    for t in range(1, n):
        x[t] = phi * x[t-1] + np.random.normal(0, sigma)
    return x

phi_true = 0.98
sigma = 1.0
lengths = [100, 500, 1000]

rows = []
for n in lengths:
    x = simulate_ar1(phi_true, sigma, n)

    # Fit AR(1) with intercept
    model = AutoReg(x, lags=1, trend='c')
    fit = model.fit()

    # Parameter extraction (NO `.index` — params is a NumPy array)
    params = fit.params      # [const, phi]
    bse = fit.bse
    ci = fit.conf_int()

    const = params[0]           # intercept
    phi_hat = params[1]         # AR(1) coefficient

    phi_se = bse[1]
    phi_ci_low = ci[1][0]
    phi_ci_high = ci[1][1]

    # ADF test (H0: unit root / nonstationary)
    adf_res = adfuller(x, regression='c', autolag='AIC')
    adf_stat, adf_pvalue = adf_res[0], adf_res[1]

    # KPSS test (H0: stationary)
    try:
        kpss_stat, kpss_pvalue, kpss_lags, kpss_crit = kpss(x, regression='c', nlags='auto')
    except Exception:
        kpss_stat, kpss_pvalue, kpss_lags, kpss_crit = kpss(x, regression='c', nlags=10)

    rows.append({
        'n': n,
        'phi_hat': phi_hat,
        'phi_se': phi_se,
        'phi_ci_low': phi_ci_low,
        'phi_ci_high': phi_ci_high,
        'adf_stat': adf_stat,
        'adf_pvalue': adf_pvalue,
        'kpss_stat': kpss_stat,
        'kpss_pvalue': kpss_pvalue
    })

    # Plot series
    plt.figure(figsize=(10, 2.5))
    plt.plot(x, lw=0.8)
    plt.title(f'AR(1) simulation φ={phi_true}, n={n}')
    plt.xlabel('t')
    plt.ylabel('x_t')
    plt.tight_layout()
    plt.show()

df = pd.DataFrame(rows)
pd.options.display.float_format = '{:,.4f}'.format
print(df[['n','phi_hat','phi_se','phi_ci_low','phi_ci_high','adf_stat','adf_pvalue','kpss_stat','kpss_pvalue']])
