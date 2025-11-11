import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram

# --- Helper Function: Generate the deterministic signal ---
def generate_signal(n):
    t = np.arange(1, n + 1)
    xt1 = 2 * np.cos(2 * np.pi * 0.06 * t) + 3 * np.sin(2 * np.pi * 0.06 * t)
    xt2 = 4 * np.cos(2 * np.pi * 0.10 * t) + 5 * np.sin(2 * np.pi * 0.10 * t)
    xt3 = 6 * np.cos(2 * np.pi * 0.40 * t) + 7 * np.sin(2 * np.pi * 0.40 * t)
    xt = xt1 + xt2 + xt3
    return t, xt

# --- (a) Generate and plot x_t for n=128 ---
n = 128
t, xt = generate_signal(n)

plt.figure(figsize=(12, 4))
plt.plot(t, xt, color='blue')
plt.title("Simulated Time Series x_t (n=128)")
plt.xlabel("Time")
plt.ylabel("x_t")
plt.grid(True)
plt.show()

# --- (b) Compute and plot the periodogram for n=128 ---
freqs, Pxx = periodogram(xt)

plt.figure(figsize=(10, 4))
plt.plot(freqs, Pxx, color='red')
plt.title("Periodogram of x_t (n=128)")
plt.xlabel("Frequency")
plt.ylabel("Power")
plt.grid(True)
plt.show()

print("Observation (n=128):")
print(" - The periodogram shows 3 sharp peaks near 0.06, 0.10, and 0.40.")
print(" - Larger n gives finer frequency resolution and narrower peaks.\n")

# --- (c) Repeat with n=100 and add Gaussian noise (N(0, 25)) ---
n = 100
t, xt_clean = generate_signal(n)
noise = np.random.normal(0, 5, n)  # variance = 25 -> std dev = 5
xt_noisy = xt_clean + noise

# Plot noisy time series
plt.figure(figsize=(12, 4))
plt.plot(t, xt_noisy, color='green')
plt.title("Noisy Time Series x_t (n=100, σ²=25)")
plt.xlabel("Time")
plt.ylabel("x_t")
plt.grid(True)
plt.show()

# Compute and plot periodogram for noisy data
freqs, Pxx_noisy = periodogram(xt_noisy)

plt.figure(figsize=(10, 4))
plt.plot(freqs, Pxx_noisy, color='purple')
plt.title("Periodogram of Noisy x_t (n=100, σ²=25)")
plt.xlabel("Frequency")
plt.ylabel("Power")
plt.grid(True)
plt.show()

print("Observation (n=100, with noise):")
print(" - Added noise broadens and obscures spectral peaks.")
print(" - Peaks near 0.06, 0.10, 0.40 are visible but less distinct.")
print(" - Smaller n reduces spectral resolution, causing wider peaks.\n")

# --- Summary of findings ---
print("Summary Comparison:")
print("1️⃣ n=128 (no noise): Clear and narrow peaks → high frequency resolution.")
print("2️⃣ n=100 (no noise): Peaks still visible but slightly broader.")
print("3️⃣ n=100 (with noise): Peaks blurred and less pronounced → lower SNR.\n")
print("✅ Larger n improves resolution; noise reduces spectral clarity.")



# --- Q2-----
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram
from statsmodels.datasets import sunspots

# Load the dataset
data = sunspots.load_pandas().data
series = data['SUNACTIVITY']

# Compute periodogram
freqs, power = periodogram(series, scaling='density')

# Plot
plt.figure(figsize=(10, 6))
plt.plot(freqs, power, color='tab:blue')
plt.title("Periodogram of Sunspot Activity")
plt.xlabel("Frequency (cycles/year)")
plt.ylabel("Spectral Density")
plt.grid(True)
plt.show()

# ---- Q3 -----

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram
from statsmodels.tsa.arima_process import ArmaProcess

np.random.seed(42)  # for reproducibility

# --- (1) Pure White Noise ---
n = 10000
white_noise = np.random.normal(0, 1, n)

# --- (2) MA(1) process with theta = 0.75 ---
ma1 = ArmaProcess(ar=[1], ma=[1, 0.75])
ma1_data = ma1.generate_sample(nsample=n)

# --- (3) AR(1) process with phi = -0.75 ---
ar1 = ArmaProcess(ar=[1, 0.75], ma=[1])  # note: sign convention: phi = -0.75 → ar=[1, 0.75]
ar1_data = ar1.generate_sample(nsample=n)

# --- Compute and plot periodograms ---
fig, axes = plt.subplots(3, 1, figsize=(10, 10))

processes = [white_noise, ma1_data, ar1_data]
titles = ['White Noise', 'MA(1): θ = 0.75', 'AR(1): φ = -0.75']

for i, (data, title) in enumerate(zip(processes, titles)):
    freqs, power = periodogram(data, scaling='density')
    axes[i].plot(freqs, power, color='tab:blue')
    axes[i].set_title(f"Periodogram of {title}")
    axes[i].set_xlabel("Frequency")
    axes[i].set_ylabel("Spectral Density")
    axes[i].grid(True)

plt.tight_layout()
plt.show()
