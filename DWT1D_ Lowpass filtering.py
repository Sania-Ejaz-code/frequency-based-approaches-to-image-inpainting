import numpy as np
import matplotlib.pyplot as plt

# CHANGED: use PyWavelets instead of DCT/FFT
import pywt     
                              # CHANGED
# Simulated 1D signal
x = np.linspace(0, 10, 500)
f_original = np.sin(x)+  np.sin(8 * x)

# Missing region D
D_start, D_end = 250, 350
f_missing = f_original.copy()
f_missing[D_start:D_end] = 0

# Estimate mean from neighbors U
U_left = f_original[D_start - 80:D_start]
U_right = f_original[D_end:D_end + 80]
m = np.mean(np.concatenate([U_left, U_right]))

# Define the inpainting region
D_indices = np.arange(D_start, D_end)

# Initialize inpainting with mean
f_inpaint = f_missing.copy()
f_inpaint[D_start:D_end] = m

# ---- Iterative frequency-domain filtering (now with WAVELETS) ----
threshold = 0.00001     # epsilon
max_iters = 450

# --- Wavelet choices (tweak these, see notes below)
wname = 'db8'            # was 'db8' – smoother, less ringing
mode  = 'periodic'   # was 'symmetric' – better for quasi-periodic signals
N     = len(x)
max_level = pywt.dwt_max_level(N, pywt.Wavelet(wname).dec_len)   # CHANGED
level = 4                    # CHANGED: cap to be safe; auto-computed maximum is fine too

# We’ll mimic your "r ramp" idea: r in [0,1] maps to how many *coarsest* detail bands we keep.
r = 0.5
growth_factor = 1.09
max_r = 2

def wavelet_lowpass_1d(sig, wavelet, level, keep_frac, mode='symmetric'):
    """
    Low-pass via DWT: keep the approximation at 'level' and the 'keep_k'
    coarsest detail bands; zero all finer (high-frequency) detail bands.
    keep_frac in [0,1] -> keep_k = round(keep_frac * level).
    """
    coeffs = pywt.wavedec(sig, wavelet=wavelet, mode=mode, level=level)
    # coeffs = [cA_L, cD_L, cD_{L-1}, ..., cD_1]
    L = len(coeffs) - 1  # number of detail bands
    keep_k = int(np.clip(np.round(keep_frac * L), 0, L))
    # Zero all detail bands except the 'keep_k' *coarsest* ones (cD_L, cD_{L-1}, ...)
    for j in range(1, L - keep_k + 1):
        coeffs[j] = np.zeros_like(coeffs[j])
    # Optional: light soft-threshold inside the kept bands (commented out by default)
    thr = 0.0
    for j in range(L - keep_k + 1, L + 1):
      coeffs[j] = pywt.threshold(coeffs[j], thr, mode='soft')
    return pywt.waverec(coeffs, wavelet=wavelet, mode=mode)

for i in range(max_iters):
    prev = f_inpaint.copy()

    # CHANGED: forward->low-pass->inverse all done via helper
    g = wavelet_lowpass_1d(f_inpaint, wname, level, keep_frac=r, mode=mode)  # CHANGED

    # Data fidelity: keep known data outside D
    f_inpaint[:D_start] = f_missing[:D_start]
    f_inpaint[D_end:]   = f_missing[D_end:]
    f_inpaint[D_start:D_end] = g[D_start:D_end]

    # Convergence on the gap only
    change = np.sqrt(np.mean((f_inpaint[D_indices] - prev[D_indices]) ** 2))
    if change < threshold:
        break

    r = min(r * growth_factor, max_r) # Gradually increase the cutoff frequency, up to a maximum


# ---- (your plotting and metrics code stays exactly the same) ----
plt.figure(figsize=(10, 5))
plt.plot(x, f_original, label="Original signal")
plt.plot(x, f_missing, label="With missing region (D)", linestyle=":")
plt.plot(x, f_inpaint, label="Reconstructed iteratively (Wavelet)", linewidth=2, linestyle="--")
plt.axvspan(x[D_start], x[D_end], color='gray', alpha=0.2, label="Inpainting region D")
plt.legend(); plt.tight_layout(); plt.show()

true_values = f_original[D_indices]
reconstructed_values = f_inpaint[D_indices]
mse = np.mean((reconstructed_values - true_values) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(reconstructed_values - true_values))
print(f"MSE  = {mse:.6f}")
print(f"RMSE = {rmse:.6f}")
print(f"MAE  = {mae:.6f}")
print("r=", r, " growth factor=", growth_factor, " max r=", max_r)
