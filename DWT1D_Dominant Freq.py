import numpy as np
import matplotlib.pyplot as plt
import pywt     
                              # CHANGED
# Simulated 1D signal
x = np.linspace(0, 10, 500)
f_original = np.sin(x) +  np.sin(8 * x)

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
# --------------------------------------------------------------------
# ---------------------------------------------------------------------

# ---- Iterative frequency-domain filtering (now with WAVELETS) ----
threshold = 0.000001
max_iters = 400
k_frac = 0.09   # keep top 6% of coefficients
eps = 1e-12
wavelet = 'sym8'     # Daubechies 4 (try also 'haar', 'sym8', 'coif5'...)
maxlevel = 12     # let pywt choose the max possible level

for i in range(max_iters):
    prev = f_inpaint.copy()

    # --- Wavelet decomposition ---
    coeffs = pywt.wavedec(f_inpaint, wavelet=wavelet, level=maxlevel)
    coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)

    # --- Select dominant coefficients (top-K) ---
    mags = np.abs(coeff_arr)
    K = max(1, int(k_frac * len(mags)))
    thresh = np.partition(mags, -K)[-K]
    keep_mask = mags >= max(thresh, eps)

    # --- Zero out non-dominant coefficients ---
    coeff_arr_dom = coeff_arr * keep_mask
    coeffs_dom = pywt.array_to_coeffs(coeff_arr_dom, coeff_slices, output_format='wavedec')

    # --- Reconstruct signal from dominant wavelet coeffs ---
    f_inpaint = pywt.waverec(coeffs_dom, wavelet=wavelet)

    # Make sure length matches original
    f_inpaint = f_inpaint[:len(f_original)]

    # --- Re-impose known values outside the missing block ---
    f_inpaint[:D_start] = f_missing[:D_start]
    f_inpaint[D_end:]   = f_missing[D_end:]

    # --- Convergence check ---
    change = np.sqrt(np.mean((f_inpaint[D_indices] - f_original[D_indices])**2))
    if change < threshold:
        print(f"[Wavelet] Converged at iter {i+1}, change={change:.3e}, K={K}")
        break

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(x, f_original, label="Original signal")
plt.plot(x, f_missing, label="With missing region (D)", linestyle=":")
plt.plot(x, f_inpaint, label="Reconstructed iteratively Wavelet", linewidth=2, linestyle="--")
plt.axvspan(x[D_start], x[D_end], color='gray', alpha=0.2, label="Inpainting region D")
plt.legend()
plt.title(" Dominant-Frequency Based Iterative Inpainting Using Wavelet")
plt.xlabel("x")
plt.ylabel("f(x) = sin(x) + sin(8x)")
plt.grid(True)
plt.tight_layout()
plt.show()


true_values = f_original[D_indices]
reconstructed_values = f_inpaint[D_indices]
mse = np.mean((reconstructed_values - true_values) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(reconstructed_values - true_values))
print(f"MSE  = {mse:.6f}")
print(f"RMSE = {rmse:.6f}")
print(f"MAE  = {mae:.6f}")
