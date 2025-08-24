import numpy as np
import matplotlib.pyplot as plt
# CHANGED: use DCT/IDCT instead of FFT/IFFT
from scipy.fft import dct, idct

# --- helper: DCT-II effective frequency bins (for thresholding like fftfreq) ---
# def dctfreq(n, dx):
#     # Cosine basis for DCT-II: cos(pi*k*(n+1/2)/N)
#     # Corresponding (nonnegative) spatial frequencies are ~ k/(2*N*dx)
#     k = np.arange(n, dtype=float)
#     return k / (2.0 * n * dx)

# # Simulated 1D signal
# x = np.linspace(0, 10, 500)
# f_original = np.sin(x) +  np.sin(8 * x)

# # Missing region D
# D_start, D_end = 250, 350
# f_missing = f_original.copy()
# f_missing[D_start:D_end] = 0

# # Estimate mean from neighbors U
# U_left = f_original[D_start - 50:D_start]
# U_right = f_original[D_end:D_end + 50]
# m = np.mean(np.concatenate([U_left, U_right]))

# # Define the inpainting region
# D_indices = np.arange(D_start, D_end)

# # Initialize inpainting with mean
# f_inpaint = f_missing.copy()
# f_inpaint[D_start:D_end] = m

# # # ---- Iterative frequency-domain filtering (now with DCT) ----
# threshold = 1e-5     # epsilon
# max_iters = 400
# k_frac = 0.12       # keep top 6% DCT coefficients (tune 0.02–0.10)
# eps = 1e-12

# for i in range(max_iters):
#     prev = f_inpaint.copy()

#     # --- DCT of current estimate (Type-II, orthonormal) ---
#     C = dct(f_inpaint, type=2, norm='ortho')
#     mags = np.abs(C)

#     # --- Select dominant coefficients (top-K by magnitude) ---
#     K = max(1, int(k_frac * len(mags)))
#     thresh = np.partition(mags, -K)[-K]
#     keep_mask = mags >= max(thresh, eps)

#     # --- Zero out the rest and invert (IDCT Type-III, orthonormal) ---
#     C_dom = C * keep_mask
#     f_inpaint = idct(C_dom, type=2, norm='ortho')

#     # --- Re-impose known samples outside the missing block ---
#     f_inpaint[:D_start] = f_missing[:D_start]
#     f_inpaint[D_end:]   = f_missing[D_end:]

#     # --- Convergence check on the gap only ---
#     change = np.sqrt(np.mean((f_inpaint[D_indices] - f_original[D_indices])**2))
#     if change < threshold:
#         print(f"[DCT] Converged at iter {i+1}, change={change:.3e}, K={K}")
#         break


# # Plot results
# plt.figure(figsize=(10, 5))
# plt.plot(x, f_original, label="Original signal")
# plt.plot(x, f_missing, label="With missing region (D)", linestyle=":")
# plt.plot(x, f_inpaint, label="Reconstructed iteratively FFT", linewidth=2, linestyle="--")
# plt.axvspan(x[D_start], x[D_end], color='gray', alpha=0.2, label="Inpainting region D")
# plt.legend()
# plt.title(" Dominant-Frequency Based Iterative Inpainting Using DCT")
# plt.xlabel("x")
# plt.ylabel("f(x) = sin(x) + sin(8x)")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # (Optional) example error metrics against ground truth on D
# true_values = f_original[D_indices]
# reconstructed_values = f_inpaint[D_indices]
# mse = np.mean((reconstructed_values - true_values) ** 2)
# rmse = np.sqrt(mse)
# mae = np.mean(np.abs(reconstructed_values - true_values))
# print(f"MSE  = {mse:.6f}")
# print(f"RMSE = {rmse:.6f}")
# print(f"MAE  = {mae:.6f}")


# --- ______________________________________________________________ ---

                    #EXAMPLE 02(More Complex Signal)
# --- ______________________________________________________________ ---


def dctfreq(n, dx):
    # Cosine basis for DCT-II: cos(pi*k*(n+1/2)/N)
    # Corresponding (nonnegative) spatial frequencies are ~ k/(2*N*dx)
    k = np.arange(n, dtype=float)
    return k / (2.0 * n * dx)

# Simulated 1D signal
x = np.linspace(0, 10, 500)
f_original = ( 0.35*np.sin(6*x + 0.6)                 # mid harmonic (phase-shifted)
   + 0.2*np.sin(14*x)                       # higher harmonic
  + 0.15*np.cos(x*(1 + 0.25*x))            # mild chirp (freq increases)
)

# Missing region D
D_start, D_end = 250, 350
f_missing = f_original.copy()
f_missing[D_start:D_end] = 0

# Estimate mean from neighbors U
U_left = f_original[D_start - 50:D_start]
U_right = f_original[D_end:D_end + 50]
m = np.mean(np.concatenate([U_left, U_right]))

# Define the inpainting region
D_indices = np.arange(D_start, D_end)

# Initialize inpainting with mean
f_inpaint = f_missing.copy()
f_inpaint[D_start:D_end] = m

# # ---- Iterative frequency-domain filtering (now with DCT) ----
threshold = 1e-5     # epsilon
max_iters = 400
# --- before the loop (replace your fixed k_frac) ---
k0     = 0.0085   # start: 2% of DCT coeffs
k_grow = 1.0007    # gentle growth factor
k_max  = 0.2  # cap (works well for your complex signal)
k_frac = k0
prev_delta = None
stall_hi, stall_lo = 3e-4, 1e-4   # progress thresholds

for i in range(max_iters):
    prev = f_inpaint.copy()

    # DCT of current estimate
    C = dct(f_inpaint, type=2, norm='ortho')
    mags = np.abs(C)

    # --- keep top-K by magnitude with the ramped k_frac ---
    K = max(1, int(k_frac * C.size))
    thr = np.partition(mags, -K)[-K]
    C_dom = np.where(mags >= max(thr, 1e-12), C, 0.0)

    # inverse DCT and data consistency (same as your code)
    g = idct(C_dom, type=2, norm='ortho')
    f_inpaint[:D_start] = f_missing[:D_start]
    f_inpaint[D_end:]   = f_missing[D_end:]
    f_inpaint[D_start:D_end] = g[D_start:D_end]

    # gap-only convergence
    delta = np.sqrt(np.mean((f_inpaint[D_indices] - prev[D_indices])**2))
    if delta < threshold: break

    # --- adaptive ramp: open bandwidth only when progress slows ---
    if prev_delta is None:
        pass
    elif delta > stall_hi and k_frac < k_max:
        k_frac = min(k_max, k_frac * k_grow)       # speed up (not improving)
    elif delta < stall_lo:
        k_frac = min(k_max, k_frac * 1.02)         # tiny growth when already smooth

    prev_delta = delta

# ---- metrics on the graph (gap-only) ----
true_values = f_original[D_indices]
reconstructed_values = f_inpaint[D_indices]
mse  = np.mean((reconstructed_values - true_values)**2)
rmse = np.sqrt(mse)
mae  = np.mean(np.abs(reconstructed_values - true_values))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x, f_original, label="Original signal")
ax.plot(x, f_missing, linestyle=":", label="With missing region (D)")
ax.plot(x, f_inpaint, linestyle="--", linewidth=2, label="Reconstructed")
ax.axvspan(x[D_start], x[D_end], color="gray", alpha=0.2, label="Inpainting region D")
ax.legend(loc='lower left')
ax.set_xlabel("x"); ax.set_ylabel("f(x)")

# Add a neat metrics box (computed on D only)
txt = f"MSE = {mse:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}"
ax.text(
    0.02, 0.98, txt,
    transform=ax.transAxes, ha="left", va="top",
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.6", alpha=0.85)
)
plt.title(" Dominant-Frequency Based Iterative Inpainting Using DCT")
plt.xlabel("x")
plt.ylabel("f(x)= (0.35sin(6x + 0.6)+ 0.2sin(14x) + 0.15cos(x + 0.25x^2)))")
plt.tight_layout()
plt.show()
