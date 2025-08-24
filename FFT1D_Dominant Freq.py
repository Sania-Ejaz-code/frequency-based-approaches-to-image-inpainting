import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq


# # Simulated 1D signal
# x = np.linspace(0, 10, 500)
# #f_original = np.sin(x) +  np.sin(8 * x)
# #f_original = np.sin(2 * np.pi * x / 5) + 0.5 * np.sin(2 * np.pi * x)
# # plotting of orignal signal
# # plt.figure(figsize=(10, 5))
# # plt.plot(x, f_original)
# # plt.title("Orignal Signal")
# # plt.show()
# # Missing region D
# D_start, D_end = 250, 350
# f_missing = f_original.copy()
# f_missing[D_start:D_end] = 0
# #Plot of missing region
# # plt.figure(figsize=(10, 5))
# # plt.plot(x, f_missing)
# # plt.title("Missing Region")
# # plt.show()

# # Estimate mean from neighbors U
# U_left = f_original[D_start - 80:D_start]  #Does that include the leftmost point of the gap?
# U_right = f_original[D_end:D_end + 80]   #Does that include the rightmost point of the gap?
# m = np.mean(np.concatenate([U_left, U_right]))

# # Define the inpainting region
# D_indices = np.arange(D_start, D_end)
# # Initialize inpainting with mean
# f_inpaint = f_missing.copy()
# f_inpaint[D_start:D_end] = m                 # No initialisation by the constant m?
# #left_val = f_missing[D_start - 50]
# #right_val = f_missing[D_end + 50]
# #f_inpaint[D_start:D_end] = np.linspace(left_val, right_val, D_end - D_start) # Is that initialisation by an affine linear function connecting the neighbors of the gap?


# #Plot of inpainted Signal
# # plt.figure(figsize=(10, 5))
# # plt.plot(x, f_inpaint)
# # plt.title("Inpainted Signal")
# # plt.show()

# #separate Plot of the Difference Between Inpainted and Missing Signal since they looks the same
# # plt.figure(figsize=(10, 4))
# # plt.plot(x, f_inpaint - f_missing, label="Difference (f_inpaint - f_missing)", color="red")
# # plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
# # plt.title("Difference Between Inpainted and Missing Signal")
# # plt.xlabel("x")
# # plt.ylabel("Difference")
# # plt.grid(True)
# # plt.legend()
# # plt.tight_layout()
# # plt.show()
# #Plot of FFT of inpainted Signal
# F_hat = fft(f_inpaint)

# # plt.figure(figsize=(10, 5))
# # plt.plot(x, F_hat)
# # plt.title("FFT of Inpainted Signal")
# # plt.show()
# # Iterative frequency-domain filtering
# threshold = 1e-7   # epsilon
# max_iters = 500
# k_frac = 0.06     # fraction of dominant frequencies to keep (e.g. 6%)
# eps = 1e-12

# for i in range(max_iters):
#     prev = f_inpaint.copy()

#     # --- FFT of current estimate ---
#     F_hat = fft(f_inpaint)
#     mags = np.abs(F_hat)

#     # --- Identify dominant frequency set ---
#     K = max(1, int(k_frac * len(mags)))
#     thresh = np.partition(mags, -K)[-K]
#     dominant_mask = mags >= max(thresh, eps)

#     # --- Zero out non-dominant frequencies ---
#     F_hat = F_hat * dominant_mask

#     # --- Inverse FFT to time domain ---
#     f_inpaint = np.real(ifft(F_hat))

#     # --- Re-impose known samples outside missing block ---
#     f_inpaint[:D_start] = f_missing[:D_start]
#     f_inpaint[D_end:] = f_missing[D_end:]

#     # --- Convergence check (RMSE on missing region) ---
#     change = np.sqrt(np.mean((f_inpaint[D_indices] - f_original[D_indices])**2))
#     if change < threshold:
#         print(f"Converged at iter {i+1}, change={change:.3e}")
#         break


# # Plot results
# plt.figure(figsize=(10, 5))
# plt.plot(x, f_original, label="Original signal")
# plt.plot(x, f_missing, label="With missing region (D)", linestyle=":")
# plt.plot(x, f_inpaint, label="Reconstructed iteratively FFT", linewidth=2, linestyle="--")
# plt.axvspan(x[D_start], x[D_end], color='gray', alpha=0.2, label="Inpainting region D")
# plt.legend()
# plt.title(" Dominant-Frequency Based Iterative Inpainting Using FFT")
# plt.xlabel("x")
# plt.ylabel("f(x) = sin(x) + sin(8x)")
# plt.grid(True)
# plt.tight_layout()
# plt.show()
# # To see how well our region is recontructed, we will compute errors. 



# # Extract true and reconstructed values in D
# true_values = f_original[D_indices]
# reconstructed_values = f_inpaint[D_indices]

# # Compute error metrics
# mse = np.mean((reconstructed_values - true_values) ** 2)
# rmse = np.sqrt(mse)
# mae = np.mean(np.abs(reconstructed_values - true_values))

# # Print results
# print(f"MSE  = {mse:.6f}") #Mean Squared Error
# print(f"RMSE = {rmse:.6f}") #Root Mean Squared Error
# print(f"MAE  = {mae:.6f}") #Mean Absolute Error

# --- ______________________________________________________________ ---

                    #EXAMPLE 02(More Complex Signal)
# --- ______________________________________________________________ ---


# Simulated 1D signal
x = np.linspace(0, 10, 500)
f_original = ( 0.35*np.sin(6*x + 0.6)                
   + 0.2*np.sin(14*x)                       
  + 0.15*np.cos(x*(1 + 0.25*x))           
)
# plotting of orignal signal
# plt.figure(figsize=(10, 5))
# plt.plot(x, f_original)
# plt.title("Orignal Signal")
# plt.show()
# Missing region D
D_start, D_end = 250, 350
f_missing = f_original.copy()
f_missing[D_start:D_end] = 0
#Plot of missing region
# plt.figure(figsize=(10, 5))
# plt.plot(x, f_missing)
# plt.title("Missing Region")
# plt.show()

# Estimate mean from neighbors U
U_left = f_original[D_start - 80:D_start]  #Does that include the leftmost point of the gap?
U_right = f_original[D_end:D_end + 80]   #Does that include the rightmost point of the gap?
m = np.mean(np.concatenate([U_left, U_right]))

# Define the inpainting region
D_indices = np.arange(D_start, D_end)
# Initialize inpainting with mean
f_inpaint = f_missing.copy()
f_inpaint[D_start:D_end] = m                 # No initialisation by the constant m?
#left_val = f_missing[D_start - 50]
#right_val = f_missing[D_end + 50]
#f_inpaint[D_start:D_end] = np.linspace(left_val, right_val, D_end - D_start) # Is that initialisation by an affine linear function connecting the neighbors of the gap?


#Plot of inpainted Signal
# plt.figure(figsize=(10, 5))
# plt.plot(x, f_inpaint)
# plt.title("Inpainted Signal")
# plt.show()

#separate Plot of the Difference Between Inpainted and Missing Signal since they looks the same
# plt.figure(figsize=(10, 4))
# plt.plot(x, f_inpaint - f_missing, label="Difference (f_inpaint - f_missing)", color="red")
# plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
# plt.title("Difference Between Inpainted and Missing Signal")
# plt.xlabel("x")
# plt.ylabel("Difference")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()
#Plot of FFT of inpainted Signal
F_hat = fft(f_inpaint)

# plt.figure(figsize=(10, 5))
# plt.plot(x, F_hat)
# plt.title("FFT of Inpainted Signal")
# plt.show()
# Iterative frequency-domain filtering
threshold = 1e-7   # epsilon
max_iters = 500

# --- before the loop (same as for DCT) ---
k0     = 0.0085   # start: 2% of 
k_grow = 1.091    # gentle growth factor
k_max  = 0.2 
k_frac = k0
prev_delta = None
stall_hi, stall_lo = 3e-4, 1e-4
eps = 1e-12

for i in range(max_iters):
    prev = f_inpaint.copy()

    # Real FFT of current estimate (non-redundant spectrum)
    F = np.fft.rfft(f_inpaint)
    mags = np.abs(F)

    # keep top-K bins (by magnitude)
    K   = max(1, int(k_frac * F.size))
    thr = np.partition(mags, -K)[-K]
    keep = mags >= max(thr, eps)
    F_dom = np.where(keep, F, 0.0)

    # inverse and data consistency
    g = np.fft.irfft(F_dom, n=f_inpaint.size)
    f_inpaint[:D_start] = f_missing[:D_start]
    f_inpaint[D_end:]   = f_missing[D_end:]
    f_inpaint[D_start:D_end] = g[D_start:D_end]

    # gap-only convergence
    delta = np.sqrt(np.mean((f_inpaint[D_indices] - prev[D_indices])**2))
    if delta < threshold:
        break

    # adaptive ramp
    if prev_delta is not None:
        if delta > stall_hi and k_frac < k_max:
            k_frac = min(k_max, k_frac * k_grow)
        elif delta < stall_lo:
            k_frac = min(k_max, k_frac * 1.02)
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
plt.title(" Dominant-Frequency Based Iterative Inpainting Using FFT")
plt.xlabel("x")
plt.ylabel("f(x)= (0.35sin(6x + 0.6)+ 0.2sin(14x) + 0.15cos(x + 0.25x^2)))")
plt.tight_layout()
plt.show()





