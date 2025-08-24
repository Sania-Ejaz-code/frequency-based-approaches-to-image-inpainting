import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq


# # Simulated 1D signal
# x = np.linspace(0, 10, 500)
# #f_original = np.sin(x) +  np.sin(8 * x)
# f_original = (
#     0.6*np.sin(x)                          # low-freq baseline
#   + 0.35*np.sin(6*x + 0.6)                 # mid harmonic (phase-shifted)
#   + 0.2*np.sin(14*x)                       # higher harmonic
#   + 0.15*np.sin(x*(1 + 0.25*x))            # mild chirp (freq increases)
#   + 0.08*(x-5)/5                           # slow linear trend
#   + 0.05*np.sin(40*x)*(0.5 + 0.5*np.cos(0.2*x))  # AM high-freq
# )


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
# U_left = f_missing[D_start - 50:D_start]  #Does that include the leftmost point of the gap?
# U_right = f_missing[D_end:D_end + 50]   #Does that include the rightmost point of the gap?
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
# threshold = 0.00001  # epsilon
# max_iters = 400
# r = 1.6          # initial frequency cutoff (start small)
# growth_factor = 1.0001 # increase rate for r, >1 only
# max_r = 100    # maximum frequency cutoff
# for i in range(max_iters):
#     prev = f_inpaint.copy()
    
#     # Apply FFT and low-pass filter
#     F_hat = fft(f_inpaint)
#     freqs = fftfreq(len(x), d=(x[1] - x[0]))
#     F_hat[np.abs(freqs) > r] = 0
#     f_inpaint = np.real(ifft(F_hat))

#     # Preserve known values outside D
#     f_inpaint[:D_start] = f_missing[:D_start]
#     f_inpaint[D_end:] = f_missing[D_end:]  
#     # Compute change
#     change = np.sqrt(np.mean((f_inpaint[D_indices] - f_original[D_indices]) ** 2))  #Just out of curiosity: Is that the Euclidean norm?
#       #Just out of curiosity: Is that the Euclidean norm?
#     if change < threshold:
#         break
#     #r *= decay  # gradually reduce frequency cutoff
    
#     r = min(r * growth_factor, max_r) # Gradually increase the cutoff frequency, up to a maximum

# # Plot results
# plt.figure(figsize=(10, 5))
# plt.plot(x, f_original, label="Original signal")
# plt.plot(x, f_missing, label="With missing region (D)", linestyle=":")
# plt.plot(x, f_inpaint, label="Reconstructed iteratively using FFT", linewidth=2, linestyle="--")
# plt.axvspan(x[D_start], x[D_end], color='gray', alpha=0.2, label="Inpainting region D")
# plt.legend()
# plt.title("Low-Pass Cutoff Filtering Iterative Inpainting using FFT")
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


x = np.linspace(0, 10, 500)
f_original = ( 0.35*np.sin(6*x + 0.6)                 
   + 0.2*np.sin(14*x)                        
  + 0.15*np.cos(x*(1 + 0.25*x))             
)

#f_original = np.sin(2 * np.pi * x / 5) + 0.5 * np.sin(2 * np.pi * x)
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
U_left = f_missing[D_start - 50:D_start]  #Does that include the leftmost point of the gap?
U_right = f_missing[D_end:D_end + 50]   #Does that include the rightmost point of the gap?
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
threshold = 0.00001  # epsilon
max_iters = 400


# normalized frequency axis (0..1, where 1 = Nyquist)
fnorm = np.fft.rfftfreq(f_inpaint.size, d=x[1]-x[0])
fnorm /= fnorm.max() if fnorm.max() else 1.0

# cutoff schedule + smooth transition
r, r_grow, r_max = 0.04, 1.007, 1 
# replace fixed tw
tw = float(np.clip(0.12*r, 0.04, 0.10))

prev_delta = None
stall_hi, stall_lo = 3e-4, 1e-4

for it in range(max_iters):
    prev_gap = f_inpaint[D_start:D_end].copy()

    # FFT and smooth low-pass mask (raised-cosine transition)
    F = np.fft.rfft(f_inpaint)
    # mask: 1 in passband, 0 in stopband, cosine roll-off in [r, r+tw]
    m = np.ones_like(F, dtype=float)
    band = (fnorm > r) & (fnorm < min(1.0, r + tw))
    m[fnorm >= r + tw] = 0.0
    if band.any():
        t = (fnorm[band] - r) / tw
        m[band] = 0.5 * (1 + np.cos(np.pi * t))   # smooth taper

    F_lp = F * m
    g = np.fft.irfft(F_lp, n=f_inpaint.size)

    # data consistency + edge pinning
    f_inpaint[:D_start] = f_missing[:D_start]
    f_inpaint[D_end:]   = f_missing[D_end:]
    f_inpaint[D_start:D_end] = g[D_start:D_end]
    f_inpaint[D_start]  = f_missing[D_start-1]    # pin values at the gap edges
    f_inpaint[D_end-1]  = f_missing[D_end]

    # convergence on the gap (no ground-truth peeking)
    delta = np.sqrt(np.mean((f_inpaint[D_start:D_end] - prev_gap)**2))
    if delta < threshold:
        print(f"Converged @ iter {it+1} | Δ={delta:.2e} | r={r:.3f}")
        break

       
    if prev_delta is not None:
        if delta > stall_hi and r < r_max:
            r = min(r_max, r * r_grow)
        elif delta < stall_lo:
            r = min(r_max, r * 1.02)  # tiny growth when already smooth
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
plt.title(" Low-Pass Cutoff Filtering Iterative Inpainting using FFT")
plt.xlabel("x")
plt.ylabel("f(x)= (0.35sin(6x + 0.6)+ 0.2sin(14x) + 0.15cos(x + 0.25x^2)))")
plt.tight_layout()
plt.show()




