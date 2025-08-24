import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct, idct

# --- ______________________________________________________________ ---

                    #EXAMPLE 01(simple Signal)
# --- ______________________________________________________________ ---
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
# U_left = f_original[D_start - 80:D_start]
# U_right = f_original[D_end:D_end + 80]
# m = np.mean(np.concatenate([U_left, U_right]))

# # Define the inpainting region
# D_indices = np.arange(D_start, D_end)

# # Initialize inpainting with mean
# f_inpaint = f_missing.copy()
# f_inpaint[D_start:D_end] = m

# ---- Iterative frequency-domain filtering (now with DCT) ----
# threshold = 1e-5       # epsilon
# max_iters = 400
# r = 1.002                # initial cutoff (same semantics as before)
# growth_factor = 1.00013    # keep your original schedule (1.0 == constant r)
# max_r = 100

# dx = x[1] - x[0]
# N = len(x)
# # Precompute the DCT "frequency" grid (nonnegative only)
# freqs = dctfreq(N, dx)

# for i in range(max_iters):
#     prev = f_inpaint.copy()

#     # CHANGED: forward transform -> DCT-II (orthonormal)
#     F_hat = dct(f_inpaint, type=2, norm="ortho")

#     # CHANGED: low-pass in DCT domain using the same cutoff variable r
#     # (zero-out cosine modes whose effective frequency exceeds r)
#     F_hat[freqs > r] = 0.0

#     # CHANGED: inverse transform -> IDCT-II (orthonormal)
#     f_inpaint = idct(F_hat, type=2, norm="ortho")

#     # Preserve known values outside D
#     f_inpaint[:D_start] = f_missing[:D_start]
#     f_inpaint[D_end:]   = f_missing[D_end:]

#     # Change on D (your original relative change idea kept intact)
#     change = np.sqrt(np.mean((f_inpaint[D_indices] - prev[D_indices]) ** 2))
#     if change < threshold:
#         break

#     r = min(r * growth_factor, max_r)

# # Plot results
# plt.figure(figsize=(10, 5))
# plt.plot(x, f_original, label="Original signal")
# plt.plot(x, f_missing, label="With missing region (D)", linestyle=":")
# plt.plot(x, f_inpaint, label="Reconstructed iteratively using FFT", linewidth=2, linestyle="--")
# plt.axvspan(x[D_start], x[D_end], color='gray', alpha=0.2, label="Inpainting region D")
# plt.legend()
# plt.title("Low-Pass Cutoff Filtering Iterative Inpainting using DCT")
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
f_original = ( 0.35*np.sin(6*x + 0.6)                 
   + 0.2*np.sin(14*x)                        
  + 0.15*np.cos(x*(1 + 0.25*x))             
)

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

# --- 2) normalized DCT "frequency" grid (fraction of Nyquist) ---
threshold = 1e-5       # epsilon
max_iters = 400
dx = x[1] - x[0]
N  = len(x)
fn = (np.arange(N) / (2*N*dx))                 # DCT-II frequencies (Hz)
fnyq = 0.5 / dx                                # Nyquist (Hz)
fnorm = fn / fnyq                              # 0..1

# --- 3) adaptive cutoff ramp (fraction of Nyquist) ---
r, r_grow, r_max = 0.04, 1.02, 1           # start 8% -> cap 40%
stall_hi, stall_lo = 3e-4, 1e-4                # progress thresholds

for i in range(max_iters):
    prev = f_inpaint.copy()

    # DCT-II (orthonormal)
    C = dct(f_inpaint, type=2, norm='ortho')

    # keep passband (fnorm <= r); soft-shrink to tame ringing
    passband = (fnorm <= r)
    C_lp = np.zeros_like(C)
    if np.any(passband):
        s = 0.012 * np.std(C[passband])        # 1.2% of std (very gentle)
        P = C[passband]
        C_lp[passband] = np.sign(P) * np.maximum(np.abs(P) - s, 0.0)

    # inverse and data consistency + edge pinning
    g = idct(C_lp, type=2, norm='ortho')
    f_inpaint[:D_start] = f_missing[:D_start]
    f_inpaint[D_end:]   = f_missing[D_end:]
    f_inpaint[D_start:D_end] = g[D_start:D_end]
    f_inpaint[D_start]  = f_missing[D_start-1]    # pin values
    f_inpaint[D_end-1]  = f_missing[D_end]

    # gap-only convergence and adaptive widening
    delta = np.sqrt(np.mean((f_inpaint[D_indices] - prev[D_indices])**2))
    if delta < threshold:
        break
    if delta > stall_hi and r < r_max:
        r = min(r_max, r * r_grow)              # widen when progress stalls
    elif delta < stall_lo:
        r = min(r_max, r * 1.02)                # tiny growth when already smooth

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
plt.title(" Low-Pass Cutoff Filtering Iterative Inpainting using DCT")
plt.xlabel("x")
plt.ylabel("f(x)= (0.35sin(6x + 0.6)+ 0.2sin(14x) + 0.15cos(x + 0.25x^2)))")
plt.tight_layout()
plt.show()

