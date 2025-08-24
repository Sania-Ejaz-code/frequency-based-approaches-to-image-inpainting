import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgba2rgb, rgb2gray
from skimage.util import img_as_float
from skimage.transform import resize
from skimage.metrics import structural_similarity as ssim
from skimage.draw import line
# CHANGED: use Wavelet instead of FFT
import pywt

# === Load grayscale image ===
def load_image(image_path):
    image = imread(image_path)
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = rgba2rgb(image)
        image = rgb2gray(image)
    return img_as_float(image)

#====== Type of Masks=====
def random_mask(shape, missing_fraction=0.3, seed=None):
    """True = missing."""
    rng = np.random.default_rng(seed)
    return (rng.random(shape) < missing_fraction)

def block_mask(shape, block_size=(50, 50), seed=None):
    """Single rectangular hole; True = missing."""
    rng = np.random.default_rng(seed)
    h, w = shape
    bh, bw = block_size
    bh = min(bh, h)
    bw = min(bw, w)
    y = rng.integers(0, h - bh + 1)
    x = rng.integers(0, w - bw + 1)
    m = np.zeros(shape, dtype=bool)
    m[y:y+bh, x:x+bw] = True
    return m

def block_mask_fraction(shape, fraction=0.5, seed=None):
    """
    One rectangular hole covering ~fraction of the image area.
    True = missing region.
    """
    rng = np.random.default_rng(seed)
    h, w = shape

    # target area in pixels
    target_area = int(fraction * h * w)

    # randomly pick width and height with area close to target
    aspect = rng.uniform(0.5, 2.0)  # random aspect ratio (w/h)
    bh = int(np.sqrt(target_area / aspect))
    bw = int(aspect * bh)

    # clip to fit inside image
    bh = min(bh, h)
    bw = min(bw, w)

    # random position
    y = rng.integers(0, h - bh + 1)
    x = rng.integers(0, w - bw + 1)

    mask = np.zeros(shape, dtype=bool)
    mask[y:y+bh, x:x+bw] = True
    return mask

def irregular_mask(shape, num_strokes=10, thickness_range=(5, 20), seed=None):
    """
    Free-form ‘scribble’ mask; True = missing.
    Uses random lines thickened by a square brush.
    """
    rng = np.random.default_rng(seed)
    h, w = shape
    m = np.zeros(shape, dtype=bool)

    for _ in range(num_strokes):
        x1, y1 = rng.integers(0, w), rng.integers(0, h)
        # random length and angle
        L = rng.integers(max(8, min(h, w)//10), max(12, min(h, w)//3))
        theta = rng.random() * 2 * np.pi
        x2 = int(np.clip(x1 + L * np.cos(theta), 0, w-1))
        y2 = int(np.clip(y1 + L * np.sin(theta), 0, h-1))
        rr, cc = line(y1, x1, y2, x2)

        t = int(rng.integers(thickness_range[0], thickness_range[1] + 1))
        rpad = t // 2
        for dy in range(-rpad, rpad + 1):
            rrs = np.clip(rr + dy, 0, h - 1)
            for dx in range(-rpad, rpad + 1):
                ccs = np.clip(cc + dx, 0, w - 1)
                m[rrs, ccs] = True
    return m

# === Load binary mask ===
def load_mask(mask_path, shape):
    """
    Returns boolean mask with True = missing.
    Supports:
      - 'random:p'                  e.g. 'random:0.3'
      - 'block:HxW'                 e.g. 'block:80x120'
      - 'blockfrac:p'               e.g. 'blockfrac:0.5'  (50% coverage rectangle)
      - 'irregular:strokes=..'      e.g. 'irregular:strokes=15,t=6-18,seed=42'
    Otherwise, treats mask_path as an image file path.
    """
    if isinstance(mask_path, str):
        key = mask_path.lower()

        # === Random mask (pixel fraction) ===
        if key.startswith("random:"):
            p = float(key.split(":")[1])
            return random_mask(shape, missing_fraction=p)

        # === Block mask (explicit dimensions) ===
        if key.startswith("block:"):
            dims = key.split(":")[1].replace(" ", "")
            h_str, w_str = dims.split("x")
            return block_mask(shape, (int(h_str), int(w_str)))

        # === Block mask by area fraction ===
        if key.startswith("blockfrac:"):
            frac = float(key.split(":")[1])
            return block_mask_fraction(shape, fraction=frac)

        # === Irregular mask (scribbles) ===
        if key.startswith("irregular:"):
            params = key.split(":", 1)[1]
            parts = {}
            for frag in params.split(","):
                if "=" in frag:
                    k, v = frag.split("=")
                    parts[k.strip()] = v.strip()

            strokes = int(parts.get("strokes", 12))
            if "t" in parts:
                lo, hi = parts["t"].split("-")
                t_range = (int(lo), int(hi))
            else:
                t_range = (5, 20)
            seed = int(parts["seed"]) if "seed" in parts else None
            return irregular_mask(shape, num_strokes=strokes,
                                  thickness_range=t_range, seed=seed)

    # === Fallback: load binary mask from file ===
    mask = imread(mask_path, as_gray=True)
    mask = resize(mask, shape, order=0, preserve_range=True, anti_aliasing=False)
    return (mask < 0.5).astype(bool)   # True = missing


# === Wavelet-based inpainting with per-subband top-K ===
def fft_inpaint(image, mask, num_iters=400, k_frac=0.07, eps=1e-12, use_window=True,
                wavelet='db2', level=None, wave_mode='symmetric', keep_approx=True,
                per_subband=True, band_alloc='energy'):
    """
    Gerchberg–Papoulis style inpainting in wavelet domain.

    per_subband : if True, pick top-K *within each wavelet detail subband*
                  (rather than one global top-K over all coefficients).
    band_alloc  : 'energy' -> allocate K to bands proportional to their energy
                  'equal'  -> allocate K equally across detail bands
    """

    img = image.astype(float)
    M = (mask > 0).astype(np.uint8)   # 1 = hole, 0 = known
    known = 1 - M

    # Optional window (usually False for periodic textures)
    if use_window:
        wy = np.hanning(img.shape[0])[:, None]
        wx = np.hanning(img.shape[1])[None, :]
        W = wy * wx
        W[W < W.max() * 1e-3] = W.max() * 1e-3
    else:
        W = np.ones_like(img)

    # Initial guess
    mean_known = img[known == 1].mean() if np.any(known) else 0.0
    x = img.copy()
    x[M == 1] = mean_known

    H, Wd = img.shape
    K_total = max(1, int(k_frac * H * Wd))  # overall coefficient budget

    wave = pywt.Wavelet(wavelet)

    for _ in range(num_iters):
        # (1) Forward 2D DWT
        coeffs = pywt.wavedec2(x * W, wavelet=wave, level=level, mode=wave_mode)

        if not per_subband:
            # --- Global top-K (your previous behaviour) ---
            arr, coeff_slices = pywt.coeffs_to_array(coeffs)
            if keep_approx:
                # protect approx from thresholding
                approx_slice = coeff_slices[0]
                approx_mask = np.zeros_like(arr, dtype=bool)
                ay, ax = approx_slice
                approx_mask[ay, ax] = True
                mags = np.abs(arr)
                mags[approx_mask] = -np.inf
                flat_idx = np.argpartition(mags.ravel(), -K_total)[-K_total:]
                keep_mask = np.zeros_like(arr, dtype=bool)
                keep_mask.ravel()[flat_idx] = True
                keep_mask[approx_mask] = True
            else:
                mags = np.abs(arr).ravel()
                flat_idx = np.argpartition(mags, -K_total)[-K_total:]
                keep_mask = np.zeros_like(arr, dtype=bool)
                keep_mask.ravel()[flat_idx] = True

            arr_kept = np.where(keep_mask, arr, 0.0)
            coeffs_kept = pywt.array_to_coeffs(arr_kept, coeff_slices, output_format='wavedec2')

        else:
            # --- Per-subband top-K ---
            # coeffs structure: [cA, (cH1,cV1,cD1), (cH2,cV2,cD2), ...]
            cA, *details = coeffs

            # 1) Count bands and their energies (LH/HL/HH at each level)
            bands = []  # list of dicts with refs
            energies = []
            for lvl, (cH, cV, cD) in enumerate(details, start=1):
                for name, c in (('H', cH), ('V', cV), ('D', cD)):
                    bands.append({'lvl': lvl, 'name': name, 'coef': c})
                    energies.append(np.sum(c**2))

            energies = np.array(energies, dtype=float)
            n_bands = len(bands)

            # 2) Allocate K per band
            if band_alloc == 'energy' and energies.sum() > 0:
                alloc = (energies / energies.sum()) * K_total
            else:  # 'equal' or zero energy fallback
                alloc = np.full(n_bands, K_total / max(n_bands, 1.0))

            # 3) Threshold per band
            kept_details = []
            idx = 0
            for (cH, cV, cD) in details:
                kept_triplet = []
                for c in (cH, cV, cD):
                    Kb = int(max(1, np.floor(alloc[idx])))
                    idx += 1
                    if Kb >= c.size:
                        c_keep = c
                    else:
                        mags = np.abs(c).ravel()
                        thresh = np.partition(mags, -Kb)[-Kb]
                        mask_keep = (np.abs(c) >= max(thresh, eps))
                        c_keep = np.where(mask_keep, c, 0.0)
                    kept_triplet.append(c_keep)
                kept_details.append(tuple(kept_triplet))

            # 4) Keep or discard the approximation
            cA_keep = cA if keep_approx else np.zeros_like(cA)

            coeffs_kept = [cA_keep] + kept_details

        # (2) Inverse 2D DWT
        x_recon = pywt.waverec2(coeffs_kept, wavelet=wave, mode=wave_mode)

        # shape guard (can differ by 1 px occasionally)
        if x_recon.shape != img.shape:
            x_recon = x_recon[:img.shape[0], :img.shape[1]]

        # (3) Undo window (approx) and project onto known pixels
        x_recon = x_recon / (W + 1e-8)
        x[M == 1] = x_recon[M == 1]
        x[known == 1] = img[known == 1]

    return np.clip(x, 0.0, 1.0)


# === Image and Mask File paths ===

image_path = "/Users/saniaejaz/Desktop/FFT Thesis/Images/P_Zigzag Horiz.png"  # Replace with  own image path

# A) 30% random pixels missing
#mask_path = "random:0.30"

# B) One 80x120 rectangular hole at random location
#mask_path = "block:150x150"

# C) Rectangle covers ~50% of the image
mask_path = "blockfrac:0.1"

# D) Free-form irregular mask with 15 strokes, thickness 6–18, fixed seed
# mask_path = "irregular:strokes=15,t=6-18,seed=42"

# === Load data ===
image = load_image(image_path)
mask = load_mask(mask_path, image.shape)

# === Apply inpainting ===
#image_inpainted = fft_inpaint(image, mask, num_iters=5000, k_frac=0.01, use_window=False)....   for PCir

# Tight, periodic zig-zag (chevron)
#image_inpainted = fft_inpaint(image, mask, num_iters=1000, k_frac=0.0010, use_window=True)

#Diamond shape Pattern
#image_inpainted = fft_inpaint(image, mask, num_iters=370, k_frac=0.00014, use_window=False)

#verticl bars
#image_inpainted = Wavelet_inpaint(image, mask, num_iters=300, k_frac=0.00001, use_window=False,  wavelet='sym4')
# Sharpest edges
image_inpainted = fft_inpaint(
    image, mask,
    num_iters=100, k_frac=0.001 , wavelet='sym8', level=4)

# # Slightly smoother
# image_inpainted = fft_inpaint(
#     image, mask,
#     num_iters=600, k_frac=0.03,
#     use_window=False, wavelet='db2', level=3,
#     wave_mode='periodization', keep_approx=True,
#     per_subband=True, band_alloc='energy'
# )

# === Evaluate ===
error = np.abs(image - image_inpainted)
mse = np.mean((image - image_inpainted) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(image - image_inpainted))
ssim_index = ssim(image, image_inpainted, data_range=1.0)

# === Visualize ===
fig, axs = plt.subplots(1, 3, figsize=(9, 3))
axs[0].imshow(image, cmap='gray')
axs[0].set_title("Original")
axs[0].axis('off')

axs[1].imshow(np.where(mask, 0, image), cmap='gray')
axs[1].set_title("With Masked Region")
axs[1].axis('off')

axs[2].imshow(image_inpainted, cmap='gray')
axs[2].set_title("Inpainted image")
axs[2].axis('off')

plt.suptitle(f"MSE = {mse:.6f}, RMSE = {rmse:.6f}, MAE = {mae:.6f}, SSIM = {ssim_index:.4f}")
plt.tight_layout()
plt.show()
