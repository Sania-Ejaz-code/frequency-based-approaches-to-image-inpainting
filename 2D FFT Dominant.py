import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgba2rgb, rgb2gray
from skimage.util import img_as_float
from skimage.transform import resize
from skimage.metrics import structural_similarity as ssim
from skimage.draw import line
from scipy.fft import fft2, ifft2, fftshift, ifftshift


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


# === FFT-based inpainting with dominant frequency preservation ===
#def fft_inpaint(image, mask, num_iters=200, preserve_ratio=1):
#     image_missing = image.copy()
#     image_missing[mask] = 0

#     image_filled = image_missing.copy()
#     image_filled[mask] = np.mean(image[~mask])  # initial guess for missing region

#     # Step 1: Compute full FFT of the image (guidance)
#     #F_image = fft2(image)
#     #magnitude = np.abs(F_image)
#     F_maskedimage = fft2(image_missing)
#     magnitude = np.abs(F_maskedimage)
#     # Step 2: Identify strongest frequency components
#     threshold = np.quantile(magnitude, 1 - preserve_ratio)
#     #threshold = 10000
#     dominant_freqs = magnitude > threshold

#     # Step 3: Defining a filter
#     M, N = image.shape
#     r = 0.00000005 # initial frequency cutoff (start small)
#     y = np.fft.fftfreq(M).reshape(-1, 1)
#     x = np.fft.fftfreq(N).reshape(1, -1)
#     radius = np.sqrt(x**2 + y**2)
#     #trying multiple filters one by one
#     gaussian_filter = np.exp(-(radius**2))#/ (2 * (r / max(M, N))**2))
#     ideal_filter = (radius <= (r / max(M, N))).astype(float)
#     n = 2  # Order of the filter
#     butterworth_filter = 1 / (1 + (radius / (r / max(M, N)))**(2 * n))
#     # Characteristic filter: set all frequencies > r to 0
#     characteristic_filter = (radius <= r).astype(float)

#     # Step 4: Iterative inpainting using frequency constraint and applying filter
#     for i in range(num_iters):
#         Ffill = fft2(image_filled)
#         F = Ffill * gaussian_filter
#         F[dominant_freqs] = F_maskedimage[dominant_freqs]  # inject known frequencies
#         image_recon = np.real(ifft2(F))
#         image_filled[mask] = image_recon[mask]  # update only missing region

#     return image_filled


def fft_inpaint(image, mask, num_iters=400, k_frac=0.07, eps=1e-12, use_window=True):
    """
    Gerchberg–Papoulis inpainting with dominant-frequency constraint.
    image: float64 in [0,1]
    mask:  1 = missing, 0 = known
    num_iters: iterations of projections
    k_frac: fraction of largest-magnitude Fourier coefficients to keep (0.02–0.10 works well for periodic textures)
    """

    img = image.astype(float)
    M = (mask > 0).astype(np.uint8)        # 1 = hole, 0 = known
    known = 1 - M

    # Optional window to reduce ringing from sharp mask edges
    if use_window:
        wy = np.hanning(img.shape[0])[:, None]
        wx = np.hanning(img.shape[1])[None, :]
        W = wy * wx
        W[W < W.max() * 1e-3] = W.max() * 1e-3  # avoid zeros
    else:
        W = np.ones_like(img)

    # Initial guess: fill hole with mean of known pixels
    mean_known = img[known == 1].mean() if np.any(known) else 0.0
    x = img.copy()
    x[M == 1] = mean_known

    # How many Fourier coefficients to keep
    H, Wd = img.shape
    K = max(1, int(k_frac * H * Wd))

    # Iterate: (A) go to FFT, keep top-K magnitudes, (B) return to space and enforce known pixels
    for _ in range(num_iters):
        X = fft2(x * W)                      # windowed FFT to tame boundaries
        mag = np.abs(X).ravel()
        if K < mag.size:
            thresh = np.partition(mag, -K)[-K]
            keep = np.abs(X) >= max(thresh, eps)
            X = X * keep
        x_recon = np.real(ifft2(X)) / (W + 1e-8)   # undo window (approx)
        # Projection: keep known pixels from the original image
        x[M == 1] = x_recon[M == 1]
        x[known == 1] = img[known == 1]

    return np.clip(x, 0.0, 1.0)

# === Image and Mask File paths ===

image_path = "/Users/saniaejaz/Desktop/FFT Thesis/Images/Tile.png"  # Replace with  own image path

# A) 30% random pixels missing
#mask_path = "random:0.30"

# B) One 80x120 rectangular hole at random location
#mask_path = "block:500x500"

# C) Rectangle covers ~50% of the image
mask_path = "blockfrac:0.10"

# D) Free-form irregular mask with 15 strokes, thickness 6–18, fixed seed
#mask_path = "irregular:strokes=30,t=50-120,seed=42"

# === Load data ===
image = load_image(image_path)
mask = load_mask(mask_path, image.shape)

# === Apply inpainting ===
#image_inpainted = fft_inpaint(image, mask, num_iters=500, k_frac=0.01, use_window=False) #....   for PCir

# Tight, periodic zig-zag (chevron)
image_inpainted = fft_inpaint(image, mask, num_iters=10, k_frac=0.0001, use_window=False)

#Diamond shape Pattern
#image_inpainted = fft_inpaint(image, mask, num_iters=370, k_frac=0.00014, use_window=False)

#verticl bars
#image_inpainted = fft_inpaint(image,    mask, num_iters=300, k_frac=0.01 , use_window= False)



# === Evaluate ===
error = np.abs(image - image_inpainted)
mse = np.mean((image - image_inpainted) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(image - image_inpainted))
ssim_index = ssim(image, image_inpainted, data_range=1.0)
psnr = 10 * np.log10(1.0 / mse) if mse > 0 else np.inf

# === Visualize ===
fig, axs = plt.subplots(1, 3, figsize=(10, 5))
axs[0].imshow(image, cmap='gray')
axs[0].set_title("Original")
axs[0].axis('off')

axs[1].imshow(np.where(mask, 0, image), cmap='gray')
axs[1].set_title("With irregular Masked Region")
axs[1].axis('off')

axs[2].imshow(image_inpainted, cmap='gray')
axs[2].set_title("Inpainted image")
axs[2].axis('off')

plt.suptitle(f"MSE = {mse:.6f}, RMSE = {rmse:.6f}, MAE = {mae:.6f}, SSIM = {ssim_index:.4f}, PSNR= {psnr:.4f}")
plt.tight_layout()
plt.show()



