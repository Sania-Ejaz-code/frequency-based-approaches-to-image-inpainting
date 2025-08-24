## Description

This repository contains the code and figures for my thesis on **frequency-based image inpainting**.  
I explored two approaches for **1D signal reconstruction**:

1) **Low-pass spectral cutoff** — iteratively filter the spectrum and re-impose known samples.  
2) **Dominant-frequency iterative inpainting** — keep a small, adaptive set of dominant coefficients in the transform domain, invert, and enforce data consistency (repeat).

Based on the 1D results, I adopted **Method 2 (dominant-frequency)** for the **2D image inpainting** experiments.  
The 2D pipeline is evaluated under three transforms:

- **FFT / DFT** (global periodic structure)  
- **DCT** (strong local energy compaction)  
- **DWT (wavelets)** using multiple bases (e.g., `sym8`, `coif5`, `bior6.8`) for multi-scale localization

The repo includes periodic and non-periodic image examples, square and irregular masks across multiple occlusion levels, and quantitative metrics (MSE, RMSE, MAE, SSIM, PSNR) with summary plots.
