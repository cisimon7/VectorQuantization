# Reconstruction evaluation

Split: **test** · Images: **802**

| Model | Step | L1 ↓ | MSE ↓ | PSNR dB ↑ | SSIM ↑ | LPIPS-Alex ↓ | rFID ↓ | KID ×1000 ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VQVAE | 10000 | **0.126362** | **0.033447** | **21.237956** | **0.531612** | **0.367521** | **108.702499** | **84.6031 ± 6.9888** |
| VQGAN | 10000 | 0.219587 | 0.086592 | 16.805020 | 0.266961 | 0.438145 | 203.350861 | 191.5519 ± 6.3526 |

## Protocol

- L1, MSE, and PSNR use unclipped model outputs; they measure exact reconstruction fidelity.
- SSIM and LPIPS use paired images clipped to `[-1, 1]`. LPIPS uses frozen pretrained AlexNet features.
- rFID and KID compare clipped reconstruction and real-image distributions using frozen Inception-v3 features.
- KID uses 50 subsets of 100 images with seed 0.
- FID is sample-size biased; with this small test split, interpret rFID together with KID and paired LPIPS.
- Bold values are best within this report. Lower is better except for PSNR and SSIM.

## Checkpoints

- **VQVAE**: `/media/cisimon7/USB/datasets/oxford_flowers102/checkpoints/flowers-vqvae.eqx`
- **VQGAN**: `/media/cisimon7/USB/datasets/oxford_flowers102/checkpoints/flowers-vqgan.eqx`
