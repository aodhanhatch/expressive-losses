# CIFAR-10, eps = 2/255: reproduction of Table 1 (De Palma et al., ICLR 2024)

Setup: README training commands unchanged except for a dedicated output folder per loss.
Verification: README command with `--ib_batch_size 500`, `bab_configs/cnn7_naive_wsl.json`
(same as `cnn7_naive.json` but `max_cpu_subdomains` 5e3 -> 1e3 to fit a 16 GB WSL2 VM),
1800 s BaB timeout, evaluated on the first 1000 test images (paper: all 10,000).
Hardware: RTX 5070 Ti Laptop (12 GB) under WSL2. Training ~5 h per model.

Numbers below are from `python reproduction/summarize_ver.py <ver_folder> 1000`.

| Loss    | Std. acc. (ours) | Std. acc. (paper) | PGD acc. (ours) | Ver. CROWN only (ours) | Ver. BaB (ours) | Ver. BaB (paper) |
|---------|-----------------:|------------------:|----------------:|-----------------------:|----------------:|-----------------:|
| CC-IBP  | 81.00 | 80.09 | 70.90 | 53.50 | 63.70 | 63.78 |
| MTL-IBP | 80.30 | 80.11 | 70.30 | 51.90 | 63.80 | 63.24 |
| Exp-IBP | (pending) | 80.61 | | | | 61.65 |

IBP-only verified accuracy is 0.00% for both models at this radius.
Standard error on a 1000-image subset is about 1.5 points for rates near 60-80%.

Result folders (one pickle per image, fields ok / pgd_ok / ibp_ok / crown_ok / ver_ok):
- `model_cifar_ccibp/cnn_fast_1790529581_ckpt_last_ver/`
- `model_cifar_mtlibp/cnn_fast_1790709178_ckpt_last_ver/`
Checkpoints are not in git (200 MB each); backed up outside the repo.
