Some random estuff to note while trying to reproduce:
auto_LiRPA pinned to release 0.5.0, commit bfb7997, and the reason. If you later want the pin to survive a fresh clone, the place to record it is your fork's README setup section.

Some key definitions for the experiment:
nat_loss = Average cross-entropy on unperturbed images, not the total CC-IBP training objective
nat_ok = Fraction of correctly classified on unperturbed images
adv_ok = Fraction correctly classified on the PGD-generated inputs
ver_ok = Frcation passing the IBP margin-bound check
eps = current perturbation size, 


On the smoke test, we ran 4 epochs on the 2/255 CIFAR 10 test,
The results we got seems resonable, we have that the natural accuracy and the advererial accuracy is the same for the first epoch where the eps valkue is 0
Also even though this was a smoke test with only for 4 epochs, we can see how the natural accuracy is falling as the peterubutation size


After training the CC-IBP model with default setting, on the CIFAR-10 dataset, we managed to run the verification up to a 1000 images (about 5-6 hour run time on the RTX 5070Ti). 

**COMPARE WITH THE PAPER when have a mom

## Results: CIFAR-10, eps = 2/255, ours vs the paper

Setup. Trained with the README commands (160 epochs, one run per loss, default seed). Verified with
verify.py: PGD attack, then IBP and CROWN, then OVAL branch-and-bound (alpha-beta-CROWN, 1800 s timeout).
Ours is the first 1000 test images. The paper is the full 10,000-image test set.
Standard error on a 1000-image subset is about 1.5 points, so differences below ~2 points are noise.

### Main comparison (paper Table 1)

| Loss    | Standard acc. ours | Standard acc. paper | Verified acc. ours | Verified acc. paper |
|---------|-------------------:|--------------------:|-------------------:|--------------------:|
| CC-IBP  |               81.00|                80.09|              63.70 |               63.78 |
| MTL-IBP |               80.30|                80.11|              63.80 |               63.24 |
| Exp-IBP |               79.90|                80.61|              62.90 |               61.65 |

### Verified accuracy by verifier (paper Table 6)

CROWN/IBP means the better of the CROWN and IBP bounds, no branch-and-bound. This is the crown_ok field in our result files.

| Loss    | BaB ours | BaB paper | CROWN/IBP ours | CROWN/IBP paper | IBP ours | IBP paper |
|---------|---------:|----------:|---------------:|----------------:|---------:|----------:|
| CC-IBP  |    63.70 |     63.78 |          53.50 |           52.37 |     0.00 |      0.00 |
| MTL-IBP |    63.80 |     63.24 |          51.90 |           51.35 |     0.00 |      0.00 |
| Exp-IBP |    62.90 |     61.65 |          51.80 |           47.87 |     0.00 |      0.00 |

### Adversarial (PGD) accuracy, ours only

The paper does not report PGD accuracy in Table 1. PGD here is 40 steps, as in the README verify command.

| Loss    | PGD acc. ours |
|---------|--------------:|
| CC-IBP  |         70.90 |
| MTL-IBP |         70.30 |
| Exp-IBP |         69.00 |

### Run times

| Loss    | Training, ours (RTX 5070 Ti laptop) | Training, paper (Titan V) | Verification of 1000 images, ours |
|---------|------------------------------------:|--------------------------:|----------------------------------:|
| CC-IBP  | ~5 h                                |          4.9 h (1.77e4 s) |                         ~6 h      |
| MTL-IBP | ~5 h                                |          4.9 h (1.76e4 s) |                         4 h 14 m  |
| Exp-IBP | 4 h 15 m |                                     not reported     |                         4 h 21 m  |

### What the comparison says

- All three Table 1 cells reproduce: every standard and BaB-verified number is within about 1.3 points of the paper.
- The ordering matches the paper: CC-IBP and MTL-IBP are effectively tied, Exp-IBP has the lowest verified accuracy.
- IBP alone verifies nothing at this radius, and BaB adds 10 to 12 points over CROWN/IBP, as in the paper.
- The one cell that stands out is Exp-IBP under CROWN/IBP: 51.80 ours vs 47.87 paper, a 3.9 point gap,
  larger than subset sampling error alone would suggest. Each model is a single training run, so
  run-to-run training variance is the likely remainder. The BaB number for the same model is within 1.3 points.

### Rest of the paper CIFAR-10 numbers, for reference (not reproduced here)

Literature baselines at eps = 2/255 from Table 1 (standard / verified):
STAPS 79.76 / 62.98, SABR 79.89 / 63.28, IBP-R 80.46 / 62.03, COLT 78.4 / 60.5,
CROWN-IBP 71.52 / 53.97, IBP 68.06 / 56.18, SortNet 67.72 / 56.94, AdvIBP 59.39 / 48.34.

The paper at eps = 8/255 (not run yet; uses config/cifar.json and 260 epochs):

| Loss    | Standard acc. paper | Verified BaB paper | CROWN/IBP paper | IBP paper |
|---------|--------------------:|-------------------:|----------------:|----------:|
| CC-IBP  | 53.71               |              35.27 |           34.02 |     34.02 |
| MTL-IBP | 53.35               |              35.44 |           34.64 |     34.64 |
| Exp-IBP | 53.97               |              35.04 |           33.88 |     33.88 |

Source: De Palma et al., Expressive Losses for Verified Robustness via Convex Combinations, ICLR 2024, Tables 1, 5 and 6.
Our numbers come from: python reproduction/summarize_ver.py <ver_folder> 1000
