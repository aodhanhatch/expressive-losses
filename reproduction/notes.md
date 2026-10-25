Some random stuff to note while trying to reproduce:
auto_LiRPA pinned to release 0.5.0, commit bfb7997, and the reason. If you later want the pin to survive a fresh clone, the place to record it is your fork's README setup section.

Some key definitions for the experiment:
nat_loss = Average cross-entropy on unperturbed images, not the total CC-IBP training objective
nat_ok = Fraction of correctly classified on unperturbed images
adv_ok = Fraction correctly classified on the PGD-generated inputs
ver_ok = Frcation passing the IBP margin-bound check
eps = current perturbation size, 


On the smoke test, we ran 4 epochs on the 2/255 CIFAR 10 test,
The results we got seems resonable, we have that the natural accuracy and the advererial accuracy is the same for the first epoch where the eps valkue is 0
Also even though this was a smoke test with only for 4 epochs, we can see how the natural accuracy is falling as the peterubutation size changes


After training the CC-IBP model with default setting, on the CIFAR-10 dataset, we managed to run the verification up to a 1000 images (about 5-6 hour run time on the RTX 5070Ti). 



## Results: CIFAR-10, eps = 2/255, ours vs the paper

Setup. Trained with the README commands (160 epochs, one run per loss, default seed). Verified with
verify.py: PGD attack, then IBP and CROWN, then OVAL branch-and-bound (alpha-beta-CROWN, 1800 s timeout).
Ours is the first 1000 test images. The paper is the full 10,000-image test set.
Standard error on a 1000-image subset is about 1.5 points, so its reasonable to say the differences below ~2 points are noise.

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



### A verification on the ineqaulities stated in section 4.2 and 4.3, proposition 4.2 and 4.3
In the propostion, whats been proved is that for every alpha in [0, 1] and every input:
L_CC(alpha) <= L_MTL(alpha), from the convexity of cross-entropy (propositon 4.2)
L_Exp(alpha) <= L_MTL(alpha), from the weighted AM-GM ineqaulity (propostion 4.3)

The immportant clarifications here are the basis this inequality hold on
The inequalities are per image and per alpha. For image x with label y, all 3 losses must be build from exactly the same 2 ingridients:
z_adv = the logit margins at one adversarial point x_adv, found by PGD inside the epsilon ball around x
z = the IBP lower bounds on those margins

The models must be the same/have the same weights theta, the same epsilon, the same alpha. The only thing that differs is the arithmetic for combining the 2 ingridients mentioned above
As we know from the paper,
- CC mixes the two vectors, then applies cross entropy on them
- MTL applies vross-entropy to each vector, then combines the tweo losses
- Exp mixes the two losses geometrically



Also note as part of the main proposition of the paper, we know that each loss sits between th adversial loss and vefified IBP lozz

In terms of the setup, you can see the scipt used on reproduction/inequality_check.py 
What the scritpt is doing is that
For each of the first 1000 CIFAR-10 test images, (40 step PGD, step size of 0.035m eps = 2/255) and IBP lower bounds and z_bar ont he same margins.
Frin that pair it builds all three losses per image, in float64, at 
MTL -  CC and MTL - Exp. A gap below -1e-6 count as a violation. I ran it on all 3 traiend models. (you can see the raw output int the txt file)

Result was reassuring, we got 0 vioaltions of both the inequality and the sandwich, on any alpha on all images for all models
We know from the paper that the ineqaulity should hold for any pair of vectors, hence if this didnt hold then it will tell us that there is a bug in our code, on top of providing an empirical soundness check it also provides the size of the gaps between the different models

We observed the the adversarial loss is nearly the same for all 3 models, theough IBP had a fairly noticable difference



### Training CC-IBP with alpha = 3e-3
So the baseline we trained for the CC-IBP was the alpha = 1e-2, using the figure given in the README 
Now in the paper, table 8 some standard and verified accuracy for different alpha values so we will do another traing run to reproduce those figures
The first one we will try will be the CC-IBP with alpha = 3e-3

Before we get our results, let us make some brief predictions,
- Given that we are usign a lower alpha, that means our training loss is more skewed towards adversarial loss,
- From that we can expect the verified accuracy to go down compared against our baseline,
- We can expect out standard accuracy to go up, IBP contraints what the network can represent by keeping most ReLUs firmly on or off for every point in the epsilon ball acting as a heavy regularisation, costing clean accuracy. Hence if we lower the alpha value, the constraint will be sligly weaker/looser helping with our standrad accuracy, so my guess is that it will go up.
- I think its resonable to expect that the adversarial accuracy will go up as well, as we are giving more weight to adversarial loss. However, it is flagged in the paper that adversarial accuracy "may first increase and then decrease with alpha" so it may not be necessary the case that it will definately go up. Although, what makes it a resonable guess is that advesarial loss must always be lower than standard accuracy, and as we expect the standard accuracy to go up, that means that the the upper bound goes up, so we may be able to expect that the advesarial accuracy goes up in accordance with the standrad accuracy.
- Also another potentially interesting thing to watch for is the gap between the PGD and verified accuracy. The gap in the 2 represnts images that can be shown that its advesarial robust but cannot be proven robust. I think we can expect this gap to widen with lower alpha as the verification/proven bounds will be looser hence the coumt of images that are empirically robusdt but not proven robust will probaly go up.


--- Results from the training and verification run
| Metric                   | Baseline α = 1e-2 | New α = 3e-3 | Change | Paper Table 8, α = 3e-3 |
|--------------------------|------------------:|-------------:|-------:|------------------------:|
| Standard accuracy        | 81.00             | 82.30        | +1.3   | 81.93                   |
| PGD accuracy             | 70.90             | 72.30        | +1.4   | –                       |
| Verified, CROWN only     | 53.50             | 45.90        | −7.6   | –                       |
| Verified, full BaB       | 63.70             | 60.30        | −3.4   | 61.12                   |
| PGD minus verified (gap) | 7.2               | 12.0         | +4.8   | –                       |

- Nicely, turns our prediction was correct!
- Accuracy went up by 1.3 points
- PGD accuracy went up by 1.4 points, more than standard accuracy
- Verified accuracy went down, with CROWN verified going down quite significantly with -7.6 points and BaB down by 3.4 points
--> A similar trait is also shown in the paper in figure 1 of section 6.2. if you look at the graph presented for CC-IBP with e=2/255, we can see how CROWN/IBP falls more sharply than BaB, hence our results seems to be in line with expectations
---> The likely explanation for this is the increased number of unstable ReLUs with the lower alpha used in traing. With CROWN, more unstable ReLUs lead to loser bounds leading to worsened performance, while for BaB, more unstable ReLUs will lead to more splitting, this was shown to be true with our verification as it look longer to do this run, most likely due to being more time outs by BaBs.
- the gap between PGD and verified has also increased by 4.8 points
- We can also see that our resulst is quite well aligned with the paper for standard accuracy and verified full BaB
