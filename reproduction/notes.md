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

images with results: 1000 (indices 0..999)
  standard accuracy        (ok):        81.00%
  adversarial accuracy     (pgd_ok):    70.90%
  verified, IBP only       (ibp_ok):     0.00%
  verified, CROWN only     (crown_ok):  53.50%
  verified, full BaB       (ver_ok):    63.70%

**COMPARE WITH THE PAPER when have a mom