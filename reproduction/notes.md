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