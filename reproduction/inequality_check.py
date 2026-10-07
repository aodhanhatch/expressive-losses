# Empirical check of Propositions 4.2 and 4.3
# Here we also do a double check on the expressivity sandwitch where the loss at alpha is in between loss of advesarial and loss of verified

# Run from the repo folder, e.g.
# python reproduction/inequality_check.py --config=config/cofar_2_255.json --model=cnn --load model_cifar_ccibp/cnn_fast_1790529581_ckpt_last --test_att_n_steps 40 --test_att_step_size 0.035 --end_idx 1000


import copy
import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())

from argparser import parse_args
from attack import pgd_attack
from certified import get_bound, get_C
from config import load_config
from datasets import load_data
from utils import compute_perturbation, prepare_model, set_seed
from auto_LiRPA import BoundedModule
from auto_LiRPA.utils import logger

# to test the inequality and multiple alphas, I am going to have and alpha grid and produce results on those
ALPHAS = [0.0, 0.004, 0.01, 0.05, 0.095, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
TOL = 1e-6 # a gap below -TOL will count as a violation

def loss_over_margins(z):
    """"Per-image cross-entropy over logit differences z (shape [B, num_class-1]).
        Same construction as certified.get_loss_over_lbs, but without the batch mean:
        pad a 0 for the true class at index 0, use fake label 0, and feed -z as logits"""
    z_padded = torch.cat([torch.zeros_like(z[:, :1]), z], dim=1)
    fake_labels = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
    return F.cross_entropy(-z_padded, fake_labels, reduction="none")

def main(args):
    config = load_config(args.config)
    set_seed(args.seed or config["seed"])

    ### --- model, in eval mode for BOTH and plain forward pass and the IBP bounds ---
    model_ori, _, _, _ = prepare_model(args, logger, config)
    model_ori.eval()
    model_ori.to(args.device)
    batch_size = args.batch_size or config["batch_size"]
    dummy_input, _, test_data = load_data(args, config["data"], batch_size, batch_size, aug=False)
    model_lirpa = BoundedModule(
        copy.deepcopy(model_ori), dummy_input, bound_opts=config["bound_params"]["bound_opts"], custom_ops={}, device=args.device)
    

    eps = args.eps or config["bound_params"]["eps"]
    data_max, data_min, std = test_data.data_max, test_data.data_min, test_data.std
    dev = args.device

    # --- pass 1: collect the two vectors for every image ---
    z_adv_all, lb_all = [], []
    n_seen = 0
    for inputs, targets in test_data:
        if args.end_idx != -1 and n_seen >= args.end_idx:
            break
        inputs, targets = inputs.to(dev), targets.to(dev)

        # perturbation box in normalised input spcace (same helper verify.py uses)
        x, data_lb, data_ub = compute_perturbation(
            args, eps, inputs, data_min.to(dev), data_max.to(dev), std.to(dev), True, False)
        with torch.no_grad():
            # IBP lower bounds on the logit differences z_bar (shape [B, 9])
            lb, _, = model_lirpa(x=(x,), method_opt="compute_bounds", IBP=True, C=get_C(args, inputs, targets), method=None, no_replicas=True)

            # adversarial point and its logit differences z_adv (shape [B, 9])
            adv = pgd_attack(
                model_ori, data_lb, data_ub,
                lambda out: F.cross_entropy(out, targets, reduction="none"),
                args.test_att_n_steps, args.test_att_step_size)
            adv_logits = model_ori(adv)
            z_adv = torch.bmm(get_C(args, inputs, targets), adv_logits.unsqueeze(-1)).squeeze(-1)

            z_adv_all.append(z_adv.double().cpu())
            lb_all.append(lb.double().cpu())
        n_seen += inputs.size(0)

    z_adv = torch.cat(z_adv_all)
    lb = torch.cat(lb_all)
    print(f"images: {z_adv.size(0)}")

    # --- pass 2: the losses, per image, for every alpha ---
    L_adv = loss_over_margins(z_adv)
    L_ver = loss_over_margins(lb)
    print(f"L_adv mean {L_adv.mean():.4f}    L_ver mean {L_ver.mean():.4f}")
    print(f"sanwich precondition z_bar <= z_adv violated on "
            f"{int(((lb - z_adv) > TOL).any(dim=1).sum())} images")
    header = f"{'alpha':>6} | {'MTL-CC mean':>11} {'MTL-CC min':>11} {'viol':>5} | " \
             f"{'MTL-Exp mean':>12} {'MTL-Exp min':>11} {'viol':>5} | {'sandwich viol':>13}"
    print(header)
    print("-" * len(header))
    for a in ALPHAS:
        L_cc = loss_over_margins(a * lb + (1 - a) * z_adv)                  # eq. (7)
        L_mtl = a * L_ver + (1 - a) * L_adv                                 # eq. (8)
        L_exp = torch.exp((1 - a) * torch.log(L_adv) + a * torch.log(L_ver))  # eq. (9), via logs

        gap_cc = L_mtl - L_cc
        gap_exp = L_mtl - L_exp
        sandwich_bad = ((L_cc < L_adv - TOL) | (L_cc > L_ver + TOL) |
                    (L_exp < L_adv - TOL) | (L_exp > L_ver + TOL)).sum()

        print(f"{a:6.3f} | {gap_cc.mean():11.5f} {gap_cc.min():11.2e} {int((gap_cc < -TOL).sum()):5d} | "
              f"{gap_exp.mean():12.5f} {gap_exp.min():11.2e} {int((gap_exp < -TOL).sum()):5d} | "
              f"{int(sandwich_bad):13d}")

if __name__ == "__main__":
    main(parse_args())
