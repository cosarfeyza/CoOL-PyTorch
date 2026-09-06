import torch
import pandas as pd

def confounding_simulation(n: int, seed: int | None = None, device=None):
    if seed is not None:
        torch.manual_seed(seed)
    if device is None:
        device = torch.device("cpu")

    def bern01(p: float):
        # 1 with prob p, else 0
        return (torch.rand(n, device=device) < p).to(torch.int64)

    A = bern01(0.3)
    B = bern01(0.3)
    D = bern01(0.3)
    E = bern01(0.3)
    F = bern01(0.3)
    C = bern01(0.3)
    Y = bern01(0.01)

    # if (C==1 & Bernoulli(0.4)) B <- 1
    mask_B = (C == 1) & (torch.rand(n, device=device) < 0.4)
    B = torch.where(mask_B, torch.ones_like(B), B)

    # if (C==1 & Bernoulli(0.3)) F <- 1
    mask_F = (C == 1) & (torch.rand(n, device=device) < 0.3)
    F = torch.where(mask_F, torch.ones_like(F), F)

    # if (C==1 & Bernoulli(0.15)) Y <- 1
    mask_Y = (C == 1) & (torch.rand(n, device=device) < 0.15)
    Y = torch.where(mask_Y, torch.ones_like(Y), Y)

    return pd.DataFrame({
        "Y": Y.cpu().numpy(),
        "A": A.cpu().numpy(),
        "B": B.cpu().numpy(),
        "C": C.cpu().numpy(),
        "D": D.cpu().numpy(),
        "E": E.cpu().numpy(),
        "F": F.cpu().numpy(),
    })
