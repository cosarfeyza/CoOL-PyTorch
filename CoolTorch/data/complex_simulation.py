import torch
import pandas as pd

def complex_simulation(n: int, seed: int | None = None, device=None):
    if seed is not None:
        torch.manual_seed(seed)

    if device is None:
        device = torch.device("cpu")

    # R: sample(1:0, n, prob=c(p, 1-p)) -> 1 with prob p
    def bern01(p: float):
        return (torch.rand(n, device=device) < p).to(torch.int64)

    Genes = bern01(0.05)
    Living_area = bern01(0.2)

    Low_SES = bern01(0.2)
    Physically_active = bern01(0.8)

    # if Low_SES==1 & Bernoulli(0.2) -> Physically_active <- 0
    mask_inactive = (Low_SES == 1) & (torch.rand(n, device=device) < 0.2)
    Physically_active = torch.where(mask_inactive, torch.zeros_like(Physically_active), Physically_active)

    # Mutation_X starts as 0; if Genes==1 & Bernoulli(0.95) -> 1
    Mutation_X = torch.zeros(n, device=device, dtype=torch.int64)
    mask_mut = (Genes == 1) & (torch.rand(n, device=device) < 0.95)
    Mutation_X = torch.where(mask_mut, torch.ones_like(Mutation_X), Mutation_X)

    # LDL baseline then Genes can force to 1 with prob 0.15
    LDL = bern01(0.3)
    mask_ldl = (Genes == 1) & (torch.rand(n, device=device) < 0.15)
    LDL = torch.where(mask_ldl, torch.ones_like(LDL), LDL)

    # Night_shifts baseline then can be set to 1 by Living_area or Low_SES (each with prob 0.1)
    Night_shifts = bern01(0.2)
    mask_ns_from_area = (Living_area == 1) & (torch.rand(n, device=device) < 0.1)
    mask_ns_from_ses  = (Low_SES == 1) & (torch.rand(n, device=device) < 0.1)
    Night_shifts = torch.where(mask_ns_from_area | mask_ns_from_ses, torch.ones_like(Night_shifts), Night_shifts)

    # Air_pollution baseline then Living_area can force to 1 with prob 0.3
    Air_pollution = bern01(0.2)
    mask_air = (Living_area == 1) & (torch.rand(n, device=device) < 0.3)
    Air_pollution = torch.where(mask_air, torch.ones_like(Air_pollution), Air_pollution)

    # Y baseline then two mechanisms can set Y to 1
    Y = bern01(0.05)

    # Mechanism 1:
    # if Physically_active==0 & LDL==1 & Night_shifts==1 & Bernoulli(0.15) -> Y=1
    mask_y1 = (Physically_active == 0) & (LDL == 1) & (Night_shifts == 1) & (torch.rand(n, device=device) < 0.15)
    Y = torch.where(mask_y1, torch.ones_like(Y), Y)

    # Mechanism 2:
    # if Mutation_X==1 & Air_pollution==1 & Bernoulli(0.1) -> Y=1
    mask_y2 = (Mutation_X == 1) & (Air_pollution == 1) & (torch.rand(n, device=device) < 0.1)
    Y = torch.where(mask_y2, torch.ones_like(Y), Y)

    df = pd.DataFrame({
        "Y": Y.cpu().numpy(),
        "Physically_active": Physically_active.cpu().numpy(),
        "Low_SES": Low_SES.cpu().numpy(),
        "Mutation_X": Mutation_X.cpu().numpy(),
        "LDL": LDL.cpu().numpy(),
        "Night_shifts": Night_shifts.cpu().numpy(),
        "Air_pollution": Air_pollution.cpu().numpy(),
    })
    return df
