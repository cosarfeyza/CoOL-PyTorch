import torch
import pandas as pd

def common_simulation(n: int, seed: int | None = None, device=None):
    if seed is not None:
        torch.manual_seed(seed)

    if device is None:
        device = torch.device("cpu")

    # R: sample(1:0, n, prob=c(p, 1-p), replace=TRUE)  -> 1 with prob p
    def bern01(p: float):
        return (torch.rand(n, device=device) < p).to(torch.int64)

    A = bern01(0.3)
    B = bern01(0.3)
    C = bern01(0.3)
    D = bern01(0.3)
    E = bern01(0.3)
    F = bern01(0.3)
    U = bern01(0.3)
    Y = bern01(0.01)

    # if(U[i]==1 & sample(1:0,1,prob=c(.4,.6))) B[i] <- 1
    # sample(...) here => 1 with prob 0.4
    mask_B_from_U = (U == 1) & (torch.rand(n, device=device) < 0.4)
    B = torch.where(mask_B_from_U, torch.ones_like(B), B)

    # if(U[i]==1 & sample(1:0,1,prob=c(.3,.7))) E[i] <- 1
    mask_E_from_U = (U == 1) & (torch.rand(n, device=device) < 0.3)
    E = torch.where(mask_E_from_U, torch.ones_like(E), E)

    # if(B[i]==1 & sample(1:0,1,prob=c(.04,.96))) Y[i] <- 1
    mask_Y_from_B = (B == 1) & (torch.rand(n, device=device) < 0.04)
    Y = torch.where(mask_Y_from_B, torch.ones_like(Y), Y)

    # if(E[i]==1 & sample(1:0,1,prob=c(.06,.94))) Y[i] <- 1
    mask_Y_from_E = (E == 1) & (torch.rand(n, device=device) < 0.06)
    Y = torch.where(mask_Y_from_E, torch.ones_like(Y), Y)

    df = pd.DataFrame({
        "Y": Y.cpu().numpy(),
        "A": A.cpu().numpy(),
        "B": B.cpu().numpy(),
        "C": C.cpu().numpy(),
        "D": D.cpu().numpy(),
        "E": E.cpu().numpy(),
        "F": F.cpu().numpy(),
    })
    return df
