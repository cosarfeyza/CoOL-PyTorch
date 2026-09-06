import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from typing import Union

def individual_effects_matrix(
    X: Union[pd.DataFrame, np.ndarray, torch.Tensor],
    model
) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        col_names = list(X.columns)
        X_t = torch.tensor(X.values, dtype=torch.float32)
    elif isinstance(X, np.ndarray):
        col_names = [f"x{i+1}" for i in range(X.shape[1])]
        X_t = torch.tensor(X, dtype=torch.float32)
    elif isinstance(X, torch.Tensor):
        col_names = [f"x{i+1}" for i in range(X.shape[1])]
        X_t = X.to(dtype=torch.float32)
    else:
        raise TypeError("X must be a DataFrame, numpy array, or torch Tensor")

    device = next(model.parameters()).device
    X_t = X_t.to(device)
    N, D = X_t.shape

    # only fc1 weights and scalar b2 from fc2 — same as R logic
    W1 = model.fc1.weight.to(device=device, dtype=torch.float32)   # (H, D)
    b1 = model.fc1.bias.to(device=device, dtype=torch.float32)     # (H,)
    b2 = model.fc2.bias.to(device=device, dtype=torch.float32).squeeze()

    ind_effect = torch.zeros((N, D + 1), dtype=torch.float32, device=device)

    # zero all columns except i, then forward through hidden layer
    for i in range(D):
        X_temp = torch.zeros_like(X_t)
        X_temp[:, i] = X_t[:, i]
        H = F.relu(X_temp @ W1.T + b1)      # (N, H)
        ind_effect[:, i] = H.sum(dim=1)

    ind_effect[:, D] = b2  # baseline column: scalar b2 broadcast to all rows

    out_cols = list(col_names) + ["Baseline_risk"]
    return pd.DataFrame(ind_effect.detach().cpu().numpy(), columns=out_cols)
