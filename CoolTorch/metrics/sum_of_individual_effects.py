# Backend: pure PyTorch (no R). R equivalent: CoOL_6_sum_of_individual_effects (CoOL_functions.R)
# Predicts each individual's risk as if the exposures acted independently: sums
# individual_effects_matrix's per-exposure-alone risks plus baseline, so it can be
# compared against the model's actual (interaction-aware) predicted risk.
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from typing import Union

def sum_of_individual_effects(
    X: Union[pd.DataFrame, np.ndarray, torch.Tensor],
    model
) -> pd.Series:
    if isinstance(X, pd.DataFrame):
        idx = X.index
        X_t = torch.tensor(X.values, dtype=torch.float32)
    elif isinstance(X, np.ndarray):
        idx = pd.RangeIndex(len(X))
        X_t = torch.tensor(X, dtype=torch.float32)
    elif isinstance(X, torch.Tensor):
        idx = pd.RangeIndex(X.shape[0])
        X_t = X.to(dtype=torch.float32)
    else:
        raise TypeError("X must be a DataFrame, numpy array, or torch Tensor")

    device = next(model.parameters()).device
    X_t = X_t.to(device)

    W1 = model.fc1.weight.to(device=device, dtype=torch.float32)   # (H, D)
    b1 = model.fc1.bias.to(device=device, dtype=torch.float32)     # (H,)
    b2 = model.fc2.bias.to(device=device, dtype=torch.float32).reshape(())

    # vectorized: Z[n, i, h] = X[n, i] * W1[h, i] + b1[h]  -> (N, D, H)
    Z = X_t[:, :, None] * W1.t()[None, :, :] + b1[None, None, :]
    contrib = F.relu(Z).sum(dim=2).sum(dim=1)  # relu -> sum hidden -> sum features -> (N,)
    out = contrib + b2

    return pd.Series(out.detach().cpu().numpy(), index=idx, name="sum_of_individual_effects")
