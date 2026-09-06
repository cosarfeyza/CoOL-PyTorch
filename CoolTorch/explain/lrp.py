import torch
import torch.nn.functional as F
import pandas as pd

@torch.no_grad()
def layerwise_relevance_propogation(
    X: torch.Tensor,     # (N, D)
    model,               # NonNegativeNN
    feature_names        # exposure_df.columns
):
    device, dtype = X.device, X.dtype
    N, D = X.shape
    Hdim = model.fc1.weight.shape[0]

    W1 = model.fc1.weight.to(device=device, dtype=dtype)
    b1 = model.fc1.bias.to(device=device, dtype=dtype)
    W2 = model.fc2.weight.squeeze(0).to(device=device, dtype=dtype)
    b2 = model.fc2.bias.to(device=device, dtype=dtype).squeeze()

    H_all = F.relu(X @ W1.T + b1)
    o_all = F.relu(H_all @ W2 + b2)

    R_X = torch.zeros((N, D), dtype=dtype, device=device)
    U_B = torch.empty((N,), dtype=dtype, device=device)

    if (W2 < 0).any():
        print("Some weights are below 0")
    if (W1 < 0).any():
        print("Some weights are below 0")

    for i in range(N):
        H = H_all[i, :]
        o = o_all[i]

        Pos1 = torch.clamp(W2, min=0)
        pos1_num = H * Pos1
        pos1_sum = pos1_num.sum()
        if not torch.isfinite(pos1_sum) or pos1_sum.item() == 0:
            pos1_sum = torch.tensor(1.0, device=device, dtype=dtype)
        Pos1_norm = pos1_num / pos1_sum

        o_adj = F.relu(o - F.relu(F.relu(b2)))
        R_H = Pos1_norm * o_adj

        Pos2 = torch.clamp(W1, min=0)
        x_i = X[i, :]

        for g in range(Hdim):
            numer = x_i * Pos2[g, :]
            denom = numer.sum()
            if not torch.isfinite(denom) or denom.item() == 0:
                denom = torch.tensor(1.0, device=device, dtype=dtype)
            R_X[i, :] += (numer / denom) * R_H[g]

        U_B[i] = b2

        if not torch.isfinite(R_X[i, :].sum()) or R_X[i, :].sum().item() == 0:
            R_X[i, :] = 0.0

    Baseline_risk = U_B.clone()
    residual = o_all - (R_X.sum(dim=1) + Baseline_risk)

    if torch.max(residual).item() > 1e-6:
        print("WARNING: Some risk contributions do not sum to the predicted value")

    df_cols = list(feature_names) + ["Baseline_risk"]

    df_out = pd.DataFrame(
        torch.cat([R_X, Baseline_risk.unsqueeze(1)], dim=1)
        .detach().cpu().numpy(),
        columns=df_cols
    )

    return df_out
