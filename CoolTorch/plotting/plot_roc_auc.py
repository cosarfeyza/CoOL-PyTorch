import torch
import pandas as pd
import matplotlib.pyplot as plt
from CoolTorch.train.predict_risks import predict_risks


def plot_roc_auc(outcome_data, exposure_data, model,
                 title="Receiver operating\ncharacteristic curve",
                 ax=None, show=True):
    if isinstance(outcome_data, pd.DataFrame):
        if outcome_data.shape[1] != 1:
            raise ValueError("outcome_data DataFrame must have exactly one column.")
        y_true = torch.tensor(outcome_data.iloc[:, 0].to_numpy(), dtype=torch.float32)
    elif isinstance(outcome_data, pd.Series):
        y_true = torch.tensor(outcome_data.to_numpy(), dtype=torch.float32)
    elif isinstance(outcome_data, torch.Tensor):
        y_true = outcome_data.to(torch.float32)
    else:
        raise TypeError("outcome_data must be a pandas Series/DataFrame or torch.Tensor.")

    if y_true.dim() != 1:
        y_true = y_true.reshape(-1)

    if isinstance(exposure_data, pd.DataFrame):
        X = torch.tensor(exposure_data.to_numpy(), dtype=torch.float32)
    elif isinstance(exposure_data, torch.Tensor):
        X = exposure_data.to(torch.float32)
    else:
        raise TypeError("exposure_data must be a pandas DataFrame or torch.Tensor.")

    if X.dim() != 2:
        raise ValueError("exposure_data must be 2D (n, p).")
    if y_true.numel() != X.shape[0]:
        raise ValueError(f"Row mismatch: outcome n={y_true.numel()} vs exposure n={X.shape[0]}.")

    y_true = (y_true > 0.5).float() if y_true.dtype.is_floating_point else y_true.clamp(0, 1).float()

    model_dict = {
        "W1": model.fc1.weight.t().detach(),
        "b1": model.fc1.bias.detach(),
        "W2": model.fc2.weight.t().detach(),
        "b2": model.fc2.bias.detach(),
    }
    y_score = predict_risks(X, model_dict)

    sorted_scores, indices = torch.sort(y_score, descending=True)
    sorted_true = y_true[indices]

    P = torch.clamp(torch.sum(sorted_true), min=1e-12)
    N = torch.clamp(len(sorted_true) - P, min=1e-12)

    TPR = torch.cumsum(sorted_true, dim=0) / P
    FPR = torch.cumsum(1 - sorted_true, dim=0) / N

    TPR = torch.cat([torch.tensor([0.0]), TPR, torch.tensor([1.0])])
    FPR = torch.cat([torch.tensor([0.0]), FPR, torch.tensor([1.0])])
    auc_val = torch.trapz(TPR, FPR).item()

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))
    else:
        fig = ax.figure

    ax.plot(FPR, TPR, label=f"AUC = {auc_val:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="lower right")
    fig.tight_layout()

    if show:
        plt.show()

    return auc_val
