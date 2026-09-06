import numpy as np
import matplotlib.pyplot as plt
from CoolTorch.train.predict_risks import predict_risks


def _to_np(a):
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    elif hasattr(a, "values"):
        a = a.values
    return np.asarray(a)


def calibration_plot(exposure_data, outcome_data, model, sub_groups,
                     ipw=1, title="Calibration curve"):
    X = _to_np(exposure_data)
    y = _to_np(outcome_data).reshape(-1)
    g = _to_np(sub_groups).astype(int).reshape(-1)

    n = X.shape[0]
    if ipw is None or np.asarray(ipw).size != n:
        w = np.ones(n, dtype=float)
        print("Equal weights are applied (assuming no selection bias)")
    else:
        w = _to_np(ipw).reshape(-1).astype(float)

    model_dict = {
        "W1": model.fc1.weight.t().detach(),
        "b1": model.fc1.bias.detach(),
        "W2": model.fc2.weight.t().detach(),
        "b2": model.fc2.bias.detach()
    }
    preds = _to_np(predict_risks(X, model_dict)).reshape(-1).astype(float)

    K = int(np.max(g))
    c_pred = np.empty(K, dtype=float)
    c_risk = np.empty(K, dtype=float)

    for i in range(1, K + 1):
        mask = (g == i)
        c_pred[i - 1] = float(np.mean(preds[mask])) if np.any(mask) else np.nan
        num = float(np.sum(y[mask] * w[mask])) if np.any(mask) else np.nan
        den = float(np.sum(w[mask])) if np.any(mask) else np.nan
        c_risk[i - 1] = num / den if den and den > 0 else np.nan

    xmax = np.nanmax(c_pred) if np.any(~np.isnan(c_pred)) else 1.0
    ymax = np.nanmax(c_risk) if np.any(~np.isnan(c_risk)) else 1.0
    xlim = (0.0, min(xmax * 1.2, 1.0))
    ylim = (0.0, min(ymax * 1.2, 1.0))

    plt.figure(figsize=(6.5, 5))
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.xlabel("Predicted risk in each sub-group")
    plt.ylabel("Actual risk in each sub-group")
    plt.title(title)
    plt.grid(False)

    for i in range(K):
        if not (np.isnan(c_pred[i]) or np.isnan(c_risk[i])):
            plt.text(c_pred[i], c_risk[i], str(i + 1), ha="center", va="center", fontsize=12)

    lo, hi = 0.0, min(xlim[1], ylim[1])
    plt.plot([lo, hi], [lo, hi], linewidth=1)
    plt.margins(x=0, y=0)
    plt.show()

    return {"c_pred": c_pred, "c_risk": c_risk, "K": K}
