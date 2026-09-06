import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def visualised_mean_risk_contributions(
    results: pd.DataFrame,
    sub_groups,
    ipw=1
):
    if not isinstance(results, pd.DataFrame):
        results = pd.DataFrame(results)

    sub_groups = np.asarray(sub_groups).reshape(-1)
    n = len(sub_groups)

    if np.isscalar(ipw):
        ipw_vec = np.ones(n, dtype=float)
    else:
        ipw_vec = np.asarray(ipw, dtype=float).reshape(-1)
        if ipw_vec.size != n:
            ipw_vec = np.ones(n, dtype=float)
            print("Equal weights are applied (assuming no selection bias)")

    G = results.shape[1]
    labels = []
    total_w = ipw_vec.sum()
    for i in range(1, G + 1):
        mask = (sub_groups == i)
        prev = 100.0 * ipw_vec[mask].sum() / total_w if total_w > 0 else 0.0
        labels.append(f"Group {i} ({prev:.1f}%)")
    results = results.copy()
    results.columns = labels

    res = results.copy()
    res = np.round(res * 1000)
    max_val = np.max(res.values)
    if max_val > 0:
        res = res * 1000.0 / max_val
    res = res.iloc[:-1, :]  # drop Baseline_risk row

    mat = res.values.astype(float)
    n_rows, n_cols = mat.shape

    farver = LinearSegmentedColormap.from_list(
        "farver",
        ["white", "orange", "orange", "red", "red", "red", "black"],
        N=1000,
    )

    fig, ax = plt.subplots(figsize=(max(6, n_cols * 1.2), max(5, n_rows * 0.6)))

    x = np.arange(n_cols + 1)
    y = np.arange(n_rows + 1)
    X, Y = np.meshgrid(x, y)

    ax.pcolormesh(
        X, Y, mat,
        cmap=farver,
        edgecolors="black",
        linewidth=0.5
    )

    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(res.columns, rotation=90)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(res.index)

    ax.invert_yaxis()  # first row at top, consistent with R output

    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.show()
