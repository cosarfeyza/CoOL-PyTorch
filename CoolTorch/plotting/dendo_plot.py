import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("MacOSX")


def cool_cluster_and_dendrogram(risk_contributions, n_subgroups=3, ipw=None,
                                 title="CoOL risk contribution dendrogram",
                                 ax=None, show=True):
    if isinstance(risk_contributions, torch.Tensor):
        risk_df = pd.DataFrame(risk_contributions.detach().cpu().numpy())
    elif isinstance(risk_contributions, np.ndarray):
        risk_df = pd.DataFrame(risk_contributions)
    elif isinstance(risk_contributions, pd.DataFrame):
        risk_df = risk_contributions.copy()
    else:
        raise TypeError("risk_contributions must be DataFrame, np.ndarray, or torch.Tensor")

    n = len(risk_df)

    if ipw is None:
        ipw = np.ones(n, dtype=float)
    else:
        ipw = np.asarray(ipw, dtype=float)
        if ipw.ndim == 0:
            ipw = np.ones(n, dtype=float) * float(ipw)
        elif len(ipw) != n:
            raise ValueError(f"Length of ipw ({len(ipw)}) must equal number of rows ({n})")

    risk_df["pfreq"] = ipw

    feature_cols = [c for c in risk_df.columns if c != "pfreq"]
    grouped = risk_df.groupby(feature_cols, as_index=False).agg(pfreq=("pfreq", "sum"))
    X_unique = grouped[feature_cols].to_numpy(dtype=float)

    dist_vec = pdist(X_unique, metric="cityblock")
    Z = linkage(dist_vec, method="average")
    clus_unique = fcluster(Z, n_subgroups, criterion="maxclust")

    merged = (
        risk_df.drop(columns=["pfreq"])
        .reset_index().rename(columns={"index": "id"})
        .merge(grouped.assign(pclus=clus_unique), on=feature_cols)
        .sort_values("id")
    )
    clus = merged["pclus"].to_numpy()

    m = X_unique.shape[0]
    if n_subgroups >= m:
        color_threshold = 0.0
    else:
        d_cut = Z[m - n_subgroups, 2]
        color_threshold = d_cut - 1e-10

    created_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
        created_fig = True

    dendrogram(Z, no_labels=True, ax=ax, color_threshold=color_threshold,
               above_threshold_color="black")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_ylabel("Manhattan distance")

    if created_fig and show:
        plt.tight_layout()
        plt.show()

    return clus, Z


def plot_cluster_sizes(clus, title="Size of CoOL sub-groups", ax=None):
    unique, counts = np.unique(clus, return_counts=True)

    created_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3))
        created_fig = True

    ax.bar(unique, counts)
    ax.set_xlabel("Sub-group")
    ax.set_ylabel("Number of individuals")
    ax.set_title(title)
    ax.set_xticks(unique)

    if created_fig:
        plt.tight_layout()
        plt.show()
