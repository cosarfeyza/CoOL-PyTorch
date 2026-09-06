import os
from scipy.spatial.distance import pdist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from CoolTorch.plotting.dendo_clustgeo import CoOl_dendrogram_clustgeo


def _mean_as_matrix_dist_cityblock(X):
    n = X.shape[0]
    if n <= 1:
        return 0.0
    d = pdist(X, metric="cityblock")
    return (2.0 * d.sum()) / (n * n)


def _elbow_k(df):
    k_vals      = df["k"].values
    dists       = df["mean_dist"].values
    first_diff  = np.diff(dists)
    second_diff = np.diff(first_diff)
    idx = int(np.argmax(second_diff))
    return int(k_vals[idx + 1])


def CoOL_6_number_of_sub_groups(risk_contributions, low_number, high_number, ipw=1,
                                  workdir="clustgeo_tmp",
                                  r_script_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils", "CoOL_dendrogram_runner.r"),
                                  plot=True, auto_elbow=False):
    X_all = risk_contributions.to_numpy(dtype=float)

    rows = []
    for k in range(int(low_number), int(high_number) + 1):
        clus = CoOl_dendrogram_clustgeo(
            risk_contributions=risk_contributions, number_of_subgroups=k,
            ipw=ipw, workdir=workdir, r_script_path=r_script_path, show=False,
        )

        temp = np.zeros(k, dtype=float)
        N    = np.zeros(k, dtype=int)
        for g in range(1, k + 1):
            idx = np.where(clus == g)[0]
            N[g - 1]    = len(idx)
            temp[g - 1] = _mean_as_matrix_dist_cityblock(X_all[idx])
        mean_dist_k = float((temp * N).sum() / max(N.sum(), 1))

        rows.append({"k": k, "mean_dist": mean_dist_k})
        print(f"{k} groups  |  mean_dist={mean_dist_k:.4f}")

    df = pd.DataFrame(rows)

    if plot:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df["k"], df["mean_dist"], marker="o", color="#1f77b4")
        ax.set_xlabel("Sub-groups")
        ax.set_ylabel("Mean difference in risk contributions")
        ax.set_title("CoOL: Mean difference vs number of sub-groups")
        ax.set_xticks(df["k"].tolist())

        if auto_elbow:
            k_elb = _elbow_k(df)
            ax.axvline(k_elb, color="red", linestyle="--", linewidth=1,
                       label=f"Auto-selected k={k_elb}")
            ax.legend()

        plt.show()
        print("Plot printed")

    optimal_k = None
    if auto_elbow:
        optimal_k = _elbow_k(df)
        print(f"Auto-selected k={optimal_k} (elbow method)")

    # always return (df, k) so unpacking works either way — k is None when
    # auto_elbow=False (diagnostic-only, analyst reads the elbow off the plot)
    return df, optimal_k
