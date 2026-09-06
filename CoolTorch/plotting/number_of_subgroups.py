import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt


def _to_np(a):
    if hasattr(a, "detach"): a = a.detach().cpu().numpy()
    if hasattr(a, "values"): a = a.values
    return np.asarray(a, dtype=float)


def _collapse_with_weights(X, w):
    Xc = np.ascontiguousarray(X)
    dt = np.dtype((np.void, Xc.dtype.itemsize * Xc.shape[1]))
    packed = Xc.view(dt).ravel()
    uniq, idx_first, inv = np.unique(packed, return_index=True, return_inverse=True)
    Xu = Xc[idx_first]
    wu = np.bincount(inv, weights=w, minlength=Xu.shape[0])
    return Xu, wu, inv


def _mean_within_cluster_cityblock(X, labels):
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels, dtype=int)
    total, count = 0.0, 0
    for k in np.unique(labels):
        idx = np.where(labels == k)[0]
        n = idx.size
        if n <= 1:
            count += n
            continue
        D = pdist(X[idx], metric="cityblock")
        total += float(D.mean()) * n
        count += n
    return np.nan if count == 0 else total / count


def _elbow_k(k_values, mean_dist):
    # sharpest bend of the mean-distance curve (same rule as the R path's _elbow_k)
    k_vals = np.asarray(k_values)
    dists = np.asarray(mean_dist, dtype=float)
    if dists.size < 3:
        return int(k_vals[0])
    second_diff = np.diff(np.diff(dists))
    return int(k_vals[int(np.argmax(second_diff)) + 1])


def number_of_sub_groups(risk_contributions, low_number=1, high_number=5, ipw=None,
                          plot=True, auto_elbow=False,
                          title="Sub-groups vs. mean difference in risk contributions"):
    X = _to_np(risk_contributions)
    if X.ndim != 2:
        raise ValueError("risk_contributions must be (n, p).")
    n = X.shape[0]

    if ipw is None:
        w = np.ones(n, float)
        print("Equal weights are applied (assuming no selection bias)")
    else:
        w = _to_np(ipw).reshape(-1)
        if w.size != n:
            w = np.ones(n, float)
            print("Equal weights are applied (assuming no selection bias)")

    Xu, wu, inv = _collapse_with_weights(X, w)
    reps = np.rint(wu).astype(int)
    keep = reps > 0

    if not keep.any():
        k_values = list(range(int(low_number), int(high_number) + 1))
        return {'k_values': k_values, 'mean_dist': [0.0] * len(k_values),
                'clusters': {k: np.ones(n, int) for k in k_values}}

    Xu_k, reps_k = Xu[keep], reps[keep]
    Xexp = np.repeat(Xu_k, reps_k, axis=0)

    D = pdist(Xexp, metric="cityblock")
    Z = linkage(D, method="ward")

    exp_to_uniq = np.concatenate([np.full(r, i, int) for i, r in enumerate(reps_k)])
    kept_indices = np.where(keep)[0]

    k_values = list(range(int(low_number), int(high_number) + 1))
    mean_dist, clusters_by_k = [], {}

    for k in k_values:
        labs_exp = fcluster(Z, k, criterion="maxclust")
        labs_u_k = np.array([np.bincount(labs_exp[exp_to_uniq == i]).argmax() for i in range(len(reps_k))])
        labs_u = np.ones(len(reps), int)
        labs_u[kept_indices] = labs_u_k
        labs_all = labs_u[inv]
        md = _mean_within_cluster_cityblock(X, labs_all)
        mean_dist.append(md)
        clusters_by_k[k] = labs_all
        print(f"{k} groups -> mean within-cluster distance: {md:.6f}")

    result = {'k_values': k_values, 'mean_dist': mean_dist, 'clusters': clusters_by_k}
    if auto_elbow:
        result['optimal_k'] = _elbow_k(k_values, mean_dist)
        print(f"Auto-selected k={result['optimal_k']} (elbow method)")

    if plot:
        xs = np.arange(1, len(mean_dist) + 1)
        plt.figure(figsize=(7, 4.5))
        plt.plot(xs, mean_dist, marker='o')
        plt.xlabel("Sub-groups")
        plt.ylabel("Mean difference in risk contributions")
        plt.title(title)
        plt.grid(alpha=0.2)
        plt.xticks(xs, k_values)
        plt.margins(y=0)
        if auto_elbow:
            x_elb = k_values.index(result['optimal_k']) + 1
            plt.axvline(x_elb, color="red", linestyle="--", linewidth=1,
                        label=f"Auto-selected k={result['optimal_k']}")
            plt.legend()
        plt.show()

    return result
