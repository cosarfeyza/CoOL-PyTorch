# Backend: pure Python/scipy (no R). R equivalent: CoOL_6_sub_groups (CoOL_functions.R)
# Assigns each individual to a sub-group by Ward-clustering the risk-contribution
# rows (weighted by ipw), then renumbers the clusters low-to-high by mean risk.
import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster


def _to_np(a):
    if hasattr(a, "detach"): a = a.detach().cpu().numpy()
    if hasattr(a, "values"): a = a.values
    return np.asarray(a, dtype=float)


def _collapse_with_weights(X, w):
    Xc = np.ascontiguousarray(X)
    dt = np.dtype((np.void, Xc.dtype.itemsize * Xc.shape[1]))
    packed = Xc.view(dt).ravel()
    _, idx_first, inv = np.unique(packed, return_index=True, return_inverse=True)
    Xu = Xc[idx_first]
    wu = np.bincount(inv, weights=w, minlength=Xu.shape[0])
    return Xu, wu, inv


def sub_groups(risk_contributions, number_of_subgroups=3, ipw=None):
    X = _to_np(risk_contributions)
    n = X.shape[0]
    if ipw is None or (np.asarray(ipw).size != n):
        print("Equal weights are applied (assuming no selection bias)")
        w = np.ones(n, float)
    else:
        w = _to_np(ipw).reshape(-1)

    Xu, wu, inv = _collapse_with_weights(X, w)

    reps = np.rint(wu).astype(int)
    keep = reps > 0
    if not keep.any():
        return np.ones(n, int)

    Xu_k, reps_k = Xu[keep], reps[keep]
    Xexp = np.repeat(Xu_k, reps_k, axis=0)

    D = pdist(Xexp, metric="cityblock")
    Z = linkage(D, method="ward")
    labs_exp = fcluster(Z, number_of_subgroups, criterion="maxclust")

    exp_to_uniq = np.concatenate([np.full(r, i, int) for i, r in enumerate(reps_k)])
    labs_u_k = np.array([np.bincount(labs_exp[exp_to_uniq==i]).argmax() for i in range(len(reps_k))])
    labs_u = np.ones(len(reps), int)
    labs_u[np.where(keep)[0]] = labs_u_k
    clus = labs_u[inv]

    k = int(number_of_subgroups)
    scores = np.full(k, np.inf)
    for i in range(1, k + 1):
        Xi = X[clus == i]
        if Xi.size:
            scores[i - 1] = Xi.mean(axis=0).sum()
    order = np.argsort(scores) + 1
    remap = {orig: new for new, orig in enumerate(order, start=1)}

    return np.array([remap[c] for c in clus], int)
