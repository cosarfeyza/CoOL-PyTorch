import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def _to_np(a):
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    elif hasattr(a, "values"):
        a = a.values
    return np.asarray(a)


def prevalence_and_mean_risk(risk_contributions, sub_groups,
                              title="Prevalence and mean risk\nof sub-groups",
                              y_max=None, colours=None, ipw=1,
                              ci_lo=None, ci_hi=None,
                              *, ax=None, show=True, return_stats=False):
    RC = risk_contributions
    RC_np = _to_np(RC)
    g = _to_np(sub_groups).astype(int).reshape(-1)
    n = RC_np.shape[0]

    if ipw is None or _to_np(ipw).reshape(-1).size != n:
        w = np.ones(n, dtype=float)
        print("Equal weights are applied (assuming no selection bias)")
    else:
        w = _to_np(ipw).reshape(-1).astype(float)

    uniq = np.unique(g)
    if not np.array_equal(uniq, np.arange(1, uniq.size + 1)):
        remap = {v: i + 1 for i, v in enumerate(sorted(uniq))}
        g = np.vectorize(remap.get)(g)
    K = int(g.max()) if g.size else 0

    risks = np.zeros(K, dtype=float)
    prevs = np.zeros(K, dtype=float)
    cols = getattr(RC, "columns", None) if hasattr(RC, "columns") else None

    for i in range(1, K + 1):
        mask = (g == i)
        if not np.any(mask):
            continue
        prevs[i - 1] = float(np.sum(w[mask]) / np.sum(w))
        risks[i - 1] = float(np.nansum(np.nanmean(RC_np[mask, :], axis=0)))

    ylim_top = max((risks.max() * 1.1) if y_max is None else float(y_max), 1e-6)

    if colours is None:
        colours = ["#B0B0B0", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                   "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    if len(colours) < K:
        colours = (colours * ((K + len(colours) - 1) // len(colours)))[:K]

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3))
    else:
        fig = ax.figure

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, ylim_top)
    ax.set_xlabel("Prevalence")
    ax.set_ylabel("Risk")
    ax.set_title(title, fontsize=11)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0.0, ylim_top, 5))
    ax.add_patch(Rectangle((0, 0), 1.0, ylim_top, fill=False, edgecolor="black"))

    x_left = 0.0
    x_centers = []
    for i in range(K):
        if prevs[i] > 0 and risks[i] > 0:
            ax.add_patch(Rectangle((x_left, 0.0), prevs[i], risks[i], color=colours[i]))
        x_centers.append(x_left + prevs[i] / 2)
        x_left += prevs[i]

    if ci_lo is not None and ci_hi is not None:
        lo = np.asarray(ci_lo, dtype=float)
        hi = np.asarray(ci_hi, dtype=float)
        for i in range(K):
            ax.errorbar(x_centers[i], risks[i],
                        yerr=[[risks[i] - lo[i]], [hi[i] - risks[i]]],
                        fmt="none", color="black", capsize=4, linewidth=1.2, capthick=1.2)

    baseline_mean = None
    if cols is not None and "Baseline_risk" in cols:
        baseline_mean = float(np.mean(np.asarray(RC["Baseline_risk"])))
        if baseline_mean <= ylim_top:
            ax.plot([0, 1], [baseline_mean, baseline_mean], linestyle="--", linewidth=1, color="black")

    fig.tight_layout()
    if show:
        plt.show()

    if return_stats:
        return {"prevalence": prevs, "risk": risks, "baseline_mean": baseline_mean, "K": K}
