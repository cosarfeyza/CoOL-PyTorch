import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _to_np(a):
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    elif hasattr(a, "values"):
        a = a.values
    return np.asarray(a)


def subgroup_profile_heatmaps(risk_contributions, exposure_df, sub_groups,
                               top_n=10, ax_prevalence=None, ax_lrp=None, show=True):
    g = _to_np(sub_groups).astype(int).reshape(-1)
    groups = sorted(np.unique(g))
    K = len(groups)
    group_labels = [f"Group {i}" for i in groups]

    features = exposure_df.columns.tolist()
    prev_matrix = pd.DataFrame(index=features, columns=group_labels, dtype=float)
    for i in groups:
        prev_matrix[f"Group {i}"] = exposure_df[g == i].mean(axis=0).values

    rc = risk_contributions if isinstance(risk_contributions, pd.DataFrame) else pd.DataFrame(_to_np(risk_contributions))
    lrp_matrix = pd.DataFrame(index=rc.columns.tolist(), columns=group_labels, dtype=float)
    for i in groups:
        lrp_matrix[f"Group {i}"] = rc[g == i].mean(axis=0).values

    top_prev = prev_matrix.max(axis=1).nlargest(top_n).index.tolist()
    top_lrp  = lrp_matrix.max(axis=1).nlargest(top_n).index.tolist()

    prev_plot = prev_matrix.loc[top_prev]
    lrp_plot  = lrp_matrix.loc[top_lrp]

    if ax_prevalence is None or ax_lrp is None:
        fig, (ax_prevalence, ax_lrp) = plt.subplots(1, 2, figsize=(6 + K * 1.2, top_n * 0.55 + 1.5))
    else:
        fig = ax_prevalence.figure

    def _draw_heatmap(ax, matrix, title, fmt=".0%", cmap="Blues"):
        data = matrix.values.astype(float)
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=data.max())
        ax.set_xticks(range(K))
        ax.set_xticklabels(matrix.columns, fontsize=9)
        ax.set_yticks(range(len(matrix)))
        ax.set_yticklabels(matrix.index, fontsize=8)
        ax.set_title(title, fontsize=10, pad=8)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                val = data[row, col]
                if np.isnan(val):
                    continue
                text = format(val, fmt) if fmt == ".0%" else f"{val:.3f}"
                color = "white" if val / (data.max() + 1e-9) > 0.6 else "black"
                ax.text(col, row, text, ha="center", va="center", fontsize=7, color=color)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)

    _draw_heatmap(ax_prevalence, prev_plot, "Who they are\n(feature prevalence %)", fmt=".0%", cmap="Blues")
    _draw_heatmap(ax_lrp, lrp_plot, "What drives their risk\n(mean LRP contribution)", fmt=".3f", cmap="Oranges")

    fig.tight_layout()
    if show:
        plt.show()

    return prev_plot, lrp_plot
