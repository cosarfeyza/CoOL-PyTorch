# Backend: pure Python/matplotlib (no R). R equivalent: CoOL_9_visualised_mean_risk_contributions_legend (CoOL_functions.R)
# Draws just the colour bar for visualised_mean_risk_contributions.py's heatmap,
# with its axis labelled in real risk-contribution units (0 to max(results)) so
# the heatmap's colours can be read as actual values.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def visualised_mean_risk_contributions_legend(results, restore_par_options=True):
    if hasattr(results, "values"):
        mat = results.values
    else:
        mat = np.asarray(results)

    max_val = np.max(mat)

    farver = LinearSegmentedColormap.from_list(
        "farver",
        ["white", "orange", "orange", "red", "red", "red", "black"],
        N=100
    )

    fig, ax = plt.subplots(figsize=(2.0, 6.0))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 100)

    ax.set_xlabel("")
    ax.set_ylabel("Average risk contributions")

    y_ticks = np.linspace(0, 100, 5)
    labels = np.round(np.linspace(0, max_val, 5), 2)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(labels)
    ax.set_xticks([])

    for i in range(1, 101):
        ax.add_patch(
            plt.Rectangle(
                (0, i - 1),
                1,
                1,
                facecolor=farver(i - 1),
                edgecolor=farver(i - 1),
            )
        )

    # intentionally NOT inverting — invert_yaxis() was flipping low/high risk labels
    plt.tight_layout()
    plt.show()
