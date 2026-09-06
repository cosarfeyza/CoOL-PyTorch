import numpy as np
import matplotlib.pyplot as plt


def plot_train_performance(train_perf, title="Performance - training data", ax=None, show=True):
    y = np.asarray(train_perf, dtype=float).ravel()
    x = np.arange(1, len(y) + 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))
    else:
        fig = ax.figure

    ax.plot(x, y, color="black", linestyle="-", linewidth=1)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Mean squared error")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(float(np.min(y)), float(np.quantile(y, 0.9)))
    ax.grid(False)
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax
