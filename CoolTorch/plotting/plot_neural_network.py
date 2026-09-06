import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def plot_neural_network(model, names, arrow_size=None,
                        title="Model connection weights and intercepts",
                        ax=None, show=True):
    W1 = model.fc1.weight.detach().cpu().numpy()
    a  = model.fc1.bias.detach().cpu().numpy().reshape(-1)
    W2 = model.fc2.weight.detach().cpu().numpy().reshape(-1)
    b0 = float(model.fc2.bias.detach().cpu().reshape(-1)[0])

    H, D = W1.shape
    assert len(names) == D, f"names length {len(names)} != n_inputs {D}"

    w1_max = float(np.max(np.abs(W1))) if W1.size else 1.0
    if arrow_size is None:
        arrow_size = 5.0 / (w1_max + 1e-12)

    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
        created_ax = True
    else:
        fig = ax.figure

    x_in, x_hid, x_out = 1.0, 2.0, 3.0
    y_inputs = -np.arange(1, D + 1)
    y_hidden = -np.arange(1, H + 1)
    y_out = -(H + 1) / 2.0

    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(-max(D, H) - 1, 0.5)
    ax.axis('off')
    ax.set_title(title, fontsize=12)

    def add_edge(x0, y0, x1, y1, w, color_pos="dodgerblue", scale=1.0):
        if w == 0:
            return
        lw = max(0.4, abs(w) * scale)
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-',
                                     linewidth=lw, color=color_pos, alpha=0.9))

    for h in range(H):
        add_edge(x_hid, y_hidden[h], x_out, y_out, W2[h], color_pos="lightgrey", scale=4.0)

    for g in range(D):
        for h in range(H):
            add_edge(x_in, y_inputs[g], x_hid, y_hidden[h], W1[h, g],
                     color_pos="dodgerblue", scale=arrow_size)

    for i in range(D):
        ax.text(x_in - 0.05, y_inputs[i], names[i], ha='right', va='center', fontsize=9)

    for h in range(H):
        ax.text(x_hid, y_hidden[h] + 0.3, f"a={a[h]:.2f}", ha='center', va='bottom', fontsize=8)

    ax.text(x_out, y_out - 0.1, r"$R^{b+}=$" + f"{b0:.2f}", ha='center', va='top', fontsize=10)

    if created_ax and show:
        plt.show()

    return fig, ax
