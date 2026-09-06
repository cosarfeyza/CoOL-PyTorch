import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def _running_mean(y, spline_df):
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 3:
        return np.arange(1, n + 1), y
    window = max(3, n // spline_df)
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window) / window
    y_pad = np.pad(y, (window // 2, window // 2), mode="edge")
    y_smooth = np.convolve(y_pad, kernel, mode="valid")
    return np.arange(1, n + 1), y_smooth


def plot_monitor_block(train_perf, weight_perf, baseline_risk, y_train_mean, spline_df=10):
    train_perf    = np.asarray(train_perf, dtype=float)
    weight_perf   = np.asarray(weight_perf, dtype=float)
    baseline_risk = np.asarray(baseline_risk, dtype=float)
    epochs = np.arange(1, len(train_perf) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax0, ax1, ax2 = axes

    q90 = np.quantile(train_perf, 0.9)
    ax0.plot(epochs, train_perf, color="black")
    ax0.set_ylim(train_perf.min(), q90)
    ax0.set_xlabel("Epochs")
    ax0.set_ylabel("Mean squared error")
    ax0.set_title("Performance on training data set")
    x_s, y_s = _running_mean(train_perf, spline_df)
    ax0.plot(x_s, y_s, "r-", linewidth=2)
    ax0.ticklabel_format(style="plain", axis="y")
    ax0.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.6f"))

    weight_safe = np.clip(weight_perf, 1e-16, None)
    logw = np.log(weight_safe)
    ax1.plot(epochs, logw, color="black")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("log of mean squared weight difference")
    ax1.set_title("Log mean squared weight difference")
    finite_mask = np.isfinite(logw)
    if finite_mask.sum() > 3:
        x_f = epochs[finite_mask]
        x_s2, y_s2 = _running_mean(logw[finite_mask], spline_df)
        ax1.plot(x_f, y_s2, "r-", linewidth=2)
    ax1.ticklabel_format(style="plain", axis="y")
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.6f"))

    ax2.plot(epochs, baseline_risk, color="black")
    ax2.axhline(y=y_train_mean, linestyle="--", color="gray")
    ax2.set_xlabel("Epochs")
    ax2.set_title("Estimated baseline risk by epoch")
    x_s3, y_s3 = _running_mean(baseline_risk, spline_df)
    ax2.plot(x_s3, y_s3, "r-", linewidth=2)
    ax2.ticklabel_format(style="plain", axis="y")
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.6f"))

    fig.tight_layout()
    plt.show()
