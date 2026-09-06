import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from CoolTorch.metrics.individual_effects_matrix import individual_effects_matrix
from CoolTorch.metrics.sum_of_individual_effects import sum_of_individual_effects


def _to_np(a):
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    elif hasattr(a, "values"):
        a = a.values
    return np.asarray(a)


def mean_risk_contributions_by_subgroup(
    risk_contributions: pd.DataFrame,
    sub_groups,
    exposure_data,
    outcome_data,
    model,
    exclude_below: float = 0.005,
    title=("Mean risk contributions by sub-group\n"
           "(Standard deviation)\n"
           "[mean risk contribution if other exposures are set to 0]"),
    colours=None,
    ipw=1,
    ax=None,
    show: bool = True,
    text_fontsize: int = 7,
    header_fontsize: int = 8,
    lrp_ci_lo: pd.DataFrame = None,
    lrp_ci_hi: pd.DataFrame = None,
    iem_ci_lo: pd.DataFrame = None,
    iem_ci_hi: pd.DataFrame = None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    RC = risk_contributions
    if not isinstance(RC, pd.DataFrame):
        RC = pd.DataFrame(_to_np(RC))

    n, K = RC.shape
    g = _to_np(sub_groups).astype(int).reshape(-1)
    y = _to_np(outcome_data).reshape(-1)

    if np.isscalar(ipw) or ipw is None:
        w = np.full(n, float(ipw) if ipw is not None else 1.0)
    else:
        w = _to_np(ipw).reshape(-1)
        if w.size != n:
            w = np.ones(n, dtype=float)
            print("Equal weights are applied (assuming no selection bias)")

    if colours is None or (isinstance(colours, float) and np.isnan(colours)):
        colours = [
            "grey", "#1f77b4", "#ff7f0e", "#2ca02c",
            "#d62728", "#9467bd", "#8c564b", "#e377c2",
            "#7f7f7f", "#bcbd22", "#17becf",
        ]

    uniq = np.unique(g)
    G = int(uniq.max()) if uniq.size else 0
    if len(colours) < G:
        colours = (colours * ((G + len(colours) - 1) // len(colours)))[:G]

    def color_for_group(i: int) -> str:
        return colours[(i - 1) % len(colours)]

    # weighted sum of group risks — used to compute excess fraction
    total = 0.0
    for i in range(1, G + 1):
        m = (g == i)
        prev = w[m].sum() / w.sum() if w.sum() > 0 else 0.0
        risk_i = RC.loc[m].mean(axis=0).sum()
        total += prev * risk_i

    d = pd.DataFrame(
        np.nan,
        index=[f"Group {i}" for i in range(1, G + 1)],
        columns=list(RC.columns),
    )
    for i in range(1, G + 1):
        m = (g == i)
        d.iloc[i - 1, :] = RC.loc[m].mean(axis=0).values

    if K >= 2:
        non_last = list(d.columns[:-1])[::-1]
        last = [d.columns[-1]]
        d = d[non_last + last]
    d.columns = list(d.columns[:-1]) + ["Baseline_risk"]

    sd_matrix = pd.DataFrame(np.nan, index=d.index, columns=d.columns)
    for i in range(1, G + 1):
        m = (g == i)
        sd_matrix.iloc[i - 1, :] = RC.loc[m, d.columns].std(axis=0, ddof=1).values

    iem = individual_effects_matrix(exposure_data, model)
    if not isinstance(iem, pd.DataFrame):
        iem = pd.DataFrame(_to_np(iem), columns=list(RC.columns))
    iem = iem.reindex(columns=RC.columns)
    iem_disp = iem.reindex(columns=d.columns)

    def _wilson_ci(succ, tot):
        if tot <= 0:
            return (np.nan, np.nan)
        z = 1.959963984540054
        p = succ / tot
        denom = 1 + z**2 / tot
        center = (p + z**2 / (2 * tot)) / denom
        half = (z / denom) * np.sqrt((p * (1 - p) / tot) + (z**2 / (4 * tot**2)))
        return (max(0.0, center - half), min(1.0, center + half))

    baseline_overall_mean = float(RC.iloc[:, -1].mean())

    sum_ie_series = sum_of_individual_effects(exposure_data, model)
    sum_ie_series = sum_ie_series.reindex(RC.index)

    left_margin = 5.0
    ax.set_xlim(-left_margin, K)
    ax.set_ylim(-(G + 1), 1)
    ax.axis("off")
    ax.set_title(title, fontsize=10)

    for j, name in enumerate(d.columns):
        ax.text(j, 0.4, name, rotation=45,
                ha="right", va="bottom", fontsize=header_fontsize)

    for i in range(1, G + 1):
        m = (g == i)
        prev = w[m].sum() / w.sum() if w.sum() > 0 else np.nan
        risk_i = RC.loc[m].mean(axis=0).sum()
        successes = float((w[m] * y[m]).sum())
        tot = float(w[m].sum())
        risk_obs = successes / tot if tot > 0 else np.nan
        lo, hi = _wilson_ci(successes, tot)

        excess = (
            prev * (risk_i - baseline_overall_mean) / total * 100.0
            if total != 0
            else np.nan
        )

        risk_sum_ie = sum_ie_series[m].mean() * 100.0 if np.any(m) else np.nan

        left_text = (
            f"Sub-group {i}: n={round(w[m].sum(),1)}, e={round(successes,1)}, "
            f"Prev={prev * 100:.1f}%, risk={risk_i * 100:.1f}%, "
            f"excess={excess:.1f}%\n"
            f"Obs risk={risk_obs * 100:.1f}% "
            f"({lo * 100:.1f}-{hi * 100:.1f}%), "
            f"Risk (sum of individual effects)={risk_sum_ie:.1f}%"
        )

        ax.text(-left_margin + 0.2, -i,
                left_text, ha="left", va="center",
                color=color_for_group(i), fontsize=text_fontsize)

        for j in range(K):
            val = float(d.iloc[i - 1, j])
            sdv = float(sd_matrix.iloc[i - 1, j])
            col = d.columns[j]

            if np.any(m):
                if col == "Baseline_risk":
                    ie_mean = float(iem_disp.loc[m, col].mean())
                else:
                    # mask by whether the exposure is actually present (0/1)
                    ie_mean = float((iem_disp.loc[m, col] * exposure_data.loc[m, col]).mean())
            else:
                ie_mean = np.nan

            grp_key = f"Group {i}"

            lrp_ci_line = ""
            if lrp_ci_lo is not None and lrp_ci_hi is not None:
                if grp_key in lrp_ci_lo.index and col in lrp_ci_lo.columns:
                    lo_v = float(lrp_ci_lo.at[grp_key, col])
                    hi_v = float(lrp_ci_hi.at[grp_key, col])
                    lrp_ci_line = f"\n{lo_v*100:.1f}-{hi_v*100:.1f}%"

            iem_ci_line = ""
            if iem_ci_lo is not None and iem_ci_hi is not None:
                if grp_key in iem_ci_lo.index and col in iem_ci_lo.columns:
                    lo_v = float(iem_ci_lo.at[grp_key, col])
                    hi_v = float(iem_ci_hi.at[grp_key, col])
                    iem_ci_line = f"\n{lo_v*100:.1f}-{hi_v*100:.1f}%"

            cell_text = (
                f"{val * 100:.1f}%"
                f"{lrp_ci_line}\n"
                f"({sdv * 100:.1f}%)\n"
                f"[{ie_mean * 100:.1f}%]"
                f"{iem_ci_line}"
            )
            alpha = 0.0 if (np.isnan(val) or val < exclude_below) else 1.0

            ax.text(
                j,
                -i + 0.1,
                cell_text,
                ha="center",
                va="center",
                color=color_for_group(i),
                alpha=alpha,
                fontsize=text_fontsize,
            )

    fig.tight_layout()
    if show:
        plt.show()

    return d.T
