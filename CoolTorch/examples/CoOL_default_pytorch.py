"""
Fully R-free / C++-free version of CoOL_default — experimental alternative to
examples/CoOL_default_R.py.

Both stages that CoOL_default_R delegates to an external backend are replaced here:
  - Training:   train_CoOL()      -> C++ (cool_ext_arma)   =>  pytorch_train_CoOL() -> pure PyTorch
  - Clustering: CoOL_6_sub_groups -> R (ClustGeo)           =>  sub_groups()         -> scipy Ward

Model, LRP and the summary plots are otherwise identical. Use this file to run the
pipeline without needing the compiled C++ extension or an R/ClustGeo install.

Like CoOL_default_R, this draws but does not save: it returns the figure objects
and leaves .savefig(...) to the caller.
"""

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from CoolTorch.explain.lrp import layerwise_relevance_propogation
from CoolTorch.plotting.plot_roc_auc import plot_roc_auc
from CoolTorch.plotting.plot_neural_network import plot_neural_network
from CoolTorch.metrics.sub_groups import sub_groups
from CoolTorch.plotting.prevalance_and_mean_risk import prevalence_and_mean_risk
from CoolTorch.metrics.mean_risk_contributions_by_subgroup import mean_risk_contributions_by_subgroup
from CoolTorch.models.NonNegativeNN import NonNegativeNN
from CoolTorch.data.binary_encode_exposure_data import binary_encode_exposure_data
from CoolTorch.data.working_example import cool_working_example
from CoolTorch.plotting.plot_train_performance import plot_train_performance
from CoolTorch.plotting.dendo_plot import plot_cluster_sizes
from CoolTorch.plotting.subgroup_profile_heatmaps import subgroup_profile_heatmaps
from CoolTorch.metrics.number_of_subgroups import number_of_sub_groups
from CoolTorch.train.pytorch_trainer import pytorch_train_CoOL


def CoOL_default_pytorch(
        data,
        num_sub_groups=3,
        low_number=2,
        high_number=6,
        input_parameter_reg=1e-3,
        hidden=10,
        monitor=False,
        epochs=10000):

    device = "cpu"
    dtype = torch.float64

    outcome_data = data.iloc[:, 0]
    exposure_df, exposure_tensor = binary_encode_exposure_data(
        data.iloc[:, 1:],
        return_tensor=True,
        device=device
    )
    outcome_tensor = torch.tensor(outcome_data.values, dtype=dtype, device=device)

    model = NonNegativeNN(
        n_inputs=exposure_tensor.shape[1],
        n_hidden=hidden,
        use_c=False,
        dtype=torch.float64
    ).to(device)

    model.initiate_neural_network(outcome_tensor, seed=123456)

    #Three-phase learning rate schedule (same as CoOL_default)
    pytorch_train_CoOL(
        X_train=exposure_tensor,
        Y_train=outcome_tensor,
        X_test=exposure_tensor,
        Y_test=outcome_tensor,
        C_train=None,
        C_test=None,
        model=model,
        lr_list=(1e-4,),
        epochs=epochs,
        patience=200,
        input_parameter_reg=input_parameter_reg,
        monitor=monitor
    )

    pytorch_train_CoOL(
        X_train=exposure_tensor,
        Y_train=outcome_tensor,
        X_test=exposure_tensor,
        Y_test=outcome_tensor,
        C_train=None,
        C_test=None,
        model=model,
        lr_list=(1e-5,),
        epochs=epochs,
        patience=100,
        input_parameter_reg=input_parameter_reg,
        monitor=monitor
    )

    pytorch_train_CoOL(
        X_train=exposure_tensor,
        Y_train=outcome_tensor,
        X_test=exposure_tensor,
        Y_test=outcome_tensor,
        C_train=None,
        C_test=None,
        model=model,
        lr_list=(1e-6,),
        epochs=epochs,
        patience=50,
        input_parameter_reg=input_parameter_reg,
        monitor=monitor
    )

    
    risk_contributions = layerwise_relevance_propogation(
        exposure_tensor,
        model,
        exposure_df.columns
    )

    # auto-select k via the elbow scan — runs exactly once (only when k not given)
    if num_sub_groups is None:
        scan = number_of_sub_groups(
            risk_contributions, low_number=low_number, high_number=high_number,
            ipw=1, plot=False, auto_elbow=True,
        )
        num_sub_groups = scan["optimal_k"]
        print(f"Auto-selected number of sub-groups: {num_sub_groups} "
              f"(searched k={low_number}..{high_number})")

    sub_groups_res = sub_groups(
        risk_contributions,
        number_of_subgroups=num_sub_groups,
        ipw=1
    )
    clus_labels = sub_groups_res


    plt.close("all")
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.6])

    ax1 = fig.add_subplot(gs[0, 0])
    plot_train_performance(model.train_performance, ax=ax1, show=False)

    ax2 = fig.add_subplot(gs[0, 1])
    plot_neural_network(
        model=model,
        names=exposure_df.columns.tolist(),
        ax=ax2,
        show=False
    )

    ax3 = fig.add_subplot(gs[0, 2])
    plot_roc_auc(
        outcome_data=outcome_data,
        exposure_data=exposure_df,
        model=model,
        ax=ax3,
        show=False
    )

    ax4 = fig.add_subplot(gs[1, 0])
    prevalence_and_mean_risk(
        risk_contributions=risk_contributions,
        sub_groups=sub_groups_res,
        ipw=1,
        ax=ax4,
        show=False
    )

    # No R/ClustGeo dendrogram PNG in this pure-Python path, so this panel is left
    # empty rather than showing something misleading.
    ax_dendo = fig.add_subplot(gs[1, 1])
    ax_dendo.set_visible(False)

    ax_sizes = fig.add_subplot(gs[1, 2])
    plot_cluster_sizes(
        clus_labels,
        title="Sub-group sizes",
        ax=ax_sizes
    )

    ax5 = fig.add_subplot(gs[2, :])
    mrcs = mean_risk_contributions_by_subgroup(
        risk_contributions=risk_contributions,
        sub_groups=sub_groups_res,
        exposure_data=exposure_df,
        outcome_data=outcome_data,
        model=model,
        ax=ax5,
        show=True,
        text_fontsize=6,
        header_fontsize=7,
    )
    print(mrcs)

    fig.tight_layout()

    fig_heatmaps, (ax_prev, ax_lrp) = plt.subplots(1, 2, figsize=(14, 7))
    subgroup_profile_heatmaps(
        risk_contributions=risk_contributions,
        exposure_df=exposure_df,
        sub_groups=sub_groups_res,
        top_n=10,
        ax_prevalence=ax_prev,
        ax_lrp=ax_lrp,
        show=False,
    )
    fig_heatmaps.tight_layout()

    # Nothing is written to disk here, matching CoOL_default_R and the R original:
    # this draws, it does not save. A caller that wants a file calls
    # fig.savefig(...) / fig_heatmaps.savefig(...) on what is returned.
    return {
        "model": model,
        "risk_contributions": risk_contributions,
        "sub_groups": sub_groups_res,
        "figure": fig,
        "subgroup_profiles_figure": fig_heatmaps,
    }


def main():
    data = cool_working_example(n=10000, seed=1)
    res = CoOL_default_pytorch(data, num_sub_groups=3)
    res["figure"].savefig("pytorch_cool_default_summary.png", dpi=250, bbox_inches="tight")
    res["subgroup_profiles_figure"].savefig("pytorch_cool_subgroup_profiles.png", dpi=250, bbox_inches="tight")
    print("saved pytorch_cool_default_summary.png, pytorch_cool_subgroup_profiles.png")


if __name__ == "__main__":
    main()
