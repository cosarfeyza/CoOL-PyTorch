import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from CoolTorch.explain.lrp import layerwise_relevance_propogation
from CoolTorch.plotting.plot_roc_auc import plot_roc_auc
from CoolTorch.plotting.plot_neural_network import plot_neural_network
from CoolTorch.plotting.sub_groups_clustgeo import CoOL_6_sub_groups
from CoolTorch.plotting.prevalance_and_mean_risk import prevalence_and_mean_risk
from CoolTorch.metrics.mean_risk_contributions_by_subgroup import mean_risk_contributions_by_subgroup
from CoolTorch.train.trainer import train_CoOL
from CoolTorch.models.NonNegativeNN import NonNegativeNN
from CoolTorch.data.binary_encode_exposure_data import binary_encode_exposure_data
from CoolTorch.data.working_example import cool_working_example
from CoolTorch.plotting.plot_train_performance import plot_train_performance
from CoolTorch.plotting.dendo_clustgeo import CoOl_dendrogram_clustgeo
from CoolTorch.plotting.dendo_plot import plot_cluster_sizes
from CoolTorch.data.complex_simulation import complex_simulation
from CoolTorch.data.confounding_simulation import confounding_simulation
from CoolTorch.data.common_simulation import common_simulation
from CoolTorch.plotting.subgroup_profile_heatmaps import subgroup_profile_heatmaps
from CoolTorch.plotting.number_of_sub_groups_clustgeo import CoOL_6_number_of_sub_groups


def CoOL_default(
        data,
        num_sub_groups=None,
        low_number=2,
        high_number=6,
        exclude_below=0.01,
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

    train_CoOL(
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

    train_CoOL(
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

    train_CoOL(
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

    # === Risk contributions & Subgroups ===
    risk_contributions = layerwise_relevance_propogation(
        exposure_tensor,
        model,
        exposure_df.columns
    )

    # auto-select k via the elbow scan — runs exactly once (only when k not given)
    if num_sub_groups is None:
        _, num_sub_groups = CoOL_6_number_of_sub_groups(
            risk_contributions, low_number=low_number, high_number=high_number,
            ipw=1, plot=False, auto_elbow=True,
        )
        print(f"Auto-selected number of sub-groups: {num_sub_groups} "
              f"(searched k={low_number}..{high_number})")

    sub_groups_res = CoOL_6_sub_groups(
        risk_contributions,
        number_of_subgroups=num_sub_groups,
        ipw=1
    )

    clus_labels = CoOl_dendrogram_clustgeo(
        risk_contributions=risk_contributions,
        number_of_subgroups=num_sub_groups,
        ipw=1,
        title="CoOL risk contribution dendrogram",
        show=False,
    )

    
    plt.close("all")
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.6])

    # 1) Train performance
    ax1 = fig.add_subplot(gs[0, 0])
    plot_train_performance(model.train_performance, ax=ax1, show=False)

    # 2) Neural network
    ax2 = fig.add_subplot(gs[0, 1])
    plot_neural_network(
        model=model,
        names=exposure_df.columns.tolist(),
        ax=ax2,
        show=False
    )

    # 3) ROC AUC
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
    

    # 5) Dendrogram (row 1, col 1) — load PNG saved by ClustGeo
    ax_dendo = fig.add_subplot(gs[1, 1])
    import matplotlib.image as mpimg
    import os
    dendo_png = os.path.join("clustgeo_tmp", "dendrogram.png")
    if os.path.exists(dendo_png):
        ax_dendo.imshow(mpimg.imread(dendo_png))
        ax_dendo.axis("off")
    else:
        ax_dendo.set_visible(False)

    # 6) Cluster size bar chart (row 1, col 2)
    ax_sizes = fig.add_subplot(gs[1, 2])
    plot_cluster_sizes(
    clus_labels,
    title="Sub-group sizes",
    ax=ax_sizes
    )

    # 5) Mean risk contributions
    ax5 = fig.add_subplot(gs[2, :])
    mrcs=mean_risk_contributions_by_subgroup(
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
    fig.savefig("cool_default_summaryyyy.png", dpi=250, bbox_inches="tight")

    # === Subgroup profile heatmaps ===
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
    fig_heatmaps.savefig("cool_subgroup_profiles.png", dpi=250, bbox_inches="tight")
    plt.close(fig_heatmaps)

    return {
        "model": model,
        "risk_contributions": risk_contributions,
        "sub_groups": sub_groups_res,
        "figure": fig
    }

def main():
        
       data = cool_working_example(n=10000, seed=1)
       #data = complex_simulation(n=50000)
       #CoOL_default(data) 
       CoOL_default(data, num_sub_groups=3)

if __name__ == "__main__": 
    main()