import os as _os
_PKG = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # CoolTorch package root
import os
import pandas as pd
import torch

from CoolTorch.models.NonNegativeNN import NonNegativeNN
from CoolTorch.train.trainer import train_CoOL
from CoolTorch.data.binary_encode_exposure_data import binary_encode_exposure_data 
from CoolTorch.plotting.plot_roc_auc import plot_roc_auc
from CoolTorch.plotting.prevalance_and_mean_risk import prevalence_and_mean_risk
from CoolTorch.plotting.dendo_clustgeo import CoOl_dendrogram_clustgeo
from CoolTorch.plotting.number_of_sub_groups_clustgeo import CoOL_6_number_of_sub_groups
from CoolTorch.plotting.sub_groups_clustgeo import CoOL_6_sub_groups
from CoolTorch.tests.load_pre_trained_model import load_cool_from_r
from CoolTorch.explain.lrp import layerwise_relevance_propogation
from CoolTorch.plotting.plot_roc_auc import plot_roc_auc
from CoolTorch.plotting.plot_neural_network import plot_neural_network
from CoolTorch.train.predict_risks import predict_risks
from CoolTorch.plotting.sub_groups import sub_groups
from CoolTorch.plotting.number_of_subgroups import number_of_sub_groups
from CoolTorch.plotting.calibration_plot import calibration_plot
from CoolTorch.plotting.prevalance_and_mean_risk import prevalence_and_mean_risk
from CoolTorch.metrics.individual_effects_matrix import individual_effects_matrix
from CoolTorch.metrics.sum_of_individual_effects import sum_of_individual_effects
from CoolTorch.metrics.mean_risk_contributions_by_subgroup import mean_risk_contributions_by_subgroup
from CoolTorch.metrics.visualised_mean_risk_contributions import visualised_mean_risk_contributions
from CoolTorch.metrics.visualised_mean_risk_contributions_legend import visualised_mean_risk_contributions_legend
from CoolTorch.plotting.dendo_plot import cool_cluster_and_dendrogram
from CoolTorch.train.trainer import train_CoOL
from CoolTorch.plotting.dendo_plot import plot_cluster_sizes
from CoolTorch.plotting.dendo_clustgeo import CoOl_dendrogram_clustgeo
from CoolTorch.plotting.number_of_sub_groups_clustgeo import CoOL_6_number_of_sub_groups
from CoolTorch.plotting.sub_groups_clustgeo import CoOL_6_sub_groups


r_data = pd.read_csv(
    _os.path.join(_PKG, "data/data_working_example_R.csv"),
    sep=";"
)
df = pd.DataFrame(r_data)

outcome_data = df.iloc[:, 0]
exposure_raw = df.iloc[:, 1:]

device = "cpu"
dtype = torch.float64

# 2) X'i binary encode et
exposure_df, X_tensor = binary_encode_exposure_data(
    exposure_raw,
    return_tensor=True,
    device=device
)
print(exposure_df.head())

Y_tensor = torch.tensor(outcome_data.values, dtype=dtype, device=device)

n_inputs = X_tensor.shape[1]
hidden = 8
epochs = 2000


model = NonNegativeNN(
    n_inputs=n_inputs,
    n_hidden=hidden,
    use_c=False,
    dtype=dtype
).to(device)

model.initiate_neural_network(Y_tensor, seed=123)

#print("=== INITIALIZED MODEL (CoOL-style) ===")
#print("fc1.weight mean:", model.fc1.weight.data.mean().item())
#print("fc1.weight min :", model.fc1.weight.data.min().item())
#print("fc1.weight max :", model.fc1.weight.data.max().item())
#print("fc1.bias mean  :", model.fc1.bias.data.mean().item())
#print("fc2.weight mean:", model.fc2.weight.data.mean().item())
#print("fc2.bias (baseline):", model.fc2.bias.data.item())
#print("Y mean:", Y_tensor.mean().item())
#print("======================================")


logs = train_CoOL(
    X_train=X_tensor,
    Y_train=Y_tensor,
    X_test=X_tensor,
    Y_test=Y_tensor,
    C_train=None,
    C_test=None,
    model=model,
    lr_list=(1e-4, 1e-5, 1e-6),
    epochs=2000,
    patience=100,
    input_parameter_reg=1e-3,
    drop_out=0,
    fix_baseline_risk=-1.0,
    ipw=1,
    dtype=dtype,
    device=device,
    spline_df=10,
)

res = layerwise_relevance_propogation(
        X_tensor,
        model,
        feature_names=exposure_df.columns)
    
    #print(outcome_data.head())
    #print(exposure_data.head())
   
   
    #plot_neural_network(
    #    model=model,
    #    names=feature_names
    #)
    
    #pred = predict_risks(
    #    X_tensor,
    #    {
    #        "W1": model.fc1.weight.t().detach(),   # (p, h)
    #        "b1": model.fc1.bias.detach(),         # (h,)
    #        "W2": model.fc2.weight.t().detach(),   # (h, 1)
    #        "b2": model.fc2.bias.detach()          # (1,)
    #    }
    #)
    #print(pred)
    #plot_roc_auc(
    #    outcome_data=outcome_data,
    #    exposure_data=exposure_data,
    #    model=model
    #)

    #sub_groups_res = sub_groups(
    #    risk_contributions=res,
    #    number_of_subgroups=3,
    #    ipw=1
    #)
    #print(">> sub_groups_res shape:", sub_groups_res.shape)
    #np.set_printoptions(threshold=np.inf)  # show everything
    #print(sub_groups_res)
    
    #done
    #number_of_sub_groups_res = number_of_sub_groups(
    #    risk_contributions=res["df"],
    #    low_number=1,
    #    high_number=5,
    #    ipw=1,
    #    plot=True,
    #    title="Sub-groups vs. mean difference in risk contributions"
    #)
    #print(">> number_of_sub_groups_res k_values:", number_of_sub_groups_res['k_values'])
    #print(">> number_of_sub_groups_res mean_dist:", number_of_sub_groups_res['mean_dist'])
    #for k, labels in number_of_sub_groups_res['clusters'].items():
    #   print(f">> k={k}, labels shape: {labels.shape}, unique: {sorted(pd.unique(labels))}")


sub_groups_clustgeo = CoOL_6_sub_groups(
       risk_contributions=res,
       number_of_subgroups=3,
       ipw=1
    )
number_of_sub_groups_clustgeo= CoOL_6_number_of_sub_groups(
       risk_contributions=res,
       low_number=1,
       high_number=5,
       ipw=1,
       plot=True)
    



    #done
clus = CoOl_dendrogram_clustgeo(
    risk_contributions=res,
    number_of_subgroups=3,
    title="Debdrogram with ClustGeo",
    ipw=1)
print(clus)
   
    #done
    #calibration_plot(
    #    outcome_data=outcome_data,
    #    exposure_data=exposure_data,
    #    model=model,
    #    sub_groups=sub_groups_res,
    #    title="Calibration Plot"
    #)
   
   #Done
prevalence_and_mean_risk(
        risk_contributions=res,
        sub_groups=sub_groups_clustgeo,
        title="Prevalence and mean risk\nof sub-groups",
        y_max=None,
        colours=None,
        ipw=1
    )
print(prevalence_and_mean_risk)

    #done
    #e_matrix =individual_effects_matrix(
    #   X=exposure_data,
    #     model=model,
       
    #)
    #print(e_matrix.head())
    #print(e_matrix.shape)
    #print(res['R_X'].shape)
    #print(res['R_X'])

    #done
    #sum_ef = sum_of_individual_effects(
    #    X=exposure_data,
    #    model=model,
    #   
    #)
    #print(sum_ef)
    
    #Need to look at subgrops, it seems inconsistent:done

    #done
mean_risk_contribs = mean_risk_contributions_by_subgroup(
        risk_contributions=res,
        sub_groups=sub_groups_clustgeo,
        exposure_data=exposure_df,
        outcome_data=outcome_data,
        model=model
    )
    #print(mean_risk_contribs)
    #print(">> [lrp_test] done")

    #done
    #vis =visualised_mean_risk_contributions(
    #results=mean_risk_contribs,
    #sub_groups=sub_groups_res,
    #ipw=1)

    #print(vis)

    #done
    #vis_legend =visualised_mean_risk_contributions_legend(results=mean_risk_contribs)
    #print(vis_legend)
    #print(">> [lrp_test] done")
    
    # risk_df is your risk contributions DataFrame (or torch tensor / ndarray)

