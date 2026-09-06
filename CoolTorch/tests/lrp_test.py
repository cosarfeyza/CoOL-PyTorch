import os as _os
_PKG = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # CoolTorch package root
import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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




DIR_EXPOSURE = _os.path.join(_PKG, "tests/R_data/exposure_data_R.csv")
DIR_OUTCOME  = _os.path.join(_PKG, "tests/R_data/outcome_data_R.csv")

'''
Function to compute LRP on a pre-trained NonNegativeNN model loaded from R,
using exposure data from a CSV file. It handles robust reading of the CSV,
aligns columns to the model's expected features, runs LRP, and returns results.

Returns:
    dict: {
        'R_X': (N, D) contribution of each input feature without baseline risk,
        'Baseline_risk': (N,) baseline contribution,
        'o_all': (N,) model outputs,
        'residual': (N,) difference o_all - (Baseline + sum(R_X)),
        'df': (optional) pandas DataFrame of contributions if return_dataframe=True
'''
def _smart_read_csv(p):
        try:
            df = pd.read_csv(p, sep=None, engine="python")
        except Exception:
            df = pd.read_csv(p)
        
        if df.shape[1] == 1:
            try:
                df = pd.read_csv(p, sep=";")
            except Exception:
                pass
        if df.shape[1] == 1:
            s = df.iloc[:, 0].astype(str)
            parts = s.str.split(",", expand=True)
            if parts.shape[1] > 1:
                df = parts
            else:
                parts = s.str.split(r"\s+", expand=True)
                if parts.shape[1] > 1:
                    df = parts
        return df

def compute_lrp():
    """
    Loads the trained model and data, reads the exposure CSV robustly,
    aligns columns to `feature_names`, runs LRP, and returns the results dict.
    """

    # 1) Get model and the expected feature names
    model, feature_names = load_cool_from_r()
    print(">> model.in_features:", getattr(model.fc1, "in_features", None))
    print(">> expected features:", feature_names)

    # 2) Read CSV 


    X_df = _smart_read_csv(DIR_EXPOSURE)

    # Drop index-like columns such as "Unnamed: 0"
    X_df = X_df.loc[:, ~X_df.columns.astype(str).str.match(r"^Unnamed")]
    X_df = X_df.apply(pd.to_numeric, errors="coerce")

    print(">> raw X_df shape after read:", X_df.shape)

    # 3) Align to the model's expected number/order of columns
    expected = len(feature_names)
    if X_df.shape[1] != expected:
        if X_df.shape[1] > expected:
            # If there are extra columns, trim to the expected count
            X_df = X_df.iloc[:, :expected]
        else:
            raise ValueError(
                f"Exposure data has {X_df.shape[1]} columns but model expects {expected}. "
                f"Check separator/header of {DIR_EXPOSURE}"
            )
    X_df.columns = list(feature_names)
    X_df = X_df.fillna(0.0)

    # 4) Run LRP
    X = torch.tensor(X_df.values, dtype=torch.float32)
    res = layerwise_relevance_propogation(X, model)



    # 5) Additivity check: o ≈ R_b + sum(R_X)
    R_X          = res["R_X"]                # (N, D)
    Baseline_risk = res["Baseline_risk"]     # (N,)
    o_all        = res["o_all"]              # (N,)
    residual     = res["residual"]           # (N,)
    
    total = R_X.sum(dim=1) + Baseline_risk
    max_gap = (o_all - total).abs().max().item()
    #print(">> X_df final shape:", X_df.shape)
    #print("max |o_all - (Baseline + Σ R_X)| =", max_gap)
    cols = list(X_df.columns) + ["Baseline_risk"]
    df_out = pd.DataFrame(
    torch.cat([R_X, Baseline_risk.unsqueeze(1)], dim=1).detach().cpu().numpy(),
    columns=cols)
    #print(df_out.head())                                                                                               

    return {
    **res,
    "df": df_out }
    
def run_equal_angle_dendrogram(res):
    """
    Draws an equal-angle dendrogram from the LRP risk contribution matrix
    and saves a PNG. Shows only 3 collapsed cluster tips.
    """
    if "df" not in res:
        raise RuntimeError("LRP result does not include 'df'. Make sure return_dataframe=True in lrp_cool_nonneg.")

    rx_df = res["df"].copy()

    # Exclude baseline/bias-like columns from clustering
    drop_like = [c for c in rx_df.columns
                 if c.lower().startswith("baseline") or "bias" in c.lower() or c.lower() in {"r_b", "rb"}]
    feature_cols = [c for c in rx_df.columns if c not in drop_like]
    if not feature_cols:
        raise ValueError("No feature columns found for dendrogram.")
    print(">> feature_cols:", feature_cols)

    risk_contribs = rx_df[feature_cols]
    print(">> risk_contribs shape:", risk_contribs.shape)

    


    # Plot
    labels, Z, order = cool_dendrogram_equal_angle(
        risk_contributions=risk_contribs,
        number_of_subgroups=3,
        title="Dendrogram",
        colours=["#F98400", "#00A08A", "#5BBCD6"],  
        tip_alpha=0.25,
        tip_size_scale=0.8,
        node_linewidth=1.8,
        rotate_deg=90,
        mirror=True,
        total_rep_target=600,
        plot=True,
        verbose=True,
        collapse_to_subgroups=True,  
    )

    # Add cluster labels + save
    rx_df["cluster"] = labels
    print(">> labels shape:", labels.shape, "unique:", sorted(pd.unique(labels)))
    print("Clusters\n", pd.Series(labels).value_counts().sort_index())

    plt.savefig("dendrogram_equal_angle.png", dpi=160, bbox_inches="tight")
    print(">> saved: dendrogram_equal_angle.png")


def main():
    print(">> [lrp_test] start")
    
    #run_equal_angle_dendrogram(res)
    model, feature_names = load_cool_from_r()

    

    #Test plot of the NN structure
    outcome_data= _smart_read_csv(DIR_OUTCOME)
    exposure_data= _smart_read_csv(DIR_EXPOSURE)

    Y_tensor = torch.tensor(outcome_data.values, dtype=torch.float32)
    X_tensor = torch.tensor(exposure_data.values, dtype=torch.float32)

    res = layerwise_relevance_propogation(
        X_tensor,
        model,
        feature_names)
    print(res.head())
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
    #    print(f">> k={k}, labels shape: {labels.shape}, unique: {sorted(pd.unique(labels))}")

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
    ipw=1,)
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
   # prevalence_and_mean_risk(
    #    risk_contributions=res["df"],
    #    sub_groups=sub_groups_res,
    #    title="Prevalence and mean risk\nof sub-groups",
    #    y_max=None,
    #    colours=None,
    #    ipw=1
    #)

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
    #mean_risk_contribs = mean_risk_contributions_by_subgroup(
    #    risk_contributions=res,
    #    sub_groups=sub_groups_res,
    #    exposure_data=exposure_data,
    #    outcome_data=outcome_data,
    #    model=model
    #)
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


    '''
    print("=== PARAMETERS LOADED FROM R MODEL ===")
    print("fc1.weight mean:", model.fc1.weight.data.mean().item())
    print("fc1.weight min :", model.fc1.weight.data.min().item())
    print("fc1.weight max :", model.fc1.weight.data.max().item())

    print("fc1.bias mean  :", model.fc1.bias.data.mean().item())
    print("fc1.bias min   :", model.fc1.bias.data.min().item())
    print("fc1.bias max   :", model.fc1.bias.data.max().item())

    print("fc2.weight mean:", model.fc2.weight.data.mean().item())
    print("fc2.weight min :", model.fc2.weight.data.min().item())
    print("fc2.weight max :", model.fc2.weight.data.max().item())

    print("fc2.bias (baseline):", model.fc2.bias.data.item())

    if model.use_c:
        print("c2 parameter:", model.c2.data.item())
    else:
        print("c2 unused")
    print("==========================================")

    
    logs = train_CoOL(
        X_tensor, Y_tensor,
        X_tensor, Y_tensor,
        C_train=None, C_test=None,
        model=model,
        lr_list=(1e-4, 1e-5, 1e-6),
        epochs=2000,
        patience=100,
        plot_and_evaluation_frequency=50,
        input_parameter_reg=1e-6,
        drop_out=0,
        fix_baseline_risk=-1.0,
        ipw=None,
        monitor=True,
        spline_df=10,
)
'''

  


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("!! [lrp_test] exception:", repr(e))
        traceback.print_exc()
