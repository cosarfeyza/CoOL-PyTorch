import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

import torch  # before cool_ext_arma
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram

from CoolTorch.data.binary_encode_exposure_data import binary_encode_exposure_data
from CoolTorch.models.NonNegativeNN import NonNegativeNN
from CoolTorch.train.trainer import train_CoOL
from CoolTorch.explain.lrp import layerwise_relevance_propogation
from CoolTorch.plotting.sub_groups import sub_groups                    # Python (Ward)
from CoolTorch.plotting.number_of_subgroups import number_of_sub_groups  # Python (Ward)
from CoolTorch.plotting.plot_train_performance import plot_train_performance
from CoolTorch.plotting.plot_neural_network import plot_neural_network
from CoolTorch.plotting.plot_roc_auc import plot_roc_auc
from CoolTorch.plotting.prevalance_and_mean_risk import prevalence_and_mean_risk
from CoolTorch.plotting.dendo_plot import plot_cluster_sizes
from CoolTorch.metrics.mean_risk_contributions_by_subgroup import mean_risk_contributions_by_subgroup

os.chdir(os.path.dirname(__file__))

path = os.path.join(PROJECT_ROOT, "CoolTorch", "data", "data_working_example_R.csv")
if not os.path.exists(path):
    path = os.path.join(PROJECT_ROOT, "data", "data_working_example_R.csv")
df = pd.read_csv(path, sep=";")
outcome_data = df.iloc[:, 0]
exposure_df, X = binary_encode_exposure_data(df.iloc[:, 1:], return_tensor=True)
Y = torch.tensor(outcome_data.values, dtype=torch.float64)

model = NonNegativeNN(n_inputs=X.shape[1], n_hidden=8, use_c=False, dtype=torch.float64)
model.initiate_neural_network(Y, seed=123)
train_CoOL(X, Y, X, Y, None, None, model, lr_list=(1e-4, 1e-5, 1e-6),
           epochs=2000, patience=100, input_parameter_reg=1e-3, monitor=False)

res = layerwise_relevance_propogation(X, model, feature_names=exposure_df.columns)
if not isinstance(res, pd.DataFrame):
    res = pd.DataFrame(res, columns=list(exposure_df.columns) + ["Baseline_risk"])

# --- PYTHON elbow -> number of clusters (same rule as the R auto-elbow) ---
nos = number_of_sub_groups(res, low_number=2, high_number=6, ipw=1, plot=False)
md, kv = np.array(nos['mean_dist']), np.array(nos['k_values'])
k = int(kv[np.argmax(np.diff(np.diff(md))) + 1])
print(f"\nPython elbow selected k={k}")

# --- PYTHON clustering labels ---
labels = np.asarray(sub_groups(res, number_of_subgroups=k, ipw=1)).astype(int)
print("sizes:", {int(g): int((labels == g).sum()) for g in np.unique(labels)})

# --- same summary layout as CoOL_default, driven by the Python labels ---
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.6])

ax1 = fig.add_subplot(gs[0, 0]); plot_train_performance(model.train_performance, ax=ax1, show=False)
ax2 = fig.add_subplot(gs[0, 1]); plot_neural_network(model=model, names=exposure_df.columns.tolist(), ax=ax2, show=False)
ax3 = fig.add_subplot(gs[0, 2]); plot_roc_auc(outcome_data=outcome_data, exposure_data=exposure_df, model=model, ax=ax3, show=False)
ax4 = fig.add_subplot(gs[1, 0]); prevalence_and_mean_risk(risk_contributions=res, sub_groups=labels, ipw=1, ax=ax4, show=False)

# dendrogram from the Python Ward linkage (unique rows, for speed)
ax_d = fig.add_subplot(gs[1, 1])
U = np.unique(res.values, axis=0)
Z = linkage(pdist(U, metric="cityblock"), method="ward")
dendrogram(Z, no_labels=True, ax=ax_d, color_threshold=0)
ax_d.set_title("CoOL dendrogram (Python, Ward)")

ax_sizes = fig.add_subplot(gs[1, 2]); plot_cluster_sizes(labels, title="Sub-group sizes", ax=ax_sizes)

ax5 = fig.add_subplot(gs[2, :])
mrcs = mean_risk_contributions_by_subgroup(risk_contributions=res, sub_groups=labels,
        exposure_data=exposure_df, outcome_data=outcome_data, model=model, ax=ax5,
        show=False, text_fontsize=6, header_fontsize=7)
print(mrcs)

fig.suptitle("Working example — CoOL summary (PYTHON clustering, Ward)", fontsize=12)
fig.tight_layout()
fig.savefig("we_cool_summary_python.png", dpi=200, bbox_inches="tight")
print("saved we_cool_summary_python.png")
