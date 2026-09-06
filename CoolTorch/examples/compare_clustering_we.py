import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

import torch  # before cool_ext_arma
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from CoolTorch.data.binary_encode_exposure_data import binary_encode_exposure_data
from CoolTorch.models.NonNegativeNN import NonNegativeNN
from CoolTorch.train.trainer import train_CoOL
from CoolTorch.explain.lrp import layerwise_relevance_propogation
from CoolTorch.plotting.sub_groups import sub_groups                 # Python, now Ward
from CoolTorch.plotting.sub_groups_clustgeo import CoOL_6_sub_groups  # R / ClustGeo

# run here so clustgeo_tmp lands in examples/
os.chdir(os.path.dirname(__file__))

# working-example data (the canonical CoOL synergy example)
path = os.path.join(PROJECT_ROOT, "CoolTorch", "data", "data_working_example_R.csv")
if not os.path.exists(path):
    path = os.path.join(PROJECT_ROOT, "data", "data_working_example_R.csv")
df = pd.read_csv(path, sep=";")
y = df.iloc[:, 0]
exposure_df, X = binary_encode_exposure_data(df.iloc[:, 1:], return_tensor=True)
Y = torch.tensor(y.values, dtype=torch.float64)
print(f"n={len(y)}, outcome rate={y.mean():.3f}, exposures={X.shape[1]}")

model = NonNegativeNN(n_inputs=X.shape[1], n_hidden=8, use_c=False, dtype=torch.float64)
model.initiate_neural_network(Y, seed=123)
train_CoOL(X, Y, X, Y, None, None, model,
           lr_list=(1e-4, 1e-5, 1e-6), epochs=2000, patience=100,
           input_parameter_reg=1e-3, monitor=False)

res = layerwise_relevance_propogation(X, model, feature_names=exposure_df.columns)
if not isinstance(res, pd.DataFrame):
    res = pd.DataFrame(res, columns=list(exposure_df.columns) + ["Baseline_risk"])

k = 3
py = np.asarray(sub_groups(res, number_of_subgroups=k, ipw=1)).astype(int)
r = np.asarray(CoOL_6_sub_groups(res, number_of_subgroups=k, ipw=1)).astype(int)

ari = adjusted_rand_score(r, py)
print(f"\nbaseline risk R^b+ = {model.fc2.bias.item():.4f}  (expected ~0.05)")
print(f"ARI (Python-Ward vs R-ClustGeo) = {ari:.3f}  (1.0 = identical)")

contrib = res.drop(columns=["Baseline_risk"]) if "Baseline_risk" in res.columns else res
def summarize(labels, name):
    print(f"\n{name}:")
    for g in sorted(np.unique(labels)):
        m = labels == g
        top = contrib[m].mean(axis=0).sort_values(ascending=False)
        s = ", ".join(f"{n}={v:.2f}" for n, v in top.head(3).items() if v > 0.02)
        print(f"  G{g}: n={m.sum():4d}, risk={y.values[m].mean():.0%}  |  {s or '(low)'}")

summarize(r, "R / ClustGeo (Ward)")
summarize(py, "Python (Ward)")
