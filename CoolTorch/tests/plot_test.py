import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from CoolTorch.plotting.plot_train_performance import plot_train_performance
from CoolTorch.plotting.plot_roc_auc import plot_roc_auc
from CoolTorch.plotting.plot_neural_network import plot_nonnegative_nn
from CoolTorch.data.working_example import cool_working_example

from CoolTorch.models.NonNegativeNN import NonNegativeNN
from CoolTorch.train.trainer import CoOL_2_train_neural_network_torch
from CoolTorch.data.binary_encode_exposure_data import binary_encode_exposure_data 
from CoolTorch.plotting.plot_roc_auc import plot_roc_auc
from CoolTorch.plotting.plot_neural_network import plot_nonnegative_nn
from CoolTorch.plotting.plot_train_performance import plot_train_performance


df = cool_working_example(10_000, seed=1)
#print(df.head())

outcome_data = df.iloc[:, 0]
exposure_raw = df.iloc[:, 1:]
#print(outcome_data.head())

exposure_df, X_tensor = binary_encode_exposure_data(
    exposure_raw,
    return_tensor=True,   
    device="cpu"          
)
#print(X_tensor)
#print(exposure_df.head())


Y_tensor = torch.tensor(outcome_data.values, dtype=torch.float32)
#print(Y_tensor)

#Initiate the model
model = NonNegativeNN(n_inputs=X_tensor.shape[1], n_hidden=8, use_c=False)


hist = {
    "train_performance": [0.08, 0.06, 0.05, 0.045, 0.043],
    "test_performance":  [0.081, 0.062, 0.052, 0.048, 0.046],
    "weight_performance": [],
    "baseline_risk_monitor": [],
    "epochs": 5 * 20,  
    "best_epoch": 5,
    "best_loss": 0.043,
}
plot_train_performance(hist, title="test")


plot_nonnegative_nn(
    model,
    names=exposure_df.columns.tolist())

#