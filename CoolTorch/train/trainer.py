# Backend: C++ Armadillo (cool_ext_arma, compiled from cool_step_arma.cpp). R equivalent:
# CoOL_2_train_neural_network (CoOL_functions.R), which itself calls the same kind
# of compiled routine (Rcpp's cpp_train_network_relu) rather than looping in R —
# this is the faithful, same-algorithm port, not a reimplementation.
# Trains by repeatedly calling into the compiled extension for a block of epochs,
# tracking the best-so-far weights and stopping early once `patience` epochs pass
# without a new best training MSE.
import math
from dataclasses import dataclass
from typing import List
import torch
import torch.nn as nn
from CoolTorch.plotting.monitor_plot import plot_monitor_block

# C++ Armadillo backend — optional. Build it with `pip install -e .` (or
# `python setup.py build_ext --inplace`). Without it the pure-PyTorch trainer in
# train.pytorch_trainer still works; only train_CoOL below needs the extension.
try:
    from cool_ext_arma import cool_train_block
except ImportError:
    cool_train_block = None





@dataclass
class TrainLogs:
    train_mse: List[float]
    test_mse: List[float]
    weight_mse_diff: List[float]
    baseline_bias: List[float]
    epochs_done: int
    lr_used: List[float]



def _get_model_params_with_c2(model: nn.Module):
    W1 = model.fc1.weight.detach().clone().t()            # (D, H)
    B1 = model.fc1.bias.detach().clone().unsqueeze(0)     # (1, H)
    W2 = model.fc2.weight.detach().clone().t()            # (H, 1)
    B2 = model.fc2.bias.detach().clone().view(1, 1)       # (1, 1)
    if getattr(model, "c2", None) is not None:
        C2 = model.c2.detach().clone().view(1, 1)         # (1, 1)
    else:
        C2 = torch.zeros(1, 1, dtype=W1.dtype, device=W1.device)
    return W1, B1, W2, B2, C2


def _set_model_params_with_c2(model, W1, B1, W2, B2, C2):
    with torch.no_grad():
        model.fc1.weight.copy_(W1.t())                  # (H, D)
        model.fc1.bias.copy_(B1.view_as(model.fc1.bias))
        model.fc2.weight.copy_(W2.t())                  # (1, H)
        model.fc2.bias.copy_(B2.view_as(model.fc2.bias))
        if getattr(model, "c2", None) is not None:
            model.c2.copy_(C2.view_as(model.c2))



def train_CoOL(
    X_train, Y_train,
    X_test, Y_test,
    C_train, C_test,
    model,
    lr_list=(1e-4, 1e-5, 1e-6),
    epochs=10000,
    patience=100,
    monitor=True,
    plot_and_evaluation_frequency=50,
    input_parameter_reg=1e-3,
    drop_out=0,
    fix_baseline_risk=-1.0,
    ipw=1,
    dtype=torch.float64,
    device=None,
    spline_df=10
):
    if cool_train_block is None:
        raise RuntimeError(
            "cool_ext_arma (C++ backend) is not built. Build it with "
            "`pip install -e .`, or use CoolTorch.train.pytorch_trainer for the "
            "pure-PyTorch trainer."
        )

    X_train = X_train.to(device, dtype).contiguous()
    Y_train = Y_train.to(device, dtype).view(-1).contiguous()
    X_test  = X_test.to(device,  dtype).contiguous()
    Y_test  = Y_test.to(device,  dtype).view(-1).contiguous()

    # Confounder
    if C_train is None:
        C_train = torch.zeros_like(Y_train, device=device, dtype=dtype)
    else:
        C_train = C_train.to(device=device, dtype=dtype).view(-1).contiguous()

    if C_test is None:
        C_test = torch.zeros_like(Y_test, device=device, dtype=dtype)
    else:
        C_test = C_test.to(device=device, dtype=dtype).view(-1).contiguous()

    
    if ipw is None or (isinstance(ipw, (int, float)) and ipw == 1):
        ipw_t = torch.ones_like(Y_train, device=device, dtype=dtype)
    else:
        ipw_t = ipw.to(device=device, dtype=dtype).view(-1)

    
    train_perf: List[float] = []
    test_perf: List[float] = []
    weight_perf: List[float] = []
    baseline_mon: List[float] = []

    best_idx: int | None = None
    stop_all = False

   
    n_rounds = math.ceil(epochs / plot_and_evaluation_frequency)

    for lr_set in lr_list:
        if stop_all:
            break

        print(f"############################## Learning rate: {lr_set} ##############################")

        for _ in range(n_rounds):
            if stop_all:
                break

            W1, B1, W2, B2, C2 = _get_model_params_with_c2(model)

            (W1, B1, W2, B2, C2,
             train_block, test_block, weight_block, baseline_block) = cool_train_block(
                X_train, Y_train, C_train,
                X_test,  Y_test,  C_test,
                W1, B1, W2, B2, C2,
                ipw_t,
                float(lr_set),
                int(plot_and_evaluation_frequency),
                float(input_parameter_reg),
                int(drop_out),
                float(fix_baseline_risk),
            )

            _set_model_params_with_c2(model, W1, B1, W2, B2, C2)

            
            train_block_list = train_block.cpu().tolist()
            test_block_list = test_block.cpu().tolist()
            weight_block_list = weight_block.cpu().tolist()
            baseline_block_list = baseline_block.cpu().tolist()

            train_perf.extend(train_block_list)
            test_perf.extend(test_block_list)
            weight_perf.extend(weight_block_list)
            baseline_mon.extend(baseline_block_list)

            
            if train_perf:
                idx_min = min(range(len(train_perf)),
                              key=lambda i: train_perf[i])
                if best_idx is None or train_perf[idx_min] < train_perf[best_idx]:
                    best_idx = idx_min

                
                if len(train_perf) - best_idx > patience:
                    stop_all = True
                    break

    if monitor:
        y_train_mean = float(Y_train.mean().item())
        plot_monitor_block(
            train_perf,
            weight_perf,
            baseline_mon,
            y_train_mean=y_train_mean,
            spline_df=spline_df,
        )
    if not hasattr(model, "train_performance"):
        model.train_performance = []
    if not hasattr(model, "test_performance"):
        model.test_performance = []
    if not hasattr(model, "weight_performance"):
        model.weight_performance = []
    if not hasattr(model, "baseline_risk_monitor"):
        model.baseline_risk_monitor = []

    model.train_performance.extend(train_perf)
    model.test_performance.extend(test_perf)
    model.weight_performance.extend(weight_perf)
    model.baseline_risk_monitor.extend(baseline_mon)

    return {
        "train_performance": train_perf,
        "test_performance": test_perf,
        "weight_performance": weight_perf,
        "baseline_risk_monitor": baseline_mon,
    }
