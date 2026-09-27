# Backend: pure PyTorch (no R, no C++ extension needed). R equivalent: CoOL_2_train_neural_network
# (CoOL_functions.R), but NOT a literal port — R/trainer.py update the model with
# full-batch gradient steps via the compiled cpp_train_network_relu / cool_train_block;
# this file reimplements training from scratch as manual mini-batch SGD (hand-written
# gradients, no autograd, so the non-negativity/sign clamps on W1/B1/B2 can be applied
# directly). Same loss and same constraints, different optimisation algorithm — use
# this when the C++ extension isn't built, not as a numerically identical substitute.
import math
from typing import List
import torch
import torch.nn as nn
from CoolTorch.plotting.monitor_plot import plot_monitor_block


def _get_params(m):
    # extract as column-major (transposed) to match R's column-major convention
    W1 = m.fc1.weight.detach().clone().t()
    B1 = m.fc1.bias.detach().clone()
    W2 = m.fc2.weight.detach().clone().t()
    B2 = m.fc2.bias.detach().clone()
    C2 = m.c2.detach().clone() if getattr(m, "c2", None) is not None else torch.zeros(1, dtype=W1.dtype, device=W1.device)
    return W1, B1, W2, B2, C2


def _set_params(m, W1, B1, W2, B2, C2):
    with torch.no_grad():
        m.fc1.weight.copy_(W1.t())
        m.fc1.bias.copy_(B1)
        m.fc2.weight.copy_(W2.t())
        m.fc2.bias.copy_(B2)
        if getattr(m, "c2", None) is not None:
            m.c2.copy_(C2.view_as(m.c2))


def _train_block(X, Y, C, X_test, Y_test, C_test, W1, B1, W2, B2, C2,
                 ipw, lr, n_epochs, reg, batch_size, fix_baseline, use_c):
    N = X.shape[0]
    dtype = W1.dtype
    device = W1.device
    mean_y = Y.mean().item()

    rng = torch.Generator(device=device)
    rng.manual_seed(123)

    W1, B1, B2, C2 = W1.clone(), B1.clone(), B2.clone(), C2.clone()
    if fix_baseline >= 0:
        B2.fill_(fix_baseline)

    W1_prev = W1.clone()
    W2v = W2.view(-1)  # fc2 weights as a flat vector — single output node

    t_perf, v_perf, w_perf, b_mon = [], [], [], []

    for ep in range(n_epochs):
        perm = torch.randperm(N, generator=rng, device=device)

        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            b = idx.shape[0]

            Xb = X[idx]; Yb = Y[idx]
            ipw_b = ipw[idx]; Cb = C[idx]

            # forward pass: relu on hidden layer, relu on output to keep predictions non-negative
            Hb = torch.relu(Xb @ W1 + B1)
            Ob = torch.relu(Hb @ W2v + B2.item() + Cb * C2.item())
            Eb = Ob - Yb  # residual (MSE gradient)

            mask = (Hb > 0).to(dtype)
            dh = mask * W2v  # backprop through hidden relu
            wE = ipw_b * Eb  # IPW-weighted residuals

            # manual gradient descent — no autograd to allow custom clamp constraints
            gW1 = Xb.t() @ (wE.unsqueeze(1) * dh) / b
            W1 -= lr * gW1
            W1 -= ipw_b.mean().item() * lr * reg  # L1 reg, per-step not scaled by batch
            W1.clamp_(min=0.0)  # non-negativity constraint

            gB1 = (wE.unsqueeze(1) * dh).mean(dim=0)
            B1 -= lr * gB1
            B1.clamp_(max=0.0)  # bias must be <=0 to keep baseline risk from being inflated

            if fix_baseline < 0:
                B2 -= (lr / 10.) * wE.mean()
                B2.clamp_(min=0.0)  # baseline risk cannot be negative
                if use_c:
                    C2 -= (lr / 10.) * (wE * Cb).mean()
                    C2.clamp_(min=0.0)

        with torch.no_grad():
            Ht = torch.relu(X @ W1 + B1)
            Ot = torch.relu(Ht @ W2v + B2.item() + C * C2.item())
            loss_tr = 0.5 * ((Y - Ot) ** 2).mean().item()

            Hv = torch.relu(X_test @ W1 + B1)
            Ov = torch.relu(Hv @ W2v + B2.item() + C_test * C2.item())
            loss_val = 0.5 * ((Y_test - Ov) ** 2).sum().item() / N

            wdiff = ((W1 - W1_prev) ** 2).mean().item()

        t_perf.append(loss_tr)
        v_perf.append(loss_val)
        w_perf.append(wdiff)
        b_mon.append(B2.item())
        W1_prev = W1.clone()

        if ep % 10 == 0:
            print(f"{ep} epochs: Train performance of {loss_tr:.6f}. Baseline risk estimated to {B2.item():.6f}.")
            if B2.item() > mean_y:
                print(f"Warning: Baseline risk ({B2.item():.4f}) > mean(Y) ({mean_y:.4f}). Consider reducing input_parameter_reg.")

    return W1, B1, W2, B2, C2, t_perf, v_perf, w_perf, b_mon


def pytorch_train_CoOL(
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
    fix_baseline_risk=-1.0,
    ipw=1,
    batch_size=32,
    dtype=torch.float64,
    device=None,
    spline_df=10,
):
    if device is None:
        device = next(model.parameters()).device

    def _to(t):
        if isinstance(t, torch.Tensor):
            return t.to(device=device, dtype=dtype).contiguous()
        return torch.tensor(t, device=device, dtype=dtype)

    X_train = _to(X_train)
    Y_train = _to(Y_train).view(-1)
    X_test = _to(X_test)
    Y_test = _to(Y_test).view(-1)

    C_train = torch.zeros_like(Y_train) if C_train is None else _to(C_train).view(-1)
    C_test = torch.zeros_like(Y_test) if C_test is None else _to(C_test).view(-1)

    if ipw is None or (isinstance(ipw, (int, float)) and ipw == 1):
        ipw_t = torch.ones_like(Y_train)
    else:
        ipw_t = _to(ipw).view(-1)

    use_c = getattr(model, "use_c", False)
    N = X_train.shape[0]
    n_batches = math.ceil(N / batch_size)

    # scale patience to mini-batch epochs
    eff_pat = max(1, round(patience * n_batches / N))
    print(f"Patience scaled: {patience} online epochs -> {eff_pat} mini-batch epochs (batch_size={batch_size}, N={N})")

    all_tr, all_val, all_w, all_b = [], [], [], []
    best_idx = None
    stop = False
    n_rounds = math.ceil(epochs / plot_and_evaluation_frequency)

    for lr in lr_list:
        if stop:
            break
        print(f"############################## Learning rate: {lr} ##############################")

        for _ in range(n_rounds):
            if stop:
                break

            W1, B1, W2, B2, C2 = _get_params(model)
            W1, B1, W2, B2, C2, tp, vp, wp, bp = _train_block(
                X_train, Y_train, C_train,
                X_test, Y_test, C_test,
                W1, B1, W2, B2, C2, ipw_t,
                float(lr), int(plot_and_evaluation_frequency),
                float(input_parameter_reg), int(batch_size),
                float(fix_baseline_risk), use_c,
            )
            _set_params(model, W1, B1, W2, B2, C2)

            all_tr.extend(tp); all_val.extend(vp)
            all_w.extend(wp); all_b.extend(bp)

            idx_min = min(range(len(all_tr)), key=lambda i: all_tr[i])
            if best_idx is None or all_tr[idx_min] < all_tr[best_idx]:
                best_idx = idx_min
            if len(all_tr) - best_idx > eff_pat:
                stop = True
                break

    if monitor:
        plot_monitor_block(all_tr, all_w, all_b,
                           y_train_mean=float(Y_train.mean().item()),
                           spline_df=spline_df)

    for attr, vals in [("train_performance", all_tr), ("test_performance", all_val),
                        ("weight_performance", all_w), ("baseline_risk_monitor", all_b)]:
        if not hasattr(model, attr):
            setattr(model, attr, [])
        getattr(model, attr).extend(vals)

    return {
        "train_performance": all_tr,
        "test_performance": all_val,
        "weight_performance": all_w,
        "baseline_risk_monitor": all_b,
    }
