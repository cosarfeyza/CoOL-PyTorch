# Backend: pure PyTorch (no R). R equivalent: CoOL_4_predict_risks (CoOL_functions.R)
# Runs the forward pass by hand from raw weight/bias tensors (or a dict of them)
# rather than a live model object, so predictions can be reproduced from saved
# parameters alone (e.g. weights loaded from R). ReLU on both the hidden layer and
# the output, matching the non-negative network's forward pass.
import torch
import torch.nn.functional as F

@torch.no_grad()
def predict_risks(X, model, dtype=torch.float64, device=None):
    if isinstance(model, dict):
        W1, b1, W2, b2 = model["W1"], model["b1"], model["W2"], model["b2"]
    else:
        W1, b1, W2, b2 = model

    if device is None:
        for t in (W1, b1, W2, b2):
            if isinstance(t, torch.Tensor):
                device = t.device
                break
        if device is None:
            device = torch.device("cpu")

    def _tt(a):
        if isinstance(a, torch.Tensor):
            return a.to(device=device, dtype=dtype)
        return torch.tensor(a, device=device, dtype=dtype)

    X  = _tt(X)
    W1 = _tt(W1); b1 = _tt(b1)
    W2 = _tt(W2); b2 = _tt(b2)

    if W1.dim() != 2:
        raise ValueError(f"W1 must be 2D (p,H); got {W1.shape}")
    p, H = W1.shape
    if X.shape[1] != p:
        raise ValueError(f"X has {X.shape[1]} cols but W1 expects {p}.")

    b1 = b1.view(-1)
    if b1.numel() != H:
        raise ValueError(f"b1 length {b1.numel()} != hidden size {H}")

    if W2.dim() == 2 and W2.shape[1] >= 1:
        w2_vec = W2[:, 0].view(H)
    elif W2.dim() == 1 and W2.numel() == H:
        w2_vec = W2.view(H)
    else:
        raise ValueError(f"W2 must be (H,1) or (H,); got {W2.shape}")

    b2 = b2.reshape(())

    H_act = F.relu(X @ W1 + b1)
    o = F.relu(H_act @ w2_vec + b2)

    if torch.any(o > 1):
        print("Warning: Some predicted risks are above 1")

    return o
