import os as _os
_PKG = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # CoolTorch package root
import os
import json
import pandas as pd
import torch
from CoolTorch.models.NonNegativeNN import NonNegativeNN


'''
The function loads a pre-trained NonNegativeNN model from CSV files exported from R.
It reads weight and bias matrices from specified CSV files, reconstructs the model architecture,
and loads the parameters into a NonNegativeNN instance.

Returns:
    model (NonNegativeNN): The loaded NonNegativeNN model with parameters from R.
    feature_names (list or None): List of feature names if available, else None.    

'''


def load_cool_from_r():
    
    W1 = torch.tensor(pd.read_csv(_os.path.join(_PKG, "tests/R_pre_trained_model/W1.csv")).values, dtype=torch.float64).T      # (H,D)
    b1 = torch.tensor(pd.read_csv(_os.path.join(_PKG, "tests/R_pre_trained_model/b1.csv")).values.squeeze(), dtype=torch.float64)  # (H,)
    W2 = torch.tensor(pd.read_csv(_os.path.join(_PKG, "tests/R_pre_trained_model/W2.csv")).values, dtype=torch.float64)      # (1,H) or (H,1) or (H,)
    b2 = torch.tensor(pd.read_csv(_os.path.join(_PKG, "tests/R_pre_trained_model/b2.csv")).values.squeeze(), dtype=torch.float64)  # ()
    
    
    try:
        meta = json.load(open(_os.path.join(_PKG, "tests/R_pre_trained_model/meta.json")))
        feature_names = meta.get("feature_names", None)
        has_confounder = bool(meta.get("has_confounder", False))
    except FileNotFoundError:
        meta = {}
        feature_names = None
        has_confounder = False 
    
    # Look if c2 exists, then set has_confounder
    try:
        c2 = torch.tensor(pd.read_csv(_os.path.join(_PKG, "tests/R_pre_trained_model/c2.csv")).values.squeeze(), dtype=torch.float32)
        has_confounder = has_confounder or (c2.numel() == 1 and float(c2) != 0.0)
    except FileNotFoundError:
        c2 = torch.tensor(0.0);  
    
    # 2) normalize shapes
    if W2.ndim == 2 and W2.shape[1] == 1:
        W2 = W2.T               # (1,H)
    elif W2.ndim == 1:
        W2 = W2.unsqueeze(0)    # (1,H)
    assert W2.ndim == 2 and W2.shape[0] == 1, f"Unexpected W2 shape: {W2.shape}"

    #print(W1.shape, b1.shape, W2.shape, b2.shape)
    
    r, c = W1.shape
    if b1.numel() == r:
        # W1 is (H, D)
        H, D = r, c
    elif b1.numel() == c:
        # W1 is (D, H) -> transpose
        W1 = W1.T
        H, D = W1.shape
    else:
        raise ValueError(
            f"b1 length ({b1.numel()}) doesn't match W1 rows ({r}) or cols ({c}). "
            "Check your exports."
        )

    # Checks for the shape of W2
    if W2.shape[1] != H:
        if W2.shape[1] == D:
            raise ValueError(f"W2 width ({W2.shape[1]}) != hidden size H ({H}).")
        else:
            raise ValueError(f"W2 width ({W2.shape[1]}) doesn't match H ({H}) or D ({D}).")

    #print(W1.shape, b1.shape, W2.shape, b2.shape)
    #print(W1)  

    # Build the model

    model = NonNegativeNN(n_inputs=D, n_hidden=H, use_c=has_confounder, dtype=torch.float64)

    with torch.no_grad():
        model.fc1.weight.copy_(W1)                 # (H, D)
        model.fc1.bias.copy_(b1.view(-1))          # (H,)
        model.fc2.weight.copy_(W2)                 # (1, H)
        model.fc2.bias.copy_(b2.view(()))          # scalar
        # c2 if present
        if has_confounder and getattr(model, "c2", None) is not None and c2.numel() == 1:
            model.c2.copy_(c2.view(()))

    model.eval()
    return model, feature_names

 
if __name__ == "__main__":
    model, feature_names = load_cool_from_r()
    
    #print(model)
    #print("Feature names:", feature_names)
    #print("w1", model.fc1.weight)
