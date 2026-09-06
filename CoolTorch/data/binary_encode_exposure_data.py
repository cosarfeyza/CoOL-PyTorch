import pandas as pd
import numpy as np
import torch

def binary_encode_exposure_data(exposure_df: pd.DataFrame,
                                return_tensor: bool = False,
                                device: str = "cpu"):
    """
    Python equivalent of:
      CoOL_0_binary_encode_exposure_data <- function(exposure_data) {
        for (i in 1:ncol(exposure_data)) {exposure_data[,i] <- factor(exposure_data[,i])}
        exposure_data <- mltools::one_hot(data.table::as.data.table(exposure_data))
        exposure_data <- as.data.frame(exposure_data)
        for (i in 1:ncol(exposure_data)) {exposure_data[,i] <- as.numeric(exposure_data[,i])}
        return(exposure_data)
      }

    - Every column is treated as categorical, regardless of numeric/continuous.
    - One-hot encodes *all* columns (no drop_first).
    - Returns numeric DataFrame (float).
    - Optionally returns a torch.FloatTensor.
    """
    # 1) force categorical for all columns (mirrors R's factor() on every column)
    df_cat = exposure_df.copy()
    for c in df_cat.columns:
        if not pd.api.types.is_categorical_dtype(df_cat[c]):
            df_cat[c] = df_cat[c].astype("category")

    # 2) one-hot all columns (like mltools::one_hot), no reference level dropped
    encoded_df = pd.get_dummies(df_cat, drop_first=False, dtype=float)

    if not return_tensor:
        return encoded_df

    X_tensor = torch.tensor(
        encoded_df.to_numpy(dtype=np.float32, copy=False),
        dtype=torch.float32,
        device=device,
    )


    X_tensor.colnames = list(encoded_df.columns)

    return encoded_df, X_tensor
