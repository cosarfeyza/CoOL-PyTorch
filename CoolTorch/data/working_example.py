import numpy as np
import pandas as pd

def cool_working_example(n: int, seed: int | None = None) -> pd.DataFrame:
    """
    Direct Python equivalent of the R function:
    
    CoOL_0_working_example <- function(n) {
      drug_a = sample(1:0,n,prob=c(0.2,0.8),replace=TRUE)
      sex = sample(1:0,n,prob=c(0.5,0.5),replace=TRUE)
      drug_b = sample(1:0,n,prob=c(0.2,0.8),replace=TRUE)
      Y <-  sample(1:0,n,prob=c(0.05,0.95),replace = TRUE)
      for (i in 1:n) {
        if (sex[i] == 0 & drug_a[i] == 1 & sample(1:0,1,prob=c(.15,0.8)) ) {
          Y[i] <- 1
        }
        if (sex[i] == 1 & drug_b[i] == 1 & sample(1:0,1,prob=c(.15,0.85)) ) {
          Y[i] <- 1
        }
      }
      data <- data.frame(Y,sex,drug_a,drug_b)
      for (i in 1:ncol(data)) data[,i] <- as.numeric(data[,i])
      return(data)
    }
    """

    rng = np.random.RandomState(seed)  

    # Replicating R's sample(1:0, n, prob=c(...), replace=TRUE)
    drug_a = rng.choice([1, 0], size=n, p=[0.2, 0.8])
    sex    = rng.choice([1, 0], size=n, p=[0.5, 0.5])
    drug_b = rng.choice([1, 0], size=n, p=[0.2, 0.8])
    Y      = rng.choice([1, 0], size=n, p=[0.05, 0.95])

    
    for i in range(n):
        
        if (sex[i] == 0) & (drug_a[i] == 1) & (rng.choice([1, 0], p=[0.15 / (0.15 + 0.8), 0.8 / (0.15 + 0.8)]) == 1):
            Y[i] = 1
        if (sex[i] == 1) & (drug_b[i] == 1) & (rng.choice([1, 0], p=[0.15, 0.85]) == 1):
            Y[i] = 1

    
    data = pd.DataFrame({
        "Y": Y.astype(int),
        "sex": sex.astype(int),
        "drug_a": drug_a.astype(int),
        "drug_b": drug_b.astype(int)
    })

    return data
