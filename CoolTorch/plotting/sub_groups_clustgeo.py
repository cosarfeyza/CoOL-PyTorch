import os
import pandas as pd
import numpy as np
from CoolTorch.plotting.dendo_clustgeo import CoOl_dendrogram_clustgeo




def CoOL_6_sub_groups(
    risk_contributions: pd.DataFrame,
    number_of_subgroups: int = 3,
    ipw=1,
    workdir: str = "clustgeo_tmp",
    r_script_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils", "CoOL_dendrogram_runner.r"),
):
    """
    Python equivalent of CoOL_6_sub_groups:
      - gets clus from ClustGeo backend
      - reorders cluster labels by sum(colMeans(risk_contributions[clus==i,]))
      - returns clus_new
    """
    K = int(number_of_subgroups)

    clus = CoOl_dendrogram_clustgeo(
        risk_contributions=risk_contributions,
        number_of_subgroups=K,
        ipw=ipw,
        workdir=workdir,
        r_script_path=r_script_path,
        show=False,
    )
    clus = np.asarray(clus, dtype=int)

    # clus_pred[i] = sum(colMeans(risk_contributions[clus==i,]))
    clus_pred = np.zeros(K, dtype=float)
    for i in range(1, K + 1):
        rows = risk_contributions.loc[clus == i]
        clus_pred[i - 1] = rows.mean(axis=0).sum() if len(rows) else np.inf

    # order ascending
    order = np.argsort(clus_pred)  # indices 0..K-1

    # map old label -> new label
    mapping = {int(order[new - 1] + 1): int(new) for new in range(1, K + 1)}

    clus_new = np.array([mapping[int(c)] for c in clus], dtype=int)
    return clus_new
