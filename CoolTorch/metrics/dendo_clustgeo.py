# Backend: R via subprocess (calls Rscript on utils/CoOL_dendrogram_runner.r, which
# runs ClustGeo). R equivalent: CoOL_6_dendrogram (CoOL_functions.R)
# The R bridge shared by sub_groups_clustgeo.py and number_of_sub_groups_clustgeo.py:
# writes the risk contributions to CSV, shells out to R for the ClustGeo clustering
# and dendrogram PNG, then reads the resulting cluster labels back into Python.
import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def CoOl_dendrogram_clustgeo(
    risk_contributions: pd.DataFrame,
    number_of_subgroups=3,
    title="Dendrogram",
    colours=None,
    ipw=1,
    workdir="clustgeo_tmp",
    r_script_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils", "CoOL_dendrogram_runner.r"),
    show=True,
):
    os.makedirs(workdir, exist_ok=True)

    risk_csv = os.path.join(workdir, "risk_contributions.csv")
    ipw_csv  = os.path.join(workdir, "ipw.csv")
    out_csv  = os.path.join(workdir, "clus_individual.csv")
    plot_png = os.path.join(workdir, "dendrogram.png")

    risk_contributions.to_csv(risk_csv, index=False)
    n = len(risk_contributions)

    if np.isscalar(ipw):
        ipw_arg = "1" if float(ipw) == 1.0 else ipw_csv
        if ipw_arg == ipw_csv:
            pd.DataFrame(np.repeat(float(ipw), n)).to_csv(ipw_csv, header=False, index=False)
    else:
        ipw = np.asarray(ipw, dtype=float)
        pd.DataFrame(ipw).to_csv(ipw_csv, header=False, index=False)
        ipw_arg = ipw_csv

    col_arg = "NA" if colours is None else ",".join(colours)

    cmd = ["Rscript", r_script_path, risk_csv, ipw_arg,
           str(int(number_of_subgroups)), title, col_arg, out_csv, plot_png]

    try:
        proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    except FileNotFoundError:
        raise RuntimeError(
            "Rscript was not found on PATH. The R/ClustGeo clustering backend needs "
            "R installed with the ClustGeo package (install.packages('ClustGeo') in "
            "R); use CoolTorch.metrics.sub_groups / CoolTorch.metrics.number_of_subgroups "
            "instead for the R-free, scipy-Ward alternative."
        )
    except subprocess.CalledProcessError as e:
        print("R stderr:\n", e.stderr)
        raise
    if proc.stdout:
        print(proc.stdout.strip())

    clus = pd.read_csv(out_csv)["cluster"].to_numpy(dtype=int)

    if show:
        img = mpimg.imread(plot_png)
        plt.figure(figsize=(7, 7))
        plt.imshow(img)
        plt.axis("off")
        plt.show()

    return clus
