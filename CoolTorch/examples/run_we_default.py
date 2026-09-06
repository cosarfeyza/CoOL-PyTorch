import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

import torch  # before cool_ext_arma
import pandas as pd
from CoolTorch.examples.CoOL_default import CoOL_default

# run here so the summary png lands in examples/ (not clobbering Adult's)
os.chdir(os.path.dirname(__file__))

path = os.path.join(PROJECT_ROOT, "CoolTorch", "data", "data_working_example_R.csv")
if not os.path.exists(path):
    path = os.path.join(PROJECT_ROOT, "data", "data_working_example_R.csv")
df = pd.read_csv(path, sep=";")
print(f"n={len(df)}, outcome rate={df.iloc[:,0].mean():.3f}")

CoOL_default(df, num_sub_groups=None, low_number=2, high_number=6,
             hidden=8, epochs=2000, monitor=False)

for src, dst in [("cool_default_summaryyyy.png", "we_cool_summary.png"),
                 ("cool_subgroup_profiles.png", "we_cool_subgroup_profiles.png")]:
    if os.path.exists(src):
        os.replace(src, dst)
print("done — saved we_cool_summary.png")
