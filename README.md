# CoolTorch — a PyTorch port of Causes of Outcome Learning (CoOL)

A clean, self-contained PyTorch implementation of CoOL (Rieckmann et al. 2022):
train a non-negative neural network, decompose each individual's risk with
Layer-wise Relevance Propagation (LRP), and cluster people by their risk-
contribution profiles into sub-groups.

Two interchangeable backends are included for each stage:

| Stage | Reference backend | Pure-Python alternative |
|---|---|---|
| Training | C++ Armadillo (`cool_ext_arma`) | `train.pytorch_trainer` |
| Clustering | R / ClustGeo (`metrics.sub_groups_clustgeo`, `metrics.number_of_sub_groups_clustgeo`) | `metrics.sub_groups`, `metrics.number_of_subgroups` (scipy Ward) |

The Python paths reproduce the R/C++ reference exactly (ARI = 1.0), so the package
runs with or without R and the C++ extension.

## Install

**Step 1 — install the package.** 


```bash
git clone <repo-url>
cd CoolTorch-pytorch
pip install -e .
```

With just this, the pure-PyTorch training path and the scipy-Ward clustering path
work fully — `CoOL_default_pytorch` runs end to end with nothing else installed.
Steps 2 and 3 are optional, and only add the R/C++ reference backends.

**Step 2 (optional) — Armadillo, for the C++ training backend.** Install it, then
re-run `pip install -e .` so it picks it up:

```bash
# macOS
brew install armadillo

# Debian/Ubuntu
sudo apt install libarmadillo-dev

# conda (any OS)
conda install -c conda-forge armadillo
```

`setup.py` looks for the headers under `/opt/homebrew/include` by default (Homebrew
on Apple Silicon); on Linux, Intel Homebrew, or a conda env, point it at the right
prefix instead: `ARMADILLO_INCLUDE=/usr/include ARMADILLO_LIB=/usr/lib pip install -e .`
(swap in wherever `apt`/`conda`/your package manager put it, e.g.
`$CONDA_PREFIX/include` and `$CONDA_PREFIX/lib` for a conda install). If Armadillo
isn't found, `pip install -e .` just skips the extension and prints a note — it
never fails the install.

**Step 3 (optional) — R + ClustGeo, for the R clustering backend.** Install R
itself, then these five R packages (`utils/CoOL_dendrogram_runner.r`, the script
that runs the clustering, ships with the package already — only R):

```r
install.packages(c("ClustGeo", "plyr", "ggplot2", "wesanderson"))
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install("ggtree")   # from Bioconductor, not CRAN
```

See the next two sections for what each backend needs and what happens if you
skip it.

## Choosing a training backend

Both trainers take the same arguments (`X_train`, `Y_train`, `model`, `lr_list`,
`epochs`, ...) and update the same `NonNegativeNN` in place, so switching is a
matter of which function you call, not a runtime flag.

**C++ Armadillo (`train.trainer.train_CoOL`)** — the reference backend, a direct
port of the compiled routine the original R package itself calls. Needs the
`cool_ext_arma` extension built (Armadillo + a C++ compiler; see Install).

```python
from CoolTorch.train.trainer import train_CoOL

train_CoOL(X_train, Y_train, X_test, Y_test, C_train=None, C_test=None,
           model=model, lr_list=(1e-4, 1e-5, 1e-6), epochs=10000)
```

**Pure PyTorch (`train.pytorch_trainer.pytorch_train_CoOL`)** — no compiled
extension, no Armadillo. Not a numerically identical port: it optimises the same
loss under the same non-negativity constraints, but with hand-written mini-batch
SGD instead of the reference's full-batch update, so expect close but not bitwise
identical results.

```python
from CoolTorch.train.pytorch_trainer import pytorch_train_CoOL

pytorch_train_CoOL(X_train, Y_train, X_test, Y_test, C_train=None, C_test=None,
                    model=model, lr_list=(1e-4, 1e-5, 1e-6), epochs=10000)
```

If `cool_ext_arma` was not built, `train_CoOL` raises a clear `RuntimeError` telling
you to use `pytorch_train_CoOL` instead, rather than failing to import or crashing
opaquely. `CoOL_default_R` (in Quick start, below) always uses `train_CoOL`;
`CoOL_default_pytorch` always uses `pytorch_train_CoOL` — pick whichever example
function matches the backend you want, or call the trainer functions directly on
your own model to mix and match.

## Choosing a clustering backend

Both take the same LRP risk-contribution matrix and a target number of sub-groups,
and return the same shape of result: one cluster label per individual.

**R / ClustGeo (`metrics.sub_groups_clustgeo.CoOL_6_sub_groups`)** — shells out to
`Rscript` to run the original package's `hclustgeo` clustering exactly. Needs
`Rscript` on `PATH` plus the five R packages listed under Install.

```python
from CoolTorch.metrics.sub_groups_clustgeo import CoOL_6_sub_groups

sub_groups_res = CoOL_6_sub_groups(risk_contributions, number_of_subgroups=3, ipw=1)
```

**scipy Ward (`metrics.sub_groups.sub_groups`)** — pure Python/scipy, no R. Verified
to reproduce the R/ClustGeo partition exactly (ARI = 1.0).

```python
from CoolTorch.metrics.sub_groups import sub_groups

sub_groups_res = sub_groups(risk_contributions, number_of_subgroups=3, ipw=1)
```

If `Rscript` isn't on `PATH`, or a required R package is missing, the R path raises
a clear `RuntimeError` pointing you at `metrics.sub_groups` instead of failing with
a raw R stack trace. `CoOL_default_R` always uses the R/ClustGeo path;
`CoOL_default_pytorch` always uses the scipy-Ward path.

## Choosing the number of sub-groups: manual or automatic elbow

`num_sub_groups` can be a fixed integer, or `None` to have it picked automatically
from the elbow of the mean-within-cluster-distance curve. This is independent of
which clustering backend you're using — both `CoOL_default_R` and
`CoOL_default_pytorch` support it:

```python
CoOL_default_R(data, num_sub_groups=None, low_number=2, high_number=6)
CoOL_default_pytorch(data, num_sub_groups=None, low_number=2, high_number=6)
```

Passing `None` scans every k from `low_number` to `high_number`
(`CoOL_6_number_of_sub_groups` for the R path, `number_of_sub_groups` for the
scipy path, both with `auto_elbow=True`), picks whichever k has the sharpest bend
in that curve, and prints which k it chose. Give an explicit integer instead to
skip the scan entirely and cluster directly at that k.

## Quick start — the working example

Reference pipeline (C++ training + R/ClustGeo clustering):

```python
from CoolTorch.data.working_example import cool_working_example
from CoolTorch.examples.CoOL_default_R import CoOL_default_R

data = cool_working_example(n=10000, seed=1)     # men+drug_a and women+drug_b co-act
res = CoOL_default_R(data, num_sub_groups=3)     # trains, LRP, clusters — draws, does not save
res["figure"].savefig("summary.png")             # that part is on the caller
```

Pure-PyTorch, R-free pipeline:

```python
python -m CoolTorch.examples.CoOL_default_pytorch
```

Both recover three sub-groups — a low-risk baseline and the two synergistic drug
groups — and show that being on drug A (for men) or drug B (for women) raises risk
by more than the sum of the individual effects.

## Things worth knowing before you use it

- **Output files land in your current working directory**, not next to the script,
  since neither function does `os.chdir()`. Run from wherever you want the PNGs to
  appear, or `cd` first.
- **Plotting functions default to `show=True`** when called without an `ax`. That's
  fine headless (`matplotlib.use("Agg")`, as the examples do) — a GUI backend on a
  machine with no display will hang on `plt.show()`. Pass `ax=...` (as the examples
  do internally) to draw into a subplot instead of popping a window.

## Layout

```
CoolTorch/
├── models/     NonNegativeNN            (the monotone non-negative network)
├── train/      trainer.py (C++)         pytorch_trainer.py (pure)   cool_step_arma.cpp
├── explain/    lrp.py                   (risk-contribution decomposition)
├── metrics/    individual_effects_matrix, sum_of_individual_effects, mean_risk_contributions_by_subgroup,
│               visualised_mean_risk_contributions(_legend); plus the clustering/k-selection logic:
│               sub_groups.py / number_of_subgroups.py (scipy Ward, no R) and their
│               *_clustgeo.py counterparts (R/ClustGeo, via dendo_clustgeo.py)
├── plotting/   the summary plots only — ROC, network, train performance, prevalence,
│               subgroup profile heatmaps, dendrogram/cluster-size bars
├── data/       working_example + the paper's simulations, binary encoding
├── utils/      CoOL_dendrogram_runner.r (the ClustGeo backend)
└── examples/   CoOL_default_R.py, CoOL_default_pytorch.py
```

## Reference

Rieckmann, A. et al. (2022). Causes of outcome learning: a causal
inference-inspired machine learning approach to disentangling common combinations
of potential causes of a health outcome. *International Journal of Epidemiology*.
