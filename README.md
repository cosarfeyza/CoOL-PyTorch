# CoolTorch — a PyTorch port of Causes of Outcome Learning (CoOL)

A clean, self-contained PyTorch implementation of CoOL (Rieckmann et al. 2022):
train a non-negative neural network, decompose each individual's risk with
Layer-wise Relevance Propagation (LRP), and cluster people by their risk-
contribution profiles into sub-groups.

Two independent stages each have two interchangeable backends:

| Stage | Reference backend | Pure-Python alternative | Needs |
|---|---|---|---|
| Training | C++ Armadillo (`train.trainer.train_CoOL`) | `train.pytorch_trainer.pytorch_train_CoOL` | Armadillo + a C++ compiler |
| Clustering | R / ClustGeo (`metrics.sub_groups_clustgeo`) | `metrics.sub_groups` (scipy Ward) | R + 5 packages |

The pure-Python paths reproduce the reference backends exactly (ARI = 1.0), so the
package installs and runs with or without R or a C++ compiler.

## Install

```bash
git clone <repo-url>
cd CoolTorch-pytorch
pip install -e .
```

That's the only required step, and it always succeeds — with or without Armadillo
or R. On its own it gives you the full pure-Python path (`CoOL_default_pytorch`
runs end to end). The two backends below are optional extras.

<details>
<summary><b>Optional — Armadillo, for the C++ training backend</b></summary>

Install it for your platform, then re-run `pip install -e .` so it gets picked up:

| Platform | Command |
|---|---|
| macOS | `brew install armadillo` |
| Debian/Ubuntu | `sudo apt install libarmadillo-dev` |
| conda (any OS) | `conda install -c conda-forge armadillo` |
| Windows | `vcpkg install armadillo` |

`setup.py` already looks in the right default place for macOS (Homebrew) and Linux
(`/usr/include`); on Windows, or if yours lives somewhere else, point at it
directly:

```bash
ARMADILLO_INCLUDE=/path/to/include ARMADILLO_LIB=/path/to/lib pip install -e .
```

If Armadillo still isn't found, or the compiler can't build it (e.g. no C++
toolchain, or an MSVC/GCC flag mismatch), `pip install -e .` just skips the
extension and prints a note — it never fails the install. `train_CoOL()` then
raises a clear error pointing you at `pytorch_train_CoOL()` instead.
</details>

<details>
<summary><b>Optional — R + ClustGeo, for the R clustering backend</b></summary>

Install R itself, then these packages from inside R (the script that actually
runs the clustering, `utils/CoOL_dendrogram_runner.r`, ships with the package
already — only R and its packages are on you):

```r
install.packages(c("ClustGeo", "plyr", "ggplot2", "wesanderson"))
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install("ggtree")   # from Bioconductor, not CRAN
```

If `Rscript` isn't on `PATH`, or a package is missing, the R path raises a clear
error pointing you at the scipy-Ward alternative instead of a raw R stack trace.
</details>

## Quick start

```python
from CoolTorch.data.working_example import cool_working_example
from CoolTorch.examples.CoOL_default_R import CoOL_default_R          # C++ + R/ClustGeo
from CoolTorch.examples.CoOL_default_pytorch import CoOL_default_pytorch  # pure PyTorch, R-free

data = cool_working_example(n=10000, seed=1)   # men+drug_a and women+drug_b co-act
res = CoOL_default_R(data, num_sub_groups=3)   # trains, LRP, clusters — draws, does not save
res["figure"].savefig("summary.png")           # saving is on the caller
```

Or, R-free end to end: `python -m CoolTorch.examples.CoOL_default_pytorch`.

Both recover three sub-groups — a low-risk baseline and the two synergistic drug
groups — and show that being on drug A (for men) or drug B (for women) raises risk
by more than the sum of the individual effects.

## Picking a backend directly

Both trainers and both clustering functions take the same arguments and return
the same shape of result, so switching is just which function you call:

```python
# Training — pick one
from CoolTorch.train.trainer import train_CoOL                  # C++, needs cool_ext_arma
from CoolTorch.train.pytorch_trainer import pytorch_train_CoOL  # pure PyTorch

# Clustering — pick one
from CoolTorch.metrics.sub_groups_clustgeo import CoOL_6_sub_groups  # R/ClustGeo
from CoolTorch.metrics.sub_groups import sub_groups                  # scipy Ward
```

`CoOL_default_R` always uses the C++ trainer and the R/ClustGeo path;
`CoOL_default_pytorch` always uses the pure-PyTorch trainer and scipy Ward —
pick whichever example matches the backend you want, or call the four functions
above directly on your own model to mix and match. The pure-PyTorch trainer is
not a numerically identical port of the C++ one (hand-written mini-batch SGD vs.
the reference's full-batch update), so expect close but not bitwise-identical
results; the scipy clustering *is* verified to reproduce the R/ClustGeo partition
exactly.

**Number of sub-groups.** `num_sub_groups` can be a fixed integer, or `None` to
pick it automatically from the elbow of the mean-within-cluster-distance curve —
supported by both `CoOL_default_R` and `CoOL_default_pytorch`:

```python
CoOL_default_R(data, num_sub_groups=None, low_number=2, high_number=6)
```

This scans every k from `low_number` to `high_number` and prints which k it
picked; give an explicit integer to skip the scan.

## Things worth knowing

- **Output files land in your current working directory**, not next to the
  script, since neither example does `os.chdir()`. Run from wherever you want
  the PNGs to appear, or `cd` first.
- **Plotting functions default to `show=True`** when called without an `ax`.
  Fine headless (`matplotlib.use("Agg")`, as the examples do) — but a GUI
  backend on a machine with no display will hang on `plt.show()`. Pass
  `ax=...` to draw into a subplot instead.

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
