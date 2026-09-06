# CoolTorch — a PyTorch port of Causes of Outcome Learning (CoOL)

A clean, self-contained PyTorch implementation of CoOL (Rieckmann et al. 2022):
train a non-negative neural network, decompose each individual's risk with
Layer-wise Relevance Propagation (LRP), and cluster people by their risk-
contribution profiles into sub-groups.

Two interchangeable backends are included for each stage:

| Stage | Reference backend | Pure-Python alternative |
|---|---|---|
| Training | C++ Armadillo (`cool_ext_arma`) | `train.pytorch_trainer` |
| Clustering | R / ClustGeo (`utils/CoOL_dendrogram_runner.r`) | `plotting.sub_groups` (scipy Ward) |

The Python paths reproduce the R/C++ reference exactly (ARI = 1.0), so the package
runs with or without R and the C++ extension.

## Install

```bash
pip install -e .
```

This builds the optional C++ Armadillo backend. It needs Armadillo:

```bash
brew install armadillo          # macOS; override paths via ARMADILLO_INCLUDE / ARMADILLO_LIB
```

The pure-PyTorch path works even if the C++ extension is not built — `train_CoOL`
raises a clear message and you use `pytorch_train_CoOL` instead. The R/ClustGeo
path needs `Rscript` with the `ClustGeo` package installed; the Python-Ward path
does not.

## Quick start — the working example

Reference pipeline (C++ training + R/ClustGeo clustering):

```python
from CoolTorch.data.working_example import cool_working_example
from CoolTorch.examples.CoOL_default import CoOL_default

data = cool_working_example(n=10000, seed=1)   # men+drug_a and women+drug_b co-act
CoOL_default(data, num_sub_groups=3)           # trains, LRP, clusters, plots the summary
```

Pure-PyTorch, R-free pipeline:

```python
python -m CoolTorch.examples.pytorch_CoOL_default
```

Both recover three sub-groups — a low-risk baseline and the two synergistic drug
groups — and show that being on drug A (for men) or drug B (for women) raises risk
by more than the sum of the individual effects.

## Layout

```
CoolTorch/
├── models/     NonNegativeNN            (the monotone non-negative network)
├── train/      trainer.py (C++)         pytorch_trainer.py (pure)   cool_step_arma.cpp
├── explain/    lrp.py                   (risk-contribution decomposition)
├── plotting/   *_clustgeo.py (R path)   sub_groups.py / number_of_subgroups.py (Python path)
│               plus the summary plots (ROC, network, prevalence, contributions)
├── metrics/    individual effects, sum of individual effects, mean contributions
├── data/       working_example + the paper's simulations, binary encoding
├── utils/      CoOL_dendrogram_runner.r (the ClustGeo backend)
├── examples/   CoOL_default, pytorch_CoOL_default, working-example runners
└── tests/      LRP and plotting checks against saved R references
```

## Reference

Rieckmann, A. et al. (2022). Causes of outcome learning: a causal
inference-inspired machine learning approach to disentangling common combinations
of potential causes of a health outcome. *International Journal of Epidemiology*.
