# fun-ml-projects

Small projects collected over the years.

## Python environment (`pyproject.toml`)

The file [`pyproject.toml`](pyproject.toml) at the **repository root** pins dependencies for **these directories only**:

| Directory | Notes |
|-----------|--------|
| [`finance/`](finance/) | Equities notebook (Yahoo Finance, Plotly, Wikipedia / `read_html`) and corporate-bond GNN notebook (SciPy, scikit-learn, NetworkX). |
| [`miscellaneous/`](miscellaneous/) | `silly_examples.ipynb` (NumPy, Pandas, Matplotlib). |
| [`time_series/`](time_series/) | ARIMA / ARCH / portfolio notebooks (`statsmodels`, `arch`, `cvxpy`, `riskfolio-lib`, `yfinance`, …) and `basic_rnn_pytorch.ipynb` (**PyTorch**). |

It does **not** apply to other folders in the repo (for example `deep_learning/`, `kaggle/`, `__udacity__/`, and so on). Those projects may need their own environments or extra packages.

### Requirements

- **Python 3.10+**

### Install

From the **root** of this repository (the directory that contains `pyproject.toml`):

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[notebook]"
```

The `[notebook]` extra installs Jupyter and `ipykernel` so you can run the `.ipynb` files. PyTorch is included in the base dependencies because `time_series/basic_rnn_pytorch.ipynb` needs it; that makes the first install larger.

### Optional: named Jupyter kernel

```bash
python -m ipykernel install --user --name fun-ml-projects --display-name "Python (fun-ml-projects)"
```

Then choose that kernel when opening notebooks under `finance/`, `miscellaneous/`, or `time_series/`.
