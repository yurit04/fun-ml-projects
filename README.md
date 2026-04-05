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

- **[`uv`](https://docs.astral.sh/uv/)** — install it using the [official install guide](https://docs.astral.sh/uv/getting-started/installation/).
- **Python 3.10 or newer** — `uv` can install and pin a version for you (`requires-python` in `pyproject.toml` is `>=3.10`).

### Install

Run everything from the **repository root** (the directory that contains `pyproject.toml`).

The **`[notebook]`** extra pulls in Jupyter and `ipykernel` so you can open the `.ipynb` files. **PyTorch** is a normal dependency because `time_series/basic_rnn_pytorch.ipynb` needs it, so the first install can be large and take a while.

```bash
# Have uv download a Python that satisfies 3.10+ (adjust 3.12 if you like)
uv python install 3.12

# Create the virtual environment
uv venv --python 3.12

source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -e ".[notebook]"
```

If you prefer not to activate the venv, target it explicitly:

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -e ".[notebook]"
```

On Windows, use `.venv\Scripts\python.exe` in place of `.venv/bin/python`.

`uv pip install` reads the same `pyproject.toml` metadata as `pip` (`dependencies` and optional extras such as `[notebook]`).

### Optional: named Jupyter kernel

After the environment is installed and activated:

```bash
python -m ipykernel install --user --name fun-ml-projects --display-name "Python (fun-ml-projects)"
```

Then choose **Python (fun-ml-projects)** when opening notebooks under `finance/`, `miscellaneous/`, or `time_series/`. In Cursor or VS Code you can instead pick the interpreter **`.venv/bin/python`** directly.
