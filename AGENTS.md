# AGENTS.md

## Project layout

This is the companion code for *Deep Learning in Quantitative Trading* by Zhang and Zohren.
`Chapter2/` through `Chapter7/` contain the numbered Jupyter notebooks. `Utilis/` contains shared
PyTorch and reporting helpers, `Data/` contains the example CSVs, and `Book/` contains the reference
PDF and analysis notes.

## Environment

- Use `uv sync --locked` from the repository root. `pyproject.toml` and `uv.lock` define the
  environment. The Conda setup in `README.md` refers to an environment file absent from this checkout.
- Use Python 3.13 for day-to-day work, as pinned in `.python-version`. Keep code compatible with
  Python 3.12, the minimum in `requires-python` and the Ruff target. The declared range is
  `>=3.12,<3.14`.
- In an existing Jupyter or IDE session, select the project's `.venv` kernel. To launch JupyterLab
  from the repository root, use `uv run --locked --with jupyterlab jupyter lab`. The project already
  includes `ipykernel`.

## Notebook changes

- Run each notebook with its chapter directory as the working directory. Some notebooks load CSVs
  from `../Data/`, add the parent directory to `sys.path` for `Utilis` imports, or write checkpoints
  to `model/`.
- Keep notebook diffs focused on the changed cells and any outputs that need refreshing. Some
  Chapter 2 examples fetch market data via `yfinance` and require network access.
- Create the chapter's `model/` directory before running training cells that save there. These
  directories are Git-ignored. `EarlyStopping` saves PyTorch state dictionaries, which the notebooks
  load into an instantiated model.
- Preserve the `Utilis` spelling and check notebook callers before changing shared helpers. Preserve
  the bundled CSVs unless the task calls for data changes. When changing a model or backtest, check
  chronological splits and the timing of features, targets, and returns to avoid look-ahead bias.

## Verification

- Execute the affected notebook from a fresh kernel when practical. For a headless smoke test from
  the repository root, this example runs a self-contained notebook and writes the executed copy to
  a temporary directory.

  ```sh
  cd Chapter3
  uv run --locked --with nbconvert jupyter nbconvert --to notebook --execute \
    --output-dir "$(mktemp -d)" 01_regression_classification.ipynb
  ```

- Run the tests in `tests/` with `uv run --locked pytest` from the repository root.
- For Python edits, follow the Ruff settings in `pyproject.toml`, including 100-character lines
  and import sorting. Run `uvx ruff check --no-fix` and `uvx ruff format --check` on the files you
  change.
- The configured Pyright scope is `src/`, which is absent. Pass `Utilis/` explicitly when running a
  type check on the shared Python code.
