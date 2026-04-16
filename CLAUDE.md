# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

Companion code for the book *Deep Learning in Quantitative Trading* (Zhang & Zohren, 2025; PDF in `Book/`). Content is almost entirely Jupyter notebooks organized by book chapter, backed by a small shared utility package and three CSV datasets. There is no application to build or deploy — the "product" is the notebooks themselves.

## Environment

- Managed with **uv** (not conda, despite what `README.md` still says). `pyproject.toml` + `uv.lock` are authoritative; `environment.yml` does not exist.
- Python 3.13 via `.python-version` (project requires `>=3.12,<3.14`).
- Setup: `uv sync` creates `.venv/`. Run notebooks with `uv run jupyter lab` (or select the `.venv` kernel from an existing Jupyter/VS Code session — `ipykernel` is already a dependency).
- Lint/format: `uv run ruff check .` and `uv run ruff format .` (line-length 100, target py312, isort enabled).
- Type check: `uv run pyright`. Note `[tool.pyright].include = ["src"]` in `pyproject.toml` — there is no `src/` directory, so pyright currently checks nothing. If you need type checking on actual code, widen `include` to `["Utilis"]` or the chapter you are touching.

## Repository layout

- `Chapter2/` … `Chapter7/` — one directory per book chapter, containing numbered `.ipynb` files that match the chapter's examples. Each chapter that trains models has a `model/` subdirectory where checkpoints are written (see "Notebook conventions" below).
- `Utilis/` — shared Python package (note the spelling: `Utilis`, not `Utils`; imports must match). Tiny surface: `early_stopper.EarlyStopping`, `loss.SharpeLoss` / `Neg_Sharpe`, `metrics.report_metrics`, `torch_data.MyDataset`.
- `Data/` — committed CSVs used by the notebooks: `aapl.csv` (single-name prices, Ch. 5), `portfolio_data.csv` (multi-asset panel, Ch. 5–6), `limit_order_book_data.csv` (Ch. 7).
- `Book/` — the book PDF (reference material).
- `locations.toml` — gitignored local config holding `benchmark_data = "/Volumes/..."`. Not read by anything in the repo today; it is a placeholder for pointing at an external dataset mount.

## Notebook conventions

All notebooks follow the same two-step bootstrap, which matters when writing or editing cells:

1. **sys.path shim.** Notebooks prepend the repo root so the `Utilis` package is importable:
   ```python
   current_dir = os.path.dirname(os.path.abspath("__file__"))
   project_dir = os.path.dirname(current_dir)
   sys.path.append(project_dir)
   from Utilis.early_stopper import EarlyStopping
   ```
   Consequence: **notebooks must be run from their own chapter directory** as cwd — the data loads (`pd.read_csv('../Data/aapl.csv')`) and the `project_dir` computation both assume this.

2. **Model checkpoints.** Training code instantiates `EarlyStopping(savepath='model/best_<name>', ...)`, which calls `torch.save(model.state_dict(), savepath)` on each improvement. Checkpoints are state-dicts (load with `model.load_state_dict(torch.load(path))`, not `torch.load` alone) and **are committed to the repo** — re-running a notebook end-to-end will overwrite them.

## Things to know before editing

- **The README is stale.** It describes a conda workflow and a different upstream repo URL; trust `pyproject.toml` over `README.md` for anything environment-related.
- **`Utilis/` is deliberately thin** and shared across chapters. Changing a signature there (e.g. `SharpeLoss.forward`, `MyDataset.__init__`) ripples into every chapter that imports it — grep before refactoring.
- **Data files are committed**, including the 8 MB LOB CSV. Do not regenerate or overwrite them casually.
