# secanalysis

Small Python library for parsing and analyzing Size Exclusion Chromatography (SEC/GPC) exports.
Current focus: WinGPC text exports (RAW/ELU/MWD blocks), baseline correction, peak/border detection, and quick plotting.

## Contents

- [Features](#features)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Supported Formats](#supported-formats)
- [Core API](#core-api)
- [Development](#development)
- [CI, Versioning, Releases](#ci-versioning-releases)
- [Contributing](#contributing)
- [License](#license)

## Features

- Parse WinGPC text exports into a consistent in-memory representation (`pandas.DataFrame`-backed).
- Automatic baseline correction (SNIP via `pybaselines`) and peak/border detection (`scipy`).
- Convert elution volume → molar mass via calibration metadata (polynomial fit with linear tails).
- Plot SEC raw signals, elugram, and MWD-style curves (matplotlib).
- Compute distribution summaries (`Mp`, `Mn`, `Mw`, `Mz`) and sample chain masses from the distribution.

## Installation

### From a local checkout (recommended for development)

This repo uses `uv` and pins dependencies in `uv.lock`.

```bash
uv sync --group dev
```

To run code without installing globally:

```bash
uv run python -c "import secanalysis; print('ok')"
```

### Editable install (optional)

```bash
uv pip install -e .
```

## Quickstart

Parse a WinGPC export, auto-detect signal borders, and plot:

```python
from pathlib import Path
from secanalysis.sec_formats.win_gpc import WinGPCData

gpc = WinGPCData.from_file(Path("tests/win_gpc_sample.txt"))
gpc.autodetect_signal_borders()

fig, axes = gpc.plot()
fig.savefig("sec_overview.png", dpi=300)
```

Compute mass parameters for a selected mass window:

```python
params = gpc.calc_mass_params(lower_mass_cutoff=8e3, upper_mass_cutoff=1e6)
print(params["Mn"], params["Mw"])
```

Sample chain masses (useful for simulations):

```python
masses = gpc.sample(by="Mw", n=10_000, random_state=0)
```

## Supported Formats

### WinGPC (`WinGPCData`)

Located at `src/secanalysis/sec_formats/win_gpc.py`.

- Reads WinGPC text exports containing `RAW` and optionally `ELU` and `MWD` blocks.
- Accepts either a volume column or a time column:
  - If the volume column is missing but time exists, volume is derived from time using the `Flow` value in metadata.
  - If neither time nor volume are present, parsing fails with `ValueError`.
- If calibration metadata is present, a molar-mass calibration function is derived automatically.

## Core API

### Base type: `SECDataBase`

All parsers derive from `SECDataBase` (`src/secanalysis/sec_formats/base.py`) and provide:

- Data access:
  - `raw_volumes` (1D), `raw_signals` (2D), `signal_names` (list)
  - `baseline_corrected_signal` (baseline-corrected signals)
- Mass calibration and distributions (when calibration exists):
  - `mass_range` (volume → mass)
  - `Wm` and `calc_logMW(lower_mass_cutoff, upper_mass_cutoff)` (dw/dlog10(M))
  - `calc_mass_params(...)` → `{"Mp", "Mn", "Mw", "Mz"}`
- Border handling:
  - `autodetect_signal_borders(order=4)`
  - `add_signal_border(vol=(...), mass=(...), index=(...))`
- Visualization:
  - `plot(...)` → `(fig, axes)` where `axes` is a dict like `{"raw", "elu", "elu2", "mwd"}`
- Sampling:
  - `sample(by="Mp"|"Mn"|"Mw", n=..., random_state=...)` → `np.ndarray` of masses

## Development

### Environment

- Python: `>=3.11` (see `pyproject.toml`)
- Dependency management: `uv` (`uv.lock` is committed)

Common commands:

```bash
uv sync --group dev
uv run pytest
uv run pre-commit run -a
```

### Tests

- Framework: `pytest` (configured in `pytest.ini`)
- Tests live in `tests/` and include small fixture files like `tests/win_gpc_sample.txt`.

### Linting / Formatting

Formatting and linting are run via `pre-commit`:

```bash
uv run pre-commit install
uv run pre-commit run -a
```

Notes:

- Ruff is used for linting and formatting (via pre-commit).
- Flake8 is also enabled with a 120 char line limit (`.flake8`).

### Outputs and local artifacts

- `test_outputs/` is ignored by git and intended for local visualizations/debugging artifacts.

## CI, Versioning, Releases

- Tests run in GitHub Actions on pushes to `dev` and `test` across multiple OSes and Python versions (see `.github/workflows/py_test.yml`).
- On pushes to `main`/`master`, CI runs tests, builds, and (if configured) publishes to PyPI and tags a release as `v<version>` (see `.github/workflows/version_publish_main.yml`).
- Package version is stored in `pyproject.toml` (`[project].version`).

## License

MIT License. See `LICENSE`.
