<div align="center">

<img src="docs/assets/banner.svg" alt="Research Multiobjective Optimizer" width="100%">

# Research Multiobjective Optimizer

**From engineering data to traceable design decisions**

![Version](https://img.shields.io/badge/version-0.2.0-2563eb)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Desktop](https://img.shields.io/badge/desktop-PyQt5-41CD52)
![Platform](https://img.shields.io/badge/tested-Windows-64748b)

[简体中文](README.md) · [Examples](examples/README.md) · [User guide (Chinese)](docs/user-guide.zh-CN.md) · [Architecture](docs/architecture.md)

</div>

A desktop research tool for surrogate modelling, constrained multi-objective optimization, and preference-based ranking of numerical experimental or simulation data. Configure variables, objective directions, response constraints, and validation settings in the interface without modifying the source for each domain.

The general workflow includes battery cooling, heat exchanger, and structural design presets. A separate MTPV/MTEG workflow retains hydrogen/air fuel-input and power/efficiency coupling.

![Actual application window with the synthetic battery example](docs/assets/overview.png)

*Actual running application. This battery example uses synthetic functions, not experimental or CFD results.*

## Getting started

Python 3.10+ is required. The graphical application has been tested on Windows.

```powershell
git clone https://github.com/Zhuayu16/research-multiobjective-optimizer.git
cd research-multiobjective-optimizer
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

After installation, double-click `run_gui.bat`. It prefers the project `.venv`, then tries Conda base or system Python. `RESEARCH_OPTIMIZER_PYTHON` can override the interpreter.

Choose a domain and click **载入合成示例** (load synthetic example), review the settings, then **开始优化** (start optimization). Alternatively, open an [example project](examples/README.md) with **打开项目** (open project). The current interface is in Chinese.

To install the package from this checkout:

```powershell
python -m pip install .
research-optimizer
```

There is no claimed PyPI release. The computation can also be called from Python:

```python
from mtpv_optimizer.presets import example
from mtpv_optimizer.general import run_general
from mtpv_optimizer.workflow import export_workbook

problem, data = example("exchanger")
result = run_general(data, problem)
export_workbook(result, "result.xlsx")
```

## Capabilities

| Stage | Supported features |
| :--- | :--- |
| Problem definition | Continuous, integer, and finite numeric-set variables; per-objective min/max; auxiliary responses used only in constraints |
| Data preparation | Excel sheets, CSV, TSV, record-list JSON; invalid-row audit; configurable replicate handling |
| Model assessment | Random, grouped, or forward time validation; independent holdout; per-response model comparison; prediction and error plots |
| Search | Feasibility-first NSGA-II; arithmetic constraints; training box or convex hull; optional nearest-sample coverage limit |
| Decision support | Separate predicted and input-data Pareto sets; IDEAL, TOPSIS, ARAS; multiple runs and preference comparisons |
| Reproducibility | Data and configuration in `.optproj`; Excel results, parameter JSON, data checksum, and SVG/PDF/TIFF/PNG figures |
| Experiment planning | Latin hypercube design with integer/set handling and retained center replicates |

Surrogate families include linear/quadratic response surfaces, Gaussian processes, SVR, MLP, random forests, ExtraTrees, and gradient boosting. General-mode Auto selects each response model by cross-validation NRMSE.

## Examples and screenshots

Each general example provides 60 synthetic samples (data seed 73), a CSV file, and a project that can be opened directly. See [example files and definitions](examples/README.md). The private 46-case MTPV/MTEG dataset is not distributed; see [input format and private-data notes](data/README.md).

<details>
<summary><b>Constraints and validation</b></summary>

![Response constraint](docs/assets/constraints.png)

![Cross-validation](docs/assets/validation.png)

The battery functions can be represented by a quadratic model; the near-zero errors shown here demonstrate the workflow and do not establish real engineering accuracy.

</details>

<details>
<summary><b>Two- and four-objective examples</b></summary>

![Heat exchanger example](docs/assets/exchanger.png)

![Structural design example](docs/assets/structure.png)

</details>

<details>
<summary><b>Dedicated MTPV/MTEG workflow</b></summary>

![MTPV/MTEG application](docs/assets/mtpv.png)

This screenshot shows the dedicated interface before loading data. Import your own table to run this workflow. See the [Chinese guide](docs/user-guide.zh-CN.md) for physics definitions and the scope of its response-holdout check.

</details>

All six screenshots come from actual application runs. [Capture script](scripts/capture_screenshots.py).

## Verification and limitations

The complete local research checkout passed 35 automated tests and both Qt interface checks. The public checkout runs 26 tests and explicitly skips nine checks requiring private data. Its installed-package startup check opens both interfaces without bundling research data. The package check reused existing Anaconda dependencies; a clean-machine and cross-platform verification is still pending.

```powershell
python -m unittest discover -s tests -v
python tests/gui_general_smoke.py
python tests/gui_smoke.py
```

- Inputs are numerical tables; unordered categorical variables are not supported. Units are annotations and are not converted automatically.
- General training requires at least ten distinct designs and sufficient variable levels. Bounds cannot exceed the training ranges.
- Nearest-sample distance describes coverage, not a predictive confidence interval.
- Auto selection and reported CV metrics share folds; nested validation is not implemented. General-mode holdout data are excluded from selection, fitting, and search-domain construction.
- Predicted candidates require experimental or simulation confirmation. The software does not launch Fluent solves or import binary `.dat.h5` files as tables.
- Project files store data and settings for rerunning, rather than serialized executable models.

## Contributing and licensing

Report reproducible issues or domain use cases through [Issues](https://github.com/Zhuayu16/research-multiobjective-optimizer/issues). See [contribution instructions](CONTRIBUTING.md) and [version history](CHANGELOG.md).

No open-source license has been specified for this repository. Use and redistribution permissions are subject to an explicit license from the repository owner.
