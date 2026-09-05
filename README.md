# CORR-VARS – Streamlining Real-World Evidence Studies and Bedside Use <img src="docs/_static/corr_favicon.png" align="right" width="100"/>
[![CI](https://img.shields.io/github/actions/workflow/status/CUB-CORR/corr_vars/tests.yml?branch=main)](https://github.com/CUB-CORR/corr_vars/actions/workflows/tests.yml)
[![DOI](https://zenodo.org/badge/DOI/UPCOMING.svg)](https://doi.org/UPCOMING)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Build and manage clinical study cohorts from ICU and hospital data sources. Add, filter and export variables whose definitions are served — and versioned — by the CORR Concepts API.

## Installation

```shell
uv add git+https://github.com/CUB-CORR/corr-vars.git
```

Requires Python ≥ 3.10.

## Quick Start

```python
from corr_vars import Cohort

cohort = Cohort(
    obs_level="icu_stay",          # "patient" | "hospital_stay" | "icu_stay" | "procedure"
    sources={"reprodicu": {"path": "/path/to/reprodICU_files"}},
    project="my_project",          # project on the Concepts API
    # api_key="corr_..."           # or set CORR_CONCEPTS_API_KEY
)

# Add a pre-defined variable
cohort.add_variable("blood_sodium")

# Apply inclusion criteria and visualise the flowchart
cohort.include_list([
    {"variable": "age_on_admission", "operation": ">= 18", "label": "Adults"},
    {"variable": "icu_length_of_stay", "operation": "> 2",  "label": "ICU stay > 2 days"},
])
cohort.figureone

# Export
cohort.save("my_cohort.corr3")
cohort.to_csv("output/")
```

## Observation Levels

| `obs_level`      | Primary key      | One row per …         |
|------------------|------------------|-----------------------|
| `"patient"`       | `patient_id`     | patient               |
| `"hospital_stay"` | `case_id`        | hospitalisation       |
| `"icu_stay"`      | `icu_stay_id`    | ICU admission         |
| `"procedure"`     | `procedure_id`   | surgical procedure    |

## Data Sources

Sources are plugins: each is a package under `src/corr_vars/sources/` that
supplies cohort data and, optionally, variable extraction. Multiple sources can
be combined in a single `Cohort` call.

Two sources ship with the package: `reprodicu`, a ready-to-use example, and
`local_datasource`, a skeleton you bind to your own store.

### ReprodICU

The plug-and-play example source. Download the
[reprodICU](https://github.com/cub-corr/reprodicu) parquet files and pass their
folder as `path` — nothing else to configure.

```python
sources = {
    "reprodicu": {
        "path": "/path/to/reprodICU_files",
        "include_datasets": [],
        "exclude_datasets": [],
    }
}
```

### Local data source (`local_datasource`)

A source skeleton, pinned local in `routing.toml`: it ships the variable
classes (`NativeStatic`, `DerivedStatic`, `DerivedDynamic`, `NativeDynamic`,
`ComplexVariable`) and the extraction pipeline, but no store. Bind it to your
own data in two places:

1. Replace `cohort.load_data(obs_level, source_kwargs)` so it returns the
   observation frame (primary key plus the level's `t_min`/`t_max` anchors).
2. Subclass `NativeExtractor`, implement `extract(cohort, id_column)`, and set
   `NativeDynamic.extractor_class` to your subclass.

Then fill `mapping/vars.json` and `mapping/variables.py` with your variable
definitions, or route the source to a Concepts API endpoint instead. See the
[local_datasource API reference](https://docs.corr-vars.de/api/local_datasource.html).

### Custom variables

The aggregation classes are source-agnostic — construct one directly and add it
to any cohort, including a reprodICU one:

```python
from corr_vars import Cohort
from corr_vars.sources.local_datasource import DerivedStatic, NativeStatic

cohort = Cohort(
    obs_level="icu_stay",
    sources={"reprodicu": {"path": "/path/to/reprodICU_files"}},
    project="my_project",
)

cohort.add_variable(NativeStatic(
    var_name="max_sodium_icu",
    select="!max value",
    base_var="blood_sodium",
    tmin="icu_admission",
    tmax="icu_discharge",
))
cohort.add_variable(DerivedStatic(
    var_name="hypernatremia",
    requires=["max_sodium_icu"],
    expression="max_sodium_icu > 145",
))
```

`base_var` and `requires` resolve through the cohort's own sources. Pass
`tmin`/`tmax` inside the constructor, not to `add_variable`. Full reference:
[aggregation variables](https://docs.corr-vars.de/api/aggregation.html).

## Variable Definitions

Variable definitions are not bundled with this package. They are fetched per
concept and per source from the CORR Concepts API, which also records the exact
version that was served. Routing is declared in
`src/corr_vars/concepts/routing.toml`:

```toml
[local]
sources = ["local_datasource"]

[[endpoint]]
url = "https://concepts.example.edu/api"
sources = ["*"]
```

A cohort authenticates against that endpoint on construction, so a missing
`project` or a bad key fails before the (slow) data load:

```python
cohort = Cohort(project="my_project", api_key="corr_...")
cohort = Cohort(project="my_project", date="2025-06-30")   # freeze to a day
cohort.add_variable("blood_sodium::v3")                    # pin one variable
```

Point a cohort at a different endpoint with `Cohort(concepts_api_url=...)`.
Sources listed under `[local]` in `routing.toml` are never fetched; their
definitions are read from the source's bundled `mapping` module and keep working
with no network and no API key. `local_datasource` is the one bundled source
listed there.

## Development

### Setup

```shell
git clone https://github.com/CUB-CORR/corr-vars.git
cd corr-vars
uv sync --group dev
uv run pre-commit install
```

Pre-commit hooks enforce uniform formatting and catch common errors. When a hook modifies a file, stage the change and commit again.

### Running Tests

If you have developed new features, please add relevant tests to the `tests/` folder.

```shell
uv run pytest tests/integration
```

Integration tests (`tests/integration`) run automatically once a pull request to `main` is opened. Results are attached to the pull request and a test report will be generated. Do not merge until tests pass.

`tests/sources` is hand-run: it needs the source's own data files plus a Concepts API project and key. CI runs `tests/integration` only, which needs neither.

### Building the Documentation

```shell
cd docs
make clean && make html
# or: rm -rf _build && uv run sphinx-build -M html . _build
```

The HTML output is written to `docs/_build/html/`. Documentation is automatically published to [docs.corr-vars.de](https://docs.corr-vars.de) on merge to `main`.

### Contributing

See the [Contributing Variables](https://docs.corr-vars.de/contributing_variables.html) guide for variable contributions in the documentation.

1. Create an issue or feature request on GitHub.
2. Branch off `main` using the naming convention `IssueNumber-IssueTitle`.
3. Commit your changes and push to your branch (i.e. variables or new features)
4. Run unit tests as described [above](#running-tests) by opening a pull request or running them manually.
5. If you have changed anything relevant to the documentation update the documentation as described [above](#building-the-documentation). **This is not done automatically!**

## License

MIT — see [LICENSE.md](LICENSE.md).
