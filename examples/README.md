# Examples

Runnable notebooks for trying `corr_vars` without data access or an account.

| notebook | what it does |
|---|---|
| [`charite_demo_quickstart.ipynb`](charite_demo_quickstart.ipynb) | Builds an ICU cohort on the public Charité demo dataset, applies inclusion criteria with a flowchart, adds variables over time windows, and ends with a Table 1 and two plots. Runs top to bottom in a few minutes. |

The notebooks ship without outputs: run them yourself (**Run → Run All Cells**).

## The demo services

- **Data**: `charite_mockdata`, a *fully synthetic* ICU dataset (20'000 ICU stays) in the
  table layout of the Charité data warehouse, at <https://demo-data.corr.icu/v1>. The
  `charite_demo` source reads it over HTTPS with DuckDB, fetching only what each query
  needs.
- **Variable definitions**: the demo Concepts API at <https://demo-concepts.corr.icu>
  (project `demo`, public key `cak_demo_0d2b53ac`, already filled in in the notebook).
  Sign in as `demo` to browse the concepts in the web app.

The data contain no real patients and are not clinically valid; results demonstrate the
workflow only.

## Running the quickstart

With [uv](https://docs.astral.sh/uv/), no installation step needed. Download
`charite_demo_quickstart.ipynb` and run, in the same folder:

```shell
uv run --with "corr_vars[charite-demo] @ git+https://github.com/CUB-CORR/corr_vars.git" \
    --with jupyterlab jupyter lab charite_demo_quickstart.ipynb
```

From a checkout of this repository:

```shell
uv run --extra charite-demo --with jupyterlab jupyter lab examples/charite_demo_quickstart.ipynb
```

The `charite-demo` extra adds `pycomorb` (Elixhauser score); everything else the notebook
uses (polars, matplotlib, tableone) comes with `corr_vars`.

Optional: to run against a mirror or a local copy of the dataset, set
`CORR_DEMO_DATA_URL` (a URL or a folder) and/or `CORR_DEMO_CONCEPTS_URL` before starting
Jupyter.

## Next steps

- The full [demo tutorial](https://docs.corr-vars.de/tutorials/charite_demo.html) in the
  documentation.
- All demo concepts in the [demo Concepts Browser](https://demo-concepts.corr.icu).
