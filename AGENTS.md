# CORR-VARS (`corr_vars`)

- Python library (≥ 3.10) for building clinical study cohorts
- Primary user-facing class is `Cohort`
    - Extracts static and dynamic (time-series) variables from one or more data sources
    - Tracks inclusion/exclusion steps
    - Exports results in several formats

## Repository layout

```
docs/                      # Sphinx documentation
├── _build/                # Documentation build output
src/corr_vars/
├── core/
│   ├── cohort.py          # Cohort class — the main user API
│   ├── variable/          # Variable classes package
│   │   ├── base.py        # Variable base class
│   │   └── free_days.py   # FreeDaysVariable (source-agnostic free-day outcome)
│   ├── steps.py           # TimeFilterStep, CleaningStep, PyFuncStep — reusable
│   │                      # extraction-pipeline stages shared by all sources
│   ├── change_tracker.py  # ChangeTrackerPipeline for inclusion/exclusion flowcharts
│   └── file_manager.py    # Temporary-directory lifecycle
├── concepts/              # CORR Concepts API — remote variable definitions
│   ├── routing.toml       # Global routing table (packaged, not user-scoped)
│   ├── routing.py         # Route / ApiRoute / LocalRoute / Endpoint, load_routing()
│   ├── spec.py            # parse_variable_spec() — "[taxonomy/]var_name[::version]"
│   ├── client.py          # ConceptsApiClient, Concept, SourceConcept, VersionInfo
│   ├── files.py           # Content-addressed file cache, materialise_files()
│   ├── compile.py         # compile_snippet() — executes served `py` source
│   ├── resolver.py        # ConceptResolver — per-cohort façade over all of it
│   └── standards.py       # Publication standards (check_source, format_findings)
├── definitions/
│   ├── typing.py          # ObsLevel enum, Protocols (VariableProtocol, VariableLoaderProtocol,
│   │                      # VariableCallable, ExtractedVariable), VariableContext,
│   │                      # RequirementsDict, TypedDicts, Literals, TypedDicts, Literals,
│   │                      # TimeAnchorColumn, DateLiteral, OffsetLike
│   ├── constants.py       # DYN_COLUMNS, COL_ORDER, PRIMARY_KEYS, UNBOUNDED_TMIN, UNBOUNDED_TMAX
│   └── exceptions.py      # CohortDataError, VariableNotFoundError, ConceptsApiError, …
├── sources/               # Source plugin system
│   ├── __init__.py        # SOURCES list (auto-discovered), get_src_module(), guess_variable_source()
│   ├── var_loader.py      # MultiSourceVariable, load_variable(), load_default_variables()
│   ├── config_loader.py   # load_default_config_data()
│   ├── cohort_loader.py   # load_cohort_data()
│   ├── local_datasource/  # Source skeleton — pinned local, no backing store
│   │   ├── config.py      # SourceDict/SourceDictPartial (database, conn_args), DEFAULTS
│   │   ├── cohort.py      # load_data() raises NotImplementedError — replace to bind a store
│   │   ├── extract.py     # aggregation classes live here: BaseDynamic, NativeExtractor,
│   │   │                  # NativeDynamic, ComplexVariable, NativeStatic, DerivedStatic,
│   │   │                  # DerivedDynamic, VariableLoader
│   │   ├── helpers.py     # parse_select()
│   │   └── mapping/
│   │       ├── loader.py     # JSON loading utilities
│   │       ├── variables.py  # `py` functions, looked up by var_name (ships empty)
│   │       └── vars.json     # bundled declarative definitions (ships empty)
│   └── reprodicu/         # Parquet-based ICU source
│       └── mapping/       # units.json
├── utils/
│   ├── base.py            # Dependency-free primitives: as_expr, columns, struct_fields,
│   │                      # col_length, row_length, is_empty, column_selector, normalize_offset
│   ├── debug.py           # print_cohort_debug_info, print_debug_info
│   ├── frames.py          # DataFrame conversion, time-series joins, cleaning
│   ├── helpers.py         # Variable-extraction primitives
│   ├── logging.py         # CustomFormatter, configure_logger_level_and_handlers
│   ├── outcomes.py        # free_days engine (source-agnostic free-day / VFD-style outcomes)
│   ├── tableone.py        # TableOne → LaTeX / Markdown / HTML / PDF export
│   └── time.py            # TimeAnchor, TimeWindow, WindowRelation, WindowOverlap
├── plot/                  # Visualisation helpers
tests/                     # Unit tests
├── assets/data/           # Static test fixtures
├── sources/               # Source-specific tests
│   └── reprodicu/         # Requires local parquet files.
└── integration/           # Hypothesis / Mock based testing (no DB, runs on PRs to `main`)
                           # incl. test_local_datasource_variables.py
```

## Key types and conventions

### `ObsLevel` (observation level)

Defined in `corr_vars/definitions/typing.py`:

| `obs_level` string  | `ObsLevel` member        | Primary key    | `t_min`              | `t_max`              |
|---------------------|--------------------------|----------------|----------------------|----------------------|
| `"patient"`         | `ObsLevel.PATIENT`       | `patient_id`   | `birthdate`          | `censordate`         |
| `"hospital_stay"`   | `ObsLevel.HOSPITAL_STAY` | `case_id`      | `hospital_admission` | `hospital_discharge` |
| `"icu_stay"`        | `ObsLevel.ICU_STAY`      | `icu_stay_id`  | `icu_admission`      | `icu_discharge`      |
| `"procedure"`       | `ObsLevel.PROCEDURE`     | `procedure_id` | `or_time_begin`      | `or_time_end`        |

- Each level also defines default `t_eligible` and `t_outcome` values
- Dynamic variable data is filtered to records between `t_min` and `t_max` by default

### `sources` dict structure

The `sources` argument to `Cohort.__init__` is a `SourceDictPartial` (TypedDict, total=False):

```python
{
    "reprodicu": {
        "path": "/path/to/reprodICU_files",
        "include_datasets": [],
        "exclude_datasets": [],
    },
}
```

**`SourceDictPartial` → `SourceDict`**: `config_loader.load_default_config_data()` converts the user config per source in four steps:
1. **Migrations** — relocate legacy top-level keys to their canonical nested position (declared in `config.py:MIGRATIONS` as `{old_key: (parent, new_key)}`)
2. **Unknown-key warning** — keys absent from `DEFAULTS` are logged as warnings
3. **Deep merge** — user values are merged onto a `deepcopy` of `DEFAULTS` (user wins)
4. **Type cast** to `SourceDict` for static type checking.

The `reprodicu` defaults (`sources/reprodicu/config.py`):

```python
DEFAULTS = {
    "path": None,
    "exclude_datasets": [],
    "include_datasets": [],
}
```

The `local_datasource` defaults (`sources/local_datasource/config.py`) are
`database` and `conn_args` (`hostname`, `username`, `password`,
`password_file`), with `MIGRATIONS` relocating a legacy top-level
`password_file`. They are deliberately minimal: a deployment that binds the
skeleton to a real store extends `SourceDict` and `DEFAULTS` with whatever that
store needs.

Each source defines its own `DEFAULTS` and optional `MIGRATIONS` in `sources/<source>/config.py`.

## Concepts API (variable definitions are fetched, not bundled)

Variable definitions no longer live only in this package. They are served per
concept and per source by the CORR Concepts API: the declarative config that
used to sit in `vars.json`, the `py` function that used to sit in
`mapping/variables.py`, the data files those functions read, and the version
metadata describing exactly what was served.

### Routing

`src/corr_vars/concepts/routing.toml` is the single global routing table. It
ships inside the package and is identical for every user — no `~/.config` file,
no environment variable. The only override is `Cohort(...)` kwargs.

| Source | Route |
|---|---|
| `reprodicu` | Concepts API (definitions not yet published → resolves to nothing) |
| `local_datasource` | pinned local (`[local].sources`) — bundled `mapping/vars.json`, no network, no key |

A source named under `[local].sources` is **pinned local**: it is never fetched,
must work offline with no key, and reads its definitions from its own bundled
`mapping` module. `local_datasource` is the one bundled source pinned there.

Resolution order: pinned-local → ordered `[[endpoint]]` list (first match wins)
→ local. Defaults for unqualified references (`corr_v1` / `latest`) live in the
same file.

**There is no fallback to `vars.json`.** A source routed to an endpoint is
served only by that endpoint; every transport, auth and server failure raises.
The one non-error case is a concept the endpoint does not publish for a source —
that source simply contributes nothing. (The only place a failure is downgraded
to a warning is the name listing behind `cohort.search_widget`, which extracts
no data.)

### Variable references

`add_variable()` takes `[taxonomy/]var_name[::version]`. The version selector is
`latest`, `vN`, an ISO date `YYYY-MM-DD` (as-of), or `draftNNNN` (a **global**
Config id, not "draft N of this concept"). Missing parts fall back to the cohort
defaults. A pin applies to that variable only — its `requires` dependencies
still resolve at the cohort default, so pin the cohort (`Cohort(date=...)`) to
freeze a whole dependency graph.

One name may resolve to a **group** of concepts (common with ATC codes). Each
member becomes its own contributor and the frames are concatenated, exactly like
a multi-source variable. Version pins do not work on a group
(`AmbiguousConceptError`).

### `Cohort` configuration

| Setting | Source | Notes |
|---|---|---|
| `project` | `Cohort(project=...)` | **required** by the API on every read |
| API key | `Cohort(api_key=...)` or `CORR_CONCEPTS_API_KEY` | never persisted by `save()` |
| `taxonomy` | `Cohort(taxonomy=...)` | defaults to `routing.toml` (`corr_v1`) |
| `version` | `Cohort(version=...)` | defaults to `routing.toml` (`latest`) |
| `date` | `Cohort(date=...)` | global as-of; **exclusive** with `version` |
| endpoint | `Cohort(concepts_api_url=...)` | overrides `routing.toml`; local stays local |
| file cache | `CORR_VARS_CACHE_DIR` | defaults to `$XDG_CACHE_HOME/corr_vars/concepts` |

`project` and the key are authenticated at construction time
(`ConceptResolver.authenticate()`, one `GET /projects` per endpoint in use), so a
bad key or unknown project raises before the slow data load. `Cohort.load()`
does not check — a saved cohort carries its own data.

`cohort.concept_versions` maps a variable name to a **list** of records, one per
contributor (per source, and per concept when a name resolved to a group).
`save()` persists it plus the resolver settings, never the key.

### Exceptions (`definitions/exceptions.py`)

| Exception | Base | Meaning |
|---|---|---|
| `ConceptsApiError` | `Exception` | any API failure; fatal under no-fallback |
| `ConceptsApiConfigurationError` | `ConceptsApiError` | misconfigured cohort |
| `ProjectNotFoundError` | `…ConfigurationError` | project unknown — fix the project, not the variable |
| `ConceptsLicenseError` | `ConceptsApiError` | 403; no corr_vars-side remedy, a lead must re-accept the license |
| `AmbiguousConceptError` | `ConceptsApiError` | version pin applied to a grouped name |
| `ConceptNotFoundError` | `ConceptsApiError`, `VariableNotFoundError` | concept does not exist |

### The exec-namespace contract

A served `py` definition arrives as **bare function source** — no imports, no
module preamble. It executes against a namespace built by the source's `py_env`
module (`sources/<source>/py_env.py`), which declares `IMPORT_NAMES`,
`SHARED_HELPER_NAMES` and `MODULE_CONSTANT_NAMES` and builds them from
`py_env.VARIABLES_MODULE` — `sources/<source>/py_namespace.py`, the module that
holds the import header, the shared helpers and the module constants a snippet
expects to find already bound. `concepts/compile.py` injects `__file__`,
`__name__`, `__builtins__` and `getfile` on top.
`concepts.compile.get_py_compiler(source)` returns `None` for a source without
a `py_env`; such a source cannot execute served snippets.

Variable functions are deliberately **not** in the namespace: a snippet must
declare what it needs via `requires` instead of calling a sibling definition.

**Adding an import, a shared helper, or a module constant to
`py_namespace.py` means updating `py_env.NAMESPACE_NAMES` in the same
change.** Two guards catch a miss — `build_py_namespace()` at fetch time, and
the `py-snippet-namespace` publication standard. Treat a failure in either as
the intended signal, not as flakiness.

### Publication standards

`concepts/standards.py` holds the checks a definition must pass to be served —
locally a function has its whole module behind it, remotely it has only what
`py_env` declares.

```python
from corr_vars.concepts.standards import check_source, format_findings

print(format_findings(check_source("reprodicu")))
```

Every standard is a no-op for a source without a `py_env` module.

### Attached data files

Definitions address their files by uuid, resolved against the manifest served
*with this config*:

```python
mapping = pl.read_csv(getfile("f5497211-1667-58ef-a16c-fb97b95b3987"))
```

`Path(__file__).parent / ...` and `var.files["postcode/postcode_mapping.csv"]`
also work (`__file__` is rebased onto the materialised file directory), but
prefer `getfile`. A **shared helper cannot call `getfile`** — it runs against its
module's globals, so the variable function does the lookup and passes the path
in. Files are cached content-addressed by sha256; a mismatch is fatal.

### Extraction cache

A cached extract is keyed by variable, contributor and id column — **never by
version** — so it survives a redraft. Clear it with
`add_variable(..., use_cache=False)` (that variable, all contributors) or
`cohort.clear_variable_cache(name)` / `cohort.clear_variable_cache()`.

### What stays local

Under `reprodicu/mapping/`: `units.json` (`UNITS`) — extraction inputs, not
definitions.

Under `local_datasource/mapping/`: `vars.json` and `variables.py` — the source is
pinned local, so both are read from the package instead of an endpoint. Both ship
empty (no `variables`, no `corr_defaults.default_vars`); a deployment fills them.

Not local: the variable definitions themselves. An API-routed source's `mapping`
module exposes no `VARS`; `ALL_VAR_NAMES` is a lazy module attribute whose first
access lists the fully-qualified names (`corr_v1/<name>`) from the API
(`ConceptsApiClient.list_concept_names`), needing `CORR_CONCEPTS_PROJECT` and
`CORR_CONCEPTS_API_KEY`.

The default variable list is the local seam: every source exposes it as
`DEFAULT_VARS` on its `mapping` module — from the `corr_defaults.default_vars`
block of `vars.json` for a pinned-local source, or its own `defaults.json` —
and `var_loader._collect_default_variable_list_by_source()` reads only that.

`cohort.search_widget` follows the same split
(`var_loader.load_raw_variable_configs()`): a local source contributes its full
`variables` mapping, an API-routed one contributes its variable *names* listed
from the endpoint, each mapped to `API_ROUTED_PLACEHOLDER_CONFIG`. Listing
failures are logged and the source is skipped — the widget is a convenience and
must never raise. Browsing a full definition happens in the Concepts Browser web
app.

### `obs` and `obsm`

- `cohort.obs` — `pl.DataFrame`, one row per observation, static columns only. Backed by `cohort._obs`.
- `cohort.obsm` — `ObsmDict` (thin `dict[str, pl.DataFrame]` wrapper with error-friendly `__getitem__`), keyed by variable name, each value a `pl.DataFrame` of dynamic (time-series) data. Backed by `cohort._obsm`.
- Assign `cohort.obs = new_df` to replace
- Assign `cohort.obs["new_var"] = new_df` to replace
- **All code must use polars only — never pandas for new code.**

Dynamic DataFrames follow `constants.COL_ORDER`:
```
[primary_key] + [recordtime, recordtime_end, recordtime_relative, recordtime_end_relative,
                 value, value_unit, attributes]
```

- Missing `DYN_COLUMNS` are filled with `null`
- Extra columns trigger a deprecation warning and must be moved to `attributes` (`pl.Struct`)

## Variable pipeline: `Cohort.add_variable`

```py
def add_variable(
    self,
    variable: str | VariableProtocol | MultiSourceVariable,
    save_as: str | None = None,
    tmin: TimeAnchorColumn | None = None,
    tmax: TimeAnchorColumn | None = None,
    use_cache: bool = True,
) -> MultiSourceVariable: ...
```

| `variable` type | What happens |
|---|---|
| `str` | A `[taxonomy/]var_name[::version]` reference. `load_variable()` collects a config per source via `_collect_variable_configs_by_source()` — from the Concepts API for a routed source (`ConceptResolver.fetch_configs()`, which also compiles the served `py` snippet) or from the bundled `vars.json` + `mapping/variables.py` for a pinned-local one. It then merges project-local overrides from `cohort.get_variable_definition()`, transfers any embedded `tmin`/`tmax` config keys into the time window, checks `compatible_with` against the cohort's obs level, calls `VariableLoader(var_name, time_window, **kwargs)` per contributor, and wraps the results in `MultiSourceVariable`. |
| `VariableProtocol` | Wrapped into a single-source `MultiSourceVariable`. |
| `MultiSourceVariable` | Passed as-is. |

- `tmin` / `tmax` are `TimeAnchorColumn`
    - Either column name string (`"icu_admission"`) or a `(column, delta)` tuple (`("icu_admission", "+2h")`)
    - Valid delta units: `ns|us|ms|s|m|h|d|w|mo|q|y` with optional `+`/`-`.

```
cohort.add_variable(variable, save_as, tmin, tmax)
         │
         ▼
 ┌── Stage 1: Resolve input ───────────────────────────────────────┐
 │  str path:                                                      │
 │    parse_variable_spec() — taxonomy / name / version            │
 │    Build TimeWindow from tmin / tmax                            │
 │    _collect_variable_configs_by_source()                        │
 │      remote source → ConceptResolver.fetch_configs()            │
 │      local source  → bundled vars.json + variables.py           │
 │      + project overrides, + concept_versions recorded           │
 │    _transfer_time_window_from_variable_config()                 │
 │      → pop tmin/tmax from config into time_window               │
 │    _ensure_compatibility() — check obs level                    │
 │    VariableLoader(var_name, time_window, **kwargs) per source   │
 │    MultiSourceVariable({src: VariableProtocol, ...})            │
 │  VariableProtocol → wrapped into a single-source MSV            │
 │  MultiSourceVariable → used as-is                               │
 └─────────────────────────────────────────────────────────────────┘
         │
         ▼
 ┌── Stage 2: Pre-validate MultiSourceVariable ────────────────────┐
 │  All per-source VariableProtocols must agree on:                │
 │    var_name, dynamic flag, time_window  →  else AssertionError  │
 └─────────────────────────────────────────────────────────────────┘
         │
         ▼
 ┌── Stage 3: Extract per source ──────────────────────────────────┐
 │  For each source variable (typically):                          │
 │    var.extract(cohort) → pl.DataFrame                           │
 │      ├── _get_required_vars()   load declared dependencies      │
 │      ├── _custom_extraction()   query DB / read parquet         │
 │      ├── _call_var_function()   apply py() transformation       │
 │      ├── _add_time_window()     join tmin/tmax anchor columns   │
 │      ├── _timefilter()          filter rows to time window      │
 │      └── _apply_cleaning()      clip/drop out-of-range values   │
 └─────────────────────────────────────────────────────────────────┘
         │
         ▼
 ┌── Stage 4: Post-validate ───────────────────────────────────────┐
 │  validate_primary_key() — primary key column must be present    │
 └─────────────────────────────────────────────────────────────────┘
         │
         ▼
 ┌── Stage 5: Post-process ────────────────────────────────────────┐
 │  Dynamic only:                                                  │
 │    add_relative_times() — compute recordtime_relative /         │
 │      recordtime_end_relative as seconds from cohort.t_min       │
 │    unify_and_order_columns() — enforce COL_ORDER, fill missing  │
 │      DYN_COLUMNS with null                                      │
 │  Multi-contributor: pl.concat(how="diagonal_relaxed"),          │
 │    optionally prepend a "data_source" column. A grouped concept  │
 │    name contributes one member per concept, all under the same  │
 │    source name.                                                 │
 └─────────────────────────────────────────────────────────────────┘
         │
         ▼
 ┌── Stage 6: Save ────────────────────────────────────────────────┐
 │  dynamic=True  → cohort._obsm[save_as] = data                   │
 │  dynamic=False → assert one value per primary key,              │
 │                  left-join to cohort._obs (drop existing col)   │
 │  cohort._validate_cohort(): obs primary-key set must be stable  │
 └─────────────────────────────────────────────────────────────────┘
```

### Hand-built variables

The `local_datasource` aggregation classes are source-agnostic and can be
constructed directly: `cohort.add_variable(NativeStatic(var_name=..., select=...,
base_var=..., tmin=..., tmax=...))`, likewise `DerivedStatic`, `DerivedDynamic`,
`ComplexVariable`. Pass `tmin`/`tmax` inside the constructor — passing them to
`add_variable` for an object warns.

`guess_variable_source()` tags such an object as `local_datasource`. Its
dependencies still resolve through the cohort's own sources, because
`Variable._dependency_sources(cohort)` (`core/variable/base.py`) restricts the
lookup to the variable's own source **only when the cohort actually uses that
source**, and returns `None` (all cohort sources) otherwise. So a hand-built
`NativeStatic` on a reprodicu cohort finds `base_var="blood_sodium"` in reprodicu.

## Shared extraction steps (`core/steps.py`)

Three reusable pipeline stages, shared by all sources rather than reimplemented
per source:

| Step | What it does |
|---|---|
| `TimeFilterStep` | Joins `tmin`/`tmax` bounds from `cohort.obs` onto variable data and filters rows to the window |
| `CleaningStep` | Applies the `cleaning` bounds (clip or nullify) via `utils.frames.apply_cleaning` |
| `PyFuncStep` | Calls the `py()` function — builds the `VariableContext` (including `files`), and handles the legacy pandas round-trip when `py_ready_polars` is false |

## Source plugin system

- Sources are auto-discovered sub-packages under `sources/<source_name>`
- `sources/__init__.py` scans for packages with an `__init__.py` and populates `SOURCES`

### Plugin layout

```
sources/<source_name>/
├── __init__.py        # public exports: SourceDictPartial, SourceDict, VariableLoader
├── config.py          # DEFAULTS, SourceDictPartial / SourceDict TypedDicts, MIGRATIONS
├── cohort.py          # load_data(obs_level, config) -> pl.DataFrame — initial Cohort.obs
├── extract.py         # VariableLoader factory + Variable subclasses
├── helpers.py         # source-specific utilities
├── py_env.py          # exec-namespace contract for API-served `py` snippets (optional)
├── py_namespace.py    # the namespace module py_env builds from (optional)
└── mapping/
    ├── loader.py      # JSON loading utilities
    ├── defaults.json  # cohort default variable list → DEFAULT_VARS
    ├── units.json     # unit table; exposes UNITS (reprodicu only)
    ├── vars.json      # declarative variable definitions — pinned-local sources only
    └── variables.py   # VariableCallable functions, looked up by var_name — same
```

An API-routed source bundles **no** `vars.json` and **no** `variables.py`: its
definitions and `py` snippets come from the endpoint, and the data files those
snippets read live in the API's per-source file store, reached with
`getfile("<uuid>")`. `py_env.py` plus `py_namespace.py` are what make a served
snippet runnable; a source that serves no `py` snippets ships neither.

The two bundled sources are the reference layouts: `local_datasource` for a
pinned-local source (bundled `vars.json` + `variables.py`, no `py_env`), and
`reprodicu` for an API-routed one.

### VariableLoader factory contract

Every source exposes a `VariableLoader` in `extract.py` satisfying `VariableLoaderProtocol`:

```python
def VariableLoader(
    var_name: str, time_window: TimeWindow, **kwargs
) -> VariableProtocol: ...
```

- All fields of the variable config — served or bundled — are forwarded as `**kwargs`:
    - e.g. `type`, `table`, `where`, `requires`, `py`, `cleaning`, `py_ready_polars`, …
- The return value must satisfy `VariableProtocol`:
    - Attributes `var_name`, `dynamic`, `time_window`
    - Method `extract(cohort) -> pl.DataFrame`

### Variable config schema

The same shape whether the config is served by the Concepts API or read from a
pinned-local `vars.json` — where it sits under a top-level `"variables"` key,
alongside the `corr_defaults.default_vars` block that source exposes as
`DEFAULT_VARS` (a source may keep that block in `defaults.json` instead):

```jsonc
{
  "variables": {
    "<var_name>": {
      // --- shared fields (all types) ---
      "type": "native_dynamic | native_static | derived_dynamic | derived_static | complex | free_days",
      "compatible_with": ["icu_stay", "hospital_stay"],
      "cleaning": {"value": {"low": 80, "high": 190}},
      "py_ready_polars": true,              // omit or false for legacy pandas functions
      // "py" is the transformation function: served with the config by the API,
      // or injected from mapping/variables.py for a pinned-local source. Never set by hand.

      // --- native_dynamic / native_static ---
      "table": "labs",
      "where": "SQL WHERE clause",
      "value_dtype": "Float64",            // polars dtype name; Python-level cast after extraction

      // --- native_static only ---
      "base_var": "dep_var",               // the existing dynamic variable to aggregate
                                           // (legacy alias: "variable"). NOT `requires`.
      "select": "!max value",              // aggregation function, see helpers.parse_select

      // --- requires: simple list form, or dict form with per-dependency time
      // overrides. Both apply to the types that take `requires`
      // (derived_dynamic, derived_static, complex, native_dynamic).
      // tmin/tmax accept:
      //   column anchor            e.g. "icu_admission" or ["icu_admission", "-1h"]
      //   "inherit"                the parent variable's own anchor (INHERIT)
      //   ["inherit", "-1h"]       that anchor shifted by a delta (deltas stack)
      //   "-1h"                    shorthand for the previous form — a bare delta
      //                            is always read relative to the parent
      //   null                     unbounded
      "requires": ["dep1", "dep2"],
      "requires": {"alias": {"template": "dep1", "tmin": "icu_admission", "tmax": "inherit"}},

      // --- derived_static only ---
      "expression": "max_sodium_icu > 145"  // SQL-like, evaluated with pl.sql_expr
    }
  }
}
```

A declarative definition is dispatched to the `VariableLoader` of **every** source
configured on the cohort, so a type only builds if a loader knows it.
`local_datasource.VariableLoader` is the one that knows the aggregation types
(`native_static`, `derived_static`, `derived_dynamic`, `complex`,
`native_dynamic`, `free_days`). `reprodicu`'s loader reads `type` only as a
restatement of `dynamic` and rejects `select` / `base_var` / `expression`, so on a
reprodicu-only cohort a declarative aggregation raises `TypeError` — construct the
class by hand instead (see **Hand-built variables**).

### `VariableCallable` / `py` functions (custom transformation functions)

```python
class VariableCallable(Protocol):
    def __call__(
        self, var: VariableContext, cohort: Cohort
    ) -> pl.DataFrame | pd.DataFrame: ...
```

A `py` function receives:
- `var` — a `VariableContext` (`definitions/typing.py`), a slim view over the source-specific `Variable`, not the instance itself.
    - `var.var_name`, `var.dynamic`, `var.time_window`
    - `var.data` — data already extracted by the extractor (for `BaseDynamic` subclasses in `local_datasource`) or the raw DB result; the `py()` function only needs to transform it.
    - `var.required_vars["alias"].data` — loaded dependency data.
    - `var.files["postcode/postcode_mapping.csv"]` — attached data files, populated by `PyFuncStep`; empty for local definitions. Prefer `getfile("<uuid>")` (see **Attached data files**).
- `cohort` — the full `Cohort` object
    - `cohort.obs`, `cohort.obsm`, `cohort.primary_key`

**`py_ready_polars`** (default `False`) controls which execution path `_call_var_function()` uses
- `True` → `py()` is called directly and must return `pl.DataFrame`
- `False` (legacy) → `var.data` and `cohort._obs` are converted to pandas before the call and back after, with a warning logged.
- Set via the variable config or a constructor kwarg
- All new functions must use `py_ready_polars=True`
- The `False` path is deliberately not type-checked for backwards compatibility

For an API-routed source a `py` function is authored and published as part of the concept, and arrives as a bare snippet named exactly like the variable. It must be self-contained against the `py_env` namespace: no sibling-variable calls, data files reached with `getfile("<uuid>")`, and any new import, shared helper or module constant added to `py_namespace.py` and declared in `py_env.NAMESPACE_NAMES` in the same change. For a pinned-local source the function goes in `mapping/variables.py` under a name exactly matching the `var_name` in `vars.json` — it is looked up by name at load time.

## Running tests

```shell
uv run pytest tests/integration/    # no DB, no network — what CI runs
```

CI (`.github/workflows/tests.yml`) runs `pytest -n auto tests/integration` on
every push to `main` and every pull request against it. It needs no credentials
and touches no network.

`tests/sources/` is hand-run: it needs the source's own data (for reprodicu, the
parquet files at `REPRODICU_PATH`) plus a Concepts API project and key.

## Building the documentation

```shell
cd docs && make clean && make html
# or: rm -rf _build && uv run sphinx-build -M html . _build
```

- Output: `docs/_build/html/index.html`
- Build must be clean (no ERRORs)
- Polars `Decimal` forward-reference warnings are suppressed via `suppress_warnings = ["ref.python"]` in `docs/conf.py` — ignore them

RST gotchas:
- Bullet lists need a blank line before the first `- item`
- trailing underscores in inline code must be escaped (`attr\_`)
- `:collapsible:` on `.. dropdown::` requires Sphinx ≥ 8.2 / Python ≥ 3.11
    - existing uses in `tutorials.rst` must not be removed
    - the sphinx pin is gated on Python version in `pyproject.toml` so uv can resolve
    - CI environment satisfies the requirement even if the local environment does not

Pages: `index.rst`, `tutorials.rst`, `contributing_variables.rst`,
`troubleshooting.rst`, plus the autogenerated `api/` tree.

## Docstring conventions

- **Google-style** docstrings (parsed by `napoleon`).
- RST code blocks: `.. code-block:: python` — not Markdown fences.
- Bullet lists inside docstrings need a blank line before the first item.

## Polars conventions

**Syntactic sugar** — unpack arguments instead of passing a list for `select`, `with_columns`, `group_by`, `sort`, `drop`, `rename`, etc. (`pl.concat` always requires a list):

```python
df.select("col_a", "col_b")    # prefer
df.select(["col_a", "col_b"])  # avoid
```

**Operators** — use named methods; boolean combinators `&`, `|`, `~` are the exception:

```python
pl.col("a").add(pl.col("b"))                # not +
pl.col("a").ge(0)                           # not >=
pl.col("a").sub(1)                          # not -
pl.col("a").eq(0)                           # not ==
pl.col("a").truediv(pl.col("b"))            # not /
pl.col("a").ne(0)                           # not !=
# floordiv, mul, gt, lt, le follow the same pattern

pl.col("a").gt(0) & pl.col("b").lt(10)      # & | ~ are fine
```

Prefer `df.remove(condition)` over `df.filter(~condition)`.

**Window ordering** — use `order_by` on `.over()` instead of pre-sorting:

```python
pl.col("value").cum_sum().over(partition_by="icu_stay_id", order_by="recordtime")
```

**Selectors** — use `import polars.selectors as cs` for dtype-based selection:

```python
df.select(cs.numeric())
df.select(cs.boolean() | cs.categorical() | cs.string())
df.select(~cs.temporal())
```

**Generic DataFrame/LazyFrame functions** — use the `PolarsFrame` TypeVar from `definitions/typing.py` with `@overload` to preserve the concrete return type.

**Pipe helpers** — reusable logic belongs in `utils/frames.py`, not inlined in variable functions. Two shapes:

```python
def my_transform(expr: pl.Expr, ...) -> pl.Expr: ...            # expression-level
def my_transform(df: pl.DataFrame, ...) -> pl.DataFrame: ...    # frame-level
```

Dependency-free primitives live in `utils/base.py` (`as_expr`, `columns`, `struct_fields`,
`col_length`, `row_length`, `is_empty`, `column_selector`, `normalize_offset`). Richer,
domain-specific helpers live in `utils/frames.py`:

| Helper | Module | What it does |
|---|---|---|
| `as_expr(value)` | `base` | Coerces `str` or `pl.Expr` to `pl.Expr` |
| `normalize_offset(offset)` | `base` | Coerces an `OffsetLike` (str or `timedelta`) to a polars-duration string |
| `time_difference(time_col, reference_col, *, unit, total)` | `frames` | Signed time delta as `pl.Expr` |
| `unique_sucessive(value)` | `frames` | Drops consecutive duplicates from a list column |
| `remove_asof(main, ref, strategy, tolerance)` | `frames` | Drops every *main* row within *tolerance* of any *ref* row (idempotent) |
| `apply_cleaning(df, cleaning)` | `frames` | Clips/nullifies values outside `cleaning` bounds |
| `interval_bucket_agg(...)` | `frames` | Aggregates time-series into fixed-width buckets |

Variable `py` functions should delegate reusable logic to these pipe helpers, keeping the custom code surface (which is untested by design) as small as possible.

## Python conventions

**Type annotations** — all parameters, keyword arguments, return types, and class attributes must be annotated. Use `from __future__ import annotations` at the top of every module.

**Import ordering** — six groups, each separated by a blank line:

```python
from __future__ import annotations          # 1. future

import logging                              # 2. standard library (alphabetical)
import os

import polars as pl                         # 3. external libraries (alphabetical)
import polars.selectors as cs

from corr_vars import logger, __            # 4. corr_vars imports
import corr_vars.utils as utils

from collections.abc import Callable        # 5. typing imports
from corr_vars.definitions.typing import VariableCallable
from polars._typing import AsofJoinStrategy
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:                           # 6. TYPE_CHECKING-only (circular / heavy imports)
    from corr_vars.core.cohort import Cohort
    from corr_vars.definitions import VariableProtocol
```

Within each group, imports are sorted alphabetically. `TYPE_CHECKING` imports are never executed at runtime; type checkers see them as always present.

**Keyword-only arguments** — separate configuration parameters from required positional ones with `*`:

```python
def add_relative_times(
    reference_df: pl.DataFrame,
    var_df: pl.DataFrame,
    *,
    reference_col: str,
    join_col: str,
    suffix: str = "_relative",
    include_time_cols: Collection[str] | None = None,
) -> pl.DataFrame: ...
```

**Logging** — use the `__` helper (`FStringCallableLogMessage`, imported from `corr_vars`) instead of f-strings. Wrap expensive formatting in a zero-argument lambda so it is only evaluated when the log level is active:

```python
logger.info(__("Extracted {var_name} from {src}", var_name=self.var_name, src=src))
```

For structured or multi-line output use the utilities in `utils/logging.py`:

| Utility | Use for |
|---|---|
| `log_collection(logger, collection, level)` | List/set as a tree (`├──` / `└──`) or indented block |
| `log_multiline_string(logger, multiline, level)` | Multi-line string logged line by line |
| `log_dict(logger, dictionary, level)` | Dict as pretty-printed JSON |
| `pretty_join(items, sep, width, sort)` | Iterable → single wrapped string (use as lambda in `__()`) |

Any pattern more complex than a single `__()` call should be extracted to `utils/logging.py`.



## What NOT to do

- Do not run `tests/sources/reprodicu/` — it needs the reprodICU parquet files and a live Concepts API key.
- Do not reintroduce a bundled `vars.json` / `variables.py` for an API-routed source, or a fallback to one, and do not soften a Concepts API failure into a warning — every transport, auth and server failure on a path that produces data is deliberately fatal. (The search widget's name listing is the sole exception, and it produces none.)
- Do not add an import, shared helper, or module constant to `py_namespace.py` without declaring it in `py_env.NAMESPACE_NAMES`.
- Do not have one served `py` snippet call another variable's function — it is not in the namespace and raises `NameError`. Declare the dependency via `requires`, or duplicate the code.
- Do not call `getfile` from a shared helper — it is not in the module's globals. Have the variable function resolve the path and pass it in.
- Do not add a user-scoped override for `concepts/routing.toml` (config file, env var) — the only override mechanism is `Cohort(...)` kwargs.
- Do not pass both `version` and `date` to `Cohort` — they are mutually exclusive.
- Do not leave a `::draftN` pin in an analysis meant to be reproducible — use `::vN` or a cohort-wide `date=`.
- Do not add `obs_level` values not in `ObsLevelName` — valid set: `"patient"`, `"hospital_stay"`, `"icu_stay"`, `"procedure"`.
- Do not use `cohort.changetracker` — use `cohort._change_tracker`; public flowchart API is `cohort.figureone` / `cohort.to_figureone()`.
- Do not use `.corr2` as the save format in new code — use `.corr3` (`.corr2` loads legacy files only).
- Do not use pandas for new code — all DataFrame operations must use polars. All new `py` functions must be polars-native; set `"py_ready_polars": true` in the variable config (or pass `py_ready_polars=True` to the constructor).
- Do not set the `"py"` key by hand — it is served with the config, or resolved from `mapping/variables.py` for a pinned-local source.
- Do not add extra columns outside `COL_ORDER` to dynamic variable output — move additional data into the `attributes` column instead.
