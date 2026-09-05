from __future__ import annotations

import copy
import logging
import warnings

import polars as pl
import polars.selectors as cs

import corr_vars.sources as corr
from corr_vars import __, logger, utils
from corr_vars.concepts.spec import contributor_source
from corr_vars.definitions.constants import COL_ORDER, DYN_COLUMNS, PRIMARY_KEYS
from corr_vars.definitions.exceptions import (
    ConceptsApiError,
    ObsLevelNotSupportedError,
    VariableNotFoundError,
)
from corr_vars.utils.time import TimeAnchor

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from corr_vars.concepts.resolver import ConceptResolver, ResolvedConcept
    from corr_vars.concepts.spec import VariableSpec
    from corr_vars.core.cohort import Cohort, ObsLevel
    from corr_vars.utils.time import TimeWindow

    from collections.abc import Collection, Iterable, Mapping, MutableMapping
    from corr_vars.definitions.typing import (
        ObsLevelName,
        VariableLoaderProtocol,
        VariableProtocol,
    )

# ---------------------------------------------------------------------------
# MultiSourceVariable with pre and post-processors
# ---------------------------------------------------------------------------


def validate_primary_key(var_df: pl.DataFrame, primary_key: str) -> None:
    """Validate that the primary key is known and present in the variable DataFrame.

    Args:
        var_df (pl.DataFrame): The variable data to validate.
        primary_key (str): The primary key column name.

    Raises:
        ValueError: If the primary key is not in `PRIMARY_KEYS` or absent from `var_df`.
    """
    if primary_key not in PRIMARY_KEYS:
        raise ValueError(f"Primary key {primary_key} must be one of {PRIMARY_KEYS}.")

    if primary_key not in var_df.columns:
        raise ValueError(f"Primary key {primary_key} not in data columns.")


def add_relative_times(
    reference_df: pl.DataFrame,
    var_df: pl.DataFrame,
    *,
    reference_col: str,
    join_col: str,
    suffix: str = "_relative",
    include_time_cols: Collection[str] | None = None,
) -> pl.DataFrame:
    """Add relative time columns to dynamic variable data.

    For each temporal column in `var_df` (optionally restricted to
    `include_time_cols`), computes the signed difference from `reference_col`
    in seconds and appends the result as a new column with `suffix` appended to
    the original column name. The reference column is dropped from the result.

    Args:
        reference_df (pl.DataFrame): DataFrame containing the reference time column.
        var_df (pl.DataFrame): DataFrame containing the variable data.
        reference_col (str): Column in `reference_df` holding the reference timestamp.
        join_col (str): Column used to join `reference_df` onto `var_df`.
        suffix (str): Suffix appended to each relative-time column name. Defaults to
            `"_relative"`.
        include_time_cols (Collection[str] | None): Restrict relative-time computation
            to these temporal columns. All temporal columns are used when `None`.

    Returns:
        pl.DataFrame: `var_df` with relative time columns added and `reference_col` dropped.
    """
    var_df = var_df.join(
        reference_df.select(join_col, reference_col),
        on=join_col,
        how="left",
    )

    selector = cs.temporal()
    if include_time_cols is not None:
        selector &= cs.by_name(include_time_cols, require_all=False)

    var_df = var_df.with_columns(
        selector.pipe(
            utils.time_difference,
            reference_col=reference_col,
            unit="s",
            total=True,
        ).name.suffix(suffix)
    )

    # Drop the reference time column
    return var_df.drop(reference_col)


def unify_and_order_columns(var_df: pl.DataFrame) -> pl.DataFrame:
    """Enforce `COL_ORDER` and fill missing dynamic columns with null.

    Columns outside `COL_ORDER` that are not in `DYN_COLUMNS` trigger a
    deprecation warning and are appended after the ordered columns until the
    `attributes` migration is complete.

    Args:
        var_df (pl.DataFrame): The variable DataFrame to standardize.

    Returns:
        pl.DataFrame: `var_df` with columns reordered and missing dynamic columns added as null.
    """
    # Add missing dynamic columns (no keys) with null values
    missing_cols = [col for col in DYN_COLUMNS if col not in var_df.columns]
    if missing_cols:
        var_df = var_df.with_columns(pl.lit(None).alias(col) for col in missing_cols)

    # Order columns according to COL_ORDER, then add remaining columns
    data_cols = set(var_df.columns)
    ordered_cols = [col for col in COL_ORDER if col in data_cols]
    remaining_cols = [col for col in data_cols if col not in COL_ORDER]
    if len(remaining_cols) > 0:
        warnings.warn(
            "🚨 DEPRECATION ALERT 🚨\n"
            "Unknown columns found.\n"
            "- Please update the variable definition to move these to attributes.\n"
            "- Unknown columns are no longer supported and might be deprecated soon!\n"
            "- Columns: " + ", ".join(remaining_cols),
            UserWarning,
        )

    var_df = var_df.select(ordered_cols + remaining_cols)

    # Uncomment after user warning
    # var_df = var_df.select(ordered_cols)
    return var_df


class MultiSourceVariable:
    """Holds one :class:`~corr_vars.definitions.typing.VariableProtocol` per contributor
    and extracts, post-validates, and post-processes each before concatenating the results.

    A contributor is normally one data source. A variable whose name resolves to
    a group of concepts has one contributor per source and group member, keyed by
    :func:`~corr_vars.concepts.spec.contributor_key`; every contributor produces
    the same columns and the frames are concatenated the same way.

    Args:
        var_name (str): Shared variable name across all contributors.
        variables (Mapping[str, VariableProtocol]): Mapping from contributor key
            to the source-specific variable implementation.
    """

    variables: Mapping[str, VariableProtocol]

    data: pl.DataFrame | None

    _var_name: str
    _dynamic: bool
    _time_window: TimeWindow

    def __init__(self, variables: Mapping[str, VariableProtocol]) -> None:
        self.variables = variables
        self.data = None
        self._pre_validate()

    def _pre_validate(self) -> None:
        if not self.variables:
            raise ValueError("No variables to concatenate")

        # Assign dynamic, tmin and tmax after veryfing variables is not empty
        first_variable = next(iter(self.variables.values()))
        self._var_name = first_variable.var_name
        self._dynamic = first_variable.dynamic
        self._time_window = first_variable.time_window

        for key, var in self.variables.items():
            assert (
                var.var_name == self._var_name
            ), f"Variable name mismatch for {key}: {var.var_name} != {self._var_name}"
            assert (
                var.dynamic == self._dynamic
            ), f"Dynamic status mismatch for {self._var_name} ({key}): {var.dynamic} != {self._dynamic}"
            assert (
                var.time_window == self._time_window
            ), f"Timebound mismatch for {self._var_name} ({key}): {var.time_window} != {self._time_window}"

    def extract(self, cohort: Cohort, with_source_column: bool = False) -> pl.DataFrame:
        var_data: list[pl.DataFrame] = []

        for key, var in self.variables.items():
            src = contributor_source(key)

            # Extract and validate data
            extracted = var.extract(cohort)
            self._post_validate(cohort, extracted)

            # Transform Data
            extracted = self._post_process(cohort, extracted)
            if with_source_column:
                extracted = extracted.insert_column(0, pl.lit(src).alias("data_source"))

            # Add to list
            var_data.append(extracted)
            logger.info(
                __(
                    "SUCCESS: Extracted {var_name} from {key}",
                    var_name=self._var_name,
                    key=key,
                )
            )

        # Concatenate data
        combined_data = pl.concat(var_data, how="diagonal_relaxed")
        self.data = combined_data
        return combined_data

    def _post_process(self, cohort: Cohort, var_data: pl.DataFrame) -> pl.DataFrame:
        if not self.dynamic:
            return var_data

        processed = var_data
        include_time_cols = ("recordtime", "recordtime_end")
        suffix = "_relative"
        if not frozenset(f"{col}{suffix}" for col in include_time_cols).issubset(
            frozenset(var_data.columns)
        ):
            processed = add_relative_times(
                cohort.obs,
                processed,
                reference_col=cohort.t_min,
                join_col=cohort.primary_key,
                suffix=suffix,
                include_time_cols=include_time_cols,
            )

        return unify_and_order_columns(processed)

    def _post_validate(self, cohort: Cohort, var_data: pl.DataFrame) -> None:
        validate_primary_key(var_data, cohort.primary_key)

    @property
    def var_name(self) -> str:
        return self._var_name

    @var_name.setter
    def var_name(self, _: str) -> None:
        raise AttributeError("Cannot set var_name")

    @property
    def dynamic(self) -> bool:
        return self._dynamic

    @dynamic.setter
    def dynamic(self, _: bool) -> bool:
        raise AttributeError("Cannot set dynamic")

    @property
    def time_window(self) -> TimeWindow:
        return self._time_window

    @time_window.setter
    def time_window(self, _: TimeWindow) -> None:
        raise ValueError("Cannot set time_window")

    @property
    def extracted(self) -> bool:
        return self.data is not None

    def __repr__(self) -> str:
        return (
            f"MultiSourceVariable(var_name='{self._var_name}', "
            f"dynamic={self._dynamic}, "
            f"time_window={self._time_window}, "
            f"contributors={list(self.variables.keys())}, "
            f"extracted={self.extracted})"
        )

    def __str__(self) -> str:
        return self.__repr__()


# ---------------------------------------------------------------------------
# load_raw_variable_configs -> dict[str, dict[str, object]]
# ---------------------------------------------------------------------------


#: Placeholder configuration used for a variable that is only known by name,
#: because its definition lives behind a Concepts API endpoint rather than in a
#: bundled ``vars.json``. The search widget renders whatever dict it is given,
#: so an empty one would show a name with a blank body; this says why.
API_ROUTED_PLACEHOLDER_CONFIG: dict[str, Any] = {"source": "concepts-api"}


def load_raw_variable_configs(
    include_sources: Iterable[str] | None = None,
    *,
    resolver: ConceptResolver | None = None,
) -> dict[str, dict[str, dict[str, object]]]:
    """Collect the searchable variable configs of every source.

    Two kinds of source are collected:

    - A source with a local ``VARS`` dict contributes its full ``vars.json``
      ``variables`` mapping, as before.
    - A source routed to a Concepts API endpoint has no local definitions at
      all. When a `resolver` is given, its variable *names* are listed from the
      endpoint so the search widget can still find them; each name maps to
      :data:`API_ROUTED_PLACEHOLDER_CONFIG` instead of a definition. Without a
      resolver, or when the endpoint cannot be reached, the source is skipped
      — the search widget is a convenience and must never raise.

    Args:
        include_sources (Iterable[str] | None): Restrict the lookup to these sources.
            All discovered sources are searched when `None`.
        resolver (ConceptResolver | None): The cohort's concept resolver, used to
            list the names of API-routed sources. When `None`, sources without a
            local ``VARS`` are skipped.

    Returns:
        dict[str, dict[str, dict[str, object]]]: Mapping from source name to its
        raw variable config dict. Sources that contribute nothing are absent.
    """
    var_dict: dict[str, dict[str, dict[str, Any]]] = {}

    for src, src_mapping_module in corr.get_src_module_mapping(
        include_sources=include_sources, submodule="mapping"
    ).items():
        src_VARS = getattr(src_mapping_module, "VARS", None)

        if src_VARS is not None:
            var_dict[src] = src_VARS["variables"]
            continue

        names = _list_remote_variable_names(src, resolver=resolver)
        if names:
            var_dict[src] = {
                name: dict(API_ROUTED_PLACEHOLDER_CONFIG) for name in names
            }

    return var_dict


def _list_remote_variable_names(
    src: str,
    *,
    resolver: ConceptResolver | None = None,
) -> list[str]:
    """List the variable names an API-routed source publishes.

    Args:
        src (str): Source name, known to have no local ``VARS``.
        resolver (ConceptResolver | None): The cohort's concept resolver. Without
            one there is nothing to ask, and the source yields no names.

    Returns:
        list[str]: Fully-qualified concept names, or an empty list when the
        source is not remote, no resolver was given, or the endpoint could not
        be reached.
    """
    if resolver is None:
        logger.debug(
            __(
                "Source {src} ships no local variable definitions and no concept "
                "resolver was given, so its variables are not listed.",
                src=src,
            )
        )
        return []

    if not resolver.is_remote(src):
        logger.debug(
            __(
                "Source {src} ships no local variable definitions and is not routed "
                "to a Concepts API endpoint, so its variables are not listed.",
                src=src,
            )
        )
        return []

    try:
        client = resolver.client_for(resolver.route(src))  # type: ignore[arg-type]
        return client.list_concept_names(resolver.default_taxonomy, source=src)
    except ConceptsApiError as exc:
        logger.warning(
            __(
                "Could not list the variable names of source {src} from the Concepts "
                "API ({error}); it is left out of the search widget.",
                src=src,
                error=exc,
            )
        )
        return []


# ---------------------------------------------------------------------------
# _load_variable_configs -> dict[str, dict[str, object]]
# ---------------------------------------------------------------------------


def _collect_local_variable_config(
    var_name: str,
    include_sources: Iterable[str],
) -> dict[str, dict[str, Any]]:
    """Read the `vars.json` entry for `var_name` from the sources' `mapping` modules.

    Also resolves the `py` callable from `mapping/variables.py` and injects it
    into the returned config dict under the `"py"` key when a matching function exists.

    A source routed to a Concepts API endpoint ships no local definitions at all
    — its `mapping` module has no `VARS` — and is skipped here by design; its
    configuration comes from :meth:`~corr_vars.concepts.resolver.ConceptResolver.fetch_configs`.

    Args:
        var_name (str): Name of the variable to look up.
        include_sources (Iterable[str]): Sources to read from.

    Returns:
        dict[str, dict[str, Any]]: Mapping from source name to its variable config
        dict (with `"py"` injected where applicable). Sources whose `vars.json`
        has no entry for `var_name`, and sources with no local definitions at
        all, are absent.
    """
    var_dict: dict[str, dict[str, Any]] = {}

    for src, src_mapping_module in corr.get_src_module_mapping(
        include_sources=include_sources, submodule="mapping"
    ).items():
        src_VARS = getattr(src_mapping_module, "VARS", None)

        if src_VARS is None:
            continue

        src_VARS_variables = src_VARS["variables"]
        src_VARCODE = getattr(src_mapping_module, "variables", None)

        if var_name not in src_VARS_variables:
            continue

        src_vardict = src_VARS_variables[var_name].copy()

        if src_VARCODE is not None and hasattr(src_VARCODE, var_name):
            src_vardict["py"] = getattr(src_VARCODE, var_name)

        var_dict[src] = src_vardict

    return var_dict


def _collect_variable_configs_by_source(
    var_name: str,
    include_sources: Iterable[str] | None = None,
    *,
    resolver: ConceptResolver | None = None,
    spec: VariableSpec | None = None,
    resolutions: MutableMapping[str, ResolvedConcept] | None = None,
) -> dict[str, dict[str, object]]:
    """Collect the variable configuration for `var_name` from every source.

    This is the routing seam. Each source is resolved independently:

    - Sources routed to a Concepts API endpoint get their definition from that
      endpoint. The declarative config, the compiled ``py`` function, and the
      variable's attached data files all come from there, and there is **no**
      fallback to the bundled ``vars.json`` — a cohort assembled from a stale
      local definition while recording a remote version number would misdescribe
      its own data.
    - Sources routed to local (pinned or unmatched, and every source when no
      `resolver` is given) behave exactly as before: the entry is read from the
      source's ``mapping`` module and the matching ``variables.py`` function is
      injected as ``py``.

    Keys are contributor keys (see
    :func:`~corr_vars.concepts.spec.contributor_key`): a plain source name, or
    ``source#concept_id`` for one member of a group the name resolved to.

    Args:
        var_name (str): Name of the variable to look up.
        include_sources (Iterable[str] | None): Restrict the lookup to these sources.
            All discovered sources are searched when `None`.
        resolver (ConceptResolver | None): The cohort's concept resolver. When
            `None`, every source is treated as local.
        spec (VariableSpec | None): The resolved variable reference, deciding
            which taxonomy and version the API serves. Defaults to the resolver's
            cohort-wide default for `var_name`.
        resolutions (MutableMapping[str, ResolvedConcept] | None): Optional
            mapping that receives the provenance record for each contributor
            that supplied a configuration.

    Returns:
        dict[str, dict[str, object]]: Mapping from contributor key to its
        variable config dict (with `"py"` resolved where applicable).

    Raises:
        AmbiguousConceptError: If a version-pinned `spec` names a group.
        ConceptsApiError: If an API-routed source could not be served.
    """
    sources = list(
        corr.SOURCES if include_sources is None else include_sources  # type: ignore[arg-type]
    )

    if resolver is None:
        var_dict = _collect_local_variable_config(var_name, sources)
        _record_local_resolutions(var_name, var_dict, resolutions=resolutions)
        return var_dict  # type: ignore[return-value]

    local_sources = [src for src in sources if not resolver.is_remote(src)]
    remote_sources = [src for src in sources if resolver.is_remote(src)]

    var_dict: dict[str, dict[str, Any]] = _collect_local_variable_config(
        var_name, local_sources
    )
    _record_local_resolutions(var_name, var_dict, resolutions=resolutions)

    if remote_sources:
        resolved_spec = spec if spec is not None else resolver.default_spec(var_name)
        for key, fetched in resolver.fetch_configs(
            resolved_spec, remote_sources
        ).items():
            var_dict[key] = fetched.config
            if resolutions is not None:
                resolutions[key] = fetched.resolution

    return var_dict  # type: ignore[return-value]


def _record_local_resolutions(
    var_name: str,
    var_dict: Mapping[str, dict[str, Any]],
    *,
    resolutions: MutableMapping[str, ResolvedConcept] | None,
) -> None:
    """Record a provenance entry for every locally resolved source.

    Args:
        var_name (str): Name of the variable.
        var_dict (Mapping[str, dict[str, Any]]): Configs collected locally.
        resolutions (MutableMapping[str, ResolvedConcept] | None): Mapping that
            receives the records; nothing happens when `None`.
    """
    if resolutions is None:
        return

    from corr_vars.concepts.resolver import ResolvedConcept

    for src in var_dict:
        resolutions[src] = ResolvedConcept(
            taxonomy="local",
            name=var_name,
            source=src,
            origin="local",
            requested="local",
        )


def _expand_override_to_contributors(
    override: Mapping[str, dict[str, Any]],
    variable_configs_by_source: Mapping[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Spread a per-source override across that source's contributors.

    Overrides are declared per source. When a grouped name gave a source several
    contributors, the override applies to each of them, so a project-local tweak
    reaches every member of the group. An override for a source that supplied no
    configuration is kept under the bare source name, which is what lets a purely
    project-local variable definition introduce a source of its own.

    Args:
        override (Mapping[str, dict[str, Any]]): Overrides keyed by source name.
        variable_configs_by_source (Mapping[str, dict[str, Any]]): Configs keyed
            by contributor key.

    Returns:
        dict[str, dict[str, Any]]: The overrides keyed by contributor key.
    """
    contributors = {contributor_source(key) for key in variable_configs_by_source}
    expanded = {
        key: override[contributor_source(key)]
        for key in variable_configs_by_source
        if contributor_source(key) in override
    }
    expanded.update(
        {src: value for src, value in override.items() if src not in contributors}
    )
    return expanded


def _load_variable_configs(
    var_name: str,
    cohort: Cohort,
    include_sources: Iterable[str] | None = None,
    spec: VariableSpec | None = None,
) -> dict[str, dict[str, object]]:
    """Load and merge variable configurations for all requested sources.

    Retrieves the config for `var_name` via
    :func:`_collect_variable_configs_by_source` — per source, from the Concepts
    API or from the bundled ``mapping`` module — then overlays any project-local
    overrides from :meth:`~corr_vars.core.cohort.Cohort.get_variable_definition`.
    Project-local overrides layer on top of API-fetched configs exactly as they
    do on top of local ones.

    The resolved ``(taxonomy, name, version)`` of each contributing source is
    recorded on the cohort, so :meth:`~corr_vars.core.cohort.Cohort.save` can
    persist what the cohort was built from.

    Args:
        var_name (str): Name of the variable to load configuration for.
        cohort (Cohort): Cohort whose resolver and project overrides are applied.
        include_sources (Iterable[str] | None): Sources to consider.
            All sources are used when `None`.
        spec (VariableSpec | None): The resolved variable reference. Defaults to
            the cohort-wide default for `var_name`.

    Returns:
        dict[str, dict[str, object]]: Merged per-source variable configuration mapping.
    """

    resolutions: dict[str, ResolvedConcept] = {}

    # Get variable configuration for each source defined in cohort
    variable_configs_by_source = _collect_variable_configs_by_source(
        var_name=var_name,
        include_sources=include_sources,
        resolver=getattr(cohort, "concepts", None),
        spec=spec,
        resolutions=resolutions,
    )

    if resolutions:
        cohort._record_concept_versions(var_name, resolutions)

    # Apply project specific overrides
    override = _expand_override_to_contributors(
        cohort.get_variable_definition(var_name), variable_configs_by_source
    )
    return utils.deep_merge(base=variable_configs_by_source, override=override)


# ---------------------------------------------------------------------------
# load_variable -> MultiSourceVariable
# ---------------------------------------------------------------------------


def _transfer_time_window_from_variable_config(
    variable_configs_by_source: dict[str, dict[str, Any]],
    time_window: TimeWindow,
) -> tuple[dict[str, dict[str, object]], TimeWindow]:
    """Extract `tmin`/`tmax` overrides from source configs into the time window.

    If any source config contains a `"tmin"` or `"tmax"` key, those values are
    popped from the config and written to a copy of `time_window`, overriding whatever
    was passed in. A warning is logged for each override.

    Args:
        variable_configs_by_source (dict[str, dict[str, Any]]): Per-source config dicts.
        time_window (TimeWindow): The original time window.

    Returns:
        tuple[dict[str, dict[str, object]], TimeWindow]: Deep copies of the config dict
        and time window, updated with any embedded overrides.
    """
    time_window_cp = copy.copy(time_window)
    variable_configs_by_source_cp = copy.deepcopy(variable_configs_by_source)
    for _, var_config in variable_configs_by_source_cp.items():
        if "tmin" in var_config:
            logger.warning(
                __(
                    "For this variable tmin is already frozen to {tmin}. Any other tmin will be ignored.",
                    tmin=var_config["tmin"],
                )
            )
            tmin = var_config.pop("tmin")
            time_window_cp.tmin = TimeAnchor(tmin)

        if "tmax" in var_config:
            logger.warning(
                __(
                    "For this variable tmax is already frozen to {tmax}. Any other tmax will be ignored.",
                    tmax=var_config["tmax"],
                )
            )
            tmax = var_config.pop("tmax")
            time_window_cp.tmax = TimeAnchor(tmax)

    return variable_configs_by_source_cp, time_window_cp


def _ensure_compatibility(
    var_name: str,
    variable_configs_by_source: dict[str, dict[str, Any]],
    obs_level: ObsLevel,
) -> None:
    """Raise if `var_name` is incompatible with the cohort's observation level.

    Pops `"compatible_with"` from each source config dict (mutates in-place) and
    raises :exc:`~corr_vars.definitions.exceptions.ObsLevelNotSupportedError` if the
    cohort's obs level is not listed.

    Args:
        var_name (str): Variable name (used in the error message).
        variable_configs_by_source (dict[str, dict[str, Any]]): Per-source config dicts.
        obs_level (ObsLevel): The cohort's current observation level.

    Raises:
        ObsLevelNotSupportedError: If `obs_level` is not in `compatible_with`.
    """
    for _, var_config in variable_configs_by_source.items():
        if "compatible_with" in var_config:
            compatible_obs_level: list[ObsLevelName] = var_config.pop("compatible_with")
            if obs_level.lower_name not in compatible_obs_level:
                raise ObsLevelNotSupportedError(
                    f"Obs level {obs_level.lower_name} is not compatible with {var_name}"
                )


def _log_variable_loader_args_kwargs(
    var_name: str,
    time_window: TimeWindow,
    var_config: dict[str, object],
) -> None:
    """Log the arguments that will be passed to a source's `VariableLoader`.

    Args:
        var_name (str): Variable name being created.
        time_window (TimeWindow): Time window used for extraction.
        var_config (dict[str, object]): Keyword arguments forwarded to the loader.
    """
    logger.info(__("Creating variable {var_name} with:", var_name=var_name))
    prefix = "└──" if not var_config else "├──"
    logger.info(
        __(
            "{prefix} time_window: {time_window}",
            prefix=prefix,
            time_window=time_window,
        )
    )
    if var_config:
        logger.info("└── kwargs:")
        utils.log_dict(logger=logger, dictionary=var_config, indent=4, json_indent=2)


def _load_variables_from_config(
    var_name: str,
    variable_configs_by_source: dict[str, dict[str, Any]],
    cohort: Cohort,
    time_window: TimeWindow,
) -> dict[str, VariableProtocol]:
    """Instantiate a variable for each entry in `variable_configs_by_source`.

    For each contributor, retrieves its source's `VariableLoader` factory, calls it
    with `var_name`, `time_window`, and the contributor's config dict as keyword
    arguments, and stores the resulting
    :class:`~corr_vars.definitions.typing.VariableProtocol` in the output mapping.
    Two contributors that share a source — the members of a concept group — are
    instantiated separately from the same factory.

    Args:
        var_name (str): Name of the variable to instantiate.
        variable_configs_by_source (dict[str, dict[str, Any]]): Config dicts keyed
            by contributor key, whose keys determine which contributors are visited.
        cohort (Cohort): The cohort (used to skip sources not in `cohort.sources`).
        time_window (TimeWindow): Time window forwarded to every `VariableLoader`.

    Returns:
        dict[str, VariableProtocol]: Mapping from contributor key to the
        instantiated variable.
    """
    src_var_mapping: dict[str, VariableProtocol] = {}

    src_modules = corr.get_src_module_mapping(
        include_sources=dict.fromkeys(
            contributor_source(key) for key in variable_configs_by_source
        )
    )

    # Load from other sources
    for key, var_config in variable_configs_by_source.items():
        src = contributor_source(key)
        if src not in cohort.sources:
            logger.info(
                __("Source {src} not requested in cohort. Skipping...", src=src)
            )

        # Load kwargs
        _log_variable_loader_args_kwargs(
            var_name=var_name, time_window=time_window, var_config=var_config
        )

        # Get VariableLoaderProtocol (adapter) to load VariableProtocol
        SrcVariableLoader: VariableLoaderProtocol = getattr(
            src_modules[src], "VariableLoader"
        )
        SrcVariable: VariableProtocol = SrcVariableLoader.__call__(
            var_name=var_name, time_window=time_window, **var_config
        )
        # Contributors sharing a source share var_name, so anything a source
        # files per variable — the extraction cache above all — needs the
        # contributor key to keep them apart.
        SrcVariable.contributor_key = key  # type: ignore[attr-defined]
        src_var_mapping.setdefault(key, SrcVariable)

        logger.info(
            __(
                "Variable {var_name} found in source {src} ({key})",
                var_name=var_name,
                src=src,
                key=key,
            )
        )

    return src_var_mapping


def load_variable(
    var_name: str,
    cohort: Cohort,
    time_window: TimeWindow,
    include_sources: Iterable[str] | None = None,
    overrides: dict[str, Any] | None = None,
    spec: VariableSpec | None = None,
) -> MultiSourceVariable:
    """Load `var_name` from all configured sources and wrap it in a :class:`MultiSourceVariable`.

    Orchestrates config loading, project-override merging, time-window transfer,
    compatibility checks, and per-source variable instantiation.

    Args:
        var_name (str): Name of the variable to load.
        cohort (Cohort): Cohort that supplies source configs and project overrides.
        time_window (TimeWindow): Time window for variable extraction.
        include_sources (Iterable[str] | None): Restrict to these sources.
            All cohort sources are used when `None`.
        overrides (dict[str, Any] | None): Config fields merged onto every source's
            `vars.json` entry, for this load only. Applied after the cohort's
            project-local overrides, so these win.
        spec (VariableSpec | None): Resolved variable reference deciding which
            taxonomy and version API-routed sources serve. Defaults to the
            cohort-wide default.

    Returns:
        MultiSourceVariable: Variable ready for extraction.

    Raises:
        VariableNotFoundError: If `var_name` is not found in any source config, or if
            no source variable could be instantiated.
        AmbiguousConceptError: If `spec` pins a version and the name resolves to a
            group of concepts.
        ConceptsApiError: If an API-routed source could not be served.
    """
    # Get lookup sources
    sources = include_sources or cohort.sources.keys()

    # Get variable configuration for each source defined in cohort
    variable_configs_by_source = _load_variable_configs(
        var_name=var_name, cohort=cohort, include_sources=sources, spec=spec
    )

    # Raise error if no source was found
    if not variable_configs_by_source:
        raise VariableNotFoundError(
            _variable_not_found_message(var_name, cohort, sources, spec)
        )

    # Per-load overrides win over the vars.json entry and the project-local overrides.
    if overrides:
        logger.debug(
            __(
                "Applying per-load overrides to {var_name}:",
                var_name=var_name,
            )
        )
        utils.log_dict(logger, overrides, level=logging.DEBUG)
        variable_configs_by_source = utils.deep_merge(
            base=variable_configs_by_source,
            override=dict.fromkeys(variable_configs_by_source, overrides),
        )

    # Debug logging
    logger.debug(__("Loaded config for {var_name}:", var_name=var_name))
    utils.log_dict(
        logger, variable_configs_by_source, level=logging.DEBUG, json_indent=2
    )

    # Transfers "tmin", "tmax" from dict to time_window (pops from config; overrides time_window)
    (
        variable_configs_by_source,
        time_window,
    ) = _transfer_time_window_from_variable_config(
        variable_configs_by_source, time_window
    )

    # STOP if not compatible with obs level
    _ensure_compatibility(var_name, variable_configs_by_source, cohort.obs_level)

    # Get variable for each variable configuration by source
    variable_by_source = _load_variables_from_config(
        var_name=var_name,
        variable_configs_by_source=variable_configs_by_source,
        cohort=cohort,
        time_window=time_window,
    )

    # Raise error if no variable was found
    if not variable_by_source:
        raise VariableNotFoundError(
            f"Variable {var_name} not found in any source (looked in {list(sources)})"
        )

    return MultiSourceVariable(variable_by_source)


def _variable_not_found_message(
    var_name: str,
    cohort: Cohort,
    sources: Iterable[str],
    spec: VariableSpec | None,
) -> str:
    """Build an actionable "variable not found" message.

    Names where each source was looked up, so a miss caused by an unpublished
    concept is not mistaken for a typo in the variable name.

    Args:
        var_name (str): The variable that could not be found.
        cohort (Cohort): Cohort whose resolver describes the routing.
        sources (Iterable[str]): Sources that were searched.
        spec (VariableSpec | None): The reference that was requested.

    Returns:
        str: A message listing the per-source lookup locations.
    """
    resolver = getattr(cohort, "concepts", None)
    reference = str(spec) if spec is not None else var_name

    if resolver is None:
        return (
            f"Variable {var_name} not found in the bundled vars.json of {list(sources)}"
        )

    locations: list[str] = []
    for src in sources:
        route = resolver.route(src)
        where = route.url if route.is_remote else "bundled vars.json"
        locations.append(f"{src} -> {where}")

    return (
        f"Variable {reference} not found. Looked in: {'; '.join(locations)}. "
        "Sources routed to a Concepts API endpoint are never served from the "
        "bundled vars.json, so a concept that has not been published there yet "
        "resolves to nothing."
    )


# ---------------------------------------------------------------------------
# load_default_variables -> list[MultiSourceVariable]
# ---------------------------------------------------------------------------


def _collect_default_variable_list_by_source(
    obs_level: ObsLevel,
    include_sources: Iterable[str] | None = None,
) -> dict[str, list[str]]:
    """Collect the default variable list each source declares.

    The defaults are corr_vars configuration, not clinical concepts, so they stay
    local even for a source whose definitions are served by the Concepts API:
    every source exposes them as a `DEFAULT_VARS` attribute on its `mapping`
    module, mapping `"global"` and each obs-level name to a list of variable
    names. Sources that declare none are skipped.

    Args:
        obs_level (ObsLevel): Observation level used to select level-specific defaults.
        include_sources (Iterable[str] | None): Restrict to these sources.
            All discovered sources are checked when `None`.

    Returns:
        dict[str, list[str]]: Mapping from source name to its list of default variable names.
    """
    src_dict: dict[str, list[str]] = {}

    for src, src_mapping_module in corr.get_src_module_mapping(
        include_sources=include_sources, submodule="mapping"
    ).items():
        default_vars: dict[str, list[str]] | None = getattr(
            src_mapping_module, "DEFAULT_VARS", None
        )

        if not default_vars:
            continue

        global_defaults = default_vars.get("global", [])
        obs_level_defaults = default_vars.get(obs_level.lower_name, [])
        all_defaults = global_defaults + obs_level_defaults
        src_dict.setdefault(src, all_defaults)
    return src_dict


def load_default_variables(
    cohort: Cohort, time_window: TimeWindow
) -> list[MultiSourceVariable]:
    """Load the default variables for the cohort's observation level.

    Aggregates the default variable lists from all cohort sources, then loads each
    variable restricted to the sources that declared it.

    Args:
        cohort (Cohort): Cohort whose observation level and sources determine which
            defaults are loaded.
        time_window (TimeWindow): Time window forwarded to every variable loader.

    Returns:
        list[MultiSourceVariable]: One :class:`MultiSourceVariable` per unique default
        variable name, ready for extraction.
    """
    # Get default variables per source
    source_default_variables_mapping = _collect_default_variable_list_by_source(
        obs_level=cohort.obs_level, include_sources=cohort.sources.keys()
    )

    # Invert mapping
    variable_source_mapping: dict[str, list[str]] = {}
    for src, default_variables in source_default_variables_mapping.items():
        for var_name in default_variables:
            variable_source_mapping.setdefault(var_name, [])
            variable_source_mapping[var_name].append(src)

    # Load variable for included sources only
    variables: list[MultiSourceVariable] = []
    for var_name, include_sources in variable_source_mapping.items():
        # TODO: Loading only included sources can be tricky,
        # if the variable is loaded in other sources later
        variable = load_variable(
            var_name=var_name,
            cohort=cohort,
            time_window=time_window,
            include_sources=include_sources,
        )
        variables.append(variable)

    return variables
