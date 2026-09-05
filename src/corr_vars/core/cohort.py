from __future__ import annotations

import gc
import json
import logging
import os
import pickle
import tarfile
import tempfile
import textwrap
import traceback
import warnings
from copy import deepcopy
from datetime import date as dt_date
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
from pickle import UnpicklingError

import polars as pl
import zstandard as zstd
from corr_vars_widget import JsonmWidget, JsonWidget, ObsmWidget, ObsWidget
from polars.exceptions import ColumnNotFoundError, PolarsError
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import corr_vars.sources as loader
from corr_vars import __, logger, utils
from corr_vars.core.change_tracker import ChangeTrackerContext, ChangeTrackerPipeline
from corr_vars.core.file_manager import TemporaryDirectoryManager
from corr_vars.definitions import (
    CohortDataError,
    ObsLevel,
    StataDateFormat,
    VariableNotFoundError,
)
from corr_vars.sources.var_loader import MultiSourceVariable
from corr_vars.utils.time import TimeWindow

from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
)
from typing import TYPE_CHECKING, Any, Final, Literal, overload

if TYPE_CHECKING:
    from itertools import chain

    import pandas as pd
    from graphviz import Digraph
    from tableone import TableOne

    from corr_vars.concepts.resolver import ConceptResolver, ResolvedConcept
    from corr_vars.concepts.spec import VariableSpec
    from corr_vars.definitions import (
        ArrayLike,
        CohortExportFormats,
        CohortSaverProtocol,
        ObsLevelName,
        SourceDict,
        SourceDictPartial,
        TimeAnchorColumn,
        VariableProtocol,
    )

    from polars._typing import (
        ColumnNameOrSelector,
        IntoExpr,
        MultiColSelector,
        MultiIndexSelector,
        SingleColSelector,
        SingleIndexSelector,
    )


def _as_concept_version_records(entry: Any) -> list[dict[str, Any]]:
    """Return one variable's provenance records as a list.

    Accepts both the list form and the source-keyed mapping form an archive may
    carry, so a cohort saved by any release loads into the same shape.

    Args:
        entry (Any): The registry value stored for one variable.

    Returns:
        list[dict[str, Any]]: One record per contributor.
    """
    if isinstance(entry, list):
        return [dict(record) for record in entry]
    if isinstance(entry, Mapping):
        return [dict(record) for record in entry.values()]
    return []


class ObsmDict(MutableMapping[str, pl.DataFrame]):
    """Dictionary-like wrapper for obsm data providing convenient access and display.

    Args:
        data (dict): Dictionary of polars DataFrames containing dynamic variable data.
    """

    def __init__(self, data: dict[str, pl.DataFrame]) -> None:
        self._data = data

    def __getitem__(self, key: str) -> pl.DataFrame:
        try:
            return self._data[key]
        except KeyError:
            available_vars = list(self._data.keys())
            utils.raise_error_with_nearest_matches(
                item=key,
                available_items=available_vars,
                label="Variable",
                error_cls=VariableNotFoundError,
            )
            error_msg = f"Variable '{key}' not found in cohort.obsm."
            if not available_vars:
                error_msg += " No variables have been extracted yet. Use cohort.add_variable() to extract variables."
            raise VariableNotFoundError(error_msg) from None

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return key in self._data

    def __setitem__(self, key: str, value: pl.DataFrame) -> None:
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def to_widget(
        self,
        *keys: str | Iterable[str],
        strict: bool = True,
        unnest_attributes: bool = True,
    ) -> ObsmWidget:
        def transform(value: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
            return {
                k: (
                    v.pipe(utils.unnest, column="attributes", prefix="attributes: ")
                    if unnest_attributes
                    and "attributes" in v.columns
                    and v["attributes"].dtype == pl.Struct
                    else v
                )
                for k, v in value.items()
            }

        if not keys:
            return ObsmWidget(transform(self._data))

        variables: chain[str] = utils.flatten_args(*keys)
        filtered_data = {
            var: self._data[var] for var in variables if strict or var in self._data
        }
        return ObsmWidget(transform(filtered_data))

    @property
    def widget(self) -> ObsmWidget:
        return self.to_widget()

    def __repr__(self) -> str:
        return self.widget.__repr__()

    def __str__(self) -> str:
        return self.widget.__str__()

    def _repr_mimebundle_(self, *args, **kwargs) -> tuple[dict, dict] | None:
        return self.widget._repr_mimebundle_(*args, **kwargs)


class Cohort:
    """Class to build a cohort in the CORR database.

    Args:
        obs_level (Literal["patient", "hospital_stay", "icu_stay", "procedure"]):
            Observation level (default: "icu_stay").

            - ``"patient"`` gives one row per patient (primary key: ``patient_id``)
            - ``"hospital_stay"`` per hospitalisation (``case_id``)
            - ``"icu_stay"`` per ICU admission (``icu_stay_id``)
            - ``"procedure"`` per surgical procedure (``procedure_id``).

        sources (dict[str, dict]):
            Dictionary of data sources to use for data extraction. Available option is "reprodicu".
            Source configurations:

            - **reprodicu**: ``path``, ``exclude_datasets``, ``include_datasets``

            Note: reprodicu does not yet implement variable extraction, only cohort data.
        project_vars (dict):
            Dictionary with local variable definitions (default: {}).
        load_default_vars (bool):
            Whether to load the default variables (default: True).
        logger_args (dict):
            Dictionary of Logging configurations [level (int), file_path (str), file_mode (str), verbose_fmt (bool), colored_output (bool), formatted_numbers (bool)] (default: {}).
        project (str | None):
            Project name on the CORR Concepts API. **Required for every source
            routed to the API** — the API rejects reads without it. Sources that
            are routed local never need it. Verified against
            the endpoint while the cohort is created, before any data is loaded:
            a missing or unknown project raises here rather than surfacing later
            as a variable that resolves to nothing.
        api_key (str | None):
            Bearer token for the Concepts API. Falls back to the
            ``CORR_CONCEPTS_API_KEY`` environment variable. Surrounding
            whitespace is stripped. Authenticated together with `project`
            against every endpoint the configured sources are routed to, so a
            rejected key is reported before the (slow) data load.
        version (str | None):
            Default version selector for every variable that does not carry its
            own ``::version`` suffix. One of ``"latest"``, ``"vN"``, an ISO date
            ``"YYYY-MM-DD"``, or ``"draftNNNN"``. Defaults to the packaged
            routing configuration, which is ``"latest"``.
        taxonomy (str | None):
            Default taxonomy for unqualified variable names. Defaults to the
            packaged routing configuration, which is ``"corr_v1"``.
        date (str | datetime.date | None):
            Global as-of date. Every variable then resolves to the definition
            that was current on that day, which makes a whole cohort
            reproducible from a single value. Mutually exclusive with an
            explicit non-``"latest"`` ``version``.
        concepts_api_url (str | None):
            Override the endpoint URL from the packaged routing configuration,
            for pointing a cohort at a staging deployment. Sources pinned local
            stay local.

    Attributes:
        obs (pl.DataFrame):
            Static data for each observation. Contains one row per observation (e.g., ICU stay)
            with columns for static variables like demographics and outcomes.

            Example:
                >>> cohort.obs
                patient_id  case_id icu_stay_id            icu_admission        icu_discharge sex   ... inhospital_death
                0  P001         C001    C001_1       2023-01-01 08:30:00  2023-01-03 12:00:00   M   ...  False
                1  P001         C001    C001_2       2023-01-03 14:20:00  2023-01-05 16:30:00   M   ...  False
                2  P002         C002    C002_1       2023-01-02 09:15:00  2023-01-04 10:30:00   F   ...  False
                3  P003         C003    C003_1       2023-01-04 11:45:00  2023-01-07 13:20:00   F   ...  True
                ...

        obsm (dict of pl.DataFrame):
            Dynamic data stored as dictionary of DataFrames. Each DataFrame contains time-series
            data for a variable with columns:

            - recordtime: Timestamp of the measurement
            - value: Value of the measurement
            - recordtime_end: End time (only for duration-based variables like therapies)
            - description: Additional information (e.g., medication names)

            Example:
                >>> cohort.obsm["blood_sodium"]
                   icu_stay_id          recordtime  value
                0  C001_1      2023-01-01 09:30:00   138
                1  C001_1      2023-01-02 10:15:00   141
                2  C001_2      2023-01-03 15:00:00   137
                3  C002_1      2023-01-02 10:00:00   142
                4  C003_1      2023-01-04 12:30:00   139
                ...

    Raises:
        ConceptsApiConfigurationError: If any configured source is routed to the
            Concepts API but no `project` or no API key is available. This is
            checked before any data is loaded.

    Notes:
        - For large cohorts, set ``load_default_vars=False`` to speed up the extraction. You can use
          pre-extracted cohorts as starting points and load them using ``Cohort.load()``.
        - Variables can be added using ``cohort.add_variable()``. Static variables will be added to ``obs``,
          dynamic variables to ``obsm``.
        - For quick prototyping, restrict the cohort with the source's own
          ``include_datasets`` / ``exclude_datasets`` options.

    Note:
        **Where variable definitions come from.** Each source is routed
        independently by the packaged routing configuration
        (``corr_vars/concepts/routing.toml``):

        - ``reprodicu`` is served by the CORR Concepts API.

        A source routed to the API is **never** served from the bundled
        ``vars.json``. Falling back would produce a cohort whose recorded
        version metadata does not describe the definitions that built it, which
        is exactly what the API exists to prevent. Every API failure therefore
        raises rather than degrading quietly.

        ``reprodicu`` is routed to the API but its definitions have not been
        imported there yet, so every reprodicu variable resolves to nothing
        until that import lands. This is a deliberate staging decision.

        ``corr_defaults.default_vars``, ``globals.json`` and ``tables.json``
        stay local in corr_vars and are not served by the API.

    Note:
        **Version pinning is not inherited by dependencies.** Pinning a variable
        with ``::v3`` pins that variable only; the variables it declares in
        ``requires`` still resolve at the cohort default (``latest``, or the
        cohort's as-of ``date``). Pass ``date=`` to the cohort to pin a whole
        dependency graph consistently.

    Note:
        **A name may cover a group of concepts.** Mostly ATC codes, where one
        code stands for a set of substances. Every member then contributes to
        the variable the way an additional data source does: identical column
        names, frames concatenated. A ``::vN`` or ``::draftN`` pin — per
        variable or cohort-wide — addresses one concept's version counter and
        therefore raises
        :exc:`~corr_vars.definitions.exceptions.AmbiguousConceptError` on a
        grouped name; ``latest`` and ``::date`` expand it.

    Examples:
        Create a new cohort:

        >>> cohort = Cohort(
        ...     obs_level="icu_stay",
        ...     load_default_vars=False,
        ...     sources={
        ...         "reprodicu": {"path": "/path/to/reprodICU_files"},
        ...     },
        ... )

        Access static data:

        >>> cohort.obs.select("age_on_admission")  # Get age for all patients
        >>> cohort.obs.filter(pl.col("sex") == "M")  # Filter for male patients

        Access time-series data:

        >>> cohort.obsm["blood_sodium"]  # Get all blood sodium measurements
        >>> # Get blood sodium measurements for a specific observation
        >>> cohort.obsm["blood_sodium"].filter(pl.col(cohort.primary_key) == "12345")

        Build against the Concepts API. ``project`` is required for every source
        routed to the API; the key may also come from ``CORR_CONCEPTS_API_KEY``:

        >>> cohort = Cohort(
        ...     obs_level="icu_stay",
        ...     project="my_project",
        ...     api_key="corr_...",
        ... )

        Freeze a whole cohort — including every ``requires`` dependency — to the
        definitions that were current on a given day:

        >>> cohort = Cohort(project="my_project", date="2025-06-30")

        Pin a single variable to an explicit version, or preview a draft. The
        pin applies to that variable only, never to its dependencies:

        >>> cohort.add_variable("blood_sodium::v3")
        >>> cohort.add_variable("sofa_score::draft1042")
        >>> cohort.add_variable("corr_v1/blood_sodium::2025-06-30")

        Inspect what the cohort was actually built from. The registry holds one
        record per contributor — per source, and per concept when a name
        resolved to a group of them:

        >>> [
        ...     (r["source"], r["concept_id"], r["version"])
        ...     for r in cohort.concept_versions["blood_sodium"]
        ... ]
        [('reprodicu', 17, 3)]
    """

    # Data variables
    _obs: pl.DataFrame
    _obsm: dict[str, pl.DataFrame]
    constant_vars: list[str]

    # Observation level variables
    # Constants
    obs_level: ObsLevel
    primary_key: str
    t_min: str
    t_max: str
    # Non constant
    t_eligible: str
    t_outcome: str

    # Changetracker
    _change_tracker: ChangeTrackerPipeline

    # Configurations
    logger_args: dict[str, Any]
    sources: SourceDict
    project_vars: dict[str, dict[str, Any]]

    # Concept resolution
    _concepts: ConceptResolver
    concept_versions: dict[str, list[dict[str, Any]]]

    # Temporary directory variables
    tmpdir_manager: Final[TemporaryDirectoryManager]
    _current_tmpdir_path: str

    # Flags
    _load_default_vars: bool
    _from_file: bool
    _data_load_time: datetime

    # Properties (defined at the bottom of Cohort class)
    # obs
    # tmpdir_path
    # stata
    # tableone

    def __init__(
        self,
        obs_level: ObsLevelName = "icu_stay",
        sources: SourceDictPartial = {"reprodicu": {}},
        project_vars: dict[str, dict[str, Any]] = {},
        load_default_vars: bool = True,
        logger_args: dict[str, Any] = {},
        project: str | None = None,
        api_key: str | None = None,
        version: str | None = None,
        taxonomy: str | None = None,
        date: str | dt_date | None = None,
        concepts_api_url: str | None = None,
    ) -> None:
        # Setup logger before FIRST
        self._setup_logger(logger_args)

        # Add flag to indicate this is a newly created cohort
        # Used in unit tests
        self._from_file = False

        self.sources = self._load_default_config_data(sources)
        self.project_vars = project_vars

        # Route variable definitions per source (Concepts API vs bundled mapping)
        self._setup_concepts(
            project=project,
            api_key=api_key,
            version=version,
            taxonomy=taxonomy,
            date=date,
            api_url=concepts_api_url,
        )

        # Fail here, not on the first variable: the data load below is slow, and
        # load_default_vars() would swallow the same error once per variable and
        # leave the user with a silently empty cohort. Credentials are both
        # checked for presence and proven against every endpoint actually in
        # use, so a bad key or an unknown project is reported now rather than as
        # a variable that mysteriously resolves to nothing.
        self.concepts.authenticate(self.sources or {})

        # Create temporary directory for storing intermediary data
        self.tmpdir_manager = TemporaryDirectoryManager()

        # Load data and set primary keys for variables
        self._set_obs_level(obs_level)
        self._load_obs_level_keys()  # Primary keys, tmin, tmax, etc.
        self._load_obs_level_data()  # Static / constant data
        self._obsm = {}  # Time series data

        # Load default variables
        self._load_default_vars = load_default_vars
        if load_default_vars:
            self.load_default_vars()

        # Create Change Tracker object - Used for inclusion and exclusion
        self._setup_change_tracker()

        logger.info(
            __(
                "SUCCESS: Extracted data. Cohort has {rows} observations ({obs_level}).",
                rows=len(self),
                obs_level=self.obs_level.lower_name,
            )
        )

    def _setup_logger(self, logger_args: dict[str, Any]) -> None:
        """Changes logger configuration."""
        self.logger_args = logger_args
        self.logger_args.pop("logger", None)
        utils.configure_logger_level_and_handlers(logger=logger, **self.logger_args)

    def _load_default_config_data(
        self,
        sources: SourceDictPartial,
    ) -> SourceDict:
        return loader.config_loader.load_default_config_data(sources=sources)

    def _setup_concepts(
        self,
        *,
        project: str | None = None,
        api_key: str | None = None,
        version: str | None = None,
        taxonomy: str | None = None,
        date: str | dt_date | None = None,
        api_url: str | None = None,
    ) -> None:
        """Build the concept resolver and the per-variable version registry.

        The resolver is created eagerly but connects lazily, so a cohort built
        only from locally routed sources needs neither a project nor a key.

        Args:
            project (str | None): Project name for the Concepts API.
            api_key (str | None): Bearer token, or ``None`` to read the
                ``CORR_CONCEPTS_API_KEY`` environment variable.
            version (str | None): Default version selector.
            taxonomy (str | None): Default taxonomy.
            date (str | datetime.date | None): Global as-of date.
            api_url (str | None): Endpoint URL override.

        Raises:
            ValueError: If both `version` and `date` select a version.
        """
        from corr_vars.concepts.resolver import ConceptResolver

        self._concepts = ConceptResolver(
            project=project,
            api_key=api_key,
            taxonomy=taxonomy,
            version=version,
            date=date,
            api_url=api_url,
        )
        if not hasattr(self, "concept_versions"):
            self.concept_versions = {}

        routes = {
            src: str(self._concepts.route(src)) for src in sorted(self.sources or {})
        }
        if routes:
            logger.debug(__("Concept routing: {routes}", routes=routes))

    @property
    def concepts(self) -> ConceptResolver:
        """The cohort's concept resolver.

        Decides, per source, whether a variable definition is fetched from a
        Concepts API endpoint or read from the source's bundled ``mapping``
        module, and holds the cohort's taxonomy and version defaults.

        Returns:
            ConceptResolver: The resolver, created with package defaults if the
            cohort was restored from a file that predates concept routing.
        """
        if getattr(self, "_concepts", None) is None:
            self._setup_concepts()
        return self._concepts

    def _record_concept_versions(
        self,
        var_name: str,
        resolutions: Mapping[str, ResolvedConcept],
    ) -> None:
        """Record where each contributor's definition of `var_name` came from.

        One record per contributor: per source, and per concept when a name
        resolved to a group of them. The registry is persisted by :meth:`save`,
        so a reloaded cohort documents exactly which taxonomy, concept, and
        version produced each of its columns.

        Args:
            var_name (str): Name of the variable.
            resolutions (Mapping[str, ResolvedConcept]): Per-contributor
                provenance, keyed by contributor key.
        """
        if not hasattr(self, "concept_versions"):
            self.concept_versions = {}

        records = {
            (record.get("source"), record.get("concept_id")): record
            for record in _as_concept_version_records(
                self.concept_versions.get(var_name)
            )
        }
        for resolution in resolutions.values():
            record = resolution.as_dict()
            records[(record.get("source"), record.get("concept_id"))] = record

        self.concept_versions[var_name] = list(records.values())

    def _set_obs_level(self, obs_level: str) -> None:
        """Set the observation level."""
        try:
            # Calling ObsLevel with a str is not an error
            # ObsLevel._missing_ will find the correct ObsLevel
            self.obs_level = ObsLevel(obs_level)  # type: ignore[call-arg]
        except ValueError as exc:
            raise ValueError(f"Observation level {obs_level} not supported.") from exc

    def _load_obs_level_keys(self) -> None:
        """Mirrors ObsLevel keys on Cohort object."""
        self.primary_key = self.obs_level.primary_key
        self.t_min = self.obs_level.t_min
        self.t_max = self.obs_level.t_max
        self.t_eligible = self.obs_level.t_eligible
        self.t_outcome = self.obs_level.t_outcome

    def _load_obs_level_data(self) -> None:
        """Load the static data for the observation level."""
        # Load timestamp for repoducibility
        self._data_load_time = datetime.now(timezone.utc)
        # Load obs data
        self._obs = loader.cohort_loader.load_cohort_data(self.sources, self.obs_level)
        # Every data load adds constant variables (not dependent on tmin tmax)
        # Save for later referencing
        self.constant_vars = [
            col for col in self._obs.columns if col not in ObsLevel.primary_keys()
        ]

    def _setup_change_tracker(self) -> None:
        self._change_tracker = ChangeTrackerPipeline(
            primary_key=self.primary_key,
            initial_df=self._obs,
            initial_description="Observations in database",
        )

    def load_default_vars(
        self, tmin: TimeAnchorColumn | None = None, tmax: TimeAnchorColumn | None = None
    ) -> None:
        """Load the default variables defined in ``vars.json``. It is recommended to use this after filtering your cohort for eligibility to speed up the process.

        Returns:
            None: Variables are loaded into the cohort.

        Examples:
            >>> # Load default variables for an ICU cohort
            >>> cohort = Cohort(obs_level="icu_stay", load_default_vars=False)
            >>> # Apply filters first (faster)
            >>> cohort.include_list(
            ...     [
            ...         {
            ...             "variable": "age_on_admission",
            ...             "operation": ">= 18",
            ...             "label": "Adults",
            ...         }
            ...     ]
            ... )
            >>> # Then load default variables
            >>> cohort.load_default_vars()
        """

        time_window = TimeWindow(
            tmin=tmin or ("hospital_admission", "-48h"),
            tmax=tmax or ("hospital_admission", "+48h"),
        )
        default_variables = loader.var_loader.load_default_variables(self, time_window)

        with logging_redirect_tqdm(loggers=[logging.root, logger]):
            for var in tqdm(
                default_variables, desc="Loading default variables", unit="var"
            ):
                try:
                    self._add_multisource_variable(var)
                    os.system("clear")
                except Exception as exc:
                    logger.error(__("Error {exc}", exc=exc))
                    logger.error(
                        __(
                            "Error traceback: {traceback}",
                            traceback=traceback.format_exc,
                        )
                    )
                    logger.info(
                        __(
                            "Could not load variable {var_name}. Continuing with next variable...",
                            var_name=var.var_name,
                        )
                    )
                    continue

    def add_variable(
        self,
        variable: str | VariableProtocol | MultiSourceVariable,
        save_as: str | None = None,
        tmin: TimeAnchorColumn | None = None,
        tmax: TimeAnchorColumn | None = None,
        use_cache: bool = True,
    ) -> MultiSourceVariable:
        """Add a variable to the cohort.

        You may specify tmin and tmax as a tuple (e.g. ("hospital_admission", "+1d")), in which case it will be relative to the hospital admission time of the patient.

        A string `variable` is a variable reference of the form
        ``[taxonomy/]var_name[::version]``. The optional ``::version`` suffix
        overrides the cohort's default version for this call only, and accepts
        ``latest``, ``vN``, an ISO date ``YYYY-MM-DD`` (as-of), or ``draftNNNN``.
        Missing parts fall back to the cohort defaults.

        Args:
            variable: Variable to add. Either a variable reference string (see above) or a Variable object.
            save_as: Name of the column to save the variable as. Defaults to the bare variable name, without taxonomy or version.
            tmin: Name of the column to use as tmin or tuple (see description).
            tmax: Name of the column to use as tmax or tuple (see description).
            use_cache: Whether an extract of this variable already cached in the
                cohort's tmpdir may be reused. Pass ``False`` to drop it and
                re-read from the source — for example after editing a draft
                definition. The fresh extract is cached again. Dependencies
                pulled in via ``requires`` keep their own cached extracts; clear
                those with ``cohort.clear_variable_cache(name)``.

        Returns:
            Variable: The variable object.

        Note:
            A ``::version`` pin applies to this variable only. Its ``requires``
            dependencies still resolve at the cohort default, so pinning a
            derived variable does not freeze the variables it is derived from.
            Use ``Cohort(date=...)`` to freeze a whole dependency graph.

        Raises:
            ValueError: If the variable reference is malformed.
            ConceptsApiError: If a source routed to the Concepts API could not
                be served. Routed sources never fall back to the bundled
                ``vars.json``.

        Examples:
            >>> cohort.add_variable("blood_sodium")

            >>> cohort.add_variable("blood_sodium::v3")  # pin to version 3

            >>> cohort.add_variable("blood_sodium::2025-06-30")  # as of a date

            >>> cohort.add_variable("blood_sodium::draft1042")  # preview a draft

            >>> # re-extract, ignoring any cached copy in the cohort's tmpdir
            >>> cohort.add_variable("blood_sodium::draft1042", use_cache=False)

            >>> cohort.add_variable(
            ...     variable="anx_dx_covid_19",
            ...     tmin=("hospital_admission", "-1d"),
            ...     tmax=cohort.t_eligible,
            ... )


            >>> from corr_vars.sources.local_datasource import NativeStatic
            >>> cohort.add_variable(
            ...     NativeStatic(
            ...         var_name="highest_hct_before_eligible",
            ...         select="!max value",
            ...         base_var="blood_hematokrit",
            ...         tmax=cohort.t_eligible,
            ...     )
            ... )


            >>> cohort.add_variable(
            ...     variable="any_med_glu",
            ...     save_as="glucose_prior_eligible",
            ...     tmin=(cohort.t_eligible, "-48h"),
            ...     tmax=cohort.t_eligible,
            ... )
        """
        var = self.load_variable(variable=variable, tmin=tmin, tmax=tmax)
        if not use_cache:
            self.clear_variable_cache(var.var_name)
        return self._add_multisource_variable(variable=var, save_as=save_as)

    def clear_variable_cache(self, var_name: str | None = None) -> None:
        """Drop cached extracts from the cohort's temporary directory.

        Sources may cache a variable's raw extract in the tmpdir and reuse it for
        the rest of the session, so a definition that changed underneath — a
        redrafted concept above all — keeps serving the old data until its cache
        is dropped.

        Args:
            var_name: Variable whose cached extract to drop, across all
                contributors. Drops every cached variable when ``None``.

        Examples:
            >>> cohort.clear_variable_cache("blood_sodium")
            >>> cohort.clear_variable_cache()  # everything
        """
        if var_name is None:
            for file_name in self.tmpdir_manager.tmpdir_variables:
                match = self.tmpdir_manager.variable_file_pattern.match(file_name)
                if match is not None:
                    self.tmpdir_manager.delete_tmpdir_variable(match.group("var_name"))
            return

        self.tmpdir_manager.delete_tmpdir_variable(var_name)

    def add_horizon_variable(
        self,
        var_name: str,
        save_as: str | None = None,
        *,
        t0: str,
        horizon: int,
    ) -> MultiSourceVariable:
        """Add a variable over a ``[t0, t0 + horizon days]`` follow-up window.

        The building block for horizon outcomes — free-day counts, horizon mortality,
        readmissions, … — it derives the window from a time-zero column and a horizon
        length and delegates to :meth:`add_variable`. Call it once per
        (variable, horizon).

        Args:
            var_name: Variable to add (a ``vars.json`` template name, or a Variable).
            save_as: Column name to save as. Defaults to ``"<var_name>_<horizon>d"``.
            t0: Obs column marking time zero (e.g. ``"icu_admission"``). Must be a plain
                column name — the horizon end is ``t0 + horizon days``.
            horizon: Follow-up length in days.

        Returns:
            MultiSourceVariable: The variable added.

        Examples:
            >>> cohort.add_horizon_variable(
            ...     "vent_free_days", t0="icu_admission", horizon=28
            ... )
            # -> saved as vent_free_days_28d over [icu_admission, icu_admission + 28d]

            >>> cohort.add_horizon_variable(
            ...     "mortality", save_as="mort_90d", t0="icu_admission", horizon=90
            ... )
        """
        if not isinstance(t0, str):
            raise TypeError(
                "add_horizon_variable requires t0 to be a column name; add a derived "
                "anchor column first if you need an offset time zero."
            )
        return self.add_variable(
            var_name,
            save_as=save_as or f"{var_name}_{horizon}d",
            tmin=t0,
            tmax=(t0, f"+{horizon}d"),
        )

    def _add_multisource_variable(
        self,
        variable: MultiSourceVariable,
        *,
        save_as: str | None = None,
    ) -> MultiSourceVariable:
        # Backup
        obs_backup = self._obs.clone()
        obs_backup_idset = set(obs_backup.select(self.primary_key).unique().to_series())

        # Extract
        data = variable.extract(self, with_source_column=variable.dynamic)

        # Save
        self._save_variable(
            var_name=variable.var_name,
            var_dynamic=variable.dynamic,
            var_data=data,
            save_as=save_as,
        )

        # Verify cohort integrity & restore if compromised
        new_obs_idset = set(self._obs.select(self.primary_key).unique().to_series())
        if new_obs_idset != obs_backup_idset:
            # If mismatch - restore cohort
            self._obs = obs_backup
            raise CohortDataError(
                "Adding variable altered cohort composition - data compromised."
            )

        # Assert no duplicates
        # Note: This is already checked inside _save_variable for static variable data.
        self._validate_cohort()

        return variable

    def load_variable(
        self,
        variable: str | tuple[str, TimeWindow] | VariableProtocol | MultiSourceVariable,
        tmin: TimeAnchorColumn | None = None,
        tmax: TimeAnchorColumn | None = None,
        include_sources: Iterable[str] | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> MultiSourceVariable:
        """Resolve `variable` into a :class:`MultiSourceVariable` without extracting it.

        A string is parsed as a ``[taxonomy/]var_name[::version]`` reference. A
        ``(var_name, time_window)`` tuple — the form ``requires`` dependencies
        arrive in — carries no version and therefore resolves at the cohort
        default, so pinning a parent does not pin its dependencies.

        Args:
            variable: A variable reference string, a ``(name, time_window)``
                tuple, a `VariableProtocol`, or an existing `MultiSourceVariable`.
            tmin: Name of the column to use as tmin, or a ``(column, delta)`` tuple.
            tmax: Name of the column to use as tmax, or a ``(column, delta)`` tuple.
            include_sources: Restrict the lookup to these sources.
            overrides: Config fields merged onto every source's entry, for this
                load only.

        Returns:
            MultiSourceVariable: The resolved variable, not yet extracted.

        Raises:
            ValueError: If a string reference is malformed.
            VariableNotFoundError: If no source defines the variable.
            ConceptsApiError: If an API-routed source could not be served.
        """
        time_window = TimeWindow(
            tmin=tmin or self.t_min,
            tmax=tmax or self.t_max,
        )
        time_window_specified = (tmin is not None, tmax is not None)

        spec: VariableSpec | None = None
        if isinstance(variable, str):
            spec = self.concepts.parse(variable)
            template: Any = (spec.name, time_window)
        else:
            template = variable

        # Warn about ignored tmin/tmax
        if (
            not isinstance(variable, str)
            and not isinstance(variable, tuple)
            and time_window_specified
        ):
            variable_type = type(template).__name__
            warnings.warn(
                f"Please specify tmin/tmax inside {variable_type}",
                UserWarning,
            )

        return self._load_multisource_variable(
            template, include_sources=include_sources, overrides=overrides, spec=spec
        )

    def _load_multisource_variable(
        self,
        template: tuple[str, TimeWindow] | VariableProtocol | MultiSourceVariable,
        include_sources: Iterable[str] | None = None,
        overrides: dict[str, Any] | None = None,
        spec: VariableSpec | None = None,
    ) -> MultiSourceVariable:
        """Loads a MultiSourceVariable from JSON template and time window or specified Variable object"""
        if isinstance(template, MultiSourceVariable):
            return template

        if isinstance(template, tuple):
            var_name, time_window = template

            if var_name in self.constant_vars:
                logger.info(
                    __(
                        "tmin/tmax are ignored for the constant variable {var_name}.",
                        var_name=var_name,
                    )
                )
                var = self._load_constant_variable(
                    var_name=var_name, include_sources=include_sources
                )
            else:
                var = loader.var_loader.load_variable(
                    var_name=var_name,
                    cohort=self,
                    time_window=time_window,
                    include_sources=include_sources,
                    overrides=overrides,
                    spec=spec,
                )
            return var

        source = loader.guess_variable_source(template) or "Not applicable"
        return MultiSourceVariable({source: template})

    def _load_constant_variable(
        self,
        var_name: str,
        include_sources: Iterable[str] | None = None,
    ) -> MultiSourceVariable:
        """Load a constant variable from obs."""

        class ObsVariable:
            def __init__(
                self,
                var_name: str,
                source: str | None = None,
            ) -> None:
                self.var_name = var_name
                self.dynamic = False
                # TODO: Fix Timewindow tmax not beeing deterministic
                # For now, set to t_min to avoid issues with non-deterministic tmax in constant variables.
                # This is not ideal but won't cause issues since tmax is not used for constant variables.
                # Setting TimeWindow without tmin and tmax will also assign a non-deterministic tmax...
                self.time_window = TimeWindow.from_obs_level(ObsLevel.PATIENT)
                self.source = source

            def extract(self, cohort: Cohort) -> pl.DataFrame:
                if self.source:
                    df = cohort._obs.select("data_source", cohort.primary_key, var_name)
                    df = df.filter(data_source=self.source).drop("data_source")
                else:
                    df = cohort._obs.select(cohort.primary_key, var_name)
                return df

        return MultiSourceVariable(
            {
                source: ObsVariable(var_name=var_name, source=source)
                for source in include_sources or self.sources.keys()
            },
        )

    def _save_variable(
        self,
        var_name: str,
        var_dynamic: bool,
        var_data: pl.DataFrame | pl.LazyFrame | pd.DataFrame,
        save_as: str | None = None,
    ) -> None:
        """Save a Variable object to the cohort.

        Will either add a column to obs or add an entry to obsm (for dynamic variables).

        Args:
            var: Variable to save. (Must be already extracted)
            save_as: Name of the column to save the variable as. Defaults to variable name.

        Returns:
            None: Variable is saved to the cohort.
        """
        # NOTE: Won't allow "" as var name
        save_as = save_as or var_name
        data = utils.convert_to_polars_df(var_data)

        if var_dynamic:
            self._obsm[save_as] = data

        else:
            # Assert single observation per primary key
            self._validate_cohort(obs=data, obs_source=save_as)

            # Find and drop obs columns that start with save_as (var_name if unspecified)
            drop_cols = self._obs.select(
                pl.selectors.starts_with(save_as).exclude(self.primary_key)
            ).columns
            if drop_cols:
                logger.info(
                    __(
                        "DROP existing variable(s): {drop_cols} (N={n_obs})",
                        drop_cols=", ".join(drop_cols),
                        n_obs=len(self._obs),
                    )
                )
                self._obs = self._obs.drop(drop_cols)
                del drop_cols  # Clear to avoid accidental re-use later

            logger.debug(
                __(
                    "Variable data size: {data_shape}, obs size: {obs_shape}",
                    data_shape=data.shape,
                    obs_shape=self._obs.shape,
                )
            )

            # Rename columns based on conditions
            if save_as != var_name:
                logger.info("Renaming columns...")
                data = data.rename(
                    lambda col: (
                        f"{save_as}{col.removeprefix(var_name)}"
                        if col != self.primary_key and col.startswith(var_name)
                        else col
                    )
                )

            # Assert that only expected columns are present
            logger.debug("Simple rename check.")
            unexpected_columns = data.drop(
                self.primary_key, pl.selectors.starts_with(save_as)
            ).columns
            if unexpected_columns:
                raise ValueError(
                    "Variable data contains unexpected columns: "
                    f"{', '.join(unexpected_columns)}"
                )

            # Join with self.obs
            logger.debug("Joining variable data with obs")
            self._obs = self._obs.join(data, on=self.primary_key, how="left")

        gc.collect()

        def nunique_info() -> int:
            return data.select(pl.col(self.primary_key)).n_unique()

        summary_config = pl.Config(
            tbl_formatting="UTF8_FULL_CONDENSED",
            tbl_hide_column_data_types=True,
            tbl_hide_dataframe_shape=True,
            tbl_width_chars=-1,
            tbl_rows=-1,
            tbl_cols=-1,
            tbl_cell_numeric_alignment="RIGHT",
            fmt_str_lengths=35,
            thousands_separator=".",
            decimal_separator=",",
            float_precision=2,
        )

        @summary_config
        def summary():
            return str(data.describe())

        logger.info(__("SUCCESS: Saved variable: {save_as}", save_as=save_as))
        logger.info(__("Data summary:\n{summary}", summary=summary))
        logger.info(
            __(
                "Number of unique non-NA entries: {nunique_info}",
                nunique_info=nunique_info,
            )
        )

    # Time anchors, inclusion and exclusion
    def set_t_eligible(self, t_eligible: str, drop_ineligible: bool = True) -> None:
        """Set the time anchor for eligibility. This can be referenced as cohort.t_eligible throughout the process and is required to add inclusion or exclusion criteria.

        Args:
            t_eligible: Name of the column to use as t_eligible.
            drop_ineligible: Whether to drop ineligible patients. Defaults to True.

        Returns:
            None: t_eligible is set.

        Examples:
            >>> # Add a suitable time-anchor variable
            >>> from corr_vars.sources.local_datasource import NativeStatic
            >>> cohort.add_variable(
            ...     NativeStatic(
            ...         var_name="spo2_lt_90",
            ...         base_var="spo2",
            ...         select="!first recordtime",
            ...         where="value < 90",
            ...     )
            ... )
            >>> # Set the time anchor for eligibility
            >>> cohort.set_t_eligible("spo2_lt_90")
        """
        self._assert_datetime_col(t_eligible)
        logger.warning(
            __(
                "t_eligible already set to {t_eligible_old}. "
                "Will overwrite and set to {t_eligible_new}.",
                t_eligible_old=self.t_eligible,
                t_eligible_new=t_eligible,
            )
        )
        self.t_eligible = t_eligible

        if drop_ineligible:
            self._drop_ineligible()

    def set_t_outcome(self, t_outcome: str) -> None:
        """Set the time anchor for outcome. This can be referenced as cohort.t_outcome throughout the process and is recommended to specify for your study.

        Args:
            t_outcome (str): Name of the column to use as t_outcome.

        Returns:
            None: t_outcome is set.

        Examples:
            >>> cohort.set_t_outcome("hospital_discharge")
        """
        self._assert_datetime_col(t_outcome)
        logger.warning(
            __(
                "t_outcome already set to {t_outcome_old}. "
                "Will overwrite and set to {t_outcome_new}.",
                t_outcome_old=self.t_outcome,
                t_outcome_new=t_outcome,
            )
        )
        self.t_outcome = t_outcome

    def _assert_datetime_col(self, col: object) -> None:
        if not isinstance(col, str):
            raise TypeError("Column name must be a string.")
        if col not in self._obs.columns:
            raise ColumnNotFoundError(f"Column {col} not found in obs.")
        if self._obs[col].dtype != pl.Datetime:
            raise TypeError(f"Column {col} is not a datetime column.")

    def _drop_ineligible(self) -> None:
        """Drops ineligible observations (where t_eligible is NaT)."""
        self._warn_on_change_tracker_misalignment()
        mask = self._obs[self.t_eligible].is_null()

        # Log absolute and relative loss
        n_dropped = int(mask.sum())
        total = len(self._obs)
        logger.info(
            __(
                "DROP: {n_dropped_count} rows ({n_dropped_percent:.2%}) due to {t_eligible} is NaT",
                n_dropped_count=n_dropped,
                n_dropped_percent=n_dropped / total,
                t_eligible=self.t_eligible,
            )
        )

        # Drop ineligible observations and update state
        self._obs = self._obs.remove(mask)
        self._change_tracker.add_step(
            description=f"Excluded {n_dropped} rows due to {self.t_eligible} is NaT",
            after_df=self._obs,
        )

        # Print warning if already dropped
        if self._change_tracker.delta_uniques != 0:
            logger.warning(
                __(
                    "Already dropped {already_dropped} rows before.\n"
                    "CAVE! Previously ineligible patients will not be restored.",
                    already_dropped=self._change_tracker.delta_uniques,
                )
            )

    def _warn_on_change_tracker_misalignment(self) -> None:
        if len(self._obs) != self._change_tracker.current_uniques:
            warnings.warn(
                "Current change tracker state is not equal to cohort size. "
                "This could indicate that the cohort was modified without proper tracking. "
                "Inclusion/Exclusion charts might be inaccurate.\n"
                f"Cohort.obs: {len(self._obs)} observations / ChangeTracker: {self._change_tracker.current_uniques} observations",
                UserWarning,
            )

    def change_tracker(
        self,
        description: str,
        group: str | None = None,
        mode: Literal["include", "exclude"] = "include",
    ) -> ChangeTrackerContext:
        """Return a context manager to group cohort edits and record a single ChangeTracker state on exit.

        Example:
            with cohort.change_tracker("Adults", mode="include") as track:
                track.filter(pl.col("age_on_admission") >= 18)
        """
        self._warn_on_change_tracker_misalignment()
        return ChangeTrackerContext(
            pipeline=self._change_tracker,
            cohort=self,
            description=description,
            group=group,
            mode=mode,
        )

    def include(self, *args, **kwargs) -> None:
        """Add an inclusion criterion to the cohort. It is recommended to use ``Cohort.include_list()`` and add all of your inclusion criteria at once. However, if you need to specify criteria at a later stage, you can use this method.

        Warning:
            You should call ``Cohort.include_list()`` before calling ``Cohort.include()`` to ensure that the inclusion criteria are properly tracked.

        Args:
            variable (str | Variable),
            operation (str),
            label (str),
            operations_done (str)
            [Optional: tmin, tmax]

        Returns:
            None: Criterion is added to the cohort.

        Note:
            ``operation`` is passed to ``pandas.DataFrame.query``, which uses a `slightly modified Python syntax <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html>`_. Also, if you specify "true"/"True" or "false"/"False" as a value for ``operation``, it will be converted to "== True" or "== False", respectively.

        Examples:
            >>> cohort.include(
            ...     variable="age_on_admission",
            ...     operation=">= 18",
            ...     label="Adult",
            ...     operations_done="Include only adult patients",
            ... )
        """
        return self._include_exclude("include", *args, **kwargs)

    def exclude(self, *args, **kwargs) -> None:
        """Add an exclusion criterion to the cohort. It is recommended to use ``Cohort.exclude_list()`` and add all of your exclusion criteria at once. However, if you need to specify criteria at a later stage, you can use this method.

        Warning:
            You should call ``Cohort.exclude_list()`` before calling ``Cohort.exclude()`` to ensure that the exclusion criteria are properly tracked.

        Args:
            variable (str | Variable),
            operation (str),
            label (str),
            operations_done (str)
            [Optional: tmin, tmax]

        Returns:
            None: Criterion is added to the cohort.

        Note:
            ``operation`` is passed to ``pandas.DataFrame.query``, which uses a `slightly modified Python syntax <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html>`_. Also, if you specify "true"/"True" or "false"/"False" as a value for ``operation``, it will be converted to "== True" or "== False", respectively.

        Examples:
            >>> cohort.exclude(
            ...     variable="elix_total",
            ...     operation="> 20",
            ...     operations_done="Exclude patients with high Elixhauser score",
            ... )
        """
        return self._include_exclude("exclude", *args, **kwargs)

    def _include_exclude(
        self,
        mode: Literal["include", "exclude"],
        variable: str | VariableProtocol | MultiSourceVariable,
        operation: str | pl.Expr,
        operations_done: str | None = None,
        label: str | None = None,
        allow_obs: bool = False,
        tmin: str | None = None,
        tmax: str | None = None,
    ) -> None:
        """Add an inclusion or exclusion criterion to the cohort.

        Args:
            mode (Literal["include", "exclude"]): Whether to add an inclusion or exclusion criterion.
            variable (str | Variable),
            operation (str),
            operations_done (str | None),
            label (str | None),
            allow_obs (bool): Allow using a stored variable in obs insteead of re-extracting. CAVE: This can be dangerous when trying to set custom time bounds.
            [Optional: tmin, tmax]

        Returns:
            None: Criterion is added to the cohort.
        """
        self._warn_on_change_tracker_misalignment()

        # Check if this is a defined variable or a custom variable
        if isinstance(variable, str) and (
            variable in self.constant_vars
            or (allow_obs and variable in self._obs.columns)
        ):
            data = self._obs.select([variable, self.primary_key])
            var_name = variable
        else:
            var = self.load_variable(variable=variable, tmin=tmin, tmax=tmax)
            data = var.extract(self)
            var_name = var.var_name

        if isinstance(operation, str):
            operation_expr = pl.sql_expr(f"{var_name} {operation}")
        else:
            operation_expr = operation

        operations_done = operations_done or operation_expr.meta.serialize(
            format="json"
        )

        try:
            operation_ids = data.filter(operation_expr)[self.primary_key]
        except PolarsError as exc:
            logger.error(
                __(
                    "Failed to add {mode} criterion ({variable}{operation}) with data:\n"
                    "{data}\n"
                    "{exc}\n"
                    "{traceback}",
                    mode=mode,
                    variable=variable,
                    operation=operation,
                    data=data,
                    exc=exc,
                    traceback=traceback.format_exc,
                )
            )

        match mode:
            case "include":
                self._obs = self._obs.filter(
                    pl.col(self.primary_key).is_in(operation_ids)
                )
                self._change_tracker.add_step(
                    description=f"Included {operations_done}",
                    group=label,
                    after_df=self._obs,
                )
            case "exclude":
                self._obs = self._obs.remove(
                    pl.col(self.primary_key).is_in(operation_ids)
                )
                self._change_tracker.add_step(
                    description=f"Excluded {operations_done}",
                    group=label,
                    after_df=self._obs,
                )

    def include_list(
        self, inclusion_list: Iterable[dict] = ()
    ) -> ChangeTrackerPipeline:
        """Add an inclusion criteria to the cohort.

        Args:
            inclusion_list (Iterable[dict]): List of inclusion criteria. Must include a dictionary with keys:
                * ``variable`` (str | Variable): Variable to use for exclusion
                * ``operation`` (str): Operation to apply (e.g., "> 5", "== True")
                * ``label`` (str): Short label for the exclusion step
                * ``operations_done`` (str): Detailed description of what this exclusion does
                * ``tmin`` (str, optional): Start time for variable extraction
                * ``tmax`` (str, optional): End time for variable extraction

        Returns:
            ct (CohortTracker): CohortTracker object, can be used to plot inclusion chart

        Note:
            Per default, all inclusion criteria are applied from ``tmin=cohort.tmin`` to ``tmax=cohort.t_eligible``. This is recommended to avoid introducing immortality biases. However, in some cases you might want to set custom time bounds.

        Examples:
            >>> ct = cohort.include_list(
            ...     [
            ...         {
            ...             "variable": "age_on_admission",
            ...             "operation": ">= 18",
            ...             "label": "Adult patients",
            ...             "operations_done": "Excluded patients under 18 years old",
            ...         }
            ...     ]
            ... )
            >>> ct.create_flowchart()
        """
        for inclusion in inclusion_list:
            self.include(**inclusion)

        return self._change_tracker

    def exclude_list(
        self, exclusion_list: Iterable[dict] = ()
    ) -> ChangeTrackerPipeline:
        """Add an exclusion criteria to the cohort.

        Args:
            exclusion_list (Iterable[dict]): List of exclusion criteria. Each criterion is a dictionary containing:
                * ``variable`` (str | Variable): Variable to use for exclusion
                * ``operation`` (str): Operation to apply (e.g., "> 5", "== True")
                * ``label`` (str): Short label for the exclusion step
                * ``operations_done`` (str): Detailed description of what this exclusion does
                * ``tmin`` (str, optional): Start time for variable extraction
                * ``tmax`` (str, optional): End time for variable extraction

        Returns:
            ct (CohortTracker): CohortTracker object, can be used to plot exclusion chart

        Note:
            Per default, all exclusion criteria are applied from ``tmin=cohort.tmin`` to ``tmax=cohort.t_eligible``. This is recommended to avoid introducing immortality biases. However, in some cases you might want to set custom time bounds.

        Examples:
            >>> ct = cohort.exclude_list(
            ...     [
            ...         {
            ...             "variable": "any_rrt_icu",
            ...             "operation": "true",
            ...             "label": "No RRT",
            ...             "operations_done": "Excluded RRT before hypernatremia",
            ...         },
            ...         {
            ...             "variable": "any_dx_tbi",
            ...             "operation": "true",
            ...             "label": "No TBI",
            ...             "operations_done": "Excluded TBI before hypernatremia",
            ...         },
            ...         {
            ...             "variable": "sodium_count",
            ...             "operation": "< 1",
            ...             "label": "Final cohort",
            ...             "operations_done": "Excluded cases with less than 1 sodium measurement after hypernatremia",
            ...             "tmin": cohort.t_eligible,
            ...             "tmax": "hospital_discharge",
            ...         },
            ...     ]
            ... )
            >>> ct.create_flowchart()  # Plot the exclusion flowchart
        """
        for exclusion in exclusion_list:
            self.exclude(**exclusion)

        return self._change_tracker

    def add_variable_definition(self, var_name: str, var_dict: dict[str, Any]) -> None:
        """Add or update a local variable definition.

        Args:
            var_name (str): Name of the variable.
            var_dict (dict[str, Any]): Dictionary containing variable definition. Can be partial -
                    missing fields will be inherited from global definition.

        Examples:
            Add a completely new variable:

            >>> cohort.add_variable_definition(
            ...     "my_new_var",
            ...     {
            ...         "type": "native_dynamic",
            ...         "table": "labs",
            ...         "where": "name LIKE '%new%'",
            ...         "value_dtype": "DOUBLE",
            ...         "cleaning": {"value": {"low": 100, "high": 150}},
            ...     },
            ... )

            Partially override existing variable:

            >>> cohort.add_variable_definition(
            ...     "blood_sodium",
            ...     {"where": "name LIKE '%custom_sodium%'"},
            ... )
        """
        self.project_vars.setdefault(var_name, {})
        self.project_vars[var_name].update(var_dict)

    def get_variable_definition(self, var_name: str) -> dict[str, dict[str, Any]]:
        """Get the variable definition for a given variable name.

        Args:
            var_name (str): Name of the variable to get variable definitions for.

        Returns:
            definition (dict[str, dict[str, Any]]): Dictionary of variable definitions per source.
        """
        definition = {}
        # Legacy mode -> Update all existing sources
        if var_name in self.project_vars:
            definition = dict.fromkeys(self.sources, self.project_vars[var_name])
        else:
            # New implementation with source specification
            definition = {
                src: self.project_vars[var_name]
                for src in self.sources
                if var_name in self.project_vars.get(src, {})
            }
        return definition

    def add_inclusion(self) -> None:
        raise NotImplementedError("Use include_list instead")

    def add_exclusion(self) -> None:
        raise NotImplementedError("Use exclude_list instead")

    def _validate_cohort(
        self, obs: pl.DataFrame | None = None, obs_source: str = "cohort"
    ) -> None:
        """Validate data integrity of an obs DataFrame"""
        # Check for unique primary keys in .obs
        df = self._obs if obs is None else obs
        if df.select(pl.col(self.primary_key)).is_unique().not_().any():
            raise CohortDataError(
                f"Duplicate entries found in obs for primary key '{self.primary_key}' ({obs_source})."
            )

    def _to_files(
        self, folder: str | PathLike, saver: CohortSaverProtocol, ext: str
    ) -> None:
        """Save the cohort to multiple files.

        Args:
            folder (str | PathLike): Path to the folder where files will be saved.
            saver (CohortSaverProtocol): Callable which saves the passed DataFrame in the specified format in the passed location
            ext (str): File extension
        """
        logger.info(__("Saving cohort to {folder}", folder=folder))
        os.makedirs(folder, exist_ok=True)

        logger.info("Preparing for export...")
        saver(os.path.join(folder, f"_obs.{ext}"), self._obs)

        logger.info("Saving variables...")
        for var_name, var_data in self._obsm.items():
            saver(os.path.join(folder, f"{var_name}.{ext}"), var_data)

        logger.info("Done!")

    def to_files(self, folder: str | PathLike, ext: CohortExportFormats) -> None:
        """Convenience method to save the cohort to various file formats

        Args:
            folder (str | PathLike[str]): Path to the folder where the files will be saved.
            ext (CohortExportFormats): File extension to use for saving the files.
        """

        def xlsx_saver(file_path: str | Path, df: pl.DataFrame) -> None:
            df.write_excel(file_path)

        ext_save_mapping: dict[CohortExportFormats, CohortSaverProtocol] = {
            "arrow": lambda file_path, df: df.write_ipc(file_path),
            # NOTE: Convert to pandas to workaround the missing support for nested data
            "csv": lambda file_path, df: df.to_pandas().to_csv(file_path),
            "json": lambda file_path, df: df.write_json(file_path),
            "jsonl": lambda file_path, df: df.write_ndjson(file_path),
            "parquet": lambda file_path, df: df.write_parquet(file_path),
            "xlsx": xlsx_saver,
        }

        return self._to_files(folder, ext_save_mapping[ext], ext)

    def to_csv(self, folder: str | PathLike[str]) -> None:
        """Save the cohort to CSV files.

        Args:
            folder (str | PathLike[str]): Path to the folder where CSV files will be saved.

        Examples:
            >>> cohort.to_csv("output_data")
            >>> # Creates:
            >>> # output_data/_obs.csv
            >>> # output_data/blood_sodium.csv
            >>> # output_data/heart_rate.csv
            >>> # ... (one file per variable)
        """
        return self.to_files(folder, "csv")

    def to_parquet(self, folder: str | PathLike[bytes]) -> None:
        """Save the cohort to parquet files.

        Args:
            folder (str | PathLike[str]): Path to the folder where parquet files will be saved.

        Examples:
            >>> cohort.to_parquet("output_data")
            >>> # Creates:
            >>> # output_data/_obs.parquet
            >>> # output_data/blood_sodium.parquet
            >>> # output_data/heart_rate.parquet
            >>> # ... (one file per variable)
        """
        return self.to_files(folder, "parquet")

    def save(self, filename: str | PathLike[str]) -> None:
        """Save the cohort to a single compressed .corr3 archive (.tar.zst equivalent). Saves cohort.__dict__ (excluding obs, obsm, variables, conn) to state.pkl, obs to obs.parquet, and each obsm DataFrame to obsm_<var_name>.parquet in a temp dir.

        Args:
            filename: Path to the .corr3 archive

        Returns:
            None
        """
        filename_path = Path(filename)
        if filename_path.suffix not in ["", ".corr3"]:
            warnings.warn(
                f"The new file format is .corr3. Will use .corr3 instead of {filename_path.suffix}",
                UserWarning,
            )
        resolved_filename_path = filename_path.with_suffix(".corr3")
        return self._save_corr3(resolved_filename_path)

    def _save_corr3(self, filename: str | PathLike[str]) -> None:
        """Save the cohort to a single compressed .corr3 archive (.tar.zst equivalent).
        Saves cohort.__dict__ (excluding obs, obsm, variables, conn) to state.pkl,
        obs to obs.parquet, and each obsm DataFrame to obsm_<var_name>.parquet in a temp dir

        Args:
            filename: Path to the .corr3 archive

        Returns:
            None

        """
        # NOTE: Default path is already set for the corr_vars module by the global tempfile.tempdir
        with tempfile.TemporaryDirectory() as tmpdir:
            # save VERSION file
            with open(os.path.join(tmpdir, "VERSION.txt"), "w", encoding="utf8") as f:
                f.write("3")

            # Save obs and obsm as parquet files
            self._obs.write_parquet(os.path.join(tmpdir, "obs.parquet"))
            for var_name, df in self._obsm.items():
                df.write_parquet(os.path.join(tmpdir, f"obsm_{var_name}.parquet"))

            # Prepare dict to save (exclude obs, obsm, variables, conn)
            state = deepcopy(self.__dict__)
            state.pop("_obs", None)
            state.pop("_obsm", None)
            state.pop("obs", None)
            state.pop("obsm", None)
            state.pop("variables", None)
            state.pop("conn", None)
            # The resolver holds live HTTP clients and the API key; only its
            # settings are persisted, via the "concepts" save key below.
            state.pop("_concepts", None)

            # Pop password before saving to avoid unencrypted password in the archive
            for source in state["sources"]:
                if (
                    "conn_args" in state["sources"][source]
                    and "password" in state["sources"][source]["conn_args"]
                ):
                    state["sources"][source]["conn_args"].pop("password")

            # Save required parts of state to recreate the cohort as JSON
            save_keys = [
                "sources",
                "project_vars",
                "logger_args",
                "constant_vars",
                "primary_key",
                "t_min",
                "t_max",
                "t_eligible",
                "t_outcome",
            ]
            save_state = {key: state[key] for key in save_keys}

            # Special objects: obs_level,tmpdir manager, change tracker pipeline

            save_state["obs_level"] = self.obs_level.lower_name

            # Document exactly which concept versions built this cohort.
            # The API key is deliberately not part of the persisted settings.
            save_state["concept_versions"] = getattr(self, "concept_versions", {})
            save_state["concepts_settings"] = self.concepts.settings()

            if self.tmpdir_manager._check_exist():
                save_state["tmpdir_path"] = self.tmpdir_path
            else:
                save_state["tmpdir_path"] = None

            state_path = os.path.join(tmpdir, "cohort.json")
            with open(state_path, "w", encoding="utf8") as f:
                json.dump(save_state, f, indent=4)

            # Change tracker pipeline is currently pickled - to be changed to JSON later
            if self._change_tracker is not None:
                change_tracker_path = os.path.join(tmpdir, "change_tracker.pkl")
                with open(change_tracker_path, "wb") as f:
                    pickle.dump(self._change_tracker, f)

            # Tar all files in tmpdir
            tar_path = os.path.join(tmpdir, "archive.tar")
            with tarfile.open(tar_path, "w") as tar:
                for file in os.listdir(tmpdir):
                    if file != "archive.tar":
                        tar.add(os.path.join(tmpdir, file), arcname=file)

            # Compress tar with zstandard
            cctx = zstd.ZstdCompressor(level=3)
            with open(tar_path, "rb") as tar_in, open(filename, "wb") as out:
                out.write(cctx.compress(tar_in.read()))

        logger.info(
            __(
                "SUCCESS: Saved cohort to {filename!r} ({n_obsm} dynamic variables, {n_obs} observations [{obs_level}])",
                filename=filename,
                n_obsm=len(self._obsm),
                n_obs=len(self._obs),
                obs_level=self.obs_level.lower_name,
            )
        )

    @classmethod
    def load(cls, filename: str | PathLike[str], password: str | None = None) -> Cohort:
        filename_path = Path(filename)
        if filename_path.suffix == ".corr2":
            return cls._load_corr2(filename_path, password)
        if filename_path.suffix == ".corr3":
            return cls._load_corr3(filename_path, password)
        raise ValueError(
            "Unsupported file format. The new Cohort module only supports .corr2 and .corr3 files."
        )

    @classmethod
    def _load_corr2(
        cls, filename: str | PathLike[str], password: str | None = None
    ) -> Cohort:
        """Load a cohort from a single compressed .corr2 archive (.tar.zst equivalent).
        Loads cohort.__dict__ from state.pkl, obs from obs.parquet, and obsm from obsm_<var_name>.parquet files
        extracted from the archive.

        Args:
            filename: Path to the .corr2 archive
            password: Password to restore for sources with password field

        Returns:
            Cohort: The loaded cohort
        """
        # BACKWARDS COMPATIBLITY SECTION:
        import io

        CLASS_RENAMES = {
            "corr_vars.utils.helpers": {
                "ChangeTracker": "corr_vars.core.change_tracker.ChangeTrackerPipeline"
            }
        }

        class _RenamingUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module in CLASS_RENAMES and name in CLASS_RENAMES[module]:
                    new_path = CLASS_RENAMES[module][name]
                    new_mod, new_cls = new_path.rsplit(".", 1)
                    mod = __import__(new_mod, fromlist=[new_cls])
                    return getattr(mod, new_cls)
                # Default
                return super().find_class(module, name)

        def load_with_rename(data: bytes):
            return _RenamingUnpickler(io.BytesIO(data)).load()

        # LOAD FILE

        # NOTE: Default path is already set for the corr_vars module by the global tempfile.tempdir
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = os.path.join(tmpdir, "archive.tar")
            dctx = zstd.ZstdDecompressor()
            with open(filename, "rb") as inp, open(tar_path, "wb") as out:
                out.write(dctx.decompress(inp.read()))

            with tarfile.open(tar_path, "r") as tar:
                tar.extractall(tmpdir)

            # Load state dict
            state_path = os.path.join(tmpdir, "state.pkl")
            with open(state_path, "rb") as f:
                raw = f.read()

            state = load_with_rename(raw)
            # Create instance
            cohort = cls.__new__(cls)
            cohort.__dict__.update(state)

            # Support older Cohort objects
            if getattr(cohort, "sources", None) is not None:
                # Passing a filled SourceDict to _load_default_config_data should be a noop
                cohort.sources = cohort._load_default_config_data(cohort.sources)  # type: ignore
            if getattr(cohort, "tmpdir_manager", None) is None:
                # Ignore Final attribute for legacy loading as the object does not exist at load
                cohort.tmpdir_manager = TemporaryDirectoryManager()  # type: ignore[misc]
            if getattr(cohort, "obs_level", None) and isinstance(cohort.obs_level, str):
                cohort.obs_level = ObsLevel(cohort.obs_level)
            old_tmpdir_path = getattr(cohort, "tmpdir_path", None)

            # Ensure tmpdir exists and is valid
            tmpdir_path = getattr(cohort, "_current_tmpdir_path", old_tmpdir_path)
            cohort._current_tmpdir_path = cohort.tmpdir_manager.create_tmpdir(
                tmpdir_path=tmpdir_path, check_existing=True
            )

            # Load parquet files
            obs_path = os.path.join(tmpdir, "obs.parquet")
            cohort._obs = pl.read_parquet(obs_path)

            cohort._obsm = {}
            for file in os.listdir(tmpdir):
                if file.startswith("obsm_") and file.endswith(".parquet"):
                    var_name = file[5:-8]  # remove 'obsm_' and '.parquet'
                    cohort._obsm[var_name] = pl.read_parquet(os.path.join(tmpdir, file))

            # Restore password, which was removed to avoid unencrypted password in the archive
            for source in state["sources"]:
                if "conn_args" in state["sources"][source]:
                    state["sources"][source]["conn_args"]["password"] = password

            # Add flag to indicate this is a old cohort
            # Used in unit tests
            cohort._from_file = True
            logger.info(
                __(
                    "SUCCESS: Loaded cohort from {filename!r} ({n_obsm} dynamic variables, {n_obs} observations [{obs_level}])",
                    filename=filename,
                    n_obsm=len(cohort._obsm),
                    n_obs=len(cohort._obs),
                    obs_level=cohort.obs_level.lower_name,
                )
            )

            # Validate
            cohort._validate_cohort()

            return cohort

    @classmethod
    def _load_corr3(
        cls, filename: str | PathLike[str], password: str | None = None
    ) -> Cohort:
        """Load a cohort from a single compressed .corr3 archive (.tar.zst equivalent).
        Loads cohort.__dict__ from cohort.json, obs from obs.parquet, and obsm from obsm_<var_name>.parquet files
        extracted from the archive.

        Args:
            filename: Path to the .corr3 archive
            password: Password to restore for sources with password field

        Returns:
            Cohort: The loaded cohort
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = os.path.join(tmpdir, "archive.tar")
            dctx = zstd.ZstdDecompressor()
            with open(filename, "rb") as inp, open(tar_path, "wb") as out:
                out.write(dctx.decompress(inp.read()))

            with tarfile.open(tar_path, "r") as tar:
                tar.extractall(tmpdir)

            # Check VERSION file
            with open(os.path.join(tmpdir, "VERSION.txt"), encoding="utf8") as f:
                version = f.read().strip()
                logger.info(
                    __(
                        "Loading cohort from {filename!r} (corr3v{version})",
                        filename=filename,
                        version=version,
                    )
                )
            match str(version):
                case "3":
                    return cls._load_file_v3(tmpdir, password)
                case _:
                    raise ValueError(f"Unsupported version: {version}")

    @classmethod
    def _load_file_v3(
        cls, tmpdir: str | PathLike[str], password: str | None = None
    ) -> Cohort:
        """Load a cohort from a single compressed .corr3 archive (.tar.zst equivalent).
        Loads cohort.__dict__ from cohort.json, obs from obs.parquet, and obsm from obsm_<var_name>.parquet files
        extracted from the archive.
        """
        # Load state dict
        state_path = os.path.join(tmpdir, "cohort.json")
        with open(state_path, encoding="utf8") as f:
            state = json.load(f)

        # Create instance
        cohort = cls.__new__(cls)  # bypass __init__
        cohort.__dict__.update(state)  # update __dict__ with state

        # Support older Cohort objects
        if getattr(cohort, "sources", None) is not None:
            cohort.sources = cohort._load_default_config_data(cohort.sources)  # type: ignore

        # Rebuild the concept resolver from the persisted settings. The API key
        # is never saved, so it has to come from CORR_CONCEPTS_API_KEY here.
        concepts_settings = cohort.__dict__.pop("concepts_settings", None) or {}
        cohort.concept_versions = {
            var_name: _as_concept_version_records(entry)
            for var_name, entry in (
                cohort.__dict__.get("concept_versions") or {}
            ).items()
        }
        cohort._setup_concepts(
            project=concepts_settings.get("project"),
            taxonomy=concepts_settings.get("taxonomy"),
            version=concepts_settings.get("version"),
        )

        if getattr(cohort, "tmpdir_manager", None) is None:
            cohort.tmpdir_manager = TemporaryDirectoryManager()  # type: ignore[misc]
        if getattr(cohort, "obs_level", None) and isinstance(cohort.obs_level, str):
            cohort.obs_level = ObsLevel(cohort.obs_level)
        old_tmpdir_path = getattr(cohort, "tmpdir_path", None)

        # Restore change tracker if exists
        if os.path.exists(os.path.join(tmpdir, "change_tracker.pkl")):
            try:
                with open(os.path.join(tmpdir, "change_tracker.pkl"), "rb") as f:
                    cohort._change_tracker = pickle.load(f)
            except (FileNotFoundError, UnpicklingError) as exc:
                logger.warning(
                    __(
                        "Failed to load change tracker: {exc}. Setting up new change tracker.",
                        exc=exc,
                    )
                )
                cohort._setup_change_tracker()
        else:
            cohort._setup_change_tracker()

        # Ensure tmpdir exists and is valid
        tmpdir_path = getattr(cohort, "_current_tmpdir_path", old_tmpdir_path)
        cohort._current_tmpdir_path = cohort.tmpdir_manager.create_tmpdir(
            tmpdir_path=tmpdir_path, check_existing=True
        )

        # Load parquet files
        obs_path = os.path.join(tmpdir, "obs.parquet")
        cohort._obs = pl.read_parquet(obs_path)

        cohort._obsm = {}
        for file in os.listdir(tmpdir):
            if file.startswith("obsm_") and file.endswith(".parquet"):
                var_name = file[5:-8]  # remove 'obsm_' and '.parquet'
                cohort._obsm[var_name] = pl.read_parquet(os.path.join(tmpdir, file))

        # Restore password, which was removed to avoid unencrypted password in the archive
        for source in state["sources"]:
            if "conn_args" in state["sources"][source]:
                state["sources"][source]["conn_args"]["password"] = password

        # Add flag to indicate this is a old cohort
        # Used in unit tests
        cohort._from_file = True
        logger.info(
            __(
                "SUCCESS: Loaded cohort(obs_level={obs_level}, len(obs)={n_obs}, len(obsm)={n_obsm})",
                obs_level=cohort.obs_level,
                n_obs=len(cohort),
                n_obsm=len(cohort._obsm),
            )
        )

        # Validate
        cohort._validate_cohort()

        return cohort

    # Property methods
    # We use _obs and _obsm to store underlying data so that the interface can be modified to pandas for the legacy version
    @property
    def obs(self) -> pl.DataFrame:
        return self._obs

    @obs.setter
    def obs(self, value: pl.DataFrame | pl.LazyFrame | pd.DataFrame) -> None:
        self._obs = utils.convert_to_polars_df(value)

    @obs.deleter
    def obs(self) -> None:
        warnings.warn("Can't delete Cohort.obs!", UserWarning)

    @property
    def obsm(self) -> ObsmDict:
        return ObsmDict(self._obsm)

    @obsm.setter
    def obsm(
        self,
        value: (
            ObsmDict
            | dict[str, pl.DataFrame]
            | dict[str, pl.LazyFrame]
            | dict[str, pd.DataFrame]
        ),
    ) -> None:
        if isinstance(value, ObsmDict):
            self._obsm = value._data
        else:
            converted_value = {
                var_name: utils.convert_to_polars_df(df)
                for var_name, df in value.items()
            }
            self._obsm = converted_value

    @obsm.deleter
    def obsm(self) -> None:
        warnings.warn("Can't delete Cohort.obsm!", UserWarning)

    @property
    def tmpdir_path(self) -> str:
        self._current_tmpdir_path = self.tmpdir_manager.path
        return self._current_tmpdir_path

    def to_widget(self, *exprs: IntoExpr | Iterable[IntoExpr]) -> ObsWidget:
        return ObsWidget(
            self._obs.select(*exprs) if exprs else self._obs,
            obs_level=self.obs_level.lower_name,
            creation_time=self._data_load_time,
        )

    @property
    def widget(self) -> ObsWidget:
        return self.to_widget()

    def to_search_widget(
        self, include_sources: Iterable[str] | None = None
    ) -> JsonWidget | JsonmWidget:
        # A cohort that never needed the Concepts API has no resolver, and
        # building one here just to render a widget would demand credentials.
        resolver = getattr(self, "_concepts", None)

        var_configs = loader.var_loader.load_raw_variable_configs(
            include_sources=include_sources, resolver=resolver
        )

        if not var_configs:
            return JsonWidget(var_configs)

        if len(var_configs) == 1:
            return JsonWidget(var_configs[list(var_configs.keys())[0]])

        return JsonmWidget(var_configs)

    @property
    def search_widget(self) -> JsonWidget | JsonmWidget:
        return self.to_search_widget(include_sources=self.sources.keys())

    # System methods
    def __repr__(self) -> str:
        return textwrap.dedent(f"""\
            Cohort(
                obs_level = {self.obs_level.lower_name!r},
                sources = {self.sources!r},
                project_vars = {self.project_vars!r},
                load_default_vars = {self._load_default_vars!r},
                logger_args = {self.logger_args!r}
            )
            """.rstrip())

    def __str__(self) -> str:
        return textwrap.dedent(f"""\
            Cohort object
            obs_level: {self.obs_level.lower_name}
            """) + str(self.obs)

    def __len__(self) -> int:
        return len(self._obs)

    def __iter__(self) -> Iterator[pl.Series]:
        return iter(self._obs)

    @overload
    def __getitem__(
        self, item: tuple[SingleIndexSelector, SingleColSelector]
    ) -> Any: ...

    @overload
    def __getitem__(  # type: ignore[overload-overlap]
        self, item: str | tuple[MultiIndexSelector, SingleColSelector]
    ) -> pl.Series: ...

    @overload
    def __getitem__(
        self,
        item: (
            SingleIndexSelector
            | MultiIndexSelector
            | MultiColSelector
            | tuple[SingleIndexSelector, MultiColSelector]
            | tuple[MultiIndexSelector, MultiColSelector]
        ),
    ) -> pl.DataFrame: ...

    def __getitem__(
        self,
        item: (
            SingleIndexSelector
            | SingleColSelector
            | MultiColSelector
            | MultiIndexSelector
            | tuple[SingleIndexSelector, SingleColSelector]
            | tuple[SingleIndexSelector, MultiColSelector]
            | tuple[MultiIndexSelector, SingleColSelector]
            | tuple[MultiIndexSelector, MultiColSelector]
        ),
    ) -> pl.DataFrame | pl.Series | Any:
        try:
            return self._obs[item]
        except ColumnNotFoundError:
            requested_vars = []
            if isinstance(item, str):
                requested_vars = [item]
            elif isinstance(item, (tuple, list)):
                requested_vars = [x for x in item if isinstance(x, str)]

            available_vars = self._obs.columns
            utils.raise_error_with_nearest_matches(
                item=requested_vars,
                available_items=available_vars,
                label="Variable",
                error_cls=VariableNotFoundError,
            )
            requested_vars_label = "Variable" + ("s" if len(requested_vars) > 1 else "")
            joined_requested_vars = ", ".join([f"'{var}'" for var in requested_vars])
            error_msg = f"{requested_vars_label} {joined_requested_vars} not found in cohort.obsm."
            raise VariableNotFoundError(error_msg) from None

    def __setitem__(
        self, item: ArrayLike | str | None, value: ArrayLike | None
    ) -> None:
        self._obs = self._obs.with_columns(pl.Series(item, value))

    def __contains__(self, value: str) -> bool:
        return value in self._obs

    def unnest(
        self,
        column: ColumnNameOrSelector,
        prefix: str = "",
        suffix: str = "",
        renamer: Sequence[str] | Callable[[str], str] | None = None,
    ) -> pl.DataFrame:
        """Unnest a struct column in the obs DataFrame.

        Args:
            column (str): The column to unnest.
            prefix (str): The prefix to add to the unnested column names.
            suffix (str): The suffix to add to the unnested column names.
            renamer (Sequence[str] | Callable[[str], str] | None): The renamer function or list of new names.

        Returns:
            obs (pl.DataFrame): The obs DataFrame with the unnested column.
        """
        return utils.unnest(
            value=self._obs,
            column=column,
            prefix=prefix,
            suffix=suffix,
            renamer=renamer,
        )

    # Other print methods
    def _repr_mimebundle_(self, *args, **kwargs) -> tuple[dict, dict] | None:
        return self.widget._repr_mimebundle_(*args, **kwargs)

    def debug_print(self) -> None:
        """Print debug information about the cohort. Please use this if you are creating a GitHub issue.

        Returns:
            None
        """
        utils.print_cohort_debug_info(self)
        utils.print_debug_info()

    # Utils methods
    def to_stata(
        self,
        df: pl.DataFrame | None = None,
        convert_dates: dict[Hashable, StataDateFormat] | None = None,
        write_index: bool = True,
        to_file: str | PathLike[str] | None = None,
    ) -> pd.DataFrame | None:
        """Convert the cohort to a Stata DataFrame. You may use cohort.stata to access the dataframe directly.
        Note that you need to save it to a top-level variable to access it via %%stata.

        Args:
            df (pd.DataFrame | None): The DataFrame to be converted to Stata format. Will default to
                the obs DataFrame if unspecified (default: None)

            convert_dates (dict[Hashable, StataDateFormat]): Dictionary of columns to convert to Stata date format.

            write_index (bool): Whether to write the index as a column.

            to_file (str | PathLike[str] | None): Path to save as .dta file. If left unspecified, the DataFrame will not be saved.

        Returns:
            pd.DataFrame: A Pandas Dataframe compatible with Stata if `to_file` is None.

        """
        stata_df = utils.convert_to_pandas_df(self._obs if df is None else df)
        return utils.convert_to_stata(
            stata_df,
            convert_dates=convert_dates,
            write_index=write_index,
            to_file=to_file,
        )

    @property
    def stata(self) -> pd.DataFrame | None:
        return self.to_stata()

    def to_tableone(
        self,
        df: pl.DataFrame | None = None,
        ignore_cols: list[str] | str | None = None,
        filter_query: str | None = None,
        replace_booleans: tuple[str, str] | None = ("Yes", "No"),
        # TableOne kwargs
        display_all: bool = True,
        groupby: str | None = None,
        normal_cols: list[str] | None = None,
        overall: bool | None = None,
        order: dict[str, list[str]] | None = None,
        pval: bool = False,
        **kwargs,
    ) -> TableOne:
        """Create a `TableOne <https://tableone.readthedocs.io/en/latest/index.html>`_ object for the cohort.

        Args:
            df (pl.DataFrame): The DataFrame to be converted to Stata format. Will default to
                the obs DataFrame if unspecified (default: None)
            ignore_cols (list | str | None): Column(s) to ignore.
            filter_query (str | None): Filter to apply to the data.
            replace_booleans (tuple[str, str] | None): Replace booleans with the given strings.
            display_all (bool): Whether to display all columns.
            groupby (str | None): Column to group by.
            normal_cols (list[str] | None): Columns to treat as normally distributed.
            overall (bool): Whether to add an “overall” column to the table. If left unspecified the overall column will be dropped if groupby is specified.
            order (dict[str, list[str]] | None): Order of categorical columns.
            pval (bool): Whether to calculate p-values.
            **kwargs: Additional arguments to pass to TableOne.

        Returns:
            TableOne: A TableOne object.


        Examples:
            >>> tableone = cohort.tableone()
            >>> print(tableone)
            >>> tableone.to_csv("tableone.csv")

            >>> tableone = cohort.tableone(groupby="sex", pval=False)
            >>> print(tableone)
            >>> tableone.to_csv("tableone_sex.csv")

        """
        return utils.convert_to_tableone(
            self._obs if df is None else df,
            ignore_cols=ignore_cols,
            filter_query=filter_query,
            replace_booleans=replace_booleans,
            # TableOne kwargs
            display_all=display_all,
            groupby=groupby,
            normal_cols=normal_cols,
            overall=overall,
            order=order,
            pval=pval,
            **kwargs,
        )

    @property
    def tableone(self) -> TableOne:
        return self.to_tableone()

    def to_figureone(self) -> Digraph:
        return self._change_tracker.create_flowchart()

    @property
    def figureone(self) -> Digraph:
        return self.to_figureone()
