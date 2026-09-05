from __future__ import annotations

import os
import pickle
from abc import ABC, abstractmethod

import pandas as pd
import polars as pl

from corr_vars import __, logger, utils
from corr_vars.core.variable import FreeDaysVariable
from corr_vars.core.variable import Variable as BaseVariable
from corr_vars.definitions.constants import UNBOUNDED_TMAX, UNBOUNDED_TMIN
from corr_vars.sources.local_datasource import helpers
from corr_vars.utils.base import column_selector, row_length
from corr_vars.utils.frames import time_difference
from corr_vars.utils.time import TimeAnchor, TimeWindow

from corr_vars.definitions.typing import ObsLevel, VariableProtocol
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from corr_vars.core import Cohort
    from corr_vars.definitions import (
        CleaningDict,
        RequirementsDict,
        RequirementsIterable,
        TimeAnchorColumn,
        VariableCallable,
    )
    from corr_vars.sources.var_loader import MultiSourceVariable

    from corr_vars.definitions.typing import ObsLevelName


def _n_rows(data: pl.DataFrame | None) -> int:
    """Row count of ``data``, or ``0`` if it has not been extracted yet."""
    return 0 if data is None else len(data)


def VariableLoader(*args, **kwargs) -> BaseVariable:
    """Factory function to create the correct variable type based on the arguments."""
    # Legacy naming error
    if kwargs.get("variable"):
        kwargs["base_var"] = kwargs.pop("variable")

    try:
        var_type = kwargs.pop("type")

        # Overwrite dynamic attribute
        if var_type in ["native_dynamic", "derived_dynamic"]:
            kwargs["dynamic"] = True
    except KeyError:
        logger.error("local_datasource VariableLoader requires the 'type' field.")

    if var_type == "complex":
        return ComplexVariable.from_time_window(*args, **kwargs)
    if var_type == "native_dynamic":
        return NativeDynamic.from_time_window(*args, **kwargs)
    if var_type == "derived_dynamic":
        return DerivedDynamic.from_time_window(*args, **kwargs)
    if var_type == "native_static":
        return NativeStatic.from_time_window(*args, **kwargs)
    if var_type == "derived_static":
        return DerivedStatic.from_time_window(*args, **kwargs)
    if var_type == "free_days":
        return FreeDaysVariable.from_time_window(*args, **kwargs)
    raise KeyError(f"Type {var_type} does not exist.")


class _ScreenedRequirements:
    """Propagate a variable's ``screened_obs_level`` down to its requirements.

    A source screens queries at the coarse key that exists in its store
    (``patient`` / ``hospital_stay``) and joins the fine-grained observation id back
    into the frame afterwards — the fine id is not in the DB. A *derived* concept
    therefore has to pass its screening on to the variables it depends on, so the
    widening reaches the leaf variable that actually hits the DB.

    The mechanism reuses the existing requirement-override channel: seeding
    ``screened_obs_level`` into each requirement dict makes
    :meth:`Variable._get_required_vars` forward it as a per-load override. Because the
    dependency it loads is itself a screened variable, propagation continues
    recursively without either core or this mixin knowing the depth of the tree.

    Only an **explicitly chosen** screening propagates (``screened_obs_level`` is not
    ``None``); an unset screening seeds nothing, so each dependency keeps its own
    default. A requirement that sets its own ``screened_obs_level`` is left untouched.
    """

    screened_obs_level: ObsLevel | Literal["patient", "hospital_stay"] | None
    requirements: dict[str, RequirementsDict]

    def _screening_override_name(self) -> ObsLevelName | None:
        level = self.screened_obs_level
        if level is None:
            return None
        return level.lower_name if isinstance(level, ObsLevel) else level

    def _get_required_vars(self, cohort: Cohort) -> None:
        name = self._screening_override_name()
        if name is not None:
            # Rebuild with fresh requirement dicts rather than mutating in place: the
            # requirement dicts can be shared with the loaded vars.json config, so an
            # in-place setdefault would leak the screening across loads. A requirement
            # that set its own screening is preserved.
            self.requirements = {
                alias: {
                    **req,
                    "screened_obs_level": req.get("screened_obs_level", name),
                }
                for alias, req in self.requirements.items()
            }
        super()._get_required_vars(cohort)  # type: ignore[misc]


class BaseDynamic(_ScreenedRequirements, BaseVariable, ABC):
    """Abstract base for all dynamic variables of this source.

    Provides the shared extraction pipeline: optimised obs-level selection,
    caching, attribute building, time-filtering, and cleaning.  Subclasses
    implement :meth:`_extract_from_db` to supply the raw data.

    Args:
        var_name: Variable name.
        dynamic: Whether the variable is dynamic (time-series).
        requires: Dependency variables loaded before extraction.
        tmin: Minimum time anchor.
        tmax: Maximum time anchor.
        py: Optional transformation function applied after raw extraction.
        py_ready_polars: Whether ``py`` accepts and returns polars DataFrames.
        cleaning: Value-range cleaning rules.
        allow_caching: Whether to cache raw data in the cohort's tmpdir.
        screened_obs_level: Default observation level used to scope DB queries
            when the time window is not bounded by hospital-stay bounds.
    """

    def __init__(
        self,
        var_name: str,
        dynamic: bool,
        requires: RequirementsIterable = [],
        tmin: TimeAnchorColumn | TimeAnchor | None = None,
        tmax: TimeAnchorColumn | TimeAnchor | None = None,
        py: VariableCallable | None = None,
        py_ready_polars: bool = False,
        cleaning: CleaningDict | None = None,
        allow_caching: bool = True,
        screened_obs_level: Literal["patient", "hospital_stay"] | None = None,
    ) -> None:
        super().__init__(
            var_name=var_name,
            dynamic=dynamic,
            requires=requires,
            tmin=tmin,
            tmax=tmax,
            py=py,
            py_ready_polars=py_ready_polars,
            cleaning=cleaning,
        )
        self.allow_caching = allow_caching
        # None means "no explicit screening": the query coalesces to HOSPITAL_STAY
        # (see _optimise_screened_obs_level) but nothing is propagated to requirements.
        self.screened_obs_level: (
            Literal[ObsLevel.HOSPITAL_STAY, ObsLevel.PATIENT] | None
        ) = (
            ObsLevel(screened_obs_level)  # type: ignore[call-arg]
            if screened_obs_level is not None
            else None
        )

    @abstractmethod
    def extract_from_db(
        self,
        cohort: Cohort,
        id_column: Literal["case_id", "patient_id"],
    ) -> pl.DataFrame:
        """Return raw variable data; called by :meth:`extract` on a cache miss.

        The returned DataFrame must **not** yet have time-window columns or
        cleaning applied — those are handled by the shared pipeline.
        """
        ...

    def _optimise_screened_obs_level(
        self, cohort: Cohort
    ) -> Literal[ObsLevel.HOSPITAL_STAY, ObsLevel.PATIENT]:
        # Ignore screened_obs_level and used fixed value for ObsLevel.PATIENT
        if cohort.obs_level == ObsLevel.PATIENT:
            logger.debug(
                __(
                    "[{var_name}] cohort is patient-level -> screening DB query at PATIENT",
                    var_name=self.var_name,
                )
            )
            return ObsLevel.PATIENT

        # Optimise search if bounded by smallest searchable level (hospital stay)
        bounded_by_hospital_stay_level = utils.bounded_by_obs_level(
            time_window=self.time_window,
            cohort=cohort,
            obs_level=ObsLevel.HOSPITAL_STAY,
        )
        # No explicit screening (None) falls back to a hospital-stay query.
        fallback = self.screened_obs_level or ObsLevel.HOSPITAL_STAY
        screened = (
            ObsLevel.HOSPITAL_STAY if bounded_by_hospital_stay_level else fallback
        )
        logger.debug(
            __(
                "[{var_name}] time_window bounded by hospital-stay bounds: {bounded} "
                "-> screening DB query at {screened} "
                "(unbounded fallback screened_obs_level={fallback})",
                var_name=self.var_name,
                bounded=bounded_by_hospital_stay_level,
                screened=screened.name,
                fallback=fallback.name,
            )
        )
        return screened

    def _clean_data(self) -> None:
        if self.data is None:
            return

        self.data = self.data.with_columns(
            (pl.selectors.date() | pl.selectors.datetime()).clip(
                UNBOUNDED_TMIN, UNBOUNDED_TMAX
            )
        )

        if self.dynamic:
            self.data = utils.build_attributes(self.data)

    def extract(self, cohort: Cohort) -> pl.DataFrame:
        narrowed_obs_level = self._optimise_screened_obs_level(cohort)
        cache_and_join_key = narrowed_obs_level.primary_key
        logger.debug(
            __(
                "[{var_name}] extraction pipeline start: dynamic={dynamic}, "
                "cache/join key={key}, time_window=[{tmin} .. {tmax}]{unbounded}",
                var_name=self.var_name,
                dynamic=self.dynamic,
                key=cache_and_join_key,
                tmin=lambda: repr(self.time_window.tmin),
                tmax=lambda: repr(self.time_window.tmax),
                unbounded=" (literal/unbounded)" if self.time_window.is_literal else "",
            )
        )

        self.data = self._get_from_cache(cohort, cache_key=cache_and_join_key)

        if self.data is None:
            logger.debug(
                __(
                    "[{var_name}] cache miss -> extracting from DB",
                    var_name=self.var_name,
                )
            )
            self._get_required_vars(cohort)
            self.data = self.extract_from_db(cohort, id_column=cache_and_join_key)  # type: ignore[arg-type]
            logger.debug(
                __(
                    "[{var_name}] extract_from_db returned {rows} rows",
                    var_name=self.var_name,
                    rows=lambda: _n_rows(self.data),
                )
            )
            self._clean_data()
            if self._call_var_function(cohort):  # Clean again afterwards
                logger.debug(
                    __(
                        "[{var_name}] py() transform produced {rows} rows",
                        var_name=self.var_name,
                        rows=lambda: _n_rows(self.data),
                    )
                )

            if self.data is None:
                raise ValueError("Variable extraction resulted in no data.")

            self._clean_data()
            self._write_to_cache(cohort, cache_key=cache_and_join_key)
        else:
            logger.debug(
                __(
                    "[{var_name}] cache hit: {rows} rows",
                    var_name=self.var_name,
                    rows=lambda: _n_rows(self.data),
                )
            )

        if self.dynamic:
            self._add_time_window(cohort, join_key=cache_and_join_key)
            after_window = len(self.data)
            self._timefilter()
            after_filter = len(self.data)
            logger.debug(
                __(
                    "[{var_name}] time-filter dropped {dropped} of {before} rows "
                    "(window=[{tmin} .. {tmax}]) -> {after} rows remain",
                    var_name=self.var_name,
                    dropped=after_window - after_filter,
                    before=after_window,
                    after=after_filter,
                    tmin=lambda: repr(self.time_window.tmin),
                    tmax=lambda: repr(self.time_window.tmax),
                )
            )
            self._apply_cleaning()
            logger.debug(
                __(
                    "[{var_name}] value-cleaning dropped {dropped} of {before} rows "
                    "-> {after} rows remain",
                    var_name=self.var_name,
                    dropped=after_filter - len(self.data),
                    before=after_filter,
                    after=lambda: _n_rows(self.data),
                )
            )

        logger.debug(
            __(
                "[{var_name}] extraction pipeline end: {rows} rows",
                var_name=self.var_name,
                rows=lambda: _n_rows(self.data),
            )
        )
        return self.data

    @property
    def _cache_name(self) -> str:
        """File stem the extraction cache is filed under.

        Two variables from the same source must not serve each other's parquet,
        and neither may two contributors sharing a var_name — the members of a
        concept group — so the stem carries both identities.
        """
        if self.contributor_key == self.var_name:
            return self.var_name
        return f"{self.var_name}@{self.contributor_key}"

    def _get_from_cache(self, cohort: Cohort, cache_key: str) -> pl.DataFrame | None:
        load_path = cohort.tmpdir_manager.create_tmpdir_variable_path(
            var_name=self._cache_name, extension=None
        )

        # Before loading, ensure that cacheinfo exists and all case_ids from cohort.obs are present
        if not (
            self.allow_caching
            and os.path.exists(f"{load_path}:{cache_key}.parquet")
            and os.path.exists(f"{load_path}:{cache_key}.cacheinfo")
        ):
            return None

        with open(f"{load_path}:{cache_key}.cacheinfo", "rb") as f:
            cached_ids = pickle.load(f)

        load_ids = set(cohort._obs.select(cache_key).unique().to_series().to_list())

        # Return none if cohort.obs contains uncached load_ids
        if not load_ids.issubset(cached_ids):
            return None

        self.data = pl.read_parquet(f"{load_path}:{cache_key}.parquet")
        logger.info(
            __(
                "Loaded {var_name} from tmpdir ({rows} rows)",
                var_name=self.var_name,
                rows=len(self.data),
            )
        )
        if cache_key in self.data.columns:
            self.data = self.data.filter(pl.col(cache_key).is_in(load_ids))
            logger.info(
                __("Found {rows} rows matching cohort.obs", rows=len(self.data))
            )

        return self.data

    def _write_to_cache(self, cohort: Cohort, cache_key: str) -> None:
        if not self.allow_caching or self.data is None:
            return

        save_path = cohort.tmpdir_manager.create_tmpdir_variable_path(
            var_name=f"{self._cache_name}:{cache_key}", extension=None
        )
        self.data.write_parquet(f"{save_path}.parquet")
        saved_ids = cohort._obs.select(cache_key).unique().to_series().to_list()

        if os.path.exists(f"{save_path}.cacheinfo"):
            os.remove(f"{save_path}.cacheinfo")

        # Write cacheinfo
        with open(f"{save_path}.cacheinfo", "wb") as f:
            pickle.dump(saved_ids, f)

        logger.info(
            __(
                "Wrote {var_name} to tmpdir ({rows} rows)",
                var_name=self.var_name,
                rows=len(self.data),
            )
        )


class NativeExtractor:
    """Raw extractor satisfying VariableProtocol.

    Fetches a variable's rows from the source's backing store. No caching,
    time-filtering, or cleaning — those are handled by the wrapping
    :class:`NativeDynamic` pipeline, which is what makes this class worth
    separating: an extractor can be swapped without touching that pipeline.

    :meth:`extract` is deliberately **not implemented here**. A deployment binds
    this source to its own store — a warehouse table, a set of parquet files, an
    HTTP service — and the query that reaches it is deployment-specific, so it
    is not part of the published package. Subclass this and implement
    :meth:`extract`, then point :class:`NativeDynamic` at the subclass through
    :attr:`NativeDynamic.extractor_class`.

    Args:
        var_name: Variable name.
        table: Table name in the backing store.
        where: Row filter, in whatever dialect the store speaks.
        columns: Column alias/cast mapping for the query.
        value_dtype: Polars dtype name the ``value`` column is cast to.
        time_window: Time window (stored to satisfy the protocol).
    """

    def __init__(
        self,
        var_name: str,
        *,
        time_window: TimeWindow | None = None,
        table: str | None = None,
        where: str | None = None,
        columns: dict[str, str | dict[str, str]] | None = None,
        value_dtype: str | None = None,
    ) -> None:
        self.var_name = var_name
        self.dynamic = True
        self.time_window = time_window or TimeWindow()
        self.table = table
        self.where = where
        self.columns = columns
        self.value_dtype: str | None = value_dtype

    def extract(
        self,
        cohort: Cohort,
        id_column: Literal["case_id", "patient_id"],
    ) -> pl.DataFrame:
        """Fetch this variable's raw rows, restricted to the cohort's ids.

        Args:
            cohort (Cohort): Cohort whose ids the extract is restricted to.
            id_column (Literal["case_id", "patient_id"]): Key the store is
                queried and joined on.

        Returns:
            pl.DataFrame: One row per record, carrying at least `id_column`,
            ``recordtime`` and ``value``.

        Raises:
            NotImplementedError: Always. See the class docstring — the query
                that reaches a deployment's store is not part of this package.
        """
        raise NotImplementedError(
            f"{type(self).__name__} ships no query for variable "
            f"{self.var_name!r}: local_datasource is a source skeleton, and the "
            "query that reaches a deployment's backing store is supplied by that "
            "deployment. Subclass NativeExtractor, implement extract(), and set "
            "NativeDynamic.extractor_class to the subclass. Variables that "
            "compute from data already in the cohort (native_static, "
            "derived_static, derived_dynamic, complex) need none of this."
        )


class NativeDynamic(BaseDynamic):
    """Store-backed dynamic variable with the full :class:`BaseDynamic` pipeline.

    Wraps a :class:`NativeExtractor` to provide caching, time-filtering,
    cleaning, and optional ``py`` post-processing.

    The extractor is the only part that touches a backing store, and the
    published :class:`NativeExtractor` raises :class:`NotImplementedError`. Bind
    this source to a store by subclassing it and setting
    :attr:`extractor_class`:

    .. code-block:: python

        class MyExtractor(NativeExtractor):
            def extract(self, cohort, id_column):
                ...  # query your store, return one row per record

        NativeDynamic.extractor_class = MyExtractor

    Args:
        table: Table name in the backing store.
        where: Row filter appended to the generated query.
        value_dtype: Optional polars dtype name applied to the ``value`` column
            after extraction (e.g. ``"Float64"``).
        columns: Column alias/cast mapping for the query.

    All other args are forwarded to :class:`BaseDynamic`.

    Note:
        Only use this class for defining custom variables.  For CORR-specified
        variables, use ``cohort.add_variable()`` directly.
    """

    #: Extractor the variable builds to reach the backing store. Rebind it on
    #: this class (or on a subclass) to plug in a deployment's own query.
    extractor_class: type[NativeExtractor] = NativeExtractor

    def __init__(
        self,
        var_name: str,
        dynamic: bool,
        requires: RequirementsIterable = [],
        tmin: TimeAnchorColumn | TimeAnchor | None = None,
        tmax: TimeAnchorColumn | TimeAnchor | None = None,
        py: VariableCallable | None = None,
        py_ready_polars: bool = False,
        cleaning: CleaningDict | None = None,
        allow_caching: bool = True,
        screened_obs_level: Literal["patient", "hospital_stay"] | None = None,
        table: str | None = None,
        where: str | None = None,
        value_dtype: str | None = None,
        columns: dict[str, str | dict[str, str]] | None = None,
    ) -> None:
        super().__init__(
            var_name=var_name,
            dynamic=dynamic,
            requires=requires,
            tmin=tmin,
            tmax=tmax,
            py=py,
            py_ready_polars=py_ready_polars,
            cleaning=cleaning,
            allow_caching=allow_caching,
            screened_obs_level=screened_obs_level,
        )
        # self.value_dtype = value_dtype
        self.extractor = self.extractor_class(
            var_name=var_name,
            time_window=self.time_window,
            table=table,
            where=where,
            columns=columns,
            value_dtype=value_dtype,
        )

    def extract_from_db(
        self,
        cohort: Cohort,
        id_column: Literal["case_id", "patient_id"],
    ) -> pl.DataFrame:
        return self.extractor.extract(cohort, id_column)


class ComplexVariable(BaseDynamic):
    """Variable extracted entirely by a custom ``py()`` function.

    No database query is issued; the variable function receives ``required_vars``
    and must return a ``pl.DataFrame``.  Time-filtering is applied only when
    the function adds ``case_tmin``/``case_tmax`` columns to the result.

    Args:
        var_name: Name of the variable.
        dynamic: Whether the variable is dynamic.
        requires: Variables loaded and passed to the function.
        tmin: Minimum time anchor.
        tmax: Maximum time anchor.
        py: Variable function that produces the data.
        py_ready_polars: Whether ``py`` accepts and returns polars DataFrames.
        cleaning: Dictionary of cleaning rules.
        allow_caching: Whether to allow caching.
        screened_obs_level: Observation level used for scoping the cache key.
    """

    def __init__(
        self,
        var_name: str,
        dynamic: bool,
        requires: RequirementsIterable = [],
        tmin: TimeAnchorColumn | TimeAnchor | None = None,
        tmax: TimeAnchorColumn | TimeAnchor | None = None,
        py: VariableCallable | None = None,
        py_ready_polars: bool = False,
        cleaning: CleaningDict | None = None,
        allow_caching: bool = True,
        screened_obs_level: Literal["patient", "hospital_stay"] | None = None,
    ) -> None:
        super().__init__(
            var_name=var_name,
            dynamic=dynamic,
            requires=requires,
            tmin=tmin,
            tmax=tmax,
            py=py,
            py_ready_polars=py_ready_polars,
            cleaning=cleaning,
            allow_caching=allow_caching,
            screened_obs_level=screened_obs_level,
        )
        self._timefilter_always = False

    def extract_from_db(
        self,
        cohort: Cohort,
        id_column: Literal["case_id", "patient_id"],
    ) -> pl.DataFrame:
        raise NotImplementedError(
            f"ComplexVariable '{self.var_name}' does not extract from the database. "
            "Data must be produced inside the variable function."
        )

    def extract(self, cohort: Cohort) -> pl.DataFrame:
        narrowed_obs_level = self._optimise_screened_obs_level(cohort)
        cache_and_join_key = narrowed_obs_level.primary_key

        data = self._get_from_cache(cohort, cache_key=cache_and_join_key)

        if (
            data is not None
            and "case_tmin" in data.columns
            and "case_tmax" in data.columns
        ):
            # For complex variables that have precomputed case_tmin and case_tmax
            # The timefilter needs to be reapplied - we will set timefilter_always to True
            # For this, we need to drop the case_tmin, case_tmax non cache_key primary keys
            drop_cols = ["case_tmin", "case_tmax"] + (
                [cohort.primary_key] if cohort.primary_key != cache_and_join_key else []
            )
            data = data.drop(drop_cols, strict=False)

            if (
                "case_id" in data.columns
            ):  # TODO Change later after ensuring that cache_key is present in all variables
                self._timefilter_always = True
            else:
                data = None

        self.data = data

        if self.data is None:
            self._get_required_vars(cohort)
            self._clean_data()
            self._call_var_function(cohort)  # Clean again afterwards

            if self.data is None:
                raise ValueError("Variable extraction resulted in no data.")

            self._clean_data()
            self._write_to_cache(cohort, cache_key=cache_and_join_key)

        if self.dynamic:
            # _get_from_cache may set _timefilter_always=True when reloading
            # cached data that previously had case_tmin/case_tmax
            if self._timefilter_always:
                self._add_time_window(cohort, join_key=cache_and_join_key)
            self._timefilter()
            self._apply_cleaning()

        return self.data


class NativeStatic(_ScreenedRequirements, BaseVariable):
    """Aggregated variables represent simple aggregations of dynamic variables.

    Args:
        var_name (str): Name of the variable.
        select (str): Select clause specifying aggregation function and columns.
        base_var (str): Name of the base variable (must be a native_dynamic variable).
        where (str, optional): Optional WHERE clause (in format for polars).
        tmin (str, optional): Minimum time for the extraction.
        tmax (str, optional): Maximum time for the extraction.

    The select argument supports several aggregation functions:

    - ``!first [columns]``: Returns the first row within this case
        >>> "!first value"  # Single column
        >>> "!first value, recordtime"  # Multiple columns

    - ``!last [columns]``: Returns the last row within this case
        >>> "!last value"
        >>> "!last value, recordtime"

    - ``!any``: Returns True if any value exists
        >>> "!any"
        >>> "!any value"

    - ``!sum [column]``: Calculates sum of values
        >>> "!sum value"

    - ``!closest(to_column, timedelta, plusminus) [columns]``: Selects value closest to specified column
        Args:
            to_column: Column to compare "recordtime" against
            timedelta: Time to add to "to_column" for comparison
            plusminus: Allowed time mismatch (can specify different before/after with space)

        >>> "!closest(hospital_admission) value, recordtime"  # Closest to admission
        >>> "!closest(hospital_admission, 0, 2h 3h) value"  # 2h before to 3h after
        >>> "!closest(first_intubation_dtime, 6h, 2h) value"  # 6h after intubation ±2h

    - ``!mean [column]``: Calculates mean value
        >>> "!mean value"

    - ``!median [column]``: Calculates median value
        >>> "!median value"

    - ``!perc(quantile) [column]``: Calculates specified percentile
        >>> "!perc(75) value"  # 75th percentile

    The where argument supports SQL-style boolean expressions. These are evaluated in the context of the base variable by column_selector().
    Where also supports magic commands (starting with !) to filter the data. Supported commands are:

        - ``!isin(column, [values])``: Filters rows where the value in column is in values
        - ``!startswith(column, [values])``: Filters rows where the value in column starts with any of the values
        - ``!endswith(column, [values])``: Filters rows where the value in column ends with any of the values

    """

    base_var: tuple[str, TimeWindow] | VariableProtocol | MultiSourceVariable

    def __init__(
        self,
        var_name: str,
        select: str,
        base_var: str | VariableProtocol | MultiSourceVariable,
        where: str | None = None,
        tmin: TimeAnchorColumn | TimeAnchor | None = None,
        tmax: TimeAnchorColumn | TimeAnchor | None = None,
        cleaning: CleaningDict | None = None,
        screened_obs_level: Literal["patient", "hospital_stay"] | None = None,
    ) -> None:
        super().__init__(
            var_name,
            dynamic=False,
            tmin=tmin,
            tmax=tmax,
            cleaning=cleaning,
        )
        # Propagated to the wrapped base_var in extract() (NativeStatic loads its
        # dependency via base_var rather than the requires mechanism).
        self.screened_obs_level = screened_obs_level
        logger.debug(
            __(
                "NativeStatic: SELECT {select} FROM {base_var} WHERE {where}",
                select=select,
                base_var=(
                    base_var
                    if isinstance(base_var, str)
                    else base_var.__class__.__name__
                ),
                where=where,
            )
        )
        self.select = select
        self.where = where
        if isinstance(base_var, str):
            _var = (base_var, self.time_window)
        else:
            _var = base_var
            _var.time_window = self.time_window
        self.base_var = _var
        self.agg_func, self.agg_params, self.select_cols = helpers.parse_select(
            self.select
        )
        if len(self.select_cols) == 0:
            self.select_cols = ["value"]

    def extract(self, cohort: Cohort) -> pl.DataFrame:
        """Extract the variable. You do not need to call this yourself, as it is called internally when you add the variable to a cohort.
        However, you may call it directly to obtain variable data independently of the cohort. You still need a cohort object for case ids and other metadata.

        Args:
            cohort: Cohort object.

        Returns:
            Extracted variable.

        After extraction, you may also access the data as ``Variable.data``.

        Examples:
            >>> var = NativeStatic(
            ...     var_name="first_sodium_recordtime",
            ...     select="!first recordtime",
            ...     base_var="blood_sodium",
            ...     tmin="hospital_admission",
            ... )
            >>> var.extract(
            ...     cohort
            ... )  # With var.extract(), the data will not be added to the cohort.
            >>> var.data  # You can access the data as a polars dataframe.
        """
        include_sources = self._dependency_sources(cohort)
        # Propagate an explicit screening to the wrapped base variable.
        overrides = (
            {"screened_obs_level": self.screened_obs_level}
            if self.screened_obs_level is not None
            else None
        )
        _var = cohort.load_variable(
            self.base_var, include_sources=include_sources, overrides=overrides
        )
        self.base_data = _var.extract(cohort=cohort)
        self.data = self._extract_aggregation(cohort)
        self._apply_cleaning()
        self._call_var_function(cohort)

        return self.data

    def _extract_aggregation(self, cohort: Cohort) -> pl.DataFrame:
        """Extract static variable using the base variable

        Args:
            cohort (Cohort): Cohort object.

        Returns:
            pl.DataFrame: Extracted variable.
        """
        base_data = self.base_data.lazy()
        obs = cohort._obs.lazy()

        # Sort by recordtime
        base_data = base_data.sort("recordtime")

        # Apply where clause filtering
        if self.where:
            if self.where.startswith("!"):
                where_func, where_params, _ = helpers.parse_select(self.where)
                match where_func:
                    case "isin":
                        base_data = base_data.filter(
                            column_selector(where_params[0]).is_in(where_params[1])
                        )
                    case "startswith":
                        # Create OR expression for multiple prefixes
                        if len(where_params[1]) == 1:
                            base_data = base_data.filter(
                                column_selector(where_params[0]).str.starts_with(
                                    where_params[1][0]
                                )
                            )
                        else:
                            starts_expr = column_selector(
                                where_params[0]
                            ).str.starts_with(where_params[1][0])
                            for prefix in where_params[1][1:]:
                                starts_expr = starts_expr | column_selector(
                                    where_params[0]
                                ).str.starts_with(prefix)
                            base_data = base_data.filter(starts_expr)
                    case "endswith":
                        # Create OR expression for multiple suffixes
                        if len(where_params[1]) == 1:
                            base_data = base_data.filter(
                                column_selector(where_params[0]).str.ends_with(
                                    where_params[1][0]
                                )
                            )
                        else:
                            ends_expr = column_selector(where_params[0]).str.ends_with(
                                where_params[1][0]
                            )
                            for suffix in where_params[1][1:]:
                                ends_expr = ends_expr | column_selector(
                                    where_params[0]
                                ).str.ends_with(suffix)
                            base_data = base_data.filter(ends_expr)
                    case _:
                        raise ValueError(
                            f"Unsupported where function: {where_func}. Supported functions are: isin, startswith, endswith"
                        )
            else:
                # Convert pandas-style where clause to polars expression
                polars_where = pl.sql_expr(self.where)
                base_data = base_data.filter(polars_where)

        # Parse time arguments
        obs = utils.add_time_window_expr(
            obs, time_window=self.time_window, aliases=("tmin", "tmax")
        )

        # Join with observation times
        base_data = base_data.join(
            obs.select([cohort.primary_key, "tmin", "tmax"]),
            on=cohort.primary_key,
            how="left",
        )

        logger.debug(
            __(
                "Base data shape (before filtering): {rows}",
                rows=utils.row_length(base_data),
            )
        )

        # TODO: Use filter_with_time_anchor and remove filled_tmin/tmax_expr from add_time_window_expr
        base_data_time = base_data.filter(
            pl.col("recordtime").is_between("tmin", "tmax")
        )

        logger.debug(
            __(
                "Base data shape (after filtering tmin and tmax): {rows}",
                rows=utils.row_length(base_data),
            )
        )

        params = self.agg_params

        # Perform aggregation using Polars syntax
        match self.agg_func.lower():
            case "first":
                res = base_data_time.group_by(cohort.primary_key).agg(
                    column_selector(col).first() for col in self.select_cols
                )
            case "last":
                res = base_data_time.group_by(cohort.primary_key).agg(
                    column_selector(col).last() for col in self.select_cols
                )
            case "mean":
                res = base_data_time.group_by(cohort.primary_key).agg(
                    column_selector(col).mean() for col in self.select_cols
                )
            case "median":
                res = base_data_time.group_by(cohort.primary_key).agg(
                    column_selector(col).median() for col in self.select_cols
                )
            case "min":
                res = base_data_time.group_by(cohort.primary_key).agg(
                    column_selector(col).min() for col in self.select_cols
                )
            case "max":
                res = base_data_time.group_by(cohort.primary_key).agg(
                    column_selector(col).max() for col in self.select_cols
                )
            case "std":
                res = base_data_time.group_by(cohort.primary_key).agg(
                    column_selector(col).std() for col in self.select_cols
                )
            case "perc":
                quantile = float(params[0]) / 100
                res = base_data_time.group_by(cohort.primary_key).agg(
                    column_selector(col).quantile(quantile) for col in self.select_cols
                )
            case "sum":
                res = base_data_time.group_by(cohort.primary_key).agg(
                    column_selector(col).sum() for col in self.select_cols
                )
            case "count":
                res = base_data_time.group_by(cohort.primary_key).agg(
                    pl.len().alias(self.select_cols[0])
                )
            case "any":
                res = base_data_time.group_by(cohort.primary_key).agg(
                    column_selector(col).is_not_null().any() for col in self.select_cols
                )
                # Right join with all observations to include cases with no data
                res = (
                    obs.select(cohort.primary_key)
                    .unique()
                    .join(res, on=cohort.primary_key, how="left")
                    .with_columns(
                        column_selector(col).fill_null(False)
                        for col in self.select_cols
                    )
                )

            case "sum_interval":
                # Use unfiltered base_data instead of base_data_time here

                if "recordtime_end" not in base_data.columns:
                    raise ValueError(
                        "sum_interval requires 'recordtime_end' column in the base variable data"
                    )

                interval_data = (
                    base_data.with_columns(
                        [
                            # Ensure recordtime_end is not null, use recordtime as fallback
                            pl.when(pl.col("recordtime_end").is_null())
                            .then(pl.col("recordtime"))
                            .otherwise(pl.col("recordtime_end"))
                            .alias("recordtime_end_clean"),
                        ]
                    )
                    .with_columns(
                        [
                            # Ensure interval is valid (end >= start), swap if necessary
                            pl.when(
                                pl.col("recordtime_end_clean") < pl.col("recordtime")
                            )
                            .then(
                                pl.col("recordtime")
                            )  # Use recordtime as both start and end if invalid
                            .otherwise(pl.col("recordtime_end_clean"))
                            .alias("recordtime_end_clean"),
                        ]
                    )
                    .with_columns(
                        [
                            # Calculate overlap start (max of interval start and tmin)
                            pl.max_horizontal(["recordtime", "tmin"]).alias(
                                "overlap_start"
                            ),
                            # Calculate overlap end (min of interval end and tmax)
                            pl.min_horizontal(["recordtime_end_clean", "tmax"]).alias(
                                "overlap_end"
                            ),
                        ]
                    )
                )

                # Filter to only intervals that have some overlap
                interval_data = interval_data.filter(
                    pl.col("overlap_start") <= pl.col("overlap_end")
                )

                # Calculate overlap duration and total interval duration
                interval_data = interval_data.with_columns(
                    [
                        # Overlap duration in seconds
                        time_difference("overlap_end", "overlap_start").alias(
                            "overlap_seconds"
                        ),
                        # Total interval duration in seconds
                        time_difference("recordtime_end_clean", "recordtime").alias(
                            "interval_seconds"
                        ),
                    ]
                )

                # Calculate proportional values for each selected column
                proportional_exprs = []
                for col in self.select_cols:
                    # Handle division by zero: if interval is instantaneous (0 duration), use full value
                    proportional_expr = (
                        pl.when(pl.col("interval_seconds") <= 0)
                        .then(pl.col(col))  # Full value for instantaneous intervals
                        .otherwise(
                            pl.col(col)
                            * pl.col("overlap_seconds")
                            / pl.col("interval_seconds")
                        )
                        .alias(f"{col}_proportional")
                    )
                    proportional_exprs.append(proportional_expr)

                interval_data = interval_data.with_columns(proportional_exprs)

                # Sum the proportional values by primary key
                sum_exprs = [
                    pl.col(f"{col}_proportional").sum().alias(col)
                    for col in self.select_cols
                ]

                res = interval_data.group_by(cohort.primary_key).agg(sum_exprs)

                # Ensure all cases are included, even those with no overlapping intervals
                # Join with all observations to include cases with zero sum
                res = (
                    obs.select(cohort.primary_key)
                    .unique()
                    .join(res, on=cohort.primary_key, how="left")
                    .with_columns(
                        [pl.col(col).fill_null(0.0) for col in self.select_cols]
                    )
                )

                logger.debug(
                    __(
                        "sum_interval processed {rows} interval records",
                        rows=lambda: row_length(interval_data),
                    )
                )

            case "closest":
                to_col = params[0]
                tdelta = params[1] if len(params) > 1 else None
                plusminus = params[2] if len(params) > 2 else "52w"

                if " " in plusminus:
                    pm_before, pm_after = plusminus.split(" ")
                else:
                    pm_before = pm_after = plusminus

                # Convert time deltas to total seconds for tolerance
                pm_before_secs = pd.to_timedelta(pm_before).total_seconds()
                pm_after_secs = pd.to_timedelta(pm_after).total_seconds()
                tdelta_secs = pd.to_timedelta(tdelta).total_seconds() if tdelta else 0.0

                # Create target times dataframe
                target_times = obs.select(
                    cohort.primary_key,
                    column_selector(to_col)
                    .add(pl.duration(seconds=int(tdelta_secs)))
                    .alias("target_time"),
                )

                # Use merge_asof for efficient time-based matching
                # For asymmetric tolerances, we need to handle this differently
                if pm_before_secs == pm_after_secs:
                    # Symmetric tolerance - use merge_asof directly

                    res = target_times.join_asof(
                        base_data_time.select(
                            [cohort.primary_key, "recordtime"] + self.select_cols
                        ),
                        left_on="target_time",
                        right_on="recordtime",
                        by=cohort.primary_key,
                        tolerance=str(int(pm_before_secs)) + "s",
                        strategy="nearest",
                    ).drop("target_time")

                else:
                    # Asymmetric tolerance - filter then use merge_asof
                    # Create time bounds
                    target_times = target_times.with_columns(
                        pl.col("target_time")
                        .sub(pl.duration(seconds=int(pm_before_secs)))
                        .alias("time_min"),
                        pl.col("target_time")
                        .add(pl.duration(seconds=int(pm_after_secs)))
                        .alias("time_max"),
                    )

                    # Filter data within the asymmetric window using a join
                    filtered_data = base_data_time.join(
                        target_times.select(
                            cohort.primary_key, "time_min", "time_max", "target_time"
                        ),
                        on=cohort.primary_key,
                        how="inner",
                    ).filter(
                        pl.col("recordtime").ge(pl.col("time_min"))
                        & pl.col("recordtime").le(pl.col("time_max"))
                    )

                    # Now use merge_asof to find the closest match within the filtered data
                    res = target_times.join_asof(
                        filtered_data.select(
                            [cohort.primary_key, "recordtime"] + self.select_cols
                        ),
                        left_on="target_time",
                        right_on="recordtime",
                        by=cohort.primary_key,
                        strategy="nearest",
                    ).drop("target_time", "time_min", "time_max")

            case _:
                raise ValueError(f"Unsupported aggregation function: {self.agg_func}")

        logger.info(
            __(
                "Extracted {var_name}.\nColumns: {columns}",
                var_name=self.var_name,
                columns=lambda: ", ".join(utils.columns(res)),
            )
        )
        res = res.select([cohort.primary_key] + self.select_cols)

        # Apply column naming
        # TODO: Remove if this is already handled by Cohort._save_variable
        if len(self.select_cols) > 1:
            rename_map = {col: f"{self.var_name}_{col}" for col in self.select_cols}
            res = res.rename(rename_map)
        else:
            res = res.rename({self.select_cols[0]: self.var_name})

        return res.collect()


class DerivedStatic(_ScreenedRequirements, BaseVariable):
    """DerivedStatic: These variables are derivations on existing columns in the cohort.obs dataframe based on the expression argument.

    Args:
        var_name: Name of the variable.
        requires: List of required variables.
        expression: Expression to extract the variable.
        tmin: Minimum time for the extraction.
        tmax: Maximum time for the extraction.

    Note that DerivedStatic variables are executed on the cohort.obs dataframe and must reference existing columns in cohort.obs.

    For DerivedStatic variables, you may either provide an SQL-Like expression (which will be parsed by `column_selector`) or a custom function in variables.py.
    Use expressions where possible, but custom functions if you require more complex logic.

    Examples:
        >>> DerivedStatic(
        ...     var_name="inhospital_death",
        ...     requires=["hospital_discharge", "death_timestamp"],
        ...     expression="hospital_discharge <= death_timestamp",
        ... )

        >>> DerivedStatic(
        ...     var_name="any_va_ecmo_icu",
        ...     requires=["ecmo_va_icu_ops", "ecmo_va_icu"],
        ...     expression=(ecmo_va_icu_ops | ecmo_va_icu),
        ... )
    """

    def __init__(
        self,
        var_name,
        requires: RequirementsIterable = [],
        expression: str | None = None,
        tmin: TimeAnchorColumn | TimeAnchor | None = None,
        tmax: TimeAnchorColumn | TimeAnchor | None = None,
        py: VariableCallable | None = None,
        py_ready_polars: bool = False,
        dynamic: bool = False,
        cleaning: CleaningDict | None = None,
        screened_obs_level: Literal["patient", "hospital_stay"] | None = None,
    ) -> None:
        assert (
            dynamic is False
        ), "DerivedStatic cannot be dynamic, please verify the configuration."

        super().__init__(
            var_name=var_name,
            dynamic=False,
            requires=requires,
            tmin=tmin,
            tmax=tmax,
            py=py,
            py_ready_polars=py_ready_polars,
            cleaning=cleaning,
        )

        # Carried for propagation to requirements (see _ScreenedRequirements).
        self.screened_obs_level = screened_obs_level
        self.expression = expression
        self.computed_expression = False

    def extract(self, cohort: Cohort) -> pl.DataFrame:
        self._get_required_vars(cohort)
        self.data = self._compute_expression(cohort)
        called_var_function = self._call_var_function(cohort)

        assert (
            self.computed_expression or called_var_function
        ), "No expression or variable function found."
        assert (
            self.var_name in self.data.columns
        ), f"DerivedStatic variable {self.var_name} not found in extracted data. ({self.data.columns})"

        # Get all columns that are either the primary key, the variable name itself, or start with the variable name
        cols_to_keep = [
            col
            for col in self.data.columns
            if col == cohort.primary_key
            or col == self.var_name
            or (isinstance(col, str) and col.startswith(f"{self.var_name}_"))
        ]
        self.data = self.data.select(cols_to_keep)
        self._apply_cleaning()

        return self.data

    def _compute_expression(self, cohort: Cohort):
        """Compute the expression using the required variables.

        Args:
            cohort (Cohort): Cohort object.

        Returns:
            pl.DataFrame: Extracted variable. (Returns the original obs dataframe if no expression is provided.)
        """
        obs = cohort._obs.clone()

        for var in self.required_vars.values():
            if not var.dynamic and var.data is not None:
                obs = obs.join(
                    var.data.select([cohort.primary_key, var.var_name]),
                    on=cohort.primary_key,
                    how="left",
                )

        if not self.expression:
            return obs
        expr = pl.sql_expr(self.expression)
        obs = obs.with_columns(expr.alias(self.var_name))
        self.computed_expression = True
        return obs


class DerivedDynamic(_ScreenedRequirements, BaseVariable):
    """Derived dynamic variables are extracted using a custom function.

    Args:
        var_name: Name of the variable.
        requires: List of required variables.
        cleaning: Cleaning parameters ({column_name: {low: int, high: int}})
        tmin: Minimum time for the extraction.
        tmax: Maximum time for the extraction.
        py: Custom function.
        py_ready_polars: Whether the custom function is already prepared for polars. Default is False (input and output are pandas dataframes).


    Examples:
        >>> def var_func(var, cohort):
        ...     # Note that this simplified example only works if blood_pao2_arterial and vent_fio2 are of the same length (which is proabably not the case).
        ...     return var.with_columns(
        ...         (
        ...             var.required_vars["blood_pao2_arterial"]
        ...             / var.required_vars["vent_fio2"]
        ...         ).alias("pf_ratio")
        ...     )

        >>> DerivedDynamic(
        ...     var_name="pf_ratio",
        ...     requires=["blood_pao2_arterial", "vent_fio2"],
        ...     py=var_func,
        ...     py_ready_polars=True,
        ... )


    """

    def __init__(
        self,
        var_name,
        requires: RequirementsIterable,
        cleaning: CleaningDict | None = None,
        tmin: TimeAnchorColumn | TimeAnchor | None = None,
        tmax: TimeAnchorColumn | TimeAnchor | None = None,
        py: VariableCallable | None = None,
        py_ready_polars: bool = False,
        dynamic: bool = True,
        screened_obs_level: Literal["patient", "hospital_stay"] | None = None,
    ) -> None:
        assert (
            dynamic is True
        ), "DerivedDynamic must be dynamic, please verify the configuration."
        super().__init__(
            var_name,
            dynamic=True,
            requires=requires,
            tmin=tmin,
            tmax=tmax,
            py=py,
            py_ready_polars=py_ready_polars,
            cleaning=cleaning,
        )
        # Carried for propagation to requirements (see _ScreenedRequirements); a
        # derived variable does not query the DB itself.
        self.screened_obs_level = screened_obs_level

    def extract(self, cohort: Cohort) -> pl.DataFrame:
        """Extract the variable.

        Args:
            cohort (Cohort): Cohort object.

        Returns:
            pl.DataFrame: Extracted variable.
        """
        self._get_required_vars(cohort)
        self._call_var_function(cohort)
        if self.data is None:
            raise ValueError("Variable extraction resulted in no data.")
        self._apply_cleaning()
        return self.data
