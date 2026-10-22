"""Variable classes of the ``charite_demo`` source.

The aggregation pipeline (caching, screening, time-filtering, cleaning, ``py``
post-processing) is the one :mod:`corr_vars.sources.local_datasource.extract`
ships; this module binds it to the DuckDB-backed mock dataset by implementing the
extractors — :class:`NativeExtractor` for plain table reads and
:class:`Co6HierarchyExtractor` for CoPRA 6 parent/child records.

The classes that need no store (:class:`NativeStatic`, :class:`DerivedStatic`,
:class:`DerivedDynamic`, :class:`ComplexVariable`) are thin subclasses, so a
variable built here is recognised as belonging to ``charite_demo`` and resolves
its dependencies inside this source.

Not ported from the production source: the generated ``medication`` and
``laboratory`` variable types, which depend on curated mapping tables.
"""

from __future__ import annotations

import polars as pl

from corr_vars import __, logger
from corr_vars.core.variable import FreeDaysVariable
from corr_vars.core.variable import Variable as BaseVariable
from corr_vars.sources.charite_demo import helpers
from corr_vars.sources.local_datasource import extract as _base
from corr_vars.utils.base import row_length
from corr_vars.utils.time import TimeWindow

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from corr_vars.core import Cohort
    from corr_vars.definitions import (
        CleaningDict,
        RequirementsIterable,
        TimeAnchorColumn,
        VariableCallable,
    )
    from corr_vars.utils.time import TimeAnchor

#: Variable types of the production source that this source does not implement.
UNSUPPORTED_TYPES: frozenset[str] = frozenset({"medication", "laboratory"})


def VariableLoader(*args, **kwargs) -> BaseVariable:
    """Factory function to create the correct variable type based on the arguments."""
    # Legacy naming error
    if kwargs.get("variable"):
        kwargs["base_var"] = kwargs.pop("variable")

    if "type" not in kwargs:
        raise KeyError("charite_demo VariableLoader requires the 'type' field.")
    var_type = kwargs.pop("type")

    # Overwrite dynamic attribute
    if var_type in ["native_dynamic", "derived_dynamic", "co6_hierarchy"]:
        kwargs["dynamic"] = True

    if var_type == "complex":
        return ComplexVariable.from_time_window(*args, **kwargs)
    if var_type == "native_dynamic":
        return NativeDynamic.from_time_window(*args, **kwargs)
    if var_type == "co6_hierarchy":
        return Co6HierarchyDynamic.from_time_window(*args, **kwargs)
    if var_type == "derived_dynamic":
        return DerivedDynamic.from_time_window(*args, **kwargs)
    if var_type == "native_static":
        return NativeStatic.from_time_window(*args, **kwargs)
    if var_type == "derived_static":
        return DerivedStatic.from_time_window(*args, **kwargs)
    if var_type == "free_days":
        return FreeDaysVariable.from_time_window(*args, **kwargs)
    if var_type in UNSUPPORTED_TYPES:
        raise NotImplementedError(
            f"charite_demo does not implement {var_type!r} variables; they rely "
            "on curated drug/lab mapping tables that are not part of the demo."
        )
    raise KeyError(f"Type {var_type} does not exist.")


class BaseDynamic(_base.BaseDynamic):
    """Abstract base for the store-backed dynamic variables of this source.

    See :class:`corr_vars.sources.local_datasource.extract.BaseDynamic`.
    """


class NativeExtractor(_base.NativeExtractor):
    """Raw table extractor satisfying VariableProtocol.

    Reads one table of the mock dataset, restricted to the cohort's ids through a
    temporary table. The ``where`` and ``columns`` fragments are raw SQL as in
    the production concept definitions; casts in them yield ``NULL`` on failure
    (see :func:`~corr_vars.sources.charite_demo.helpers.try_cast_sql`). No
    caching, time-filtering, or cleaning — those are handled by the wrapping
    :class:`NativeDynamic` pipeline.

    Args:
        var_name: Variable name.
        table: Table name in the dataset (a key of ``mapping/tables.json``).
        where: SQL WHERE clause applied before the columns are aliased.
        columns: Column alias/cast overrides for the table's default mapping.
        value_dtype: Type the ``value`` column is converted to.
        time_window: Time window (stored to satisfy the protocol).
    """

    def extract(
        self,
        cohort: Cohort,
        id_column: Literal["case_id", "patient_id"],
    ) -> pl.DataFrame:
        """Fetch this variable's raw rows, restricted to the cohort's ids.

        Args:
            cohort (Cohort): Cohort whose ids the extract is restricted to.
            id_column (Literal["case_id", "patient_id"]): Key the table is
                queried and joined on.

        Returns:
            pl.DataFrame: One row per record, carrying `id_column`,
            ``recordtime``, ``value`` and the table's optional columns.

        Raises:
            ValueError: If no table is configured.
        """
        if not self.table:
            raise ValueError(f"Table not specified for variable {self.var_name}")

        with helpers.temporary_table(cohort, id_column=id_column) as temp_tbl:
            logger.debug(__("Building query for {var_name}", var_name=self.var_name))
            dbs = helpers.get_databases(cohort)
            conn = helpers.reinit_conn(cohort)

            expr, alias_mapping, datetime_cols = helpers.build_native_extractor_expr(
                table=conn.conn.table(self.table, database=dbs["prepared"]),
                table_info=helpers.get_table_info(self.table),
                id_column=id_column,
                columns=self.columns,
                value_dtype=self.value_dtype,
                where=self.where,
            )
            expr = helpers.filter_by_tmp_table(
                conn,
                expr,
                tmp_tbl=temp_tbl,
                workbench_db=dbs["workbench"],
                id_column=id_column,
            )
            result_df = conn.df_from_expr(
                expr, datetime_cols=datetime_cols, keep_cols=alias_mapping.keys()
            )

        logger.info(__("Extracted {rows} rows", rows=len(result_df)))
        return result_df


class NativeDynamic(_base.NativeDynamic):
    """Table-backed dynamic variable with the full :class:`BaseDynamic` pipeline.

    Wraps a :class:`NativeExtractor` to provide caching, time-filtering,
    cleaning, and optional ``py`` post-processing.

    Args:
        table: Table name in the dataset.
        where: SQL WHERE clause appended to the generated query.
        value_dtype: Optional type applied to the ``value`` column in the query
            (values that do not convert become null).
        columns: Column alias/cast mapping for the query.

    All other args are forwarded to :class:`BaseDynamic`.
    """

    extractor_class: type[_base.NativeExtractor] = NativeExtractor


class ComplexVariable(_base.ComplexVariable):
    """Variable extracted entirely by a custom ``py()`` function.

    See :class:`corr_vars.sources.local_datasource.extract.ComplexVariable`.
    """


class NativeStatic(_base.NativeStatic):
    """Static aggregation over a dynamic variable.

    See :class:`corr_vars.sources.local_datasource.extract.NativeStatic`.
    """


class DerivedStatic(_base.DerivedStatic):
    """Static variable derived from other static variables.

    See :class:`corr_vars.sources.local_datasource.extract.DerivedStatic`.
    """


class DerivedDynamic(_base.DerivedDynamic):
    """Dynamic variable derived from other variables.

    See :class:`corr_vars.sources.local_datasource.extract.DerivedDynamic`.
    """


class Co6HierarchyExtractor:
    """Raw CoPRA 6 hierarchy extractor satisfying VariableProtocol.

    Fetches data by joining child nodes under a parent in the CoPRA 6 hierarchy
    table.  No caching, time-filtering, or cleaning — those are handled by the
    wrapping :class:`Co6HierarchyDynamic` pipeline.

    Args:
        var_name: Variable name.
        parent_name: CoPRA 6 parent node name (e.g. ``"Score_SOFA"``).
        suffixes: Maps ``"_Suffix"`` → output column name.  Must include a
            suffix that resolves to ``"recordtime"``.
        table: Source table. Defaults to ``"it_copra6_hierarchy_v2"``.
        not_strict: Suffixes to LEFT JOIN instead of INNER JOIN.
        time_window: Time window used to scope the SQL query.
    """

    def __init__(
        self,
        var_name: str,
        *,
        parent_name: str,
        suffixes: dict[str, str],
        not_strict: list[str] | None = None,
        table: str = "it_copra6_hierarchy_v2",
        time_window: TimeWindow | None = None,
    ) -> None:
        self.var_name = var_name
        self.dynamic = True
        self.time_window = time_window or TimeWindow()
        self.parent_name = parent_name
        self.suffixes = suffixes
        self.not_strict: list[str] = not_strict or []
        self.table = table

    def extract(
        self,
        cohort: Cohort,
        id_column: Literal["case_id", "patient_id"] = "case_id",
    ) -> pl.DataFrame:
        with helpers.temporary_table(cohort, id_column=id_column) as temp_tbl:
            logger.debug(__("Building query for {var_name}", var_name=self.var_name))
            dbs = helpers.get_databases(cohort)
            conn = helpers.reinit_conn(cohort)

            (
                expr,
                alias_mapping,
                datetime_cols,
            ) = helpers.build_co6hierarchy_extractor_expr(
                table=conn.conn.table(self.table, database=dbs["prepared"]),
                id_column=id_column,
                parent_name=self.parent_name,
                suffixes=self.suffixes,
                not_strict=self.not_strict,
            )
            expr = helpers.filter_by_tmp_table(
                conn,
                expr,
                tmp_tbl=temp_tbl,
                workbench_db=dbs["workbench"],
                id_column=id_column,
            )
            result_df = conn.lf_from_expr(
                expr,
                datetime_cols=datetime_cols,
                keep_cols=[id_column, *alias_mapping],
            )

            # Timestamps stored in c_value are UTC, while everything else
            # (c_var_timestamp) is local time: convert them to local time.
            result_df = result_df.with_columns(
                pl.col(col)
                .dt.replace_time_zone("UTC")
                .dt.convert_time_zone("Europe/Berlin")
                .dt.cast_time_unit("us")
                .dt.replace_time_zone(None)
                for col in alias_mapping
                if col.startswith("recordtime")
                and result_df.collect_schema()[col] == pl.Datetime
            )
            logger.info(__("Extracted {rows} rows", rows=lambda: row_length(result_df)))

        return result_df.collect()


class Co6HierarchyDynamic(_base.BaseDynamic):
    """CoPRA 6 hierarchy variable with the full :class:`BaseDynamic` pipeline.

    Wraps a :class:`Co6HierarchyExtractor` to provide caching, time-filtering,
    cleaning, and optional ``py`` post-processing.

    Args:
        parent_name: CoPRA 6 parent node name.
        suffixes: Maps ``"_Suffix"`` → output column name.
        table: Source table. Defaults to ``"it_copra6_hierarchy_v2"``.
        not_strict: Suffixes to LEFT JOIN instead of INNER JOIN.
        value_dtype: Optional polars dtype name applied to the ``value`` column
            after extraction (e.g. ``"Float64"``, ``"Int64"``).

    All other args are forwarded to :class:`BaseDynamic`.
    """

    def __init__(
        self,
        var_name: str,
        dynamic: bool = True,
        requires: RequirementsIterable = [],
        tmin: TimeAnchorColumn | TimeAnchor | None = None,
        tmax: TimeAnchorColumn | TimeAnchor | None = None,
        py: VariableCallable | None = None,
        py_ready_polars: bool = False,
        cleaning: CleaningDict | None = None,
        allow_caching: bool = True,
        screened_obs_level: Literal["patient", "hospital_stay"] | None = None,
        *,
        parent_name: str,
        suffixes: dict[str, str],
        table: str = "it_copra6_hierarchy_v2",
        not_strict: list[str] | None = None,
        value_dtype: str | None = None,
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
        self.value_dtype = value_dtype
        self.extractor = Co6HierarchyExtractor(
            var_name=var_name,
            parent_name=parent_name,
            suffixes=suffixes,
            table=table,
            not_strict=not_strict,
            time_window=self.time_window,
        )

    def extract_from_db(
        self, cohort: Cohort, id_column: Literal["case_id", "patient_id"]
    ) -> pl.DataFrame:
        raw = self.extractor.extract(cohort, id_column)
        if self.value_dtype is not None:
            dtype = getattr(pl, self.value_dtype)
            raw = raw.with_columns(pl.col("value").cast(dtype, strict=False))
        return raw
