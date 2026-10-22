"""Query helpers shared by the charite_demo loaders, extractors and ``py`` snippets.

A port of the production ``cub_hdp`` helpers from Impala to DuckDB. The query
builders are unchanged ibis expressions; what differs is the connection
(:class:`~corr_vars.sources.charite_demo.connector.DuckDBConnector`, cached per
dataset for the process), the temporary tables (DuckDB ``TEMP`` tables rather
than workbench tables), and casting: Impala returns ``NULL`` for a text value
that does not parse as a number, DuckDB raises, so every cast here is a
``TRY_CAST``.
"""

from __future__ import annotations

import logging
import re
import threading
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import ibis
import polars as pl

from corr_vars import __, logger, utils
from corr_vars.sources.charite_demo.connector import (
    MANIFEST_FILE,
    DuckDBConnector,
    is_url,
    normalize_path,
)
from corr_vars.sources.charite_demo.mapping import TABLES

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, TypedDict, cast

if TYPE_CHECKING:
    from corr_vars.core import Cohort
    from corr_vars.sources.charite_demo.config import SourceDict as DemoSourceDict

    from collections.abc import Iterable, Mapping
    from types import TracebackType

#: Source key this module reads its configuration from.
SOURCE: Final[str] = "charite_demo"

#: DuckDB schema temporary tables live in (``temp`` catalog, ``main`` schema).
WORKBENCH: Final[str] = "temp.main"


# ------------------------
# Source helper
# ------------------------
class SourceDatabases(TypedDict):
    """Database names a query against the charite_demo dataset uses.

    The keys mirror the production source, so code written against it ports
    without renaming.

    Attributes:
        prepared: Schema holding the dataset's tables (``charite_mockdata``).
        workbench: Schema temporary tables are created in (``temp.main``).
    """

    prepared: str
    workbench: str


def get_source_config(cohort_or_config: Cohort | DemoSourceDict) -> DemoSourceDict:
    """Return the charite_demo source configuration.

    Accepts either a :class:`Cohort` (extracts ``sources["charite_demo"]``) or
    an already-resolved source config.

    Args:
        cohort_or_config: A cohort object or a charite_demo source config dict.

    Returns:
        The charite_demo source configuration dictionary.

    Raises:
        ValueError: If *cohort_or_config* is a :class:`Cohort` that has no
            ``charite_demo`` source configured.
    """
    if isinstance(cohort_or_config, dict):
        return cohort_or_config
    config = cohort_or_config.sources.get(SOURCE)
    if not config:
        raise ValueError(
            "charite_demo source not configured in cohort.sources. "
            "Make sure to initialize the cohort with the charite_demo source."
        )
    return config


def get_databases(cohort_or_config: Cohort | DemoSourceDict) -> SourceDatabases:
    """Return the database names a charite_demo query uses.

    ``conn.conn.table(name, database=dbs["prepared"])`` addresses a dataset
    table, ``conn.conn.table(tmp, database=dbs["workbench"])`` a temporary table
    created by :class:`temporary_table`.

    Args:
        cohort_or_config: A cohort object or a charite_demo source config dict.

    Returns:
        A :class:`SourceDatabases` dict with ``prepared`` and ``workbench`` keys.
    """
    config = get_source_config(cohort_or_config)
    return {"prepared": config["database"], "workbench": WORKBENCH}


#: Production-source names of :func:`get_source_config` / :func:`get_databases`,
#: kept so ``py`` snippets ported from ``cub_hdp`` run unchanged. Prefer the new
#: names in new code.
get_cub_hdp_config = get_source_config
get_cub_hdp_database = get_databases

#: Cached connectors by ``(path, database)``, each with the identity of the
#: manifest it was built from (see :func:`_manifest_identity`).
_CONNECTORS: dict[tuple[str, str], tuple[object, DuckDBConnector]] = {}
_CONNECTORS_LOCK = threading.Lock()


def _manifest_identity(path: str) -> object:
    """Identify the manifest a connector for `path` would read.

    A local dataset is identified by its ``manifest.json``'s modification time
    and size, so a folder refreshed by :func:`download` (or by hand) is picked
    up; a URL is assumed to be stable for the life of the process.

    Args:
        path (str): Normalised dataset location.

    Returns:
        object: A comparable token; ``None`` for a URL or a missing manifest.
    """
    if is_url(path):
        return None
    try:
        stat = (Path(path) / MANIFEST_FILE).stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


def reinit_conn(cohort_or_config: Cohort | DemoSourceDict) -> DuckDBConnector:
    """Return the process-wide connector for the configured dataset.

    Unlike the production source, which opens a new Impala session per call,
    this returns one cached :class:`DuckDBConnector` per ``(path, database)``.
    It has to: the views and the temporary tables created by
    :class:`temporary_table` live inside that one in-process DuckDB, and a
    second connection would see neither. The name is kept so ported code reads
    the same. When the ``manifest.json`` of a local dataset has changed since
    the connector was built, the connector is closed and rebuilt.

    Args:
        cohort_or_config: A cohort object or a charite_demo source config dict.

    Returns:
        The connector for the configured ``path`` and ``database``.

    Raises:
        ConnectionError: If the dataset cannot be reached.
        FileNotFoundError: If a local ``path`` holds no dataset.
    """
    config = get_source_config(cohort_or_config)
    key = (normalize_path(config["path"]), config["database"])
    identity = _manifest_identity(key[0])
    with _CONNECTORS_LOCK:
        cached = _CONNECTORS.get(key)
        if cached is not None and cached[0] == identity:
            return cached[1]
        if cached is not None:
            logger.info(
                __("charite_demo dataset at {path} changed; reconnecting", path=key[0])
            )
            cached[1].close()
            del _CONNECTORS[key]
        conn = DuckDBConnector(path=key[0], database=key[1])
        _CONNECTORS[key] = (identity, conn)
        return conn


def clear_connections(path: str | None = None) -> None:
    """Close and forget cached connectors (and with them their temporary tables).

    Args:
        path (str | None): Only forget the connectors of this dataset location;
            every connector when ``None``.
    """
    target = normalize_path(path) if path is not None else None
    with _CONNECTORS_LOCK:
        for key in list(_CONNECTORS):
            if target is None or key[0] == target:
                _CONNECTORS.pop(key)[1].close()


# ------------------------
# TABLE helper
# ------------------------
def get_table_info(table: str) -> dict[str, str]:
    """Get the table info for a given table.

    Args:
        table (str): The table name.

    Returns:
        dict: The table info.
    """
    return TABLES[table].copy()


# ------------------------
# TMP Table
# ------------------------
def create_tmp_table(
    cohort: Cohort,
    prefix: str = "cv_tmp_",
    id_column: Literal["case_id", "patient_id"] = "case_id",
    batch_size: int = 250_000,
) -> str:
    """Create a temporary ID-filter table holding the cohort's ids.

    The table contains a single ``VARCHAR`` column (*id_column*) populated with
    the unique IDs currently in ``cohort._obs``. It is a DuckDB ``TEMP`` table
    in :data:`WORKBENCH`, filled in one statement from the registered id frame.

    Args:
        cohort: Cohort whose observation IDs are written to the table.
        prefix: Name prefix for the generated table (default ``"cv_tmp_"``).
        id_column: ID column to create and populate.
        batch_size: Unused; kept for call compatibility with the production
            source, which uploads ids in batches.

    Returns:
        The generated temporary table name (without database qualifier).
    """
    tmp_table_name = prefix + str(uuid4()).replace("-", "")
    logger.debug(
        __("Creating temporary table {tmp_table_name}", tmp_table_name=tmp_table_name)
    )

    ids = (
        cohort._obs.select(pl.col(id_column).cast(pl.Utf8))
        .unique()
        .drop_nulls()
        .to_arrow()
    )
    conn = reinit_conn(cohort)
    view = f"_{tmp_table_name}_ids"
    with conn._lock:
        raw = conn.conn.con
        raw.register(view, ids)
        try:
            raw.execute(
                f'CREATE TEMP TABLE "{tmp_table_name}" AS '
                f'SELECT CAST("{id_column}" AS VARCHAR) AS "{id_column}" FROM "{view}"'
            )
        finally:
            raw.unregister(view)
    logger.debug(
        __(
            "Inserted {n} ids into {tmp_table_name}",
            n=ids.num_rows,
            tmp_table_name=tmp_table_name,
        )
    )
    return tmp_table_name


def drop_tmp_table(cohort: Cohort, temp_tbl: str) -> None:
    """Drop a temporary table created by :func:`create_tmp_table`.

    Args:
        cohort: Cohort used to resolve the connection.
        temp_tbl: Unqualified table name returned by :func:`create_tmp_table`.
    """
    logger.debug(__("Dropping temporary table {temp_tbl}", temp_tbl=temp_tbl))
    reinit_conn(cohort).execute_dml(f'DROP TABLE IF EXISTS {WORKBENCH}."{temp_tbl}"')


@dataclass
class temporary_table:
    """Context manager that creates and drops a temporary id table.

    On entry the table is populated with the current cohort observation IDs;
    on exit it is unconditionally dropped — even if an exception was raised
    inside the ``with`` block.

    Args:
        cohort: Cohort whose observation IDs populate the table.
        prefix: Name prefix for the generated table (default ``"cv_tmp_"``).
        id_column: ID column to create and populate.

    Example:

    .. code-block:: python

        conn = reinit_conn(cohort)
        dbs = get_databases(cohort)
        with temporary_table(cohort, id_column="case_id") as tbl:
            table = conn.conn.table("some_table", database=dbs["prepared"])
            table = filter_by_tmp_table(
                conn,
                table,
                tmp_tbl=tbl,
                workbench_db=dbs["workbench"],
                id_column="case_id",
            )
            result = conn.lf_from_expr(table)
    """

    cohort: Cohort
    prefix: str = "cv_tmp_"
    id_column: Literal["case_id", "patient_id"] = "case_id"
    batch_size: int = 300_000

    def __enter__(self) -> str:
        """Create the temporary table and return its unqualified name."""
        self.temp_tbl = create_tmp_table(
            cohort=self.cohort,
            prefix=self.prefix,
            id_column=self.id_column,
            batch_size=self.batch_size,
        )
        return self.temp_tbl

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        """Drop the temporary table; always returns ``False`` to propagate exceptions."""
        drop_tmp_table(self.cohort, self.temp_tbl)
        return False


# ------------------------
# Query builder
# ------------------------
def filter_by_tmp_table(
    conn: DuckDBConnector,
    table: ibis.Table,
    *,
    tmp_tbl: str,
    workbench_db: str,
    id_column: str,
) -> ibis.Table:
    """Filter an ibis table to the IDs stored in a temporary table.

    Args:
        conn: Active connector used to reference the temporary table.
        table: Table expression to filter; must contain *id_column*.
        tmp_tbl: Unqualified temporary-table name (see :class:`temporary_table`).
        workbench_db: Database that owns *tmp_tbl* (``get_databases()["workbench"]``).
        id_column: Column present in both *table* and the temporary table.

    Returns:
        *table* restricted to rows whose *id_column* appears in *tmp_tbl*.
    """
    tmp = conn.conn.table(tmp_tbl, database=workbench_db)
    return table.filter(table[id_column].isin(tmp[id_column]))


_CAST_PATTERN = re.compile(r"(?<![\w.])CAST\s*\(", flags=re.IGNORECASE)

#: A quoted span: a string literal (``''`` escapes a quote) or a quoted
#: identifier (``""`` escapes a quote). An unterminated span runs to the end.
_QUOTED_PATTERN = re.compile(r"'(?:[^']|'')*(?:'|$)|\"(?:[^\"]|\"\")*(?:\"|$)")


def try_cast_sql(fragment: str) -> str:
    """Rewrite ``CAST(`` to ``TRY_CAST(`` in a raw SQL fragment.

    Concept definitions were written for Impala, where a cast that fails yields
    ``NULL``. DuckDB's ``CAST`` raises instead, so a single malformed value in a
    free-text column would abort the whole extraction; ``TRY_CAST`` restores the
    Impala behaviour. ``TRY_CAST`` itself, identifiers merely ending in
    ``cast`` and text inside string literals (``'...'``) or quoted identifiers
    (``"..."``) are left alone.

    Args:
        fragment: Raw SQL (a ``WHERE`` body or a projected expression).

    Returns:
        The fragment with every ``CAST(`` call turned into ``TRY_CAST(``.
    """
    out: list[str] = []
    pos = 0
    for quoted in _QUOTED_PATTERN.finditer(fragment):
        out.append(_CAST_PATTERN.sub("TRY_CAST(", fragment[pos : quoted.start()]))
        out.append(quoted.group())
        pos = quoted.end()
    out.append(_CAST_PATTERN.sub("TRY_CAST(", fragment[pos:]))
    return "".join(out)


def apply_sql_where(table: ibis.Table, where: str | None) -> ibis.Table:
    """Apply a raw SQL ``WHERE`` fragment to an ibis table expression.

    Concept filters are freeform SQL strings referencing the source table's
    original columns, so they cannot be expressed as native ibis predicates
    without parsing.  This wraps the fragment via :meth:`ibis.Table.sql`,
    keeping the result a composable ibis table.  Apply it before any column
    aliasing, while the original column names are still valid. Casts in the
    fragment are made ``NULL``-on-failure (see :func:`try_cast_sql`).

    Args:
        table: Table expression whose original columns *where* references.
        where: SQL boolean expression (everything after ``WHERE``); ``None`` or
            empty returns *table* unchanged.

    Returns:
        *table* restricted to rows matching *where*.
    """
    if not where:
        return table
    return table.alias("_src").sql(f"SELECT * FROM _src WHERE {try_cast_sql(where)}")


def apply_sql_select(
    table: ibis.Table,
    projections: Mapping[str, str],
    where: str | None = None,
) -> ibis.Table:
    """Project raw SQL fragments onto output aliases.

    The values of ``tables.json`` / concept column mappings are freeform SQL,
    not merely column names — a plain column (``c_falnr``), a string literal
    (``'ward_id'``), or an expression (``CAST(c_systolic AS float)``).  They must
    therefore be interpolated into a ``SELECT`` rather than looked up with
    ``table[name]``, which only resolves existing column names.

    *where* is applied in the same statement: SQL evaluates ``WHERE`` before the
    ``SELECT`` aliases, so the fragment still refers to the source column names.
    Casts in either are made ``NULL``-on-failure (see :func:`try_cast_sql`).

    Args:
        table: Source table expression whose original columns the fragments
            reference.
        projections: Mapping of output alias → raw SQL fragment.
        where: Optional SQL boolean expression (everything after ``WHERE``).

    Returns:
        *table* projected to *projections*, restricted by *where*.

    Raises:
        ValueError: If *projections* is empty.
    """
    if not projections:
        raise ValueError("apply_sql_select requires at least one projection")

    select_list = ", ".join(
        f'{try_cast_sql(expr)} AS "{alias}"' for alias, expr in projections.items()
    )
    query = f"SELECT {select_list} FROM _src"
    if where:
        query += f" WHERE {try_cast_sql(where)}"
    return table.alias("_src").sql(query)


def build_native_extractor_expr(
    table: ibis.Table,
    table_info: dict[str, str],
    id_column: Literal["case_id", "patient_id"],
    columns: dict[str, str | dict[str, str]] | None = None,
    value_dtype: str | None = None,
    where: str | None = None,
) -> tuple[ibis.Table, dict[str, str], list[str]]:
    """Select and alias a native dynamic variable from a source table.

    Pure ``ibis.Table -> ibis.Table`` transform: applies the optional raw WHERE
    fragment, then aliases the standard ``(id_column, recordtime, value, …)``
    output columns.  Row-level filtering by cohort IDs is left to the caller
    via :func:`filter_by_tmp_table` on the returned table.

    Args:
        table: Source table expression (e.g. ``conn.conn.table(name, database=...)``).
        table_info: Logical→physical column mapping for *table* (from
            :func:`get_table_info`).  Mutated in place by *columns*.
        id_column: Primary-key column (``"case_id"`` or ``"patient_id"``).
        columns: Optional column alias/cast overrides.
        value_dtype: Optional polars/SQL type for the ``value`` column; values
            that do not convert become null.
        where: Optional raw SQL WHERE body.  Applied before column aliasing so
            the original column names it references remain valid.

    Returns:
        A ``(expr, alias_mapping, datetime_cols)`` triple where *expr* is an
        ibis table expression ready for :meth:`DuckDBConnector.lf_from_expr`.
    """
    logger.debug("Table info:")
    utils.log_dict(logger=logger, dictionary=table_info, level=logging.DEBUG)

    # Custom column and attributes override
    if columns:
        logger.debug("Custom columns:")
        utils.log_dict(logger=logger, dictionary=columns, level=logging.DEBUG)
        for col, override in columns.items():
            if col == "attributes":
                if isinstance(override, dict):
                    for attr_name, attr_value in override.items():
                        table_info[f"attributes_{attr_name}"] = attr_value
            elif isinstance(override, str):
                table_info[col] = override

    # Read table_info
    optional_columns = {
        alias: col
        for alias, col in table_info.items()
        if alias not in ["value", "recordtime", id_column]
    }

    # The mapping values are raw SQL fragments, not just column names: a plain
    # column, a literal ('ward_id'), or an expression (CAST(c_systolic AS float)).
    # Project them via SQL — table[name] would only resolve existing column names.
    # The WHERE is applied in the same statement, where the original column names
    # it references are still in scope.
    expr = apply_sql_select(
        table,
        {
            id_column: table_info[id_column],
            "recordtime": table_info["recordtime"],
            "value": table_info["value"],
            **optional_columns,
        },
        where=where,
    )

    # Cast the projected value in ibis rather than inside the SQL fragment, so the
    # dtype stays expressed in backend-portable ibis types.
    if value_dtype:
        legacy_mapping = {"varchar": "string", "utf8": "string"}
        dtype = legacy_mapping.get(value_dtype.lower(), value_dtype.lower())
        expr = expr.mutate(value=cast("ibis.Value", expr["value"]).try_cast(dtype))

    alias_mapping = dict(table_info.items())
    datetime_cols = [
        col for col in ("recordtime", "attributes_storetime") if col in alias_mapping
    ]
    return expr, alias_mapping, datetime_cols


def build_co6hierarchy_extractor_expr(
    table: ibis.Table,
    id_column: Literal["case_id", "patient_id"],
    parent_name: str,
    suffixes: dict[str, str],
    not_strict: list[str] | None = None,
) -> tuple[ibis.Table, dict[str, str], list[str]]:
    """Select and merge a CoPRA 6 hierarchy variable from a source table.

    Pure ``ibis.Table -> ibis.Table`` transform: merges child elements of a
    parent variable by self-joining the hierarchy table once per suffix.
    Row-level filtering by cohort IDs is left to the caller via
    :func:`filter_by_tmp_table` on the returned table.

    Args:
        table: Hierarchy source table expression (usually
            ``conn.conn.table("it_copra6_hierarchy_v2", database=...)``).
        id_column: Primary-key column to return and join on (``"case_id"`` or
            ``"patient_id"``).
        parent_name: CoPRA 6 parent concept name.
        suffixes: Mapping of child suffix → output column name.  Must include a
            suffix that maps to ``"recordtime"``.
        not_strict: Suffixes that may be absent per row (``LEFT JOIN`` instead
            of ``INNER JOIN``).

    Returns:
        A ``(expr, alias_mapping, datetime_cols)`` triple where *expr* is an
        ibis table expression ready for :meth:`DuckDBConnector.lf_from_expr`.
    """
    if "recordtime" not in suffixes.values():
        raise ValueError("suffixes must include one that maps to 'recordtime'")

    db_id_col = "c_patnr" if id_column == "patient_id" else "c_falnr"

    # Parent view: rows where c_var_name matches the parent concept
    p = table.view()
    expr: ibis.Table = p.filter(p.c_var_name == parent_name)

    # One view per suffix; chain inner/left joins onto the parent
    child_views = {suffix: table.view() for suffix in suffixes}
    for suffix in suffixes:
        xc = child_views[suffix]
        how = "left" if not_strict and suffix in not_strict else "inner"
        expr = expr.join(
            xc,
            (xc.c_parent_id == p.c_id) & (xc.c_var_name == f"{parent_name}{suffix}"),
            how=how,
        )

    # SELECT: id column from parent + c_value from each child (cast where needed)
    select_exprs: list[ibis.Expr] = [p[db_id_col].name(id_column)]
    for suffix, col_name in suffixes.items():
        xc = child_views[suffix]
        val = (
            xc.c_value.try_cast("timestamp") if "recordtime" in col_name else xc.c_value
        )
        select_exprs.append(val.name(col_name))

    result = expr.select(*select_exprs)

    alias_mapping = dict.fromkeys(suffixes.values(), "c_value")
    datetime_cols = ["recordtime"]
    return result, alias_mapping, datetime_cols


# ------------------------
# Static-load expressions (cohort.py)
# ------------------------
def is_present(column: ibis.Column) -> ibis.ir.BooleanColumn:
    """Predicate: *column* holds a usable value.

    ``True`` when the column is neither SQL ``NULL`` nor the literal string
    ``"NULL"`` — a sentinel that the ishmed tables use for missing ids.

    Args:
        column: Column expression to test.

    Returns:
        A boolean column expression.
    """
    return column.notnull() & (column != "NULL")


def now_expr(
    end_date: datetime | None, *, reference_now: datetime | None = None
) -> ibis.ir.TimestampScalar:
    """Return the reference "now" timestamp used by the static loaders.

    In order of precedence: the configured extraction end date (so the cohort
    stays reproducible), the dataset's own frozen "now"
    (:attr:`DuckDBConnector.reference_now`, from the manifest), else the
    backend's current timestamp — Europe/Berlin wall-clock time, the time zone
    the connector's session runs in.

    Args:
        end_date: Optional extraction end date.
        reference_now: Optional dataset reference time.

    Returns:
        A timestamp scalar expression.
    """
    if end_date is not None:
        return ibis.timestamp(end_date)
    if reference_now is not None:
        return ibis.timestamp(reference_now)
    return ibis.now()


def stay_status_expr(
    begin: ibis.Value, end: ibis.Value, now: ibis.ir.TimestampScalar
) -> ibis.Value:
    """Classify a stay interval relative to *now*.

    Shared by the hospital-stay (``it_ishmed_fall``) and ICU-stay
    (``it_ishmed_bewegung``) loaders, which apply identical status logic to
    different begin/end columns.

    Args:
        begin: Stay start timestamp expression.
        end: Stay end timestamp expression.
        now: Reference "now" timestamp (see :func:`now_expr`).

    Returns:
        A string expression: one of ``"Implausible"``, ``"Incomplete"``,
        ``"Planned"``, ``"Ongoing"``, or ``"Finalised"``.
    """
    ninety_days_ago = now - ibis.interval(days=90)
    return ibis.cases(
        (begin > end, "Implausible"),
        (begin.isnull(), "Incomplete"),
        ((begin > now) & end.isnull(), "Planned"),
        ((ninety_days_ago <= begin) & end.isnull(), "Ongoing"),
        ((ninety_days_ago > begin) & end.isnull(), "Incomplete"),
        ((begin <= now) & (end <= now), "Finalised"),
        ((begin <= now) & (end > now), "Ongoing"),
        ((begin > now) & (end > now), "Planned"),
        else_="Incomplete",
    )


def within_date_window(
    start_col: ibis.Value,
    end_col: ibis.Value,
    *,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> ibis.ir.BooleanValue:
    """Predicate bounding a ``[start_col, end_col]`` interval to the extraction window.

    *start_col* is bounded below by *start_date* and *end_col* above by
    *end_date*.  A missing bound is treated as open — the row is **kept** — so the
    caller decides how to handle ongoing or unknown intervals (ongoing stays,
    living patients, …).  Returns an always-true predicate when neither date is
    given.

    Args:
        start_col: Interval-start column (e.g. admission, birthdate).
        end_col: Interval-end column (e.g. discharge, death).
        start_date: Optional lower bound applied to *start_col*.
        end_date: Optional upper bound applied to *end_col*.

    Returns:
        A boolean predicate expression.
    """
    pred: ibis.ir.BooleanValue = ibis.literal(True)
    if start_date is not None:
        pred = pred & (start_col.isnull() | (start_col >= ibis.timestamp(start_date)))
    if end_date is not None:
        pred = pred & (end_col.isnull() | (end_col <= ibis.timestamp(end_date)))
    return pred


def filtered_movements(
    bewegung: ibis.Table,
    *,
    icu_ids: Iterable[str] = (),
    exclude_movement_ids: Iterable[str] = (),
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> ibis.Table:
    """Filter ``it_ishmed_bewegung`` to qualifying ICU movements.

    Keeps admission movements (``c_bewegungstyp`` ``"3"``) and internal
    transfers (type ``"1"`` whose ``c_bewegungsart`` is not excluded), optionally
    restricted to the given ICU wards and bounded by the extraction date range.
    Shared by the patient, fall, and bewegung loaders — the first two use the
    result as an ``EXISTS`` filter via an ``isin`` subquery on the id column.

    The movement-art filter always applies.  The ICU-ward filter acts on a
    different column (``c_pflege_oe_id``) and is only applied when *icu_ids* is
    non-empty, so an empty list means "any ward" rather than "no rows".

    Args:
        bewegung: ``it_ishmed_bewegung`` source table expression.
        icu_ids: ICU ward ids (``c_pflege_oe_id``) to keep; empty means any ward.
        exclude_movement_ids: ``c_bewegungsart`` values to exclude for type-1
            movements.
        start_date: Optional lower bound on ``c_begin`` (missing bounds kept; see
            :func:`within_date_window`).
        end_date: Optional upper bound on ``c_ende`` (missing bounds kept).

    Returns:
        The filtered ``it_ishmed_bewegung`` table expression.
    """
    icu_ids = list(icu_ids)
    pred = (bewegung.c_bewegungstyp == "3") | (
        (bewegung.c_bewegungstyp == "1")
        & ~bewegung.c_bewegungsart.isin(list(exclude_movement_ids))
    )
    if icu_ids:
        pred = pred & bewegung.c_pflege_oe_id.isin(icu_ids)
    pred = pred & within_date_window(
        bewegung.c_begin, bewegung.c_ende, start_date=start_date, end_date=end_date
    )
    return bewegung.filter(pred)


def build_procedures(conn: DuckDBConnector, database: str) -> pl.LazyFrame:
    """Build the per-procedure frame from COPRA OP-times and SAP procedures.

    Used by the ``surgery`` variable.  Returns one row per COPRA OP-time
    (behandlung), enriched with the nearest SAP procedure (OPS codes + ward) via
    an as-of join.

    Output columns: ``case_id``, ``ops_code`` (list), ``or_time_begin``,
    ``or_time_end``, ``or_time_ops``, ``or_time_anes_begin``,
    ``or_time_anes_end``, ``or_time_incision``, ``or_time_suture``,
    ``or_time_anes_ready``, ``bw_pflege_oe``, ``bw_fach_oe``.

    Args:
        conn: Active :class:`DuckDBConnector`.
        database: Database name owning the source tables.

    Returns:
        A polars :class:`~polars.LazyFrame`, one row per procedure.
    """
    # COPRA hierarchy: Behandlung (var 30) -> OP-Zeiten (var 101322) -> event
    # name (var 101324) + event timestamp (var 101323); behandlung begin/end are
    # vars 1263/1264.
    event_mapping = {
        "began": "or_time_anes_begin",
        "endan": "or_time_anes_end",
        "schni": "or_time_incision",
        "naht": "or_time_suture",
        "freig": "or_time_anes_ready",
    }

    h_table = conn.conn.table("it_copra6_hierarchy_v2", database=database)

    # Behandlung (treatment) entries (var 30).
    hp = h_table.view()
    procedures = hp.filter(hp.c_var_id == 30).select(
        case_id=hp.c_falnr, procedure_id=hp.c_id
    )

    # Behandlung begin/end timestamps (var 1263 / 1264).
    hpt = h_table.view()
    procedure_times = hpt.filter(hpt.c_var_id.isin([1263, 1264])).select(
        procedure_id=hpt.c_parent_id,
        procedure_event_type=hpt.c_var_id,
        procedure_event_value=hpt.c_value,
    )

    # OP-Zeiten (operation-time) entries (var 101322).
    hot = h_table.view()
    or_times = hot.filter(hot.c_var_id == 101322).select(
        or_time_id=hot.c_id,
        procedure_id=hot.c_parent_id,
    )

    # One row per OP-time carrying its behandlung begin/end.  Begin/end times are
    # LEFT-joined so behandlungen without 1263/1264 timestamps are kept with null
    # or_time_begin/or_time_end rather than dropped.
    base_joined = (
        or_times.join(procedures, "procedure_id", how="inner")
        .join(procedure_times, "procedure_id", how="left")
        .select(
            or_times.or_time_id,
            procedures.case_id,
            procedures.procedure_id,
            procedure_times.procedure_event_type,
            procedure_times.procedure_event_value,
        )
    )
    base = base_joined.group_by("or_time_id", "case_id", "procedure_id").agg(
        or_time_begin=base_joined.procedure_event_value.max(
            where=base_joined.procedure_event_type == 1263
        ),
        or_time_end=base_joined.procedure_event_value.max(
            where=base_joined.procedure_event_type == 1264
        ),
    )

    # Event name (var 101324) paired with its timestamp (var 101323) per OP-time.
    # Accepts any event name (the 5 milestones are picked by the pivot below) so
    # behandlungen with non-standard events are kept.
    het = h_table.view()
    event_types = het.filter(het.c_var_id == 101324).select(
        or_time_id=het.c_parent_id,
        procedure_event_type=het.c_value,
    )

    hev = h_table.view()
    event_times = hev.filter(hev.c_var_id == 101323).select(
        or_time_id=hev.c_parent_id,
        procedure_event_value=hev.c_value,
    )

    # INNER join: an event is only kept when it has a recorded timestamp.
    events = event_types.join(event_times, "or_time_id", how="inner").select(
        event_types.or_time_id,
        event_types.procedure_event_type,
        event_times.procedure_event_value,
    )

    # Pivot the events into one timestamp column per event type, per behandlung.
    op_joined = base.join(events, "or_time_id", how="inner").select(
        base.case_id,
        base.procedure_id,
        base.or_time_begin,
        base.or_time_end,
        events.procedure_event_type,
        events.procedure_event_value,
    )
    op_expr = op_joined.group_by(
        "case_id", "procedure_id", "or_time_begin", "or_time_end"
    ).agg(
        **{
            alias: op_joined.procedure_event_value.max(
                where=op_joined.procedure_event_type == raw.upper()
            )
            for raw, alias in event_mapping.items()
        }
    )

    dt_cols = list(event_mapping.values()) + ["or_time_begin", "or_time_end"]
    lf_op_times = conn.lf_from_expr(op_expr, datetime_cols=dt_cols)

    # Timestamps stored in c_value are UTC, everything else is local time —
    # convert the c_value timestamps to local time.
    lf_op_times = lf_op_times.with_columns(
        pl.col(dt_cols)
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone("Europe/Berlin")
        .dt.cast_time_unit("us")
        .dt.replace_time_zone(None)
    )

    # SAP surgeries (OPS code starting with 5), aggregated per movement.
    proc = conn.conn.table("it_ishmed_procedure", database=database)
    sap_ops = (
        proc.select(
            proc.c_falnr.name("case_id"),
            proc.c_lfd_bewegung.name("movement_counter"),
            proc.c_prozedur_code,
            proc.c_prozedur_begin,
        )
        .filter(proc.c_prozedur_code.like("5%"))
        .group_by("case_id", "movement_counter")
        .agg(
            ops_code=proc.c_prozedur_code.group_concat(),
            or_time_ops=proc.c_prozedur_begin.min(),
        )
    )

    bw = conn.conn.table("it_ishmed_bewegung", database=database)
    sap_expr = sap_ops.join(
        bw.select(
            bw.c_falnr.name("case_id"),
            bw.c_laufende_nummer.name("movement_counter"),
            bw.c_pflege_oe_id,
            bw.c_fachliche_oe_id,
        ),
        ["case_id", "movement_counter"],
        how="left",
    ).select(
        sap_ops.case_id,
        sap_ops.ops_code,
        sap_ops.or_time_ops,
        bw.c_pflege_oe_id.name("bw_pflege_oe"),
        bw.c_fachliche_oe_id.name("bw_fach_oe"),
    )
    lf_sap = conn.lf_from_expr(sap_expr, datetime_cols=["or_time_ops"])

    lf_sap = lf_sap.with_columns(pl.col("or_time_ops").dt.cast_time_unit("us")).sort(
        "or_time_ops"
    )
    lf_op_times = lf_op_times.sort("or_time_begin")

    lf_proc = lf_op_times.join_asof(
        lf_sap,
        left_on="or_time_begin",
        right_on="or_time_ops",
        by="case_id",
        strategy="nearest",
        tolerance="4h",
        # Both sides are sorted on their time key above; polars cannot verify
        # that per `by` group and would warn otherwise.
        check_sortedness=False,
    )

    # GROUP_CONCAT returns comma separated values (Impala: ", ", DuckDB: ",")
    # -> parse into a list of stripped codes.
    lf_proc = lf_proc.with_columns(
        pl.col("ops_code")
        .str.split(",")
        .list.eval(pl.element().str.strip_chars())
        .fill_null(pl.lit([], dtype=pl.List(pl.Utf8)))
    )

    return lf_proc.drop("procedure_id")


# ------------------------
# Variable data helpers
# ------------------------
def attach_obs_ids(
    df: pl.DataFrame,
    cohort: Cohort,
    *,
    columns: Iterable[str] | None = None,
) -> pl.DataFrame:
    """Complete *df*'s id columns from ``cohort.obs`` and scope it to the cohort.

    Dynamic charite_demo variables are extracted at the screened observation
    level (``case_id`` / ``patient_id``), so a ``py()`` function receives data
    *without* the cohort primary key — the pipeline only adds it afterwards, in
    the time-window step.  Conversely, data taken from an already-extracted
    ``required_vars`` entry carries the primary key but not ``case_id``.

    The obs id map is inner-joined onto *df* on the finest id column *df* already
    has (``cohort.primary_key``, else ``case_id``, else ``patient_id``), adding any
    of *columns* that are missing.  Because the join is an inner join it also
    restricts *df* to the cohort — relied upon by callers whose source data is not
    already cohort-scoped.

    Rows fan out when one id maps to several observations (e.g. a case with
    multiple ICU stays).

    Args:
        df: Variable data carrying at least one obs id column.
        cohort: Cohort providing ``obs`` and the primary key.
        columns: Id columns to ensure are present.  Defaults to
            ``(cohort.primary_key,)``.

    Returns:
        *df* restricted to the cohort, with *columns* present.

    Raises:
        ValueError: If *df* has no obs id column to join on.
    """
    targets = tuple(columns) if columns is not None else (cohort.primary_key,)

    join_col = next(
        (
            col
            for col in (cohort.primary_key, "case_id", "patient_id")
            if col in df.columns
        ),
        None,
    )
    if join_col is None:
        raise ValueError(
            f"Cannot attach {targets}: data has no "
            f"{cohort.primary_key!r}, 'case_id' or 'patient_id' column to join on."
        )

    missing = [col for col in targets if col not in df.columns]
    keys = cohort.obs.select(join_col, *missing).unique()
    return df.join(keys, on=join_col)


# ------------------------
# Parsers
# ------------------------
def parse_date(date_str: str) -> datetime:
    """Parse a date string in ``YYYY-MM-DD`` or ``YYYY-MM-DD HH:MM:SS`` format.

    Args:
        date_str: Date string to parse.

    Returns:
        Parsed :class:`~datetime.datetime` object.

    Raises:
        ValueError: If the time component has an unsupported number of parts.
    """
    date, _, time = date_str.partition(" ")
    if time:
        time_components = len(time.split(":"))
        if time_components == 3:
            return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        if time_components == 2:
            return datetime.strptime(date_str, "%Y-%m-%d %H:%M")
        raise ValueError(f"Invalid time format: {time}")
    return datetime.strptime(date, "%Y-%m-%d")


def parse_filters(filters: str) -> str:
    """Parse additional SQL filter strings, including debug shorthands.

    The shorthand ``_dN`` expands to a ``c_aufnahme BETWEEN`` clause spanning
    *N* months of data starting from 2023-01-01.

    Args:
        filters: Raw filter string or ``_dN`` shorthand.

    Returns:
        Resolved SQL filter string.
    """
    if filters.startswith("_d"):
        months = int(filters[2:])
        start_date = "2023-01-01"
        end_date = (
            datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=30 * months)
        ).strftime("%Y-%m-%d")
        filters = f"c_aufnahme BETWEEN '{start_date}' AND '{end_date}'"
        logger.debug(__("Filters set to {filters}", filters=filters))
    return filters


# ------------------------
# Frame helper
# ------------------------
def merge_consecutive_stays(
    stays: pl.LazyFrame,
    treshold_hours: int = 24,
    strategy: Literal["any", "same_icu_only"] = "any",
) -> pl.LazyFrame:
    """Merge consecutive ICU stays in a lazyframe, including stays across different ICUs.

    Args:
        stays (pl.LazyFrame): LazyFrame containing at least
                ['case_id', 'icu_admission', 'icu_discharge', 'icu_status']
        treshold_hours (int): Maximum gap (in hours) to merge consecutive stays.
        strategy (str): Merges any stay if 'any' is selected, or only stays in the same icu_id if 'same_icu_only' is selected.

    Returns:
        merged_stays (pl.Lazyframe): A LazyFrame with one row per merged stay, having at least these columns:\
            ['case_id', 'icu_admission', 'icu_discharge', 'icu_status']
            All other columns will be converted to ordered lists.
    """
    treshold = timedelta(hours=treshold_hours)

    grouped = (
        stays.sort(by=["case_id", "icu_admission", "icu_discharge"], nulls_last=True)
        .with_columns(
            pl.col("icu_discharge").shift(1).over("case_id").alias("last_discharge"),
            pl.col("icu_id").shift(1).over("case_id").alias("last_icu_id"),
        )
        .with_columns(
            pl.col("icu_admission")
            .sub(pl.col("last_discharge"))
            .alias("time_since_last_discharge")
        )
        .with_columns(
            pl.col("time_since_last_discharge")
            .ge(treshold)
            .or_(
                pl.lit(False)
                if strategy == "any"
                else pl.col("icu_id").ne(pl.col("last_icu_id"))
            )
            .fill_null(True)  # first stay always starts a new group
            .alias("break_group")
        )
        .with_columns(
            pl.col("break_group").cum_sum().over("case_id").alias("merge_group")
        )
    )

    return (
        grouped.group_by("case_id", "merge_group")
        .agg(
            pl.col("icu_admission").first(),
            pl.col("icu_discharge").last(),
            pl.col("icu_status").last(),
            # NOTE: It is very important to explode the other values,
            # even if there is only one ICU Stay per group when strategy == "same_icu_only".
            # i.e. icu_id, bw_patient_origin and bw_fach_oe are expected to be a list!
            pl.all().exclude("icu_admission", "icu_discharge", "icu_status"),
        )
        .drop(
            "merge_group",
            "last_discharge",
            "last_icu_id",
            "time_since_last_discharge",
            "break_group",
        )
        .sort("case_id", "icu_admission")
    )
