"""In-process DuckDB connection over the CHARITÉ mock dataset.

The dataset is a directory of Parquet files described by a ``manifest.json``,
published at a URL and downloadable to a local folder (see
:func:`corr_vars.sources.charite_demo.download`). Both layouts are identical:

.. code-block:: text

    <base>/manifest.json
    <base>/charite_mockdata/<table>.parquet
    <base>/charite_mockdata/<table>/<part>.parquet

:class:`DuckDBConnector` registers one view per manifest table in a schema named
after the database (``charite_mockdata.<table>``), so the queries the source
builds with ibis address the tables exactly as they would on the Health Data
Platform. Remote files are read lazily with HTTP range requests through DuckDB's
``httpfs`` extension; only the row groups a query touches are transferred.
"""

from __future__ import annotations

import json
import re
import textwrap
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path

import httpx
import ibis
import polars as pl

from corr_vars import __, logger
from corr_vars.utils.base import columns, row_length

from typing import TYPE_CHECKING, Any, Final, Literal, overload

if TYPE_CHECKING:
    from ibis.backends.duckdb import Backend as DuckDBBackend

    from corr_vars.definitions import PandasDataframe, PolarsDataframe, PolarsLazyframe

    from collections.abc import Iterable

#: Name of the dataset description at the root of a dataset directory or URL.
MANIFEST_FILE: Final[str] = "manifest.json"

#: Timeout (seconds) for fetching the manifest over HTTP.
MANIFEST_TIMEOUT: Final[float] = 30.0

#: Attempts for fetching the manifest over HTTP before giving up.
MANIFEST_RETRIES: Final[int] = 3

#: DuckDB settings applied to a connection that reads over HTTP. A setting the
#: installed DuckDB does not know is skipped, so older and newer releases both
#: work.
HTTP_SETTINGS: Final[dict[str, object]] = {
    # Cache HEAD responses and Parquet footers: every query re-binds the views,
    # and without these each one would re-fetch the metadata of every file.
    "enable_http_metadata_cache": True,
    "parquet_metadata_cache": True,
    "enable_object_cache": True,
    "http_keep_alive": True,
    "http_timeout": 120,
    "http_retries": 5,
    "http_retry_wait_ms": 500,
}

#: Time zone of the dataset's naive timestamps. The session runs in it so that
#: ``CURRENT_TIMESTAMP`` compares like a wall-clock time recorded at the Charité.
SESSION_TIME_ZONE: Final[str] = "Europe/Berlin"


def is_url(path: str) -> bool:
    """Whether `path` addresses a remote dataset rather than a local directory.

    Args:
        path (str): A dataset location as given in the source config.

    Returns:
        bool: ``True`` for ``http(s)://`` and ``s3://`` locations.
    """
    return re.match(r"^(https?|s3)://", path, flags=re.IGNORECASE) is not None


def normalize_path(path: str) -> str:
    """Normalise a dataset location so equal locations compare equal.

    A URL loses its trailing slash; a local directory has ``~`` expanded and is
    made absolute.

    Args:
        path (str): A dataset location as given in the source config.

    Returns:
        str: The canonical form of `path`.
    """
    if is_url(path):
        return path.rstrip("/")
    return str(Path(path).expanduser().resolve())


def fetch_manifest(path: str) -> dict[str, Any]:
    """Read the ``manifest.json`` of a dataset.

    Args:
        path (str): Normalised dataset location (see :func:`normalize_path`).

    Returns:
        dict[str, Any]: The parsed manifest.

    Raises:
        FileNotFoundError: If a local directory holds no manifest.
        ConnectionError: If the manifest cannot be fetched over HTTP.
        ValueError: If the manifest has no ``tables`` mapping.
    """
    if is_url(path):
        url = f"{path}/{MANIFEST_FILE}"
        last_error: Exception | None = None
        for attempt in range(1, MANIFEST_RETRIES + 1):
            try:
                response = httpx.get(
                    url, timeout=MANIFEST_TIMEOUT, follow_redirects=True
                )
                response.raise_for_status()
                manifest = response.json()
                break
            except (httpx.TransportError, httpx.HTTPStatusError) as error:
                last_error = error
                logger.debug(
                    __(
                        "Fetching {url} failed (attempt {attempt}): {error}",
                        url=url,
                        attempt=attempt,
                        error=error,
                    )
                )
                time.sleep(0.5 * attempt)
        else:
            raise ConnectionError(
                f"Could not fetch the charite_demo manifest from {url}: {last_error}. "
                "Check your network connection, or download the dataset with "
                "corr_vars.sources.charite_demo.download() and pass its folder as "
                "'path'."
            )
    else:
        manifest_path = Path(path) / MANIFEST_FILE
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"No {MANIFEST_FILE} in {path}. 'path' must point at a dataset "
                "folder created by corr_vars.sources.charite_demo.download(), or "
                "at the dataset URL."
            )
        manifest = json.loads(manifest_path.read_text())

    if not isinstance(manifest, dict) or not isinstance(manifest.get("tables"), dict):
        raise ValueError(f"Malformed manifest at {path}: missing 'tables' mapping.")
    return manifest


def resolve_files(path: str, files: Iterable[str]) -> list[str]:
    """Turn manifest-relative file paths into absolute URLs or paths.

    Args:
        path (str): Normalised dataset location.
        files (Iterable[str]): File paths relative to the manifest.

    Returns:
        list[str]: Absolute URLs for a remote dataset, absolute paths otherwise.
    """
    if is_url(path):
        return [f"{path}/{f.lstrip('/')}" for f in files]
    return [str(Path(path) / f) for f in files]


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _sql_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def string_to_datetime(expr: pl.Expr) -> pl.Expr:
    """Parse a string column the way timestamps are written in the dataset.

    Args:
        expr (pl.Expr): String (or castable) expression.

    Returns:
        pl.Expr: A datetime expression; unparseable values become null.
    """
    string_formats = (
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%.f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    )
    return pl.coalesce(
        [
            expr.cast(pl.Utf8).str.strptime(
                pl.Datetime("us"), format=fmt, strict=False, cache=True
            )
            for fmt in string_formats
        ]
    )


class DuckDBConnector:
    """In-memory DuckDB with one view per table of a charite_demo dataset.

    Exposes the surface the source's queries use: the ibis backend as
    :attr:`conn`, raw SQL through :meth:`execute` / :meth:`execute_dml`, and
    polars results through :meth:`lf_read_sql` / :meth:`df_read_sql` (SQL) and
    :meth:`lf_from_expr` / :meth:`df_from_expr` (ibis expressions).

    Temporary tables created on :attr:`conn` live as long as this connector, so
    the source keeps one connector per dataset for the whole process — see
    :func:`corr_vars.sources.charite_demo.helpers.reinit_conn`. The methods here
    serialise access with a lock; code driving :attr:`conn` directly from
    several threads has to do the same.

    Args:
        path (str): Dataset URL or local directory (``~`` is expanded).
        database (str): Schema the tables are registered under.
    """

    conn: DuckDBBackend
    path: str
    database: str
    manifest: dict[str, Any]

    def __init__(self, path: str, database: str = "charite_mockdata") -> None:
        self.path = normalize_path(path)
        self.database = database
        self._lock = threading.RLock()

        self.manifest = fetch_manifest(self.path)
        self.conn = ibis.duckdb.connect()
        self._configure()
        self._register_views()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(path={self.path!r}, database={self.database!r})"

    @property
    def reference_now(self) -> datetime | None:
        """The dataset's frozen "now" (manifest ``reference_now``), if it declares one.

        A synthetic dataset is generated up to a fixed date; using that date as
        "now" keeps stay statuses (``Ongoing`` / ``Finalised``) and open cases
        reproducible regardless of when the cohort is built.
        """
        value = self.manifest.get("reference_now")
        if not value:
            return None
        return datetime.fromisoformat(str(value).rstrip("Z")).replace(tzinfo=None)

    @property
    def tables(self) -> list[str]:
        """Names of the tables registered from the manifest."""
        return sorted(self.manifest["tables"])

    def close(self) -> None:
        """Close the DuckDB connection; temporary tables are dropped with it."""
        with self._lock:
            self.conn.disconnect()

    def _configure(self) -> None:
        raw = self.conn.con
        raw.execute(f"SET TimeZone = {_sql_string(SESSION_TIME_ZONE)}")
        if not is_url(self.path):
            return

        raw.execute("SET autoinstall_known_extensions = true")
        raw.execute("SET autoload_known_extensions = true")
        try:
            raw.execute("INSTALL httpfs")
            raw.execute("LOAD httpfs")
        except Exception as error:
            raise ConnectionError(
                "Reading the charite_demo dataset over HTTP needs DuckDB's httpfs "
                f"extension, which could not be installed or loaded: {error}. "
                "Download the dataset with corr_vars.sources.charite_demo.download() "
                "and pass its folder as 'path' to work without it."
            ) from error

        known = {
            name
            for (name,) in raw.execute("SELECT name FROM duckdb_settings()").fetchall()
        }
        for name, value in HTTP_SETTINGS.items():
            if name not in known:
                logger.debug(__("DuckDB has no setting {name}; skipped", name=name))
                continue
            literal = str(value).lower() if isinstance(value, bool) else str(value)
            raw.execute(f"SET {name} = {literal}")

    def _register_views(self) -> None:
        raw = self.conn.con
        schema = _sql_identifier(self.database)
        raw.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        for table, info in self.manifest["tables"].items():
            files = resolve_files(self.path, info.get("files", []))
            if not files:
                logger.warning(
                    __(
                        "Manifest lists no files for table {table}; skipped",
                        table=table,
                    )
                )
                continue
            file_list = ", ".join(_sql_string(f) for f in files)
            raw.execute(
                f"CREATE OR REPLACE VIEW {schema}.{_sql_identifier(table)} AS "
                f"SELECT * FROM read_parquet([{file_list}])"
            )
        logger.debug(
            __(
                "Registered {n} charite_demo tables from {path}",
                n=len(self.manifest["tables"]),
                path=self.path,
            )
        )

    def execute(self, query: str) -> list:
        """Execute a SQL query and return the result as a list of row tuples.

        Args:
            query: SQL query to execute.

        Returns:
            List of tuples (one per row).
        """
        with self._lock:
            return self.conn.con.execute(query).fetchall()

    def execute_dml(self, query: str) -> None:
        """Execute a DDL/DML statement without returning results.

        Args:
            query: SQL DDL or DML statement (CREATE, INSERT, DROP, etc.).
        """
        with self._lock:
            self.conn.con.execute(query)

    def lf_read_sql(
        self,
        query: str,
        datetime_cols: Iterable[str] = [],
        keep_cols: Iterable[str] = [],
    ) -> PolarsLazyframe:
        """Execute a SQL query and return the result as a polars LazyFrame.

        The query runs eagerly; the LazyFrame wraps the materialised result.

        Args:
            query: SQL query to execute (DuckDB dialect).
            datetime_cols: Columns to cast to :class:`~polars.Datetime` when the
                backend returns them as strings.
            keep_cols: Columns to retain in the result.  All columns are kept
                when empty.

        Returns:
            Query result as a :class:`~polars.LazyFrame`.
        """
        logger.debug(
            __(
                "[lf_read_sql] Reading SQL query:\n{query}",
                query=lambda: textwrap.indent(textwrap.dedent(query).strip(), "  "),
            )
        )
        try:
            with self._lock:
                df = self.conn.con.execute(query).pl()
        except Exception as exc:
            logger.error(
                __("Error executing SQL query:\n{query}\n{exc}", query=query, exc=exc)
            )
            raise
        return self._postprocess(df, datetime_cols=datetime_cols, keep_cols=keep_cols)

    # return_polars False
    @overload
    def df_read_sql(
        self,
        query: str,
        datetime_cols: list[str] = ...,
        keep_cols: list[str] = ...,
        return_polars: Literal[False] = ...,
    ) -> PandasDataframe: ...

    # return_polars True
    @overload
    def df_read_sql(
        self,
        query: str,
        datetime_cols: list[str] = ...,
        keep_cols: list[str] = ...,
        *,
        return_polars: Literal[True],
    ) -> PolarsDataframe: ...

    # return_polars bool
    @overload
    def df_read_sql(
        self,
        query: str,
        datetime_cols: list[str] = ...,
        keep_cols: list[str] = ...,
        return_polars: bool = ...,
    ) -> PolarsDataframe | PandasDataframe: ...

    def df_read_sql(
        self,
        query: str,
        datetime_cols: list[str] = [],
        keep_cols: list[str] = [],
        return_polars: bool = False,
    ) -> PolarsDataframe | PandasDataframe:
        """Execute a SQL query and return the result as a pandas or polars DataFrame.

        Args:
            query: SQL query to execute (DuckDB dialect).
            datetime_cols: Columns to cast to :class:`~polars.Datetime` when the
                backend returns them as strings.
            keep_cols: Columns to retain in the result.  All columns are kept
                when empty.
            return_polars: Return a :class:`~polars.DataFrame` when ``True``,
                otherwise a :class:`~pandas.DataFrame`.

        Returns:
            Query result as a DataFrame.
        """
        df = self.lf_read_sql(
            query, datetime_cols=datetime_cols, keep_cols=keep_cols
        ).collect()
        return df if return_polars else df.to_pandas()

    def lf_from_expr(
        self,
        expr: ibis.Table,
        datetime_cols: Iterable[str] = [],
        keep_cols: Iterable[str] = [],
    ) -> PolarsLazyframe:
        """Execute an ibis table expression and return the result as a polars LazyFrame.

        Args:
            expr: ibis table expression to execute.
            datetime_cols: Columns to cast to :class:`~polars.Datetime` when the
                backend returns them as strings.
            keep_cols: Columns to retain in the result.

        Returns:
            Query result as a :class:`~polars.LazyFrame`.
        """
        logger.debug(
            __(
                "[lf_from_expr] Executing:\n{query}",
                query=lambda: textwrap.indent(str(self.conn.compile(expr)), "  "),
            )
        )
        try:
            with self._lock:
                df = self.conn.to_polars(expr)
        except Exception as exc:
            logger.error(
                __(
                    "Error executing query:\n{query}\n{exc}",
                    query=lambda: self.conn.compile(expr),
                    exc=exc,
                )
            )
            raise
        return self._postprocess(df, datetime_cols=datetime_cols, keep_cols=keep_cols)

    def df_from_expr(
        self,
        expr: ibis.Table,
        datetime_cols: Iterable[str] = [],
        keep_cols: Iterable[str] = [],
    ) -> PolarsDataframe:
        """Execute an ibis table expression and return the result as a polars DataFrame.

        The eager counterpart to :meth:`lf_from_expr`.

        Args:
            expr: ibis table expression to execute.
            datetime_cols: Columns to cast to :class:`~polars.Datetime` when the
                backend returns them as strings.
            keep_cols: Columns to retain in the result.

        Returns:
            Query result as a :class:`~polars.DataFrame`.
        """
        return self.lf_from_expr(
            expr, datetime_cols=datetime_cols, keep_cols=keep_cols
        ).collect()

    @staticmethod
    def _postprocess(
        df: pl.DataFrame,
        *,
        datetime_cols: Iterable[str],
        keep_cols: Iterable[str],
    ) -> pl.LazyFrame:
        """Apply the ``datetime_cols`` / ``keep_cols`` contract to a result.

        Args:
            df: Raw query result.
            datetime_cols: Columns that must be datetimes; string columns among
                them are parsed. Missing ones warn.
            keep_cols: Columns to keep, in this order. Missing ones warn; all
                columns are kept when none of them exist.

        Returns:
            The processed result, as a LazyFrame.
        """
        datetime_cols = list(dict.fromkeys(datetime_cols))
        keep_cols = list(dict.fromkeys(keep_cols))
        schema = df.schema

        missing_datetime = [c for c in datetime_cols if c not in schema]
        if missing_datetime:
            warnings.warn(
                f"Column(s) {', '.join(sorted(missing_datetime))} specified as datetime column(s) but not found in dataframe",
                UserWarning,
                stacklevel=3,
            )
        to_convert = [
            c
            for c in datetime_cols
            if c in schema and not isinstance(schema[c], (pl.Datetime, pl.Date))
        ]
        if to_convert:
            df = df.with_columns(
                pl.col(c).pipe(string_to_datetime).alias(c) for c in to_convert
            )

        # DuckDB hands CURRENT_TIMESTAMP-derived columns over in nanoseconds;
        # keep every datetime at the microsecond resolution the tables use.
        df = df.with_columns(
            pl.col(name).dt.cast_time_unit("us")
            for name, dtype in df.schema.items()
            if isinstance(dtype, pl.Datetime) and dtype.time_unit == "ns"
        )

        missing_keep = [c for c in keep_cols if c not in schema]
        if missing_keep:
            warnings.warn(
                f"Column(s) {', '.join(sorted(missing_keep))} specified as keep column(s) but not found in dataframe",
                UserWarning,
                stacklevel=3,
            )
        present_keep = [c for c in keep_cols if c in schema]
        if present_keep:
            df = df.select(present_keep)

        lf = df.lazy()
        logger.debug(
            __(
                "DuckDB returning {columns} columns and {rows} rows.",
                columns=lambda: columns(lf),
                rows=lambda: row_length(lf),
            )
        )
        return lf
