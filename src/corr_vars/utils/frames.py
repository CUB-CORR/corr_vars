from datetime import timedelta
from io import BytesIO
from os import PathLike

import pandas as pd
import polars as pl
import polars.selectors as cs
from tableone import TableOne

from corr_vars import __, logger
from corr_vars.definitions import (
    ObsLevel,
    OffsetLike,
    PolarsFrame,
    StataDateFormat,
    TimeAnchorColumn,
)
from corr_vars.utils.base import (
    as_expr,
    columns,
    is_empty,
    normalize_offset,
    struct_fields,
)

from collections.abc import Callable, Hashable, Iterable, Sequence
from corr_vars.definitions.typing import CleaningDict, TimeDifferenceUnit
from polars._typing import AsofJoinStrategy, ColumnNameOrSelector, PolarsDataType
from typing import Literal, cast, overload

__all__ = [
    "absolute_and_relative_value_counts",
    "apply_cleaning",
    "attach_asof_value",
    "build_attributes",
    "convert_to_pandas_df",
    "convert_to_polars_df",
    "convert_to_polars_lf",
    "convert_to_stata",
    "convert_to_tableone",
    "find_nearest",
    "interval_bucket_agg",
    "remove_asof",
    "threshold_score",
    "time_difference",
    "unique_sucessive",
    "unnest",
]


# ------------------------
# Converter
# ------------------------
def convert_to_polars_df(
    value: pl.DataFrame | pl.LazyFrame | pd.DataFrame,
) -> pl.DataFrame:
    if isinstance(value, pl.DataFrame):
        return value
    if isinstance(value, pl.LazyFrame):
        return value.collect()
    if isinstance(value, pd.DataFrame):
        return pl.from_pandas(value)
    raise TypeError(f"Conversion failed. Unsupported data type: {type(value)}")


def convert_to_polars_lf(
    value: pl.DataFrame | pl.LazyFrame | pd.DataFrame,
) -> pl.LazyFrame:
    if isinstance(value, pl.DataFrame):
        return value.lazy()
    if isinstance(value, pl.LazyFrame):
        return value
    if isinstance(value, pd.DataFrame):
        return pl.from_pandas(value).lazy()
    raise TypeError(f"Conversion failed. Unsupported data type: {type(value)}")


def convert_to_pandas_df(
    value: pl.DataFrame | pl.LazyFrame | pd.DataFrame,
) -> pd.DataFrame:
    if isinstance(value, pl.DataFrame):
        return value.to_pandas()
    if isinstance(value, pl.LazyFrame):
        return value.collect().to_pandas()
    if isinstance(value, pd.DataFrame):
        return value
    raise TypeError(f"Conversion failed. Unsupported data type: {type(value)}")


def convert_to_stata(
    df: pd.DataFrame,
    convert_dates: (
        dict[
            Hashable,
            StataDateFormat,
        ]
        | None
    ) = None,
    write_index: bool = True,
    to_file: str | PathLike[str] | None = None,
) -> pd.DataFrame | None:
    """Convert a pandas DataFrame to a Stata compatible DataFrame or writes the DataFrame to a Stata file.

    Args:
        df (pd.DataFrame): The DataFrame to be converted to Stata format.

        convert_dates (dict[Hashable, str]): Dictionary of columns to convert to Stata date format.

        write_index (bool): Whether to write the index as a column.

        to_file (str | None): Path to save as .dta file. If left unspecified, the DataFrame will not be saved.

    Returns:
        pd.DataFrame: A Pandas Dataframe compatible with Stata if `to_file` is None.

    """
    df[df.select_dtypes(include=["object"]).columns] = df.select_dtypes(
        include=["object"]
    ).astype(str)

    format_guidance = []
    for col in df.select_dtypes(include=["datetime64"]).columns:
        format_guidance.append(f"format {col} %tc")
        df.loc[:, col] = df[col].mask(
            (df[col] < pd.Timestamp.min) | (df[col] > pd.Timestamp.max),
            pd.NA,
        )
    format_guidance_string = "\n".join(format_guidance)

    max_string_length = bool(
        df.select_dtypes(include=["object", "string"]).map(len).max(axis=None).item()
    )
    version = 114 if max_string_length < 245 else None
    logger.info(
        __(
            "Creating a DataFrame compatible with {version} based on max string length {max_string_length}",
            version=version or "118 or 119",
            max_string_length=max_string_length,
        )
    )
    logger.info(
        __(
            "You can load the data with a command like this inside a Notebook:\n"
            "%%stata -d stata_df -force\n{format_guidance_string}",
            format_guidance_string=format_guidance_string,
        )
    )

    if to_file:
        df.to_stata(
            to_file,
            convert_dates=convert_dates,
            write_index=write_index,
            version=version,
        )
        return None
    output = BytesIO()

    df.to_stata(
        output,
        convert_dates=convert_dates,
        write_index=write_index,
        version=version,
    )

    output.seek(0)

    return pd.read_stata(
        output,
        convert_dates=False,
        convert_missing=False,
        convert_categoricals=False,
    )


def convert_to_tableone(
    df: pl.DataFrame,
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
    """Create a `TableOne <https://tableone.readthedocs.io/en/latest/index.html>`_ object for the pandas DataFrame.

    Args:
        df (pl.DataFrame): The input.
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
    """
    # Ignore userspecified columns
    exclude_cols: list[str] = []
    if ignore_cols is not None:
        if isinstance(ignore_cols, str):
            exclude_cols.append(ignore_cols)
        else:
            exclude_cols.extend(ignore_cols)

    # Also ignore datetime, list/array, dict columns, etc.
    unhashable_cols: list[str] = df.select(
        ~(
            cs.boolean()
            | cs.categorical()
            | cs.numeric()
            | cs.string()
            | cs.by_dtype(pl.Enum)
        )
    ).columns
    logger.info(
        __(
            "Skipping unhashable columns: {unhashable_cols}",
            unhashable_cols=lambda: ",".join(unhashable_cols),
        )
    )
    exclude_cols.extend(unhashable_cols)

    # Ignore ID columns
    id_columns: list[str] = [obs_level.primary_key for obs_level in ObsLevel]
    exclude_cols.extend(id_columns)

    # Ignore groupby column (added as special argument)
    if groupby is not None:
        exclude_cols.append(groupby)

    # Collect included columns
    include_cols: list[str] = df.select(pl.all().exclude(exclude_cols)).columns

    # Reorder columns to put icu_id last (because it is so long...)
    if "icu_id" in include_cols:
        include_cols.append(include_cols.pop(include_cols.index("icu_id")))

    # Set non-normal distribution for all numeric columns, except for user specififed normal columns
    normal: list[str] = normal_cols or []
    nonnormal: list[str] = (
        df.select(pl.col(include_cols))
        .select(cs.numeric().exclude("inhospital_death"))
        .columns
    )

    # Explicitly set booleans and strings to categorical
    categorical_order = order or {}
    categorical: list[str] = (
        df.select(pl.col(include_cols))
        .select(cs.boolean() | cs.categorical() | cs.string() | cs.by_dtype(pl.Enum))
        .columns
    )

    # Transform / filter data
    boolean: list[str] = df.select(pl.col(include_cols)).select(cs.boolean()).columns
    if replace_booleans is not None:
        df = df.with_columns(
            pl.col(boolean).replace_strict(
                {
                    True: replace_booleans[0],
                    False: replace_booleans[1],
                },
                return_dtype=pl.Utf8,
            )
        )

        for b in boolean:
            # Only override order if unspecified
            categorical_order.setdefault(b, [replace_booleans[0], replace_booleans[1]])

    pd_df = convert_to_pandas_df(df)
    if filter_query:
        pd_df = pd_df.query(filter_query)

    logger.info(
        __(
            "Columns: {include_cols}",
            include_cols=lambda: ",".join(include_cols),
        )
    )
    logger.debug(
        __(
            "Unhashable Columns: {unhashable_cols}",
            unhashable_cols=lambda: ",".join(unhashable_cols),
        )
    )
    logger.debug(
        __(
            "Excluded Columns: {exclude_cols}",
            exclude_cols=lambda: ",".join(exclude_cols),
        )
    )
    logger.debug(
        __(
            "Included Columns: {include_cols}",
            include_cols=lambda: ",".join(include_cols),
        )
    )
    logger.info(
        __(
            "Normal: {normal}",
            normal=lambda: ",".join(normal),
        )
    )
    logger.info(
        __(
            "Nonnormal: {nonnormal}",
            nonnormal=lambda: ",".join(nonnormal),
        )
    )
    logger.info(
        __(
            "Categorical: {categorical}",
            categorical=lambda: ",".join(categorical),
        )
    )
    logger.info(
        __(
            "Groupby: {groupby}",
            groupby=groupby,
        )
    )

    # Overall
    add_overall = overall or (groupby is None)

    return TableOne(
        pd_df,
        columns=include_cols,
        nonnormal=nonnormal,
        categorical=categorical,
        # Pass through kwargs
        display_all=display_all,
        groupby=groupby,
        order=categorical_order,
        overall=add_overall,
        pval=pval,
        **kwargs,
    )


# ------------------------
# Quality of life improvements
# ------------------------
def absolute_and_relative_value_counts(
    df: pl.LazyFrame | pl.DataFrame,
    cols: list[str] | str,
    sort: Literal["cols", "counts"] | None = "counts",
    decimals: int = 2,
) -> pl.DataFrame:
    """Returns the absolute and relative value_counts for the column(s) `cols` of a `pl.LazyFrame` or `pl.DataFrame`."""
    df = df.lazy() if isinstance(df, pl.DataFrame) else df

    value_counts = (
        df.group_by(cols)
        .agg(
            pl.len().alias("absolute (int)"),
            (100 * pl.len() / df.select(pl.len()).collect().item())
            .round(decimals)
            .alias("relative (%)"),
        )
        .collect()
    )
    if sort == "cols":
        value_counts = value_counts.sort(cols)
    elif sort == "counts":
        value_counts = value_counts.sort("absolute (int)", descending=True)
    return value_counts


# ------------------------
# Expression pipes
# ------------------------
def unique_sucessive(value: pl.Expr | str) -> pl.Expr:
    """Drop sucessive duplicate values of a list column."""
    return as_expr(value).list.filter(
        pl.element().ne(pl.element().shift(1)) | pl.element().shift(1).is_null()
    )


def time_difference(
    time_col: pl.Expr | str,
    reference_col: pl.Expr | str,
    *,
    unit: TimeDifferenceUnit = "s",
    total: bool = True,
) -> pl.Expr:
    """Signed time difference ``time_col - reference_col`` as a numeric expression.

    Computes the elapsed time from *reference_col* to *time_col* — positive when
    *time_col* is the later timestamp, negative otherwise — and converts it to the
    requested *unit*. Both operands may be column names or expressions, and the
    result is a :class:`polars.Expr`, so it composes inside ``select`` /
    ``with_columns``.

    Args:
        time_col: The (typically later) timestamp, as a column name or expression.
        reference_col: The timestamp subtracted from *time_col*, as a column name or
            expression.
        unit: Output unit — one of ``"s"`` (seconds), ``"m"`` (minutes), ``"h"``
            (hours), ``"d"`` (days = 24 hours), ``"w"`` (weeks = 7 days). Defaults
            to ``"s"``.
        total: How sub-unit remainders are handled for non-second units. ``True``
            (default) returns whole units via floor division; ``False`` returns the
            fractional value via true division. Ignored for ``unit="s"``, which
            always yields total whole seconds.

    Returns:
        A :class:`polars.Expr` yielding the signed difference expressed in *unit*.

    Raises:
        ValueError: If *unit* is not one of ``"s"``, ``"m"``, ``"h"``, ``"d"``, ``"w"``.

    Example:
        .. code-block:: python

            # Whole hours from ICU admission to first lab
            time_difference("first_lab_time", "icu_admission", unit="h")

            # Fractional days of ICU stay
            time_difference("icu_discharge", "icu_admission", unit="d", total=False)
    """
    time = as_expr(time_col)
    reference = as_expr(reference_col)
    expr = time.sub(reference).dt.total_seconds()
    if unit == "s":
        return expr

    conversions = {
        "m": 60,
        "h": 60 * 60,
        "d": 24 * 60 * 60,
        "w": 7 * 24 * 60 * 60,
    }
    if unit not in conversions:
        raise ValueError(f"Invalid unit: {unit}")

    def op(e: pl.Expr, v: int) -> pl.Expr:
        return e.floordiv(v) if total else e.truediv(v)

    return op(expr, conversions[unit])


# ------------------------
# DataFrame / LazyFrame pipes
# ------------------------
@overload
def remove_asof(
    main: pl.DataFrame,
    ref: pl.DataFrame,
    strategy: AsofJoinStrategy = ...,
    by: str = ...,
    on: str = ...,
    tolerance: str | timedelta = ...,
) -> pl.DataFrame: ...


@overload
def remove_asof(
    main: pl.LazyFrame,
    ref: pl.LazyFrame,
    strategy: AsofJoinStrategy = ...,
    by: str = ...,
    on: str = ...,
    tolerance: str | timedelta = ...,
) -> pl.LazyFrame: ...


def remove_asof(
    main: PolarsFrame,
    ref: PolarsFrame,
    strategy: AsofJoinStrategy = "nearest",
    by: str = "case_id",
    on: str = "recordtime",
    tolerance: str | timedelta = "1d",
) -> PolarsFrame:
    """Remove every *main* entry that has a *ref* entry within *tolerance*.

    Rows are keyed by *by* and ordered by *on*.  A *main* row is dropped when at
    least one *ref* row (same *by*) lies within *tolerance* of it, following
    *strategy* (``"backward"`` / ``"forward"`` restrict the match to ref rows
    before / after the main row; ``"nearest"`` allows either direction).  Used to
    prefer a higher-priority source whenever it has a nearby measurement.

    Unlike a plain as-of join — which pairs each row only once — this removes
    *all* matching *main* rows and is therefore idempotent.
    """
    if type(main) is not type(ref):
        raise TypeError("Types of main and ref are not the same.")

    marker = "__ref_match"
    ref_marked = (
        ref.sort(by, on).select(by, on).with_columns(pl.lit(True).alias(marker))
    )
    # As-of join main -> ref: each main row is matched to its nearest ref within
    # tolerance.  A match exists iff *any* ref row is within tolerance, so the
    # unmatched rows are exactly those to keep.
    matched = main.sort(by, on).join_asof(
        ref_marked,  # type: ignore[arg-type]
        on=on,
        by=by,
        strategy=strategy,
        tolerance=tolerance,
    )
    return matched.filter(pl.col(marker).is_null()).drop(marker)


def threshold_score(
    value: pl.Expr | str,
    thresholds: Sequence[tuple[float, int]],
    *,
    below: int,
    dtype: PolarsDataType | pl.DataTypeExpr = pl.Int8,
) -> pl.Expr:
    """Map a measurement onto the points of a threshold table.

    *thresholds* are ``(lower_bound, points)`` pairs in **descending** bound
    order; the first bound the value reaches (``value >= bound``) wins, and
    *below* applies underneath the last one.  Null values score null, so a
    missing measurement is never silently treated as normal.

    Clinical score tables are usually written with gaps (``38.5 - 38.9``,
    ``39.0 - 40.9``); expressing them as half-open thresholds closes those gaps
    and keeps the chain readable.

    Args:
        value: Column name or expression holding the measurement.
        thresholds: ``(lower_bound, points)`` pairs, highest bound first.
        below: Points awarded below the lowest bound.
        dtype: Output dtype of the points.

    Returns:
        Expression yielding the points for each row.

    Examples:
        >>> # APACHE II serum potassium
        >>> threshold_score(
        ...     "value", [(7, 4), (6, 3), (5.5, 1), (3.5, 0), (3, 1), (2.5, 2)], below=4
        ... )
    """
    value = as_expr(value)
    chain = pl.when(value.ge(thresholds[0][0])).then(pl.lit(thresholds[0][1], dtype))
    for bound, points in thresholds[1:]:
        chain = chain.when(value.ge(bound)).then(pl.lit(points, dtype))
    return chain.when(value.is_not_null()).then(pl.lit(below, dtype)).otherwise(None)


@overload
def attach_asof_value(
    main: pl.DataFrame,
    ref: pl.DataFrame,
    *,
    name: str,
    by: str = ...,
    on: str = ...,
    value: str = ...,
    strategy: AsofJoinStrategy = ...,
    tolerance: str | timedelta | None = ...,
    with_matched_time: bool = ...,
) -> pl.DataFrame: ...


@overload
def attach_asof_value(
    main: pl.LazyFrame,
    ref: pl.LazyFrame,
    *,
    name: str,
    by: str = ...,
    on: str = ...,
    value: str = ...,
    strategy: AsofJoinStrategy = ...,
    tolerance: str | timedelta | None = ...,
    with_matched_time: bool = ...,
) -> pl.LazyFrame: ...


def attach_asof_value(
    main: PolarsFrame,
    ref: PolarsFrame,
    *,
    name: str,
    by: str = "case_id",
    on: str = "recordtime",
    value: str = "value",
    strategy: AsofJoinStrategy = "backward",
    tolerance: str | timedelta | None = "1h",
    with_matched_time: bool = False,
) -> PolarsFrame:
    """Attach the as-of matching *ref* measurement to every *main* row.

    Rows are keyed by *by* and ordered by *on*.  Each *main* row is paired with
    at most one *ref* row (same *by*) within *tolerance*, following *strategy*
    (``"backward"`` matches the last ref row at or before the main row,
    ``"forward"`` the next one, ``"nearest"`` either direction — an exact
    timestamp match always wins, since its distance is zero).

    The matched measurement lands in *name*, null where nothing matched.  Set
    *with_matched_time* to also keep the timestamp it was measured at, as
    ``f"{name}_{on}"`` — useful to check how far a match had to reach, but off
    by default so the result gains exactly one column, like ``join_asof``
    itself, which does not return the right-hand key either.

    Args:
        main: Frame whose rows are enriched.
        ref: Frame supplying the matched measurements.
        name: Output column name for the matched value.
        by: Grouping key both frames are matched within.
        on: Ordering (time) column present in both frames.
        value: Column of *ref* to attach.
        strategy: As-of join direction.
        tolerance: Maximum distance of a match, ``None`` for unbounded.
        with_matched_time: Whether to also add the matched row's timestamp.

    Returns:
        *main* with the additional column(s), sorted by *by*, *on*.
    """
    if type(main) is not type(ref):
        raise TypeError("Types of main and ref are not the same.")

    matched_on = f"{name}_{on}"
    added = {name, matched_on} if with_matched_time else {name}
    conflicts = set(columns(main)).intersection(added)
    if conflicts:
        raise ValueError(f"Columns {sorted(conflicts)} already exist in main.")

    ref_renamed = ref.sort(by, on).select(
        by,
        on,
        pl.col(value).alias(name),
        *([pl.col(on).alias(matched_on)] if with_matched_time else []),
    )
    return main.sort(by, on).join_asof(
        ref_renamed,  # type: ignore[arg-type]
        on=on,
        by=by,
        strategy=strategy,
        tolerance=tolerance,
    )


def find_nearest(
    df: pl.DataFrame,
    time: TimeAnchorColumn,
    pm_before: OffsetLike = "52w",
    pm_after: OffsetLike = "52w",
    recordtime: str = "recordtime",
) -> pl.DataFrame:
    """Find the closest record to the target time.

    Args:
        df (pl.DataFrame): The input dataframe.
        to_col (str): The column containing the target time.
        tdelta (str): The time delta. Defaults to "0".
        pm_before (OffsetLike): The time range before the target time. Defaults to
            ``"52w"``. A polars-duration string or a timedelta.
        pm_after (OffsetLike): The time range after the target time. Defaults to
            ``"52w"``.

    Returns:
        pl.DataFrame: The closest record.
    """
    from corr_vars.utils.time import TimeAnchor

    before = normalize_offset(pm_before).lstrip("+")
    after = normalize_offset(pm_after)
    target_time_anchor = TimeAnchor(time)
    time_range = pl.col(recordtime).sub(target_time_anchor.expr).abs()
    within_range = pl.col(recordtime).is_between(
        target_time_anchor.expr.dt.offset_by(f"-{before}"),
        target_time_anchor.expr.dt.offset_by(after),
    )

    filtered_df = df.filter(within_range)
    if filtered_df.is_empty():
        return filtered_df

    return filtered_df.sort(time_range).head(1)


def apply_cleaning(df: PolarsFrame, cleaning: CleaningDict) -> PolarsFrame:
    """Apply cleaning rules to filter out invalid values.

    Args:
        df (pl.DataFrame | pl.LazyFrame): The input dataframe.
        cleaning (dict[str, dict[str, float]]): A dictionary where keys are column names and values are dictionaries with optional "low" and "high" keys specifying the bounds for valid values.

    Returns:
        cleaned (pl.DataFrame | pl.LazyFrame): Cleaned DataFrame
    """
    if not cleaning or is_empty(df):
        return df

    for col, bounds in cleaning.items():
        if "low" in bounds:
            df = df.filter(pl.col(col).ge(bounds["low"]))  # type: ignore[assignment]
        if "high" in bounds:
            df = df.filter(pl.col(col).le(bounds["high"]))  # type: ignore[assignment]

    return df


def build_attributes(data: PolarsFrame) -> PolarsFrame:
    """Combines columns prefixed by ``attributes_`` into a struct column called ``attributes``.

    Args:
        data (pl.DataFrame | pl.LazyFrame): The input dataframe.

    Returns:
        pl.DataFrame | pl.LazyFrame: The input dataframe with the new ``attributes`` struct column and without the original ``attributes_`` columns.
    """
    attr_cols = [col for col in columns(data) if col.startswith("attributes_")]

    if attr_cols:
        struct_expr = {
            col.removeprefix("attributes_"): pl.col(col) for col in attr_cols
        }
        # Mypy does not recognise the kwargs correctly as an overload
        data = data.with_columns(pl.struct(**struct_expr).alias("attributes"))  # type: ignore[call-overload, assignment]
        data = data.drop(attr_cols)  # type: ignore[assignment]

    return data


def unnest(
    value: PolarsFrame,
    column: ColumnNameOrSelector,
    prefix: str = "",
    suffix: str = "",
    renamer: Sequence[str] | Callable[[str], str] | None = None,
) -> PolarsFrame:
    """Unnest a struct column.

    Args:
        value (pl.DataFrame | pl.LazyFrame): The input dataframe.
        column (str): The column to unnest.
        prefix (str): The prefix to add to the unnested column names.
        suffix (str): The suffix to add to the unnested column names.
        renamer (Sequence[str] | Callable[[str], str] | None): The renamer function or list of new names.

    Returns:
        value (pl.DataFrame | pl.LazyFrame): The input dataframe with the unnested column.
    """
    parsed_renamer = renamer or (lambda s: s)
    fields = struct_fields(value, column)
    renamed_fields = (
        [f"{prefix}{parsed_renamer(f)}{suffix}" for f in fields]
        if callable(parsed_renamer)
        else parsed_renamer
    )
    unnested = value.with_columns(
        pl.col(column).struct.rename_fields(renamed_fields)  # type: ignore[arg-type]
    ).unnest(column)
    return cast("PolarsFrame", unnested)


def interval_bucket_agg(
    data: PolarsFrame,
    primary_key: str = "case_id",
    recordtime: str = "recordtime",
    recordtime_end: str | None = None,
    recordtime_end_gap_treshold: str | None = None,
    recordtime_end_gap_behavior: Literal["impute", "drop"] = "drop",
    duration_empty_impute_duration: str | None = None,
    value: str = "value",
    tmin: str | None = None,
    tmax: str | None = None,
    t0: str | None = None,
    t_missing_behavior: Literal["keep", "drop"] = "drop",
    granularity: str | None = "1h",
    same_bin_aggregation: Callable[[pl.Expr], pl.Expr] | None = None,
) -> PolarsFrame:
    """Bins intervals clipped between tmin and tmax into bins aligned to t0 and aggregates on these bins.

    Args:
        data (pl.DataFrame | pl.LazyFrame): The data to transform.
        primary_key (str): The primary key column. Defaults to "case_id".
        recordtime (str): The recordtime column. Defaults to "recordtime".
        recordtime_end (str | None): The recordtime_end column. Defaults to `None`. Will impute recordtime_end from the next recordtime if set to `None`.
        recordtime_end_gap_treshold (str | None): Maximum allowed gap for recordtime_end. Only applies if recordtime_end is set to `None`.
        recordtime_end_gap_behavior (str | None): Determines whether to impute or drop rows when the `recordtime_end_gap_treshold` is exceeded.
        duration_empty_impute_duration (str | None): The duration to be imputed by offsetting recordtime_end. Drops rows if set to default of `None` or ≤0s.
        value (str): The value column. Defaults to "value".
        tmin (str | None): The tmin column. Defaults to `None`.
        tmax (str | None): The tmax column. Defaults to `None`.
        t0 (str | None): The t0 column. Bins will align to this datetime col if specified. Defaults to `None`.
        t_missing_behavior (str | None): Determines whether to drop or keep rows when `t0`, `tmin` or `tmax` columns are specified and values are missing in `data`. Defaults to "drop".
        granularity (str | None): The size of the bucket intervals. Defaults to 1 hour.
        same_bin_aggregation: (Callable[[pl.Expr], pl.Expr] | None): The aggregation to perform on `value` for each bin. Uses `pl.sum()` if set to `None`.

    .. Note::
        The `granularity`, `recordtime_end_impute_gap_treshold` and `duration_empty_impute_duration` arguments are created with
        the following string language:

        - 1ns   (1 nanosecond) # nanoseconds are not supported by our DataFrames.
        - 1us   (1 microsecond)
        - 1ms   (1 millisecond)
        - 1s    (1 second)
        - 1m    (1 minute)
        - 1h    (1 hour)
        - 1d    (1 calendar day)
        - 1w    (1 calendar week)
        - 1mo   (1 calendar month)
        - 1q    (1 calendar quarter)
        - 1y    (1 calendar year)

        Or combine them: "3d12h4m25s" # 3 days, 12 hours, 4 minutes, and 25 seconds

        By "calendar day", we mean the corresponding time on the next day (which may
        not be 24 hours, due to daylight savings). Similarly for "calendar week",
        "calendar month", "calendar quarter", and "calendar year".

        If you do not wish to use "calendar day" or any other "calendar" durations, specify the duration in hours or smaller units instead.

    Returns:
        clip_bin_data (pl.DataFrame | pl.LazyFrame): The clipped and binned data. Same type as `data`.
    """
    lazy_input = data if isinstance(data, pl.LazyFrame) else data.lazy()

    START = "start_time"
    END = "end_time"
    DURATION = "duration_time"
    VALUE = "value"

    TMIN = "tmin"
    TMAX = "tmax"
    T0 = "t0"

    START_CLIPPED_TMIN_TMAX = f"clipped_tmin_tmax_{START}"
    END_CLIPPED_TMIN_TMAX = f"clipped_tmin_tmax_{END}"
    DURATION_CLIPPED_TMIN_TMAX = f"clipped_tmin_tmax_{DURATION}"
    VALUE_CLIPPED_TMIN_TMAX = f"clipped_tmin_tmax_{VALUE}"

    START_SPLIT = f"split_{START}"
    END_SPLIT = f"split_{END}"

    START_OVERLAP = f"overlap_{START}"
    END_OVERLAP = f"overlap_{END}"
    DURATION_OVERLAP = f"overlap_{DURATION}"
    VALUE_OVERLAP = f"overlap_{VALUE}"

    BEFORE_OR_AFTER_T0_SIGNED = "t0_side_signed"
    T0_GRANULARITY_DISTANCE = "t0_granularity_distance"

    AGG_START_OVERLAP = f"first_{START_OVERLAP}"
    AGG_END_OVERLAP = f"last_{END_OVERLAP}"
    AGG_DURATION_OVERLAP_S = f"max_{DURATION_OVERLAP}_s"
    AGG_VALUE = f"agg_{VALUE}"

    ATTRIBUTES = "attributes"

    # Select required columns from input and rename them
    # Fill recordtime_end, tmin, tmax and t0 with None if unspecified
    lf = lazy_input.select(
        primary_key,
        pl.col(recordtime).alias(START),
        (
            pl.lit(None).alias(END)
            if recordtime_end is None
            else pl.col(recordtime_end).alias(END)
        ),
        pl.col(value).alias(VALUE),
        pl.lit(None).alias(TMIN) if tmin is None else pl.col(tmin).alias(TMIN),
        pl.lit(None).alias(TMAX) if tmax is None else pl.col(tmax).alias(TMAX),
        pl.lit(None).alias(T0) if t0 is None else pl.col(t0).alias(T0),
    )

    # impute recordtime_end if necessary
    if recordtime_end is None:
        # Sort and calculate end_time by shifting over partition
        lf = lf.sort(primary_key, START).with_columns(
            # NOTE: Create nulls for the last entry of each partition
            # Will be dropped in the next step
            pl.col(START)
            .shift(-1)
            .over(primary_key)
            .alias(END)
        )

        # Enforce max gap treshold
        if recordtime_end_gap_treshold:
            if recordtime_end_gap_behavior == "drop":
                lf = lf.remove(
                    pl.col(START)
                    .dt.offset_by(recordtime_end_gap_treshold)
                    .lt(pl.col(END))
                )
            else:
                lf = lf.with_columns(
                    pl.min_horizontal(
                        pl.col(START).dt.offset_by(recordtime_end_gap_treshold),
                        pl.col(END),
                    ).alias(END)
                )

    # Drop nulls and calculate duration
    lf = lf.drop_nulls(
        [primary_key, START, END, VALUE]
        + ([TMIN] if tmin is not None and t_missing_behavior == "drop" else [])
        + ([TMAX] if tmax is not None and t_missing_behavior == "drop" else [])
        + ([T0] if t0 is not None and t_missing_behavior == "drop" else [])
    ).with_columns(pl.col(END).sub(pl.col(START)).alias(DURATION))

    if duration_empty_impute_duration:
        lf = lf.with_columns(
            pl.when(pl.col(DURATION).eq(0))
            .then(pl.col(END).dt.offset_by(duration_empty_impute_duration))
            .otherwise(pl.col(END))
            .alias(END)
        )

    # Filter out empty intervals and intervals not intersecting [tmin, tmax]
    lf_clipped = (
        lf.filter(
            pl.col(DURATION).gt(0)
            & pl.col(START).lt(pl.col(TMAX)).or_(pl.col(TMAX).is_null())
            & pl.col(END).gt(pl.col(TMIN)).or_(pl.col(TMIN).is_null())
        )
        .with_columns(
            pl.col(START).clip(TMIN, TMAX).alias(START_CLIPPED_TMIN_TMAX),
            pl.col(END).clip(TMIN, TMAX).alias(END_CLIPPED_TMIN_TMAX),
        )
        .drop(TMIN, TMAX)
        .with_columns(
            pl.col(END_CLIPPED_TMIN_TMAX)
            .sub(pl.col(START_CLIPPED_TMIN_TMAX))
            .alias(DURATION_CLIPPED_TMIN_TMAX)
        )
    )

    # Calculate corrected values for intervals clipped to [tmin, tmax]
    # Skip this step when tmin or tmax unspecified for performance
    # It is practically a no-op in that case
    corrected_value_expr: pl.Expr
    if tmin is not None or tmax is not None:
        corrected_value_expr = (
            pl.col(VALUE)
            .cast(pl.Float64)
            .mul(pl.col(DURATION_CLIPPED_TMIN_TMAX))
            .truediv(pl.col(DURATION))
        )
    else:
        corrected_value_expr = pl.col(VALUE).cast(pl.Float64)

    lf_clipped = (
        lf_clipped
        # Shouldn't be possible after the previous filter step
        .filter(pl.col(DURATION_CLIPPED_TMIN_TMAX).gt(0)).with_columns(
            corrected_value_expr.alias(VALUE_CLIPPED_TMIN_TMAX)
        )
    )

    # NOTE: Empty string is also not valid
    correction_expr: pl.Expr
    datetime_ranges_expr: pl.Expr
    intersection_condition_expr: pl.Expr
    lf_distributed: pl.LazyFrame

    if granularity:
        # Use t0 as relative anchor for the buckets
        if t0 is not None:
            # Apply correction offset to align bins to a grid starting from t0
            correction_expr = pl.col(T0).sub(pl.col(T0).dt.truncate(granularity))
            datetime_ranges_expr = pl.datetime_ranges(
                pl.col(START_CLIPPED_TMIN_TMAX)
                .sub(correction_expr)
                .dt.truncate(granularity),
                pl.col(END_CLIPPED_TMIN_TMAX).sub(correction_expr),
                granularity,
            )

            lf_distributed = (
                # Generate buckets
                lf_clipped.with_columns(
                    datetime_ranges_expr.alias(START_SPLIT)
                ).explode(START_SPLIT)
                # Revert correction offset
                .with_columns(
                    pl.col(START_SPLIT).add(correction_expr),
                )
            )

        # Snap buckets to previous grid point
        else:
            datetime_ranges_expr = pl.datetime_ranges(
                pl.col(START_CLIPPED_TMIN_TMAX).dt.truncate(granularity),
                END_CLIPPED_TMIN_TMAX,
                granularity,
            )

            lf_distributed = (
                # Generate buckets
                lf_clipped.with_columns(
                    datetime_ranges_expr.alias(START_SPLIT)
                ).explode(START_SPLIT)
            )

        lf_distributed = (
            lf_distributed
            # Calculate end_granularity
            .with_columns(
                pl.col(START_SPLIT).dt.offset_by(granularity).alias(END_SPLIT),
            )
        )
    # Split intervals containing t0 into two intervals
    elif t0 is not None:
        # Find intervals containing t0
        intersection_condition_expr = (
            pl.col(START_CLIPPED_TMIN_TMAX)
            .lt(pl.col(T0))
            .and_(pl.col(END_CLIPPED_TMIN_TMAX).gt(pl.col(T0)))
        )

        # Split intervals containing t0
        lf_distributed = lf_clipped.with_columns(
            pl.when(intersection_condition_expr)
            .then(
                pl.concat_list(
                    pl.col(START_CLIPPED_TMIN_TMAX).cast(pl.List(pl.Datetime)),
                    pl.col(T0).cast(pl.List(pl.Datetime)),
                )
            )
            .otherwise(pl.col(START_CLIPPED_TMIN_TMAX).cast(pl.List(pl.Datetime)))
            .alias(START_SPLIT),
            pl.when(intersection_condition_expr)
            .then(
                pl.concat_list(
                    pl.col(T0).cast(pl.List(pl.Datetime)),
                    pl.col(END_CLIPPED_TMIN_TMAX).cast(pl.List(pl.Datetime)),
                )
            )
            .otherwise(pl.col(END_CLIPPED_TMIN_TMAX).cast(pl.List(pl.Datetime)))
            .alias(END_SPLIT),
        ).explode(START_SPLIT, END_SPLIT)

    # Do nothing: Split times are equal to previous times and not exploded
    else:
        # Use clipped times as bucket times
        lf_distributed = (
            # Generate buckets
            lf_clipped.with_columns(
                pl.col(START_CLIPPED_TMIN_TMAX).alias(START_SPLIT),
                pl.col(END_CLIPPED_TMIN_TMAX).alias(END_SPLIT),
            )
        )

    # Calculate corrected values for intervals clipped to [tmin, tmax]
    # Skip this step when tmin or tmax unspecified for performance
    # It is practically a no-op in that case
    if granularity or t0 is not None:
        corrected_value_expr = (
            pl.col(VALUE_CLIPPED_TMIN_TMAX)  # Already cast to Float64
            .mul(pl.col(DURATION_OVERLAP))
            .truediv(pl.col(DURATION_CLIPPED_TMIN_TMAX))
        )
    else:
        corrected_value_expr = pl.col(VALUE_CLIPPED_TMIN_TMAX)

    lf_distributed = (
        lf_distributed.with_columns(
            pl.max_horizontal(START_CLIPPED_TMIN_TMAX, START_SPLIT).alias(
                START_OVERLAP
            ),
            pl.min_horizontal(END_CLIPPED_TMIN_TMAX, END_SPLIT).alias(END_OVERLAP),
        )
        .with_columns(
            pl.col(END_OVERLAP).sub(pl.col(START_OVERLAP)).alias(DURATION_OVERLAP)
        )
        .with_columns(corrected_value_expr.alias(VALUE_OVERLAP))
        .select(
            primary_key,
            START_SPLIT,
            END_SPLIT,
            T0,
            START_OVERLAP,
            END_OVERLAP,
            VALUE_OVERLAP,
        )
    )

    # Importantly this data is not yet aggregated on the intervals...
    # There can be multiple entries per interval

    # Aggregate interval
    def agg_interval(
        df: pl.LazyFrame, more_group_by_cols: Iterable[pl.Expr | str] = []
    ):
        return (
            df.group_by(primary_key, *more_group_by_cols)
            .agg(
                pl.min(START_OVERLAP).alias(AGG_START_OVERLAP),
                pl.max(END_OVERLAP).alias(AGG_END_OVERLAP),
                (
                    pl.col(VALUE_OVERLAP).pipe(same_bin_aggregation).alias(AGG_VALUE)
                    if same_bin_aggregation
                    else pl.col(VALUE_OVERLAP).sum().alias(AGG_VALUE)
                ),
            )
            # Recalculate overlap_duration_s with new aggregated timestamps
            .with_columns(
                pl.col(AGG_END_OVERLAP)
                .sub(pl.col(AGG_START_OVERLAP))
                .dt.total_seconds()
                .alias(AGG_DURATION_OVERLAP_S)
            )
            .sort(primary_key, AGG_START_OVERLAP)
        )

    group_cols: list[str | pl.Expr] = []
    if granularity:
        group_cols += [START_SPLIT, END_SPLIT] + ([T0] if t0 is not None else [])
    elif t0 is not None:
        group_cols += [
            pl.when(pl.col(START_SPLIT).lt(pl.col(T0)))
            .then(pl.lit(-1))
            .otherwise(pl.lit(1))
            .alias(BEFORE_OR_AFTER_T0_SIGNED),
        ]

    agg_df = lf_distributed.pipe(agg_interval, group_cols)

    extra_attributes: list[pl.Expr | str] = [
        AGG_START_OVERLAP,
        AGG_END_OVERLAP,
        AGG_DURATION_OVERLAP_S,
    ]
    if granularity and t0 is not None:
        extra_attributes += [
            pl.col(START_SPLIT)
            .sub(pl.col(T0))
            .truediv(pl.col(END_SPLIT).sub(pl.col(START_SPLIT)))
            .alias(T0_GRANULARITY_DISTANCE)
        ]

    elif t0 is not None:
        extra_attributes += [BEFORE_OR_AFTER_T0_SIGNED]

    agg_df_packed = agg_df.with_columns(
        pl.struct(extra_attributes).alias(ATTRIBUTES)
    ).select(primary_key, *group_cols, AGG_VALUE, ATTRIBUTES)

    return agg_df_packed if isinstance(data, pl.LazyFrame) else agg_df_packed.collect()
