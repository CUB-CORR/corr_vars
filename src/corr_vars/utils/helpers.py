from __future__ import annotations

import warnings
from difflib import get_close_matches
from itertools import chain

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

from corr_vars import __, logger
from corr_vars.definitions import (
    IntoExprColumn,
    PandasDataframe,
    PolarsDataframe,
    PolarsFrame,
    PolarsLazyframe,
)

from collections.abc import Iterable, Mapping, MutableMapping
from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

if TYPE_CHECKING:
    from pandas.core.groupby import DataFrameGroupBy

    from corr_vars.utils.time import TimeWindow

A = TypeVar("A")
T = TypeVar("T", bound=MutableMapping[str, Any])
F = TypeVar("F", bound=PandasDataframe | PolarsDataframe | PolarsLazyframe)

__all__ = [
    "get_cb",
    "add_time_window_expr",
    "harmonize_str_list_cols",
    "deep_merge",
    "flatten_args",
    "raise_error_with_nearest_matches",
    # TODO: Remove/migrate the functions below
    "merge_consecutive",
    "clean_timeseries",
    "extract_df_data",
    "filter_by_condition",
    "df_find_closest",
    "aggregate_column",
]


@overload
def get_cb(
    df: PolarsDataframe,
    join_key: str = ...,
    primary_key: str = ...,
    tmin_col: str = ...,
    tmax_col: str = ...,
    ttarget_col: str | None = ...,
    aliases: tuple[str, str] = ...,
) -> PolarsDataframe: ...


@overload
def get_cb(
    df: PandasDataframe,
    join_key: str = ...,
    primary_key: str = ...,
    tmin_col: str = ...,
    tmax_col: str = ...,
    ttarget_col: str | None = ...,
    aliases: tuple[str, str] = ...,
) -> PandasDataframe: ...


def get_cb(
    df: PandasDataframe | PolarsDataframe,
    join_key: str = "case_id",
    primary_key: str = "case_id",
    tmin_col: str = "case_tmin",
    tmax_col: str = "case_tmax",
    ttarget_col: str | None = None,
    aliases: tuple[str, str] = ("case_tmin", "case_tmax"),
) -> PandasDataframe | PolarsDataframe:
    """Get the case bounds Dataframe. This is used to apply a time filter to a native dynamic variable.

    Args:
        df (pd.DataFrame | pl.DataFrame): The cohort dataframe.
        join_key (str): The join key column. Defaults to "case_id".
        primary_key (str): The primary key column. Defaults to "case_id".
        tmin_col (str): The column containing the tmin. Defaults to "tmin".
        tmax_col (str): The column containing the tmax. Defaults to "tmax".
        ttarget_col (str | None): The column containing the target time. Defaults to None. (For !closest)
        aliases (tuple[str, str]): The aliases for tmin and tmax. Defaults to ("tmin", "tmax").

    Returns:
        result: The case bounds Dataframe with the same type as `df`.
    """
    # Build columns list
    columns = [join_key, tmin_col, tmax_col]
    if ttarget_col:
        columns.append(ttarget_col)
    if primary_key != join_key:
        columns.append(primary_key)

    # Create renamer
    tmin_alias, tmax_alias = aliases
    column_rename = {tmin_col: tmin_alias, tmax_col: tmax_alias}

    # Select columns and rename
    if isinstance(df, pl.DataFrame):
        return df.select(columns).rename(column_rename)
    return df[columns].rename(columns=column_rename)


# "Compiles" and adds a tmin/tmax expressions from TimeWindo to a DataFrame
# Used for cohort timefiltering
def add_time_window_expr(
    obs: PolarsFrame,
    time_window: TimeWindow,
    aliases: tuple[str, str] | None = ("case_tmin", "case_tmax"),
) -> PolarsFrame:
    if aliases:
        tmin_alias, tmax_alias = aliases
        obs = obs.with_columns(
            time_window.filled_tmin_expr.alias(tmin_alias),
            time_window.filled_tmax_expr.alias(tmax_alias),
        )  # type: ignore[assignment]
    else:
        obs = obs.with_columns(
            time_window.filled_tmin_expr, time_window.filled_tmax_expr
        )  # type: ignore[assignment]
    return obs


# Used to harmonize obs dfs when loading cohorts from different sources
def harmonize_str_list_cols(dfs: Iterable[pl.DataFrame]) -> list[pl.DataFrame]:
    """Harmonises columns with str and list[str] dtypes in different DataFrames
    by converting str to list[str] as well.
    """
    cols = set().union(*(df.columns for df in dfs))
    if not cols:
        raise ValueError("All dataframes are empty.")

    harmonized = []
    for col in cols:
        types = {df.schema[col] for df in dfs if col in df.columns}
        if pl.String in types and pl.List(pl.String) in types:
            harmonized = [
                df.with_columns(
                    pl.col(col).cast(pl.List(pl.String), strict=False).alias(col)
                    if col in df.columns
                    else []
                )
                for df in dfs
            ]
    return harmonized or list(dfs)


# Used to merge overrides for default cohort source configurations
def deep_merge(base: T, override: Mapping[str, Any]) -> T:
    """Recursively merge `override` into `base` with override taking precedence.
    Works on normal dictionaries, compatible with TypedDict runtime dicts.
    """
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], MutableMapping)
            and isinstance(value, Mapping)
        ):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


# Flattens arguments similar to polars
def flatten_args(*args: A | Iterable[A]) -> chain[A]:
    return chain.from_iterable(
        [k] if not isinstance(k, Iterable) or isinstance(k, (str, bytes)) else k
        for k in args
    )  # type: ignore


# Used to suggest nearest columns when a column is missed in obs/obsm
def raise_error_with_nearest_matches(
    item: str | list[str],
    available_items: list[str],
    label: str = "Item",
    n: int = 3,
    cutoff: float = 0.6,
    error_cls: type[Exception] = LookupError,
):
    """Finds closest matches and raises a descriptive error."""
    packed_items = item if isinstance(item, list) else [item]

    suggestions = {}
    for it in packed_items:
        if it not in available_items:
            matches = get_close_matches(it, available_items, n=n, cutoff=cutoff)
            if matches:
                suggestions[it] = matches

    if suggestions:
        msg = "\n".join(
            [
                f"{label.capitalize()} '{k}' not found. Did you mean: {', '.join(v)}?"
                for k, v in suggestions.items()
            ]
        )
        raise error_cls(msg)


#################################################
#                 Pandas pipes                  #
#################################################


def merge_consecutive(
    data: pd.DataFrame,
    primary_key: str,  # Deprecated!
    recordtime: str = "recordtime",
    recordtime_end: str = "recordtime_end",
    time_threshold: pd.Timedelta = pd.Timedelta(minutes=30),
) -> pd.Series:
    """Combine consecutive sessions (<30min separation) of ecmo_vv_icu into a single session.

    Args:
        data (pd.DataFrame): The data to merge.
        primary_key (str): The primary key column. Deprecated! Will not be used, case_id will be used instead.
        recordtime (str): The recordtime column. Defaults to "recordtime".
        recordtime_end (str): The recordtime_end column. Defaults to "recordtime_end".
        time_threshold (pd.Timedelta): The time threshold. Defaults to 30 minutes.

    Returns:
        pd.DataFrame: The merged data.
    """
    # Sort by primary_key and recordtime

    data = data.sort_values(by=["case_id", recordtime]).reset_index(drop=True)

    def combine_consecutive_sessions(group):
        group["last_event_recordtime_end"] = group["recordtime_end"].shift(1)
        group["time_since_last_event"] = (
            group["recordtime"] - group["last_event_recordtime_end"]
        )
        group["merge_with_previous"] = group["time_since_last_event"] < time_threshold

        group["merge_group"] = (~group["merge_with_previous"]).cumsum()

        return (
            group.groupby(by=["case_id", "merge_group"])
            .agg({"value": "first", "recordtime": "first", "recordtime_end": "last"})
            .reset_index()
        )

    grouped = data.groupby("case_id")
    grouped = grouped.apply(combine_consecutive_sessions)

    return grouped.reset_index(drop=True)


@overload
def clean_timeseries(
    data: PolarsDataframe,
    max_roc: float,
    time_col: str = ...,
    value_col: str = ...,
    identifier_col: str = ...,
) -> PolarsDataframe: ...


@overload
def clean_timeseries(
    data: PandasDataframe,
    max_roc: float,
    time_col: str = ...,
    value_col: str = ...,
    identifier_col: str = ...,
) -> PandasDataframe: ...


def clean_timeseries(
    data: PandasDataframe | PolarsDataframe,
    max_roc: float,
    time_col: str = "recordtime",
    value_col: str = "value",
    identifier_col: str = "icu_stay_id",
) -> PandasDataframe | PolarsDataframe:
    """Clean timeseries by removing values that create impossible rates of change
    using vectorized operations with Polars for maximum performance.

    Args:
        data (pl.DataFrame): The data to clean.
        max_roc (float): The maximum rate of change.
        time_col (str): The time column.
        value_col (str): The value column.
        identifier_col (str): The identifier column.

    Returns:
        result: The cleaned data with the same type as `data`.
    """
    return_pandas = False
    # Convert to Polars if necessary
    if isinstance(data, pd.DataFrame):
        data = pl.from_pandas(data)
        return_pandas = True

    def clean_group(group):
        if len(group) <= 1:
            return group.with_columns(pl.lit(None).alias("current_roc"))

        group = group.sort(time_col)

        times = group.select(time_col).to_numpy().flatten()
        values = group.select(value_col).to_numpy().flatten()

        keep_mask = np.ones(len(group), dtype=bool)

        i = 1
        while i < len(group):
            last_kept_idx = np.where(keep_mask[:i])[0][-1]

            time_diff = (
                times[i] - times[last_kept_idx]
            ) / 3600  # Convert seconds to hours
            value_diff = values[i] - values[last_kept_idx]
            current_roc = abs(value_diff / time_diff) if time_diff > 0 else float("inf")

            if np.isnan(current_roc) or current_roc >= max_roc:
                keep_mask[i] = False

            i += 1

        filtered_group = group.filter(pl.lit(keep_mask))

        # Calculate RoC for kept values
        if len(filtered_group) <= 1:
            return filtered_group.with_columns(pl.lit(None).alias("current_roc"))

        return filtered_group.with_columns(
            (
                (pl.col("value") - pl.col("value").shift(1))
                / ((pl.col(time_col) - pl.col(time_col).shift(1)) / 3600)
            ).alias("current_roc")
        )

    result = data.partition_by(identifier_col)
    cleaned_groups = [
        clean_group(group) for group in tqdm(result, desc="Cleaning timeseries")
    ]

    res = (
        pl.concat(cleaned_groups, how="vertical_relaxed")
        if cleaned_groups
        else data.clear()
    )

    return res.to_pandas() if return_pandas else res


#################################################
#                    Unused                     #
#################################################


# Unused. Written for pandas.
def extract_df_data(
    df: pd.DataFrame,
    col_dict: dict[str, str] | None = None,
    filter_dict: dict[str, list[str]] | None = None,
    exact_match: bool = False,
    remove_prefix: bool = False,
    drop: bool = False,
) -> pd.DataFrame:
    """Extracts data from a DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame to operate on.
        col_dict (dict, optional): A dictionary mapping column names to new names. Defaults to None.
        filter_dict (dict[str, list:str], optional): A dictionary where keys are column names and values are lists of values to filter rows by (may include regex pattern for exact_match=False). Defaults to None.
        exact_match (bool, optional): If True, performs exact matching when filtering. Defaults to False.
        remove_prefix (bool, optional): If True, removes prefix from default_key. Defaults to False.
        drop (bool, optional): If True, drop all columns not specified in col_dict.

    Returns:
        pandas.DataFrame: A DataFrame containing the extracted data from the original DataFrame.
    """
    # rename columns if applicable
    if col_dict:
        df = df.rename(columns=col_dict)

        if drop:
            df = df.filter(items=list(col_dict.values()))

    # filter rows for given values
    if filter_dict:
        for column, values in filter_dict.items():
            if exact_match:
                df = df[df[column].isin(values)]
            elif df[column].dtype == "object":
                regex = "|".join(values)
                df = df[df[column].str.contains(regex, na=False, regex=True)]
            else:
                df = df[df[column].isin(values)]

    return df


# Unused.
def filter_by_condition(
    df: pl.DataFrame,
    expression: (
        IntoExprColumn
        | Iterable[IntoExprColumn]
        | bool
        | list[bool]
        | np.ndarray[Any, Any]
    ),
    description: str = "filter_by_condition call",
    verbose: bool = True,
    mode: Literal["drop", "keep"] = "drop",
) -> pl.DataFrame:
    """Drop rows from a DataFrame based on a condition function.

    Parameters:
        df (pl.DataFrame): The input DataFrame.
        expression: Expression(s) that evaluate to a boolean Series.
        description (str): Description of the condition.
        mode (str): Whether to drop or keep the rows. Can be "drop" or "keep".

    Returns:
        pd.DataFrame: The DataFrame with rows dropped based on the condition.
    """
    mask = df.select(expression)
    if any(dtype != pl.Boolean for dtype in mask.schema.values()):
        raise ValueError(
            "Conditional expression must return a boolean Series/Dataframe."
        )
    if mask.null_count().select(pl.any_horizontal(pl.all().gt(0))).item():
        warnings.warn(
            "Condition expression should return a boolean Series/Dataframe without NA values. Consider the use of `fillna`.",
            UserWarning,
        )

    if mode == "drop":
        df2 = df.remove(expression)
    elif mode == "keep":
        df2 = df.filter(expression)
    else:
        raise ValueError(f"Mode must be 'drop' or 'keep'. Got {mode}")

    if verbose:
        true_count = mask.sum().sum_horizontal().item()
        logger.info(
            __(
                "{mode}: {true_count} rows ({true_count_pct:.2%}) due to {description}",
                mode=mode.upper(),
                true_count=true_count,
                true_count_pct=true_count / len(df),
                description=description,
            )
        )

    return df2


# Unused. Written for pandas.
def df_find_closest(
    group: pd.DataFrame,
    to_col: str,
    tdelta: str = "0",
    pm_before: str = "52w",
    pm_after: str = "52w",
) -> pd.Series[Any] | pd.DataFrame:
    """Find the closest record to the target time.

    Args:
        group (pd.DataFrame): The input dataframe.
        to_col (str): The column containing the target time.
        tdelta (str): The time delta. Defaults to "0".
        pm_before (str): The time range before the target time. Defaults to "52w".
        pm_after (str): The time range after the target time. Defaults to "52w".

    Returns:
        pd.Series[Any] | pd.DataFrame: The closest record.
    """
    parsed_tdelta = pd.to_timedelta(tdelta)
    parsed_pm_before = pd.to_timedelta(pm_before)
    parsed_pm_after = pd.to_timedelta(pm_after)

    target_time = group[to_col] + parsed_tdelta

    group["timediff"] = (group["recordtime"] - target_time).abs()

    within_range = group[
        (group["recordtime"] >= (target_time - parsed_pm_before))
        & (group["recordtime"] <= (target_time + parsed_pm_after))
    ]

    if within_range.empty:
        return pd.Series([np.nan] * len(group.columns), index=group.columns)

    closest = within_range.loc[within_range["timediff"].idxmin()]
    closest.drop(columns=["timediff"], inplace=True)

    return closest


# Unused. Written for pandas.
def aggregate_column(
    group: DataFrameGroupBy, agg_func: str, column: str, params: list[str] = []
) -> pd.Series[Any]:
    match agg_func:
        case "median":
            return group[column].median()
        case "mean":
            return group[column].mean()
        case "perc":
            return group[column].quantile(float(params[0]) / 100)
        case "count":
            return group.size()
        case _:
            raise ValueError(f"Unknown aggregation function: {agg_func}")
