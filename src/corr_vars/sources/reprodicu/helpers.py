from __future__ import annotations

import os
import textwrap

import polars as pl

from corr_vars import __, logger
from corr_vars.definitions.exceptions import VariableDefinitionError
from corr_vars.sources.reprodicu.mapping import UNITS

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from corr_vars.sources.reprodicu.extract import Variable

    from collections.abc import Iterable


def dpath(path, filename, file_extension=".parquet"):
    return os.path.join(path, filename + file_extension)


def read_parquet(path, filename, file_extension=".parquet"):
    return pl.read_parquet(dpath(path, filename, file_extension))


def scan_parquet(path, filename, file_extension=".parquet"):
    return pl.scan_parquet(dpath(path, filename, file_extension))


def pretty_join(
    values: Iterable[str], sep: str = ", ", width: int = 120, prefix: str = "\t"
) -> str:
    return textwrap.indent(
        text=textwrap.fill(sep.join(sorted(values)), width=width), prefix=prefix
    )


def check_for_required_columns(columns: Iterable[str], dynamic: bool = True) -> None:
    required_columns = [
        "icu_stay_id",
        "value",
        "value_unit",
        "attributes",
    ]
    if dynamic:
        required_columns.append("recordtime_relative")
    for col in required_columns:
        if col not in columns:
            raise ValueError(
                f"Missing required column '{col}' after processing variable"
            )


def extract_variable_from_parquet(
    var: Variable, data_path: str, cohort_obs: pl.DataFrame
) -> pl.DataFrame:
    """Extract a variable from parquet files based on the variable configuration.

    Args:
        var: The Variable instance with extraction configuration
        data_path (str): Path to the reprodicu data files
        cohort_obs (pl.DataFrame): The cohort observations dataframe

    Returns:
        pl.DataFrame: The extracted variable data with standardized columns
    """
    path = var.path
    column = var.column
    is_struct = var.is_struct
    filter_expr = var.filter
    calculation_expr = var.calculation
    calculation_operation: pl.Expr | None = None

    if path is None:
        raise ValueError(f"No path specified for variable '{var.var_name}'")
    if column is None:
        raise ValueError(f"No column specified for variable '{var.var_name}'")

    # Read the parquet file
    file_path = dpath(data_path, path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Parquet file not found: {file_path}")

    logger.debug(__("Reading parquet file: {file_path}", file_path=file_path))
    data = pl.scan_parquet(file_path)

    # Filter to only include patients in the cohort
    case_ids = cohort_obs.lazy().select(
        pl.col("icu_stay_id").alias("Global ICU Stay ID")
    )
    data = data.join(case_ids, on="Global ICU Stay ID")

    # Select relevant columns - we'll build a base structure first
    base_columns = ["Global ICU Stay ID"]

    if var.dynamic:
        base_columns.append("Time Relative to Admission (seconds)")

    extra_columns = []

    if calculation_expr:
        try:
            # Convert string calculation to polars expression
            calculation_operation = eval(calculation_expr)
        except Exception as exc:
            raise VariableDefinitionError(
                f"The 'calculation' expression of variable '{var.var_name}' could "
                f"not be evaluated: {calculation_expr!r}"
            ) from exc

        if not isinstance(calculation_operation, pl.Expr):
            raise VariableDefinitionError(
                f"The 'calculation' expression of variable '{var.var_name}' is a "
                f"{type(calculation_operation).__name__}, not a polars expression: "
                f"{calculation_expr!r}"
            )

        extra_columns.extend(calculation_operation.meta.root_names())
    else:
        extra_columns.append(column)

    # Check if columns exist
    data_schema = data.collect_schema()
    available_columns = list(data_schema.keys())
    missing_columns = set(extra_columns) - set(available_columns)
    if missing_columns:
        raise ValueError(
            f"Some columns were not not found in {path}.\nAvailable:\n{pretty_join(available_columns)}\nMissing:\n{pretty_join(missing_columns)}"
        )

    # Apply calculation if specified
    if calculation_expr and isinstance(calculation_operation, pl.Expr):
        logger.debug(
            __(
                "Applying calculation: {calculation_expr}",
                calculation_expr=calculation_expr,
            )
        )
        data = data.with_columns(calculation_operation.alias(column))

    data = data.select(*base_columns, column)

    # Apply filter if specified
    if filter_expr:
        logger.debug(
            __(
                "Applying filter: {filter_expr}",
                filter_expr=filter_expr,
            )
        )
        try:
            # Convert string filter to polars expression
            filter_condition = eval(filter_expr)
            data = data.filter(filter_condition)
        except Exception as exc:
            raise VariableDefinitionError(
                f"The 'filter' expression of variable '{var.var_name}' could not be "
                f"applied: {filter_expr!r}"
            ) from exc

    # Handle structured data and extract additional attributes
    if is_struct:
        # For structured columns, extract value, unit, and other attributes
        data = data.filter(pl.col(column).is_not_null())

        # Extract the main value - be defensive about this
        try:
            data = data.with_columns(
                pl.col(column).struct.field("value").alias("value")
            )
        except Exception as exc:
            logger.error(
                __(
                    "Failed to extract 'value' field from struct column '{column}': {exc}",
                    column=column,
                    exc=exc,
                )
            )
            # Try to use the struct directly as value if it's a simple case
            try:
                data = data.with_columns(pl.col(column).alias("value"))
                logger.warning(
                    __(
                        "Using entire struct as value for column '{column}'",
                        column=column,
                    )
                )
            except Exception as exc2:
                logger.error(
                    __(
                        "Cannot process struct column '{column}': {exc}",
                        column=column,
                        exc=exc2,
                    )
                )
                raise ValueError(
                    f"Cannot extract value from struct column '{column}': {exc}"
                )

        # Extract other attributes into a struct - be more defensive about field access
        struct_fields = []
        try:
            struct_column = data.collect_schema()[column]
            if isinstance(struct_column, pl.Struct):
                available_fields = struct_column.to_schema().keys()
                logger.debug(
                    __(
                        "Available struct fields for '{column}': {available_fields}",
                        column=column,
                        available_fields=available_fields,
                    )
                )
            else:
                raise TypeError(f"Column '{column}' is not a struct")
        except Exception as exc:
            logger.debug(
                __(
                    "Could not introspect struct fields for '{column}': {exc}",
                    column=column,
                    exc=exc,
                )
            )

        common_fields = {
            "system",
            "timestamp",
            "source",
            "status",
            "method",
            "unit_concept_id",
        }

        struct_fields.append(
            pl.struct(
                pl.col(column).struct.field(
                    list(common_fields.intersection(available_fields))
                )
            ).alias("attributes")
        )

        # Create attributes column from available struct fields
        if struct_fields:
            data = data.with_columns(struct_fields)
        else:
            # No struct fields found, create empty attributes
            data = data.with_columns([pl.lit(None).alias("attributes")])

        # Drop the original struct column if it still exists
        if column != "attributes" and column in data.columns:
            data = data.drop(column)
    else:
        # For non-structured data, rename the column to 'value' and add standard columns
        data = data.rename({column: "value"})
        data = data.with_columns([pl.lit(None).alias("attributes")])

    # Get the unit from the definition, falling back to the packaged units.json (either
    # overrides any struct unit). A config served by the Concepts API carries its own
    # unit; a bundled one does not, and looks it up here as it always has.
    var_unit = getattr(var, "unit", None) or UNITS.get(var.var_name)
    data = data.with_columns([pl.lit(var_unit).cast(pl.Utf8).alias("value_unit")])

    # Rename columns to match expected format
    data = data.rename(
        {
            "Global ICU Stay ID": "icu_stay_id",
            "Time Relative to Admission (seconds)": "recordtime_relative",
        },
        strict=False,
    )

    # Drop struct_unit column as we're using value_unit from units.json
    if "struct_unit" in data.collect_schema():
        data = data.drop("struct_unit")

    # Ensure we have the minimum required columns
    check_for_required_columns(data.collect_schema(), var.dynamic)

    return data.collect()


def standardize_variable_columns(data: pl.DataFrame, dynamic: bool) -> pl.DataFrame:
    """Standardize variable data columns to match the expected format.

    Args:
        data (pl.DataFrame): Variable data to standardize
        primary_key (str): Primary key column name (default: "icu_stay_id")

    Returns:
        pl.DataFrame: Standardized data with all required columns
    """
    # Define the standard column order
    standard_columns = [
        "case_id",
        "icu_stay_id",
        "recordtime",
        "recordtime_relative",
        "recordtime_end",
        "recordtime_end_relative",
        "value",
        "value_unit",
        "attributes",
    ]

    # Ensure required columns exists
    check_for_required_columns(data.collect_schema(), dynamic)

    # Add missing columns with appropriate null values
    for col in standard_columns:
        if col not in data.columns:
            if col in ["recordtime", "recordtime_end"]:
                data = data.with_columns([pl.lit(None).cast(pl.Datetime).alias(col)])
            elif col in ["recordtime_relative", "recordtime_end_relative"]:
                data = data.with_columns([pl.lit(None).cast(pl.Float64).alias(col)])
            elif col in ["value"]:
                # Should never be missing - it can be any type
                continue
            elif col in ["value_unit"]:
                data = data.with_columns([pl.lit(None).cast(pl.Utf8).alias(col)])
            elif col in ["attributes"]:
                data = data.with_columns([pl.lit(None).alias(col)])
            else:
                data = data.with_columns([pl.lit(None).cast(pl.Utf8).alias(col)])

    # Reorder columns to match standard order, keeping extra columns at the end
    data_cols = set(data.columns)
    ordered_cols = [col for col in standard_columns if col in data_cols]
    remaining_cols = [col for col in data_cols if col not in standard_columns]

    return data.select(ordered_cols + remaining_cols)
