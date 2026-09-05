from __future__ import annotations

import datetime

import polars as pl

from corr_vars import __, logger
from corr_vars.core.variable import Variable as BaseVariable
from corr_vars.sources.reprodicu import helpers

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from corr_vars.core.cohort import Cohort
    from corr_vars.definitions import (
        CleaningDict,
        RequirementsIterable,
        TimeAnchorColumn,
        VariableCallable,
    )
    from corr_vars.utils.time import TimeAnchor


#: Served ``type`` values that mean the variable carries a time dimension.
_DYNAMIC_TYPES = frozenset({"native_dynamic", "derived_dynamic"})


def VariableLoader(*args, **kwargs) -> BaseVariable:
    """Factory function to create the correct variable type based on the arguments.

    A config served by the Concepts API carries two keys the bundled ``vars.json``
    never had: ``type``, which the importer derives from the presence of
    ``calculation`` and ``dynamic``, and ``unit``, which it folds in from
    ``units.json``. ``type`` is consumed here — like the loader does — because
    it restates ``dynamic`` rather than describing a distinct class; reprodICU has
    only one Variable type.
    """
    var_type = kwargs.pop("type", None)
    if var_type in _DYNAMIC_TYPES:
        kwargs["dynamic"] = True

    return Variable.from_time_window(*args, **kwargs)


class Variable(BaseVariable):
    """Variable class for the reprodicu source.

    Args:
        var_name: Name of the variable
        dynamic: Whether the variable is dynamic
        requires: List of required variables
        tmin: Minimum time pointx
        tmax: Maximum time point
        py: Python function to apply to the variable
        py_ready_polars: Whether the variable is already polars ready
        cleaning: Dictionary of cleaning functions
        path: Name of the parquet file/table to extract from (e.g., "timeseries_labs")
        column: Name of the column to extract
        is_struct: Whether the column contains structured data
        filter: Polars filter expression to apply
        complex: Whether the variable is complex (Will skip the parquet extraction -> Must specify query in py)
        unit: Unit of the extracted value. Carried by configs served from the Concepts
            API; None for bundled definitions, which look it up in units.json

    Returns:
        None

    Note:
        Only use this class for defining custom variables. For CORR-specified variables, use cohort.add_variable() directly.
    """

    def __init__(
        self,
        # core.variable.Variable
        var_name: str,
        dynamic: bool = True,
        requires: RequirementsIterable = [],
        tmin: TimeAnchorColumn | TimeAnchor | None = None,
        tmax: TimeAnchorColumn | TimeAnchor | None = None,
        py: VariableCallable | None = None,
        py_ready_polars: bool = False,
        cleaning: CleaningDict | None = None,
        # sources.reprodicu.extract.Variable(NativeVariable)
        path: str | None = None,
        column: str | None = None,
        is_struct: bool = False,
        filter: str | None = None,
        calculation: str | None = None,
        complex: bool = False,
        unit: str | None = None,
    ):
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

        self.path = path
        self.column = column
        self.is_struct = is_struct
        self.filter = filter
        self.calculation = calculation
        self.complex = complex
        # Served configs carry their unit; bundled ones leave this None and fall back
        # to the packaged units.json (see helpers.extract_variable_from_parquet).
        self.unit = unit

    def extract(self, cohort: Cohort) -> pl.DataFrame:
        # TODO: Split into classes or try to reuse the shared aggregation logic

        self._get_required_vars(cohort)

        # NativeDynamic variables
        if not self.complex:
            self.data = self._extract_from_parquet(cohort)

        function_called = self._call_var_function(cohort)
        if self.data is None:
            if self.complex and not function_called:
                raise ValueError(
                    "Complex variables need to extract data inside a custom variable function."
                )
            raise ValueError("Variable extraction resulted in no data.")

        if self.dynamic:
            self._add_relative_times(cohort)
            if not self.complex:
                self._add_time_window(cohort)
            self._timefilter()
            self._apply_cleaning()
        else:
            self._apply_cleaning()
            self.data = self.data.select(
                cohort.primary_key, pl.col("value").alias(self.var_name)
            )

        return self.data

    def _add_relative_times(self, cohort: Cohort) -> None:
        # TODO: Move logic to add_relative_times
        # "recordtime_relative" is a required column and should be always present at this point
        if self.data is None:
            return

        # Join with cohort data to get the admission timestamp
        admission_times = cohort._obs.select(
            cohort.primary_key,
            cohort.t_min,  # This should be the admission time column
        )

        self.data = self.data.join(
            admission_times,
            left_on="icu_stay_id",
            right_on=cohort.primary_key,
            how="left",
        )

        # Convert relative seconds to absolute datetime
        self.data = self.data.with_columns(
            pl.coalesce(cohort.t_min, pl.lit(datetime.datetime(1900, 1, 1, 0, 0, 0)))
            .add(pl.duration(seconds="recordtime_relative"))
            .alias("recordtime")
        )

        # Drop the admission time column
        self.data = self.data.drop(cohort.t_min)

    def _extract_from_parquet(self, cohort) -> pl.DataFrame:
        """Extract data from parquet files."""
        data_path = cohort.sources["reprodicu"]["path"]

        # Build the extraction query for parquet files
        result_df = helpers.extract_variable_from_parquet(self, data_path, cohort._obs)

        # Ensure all standardized columns are present and properly ordered
        result_df = helpers.standardize_variable_columns(result_df, self.dynamic)

        logger.info(
            __(
                "Extracted {rows} rows for {var_name}",
                rows=len(result_df),
                var_name=self.var_name,
            )
        )

        return result_df
