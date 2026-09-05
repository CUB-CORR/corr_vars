from __future__ import annotations

import copy

import pandas as pd
import polars as pl

from corr_vars import __, logger, utils
from corr_vars.utils.frames import apply_cleaning
from corr_vars.utils.time import (
    TimeAnchor,
    TimeWindow,
    WindowRelation,
    compatible_with_time_window,
    filter_with_time_anchor,
    filter_with_time_window,
)

from corr_vars.definitions.typing import (
    ExtractedVariable,
    VariableContext,
    VariableProtocol,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from corr_vars.core import Cohort

    from collections.abc import Mapping
    from corr_vars.definitions.typing import CleaningDict, VariableCallable
    from polars._typing import ClosedInterval


class TimeFilterStep:
    """Joins time-window bounds from cohort.obs onto variable data and filters rows.

    Args:
        time_window (TimeWindow): The time window defining tmin/tmax anchors.
        var_name (str): Variable name used in warning messages.
        aliases (tuple[str, str]): Names for the tmin/tmax bound columns that are
            added by :meth:`add_window_columns` and consumed by :meth:`filter`.
            Defaults to ``("case_tmin", "case_tmax")``.
        join_key: Column to join on; defaults to cohort.primary_key.
        closed: Boundary inclusion for the point-in-time path. Defaults to ``"both"``.
        relation: Overlap relation for the interval path. Defaults to ``ANY_OVERLAP``.
    """

    def __init__(
        self,
        time_window: TimeWindow,
        aliases: tuple[str, str] = ("case_tmin", "case_tmax"),
        join_key: str | None = None,
        closed: ClosedInterval = "both",
        relation: WindowRelation = WindowRelation.ANY_OVERLAP,
    ) -> None:
        self.time_window = time_window
        self.aliases = aliases
        self.join_key = join_key
        self.closed = closed
        self.relation = relation

    def add_window_columns(
        self,
        data: pl.DataFrame,
        cohort: Cohort,
    ) -> pl.DataFrame:
        """Join tmin/tmax bound columns from cohort.obs onto data.

        The names of the added columns are determined by :attr:`aliases`.
        The join key is determined by :attr:`join_key` or defaults to ``cohort.primary_key``.

        Args:
            data: Variable data to enrich with time bounds.
            cohort: Cohort providing obs and primary key.

        Returns:
            pl.DataFrame: data with added tmin/tmax bound columns.
        """
        tmin, tmax = self.aliases

        # If the data already carries the cohort primary key (typical for derived
        # variables whose dependencies are keyed at the observation level), join on
        # it directly. Joining on a coarser screened key (e.g. case_id) would both
        # duplicate rows (many primary keys per screened key) and collide with the
        # existing primary-key column, leaking a ``<primary_key>_right`` column.
        if cohort.primary_key in data.columns:
            join_key = cohort.primary_key
        else:
            join_key = self.join_key or cohort.primary_key

        cb = cohort._obs.pipe(
            utils.add_time_window_expr,
            time_window=self.time_window,
            aliases=(tmin, tmax),
        ).select(*{join_key, cohort.primary_key}, tmin, tmax)

        return data.join(cb, on=join_key)

    def filter(self, data: pl.DataFrame) -> pl.DataFrame:
        """Filter rows to those overlapping the time window.

        Expects the tmin/tmax bound columns named by :attr:`aliases` to be present
        (added by :meth:`add_window_columns`). Drops them after filtering.

        When ``recordtime_end`` is present, interval overlap is tested via
        :attr:`relation`; otherwise a point-in-time containment check is used with
        the :attr:`closed` argument controlling boundary inclusion.

        Args:
            data: Variable data containing ``recordtime`` and optionally ``recordtime_end``.

        Returns:
            pl.DataFrame: Filtered data with time-bound columns dropped.
        """
        tmin, tmax = self.aliases
        window = TimeWindow(tmin, tmax)

        if not compatible_with_time_window(data, window):
            logger.warning(
                __(
                    "Time-filter not applied as there is either no {tmin} or {tmax} column in the data",
                    tmin=tmin,
                    tmax=tmax,
                )
            )
            return data

        start_col = "recordtime"
        if "recordtime_end" in data.columns:
            end_col = "recordtime_end"
            end_cleaned_col = f"{end_col}_clean"
            data = data.with_columns(
                pl.max_horizontal(start_col, end_col).alias(end_cleaned_col),
            )
            record_window = TimeWindow(start_col, end_cleaned_col)
            data = data.pipe(
                filter_with_time_window,
                time_window=record_window,
                reference_window=window,
                relation=self.relation,
            )
            data = data.drop(end_cleaned_col)
        else:
            data = data.pipe(
                filter_with_time_anchor,
                time_anchor=TimeAnchor(start_col),
                reference_window=window,
                closed=self.closed,  # pyright: ignore[reportArgumentType]
            )

        data = data.drop(tmin, tmax)
        logger.info(__("Time-filter selected {n_data} rows", n_data=len(data)))
        return data

    def apply(self, data: pl.DataFrame, cohort: Cohort) -> pl.DataFrame:
        data = self.add_window_columns(data, cohort)
        return self.filter(data)


class CleaningStep:
    """Applies value-range cleaning rules to a DataFrame.

    Args:
        cleaning (CleaningDict | None): Column-to-bounds mapping.
        aliases (dict[str, str] | None): Mapping from cleaning-key aliases to actual
            column names (e.g. ``{"value": "heart_rate"}``). Applied unconditionally
            before passing to :func:`apply_cleaning`.
    """

    def __init__(
        self,
        cleaning: CleaningDict | None,
        *,
        aliases: dict[str, str] | None = None,
    ) -> None:
        self.cleaning = cleaning
        self.aliases: dict[str, str] = aliases or {}

    def clean(self, data: pl.DataFrame) -> tuple[pl.DataFrame, bool]:
        """Apply cleaning rules to filter out-of-range values.

        Keys present in :attr:`aliases` are resolved to their target column names
        before the bounds are applied.

        Args:
            data: Variable data to clean.

        Returns:
            tuple[pl.DataFrame, bool]: Cleaned data and a flag indicating whether
            any rules were applied.
        """
        if not self.cleaning:
            return data, False

        cleaning: CleaningDict = {}
        for col, config in self.cleaning.items():
            alias = self.aliases.get(col, col)
            if col not in data.columns and alias in data.columns:
                cleaning[alias] = config
            else:
                cleaning[col] = config

        return apply_cleaning(data, cleaning), True

    def apply(self, data: pl.DataFrame, _: Cohort) -> pl.DataFrame:
        cleaned_data, __ = self.clean(data)
        return cleaned_data


class PyFuncStep:
    """Calls a :class:`~corr_vars.definitions.typing.VariableCallable` with a slim
    :class:`~corr_vars.definitions.typing.VariableContext` instead of the full Variable.

    Args:
        py (VariableCallable): The function to call.
        var (VariableProtocol): The variable instance.
        py_ready_polars (bool): If ``True``, the function receives polars DataFrames
            directly; if ``False``, data is converted to pandas first (legacy path).
        files (Mapping[str, Path] | None): Data files attached to the variable
            definition, keyed by their path relative to the variable's file
            directory. Exposed to the function as ``var.files``. When ``None``,
            the mapping is taken from the callable's own ``files`` attribute —
            functions compiled from a Concepts API ``py`` snippet carry it, and
            locally defined functions do not, which yields an empty mapping.
    """

    def __init__(
        self,
        py: VariableCallable,
        var: VariableProtocol,
        *,
        py_ready_polars: bool = False,
        files: Mapping[str, Path] | None = None,
    ) -> None:
        self.py = py
        self.var = var
        self.py_ready_polars = py_ready_polars
        self.files: Mapping[str, Path] = (
            files if files is not None else dict(getattr(py, "files", {}) or {})
        )

    def call(
        self,
        data: pl.DataFrame | None,
        cohort: Cohort,
        *,
        required_vars: dict[str, ExtractedVariable],
    ) -> pl.DataFrame:
        """Build a :class:`~corr_vars.definitions.typing.VariableContext` and invoke the py function.

        Args:
            data: Current variable data; ``None`` for complex variables.
            cohort: The cohort passed through to the py function.
            required_vars: Loaded dependency variables keyed by alias.

        Returns:
            pl.DataFrame: The DataFrame returned by the py function.

        Raises:
            TypeError: If the py function returns the wrong DataFrame type.
        """
        logger.debug(
            __("Calling Variable function {var_name}", var_name=self.var.var_name)
        )

        if self.py_ready_polars:
            context = VariableContext(
                var_name=self.var.var_name,
                dynamic=self.var.dynamic,
                time_window=self.var.time_window,
                required_vars=required_vars,
                data=data,
                files=self.files,
            )
            result = self.py(var=context, cohort=copy.deepcopy(cohort))
            if not isinstance(result, pl.DataFrame):
                raise TypeError("Variable function did not return a polars DataFrame")
            return result

        # Legacy pandas path
        logger.warning(
            __(
                "Converting {var_name} to pandas. Consider rewriting to polars and setting py_ready_polars=True",
                var_name=self.var.var_name,
            )
        )
        pd_required_vars = {
            k: ExtractedVariable(
                var_name=v.var_name,
                dynamic=v.dynamic,
                data=utils.convert_to_pandas_df(v.data),  # type: ignore[arg-type]
            )
            for k, v in required_vars.items()
        }
        pd_cohort = copy.deepcopy(cohort)
        pd_cohort._obs = utils.convert_to_pandas_df(pd_cohort._obs)  # type: ignore[assignment]
        context = VariableContext(
            var_name=self.var.var_name,
            dynamic=self.var.dynamic,
            time_window=self.var.time_window,
            required_vars=pd_required_vars,
            data=utils.convert_to_pandas_df(data) if data is not None else None,
            files=self.files,
        )
        result = self.py(var=context, cohort=pd_cohort)
        if not isinstance(result, pd.DataFrame):
            raise TypeError("Variable function did not return a pandas DataFrame")
        return utils.convert_to_polars_df(result)
