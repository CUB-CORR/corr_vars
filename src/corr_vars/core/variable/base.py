from __future__ import annotations

import textwrap
import warnings

from corr_vars import __, logger, utils
from corr_vars.core.steps import CleaningStep, PyFuncStep, TimeFilterStep
from corr_vars.sources import guess_variable_source
from corr_vars.utils.time import TimeAnchor, TimeWindow

from corr_vars.definitions.typing import INHERIT, ExtractedVariable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import polars as pl

    from corr_vars.core import Cohort

    from collections.abc import Mapping
    from corr_vars.definitions.typing import (
        CleaningDict,
        RequirementsDict,
        RequirementsIterable,
        TimeAnchorColumn,
        VariableCallable,
    )


class Variable:
    """Base class for all variables.

    Args:
        var_name (str): The variable name.
        dynamic (bool): True if the variable is dynamic (time-series).
        requires (RequirementsIterable): List of variables or dict of variables with tmin/tmax required to calculate the variable (default: []).
        tmin (TimeAnchorColumn | None): The tmin argument. Can either be a string (column name) or a tuple of (column name, timedelta).
        tmax (TimeAnchorColumn | None): The tmax argument. Can either be a string (column name) or a tuple of (column name, timedelta).
        py (Callable): The function to call to calculate the variable.
        py_ready_polars (bool): True if the variable code can accept polars dataframes as input and return a polars dataframe.
        cleaning (CleaningDict): Dictionary with cleaning rules for the variable.

    Attributes:
        contributor_key (str): Identity of this variable within its
            :class:`~corr_vars.sources.var_loader.MultiSourceVariable`. Defaults
            to `var_name` and is overwritten with the contributor key when the
            variable is one of several sharing a name, such as the members of a
            concept group.

    Note:
        tmin and tmax can be None when you create a Variable object, but must be set before extraction.
        If you add the variable via cohort.add_variable(), it will be automatically set to the cohort's tmin and tmax.

        This base class should not be used directly; use one of the subclasses instead. These are specified in the sources submodule.

    Examples:
        Basic cleaning configuration:

        >>> cleaning = {"value": {"low": 10, "high": 80}}

        Time constraints with relative offsets:

        >>> # Extract data from 2 hours before to 6 hours after ICU admission
        >>> tmin = ("icu_admission", "-2h")
        >>> tmax = ("icu_admission", "+6h")

        Variable with dependencies:

        >>> # This variable requires other variables to be loaded first
        >>> requires = ["blood_pressure_sys", "blood_pressure_dia"]

        >>> # This variable requires other variables with fixed tmin/tmax to be loaded first
        >>> requires = {
        ...     "blood_pressure_sys_hospital": {
        ...         "template": "blood_pressure_sys"
        ...         "tmin": "hospital_admission",
        ...         "tmax": "hospital_discharge"
        ...     },
        ...     "blood_pressure_dia_hospital": {
        ...         "template": "blood_pressure_dia"
        ...         "tmin": "hospital_admission",
        ...         "tmax": "hospital_discharge"
        ...     }
        ... }
    """

    # VariableContext
    var_name: str
    dynamic: bool
    time_window: TimeWindow
    required_vars: dict[str, ExtractedVariable]
    data: pl.DataFrame | None

    # Additional fields
    requirements: Mapping[str, RequirementsDict]
    py: VariableCallable | None
    py_ready_polars: bool
    cleaning: CleaningDict | None

    def __init__(
        self,
        var_name: str,
        dynamic: bool,
        tmin: TimeAnchorColumn | TimeAnchor | None,
        tmax: TimeAnchorColumn | TimeAnchor | None,
        requires: RequirementsIterable = [],
        py: VariableCallable | None = None,
        py_ready_polars: bool = False,
        cleaning: CleaningDict | None = None,
    ) -> None:
        self.var_name = var_name
        self.dynamic = dynamic

        # Identity within the variable. Several contributors may share a
        # var_name — the members of a concept group do — and anything filed per
        # variable, such as the extraction cache, has to tell them apart.
        self.contributor_key = var_name

        self.time_window = TimeWindow(tmin=tmin, tmax=tmax)

        self.requirements = (
            requires
            if isinstance(requires, dict)
            else {
                r: {"template": r, "tmin": INHERIT, "tmax": INHERIT} for r in requires
            }
        )
        self.required_vars = {}

        self.py = py
        self.py_ready_polars = py_ready_polars

        self.cleaning = cleaning
        self.data = None

    @property
    def guessed_source(self) -> str | None:
        return guess_variable_source(self)

    def _dependency_sources(self, cohort: Cohort) -> list[str] | None:
        """Sources a dependency of this variable is looked up in.

        A variable that belongs to one of the cohort's configured sources keeps
        its dependencies inside that source. One that does not — above all the
        source-agnostic aggregation classes of ``local_datasource``, constructed
        by hand over a cohort built from other sources — lets them resolve
        across every source the cohort was configured with.

        Args:
            cohort (Cohort): Cohort whose configured sources decide the scope.

        Returns:
            list[str] | None: The single source to restrict the lookup to, or
            ``None`` for all of the cohort's sources.
        """
        source = self.guessed_source
        configured = getattr(cohort, "sources", None) or {}
        if source is not None and source in configured:
            return [source]
        return None

    @classmethod
    def from_time_window(cls, time_window: TimeWindow, *args, **kwargs) -> Variable:
        kwargs["tmin"] = time_window.tmin
        kwargs["tmax"] = time_window.tmax
        return cls(*args, **kwargs)

    def extract(self, cohort: Cohort) -> pl.DataFrame:
        """Extracts data from the datasource. Usually follows this pattern.

        .. code-block:: python

            # Load & Extract required variables
            self._get_required_vars(cohort)

            # This should change self.data either by returning or side effect
            self.data = self._custom_extraction(cohort)

            # Calls variable function to transform extracted data
            self._call_var_function(cohort)

            if self.dynamic:
                self._add_time_window(cohort)
                # Expects case_tmin, case_tmax for each primary key
                self._timefilter()
                self._apply_cleaning()
        """
        raise NotImplementedError

    def _get_required_vars(self, cohort: Cohort) -> None:
        """Load required variables for this variable.

        Args:
            cohort (Cohort): The cohort instance to load variables from.

        Returns:
            None: Sets self.required_vars with loaded Variable objects.
        """
        if len(self.requirements) == 0:
            logger.debug(
                __("No dependencies loaded for '{var_name}'.", var_name=self.var_name)
            )
            return

        # Print dependency tree before extraction
        logger.info(
            __("Loading dependencies for '{var_name}':", var_name=self.var_name)
        )
        utils.log_collection(logger=logger, collection=self.requirements)

        for save_as, config in self.requirements.items():
            var_name = config["template"]

            include_sources = self._dependency_sources(cohort)

            # "inherit" (optionally with a delta) -> parent's window; otherwise
            # the configured anchor, or None (TimeWindow fills
            # datetime.min / datetime.max -> unbounded).
            tmin = utils.resolve_time_anchor(config["tmin"], self.time_window.tmin)
            tmax = utils.resolve_time_anchor(config["tmax"], self.time_window.tmax)

            # Any remaining key overrides that field of the required variable's own
            # vars.json entry, for this requirement only (e.g. screened_obs_level).
            overrides = {
                key: value
                for key, value in config.items()
                if key not in ("template", "tmin", "tmax")
            }

            var = cohort.load_variable(
                variable=(
                    var_name,
                    TimeWindow(tmin, tmax),
                ),
                include_sources=include_sources,
                overrides=overrides or None,
            )
            data = var.extract(cohort)
            self.required_vars[save_as] = ExtractedVariable(
                var_name=var.var_name, dynamic=var.dynamic, data=data
            )

    def _call_var_function(self, cohort: Cohort) -> bool:
        """Call the variable function if it exists.

        Args:
            cohort (Cohort)

        Returns:
            True if the function was called, False otherwise.
            Operations are applied to Variable.data directly.
        """
        if self.py is None:
            logger.debug(
                __(
                    "Variable function not called for {var_name}",
                    var_name=self.var_name,
                )
            )
            return False

        py_func_step = PyFuncStep(self.py, self, py_ready_polars=self.py_ready_polars)
        self.data = py_func_step.call(
            self.data, cohort, required_vars=self.required_vars
        )
        return True

    def _add_time_window(self, cohort: Cohort, join_key: str | None = None) -> None:
        """Adds tmin / tmax / join key / primary key from cohort.obs to self.data.

        This might duplicate data if the join_key (default: cohort.primary_key) is not unique in obs.

        Args:
            cohort (Cohort): The cohort instance containing time constraint information.
            join_key (str | None): Whether to apply filtering even if case_tmin/case_tmax columns don't exist (default: cohort.primary_key if set to None).

        Returns:
            None: Modifies self.data to add tmin / tmax / join key / primary key columns.
        """
        if self.data is None:
            warnings.warn(
                "_add_time_window was called before extraction, skipping.", UserWarning
            )
            return

        time_filter_step = TimeFilterStep(self.time_window, join_key=join_key)
        self.data = time_filter_step.add_window_columns(self.data, cohort)

    def _timefilter(self) -> None:
        """Apply time-based filtering to variable data based on tmin/tmax constraints."""
        if self.data is None:
            warnings.warn(
                "_timefilter was called before extraction, skipping.", UserWarning
            )
            return

        time_filter_step = TimeFilterStep(self.time_window)
        self.data = time_filter_step.filter(self.data)

    def _apply_cleaning(self) -> bool:
        """Apply cleaning rules to filter out invalid values.

        Returns:
            True if the data was cleaned,
            False otherwise.

            Operations will be applied to Variable.data directly.
        """
        if self.data is None:
            warnings.warn(
                "_apply_cleaning was called before extraction, skipping.", UserWarning
            )
            return False

        cleaning_step = CleaningStep(self.cleaning, aliases={"value": self.var_name})
        self.data, applied = cleaning_step.clean(self.data)
        return applied

    # System methods
    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return textwrap.dedent(f"""\
            Variable: {self.var_name} ({self.__class__.__name__}, {"dynamic" if self.dynamic else "static"})
            ├── time_window: {self.time_window}
            └── data: {"Not extracted" if self.data is None else self.data.shape}
            """).strip()

    def __getstate__(self) -> dict[str, Any]:
        return self.__dict__

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
