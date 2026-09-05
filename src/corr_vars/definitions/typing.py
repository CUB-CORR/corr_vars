from __future__ import annotations

import re
import textwrap
from datetime import date, datetime, timedelta
from enum import Enum

import pandas as pd
import polars as pl

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NamedTuple,
    Protocol,
    TypeAlias,
    TypedDict,
    TypeGuard,
    TypeVar,
    Union,
    cast,
    get_args,
)

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

    from corr_vars.core.cohort import Cohort

    # TODO: Import dynamically
    from corr_vars.sources.local_datasource import (
        SourceDict as LocalDatasourceSourceDict,
    )
    from corr_vars.sources.local_datasource import (
        SourceDictPartial as LocalDatasourceSourceDictPartial,
    )
    from corr_vars.sources.reprodicu import (
        SourceDict as ReprodIcuSourceDict,
    )
    from corr_vars.sources.reprodicu import (
        SourceDictPartial as ReprodIcuSourceDictPartial,
    )
    from corr_vars.utils.time import TimeWindow


####################
#      Polars      #
####################
ArrayLike: TypeAlias = Union[
    Sequence[Any],
    "pl.Series",
    "np.ndarray[Any, Any]",
    "pd.Series[Any]",
    "pd.DatetimeIndex",
]
IntoExprColumn: TypeAlias = Union["pl.Expr", "pl.Series", str]

PandasDataframe: TypeAlias = pd.DataFrame
PolarsDataframe: TypeAlias = pl.DataFrame
PolarsLazyframe: TypeAlias = pl.LazyFrame

PolarsFrame = TypeVar("PolarsFrame", bound=pl.DataFrame | pl.LazyFrame)

####################
#   Pandas/Stata   #
####################
StataDateFormat: TypeAlias = Literal[
    "tc",
    "%tc",
    "td",
    "%td",
    "tw",
    "%tw",
    "tm",
    "%tm",
    "tq",
    "%tq",
    "th",
    "%th",
    "ty",
    "%ty",
]


####################
#      Helper      #
####################
# A reference to a time column with an optional offset in polars duration language (see below).
TimeAnchorColumn: TypeAlias = str | tuple[str, str]
# A fixed point in time an anchor can be pinned to.
DateLiteral: TypeAlias = datetime | date
# An offset in the polars duration language (``"28d"``, ``"6h"``) or a timedelta.
OffsetLike: TypeAlias = str | timedelta
# Polars duration units consumed by time_difference
TimeDifferenceUnit: TypeAlias = Literal["s", "m", "h", "d", "w"]

TIME_ANCHOR_DELTA_PATTERN = re.compile(
    r"(\+|-)?(?:(?:\d+)(?:ns|us|ms|s|m|h|d|w|mo|q|y))+"
)


def is_time_anchor_column(value: object) -> TypeGuard[TimeAnchorColumn]:
    return isinstance(value, str) or (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], str)
        and TIME_ANCHOR_DELTA_PATTERN.fullmatch(value[1]) is not None
    )


def is_time_anchor_delta(value: object) -> TypeGuard[str]:
    """Whether *value* is a bare delta such as ``"24h"`` or ``"-30m"``."""
    return (
        isinstance(value, str)
        and TIME_ANCHOR_DELTA_PATTERN.fullmatch(value) is not None
    )


####################
#      Cohort      #
####################
class CohortSaverProtocol(Protocol):
    def __call__(self, file_path: str | Path, df: pl.DataFrame) -> None: ...


CohortExportFormats: TypeAlias = Literal[
    "arrow", "csv", "json", "jsonl", "parquet", "xlsx"
]


####################
#     Variables    #
####################
InheritSentry: TypeAlias = Literal["inherit"]
# Sentinel value for RequirementsDict tmin/tmax: inherit the parent's time window.
INHERIT: InheritSentry = "inherit"

_inherit_sentry_values = set(get_args(InheritSentry))
assert _inherit_sentry_values == {
    INHERIT
}, f"The INHERIT constant {INHERIT!r} does not match InheritSentry {sorted(_inherit_sentry_values)}"


# TODO: Make this more generalised...
class _RequirementsRequired(TypedDict):
    """Keys every dict-form ``requires`` entry must define."""

    template: str
    tmin: TimeAnchorColumn | InheritSentry | None
    tmax: TimeAnchorColumn | InheritSentry | None


class RequirementsDict(_RequirementsRequired, total=False):
    """A dict-form ``requires`` entry.

    Beyond the required ``template`` / ``tmin`` / ``tmax``, **any further key
    overrides the field of the same name in the required variable's own
    ``vars.json`` entry, for this requirement only** — the loader forwards unknown
    keys verbatim (see :func:`corr_vars.sources.var_loader.load_variable`).

    A :class:`~typing.TypedDict` cannot express "and arbitrary extra keys" before
    :pep:`728`, so the optional keys that are actually useful are enumerated here to
    keep them type-checked.  Extra keys beyond these still work at runtime; they are
    simply not statically known.

    Optional keys:
        screened_obs_level: Observation level used to scope the DB query.
            Set it to ``"patient"`` to look data up across *all* of a patient's
            cases (e.g. a creatinine baseline from a prior admission).
    """

    screened_obs_level: Literal["patient", "hospital_stay"]


RequirementsIterable: TypeAlias = list[str] | dict[str, RequirementsDict]
CleaningDict: TypeAlias = dict[str, dict[Literal["low", "high"], Any]]


class VariableLoaderProtocol(Protocol):
    def __call__(
        self, var_name: str, time_window: TimeWindow, **kwargs
    ) -> VariableProtocol:
        """Load a variable for the given variable name, time window and optional keyword arguments.

        Args:
            var_name (str): The name of the variable to load.
            time_window (TimeWindow): Time window for extraction.
            **kwargs: Optional keyword arguments for loading the variable.

        Returns:
            VariableProtocol: The loaded variable.
        """
        ...


class VariableProtocol(Protocol):
    var_name: str
    dynamic: bool
    time_window: TimeWindow

    def extract(self, cohort: Cohort) -> pl.DataFrame:
        """Extract the variable data for the given cohort.

        Args:
            cohort (Cohort): The cohort to extract the variable for.

        Returns:
            pl.DataFrame: The extracted variable data.
        """
        ...


@dataclass
class VariableContext:
    """Slim context passed to :class:`VariableCallable` py functions.

    Exposes only minimal set of attributes accessed by py functions, avoiding
    exposure of the full :class:`Variable` implementation.

    Args:
        var_name (str): The variable name.
        dynamic (bool): Whether the variable is dynamic (time-series).
        time_window (TimeWindow): Active time window for extraction.
        required_vars (dict[str, ExtractedVariable]): Loaded dependency variables keyed by alias.
        data (pl.DataFrame | pd.DataFrame | None): Current variable data; ``None`` for complex variables.
        files (Mapping[str, Path]): Data files attached to the variable definition,
            keyed by their path relative to the variable's file directory
            (e.g. ``"postcode/postcode_mapping.csv"``). Empty for variables
            defined locally, which read their files from the package directory.

    Examples:
        Read an attached data file:

        >>> mapping = pl.read_csv(var.files["postcode/postcode_mapping.csv"])
    """

    var_name: str
    dynamic: bool
    time_window: TimeWindow
    required_vars: dict[str, ExtractedVariable]
    data: pl.DataFrame | pd.DataFrame | None
    files: Mapping[str, Path] = field(default_factory=dict)


class VariableCallable(Protocol):
    def __call__(
        self, var: VariableContext, cohort: Cohort
    ) -> pl.DataFrame | pd.DataFrame: ...


@dataclass(repr=False)
class ExtractedVariable:
    """Class for transferring variables from sources to the Cohort object.

    Args:
        var_name (str): Name of the variable.
        dynamic (bool): Whether the variable is dynamic (time-series) (default: True).
        data (pl.DataFrame): The extracted data (default: None).
    """

    var_name: str
    dynamic: bool = True
    data: pl.DataFrame = field(default_factory=pl.DataFrame)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return textwrap.dedent(f"""\
            Variable: {self.var_name} ({self.__class__.__name__}, {"dynamic" if self.dynamic else "static"})
            └── data: {"Nothing extracted" if self.data.is_empty() else self.data.shape}
            """).strip()


####################
#     Obs Level    #
####################
class ObsLevelKeys(NamedTuple):
    primary_key: str
    t_min: str
    t_max: str
    t_eligible: str
    t_outcome: str


class ObsLevel(ObsLevelKeys, Enum):
    PATIENT = (
        "patient_id",
        "birthdate",
        "censordate",
        "birthdate",
        "censordate",
    )

    HOSPITAL_STAY = (
        "case_id",
        "hospital_admission",
        "hospital_discharge",
        "hospital_admission",
        "hospital_discharge",
    )

    ICU_STAY = (
        "icu_stay_id",
        "icu_admission",
        "icu_discharge",
        "icu_admission",
        "icu_discharge",
    )

    PROCEDURE = (
        "procedure_id",
        "or_time_begin",
        "or_time_end",
        "or_time_begin",
        "hospital_discharge",
    )

    @property
    def lower_name(self) -> ObsLevelName:
        return cast("ObsLevelName", self.name.lower())

    @classmethod
    def primary_keys(cls) -> list[str]:
        return [level.primary_key for level in cls]

    @classmethod
    def _missing_(cls, value: object) -> ObsLevel | None:
        def normalise(value: str) -> str:
            return value.lower().replace("_stay", "")

        for member in cls:
            if normalise(member.name) == normalise(str(value)):
                return member
        return None


ObsLevelName: TypeAlias = Literal["patient", "hospital_stay", "icu_stay", "procedure"]

_obs_level_literal_values = set(get_args(ObsLevelName))
_obs_level_enum_values = {level.lower_name for level in ObsLevel}
assert (
    _obs_level_literal_values == _obs_level_enum_values
), f"The ObsLevelName literal values {sorted(_obs_level_literal_values)} don't match {sorted(_obs_level_enum_values)}"


####################
#   Source Dicts   #
####################
class SourceDictPartial(TypedDict, total=False):
    local_datasource: LocalDatasourceSourceDictPartial
    reprodicu: ReprodIcuSourceDictPartial


class SourceDict(TypedDict, total=False):
    local_datasource: LocalDatasourceSourceDict
    reprodicu: ReprodIcuSourceDict
