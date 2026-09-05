from __future__ import annotations

import copy
from datetime import date, datetime
from enum import Flag, auto

import polars as pl

from corr_vars.definitions.constants import UNBOUNDED_TMAX, UNBOUNDED_TMIN
from corr_vars.utils.frames import time_difference

from .base import columns, normalize_offset

from corr_vars.definitions.typing import (
    INHERIT,
    DateLiteral,
    InheritSentry,
    ObsLevel,
    OffsetLike,
    PolarsFrame,
    TimeAnchorColumn,
    TimeDifferenceUnit,
    is_time_anchor_column,
    is_time_anchor_delta,
)
from typing import TYPE_CHECKING, cast

__all__ = [
    "TimeAnchor",
    "TimeWindow",
    "WindowOverlap",
    "WindowRelation",
    "bounded_by_obs_level",
    "compatible_with_time_anchor",
    "compatible_with_time_window",
    "filter_with_time_anchor",
    "filter_with_time_window",
    "format_delta",
    "resolve_time_anchor",
    "truncate_to_day",
]

if TYPE_CHECKING:
    from polars.expr.whenthen import ChainedThen, Then

    from corr_vars.core import Cohort

    from collections.abc import Sequence
    from polars._typing import ClosedInterval


def format_delta(delta: str | Sequence[str]) -> str:
    """Render a chain of offsets with an explicit sign in front of each.

    A positive offset is stored without a sign, so spelling every sign out is
    what keeps a chain readable: ``("2h", "-1h", "3d")`` becomes
    ``"+ 2h - 1h + 3d"``.

    Args:
        delta: The offsets, or a single one.

    Returns:
        The chain with an operator in front of every offset.
    """
    offsets = (delta,) if isinstance(delta, str) else delta
    return " ".join(
        f"{'-' if offset.startswith('-') else '+'} {offset.removeprefix('-')}"
        for offset in offsets
    )


class TimeAnchor:
    column: str | None
    # Every offset applied to this anchor, in order.
    delta: tuple[str, ...] | None
    expr: pl.Expr

    def __init__(self, bound: TimeAnchorColumn | DateLiteral | pl.Expr) -> None:
        if (
            not is_time_anchor_column(bound)
            and not isinstance(bound, pl.Expr)
            and not isinstance(bound, datetime)
            and not isinstance(bound, date)
        ):
            raise TypeError(
                "Expected TimeAnchorColumn to be a string or a tuple of two strings, a polars expression or a date/datetime literal."
            )

        if isinstance(bound, pl.Expr):
            # An already-built expression (e.g. a day-snapped anchor). Literalness is
            # read from the expression itself so column-referencing exprs are not
            # mistaken for literals.
            self.column = (
                None if bound.meta.is_literal(allow_aliasing=True) else str(bound)
            )
            self.delta = None
            self.expr = bound
        elif isinstance(bound, (datetime, date)):
            self.column = None
            self.delta = None
            self.expr = pl.lit(bound)
        elif isinstance(bound, str):
            self.column = bound
            self.delta = None
            self.expr = pl.col(self.column)
        else:
            column, offset = bound
            offset = offset.lstrip("+")
            if not is_time_anchor_delta(offset):
                raise ValueError(f"Invalid time anchor delta: {offset!r}")

            self.column = column
            self.delta = (offset,)
            self.expr = pl.col(self.column).dt.offset_by(offset)

        self.expr = self.expr.alias("time")

    def __repr__(self) -> str:
        if self.is_literal:
            # A shifted literal already carries the offset in its timestamp, so
            # the delta is not repeated here. The expression is always aliased,
            # which is why plain is_literal() would be False. The else branch
            # should never be hit, but if it is, the root names are printed
            # so the user can see what the anchor is built from.
            return (
                f"Literal: {pl.select(self.expr).item()}"
                if self.expr.meta.is_literal(allow_aliasing=True)
                else str(self.expr)
            )

        if not self.is_relative:
            return cast("str", self.column)

        return f"{self.column} {format_delta(cast('tuple', self.delta))}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TimeAnchor):
            return False

        if self.column != other.column or self.delta != other.delta:
            return False

        # Column and delta fully determine a column anchor's expression, so
        # comparing the expression tree on top would only compare how the anchor
        # was *built*: shifting a plain column wraps its already aliased
        # expression, while TimeAnchor((column, delta)) offsets the bare column.
        # Both mean the same thing. A literal keeps its timestamp nowhere else
        # than in the expression, so there it stays the deciding comparison.
        return not self.is_literal or self.expr.meta.eq(other.expr)

    @property
    def is_relative(self) -> bool:
        """Whether a delta was applied — independently of what it was applied to."""
        return self.delta is not None

    @property
    def is_literal(self) -> bool:
        """Whether this anchor is a fixed timestamp rather than a column."""
        return self.column is None

    def copy(self) -> TimeAnchor:
        """Return a copy of this anchor.

        The copy shares only immutable state — column, delta and the polars
        expression — so re-pointing one of them leaves the original alone.
        """
        return copy.copy(self)

    def shifted_by(self, delta: OffsetLike) -> TimeAnchor:
        """Return a copy of this anchor moved by *delta*.

        Deltas stack: shifting ``("icu_admission", "+2h")`` by ``"-1h"`` keeps
        the column, applies both offsets and appends to :attr:`delta`.  A fixed
        timestamp is moved to the offset timestamp.

        Either way the result reports :attr:`is_relative`, and a shifted literal
        stays :attr:`is_literal`: the two describe different things — what the
        anchor is fixed to, and whether it was moved.

        Args:
            delta: Offset such as ``"24h"`` or ``"-30m"``.

        Returns:
            TimeAnchor: The shifted anchor.

        Raises:
            ValueError: If *delta* is not a valid offset, or if shifting a fixed
                timestamp leaves the representable datetime range — which is what
                moving :data:`UNBOUNDED_TMIN` further back or
                :data:`UNBOUNDED_TMAX` further forward does.
        """
        delta = normalize_offset(delta).lstrip("+")
        if not is_time_anchor_delta(delta):
            raise ValueError(f"Invalid time anchor delta: {delta!r}")

        shifted = self.copy()
        if self.is_literal:
            # The offset lands in the timestamp itself, which is then put back
            # into a plain literal so the anchor keeps reading as one.  Casting
            # keeps a sub-day offset from being silently dropped on a date.
            moved = pl.select(
                self.expr.cast(pl.Datetime("us")).dt.offset_by(delta)
            ).item()
            shifted.expr = pl.lit(moved).alias("time")
        else:
            shifted.expr = self.expr.dt.offset_by(delta).alias("time")

        shifted.delta = (delta,) if self.delta is None else self.delta + (delta,)
        return shifted


def truncate_to_day(expr: pl.Expr, *, day_start: OffsetLike | None = None) -> pl.Expr:
    """Snap a datetime expression to the start of the day it falls in.

    ``day_start`` shifts where a "day" begins, letting the same helper express
    different day conventions:

    - ``None`` — calendar day: the boundary is midnight (``00:00``).
    - ``"6h"`` — ICU day: the boundary is ``06:00``, so timestamps between midnight
      and ``06:00`` roll into the *previous* day (``05:59`` → ``06:00`` yesterday,
      ``06:00`` → ``06:00`` today). The offset-truncate-offset form handles this wrap
      without any hour-of-day branching.

    The result keeps the ``Datetime`` dtype (at the boundary time), so downstream
    datetime arithmetic is unaffected. ``day_start`` is a non-negative offset.
    """
    if day_start is None:
        return expr.dt.truncate("1d")
    offset = normalize_offset(day_start).lstrip("+")
    return expr.dt.offset_by(f"-{offset}").dt.truncate("1d").dt.offset_by(offset)


def resolve_time_anchor(
    value: TimeAnchorColumn | InheritSentry | None,
    parent: TimeAnchor,
) -> TimeAnchor | TimeAnchorColumn | None:
    """Resolve a requirement's ``tmin`` / ``tmax`` against the parent's anchor.

    Four notations are accepted:

    - ``"inherit"`` — the parent's anchor unchanged.
    - ``("inherit", "-1h")`` — the parent's anchor shifted by the delta.
      In ``vars.json`` this is written as ``["inherit", "-1h"]``.
    - ``"-1h"`` — shorthand for the previous form; any bare delta is read
      relative to the parent, since no column is named like one.
    - anything else — an anchor of its own (column, ``(column, delta)`` or
      ``None`` for unbounded), returned untouched.

    Args:
        value: The requirement's configured anchor.
        parent: The anchor of the same side of the parent variable's window.

    Returns:
        The resolved anchor, ready to be passed to :class:`TimeWindow`.
    """
    if value == INHERIT:
        return parent

    if is_time_anchor_delta(value):
        return parent.shifted_by(cast("str", value))

    if isinstance(value, tuple) and len(value) == 2 and value[0] == INHERIT:
        return parent.shifted_by(value[1])

    return value


class TimeWindow:
    tmin: TimeAnchor
    tmax: TimeAnchor

    def __init__(
        self,
        tmin: TimeAnchor | TimeAnchorColumn | DateLiteral | None = None,
        tmax: TimeAnchor | TimeAnchorColumn | DateLiteral | None = None,
    ) -> None:
        self.tmin = (
            tmin if isinstance(tmin, TimeAnchor) else TimeAnchor(tmin or UNBOUNDED_TMIN)
        )
        self.tmax = (
            tmax if isinstance(tmax, TimeAnchor) else TimeAnchor(tmax or UNBOUNDED_TMAX)
        )

    @classmethod
    def from_obs_level(cls, obs_level: ObsLevel) -> TimeWindow:
        """Build a :class:`TimeWindow` from an observation level's ``t_min`` / ``t_max``."""
        return cls(obs_level.t_min, obs_level.t_max)

    @classmethod
    def from_anchor_and_duration(
        cls,
        anchor: TimeAnchor | TimeAnchorColumn | DateLiteral,
        duration: OffsetLike,
    ) -> TimeWindow:
        """Build ``[anchor, anchor + duration]`` from one anchor and a length.

        Spares callers from repeating the anchor on both bounds — instead of
        ``TimeWindow("icu_admission", ("icu_admission", "+28d"))`` write
        ``TimeWindow.from_anchor_and_duration("icu_admission", "28d")``. This is the
        natural shape for a T0-plus-horizon follow-up window.

        Args:
            anchor: The window start (T0). A column name, ``(column, delta)`` tuple,
                datetime/date literal, or an existing :class:`TimeAnchor`.
            duration: Horizon length as a polars-duration string (``"28d"``) or a
                :class:`~datetime.timedelta`. A leading sign is optional (``+``).
        """
        offset = normalize_offset(duration)

        # A plain column start keeps its column/delta metadata via the tuple form.
        if isinstance(anchor, str):
            return cls(anchor, (anchor, offset))

        tmin = anchor if isinstance(anchor, TimeAnchor) else TimeAnchor(anchor)
        return cls(tmin, tmin.shifted_by(offset))

    def anchored_to_day(self, day_start: OffsetLike | None = None) -> TimeWindow:
        """Return a copy with both bounds snapped to a day boundary.

        Snaps ``tmin`` and ``tmax`` via :func:`truncate_to_day` (``day_start=None``
        for calendar days, ``"6h"`` for ICU days). The original window is unchanged, so
        only the caller that wants day-anchored bounds is affected. When ``tmax`` is
        ``tmin + Nd`` both bounds shift by the same intra-day amount, so the horizon
        length is preserved.
        """
        return TimeWindow(
            TimeAnchor(truncate_to_day(self.tmin.expr, day_start=day_start)),
            TimeAnchor(truncate_to_day(self.tmax.expr, day_start=day_start)),
        )

    def time_difference(
        self, *, unit: TimeDifferenceUnit = "s", total: bool = True
    ) -> pl.Expr:
        return time_difference(self.tmax_expr, self.tmin_expr, unit=unit, total=total)

    def __repr__(self) -> str:
        return f"TimeWindow(tmin={self.tmin}, tmax={self.tmax})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TimeWindow):
            return False
        return self.tmin == other.tmin and self.tmax == other.tmax

    ####################
    #   Time Anchors   #
    ####################
    @property
    def tmin_expr(self) -> pl.Expr:
        return self.tmin.expr.alias("tmin")

    @property
    def filled_tmin_expr(self) -> pl.Expr:
        return self.tmin.expr.fill_null(UNBOUNDED_TMIN).alias("tmin")

    @property
    def tmax_expr(self) -> pl.Expr:
        return self.tmax.expr.alias("tmax")

    @property
    def filled_tmax_expr(self) -> pl.Expr:
        return self.tmax.expr.fill_null(UNBOUNDED_TMAX).alias("tmax")

    ####################
    #   Time Concepts  #
    ####################
    # TODO: Use or discard
    @property
    def tmid_expr(self) -> pl.Expr:
        return self.tmin_expr.add(self.duration_expr.truediv(2)).alias("tmid")

    # TODO: Use or discard
    @property
    def duration_expr(self) -> pl.Expr:
        return (self.tmax_expr - self.tmin_expr).alias("duration")

    @property
    def is_relative(self) -> bool:
        return self.tmin.is_relative or self.tmax.is_relative

    @property
    def is_literal(self) -> bool:
        return self.tmin.is_literal and self.tmax.is_literal

    ####################
    #     Validity     #
    ####################
    @property
    def any_timebound_in_year_9999_expr(self) -> pl.Expr:
        return pl.any_horizontal(
            self.tmin_expr.dt.year().eq(pl.lit(9999)).fill_null(False),
            self.tmax_expr.dt.year().eq(pl.lit(9999)).fill_null(False),
        )

    @property
    def any_timebound_null_expr(self) -> pl.Expr:
        return pl.any_horizontal(
            self.tmin_expr.is_null(),
            self.tmax_expr.is_null(),
        )

    @property
    def tmin_after_tmax_expr(self) -> pl.Expr:
        return self.tmin_expr > self.tmax_expr


class WindowRelation(Flag):
    CONTAINED = auto()  # fully inside [tmin, tmax]
    OVERLAPS_LEFT = auto()  # starts before tmin and ends inside
    OVERLAPS_RIGHT = auto()  # starts inside and ends after tmax
    COVERS_WINDOW = auto()  # spans over both boundaries

    # Common combinations
    ANY_LEFT_OVERLAP = OVERLAPS_LEFT | CONTAINED | COVERS_WINDOW
    ANY_RIGHT_OVERLAP = OVERLAPS_RIGHT | CONTAINED | COVERS_WINDOW
    ANY_OVERLAP = CONTAINED | OVERLAPS_LEFT | OVERLAPS_RIGHT | COVERS_WINDOW


class WindowOverlap:
    @staticmethod
    def contains_expr(
        record: TimeAnchor, window: TimeWindow, closed: ClosedInterval = "both"
    ) -> pl.Expr:
        return record.expr.is_between(window.tmin_expr, window.tmax_expr, closed=closed)

    @staticmethod
    def relation_expr(
        record: TimeWindow,
        window: TimeWindow,
    ) -> dict[WindowRelation, pl.Expr]:
        start = record.tmin_expr
        end = record.tmax_expr
        tmin = window.tmin_expr
        tmax = window.tmax_expr

        return {
            WindowRelation.CONTAINED: (start >= tmin) & (end <= tmax),
            WindowRelation.OVERLAPS_LEFT: (start < tmin)
            & (end >= tmin)
            & (end <= tmax),
            WindowRelation.OVERLAPS_RIGHT: (start >= tmin)
            & (start <= tmax)
            & (end > tmax),
            WindowRelation.COVERS_WINDOW: (start < tmin) & (end > tmax),
        }

    @staticmethod
    def overlap_expr(
        record: TimeWindow,
        window: TimeWindow,
        relation: WindowRelation,
    ) -> pl.Expr:
        expr_map = WindowOverlap.relation_expr(record, window)

        expr: pl.Expr | None = None
        for flag, cond in expr_map.items():
            if relation & flag:
                expr = cond if expr is None else expr | cond

        if expr is None:
            raise ValueError(f"Invalid relation: {relation}")

        return expr

    # TODO: Use or discard
    @staticmethod
    def classify_expr(
        record: TimeWindow,
        window: TimeWindow,
        *,
        default: WindowRelation | None = None,
    ) -> pl.Expr:
        exprs = WindowOverlap.relation_expr(record, window)

        when: Then | ChainedThen = pl.when(exprs[WindowRelation.CONTAINED]).then(
            pl.lit(WindowRelation.CONTAINED.value)
        )

        for rel in (
            WindowRelation.OVERLAPS_LEFT,
            WindowRelation.OVERLAPS_RIGHT,
            WindowRelation.COVERS_WINDOW,
        ):
            when = when.when(exprs[rel]).then(pl.lit(rel.value))

        if default is not None:
            expr = when.otherwise(pl.lit(default.value))
        else:
            expr = when.otherwise(pl.lit(None))

        return expr


# ------------------------
# Compability checks for time anchors and windows
# ------------------------
def compatible_with_time_anchor(
    df: pl.DataFrame | pl.LazyFrame, time_anchor: TimeAnchor
) -> bool:
    if time_anchor.is_literal:
        return True

    # Non-literal anchor — a column, a shifted column, or an arbitrary expression such
    # as a day-snapped bound: every column it references must be present. Reading the
    # columns off the expression covers all three (``column`` is not usable here, as it
    # holds the expression's repr, not a column name, for expression anchors).
    available = columns(df)
    return all(name in available for name in time_anchor.expr.meta.root_names())


def compatible_with_time_window(
    df: pl.DataFrame | pl.LazyFrame, time_window: TimeWindow
) -> bool:
    return compatible_with_time_anchor(
        df, time_window.tmin
    ) and compatible_with_time_anchor(df, time_window.tmax)


# ------------------------
# DataFrame time-filters for time anchors and windows
# ------------------------
def filter_with_time_anchor(
    df: PolarsFrame,
    time_anchor: TimeAnchor,
    reference_window: TimeWindow,
    *,
    closed: ClosedInterval = "both",
    keep_null: bool = True,
) -> PolarsFrame:
    return cast(
        "PolarsFrame",
        df.filter(
            WindowOverlap.contains_expr(
                time_anchor, reference_window, closed=closed
            ).fill_null(
                keep_null
            )  # Kleene logic fill
        ),
    )


def filter_with_time_window(
    df: PolarsFrame,
    time_window: TimeWindow,
    reference_window: TimeWindow,
    *,
    relation: WindowRelation = WindowRelation.ANY_OVERLAP,
    keep_null: bool = True,
) -> PolarsFrame:
    return cast(
        "PolarsFrame",
        df.filter(
            WindowOverlap.overlap_expr(
                time_window, reference_window, relation=relation
            ).fill_null(
                keep_null
            )  # Kleene logic fill
        ),
    )


# ------------------------
# Boundary checker
# ------------------------
def bounded_by_obs_level(
    time_window: TimeWindow, cohort: Cohort, obs_level: ObsLevel | None = None
) -> bool:
    _obs_level = obs_level or cohort.obs_level
    obs_level_time_window = TimeWindow.from_obs_level(_obs_level)
    return bool(
        cohort._obs.select(
            WindowOverlap.overlap_expr(
                time_window,
                obs_level_time_window,
                WindowRelation.CONTAINED,
            )
            .fill_null(True)  # Kleene logic fill
            .all()
        ).item()
    )
