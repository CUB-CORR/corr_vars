from __future__ import annotations

import polars as pl

from corr_vars import __, logger
from corr_vars.utils.base import as_expr, columns, normalize_offset
from corr_vars.utils.frames import time_difference

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from corr_vars.utils.time import TimeWindow

    from collections.abc import Collection
    from corr_vars.definitions.typing import OffsetLike

__all__ = [
    "DeathRule",
    "EventType",
    "FreeDaysMethod",
    "free_days",
]

FreeDaysMethod = Literal["last_event", "day_grid"]
DeathRule = Literal["zero_if_dead", "alive_days", "negative_if_death"]


def _optional_column_expr(
    value: str | pl.Expr | None, available: Collection[str]
) -> pl.Expr | None:
    """Coerce a column name or expression to an expression, or ``None`` if unset/absent.

    Unifies the two ways a column argument (``death_col`` / ``censor_col``) can be
    given — a column name or a ready polars expression — via
    :func:`~corr_vars.utils.base.as_expr`. A name referring to a column not present in
    ``obs`` yields ``None`` so the feature is simply skipped (a source without a death or
    censor column is handled gracefully rather than erroring).
    """
    if value is None or (isinstance(value, str) and value not in available):
        return None
    return as_expr(value)


# Canonical column names used inside the day-grid pipeline.
_T0 = "__t0"
_END = "__end"
_N_DAYS = "__n_days"
_DEATH = "__death"
_CENSORED = "__censored"
_DAY = "__day"
_BLOCK_START = "__block_start"
_BLOCK_END = "__block_end"
_EP_START = "__ep_start"
_EP_END = "__ep_end"
_BLOCKED = "__blocked"
_FREE = "__free"
_LAST_END = "__last_end"
_N_EPISODES = "__n_episodes"
_MERGE_GROUP = "__merge_group"
_STARTS_NEW = "__starts_new"
_OBSERVED = "__observed"
_UNOBSERVED = "__unobserved"

_ONE_DAY = pl.duration(days=1)


def _merge_episodes(
    episodes: pl.DataFrame,
    *,
    primary_key: str,
    recordtime: str,
    recordtime_end: str,
    gap: OffsetLike,
) -> pl.DataFrame:
    """Merge episodes separated by at most ``gap`` into one continuous interval.

    Interval union with a tolerance, per key: sorted by start, a new merged interval
    begins only when an episode starts more than ``gap`` after the furthest end seen so
    far. This bridges short gaps — e.g. a reintubation within 48h — so the brief window
    between the two runs is not treated as event-free. Genuine long gaps (beyond
    ``gap``) are left as separate episodes.

    Returns the merged intervals under the original ``recordtime`` / ``recordtime_end``
    names (other columns are dropped; only the interval is used downstream).
    """
    intervals = episodes.select(
        primary_key,
        pl.col(recordtime).alias(_EP_START),
        pl.col(recordtime_end).alias(_EP_END),
    ).filter(pl.col(_EP_START).is_not_null() & pl.col(_EP_END).is_not_null())

    # Furthest end among all earlier episodes of this key (nested/overlapping safe).
    prior_max_end = (
        pl.col(_EP_END).cum_max().shift(1).over(primary_key, order_by=_EP_START)
    )
    starts_new = (
        pl.col(_EP_START)
        .gt(prior_max_end.dt.offset_by(normalize_offset(gap)))
        .fill_null(True)
    )

    # ``starts_new`` is itself a windowed expression (via ``prior_max_end``), so it
    # must be materialised into a plain column before the second window — nesting a
    # window inside another window's context is rejected by polars
    return (
        intervals.with_columns(starts_new.alias(_STARTS_NEW))
        .with_columns(
            pl.col(_STARTS_NEW)
            .cum_sum()
            .over(primary_key, order_by=_EP_START)
            .alias(_MERGE_GROUP)
        )
        .group_by(primary_key, _MERGE_GROUP)
        .agg(
            pl.min(_EP_START).alias(recordtime),
            pl.max(_EP_END).alias(recordtime_end),
        )
        .drop(_MERGE_GROUP)
    )


#: Terminal follow-up status recorded in the detail struct. Concept-neutral so the
#: same schema serves ventilation, vasopressor, RRT, ... — the "event" is named by the
#: variable itself.
EventType = Literal["liberated", "died", "censored", "ongoing_at_horizon"]


def _horizon_bounds(
    obs: pl.DataFrame,
    *,
    primary_key: str,
    window: TimeWindow,
    death_col: str | pl.Expr | None,
    censor_col: str | pl.Expr | None,
) -> pl.DataFrame:
    """One row per observation carrying the follow-up window and its censoring state.

    ``T0`` is the window's ``tmin`` (time zero) and ``END`` is its ``tmax``
    (``T0 + horizon``); the horizon length in whole 24h days is derived from them.
    ``censored`` marks observations whose horizon reaches past the last observable
    time, so the outcome cannot be known and must become null.
    """
    available = columns(obs)
    death_expr = _optional_column_expr(death_col, available)
    death = (
        death_expr.alias(_DEATH)
        if death_expr is not None
        else pl.lit(None, dtype=pl.Datetime).alias(_DEATH)
    )
    censor_expr = _optional_column_expr(censor_col, available)
    censored = (
        censor_expr.lt(window.tmax_expr).fill_null(False).alias(_CENSORED)
        if censor_expr is not None
        else pl.lit(False).alias(_CENSORED)
    )

    return obs.select(
        primary_key,
        window.tmin_expr.alias(_T0),
        window.tmax_expr.alias(_END),
        # Round (not floor) so a DST transition inside the horizon (a 28-day window
        # spanning a clock change is 671h or 673h, not 672h) still yields 28 days.
        window.time_difference(unit="d", total=False)
        .round()
        .cast(pl.Int64)
        .alias(_N_DAYS),
        death,
        censored,
    )


def _day_grid(bounds: pl.DataFrame, *, primary_key: str) -> pl.DataFrame:
    """Explode each observation into its ``n_days`` fixed 24h blocks anchored at T0.

    Block ``d`` covers ``[T0 + d*24h, T0 + (d+1)*24h)``. Observations with a
    non-positive horizon contribute no blocks (and score 0 free days).
    """
    return (
        bounds.with_columns(
            pl.int_ranges(0, pl.max_horizontal(pl.col(_N_DAYS), 0)).alias(_DAY)
        )
        .explode(_DAY)
        # Exploding an empty range yields a single null row, not zero rows; a
        # non-positive horizon must contribute no day-blocks at all.
        .drop_nulls(_DAY)
        .with_columns(
            pl.col(_T0).add(pl.duration(days=pl.col(_DAY))).alias(_BLOCK_START)
        )
        .with_columns(pl.col(_BLOCK_START).add(_ONE_DAY).alias(_BLOCK_END))
    )


def _relevant_episodes(
    episodes: pl.DataFrame,
    bounds: pl.DataFrame,
    *,
    primary_key: str,
    recordtime: str,
    recordtime_end: str,
) -> pl.DataFrame:
    """Episodes overlapping the horizon, with their end clipped to ``END``.

    Clipping the end to ``END`` implements the "within horizon" cap: an episode
    running past the horizon counts only up to it. Starts are left untouched — the
    daily predicate only compares them against block ends.
    """
    return (
        episodes.select(
            primary_key,
            pl.col(recordtime).alias(_EP_START),
            pl.col(recordtime_end).alias(_EP_END),
        )
        # An episode needs a genuine interval end. A null end cannot be clipped
        # (min_horizontal(null, END) would spuriously stretch it to END and block
        # the rest of the horizon), so such rows are dropped: the episode source
        # must supply recordtime_end, not point events.
        .filter(pl.col(_EP_START).is_not_null() & pl.col(_EP_END).is_not_null())
        .join(bounds.select(primary_key, _T0, _END), on=primary_key, how="inner")
        .with_columns(pl.min_horizontal(_EP_END, _END).alias(_EP_END))
        .filter(pl.col(_EP_END).gt(pl.col(_T0)) & pl.col(_EP_START).lt(pl.col(_END)))
        .select(primary_key, _EP_START, _EP_END)
    )


def _observed_days(
    presence: pl.DataFrame,
    grid: pl.DataFrame,
    bounds: pl.DataFrame,
    *,
    primary_key: str,
    recordtime: str,
    recordtime_end: str,
) -> pl.DataFrame:
    """Per day-block flag: is the block covered by an observation (presence) interval.

    Presence intervals are clipped to the horizon; an open (null-end) interval is read
    as observed through ``END`` (still admitted at the horizon end) rather than dropped
    as :func:`_relevant_episodes` does for events. A block is observed when any presence
    interval overlaps it.
    """
    intervals = (
        presence.select(
            primary_key,
            pl.col(recordtime).alias(_EP_START),
            pl.col(recordtime_end).alias(_EP_END),
        )
        .filter(pl.col(_EP_START).is_not_null())
        .join(bounds.select(primary_key, _T0, _END), on=primary_key, how="inner")
        # Open interval (still admitted) -> observed through END, not dropped.
        .with_columns(
            pl.min_horizontal(
                pl.col(_EP_END).fill_null(pl.col(_END)), pl.col(_END)
            ).alias(_EP_END)
        )
        .filter(pl.col(_EP_END).gt(pl.col(_T0)) & pl.col(_EP_START).lt(pl.col(_END)))
        .select(primary_key, _EP_START, _EP_END)
    )
    overlaps = pl.col(_EP_START).lt(pl.col(_BLOCK_END)) & pl.col(_EP_END).gt(
        pl.col(_BLOCK_START)
    )
    return (
        grid.join(intervals, on=primary_key, how="left")
        .group_by(primary_key, _DAY)
        .agg(overlaps.any().fill_null(False).alias(_OBSERVED))
    )


def _blocked_predicate(
    *, method: FreeDaysMethod, recovery_threshold: OffsetLike | None
) -> pl.Expr:
    """Whether an episode blocks (makes non-free) the day it is joined against.

    - ``last_event``: any episode ending after the block start blocks the day, so
      the free tail is everything after the single last episode end — gaps do not
      restore freedom.
    - ``day_grid``: an episode blocks the day if it starts before the block ends and
      ends after ``block_start - recovery_threshold``. With no threshold this is plain
      overlap; with one, a recently ended episode keeps the day non-free until the
      recovery period has elapsed.
    """
    recovery_start = pl.col(_BLOCK_START)
    if recovery_threshold is not None:
        offset = normalize_offset(recovery_threshold).lstrip("+")
        recovery_start = recovery_start.dt.offset_by(f"-{offset}")

    if method == "last_event":
        return pl.col(_EP_END).gt(pl.col(_BLOCK_START))

    return pl.col(_EP_START).lt(pl.col(_BLOCK_END)) & pl.col(_EP_END).gt(recovery_start)


def _detail_frame(
    bounds: pl.DataFrame,
    relevant: pl.DataFrame,
    *,
    primary_key: str,
    output_name: str,
    unobserved: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build the ``{output_name}_detail`` struct of method-agnostic components.

    The struct reports what the paper (Yehya et al. 2019) insists be reported
    alongside the composite: the mortality component, the duration component, and the
    follow-up status — the inputs to a competing-risk (Fine-Gray) analysis. These are
    *ground truth*, independent of the scalar's ``method`` / ``death_rule``.

    Fields:
        event_time_days: Elapsed days T0 → final liberation, i.e. the duration
            component. Populated only for the ``"liberated"`` terminal status (survivors
            who came off the event), matching the paper's "ventilator days in 28-d
            survivors"; ``null`` otherwise.
        event_type: Terminal status, in precedence order ``censored`` >
            ``died`` > ``ongoing_at_horizon`` > ``liberated``.
        died_in_horizon: A recorded death within ``[T0, END]`` (the mortality
            component); an independent fact, so it may be ``True`` even when
            ``event_type`` is ``"censored"``.
        censored: The horizon was not fully observed.
        n_episodes: Number of (post-merge) event runs overlapping the horizon — e.g.
            distinct ventilation runs. ``> 1`` marks a genuine re-event (a gap wider
            than any ``merge_gap``); short reintubations are already merged away.
        unobserved_days: Days that would have counted as free but were removed because
            the patient was not observed (only when a ``presence`` window is given, else
            ``0``) — the size of the observation-gap correction. See :func:`free_days`.
    """
    per_key = relevant.group_by(primary_key).agg(
        pl.max(_EP_END).alias(_LAST_END),
        pl.len().alias(_N_EPISODES),
    )
    base = bounds.select(primary_key, _T0, _END, _DEATH, _CENSORED).join(
        per_key, on=primary_key, how="left"
    )
    if unobserved is not None:
        base = base.join(unobserved, on=primary_key, how="left")
    unobserved_days = (
        pl.col(_UNOBSERVED).fill_null(0).cast(pl.Int32)
        if unobserved is not None
        else pl.lit(0, dtype=pl.Int32)
    )

    died = pl.col(_DEATH).is_not_null() & pl.col(_DEATH).le(pl.col(_END))
    # Episode ends are clipped to END, so last_end == END means the patient was still
    # on the event at the horizon end (never liberated within it).
    ongoing = pl.col(_LAST_END).is_not_null() & pl.col(_LAST_END).ge(pl.col(_END))
    event_type = (
        pl.when(pl.col(_CENSORED))
        .then(pl.lit("censored"))
        .when(died)
        .then(pl.lit("died"))
        .when(ongoing)
        .then(pl.lit("ongoing_at_horizon"))
        .otherwise(pl.lit("liberated"))
    )

    # No episodes at all -> never on the event -> liberated at day 0.
    liberation_days = (
        pl.when(pl.col(_LAST_END).is_null())
        .then(pl.lit(0.0))
        .otherwise(time_difference(_LAST_END, _T0, unit="d", total=False))
    )
    event_time_days = (
        pl.when(event_type.eq("liberated")).then(liberation_days).otherwise(None)
    )

    detail = pl.struct(
        event_time_days.cast(pl.Float64).alias("event_time_days"),
        event_type.alias("event_type"),
        died.alias("died_in_horizon"),
        pl.col(_CENSORED).alias("censored"),
        pl.col(_N_EPISODES).fill_null(0).cast(pl.Int32).alias("n_episodes"),
        unobserved_days.alias("unobserved_days"),
    ).alias(f"{output_name}_detail")

    return base.select(primary_key, detail)


def free_days(
    obs: pl.DataFrame,
    episodes: pl.DataFrame,
    *,
    primary_key: str,
    window: TimeWindow,
    method: FreeDaysMethod = "last_event",
    death_rule: DeathRule = "zero_if_dead",
    recovery_threshold: OffsetLike | None = None,
    death_col: str | pl.Expr | None = None,
    censor_col: str | pl.Expr | None = None,
    presence: pl.DataFrame | None = None,
    recordtime: str = "recordtime",
    recordtime_end: str = "recordtime_end",
    merge_gap: OffsetLike | None = None,
    output_name: str = "free_days",
    with_detail: bool = False,
) -> pl.DataFrame:
    """Count the days in a follow-up horizon free of some clinical event.

    The horizon ``[T0, T0 + H]`` is taken from ``window`` (``T0 = tmin``,
    ``H = tmax - tmin`` in whole 24h days) and split into ``H`` day-blocks anchored at
    ``T0``. Each block is scored free or not by ``method``; the free blocks are summed.

    Args:
        obs: One row per observation, providing the window anchor columns, the death
            column and (optionally) the censoring expression's inputs.
        episodes: Event intervals (``recordtime`` / ``recordtime_end``) for the
            underlying concept, e.g. ventilation or vasopressor episodes. May hold
            several rows per observation, or none (then every day is free).
        primary_key: The observation key shared by ``obs`` and ``episodes``.
        window: The follow-up window; ``tmin`` is T0, ``tmax`` is ``T0 + horizon``.
        method: ``"last_event"`` (free tail after the last episode end) or
            ``"day_grid"`` (per-day union; gaps restore freedom).
        death_rule: ``"zero_if_dead"`` (any death in the horizon scores 0, ARDSnet),
            ``"negative_if_death"`` (death scores -1, the newer ordinal definition that
            ranks death below every survivor), or ``"alive_days"`` (days before death
            still count).
        recovery_threshold: ``day_grid`` only — a day stays non-free until this long
            after the last episode end (e.g. ``"7d"`` for RRT recovery).
        death_col: Column name or expression over ``obs`` giving the death timestamp.
            ``None`` (the default, source-agnostic) or a name absent from ``obs``
            disables the death rule.
        censor_col: Column name or expression over ``obs`` giving the last observable
            time per observation; when it precedes ``END`` the horizon is unobserved and
            the result is null. ``None`` or a name absent from ``obs`` disables
            censoring.
        presence: Optional observation-window intervals (``recordtime`` /
            ``recordtime_end``, e.g. in-hospital stays), keyed by ``primary_key``. When
            given, a day counts as free only if it is **also** observed (covered by a
            presence interval) — the standard scoring intersected with presence — so an
            unobserved gap (discharge → readmission) is not miscredited as free. An open
            (null-end) interval is treated as observed through ``END``. ``None`` keeps
            the standard behaviour (every day in ``[T0, END]`` is a candidate).
        recordtime / recordtime_end: Episode interval columns in ``episodes``.
        merge_gap: When set (e.g. ``"48h"``), episodes separated by at most this long
            are first merged into one continuous interval (:func:`_merge_episodes`), so
            a short gap — a reintubation within 48h — does not count as free. Under
            ``last_event`` this bites when a re-event just past the horizon bridges the
            last in-horizon episode; under ``day_grid`` it also drops short interior
            gaps while keeping genuine long ones.
        output_name: Name of the produced count column.
        with_detail: Also emit a ``{output_name}_detail`` struct carrying the
            competing-risk components (``event_time_days``, ``event_type``,
            ``died_in_horizon``, ``censored``, ``n_episodes``) plus ``unobserved_days``
            (free days removed by the ``presence`` intersection, else ``0``). See
            :func:`_detail_frame`.

    Returns:
        pl.DataFrame: ``[primary_key, output_name]`` (one row per observation, an
        ``Int32`` free-day count, null where censored), plus a
        ``{output_name}_detail`` struct column when ``with_detail`` is set.
    """
    if recovery_threshold is not None and method != "day_grid":
        logger.warning(
            __(
                "recovery_threshold is only used by method='day_grid'; ignoring it "
                "for method='{method}'.",
                method=method,
            )
        )
        recovery_threshold = None

    if merge_gap is not None:
        episodes = _merge_episodes(
            episodes,
            primary_key=primary_key,
            recordtime=recordtime,
            recordtime_end=recordtime_end,
            gap=merge_gap,
        )

    bounds = _horizon_bounds(
        obs,
        primary_key=primary_key,
        window=window,
        death_col=death_col,
        censor_col=censor_col,
    )
    grid = _day_grid(bounds, primary_key=primary_key)
    relevant = _relevant_episodes(
        episodes,
        bounds,
        primary_key=primary_key,
        recordtime=recordtime,
        recordtime_end=recordtime_end,
    )

    blocked = _blocked_predicate(method=method, recovery_threshold=recovery_threshold)

    # A day is free unless some episode blocks it. Days survive death only under
    # alive_days; under zero_if_dead that is overridden to 0 below.
    day_is_free = pl.col(_BLOCKED).not_()
    if death_rule == "alive_days":
        day_is_free = day_is_free & (
            pl.col(_DEATH).is_null() | pl.col(_DEATH).ge(pl.col(_BLOCK_END))
        )

    per_day = (
        grid.join(relevant, on=primary_key, how="left")
        .group_by(primary_key, _DAY)
        .agg(
            blocked.any().fill_null(False).alias(_BLOCKED),
            pl.first(_BLOCK_END).alias(_BLOCK_END),
            pl.first(_DEATH).alias(_DEATH),
        )
        .with_columns(day_is_free.alias(_FREE))
    )

    # Presence intersection: a day counts as free only if it is also observed. Only
    # days inside a presence interval survive; the standard scoring is otherwise kept.
    unobserved_counts = None
    if presence is not None:
        observed = _observed_days(
            presence,
            grid,
            bounds,
            primary_key=primary_key,
            recordtime=recordtime,
            recordtime_end=recordtime_end,
        )
        per_day = per_day.join(
            observed, on=[primary_key, _DAY], how="left"
        ).with_columns(pl.col(_OBSERVED).fill_null(False).alias(_OBSERVED))
        if with_detail:
            # Days that would have counted as free but fall outside the observation
            # window (removed by the intersection): for day_grid every such day, for
            # last_event those in the free tail after the last episode.
            unobserved_counts = per_day.group_by(primary_key).agg(
                pl.col(_FREE)
                .and_(pl.col(_OBSERVED).not_())
                .sum()
                .cast(pl.Int32)
                .alias(_UNOBSERVED)
            )
        per_day = per_day.with_columns(
            pl.col(_FREE).and_(pl.col(_OBSERVED)).alias(_FREE)
        )

    counts = per_day.group_by(primary_key).agg(
        pl.col(_FREE).sum().cast(pl.Int32).alias(output_name)
    )

    # Observations with no day-blocks (non-positive horizon) dropped out of the grid;
    # restore them at 0, then apply the death and censoring overrides.
    result = bounds.select(primary_key, _END, _DEATH, _CENSORED).join(
        counts, on=primary_key, how="left"
    )

    free = pl.col(output_name).fill_null(0)
    if death_rule in ("zero_if_dead", "negative_if_death"):
        # Death in the horizon overrides the count: 0 (ARDSnet) or -1 (the newer
        # ordinal definition that weights death below every survivor outcome).
        dead_score = -1 if death_rule == "negative_if_death" else 0
        died_in_horizon = pl.col(_DEATH).is_not_null() & pl.col(_DEATH).le(pl.col(_END))
        free = (
            pl.when(died_in_horizon)
            .then(pl.lit(dead_score, dtype=pl.Int32))
            .otherwise(free)
        )
    # Censored horizons are unobservable -> null wins over any count.
    free = pl.when(pl.col(_CENSORED)).then(None).otherwise(free)

    out = result.select(primary_key, free.alias(output_name))
    if with_detail:
        detail = _detail_frame(
            bounds,
            relevant,
            primary_key=primary_key,
            output_name=output_name,
            unobserved=unobserved_counts,
        )
        out = out.join(detail, on=primary_key, how="left")
    return out
