from __future__ import annotations

from corr_vars.core.variable.base import Variable
from corr_vars.utils.outcomes import free_days
from corr_vars.utils.time import TimeWindow

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import polars as pl

    from corr_vars.core import Cohort
    from corr_vars.utils.outcomes import DeathRule, FreeDaysMethod
    from corr_vars.utils.time import TimeAnchor

    from corr_vars.definitions.typing import (
        OffsetLike,
        RequirementsIterable,
        TimeAnchorColumn,
    )

__all__ = ["FreeDaysVariable"]


class FreeDaysVariable(Variable):
    """A source-agnostic "free-days-through-day-N" outcome.

    Counts the days in a follow-up horizon that are free of some clinical event —
    ventilator-free days, RRT-free days, vasopressor-free days, and so on. The horizon
    is the variable's time window: ``tmin`` is time zero (T0) and ``tmax`` is
    ``T0 + horizon``, so the usual ``add_variable(..., tmin=..., tmax=...)`` machinery
    selects the horizon (e.g. ``tmax=("icu_admission", "+28d")`` for a 28-day window).

    The variable declares exactly one dependency — the event's episode intervals — and
    delegates all counting to :func:`corr_vars.utils.outcomes.free_days`, so the same
    definition holds across every data source.

    Args:
        var_name: Name of the produced outcome column.
        requires: The episode source (e.g. ``invasive_vent``), aliased ``episodes``,
            plus an **optional** second requirement aliased ``presence`` carrying
            observation-window intervals (e.g. ``hospital_stays``). Declare them in dict
            form so they inherit the horizon and are screened across the patient's
            cases::

                "requires": {"episodes": {"template": "invasive_vent",
                                          "tmin": "inherit", "tmax": "inherit",
                                          "screened_obs_level": "patient"}}

            When ``presence`` is supplied, a day counts as free only if it is also
            observed (inside a presence interval), so an unobserved discharge →
            readmission gap is not miscredited as free; omit it to keep the standard
            convention.
        method: ``"last_event"`` (free tail after the last episode) or ``"day_grid"``
            (per-day union; gaps restore free days).
        death_rule: ``"zero_if_dead"`` or ``"alive_days"``.
        recovery_threshold: ``day_grid`` only — keep days non-free until this long after
            the last episode ends (e.g. ``"7d"`` for RRT).
        merge_gap: Merge episodes separated by at most this long into one before
            counting (e.g. ``"48h"`` so a reintubation within 48h does not count as
            ventilator-free).
        death_col: Column name or expression over ``cohort.obs`` giving the death
            timestamp. ``None`` (source-agnostic default) or a name absent from obs
            disables the death rule.
        censor_col: Column name or expression over ``cohort.obs`` giving the last
            observable time; when it precedes the horizon end the outcome is null.
            ``None`` or a name absent from obs disables censoring.
        with_detail: Also save a ``{save_as}_detail`` struct with the competing-risk
            components (``event_time_days``, ``event_type``, ``died_in_horizon``,
            ``censored``) that the VFD literature recommends reporting alongside the
            composite.
    """

    def __init__(
        self,
        var_name: str,
        requires: RequirementsIterable = [],
        *,
        method: FreeDaysMethod = "last_event",
        death_rule: DeathRule = "zero_if_dead",
        recovery_threshold: OffsetLike | None = None,
        merge_gap: OffsetLike | None = None,
        death_col: str | pl.Expr | None = None,
        censor_col: str | pl.Expr | None = None,
        with_detail: bool = False,
        tmin: TimeAnchorColumn | TimeAnchor | None = None,
        tmax: TimeAnchorColumn | TimeAnchor | None = None,
        dynamic: bool = False,
        **_ignored: Any,
    ) -> None:
        assert dynamic is False, "FreeDaysVariable is always static."
        assert 1 <= len(requires) <= 2, (
            f"FreeDaysVariable '{var_name}' needs one (episode) requirement, "
            "optionally plus a 'presence' observation-window requirement."
        )

        super().__init__(
            var_name=var_name,
            dynamic=False,
            requires=requires,
            tmin=tmin,
            tmax=tmax,
        )

        self.method: FreeDaysMethod = method
        self.death_rule: DeathRule = death_rule
        self.recovery_threshold = recovery_threshold
        self.merge_gap = merge_gap
        self.death_col = death_col
        self.censor_col = censor_col
        self.with_detail = with_detail

    def _get_required_vars(self, cohort: Cohort) -> None:
        """Load the episode dependency, widening its window by ``merge_gap``.

        Merging bridges an episode that resumes just past the horizon back onto the
        last in-horizon episode (e.g. a reintubation the day after the horizon ends
        makes the day-before-horizon extubation unsuccessful, so it should not count as
        free). For that episode to be available to merge, the dependency must be loaded
        a little past ``tmax`` — otherwise the pipeline filters it out before the engine
        sees it. The engine still clips to the true horizon, so widening only feeds the
        merge; it never extends the counted window.
        """
        if self.merge_gap is None:
            super()._get_required_vars(cohort)
            return

        true_window = self.time_window
        self.time_window = TimeWindow(
            true_window.tmin, true_window.tmax.shifted_by(self.merge_gap)
        )
        try:
            super()._get_required_vars(cohort)
        finally:
            self.time_window = true_window

    def extract(self, cohort: Cohort) -> pl.DataFrame:
        self._get_required_vars(cohort)
        # An optional "presence" requirement carries observation-window intervals; the
        # remaining single requirement is the event episode source.
        presence = (
            self.required_vars["presence"].data
            if "presence" in self.required_vars
            else None
        )
        episode_reqs = [v for k, v in self.required_vars.items() if k != "presence"]
        assert len(episode_reqs) == 1, (
            f"FreeDaysVariable '{self.var_name}' needs exactly one episode requirement "
            "(alias the observation window 'presence')."
        )
        episodes = episode_reqs[0].data

        self.data = free_days(
            cohort._obs,
            episodes,
            primary_key=cohort.primary_key,
            window=self.time_window,
            method=self.method,
            death_rule=self.death_rule,
            recovery_threshold=self.recovery_threshold,
            merge_gap=self.merge_gap,
            death_col=self.death_col,
            censor_col=self.censor_col,
            presence=presence,
            output_name=self.var_name,
            with_detail=self.with_detail,
        )
        return self.data
