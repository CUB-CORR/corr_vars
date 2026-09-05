from __future__ import annotations

import datetime

import polars as pl
import pytest

from corr_vars.core.variable import FreeDaysVariable
from corr_vars.utils.outcomes import free_days
from corr_vars.utils.time import TimeWindow

import types
from corr_vars.definitions.typing import ExtractedVariable


def _dt(iso: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(iso)


_T0 = _dt("2024-03-05T07:42:13")  # deliberately not midnight: blocks anchor to T0


def _h(hours: float) -> datetime.datetime:
    return _T0 + datetime.timedelta(hours=hours)


def _d(days: float) -> datetime.datetime:
    return _h(days * 24)


def _obs(
    *ids: str,
    horizon: str = "28d",
    death: dict[str, datetime.datetime] | None = None,
    censor: dict[str, datetime.datetime] | None = None,
) -> pl.DataFrame:
    """One observation per id, all sharing T0; optional per-id death / censor times."""
    death = death or {}
    censor = censor or {}
    return pl.DataFrame(
        {
            "icu_stay_id": list(ids),
            "t0": [_T0] * len(ids),
            "death_timestamp": [death.get(i) for i in ids],
            "censordate": [censor.get(i) for i in ids],
        },
        schema_overrides={
            "death_timestamp": pl.Datetime,
            "censordate": pl.Datetime,
        },
    ).with_columns(pl.col("t0").dt.offset_by(horizon).alias("end"))


def _episodes(*rows: tuple[str, float, float]) -> pl.DataFrame:
    """(id, start_day, end_day) triples -> an episodes frame keyed like obs."""
    return pl.DataFrame(
        [(case, _d(start), _d(end)) for case, start, end in rows],
        schema={
            "icu_stay_id": pl.String,
            "recordtime": pl.Datetime,
            "recordtime_end": pl.Datetime,
        },
        orient="row",
    )


_WINDOW = TimeWindow("t0", "end")


def _run(obs: pl.DataFrame, episodes: pl.DataFrame, **kwargs) -> dict[str, int | None]:
    out = free_days(
        obs,
        episodes,
        primary_key="icu_stay_id",
        window=_WINDOW,
        output_name="fd",
        **kwargs,
    )
    return dict(zip(out["icu_stay_id"].to_list(), out["fd"].to_list()))


class TestFreeDays:
    # --- the two methods diverge exactly on gaps ---
    def test_last_event_counts_only_the_tail_after_the_last_episode(self) -> None:
        """Extubated d5, re-intubated d10-12: free tail is the 16 days after d12."""
        res = _run(
            _obs("a"), _episodes(("a", 0, 5), ("a", 10, 12)), method="last_event"
        )
        assert res == {"a": 16}

    def test_day_grid_credits_the_gap_between_episodes(self) -> None:
        """Same episodes, but the free days d5-9 in the gap now count: 28 - 7 = 21."""
        res = _run(_obs("a"), _episodes(("a", 0, 5), ("a", 10, 12)), method="day_grid")
        assert res == {"a": 21}

    def test_no_episodes_means_the_whole_horizon_is_free(self) -> None:
        assert _run(_obs("a"), _episodes(), method="last_event") == {"a": 28}
        assert _run(_obs("a"), _episodes(), method="day_grid") == {"a": 28}

    def test_episode_covering_the_whole_horizon_gives_zero(self) -> None:
        res = _run(_obs("a"), _episodes(("a", 0, 40)), method="last_event")
        assert res == {"a": 0}

    # --- blocks are anchored at T0, not the wall clock ---
    def test_blocks_are_24h_from_t0_not_calendar_days(self) -> None:
        """T0 is 07:42; an episode ending at day-index 12 frees exactly days 12..27."""
        res = _run(_obs("a"), _episodes(("a", 0, 12)), method="last_event")
        assert res == {"a": 16}

    # --- death handling ---
    def test_zero_if_dead_wipes_the_count_for_any_death_in_horizon(self) -> None:
        res = _run(
            _obs("a", death={"a": _d(20)}),
            _episodes(("a", 0, 3)),
            method="last_event",
            death_rule="zero_if_dead",
            death_col="death_timestamp",
        )
        assert res == {"a": 0}

    def test_negative_if_death_scores_minus_one(self) -> None:
        """The newer ordinal definition: any death in the horizon scores -1."""
        res = _run(
            _obs("a", death={"a": _d(20)}),
            _episodes(("a", 0, 3)),
            method="last_event",
            death_rule="negative_if_death",
            death_col="death_timestamp",
        )
        assert res == {"a": -1}

    def test_negative_if_death_after_horizon_keeps_the_count(self) -> None:
        """Death beyond the horizon is not counted, so the normal tail is returned."""
        res = _run(
            _obs("a", death={"a": _d(40)}),
            _episodes(("a", 0, 3)),
            method="last_event",
            death_rule="negative_if_death",
            death_col="death_timestamp",
        )
        assert res == {"a": 25}

    def test_alive_days_keeps_free_days_before_death(self) -> None:
        """Off support after d3, dies d10: days 3..9 are free (7), the rest are dead."""
        res = _run(
            _obs("a", death={"a": _d(10)}),
            _episodes(("a", 0, 3)),
            method="last_event",
            death_rule="alive_days",
            death_col="death_timestamp",
        )
        assert res == {"a": 7}

    def test_death_after_horizon_does_not_zero_the_count(self) -> None:
        res = _run(
            _obs("a", death={"a": _d(40)}),
            _episodes(("a", 0, 3)),
            method="last_event",
            death_rule="zero_if_dead",
        )
        assert res == {"a": 25}

    # --- censoring ---
    def test_censored_horizon_is_null(self) -> None:
        res = _run(
            _obs("a", censor={"a": _d(5)}),
            _episodes(("a", 0, 3)),
            method="last_event",
            censor_col="censordate",
        )
        assert res == {"a": None}

    def test_censor_at_or_after_horizon_end_is_observed(self) -> None:
        res = _run(
            _obs("a", censor={"a": _d(40)}),
            _episodes(("a", 0, 3)),
            method="last_event",
            censor_col="censordate",
        )
        assert res == {"a": 25}

    def test_censoring_wins_over_zero_if_dead(self) -> None:
        """An unobservable horizon is null even if a death is recorded within it."""
        res = _run(
            _obs("a", death={"a": _d(4)}, censor={"a": _d(5)}),
            _episodes(("a", 0, 3)),
            method="last_event",
            death_rule="zero_if_dead",
            censor_col="censordate",
        )
        assert res == {"a": None}

    # --- day_grid recovery threshold (RRT) ---
    def test_recovery_threshold_keeps_days_blocked_after_the_episode_ends(self) -> None:
        """RRT ends d2; with a 7d recovery window days 2..8 stay non-free."""
        no_threshold = _run(_obs("a"), _episodes(("a", 0, 2)), method="day_grid")
        with_threshold = _run(
            _obs("a"),
            _episodes(("a", 0, 2)),
            method="day_grid",
            recovery_threshold="7d",
        )
        assert no_threshold == {"a": 26}  # only days 0,1 blocked
        assert with_threshold == {"a": 19}  # days 0..8 blocked (9 days)

    def test_recovery_threshold_ignored_for_last_event(self) -> None:
        res = _run(
            _obs("a"),
            _episodes(("a", 0, 2)),
            method="last_event",
            recovery_threshold="7d",
        )
        assert res == {"a": 26}

    # --- multiple observations are independent ---
    def test_observations_are_scored_independently(self) -> None:
        obs = _obs("a", "b", "c", death={"b": _d(10)})
        episodes = _episodes(("a", 0, 5), ("a", 10, 12), ("b", 0, 5), ("c", 0, 40))
        res = _run(
            obs,
            episodes,
            method="last_event",
            death_rule="zero_if_dead",
            death_col="death_timestamp",
        )
        assert res == {"a": 16, "b": 0, "c": 0}

    # --- edge cases ---
    def test_non_positive_horizon_scores_zero(self) -> None:
        obs = _obs("a", horizon="0d")
        assert _run(obs, _episodes(("a", 0, 1)), method="last_event") == {"a": 0}

    def test_null_end_episodes_are_dropped_not_stretched(self) -> None:
        """A point event (null recordtime_end) must not block the horizon."""
        eps = pl.DataFrame(
            {"icu_stay_id": ["a"], "recordtime": [_d(2)], "recordtime_end": [None]},
            schema_overrides={"recordtime": pl.Datetime, "recordtime_end": pl.Datetime},
        )
        assert _run(_obs("a"), eps, method="last_event") == {"a": 28}

    def test_episode_before_horizon_does_not_affect_last_event(self) -> None:
        """An interval ending before T0 is not relevant to the follow-up window."""
        res = _run(_obs("a"), _episodes(("a", -5, -1)), method="last_event")
        assert res == {"a": 28}

    def test_missing_death_column_disables_zero_if_dead(self) -> None:
        obs = _obs("a").drop("death_timestamp")
        res = _run(
            obs, _episodes(("a", 0, 3)), method="last_event", death_rule="zero_if_dead"
        )
        assert res == {"a": 25}

    def test_last_event_value_matches_horizon_minus_last_end(self) -> None:
        """Free days = clip((T0+H) - last_end, 0, H), in whole day-blocks."""
        for last_day in (0, 4, 15, 27, 28):
            res = _run(_obs("a"), _episodes(("a", 0, last_day)), method="last_event")
            assert res == {"a": max(0, 28 - last_day)}


class TestFreeDaysVariable:
    """The variable type is thin glue over the tested engine: it must load the single
    episode dependency, thread the window through, and return the named column.
    """

    def _make(self, **kwargs) -> FreeDaysVariable:
        var = FreeDaysVariable(
            "vfd",
            requires={
                "episodes": {
                    "template": "invasive_vent",
                    "tmin": "inherit",
                    "tmax": "inherit",
                }
            },
            **kwargs,
        )
        var.time_window = _WINDOW
        return var

    def _drive(
        self, var: FreeDaysVariable, obs: pl.DataFrame, episodes: pl.DataFrame
    ) -> dict[str, int | None]:
        # Bypass source loading: inject the episode dependency directly.
        var._get_required_vars = lambda cohort: None  # type: ignore[method-assign]
        var.required_vars = {
            "episodes": ExtractedVariable(
                var_name="invasive_vent", dynamic=True, data=episodes
            )
        }
        cohort = types.SimpleNamespace(_obs=obs, primary_key="icu_stay_id")
        out = var.extract(cohort)  # type: ignore[arg-type]
        assert out.columns == ["icu_stay_id", "vfd"]
        return dict(zip(out["icu_stay_id"].to_list(), out["vfd"].to_list()))

    def test_extract_delegates_to_engine(self) -> None:
        var = self._make(method="last_event")
        res = self._drive(var, _obs("a"), _episodes(("a", 0, 5), ("a", 10, 12)))
        assert res == {"a": 16}

    def test_extract_honours_method_and_death_rule(self) -> None:
        var = self._make(
            method="day_grid", death_rule="alive_days", death_col="death_timestamp"
        )
        res = self._drive(
            var, _obs("a", death={"a": _d(10)}), _episodes(("a", 0, 5), ("a", 10, 12))
        )
        assert res == {"a": 5}

    def test_censor_col_absent_is_tolerated(self) -> None:
        var = self._make(method="last_event", censor_col="nonexistent")
        res = self._drive(var, _obs("a").drop("censordate"), _episodes(("a", 0, 3)))
        assert res == {"a": 25}

    def test_censor_col_present_nulls_censored(self) -> None:
        var = self._make(method="last_event", censor_col="censordate")
        res = self._drive(var, _obs("a", censor={"a": _d(5)}), _episodes(("a", 0, 3)))
        assert res == {"a": None}

    def test_merge_gap_widens_the_episode_load_window(self) -> None:
        """merge_gap loads episodes a little past the horizon (so a re-event can bridge)
        then restores the true window for the engine.
        """
        var = self._make(method="last_event", merge_gap="48h")
        seen = {}

        def capture(*args):
            seen["tmax"] = repr(var.time_window.tmax)

        # Patch the base loader to record the window it is called with.
        from corr_vars.core.variable.base import Variable as _Base

        original = _Base._get_required_vars
        _Base._get_required_vars = capture  # type: ignore[method-assign]
        try:
            var._get_required_vars(cohort=object())  # type: ignore[arg-type]
        finally:
            _Base._get_required_vars = original  # type: ignore[method-assign]

        assert seen["tmax"] == "end + 48h"  # widened during load
        assert repr(var.time_window.tmax) == "end"  # restored after

    def test_requires_one_or_two_with_presence(self) -> None:
        # Three requirements is always too many (caught at construction).
        with pytest.raises(AssertionError):
            FreeDaysVariable("vfd", requires=["a", "b", "c"])

        # Two requirements are allowed only when one is the 'presence' observation
        # window; two plain episode requirements are rejected at extraction.
        var = self._make(method="last_event")
        var._get_required_vars = lambda cohort: None  # type: ignore[method-assign]
        var.required_vars = {
            "episodes": ExtractedVariable(
                var_name="a", dynamic=True, data=_episodes(("a", 0, 3))
            ),
            "other": ExtractedVariable(
                var_name="b", dynamic=True, data=_episodes(("a", 5, 6))
            ),
        }
        cohort = types.SimpleNamespace(_obs=_obs("a"), primary_key="icu_stay_id")
        with pytest.raises(AssertionError):
            var.extract(cohort)  # type: ignore[arg-type]

    def test_presence_masks_unobserved_free_days(self) -> None:
        """A day counts as free only if it is also observed (inside a presence
        interval); the unobserved tail after discharge no longer counts."""
        obs = _obs("a")
        episodes = _episodes(("a", 0, 3))  # last_event tail d3..d28 -> 25 free days
        # Observed only d0..d7: free-and-observed days are d3..d7 -> 4.
        presence = _episodes(("a", 0, 7))
        assert _run(obs, episodes, method="last_event")["a"] == 25
        assert _run(obs, episodes, method="last_event", presence=presence)["a"] == 4

    def test_extract_with_presence_requirement(self) -> None:
        var = self._make(method="last_event")
        var._get_required_vars = lambda cohort: None  # type: ignore[method-assign]
        var.required_vars = {
            "episodes": ExtractedVariable(
                var_name="invasive_vent", dynamic=True, data=_episodes(("a", 0, 3))
            ),
            "presence": ExtractedVariable(
                var_name="hospital_stays", dynamic=True, data=_episodes(("a", 0, 7))
            ),
        }
        cohort = types.SimpleNamespace(_obs=_obs("a"), primary_key="icu_stay_id")
        out = var.extract(cohort)  # type: ignore[arg-type]
        assert dict(zip(out["icu_stay_id"].to_list(), out["vfd"].to_list())) == {"a": 4}

    def test_with_detail_adds_named_struct_column(self) -> None:
        var = self._make(method="last_event", with_detail=True)
        var._get_required_vars = lambda cohort: None  # type: ignore[method-assign]
        var.required_vars = {
            "episodes": ExtractedVariable(
                var_name="invasive_vent", dynamic=True, data=_episodes(("a", 0, 10))
            )
        }
        cohort = types.SimpleNamespace(_obs=_obs("a"), primary_key="icu_stay_id")
        out = var.extract(cohort)  # type: ignore[arg-type]

        assert out.columns == ["icu_stay_id", "vfd", "vfd_detail"]
        assert out.schema["vfd_detail"] == pl.Struct
        detail = out.unnest("vfd_detail").to_dicts()[0]
        assert detail["event_type"] == "liberated"
        assert detail["event_time_days"] == 10.0


def _detail(obs, episodes, **kwargs):
    """Return {id: detail-dict} for the with_detail struct column."""
    out = free_days(
        obs,
        episodes,
        primary_key="icu_stay_id",
        window=_WINDOW,
        output_name="fd",
        with_detail=True,
        **kwargs,
    )
    assert "fd_detail" in out.columns
    # Drop the scalar so callers compare only the (method-agnostic) struct fields.
    rows = out.drop("fd").unnest("fd_detail").to_dicts()
    return {r["icu_stay_id"]: r for r in rows}


class TestFreeDaysDetail:
    """The {output_name}_detail struct: method-agnostic competing-risk components."""

    def test_liberated_survivor_reports_duration(self) -> None:
        """Extubated d10, survives: event_time = 10, terminal status 'liberated'."""
        det = _detail(_obs("a"), _episodes(("a", 0, 10)), method="last_event")["a"]
        assert det["event_type"] == "liberated"
        assert det["event_time_days"] == 10.0
        assert det["died_in_horizon"] is False
        assert det["censored"] is False

    def test_never_on_event_is_liberated_at_day_zero(self) -> None:
        det = _detail(_obs("a"), _episodes(), method="last_event")["a"]
        assert det["event_type"] == "liberated"
        assert det["event_time_days"] == 0.0

    def test_died_after_coming_off_has_no_duration(self) -> None:
        """Off d5, dies d20: status 'died', duration null (survivor-only component)."""
        det = _detail(
            _obs("a", death={"a": _d(20)}),
            _episodes(("a", 0, 5)),
            method="last_event",
            death_col="death_timestamp",
        )["a"]
        assert det["event_type"] == "died"
        assert det["event_time_days"] is None
        assert det["died_in_horizon"] is True

    def test_died_while_on_event(self) -> None:
        det = _detail(
            _obs("a", death={"a": _d(20)}),
            _episodes(("a", 0, 20)),
            method="last_event",
            death_col="death_timestamp",
        )["a"]
        assert det["event_type"] == "died"
        assert det["event_time_days"] is None

    def test_ongoing_at_horizon(self) -> None:
        """Still on the event past the horizon end -> 'ongoing_at_horizon'."""
        det = _detail(_obs("a"), _episodes(("a", 0, 40)), method="last_event")["a"]
        assert det["event_type"] == "ongoing_at_horizon"
        assert det["event_time_days"] is None
        assert det["died_in_horizon"] is False

    def test_censored_status(self) -> None:
        det = _detail(
            _obs("a", censor={"a": _d(5)}),
            _episodes(("a", 0, 3)),
            method="last_event",
            censor_col="censordate",
        )["a"]
        assert det["event_type"] == "censored"
        assert det["event_time_days"] is None
        assert det["censored"] is True

    def test_censored_takes_precedence_over_death(self) -> None:
        """Censored wins the terminal status, but the recorded death is still flagged."""
        det = _detail(
            _obs("a", death={"a": _d(4)}, censor={"a": _d(5)}),
            _episodes(("a", 0, 3)),
            method="last_event",
            censor_col="censordate",
            death_col="death_timestamp",
        )["a"]
        assert det["event_type"] == "censored"
        assert det["died_in_horizon"] is True

    def test_detail_is_method_agnostic(self) -> None:
        """Struct is ground truth: identical under last_event and day_grid."""
        obs, eps = _obs("a"), _episodes(("a", 0, 5), ("a", 10, 12))
        le = _detail(obs, eps, method="last_event")["a"]
        dg = _detail(obs, eps, method="day_grid")["a"]
        assert le == dg
        assert le["event_type"] == "liberated"
        assert le["event_time_days"] == 12.0  # final liberation, not the gap

    def test_unobserved_days_zero_without_presence(self) -> None:
        det = _detail(_obs("a"), _episodes(("a", 0, 3)), method="last_event")["a"]
        assert det["unobserved_days"] == 0

    def test_unobserved_days_counts_free_days_removed_by_presence(self) -> None:
        """With presence, unobserved_days = free days dropped for being unobserved;
        the competing-risk components stay exactly as without presence."""
        obs, eps = _obs("a"), _episodes(("a", 0, 3))
        presence = _episodes(("a", 0, 7))  # observed only d0..d7
        no_p = _detail(obs, eps, method="last_event")["a"]
        with_p = _detail(obs, eps, method="last_event", presence=presence)["a"]
        # 25 free without presence, 4 observed-free with it -> 21 removed.
        assert _run(obs, eps, method="last_event") == {"a": 25}
        assert _run(obs, eps, method="last_event", presence=presence) == {"a": 4}
        assert with_p["unobserved_days"] == 21
        assert no_p["unobserved_days"] == 0
        # presence must not touch the ground-truth fields:
        assert with_p["event_type"] == no_p["event_type"] == "liberated"
        assert with_p["event_time_days"] == no_p["event_time_days"] == 3.0
        assert with_p["died_in_horizon"] == no_p["died_in_horizon"] is False

    def test_unobserved_days_day_grid_counts_all_masked_days(self) -> None:
        """day_grid: every free-but-unobserved day is counted, not only a tail."""
        obs = _obs("a")
        eps = _episodes(("a", 0, 2), ("a", 20, 22))
        presence = _episodes(("a", 0, 10))  # observed only d0..d10
        with_p = _detail(obs, eps, method="day_grid", presence=presence)["a"]
        full = _run(obs, eps, method="day_grid")["a"]
        observed = _run(obs, eps, method="day_grid", presence=presence)["a"]
        assert with_p["unobserved_days"] == full - observed

    def test_without_detail_has_no_struct_column(self) -> None:
        out = free_days(
            _obs("a"),
            _episodes(("a", 0, 5)),
            primary_key="icu_stay_id",
            window=_WINDOW,
            output_name="fd",
        )
        assert out.columns == ["icu_stay_id", "fd"]


class TestFreeDaysMerge:
    """merge_gap unions episodes separated by at most the gap, before counting."""

    def test_reintubation_straddling_horizon_end_under_last_event(self) -> None:
        """The VFD case: extubated d27 (1d before horizon), reintubated d29-31 (1d
        after). Plain last_event credits the day-27 window (1); merging bridges the
        two runs so the clipped end reaches the horizon -> 0 free days.
        """
        eps = _episodes(("a", 0, 27), ("a", 29, 31))
        assert _run(_obs("a"), eps, method="last_event") == {"a": 1}
        assert _run(_obs("a"), eps, method="last_event", merge_gap="48h") == {"a": 0}

    def test_gap_wider_than_merge_is_kept(self) -> None:
        """A genuine >48h free window is not merged: the day-27 extubation stands."""
        eps = _episodes(("a", 0, 27), ("a", 29.5, 31))  # 2.5d gap > 48h
        assert _run(_obs("a"), eps, method="last_event", merge_gap="48h") == {"a": 1}

    def test_merge_is_noop_for_in_horizon_episodes_under_last_event(self) -> None:
        """last_event already ignores interior gaps, so merging in-horizon episodes
        does not change the count — only a bridge past the horizon does.
        """
        eps = _episodes(("a", 0, 5), ("a", 6, 8))
        assert _run(_obs("a"), eps, method="last_event") == _run(
            _obs("a"), eps, method="last_event", merge_gap="48h"
        )

    def test_day_grid_merge_drops_short_gaps_keeps_long_ones(self) -> None:
        short = _episodes(("a", 0, 5), ("a", 6, 8))  # 1d gap -> merged
        long = _episodes(("a", 0, 5), ("a", 20, 22))  # 15d gap -> kept
        assert _run(_obs("a"), short, method="day_grid") == {"a": 21}
        assert _run(_obs("a"), short, method="day_grid", merge_gap="48h") == {"a": 20}
        assert _run(_obs("a"), long, method="day_grid", merge_gap="48h") == {"a": 21}

    def test_n_episodes_differentiates_merged_runs(self) -> None:
        def n_eps(eps, **kw):
            out = free_days(
                _obs("a"),
                eps,
                primary_key="icu_stay_id",
                window=_WINDOW,
                output_name="fd",
                with_detail=True,
                **kw,
            )
            return out.unnest("fd_detail")["n_episodes"][0]

        merged = _episodes(("a", 0, 5), ("a", 6, 8))  # short gap -> 1 run
        distinct = _episodes(("a", 0, 5), ("a", 20, 22))  # long gap -> 2 runs
        assert n_eps(merged, merge_gap="48h") == 1
        assert n_eps(distinct, merge_gap="48h") == 2
        assert n_eps(_episodes()) == 0
