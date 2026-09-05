from datetime import datetime, timedelta

import polars as pl
import pytest

from corr_vars.utils.time import (
    TimeAnchor,
    TimeWindow,
    compatible_with_time_anchor,
    format_delta,
    resolve_time_anchor,
    truncate_to_day,
)


def _eval(expr: pl.Expr, **cols) -> object:
    """Evaluate a single-value expression against a one-row frame."""
    return pl.DataFrame(cols).select(expr.alias("__r"))["__r"][0]


def test_timeanchor_string_expr_and_repr():
    anchor = TimeAnchor("recordtime")
    assert repr(anchor) == "recordtime"

    df = pl.DataFrame({"recordtime": [1, 2, 3]})
    res = df.select(anchor.expr.alias("time"))["time"].to_list()
    assert res == [1, 2, 3]


def test_timeanchor_tuple_expr_and_repr():
    # offset by one day
    anchor_plus = TimeAnchor(("ref", "1d"))
    r = repr(anchor_plus)
    assert r == "ref + 1d"

    # offset by minus one day
    anchor_minus = TimeAnchor(("ref", "-1d"))
    r = repr(anchor_minus)
    assert r == "ref - 1d"

    now = datetime(2020, 1, 1)
    df = pl.DataFrame({"ref": [now, now + timedelta(days=2)]}).select(
        anchor_plus.expr.alias("time_plus"), anchor_minus.expr.alias("time_minus")
    )
    res_plus = df["time_plus"].to_list()
    assert res_plus[0] == now + timedelta(days=1)
    assert res_plus[1] == now + timedelta(days=3)
    res_minus = df["time_minus"].to_list()
    assert res_minus[0] == now - timedelta(days=1)
    assert res_minus[1] == now + timedelta(days=1)


def test_timeanchor_is_relative_property():
    anchor = TimeAnchor("ref")
    assert anchor.is_relative is False

    anchor = TimeAnchor(("ref", "1d"))
    assert anchor.is_relative is True


def test_timeanchor_literal_and_relative_are_independent():
    """`is_literal` describes what the anchor is fixed to, `is_relative` whether
    it was moved — a shifted literal is both.
    """
    column = TimeAnchor("ref")
    assert (column.is_literal, column.is_relative) == (False, False)
    assert (
        column.shifted_by("1d").is_literal,
        column.shifted_by("1d").is_relative,
    ) == (
        False,
        True,
    )

    literal = TimeAnchor(datetime(2020, 1, 1))
    assert (literal.is_literal, literal.is_relative) == (True, False)

    shifted = literal.shifted_by("1d")
    assert (shifted.is_literal, shifted.is_relative) == (True, True)
    # A shifted literal still needs no column, so it fits any DataFrame.
    assert compatible_with_time_anchor(pl.DataFrame({"other": [1]}), shifted)


def test_format_delta_signs_every_offset():
    assert format_delta(("1h",)) == "+ 1h"
    assert format_delta(("-1h",)) == "- 1h"
    assert format_delta(("2h", "-1h")) == "+ 2h - 1h"
    assert format_delta(("1d", "-2h", "30m")) == "+ 1d - 2h + 30m"
    # A bare string is read as a single offset.
    assert format_delta("-1h") == "- 1h"


def test_timeanchor_equality_ignores_how_it_was_built():
    """Column and delta define a column anchor; the expression tree also records
    which construction path produced it, which must not decide equality.
    """
    assert TimeAnchor("ref").shifted_by("1d") == TimeAnchor(("ref", "1d"))
    assert TimeAnchor("ref").shifted_by("+1d") == TimeAnchor(("ref", "1d"))

    assert TimeAnchor("ref").shifted_by("1d") != TimeAnchor(("ref", "2d"))
    assert TimeAnchor("a").shifted_by("1d") != TimeAnchor(("b", "1d"))
    assert TimeAnchor("ref") != TimeAnchor("ref").shifted_by("1d")
    # Stacked offsets are their own chain, even when they add up to another one.
    assert TimeAnchor(("ref", "2h")).shifted_by("-1h") != TimeAnchor(("ref", "1h"))

    # A literal has nothing but its expression to compare.
    now = datetime(2020, 1, 1)
    assert TimeAnchor(now) == TimeAnchor(now)
    assert TimeAnchor(now) != TimeAnchor(now + timedelta(days=1))
    assert TimeAnchor(now) != TimeAnchor("ref")
    assert TimeWindow().tmin != TimeWindow().tmax


def test_timeanchor_delta_keeps_offsets_apart():
    """A chain is not a polars duration, so the offsets stay separate values."""
    assert TimeAnchor("ref").delta is None
    assert TimeAnchor(("ref", "+2h")).delta == ("2h",)
    assert TimeAnchor(("ref", "+2h")).shifted_by("-1h").delta == ("2h", "-1h")
    assert TimeAnchor(("ref", "1d")).shifted_by("-2h").shifted_by("30m").delta == (
        "1d",
        "-2h",
        "30m",
    )


def test_timeanchor_copy_is_independent():
    anchor = TimeAnchor(("ref", "2h"))
    clone = anchor.copy()
    clone.delta = ("99d",)
    clone.expr = pl.col("other").alias("time")

    assert clone is not anchor
    assert (anchor.column, anchor.delta) == ("ref", ("2h",))
    assert repr(anchor) == "ref + 2h"


def test_timeanchor_shifted_by():
    now = datetime(2020, 1, 1)
    df = pl.DataFrame({"ref": [now]})

    shifted = TimeAnchor("ref").shifted_by("-1h")
    assert repr(shifted) == "ref - 1h"
    assert df.select(shifted.expr).item() == now - timedelta(hours=1)

    # Deltas stack instead of replacing each other, and each keeps its sign.
    stacked = TimeAnchor(("ref", "+2h")).shifted_by("-1h")
    assert stacked.column == "ref"
    assert repr(stacked) == "ref + 2h - 1h"
    assert df.select(stacked.expr).item() == now + timedelta(hours=1)

    triple = TimeAnchor(("ref", "1d")).shifted_by("-2h").shifted_by("30m")
    assert repr(triple) == "ref + 1d - 2h + 30m"
    assert df.select(triple.expr).item() == now + timedelta(
        days=1, hours=-2, minutes=30
    )

    # A fixed timestamp is moved to the offset timestamp.
    literal = TimeAnchor(now).shifted_by("90m")
    assert df.select(literal.expr).item() == now + timedelta(minutes=90)

    # A date literal keeps a sub-day delta instead of dropping it.
    from_date = TimeAnchor(now.date()).shifted_by("1h")
    assert df.select(from_date.expr).item() == now + timedelta(hours=1)

    # An unbounded window bound is only a sentinel timestamp: moving it further
    # out leaves the representable datetime range, moving it inwards leaves it
    # unbounded for every practical purpose.
    with pytest.raises(ValueError):
        TimeWindow().tmin.shifted_by("-1h")
    with pytest.raises(ValueError):
        TimeWindow().tmax.shifted_by("1h")
    assert TimeWindow().tmax.shifted_by("-1h").is_literal

    with pytest.raises(ValueError):
        TimeAnchor("ref").shifted_by("not-a-delta")


def test_resolve_time_anchor_notations():
    parent = TimeAnchor("icu_admission")

    # "inherit" -> the parent's anchor itself
    assert resolve_time_anchor("inherit", parent) is parent

    # ("inherit", delta) and the bare-delta shorthand are equivalent
    tuple_form = resolve_time_anchor(("inherit", "-1h"), parent)
    bare_form = resolve_time_anchor("-1h", parent)
    assert tuple_form == bare_form
    assert repr(tuple_form) == "icu_admission - 1h"

    # Anything else is an anchor of its own and passes through untouched
    assert resolve_time_anchor("hospital_admission", parent) == "hospital_admission"
    assert resolve_time_anchor(("hospital_admission", "2h"), parent) == (
        "hospital_admission",
        "2h",
    )
    assert resolve_time_anchor(None, parent) is None


def test_timewindow_tmid_expr():
    window = TimeWindow(tmin="a", tmax="b")
    df = pl.DataFrame({"a": [1, 5], "b": [3, 7]})
    res = df.select(window.tmid_expr.alias("tmid"))["tmid"].to_list()
    assert res == [2, 6]


def test_timewindow_duration_expr():
    window = TimeWindow(tmin="a", tmax="b")
    df = pl.DataFrame({"a": [1, 5], "b": [3, 7]})
    res = df.select(window.duration_expr.alias("duration"))["duration"].to_list()
    assert res == [2, 2]


def test_timewindow_is_relative_property():
    window = TimeWindow(tmin="ref", tmax="b")
    assert window.is_relative is False

    window = TimeWindow(tmin=("ref", "+1d"), tmax="b")
    assert window.is_relative is True


def test_timewindow_any_timebound_invalid_checks():
    window = TimeWindow(tmin="tmin_dt", tmax="tmax_dt")

    d1 = datetime(9999, 1, 1)
    d2 = datetime(2000, 1, 1)
    df = pl.DataFrame({"tmin_dt": [d1, d2], "tmax_dt": [d2, None]})

    in_year = df.select(window.any_timebound_in_year_9999_expr.alias("flag"))[
        "flag"
    ].to_list()
    assert in_year == [True, False]

    is_null = df.select(window.any_timebound_null_expr.alias("flag"))["flag"].to_list()
    assert is_null == [False, True]


# ------------------------
# TimeAnchor: polars-expression bound
# ------------------------
def test_timeanchor_accepts_expr():
    anchor = TimeAnchor(pl.col("icu_admission").dt.truncate("1d"))
    assert anchor.is_relative is False
    assert anchor.is_literal is False  # references a column, not a literal
    assert anchor.expr.meta.root_names() == ["icu_admission"]


def test_timeanchor_literalness_preserved():
    # datetime/date literal -> is_literal True (unchanged behaviour)
    assert TimeAnchor(datetime(2020, 1, 1)).is_literal is True
    # a literal polars expression is also literal
    assert TimeAnchor(pl.lit(datetime(2020, 1, 1))).is_literal is True
    # a plain column is not
    assert TimeAnchor("ref").is_literal is False


def test_compatible_with_expr_anchor():
    df = pl.DataFrame({"icu_admission": [datetime(2024, 1, 1)]})
    present = TimeAnchor(pl.col("icu_admission").dt.truncate("1d"))
    absent = TimeAnchor(pl.col("missing").dt.truncate("1d"))
    assert compatible_with_time_anchor(df, present) is True
    assert compatible_with_time_anchor(df, absent) is False
    # a literal anchor is always compatible
    assert compatible_with_time_anchor(df, TimeAnchor(datetime(2024, 1, 1))) is True


# ------------------------
# truncate_to_day
# ------------------------
def test_truncate_to_day_calendar():
    for hour in (0, 5, 6, 22):
        t = datetime(2024, 3, 5, hour, 42, 13)
        assert _eval(truncate_to_day(pl.col("t")), t=[t]) == datetime(2024, 3, 5)


def test_truncate_to_day_icu_boundary_wraps_night_hours():
    # Before 06:00 rolls into the previous ICU day; 06:00 and after stay.
    cases = {
        datetime(2024, 3, 5, 0, 42): datetime(2024, 3, 4, 6, 0),
        datetime(2024, 3, 5, 5, 59): datetime(2024, 3, 4, 6, 0),
        datetime(2024, 3, 5, 6, 0): datetime(2024, 3, 5, 6, 0),
        datetime(2024, 3, 5, 22, 0): datetime(2024, 3, 5, 6, 0),
    }
    for t, expected in cases.items():
        assert _eval(truncate_to_day(pl.col("t"), day_start="6h"), t=[t]) == expected


def test_truncate_to_day_accepts_timedelta():
    t = datetime(2024, 3, 5, 5, 42)
    as_str = _eval(truncate_to_day(pl.col("t"), day_start="6h"), t=[t])
    as_td = _eval(truncate_to_day(pl.col("t"), day_start=timedelta(hours=6)), t=[t])
    assert as_str == as_td == datetime(2024, 3, 4, 6, 0)


def test_truncate_to_day_keeps_datetime_dtype():
    out = pl.DataFrame({"t": [datetime(2024, 3, 5, 12)]}).select(
        truncate_to_day(pl.col("t")).alias("d")
    )
    assert out.schema["d"] == pl.Datetime


# ------------------------
# TimeWindow.from_anchor_and_duration
# ------------------------
def test_from_anchor_and_duration_column_keeps_metadata():
    window = TimeWindow.from_anchor_and_duration("icu_admission", "28d")
    assert repr(window.tmin) == "icu_admission"
    assert repr(window.tmax) == "icu_admission + 28d"
    t0 = datetime(2024, 3, 5, 10, 0)
    row = {"icu_admission": [t0]}
    assert _eval(window.tmin_expr, **row) == t0
    assert _eval(window.tmax_expr, **row) == t0 + timedelta(days=28)


def test_from_anchor_and_duration_accepts_timedelta_and_tuple():
    window = TimeWindow.from_anchor_and_duration(
        ("icu_admission", "+2h"), timedelta(days=28)
    )
    t0 = datetime(2024, 3, 5, 10, 0)
    tmax = _eval(window.tmax_expr, icu_admission=[t0])
    assert tmax == t0 + timedelta(hours=2, days=28)


def test_from_anchor_and_duration_accepts_leading_sign():
    window = TimeWindow.from_anchor_and_duration("t0", "+28d")
    t0 = datetime(2024, 1, 1)
    assert _eval(window.tmax_expr, t0=[t0]) == t0 + timedelta(days=28)


# ------------------------
# TimeWindow.anchored_to_day
# ------------------------
def test_anchored_to_day_snaps_both_bounds_and_preserves_horizon():
    window = TimeWindow.from_anchor_and_duration("t0", "28d").anchored_to_day()
    t0 = datetime(2024, 3, 5, 22, 0)
    tmin = _eval(window.tmin_expr, t0=[t0])
    tmax = _eval(window.tmax_expr, t0=[t0])
    assert tmin == datetime(2024, 3, 5)
    assert tmax == datetime(2024, 4, 2)
    assert (tmax - tmin).days == 28


def test_anchored_to_day_icu_boundary():
    window = TimeWindow.from_anchor_and_duration("t0", "28d").anchored_to_day("6h")
    t0 = datetime(2024, 3, 5, 3, 0)  # 03:00 -> previous ICU day 06:00
    tmin = _eval(window.tmin_expr, t0=[t0])
    tmax = _eval(window.tmax_expr, t0=[t0])
    assert tmin == datetime(2024, 3, 4, 6, 0)
    assert (tmax - tmin).days == 28


def test_anchored_to_day_does_not_mutate_original():
    window = TimeWindow.from_anchor_and_duration("t0", "28d")
    snapped = window.anchored_to_day()
    t0 = datetime(2024, 3, 5, 22, 0)
    assert _eval(window.tmin_expr, t0=[t0]) == t0
    assert _eval(snapped.tmin_expr, t0=[t0]) == datetime(2024, 3, 5)
