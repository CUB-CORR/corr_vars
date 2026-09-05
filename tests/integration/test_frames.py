import datetime
from string import ascii_uppercase

import hypothesis.strategies as st
import pandas.testing as pd_testing
import polars as pl
import polars.testing as pl_testing
import pytest
from hypothesis import example, given
from polars.testing.parametric import column, dataframes, lists

from corr_vars.utils.base import (
    as_expr,
    col_length,
    columns,
    is_empty,
    row_length,
    struct_fields,
)
from corr_vars.utils.frames import (
    absolute_and_relative_value_counts,
    apply_cleaning,
    attach_asof_value,
    build_attributes,
    convert_to_pandas_df,
    convert_to_polars_df,
    convert_to_polars_lf,
    remove_asof,
    time_difference,
    unique_sucessive,
    unnest,
)

from typing import cast

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

col_name_st = st.text(alphabet=ascii_uppercase, min_size=1, max_size=10)

plain_df_st = dataframes(
    min_cols=1,
    max_cols=6,
    min_size=0,
    max_size=20,
    allowed_dtypes=[
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
        pl.Boolean,
        pl.String,
    ],
)


def _dt(iso: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(iso)


_DT_MIN = _dt("2000-01-01T00:00:00")
_DT_MAX = _dt("2030-12-31T00:00:00")
datetime_st = st.datetimes(min_value=_DT_MIN, max_value=_DT_MAX)


# ------------------------
# Converter
# ------------------------
@given(
    df=dataframes(
        min_size=5,
        max_size=10,
        allow_null=False,
        allowed_dtypes=[pl.Int64, pl.Utf8, pl.Datetime],
    )
)
def test_convert_to_polars_df(
    df: pl.DataFrame,
) -> None:
    df_pl = df

    # polars DataFrame passes through
    out = convert_to_polars_df(df_pl)
    pl_testing.assert_frame_equal(out, df_pl)

    # polars LazyFrame -> polars DataFrame
    df_lazy = df.lazy()
    out_lazy = convert_to_polars_df(df_lazy)
    pl_testing.assert_frame_equal(out_lazy, df_pl)

    # pandas DataFrame -> polars DataFrame
    df_pd = df.to_pandas()
    out_pd = convert_to_polars_df(df_pd)
    pl_testing.assert_frame_equal(out_pd, df_pl)

    # unsupported type
    with pytest.raises(TypeError):
        convert_to_polars_df("not a df")  # type: ignore


@given(
    lf=dataframes(
        min_size=5,
        max_size=10,
        allow_null=False,
        allowed_dtypes=[pl.Int64, pl.Utf8, pl.Datetime],
        lazy=True,
    )
)
def test_convert_to_polars_lf(
    lf: pl.LazyFrame,
) -> None:
    lf_pl = lf

    # polars DataFrame -> polars LazyFrame
    df_pl = lf_pl.collect()
    out_pl = convert_to_polars_lf(df_pl)
    pl_testing.assert_frame_equal(out_pl, lf_pl)

    # polars LazyFrame passes through
    out = convert_to_polars_lf(lf_pl)
    pl_testing.assert_frame_equal(out, lf_pl)

    # pandas DataFrame -> polars LazyFrame
    df_pd = lf.collect().to_pandas()
    out_pd = convert_to_polars_lf(df_pd)
    pl_testing.assert_frame_equal(out_pd, lf_pl)

    # unsupported type
    with pytest.raises(TypeError):
        convert_to_polars_lf("not a df")  # type: ignore


@given(
    df=dataframes(
        min_size=5,
        max_size=10,
        allow_null=False,
        allowed_dtypes=[pl.Int64, pl.Utf8, pl.Datetime],
    )
)
def test_convert_to_pandas_df(df: pl.DataFrame) -> None:
    df_pd = df.to_pandas()

    # polars DataFrame -> pandas DataFrame
    df_pl = df
    out = convert_to_pandas_df(df_pl)
    pd_testing.assert_frame_equal(out, df_pd)

    # polars LazyFrame -> pandas DataFrame
    df_lazy = df_pl.lazy()
    out_lazy = convert_to_pandas_df(df_lazy)
    pd_testing.assert_frame_equal(out_lazy, df_pd)

    # pandas DataFrame passes through
    out_pd = convert_to_pandas_df(df_pd)
    pd_testing.assert_frame_equal(out_pd, df_pd)

    # unsupported
    with pytest.raises(TypeError):
        convert_to_pandas_df(123)  # type: ignore


# ------------------------
# Quality of life improvements
# ------------------------


# ===========================================================================
# absolute_and_relative_value_counts
# ===========================================================================
class TestAbsoluteAndRelativeValueCounts:
    # ------------------------------------------------------------------
    # Output schema
    # ------------------------------------------------------------------

    def test_output_columns_single_col(self) -> None:
        df = pl.DataFrame({"grp": ["a", "b", "b"]})
        res = absolute_and_relative_value_counts(df, cols="grp")
        assert {"grp", "absolute (int)", "relative (%)"}.issubset(set(res.columns))

    def test_output_columns_multi_col(self) -> None:
        df = pl.DataFrame({"grp": ["a", "a"], "sub": ["x", "y"]})
        res = absolute_and_relative_value_counts(df, cols=["grp", "sub"])
        assert {"grp", "sub", "absolute (int)", "relative (%)"}.issubset(
            set(res.columns)
        )

    def test_relative_column_is_float(self) -> None:
        df = pl.DataFrame({"grp": ["a", "b"]})
        res = absolute_and_relative_value_counts(df, cols="grp")
        assert res["relative (%)"].dtype == pl.Float64

    # ------------------------------------------------------------------
    # Absolute counts
    # ------------------------------------------------------------------

    @given(
        df=dataframes(
            cols=[
                column("grp", strategy=st.sampled_from(list(ascii_uppercase[:5]))),
                column("val", strategy=st.integers(min_value=1, max_value=5)),
            ],
            min_size=5,
            max_size=10,
            allow_null=False,
        ),
    )
    def test_absolute_counts_sum_to_input_length(self, df: pl.DataFrame) -> None:
        res = absolute_and_relative_value_counts(df, cols="grp", sort="cols")
        assert sum(res["absolute (int)"].to_list()) == len(df)

    def test_absolute_counts_hardcoded(self) -> None:
        df = pl.DataFrame({"grp": ["a", "a", "b", "b", "b"], "val": [1, 2, 3, 4, 5]})
        res = absolute_and_relative_value_counts(
            df.lazy(), cols="grp", sort="cols", decimals=1
        )
        assert res.shape[0] == 2
        abs_counts = dict(zip(res["grp"].to_list(), res["absolute (int)"].to_list()))
        assert abs_counts == {"a": 2, "b": 3}

    def test_absolute_count_single_group(self) -> None:
        df = pl.DataFrame({"grp": ["x", "x", "x"]})
        res = absolute_and_relative_value_counts(df, cols="grp")
        assert res["absolute (int)"][0] == 3

    # ------------------------------------------------------------------
    # Relative counts
    # ------------------------------------------------------------------

    @given(
        df=dataframes(
            cols=[
                column("grp", strategy=st.sampled_from(list(ascii_uppercase[:5]))),
                column("val", strategy=st.integers(min_value=1, max_value=5)),
            ],
            min_size=5,
            max_size=10,
            allow_null=False,
        ),
        decimals=st.integers(min_value=0, max_value=10),
    )
    def test_relative_sums_to_100(self, df: pl.DataFrame, decimals: int) -> None:
        res = absolute_and_relative_value_counts(
            df, cols="grp", sort="cols", decimals=decimals
        )
        total = sum(res["relative (%)"].to_list())
        # rounding means we allow one ULP of slack per group
        assert pytest.approx(total, abs=len(res) * 10**-decimals) == 100.0

    def test_relative_single_group_is_100(self) -> None:
        df = pl.DataFrame({"grp": ["a", "a", "a"]})
        res = absolute_and_relative_value_counts(df, cols="grp")
        assert res["relative (%)"][0] == pytest.approx(100.0)

    @given(decimals=st.integers(min_value=0, max_value=10))
    def test_relative_respects_decimals(self, decimals: int) -> None:
        df = pl.DataFrame({"grp": ["a", "b", "b"]})
        res = absolute_and_relative_value_counts(df, cols="grp", decimals=decimals)
        for val in res["relative (%)"].to_list():
            # rounding to `decimals` places should not change the value
            assert round(val, decimals) == pytest.approx(val, abs=1e-9)

    # ------------------------------------------------------------------
    # Sorting
    # ------------------------------------------------------------------

    @given(
        df=dataframes(
            cols=[
                column("grp", strategy=st.sampled_from(list(ascii_uppercase[:5]))),
                column("val", strategy=st.integers(min_value=1, max_value=5)),
            ],
            min_size=5,
            max_size=10,
            allow_null=False,
        ),
        decimals=st.integers(min_value=0, max_value=10),
        hardcoded=st.just(False),
    )
    @example(
        df=pl.DataFrame({"grp": ["a", "a", "b", "b", "b"], "val": [1, 2, 3, 4, 5]}),
        decimals=1,
        hardcoded=True,
    ).via("Hard-coded example")
    def test_sort_cols_ordering(
        self, df: pl.DataFrame, decimals: int, hardcoded: bool
    ) -> None:
        res = absolute_and_relative_value_counts(
            df.lazy(), cols="grp", sort="cols", decimals=decimals
        )

        if hardcoded:
            assert res.shape[0] == 2
            abs_counts = dict(
                zip(res["grp"].to_list(), res["absolute (int)"].to_list())
            )
            assert abs_counts == {"a": 2, "b": 3}

        grp = res["grp"].to_list()
        assert grp == df["grp"].unique().sort().to_list()

    def test_sort_counts_descending(self) -> None:
        df = pl.DataFrame({"grp": ["a", "b", "b", "b", "c", "c"]})
        res = absolute_and_relative_value_counts(df, cols="grp", sort="counts")
        counts = res["absolute (int)"].to_list()
        assert counts == sorted(counts, reverse=True)

    def test_sort_none_does_not_raise(self) -> None:
        df = pl.DataFrame({"grp": ["a", "b", "b"]})
        res = absolute_and_relative_value_counts(df, cols="grp", sort=None)
        assert res.shape[0] == 2

    # ------------------------------------------------------------------
    # Multiple grouping columns
    # ------------------------------------------------------------------

    @given(
        df=dataframes(
            cols=[
                column("grp", strategy=st.sampled_from(list(ascii_uppercase[:5]))),
                column("sub", strategy=st.sampled_from(list(ascii_uppercase[-5:]))),
                column("val", strategy=st.integers(min_value=1, max_value=5)),
            ],
            min_size=5,
            max_size=10,
            allow_null=False,
        ),
    )
    @example(
        df=pl.DataFrame(
            {
                "grp": ["a", "a", "b", "b", "b"],
                "sub": ["x", "y", "x", "x", "y"],
                "val": [1, 2, 3, 4, 5],
            }
        ),
    ).via("Hard-coded example")
    def test_multi_col_row_count_matches_unique_combos(self, df: pl.DataFrame) -> None:
        res = absolute_and_relative_value_counts(
            df.lazy(), cols=["grp", "sub"], sort="counts", decimals=1
        )
        unique_combos = df.select(["grp", "sub"]).unique().height
        assert res.shape[0] == unique_combos

    @given(
        df=dataframes(
            cols=[
                column("grp", strategy=st.sampled_from(list(ascii_uppercase[:5]))),
                column("sub", strategy=st.sampled_from(list(ascii_uppercase[-5:]))),
                column("val", strategy=st.integers(min_value=1, max_value=5)),
            ],
            min_size=5,
            max_size=10,
            allow_null=False,
        ),
    )
    @example(
        df=pl.DataFrame(
            {
                "grp": ["a", "a", "b", "b", "b"],
                "sub": ["x", "y", "x", "x", "y"],
            }
        ),
    ).via("Hard-coded example")
    def test_multi_col_absolute_sum(self, df: pl.DataFrame) -> None:
        res = absolute_and_relative_value_counts(
            df, cols=["grp", "sub"], sort="counts", decimals=1
        )
        assert sum(res["absolute (int)"].to_list()) == df.height

    def test_multi_col_grouping_columns_present(self) -> None:
        df = pl.DataFrame(
            {
                "grp": ["a", "a", "b", "b", "b"],
                "sub": ["x", "y", "x", "x", "y"],
            }
        )
        res = absolute_and_relative_value_counts(
            df, cols=["grp", "sub"], sort="counts", decimals=1
        )
        assert "grp" in res.columns and "sub" in res.columns

    # ------------------------------------------------------------------
    # Accepts both DataFrame and LazyFrame
    # ------------------------------------------------------------------

    def test_accepts_dataframe(self) -> None:
        df = pl.DataFrame({"grp": ["a", "a", "b"]})
        res = absolute_and_relative_value_counts(df, cols="grp")
        assert res.shape[0] == 2

    def test_accepts_lazyframe(self) -> None:
        df = pl.DataFrame({"grp": ["a", "a", "b"]})
        res = absolute_and_relative_value_counts(df.lazy(), cols="grp")
        assert res.shape[0] == 2

    def test_eager_and_lazy_produce_same_result(self) -> None:
        df = pl.DataFrame({"grp": ["a", "b", "b", "c"]})
        eager = absolute_and_relative_value_counts(df, cols="grp", sort="cols")
        lazy = absolute_and_relative_value_counts(df.lazy(), cols="grp", sort="cols")
        assert eager.equals(lazy)


# ------------------------
# Basic helpers
# ------------------------


# ---------------------------------------------------------------------------
# as_expr
# ---------------------------------------------------------------------------
class TestAsExpr:
    @given(col=col_name_st)
    def test_string_input_produces_col_expr(self, col: str) -> None:
        assert pl.col(col).meta.eq(as_expr(col))

    @given(col=col_name_st)
    def test_expr_passthrough(self, col: str) -> None:
        expr = pl.col(col)
        assert expr.meta.eq(as_expr(expr))

    @given(col=col_name_st)
    def test_idempotent_on_expr(self, col: str) -> None:
        """Calling as_expr twice on a string yields the same expression."""
        assert as_expr(pl.col(col)).meta.eq(as_expr(as_expr(col)))

    def test_arithmetic_expr_preserved(self) -> None:
        """Arithmetic expressions are returned as-is."""
        expr = pl.col("A") + pl.col("B")
        assert as_expr(expr).meta.eq(expr)

    def test_literal_expr_preserved(self) -> None:
        """Literal expressions are returned as-is."""
        expr = pl.lit(42)
        assert as_expr(expr).meta.eq(expr)


# ---------------------------------------------------------------------------
# columns
# ---------------------------------------------------------------------------
class TestColumns:
    @given(df=plain_df_st)
    def test_dataframe_matches_native_columns(self, df: pl.DataFrame) -> None:
        assert columns(df) == df.columns

    @given(df=plain_df_st)
    def test_lazyframe_matches_dataframe(self, df: pl.DataFrame) -> None:
        assert columns(df.lazy()) == columns(df)

    @given(df=plain_df_st)
    def test_column_names_are_unique(self, df: pl.DataFrame) -> None:
        result = columns(df)
        assert len(result) == len(set(result))

    def test_empty_dataframe_returns_empty_list(self) -> None:
        df = pl.DataFrame()
        assert columns(df) == []


# ---------------------------------------------------------------------------
# struct_fields
# ---------------------------------------------------------------------------
class TestStructFields:
    def _struct_df(self, field_names: list[str]) -> pl.DataFrame:
        """Build a DataFrame with a single Struct column whose sub-fields
        are the given names (all Int32).
        """
        struct_dtype = pl.Struct(dict.fromkeys(field_names, pl.Int32))
        data = [dict.fromkeys(field_names, 0)]
        return pl.DataFrame({"s": pl.Series(data, dtype=struct_dtype)})

    @given(field_names=st.lists(col_name_st, min_size=1, max_size=5, unique=True))
    def test_returns_correct_sub_field_names(self, field_names: list[str]) -> None:
        df = self._struct_df(field_names)
        result = struct_fields(df, "s")
        assert result == field_names

    @given(field_names=st.lists(col_name_st, min_size=1, max_size=5, unique=True))
    def test_lazyframe_matches_dataframe(self, field_names: list[str]) -> None:
        df = self._struct_df(field_names)
        assert struct_fields(df, "s") == struct_fields(df.lazy(), "s")


# ---------------------------------------------------------------------------
# col_length
# ---------------------------------------------------------------------------
class TestColLength:
    @given(df=plain_df_st)
    def test_matches_len_of_columns(self, df: pl.DataFrame) -> None:
        assert col_length(df) == len(df.columns)

    @given(df=plain_df_st)
    def test_dataframe_and_lazyframe_agree(self, df: pl.DataFrame) -> None:
        assert col_length(df) == col_length(df.lazy())

    def test_empty_dataframe_is_zero(self) -> None:
        assert col_length(pl.DataFrame()) == 0


# ---------------------------------------------------------------------------
# row_length
# ---------------------------------------------------------------------------
class TestRowLength:
    @given(df=plain_df_st)
    def test_dataframe_matches_native_len(self, df: pl.DataFrame) -> None:
        assert row_length(df) == len(df)

    @given(df=plain_df_st)
    def test_dataframe_and_lazyframe_agree(self, df: pl.DataFrame) -> None:
        assert row_length(df) == row_length(df.lazy())

    def test_empty_dataframe_is_zero(self) -> None:
        df = pl.DataFrame({"a": pl.Series([], dtype=pl.Int32)})
        assert row_length(df) == 0


# ---------------------------------------------------------------------------
# is_empty
# ---------------------------------------------------------------------------
class TestIsEmpty:
    @given(df=plain_df_st)
    def test_consistent_with_row_length(self, df: pl.DataFrame) -> None:
        assert is_empty(df) == (row_length(df) == 0)

    @given(df=plain_df_st)
    def test_consistent_with_is_empty_native(self, df: pl.DataFrame) -> None:
        assert is_empty(df) == df.is_empty()

    @given(df=plain_df_st)
    def test_dataframe_and_lazyframe_agree(self, df: pl.DataFrame) -> None:
        assert is_empty(df) == is_empty(df.lazy())

    @given(df=dataframes(min_cols=1, max_cols=4, min_size=1, max_size=20))
    def test_non_empty_dataframe_is_false_and_cleared_is_true(
        self, df: pl.DataFrame
    ) -> None:
        assert not is_empty(df)
        assert is_empty(df.clear())


# ------------------------
# Expression pipes
# ------------------------


# ===========================================================================
# unique_sucessive
# ===========================================================================
class TestUniqueSucessive:
    # --- hardcoded correctness ---
    def test_known_output(self) -> None:
        df = pl.DataFrame(
            {"lst": [[1, 1, 2, 2, 3, 2, 2, 1], [1, 2, 2, 1, 2], [3, 3, 3, 4, 2, 1]]}
        )
        out = df.with_columns(unique_sucessive(pl.col("lst")).alias("filtered"))
        expected = pl.DataFrame(
            {
                "lst": [[1, 1, 2, 2, 3, 2, 2, 1], [1, 2, 2, 1, 2], [3, 3, 3, 4, 2, 1]],
                "filtered": [[1, 2, 3, 2, 1], [1, 2, 1, 2], [3, 4, 2, 1]],
            }
        )
        pl_testing.assert_frame_equal(out, expected)

    def test_all_same_values_collapses_to_one(self) -> None:
        df = pl.DataFrame({"lst": [[5, 5, 5, 5]]})
        out = df.with_columns(unique_sucessive(pl.col("lst")).alias("out"))
        assert out["out"][0].to_list() == [5]

    def test_accepts_string_column_name_and_empty_list_stays_empty(self) -> None:
        df = pl.DataFrame({"lst": [[]]}, schema={"lst": pl.List(pl.Int32)})
        out = df.with_columns(unique_sucessive("lst").alias("out"))
        assert out["out"][0].to_list() == []

    def test_no_duplicates_unchanged(self) -> None:
        df = pl.DataFrame({"lst": [[1, 2, 3, 4]]})
        out = df.with_columns(unique_sucessive(pl.col("lst")).alias("out"))
        assert out["out"][0].to_list() == [1, 2, 3, 4]

    # --- properties ---
    @given(
        df=st.one_of(
            dataframes(
                min_size=5,
                max_size=10,
                allow_null=False,
                allowed_dtypes=[
                    pl.List(pl.Int64),
                    pl.List(pl.Utf8),
                    pl.List(pl.Float64),
                ],
            ),
            dataframes(
                cols=column(
                    "test",
                    strategy=lists(
                        inner_dtype=pl.Int8(),
                        select_from=[0, 1],
                        min_size=2,
                        max_size=20,
                    ),
                ),
                min_size=5,
                max_size=10,
                allow_null=False,
            ),
        )
    )
    def test_idempotent_and_length_non_increasing(self, df: pl.DataFrame) -> None:
        """One pass removes consecutive duplicates; a second pass changes nothing."""
        one_pass = df.with_columns(pl.all().pipe(unique_sucessive).list.len())
        double_pass = df.with_columns(
            pl.all().pipe(unique_sucessive).pipe(unique_sucessive).list.len()
        )
        original_lens = df.with_columns(pl.all().list.len())

        # Idempotent: second application is a no-op
        pl_testing.assert_frame_equal(one_pass, double_pass)

        # Non-increasing: filtered lists are never longer than the originals
        diff = (
            original_lens.with_row_index()
            .join(
                one_pass.with_row_index(),
                on="index",
                suffix="_filtered",
            )
            .drop("index")
            .select(pl.sum_horizontal(pl.all()).alias("diff"))["diff"]
        )
        assert diff.ge(0).all()

    @given(
        df=dataframes(
            cols=column(
                "lst",
                strategy=lists(
                    inner_dtype=pl.Int8(),
                    select_from=[0, 1],
                    min_size=0,
                    max_size=15,
                ),
            ),
            min_size=1,
            max_size=10,
            allow_null=False,
        )
    )
    def test_result_values_subset_of_original(self, df: pl.DataFrame) -> None:
        """Every value in the filtered list was present in the original."""
        out = df.with_columns(unique_sucessive(pl.col("lst")).alias("filtered"))
        for orig, filtered in zip(out["lst"].to_list(), out["filtered"].to_list()):
            assert set(filtered).issubset(set(orig))

    @given(
        df=dataframes(
            cols=column(
                "lst",
                strategy=lists(
                    inner_dtype=pl.Int8(),
                    select_from=[0, 1],
                    min_size=2,
                    max_size=15,
                ),
            ),
            min_size=1,
            max_size=10,
            allow_null=False,
        )
    )
    def test_no_consecutive_duplicates_after_filter(self, df: pl.DataFrame) -> None:
        """The output list contains no two consecutive equal elements."""
        out = df.with_columns(unique_sucessive(pl.col("lst")).alias("filtered"))
        for row in out["filtered"].to_list():
            if row is not None:
                for a, b in zip(row, row[1:]):
                    assert a != b, f"Consecutive duplicate found: {row}"


# ===========================================================================
# time_difference
# ===========================================================================
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


class TestTimeDifference:
    def _df(
        self,
        t: datetime.datetime,
        ref: datetime.datetime,
    ) -> pl.DataFrame:
        return pl.DataFrame({"t": [t], "ref": [ref]})

    # --- hardcoded correctness ---

    def test_seconds_exact(self) -> None:
        t = _dt("2020-01-01T02:00:00")
        ref = _dt("2020-01-01T00:00:00")
        df = self._df(t, ref)
        result = cast("int", df.select(time_difference("t", "ref", unit="s")).item())
        assert result == 7200

    def test_minutes_exact(self) -> None:
        t = _dt("2020-01-01T02:00:00")
        ref = _dt("2020-01-01T00:00:00")
        df = self._df(t, ref)
        result = cast("int", df.select(time_difference("t", "ref", unit="m")).item())
        assert result == 120  # floor(7200 / 60)

    def test_hours_floor(self) -> None:
        # 90 minutes => floor = 1 hour
        t = _dt("2020-01-01T01:30:00")
        ref = _dt("2020-01-01T00:00:00")
        df = self._df(t, ref)
        result = cast("int", df.select(time_difference("t", "ref", unit="h")).item())
        assert result == 1

    def test_hours_total(self) -> None:
        t = _dt("2020-01-01T01:30:00")
        ref = _dt("2020-01-01T00:00:00")
        df = self._df(t, ref)
        result = cast(
            "int", df.select(time_difference("t", "ref", unit="h", total=False)).item()
        )
        assert result == pytest.approx(1.5)

    def test_negative_when_ref_after_time(self) -> None:
        t = _dt("2020-01-01T00:00:00")
        ref = _dt("2020-01-01T02:00:00")
        df = self._df(t, ref)
        result = cast("int", df.select(time_difference("t", "ref", unit="h")).item())
        assert result == -2

    def test_zero_when_equal(self) -> None:
        t = _dt("2020-06-15T12:00:00")
        df = self._df(t, t)
        result = cast("int", df.select(time_difference("t", "ref", unit="s")).item())
        assert result == 0

    def test_invalid_unit_raises(self) -> None:
        df = self._df(_dt("2020-01-02T00:00:00"), _dt("2020-01-01T00:00:00"))
        with pytest.raises(ValueError, match="Invalid unit"):
            df.select(time_difference("t", "ref", unit="x"))  # type: ignore[arg-type]

    def test_accepts_expr_input(self) -> None:
        df = self._df(
            _dt("2020-01-01T01:00:00"),
            _dt("2020-01-01T00:00:00"),
        )
        result_str = cast(
            "int", df.select(time_difference("t", "ref", unit="h")).item()
        )
        result_expr = cast(
            "int",
            df.select(time_difference(pl.col("t"), pl.col("ref"), unit="h")).item(),
        )
        assert result_str == result_expr

    # --- properties ---
    @given(delta_seconds=st.integers(min_value=-86400 * 365, max_value=86400 * 365))
    def test_seconds_equals_raw_delta(self, delta_seconds: int) -> None:
        """time_difference in 's' must equal the raw second count."""
        base = _dt("2015-06-01T00:00:00")
        t = base + datetime.timedelta(seconds=delta_seconds)
        df = pl.DataFrame({"t": [t], "ref": [base]})
        result = cast("int", df.select(time_difference("t", "ref", unit="s")).item())
        assert result == delta_seconds

    @given(
        delta_seconds=st.integers(min_value=0, max_value=86400 * 30),
        unit=st.sampled_from(list(_UNIT_SECONDS.keys())),
    )
    def test_floor_unit_consistent_with_seconds(
        self, delta_seconds: int, unit: str
    ) -> None:
        """Floor result must equal seconds // conversion factor."""
        base = _dt("2015-01-01T00:00:00")
        t = base + datetime.timedelta(seconds=delta_seconds)
        df = pl.DataFrame({"t": [t], "ref": [base]})
        result = cast("int", df.select(time_difference("t", "ref", unit=unit)).item())
        expected = delta_seconds // _UNIT_SECONDS[unit]
        assert result == expected

    @given(
        delta_seconds=st.integers(min_value=0, max_value=86400 * 30),
        unit=st.sampled_from([u for u in _UNIT_SECONDS if u != "s"]),
    )
    def test_total_unit_consistent_with_seconds(
        self, delta_seconds: int, unit: str
    ) -> None:
        """Non-total (float) result must equal seconds / conversion factor."""
        base = _dt("2015-01-01T00:00:00")
        t = base + datetime.timedelta(
            seconds=delta_seconds, milliseconds=100
        )  # milliseconds will be dropped as they are smaller than seconds
        df = pl.DataFrame({"t": [t], "ref": [base]})
        result = cast(
            "int", df.select(time_difference("t", "ref", unit=unit, total=False)).item()
        )
        expected = delta_seconds / _UNIT_SECONDS[unit]
        assert result == pytest.approx(expected, rel=1e-9)

    @given(delta_seconds=st.integers(min_value=0, max_value=86400 * 365))
    def test_floor_le_total(self, delta_seconds: int) -> None:
        """Floor value is always <= the true (non-total) value for non-negative deltas."""
        base = _dt("2015-01-01T00:00:00")
        t = base + datetime.timedelta(seconds=delta_seconds)
        df = pl.DataFrame({"t": [t], "ref": [base]})
        for unit in [u for u in _UNIT_SECONDS if u != "s"]:
            floor = cast(
                "int", df.select(time_difference("t", "ref", unit=unit)).item()
            )
            total = cast(
                "int",
                df.select(time_difference("t", "ref", unit=unit, total=False)).item(),
            )
            assert floor <= total


# ===========================================================================
# attach_asof_value
# ===========================================================================
class TestAttachAsofValue:
    """The helper exists so that the sort join_asof silently requires cannot be
    forgotten: polars cannot verify sortedness once `by` groups are given and
    returns wrong values instead of raising.
    """

    _main = pl.DataFrame(
        {
            "case_id": [1, 1, 2],
            "recordtime": [
                _dt("2020-01-01T10:00:00"),
                _dt("2020-01-01T14:00:00"),
                _dt("2020-01-01T12:00:00"),
            ],
            "value": [10.0, 20.0, 30.0],
        }
    )
    _ref = pl.DataFrame(
        {
            "case_id": [1, 1, 2],
            "recordtime": [
                _dt("2020-01-01T09:00:00"),
                _dt("2020-01-01T13:00:00"),
                _dt("2020-01-01T11:00:00"),
            ],
            "value": [0.9, 1.3, 1.1],
        }
    )

    # --- correctness ---
    def test_attaches_matching_value(self) -> None:
        out = attach_asof_value(self._main, self._ref, name="ref_value")
        assert out["ref_value"].to_list() == [0.9, 1.3, 1.1]

    def test_unsorted_input_matches_the_same(self) -> None:
        """Neither side needs to arrive sorted — that is the point."""
        shuffled_main = self._main.sort("value", descending=True)
        shuffled_ref = self._ref.sort("value", descending=True)
        out = attach_asof_value(shuffled_main, shuffled_ref, name="ref_value")
        pl_testing.assert_frame_equal(
            out, attach_asof_value(self._main, self._ref, name="ref_value")
        )

    def test_output_is_sorted_by_key_and_time(self) -> None:
        out = attach_asof_value(self._main, self._ref, name="ref_value")
        pl_testing.assert_frame_equal(out, out.sort("case_id", "recordtime"))

    def test_tolerance_limits_the_match(self) -> None:
        out = attach_asof_value(
            self._main, self._ref, name="ref_value", tolerance="30m"
        )
        assert out["ref_value"].to_list() == [None, None, None]

    def test_unbounded_tolerance(self) -> None:
        """With every ref 100 days back, the latest preceding one still matches."""
        far_ref = self._ref.with_columns(pl.col("recordtime").dt.offset_by("-100d"))
        out = attach_asof_value(self._main, far_ref, name="ref_value", tolerance=None)
        assert out["ref_value"].to_list() == [1.3, 1.3, 1.1]

        bounded = attach_asof_value(self._main, far_ref, name="ref_value")
        assert bounded["ref_value"].to_list() == [None, None, None]

    def test_forward_strategy(self) -> None:
        out = attach_asof_value(
            self._main, self._ref, name="ref_value", strategy="forward", tolerance="4h"
        )
        assert out["ref_value"].to_list() == [1.3, None, None]

    def test_no_match_is_null(self) -> None:
        out = attach_asof_value(self._main, self._ref.clear(), name="ref_value")
        assert out["ref_value"].to_list() == [None, None, None]

    # --- matched timestamp ---
    def test_matched_time_is_omitted_by_default(self) -> None:
        out = attach_asof_value(self._main, self._ref, name="ref_value")
        assert columns(out) == columns(self._main) + ["ref_value"]

    def test_matched_time_can_be_kept(self) -> None:
        out = attach_asof_value(
            self._main, self._ref, name="ref_value", with_matched_time=True
        )
        assert columns(out) == columns(self._main) + [
            "ref_value",
            "ref_value_recordtime",
        ]
        assert out["ref_value_recordtime"].to_list() == [
            _dt("2020-01-01T09:00:00"),
            _dt("2020-01-01T13:00:00"),
            _dt("2020-01-01T11:00:00"),
        ]

    # --- lazy / eager parity ---
    def test_lazy_and_eager_same_result(self) -> None:
        eager = attach_asof_value(self._main, self._ref, name="ref_value")
        lazy = attach_asof_value(
            self._main.lazy(), self._ref.lazy(), name="ref_value"
        ).collect()  # type: ignore[union-attr]
        pl_testing.assert_frame_equal(eager, lazy)

    # --- guards ---
    def test_mixed_types_raises(self) -> None:
        with pytest.raises(TypeError, match="Types of main and ref are not the same"):
            attach_asof_value(self._main, self._ref.lazy(), name="ref_value")

    def test_existing_column_raises(self) -> None:
        with pytest.raises(ValueError, match="already exist in main"):
            attach_asof_value(self._main, self._ref, name="value")

    def test_matched_time_conflict_only_checked_when_kept(self) -> None:
        main = self._main.with_columns(pl.lit(1).alias("ref_value_recordtime"))
        attach_asof_value(main, self._ref, name="ref_value")
        with pytest.raises(ValueError, match="already exist in main"):
            attach_asof_value(main, self._ref, name="ref_value", with_matched_time=True)


# ===========================================================================
# remove_asof
# ===========================================================================
class TestRemoveAsof:
    _main = pl.DataFrame(
        {
            "case_id": [1, 1, 2],
            "recordtime": [
                _dt("2020-01-01T10:00:00"),
                _dt("2020-01-02T10:00:00"),
                _dt("2020-01-01T12:00:00"),
            ],
            "value": [10, 20, 30],
        }
    )
    _ref = pl.DataFrame(
        {
            "case_id": [1],
            "recordtime": [_dt("2020-01-01T10:30:00")],
        }
    )

    # --- correctness ---
    def test_removes_nearby_row(self) -> None:
        out = remove_asof(self._main, self._ref, tolerance="1h")
        assert out.height == 2
        remaining = list(zip(out["case_id"].to_list(), out["recordtime"].to_list()))
        assert (1, _dt("2020-01-02T10:00:00")) in remaining
        assert (2, _dt("2020-01-01T12:00:00")) in remaining

    def test_tight_tolerance_removes_nothing(self) -> None:
        """With a 1-second tolerance the ref (30 min away) matches nothing."""
        out = remove_asof(self._main, self._ref, tolerance="1s")
        assert out.height == self._main.height

    def test_ref_far_away_removes_nothing(self) -> None:
        """Reference 9 days away removes nothing even with the main fixture."""
        far_ref = pl.DataFrame(
            {
                "case_id": [1],
                "recordtime": [_dt("2020-01-10T10:30:00")],
            }
        )
        out = remove_asof(self._main, far_ref, tolerance="3d")
        assert out.height == self._main.height
        pl_testing.assert_frame_equal(out, self._main)

    def test_wide_tolerance_removes_all_matching_case(self) -> None:
        """48h tolerance covers both rows for case_id=1, so both are removed."""
        main_case1 = self._main.filter(pl.col("case_id") == 1)
        out = remove_asof(main_case1, self._ref, tolerance="48h")
        # both rows are within 48h of the ref -> everything removed
        assert out.height == 0

    def test_empty_ref_removes_nothing(self) -> None:
        empty_ref = self._ref.clear()
        out = remove_asof(self._main, empty_ref, tolerance="99d")
        assert out.height == self._main.height

    def test_result_is_subset_of_main(self) -> None:
        out = remove_asof(self._main, self._ref, tolerance="1h")
        joined = out.join(
            self._main, on=["case_id", "recordtime", "value"], how="inner"
        )
        assert joined.height == out.height

    # --- lazy / eager parity ---
    def test_lazy_and_eager_same_result(self) -> None:
        eager = remove_asof(self._main, self._ref, tolerance="1h")
        lazy = remove_asof(
            self._main.lazy(), self._ref.lazy(), tolerance="1h"
        ).collect()  # type: ignore[union-attr]
        pl_testing.assert_frame_equal(
            eager.sort(["case_id", "recordtime"]),
            lazy.sort(["case_id", "recordtime"]),
        )

    # --- type safety ---
    def test_mixed_types_raises(self) -> None:
        with pytest.raises(TypeError, match="Types of main and ref are not the same"):
            remove_asof(self._main, self._ref.lazy())

    # --- strategies ---
    @given(
        strategy=st.sampled_from(["backward", "forward", "nearest"]),
    )
    def test_result_is_subset_for_all_strategies(self, strategy: str) -> None:
        out = remove_asof(
            self._main,
            self._ref,
            strategy=strategy,
            tolerance="1h",  # type: ignore[arg-type]
        )
        assert out.height <= self._main.height
        assert set(out["case_id"].to_list()).issubset(
            set(self._main["case_id"].to_list())
        )

    # --- property-based ---
    @given(
        df_main=dataframes(
            cols=[
                column("case_id", strategy=st.integers(min_value=1, max_value=5)),
                column(
                    "recordtime",
                    strategy=st.datetimes(
                        min_value=datetime.datetime(2020, 1, 1),
                        max_value=datetime.datetime(2020, 12, 31),
                    ),
                ),
                column("value", strategy=st.integers(min_value=1, max_value=100)),
            ],
            min_size=5,
            max_size=10,
        ),
        df_ref=dataframes(
            cols=[
                column("case_id", strategy=st.integers(min_value=1, max_value=5)),
                column(
                    "recordtime",
                    strategy=st.datetimes(
                        min_value=datetime.datetime(2020, 1, 1),
                        max_value=datetime.datetime(2020, 12, 31),
                    ),
                ),
            ],
            min_size=3,
            max_size=5,
        ),
        tolerance=st.sampled_from(["1h", "6h", "12h", "1d"]),
    )
    def test_idempotent_and_reduces(
        self, df_main: pl.DataFrame, df_ref: pl.DataFrame, tolerance: str
    ) -> None:
        """Applying remove_asof twice yields the same result as once."""
        out = remove_asof(df_main, df_ref, tolerance=tolerance)
        assert out.height <= df_main.height
        out2 = remove_asof(out, df_ref, tolerance=tolerance)
        pl_testing.assert_frame_equal(out, out2)


# ===========================================================================
# apply_cleaning
# ===========================================================================
class TestApplyCleaning:
    def test_low_bound_filters_below(self) -> None:
        df = pl.DataFrame({"val": [1.0, 5.0, 10.0]})
        out = apply_cleaning(df, {"val": {"low": 5.0}})
        assert out["val"].to_list() == [5.0, 10.0]

    def test_high_bound_filters_above(self) -> None:
        df = pl.DataFrame({"val": [1.0, 5.0, 10.0]})
        out = apply_cleaning(df, {"val": {"high": 5.0}})
        assert out["val"].to_list() == [1.0, 5.0]

    def test_both_bounds(self) -> None:
        df = pl.DataFrame({"val": [1.0, 5.0, 10.0, -1.0, 100.0]})
        out = apply_cleaning(df, {"val": {"low": 2.0, "high": 9.0}})
        assert out["val"].to_list() == [5.0]

    def test_empty_cleaning_returns_original(self) -> None:
        df = pl.DataFrame({"val": [1.0, 2.0, 3.0]})
        out = apply_cleaning(df, {})
        pl_testing.assert_frame_equal(out, df)

    def test_empty_dataframe_returns_unchanged(self) -> None:
        df = pl.DataFrame({"val": pl.Series([], dtype=pl.Float64)})
        out = apply_cleaning(df, {"val": {"low": 0.0}})
        assert out.is_empty()

    def test_multi_column_cleaning(self) -> None:
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 5.0, 1.0]})
        out = apply_cleaning(df, {"a": {"low": 2.0}, "b": {"high": 6.0}})
        # a >= 2 AND b <= 6 => rows (2.0, 5.0) and (3.0, 1.0)
        assert out.shape == (2, 2)
        assert set(out["a"].to_list()) == {2.0, 3.0}

    def test_lazy_and_eager_same_result(self) -> None:
        df = pl.DataFrame({"val": [1.0, 2.0, 3.0]})
        eager = apply_cleaning(df, {"val": {"low": 2.0}})
        lazy = apply_cleaning(df.lazy(), {"val": {"low": 2.0}})
        pl_testing.assert_frame_equal(eager, lazy.collect())

    # --- properties ---
    @given(
        rows=st.lists(
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
            min_size=1,
            max_size=30,
        ),
        low=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
        high=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
    )
    def test_all_results_respect_bounds(
        self, rows: list[float], low: float, high: float
    ) -> None:
        if low > high:
            low, high = high, low
        df = pl.DataFrame({"val": rows})
        out = apply_cleaning(df, {"val": {"low": low, "high": high}})
        vals = out["val"].to_list()
        assert all(v >= low for v in vals)
        assert all(v <= high for v in vals)

    @given(
        rows=st.lists(
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
            min_size=1,
            max_size=30,
        ),
        low=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
    )
    def test_result_is_never_larger_than_input(
        self, rows: list[float], low: float
    ) -> None:
        df = pl.DataFrame({"val": rows})
        out = apply_cleaning(df, {"val": {"low": low}})
        assert out.height <= df.height

    @given(
        rows=st.lists(st.integers(min_value=0, max_value=100), min_size=0, max_size=20)
    )
    def test_none_cleaning_returns_all_rows(self, rows: list[int]) -> None:
        df = pl.DataFrame({"val": rows})
        out = apply_cleaning(df, {})
        pl_testing.assert_frame_equal(out, df)


# ===========================================================================
# build_attributes
# ===========================================================================
class TestBuildAttributes:
    def test_creates_struct_column(self) -> None:
        df = pl.DataFrame({"id": [1], "attributes_x": [1.0], "attributes_y": ["a"]})
        out = build_attributes(df)
        assert "attributes" in out.columns

    def test_removes_prefix_columns(self) -> None:
        df = pl.DataFrame({"id": [1], "attributes_x": [1.0], "attributes_y": ["a"]})
        out = build_attributes(df)
        assert "attributes_x" not in out.columns
        assert "attributes_y" not in out.columns

    def test_struct_fields_match_stripped_names(self) -> None:
        df = pl.DataFrame({"id": [1], "attributes_x": [1.0], "attributes_y": ["a"]})
        out = build_attributes(df)
        assert set(out["attributes"].struct.fields) == {"x", "y"}

    def test_no_attributes_columns_unchanged(self) -> None:
        df = pl.DataFrame({"id": [1], "val": [42]})
        out = build_attributes(df)
        pl_testing.assert_frame_equal(out, df)

    def test_lazy_and_eager_same_result(self) -> None:
        df = pl.DataFrame({"id": [1], "attributes_a": [10]})
        eager = build_attributes(df)
        lazy = build_attributes(df.lazy())
        pl_testing.assert_frame_equal(eager, lazy.collect())

    def test_struct_values_preserved(self) -> None:
        df = pl.DataFrame({"attributes_score": [0.9], "attributes_label": ["ok"]})
        out = build_attributes(df)
        row = out["attributes"][0]
        assert row["score"] == pytest.approx(0.9)
        assert row["label"] == "ok"

    @given(
        field_names=st.lists(
            st.text(alphabet=ascii_uppercase, min_size=1, max_size=6),
            min_size=1,
            max_size=4,
            unique=True,
        )
    )
    def test_all_attr_cols_collapsed_into_struct(self, field_names: list[str]) -> None:
        df = pl.DataFrame({"id": [0], **{f"attributes_{n}": [0] for n in field_names}})
        out = build_attributes(df)
        assert set(out["attributes"].struct.fields) == set(field_names)
        remaining = [c for c in out.columns if c.startswith("attributes_")]
        assert remaining == []


# ===========================================================================
# unnest
# ===========================================================================
class TestUnnest:
    @staticmethod
    def _struct_df(fields: dict[str, pl.DataType]) -> pl.DataFrame:
        dtype = pl.Struct(fields)
        row = dict.fromkeys(fields, 0)
        return pl.DataFrame({"s": pl.Series([row], dtype=dtype)})

    # --- schema correctness ---
    def test_struct_column_removed(self) -> None:
        df = self._struct_df({"a": pl.Int32, "b": pl.Int32})
        out = unnest(df, "s")
        assert "s" not in out.columns

    def test_plain_fields_present(self) -> None:
        df = self._struct_df({"a": pl.Int32, "b": pl.Float32})
        out = unnest(df, "s")
        assert "a" in out.columns and "b" in out.columns

    def test_prefix_applied(self) -> None:
        df = self._struct_df({"x": pl.Int32, "y": pl.Int32})
        out = unnest(df, "s", prefix="col_")
        assert "col_x" in out.columns and "col_y" in out.columns

    def test_suffix_applied(self) -> None:
        df = self._struct_df({"x": pl.Int32, "y": pl.Int32})
        out = unnest(df, "s", suffix="_val")
        assert "x_val" in out.columns and "y_val" in out.columns

    def test_callable_renamer(self) -> None:
        df = self._struct_df({"x": pl.Int32, "y": pl.Int32})
        out = unnest(df, "s", renamer=str.upper)
        assert "X" in out.columns and "Y" in out.columns

    def test_sequence_renamer(self) -> None:
        df = self._struct_df({"a": pl.Int32, "b": pl.Int32})
        out = unnest(df, "s", renamer=["first", "second"])
        assert "first" in out.columns and "second" in out.columns

    def test_column_count_preserved(self) -> None:
        """Unnest replaces 1 struct col with N field cols; total is unchanged."""
        extra_cols = {"id": [1]}
        fields = {"a": pl.Int32, "b": pl.Int32, "c": pl.Float32}
        dtype = pl.Struct(fields)
        df = pl.DataFrame(
            {**extra_cols, "s": pl.Series([{"a": 0, "b": 0, "c": 0.0}], dtype=dtype)}
        )
        out = unnest(df, "s")
        # 1 (id) + 3 (fields) = 4; original was 1 (id) + 1 (s) = 2
        assert col_length(out) == col_length(df) - 1 + len(fields)

    def test_lazy_and_eager_same_result(self) -> None:
        df = self._struct_df({"a": pl.Int32})
        eager = unnest(df, "s")
        lazy = unnest(df.lazy(), "s")
        pl_testing.assert_frame_equal(eager, right=lazy.collect())

    def test_values_preserved(self) -> None:
        dtype = pl.Struct({"a": pl.Int32, "b": pl.Float32})
        df = pl.DataFrame({"s": pl.Series([{"a": 7, "b": 3.14}], dtype=dtype)})
        out = unnest(df, "s")
        assert out["a"][0] == 7
        assert out["b"][0] == pytest.approx(3.14)

    # --- properties ---
    @given(
        field_names=st.lists(col_name_st, min_size=1, max_size=5, unique=True),
        prefix=st.text(alphabet=ascii_uppercase, min_size=0, max_size=3),
        suffix=st.text(alphabet=ascii_uppercase, min_size=0, max_size=3),
    )
    def test_prefix_suffix_naming(
        self, field_names: list[str], prefix: str, suffix: str
    ) -> None:
        dtype = pl.Struct(dict.fromkeys(field_names, pl.Int32))
        df = pl.DataFrame(
            {"s": pl.Series([dict.fromkeys(field_names, 0)], dtype=dtype)}
        )
        out = unnest(df, "s", prefix=prefix, suffix=suffix)
        expected = {f"{prefix}{n}{suffix}" for n in field_names}
        assert expected.issubset(set(out.columns))
