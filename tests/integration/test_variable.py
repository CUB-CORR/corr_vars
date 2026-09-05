from copy import deepcopy

import pandas as pd
import polars as pl
import polars.testing as pl_testing
import pytest
from hypothesis import given, settings
from hypothesis_cohort import obs_dataframe
from hypothesis_var import var_dataframe
from polars.testing.parametric import dataframes

from conftest import EmptyCohort
from corr_vars.core.cohort import Cohort
from corr_vars.core.variable import Variable
from corr_vars.definitions import ObsLevel, TimeAnchorColumn
from corr_vars.definitions.constants import COL_ORDER, DYN_COLUMNS
from corr_vars.sources.var_loader import (
    add_relative_times,
    unify_and_order_columns,
)

from corr_vars.definitions.typing import VariableContext
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from corr_vars.utils.time import TimeWindow


####################
#      Helpers     #
####################
def static_variable(
    var_name: str = "test_var",
    tmin: TimeAnchorColumn | None = None,
    tmax: TimeAnchorColumn | None = None,
    **kwargs,
) -> Variable:
    var_tmin = tmin or "tmin"
    var_tmax = tmax or "tmax"
    return Variable(
        var_name=var_name, dynamic=False, tmin=var_tmin, tmax=var_tmax, **kwargs
    )


def dynamic_variable(
    var_name: str = "test_var",
    tmin: TimeAnchorColumn | None = None,
    tmax: TimeAnchorColumn | None = None,
    **kwargs,
) -> Variable:
    var_tmin = tmin or "tmin"
    var_tmax = tmax or "tmax"
    return Variable(
        var_name=var_name, dynamic=True, tmin=var_tmin, tmax=var_tmax, **kwargs
    )


####################
#   Requirements   #
####################
def test_get_required_vars_resolves_relative_windows() -> None:
    """`requires` deltas are resolved against the parent variable's window."""
    captured: dict[str, TimeWindow] = {}

    class LoadedVariable:
        var_name = "dep"
        dynamic = True

        def extract(self, cohort) -> pl.DataFrame:
            return pl.DataFrame()

    class SpyCohort:
        """Records the time window each requirement is loaded with."""

        def load_variable(self, variable, include_sources=None, overrides=None):
            _, time_window = variable
            captured[str(len(captured))] = time_window
            return LoadedVariable()

    var = dynamic_variable(
        tmin="icu_admission",
        tmax="icu_discharge",
        requires={
            "inherited": {"template": "dep", "tmin": "inherit", "tmax": "inherit"},
            "tuple_delta": {
                "template": "dep",
                "tmin": ("inherit", "-1h"),
                "tmax": "inherit",
            },
            "bare_delta": {"template": "dep", "tmin": "-1h", "tmax": "24h"},
        },
    )
    var._get_required_vars(SpyCohort())  # type: ignore[arg-type]

    inherited, tuple_delta, bare_delta = captured.values()
    assert repr(inherited.tmin) == "icu_admission"
    assert repr(tuple_delta.tmin) == "icu_admission - 1h"
    # The bare delta is the shorthand of the tuple form.
    assert repr(bare_delta.tmin) == "icu_admission - 1h"
    assert repr(bare_delta.tmax) == "icu_discharge + 24h"


####################
# Variable extract #
####################
def test_extract(dummy_cohort: Cohort) -> None:
    """Test variable extract operation"""
    # By default this is not implemented.
    with pytest.raises(NotImplementedError):
        var = dynamic_variable()
        var.extract(dummy_cohort)


# def test_get_required_vars_constant_branch() -> None:
#     cohort = EmptyCohort()
#     # make a constant variable present in cohort
#     cohort.constant_vars = ["const1"]
#     cohort._obs = pl.DataFrame({cohort.primary_key: [1, 2], "const1": [10, 20]})

#     v = static_variable(requires=["const1"])
#     v._get_required_vars(cohort)

#     assert "const1" in v.required_vars
#     expected = cohort._obs.select(cohort.primary_key, "const1")
#     assert v.required_vars["const1"].data is not None
#     pl_testing.assert_frame_equal(v.required_vars["const1"].data, expected)


# # TODO: Unify all cohort and variable helper classes and override on the lowest level
# # to avoid problems like missing .constant_vars on EmptyCohort
# def test_get_required_vars_calls_load_variable_spy(spy: Spy) -> None:
#     cohort = EmptyCohort()
#     cohort.constant_vars = []

#     def fake_load_variable(var_name, *args, **kwargs):
#         class SpyVar:
#             def __init__(self, name):
#                 self.var_name = name
#                 self.dynamic = True

#             def extract(self, *args, **kwargs):
#                 # return a simple polars frame to be stored
#                 return pl.DataFrame({"val": [1, 2]})

#         return SpyVar(var_name)

#     captured = spy("corr_vars.core.variable.load_variable", wrap=fake_load_variable)

#     requires = {
#         "req1": {
#             "template": "template1",
#             "tmin": "tmin_override",
#             "tmax": "tmax_override",
#         }
#     }
#     v = dynamic_variable(requires=requires)

#     v._get_required_vars(cohort)

#     # ensure loader was called with the template name
#     assert "last_kwargs" in captured
#     assert captured["last_kwargs"]["var_name"] == "template1"
#     # ensure time_window passed through use the overrides
#     tb = captured["last_kwargs"]["time_window"]
#     assert isinstance(tb, TimeWindow)
#     assert tb.tmin.column == "tmin_override"
#     assert tb.tmax.column == "tmax_override"

#     # required var should be present and data stored
#     assert "req1" in v.required_vars
#     assert v.required_vars["req1"].data is not None
#     pl_testing.assert_frame_equal(
#         v.required_vars["req1"].data, pl.DataFrame({"val": [1, 2]})
#     )


@settings(max_examples=1)
@given(
    obs_df=obs_dataframe(size=10),
    var_df=var_dataframe(vtype="dynamic_ts_snapshot", size=10),
)
def test_call_var_function_sets_data(
    obs_df: pl.DataFrame, var_df: pl.DataFrame
) -> None:
    class CalledArgs(TypedDict, total=False):
        var: VariableContext
        cohort: Cohort

    called: CalledArgs = {}

    def fn_polars(var: VariableContext, cohort: Cohort):
        called.update(var=deepcopy(var), cohort=deepcopy(cohort))
        return var_df

    def fn_pandas(var: VariableContext, cohort: Cohort):
        called.update(var=deepcopy(var), cohort=deepcopy(cohort))
        return var_df.to_pandas()

    cohort = EmptyCohort(obs_level="hospital_stay")
    cohort.obs = obs_df

    v = dynamic_variable(var_name="v_no_py", py=None)
    result = v._call_var_function(cohort)
    assert result is False
    assert v.data is None

    v = dynamic_variable(var_name="v_pl_py", py=fn_polars, py_ready_polars=True)
    result = v._call_var_function(cohort)
    assert v.data is not None
    pl_testing.assert_frame_equal(v.data, var_df)
    assert result is True
    assert "var" in called
    assert called["var"].data is None
    assert "cohort" in called
    assert isinstance(called["cohort"]._obs, pl.DataFrame)
    pl_testing.assert_frame_equal(called["cohort"]._obs, cohort._obs)

    v = dynamic_variable(var_name="v_pd_py", py=fn_pandas, py_ready_polars=False)
    result = v._call_var_function(cohort)
    assert v.data is not None
    pl_testing.assert_frame_equal(
        v.data.with_columns(pl.col("recordtime_end").cast(pl.Null)), var_df
    )
    assert result is True
    assert "var" in called
    assert called["var"].data is None
    assert "cohort" in called
    assert isinstance(called["cohort"]._obs, pd.DataFrame)
    pl_testing.assert_frame_equal(pl.from_pandas(called["cohort"]._obs), cohort._obs)

    # TODO: Fix recordtime_end conversion to string by providing a schema
    # v = Variable(var_name="v_pd_py", dynamic=True, py=fn, py_ready_polars=False)
    # result = v._call_var_function(cohort)
    # pl_testing.assert_frame_equal(v.data, df)


@settings(max_examples=1)
@given(df=var_dataframe(vtype="dynamic_ts_snapshot", size=10))
def test_call_var_function_return_type_error(df: pl.DataFrame) -> None:
    cohort = EmptyCohort(obs_level="hospital_stay")

    def bad_fn_polars(var: VariableContext, cohort: Cohort):
        return df.to_pandas()  # wrong: polars path expects pl.DataFrame

    def bad_fn_pandas(var: VariableContext, cohort: Cohort):
        return df  # wrong: pandas path expects pd.DataFrame

    v = dynamic_variable(var_name="v_bad_py", py=bad_fn_polars, py_ready_polars=True)
    with pytest.raises(TypeError):
        v._call_var_function(cohort)

    v = dynamic_variable(var_name="v_bad_py", py=bad_fn_pandas, py_ready_polars=False)
    with pytest.raises(TypeError):
        v._call_var_function(cohort)


def test_variable_timefilter() -> None:
    """Test variable timefilter operation"""
    var = dynamic_variable()
    with pytest.warns(UserWarning, match="called before extraction"):
        var._timefilter()

    # Should not fail if case_tmin/case_tmax not present
    var.data = pl.DataFrame({"recordtime": [1, 2, 3]})
    var._timefilter()
    pl_testing.assert_frame_equal(var.data, pl.DataFrame({"recordtime": [1, 2, 3]}))

    # Should filter if case_tmin/case_tmax present
    var.data = pl.DataFrame(
        {
            "recordtime": [1, 2, 3],
            "case_tmin": [1.5, 1.5, 1.5],
            "case_tmax": [2.5, 2.5, 2.5],
        }
    )

    var._timefilter()
    assert len(var.data) == 1
    assert var.data["recordtime"].item() == 2
    # TODO: Add always=True test case with missing case_tmin/case_tmax


@given(df=var_dataframe(vtype="dynamic_ts_snapshot", size=8))
def test_timefilter_removes_case_columns(df: pl.DataFrame) -> None:
    v = dynamic_variable()

    # Add case_tmin/case_tmax around recordtime so rows survive but columns should be dropped after filtering
    df_with_case = df.with_columns(
        (pl.col("recordtime") - pl.duration(seconds=3600)).alias("case_tmin"),
        (pl.col("recordtime") + pl.duration(seconds=3600)).alias("case_tmax"),
    )
    v.data = df_with_case

    v._timefilter()

    assert "case_tmin" not in v.data.columns
    assert "case_tmax" not in v.data.columns


@given(df=var_dataframe(vtype="dynamic_ts_snapshot", size=12))
def test_apply_cleaning_filters_values(df: pl.DataFrame) -> None:
    """Test varibale cleaning operation"""
    v = dynamic_variable()
    v.data = df
    v.cleaning = {"value": {"low": 10.0, "high": 90.0}}
    v._apply_cleaning()

    # size should decrease
    assert len(v.data) <= len(df)
    current_size = len(v.data)

    # values should be within bounds
    min_val = v.data.select(pl.col("value").min()).item() or float("inf")
    max_val = v.data.select(pl.col("value").max()).item() or float("-inf")
    assert min_val >= 10.0
    assert max_val <= 90.0

    # size should remain the same
    v._apply_cleaning()
    assert len(v.data) == current_size


@given(
    var_df=var_dataframe(
        vtype="dynamic_ts_snapshot",
        obs_level=ObsLevel.ICU_STAY,
        size=10,
    )
)
def test_add_relative_times_adds_expected_columns(var_df: pl.DataFrame) -> None:
    primary_key = ObsLevel.ICU_STAY.primary_key

    # Build cohort._obs with a reference time (icu_admission) per primary_key earlier than recordtime
    tmins = (
        var_df.group_by(primary_key)
        .agg(pl.col("recordtime").min().alias("icu_admission"))
        .with_columns(
            (pl.col("icu_admission") - pl.duration(seconds=3600)).alias("icu_admission")
        )
    )

    result = add_relative_times(
        reference_df=tmins,
        var_df=var_df,
        reference_col="icu_admission",
        join_col=primary_key,
        suffix="_relative",
        include_time_cols=["recordtime"],
    )

    assert "recordtime_relative" in result.columns
    # relative times should be numeric and 3600 (since t_min was set before recordtime)
    min_rel = result.select(pl.col("recordtime_relative").min()).item()
    assert min_rel == 3600


@given(df=var_dataframe(vtype="dynamic_ts_interval", size=8))
def test_unify_and_order_columns_adds_missing_and_orders(df: pl.DataFrame) -> None:
    primary_key = ObsLevel.HOSPITAL_STAY.primary_key

    # start with dataframe that may not contain all DYN_COLUMNS
    result = unify_and_order_columns(df.select(primary_key, "value", "recordtime"))

    # All dynamic columns should now exist
    for col in DYN_COLUMNS:
        assert col in result.columns

    # primary key should appear before dynamic columns in the ordering
    cols = result.columns
    pk_index = cols.index(primary_key)
    first_dyn_index = min(cols.index(c) for c in DYN_COLUMNS)
    assert pk_index < first_dyn_index

    # COL_ORDER columns that exist should be the leading segment of the dataframe columns
    expected_leading = [c for c in COL_ORDER if c in set(cols)]
    assert cols[: len(expected_leading)] == expected_leading

    # Other columns should be appended to the end of the list and warn
    with pytest.warns(UserWarning):
        result = unify_and_order_columns(
            df.select(primary_key, "value", "recordtime", pl.lit(None).alias("extra"))
        )
    cols = result.columns
    assert cols[-1] == "extra"


####################
#  Variable props  #
####################
@given(
    data=dataframes(
        max_cols=10,
        max_size=10,
    )
)
def test_variable_repr_and_str(data: pl.DataFrame) -> None:
    """Test __repr__ and __str__ methods"""
    var = dynamic_variable(var_name="test_var")
    s = str(var)
    assert "Variable: test_var" in s
    assert "time_window" in s
    assert "data: Not extracted" in s
    assert "dynamic" in s
    assert var.__repr__() == s

    var = static_variable(var_name="test_var_static")
    var.data = data
    s = str(var)
    assert "Variable: test_var" in s
    assert "time_window" in s
    assert f"data: {data.shape}" in s
    assert "static" in s
    assert var.__repr__() == s


def test_variable_getstate_setstate() -> None:
    """Test __getstate__ and __setstate__ methods"""
    var = dynamic_variable(var_name="test_var")
    state = var.__getstate__()
    new_var = static_variable(var_name="other")
    new_var.__setstate__(state)
    assert new_var.var_name == "test_var"
    assert new_var.dynamic is True
