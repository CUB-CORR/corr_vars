# tests/integration/test_cohort.py
import os
from datetime import datetime, timezone
from pathlib import Path

import hypothesis.strategies as st
import pandas as pd
import polars as pl
import polars.testing as pl_testing
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis_cohort import obs_dataframe
from polars.exceptions import ColumnNotFoundError
from polars.testing.parametric import column, dataframes
from tableone import TableOne

from conftest import EmptyCohort, Spy
from corr_vars.core.change_tracker import ChangeTrackerPipeline
from corr_vars.core.cohort import (
    Cohort,
    CohortDataError,
    ObsmDict,
    VariableNotFoundError,
)
from corr_vars.core.file_manager import TemporaryDirectoryManager
from corr_vars.definitions import ObsLevel, VariableProtocol
from corr_vars.sources.var_loader import MultiSourceVariable
from corr_vars.utils.time import TimeWindow

from collections.abc import Callable


def wrap_as_multisource_var(
    var: VariableProtocol, *, source_name: str = "pytest"
) -> MultiSourceVariable:
    return MultiSourceVariable(variables={source_name: var})


class FakeVariable(VariableProtocol):
    """FakeVariable following the VariableProtocol.
    This will simply return a `pl.Dataframe` with an row index as the value column for the given size by default.
    The return type can be specified by a predefined return value or a return function which will receive the cohort as an argument.
    Its also possible to inject side-effects into the extract method.
    """

    def __init__(
        self,
        # VariableProtocol
        var_name: str,
        dynamic: bool = True,
        time_window: TimeWindow | None = None,
        # FakeVar
        size: int = 5,
        return_value: pl.DataFrame | None = None,
        side_effect: Callable[[Cohort], None] | None = None,
        wrap: Callable[[Cohort], pl.DataFrame] | None = None,
    ):
        self.var_name = var_name
        self.dynamic = dynamic
        self.time_window = time_window or TimeWindow(tmin="X", tmax="Y")

        self.size = size
        self.return_value = return_value
        self.side_effect = side_effect
        self.wrap = wrap

    def extract(self, cohort: Cohort):
        if self.side_effect is not None:
            self.side_effect(cohort)
        if self.wrap is not None:
            return self.wrap(cohort)
        if self.return_value is not None:
            return self.return_value
        # Return a small DF with primary key and a value column
        pk = cohort.primary_key
        value_column = "value" if self.dynamic else self.var_name
        return cohort.obs.head(self.size).select(pk).with_row_index(name=value_column)


####################
#     ObsmDict     #
####################
@st.composite
def obsm(
    draw: Callable[[st.SearchStrategy[pl.DataFrame]], pl.DataFrame],
    size: int = 10,
) -> ObsmDict:
    return ObsmDict(
        {
            "X_pca": draw(
                dataframes(
                    cols=[
                        column(
                            "a",
                            strategy=st.floats(
                                min_value=-10, max_value=10, allow_subnormal=False
                            ),
                        ),
                        column(
                            "b",
                            strategy=st.floats(
                                min_value=-10, max_value=10, allow_subnormal=False
                            ),
                        ),
                    ],
                    min_size=size,
                    max_size=size,
                )
            ),
            "X_umap": draw(
                dataframes(
                    cols=[
                        column(
                            "x",
                            strategy=st.floats(
                                min_value=-10, max_value=10, allow_subnormal=False
                            ),
                        ),
                        column(
                            "y",
                            strategy=st.floats(
                                min_value=-10, max_value=10, allow_subnormal=False
                            ),
                        ),
                    ],
                    min_size=size,
                    max_size=size,
                )
            ),
        }
    )


@given(obsm=obsm(), new_df=dataframes(min_cols=1, max_cols=1, min_size=5, max_size=5))
def test_obsm_get_and_set_item(obsm: ObsmDict, new_df: pl.DataFrame) -> None:
    """Test __getitem__ and __setitem__ methods"""
    df = obsm["X_pca"]
    assert df.shape == (10, 2)

    obsm["new"] = new_df
    assert "new" in obsm
    assert obsm["new"].shape == (5, 1)


@given(obsm=obsm())
def test_obsm_getitem_error(obsm: ObsmDict) -> None:
    """Test __getitem__ method with missing key"""
    with pytest.raises(VariableNotFoundError) as exc_info:
        obsm["missing_key"]

    exc_info.match("missing_key")
    exc_info.match("not found in cohort.obsm")


@given(obsm=obsm())
def test_obsm_getitem_similar_key_error(obsm: ObsmDict) -> None:
    """Test __getitem__ method with missing key, which is similar to existing key"""
    with pytest.raises(VariableNotFoundError) as exc_info:
        obsm["X_pcb"]

    exc_info.match("X_pcb")
    exc_info.match("not found")
    exc_info.match("Did you mean")
    exc_info.match("X_pca")


def test_empty_obsm_getitem_error() -> None:
    """Test __getitem__ method with missing key for empty obsm"""
    obsm = ObsmDict(data={})
    with pytest.raises(VariableNotFoundError) as exc_info:
        obsm["missing_key"]

    exc_info.match("missing_key")
    exc_info.match("not found in cohort.obsm")
    exc_info.match("No variables have been extracted yet")


@given(obsm=obsm())
def test_obsm_len_and_iter(obsm: ObsmDict) -> None:
    """Test __len__ and __iter__ methods"""
    assert len(obsm) == 2
    assert set(iter(obsm)) == {"X_pca", "X_umap"}


@given(obsm=obsm())
def test_obsm_delitem(obsm: ObsmDict) -> None:
    """Test __delitem__ method"""
    del obsm["X_pca"]
    assert "X_pca" not in obsm
    assert len(obsm) == 1


# @given(obsm=obsm())
# def test_obsm_str(obsm: ObsmDict) -> None:
#     """Test __str__ method"""
#     s = str(obsm)
#     assert "X_pca" in s
#     assert "shape=" in s


####################
#   Cohort Setup   #
####################
def test_cohort_creation_fields(dummy_cohort: Cohort) -> None:
    """Test creation of Cohort fields"""
    assert not dummy_cohort._from_file
    assert dummy_cohort.project_vars == {}
    assert dummy_cohort.obsm == ObsmDict({})
    assert dummy_cohort.logger_args == {}
    assert isinstance(dummy_cohort.tmpdir_manager, TemporaryDirectoryManager)
    assert isinstance(dummy_cohort._change_tracker, ChangeTrackerPipeline)


def test_setup_logger_calls_utils_and_sets_logger_args(
    dummy_cohort: Cohort, spy: Spy
) -> None:
    """Ensure _setup_logger stores logger_args and delegates to utils.configure_logger_level_and_handlers."""
    called = spy("corr_vars.core.cohort.utils.configure_logger_level_and_handlers")

    logger_args = {"level": 20, "file_path": "/tmp/test_logger.log", "file_mode": "a"}
    dummy_cohort._setup_logger(logger_args)

    # Cohort should store the provided args
    assert dummy_cohort.logger_args == logger_args

    # The utils function should have been called
    assert "logger" in called["last_kwargs"]

    # Other kwargs should match the provided logger_args exactly
    passed_kwargs = {k: v for k, v in called["last_kwargs"].items() if k != "logger"}
    assert passed_kwargs == logger_args


def test_setup_logger_with_empty_args_passes_only_logger(
    dummy_cohort: Cohort, spy: Spy
) -> None:
    """When passed an empty logger_args dict, the utils function should be invoked with only logger kwarg."""
    called = spy("corr_vars.core.cohort.utils.configure_logger_level_and_handlers")

    dummy_cohort._setup_logger({})

    # Cohort should store the provided args
    assert dummy_cohort.logger_args == {}

    # Only 'logger' key should be present
    assert set(called["last_kwargs"].keys()) == {"logger"}


def test_obs_level_loading_error_handling() -> None:
    """Test error cases for cohort loading"""
    with pytest.raises(ValueError, match=r"Observation level .* not supported."):
        EmptyCohort(obs_level="invalid_level")  # type: ignore

    with pytest.raises(ValueError, match="No source returned data."):
        Cohort(obs_level="hospital_stay", sources={})


def test_load_obs_level_and_obs_level_keys() -> None:
    """Test primary key setting for different observation levels"""
    cohort = EmptyCohort(obs_level="icu_stay")
    assert cohort.primary_key == "icu_stay_id"
    assert cohort.t_min == "icu_admission"
    assert cohort.t_max == "icu_discharge"
    assert cohort.t_eligible == "icu_admission"
    assert cohort.t_outcome == "icu_discharge"

    cohort = EmptyCohort(obs_level="hospital_stay")
    assert cohort.primary_key == "case_id"
    assert cohort.t_min == "hospital_admission"
    assert cohort.t_max == "hospital_discharge"
    assert cohort.t_eligible == "hospital_admission"
    assert cohort.t_outcome == "hospital_discharge"

    cohort = EmptyCohort(obs_level="procedure")
    assert cohort.primary_key == "procedure_id"
    assert cohort.t_min == "or_time_begin"
    assert cohort.t_max == "or_time_end"
    assert cohort.t_eligible == "or_time_begin"
    assert cohort.t_outcome == "hospital_discharge"


@given(obs=obs_dataframe(obs_level=ObsLevel.ICU_STAY, size=5))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_load_obs_level_data_sets_time_and_constant_vars(
    obs: pl.DataFrame,
    spy: Spy,
) -> None:
    called = spy(
        "corr_vars.core.cohort.loader.cohort_loader.load_cohort_data",
        return_value=obs,
    )

    # Create cohort instance without running __init__
    cohort = Cohort.__new__(Cohort)

    # Override class attributes used inside _load_obs_level_data
    cohort.obs_level = ObsLevel("icu_stay")
    cohort.sources = {}

    # Call the method under test
    cohort._load_obs_level_data()

    # _data_load_time set and timezone-aware UTC
    assert isinstance(cohort._data_load_time, datetime)
    assert cohort._data_load_time.tzinfo is timezone.utc

    # obs stored as returned DataFrame
    pl_testing.assert_frame_equal(cohort._obs, obs)

    # constant_vars = all columns except primary keys
    expected_constants = [c for c in obs.columns if c not in ObsLevel.primary_keys()]
    assert cohort.constant_vars == expected_constants

    # ensure fake loader was called with the expected obs_level
    assert len(called["last_args"]) > 0
    assert called["last_args"][1] == cohort.obs_level


def test_setup_change_tracker(dummy_cohort: Cohort) -> None:
    """Simple test to see whether two properties were set"""
    del dummy_cohort._change_tracker
    assert not hasattr(dummy_cohort, "_change_tracker")
    dummy_cohort._setup_change_tracker()
    assert isinstance(dummy_cohort._change_tracker, ChangeTrackerPipeline)


####################
#  Cohort Variable #
####################

# TODO: Test load_default_vars
# def test_load_default_vars():
#     pass


def test_add_variable_with_variable_object_dynamic(dummy_cohort: Cohort) -> None:
    # Fake variable that is dynamic (should be stored in obsm)
    fv = FakeVariable("dyn_var_test", dynamic=True, size=5)
    # Ensure not present before
    assert "dyn_var_test" not in dummy_cohort.obsm
    dummy_cohort.add_variable(wrap_as_multisource_var(fv))
    # After adding should be present in obsm
    assert "dyn_var_test" in dummy_cohort.obsm
    df_saved = dummy_cohort.obsm["dyn_var_test"]
    assert df_saved.shape[0] == 5
    assert "value" in df_saved.columns
    assert df_saved["value"].to_list() == [*range(5)]


def test_add_variable_with_variable_object_static(dummy_cohort: Cohort) -> None:
    # Fake variable that is static (should be joined into obs)
    fv = FakeVariable("stat_var_test", dynamic=False, size=5)
    # Ensure column not present before
    assert "stat_var_test" not in dummy_cohort._obs.columns
    dummy_cohort.add_variable(wrap_as_multisource_var(fv))
    # After adding should have new column in obs
    assert "stat_var_test" in dummy_cohort._obs.columns
    values = dummy_cohort._obs.select("stat_var_test").to_series().to_list()[:5]
    assert values == [*range(5)]


@given(obs=obs_dataframe(size=(10, 20)))
def test_add_variable_restores_obs_on_compromised_cohort(obs: pl.DataFrame) -> None:
    """Simulate a variable extraction that mutates cohort._obs (thus changing the primary-key set).
    Expect add_variable to restore the original _obs and raise CohortDataError.
    """
    # Create minimal cohort instance
    cohort = EmptyCohort.with_obs(obs=obs, obs_level="hospital_stay")
    # Fake variable that mutates cohort._obs when extracted

    # Run and assert restore + exception
    with pytest.raises(CohortDataError):

        def corrupt(cohort: Cohort) -> None:
            cohort._obs = cohort._obs.head(5)

        bad_var = FakeVariable(
            var_name="badvar", dynamic=False, size=5, side_effect=corrupt
        )
        cohort.add_variable(wrap_as_multisource_var(bad_var))

    # Cohort._obs should have been restored to original state
    pl_testing.assert_frame_equal(cohort._obs, obs)


def test_load_variable_string_calls_loader(dummy_cohort: Cohort, spy: Spy) -> None:
    def fake_load_variable(
        var_name, cohort, time_window, include_sources, overrides=None, spec=None
    ):
        fv = FakeVariable(var_name, dynamic=True)
        return wrap_as_multisource_var(var=fv, source_name="fake")

    called = spy(
        "corr_vars.core.cohort.loader.var_loader.load_variable", wrap=fake_load_variable
    )

    # Call _load_variable with string and no explicit tmin/tmax
    var_obj = dummy_cohort.load_variable("some_loaded_var")
    assert isinstance(var_obj.variables["fake"], FakeVariable)
    # And loader should have been called with those values
    assert called["last_kwargs"]["var_name"] == "some_loaded_var"
    assert called["last_kwargs"]["cohort"] is dummy_cohort
    assert called["last_kwargs"]["time_window"] == TimeWindow(
        tmin=dummy_cohort.t_min, tmax=dummy_cohort.t_max
    )


def test_load_variable_with_multisource_variable(dummy_cohort: Cohort) -> None:
    fv = FakeVariable("some_loaded_var", dynamic=True)
    var = wrap_as_multisource_var(var=fv, source_name="fake")

    # Call _load_variable with variable
    var_obj = dummy_cohort.load_variable(var)
    assert isinstance(var_obj.variables["fake"], FakeVariable)
    assert var_obj == var

    # Call _load_variable with variable
    with pytest.warns(UserWarning, match="Please specify tmin/tmax inside"):
        var_obj = dummy_cohort.load_variable(var, tmin="X", tmax="Y")


def test_save_variable_static(dummy_cohort: Cohort) -> None:
    # Prepare var_data with multiple non-primary columns that require renaming
    pk = dummy_cohort.primary_key
    ids = dummy_cohort._obs.select(pk).to_series().to_list()
    # Create var_data where var_name is 'static' and columns are 'static'
    var_data = pl.DataFrame({pk: ids[:3], "static": [1, 2, 3]})

    # Ensure columns do not yet exist in obs
    assert "static" not in dummy_cohort._obs.columns

    # Save with same name
    dummy_cohort._save_variable(var_name="static", var_dynamic=False, var_data=var_data)

    # Assert new columns are present with same values
    assert "static" in dummy_cohort._obs.columns

    # Compare values for the first three rows we provided
    saved_static = dummy_cohort._obs.select("static").to_series().to_list()[:3]
    assert saved_static == [1, 2, 3]


def test_save_variable_dynamic(dummy_cohort: Cohort) -> None:
    # Prepare var_data with multiple non-primary columns that require renaming
    pk = dummy_cohort.primary_key
    ids = dummy_cohort._obs.select(pk).to_series().to_list()
    # Create var_data where var_name is 'dynamic' and columns are 'recordtime' and 'value'
    var_data = pl.DataFrame(
        {
            pk: ids[:3] * 3,
            "recordtime": [
                datetime(2000, 1, 1, 0, 0, 0),
                datetime(2000, 1, 1, 0, 1, 0),
                datetime(2000, 1, 1, 0, 2, 0),
            ]
            * 3,
            "value": [1, 2, 3] * 3,
        }
    )

    # Ensure data do not yet exist in obsm
    assert "dynamic" not in dummy_cohort._obsm

    # Save with var_name 'dynamic'
    dummy_cohort._save_variable(var_name="dynamic", var_dynamic=True, var_data=var_data)

    # Assert new columns are present with same values
    assert "dynamic" in dummy_cohort._obsm

    # Compare values for the first three rows we provided
    saved_dynamic = (
        dummy_cohort._obsm["dynamic"].select("value").to_series().to_list()[:3]
    )
    assert saved_dynamic == [1, 2, 3]


#: `(var_name, save_as)` pairs covering every overlap between the two names.
#: `save_as` extending `var_name` is the interesting one: renaming used to add a
#: renamed copy and then drop every column starting with `var_name`, which also
#: matched the freshly renamed column and silently dropped the variable.
SAVE_AS_NAMING = [
    ("orig", "new"),
    ("test", "test_new"),
    ("test_new", "test"),
]
SAVE_AS_NAMING_IDS = [
    "disjoint",
    "save_as_extends_var_name",
    "var_name_extends_save_as",
]


@pytest.mark.parametrize(
    "var_name, save_as",
    [("static", None), *SAVE_AS_NAMING],
    ids=["no_save_as", *SAVE_AS_NAMING_IDS],
)
def test_save_variable_obs_conflict(
    dummy_cohort: Cohort, var_name: str, save_as: str | None
) -> None:
    """Re-saving over an existing column must overwrite it, whatever the naming."""
    pk = dummy_cohort.primary_key
    target = save_as or var_name
    dummy_cohort.obs = dummy_cohort.obs.head(3).with_columns(pl.lit(0).alias(target))

    # Create var_data carrying the variable under its own name
    ids = dummy_cohort._obs.select(pk).to_series().to_list()
    var_data = pl.DataFrame({pk: ids, var_name: [1, 2, 3]})

    # Assert old columns are present with same values
    assert target in dummy_cohort._obs.columns
    saved_static = dummy_cohort._obs.select(target).to_series().to_list()
    assert saved_static == [0, 0, 0]

    # Save -> should override the original column
    dummy_cohort._save_variable(
        var_name=var_name, var_dynamic=False, var_data=var_data, save_as=save_as
    )

    # Assert new columns are present with changed values
    assert target in dummy_cohort._obs.columns
    saved_static = dummy_cohort._obs.select(target).to_series().to_list()
    assert saved_static == [1, 2, 3]


@pytest.mark.parametrize("var_name, save_as", SAVE_AS_NAMING, ids=SAVE_AS_NAMING_IDS)
def test_save_variable_rename_conflict(
    dummy_cohort: Cohort, var_name: str, save_as: str
) -> None:
    """Saving under save_as renames the variable column and its sub-columns.

    Covers the exact-name column (where ``removeprefix`` leaves an empty remainder)
    as well as ``<var_name>_*`` sub-columns, for every overlap between the two names.
    """
    # Prepare var_data with multiple non-primary columns that require renaming
    pk = dummy_cohort.primary_key
    ids = dummy_cohort._obs.select(pk).to_series().to_list()
    var_data = pl.DataFrame(
        {
            pk: ids[:3],
            var_name: [1, 2, 3],
            f"{var_name}_mean": [0.1, 0.2, 0.3],
            f"{var_name}_count": [10, 20, 30],
        }
    )

    # Ensure neither the source nor the target columns exist in obs yet
    for col in (var_name, save_as):
        assert col not in dummy_cohort._obs.columns
        assert f"{col}_mean" not in dummy_cohort._obs.columns
        assert f"{col}_count" not in dummy_cohort._obs.columns

    dummy_cohort._save_variable(
        var_name=var_name, var_dynamic=False, var_data=var_data, save_as=save_as
    )

    # Assert renamed columns are present with the same values
    assert dummy_cohort._obs.select(save_as).to_series().to_list()[:3] == [1, 2, 3]
    assert dummy_cohort._obs.select(f"{save_as}_mean").to_series().to_list()[:3] == [
        0.1,
        0.2,
        0.3,
    ]
    assert dummy_cohort._obs.select(f"{save_as}_count").to_series().to_list()[:3] == [
        10,
        20,
        30,
    ]

    # Renamed, not copied: the original names must not linger alongside
    assert var_name not in dummy_cohort._obs.columns
    assert f"{var_name}_mean" not in dummy_cohort._obs.columns
    assert f"{var_name}_count" not in dummy_cohort._obs.columns


def test_add_variable_definition(dummy_cohort: Cohort) -> None:
    """Test adding and updating variable definitions"""
    # Add a new variable definition
    var_name = "my_new_var"
    var_dict = {"type": "py_test", "table": "unittest", "where": "value > 5"}
    dummy_cohort.add_variable_definition(var_name, var_dict)
    assert var_name in dummy_cohort.project_vars
    for k, v in var_dict.items():
        assert dummy_cohort.project_vars[var_name][k] == v

    # Update existing variable definition
    update_dict = {"where": "value > 10"}
    dummy_cohort.add_variable_definition(var_name, update_dict)
    assert dummy_cohort.project_vars[var_name]["where"] == "value > 10"
    # Other keys should remain unchanged
    assert dummy_cohort.project_vars[var_name]["type"] == "py_test"
    assert dummy_cohort.project_vars[var_name]["table"] == "unittest"


def test_get_variable_definition(dummy_cohort: Cohort) -> None:
    """Test getting variable definitions"""
    var_name = "my_new_var"
    var_dict = {"type": "py_test", "table": "unittest", "where": "value > 5"}
    dummy_cohort.add_variable_definition(var_name, var_dict)

    dummy_cohort.sources = {
        "pytest": {"config": "key"},
        "unittest": {"config": "key"},
    }  # type: ignore
    var_def = dummy_cohort.get_variable_definition("my_new_var")
    assert var_def == {"pytest": var_dict, "unittest": var_dict}


@given(obs=obs_dataframe(size=10, obs_level=ObsLevel.HOSPITAL_STAY))
def test_validate_cohort(obs: pl.DataFrame) -> None:
    cohort = EmptyCohort.with_obs(obs=obs, obs_level="hospital_stay")

    # Should not raise an error
    cohort._validate_cohort()

    # Should raise an error
    cohort.obs = pl.concat([obs, obs])
    with pytest.raises(CohortDataError):
        cohort._validate_cohort()


####################
#  Cohort Inc/Excl #
####################
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=10)
@given(
    obs=dataframes(
        cols=[
            # Allow filtering by providing ID columns
            column("patient_id", dtype=pl.Utf8),
            column("case_id", dtype=pl.Utf8),
            column("icu_stay_id", dtype=pl.Utf8),
            column("procedure_id", dtype=pl.Utf8),
            column("eligible_time", dtype=pl.Datetime),
            column("wrong_type", dtype=pl.Float64),
        ],
        min_size=1,
        allow_null=False,
    )
)
def test_set_t_eligible_and_drop_ineligible(
    dummy_cohort: Cohort, obs: pl.DataFrame
) -> None:
    """Test setting t_eligible column"""
    # Add a datetime column to obs
    dummy_cohort._obs = obs
    dummy_cohort.set_t_eligible("eligible_time", drop_ineligible=False)
    assert dummy_cohort.t_eligible == "eligible_time"

    # TODO: Can be removed as this is already tested?
    # Should raise if column does not exist
    with pytest.raises(ColumnNotFoundError, match="not found"):
        dummy_cohort.set_t_eligible("not_a_column", drop_ineligible=False)

    # Should raise if column is not datetime
    with pytest.raises(TypeError, match="not a datetime"):
        dummy_cohort.set_t_eligible("wrong_type", drop_ineligible=False)

    # Test dropping ineligible
    # TODO: Test whether correct rows were dropped
    tracker_steps = len(dummy_cohort._change_tracker.steps)
    before_size = len(dummy_cohort)
    dummy_cohort.set_t_eligible("eligible_time", drop_ineligible=True)
    after_size = len(dummy_cohort)
    assert after_size <= before_size
    assert tracker_steps + 1 == len(dummy_cohort._change_tracker.steps)

    # Dropping a second time should not change the size
    dummy_cohort.set_t_eligible("eligible_time", drop_ineligible=True)
    assert len(dummy_cohort) == after_size
    # TODO: Don't update if there is no change?
    assert tracker_steps + 2 == len(dummy_cohort._change_tracker.steps)


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=10)
@given(
    obs=dataframes(
        cols=[
            column("outcome_time", dtype=pl.Datetime),
            column("wrong_type", dtype=pl.Float64),
        ],
        min_size=1,
        allow_null=False,
    )
)
def test_set_t_outcome(dummy_cohort: Cohort, obs: pl.DataFrame) -> None:
    """Test setting t_outcome column"""
    # Add a datetime column to obs
    dummy_cohort._obs = obs
    dummy_cohort.set_t_outcome("outcome_time")
    assert dummy_cohort.t_outcome == "outcome_time"

    # TODO: Can be removed as this is already tested?
    # Should raise if column does not exist
    with pytest.raises(ColumnNotFoundError, match="not found"):
        dummy_cohort.set_t_outcome("not_a_column")

    # Should raise if column is not datetime
    with pytest.raises(TypeError, match="not a datetime"):
        dummy_cohort.set_t_outcome("wrong_type")


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(obs=dataframes(cols=column("datetime_col"), allowed_dtypes=pl.Datetime))
def test_assert_datetime_col_success(dummy_cohort: Cohort, obs: pl.DataFrame) -> None:
    """Tests that _assert_datetime_col runs without error for a datetime column."""
    dummy_cohort._obs = obs
    dummy_cohort._assert_datetime_col("datetime_col")


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(obs=dataframes(cols=column("datetime_col"), allowed_dtypes=pl.Datetime))
def test_assert_datetime_col_raises_error_if_col_not_found(
    dummy_cohort: Cohort, obs: pl.DataFrame
) -> None:
    """Tests that _assert_datetime_col raises an AssertionError if the column does not exist."""
    dummy_cohort._obs = obs
    with pytest.raises(ColumnNotFoundError, match="not found"):
        dummy_cohort._assert_datetime_col("non_existent_col")


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(obs=dataframes(cols=column("not_datetime_col"), excluded_dtypes=pl.Datetime))
def test_assert_datetime_col_raises_error_if_not_datetime(
    dummy_cohort: Cohort, obs: pl.DataFrame
) -> None:
    """Tests that _assert_datetime_col raises an AssertionError if the column is not a datetime type."""
    dummy_cohort._obs = obs
    with pytest.raises(TypeError, match="not a datetime"):
        dummy_cohort._assert_datetime_col("not_datetime_col")


# TODO: Test change_tracker here
# def test_change_tracker():
#     pass


def test_inclusion(dummy_cohort: Cohort) -> None:
    """Test inclusion/exclusion"""
    dummy_cohort.add_variable("age_on_admission")

    dummy_cohort.include_list(
        [
            {
                "variable": "age_on_admission",
                "operation": ">= 18",
                "label": "Adult patients",
                "operations_done": "Excluded patients under 18 years old",
            }
        ]
    )

    assert len(dummy_cohort._change_tracker.steps) == 1
    dummy_cohort.include(variable="sex", operation="== 'M'")
    assert len(dummy_cohort._change_tracker.steps) == 2


def test_exclusion(dummy_cohort: Cohort) -> None:
    """Test exclusion"""
    dummy_cohort.add_variable("age_on_admission")

    dummy_cohort.exclude_list(
        [
            {
                "variable": "age_on_admission",
                "operation": ">= 18",
                "label": "Adult patients",
                "operations_done": "Excluded patients under 18 years old",
            }
        ]
    )

    assert len(dummy_cohort._change_tracker.steps) == 1
    dummy_cohort.exclude(variable="sex", operation="== 'M'")
    assert len(dummy_cohort._change_tracker.steps) == 2


def test_cohort_inclusion_exclusion_error_handling(dummy_cohort: Cohort) -> None:
    """Test error handling for inclusion/exclusion methods"""
    with pytest.raises(NotImplementedError, match="include_list"):
        dummy_cohort.add_inclusion()

    with pytest.raises(NotImplementedError, match="exclude_list"):
        dummy_cohort.add_exclusion()


# TODO: Test include_list / exclude_list here
# def test_include_list():
#     pass
# def test_exclude_list():
#     pass


####################
# Cohort Load/Save #
####################
@pytest.mark.parametrize(
    "extension",
    ["arrow", "json", "jsonl", "xlsx"],
)
def test_to_file(dummy_cohort: Cohort, tmp_path: Path, extension: str) -> None:
    """Test saving and loading a cohort (csv and parquet will be tested seperatly; parquet will also test loading from parquet)"""
    # Add some variables
    sample_vars = ["age_on_admission", "blood_sodium_dynamic"]
    for v in sample_vars:
        dummy_cohort.add_variable(v)

    # Save
    folder = tmp_path / f"test_cohort_{dummy_cohort.obs_level.lower_name}"
    dummy_cohort.to_files(folder.as_posix(), ext=extension)

    # Assert
    assert (
        folder / f"_obs.{extension}"
    ).exists(), f"Expected _obs.{extension} file not found"
    assert (
        folder / f"blood_sodium_dynamic.{extension}"
    ).exists(), f"Expected blood_sodium_dynamic.{extension} file not found"


def test_to_csv(dummy_cohort: Cohort, tmp_path: Path) -> None:
    """Test saving and loading a cohort"""
    # Add some variables
    sample_vars = ["age_on_admission", "blood_sodium_dynamic"]
    for v in sample_vars:
        dummy_cohort.add_variable(v)

    # Save
    folder = tmp_path / f"test_cohort_{dummy_cohort.obs_level.lower_name}"
    dummy_cohort.to_csv(folder.as_posix())

    # Assert
    assert (folder / "_obs.csv").exists(), "Expected _obs.csv file not found"
    assert (
        folder / "blood_sodium_dynamic.csv"
    ).exists(), "Expected blood_sodium_dynamic.csv file not found"


def test_to_parquet(dummy_cohort: Cohort, tmp_path: Path) -> None:
    """Test saving and loading a cohort"""
    # Add some variables
    sample_vars = ["age_on_admission", "blood_sodium_dynamic"]
    for v in sample_vars:
        dummy_cohort.add_variable(v)

    # Save
    folder = tmp_path / f"test_cohort_{dummy_cohort.obs_level.lower_name}"
    dummy_cohort.to_parquet(folder.as_posix())

    # Assert
    assert (folder / "_obs.parquet").exists(), "Expected _obs.parquet file not found"
    assert (
        folder / "blood_sodium_dynamic.parquet"
    ).exists(), "Expected blood_sodium_dynamic.parquet file not found"

    # Load
    obs = pl.read_parquet(
        folder / "_obs.parquet",
    )
    blood_sodium = pl.read_parquet(folder / "blood_sodium_dynamic.parquet")

    # Assert
    pl_testing.assert_series_equal(
        obs[dummy_cohort.primary_key], dummy_cohort.obs[dummy_cohort.primary_key]
    )
    pl_testing.assert_series_equal(
        obs["age_on_admission"], dummy_cohort.obs["age_on_admission"]
    )
    pl_testing.assert_series_equal(
        blood_sodium[dummy_cohort.primary_key],
        dummy_cohort.obsm["blood_sodium_dynamic"][dummy_cohort.primary_key],
    )
    pl_testing.assert_series_equal(
        blood_sodium["value"],
        dummy_cohort.obsm["blood_sodium_dynamic"]["value"],
    )


def test_save_and_load(dummy_cohort: Cohort, tmp_path: Path) -> None:
    """Test saving and loading a cohort"""
    # Add some variables
    sample_vars = ["age_on_admission", "blood_sodium_dynamic"]
    for v in sample_vars:
        dummy_cohort.add_variable(v)

    # Save
    file = tmp_path / f"test_cohort_{dummy_cohort.obs_level}.corr3"
    # TODO: Add path support
    dummy_cohort.save(file.as_posix())

    # Load
    loaded_cohort = Cohort.load(file.as_posix())

    # Assert
    assert (
        loaded_cohort.obs_level == dummy_cohort.obs_level
    ), f"{loaded_cohort.obs_level.lower_name} != {dummy_cohort.obs_level.lower_name}"
    pl_testing.assert_frame_equal(loaded_cohort.obs, dummy_cohort.obs)
    pl_testing.assert_frame_equal(
        loaded_cohort.obsm["blood_sodium_dynamic"],
        dummy_cohort.obsm["blood_sodium_dynamic"],
    )


def test_load_error_handling(dummy_cohort: Cohort) -> None:
    """Test error handling in load method"""
    # Invalid file extension
    with pytest.raises(ValueError, match="Unsupported file format"):
        dummy_cohort.load("invalid_file.txt")


def test_save_rename_handling(dummy_cohort: Cohort, spy: Spy) -> None:
    called = spy("corr_vars.core.cohort.Cohort._save_corr3")

    with pytest.warns(UserWarning, match="The new file format is .corr3"):
        dummy_cohort.save("oldformat.corr")
    assert isinstance(called["last_args"][1], Path)
    assert called["last_args"][1].name == "oldformat.corr3"

    with pytest.warns(UserWarning, match="The new file format is .corr3"):
        dummy_cohort.save("otherformat.txt")
    assert isinstance(called["last_args"][1], Path)
    assert called["last_args"][1].name == "otherformat.corr3"

    dummy_cohort.save("noextension")
    assert isinstance(called["last_args"][1], Path)
    assert called["last_args"][1].name == "noextension.corr3"


####################
#  Cohort tmp dir  #
####################
# TODO: Test file_manager.py seperatly
def test_tmpdir_variables(dummy_cohort: Cohort) -> None:
    """Test tmpdir_variables property"""
    import shutil

    # Create a dummy file in the cohort's tmpdir
    tmp_path = dummy_cohort.tmpdir_path
    test_file = os.path.join(tmp_path, "var_c_test.parquet")
    with open(test_file, "w") as f:
        f.write("dummy content")

    files = dummy_cohort.tmpdir_manager.tmpdir_variables
    assert any("var_c_test.parquet" in f for f in files)

    # Clean up and remove tmpdir_path attribute
    shutil.rmtree(tmp_path)
    dummy_cohort.tmpdir_manager.delete_tmpdir()

    # Now property should return empty list
    assert dummy_cohort.tmpdir_manager.tmpdir_variables == []


def test_create_tmpdir_warning(dummy_cohort: Cohort) -> None:
    """Test create_tmpdir with invalid path raises UserWarning"""
    with pytest.warns(UserWarning, match="does not exist"):
        dummy_cohort.tmpdir_manager.create_tmpdir("/invalid/path/that/does/not/exist")


def test_delete_tmpdir(dummy_cohort: Cohort) -> None:
    """Test delete_tmpdir method"""
    # Ensure tmpdir exists before deletion
    tmpdir_path = dummy_cohort.tmpdir_path
    assert os.path.exists(tmpdir_path)

    dummy_cohort.tmpdir_manager.delete_tmpdir()

    # After deletion, the directory should not exist
    assert not os.path.exists(tmpdir_path)

    # Delete folder in case of test failure
    import shutil

    shutil.rmtree(tmpdir_path, ignore_errors=True)


####################
#   Cohort utils   #
####################
@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=None,
    max_examples=10,
)
@given(obs=obs_dataframe(size=10, obs_level=ObsLevel.HOSPITAL_STAY))
def test_tableone_basic(dummy_cohort: Cohort, obs: pl.DataFrame) -> None:
    """Test basic TableOne functionality"""
    dummy_cohort.obs = obs.select(
        "age_on_admission",
        "hospital_admission",
        "hospital_discharge",
        "sex",
        "inhospital_death",
    )
    tab1 = dummy_cohort.tableone
    assert isinstance(tab1, TableOne)

    # Optionally check columns
    for col in ["age_on_admission", "inhospital_death"]:
        assert col in tab1._columns


@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=None,
    max_examples=10,
)
@given(obs=obs_dataframe(size=10, obs_level=ObsLevel.HOSPITAL_STAY))
def test_tableone_groupby(dummy_cohort: Cohort, obs: pl.DataFrame) -> None:
    dummy_cohort.obs = obs.select(
        "age_on_admission",
        "hospital_admission",
        "hospital_discharge",
        "sex",
        "inhospital_death",
    )
    tab1 = dummy_cohort.to_tableone(groupby="sex")
    assert isinstance(tab1, TableOne)
    assert "sex" in tab1._groupby


@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=None,
    max_examples=10,
)
@given(obs=obs_dataframe(size=10, obs_level=ObsLevel.HOSPITAL_STAY))
def test_tableone_ignore_cols(dummy_cohort: Cohort, obs: pl.DataFrame) -> None:
    """Test TableOne with ignore_cols parameter"""
    dummy_cohort.obs = obs.select(
        "age_on_admission",
        "hospital_admission",
        "hospital_discharge",
        "sex",
        "inhospital_death",
    )
    # Ignore 'age_on_admission' and 'hospital_admission'
    tab1 = dummy_cohort.to_tableone(ignore_cols=["age_on_admission"])

    # age_on_admission (user specified) and hospital_admission (datetime) should not be in the columns
    assert "age_on_admission" not in tab1._columns
    assert "hospital_admission" not in tab1._columns
    # hospital_discharge should still be present
    assert "inhospital_death" in tab1._columns


# TODO: parametrise (very slow and prone to breaking)
def test_stata(dummy_cohort: Cohort) -> None:
    """Test Stata export functionality"""
    stata_df = dummy_cohort.stata
    assert isinstance(stata_df, pd.DataFrame)
    assert stata_df.empty is False
    stata_df = dummy_cohort.to_stata()
    assert isinstance(stata_df, pd.DataFrame)
    assert stata_df.empty is False

    # Write Stata file
    file = os.path.join(
        os.path.dirname(__file__), f"test_stata_{dummy_cohort.obs_level.lower_name}.dta"
    )
    dummy_cohort.to_stata(to_file=file)

    # Check that file was created and is readable
    stata_df = pd.read_stata(file)
    assert isinstance(stata_df, pd.DataFrame)
    assert stata_df.empty is False

    os.remove(file)


####################
#   Cohort props   #
####################
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    obs=dataframes(
        min_cols=1,
        max_cols=10,
        min_size=1,
        max_size=10,
        allow_null=False,
        allowed_dtypes=[pl.Int64, pl.Utf8, pl.Boolean],
    )
)
def test_obs_setter(dummy_cohort: Cohort, obs: pl.DataFrame) -> None:
    """Test obs properties setter function"""
    # Assign a Polars DataFrame
    dummy_cohort.obs = obs
    assert isinstance(dummy_cohort._obs, pl.DataFrame)
    pl_testing.assert_frame_equal(dummy_cohort._obs, obs)

    # Assign a Pandas DataFrame
    df_pandas = obs.to_pandas()
    dummy_cohort.obs = df_pandas
    assert isinstance(dummy_cohort._obs, pl.DataFrame)
    pl_testing.assert_frame_equal(dummy_cohort._obs, obs)

    # Assign a Polars LazyFrame
    df_lazy = obs.lazy()
    dummy_cohort.obs = df_lazy
    assert isinstance(dummy_cohort._obs, pl.DataFrame)
    pl_testing.assert_frame_equal(dummy_cohort._obs, obs)

    # Assign an invalid type
    with pytest.raises(TypeError):
        dummy_cohort.obs = "not a dataframe"  # type: ignore


def test_obs_deleter(dummy_cohort: Cohort) -> None:
    """Test obs properties deleter function"""
    with pytest.warns(UserWarning, match="Can't delete Cohort.obs"):
        del dummy_cohort.obs


def test_cohort_repr_and_str(dummy_cohort: Cohort) -> None:
    """Test __repr__, __str__, and _repr_html_ methods"""
    s = str(dummy_cohort)
    r = repr(dummy_cohort)
    assert "Cohort" in r
    assert "obs_level" in s


def test_cohort_len_and_iter(dummy_cohort: Cohort) -> None:
    """Test __len__ and __iter__ methods"""
    assert len(dummy_cohort) == len(dummy_cohort.obs)
    assert [col.name for col in iter(dummy_cohort)] == dummy_cohort.obs.columns


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(obs=obs_dataframe(size=(1, 100), obs_level=ObsLevel.HOSPITAL_STAY))
def test_cohort_getitem_and_setitem_and_contains(
    dummy_cohort: Cohort, obs: pl.DataFrame
) -> None:
    """Test __getitem__, __setitem__, and __contains__ methods"""
    dummy_cohort.obs = obs

    first = dummy_cohort[:, 0]
    assert isinstance(first, pl.Series)
    pl_testing.assert_series_equal(first, obs[:, 0])
    dummy_cohort["new_col"] = [1] * len(dummy_cohort)
    assert "new_col" in dummy_cohort


def test_debug_print_stdout(dummy_cohort: Cohort, capsys: pytest.CaptureFixture):
    """Test debug_print outputs to stdout"""
    dummy_cohort.debug_print()
    captured = capsys.readouterr()
    assert "Cohort (repr)" in captured.out
    assert f"obs_level = '{dummy_cohort.obs_level.lower_name}'" in captured.out
