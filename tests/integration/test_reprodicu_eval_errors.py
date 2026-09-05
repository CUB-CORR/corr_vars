"""A reprodICU ``filter``/``calculation`` expression that cannot be evaluated must raise.

Both fields are polars expression strings carried by the variable definition and turned
into expressions at extraction time. A failure there used to be logged and swallowed: the
extraction then ran *without* the filter and returned every row of the column — a silently
different variable than the definition describes. These tests pin the failure down to the
expression itself instead.
"""

from __future__ import annotations

import polars as pl
import pytest

from corr_vars.definitions.exceptions import VariableDefinitionError
from corr_vars.sources.reprodicu.extract import VariableLoader
from corr_vars.sources.reprodicu.helpers import extract_variable_from_parquet
from corr_vars.utils.time import TimeWindow


@pytest.fixture
def data_path(tmp_path):
    """A minimal reprodICU-shaped parquet with two rows, one of them null."""
    pl.DataFrame(
        {
            "Global ICU Stay ID": ["stay-1", "stay-1"],
            "Time Relative to Admission (seconds)": [0, 3600],
            "Sodium": [140.0, None],
            "Factor": [2.0, 2.0],
        }
    ).write_parquet(tmp_path / "timeseries_labs.parquet")
    return str(tmp_path)


@pytest.fixture
def cohort_obs():
    return pl.DataFrame({"icu_stay_id": ["stay-1"]})


def _var(**config):
    config.setdefault("path", "timeseries_labs")
    config.setdefault("column", "Sodium")
    return VariableLoader(TimeWindow(), var_name="blood_sodium", **config)


def test_valid_filter_is_applied(data_path, cohort_obs):
    """The working case: the filter drops the null row."""
    var = _var(filter="pl.col('Sodium').is_not_null()")

    data = extract_variable_from_parquet(var, data_path, cohort_obs)

    assert data["value"].to_list() == [140.0]


def test_valid_calculation_is_applied(data_path, cohort_obs):
    """A calculation writes into the extracted column and its roots must exist."""
    var = _var(calculation="pl.col('Sodium') * pl.col('Factor')")

    data = extract_variable_from_parquet(var, data_path, cohort_obs)

    assert data["value"].to_list() == [280.0, None]


def test_malformed_filter_raises_instead_of_returning_unfiltered_data(
    data_path, cohort_obs
):
    """The regression: a broken filter must not quietly yield every row."""
    var = _var(filter="pl.col('Sodium').is_not_nul()")  # typo: no such method

    with pytest.raises(VariableDefinitionError) as excinfo:
        extract_variable_from_parquet(var, data_path, cohort_obs)

    assert "blood_sodium" in str(excinfo.value)
    assert "is_not_nul()" in str(excinfo.value)
    assert excinfo.value.__cause__ is not None, "the original error must be chained"


def test_filter_referencing_an_unknown_name_raises(data_path, cohort_obs):
    """A name the expression namespace does not hold is a definition error too."""
    var = _var(filter="polars.col('Sodium').is_not_null()")

    with pytest.raises(VariableDefinitionError, match="blood_sodium"):
        extract_variable_from_parquet(var, data_path, cohort_obs)


def test_malformed_calculation_raises(data_path, cohort_obs):
    """A broken calculation used to surface as a ColumnNotFoundError further down."""
    var = _var(calculation="pl.col('Sodium') * (")  # SyntaxError

    with pytest.raises(VariableDefinitionError) as excinfo:
        extract_variable_from_parquet(var, data_path, cohort_obs)

    assert "blood_sodium" in str(excinfo.value)
    assert "pl.col('Sodium') * (" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, SyntaxError)


def test_calculation_that_is_not_an_expression_raises(data_path, cohort_obs):
    """A syntactically fine expression yielding a non-Expr is equally unusable."""
    var = _var(calculation="'Sodium'")

    with pytest.raises(VariableDefinitionError, match="not a polars expression"):
        extract_variable_from_parquet(var, data_path, cohort_obs)
