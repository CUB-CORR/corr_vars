import polars as pl
import pytest

from corr_vars.utils.helpers import filter_by_condition


def test_filter_by_condition_drop_and_keep() -> None:
    """Test filter_by_condition dropping and keeping rows"""
    df = pl.DataFrame({"a": [1, 2, 3, 4]})
    # Drop rows where a > 2
    result_drop = filter_by_condition(df, pl.col("a") > 2, mode="drop", verbose=False)
    assert result_drop.shape[0] == 2
    assert set(result_drop["a"]) == {1, 2}
    # Keep rows where a > 2
    result_keep = filter_by_condition(df, pl.col("a") > 2, mode="keep", verbose=False)
    assert result_keep.shape[0] == 2
    assert set(result_keep["a"]) == {3, 4}


def test_filter_by_condition_error() -> None:
    """Test filter_by_condition detecting incorrect expression type and raising exception"""
    df = pl.DataFrame({"a": [1, 2, 3]})
    # Expression does not return boolean
    with pytest.raises(
        ValueError, match="Conditional expression must return a boolean"
    ):
        filter_by_condition(df, pl.col("a"), verbose=False)


def test_filter_by_condition_warning() -> None:
    """Test filter_by_condition detecting null values and warning user"""
    df = pl.DataFrame({"a": [True, False, None]})
    # Expression returns boolean but contains nulls
    with pytest.warns(
        UserWarning, match="Condition expression should return a boolean"
    ):
        filter_by_condition(df, pl.col("a"), verbose=False)


def test_filter_by_condition_verbose(caplog: pytest.LogCaptureFixture) -> None:
    """Test filter_by_condition logging operation"""
    df = pl.DataFrame({"a": [1, 2, 3, 4]})
    # Should log info when verbose is True
    filter_by_condition(df, pl.col("a") > 2, verbose=True)
    assert any("DROP" in m for m in caplog.messages)


def test_filter_by_condition_invalid_mode() -> None:
    """Test filter_by_condition catching wrong mode"""
    df = pl.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError):
        filter_by_condition(df, pl.col("a") > 1, mode="invalid", verbose=False)  # type: ignore
