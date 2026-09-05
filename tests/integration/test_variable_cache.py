# tests/integration/test_variable_cache.py
"""Tests for the per-variable extraction cache in the cohort's tmpdir.

A source caches a variable's raw extract in the cohort's temporary directory
under a name the :class:`~corr_vars.core.file_manager.TemporaryDirectoryManager`
owns: ``var_c_<var_name>[@<contributor_key>][:<cache_key>].parquet``. What is
covered here is the bookkeeping around those files — which ones
``Cohort.clear_variable_cache`` drops, and that ``use_cache=False`` drops one
before extracting. Writing and reading the extract itself belongs to the source.
"""

from pathlib import Path

import polars as pl
import pytest

from conftest import EmptyCohort
from corr_vars.core.cohort import Cohort


@pytest.fixture
def cache_cohort() -> Cohort:
    """Cohort carrying just enough obs for the cache's id bookkeeping."""
    cohort = EmptyCohort(obs_level="hospital_stay")
    cohort._obs = pl.DataFrame({"case_id": ["1", "2", "3"]})
    return cohort


def _cache_file(
    cohort: Cohort, var_name: str, contributor_key: str | None = None
) -> Path:
    """Write a cached extract the way a source would, and return its path."""
    cache_name = (
        var_name if contributor_key is None else f"{var_name}@{contributor_key}"
    )
    path = Path(
        cohort.tmpdir_manager.create_tmpdir_variable_path(
            var_name=f"{cache_name}:case_id"
        )
    )
    pl.DataFrame({"case_id": ["1", "2", "3"], "value": [1.0, 1.0, 1.0]}).write_parquet(
        path
    )
    return path


def test_a_cached_extract_is_listed(cache_cohort: Cohort) -> None:
    path = _cache_file(cache_cohort, "heart_rate")

    assert path.exists()
    assert cache_cohort.tmpdir_manager.tmpdir_variables == [path.name]


def test_clear_variable_cache_drops_only_the_named_variable(
    cache_cohort: Cohort,
) -> None:
    """Clearing one variable leaves a similarly named one cached."""
    dropped = _cache_file(cache_cohort, "heart_rate", "demo_source")
    kept = _cache_file(cache_cohort, "heart_rate_variability", "demo_source")

    cache_cohort.clear_variable_cache("heart_rate")

    assert not dropped.exists()
    assert kept.exists()


def test_clear_variable_cache_drops_every_contributor(cache_cohort: Cohort) -> None:
    """Clearing a grouped variable drops all of its members' extracts."""
    members = [
        _cache_file(cache_cohort, "heart_rate", "demo_source#1"),
        _cache_file(cache_cohort, "heart_rate", "demo_source#2"),
    ]

    cache_cohort.clear_variable_cache("heart_rate")

    assert not any(member.exists() for member in members)


def test_clear_variable_cache_without_name_drops_everything(
    cache_cohort: Cohort,
) -> None:
    """Called bare, the whole extraction cache goes."""
    _cache_file(cache_cohort, "heart_rate", "demo_source")
    _cache_file(cache_cohort, "blood_sodium", "demo_source")

    cache_cohort.clear_variable_cache()

    assert cache_cohort.tmpdir_manager.tmpdir_variables == []


def test_add_variable_use_cache_false_clears_the_cache(
    dummy_cohort: Cohort, spy
) -> None:
    """``use_cache=False`` drops the cached extract before extracting."""
    called = spy(
        "corr_vars.core.cohort.Cohort.clear_variable_cache",
        wrap=lambda *args, **kwargs: None,
    )

    dummy_cohort.add_variable("dummy_static_var")
    assert called["count"] == 0

    dummy_cohort.add_variable("dummy_static_var_2", use_cache=False)
    assert called["count"] == 1
    assert called["last_args"][1] == "dummy_static_var_2"
