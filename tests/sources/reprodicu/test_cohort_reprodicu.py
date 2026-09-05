import polars as pl
import pytest


@pytest.mark.parametrize(
    "obs_level",
    ["icu_stay", "hospital_stay", "patient"],
)
def test_cohort_creation(sample_cohort_factory, obs_level):
    """Test different observation levels"""
    cohort = sample_cohort_factory(obs_level=obs_level)

    assert not cohort._from_file
    assert cohort.obs_level == obs_level
    assert isinstance(cohort.obs, pl.DataFrame)
    assert len(cohort.obs) > 0
    for col in [
        cohort.primary_key,
        cohort.t_min,
        cohort.t_max,
        cohort.t_eligible,
        cohort.t_outcome,
    ]:
        assert col in cohort.obs.columns
