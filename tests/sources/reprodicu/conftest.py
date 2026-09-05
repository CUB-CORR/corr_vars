import os

import pytest

from corr_vars import Cohort


@pytest.fixture
def cohort_kwargs():
    """Base fixture for cohort creation with default arguments"""
    return {
        "sources": {
            "reprodicu": {"path": os.environ["REPRODICU_PATH"]},
        },
        "obs_level": "icu_stay",
        "load_default_vars": False,
        # reprodicu is served by the Concepts API: every read is scoped to a
        # project, and the key comes from CORR_CONCEPTS_API_KEY.
        "project": os.getenv("CORR_CONCEPTS_PROJECT"),
    }


@pytest.fixture(
    params=[
        ("icu_stay",),
    ],
    ids=lambda param: f"obs={param[0]}",
)
def sample_cohort(cohort_kwargs, request):
    """Fixture for a basic cohort instance for all valid database/observation level combinations.
    Note: procedure observation level is only available in db_corror.
    """
    (obs_level,) = request.param

    # Update kwargs
    cohort_kwargs = cohort_kwargs.copy()
    cohort_kwargs["obs_level"] = obs_level

    return Cohort(**cohort_kwargs)


@pytest.fixture
def sample_cohort_factory(cohort_kwargs):
    """Build a reprodICU cohort at an arbitrary observation level."""

    def factory(**overrides):
        kwargs = {**cohort_kwargs, **overrides}
        return Cohort(**kwargs)

    return factory
