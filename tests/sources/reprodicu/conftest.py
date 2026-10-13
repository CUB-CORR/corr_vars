import os

import pytest

from corr_vars import Cohort
from corr_vars.concepts.client import API_KEY_ENV_VAR

#: Environment these hand-run tests need: the reprodICU parquet files plus a
#: Concepts API project and key. Without them the tests skip instead of erroring.
REQUIRED_ENV_VARS = ("REPRODICU_PATH", "CORR_CONCEPTS_PROJECT", API_KEY_ENV_VAR)


@pytest.fixture
def cohort_kwargs():
    """Base fixture for cohort creation with default arguments"""
    missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]
    if missing:
        pytest.skip(f"reprodICU source tests need {', '.join(missing)}")
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
