from __future__ import annotations

import polars as pl

from corr_vars.sources.local_datasource.extract import (
    DerivedDynamic,
    DerivedStatic,
    NativeDynamic,
)
from corr_vars.utils.time import TimeWindow


class _FakeVar:
    """Stand-in for a loaded dependency; records nothing, extracts to empty."""

    var_name = "dep"
    dynamic = True

    def extract(self, cohort: object) -> pl.DataFrame:
        return pl.DataFrame()


class _CapturingCohort:
    """Captures the `overrides` each requirement load is called with."""

    primary_key = "icu_stay_id"

    def __init__(self) -> None:
        self.overrides: list[dict | None] = []

    def load_variable(
        self,
        variable: object,
        include_sources: object = None,
        overrides: dict | None = None,
    ) -> _FakeVar:
        self.overrides.append(overrides)
        return _FakeVar()


def _requirement(template: str, **extra: object) -> dict:
    return {template: {"template": template, "tmin": None, "tmax": None, **extra}}


def _drive(var: DerivedDynamic | DerivedStatic) -> dict | None:
    var.time_window = TimeWindow("icu_admission", ("icu_admission", "+28d"))
    cohort = _CapturingCohort()
    var._get_required_vars(cohort)  # type: ignore[arg-type]
    assert len(cohort.overrides) == 1
    return cohort.overrides[0]


class TestScreenedRequirementsPropagation:
    def test_patient_screening_reaches_the_requirement(self) -> None:
        """The crash path: a derived concept forwards its screening as an override."""
        var = DerivedDynamic(
            "x", requires=_requirement("leaf"), screened_obs_level="patient"
        )
        assert _drive(var) == {"screened_obs_level": "patient"}

    def test_no_screening_forwards_nothing(self) -> None:
        var = DerivedDynamic("x", requires=_requirement("leaf"))
        assert _drive(var) is None

    def test_requirement_keeps_its_own_screening(self) -> None:
        """Setdefault must not clobber a requirement that set its own scope."""
        var = DerivedDynamic(
            "x",
            requires=_requirement("leaf", screened_obs_level="hospital_stay"),
            screened_obs_level="patient",
        )
        assert _drive(var) == {"screened_obs_level": "hospital_stay"}

    def test_propagation_does_not_mutate_the_shared_requirement_dict(self) -> None:
        """Requirement dicts can be shared with vars.json config; must not be mutated."""
        shared = _requirement("leaf")
        var = DerivedDynamic("x", requires=shared, screened_obs_level="patient")
        _drive(var)
        assert "screened_obs_level" not in shared["leaf"]

    def test_derived_static_propagates_too(self) -> None:
        var = DerivedStatic(
            "x", requires=_requirement("leaf"), screened_obs_level="patient"
        )
        assert _drive(var) == {"screened_obs_level": "patient"}

    def test_leaf_native_dynamic_accepts_and_stores_patient(self) -> None:
        """The leaf that actually queries the DB consumes the propagated scope."""
        from corr_vars.definitions.typing import ObsLevel

        var = NativeDynamic("leaf", dynamic=True, screened_obs_level="patient")
        assert var.screened_obs_level is ObsLevel.PATIENT

    def test_leaf_default_is_none_but_falls_back_to_hospital_stay(self) -> None:
        var = NativeDynamic("leaf", dynamic=True)
        assert var.screened_obs_level is None
