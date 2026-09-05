"""The variable classes local_datasource ships, exercised offline.

``NativeStatic``, ``DerivedStatic`` and ``DerivedDynamic`` compute from data
already in the cohort, so none of them needs a backing store — which is what
makes them testable here. ``NativeDynamic`` is the one that does need one, and
the published ``NativeExtractor`` deliberately raises rather than shipping a
query; that contract is checked too.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from conftest import EmptyCohort
from corr_vars.core import cohort as cohort_module
from corr_vars.sources.local_datasource import (
    ComplexVariable,
    DerivedDynamic,
    DerivedStatic,
    NativeDynamic,
    NativeExtractor,
    NativeStatic,
    VariableLoader,
)
from corr_vars.sources.var_loader import MultiSourceVariable
from corr_vars.utils.time import TimeWindow

ADMISSION = datetime(2024, 1, 1, 12, 0)


class _StubVariable:
    """A pre-extracted dependency, standing in for a loaded variable."""

    def __init__(self, var_name: str, data: pl.DataFrame, dynamic: bool = True) -> None:
        self.var_name = var_name
        self.dynamic = dynamic
        self.data = data
        self.time_window = TimeWindow()

    def extract(self, cohort: object) -> pl.DataFrame:
        return self.data


class _StubCohort(EmptyCohort):
    """Cohort whose variable loading is served from a dict of stubs."""

    @classmethod
    def build(cls, obs: pl.DataFrame, **stubs: _StubVariable) -> _StubCohort:
        """Construct a cohort over `obs`, resolving variables from `stubs`.

        ``EmptyCohort`` keeps ``_obs`` as a *class* attribute, so a cohort built
        by another test module would otherwise be inherited here and fail the
        change tracker's primary-key count. Setting it on this subclass first
        shadows whatever is on the parent.
        """
        cls._obs = obs
        cohort = cls(obs_level="icu_stay")
        cohort._obs = obs
        cohort._stubs = stubs
        return cohort

    def load_variable(self, variable, include_sources=None, overrides=None, **kwargs):
        name = variable[0] if isinstance(variable, tuple) else variable
        return self._stubs[name]


@pytest.fixture
def sodium_cohort() -> _StubCohort:
    """Three ICU stays with four sodium measurements between them."""
    obs = pl.DataFrame(
        {
            "icu_stay_id": ["1", "2", "3"],
            "icu_admission": [ADMISSION] * 3,
            "icu_discharge": [ADMISSION + timedelta(days=5)] * 3,
        }
    )
    sodium = pl.DataFrame(
        {
            "icu_stay_id": ["1", "1", "2", "3"],
            "recordtime": [
                ADMISSION + timedelta(hours=1),
                ADMISSION + timedelta(hours=6),
                ADMISSION + timedelta(hours=2),
                ADMISSION + timedelta(hours=3),
            ],
            "value": [140.0, 152.0, 133.0, 148.0],
        }
    )
    return _StubCohort.build(obs, blood_sodium=_StubVariable("blood_sodium", sodium))


class TestNativeStatic:
    """One dynamic variable in, one value per observation out."""

    def _extract(self, cohort: _StubCohort, **kwargs) -> pl.DataFrame:
        var = NativeStatic(base_var="blood_sodium", tmin="icu_admission", **kwargs)
        return var.extract(cohort)  # type: ignore[arg-type]

    def test_first_value(self, sodium_cohort: _StubCohort) -> None:
        data = self._extract(
            sodium_cohort, var_name="first_sodium", select="!first value"
        )
        assert dict(zip(data["icu_stay_id"], data["first_sodium"])) == {
            "1": 140.0,
            "2": 133.0,
            "3": 148.0,
        }

    def test_max_value(self, sodium_cohort: _StubCohort) -> None:
        data = self._extract(sodium_cohort, var_name="max_sodium", select="!max value")
        assert data.filter(pl.col("icu_stay_id").eq("1"))["max_sodium"].item() == 152.0

    def test_mean_value(self, sodium_cohort: _StubCohort) -> None:
        data = self._extract(
            sodium_cohort, var_name="mean_sodium", select="!mean value"
        )
        assert data.filter(pl.col("icu_stay_id").eq("1"))["mean_sodium"].item() == 146.0

    def test_count_value(self, sodium_cohort: _StubCohort) -> None:
        data = self._extract(
            sodium_cohort, var_name="sodium_count", select="!count value"
        )
        assert dict(zip(data["icu_stay_id"], data["sodium_count"])) == {
            "1": 2,
            "2": 1,
            "3": 1,
        }

    def test_first_recordtime_is_a_usable_time_anchor(
        self, sodium_cohort: _StubCohort
    ) -> None:
        data = self._extract(
            sodium_cohort, var_name="first_sodium_time", select="!first recordtime"
        )
        assert data["first_sodium_time"].dtype == pl.Datetime
        assert data.filter(pl.col("icu_stay_id").eq("1"))[
            "first_sodium_time"
        ].item() == ADMISSION + timedelta(hours=1)

    def test_where_filters_before_aggregating(self, sodium_cohort: _StubCohort) -> None:
        """Hypernatremia onset: the first measurement above the threshold."""
        data = self._extract(
            sodium_cohort,
            var_name="hypernatremia_onset",
            select="!first recordtime",
            where="value > 145",
        )
        by_id = dict(zip(data["icu_stay_id"], data["hypernatremia_onset"]))
        assert by_id["1"] == ADMISSION + timedelta(hours=6)  # not the 140.0 at +1h
        assert by_id["3"] == ADMISSION + timedelta(hours=3)
        # Stay 2 never crosses the threshold, so it contributes no row at all —
        # which is what makes this usable as a `set_t_eligible` anchor.
        assert "2" not in by_id

    def test_observations_without_measurements_are_absent(
        self, sodium_cohort: _StubCohort
    ) -> None:
        """An observation the base variable has no rows for contributes none."""
        sodium_cohort._obs = sodium_cohort._obs.vstack(
            pl.DataFrame(
                {
                    "icu_stay_id": ["4"],
                    "icu_admission": [ADMISSION],
                    "icu_discharge": [ADMISSION + timedelta(days=5)],
                }
            )
        )
        data = self._extract(
            sodium_cohort, var_name="first_sodium", select="!first value"
        )
        assert sorted(data["icu_stay_id"].to_list()) == ["1", "2", "3"]


class TestDerivedStatic:
    """A polars expression over columns already in ``cohort.obs``."""

    @pytest.fixture
    def bmi_cohort(self) -> _StubCohort:
        return _StubCohort.build(
            pl.DataFrame(
                {
                    "icu_stay_id": ["1", "2"],
                    "body_weight": [80.0, 60.0],
                    "body_height": [200.0, 150.0],
                }
            )
        )

    def test_expression_is_computed(self, bmi_cohort: _StubCohort) -> None:
        var = DerivedStatic(
            var_name="bmi",
            requires=[],
            expression="body_weight / (body_height / 100 * body_height / 100)",
        )
        data = var.extract(bmi_cohort)  # type: ignore[arg-type]
        assert data["bmi"].to_list() == [20.0, pytest.approx(26.666667)]

    def test_output_keeps_only_the_key_and_the_variable(
        self, bmi_cohort: _StubCohort
    ) -> None:
        var = DerivedStatic(
            var_name="is_tall", requires=[], expression="body_height >= 180"
        )
        data = var.extract(bmi_cohort)  # type: ignore[arg-type]
        assert data.columns == ["icu_stay_id", "is_tall"]
        assert data["is_tall"].to_list() == [True, False]

    def test_a_variable_needs_an_expression_or_a_py_function(
        self, bmi_cohort: _StubCohort
    ) -> None:
        var = DerivedStatic(var_name="nothing", requires=[])
        with pytest.raises(AssertionError, match="No expression or variable function"):
            var.extract(bmi_cohort)  # type: ignore[arg-type]

    def test_dynamic_is_rejected(self) -> None:
        with pytest.raises(AssertionError, match="cannot be dynamic"):
            DerivedStatic(var_name="x", dynamic=True)


class TestDerivedDynamic:
    """A time-series computed from other variables by a ``py`` function."""

    def test_py_function_output_is_the_variable_data(
        self, sodium_cohort: _StubCohort
    ) -> None:
        def double_sodium(var, cohort):
            return var.required_vars["blood_sodium"].data.with_columns(
                pl.col("value").mul(2)
            )

        var = DerivedDynamic(
            var_name="double_sodium",
            requires={
                "blood_sodium": {
                    "template": "blood_sodium",
                    "tmin": None,
                    "tmax": None,
                }
            },
            py=double_sodium,
            py_ready_polars=True,
        )
        data = var.extract(sodium_cohort)  # type: ignore[arg-type]
        assert data["value"].to_list() == [280.0, 304.0, 266.0, 296.0]

    def test_no_data_is_an_error(self, sodium_cohort: _StubCohort) -> None:
        var = DerivedDynamic(var_name="empty", requires=[])
        with pytest.raises(ValueError, match="resulted in no data"):
            var.extract(sodium_cohort)  # type: ignore[arg-type]


class TestNativeExtractorIsUnbound:
    """The store-backed path ships no query — that is the published contract."""

    def test_extract_raises_and_says_what_to_do(self) -> None:
        extractor = NativeExtractor("blood_sodium", table="labs")
        with pytest.raises(NotImplementedError) as excinfo:
            extractor.extract(None, "case_id")  # type: ignore[arg-type]
        message = str(excinfo.value)
        assert "blood_sodium" in message
        assert "Subclass NativeExtractor" in message

    def test_native_dynamic_builds_the_bound_extractor_class(self) -> None:
        class _BoundExtractor(NativeExtractor):
            def extract(self, cohort, id_column):
                return pl.DataFrame({"case_id": ["1"], "value": [1.0]})

        var = NativeDynamic("x", dynamic=True, table="labs")
        assert isinstance(var.extractor, NativeExtractor)
        assert type(var.extractor) is NativeExtractor

        try:
            NativeDynamic.extractor_class = _BoundExtractor
            bound = NativeDynamic("x", dynamic=True, table="labs")
            assert isinstance(bound.extractor, _BoundExtractor)
            assert bound.extract_from_db(None, "case_id")["value"].to_list() == [1.0]  # type: ignore[arg-type]
        finally:
            NativeDynamic.extractor_class = NativeExtractor


class TestVariableLoader:
    """The factory the config's ``type`` selects a class with."""

    @pytest.mark.parametrize(
        "var_type,expected,kwargs",
        [
            ("native_dynamic", NativeDynamic, {"dynamic": False}),
            (
                "native_static",
                NativeStatic,
                {"select": "!first value", "base_var": "dep"},
            ),
            ("derived_static", DerivedStatic, {}),
            ("derived_dynamic", DerivedDynamic, {"requires": []}),
            ("complex", ComplexVariable, {"dynamic": False}),
        ],
    )
    def test_type_selects_the_class(
        self, var_type: str, expected: type, kwargs: dict
    ) -> None:
        var = VariableLoader(
            var_name="x", type=var_type, time_window=TimeWindow(), **kwargs
        )
        assert isinstance(var, expected)

    def test_dynamic_types_force_the_dynamic_flag(self) -> None:
        # `dynamic=False` in the config loses to the type, which restates it.
        var = VariableLoader(
            var_name="x",
            type="native_dynamic",
            time_window=TimeWindow(),
            dynamic=False,
        )
        assert var.dynamic is True

    def test_unknown_type_is_rejected(self) -> None:
        with pytest.raises(KeyError, match="not_a_type"):
            VariableLoader(
                var_name="x",
                type="not_a_type",
                time_window=TimeWindow(),
                dynamic=False,
            )


class _ForeignCohort(EmptyCohort):
    """Cohort whose own ``_obs`` never touches ``EmptyCohort``'s class attribute."""


class TestHandBuiltVariablesOnForeignCohort:
    """A hand-built aggregation on a cohort configured with another source.

    ``NativeStatic`` and friends are tagged ``local_datasource`` by
    :func:`guess_variable_source`, but the cohort they run on may be built from
    entirely different sources. ``Variable._dependency_sources`` is what keeps
    their dependency lookups from being restricted to a source the cohort does
    not have, so these go through the real
    ``Cohort.add_variable`` -> ``Cohort.load_variable`` path rather than the
    stubbed ``load_variable`` of ``_StubCohort``.
    """

    @pytest.fixture
    def obs(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "icu_stay_id": ["1", "2"],
                "icu_admission": [ADMISSION] * 2,
                "icu_discharge": [ADMISSION + timedelta(days=5)] * 2,
            }
        )

    @pytest.fixture
    def reprodicu_cohort(self, obs: pl.DataFrame) -> _ForeignCohort:
        """A cohort over `obs` that claims a single, foreign source."""
        # `_obs` is a class attribute on EmptyCohort. Building through the
        # subclass keeps this frame off the parent, where other test modules
        # would inherit it (see `_StubCohort.build`).
        cohort = _ForeignCohort.with_obs(obs, obs_level="icu_stay")
        cohort._obs = obs
        # EmptyCohort loads no defaults, so neither of these is set for it.
        cohort.constant_vars = []
        cohort.sources = {"reprodicu": {}}
        return cohort

    def _patch_loader(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stub: _StubVariable,
    ) -> list[tuple[str, object]]:
        """Serve every dependency from `stub`, recording the sources asked for."""
        calls: list[tuple[str, object]] = []

        def fake_load_variable(
            var_name, cohort, time_window, include_sources=None, **kwargs
        ):
            calls.append((var_name, include_sources))
            return MultiSourceVariable({"reprodicu": stub})

        monkeypatch.setattr(
            cohort_module.loader.var_loader, "load_variable", fake_load_variable
        )
        return calls

    def test_native_static_resolves_its_base_var_across_all_sources(
        self, reprodicu_cohort: _ForeignCohort, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sodium = pl.DataFrame(
            {
                "icu_stay_id": ["1", "1", "2"],
                "recordtime": [
                    ADMISSION + timedelta(hours=1),
                    ADMISSION + timedelta(hours=6),
                    ADMISSION + timedelta(hours=2),
                ],
                "value": [140.0, 152.0, 133.0],
            }
        )
        calls = self._patch_loader(
            monkeypatch, _StubVariable("blood_sodium", sodium, dynamic=True)
        )

        var = reprodicu_cohort.add_variable(
            NativeStatic(
                var_name="max_sodium",
                select="!max value",
                base_var="blood_sodium",
                tmin="icu_admission",
                tmax="icu_discharge",
            )
        )

        assert list(var.variables) == ["local_datasource"]
        assert dict(
            zip(
                reprodicu_cohort.obs["icu_stay_id"],
                reprodicu_cohort.obs["max_sodium"],
            )
        ) == {"1": 152.0, "2": 133.0}
        # The lookup must not be pinned to ["local_datasource"], which the
        # cohort does not have configured.
        assert calls == [("blood_sodium", None)]

    def test_derived_static_resolves_its_requirements_across_all_sources(
        self, reprodicu_cohort: _ForeignCohort, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        max_sodium = pl.DataFrame(
            {"icu_stay_id": ["1", "2"], "max_sodium": [152.0, 133.0]}
        )
        calls = self._patch_loader(
            monkeypatch, _StubVariable("max_sodium", max_sodium, dynamic=False)
        )

        var = reprodicu_cohort.add_variable(
            DerivedStatic(
                var_name="high_sodium",
                requires=["max_sodium"],
                expression="max_sodium > 145",
            )
        )

        assert list(var.variables) == ["local_datasource"]
        assert dict(
            zip(
                reprodicu_cohort.obs["icu_stay_id"],
                reprodicu_cohort.obs["high_sodium"],
            )
        ) == {"1": True, "2": False}
        assert calls == [("max_sodium", None)]


class TestDependencySources:
    """``Variable._dependency_sources`` in isolation."""

    @pytest.fixture
    def variable(self) -> NativeStatic:
        return NativeStatic(
            var_name="max_sodium", select="!max value", base_var="blood_sodium"
        )

    def test_the_variables_own_source_pins_the_lookup(
        self, variable: NativeStatic
    ) -> None:
        cohort = _ForeignCohort(obs_level="icu_stay")
        cohort.sources = {"local_datasource": {}}
        assert variable._dependency_sources(cohort) == ["local_datasource"]

    def test_a_foreign_cohort_leaves_the_lookup_open(
        self, variable: NativeStatic
    ) -> None:
        cohort = _ForeignCohort(obs_level="icu_stay")
        cohort.sources = {"reprodicu": {}}
        assert variable._dependency_sources(cohort) is None

    @pytest.mark.parametrize("sources", [None, {}])
    def test_a_cohort_without_sources_leaves_the_lookup_open(
        self, variable: NativeStatic, sources: dict | None
    ) -> None:
        cohort = _ForeignCohort(obs_level="icu_stay")
        cohort.sources = sources  # type: ignore[assignment]
        assert variable._dependency_sources(cohort) is None
