from __future__ import annotations

import polars as pl
import pytest
from hypothesis_cohort import obs_dataframe
from hypothesis_var import var_dataframe

import corr_vars.core.cohort as cohort_module
from corr_vars.concepts.client import ConceptsApiClient
from corr_vars.core.cohort import Cohort
from corr_vars.core.variable import Variable
from corr_vars.sources.var_loader import MultiSourceVariable

from typing import TYPE_CHECKING, Any, Literal, TypedDict

if TYPE_CHECKING:
    from corr_vars.concepts.spec import VariableSpec
    from corr_vars.definitions import ObsLevel
    from corr_vars.utils.time import TimeWindow

    from collections.abc import Callable, Iterable
    from corr_vars.definitions.typing import ObsLevelName, SourceDict


def _dummy_load_cohort_data(sources: SourceDict, obs_level: ObsLevel):
    return obs_dataframe(obs_level=obs_level).example()


class EmptyCohort(Cohort):
    """Dummy subclass for testing Cohort logic without static data (obs) loading.

    This class should only be used for testing functionality, which doesn't require any variable data.
    Obs data can be easily overwritten by assignment to cohort._obs or cohort.obs.

    Please note that the changetracker is disabled for this cohort if obs is empty.

    Use the dummy_cohort fixture to load actual dummy data.
    """

    def __init__(
        self,
        obs_level: ObsLevelName = "icu_stay",
    ) -> None:
        return super().__init__(
            obs_level=obs_level,
            sources={},
            project_vars={},
            load_default_vars=False,
            logger_args={},
        )

    @classmethod
    def with_obs(
        cls, obs: pl.DataFrame, obs_level: ObsLevelName = "hospital_stay"
    ) -> EmptyCohort:
        cls._obs = obs
        return cls(obs_level=obs_level)

    def _load_obs_level_data(self):
        if not hasattr(self, "_obs"):
            self._obs = pl.DataFrame()

    def _setup_change_tracker(self):
        if not self._obs.is_empty():
            super()._setup_change_tracker()


class DummyVariable(Variable):
    """Used internally in the dummy_cohort fixture to add dummy Variables loading dummy data."""

    dynamic: bool
    df_type: Literal["dynamic_ts_interval", "dynamic_ts_snapshot", "static"]

    def __init__(
        self,
        var_name: str,
        df_type: Literal[
            "dynamic_ts_interval", "dynamic_ts_snapshot", "static"
        ] = "static",
        # id_overlap: float = 0.8,
    ) -> None:
        dynamic = df_type != "static"
        self.df_type = df_type

        # self.id_overlap = max(0, min(1, id_overlap))
        # (
        #     cohort.obs[cohort.primary_key]
        #     .unique()
        #     .sample(
        #         fraction=self.id_overlap,
        #         with_replacement=False,
        #         seed=42,
        #         shuffle=False,
        #     )
        # )

        super().__init__(
            var_name=var_name,
            dynamic=dynamic,
            requires=[],
            tmin="X",
            tmax="Y",
            py=None,
            py_ready_polars=False,
            cleaning=None,
        )

    def extract(self, cohort: Cohort) -> pl.DataFrame:
        self.data = var_dataframe(
            vtype=self.df_type, obs_level=cohort.obs_level
        ).example()

        if self.dynamic:
            # Expects case_tmin, case_tmax for each primary key
            self._timefilter()
            # self._apply_cleaning()
        else:
            self.data = self.data.rename({"value": self.var_name})

        return self.data


def _dummy_load_variable(
    var_name: str,
    cohort: Cohort,
    time_window: TimeWindow,
    include_sources: Iterable[str] | None,
    overrides: dict[str, Any] | None = None,
    spec: VariableSpec | None = None,
) -> MultiSourceVariable:
    # df_type can be inferred from var name pattern for tests
    dynamic = "dynamic" in var_name
    interval = "interval" in var_name
    df_type = (
        ("dynamic_ts_interval" if interval else "dynamic_ts_snapshot")
        if dynamic
        else "static"
    )
    return MultiSourceVariable(
        {"dummy": DummyVariable(var_name=var_name, df_type=df_type)}
    )


@pytest.fixture(
    params=[
        ("patient",),
        ("hospital_stay",),
        ("icu_stay",),
        ("procedure",),
    ],
    ids=lambda param: f"obs={param[0]}",
)
def dummy_cohort(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Cohort:
    """Fixture that monkeypatches the cohort loader used by Cohort._load_obs_level_data
    and the variable loader used by Cohort._load_variable.

    This allows testing the real Cohort._load_obs_level_data behavior (e.g. constant_vars)
    without relying on external data sources.
    """
    obs_level = request.param[0]

    # Patch the loader that Cohort imports
    monkeypatch.setattr(
        cohort_module.loader.cohort_loader,
        "load_cohort_data",
        _dummy_load_cohort_data,
    )

    monkeypatch.setattr(
        cohort_module.loader.var_loader,
        "load_variable",
        _dummy_load_variable,
    )

    monkeypatch.setattr(
        cohort_module.loader.var_loader,
        "load_default_variables",
        lambda _cohort, _tmin, _tmax: [],
    )

    monkeypatch.setattr(
        cohort_module.loader.config_loader,
        "load_default_config_data",
        lambda sources: sources,
    )

    # "dummy" is not pinned local, so it matches the wildcard endpoint and the
    # cohort verifies its Concepts API credentials on construction. The loaders
    # above are patched out, and so is the authentication handshake: there is no
    # endpoint behind this made-up source, and an offline test must never reach
    # the real one.
    monkeypatch.setattr(ConceptsApiClient, "verify_access", lambda self: None)

    return Cohort(
        sources={"dummy": {}},
        obs_level=obs_level,
        project_vars={},
        load_default_vars=False,
        logger_args={},
        project="test_project",
        api_key="test_key",
    )


class SpyDict(TypedDict):
    count: int
    args_list: list[tuple[Any]]
    kwargs_list: list[dict[str, Any]]
    last_args: tuple
    last_kwargs: dict[str, Any]


class Spy:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.monkeypatch = monkeypatch

    def __call__(
        self,
        target: str | object,
        *,
        return_value: Any | None = None,
        side_effect: Callable[..., None] | None = None,
        wrap: Callable[..., Any] | None = None,
    ) -> SpyDict:
        called: SpyDict = {
            "count": 0,
            "args_list": [],
            "kwargs_list": [],
            "last_args": (),
            "last_kwargs": {},
        }

        def _recorder(*args, **kwargs):
            called["count"] += 1
            called["args_list"].append(args)
            called["kwargs_list"].append(kwargs)
            called["last_args"] = args
            called["last_kwargs"] = kwargs
            if side_effect is not None:
                side_effect(*args, **kwargs)
            if wrap is not None:
                return wrap(*args, **kwargs)
            return return_value

        self.monkeypatch.setattr(target, _recorder)  # type: ignore
        return called


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> Spy:
    """Factory to monkeypatch a target with a recorder wrapper.

    Usage:
      called, fake = spy("module.attr")                       # default: fake returns None
      called, fake = spy("module.attr", return_value=42)      # fake returns 42
      called, fake = spy("module.attr", side_effect=fn)       # fake calls fn(...), returns None
      called, fake = spy("module.attr", wrap=real_fn)         # fake records and calls real_fn(...)

    Returned `called` dict contains:
      - count: number of calls
      - args_list: list of positional-args tuples
      - kwargs_list: list of kwargs dicts
      - last_args, last_kwargs: most recent call
    """
    return Spy(monkeypatch=monkeypatch)
