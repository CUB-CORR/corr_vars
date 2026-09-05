from __future__ import annotations

from datetime import date

import httpx
import polars as pl
import pytest

from corr_vars.concepts.client import API_KEY_ENV_VAR, ConceptsApiClient
from corr_vars.concepts.resolver import ConceptResolver
from corr_vars.concepts.routing import ConceptRouting, Endpoint
from corr_vars.concepts.spec import VersionSelector
from corr_vars.core.cohort import Cohort
from corr_vars.definitions.exceptions import (
    AmbiguousConceptError,
    CohortDataError,
    ConceptsApiConfigurationError,
    ConceptsApiError,
    ConceptsLicenseError,
    ProjectNotFoundError,
    VariableNotFoundError,
)
from corr_vars.sources import var_loader

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

ENDPOINT = "https://concepts.test/api"

#: Every source is remote here, so a cohort exercises the API path end to end.
REMOTE_ROUTING = ConceptRouting(
    taxonomy="corr_v1",
    version="latest",
    local_sources=(),
    endpoints=(Endpoint(url=ENDPOINT, sources=("*",)),),
)

BLOOD_SODIUM_JSON = {
    "type": "native_dynamic",
    "table": "labs",
    "where": "name = 'sodium'",
    "compatible_with": ["icu_stay", "hospital_stay", "patient", "procedure"],
    "tmin": ["icu_admission", "-2h"],
    "tmax": ["icu_admission", "+48h"],
}


#: The bundled source that is routed local. Its ``vars.json`` ships empty — it is
#: a source skeleton — so the ``local_source`` fixture fills it in for the length
#: of a test to exercise the local half of the routing seam.
LOCAL_SOURCE = "local_datasource"

LOCAL_VARS = {
    "corr_defaults": {"default_vars": {"global": ["age"], "icu_stay": ["los_icu"]}},
    "variables": {
        "blood_sodium": {
            "type": "native_dynamic",
            "table": "labs",
            "compatible_with": ["icu_stay", "hospital_stay", "patient", "procedure"],
        }
    },
}


@pytest.fixture
def local_source(monkeypatch: pytest.MonkeyPatch) -> str:
    """Give ``local_datasource`` a non-empty ``vars.json`` for one test."""
    from corr_vars.sources.local_datasource import mapping

    monkeypatch.setattr(mapping, "VARS", LOCAL_VARS)
    monkeypatch.setattr(
        mapping, "DEFAULT_VARS", LOCAL_VARS["corr_defaults"]["default_vars"]
    )
    return LOCAL_SOURCE


def element(
    *,
    json_config: dict[str, Any] | None = None,
    py: str | None = None,
    version: int = 4,
    warning: dict[str, Any] | None = None,
    source: str = "reprodicu",
    name: str = "blood_sodium",
    concept_id: int = 1,
    deprecated_at: str | None = None,
    pointer_deprecated_at: str | None = None,
) -> dict[str, Any]:
    return {
        "id": concept_id,
        "taxonomy": "corr_v1",
        "name": name,
        "version": version,
        "requested": {},
        "sources": {
            source: {
                "json": (
                    json_config if json_config is not None else dict(BLOOD_SODIUM_JSON)
                ),
                "py": py,
                "files": [],
                "version_info": {
                    "source_version": version,
                    "status": "committed",
                    "committed_at": "2024-05-06T10:00:00Z",
                    "warning": warning,
                },
            }
        },
        "pointer": {
            "id": 100 + concept_id,
            "identifier": name,
            "display_name": name,
            "relationship": "primary",
            "created_at": "2024-01-01T00:00:00Z",
            "deprecated_at": pointer_deprecated_at,
        },
        "deprecated_at": deprecated_at,
        "successor_id": None,
    }


def document(**kwargs: Any) -> list[dict[str, Any]]:
    return [element(**kwargs)]


def group(*elements: dict[str, Any]) -> list[dict[str, Any]]:
    return list(elements)


def ambiguous_response() -> httpx.Response:
    return httpx.Response(
        400,
        json={
            "detail": {
                "error": "ambiguous_name",
                "members": [
                    {"id": 12, "name": "atc_c07ab", "display_name": "Metoprolol"},
                    {"id": 13, "name": "atc_c07ab", "display_name": "Bisoprolol"},
                ],
            }
        },
    )


def records_by_source(cohort: Any, var_name: str) -> dict[str, dict[str, Any]]:
    """Index a variable's provenance records by contributor source."""
    return {record["source"]: record for record in cohort.concept_versions[var_name]}


def make_resolver(
    handler: Callable[[httpx.Request], httpx.Response], **kwargs: Any
) -> ConceptResolver:
    def factory(url: str) -> ConceptsApiClient:
        return ConceptsApiClient(
            url,
            project=kwargs.get("project") or "demo",
            api_key="k",
            transport=httpx.MockTransport(handler),
            backoff=0.0,
        )

    kwargs.setdefault("project", "demo")
    return ConceptResolver(routing=REMOTE_ROUTING, client_factory=factory, **kwargs)


class StubCohort:
    """Minimal stand-in exposing what ``_load_variable_configs`` touches."""

    def __init__(self, resolver: ConceptResolver | None, **project_vars: Any) -> None:
        self.concepts = resolver
        self.project_vars = dict(project_vars)
        self.sources = {"reprodicu": {}}
        self.concept_versions: dict[str, list[dict[str, Any]]] = {}

    def get_variable_definition(self, var_name: str) -> dict[str, dict[str, Any]]:
        if var_name in self.project_vars:
            return dict.fromkeys(self.sources, self.project_vars[var_name])
        return {}

    def _record_concept_versions(self, var_name: str, resolutions: Any) -> None:
        self.concept_versions[var_name] = [
            resolution.as_dict() for resolution in resolutions.values()
        ]


# ---------------------------------------------------------------------------
# The routing seam in var_loader
# ---------------------------------------------------------------------------


class TestCollectVariableConfigs:
    def test_without_a_resolver_everything_is_local(self, local_source: str) -> None:
        configs = var_loader._collect_variable_configs_by_source(
            "blood_sodium", [local_source]
        )
        assert local_source in configs
        assert configs[local_source]["type"]

    def test_local_behaviour_is_preserved_for_locally_routed_sources(
        self, local_source: str
    ) -> None:
        local_routing = ConceptRouting(
            taxonomy="corr_v1",
            version="latest",
            local_sources=(local_source,),
            endpoints=(),
        )
        resolver = ConceptResolver(routing=local_routing, project="demo")
        with_resolver = var_loader._collect_variable_configs_by_source(
            "blood_sodium", [local_source], resolver=resolver
        )
        without = var_loader._collect_variable_configs_by_source(
            "blood_sodium", [local_source]
        )
        assert with_resolver == without

    def test_routed_source_comes_from_the_api(self) -> None:
        resolver = make_resolver(lambda _: httpx.Response(200, json=document()))
        configs = var_loader._collect_variable_configs_by_source(
            "blood_sodium", ["reprodicu"], resolver=resolver
        )
        assert configs["reprodicu"]["table"] == "labs"

    def test_tmin_tmax_are_coerced_back_to_tuples(self) -> None:
        resolver = make_resolver(lambda _: httpx.Response(200, json=document()))
        config = var_loader._collect_variable_configs_by_source(
            "blood_sodium", ["reprodicu"], resolver=resolver
        )["reprodicu"]
        assert config["tmin"] == ("icu_admission", "-2h")
        assert config["tmax"] == ("icu_admission", "+48h")

    def test_routed_source_never_falls_back_to_local(self) -> None:
        resolver = make_resolver(lambda _: httpx.Response(500))
        with pytest.raises(ConceptsApiError):
            var_loader._collect_variable_configs_by_source(
                "blood_sodium", ["reprodicu"], resolver=resolver
            )

    def test_unpublished_concept_yields_nothing_not_the_local_entry(self) -> None:
        resolver = make_resolver(lambda _: httpx.Response(404, json={}))
        configs = var_loader._collect_variable_configs_by_source(
            "blood_sodium", ["reprodicu"], resolver=resolver
        )
        assert configs == {}

    def test_resolutions_are_recorded_for_api_sources(self) -> None:
        resolver = make_resolver(lambda _: httpx.Response(200, json=document()))
        resolutions: dict[str, Any] = {}
        var_loader._collect_variable_configs_by_source(
            "blood_sodium",
            ["reprodicu"],
            resolver=resolver,
            resolutions=resolutions,
        )
        assert resolutions["reprodicu"].origin == "api"
        assert resolutions["reprodicu"].version == 4

    def test_resolutions_are_recorded_for_local_sources(
        self, local_source: str
    ) -> None:
        resolutions: dict[str, Any] = {}
        var_loader._collect_variable_configs_by_source(
            "blood_sodium", [local_source], resolutions=resolutions
        )
        assert resolutions[local_source].origin == "local"


class TestProjectOverrides:
    def test_overrides_layer_on_top_of_api_configs(self) -> None:
        resolver = make_resolver(lambda _: httpx.Response(200, json=document()))
        cohort = StubCohort(resolver, blood_sodium={"where": "name = 'na'"})
        configs = var_loader._load_variable_configs(
            "blood_sodium",
            cohort,
            ["reprodicu"],  # type: ignore[arg-type]
        )
        assert configs["reprodicu"]["where"] == "name = 'na'"
        assert configs["reprodicu"]["table"] == "labs"

    def test_overrides_layer_on_top_of_local_configs(self, local_source: str) -> None:
        cohort = StubCohort(None, blood_sodium={"where": "name = 'na'"})
        cohort.sources = {local_source: {}}
        configs = var_loader._load_variable_configs(
            "blood_sodium",
            cohort,
            [local_source],  # type: ignore[arg-type]
        )
        assert configs[local_source]["where"] == "name = 'na'"

    def test_versions_are_recorded_on_the_cohort(self) -> None:
        resolver = make_resolver(lambda _: httpx.Response(200, json=document()))
        cohort = StubCohort(resolver)
        var_loader._load_variable_configs(
            "blood_sodium",
            cohort,
            ["reprodicu"],  # type: ignore[arg-type]
        )
        record = records_by_source(cohort, "blood_sodium")["reprodicu"]
        assert record["version"] == 4
        assert record["concept_id"] == 1


# ---------------------------------------------------------------------------
# Cohort
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_sources(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Source config for a cohort whose obs frame is generated, not read.

    reprodicu reads parquet files from disk, which an offline test has none of,
    so the cohort loader is patched to the same synthetic obs frame the
    ``dummy_cohort`` fixture uses. Everything downstream of the obs load — the
    resolver, ``var_loader``, extraction — stays real.
    """
    import corr_vars.core.cohort as cohort_module
    from conftest import _dummy_load_cohort_data
    from corr_vars.concepts import resolver as resolver_module

    monkeypatch.setattr(
        cohort_module.loader.cohort_loader,
        "load_cohort_data",
        _dummy_load_cohort_data,
    )
    # Pinned local, so the cohort needs neither a project nor a key to be built
    # and no test here reaches an endpoint by accident. Tests that want the API
    # path replace `cohort._concepts` with a resolver over `REMOTE_ROUTING`.
    monkeypatch.setattr(
        resolver_module,
        "load_routing",
        lambda: ConceptRouting(
            taxonomy="corr_v1",
            version="latest",
            local_sources=("reprodicu",),
            endpoints=(
                Endpoint(url="https://concepts.example.edu/api", sources=("*",)),
            ),
        ),
    )
    return {"reprodicu": {}}


def local_cohort(dummy_sources: dict[str, Any], **kwargs: Any) -> Cohort:
    return Cohort(
        obs_level="icu_stay",
        sources=dummy_sources,
        load_default_vars=False,
        **kwargs,
    )


class TestCohortConceptSettings:
    def test_defaults_come_from_the_packaged_routing_file(
        self, dummy_sources: dict[str, Any]
    ) -> None:
        cohort = local_cohort(dummy_sources)
        assert cohort.concepts.default_taxonomy == "corr_v1"
        assert str(cohort.concepts.default_version) == "latest"

    def test_kwargs_override_the_file(self, dummy_sources: dict[str, Any]) -> None:
        cohort = local_cohort(dummy_sources, taxonomy="corr_v2", version="v5")
        assert cohort.concepts.default_taxonomy == "corr_v2"
        assert cohort.concepts.default_version == VersionSelector("version", 5)

    def test_global_as_of_date(self, dummy_sources: dict[str, Any]) -> None:
        cohort = local_cohort(dummy_sources, date="2025-06-30")
        assert cohort.concepts.default_version == VersionSelector(
            "date", date(2025, 6, 30)
        )

    def test_version_and_date_together_are_rejected(
        self, dummy_sources: dict[str, Any]
    ) -> None:
        with pytest.raises(ValueError, match="only one"):
            local_cohort(dummy_sources, version="v3", date="2025-06-30")

    def test_api_url_override(self, dummy_sources: dict[str, Any]) -> None:
        cohort = local_cohort(dummy_sources, concepts_api_url="https://staging/api")
        assert cohort.concepts.route("demo_source").url == "https://staging/api"


class TestCredentialVerificationOnConstruction:
    """A cohort checks its Concepts API credentials before it loads any data."""

    def test_locally_routed_sources_need_no_credentials(
        self, dummy_sources: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CORR_CONCEPTS_API_KEY", raising=False)
        assert local_cohort(dummy_sources).concepts.project is None

    def test_missing_project_is_reported_before_any_data_is_loaded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CORR_CONCEPTS_API_KEY", "k")
        # reprodicu reads parquet from disk, so reaching the data load at all
        # would fail the test with a file error instead.
        with pytest.raises(ConceptsApiConfigurationError, match="project"):
            Cohort(
                obs_level="icu_stay",
                sources={"reprodicu": {}},
                load_default_vars=False,
            )

    def test_missing_api_key_is_reported_before_any_data_is_loaded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CORR_CONCEPTS_API_KEY", raising=False)
        with pytest.raises(ConceptsApiConfigurationError, match=API_KEY_ENV_VAR):
            Cohort(
                obs_level="icu_stay",
                sources={"reprodicu": {}},
                load_default_vars=False,
                project="demo",
            )

    def test_the_error_names_the_sources_that_need_credentials(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CORR_CONCEPTS_API_KEY", raising=False)
        resolver = ConceptResolver(routing=REMOTE_ROUTING, project="demo")
        with pytest.raises(
            ConceptsApiConfigurationError, match="demo_source, reprodicu"
        ):
            resolver.verify_credentials(["demo_source", "reprodicu"])

    def test_a_blank_project_is_not_a_project(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CORR_CONCEPTS_API_KEY", "k")
        resolver = ConceptResolver(routing=REMOTE_ROUTING, project="   ")
        with pytest.raises(ConceptsApiConfigurationError, match="project"):
            resolver.verify_credentials(["demo_source"])

    def test_credentials_from_the_environment_satisfy_the_check(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CORR_CONCEPTS_API_KEY", "  k  ")
        resolver = ConceptResolver(routing=REMOTE_ROUTING, project=" demo ")
        resolver.verify_credentials(["demo_source"])
        assert resolver.client_for(resolver.route("demo_source")).project == "demo"

    def test_the_check_makes_no_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CORR_CONCEPTS_API_KEY", "k")

        def refuse(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("verify_credentials must not touch the network")

        monkeypatch.setattr(httpx.Client, "get", refuse)
        ConceptResolver(routing=REMOTE_ROUTING, project="demo").verify_credentials(
            ["demo_source"]
        )


class TestAuthenticationOnConstruction:
    """The credentials are proven against the endpoint before any data loads."""

    @staticmethod
    def serve(*rows: dict[str, Any]) -> Callable[[httpx.Request], httpx.Response]:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path.endswith("/projects"), (
                f"authentication must not read anything but the project "
                f"listing, but asked for {request.url.path}"
            )
            return httpx.Response(200, json=list(rows))

        return handler

    def test_a_listed_project_authenticates(self) -> None:
        resolver = make_resolver(self.serve({"name": "demo", "license_ok": True}))
        resolver.authenticate(["reprodicu"])

    def test_an_unknown_project_is_refused(self) -> None:
        resolver = make_resolver(self.serve({"name": "other"}))
        with pytest.raises(ProjectNotFoundError, match="'demo'"):
            resolver.authenticate(["reprodicu"])

    def test_an_unlicensed_project_is_refused(self) -> None:
        resolver = make_resolver(self.serve({"name": "demo", "license_ok": False}))
        with pytest.raises(ConceptsLicenseError):
            resolver.authenticate(["reprodicu"])

    def test_locally_routed_sources_touch_no_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            httpx.Client,
            "get",
            lambda *a, **k: pytest.fail("a local source must not authenticate"),
        )
        local_routing = ConceptRouting(
            taxonomy="corr_v1",
            version="latest",
            local_sources=("reprodicu",),
            endpoints=(Endpoint(url=ENDPOINT, sources=("*",)),),
        )
        ConceptResolver(
            routing=local_routing, project="demo", api_key="k"
        ).authenticate(["reprodicu"])

    def test_one_request_per_endpoint(self) -> None:
        calls = {"n": 0}

        def handler(_: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(200, json=[{"name": "demo"}])

        make_resolver(handler).authenticate(
            ["demo_source", "other_source", "reprodicu"]
        )
        assert calls["n"] == 1

    def test_a_missing_key_is_reported_without_a_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)
        monkeypatch.setattr(
            httpx.Client,
            "get",
            lambda *a, **k: pytest.fail("no key means no request to make"),
        )
        resolver = ConceptResolver(routing=REMOTE_ROUTING, project="demo")
        with pytest.raises(ConceptsApiConfigurationError, match=API_KEY_ENV_VAR):
            resolver.authenticate(["demo_source"])

    def test_an_unknown_project_is_reported_before_any_data_is_loaded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # reprodicu reads parquet from disk, so reaching the data load at all
        # would fail with a file error instead.
        def serve(*args: Any, **kwargs: Any) -> httpx.Response:
            return httpx.Response(200, json=[{"name": "other", "license_ok": True}])

        monkeypatch.setattr(httpx.Client, "get", serve)
        with pytest.raises(ProjectNotFoundError, match="'demo'"):
            Cohort(
                obs_level="icu_stay",
                sources={"reprodicu": {}},
                load_default_vars=False,
                project="demo",
                api_key="k",
            )

    def test_a_rejected_key_is_reported_before_any_data_is_loaded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def refuse(*args: Any, **kwargs: Any) -> httpx.Response:
            return httpx.Response(401, json={"detail": "invalid token"})

        monkeypatch.setattr(httpx.Client, "get", refuse)
        with pytest.raises(ConceptsApiError, match="rejected the API key"):
            Cohort(
                obs_level="icu_stay",
                sources={"reprodicu": {}},
                load_default_vars=False,
                project="demo",
                api_key="k",
            )


#: An extractable ``blood_sodium``: a complex variable whose served ``py``
#: snippet builds its own frame, so no source has to read a database or a
#: parquet file for it. :data:`BLOOD_SODIUM_JSON` describes the same concept
#: declaratively, for the tests that only look at the config.
BLOOD_SODIUM_COMPLEX_JSON = {
    "type": "complex",
    "complex": True,
    "dynamic": True,
    "py_ready_polars": True,
    "compatible_with": ["icu_stay", "hospital_stay", "patient", "procedure"],
}

BLOOD_SODIUM_PY = (
    "def blood_sodium(var, cohort):\n"
    "    return cohort.obs.select(\n"
    "        cohort.primary_key,\n"
    "        pl.lit(0).alias('recordtime_relative'),\n"
    "        pl.lit(140.0).alias('value'),\n"
    "    )\n"
)


def serve_blood_sodium(cohort: Cohort, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point `cohort` at a mock endpoint serving an extractable ``blood_sodium``."""
    served = document(json_config=dict(BLOOD_SODIUM_COMPLEX_JSON), py=BLOOD_SODIUM_PY)
    monkeypatch.setattr(
        cohort,
        "_concepts",
        make_resolver(lambda _: httpx.Response(200, json=served)),
    )


@pytest.mark.usefixtures("dummy_py_compiler")
class TestAddVariableSpec:
    def test_bare_name(
        self, dummy_sources: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cohort = local_cohort(dummy_sources)
        serve_blood_sodium(cohort, monkeypatch)
        assert cohort.add_variable("blood_sodium").var_name == "blood_sodium"

    def test_version_suffix_is_stripped_from_the_saved_name(
        self, dummy_sources: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cohort = local_cohort(dummy_sources)
        serve_blood_sodium(cohort, monkeypatch)
        var = cohort.add_variable("blood_sodium::latest")
        assert var.var_name == "blood_sodium"
        assert "blood_sodium" in cohort.obsm
        assert "blood_sodium::latest" not in cohort.obsm

    def test_taxonomy_prefix_is_stripped(
        self, dummy_sources: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cohort = local_cohort(dummy_sources)
        serve_blood_sodium(cohort, monkeypatch)
        assert cohort.add_variable("corr_v1/blood_sodium").var_name == "blood_sodium"

    def test_malformed_spec_is_rejected(self, dummy_sources: dict[str, Any]) -> None:
        cohort = local_cohort(dummy_sources)
        with pytest.raises(ValueError, match="Cannot parse version"):
            cohort.add_variable("blood_sodium::newest")

    def test_spec_reaches_the_loader(
        self, dummy_sources: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cohort = local_cohort(dummy_sources)
        serve_blood_sodium(cohort, monkeypatch)
        seen: list[Any] = []
        original = var_loader.load_variable

        def spy(*args: Any, **kwargs: Any):
            seen.append(kwargs.get("spec"))
            return original(*args, **kwargs)

        monkeypatch.setattr(var_loader, "load_variable", spy)
        cohort.add_variable("blood_sodium::v3")

        assert seen[0] is not None
        assert seen[0].version == VersionSelector("version", 3)
        assert seen[0].name == "blood_sodium"


class TestRequiresResolveAtCohortDefault:
    def test_dependency_spec_is_the_cohort_default(self) -> None:
        resolver = make_resolver(
            lambda _: httpx.Response(200, json=document()), date="2025-06-30"
        )
        # A pinned parent does not pin its dependencies: requires arrive as bare
        # names and therefore take the cohort-wide default selector.
        assert resolver.parse("parent::v3").version == VersionSelector("version", 3)
        assert resolver.default_spec("dependency").version == VersionSelector(
            "date", date(2025, 6, 30)
        )

    def test_dependencies_hit_the_cache_instead_of_refetching(self) -> None:
        calls = {"n": 0}

        def handler(_: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(200, json=document())

        resolver = make_resolver(handler)
        for _ in range(4):
            var_loader._collect_variable_configs_by_source(
                "blood_sodium", ["reprodicu"], resolver=resolver
            )
        assert calls["n"] == 1


class TestVariableNotFoundMessage:
    def test_message_names_the_endpoint(self) -> None:
        resolver = make_resolver(lambda _: httpx.Response(404, json={}))
        cohort = StubCohort(resolver)
        with pytest.raises(VariableNotFoundError, match=ENDPOINT):
            var_loader.load_variable(
                var_name="not_published",
                cohort=cohort,  # type: ignore[arg-type]
                time_window=None,  # type: ignore[arg-type]
                include_sources=["reprodicu"],
                spec=resolver.parse("not_published"),
            )

    def test_message_explains_the_no_fallback_policy(self) -> None:
        resolver = make_resolver(lambda _: httpx.Response(404, json={}))
        cohort = StubCohort(resolver)
        with pytest.raises(VariableNotFoundError, match="never served from"):
            var_loader.load_variable(
                var_name="not_published",
                cohort=cohort,  # type: ignore[arg-type]
                time_window=None,  # type: ignore[arg-type]
                include_sources=["reprodicu"],
                spec=resolver.parse("not_published"),
            )


@pytest.fixture
def dummy_py_compiler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let served ``py`` snippets compile for a source that ships no ``py_env``.

    The resolver reaches a source's compiler through ``get_py_compiler``, which
    imports ``corr_vars.sources.<source>.py_env``. reprodicu ships none, so the
    lookup is patched to a compiler over the namespace a snippet here needs.
    """
    from corr_vars.concepts import resolver as resolver_module
    from corr_vars.concepts.compile import compile_snippet

    def compile_py(snippet, var_name, files_dir, *, files_by_uuid=None):
        return compile_snippet(
            snippet,
            var_name,
            files_dir,
            namespace={"pl": pl},
            files_by_uuid=files_by_uuid,
        )

    monkeypatch.setattr(resolver_module, "get_py_compiler", lambda source: compile_py)


class TestApiBackedExtraction:
    """The full path: API config, compiled py, and extraction into the cohort."""

    @pytest.mark.usefixtures("dummy_py_compiler")
    def test_variable_from_the_api_is_extracted(
        self, dummy_sources: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cohort = local_cohort(dummy_sources)
        served = document(
            json_config={
                "type": "complex",
                "complex": True,
                "dynamic": False,
                "py_ready_polars": True,
                "compatible_with": ["icu_stay"],
            },
            py=(
                "def api_var(var, cohort):\n"
                "    return cohort.obs.select(cohort.primary_key).with_columns(\n"
                "        pl.lit(42).alias('value')\n"
                "    )\n"
            ),
            version=9,
            name="api_var",
        )
        monkeypatch.setattr(
            cohort,
            "_concepts",
            make_resolver(lambda _: httpx.Response(200, json=served)),
        )

        cohort.add_variable("api_var::v9")

        assert cohort.obs["api_var"].unique().to_list() == [42]
        recorded = records_by_source(cohort, "api_var")["reprodicu"]
        assert recorded["origin"] == "api"
        assert recorded["version"] == 9
        assert recorded["requested"] == "v9"
        assert recorded["concept_id"] == 1

    @pytest.mark.usefixtures("dummy_py_compiler")
    def test_version_warning_reaches_the_user(
        self,
        dummy_sources: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging

        cohort = local_cohort(dummy_sources)
        served = document(
            json_config={
                "type": "complex",
                "complex": True,
                "dynamic": False,
                "py_ready_polars": True,
                "compatible_with": ["icu_stay"],
            },
            py=(
                "def api_var(var, cohort):\n"
                "    return cohort.obs.select(cohort.primary_key).with_columns(\n"
                "        pl.lit(1).alias('value')\n"
                "    )\n"
            ),
            warning={
                "type": "critical_update",
                "corrected_in_version": 11,
                "message": "the denominator was wrong.",
            },
            name="api_var",
        )
        monkeypatch.setattr(
            cohort,
            "_concepts",
            make_resolver(lambda _: httpx.Response(200, json=served)),
        )

        with caplog.at_level(logging.WARNING, logger="corr_vars"):
            cohort.add_variable("api_var::v4")

        assert any(
            "critical correction" in record.getMessage()
            and record.levelno == logging.WARNING
            for record in caplog.records
        )
        assert records_by_source(cohort, "api_var")["reprodicu"]["warning"]

    @pytest.mark.usefixtures("dummy_py_compiler")
    def test_deprecated_pointer_still_loads_the_variable(
        self, dummy_sources: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cohort = local_cohort(dummy_sources)
        served = document(
            json_config={
                "type": "complex",
                "complex": True,
                "dynamic": False,
                "py_ready_polars": True,
                "compatible_with": ["icu_stay"],
            },
            py=(
                "def api_var(var, cohort):\n"
                "    return cohort.obs.select(cohort.primary_key).with_columns(\n"
                "        pl.lit(7).alias('value')\n"
                "    )\n"
            ),
            name="api_var",
            deprecated_at="2025-01-01T00:00:00Z",
            pointer_deprecated_at="2025-02-02T00:00:00Z",
        )
        monkeypatch.setattr(
            cohort,
            "_concepts",
            make_resolver(lambda _: httpx.Response(200, json=served)),
        )

        cohort.add_variable("api_var")

        assert cohort.obs["api_var"].unique().to_list() == [7]
        recorded = records_by_source(cohort, "api_var")["reprodicu"]
        assert recorded["deprecated_at"] == "2025-01-01T00:00:00Z"
        assert recorded["pointer_deprecated_at"] == "2025-02-02T00:00:00Z"

    def test_missing_project_is_reported_actionably(
        self, dummy_sources: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CORR_CONCEPTS_API_KEY", raising=False)
        cohort = local_cohort(dummy_sources)
        monkeypatch.setattr(
            cohort, "_concepts", ConceptResolver(routing=REMOTE_ROUTING, project=None)
        )
        with pytest.raises(ConceptsApiConfigurationError, match="project"):
            cohort.add_variable("blood_sodium")


GROUP_DYNAMIC_JSON = {
    "type": "complex",
    "complex": True,
    "dynamic": True,
    "py_ready_polars": True,
    "compatible_with": ["icu_stay"],
}

GROUP_STATIC_JSON = {
    "type": "complex",
    "complex": True,
    "dynamic": False,
    "py_ready_polars": True,
    "compatible_with": ["icu_stay"],
}


def dynamic_member(concept_id: int, value: int) -> dict[str, Any]:
    """A group member emitting one dynamic record per observation."""
    return element(
        name="atc_c07ab",
        concept_id=concept_id,
        json_config=dict(GROUP_DYNAMIC_JSON),
        py=(
            "def atc_c07ab(var, cohort):\n"
            "    return cohort.obs.select(\n"
            "        cohort.primary_key,\n"
            "        pl.lit(0).alias('recordtime_relative'),\n"
            f"        pl.lit({value}).alias('value'),\n"
            "    )\n"
        ),
    )


def static_member(concept_id: int, value: int) -> dict[str, Any]:
    """A group member emitting one static value per observation."""
    return element(
        name="atc_c07ab",
        concept_id=concept_id,
        json_config=dict(GROUP_STATIC_JSON),
        py=(
            "def atc_c07ab(var, cohort):\n"
            "    return cohort.obs.select(cohort.primary_key).with_columns(\n"
            f"        pl.lit({value}).alias('value')\n"
            "    )\n"
        ),
    )


class TestGroupAutoExpansion:
    """A grouped name loads like a multi-source variable: same columns, concatenated."""

    @pytest.mark.usefixtures("dummy_py_compiler")
    def test_dynamic_members_are_concatenated(
        self, dummy_sources: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cohort = local_cohort(dummy_sources)
        served = group(dynamic_member(12, 1), dynamic_member(13, 2))
        monkeypatch.setattr(
            cohort,
            "_concepts",
            make_resolver(lambda _: httpx.Response(200, json=served)),
        )

        cohort.add_variable("atc_c07ab")

        data = cohort.obsm["atc_c07ab"]
        assert sorted(data["value"].unique().to_list()) == [1, 2]
        assert len(data) == 2 * len(cohort.obs)
        # Both members produce the same columns, so nothing is widened.
        assert data.select("value").width == 1

    @pytest.mark.usefixtures("dummy_py_compiler")
    def test_members_share_the_source_in_the_data_source_column(
        self, dummy_sources: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cohort = local_cohort(dummy_sources)
        served = group(dynamic_member(12, 1), dynamic_member(13, 2))
        monkeypatch.setattr(
            cohort,
            "_concepts",
            make_resolver(lambda _: httpx.Response(200, json=served)),
        )

        cohort.add_variable("atc_c07ab")

        assert cohort.obsm["atc_c07ab"]["data_source"].unique().to_list() == [
            "reprodicu"
        ]

    @pytest.mark.usefixtures("dummy_py_compiler")
    def test_one_variable_holds_one_contributor_per_member(
        self, dummy_sources: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cohort = local_cohort(dummy_sources)
        served = group(dynamic_member(12, 1), dynamic_member(13, 2))
        monkeypatch.setattr(
            cohort,
            "_concepts",
            make_resolver(lambda _: httpx.Response(200, json=served)),
        )

        var = cohort.load_variable("atc_c07ab")
        assert set(var.variables) == {"reprodicu#12", "reprodicu#13"}

    @pytest.mark.usefixtures("dummy_py_compiler")
    def test_provenance_carries_one_record_per_member(
        self, dummy_sources: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cohort = local_cohort(dummy_sources)
        served = group(dynamic_member(12, 1), dynamic_member(13, 2))
        monkeypatch.setattr(
            cohort,
            "_concepts",
            make_resolver(lambda _: httpx.Response(200, json=served)),
        )

        cohort.add_variable("atc_c07ab")

        records = cohort.concept_versions["atc_c07ab"]
        assert [record["concept_id"] for record in records] == [12, 13]
        assert {record["source"] for record in records} == {"reprodicu"}
        assert {record["name"] for record in records} == {"atc_c07ab"}

    @pytest.mark.usefixtures("dummy_py_compiler")
    def test_static_members_concatenate_and_hit_the_duplicate_guard(
        self, dummy_sources: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Static contributors are concatenated row-wise, exactly as two data
        # sources are. Two members that both describe the same observations
        # therefore collide on the primary key, which the obs integrity check
        # rejects — the same outcome two overlapping sources would produce.
        cohort = local_cohort(dummy_sources)
        served = group(static_member(12, 1), static_member(13, 2))
        monkeypatch.setattr(
            cohort,
            "_concepts",
            make_resolver(lambda _: httpx.Response(200, json=served)),
        )

        with pytest.raises(CohortDataError, match="Duplicate entries"):
            cohort.add_variable("atc_c07ab")

    @pytest.mark.usefixtures("dummy_py_compiler")
    def test_disjoint_static_members_are_stacked_into_one_column(
        self, dummy_sources: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The static shape that does work: members covering disjoint
        # observations, which is how two data sources behave.
        cohort = local_cohort(dummy_sources)

        def half(concept_id: int, value: int, keep_first: bool) -> dict[str, Any]:
            comparison = "lt" if keep_first else "ge"
            return element(
                name="atc_c07ab",
                concept_id=concept_id,
                json_config=dict(GROUP_STATIC_JSON),
                py=(
                    "def atc_c07ab(var, cohort):\n"
                    "    half = cohort.obs.height // 2\n"
                    "    return (\n"
                    "        cohort.obs.select(cohort.primary_key)\n"
                    "        .with_row_index('_i')\n"
                    f"        .filter(pl.col('_i').{comparison}(half))\n"
                    "        .drop('_i')\n"
                    f"        .with_columns(pl.lit({value}).alias('value'))\n"
                    "    )\n"
                ),
            )

        served = group(half(12, 1, True), half(13, 2, False))
        monkeypatch.setattr(
            cohort,
            "_concepts",
            make_resolver(lambda _: httpx.Response(200, json=served)),
        )

        cohort.add_variable("atc_c07ab")

        assert sorted(cohort.obs["atc_c07ab"].unique().to_list()) == [1, 2]
        assert cohort.obs["atc_c07ab"].null_count() == 0

    @pytest.mark.usefixtures("dummy_py_compiler")
    def test_members_disagreeing_on_dynamic_are_rejected(
        self, dummy_sources: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cohort = local_cohort(dummy_sources)
        served = group(dynamic_member(12, 1), static_member(13, 2))
        monkeypatch.setattr(
            cohort,
            "_concepts",
            make_resolver(lambda _: httpx.Response(200, json=served)),
        )

        with pytest.raises(AssertionError, match="reprodicu#13"):
            cohort.load_variable("atc_c07ab")

    @pytest.mark.usefixtures("dummy_py_compiler")
    def test_project_overrides_reach_every_member(
        self, dummy_sources: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        resolver = make_resolver(
            lambda _: httpx.Response(
                200, json=group(dynamic_member(12, 1), dynamic_member(13, 2))
            )
        )
        cohort = StubCohort(resolver, atc_c07ab={"cleaning": {"value": {"low": 0}}})
        configs = var_loader._load_variable_configs(
            "atc_c07ab",
            cohort,
            ["reprodicu"],  # type: ignore[arg-type]
        )
        assert set(configs) == {"reprodicu#12", "reprodicu#13"}
        assert all(
            config["cleaning"] == {"value": {"low": 0}} for config in configs.values()
        )


class TestVersionPinsOnGroups:
    def test_per_variable_pin_names_the_variable(self) -> None:
        resolver = make_resolver(lambda _: ambiguous_response())
        cohort = StubCohort(resolver)
        with pytest.raises(AmbiguousConceptError) as excinfo:
            var_loader._load_variable_configs(
                "atc_c07ab",
                cohort,  # type: ignore[arg-type]
                ["reprodicu"],
                spec=resolver.parse("atc_c07ab::v3"),
            )
        assert "atc_c07ab" in str(excinfo.value)
        assert [member["id"] for member in excinfo.value.members] == [12, 13]

    def test_cohort_wide_pin_raises_through_add_variable(
        self, dummy_sources: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cohort = local_cohort(dummy_sources, version="v5")
        monkeypatch.setattr(
            cohort,
            "_concepts",
            make_resolver(lambda _: ambiguous_response(), version="v5"),
        )
        with pytest.raises(AmbiguousConceptError, match="v5"):
            cohort.add_variable("atc_c07ab")


@pytest.mark.usefixtures("dummy_py_compiler")
class TestSaveRoundTrip:
    def test_concept_versions_survive_save_and_load(
        self, dummy_sources: dict[str, Any], tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cohort = local_cohort(dummy_sources, project="demo", taxonomy="corr_v1")
        serve_blood_sodium(cohort, monkeypatch)
        cohort.add_variable("blood_sodium")

        path = tmp_path / "cohort.corr3"
        cohort.save(path)
        reloaded = Cohort.load(path)

        assert reloaded.concept_versions == cohort.concept_versions
        assert reloaded.concepts.project == "demo"
        assert reloaded.concepts.default_taxonomy == "corr_v1"

    def test_concept_versions_are_a_list_of_records(
        self, dummy_sources: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cohort = local_cohort(dummy_sources, project="demo")
        serve_blood_sodium(cohort, monkeypatch)
        cohort.add_variable("blood_sodium")

        records = cohort.concept_versions["blood_sodium"]
        assert isinstance(records, list)
        assert {record["source"] for record in records} == {"reprodicu"}

    def test_source_keyed_archives_load_into_the_list_form(
        self, dummy_sources: dict[str, Any], tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import json
        import tarfile

        import zstandard as zstd

        cohort = local_cohort(dummy_sources, project="demo")
        serve_blood_sodium(cohort, monkeypatch)
        cohort.add_variable("blood_sodium")
        path = tmp_path / "cohort.corr3"
        cohort.save(path)

        # Rewrite the registry into the source-keyed mapping form an older
        # archive carries and repack the archive around it.
        extracted = tmp_path / "extracted"
        extracted.mkdir()
        tar_path = extracted / "archive.tar"
        with open(path, "rb") as src, open(tar_path, "wb") as dst:
            dst.write(zstd.ZstdDecompressor().decompress(src.read()))
        with tarfile.open(tar_path) as tar:
            tar.extractall(extracted)
        tar_path.unlink()

        state_path = extracted / "cohort.json"
        state = json.loads(state_path.read_text())
        state["concept_versions"] = {
            var_name: {record["source"]: record for record in records}
            for var_name, records in state["concept_versions"].items()
        }
        state_path.write_text(json.dumps(state))

        with tarfile.open(tar_path, "w") as tar:
            for entry in extracted.iterdir():
                if entry.name != "archive.tar":
                    tar.add(entry, arcname=entry.name)
        with open(tar_path, "rb") as src, open(path, "wb") as dst:
            dst.write(zstd.ZstdCompressor(level=3).compress(src.read()))

        reloaded = Cohort.load(path)
        assert reloaded.concept_versions == cohort.concept_versions

    def test_as_of_date_survives_save_and_load(
        self, dummy_sources: dict[str, Any], tmp_path
    ) -> None:
        cohort = local_cohort(dummy_sources, project="demo", date="2025-06-30")
        path = tmp_path / "cohort.corr3"
        cohort.save(path)
        reloaded = Cohort.load(path)
        assert reloaded.concepts.default_version == VersionSelector(
            "date", date(2025, 6, 30)
        )

    def test_api_key_is_never_persisted(
        self, dummy_sources: dict[str, Any], tmp_path
    ) -> None:
        import json
        import tarfile
        import tempfile

        import zstandard as zstd

        cohort = local_cohort(dummy_sources, project="demo", api_key="super-secret")
        path = tmp_path / "cohort.corr3"
        cohort.save(path)

        with tempfile.TemporaryDirectory() as extracted:
            tar_path = f"{extracted}/archive.tar"
            with open(path, "rb") as src, open(tar_path, "wb") as dst:
                dst.write(zstd.ZstdDecompressor().decompress(src.read()))
            with tarfile.open(tar_path) as tar:
                tar.extractall(extracted)
            with open(f"{extracted}/cohort.json") as f:
                state = json.loads(f.read())

        assert "super-secret" not in json.dumps(state)
        assert state["concepts_settings"] == {
            "project": "demo",
            "taxonomy": "corr_v1",
            "version": "latest",
        }


class TestDefaultsStayLocal:
    def test_default_vars_are_read_from_the_mapping_module(
        self, local_source: str
    ) -> None:
        from corr_vars.definitions import ObsLevel

        defaults = var_loader._collect_default_variable_list_by_source(
            obs_level=ObsLevel.ICU_STAY, include_sources=[local_source]
        )
        assert defaults[local_source] == ["age", "los_icu"]

    def test_default_vars_stay_local_for_an_api_routed_source(
        self, local_source: str
    ) -> None:
        """A source's default list is read locally even when it is API-routed."""
        from corr_vars.definitions import ObsLevel

        resolver = ConceptResolver(routing=REMOTE_ROUTING, project="demo")
        assert resolver.is_remote(local_source) is True
        defaults = var_loader._collect_default_variable_list_by_source(
            obs_level=ObsLevel.ICU_STAY, include_sources=[local_source]
        )
        assert defaults[local_source]

    def test_a_source_without_defaults_is_skipped(self) -> None:
        from corr_vars.definitions import ObsLevel

        defaults = var_loader._collect_default_variable_list_by_source(
            obs_level=ObsLevel.ICU_STAY, include_sources=["reprodicu"]
        )
        assert defaults == {}


# ---------------------------------------------------------------------------
# load_raw_variable_configs (the search widget)
# ---------------------------------------------------------------------------


class NameListingClient:
    """Stub client exposing only what the name listing needs."""

    def __init__(self, names: list[str] | None = None, error: Exception | None = None):
        self._names = names or []
        self._error = error
        self.calls: list[tuple[str, str | None]] = []

    def list_concept_names(self, taxonomy: str, *, source: str | None = None):
        self.calls.append((taxonomy, source))
        if self._error is not None:
            raise self._error
        return list(self._names)


def name_listing_resolver(client: NameListingClient) -> ConceptResolver:
    return ConceptResolver(
        routing=REMOTE_ROUTING, client_factory=lambda url: client, project="demo"
    )


class TestLoadRawVariableConfigs:
    def test_a_local_source_contributes_its_full_configs(
        self, local_source: str
    ) -> None:
        configs = var_loader.load_raw_variable_configs(include_sources=[local_source])
        assert configs[local_source]["blood_sodium"]["type"]

    def test_a_remote_source_contributes_its_names(self) -> None:
        client = NameListingClient(["corr_v1/blood_sodium", "corr_v1/heart_rate"])
        configs = var_loader.load_raw_variable_configs(
            include_sources=["reprodicu"], resolver=name_listing_resolver(client)
        )
        assert set(configs["reprodicu"]) == {
            "corr_v1/blood_sodium",
            "corr_v1/heart_rate",
        }
        assert configs["reprodicu"]["corr_v1/heart_rate"] == {"source": "concepts-api"}
        assert client.calls == [("corr_v1", "reprodicu")]

    def test_an_api_error_skips_the_source_with_a_warning(self, caplog) -> None:
        client = NameListingClient(error=ConceptsApiError("endpoint is down"))
        with caplog.at_level("WARNING"):
            configs = var_loader.load_raw_variable_configs(
                include_sources=["reprodicu"], resolver=name_listing_resolver(client)
            )
        assert "reprodicu" not in configs
        assert "endpoint is down" in caplog.text

    def test_without_a_resolver_a_remote_source_is_skipped(self) -> None:
        configs = var_loader.load_raw_variable_configs(include_sources=["reprodicu"])
        assert configs == {}

    def test_the_search_widget_passes_the_cohorts_resolver(
        self, dummy_sources: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}

        def spy(include_sources=None, *, resolver=None):
            seen["resolver"] = resolver
            return {}

        cohort = local_cohort(dummy_sources)
        monkeypatch.setattr(var_loader, "load_raw_variable_configs", spy)
        cohort.to_search_widget(include_sources=["reprodicu"])
        assert seen["resolver"] is cohort.concepts


class TestPyFuncStepFiles:
    def test_files_default_to_empty_for_local_callables(self) -> None:
        from corr_vars.core.steps import PyFuncStep

        def local_py(var: Any, cohort: Any) -> pl.DataFrame:
            return pl.DataFrame()

        step = PyFuncStep(local_py, var=None, py_ready_polars=True)  # type: ignore[arg-type]
        assert step.files == {}

    def test_files_are_taken_from_a_compiled_callable(self, tmp_path) -> None:
        from corr_vars.core.steps import PyFuncStep
        from corr_vars.concepts.compile import compile_snippet

        (tmp_path / "a.csv").write_text("x\n")
        compiled = compile_snippet(
            "def demo(var, cohort):\n    return pl.DataFrame()\n",
            "demo",
            tmp_path,
            namespace={"pl": pl},
        )
        step = PyFuncStep(compiled, var=None, py_ready_polars=True)  # type: ignore[arg-type]
        assert set(step.files) == {"a.csv"}
