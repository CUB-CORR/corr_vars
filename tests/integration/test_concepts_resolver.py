from __future__ import annotations

import copy
import hashlib
import logging
from datetime import date

import httpx
import pytest

from corr_vars.concepts.client import ConceptsApiClient
from corr_vars.concepts.compile import compile_snippet
from corr_vars.concepts.resolver import ConceptResolver, _timebounds_to_tuple
from corr_vars.concepts.routing import ConceptRouting, Endpoint
from corr_vars.concepts.spec import LATEST_SELECTOR, VersionSelector
from corr_vars.definitions.exceptions import (
    AmbiguousConceptError,
    ConceptsApiError,
    VariableDefinitionError,
)

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

ENDPOINT = "https://concepts.test/api"
FILE_UUID = "2f3a9c1e-5b7d-4a10-9f2c-8e6d1b4a7c30"

ROUTING = ConceptRouting(
    taxonomy="corr_v1",
    version="latest",
    local_sources=("local_source",),
    endpoints=(Endpoint(url=ENDPOINT, sources=("*",)),),
)


def element(
    *,
    sources: dict[str, Any] | None = None,
    version: int = 4,
    name: str = "blood_sodium",
    concept_id: int = 1,
    pointer: dict[str, Any] | None = None,
    deprecated_at: str | None = None,
    successor_id: int | None = None,
) -> dict[str, Any]:
    return {
        "id": concept_id,
        "taxonomy": "corr_v1",
        "name": name,
        "version": version,
        "requested": {},
        "sources": (
            sources
            if sources is not None
            else {
                "demo_source": {
                    "json": {"type": "native_dynamic", "table": "labs"},
                    "py": None,
                    "files": [],
                    "version_info": {
                        "source_version": 4,
                        "status": "committed",
                        "committed_at": "2024-05-06T10:00:00Z",
                        "warning": None,
                    },
                }
            }
        ),
        "pointer": (
            pointer
            if pointer is not None
            else {
                "id": 100 + concept_id,
                "identifier": name,
                "display_name": name,
                "relationship": "primary",
                "created_at": "2024-01-01T00:00:00Z",
                "deprecated_at": None,
            }
        ),
        "deprecated_at": deprecated_at,
        "successor_id": successor_id,
    }


def payload(**kwargs: Any) -> list[dict[str, Any]]:
    return [element(**kwargs)]


def group_payload(*elements: dict[str, Any]) -> list[dict[str, Any]]:
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


@pytest.fixture(autouse=True)
def demo_source_py_compiler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give ``demo_source`` a ``py_env``-style compiler.

    ``ConceptResolver`` reaches a source's compiler through
    :func:`~corr_vars.concepts.compile.get_py_compiler`, which imports
    ``corr_vars.sources.<source>.py_env``. ``demo_source`` is a name invented for
    these tests, so the lookup is patched to a compiler over an empty namespace —
    what a source shipping a ``py_env`` would supply.
    """

    def compiler(snippet, var_name, files_dir, *, files_by_uuid=None):
        return compile_snippet(
            snippet, var_name, files_dir, namespace={}, files_by_uuid=files_by_uuid
        )

    monkeypatch.setattr(
        "corr_vars.concepts.resolver.get_py_compiler",
        lambda source: compiler if source == "demo_source" else None,
    )


def make_resolver(
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> ConceptResolver:
    def factory(url: str) -> ConceptsApiClient:
        return ConceptsApiClient(
            url,
            project="demo",
            api_key="k",
            transport=httpx.MockTransport(handler),
            backoff=0.0,
        )

    kwargs.setdefault("project", "demo")
    return ConceptResolver(routing=ROUTING, client_factory=factory, **kwargs)


class TestDefaults:
    def test_defaults_come_from_the_routing_table(self) -> None:
        resolver = make_resolver(lambda _: httpx.Response(200, json=payload()))
        assert resolver.default_taxonomy == "corr_v1"
        assert resolver.default_version == LATEST_SELECTOR

    def test_cohort_kwargs_override_the_file(self) -> None:
        resolver = make_resolver(
            lambda _: httpx.Response(200, json=payload()),
            taxonomy="corr_v2",
            version="v9",
        )
        assert resolver.default_taxonomy == "corr_v2"
        assert resolver.default_version == VersionSelector("version", 9)

    def test_as_of_date_becomes_the_default_selector(self) -> None:
        resolver = make_resolver(
            lambda _: httpx.Response(200, json=payload()), date="2024-05-06"
        )
        assert resolver.default_version == VersionSelector("date", date(2024, 5, 6))

    def test_version_and_date_together_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="only one"):
            make_resolver(
                lambda _: httpx.Response(200, json=payload()),
                version="v3",
                date="2024-05-06",
            )

    def test_parse_applies_defaults(self) -> None:
        resolver = make_resolver(
            lambda _: httpx.Response(200, json=payload()), date="2024-05-06"
        )
        spec = resolver.parse("blood_sodium")
        assert spec.taxonomy == "corr_v1"
        assert spec.version == VersionSelector("date", date(2024, 5, 6))

    def test_default_spec_ignores_any_pin(self) -> None:
        resolver = make_resolver(lambda _: httpx.Response(200, json=payload()))
        assert resolver.default_spec("dep").version == LATEST_SELECTOR


class TestRouting:
    def test_is_remote(self) -> None:
        resolver = make_resolver(lambda _: httpx.Response(200, json=payload()))
        assert resolver.is_remote("demo_source") is True
        assert resolver.is_remote("reprodicu") is True
        assert resolver.is_remote("local_source") is False

    def test_local_sources_are_never_fetched(self) -> None:
        calls = {"n": 0}

        def handler(_: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(200, json=payload())

        resolver = make_resolver(handler)
        spec = resolver.parse("blood_sodium")
        assert resolver.fetch_configs(spec, ["local_source"]) == {}
        assert calls["n"] == 0

    def test_sources_sharing_an_endpoint_cost_one_request(self) -> None:
        calls = {"n": 0}

        def handler(_: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(200, json=payload())

        resolver = make_resolver(handler)
        spec = resolver.parse("blood_sodium")
        resolver.fetch_configs(spec, ["demo_source", "reprodicu"])
        assert calls["n"] == 1


class TestFetchConfigs:
    def test_config_and_resolution(self) -> None:
        resolver = make_resolver(lambda _: httpx.Response(200, json=payload()))
        spec = resolver.parse("blood_sodium")
        fetched = resolver.fetch_configs(spec, ["demo_source"])["demo_source"]

        assert fetched.config == {
            "type": "native_dynamic",
            "table": "labs",
        }
        assert fetched.resolution.origin == "api"
        assert fetched.resolution.endpoint == ENDPOINT
        assert fetched.resolution.version == 4
        assert fetched.resolution.source_version == 4
        assert fetched.resolution.requested == "latest"
        assert fetched.resolution.as_dict()["name"] == "blood_sodium"

    def test_source_absent_from_the_document_is_skipped(self) -> None:
        resolver = make_resolver(lambda _: httpx.Response(200, json=payload()))
        spec = resolver.parse("blood_sodium")
        fetched = resolver.fetch_configs(spec, ["demo_source", "reprodicu"])
        assert set(fetched) == {"demo_source"}

    def test_unknown_concept_yields_nothing(self) -> None:
        resolver = make_resolver(lambda _: httpx.Response(404, json={}))
        spec = resolver.parse("blood_sodium")
        assert resolver.fetch_configs(spec, ["demo_source"]) == {}

    def test_unknown_concept_warns_at_warning_level(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # An endpoint that has imported nothing 404s every concept, so each of its
        # sources drops out of every variable. Silently narrowing a multi-source
        # variable is the divergence the hard-fail policy exists to prevent, so it
        # has to be visible at the default log level.
        resolver = make_resolver(lambda _: httpx.Response(404, json={}))
        spec = resolver.parse("blood_sodium")

        with caplog.at_level(logging.WARNING, logger="corr_vars"):
            resolver.fetch_configs(spec, ["demo_source"])

        assert any(
            "is not published at" in record.getMessage()
            and record.levelno == logging.WARNING
            for record in caplog.records
        )

    def test_transport_failure_is_fatal(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down", request=request)

        resolver = make_resolver(handler)
        spec = resolver.parse("blood_sodium")
        with pytest.raises(ConceptsApiError):
            resolver.fetch_configs(spec, ["demo_source"])

    def test_never_falls_back_to_the_bundled_definition(self) -> None:
        # blood_sodium exists in the bundled demo_source vars.json; a server error
        # must still be fatal rather than quietly serving the local entry.
        resolver = make_resolver(lambda _: httpx.Response(500))
        spec = resolver.parse("blood_sodium")
        with pytest.raises(ConceptsApiError):
            resolver.fetch_configs(spec, ["demo_source"])

    def test_version_warning_is_surfaced_at_warning_level(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        document = payload()
        document[0]["sources"]["demo_source"]["version_info"]["warning"] = {
            "type": "critical_update",
            "corrected_in_version": 6,
            "message": "unit conversion was wrong.",
        }
        resolver = make_resolver(lambda _: httpx.Response(200, json=document))
        spec = resolver.parse("blood_sodium::v4")

        with caplog.at_level(logging.WARNING, logger="corr_vars"):
            fetched = resolver.fetch_configs(spec, ["demo_source"])

        assert any(
            "critical correction" in record.getMessage()
            and record.levelno == logging.WARNING
            for record in caplog.records
        )
        assert fetched["demo_source"].resolution.warning == {
            "type": "critical_update",
            "corrected_in_version": 6,
            "message": "unit conversion was wrong.",
        }

    def test_single_member_keeps_the_bare_source_key(self) -> None:
        resolver = make_resolver(lambda _: httpx.Response(200, json=payload()))
        spec = resolver.parse("blood_sodium")
        assert set(resolver.fetch_configs(spec, ["demo_source"])) == {"demo_source"}

    def test_concept_id_is_recorded(self) -> None:
        resolver = make_resolver(
            lambda _: httpx.Response(200, json=payload(concept_id=42))
        )
        spec = resolver.parse("blood_sodium")
        fetched = resolver.fetch_configs(spec, ["demo_source"])["demo_source"]
        assert fetched.resolution.concept_id == 42
        assert fetched.resolution.as_dict()["concept_id"] == 42


def group_source(table: str) -> dict[str, Any]:
    return {
        "demo_source": {
            "json": {"type": "native_dynamic", "table": table},
            "py": None,
            "files": [],
            "version_info": {"source_version": 4, "status": "committed"},
        }
    }


class TestGroupExpansion:
    """A name resolving to several concepts expands like an extra source."""

    def group(self) -> list[dict[str, Any]]:
        return group_payload(
            element(
                concept_id=12, name="atc_c07ab", sources=group_source("metoprolol")
            ),
            element(
                concept_id=13, name="atc_c07ab", sources=group_source("bisoprolol")
            ),
        )

    def test_every_member_becomes_its_own_contributor(self) -> None:
        resolver = make_resolver(lambda _: httpx.Response(200, json=self.group()))
        spec = resolver.parse("atc_c07ab")
        fetched = resolver.fetch_configs(spec, ["demo_source"])

        assert set(fetched) == {"demo_source#12", "demo_source#13"}
        assert fetched["demo_source#12"].config["table"] == "metoprolol"
        assert fetched["demo_source#13"].config["table"] == "bisoprolol"

    def test_each_member_carries_its_own_provenance(self) -> None:
        resolver = make_resolver(lambda _: httpx.Response(200, json=self.group()))
        spec = resolver.parse("atc_c07ab")
        fetched = resolver.fetch_configs(spec, ["demo_source"])

        assert [
            fetched[key].resolution.concept_id
            for key in ("demo_source#12", "demo_source#13")
        ] == [12, 13]
        assert all(fetched[key].resolution.source == "demo_source" for key in fetched)

    def test_group_costs_one_request(self) -> None:
        calls = {"n": 0}

        def handler(_: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(200, json=self.group())

        resolver = make_resolver(handler)
        spec = resolver.parse("atc_c07ab")
        resolver.fetch_configs(spec, ["demo_source"])
        resolver.fetch_configs(spec, ["demo_source"])
        assert calls["n"] == 1

    def test_a_member_not_defining_the_source_drops_out(self) -> None:
        document = group_payload(
            element(
                concept_id=12, name="atc_c07ab", sources=group_source("metoprolol")
            ),
            element(concept_id=13, name="atc_c07ab", sources={}),
        )
        resolver = make_resolver(lambda _: httpx.Response(200, json=document))
        spec = resolver.parse("atc_c07ab")
        assert set(resolver.fetch_configs(spec, ["demo_source"])) == {"demo_source#12"}

    def test_py_is_compiled_per_member(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CORR_VARS_CACHE_DIR", str(tmp_path))

        def py_source(body: str) -> dict[str, Any]:
            return {
                "demo_source": {
                    "json": {"type": "complex", "py_ready_polars": True},
                    "py": f"def atc_c07ab(var, cohort):\n    return {body!r}\n",
                    "files": [],
                    "version_info": {"source_version": 2},
                }
            }

        document = group_payload(
            element(concept_id=12, name="atc_c07ab", sources=py_source("twelve")),
            element(concept_id=13, name="atc_c07ab", sources=py_source("thirteen")),
        )
        resolver = make_resolver(lambda _: httpx.Response(200, json=document))
        spec = resolver.parse("atc_c07ab")
        fetched = resolver.fetch_configs(spec, ["demo_source"])

        first = fetched["demo_source#12"].config["py"]
        second = fetched["demo_source#13"].config["py"]
        assert first is not second
        assert first(None, None) == "twelve"
        assert second(None, None) == "thirteen"

    def test_members_get_separate_materialised_directories(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CORR_VARS_CACHE_DIR", str(tmp_path))
        contents = {12: b"twelve\n", 13: b"thirteen\n"}

        def files_source(concept_id: int) -> dict[str, Any]:
            content = contents[concept_id]
            return {
                "demo_source": {
                    "json": {"type": "complex", "py_ready_polars": True},
                    "py": "def atc_c07ab(var, cohort):\n    return None\n",
                    "files": [
                        {
                            "uuid": f"members-{concept_id}",
                            "path": "members.csv",
                            "version_no": 1,
                            "size": len(content),
                            "sha256": hashlib.sha256(content).hexdigest(),
                            "media_type": "text/csv",
                            "url": (
                                f"{ENDPOINT}/concept/id/{concept_id}/files/"
                                f"members-{concept_id}?v=4"
                            ),
                        }
                    ],
                    "version_info": {"source_version": 2},
                }
            }

        document = group_payload(
            element(concept_id=12, name="atc_c07ab", sources=files_source(12)),
            element(concept_id=13, name="atc_c07ab", sources=files_source(13)),
        )

        def handler(request: httpx.Request) -> httpx.Response:
            if "/files/" in request.url.path:
                concept_id = int(request.url.path.split("/id/")[1].split("/")[0])
                return httpx.Response(200, content=contents[concept_id])
            return httpx.Response(200, json=document)

        resolver = make_resolver(handler)
        spec = resolver.parse("atc_c07ab")
        fetched = resolver.fetch_configs(spec, ["demo_source"])

        first = fetched["demo_source#12"].config["py"]
        second = fetched["demo_source#13"].config["py"]
        assert first.files_dir != second.files_dir
        assert first.files_dir.parent.name == "12"
        assert second.files_dir.parent.name == "13"
        assert first.files["members.csv"].read_bytes() == b"twelve\n"
        assert second.files["members.csv"].read_bytes() == b"thirteen\n"


class TestVersionPinsOnGroups:
    def test_pinned_version_raises_ambiguous_concept_error(self) -> None:
        resolver = make_resolver(lambda _: ambiguous_response())
        spec = resolver.parse("atc_c07ab::v3")
        with pytest.raises(AmbiguousConceptError) as excinfo:
            resolver.fetch_configs(spec, ["demo_source"])

        message = str(excinfo.value)
        assert "atc_c07ab" in message
        assert "v3" in message
        assert "12" in message and "13" in message
        assert [member["id"] for member in excinfo.value.members] == [12, 13]

    def test_pinned_draft_raises_ambiguous_concept_error(self) -> None:
        resolver = make_resolver(lambda _: ambiguous_response())
        spec = resolver.parse("atc_c07ab::draft77")
        with pytest.raises(AmbiguousConceptError, match="draft77"):
            resolver.fetch_configs(spec, ["demo_source"])

    def test_cohort_wide_pin_raises_too(self) -> None:
        resolver = make_resolver(lambda _: ambiguous_response(), version="v5")
        spec = resolver.parse("atc_c07ab")
        with pytest.raises(AmbiguousConceptError, match="v5"):
            resolver.fetch_configs(spec, ["demo_source"])

    def test_as_of_date_expands_instead(self) -> None:
        document = group_payload(
            element(concept_id=12, name="atc_c07ab", sources=group_source("a")),
            element(concept_id=13, name="atc_c07ab", sources=group_source("b")),
        )
        resolver = make_resolver(
            lambda _: httpx.Response(200, json=document), date="2025-06-30"
        )
        spec = resolver.parse("atc_c07ab")
        assert set(resolver.fetch_configs(spec, ["demo_source"])) == {
            "demo_source#12",
            "demo_source#13",
        }


class TestDeprecation:
    def test_deprecated_concept_still_loads_and_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        document = payload(deprecated_at="2025-01-01T00:00:00Z", successor_id=99)
        resolver = make_resolver(lambda _: httpx.Response(200, json=document))
        spec = resolver.parse("blood_sodium")

        with caplog.at_level(logging.WARNING, logger="corr_vars"):
            fetched = resolver.fetch_configs(spec, ["demo_source"])

        assert fetched["demo_source"].config["table"] == "labs"
        assert fetched["demo_source"].resolution.deprecated_at == "2025-01-01T00:00:00Z"
        assert fetched["demo_source"].resolution.successor_id == 99
        assert any(
            "was deprecated on" in record.getMessage()
            and record.levelno == logging.WARNING
            for record in caplog.records
        )

    def test_deprecated_pointer_still_resolves_and_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        document = payload(
            pointer={
                "id": 101,
                "identifier": "blood_sodium",
                "display_name": "Blood sodium",
                "relationship": "primary",
                "created_at": "2024-01-01T00:00:00Z",
                "deprecated_at": "2025-02-02T00:00:00Z",
            }
        )
        resolver = make_resolver(lambda _: httpx.Response(200, json=document))
        spec = resolver.parse("blood_sodium")

        with caplog.at_level(logging.WARNING, logger="corr_vars"):
            fetched = resolver.fetch_configs(spec, ["demo_source"])

        assert fetched["demo_source"].config["table"] == "labs"
        assert (
            fetched["demo_source"].resolution.pointer_deprecated_at
            == "2025-02-02T00:00:00Z"
        )
        assert any(
            "retired on" in record.getMessage() and record.levelno == logging.WARNING
            for record in caplog.records
        )


class TestPySnippets:
    def test_snippet_is_compiled_into_the_config(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CORR_VARS_CACHE_DIR", str(tmp_path))
        document = payload(
            sources={
                "demo_source": {
                    "json": {"type": "complex", "py_ready_polars": True},
                    "py": "def blood_sodium(var, cohort):\n    return pl.DataFrame()\n",
                    "files": [],
                    "version_info": {"source_version": 2},
                }
            }
        )
        resolver = make_resolver(lambda _: httpx.Response(200, json=document))
        spec = resolver.parse("blood_sodium")
        config = resolver.fetch_configs(spec, ["demo_source"])["demo_source"].config
        assert callable(config["py"])
        assert config["py"].var_name == "blood_sodium"

    def test_attached_files_are_materialised_and_exposed(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CORR_VARS_CACHE_DIR", str(tmp_path))
        content = b"a\n1\n"
        document = payload(
            sources={
                "demo_source": {
                    "json": {"type": "complex", "py_ready_polars": True},
                    "py": "def blood_sodium(var, cohort):\n    return pl.DataFrame()\n",
                    "files": [
                        {
                            "uuid": FILE_UUID,
                            "path": "postcode/postcode_mapping.csv",
                            "version_no": 2,
                            "size": len(content),
                            "sha256": hashlib.sha256(content).hexdigest(),
                            "media_type": "text/csv",
                            "url": (
                                f"{ENDPOINT}/concept/corr_v1/blood_sodium/files/"
                                f"{FILE_UUID}?source=demo_source&v=4"
                            ),
                        }
                    ],
                    "version_info": {"source_version": 2},
                }
            }
        )
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            if "/files/" in request.url.path:
                return httpx.Response(200, content=content)
            return httpx.Response(200, json=document)

        resolver = make_resolver(handler)
        spec = resolver.parse("blood_sodium")
        py = resolver.fetch_configs(spec, ["demo_source"])["demo_source"].config["py"]

        assert "postcode/postcode_mapping.csv" in py.files
        assert py.files["postcode/postcode_mapping.csv"].read_bytes() == content
        assert py.files_dir == py.files["postcode/postcode_mapping.csv"].parent.parent

        # The served url is used verbatim; only project is merged in.
        file_request = seen[-1]
        assert file_request.url.params["source"] == "demo_source"
        assert file_request.url.params["v"] == "4"
        assert file_request.url.params["project"] == "demo"

    @staticmethod
    def _getfile_document(
        content: bytes, uuid: str = FILE_UUID
    ) -> list[dict[str, Any]]:
        """A concept whose snippet reaches its data file by uuid."""
        return payload(
            sources={
                "demo_source": {
                    "json": {"type": "complex", "py_ready_polars": True},
                    "py": (
                        "def blood_sodium(var, cohort):\n"
                        f"    return getfile({uuid!r}).read_bytes()\n"
                    ),
                    "files": [
                        {
                            "uuid": FILE_UUID,
                            "path": "postcode/postcode_mapping.csv",
                            "version_no": 2,
                            "size": len(content),
                            "sha256": hashlib.sha256(content).hexdigest(),
                            "media_type": "text/csv",
                            "url": (
                                f"{ENDPOINT}/concept/corr_v1/blood_sodium/files/"
                                f"{FILE_UUID}?v=4"
                            ),
                        }
                    ],
                    "version_info": {"source_version": 2},
                }
            }
        )

    def test_getfile_resolves_the_pinned_bytes(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CORR_VARS_CACHE_DIR", str(tmp_path))
        content = b"postal_code,state\n10115,Berlin\n"
        document = self._getfile_document(content)

        def handler(request: httpx.Request) -> httpx.Response:
            if "/files/" in request.url.path:
                return httpx.Response(200, content=content)
            return httpx.Response(200, json=document)

        resolver = make_resolver(handler)
        spec = resolver.parse("blood_sodium")
        py = resolver.fetch_configs(spec, ["demo_source"])["demo_source"].config["py"]

        assert py(None, None) == content
        assert py.files_by_uuid[FILE_UUID].read_bytes() == content

    def test_getfile_for_a_uuid_the_config_does_not_pin(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CORR_VARS_CACHE_DIR", str(tmp_path))
        content = b"a\n1\n"
        # The snippet asks for a uuid the manifest does not carry — a file added
        # to the source's library after this version was published, or a typo.
        document = self._getfile_document(content, uuid="00000000-dead-beef")

        def handler(request: httpx.Request) -> httpx.Response:
            if "/files/" in request.url.path:
                return httpx.Response(200, content=content)
            return httpx.Response(200, json=document)

        resolver = make_resolver(handler)
        spec = resolver.parse("blood_sodium")
        py = resolver.fetch_configs(spec, ["demo_source"])["demo_source"].config["py"]

        with pytest.raises(VariableDefinitionError) as excinfo:
            py(None, None)
        assert "00000000-dead-beef" in str(excinfo.value)
        assert "blood_sodium" in str(excinfo.value)

    def test_pinned_bytes_are_downloaded_once_and_then_served_offline(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CORR_VARS_CACHE_DIR", str(tmp_path))
        content = b"postal_code,state\n10115,Berlin\n"
        document = self._getfile_document(content)
        downloads = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            if "/files/" in request.url.path:
                downloads["n"] += 1
                return httpx.Response(200, content=content)
            return httpx.Response(200, json=document)

        resolver = make_resolver(handler)
        spec = resolver.parse("blood_sodium")
        resolver.fetch_configs(spec, ["demo_source"])

        # A fresh resolver, so the client's in-process blob cache is gone too:
        # what makes the second resolve free is the content-addressed cache on
        # disk, keyed by the sha256 the manifest pins.
        second = make_resolver(handler)
        py = second.fetch_configs(second.parse("blood_sodium"), ["demo_source"])[
            "demo_source"
        ].config["py"]

        assert downloads["n"] == 1
        assert py(None, None) == content

    def test_a_manifest_entry_without_a_uuid_is_fatal(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CORR_VARS_CACHE_DIR", str(tmp_path))
        document = self._getfile_document(b"a\n1\n")
        del document[0]["sources"]["demo_source"]["files"][0]["uuid"]
        resolver = make_resolver(lambda _: httpx.Response(200, json=document))
        with pytest.raises(ConceptsApiError, match="carries no uuid"):
            resolver.fetch_configs(resolver.parse("blood_sodium"), ["demo_source"])

    def test_snippet_for_a_source_without_py_env_is_fatal(self) -> None:
        document = payload(
            sources={
                "reprodicu": {
                    "json": {"type": "complex"},
                    "py": "def blood_sodium(var, cohort):\n    return None\n",
                    "files": [],
                    "version_info": {},
                }
            }
        )
        resolver = make_resolver(lambda _: httpx.Response(200, json=document))
        spec = resolver.parse("blood_sodium")
        with pytest.raises(ConceptsApiError, match="py_env"):
            resolver.fetch_configs(spec, ["reprodicu"])


class TestTimeBoundCoercion:
    def test_lists_become_tuples(self) -> None:
        config = _timebounds_to_tuple(
            {"tmin": ["icu_admission", "-2h"], "tmax": ["icu_admission", "+6h"]}
        )
        assert config["tmin"] == ("icu_admission", "-2h")
        assert config["tmax"] == ("icu_admission", "+6h")

    def test_strings_are_left_alone(self) -> None:
        config = _timebounds_to_tuple({"tmin": "icu_admission", "tmax": None})
        assert config == {"tmin": "icu_admission", "tmax": None}

    def test_nested_requires_are_coerced(self) -> None:
        config = _timebounds_to_tuple(
            {
                "requires": {
                    "dep": {
                        "template": "x",
                        "tmin": ["icu_admission", "-1d"],
                        "tmax": "inherit",
                    }
                }
            }
        )
        assert config["requires"]["dep"]["tmin"] == ("icu_admission", "-1d")

    def test_list_form_requires_untouched(self) -> None:
        config = _timebounds_to_tuple({"requires": ["a", "b"]})
        assert config["requires"] == ["a", "b"]

    def test_coerced_bounds_are_valid_time_anchor_columns(self) -> None:
        from corr_vars.definitions.typing import is_time_anchor_column

        config = _timebounds_to_tuple({"tmin": ["icu_admission", "-2h"]})
        assert is_time_anchor_column(config["tmin"])


# The real shapes from sources/demo_source/mapping/vars.json. The APACHE-II variables
# are the only definitions with list-valued tmin/tmax nested inside `requires`,
# and therefore the only ones exercising the recursion of _timebounds_to_tuple
# over the API path.
APACHE2_CHRONIC_HEALTH = {
    "type": "derived_static",
    "compatible_with": ["icu_stay"],
    "requires": {
        "dx_liver_cirrhosis": {
            "template": "dx_liver_cirrhosis",
            "tmin": "hospital_admission",
            "tmax": "hospital_discharge",
        },
        "any_surgery_ops": {
            "template": "any_surgery_ops",
            "tmin": ["icu_admission", "-24h"],
            "tmax": ["icu_admission", "6h"],
        },
        "surgery_urgency": {
            "template": "surgery_urgency",
            "tmin": ["icu_admission", "-24h"],
            "tmax": ["icu_admission", "6h"],
        },
    },
    "py_ready_polars": True,
}

ALVEOLAR_ARTERIAL_GRADIENT = {
    "type": "derived_dynamic",
    "requires": {
        "blood_pao2_arterial": {
            "template": "blood_pao2_arterial",
            "tmin": "inherit",
            "tmax": "inherit",
        },
        "vent_fio2": {
            "template": "vent_fio2",
            "tmin": ["inherit", "-1h"],
            "tmax": "inherit",
        },
    },
    "py_ready_polars": True,
    "cleaning": {"value": {"low": -100, "high": 700}},
}


def served(name: str, json: dict[str, Any]) -> list[dict[str, Any]]:
    return payload(
        name=name,
        sources={
            "demo_source": {
                "json": json,
                "py": None,
                "files": [],
                "version_info": {"source_version": 1},
            }
        },
    )


class TestNestedRequiresTimeBounds:
    """Nested `requires` bounds must survive the API path as tuples.

    JSON has no tuple type, so ``("icu_admission", "-24h")`` arrives as a list. A
    list that reaches the loader is not merely rejected: ``TimeAnchor`` only
    accepts a string or a two-string tuple, so the variable resolves against a
    different window than its definition names. The local path restores the tuple
    in ``mapping/loader.py``'s object_hook; over the API it is the resolver's job.
    """

    def test_nested_bounds_of_apache2_chronic_health_are_tuples(self) -> None:
        resolver = make_resolver(
            lambda _: httpx.Response(
                200, json=served("apache2_chronic_health", APACHE2_CHRONIC_HEALTH)
            )
        )
        spec = resolver.parse("apache2_chronic_health")

        requires = resolver.fetch_configs(spec, ["demo_source"])["demo_source"].config[
            "requires"
        ]

        assert requires["any_surgery_ops"]["tmin"] == ("icu_admission", "-24h")
        assert requires["any_surgery_ops"]["tmax"] == ("icu_admission", "6h")
        assert requires["surgery_urgency"]["tmin"] == ("icu_admission", "-24h")
        assert requires["surgery_urgency"]["tmax"] == ("icu_admission", "6h")
        for bound in ("tmin", "tmax"):
            assert isinstance(requires["any_surgery_ops"][bound], tuple)
            assert isinstance(requires["surgery_urgency"][bound], tuple)

        # String bounds on the sibling requirement are left exactly as served.
        assert requires["dx_liver_cirrhosis"]["tmin"] == "hospital_admission"
        assert requires["dx_liver_cirrhosis"]["tmax"] == "hospital_discharge"

    def test_nested_inherit_offset_of_alveolar_arterial_gradient_is_a_tuple(
        self,
    ) -> None:
        resolver = make_resolver(
            lambda _: httpx.Response(
                200,
                json=served("alveolar_arterial_gradient", ALVEOLAR_ARTERIAL_GRADIENT),
            )
        )
        spec = resolver.parse("alveolar_arterial_gradient")

        requires = resolver.fetch_configs(spec, ["demo_source"])["demo_source"].config[
            "requires"
        ]

        assert requires["vent_fio2"]["tmin"] == ("inherit", "-1h")
        assert isinstance(requires["vent_fio2"]["tmin"], tuple)
        assert requires["vent_fio2"]["tmax"] == "inherit"
        assert requires["blood_pao2_arterial"]["tmin"] == "inherit"

    def test_a_nested_bound_that_stayed_a_list_would_not_build_a_time_anchor(
        self,
    ) -> None:
        """Why the coercion matters, stated as the failure it prevents."""
        from corr_vars.utils.time import TimeAnchor

        with pytest.raises(TypeError):
            TimeAnchor(["icu_admission", "-24h"])  # type: ignore[arg-type]

        assert TimeAnchor(("icu_admission", "-24h")).delta == ("-24h",)


class TestLifecycle:
    def test_deepcopy_returns_the_same_resolver(self) -> None:
        resolver = make_resolver(lambda _: httpx.Response(200, json=payload()))
        assert copy.deepcopy(resolver) is resolver

    def test_settings_exclude_the_api_key(self) -> None:
        resolver = make_resolver(lambda _: httpx.Response(200, json=payload()))
        assert resolver.settings() == {
            "project": "demo",
            "taxonomy": "corr_v1",
            "version": "latest",
        }

    def test_getstate_drops_credentials(self) -> None:
        resolver = make_resolver(lambda _: httpx.Response(200, json=payload()))
        state = resolver.__getstate__()
        assert state["_api_key"] is None
        assert state["_clients"] == {}
