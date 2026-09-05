from __future__ import annotations

import httpx
import pytest

from corr_vars.concepts.client import (
    API_KEY_ENV_VAR,
    Concept,
    ConceptFile,
    ConceptsApiClient,
)
from corr_vars.concepts.spec import VersionSelector, parse_version_selector
from corr_vars.definitions.exceptions import (
    AmbiguousConceptError,
    ConceptNotFoundError,
    ConceptsApiConfigurationError,
    ConceptsApiError,
    ConceptsLicenseError,
    ProjectNotFoundError,
    VariableNotFoundError,
)

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

BASE_URL = "https://concepts.test/api"
FILE_UUID = "2f3a9c1e-5b7d-4a10-9f2c-8e6d1b4a7c30"


def concept_element(**overrides: Any) -> dict[str, Any]:
    """One element of the list ``GET /concept/{taxonomy}/{name}`` serves."""
    payload: dict[str, Any] = {
        "id": 17,
        "taxonomy": "corr_v1",
        "name": "blood_sodium",
        "version": 4,
        "requested": {"v": None, "date": None, "draft": None},
        "sources": {
            "demo_source": {
                "json": {"type": "native_dynamic", "table": "labs"},
                "py": None,
                "files": [],
                "version_info": {
                    "source_version": 4,
                    "type": "native_dynamic",
                    "read_only": False,
                    "change_type": "minor",
                    "message": "widen the where clause",
                    "author": "someone",
                    "committed_at": "2024-05-06T10:00:00Z",
                    "status": "committed",
                    "warning": None,
                },
            }
        },
        "pointer": {
            "id": 101,
            "identifier": "blood_sodium",
            "display_name": "Blood sodium",
            "relationship": "primary",
            "created_at": "2024-01-01T00:00:00Z",
            "deprecated_at": None,
        },
        "deprecated_at": None,
        "successor_id": None,
        "doc_clinical": "Serum sodium.",
        "doc_implementation": None,
        "doc_caveats": None,
        "doc_status": "reviewed",
        "notion_url": None,
    }
    payload.update(overrides)
    return payload


def concept_payload(*elements: dict[str, Any]) -> list[dict[str, Any]]:
    """The list body of a name resolving to one concept, or to a group."""
    return list(elements) if elements else [concept_element()]


def ambiguous_body(*members: dict[str, Any]) -> dict[str, Any]:
    return {
        "detail": {
            "error": "ambiguous_name",
            "members": list(members)
            or [
                {
                    "id": 12,
                    "name": "atc_c07ab",
                    "display_name": "Metoprolol",
                    "description": "",
                },
                {
                    "id": 13,
                    "name": "atc_c07ab",
                    "display_name": "Bisoprolol",
                    "description": "",
                },
            ],
        }
    }


def make_client(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    project: str = "demo",
    max_retries: int = 3,
) -> ConceptsApiClient:
    return ConceptsApiClient(
        BASE_URL,
        project=project,
        api_key="secret-token",
        transport=httpx.MockTransport(handler),
        max_retries=max_retries,
        backoff=0.0,
    )


class TestConfiguration:
    def test_missing_project_is_rejected(self) -> None:
        with pytest.raises(ConceptsApiConfigurationError, match="project"):
            ConceptsApiClient(BASE_URL, project="", api_key="k")

    def test_missing_api_key_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)
        with pytest.raises(ConceptsApiConfigurationError, match=API_KEY_ENV_VAR):
            ConceptsApiClient(BASE_URL, project="demo")

    def test_api_key_from_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(API_KEY_ENV_VAR, "from-env")
        client = ConceptsApiClient(BASE_URL, project="demo")
        assert client.http.headers["Authorization"] == "Bearer from-env"

    def test_trailing_slash_stripped(self) -> None:
        client = ConceptsApiClient(f"{BASE_URL}/", project="demo", api_key="k")
        assert client.base_url == BASE_URL


class TestGetConcept:
    def test_request_shape(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=concept_payload())

        client = make_client(handler)
        client.get_concepts("corr_v1", "blood_sodium", parse_version_selector("v3"))

        request = seen[0]
        assert request.url.path == "/api/concept/corr_v1/blood_sodium"
        assert dict(request.url.params) == {"project": "demo", "v": "3"}
        assert request.headers["Authorization"] == "Bearer secret-token"

    def test_project_is_always_sent(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=concept_payload())

        client = make_client(handler)
        client.get_concepts("corr_v1", "x", parse_version_selector("latest"))
        assert seen[0].url.params["project"] == "demo"

    def test_date_selector_uses_date_param(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=concept_payload())

        client = make_client(handler)
        client.get_concepts("corr_v1", "x", parse_version_selector("2024-05-06"))
        assert seen[0].url.params["date"] == "2024-05-06"
        assert "vd" not in seen[0].url.params

    def test_draft_selector_uses_draft_param(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=concept_payload())

        client = make_client(handler)
        client.get_concepts("corr_v1", "x", parse_version_selector("draft55"))
        assert seen[0].url.params["draft"] == "55"

    def test_payload_is_parsed(self) -> None:
        client = make_client(lambda _: httpx.Response(200, json=concept_payload()))
        concepts = client.get_concepts(
            "corr_v1", "blood_sodium", parse_version_selector("latest")
        )
        assert len(concepts) == 1
        concept = concepts[0]
        assert isinstance(concept, Concept)
        assert concept.id == 17
        assert concept.version == 4
        assert concept.doc_clinical == "Serum sodium."
        source = concept.sources["demo_source"]
        assert source.json["table"] == "labs"
        assert source.py is None
        assert source.version_info.status == "committed"
        assert source.version_info.warning is None

    def test_pointer_is_parsed(self) -> None:
        client = make_client(lambda _: httpx.Response(200, json=concept_payload()))
        concept = client.get_concepts(
            "corr_v1", "blood_sodium", parse_version_selector("latest")
        )[0]
        assert concept.pointer is not None
        assert concept.pointer.identifier == "blood_sodium"
        assert concept.pointer.relationship == "primary"
        assert concept.pointer.deprecated is False
        assert concept.deprecated is False
        assert concept.successor_id is None

    def test_deprecation_fields_are_parsed(self) -> None:
        element = concept_element(deprecated_at="2025-01-01T00:00:00Z", successor_id=99)
        element["pointer"]["deprecated_at"] = "2024-12-01T00:00:00Z"
        client = make_client(
            lambda _: httpx.Response(200, json=concept_payload(element))
        )
        concept = client.get_concepts("corr_v1", "x", parse_version_selector("latest"))[
            0
        ]
        assert concept.deprecated is True
        assert concept.successor_id == 99
        assert concept.pointer is not None
        assert concept.pointer.deprecated is True

    def test_missing_pointer_is_tolerated(self) -> None:
        element = concept_element(pointer=None)
        client = make_client(
            lambda _: httpx.Response(200, json=concept_payload(element))
        )
        concept = client.get_concepts("corr_v1", "x", parse_version_selector("latest"))[
            0
        ]
        assert concept.pointer is None

    def test_group_returns_every_member(self) -> None:
        body = concept_payload(
            concept_element(id=12, name="atc_c07ab"),
            concept_element(id=13, name="atc_c07ab"),
        )
        client = make_client(lambda _: httpx.Response(200, json=body))
        concepts = client.get_concepts(
            "corr_v1", "atc_c07ab", parse_version_selector("latest")
        )
        assert [concept.id for concept in concepts] == [12, 13]

    def test_object_body_is_rejected(self) -> None:
        client = make_client(lambda _: httpx.Response(200, json=concept_element()))
        with pytest.raises(ConceptsApiError, match="documented list"):
            client.get_concepts("corr_v1", "x", parse_version_selector("latest"))

    def test_version_warning_is_parsed(self) -> None:
        payload = concept_payload()
        payload[0]["sources"]["demo_source"]["version_info"]["warning"] = {
            "type": "critical_update",
            "corrected_in_version": 6,
            "message": "unit conversion was wrong",
        }
        client = make_client(lambda _: httpx.Response(200, json=payload))
        concept = client.get_concepts("corr_v1", "x", parse_version_selector("v4"))[0]
        warning = concept.sources["demo_source"].version_info.warning
        assert warning is not None
        assert warning.corrected_in_version == 6
        assert "unit conversion" in warning.message

    def test_unknown_keys_are_tolerated(self) -> None:
        payload = concept_payload()
        payload[0]["brand_new_field"] = "surprise"
        payload[0]["sources"]["demo_source"]["version_info"]["also_new"] = 1
        client = make_client(lambda _: httpx.Response(200, json=payload))
        assert client.get_concepts("corr_v1", "x", parse_version_selector("latest"))


class TestFileManifest:
    """The ``files`` block of a source config: what ``getfile`` may reach.

    A file lives in the source's library and is versioned there; a config
    version pins one of its versions. The manifest is that pinning, and it is
    the only thing that answers a ``getfile("<uuid>")`` in the snippet.
    """

    @staticmethod
    def _manifest_payload(**overrides: Any) -> dict[str, Any]:
        entry = {
            "uuid": FILE_UUID,
            "path": "postcode/postcode_mapping.csv",
            "version_no": 3,
            "size": 42,
            "sha256": "ab" * 32,
            "media_type": "text/csv",
            "url": f"/concept/id/17/files/{FILE_UUID}?v=4",
        }
        entry.update(overrides)
        payload = concept_payload()
        payload[0]["sources"]["demo_source"]["files"] = [entry]
        return payload

    def _fetch(self, payload: dict[str, Any]) -> ConceptFile:
        client = make_client(lambda _: httpx.Response(200, json=payload))
        concept = client.get_concepts(
            "corr_v1", "blood_sodium", parse_version_selector("latest")
        )[0]
        return concept.sources["demo_source"].files[0]

    def test_manifest_entry_is_parsed(self) -> None:
        file = self._fetch(self._manifest_payload())
        assert file.uuid == FILE_UUID
        assert file.version_no == 3
        assert file.path == "postcode/postcode_mapping.csv"
        assert file.sha256 == "ab" * 32
        assert file.media_type == "text/csv"
        assert file.url == f"/concept/id/17/files/{FILE_UUID}?v=4"

    def test_entry_without_a_uuid_is_fatal(self) -> None:
        payload = self._manifest_payload()
        del payload[0]["sources"]["demo_source"]["files"][0]["uuid"]
        with pytest.raises(ConceptsApiError, match="carries no uuid"):
            self._fetch(payload)

    def test_optional_fields_fall_back(self) -> None:
        payload = concept_payload()
        payload[0]["sources"]["demo_source"]["files"] = [
            {"uuid": FILE_UUID, "path": "a.csv"}
        ]
        file = self._fetch(payload)
        assert (file.version_no, file.size, file.sha256, file.url) == (0, 0, "", "")
        assert file.media_type == "application/octet-stream"


class TestCache:
    def test_repeat_requests_are_served_from_cache(self) -> None:
        calls = {"n": 0}

        def handler(_: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(200, json=concept_payload())

        client = make_client(handler)
        selector = parse_version_selector("latest")
        for _ in range(5):
            client.get_concepts("corr_v1", "blood_sodium", selector)
        assert calls["n"] == 1

    def test_different_selectors_are_cached_separately(self) -> None:
        calls = {"n": 0}

        def handler(_: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(200, json=concept_payload())

        client = make_client(handler)
        client.get_concepts("corr_v1", "x", parse_version_selector("latest"))
        client.get_concepts("corr_v1", "x", parse_version_selector("v3"))
        client.get_concepts("corr_v1", "x", parse_version_selector("latest"))
        assert calls["n"] == 2

    def test_different_names_are_cached_separately(self) -> None:
        calls = {"n": 0}

        def handler(_: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(200, json=concept_payload())

        client = make_client(handler)
        selector = parse_version_selector("latest")
        client.get_concepts("corr_v1", "a", selector)
        client.get_concepts("corr_v1", "b", selector)
        assert calls["n"] == 2


class TestFailurePolicy:
    def test_404_raises_concept_not_found(self) -> None:
        client = make_client(lambda _: httpx.Response(404, json={"detail": "nope"}))
        with pytest.raises(ConceptNotFoundError, match="blood_sodium"):
            client.get_concepts(
                "corr_v1", "blood_sodium", parse_version_selector("latest")
            )

    @pytest.mark.parametrize("status", [401, 403])
    def test_auth_failures_are_fatal_and_not_retried(self, status: int) -> None:
        calls = {"n": 0}

        def handler(_: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(status, json={"detail": "invalid token"})

        client = make_client(handler)
        with pytest.raises(ConceptsApiError, match="API key"):
            client.get_concepts("corr_v1", "x", parse_version_selector("latest"))
        assert calls["n"] == 1

    def test_license_rejection_is_its_own_error(self) -> None:
        client = make_client(
            lambda _: httpx.Response(
                403,
                json={"detail": "Project has not accepted the current CORR License."},
            )
        )
        with pytest.raises(ConceptsLicenseError, match="re-accept the CORR license"):
            client.get_concepts("corr_v1", "x", parse_version_selector("latest"))

    def test_license_rejection_names_the_project(self) -> None:
        client = make_client(
            lambda _: httpx.Response(403, json={"detail": "license_approval is 0"})
        )
        with pytest.raises(ConceptsLicenseError, match="'demo'"):
            client.get_concepts("corr_v1", "x", parse_version_selector("latest"))

    def test_license_error_is_a_concepts_api_error(self) -> None:
        assert issubclass(ConceptsLicenseError, ConceptsApiError)

    def test_license_rejection_is_not_retried(self) -> None:
        calls = {"n": 0}

        def handler(_: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(403, json={"detail": "License not accepted"})

        client = make_client(handler)
        with pytest.raises(ConceptsLicenseError):
            client.get_concepts("corr_v1", "x", parse_version_selector("latest"))
        assert calls["n"] == 1

    def test_client_error_is_fatal(self) -> None:
        client = make_client(lambda _: httpx.Response(400, text="bad selector"))
        with pytest.raises(ConceptsApiError, match="400"):
            client.get_concepts("corr_v1", "x", parse_version_selector("latest"))

    def test_plain_400_is_not_an_ambiguity(self) -> None:
        client = make_client(
            lambda _: httpx.Response(400, json={"detail": "unknown selector"})
        )
        with pytest.raises(ConceptsApiError) as excinfo:
            client.get_concepts("corr_v1", "x", parse_version_selector("v3"))
        assert not isinstance(excinfo.value, AmbiguousConceptError)

    def test_ambiguous_name_is_its_own_error(self) -> None:
        client = make_client(lambda _: httpx.Response(400, json=ambiguous_body()))
        with pytest.raises(AmbiguousConceptError) as excinfo:
            client.get_concepts("corr_v1", "atc_c07ab", parse_version_selector("v3"))

        message = str(excinfo.value)
        assert "12" in message and "13" in message
        assert "version pin" in message
        assert [member["id"] for member in excinfo.value.members] == [12, 13]

    def test_ambiguity_error_is_a_concepts_api_error(self) -> None:
        assert issubclass(AmbiguousConceptError, ConceptsApiError)

    def test_ambiguous_name_is_not_retried(self) -> None:
        calls = {"n": 0}

        def handler(_: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(400, json=ambiguous_body())

        client = make_client(handler)
        with pytest.raises(AmbiguousConceptError):
            client.get_concepts("corr_v1", "atc_c07ab", parse_version_selector("v3"))
        assert calls["n"] == 1

    def test_ambiguous_name_without_members_still_raises(self) -> None:
        body = {"detail": {"error": "ambiguous_name"}}
        client = make_client(lambda _: httpx.Response(400, json=body))
        with pytest.raises(AmbiguousConceptError) as excinfo:
            client.get_concepts("corr_v1", "atc_c07ab", parse_version_selector("v3"))
        assert excinfo.value.members == []

    def test_server_errors_are_retried_then_fatal(self) -> None:
        calls = {"n": 0}

        def handler(_: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(503)

        client = make_client(handler, max_retries=3)
        with pytest.raises(ConceptsApiError, match="3 attempts"):
            client.get_concepts("corr_v1", "x", parse_version_selector("latest"))
        assert calls["n"] == 3

    def test_transient_server_error_recovers(self) -> None:
        calls = {"n": 0}

        def handler(_: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(502)
            return httpx.Response(200, json=concept_payload())

        client = make_client(handler)
        assert client.get_concepts("corr_v1", "x", parse_version_selector("latest"))
        assert calls["n"] == 2

    def test_network_error_is_retried_then_fatal(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            raise httpx.ConnectError("no route to host", request=request)

        client = make_client(handler, max_retries=2)
        with pytest.raises(ConceptsApiError, match="never served from"):
            client.get_concepts("corr_v1", "x", parse_version_selector("latest"))
        assert calls["n"] == 2


class TestHistoryAndFiles:
    def test_history_is_addressed_by_id(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=[{"version": 1}])

        client = make_client(handler)
        assert client.get_history(17) == [{"version": 1}]
        assert seen[0].url.path == "/api/concept/id/17/history"
        assert seen[0].url.params["project"] == "demo"

    def test_history_object_payload(self) -> None:
        client = make_client(
            lambda _: httpx.Response(200, json={"history": [{"version": 2}]})
        )
        assert client.get_history(17) == [{"version": 2}]

    def test_fetch_file_without_url_addresses_the_uuid(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, content=b"a,b\n1,2\n")

        client = make_client(handler)
        file = ConceptFile(
            path="postcode/postcode_mapping.csv", uuid=FILE_UUID, sha256="abc"
        )
        content = client.fetch_file(
            "corr_v1", "patient_state_de", VersionSelector("version", 3), file
        )
        assert content == b"a,b\n1,2\n"
        assert seen[0].url.path == (
            f"/api/concept/corr_v1/patient_state_de/files/{FILE_UUID}"
        )
        assert seen[0].url.params["v"] == "3"
        assert seen[0].url.params["project"] == "demo"

    def test_fetch_file_without_url_or_uuid_is_fatal(self) -> None:
        client = make_client(lambda _: httpx.Response(200, content=b""))
        with pytest.raises(ConceptsApiError, match="neither a download url nor a uuid"):
            client.fetch_file(
                "corr_v1",
                "patient_state_de",
                VersionSelector("latest"),
                ConceptFile(path="a.csv", sha256="abc"),
            )

    def test_served_url_is_used_verbatim(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, content=b"payload")

        client = make_client(handler)
        file = ConceptFile(
            path="postcode/postcode_mapping.csv",
            sha256="abc",
            url=(
                f"{BASE_URL}/concept/corr_v1/patient_state_de/files/"
                "postcode/postcode_mapping.csv?source=demo_source&v=7"
            ),
        )
        # A different selector on the call must not override the pinned url.
        client.fetch_file(
            "corr_v1", "patient_state_de", VersionSelector("latest"), file
        )
        params = seen[0].url.params
        assert params["source"] == "demo_source"
        assert params["v"] == "7"
        assert params["project"] == "demo"

    def test_served_url_disambiguates_two_sources_sharing_a_path(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, content=b"payload")

        client = make_client(handler)
        for source, sha in (("demo_source", "a"), ("reprodicu", "b")):
            client.fetch_file(
                "corr_v1",
                "x",
                VersionSelector("latest"),
                ConceptFile(
                    path="shared.csv",
                    sha256=sha,
                    url=f"{BASE_URL}/concept/corr_v1/x/files/shared.csv?source={source}&v=2",
                ),
            )
        assert [r.url.params["source"] for r in seen] == ["demo_source", "reprodicu"]

    def test_fetch_file_without_url_prefers_the_id_route(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, content=b"payload")

        client = make_client(handler)
        client.fetch_file(
            "corr_v1",
            "atc_c07ab",
            VersionSelector("latest"),
            ConceptFile(path="a.csv", uuid=FILE_UUID, sha256="id-route"),
            concept_id=13,
        )
        assert seen[0].url.path == f"/api/concept/id/13/files/{FILE_UUID}"

    def test_fetch_file_is_cached_by_sha(self) -> None:
        calls = {"n": 0}

        def handler(_: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(200, content=b"payload")

        client = make_client(handler)
        file = ConceptFile(path="a.csv", uuid=FILE_UUID, sha256="deadbeef")
        client.fetch_file("corr_v1", "x", VersionSelector("latest"), file)
        client.fetch_file("corr_v1", "x", VersionSelector("latest"), file)
        assert calls["n"] == 1


class TestLifecycle:
    def test_deepcopy_returns_the_same_client(self) -> None:
        import copy

        client = make_client(lambda _: httpx.Response(200, json=concept_payload()))
        assert copy.deepcopy(client) is client

    def test_getstate_drops_the_api_key_and_caches(self) -> None:
        client = make_client(lambda _: httpx.Response(200, json=concept_payload()))
        client.get_concepts("corr_v1", "x", parse_version_selector("latest"))
        state = client.__getstate__()
        assert state["_api_key"] is None
        assert state["_http"] is None
        assert state["_concept_cache"] == {}


def projects_handler(
    *rows: dict[str, Any],
    status: int = 200,
    body: Any = None,
) -> Callable[[httpx.Request], httpx.Response]:
    """Answer ``GET /projects`` with `rows`, and 404 everything else.

    Args:
        *rows: Project rows the listing serves.
        status (int): Status for the listing.
        body (Any): Body for the listing, defaulting to `rows`.

    Returns:
        Callable[[httpx.Request], httpx.Response]: A mock transport handler.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/projects"):
            return httpx.Response(status, json=list(rows) if body is None else body)
        return httpx.Response(404, json={"detail": "no such concept"})

    return handler


class TestVerifyAccess:
    def test_a_listed_project_is_accepted(self) -> None:
        client = make_client(projects_handler({"name": "demo", "license_ok": True}))
        client.verify_access()

    def test_an_unknown_project_is_a_project_error(self) -> None:
        client = make_client(projects_handler({"name": "other", "license_ok": True}))
        with pytest.raises(ProjectNotFoundError, match="'demo'") as caught:
            client.verify_access()
        assert caught.value.project == "demo"
        assert caught.value.available == ["other"]

    def test_an_unknown_project_is_not_a_missing_variable(self) -> None:
        # The whole point: a misspelled project used to arrive as
        # VariableNotFoundError and sent users hunting for the wrong problem.
        client = make_client(projects_handler({"name": "other"}))
        with pytest.raises(ProjectNotFoundError) as caught:
            client.verify_access()
        assert not isinstance(caught.value, VariableNotFoundError)

    def test_the_error_lists_the_readable_projects(self) -> None:
        client = make_client(projects_handler({"name": "a"}, {"name": "b"}))
        with pytest.raises(ProjectNotFoundError, match="a, b"):
            client.verify_access()

    def test_a_case_mismatch_is_pointed_out(self) -> None:
        client = make_client(projects_handler({"name": "Demo"}))
        with pytest.raises(ProjectNotFoundError, match="Did you mean 'Demo'"):
            client.verify_access()

    def test_a_project_row_may_name_itself_by_slug(self) -> None:
        client = make_client(projects_handler({"slug": "demo"}))
        client.verify_access()

    def test_an_unlicensed_project_is_a_license_error(self) -> None:
        client = make_client(projects_handler({"name": "demo", "license_ok": False}))
        with pytest.raises(ConceptsLicenseError, match="re-accept the CORR license"):
            client.verify_access()

    def test_a_rejected_key_is_an_api_error(self) -> None:
        client = make_client(projects_handler(status=401, body={"detail": "nope"}))
        with pytest.raises(ConceptsApiError, match="rejected the API key") as caught:
            client.verify_access()
        assert "secret-token" not in str(caught.value)

    @pytest.mark.parametrize("status", [403, 404])
    def test_a_deployment_without_the_listing_does_not_fail(self, status: int) -> None:
        # Cannot tell, which must never read as "the project does not exist".
        client = make_client(projects_handler(status=status, body={"detail": "no"}))
        client.verify_access()

    def test_a_license_refusal_of_the_listing_is_reported(self) -> None:
        client = make_client(
            projects_handler(status=403, body={"detail": "License not accepted"})
        )
        with pytest.raises(ConceptsLicenseError):
            client.verify_access()

    def test_an_unreachable_endpoint_is_an_api_error(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route to host")

        client = make_client(handler, max_retries=1)
        with pytest.raises(ConceptsApiError, match="Could not reach"):
            client.verify_access()

    def test_the_listing_is_not_project_scoped(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=[{"name": "demo"}])

        make_client(handler).verify_access()
        assert seen[0].url.path == "/api/projects"
        assert "project" not in seen[0].url.params

    def test_the_result_is_remembered(self) -> None:
        calls = {"n": 0}

        def handler(_: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(200, json=[{"name": "demo"}])

        client = make_client(handler)
        client.verify_access()
        client.verify_access()
        assert calls["n"] == 1


class TestUnknownProjectOnRead:
    def test_a_404_under_an_unknown_project_names_the_project(self) -> None:
        client = make_client(projects_handler({"name": "other"}))
        with pytest.raises(ProjectNotFoundError, match="'demo'"):
            client.get_concepts(
                "corr_v1", "blood_sodium", parse_version_selector("latest")
            )

    def test_a_404_under_a_known_project_is_still_a_missing_concept(self) -> None:
        client = make_client(projects_handler({"name": "demo"}))
        with pytest.raises(ConceptNotFoundError, match="blood_sodium"):
            client.get_concepts(
                "corr_v1", "blood_sodium", parse_version_selector("latest")
            )

    def test_a_404_stays_a_missing_concept_when_the_check_cannot_run(self) -> None:
        # No project listing served: the 404 must be left as it was rather than
        # replaced by a guess.
        client = make_client(lambda _: httpx.Response(404, json={"detail": "no"}))
        with pytest.raises(ConceptNotFoundError):
            client.get_concepts(
                "corr_v1", "blood_sodium", parse_version_selector("latest")
            )


def concept_row(**overrides: Any) -> dict[str, Any]:
    """One element of the list ``GET /concepts`` serves."""
    payload: dict[str, Any] = {
        "id": 17,
        "taxonomy": "corr_v1",
        "name": "blood_sodium",
        "display_name": "Blood sodium",
        "sources": ["demo_source"],
        "types": ["native_dynamic"],
        "deprecated_at": None,
        "group_size": 1,
    }
    payload.update(overrides)
    return payload


class TestListConceptNames:
    def test_request_shape(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=[concept_row()])

        client = make_client(handler)
        client.list_concept_names("corr_v1")

        request = seen[0]
        assert request.url.path == "/api/concepts"
        assert dict(request.url.params) == {"project": "demo", "taxonomy": "corr_v1"}

    def test_names_are_fully_qualified(self) -> None:
        client = make_client(
            lambda _: httpx.Response(
                200,
                json=[
                    concept_row(name="blood_sodium"),
                    concept_row(id=18, name="heart_rate"),
                ],
            )
        )
        assert client.list_concept_names("corr_v1") == [
            "corr_v1/blood_sodium",
            "corr_v1/heart_rate",
        ]

    def test_deprecated_rows_are_skipped(self) -> None:
        client = make_client(
            lambda _: httpx.Response(
                200,
                json=[
                    concept_row(name="blood_sodium"),
                    concept_row(
                        id=18, name="old_sodium", deprecated_at="2024-05-06T10:00:00Z"
                    ),
                ],
            )
        )
        assert client.list_concept_names("corr_v1") == ["corr_v1/blood_sodium"]

    def test_duplicate_names_are_collapsed_in_order(self) -> None:
        client = make_client(
            lambda _: httpx.Response(
                200,
                json=[
                    concept_row(id=18, name="atc_c07ab", group_size=2),
                    concept_row(id=19, name="atc_c07ab", group_size=2),
                    concept_row(id=20, name="heart_rate"),
                ],
            )
        )
        assert client.list_concept_names("corr_v1") == [
            "corr_v1/atc_c07ab",
            "corr_v1/heart_rate",
        ]

    def test_source_filter_keeps_only_matching_rows(self) -> None:
        client = make_client(
            lambda _: httpx.Response(
                200,
                json=[
                    concept_row(name="blood_sodium", sources=["demo_source", "mimic"]),
                    concept_row(id=18, name="heart_rate", sources=["mimic"]),
                    concept_row(id=19, name="urine_output", sources=[]),
                ],
            )
        )
        assert client.list_concept_names("corr_v1", source="demo_source") == [
            "corr_v1/blood_sodium"
        ]

    def test_repeat_listings_are_served_from_cache(self) -> None:
        calls = {"n": 0}

        def handler(_: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(200, json=[concept_row()])

        client = make_client(handler)
        for _ in range(5):
            client.list_concept_names("corr_v1")
        assert calls["n"] == 1

    def test_taxonomy_and_source_are_cached_separately(self) -> None:
        calls = {"n": 0}

        def handler(_: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(200, json=[concept_row()])

        client = make_client(handler)
        client.list_concept_names("corr_v1")
        client.list_concept_names("corr_v2")
        client.list_concept_names("corr_v1", source="demo_source")
        assert calls["n"] == 3

    def test_object_body_is_rejected(self) -> None:
        client = make_client(lambda _: httpx.Response(200, json={"concepts": []}))
        with pytest.raises(ConceptsApiError, match="older than this client"):
            client.list_concept_names("corr_v1")

    def test_an_unknown_taxonomy_is_a_missing_concept(self) -> None:
        client = make_client(lambda _: httpx.Response(404, json={"detail": "nope"}))
        with pytest.raises(ConceptNotFoundError, match="corr_v1"):
            client.list_concept_names("corr_v1")

    def test_the_listing_cache_is_dropped_from_the_state(self) -> None:
        client = make_client(lambda _: httpx.Response(200, json=[concept_row()]))
        client.list_concept_names("corr_v1")
        assert client.__getstate__()["_concept_name_cache"] == {}
