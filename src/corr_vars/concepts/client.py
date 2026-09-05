from __future__ import annotations

import os
import random
import time

import httpx

from corr_vars import __, logger
from corr_vars.definitions.exceptions import (
    AmbiguousConceptError,
    ConceptNotFoundError,
    ConceptsApiConfigurationError,
    ConceptsApiError,
    ConceptsLicenseError,
    ProjectNotFoundError,
)

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from corr_vars.concepts.spec import VersionSelector

API_KEY_ENV_VAR: Final[str] = "CORR_CONCEPTS_API_KEY"

DEFAULT_TIMEOUT: Final[httpx.Timeout] = httpx.Timeout(
    connect=5.0, read=30.0, write=30.0, pool=5.0
)
DEFAULT_MAX_RETRIES: Final[int] = 3
DEFAULT_BACKOFF: Final[float] = 0.5
RETRY_STATUS_CODES: Final[frozenset[int]] = frozenset(
    {408, 425, 429, 500, 502, 503, 504}
)

#: Substrings that identify a 403 caused by the CORR license rather than by the
#: API key. The API refuses external reads for a project whose license approval
#: is missing or older than the current license version.
LICENSE_MARKERS: Final[tuple[str, ...]] = ("license",)

#: ``detail.error`` value the API sends when a name resolves to several concepts
#: and the request pins a version.
AMBIGUOUS_NAME_ERROR: Final[str] = "ambiguous_name"

#: Listing of the projects an API key may read. Not project-scoped: it is the
#: one route that answers before a project name has been established, which is
#: what makes it usable as an authentication handshake.
PROJECTS_PATH: Final[str] = "/projects"

#: Listing of every concept in a taxonomy. Project-scoped like the other reads;
#: the ``taxonomy`` query parameter is optional and the API defaults it.
CONCEPTS_PATH: Final[str] = "/concepts"

#: Keys a project row may carry its name under, in preference order.
PROJECT_NAME_KEYS: Final[tuple[str, ...]] = ("name", "slug", "key", "project")


def _is_license_rejection(response: httpx.Response) -> bool:
    """Report whether a 403 is a license refusal rather than an auth failure.

    Args:
        response (httpx.Response): The rejected response.

    Returns:
        bool: ``True`` when the body mentions the license.
    """
    body = response.text.lower()
    return any(marker in body for marker in LICENSE_MARKERS)


def _ambiguous_members(response: httpx.Response) -> list[dict[str, Any]] | None:
    """Extract the group members from an ``ambiguous_name`` rejection.

    Args:
        response (httpx.Response): A 400 response.

    Returns:
        list[dict[str, Any]] | None: The members the name resolved to, or
        ``None`` when the body is an ordinary 400 rather than this one.
    """
    try:
        body = response.json()
    except ValueError:
        return None

    if not isinstance(body, Mapping):
        return None

    detail = body.get("detail")
    if not isinstance(detail, Mapping) or detail.get("error") != AMBIGUOUS_NAME_ERROR:
        return None

    members = detail.get("members")
    if not isinstance(members, list):
        return []
    return [dict(member) for member in members if isinstance(member, Mapping)]


def _project_names(row: Any) -> list[str]:
    """Names one project row can be addressed by.

    A deployment may file the addressable slug under any of
    :data:`PROJECT_NAME_KEYS`, so every candidate is collected rather than one
    key being guessed at.

    Args:
        row (Any): One element of the ``GET /projects`` payload.

    Returns:
        list[str]: The non-empty names, stripped.
    """
    if not isinstance(row, Mapping):
        return [str(row).strip()] if str(row).strip() else []
    names = []
    for key in PROJECT_NAME_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            names.append(value.strip())
    return names


def _describe_members(members: Sequence[Mapping[str, Any]]) -> str:
    """Render group members as ``id (name)`` pairs for an error message.

    Args:
        members (Sequence[Mapping[str, Any]]): Member records from the API.

    Returns:
        str: A comma-separated listing, or ``"none reported"`` when empty.
    """
    if not members:
        return "none reported"
    return ", ".join(
        f"{member.get('id')} ({member.get('name') or member.get('display_name')})"
        for member in members
    )


@dataclass(frozen=True)
class VersionWarning:
    """A notice that a later critical version supersedes the version served.

    Args:
        type (str): Machine-readable warning kind reported by the API.
        corrected_in_version (int | None): Version that carries the correction.
        message (str): Human-readable explanation, shown to the user verbatim.
    """

    type: str
    corrected_in_version: int | None
    message: str

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> VersionWarning | None:
        """Build a warning from the ``version_info.warning`` payload.

        Args:
            payload (Mapping[str, Any] | None): The raw warning object, or ``None``.

        Returns:
            VersionWarning | None: The parsed warning, or ``None`` when absent.
        """
        if not payload:
            return None
        return cls(
            type=str(payload.get("type", "critical_update")),
            corrected_in_version=payload.get("corrected_in_version"),
            message=str(payload.get("message", "")),
        )


@dataclass(frozen=True)
class VersionInfo:
    """Provenance of the source definition the API served.

    Args:
        source_version (int | None): Version number of this source's definition.
        type (str | None): Definition type as recorded upstream.
        read_only (bool): Whether the upstream definition is frozen.
        change_type (str | None): Classification of the last change.
        message (str | None): Commit message of the served version.
        author (str | None): Who committed the served version.
        committed_at (str | None): ISO timestamp of the commit.
        status (str | None): Lifecycle status, e.g. ``"committed"`` or ``"draft"``.
        warning (VersionWarning | None): Set when a later critical version
            supersedes the one served.
    """

    source_version: int | None = None
    type: str | None = None
    read_only: bool = False
    change_type: str | None = None
    message: str | None = None
    author: str | None = None
    committed_at: str | None = None
    status: str | None = None
    warning: VersionWarning | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> VersionInfo:
        """Build version info from the ``version_info`` payload.

        Unknown keys are ignored so the client survives additive API changes.

        Args:
            payload (Mapping[str, Any] | None): The raw ``version_info`` object.

        Returns:
            VersionInfo: The parsed provenance record.
        """
        payload = payload or {}
        return cls(
            source_version=payload.get("source_version"),
            type=payload.get("type"),
            read_only=bool(payload.get("read_only", False)),
            change_type=payload.get("change_type"),
            message=payload.get("message"),
            author=payload.get("author"),
            committed_at=payload.get("committed_at"),
            status=payload.get("status"),
            warning=VersionWarning.from_payload(payload.get("warning")),
        )


@dataclass(frozen=True)
class ConceptFile:
    """One entry of a source config's file manifest.

    Data files live in the **source's** library and are versioned there; a
    snippet names one by ``uuid`` (``getfile("<uuid>")``) and the config version
    it belongs to pins that file at ``version_no``. The manifest is therefore the
    complete set of files a snippet may reach, at the exact bytes it was
    published against.

    Args:
        uuid (str): The file's stable id — what ``getfile("…")`` names.
        path (str): Path relative to the variable's file directory, e.g.
            ``"postcode/postcode_mapping.csv"``. Only a layout hint now: nothing
            addresses a file by it.
        version_no (int): File version this config version pins.
        size (int): Size in bytes, as reported by the API.
        sha256 (str): Content hash, used as the on-disk cache key.
        media_type (str): MIME type reported by the API.
        url (str): Absolute or endpoint-relative download URL.
    """

    path: str
    uuid: str = ""
    version_no: int = 0
    size: int = 0
    sha256: str = ""
    media_type: str = "application/octet-stream"
    url: str = ""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ConceptFile:
        """Build a file record from one entry of the ``files`` list.

        Args:
            payload (Mapping[str, Any]): The raw file object.

        Returns:
            ConceptFile: The parsed record.

        Raises:
            ConceptsApiError: If the entry carries no ``uuid``. Downloads are
                uuid-addressed and ``getfile`` looks a file up by it, so a
                manifest entry without one names nothing.
        """
        uuid = str(payload.get("uuid") or "")
        path = str(payload.get("path") or "")
        if not uuid:
            raise ConceptsApiError(
                f"Concept file entry {path or '<no path>'!r} carries no uuid. The "
                "Concepts API addresses data files by uuid; an entry without one "
                "cannot be downloaded or reached by getfile()."
            )
        return cls(
            path=path,
            uuid=uuid,
            version_no=int(payload.get("version_no", 0) or 0),
            size=int(payload.get("size", 0)),
            sha256=str(payload.get("sha256", "")),
            media_type=str(payload.get("media_type", "application/octet-stream")),
            url=str(payload.get("url", "")),
        )


@dataclass(frozen=True)
class SourceConcept:
    """The per-source payload of a concept document.

    Args:
        source (str): Source key this payload was filed under.
        json (dict[str, Any]): The declarative definition, equivalent to a
            ``vars.json`` entry.
        py (str | None): Bare function source text, or ``None``.
        files (tuple[ConceptFile, ...]): Attached data files.
        version_info (VersionInfo): Provenance of the served definition.
    """

    source: str
    json: dict[str, Any] = field(default_factory=dict)
    py: str | None = None
    files: tuple[ConceptFile, ...] = ()
    version_info: VersionInfo = field(default_factory=VersionInfo)

    @classmethod
    def from_payload(cls, source: str, payload: Mapping[str, Any]) -> SourceConcept:
        """Build a source payload from one entry of the ``sources`` mapping.

        Args:
            source (str): Source key.
            payload (Mapping[str, Any]): The raw per-source object.

        Returns:
            SourceConcept: The parsed payload.
        """
        return cls(
            source=source,
            json=dict(payload.get("json") or {}),
            py=payload.get("py"),
            files=tuple(
                ConceptFile.from_payload(item) for item in payload.get("files") or ()
            ),
            version_info=VersionInfo.from_payload(payload.get("version_info")),
        )


@dataclass(frozen=True)
class ConceptPointer:
    """The taxonomy entry a concept was resolved through.

    Taxonomy entries are append-only pointers, so one name can point at several
    concepts and one concept can carry several names. The pointer records which
    of those entries served this element, and whether it is still active.

    Args:
        id (str | int | None): Upstream pointer id.
        identifier (str): The name this pointer files the concept under.
        display_name (str | None): Label shown for the pointer.
        relationship (str | None): How the name relates to the concept, e.g.
            ``"primary"`` or ``"alias"``.
        created_at (str | None): ISO timestamp the pointer became active.
        deprecated_at (str | None): ISO timestamp the pointer was retired, or
            ``None`` while it is active.
    """

    id: str | int | None = None
    identifier: str = ""
    display_name: str | None = None
    relationship: str | None = None
    created_at: str | None = None
    deprecated_at: str | None = None

    @property
    def deprecated(self) -> bool:
        """Report whether this pointer has been retired.

        Returns:
            bool: ``True`` when the pointer carries a ``deprecated_at``.
        """
        return self.deprecated_at is not None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> ConceptPointer | None:
        """Build a pointer from the ``pointer`` payload.

        Args:
            payload (Mapping[str, Any] | None): The raw pointer object.

        Returns:
            ConceptPointer | None: The parsed pointer, or ``None`` when absent.
        """
        if not payload:
            return None
        return cls(
            id=payload.get("id"),
            identifier=str(payload.get("identifier", "")),
            display_name=payload.get("display_name"),
            relationship=payload.get("relationship"),
            created_at=payload.get("created_at"),
            deprecated_at=payload.get("deprecated_at"),
        )


@dataclass(frozen=True)
class Concept:
    """One element of the list served by ``GET /concept/{taxonomy}/{name}``.

    Args:
        id (str | int | None): Upstream concept id, which is the concept's
            identity — a name may resolve to several of them.
        taxonomy (str): Taxonomy the concept lives in.
        name (str): Concept name.
        version (int | None): Version number of the served document.
        requested (dict[str, Any]): Echo of the selector the API resolved.
        sources (dict[str, SourceConcept]): Per-source definitions.
        pointer (ConceptPointer | None): The taxonomy entry this element
            resolved through.
        deprecated_at (str | None): ISO timestamp the concept itself was
            deprecated, or ``None``.
        successor_id (str | int | None): Concept that replaces a deprecated one,
            already resolved to the final successor.
        doc_clinical (str | None): Clinical documentation.
        doc_implementation (str | None): Implementation notes.
        doc_caveats (str | None): Known caveats.
        doc_status (str | None): Documentation status.
        notion_url (str | None): Link to the upstream Notion page.
    """

    id: str | int | None
    taxonomy: str
    name: str
    version: int | None = None
    requested: dict[str, Any] = field(default_factory=dict)
    sources: dict[str, SourceConcept] = field(default_factory=dict)
    pointer: ConceptPointer | None = None
    deprecated_at: str | None = None
    successor_id: str | int | None = None
    doc_clinical: str | None = None
    doc_implementation: str | None = None
    doc_caveats: str | None = None
    doc_status: str | None = None
    notion_url: str | None = None

    @property
    def deprecated(self) -> bool:
        """Report whether the concept itself has been deprecated.

        Returns:
            bool: ``True`` when the concept carries a ``deprecated_at``.
        """
        return self.deprecated_at is not None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Concept:
        """Build a concept from one element of a decoded API response body.

        Args:
            payload (Mapping[str, Any]): The decoded JSON object.

        Returns:
            Concept: The parsed document.
        """
        return cls(
            id=payload.get("id"),
            taxonomy=str(payload.get("taxonomy", "")),
            name=str(payload.get("name", "")),
            version=payload.get("version"),
            requested=dict(payload.get("requested") or {}),
            sources={
                source: SourceConcept.from_payload(source, source_payload)
                for source, source_payload in (payload.get("sources") or {}).items()
            },
            pointer=ConceptPointer.from_payload(payload.get("pointer")),
            deprecated_at=payload.get("deprecated_at"),
            successor_id=payload.get("successor_id"),
            doc_clinical=payload.get("doc_clinical"),
            doc_implementation=payload.get("doc_implementation"),
            doc_caveats=payload.get("doc_caveats"),
            doc_status=payload.get("doc_status"),
            notion_url=payload.get("notion_url"),
        )


class ConceptsApiClient:
    """Thin HTTP client for the Concepts API.

    One instance serves one endpoint for one project. Concept documents are
    memoised in-process under ``(endpoint, taxonomy, name, selector)`` because
    ``requires`` chains request the same concepts over and over — without the
    cache a single derived variable would re-fetch each dependency for every
    parent that declares it. One cache entry holds the whole list a name
    resolves to, so a grouped name costs a single request too.

    Every failure is fatal: a source routed to an endpoint is never satisfied
    from the bundled ``vars.json``, so callers get a
    :exc:`~corr_vars.definitions.exceptions.ConceptsApiError` rather than a
    cohort whose recorded version metadata does not describe its data.

    Args:
        base_url (str): Endpoint base URL, e.g. ``"https://concepts.example.edu/api"``.
        project (str): Project name; the API requires it on every read.
        api_key (str | None): Bearer token. Falls back to the
            ``CORR_CONCEPTS_API_KEY`` environment variable.
        timeout (httpx.Timeout | float | None): Per-request timeouts.
        max_retries (int): Attempts for transient failures (network errors and
            the retryable status codes in :data:`RETRY_STATUS_CODES`).
        backoff (float): Base delay in seconds for exponential backoff.
        transport (httpx.BaseTransport | None): Injected transport, used by the
            test suite to serve responses without a live API.

    Raises:
        ConceptsApiConfigurationError: If `project` is empty or no API key is
            available.
    """

    def __init__(
        self,
        base_url: str,
        *,
        project: str,
        api_key: str | None = None,
        timeout: httpx.Timeout | float | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff: float = DEFAULT_BACKOFF,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not isinstance(project, str) or not project.strip():
            raise ConceptsApiConfigurationError(
                "The Concepts API requires a project on every read. "
                "Pass Cohort(project='<name>')."
            )

        # Keys read from an environment file or pasted into a notebook routinely
        # carry surrounding whitespace, which the endpoint answers with an
        # opaque 401 rather than anything pointing at the key.
        resolved_key = (api_key or os.environ.get(API_KEY_ENV_VAR) or "").strip()
        if not resolved_key:
            raise ConceptsApiConfigurationError(
                "No Concepts API key. Pass Cohort(api_key='...') or set the "
                f"{API_KEY_ENV_VAR} environment variable."
            )

        self.base_url = base_url.rstrip("/")
        self.project = project.strip()
        self.max_retries = max(1, max_retries)
        self.backoff = backoff

        self._api_key = resolved_key
        self._timeout = DEFAULT_TIMEOUT if timeout is None else timeout
        self._transport = transport
        self._http: httpx.Client | None = None
        self._concept_cache: dict[
            tuple[str, str, str, tuple[str, str | None]], list[Concept]
        ] = {}
        self._concept_name_cache: dict[tuple[str, str, str | None], list[str]] = {}
        self._blob_cache: dict[str, bytes] = {}
        self._access_verified = False

    # -- HTTP plumbing -----------------------------------------------------

    @property
    def http(self) -> httpx.Client:
        """The lazily created underlying HTTP client.

        Returns:
            httpx.Client: A client carrying the bearer token and base URL.
        """
        if self._http is None:
            self._http = httpx.Client(
                base_url=self.base_url,
                timeout=self._timeout,
                transport=self._transport,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Accept": "application/json",
                },
            )
        return self._http

    def close(self) -> None:
        """Close the underlying HTTP client, if one was created."""
        if self._http is not None:
            self._http.close()
            self._http = None

    def _sleep(self, attempt: int) -> None:
        delay = self.backoff * (2**attempt) * (1 + random.random() * 0.1)
        time.sleep(delay)

    def _request(
        self,
        url: str,
        params: Mapping[str, str],
        *,
        context: str,
        project_scoped: bool = True,
        soft_statuses: frozenset[int] = frozenset(),
    ) -> httpx.Response:
        """Issue a GET with retries, translating failures into fatal errors.

        Args:
            url (str): Endpoint-relative path beginning with ``/``, or an
                absolute URL served by the API. Query parameters already present
                on `url` are preserved and `params` is merged on top.
            params (Mapping[str, str]): Query parameters, without ``project``.
            context (str): Human-readable description used in error messages.
            project_scoped (bool): Whether to attribute the read to this
                client's project. ``False`` for the routes that answer before a
                project has been established, such as the project listing.
            soft_statuses (frozenset[int]): Statuses returned to the caller
                instead of raising, so it can phrase its own error. Transient
                statuses stay retryable and are only handed over once the retry
                budget is spent.

        Raises:
            ConceptNotFoundError: On 404.
            ConceptsLicenseError: On a 403 caused by the CORR license.
            AmbiguousConceptError: On a 400 whose body reports that the name
                resolves to several concepts.
            ConceptsApiError: On any other non-2xx status, or after the retry
                budget for transient failures is exhausted.
        """
        merged = {"project": self.project, **params} if project_scoped else dict(params)
        target = httpx.URL(url).copy_merge_params(merged)
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = self.http.get(target)
            except httpx.TransportError as exc:
                last_error = exc
                logger.debug(
                    __(
                        "Concepts API transport error for {context} (attempt {attempt}/{total}): {exc}",
                        context=context,
                        attempt=attempt + 1,
                        total=self.max_retries,
                        exc=exc,
                    )
                )
                if attempt + 1 < self.max_retries:
                    self._sleep(attempt)
                    continue
                break

            if (
                response.status_code in soft_statuses
                and response.status_code not in RETRY_STATUS_CODES
            ):
                return response

            if response.status_code == 404:
                raise ConceptNotFoundError(
                    f"The Concepts API at {self.base_url} has no {context} "
                    f"for project {self.project!r}."
                )

            if response.status_code == 403 and _is_license_rejection(response):
                raise ConceptsLicenseError(
                    f"Project {self.project!r} may not read concepts: the CORR "
                    f"license has not been accepted for it, or the accepted "
                    f"version is older than the current one. A project lead must "
                    f"re-accept the CORR license on the project page before this "
                    f"project can query definitions. "
                    f"(HTTP 403 from {self.base_url} for {context}: "
                    f"{response.text[:300]})"
                )

            if response.status_code in (401, 403):
                raise ConceptsApiError(
                    f"The Concepts API at {self.base_url} rejected the request for "
                    f"{context} (HTTP {response.status_code}). Check the API key "
                    f"and that project {self.project!r} is readable for it."
                )

            if response.status_code in RETRY_STATUS_CODES:
                last_error = ConceptsApiError(
                    f"HTTP {response.status_code} from {self.base_url} for {context}"
                )
                if attempt + 1 < self.max_retries:
                    self._sleep(attempt)
                    continue
                if response.status_code in soft_statuses:
                    return response
                break

            if response.status_code == 400:
                members = _ambiguous_members(response)
                if members is not None:
                    raise AmbiguousConceptError(
                        f"The name behind {context} points at several concepts: "
                        f"{_describe_members(members)}. A version pin addresses one "
                        f"concept's version counter, so it cannot target a grouped "
                        f"name. Drop the '::vN'/'::draftN' pin to load every member, "
                        f"pin by date instead, or name a single member.",
                        members=members,
                    )

            if response.status_code >= 400:
                raise ConceptsApiError(
                    f"The Concepts API at {self.base_url} returned HTTP "
                    f"{response.status_code} for {context}: {response.text[:500]}"
                )

            return response

        raise ConceptsApiError(
            f"Could not reach the Concepts API at {self.base_url} for {context} "
            f"after {self.max_retries} attempts. Definitions are never served from "
            f"the bundled vars.json for a routed source, so this is fatal. "
            f"Last error: {last_error}"
        ) from last_error

    # -- Authentication ----------------------------------------------------

    def list_projects(self) -> list[dict[str, Any]] | None:
        """Fetch the projects this API key may read.

        Returns:
            list[dict[str, Any]] | None: One row per project, or ``None`` when
            the deployment does not serve the listing (404) or refuses it to
            this key (403). ``None`` means "cannot tell", never "no projects".

        Raises:
            ConceptsApiError: If the key itself is rejected (401), or the
                endpoint cannot be reached at all.
            ConceptsLicenseError: If the refusal names the CORR license.
        """
        response = self._request(
            PROJECTS_PATH,
            {},
            context="the project listing",
            project_scoped=False,
            soft_statuses=frozenset({401, 403, 404}),
        )

        if response.status_code == 401:
            raise ConceptsApiError(
                f"The Concepts API at {self.base_url} rejected the API key "
                f"(HTTP 401). Pass a valid Cohort(api_key='...') or set the "
                f"{API_KEY_ENV_VAR} environment variable."
            )

        if response.status_code == 403 and _is_license_rejection(response):
            raise ConceptsLicenseError(
                f"The Concepts API at {self.base_url} refused the project "
                f"listing on license grounds: {response.text[:300]}"
            )

        if response.status_code in (403, 404):
            # An older deployment, or a key without the permission to list.
            # Either way the project name cannot be checked here; the first
            # concept read will still report an unknown project.
            logger.debug(
                __(
                    "The Concepts API at {url} does not serve {path} (HTTP "
                    "{status}); the project name cannot be verified up front.",
                    url=self.base_url,
                    path=PROJECTS_PATH,
                    status=response.status_code,
                )
            )
            return None

        try:
            payload = response.json()
        except ValueError:
            payload = None

        if not isinstance(payload, list):
            logger.debug(
                __(
                    "The Concepts API at {url} answered {path} with a {kind} "
                    "rather than a list; skipping the project check.",
                    url=self.base_url,
                    path=PROJECTS_PATH,
                    kind=type(payload).__name__,
                )
            )
            return None
        return [
            dict(row) if isinstance(row, Mapping) else {"name": row} for row in payload
        ]

    def _raise_if_project_unknown(self) -> None:
        """Re-report a 404 as a project error when the project is the cause.

        Called while a :exc:`ConceptNotFoundError` is being handled, so a
        failure to answer the question leaves that error in place rather than
        replacing it with a second, less relevant one.

        Raises:
            ProjectNotFoundError: If the endpoint has no project of this name.
            ConceptsLicenseError: If the project's CORR license approval is
                missing or outdated.
        """
        try:
            self.verify_access()
        except (ProjectNotFoundError, ConceptsLicenseError):
            raise
        except ConceptsApiError as exc:
            logger.debug(
                __(
                    "Could not check whether project {project} exists: {exc}",
                    project=self.project,
                    exc=exc,
                )
            )

    def verify_access(self) -> None:
        """Authenticate the key and check the project exists, before any read.

        One request. It answers the two questions a misconfigured cohort would
        otherwise ask far too late: is the key accepted, and does this project
        exist for it. Both are checked up front because the endpoint answers a
        read for an unknown project with the same 404 it uses for an unknown
        concept — which the resolver treats as "this source does not define the
        variable" and turns into an empty cohort.

        The result is remembered, so repeated calls across a cohort's sources
        cost a single request.

        Raises:
            ConceptsApiError: If the key is rejected or the endpoint cannot be
                reached.
            ProjectNotFoundError: If the endpoint has no project of this name.
            ConceptsLicenseError: If the project exists but its CORR license
                approval is missing or outdated.
        """
        if self._access_verified:
            return

        projects = self.list_projects()
        if projects is None:
            self._access_verified = True
            return

        known = {name: row for row in projects for name in _project_names(row)}
        row = known.get(self.project)
        if row is None:
            folded = {name.casefold(): name for name in known}
            match = folded.get(self.project.casefold())
            hint = (
                f" Did you mean {match!r}? Project names are case-sensitive."
                if match
                else (
                    f" Projects readable with this key: "
                    f"{', '.join(sorted(known)) or 'none'}."
                )
            )
            raise ProjectNotFoundError(
                f"The Concepts API at {self.base_url} has no project "
                f"{self.project!r}.{hint}",
                project=self.project,
                available=sorted(known),
            )

        if row.get("license_ok") is False:
            raise ConceptsLicenseError(
                f"Project {self.project!r} may not read concepts: the CORR "
                f"license has not been accepted for it, or the accepted version "
                f"is older than the current one. A project lead must re-accept "
                f"the CORR license on the project page before this project can "
                f"query definitions."
            )

        logger.debug(
            __(
                "Concepts API at {url} authenticated for project {project}.",
                url=self.base_url,
                project=self.project,
            )
        )
        self._access_verified = True

    # -- Reads -------------------------------------------------------------

    def get_concepts(
        self, taxonomy: str, name: str, version: VersionSelector
    ) -> list[Concept]:
        """Fetch every concept a name resolves to, serving repeats from the cache.

        A name normally resolves to exactly one concept. It resolves to several
        when the taxonomy files a group of them under one identifier, in which
        case every member is returned, ordered by concept id.

        Args:
            taxonomy (str): Taxonomy to resolve in.
            name (str): Concept name.
            version (VersionSelector): Which version to request.

        Returns:
            list[Concept]: The parsed documents, one per resolved concept.

        Raises:
            ConceptNotFoundError: If the name resolves to nothing.
            AmbiguousConceptError: If `version` pins a version and the name
                resolves to several concepts.
            ConceptsApiError: On any other API or transport failure, including a
                body that is not the documented list.
        """
        key = (self.base_url, taxonomy, name, version.cache_key)
        cached = self._concept_cache.get(key)
        if cached is not None:
            logger.debug(
                __(
                    "Concepts API cache hit for {taxonomy}/{name}::{version}",
                    taxonomy=taxonomy,
                    name=name,
                    version=version,
                )
            )
            return cached

        context = f"concept {taxonomy}/{name}::{version}"
        try:
            response = self._request(
                f"/concept/{taxonomy}/{name}", version.query_params, context=context
            )
        except ConceptNotFoundError:
            # The endpoint answers an unknown project with the same 404 as an
            # unknown concept, and the resolver reads that as "this source does
            # not define the variable". Ask once which of the two it was, so a
            # misspelled project does not surface as a missing variable.
            self._raise_if_project_unknown()
            raise
        payload = response.json()
        if not isinstance(payload, list):
            raise ConceptsApiError(
                f"The Concepts API at {self.base_url} answered {context} with a "
                f"{type(payload).__name__} rather than the documented list of "
                f"concepts. The endpoint is older than this client."
            )

        concepts = [Concept.from_payload(element) for element in payload]
        self._concept_cache[key] = concepts
        return concepts

    def list_concept_names(
        self, taxonomy: str, *, source: str | None = None
    ) -> list[str]:
        """List the fully-qualified names of a taxonomy's live concepts.

        Deprecated rows are dropped: they are still served so that an existing
        pin keeps resolving, but nothing new should be written against them.
        Names are returned fully qualified (``"<taxonomy>/<name>"``) because a
        bare name only addresses a concept once a taxonomy is known, and
        de-duplicated in order, since a taxonomy may file several concepts under
        one name.

        The listing is memoised in-process under
        ``(endpoint, taxonomy, source)``, like the concept documents are.

        Args:
            taxonomy (str): Taxonomy to list.
            source (str | None): When given, keep only concepts that carry a
                definition for this source key.

        Returns:
            list[str]: The fully-qualified names, in the order the API served
            them.

        Raises:
            ConceptNotFoundError: If the taxonomy is unknown to the endpoint.
            ConceptsLicenseError: If the project may not read concepts.
            ConceptsApiError: On any other API or transport failure, including a
                body that is not the documented list.
        """
        key = (self.base_url, taxonomy, source)
        cached = self._concept_name_cache.get(key)
        if cached is not None:
            logger.debug(
                __(
                    "Concepts API cache hit for the {taxonomy} listing",
                    taxonomy=taxonomy,
                )
            )
            return cached

        context = f"the concept listing of taxonomy {taxonomy!r}"
        response = self._request(CONCEPTS_PATH, {"taxonomy": taxonomy}, context=context)
        payload = response.json()
        if not isinstance(payload, list):
            raise ConceptsApiError(
                f"The Concepts API at {self.base_url} answered {context} with a "
                f"{type(payload).__name__} rather than the documented list of "
                f"concepts. The endpoint is older than this client."
            )

        names: list[str] = []
        seen: set[str] = set()
        for row in payload:
            if not isinstance(row, Mapping):
                continue
            if row.get("deprecated_at") is not None:
                continue
            if source is not None and source not in (row.get("sources") or ()):
                continue
            name = str(row.get("name") or "").strip()
            if not name:
                continue
            qualified = f"{taxonomy}/{name}"
            if qualified in seen:
                continue
            seen.add(qualified)
            names.append(qualified)

        logger.debug(
            __(
                "Concepts API listed {count} concepts for taxonomy {taxonomy}",
                count=len(names),
                taxonomy=taxonomy,
            )
        )
        self._concept_name_cache[key] = names
        return names

    def get_history(self, concept_id: str | int) -> list[dict[str, Any]]:
        """Fetch the version history of one concept.

        Addressed by id rather than by name, since a name may point at several
        concepts and a history belongs to exactly one of them.

        Args:
            concept_id (str | int): Upstream concept id.

        Returns:
            list[dict[str, Any]]: The history entries as returned by the API.

        Raises:
            ConceptNotFoundError: If the concept does not exist.
            ConceptsApiError: On any other API or transport failure.
        """
        response = self._request(
            f"/concept/id/{concept_id}/history",
            {},
            context=f"history of concept {concept_id}",
        )
        payload = response.json()
        if isinstance(payload, dict):
            return list(payload.get("history") or [])
        return list(payload)

    def fetch_file(
        self,
        taxonomy: str,
        name: str,
        version: VersionSelector,
        file: ConceptFile,
        concept_id: str | int | None = None,
    ) -> bytes:
        """Download one file of a source config's manifest.

        Downloads are addressed by file **uuid**, never by path: a file belongs
        to the source's library and the config version pins one of its versions,
        so the path is a layout hint and nothing more.

        The ``url`` on the file record is used verbatim when present. The API
        pins it to the version that was actually served — reconstructing the URL
        from `taxonomy`, `name` and `version` would be both redundant and wrong,
        since a ``date`` selector re-resolved later can point at other bytes.
        Only ``project`` is merged in, since the API requires it on every read.
        A record without a ``url`` is fetched from the id route when
        `concept_id` is known, since a name may point at several concepts.

        Args:
            taxonomy (str): Taxonomy to resolve in.
            name (str): Concept name.
            version (VersionSelector): Version the file belongs to. Used only
                when the record carries no ``url``.
            file (ConceptFile): The manifest entry from the concept document.
            concept_id (str | int | None): Id of the concept the file belongs
                to. Used only when the record carries no ``url``.

        Returns:
            bytes: The raw file content.

        Raises:
            ConceptNotFoundError: If the file does not exist.
            ConceptsLicenseError: If the project may not read concepts.
            ConceptsApiError: On any other API or transport failure, or if the
                record carries neither a ``url`` nor a ``uuid`` to build one
                from.
        """
        if file.sha256 and file.sha256 in self._blob_cache:
            return self._blob_cache[file.sha256]

        if file.url:
            url, params = file.url, {}
        elif not file.uuid:
            raise ConceptsApiError(
                f"Concept file {file.path!r} of {taxonomy}/{name} carries neither a "
                "download url nor a uuid to build one from."
            )
        elif concept_id is not None:
            url = f"/concept/id/{concept_id}/files/{file.uuid}"
            params = version.query_params
        else:
            url = f"/concept/{taxonomy}/{name}/files/{file.uuid}"
            params = version.query_params

        response = self._request(
            url,
            params,
            context=(
                f"file {file.uuid or file.path!r} of concept "
                f"{taxonomy}/{name}::{version}"
            ),
        )
        content = response.content
        if file.sha256:
            self._blob_cache[file.sha256] = content
        return content

    # -- Lifecycle ---------------------------------------------------------

    def __deepcopy__(self, memo: dict[int, Any]) -> ConceptsApiClient:
        """Return the receiver unchanged.

        The client wraps a live connection pool and an in-process cache; both
        are shared services, and ``PyFuncStep`` deep-copies the cohort on every
        ``py`` call.

        Args:
            memo (dict[int, Any]): Standard :mod:`copy` memo, unused.

        Returns:
            ConceptsApiClient: The receiver itself.
        """
        return self

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_http"] = None
        state["_transport"] = None
        state["_api_key"] = None
        state["_concept_cache"] = {}
        state["_concept_name_cache"] = {}
        state["_blob_cache"] = {}
        state["_access_verified"] = False
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        # Archives written before the access check carry no such flag.
        self.__dict__.setdefault("_access_verified", False)
        # Archives written before the taxonomy listing carry no such cache.
        self.__dict__.setdefault("_concept_name_cache", {})

    def __repr__(self) -> str:
        return (
            f"ConceptsApiClient(base_url={self.base_url!r}, project={self.project!r}, "
            f"cached={len(self._concept_cache)})"
        )
