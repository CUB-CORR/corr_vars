from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class CohortDataError(Exception):
    """Raised upon cohort data errors."""


class ObsLevelNotSupportedError(Exception):
    """Raised upon incompatible obs level."""


class VariableNotFoundError(LookupError):
    """Raised when a variable is not found in the cohort."""


class VariableDefinitionError(Exception):
    """Raised when a variable definition cannot be turned into an extraction.

    Covers the parts of a definition that only become executable at extraction
    time — a source's ``filter`` or ``calculation`` expression, for instance.
    Such a failure is never recoverable locally: continuing without the failed
    piece would extract something other than what the definition describes, and
    the resulting cohort would carry version metadata for a definition it does
    not match.
    """


class ConceptsApiError(Exception):
    """Raised when a source routed to the Concepts API cannot be served.

    Sources routed to an endpoint never fall back to the bundled ``vars.json``:
    a cohort built from a stale local definition would record version metadata
    that does not describe the code that produced it. Every transport, auth, or
    server error therefore surfaces as this exception instead.
    """


class ConceptsApiConfigurationError(ConceptsApiError):
    """Raised when the Concepts API client is missing mandatory configuration.

    Typically a missing ``project`` (required on every read) or a missing API
    key (``Cohort(api_key=...)`` or the ``CORR_CONCEPTS_API_KEY`` environment
    variable).
    """


class ProjectNotFoundError(ConceptsApiConfigurationError):
    """Raised when the Concepts API does not know the configured project.

    Deliberately **not** a :class:`VariableNotFoundError`: the endpoint answers
    a read for an unknown project with the same 404 it uses for an unknown
    concept, which used to surface as "variable not found" and sent users
    looking for a misspelled variable instead of a misspelled project.

    Args:
        message (str): Human-readable explanation, shown to the user verbatim.
        project (str): The project name that was rejected.
        available (Sequence[str]): Project names the API key can see, when the
            deployment lists them.

    Attributes:
        project (str): The project name that was rejected.
        available (list[str]): Project names the API key can see.
    """

    def __init__(
        self, message: str, project: str = "", available: Sequence[str] = ()
    ) -> None:
        super().__init__(message)
        self.project: str = project
        self.available: list[str] = [str(name) for name in available]


class ConceptsLicenseError(ConceptsApiError):
    """Raised when a project may not read concepts because of the CORR license.

    The API refuses every external read for a project whose license approval is
    missing or older than the current license version. No corr_vars setting can
    work around it: a project lead has to re-accept the CORR license on the
    project page in the browser app.
    """


class AmbiguousConceptError(ConceptsApiError):
    """Raised when a version-pinned request names a concept group.

    A name may point at several concepts — mostly ATC codes, where one code
    covers a set of substances. Such a group expands like an extra data source
    and loads fine at ``latest`` or at an as-of date, but ``::vN`` and
    ``::draftN`` address a single concept's version counter and therefore have
    no meaning for the group as a whole. The API refuses those requests and this
    carries the members it refused over.

    Args:
        message (str): Human-readable explanation, shown to the user verbatim.
        members (Sequence[Mapping[str, Any]]): The concepts the name resolved
            to, each with ``id``, ``name``, ``display_name`` and ``description``.

    Attributes:
        members (list[dict[str, Any]]): The concepts the name resolved to.
    """

    def __init__(self, message: str, members: Sequence[Mapping[str, Any]] = ()) -> None:
        super().__init__(message)
        self.members: list[dict[str, Any]] = [dict(member) for member in members]


class ConceptNotFoundError(ConceptsApiError, VariableNotFoundError):
    """Raised when the Concepts API has no concept under the requested reference.

    Inherits from :class:`VariableNotFoundError` so callers that already handle
    "unknown variable" keep working, and from :class:`ConceptsApiError` so the
    remote origin of the failure stays visible.
    """
