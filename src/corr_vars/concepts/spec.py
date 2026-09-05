from __future__ import annotations

import re
from datetime import date

from dataclasses import dataclass
from typing import Final, Literal

VERSION_SEPARATOR: Final[str] = "::"
TAXONOMY_SEPARATOR: Final[str] = "/"
MEMBER_SEPARATOR: Final[str] = "#"

LATEST: Final[str] = "latest"

_VERSION_PATTERN: Final[re.Pattern[str]] = re.compile(r"^v(?P<number>\d+)$")
_DRAFT_PATTERN: Final[re.Pattern[str]] = re.compile(r"^draft(?P<number>\d+)$")
_DATE_PATTERN: Final[re.Pattern[str]] = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]*$")

_ACCEPTED_VERSIONS: Final[str] = (
    "'latest', 'vN' (e.g. 'v3'), an ISO date 'YYYY-MM-DD', or 'draftNNNN'"
)

VersionKind = Literal["latest", "version", "date", "draft"]


@dataclass(frozen=True)
class VersionSelector:
    """A resolved ``::version`` suffix, ready to be turned into query parameters.

    Args:
        kind (VersionKind): Which selector form was requested.
        value (int | date | None): The parsed payload — the version number for
            ``"version"``, the as-of date for ``"date"``, the Config id for
            ``"draft"``, and ``None`` for ``"latest"``.
    """

    kind: VersionKind
    value: int | date | None = None

    @property
    def query_params(self) -> dict[str, str]:
        """Query parameters selecting this version on the Concepts API.

        ``latest`` sends no selector at all; the API serves the newest
        committed version when none is given.

        Returns:
            dict[str, str]: Either ``{}``, ``{"v": ...}``, ``{"date": ...}``, or
            ``{"draft": ...}``.
        """
        if self.kind == "version":
            return {"v": str(self.value)}
        if self.kind == "date":
            return {"date": str(self.value)}
        if self.kind == "draft":
            return {"draft": str(self.value)}
        return {}

    @property
    def cache_key(self) -> tuple[str, str | None]:
        """Hashable identity of this selector, for the client response cache.

        Returns:
            tuple[str, str | None]: ``(kind, stringified value)``.
        """
        return (self.kind, None if self.value is None else str(self.value))

    def __str__(self) -> str:
        if self.kind == "version":
            return f"v{self.value}"
        if self.kind == "draft":
            return f"draft{self.value}"
        if self.kind == "date":
            return str(self.value)
        return LATEST


LATEST_SELECTOR: Final[VersionSelector] = VersionSelector(kind="latest")


@dataclass(frozen=True)
class VariableSpec:
    """A fully resolved variable reference.

    Args:
        name (str): Bare concept name, with taxonomy and version stripped.
        taxonomy (str): Taxonomy the concept is resolved in.
        version (VersionSelector): Which version of the concept to serve.
    """

    name: str
    taxonomy: str
    version: VersionSelector

    def __str__(self) -> str:
        return f"{self.taxonomy}/{self.name}::{self.version}"


def contributor_key(source: str, concept_id: object | None = None) -> str:
    """Build the key one contributor is filed under in a variable's config map.

    A variable is assembled from one configuration per contributor. Normally
    there is one contributor per source and the key is just the source name.
    When a name resolves to a group of concepts, each member contributes
    separately for the same source and is distinguished by its concept id.

    Args:
        source (str): Source name.
        concept_id (object | None): Concept id, for a group member. ``None``
            for a source's single contribution.

    Returns:
        str: Either ``source`` or ``"source#concept_id"``.
    """
    if concept_id is None:
        return source
    return f"{source}{MEMBER_SEPARATOR}{concept_id}"


def contributor_source(key: str) -> str:
    """Return the source name a contributor key belongs to.

    Args:
        key (str): A key produced by :func:`contributor_key`.

    Returns:
        str: The source name, with any group-member suffix stripped.
    """
    return key.split(MEMBER_SEPARATOR, maxsplit=1)[0]


def contributor_concept_id(key: str) -> str | None:
    """Return the concept id a contributor key carries, if any.

    Args:
        key (str): A key produced by :func:`contributor_key`.

    Returns:
        str | None: The concept id, or ``None`` for a plain source key.
    """
    parts = key.split(MEMBER_SEPARATOR, maxsplit=1)
    return parts[1] if len(parts) == 2 else None


def parse_version_selector(text: str) -> VersionSelector:
    """Parse the part of a reference that follows ``::``.

    Args:
        text (str): One of ``latest``, ``vN``, ``YYYY-MM-DD``, or ``draftNNNN``.

    Returns:
        VersionSelector: The parsed selector.

    Raises:
        ValueError: If `text` is empty, uses an unknown form, carries a
            non-positive number, or looks like a date but is not a real one.
    """
    stripped = text.strip()
    if not stripped:
        raise ValueError(
            f"Empty version in variable reference. Expected one of {_ACCEPTED_VERSIONS}."
        )
    if stripped != text:
        raise ValueError(
            f"Version {text!r} must not contain surrounding whitespace. "
            f"Expected one of {_ACCEPTED_VERSIONS}."
        )

    if stripped == LATEST:
        return LATEST_SELECTOR

    match = _VERSION_PATTERN.match(stripped)
    if match is not None:
        number = int(match.group("number"))
        if number < 1:
            raise ValueError(
                f"Version {stripped!r} is not valid: version numbers start at v1."
            )
        return VersionSelector(kind="version", value=number)

    match = _DRAFT_PATTERN.match(stripped)
    if match is not None:
        number = int(match.group("number"))
        if number < 1:
            raise ValueError(
                f"Draft {stripped!r} is not valid: draft ids are positive integers."
            )
        return VersionSelector(kind="draft", value=number)

    if _DATE_PATTERN.match(stripped) is not None:
        try:
            as_of = date.fromisoformat(stripped)
        except ValueError as exc:
            raise ValueError(
                f"Version {stripped!r} is not a valid calendar date (YYYY-MM-DD)."
            ) from exc
        return VersionSelector(kind="date", value=as_of)

    raise ValueError(
        f"Cannot parse version {stripped!r}. Expected one of {_ACCEPTED_VERSIONS}."
    )


def parse_variable_spec(
    reference: str,
    *,
    default_taxonomy: str,
    default_version: VersionSelector = LATEST_SELECTOR,
) -> VariableSpec:
    """Parse a ``[taxonomy/]var_name[::version]`` reference.

    Missing parts fall back to the cohort defaults, which themselves default to
    the values in the packaged routing configuration (``corr_v1`` / ``latest``).

    Args:
        reference (str): The reference to parse, e.g. ``"blood_sodium"``,
            ``"corr_v1/blood_sodium"``, or ``"blood_sodium::v3"``.
        default_taxonomy (str): Taxonomy used when the reference has no ``/``.
        default_version (VersionSelector): Selector used when the reference has
            no ``::`` suffix. Defaults to :data:`LATEST_SELECTOR`.

    Returns:
        VariableSpec: The reference with every part filled in.

    Raises:
        ValueError: If the reference is empty, carries more than one ``::`` or
            ``/``, has an empty taxonomy or name, or has an unparsable version.
    """
    if not isinstance(reference, str):
        raise ValueError(f"Variable reference must be a string, got {type(reference)}.")

    if not reference.strip():
        raise ValueError("Variable reference must not be empty.")

    parts = reference.split(VERSION_SEPARATOR)
    if len(parts) > 2:
        raise ValueError(
            f"Variable reference {reference!r} contains more than one "
            f"{VERSION_SEPARATOR!r} separator."
        )

    qualified_name = parts[0]
    version = default_version if len(parts) == 1 else parse_version_selector(parts[1])

    name_parts = qualified_name.split(TAXONOMY_SEPARATOR)
    if len(name_parts) > 2:
        raise ValueError(
            f"Variable reference {reference!r} contains more than one "
            f"{TAXONOMY_SEPARATOR!r} separator. Expected '[taxonomy/]var_name[::version]'."
        )

    if len(name_parts) == 2:
        taxonomy, name = name_parts
        if not taxonomy:
            raise ValueError(
                f"Variable reference {reference!r} has an empty taxonomy. "
                "Drop the leading '/' to use the cohort default."
            )
    else:
        taxonomy, name = default_taxonomy, name_parts[0]

    if not name:
        raise ValueError(
            f"Variable reference {reference!r} has an empty variable name."
        )

    if _NAME_PATTERN.match(name) is None:
        raise ValueError(
            f"Variable name {name!r} in reference {reference!r} is not a valid "
            "concept name (letters, digits, '_', '.' and '-' only)."
        )

    return VariableSpec(name=name, taxonomy=taxonomy, version=version)


def resolve_default_version(
    version: str | None,
    as_of_date: str | date | None,
) -> VersionSelector:
    """Build the cohort-wide default selector from ``version`` and ``date``.

    Args:
        version (str | None): A version selector string, e.g. ``"latest"`` or
            ``"v3"``. ``None`` is treated as ``"latest"``.
        as_of_date (str | date | None): A global as-of date. When given, every
            unqualified reference resolves to the version that was current on
            that date.

    Returns:
        VersionSelector: The default selector for the cohort.

    Raises:
        ValueError: If both an explicit non-``latest`` `version` and an
            `as_of_date` are given — the two select different things and the
            combination is ambiguous.
    """
    selector = LATEST_SELECTOR if version is None else parse_version_selector(version)

    if as_of_date is None:
        return selector

    if selector.kind != "latest":
        raise ValueError(
            f"Cohort was given both version={version!r} and date={as_of_date!r}. "
            "Pass only one: 'date' pins every variable to the version current on "
            "that day, 'version' pins them to an explicit version."
        )

    if isinstance(as_of_date, date):
        return VersionSelector(kind="date", value=as_of_date)

    return parse_version_selector(str(as_of_date))
