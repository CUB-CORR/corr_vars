from __future__ import annotations

import copy
from importlib import resources

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Final

try:  # Python >= 3.11
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10 only
    import tomli as tomllib  # type: ignore[no-redef]


ROUTING_FILE: Final[str] = "routing.toml"
WILDCARD: Final[str] = "*"


@dataclass(frozen=True)
class LocalRoute:
    """A source whose definitions are read from its bundled ``mapping`` module.

    Args:
        reason (str): Why the source resolved to local — ``"pinned"`` for an
            entry in ``[local].sources``, ``"unmatched"`` for the fallback.
    """

    reason: str = "unmatched"

    @property
    def is_remote(self) -> bool:
        return False


@dataclass(frozen=True)
class ApiRoute:
    """A source whose definitions are fetched from a Concepts API endpoint.

    Args:
        url (str): Base URL of the endpoint, without a trailing slash.
    """

    url: str

    @property
    def is_remote(self) -> bool:
        return True


Route = LocalRoute | ApiRoute


@dataclass(frozen=True)
class Endpoint:
    """One ``[[endpoint]]`` table from :data:`ROUTING_FILE`.

    Args:
        url (str): Base URL of the Concepts API.
        sources (tuple[str, ...]): Source names served by this endpoint, or
            ``("*",)`` for every source not pinned local.
    """

    url: str
    sources: tuple[str, ...]

    def matches(self, source: str) -> bool:
        """Report whether this endpoint claims `source`.

        Args:
            source (str): Source name to test.

        Returns:
            bool: ``True`` if the source is listed explicitly or covered by the
            ``"*"`` wildcard.
        """
        return source in self.sources or WILDCARD in self.sources


@dataclass(frozen=True)
class ConceptRouting:
    """Resolved contents of the packaged routing configuration.

    A source resolves to the first matching rule: pinned-local first, then the
    ordered endpoint list, then local as the fallback for anything unmatched.

    Warning:
        ``reprodicu`` matches the wildcard endpoint and therefore routes to the
        Concepts API — **but its definitions have not been imported there yet**.
        Together with the hard-fail policy (a routed source never falls back to
        the bundled ``vars.json``) this means every reprodicu variable resolves
        to nothing until that import lands, and a reprodicu-only variable raises
        :exc:`~corr_vars.definitions.exceptions.ConceptNotFoundError`. This is a
        deliberate, approved staging decision — not a bug to debug.

    Args:
        taxonomy (str): Default taxonomy for unqualified variable references.
        version (str): Default version selector for references without ``::``.
        local_sources (tuple[str, ...]): Sources pinned to their bundled mapping.
        endpoints (tuple[Endpoint, ...]): Endpoint rules in file order.
    """

    taxonomy: str
    version: str
    local_sources: tuple[str, ...]
    endpoints: tuple[Endpoint, ...]

    def resolve(self, source: str) -> Route:
        """Route a single source.

        Args:
            source (str): Source name, e.g. ``"reprodicu"``.

        Returns:
            Route: :class:`ApiRoute` when an endpoint claims the source,
            :class:`LocalRoute` when it is pinned or unmatched.
        """
        if source in self.local_sources:
            return LocalRoute(reason="pinned")

        for endpoint in self.endpoints:
            if endpoint.matches(source):
                return ApiRoute(url=endpoint.url)

        return LocalRoute(reason="unmatched")

    def resolve_all(self, sources: Iterable[str]) -> dict[str, Route]:
        """Route every source in `sources`.

        Args:
            sources (Iterable[str]): Source names to route.

        Returns:
            dict[str, Route]: Mapping from source name to its route.
        """
        return {source: self.resolve(source) for source in sources}

    def with_overrides(
        self,
        *,
        taxonomy: str | None = None,
        version: str | None = None,
        api_url: str | None = None,
    ) -> ConceptRouting:
        """Return a copy with ``Cohort`` keyword arguments applied.

        This is the only supported way to deviate from the packaged file.

        Args:
            taxonomy (str | None): Replacement default taxonomy.
            version (str | None): Replacement default version selector.
            api_url (str | None): Replacement base URL for **every** endpoint,
                for pointing a cohort at a staging or on-premise deployment.
                Pinned-local sources are unaffected.

        Returns:
            ConceptRouting: A new routing object; the receiver is unchanged.
        """
        endpoints = self.endpoints
        if api_url is not None:
            url = api_url.rstrip("/")
            endpoints = tuple(replace(ep, url=url) for ep in endpoints)

        return ConceptRouting(
            taxonomy=taxonomy if taxonomy is not None else self.taxonomy,
            version=version if version is not None else self.version,
            local_sources=self.local_sources,
            endpoints=endpoints,
        )


def _as_source_tuple(value: Any, context: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{ROUTING_FILE}: {context} must be a list of source names")
    if not all(isinstance(item, str) for item in value):
        raise TypeError(f"{ROUTING_FILE}: {context} must contain only strings")
    return tuple(value)


def parse_routing(raw: Mapping[str, Any]) -> ConceptRouting:
    """Validate and convert the parsed TOML document.

    Args:
        raw (Mapping[str, Any]): Parsed contents of :data:`ROUTING_FILE`.

    Returns:
        ConceptRouting: The validated routing table.

    Raises:
        KeyError: If ``[defaults]`` is missing ``taxonomy`` or ``version``.
        TypeError: If a ``sources`` list or an endpoint ``url`` has the wrong type.
    """
    defaults = raw.get("defaults", {})
    for key in ("taxonomy", "version"):
        if key not in defaults:
            raise KeyError(f"{ROUTING_FILE}: [defaults] is missing {key!r}")

    local_sources = _as_source_tuple(
        raw.get("local", {}).get("sources", []), "[local].sources"
    )

    endpoints: list[Endpoint] = []
    for index, entry in enumerate(raw.get("endpoint", [])):
        url = entry.get("url")
        if not isinstance(url, str) or not url:
            raise TypeError(f"{ROUTING_FILE}: [[endpoint]] #{index} needs a string url")
        endpoints.append(
            Endpoint(
                url=url.rstrip("/"),
                sources=_as_source_tuple(
                    entry.get("sources", []), f"[[endpoint]] #{index} sources"
                ),
            )
        )

    return ConceptRouting(
        taxonomy=str(defaults["taxonomy"]),
        version=str(defaults["version"]),
        local_sources=local_sources,
        endpoints=tuple(endpoints),
    )


def load_routing_document() -> dict[str, Any]:
    """Read :data:`ROUTING_FILE` from the installed package.

    Returns:
        dict[str, Any]: The raw parsed TOML document.
    """
    package = ".".join(__name__.split(".")[:-1])
    with resources.files(package).joinpath(ROUTING_FILE).open("rb") as handle:
        return copy.deepcopy(tomllib.load(handle))


def load_routing() -> ConceptRouting:
    """Load and validate the packaged routing configuration.

    Returns:
        ConceptRouting: The global routing table, identical for every user of
        the package.
    """
    return parse_routing(load_routing_document())
