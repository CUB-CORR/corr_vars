from __future__ import annotations

import copy

from corr_vars import __, logger
from corr_vars.concepts.client import (
    Concept,
    ConceptsApiClient,
    SourceConcept,
    VersionWarning,
)
from corr_vars.concepts.compile import get_py_compiler
from corr_vars.concepts.files import materialise_files
from corr_vars.concepts.routing import ApiRoute, ConceptRouting, Route, load_routing
from corr_vars.concepts.spec import (
    LATEST_SELECTOR,
    VariableSpec,
    VersionSelector,
    contributor_key,
    parse_variable_spec,
    resolve_default_version,
)
from corr_vars.definitions.exceptions import (
    AmbiguousConceptError,
    ConceptNotFoundError,
    ConceptsApiConfigurationError,
    ConceptsApiError,
)

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datetime import date

    from collections.abc import Iterable


@dataclass(frozen=True)
class ResolvedConcept:
    """What a cohort recorded about one variable from one source.

    Persisted with the cohort so a reloaded cohort documents exactly which
    definitions produced its data.

    Args:
        taxonomy (str): Taxonomy the concept was resolved in.
        name (str): Concept name.
        source (str): Source key the definition came from.
        origin (str): ``"api"`` or ``"local"``.
        concept_id (str | int | None): Upstream concept id, which identifies the
            member a grouped name expanded into.
        endpoint (str | None): Base URL of the serving endpoint, for ``"api"``.
        requested (str): The selector the user asked for, e.g. ``"latest"``.
        version (int | None): Concept version the API served.
        source_version (int | None): Version of this source's definition.
        status (str | None): Lifecycle status of the served version.
        committed_at (str | None): ISO commit timestamp of the served version.
        deprecated_at (str | None): ISO timestamp the concept was deprecated.
        successor_id (str | int | None): Concept that replaces a deprecated one.
        pointer_deprecated_at (str | None): ISO timestamp the taxonomy entry the
            name resolved through was retired.
        warning (dict[str, Any] | None): ``version_info.warning``, when set.
    """

    taxonomy: str
    name: str
    source: str
    origin: str
    concept_id: str | int | None = None
    endpoint: str | None = None
    requested: str = "latest"
    version: int | None = None
    source_version: int | None = None
    status: str | None = None
    committed_at: str | None = None
    deprecated_at: str | None = None
    successor_id: str | int | None = None
    pointer_deprecated_at: str | None = None
    warning: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy, for ``Cohort.save()``.

        Returns:
            dict[str, Any]: The record as a plain dict.
        """
        return asdict(self)


@dataclass
class FetchedConfig:
    """A variable configuration plus the provenance of where it came from.

    Args:
        config (dict[str, Any]): Loader keyword arguments, equivalent to a
            ``vars.json`` entry with ``py`` resolved.
        resolution (ResolvedConcept): Provenance of the definition.
    """

    config: dict[str, Any]
    resolution: ResolvedConcept


def _timebounds_to_tuple(config: dict[str, Any]) -> dict[str, Any]:
    """Coerce JSON list ``tmin``/``tmax`` back into tuples, in place.

    JSON has no tuple type, so a ``("icu_admission", "+2h")`` anchor arrives as a
    two-element list. ``_transfer_time_window_from_variable_config`` and
    :func:`~corr_vars.definitions.typing.is_time_anchor_column` both require a
    tuple, so the API path has to restore it — the local path does the same via
    the ``object_hook`` in ``mapping/loader.py``.

    Args:
        config (dict[str, Any]): A variable configuration, mutated in place.

    Returns:
        dict[str, Any]: The same dict, for chaining.
    """
    for key in ("tmin", "tmax"):
        value = config.get(key)
        if isinstance(value, list):
            config[key] = tuple(value)

    requires = config.get("requires")
    if isinstance(requires, dict):
        for requirement in requires.values():
            if isinstance(requirement, dict):
                _timebounds_to_tuple(requirement)

    return config


def _format_version_warning(
    spec: VariableSpec, source: str, warning: VersionWarning
) -> str:
    corrected = (
        f"v{warning.corrected_in_version}"
        if warning.corrected_in_version is not None
        else "a later version"
    )
    return (
        f"Variable {spec.name!r} ({source}) was served at {spec.version} but "
        f"{corrected} contains a critical correction that supersedes it. "
        f"{warning.message}"
    )


class ConceptResolver:
    """Routes per-source variable definitions to an endpoint or the local mapping.

    One resolver belongs to one cohort. It owns the routing table, the cohort's
    spec defaults, and one :class:`~corr_vars.concepts.client.ConceptsApiClient`
    per endpoint. The clients memoise concept documents, so a ``requires`` chain
    that walks the same dependency from several parents costs one request.

    Warning:
        ``reprodicu`` routes to the Concepts API, but its definitions have not
        been imported there yet. Since a routed source never falls back to the
        bundled ``vars.json``, every reprodicu concept resolves to nothing until
        that import lands. This is a deliberate staging decision.

    Args:
        project (str): Project name, required by the API on every read.
        api_key (str | None): Bearer token; falls back to ``CORR_CONCEPTS_API_KEY``.
        taxonomy (str | None): Overrides the packaged default taxonomy.
        version (str | None): Overrides the packaged default version selector.
        date (str | datetime.date | None): Global as-of date. Mutually exclusive
            with an explicit non-``latest`` `version`.
        api_url (str | None): Overrides the endpoint URL from the routing file.
        routing (ConceptRouting | None): Injected routing table; loaded from the
            packaged ``routing.toml`` when ``None``.
        client_factory (Any): Callable ``(url) -> ConceptsApiClient``, injected by
            the test suite to avoid a live API.

    Raises:
        ValueError: If both `version` and `date` select a version.
    """

    def __init__(
        self,
        *,
        project: str | None = None,
        api_key: str | None = None,
        taxonomy: str | None = None,
        version: str | None = None,
        date: str | date | None = None,
        api_url: str | None = None,
        routing: ConceptRouting | None = None,
        client_factory: Any = None,
    ) -> None:
        base_routing = routing if routing is not None else load_routing()
        self.routing = base_routing.with_overrides(
            taxonomy=taxonomy, version=version, api_url=api_url
        )
        self.project = project.strip() if isinstance(project, str) else project
        self.default_taxonomy = self.routing.taxonomy
        self.default_version = resolve_default_version(self.routing.version, date)

        self._api_key = api_key
        self._client_factory = client_factory
        self._clients: dict[str, ConceptsApiClient] = {}

    # -- Routing -----------------------------------------------------------

    def route(self, source: str) -> Route:
        """Route a single source.

        Args:
            source (str): Source name.

        Returns:
            Route: The resolved route.
        """
        return self.routing.resolve(source)

    def is_remote(self, source: str) -> bool:
        """Report whether `source` is served by a Concepts API endpoint.

        Args:
            source (str): Source name.

        Returns:
            bool: ``True`` for API-routed sources.
        """
        return self.route(source).is_remote

    def client_for(self, route: ApiRoute) -> ConceptsApiClient:
        """Return the (cached) client for an endpoint.

        Args:
            route (ApiRoute): The endpoint to talk to.

        Returns:
            ConceptsApiClient: A client bound to the endpoint and project.

        Raises:
            ConceptsApiConfigurationError: If no project or API key is available.
        """
        client = self._clients.get(route.url)
        if client is not None:
            return client

        if self._client_factory is not None:
            client = self._client_factory(route.url)
        else:
            client = ConceptsApiClient(
                route.url, project=self.project or "", api_key=self._api_key
            )
        self._clients[route.url] = client
        return client

    def verify_credentials(self, sources: Iterable[str]) -> None:
        """Check that every API-routed source in `sources` has usable credentials.

        Building the client validates the project name and resolves the API key
        — including the ``CORR_CONCEPTS_API_KEY`` fallback — without issuing a
        request. Calling this while the cohort is being created turns a
        misconfiguration into an immediate, actionable error instead of one that
        surfaces on the first variable: after the (slow) initial data load, and
        after ``load_default_vars`` has swallowed the same error once per
        variable.

        Only the presence of the credentials is checked. Whether the endpoint
        accepts them is not, since that would need a request per cohort.

        Args:
            sources (Iterable[str]): Source names configured on the cohort.
                Sources routed local are ignored, so a
                cohort built only from those needs neither a project nor a key.

        Raises:
            ConceptsApiConfigurationError: If a source served by an endpoint has
                no project name, or no API key is available.
        """
        for route, endpoint_sources in self._api_routes(sources).items():
            try:
                self.client_for(route)
            except ConceptsApiConfigurationError as exc:
                raise ConceptsApiConfigurationError(
                    f"{exc} The source(s) {', '.join(sorted(endpoint_sources))} "
                    f"are served by the Concepts API at {route.url}."
                ) from exc

    def authenticate(self, sources: Iterable[str]) -> None:
        """Verify credentials, then prove them against every endpoint in use.

        Runs :meth:`verify_credentials` first, so a missing project or key is
        reported without a request, and then asks each endpoint that actually
        serves one of `sources` to confirm the key is accepted and the project
        exists. One request per endpoint, and only for endpoints in use — a
        cohort built solely from locally routed sources still touches no
        network.

        Args:
            sources (Iterable[str]): Source names configured on the cohort.

        Raises:
            ConceptsApiConfigurationError: If a source served by an endpoint has
                no project name, or no API key is available.
            ProjectNotFoundError: If an endpoint has no project of this name.
            ConceptsLicenseError: If the project's CORR license approval is
                missing or outdated.
            ConceptsApiError: If an endpoint rejects the key or cannot be
                reached.
        """
        sources = list(sources)
        self.verify_credentials(sources)

        for route in self._api_routes(sources):
            self.client_for(route).verify_access()

    def _api_routes(self, sources: Iterable[str]) -> dict[ApiRoute, list[str]]:
        """Group the API-routed members of `sources` by endpoint.

        Args:
            sources (Iterable[str]): Source names configured on the cohort.

        Returns:
            dict[ApiRoute, list[str]]: Sources per endpoint. Locally routed
            sources are absent.
        """
        by_endpoint: dict[ApiRoute, list[str]] = {}
        for source in sources:
            route = self.route(source)
            if isinstance(route, ApiRoute):
                by_endpoint.setdefault(route, []).append(source)
        return by_endpoint

    # -- Spec --------------------------------------------------------------

    def parse(self, reference: str) -> VariableSpec:
        """Parse a variable reference against this cohort's defaults.

        Args:
            reference (str): ``[taxonomy/]var_name[::version]``.

        Returns:
            VariableSpec: The fully resolved reference.

        Raises:
            ValueError: If the reference is malformed.
        """
        return parse_variable_spec(
            reference,
            default_taxonomy=self.default_taxonomy,
            default_version=self.default_version,
        )

    def default_spec(self, var_name: str) -> VariableSpec:
        """Build the spec a bare variable name resolves to.

        Used for ``requires`` dependencies, which always resolve at the cohort
        default rather than inheriting a pinned parent version.

        Args:
            var_name (str): Bare concept name.

        Returns:
            VariableSpec: The reference with cohort defaults filled in.
        """
        return VariableSpec(
            name=var_name,
            taxonomy=self.default_taxonomy,
            version=self.default_version,
        )

    # -- Fetching ----------------------------------------------------------

    def fetch_configs(
        self, spec: VariableSpec, sources: Iterable[str]
    ) -> dict[str, FetchedConfig]:
        """Fetch the definition of `spec` for every API-routed source in `sources`.

        The API serves one document per concept covering every source, so the
        sources sharing an endpoint cost a single request.

        A name resolving to several concepts is a group, and each member is
        expanded into its own contributor for every source it defines — the same
        shape an extra data source takes, so the frames concatenate downstream
        under identical column names. Members are keyed by
        :func:`~corr_vars.concepts.spec.contributor_key`, and each member's
        ``py`` snippet is compiled on its own.

        Args:
            spec (VariableSpec): The resolved variable reference.
            sources (Iterable[str]): Sources to fetch; locally routed ones are
                ignored.

        Returns:
            dict[str, FetchedConfig]: Per-contributor configuration and
            provenance. A source the concept does not define is absent, exactly
            as a source whose ``vars.json`` lacks the entry would be — but a
            concept the endpoint does not publish at all is logged at warning
            level, since that silently narrows a multi-source variable.

        Raises:
            AmbiguousConceptError: If `spec` pins a version and the name is a
                group. ``latest`` and ``::date`` expand instead.
            ConceptsApiError: On any transport, auth, or server failure. A
                routed source is never satisfied from the bundled ``vars.json``.
        """
        by_endpoint: dict[str, list[str]] = {}
        for source in sources:
            route = self.route(source)
            if isinstance(route, ApiRoute):
                by_endpoint.setdefault(route.url, []).append(source)

        fetched: dict[str, FetchedConfig] = {}
        for url, endpoint_sources in by_endpoint.items():
            client = self.client_for(ApiRoute(url=url))
            try:
                concepts = client.get_concepts(spec.taxonomy, spec.name, spec.version)
            except AmbiguousConceptError as exc:
                raise AmbiguousConceptError(
                    f"Variable {spec.name!r} was requested at {spec.version}, but "
                    f"{exc}",
                    members=exc.members,
                ) from exc
            except ConceptNotFoundError:
                # A 404 is indistinguishable from "this source has no entry for the
                # variable", which is the local behaviour we mirror. It is also what a
                # source routed to an endpoint that has imported nothing yet returns for
                # *every* concept — so the cohort would quietly be assembled from the
                # remaining sources alone. Loud by default: silently dropping a source
                # from a multi-source variable is the divergence the hard-fail policy
                # exists to prevent.
                logger.warning(
                    __(
                        "Concept {spec} is not published at {url}; sources {sources} "
                        "contribute nothing to this variable. The variable is built "
                        "from the remaining sources only.",
                        spec=spec,
                        url=url,
                        sources=sorted(endpoint_sources),
                    )
                )
                continue

            grouped = len(concepts) > 1
            if grouped:
                logger.info(
                    __(
                        "Concept {spec} is a group of {count} concepts ({ids}); each "
                        "member contributes to the variable like an additional source.",
                        spec=spec,
                        count=len(concepts),
                        ids=[concept.id for concept in concepts],
                    )
                )

            for concept in concepts:
                self._warn_on_deprecation(spec, concept)
                for source in endpoint_sources:
                    source_concept = concept.sources.get(source)
                    if source_concept is None:
                        continue
                    key = contributor_key(source, concept.id if grouped else None)
                    fetched[key] = self._build_config(
                        spec, source, source_concept, concept, url, client
                    )

        return fetched

    def _warn_on_deprecation(self, spec: VariableSpec, concept: Concept) -> None:
        """Log the deprecation state a concept was served with.

        A deprecated concept and a retired taxonomy entry both still load — a
        name that once worked keeps working — so the flags only reach the user
        as warnings.

        Args:
            spec (VariableSpec): The resolved reference.
            concept (Concept): The served document.
        """
        if concept.deprecated:
            logger.warning(
                __(
                    "Concept {spec} (id {concept_id}) was deprecated on {deprecated_at}"
                    "{successor}. Its definition still loads.",
                    spec=spec,
                    concept_id=concept.id,
                    deprecated_at=concept.deprecated_at,
                    successor=(
                        f" and replaced by concept {concept.successor_id}"
                        if concept.successor_id is not None
                        else ""
                    ),
                )
            )

        if concept.pointer is not None and concept.pointer.deprecated:
            logger.warning(
                __(
                    "Name {name!r} resolves to concept {concept_id} only through a "
                    "taxonomy entry retired on {deprecated_at}. The definition still "
                    "loads under that name.",
                    name=spec.name,
                    concept_id=concept.id,
                    deprecated_at=concept.pointer.deprecated_at,
                )
            )

    def _build_config(
        self,
        spec: VariableSpec,
        source: str,
        source_concept: SourceConcept,
        concept: Concept,
        endpoint: str,
        client: ConceptsApiClient,
    ) -> FetchedConfig:
        """Turn one served source payload into loader keyword arguments.

        Runs once per concept, so a group member's ``py`` snippet is compiled
        against that member's own attached files and never merged with another
        member's.

        Args:
            spec (VariableSpec): The resolved reference.
            source (str): Source key.
            source_concept (SourceConcept): The served per-source payload.
            concept (Concept): The full document, for version metadata.
            endpoint (str): Base URL that served the document.
            client (ConceptsApiClient): Client used to download attached files.

        Returns:
            FetchedConfig: Configuration plus provenance.

        Raises:
            ConceptsApiError: If a ``py`` snippet is served for a source with no
                ``py_env`` module, or if it fails to compile.
        """
        config = _timebounds_to_tuple(copy.deepcopy(source_concept.json))

        if source_concept.py:
            compiler = get_py_compiler(source)
            if compiler is None:
                raise ConceptsApiError(
                    f"Concept {spec} carries a py snippet for source {source!r}, but "
                    f"corr_vars.sources.{source} ships no py_env module declaring the "
                    "namespace such snippets execute against."
                )
            # Everything the manifest pins is downloaded before the snippet is
            # compiled, so `getfile` is a dict lookup at extraction time and a
            # second resolve of the same definition needs no network at all.
            files_dir, files_by_uuid = materialise_files(
                source_concept.files,
                spec=spec,
                source=source,
                client=client,
                concept_id=concept.id,
            )
            config["py"] = compiler(
                source_concept.py,
                spec.name,
                files_dir,
                files_by_uuid=files_by_uuid,
            )

        info = source_concept.version_info
        resolution = ResolvedConcept(
            taxonomy=spec.taxonomy,
            name=spec.name,
            source=source,
            origin="api",
            concept_id=concept.id,
            endpoint=endpoint,
            requested=str(spec.version),
            version=concept.version,
            source_version=info.source_version,
            status=info.status,
            committed_at=info.committed_at,
            deprecated_at=concept.deprecated_at,
            successor_id=concept.successor_id,
            pointer_deprecated_at=(
                concept.pointer.deprecated_at if concept.pointer is not None else None
            ),
            warning=(
                {
                    "type": info.warning.type,
                    "corrected_in_version": info.warning.corrected_in_version,
                    "message": info.warning.message,
                }
                if info.warning
                else None
            ),
        )

        if info.warning is not None:
            logger.warning(_format_version_warning(spec, source, info.warning))

        return FetchedConfig(config=config, resolution=resolution)

    # -- Lifecycle ---------------------------------------------------------

    def settings(self) -> dict[str, Any]:
        """Return the JSON-serialisable settings needed to rebuild the resolver.

        The API key is deliberately excluded so it never reaches a saved cohort.

        Returns:
            dict[str, Any]: Project, taxonomy, and the default version selector.
        """
        return {
            "project": self.project,
            "taxonomy": self.default_taxonomy,
            "version": str(self.default_version),
        }

    def __deepcopy__(self, memo: dict[int, Any]) -> ConceptResolver:
        """Return the receiver unchanged.

        The resolver owns live HTTP clients and their caches, and the cohort is
        deep-copied on every ``py`` call.

        Args:
            memo (dict[int, Any]): Standard :mod:`copy` memo, unused.

        Returns:
            ConceptResolver: The receiver itself.
        """
        return self

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_api_key"] = None
        state["_client_factory"] = None
        state["_clients"] = {}
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def __repr__(self) -> str:
        return (
            f"ConceptResolver(project={self.project!r}, "
            f"taxonomy={self.default_taxonomy!r}, "
            f"version={self.default_version}, "
            f"endpoints={[ep.url for ep in self.routing.endpoints]})"
        )


__all__ = [
    "ConceptResolver",
    "FetchedConfig",
    "ResolvedConcept",
    "LATEST_SELECTOR",
    "VersionSelector",
]
