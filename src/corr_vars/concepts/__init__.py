"""Fetching variable definitions from the CORR Concepts API.

A variable definition has two halves: a declarative ``json`` config (what
``vars.json`` holds locally) and an optional ``py`` transformation function
(what ``mapping/variables.py`` holds locally). The Concepts API serves both,
per concept and per source, together with the data files a definition needs and
the version metadata describing exactly what was served.

The package is organised as:

- :mod:`~corr_vars.concepts.routing` — the packaged ``routing.toml``, deciding
  per source whether definitions come from an endpoint or stay local.
- :mod:`~corr_vars.concepts.spec` — parsing ``[taxonomy/]var_name[::version]``.
- :mod:`~corr_vars.concepts.client` — the HTTP client and response types.
- :mod:`~corr_vars.concepts.files` — the content-addressed file cache.
- :mod:`~corr_vars.concepts.compile` — executing served ``py`` snippets.
- :mod:`~corr_vars.concepts.resolver` — the per-cohort façade over all of it.

Warning:
    Sources routed to an endpoint **never** fall back to the bundled
    ``vars.json``. A cohort assembled from a stale local definition while
    recording a remote version number would misdescribe its own data, which is
    the problem the API exists to solve. Every failure is therefore fatal.

Warning:
    ``reprodicu`` is routed to the API but its definitions have not been
    imported there yet, so every reprodicu concept resolves to nothing until
    that import lands. This is a deliberate staging decision, not a bug.
"""

from corr_vars.concepts.client import (
    Concept,
    ConceptFile,
    ConceptsApiClient,
    SourceConcept,
    VersionInfo,
    VersionWarning,
)
from corr_vars.concepts.compile import (
    CompiledVariableFunction,
    compile_snippet,
    get_py_compiler,
)
from corr_vars.concepts.files import cache_root, materialise_files
from corr_vars.concepts.resolver import (
    ConceptResolver,
    FetchedConfig,
    ResolvedConcept,
)
from corr_vars.concepts.routing import (
    ApiRoute,
    ConceptRouting,
    Endpoint,
    LocalRoute,
    Route,
    load_routing,
)
from corr_vars.concepts.spec import (
    LATEST_SELECTOR,
    VariableSpec,
    VersionSelector,
    parse_variable_spec,
    parse_version_selector,
    resolve_default_version,
)

__all__ = [
    # Routing
    "ApiRoute",
    "ConceptRouting",
    "Endpoint",
    "LocalRoute",
    "Route",
    "load_routing",
    # Spec
    "LATEST_SELECTOR",
    "VariableSpec",
    "VersionSelector",
    "parse_variable_spec",
    "parse_version_selector",
    "resolve_default_version",
    # Client
    "Concept",
    "ConceptFile",
    "ConceptsApiClient",
    "SourceConcept",
    "VersionInfo",
    "VersionWarning",
    # Files
    "cache_root",
    "materialise_files",
    # Compilation
    "CompiledVariableFunction",
    "compile_snippet",
    "get_py_compiler",
    # Resolver
    "ConceptResolver",
    "FetchedConfig",
    "ResolvedConcept",
]
