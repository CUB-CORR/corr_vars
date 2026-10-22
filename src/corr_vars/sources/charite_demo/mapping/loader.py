"""What the charite_demo ``mapping`` package reads from disk, and what it fetches.

The variable definitions are served by the Concepts API: there is no bundled
``vars.json`` and no ``variables.py``, and nothing here falls back to one. What stays local is the configuration that is a corr_vars decision rather
than a clinical concept — ``globals.json``, ``tables.json`` and the default
variable list in ``defaults.json`` — and it is loaded eagerly, so importing the
package needs no network, no project and no API key.

The one thing that *is* remote is the list of variable names charite_demo offers.
:func:`fetch_all_var_names` asks the endpoint for it, and
:mod:`corr_vars.sources.charite_demo.mapping` exposes it as the lazy module attribute
``ALL_VAR_NAMES`` so that merely importing the package never reaches out.
"""

from __future__ import annotations

import copy
import json
import os
from importlib import resources

from corr_vars.definitions.exceptions import ConceptsApiConfigurationError

from typing import TYPE_CHECKING, Any, Final, TypeGuard

if TYPE_CHECKING:
    from collections.abc import Callable

#: Source key charite_demo's concepts are filed under in the Concepts API.
SOURCE: Final[str] = "charite_demo"

#: Environment variable naming the project every Concepts API read is scoped to.
PROJECT_ENV_VAR: Final[str] = "CORR_CONCEPTS_PROJECT"

#: Memoised result of :func:`fetch_all_var_names`, for the life of the process.
#: Keyed by project, since two projects may see different concepts.
_ALL_VAR_NAMES_CACHE: dict[str, list[str]] = {}


def load_json(
    name: str, object_hook: Callable[[dict[Any, Any]], Any] | None = None
) -> Any:
    # For Python 3.9+
    with (
        resources.files(".".join(__name__.split(".")[:-1]))
        .joinpath(f"{name}.json")
        .open("r") as f
    ):
        return copy.deepcopy(json.load(f, object_hook=object_hook))


def _is_table_type(value: Any) -> TypeGuard[dict[str, dict[str, str]]]:
    return isinstance(value, dict) and all(
        isinstance(k, str)
        and isinstance(v, dict)
        and all(isinstance(kk, str) and isinstance(vv, str) for kk, vv in v.items())
        for k, v in value.items()
    )


def _check_tables_type(value: Any) -> dict[str, dict[str, str]]:
    if not _is_table_type(value):
        raise TypeError("tables.json must be dict[str, dict[str, str]]")
    return value


def _resolve_project(project: str | None) -> str:
    """Resolve the Concepts API project to list charite_demo's concepts for.

    Args:
        project (str | None): Explicit project, usually from the caller's cohort.

    Returns:
        str: The project name, stripped.

    Raises:
        ConceptsApiConfigurationError: If no project was given and the
            environment does not name one.
    """
    resolved = (project or os.environ.get(PROJECT_ENV_VAR) or "").strip()
    if not resolved:
        raise ConceptsApiConfigurationError(
            "Listing the charite_demo variable names is a Concepts API read, and every "
            "read is scoped to a project. Pass a project explicitly or set the "
            f"{PROJECT_ENV_VAR} environment variable."
        )
    return resolved


def fetch_all_var_names(project: str | None = None) -> list[str]:
    """List every charite_demo variable name the Concepts API serves.

    The names come back fully qualified (``"<taxonomy>/<name>"``), which is what
    a reference needs once a taxonomy is in play. The result is memoised per
    project for the life of the process: this replaces a constant that used to
    be read from a bundled file, so callers treat it as one.

    Args:
        project (str | None): Project to read as. Defaults to the
            :data:`PROJECT_ENV_VAR` environment variable.

    Returns:
        list[str]: The fully-qualified names, in the order the API served them.

    Raises:
        ConceptsApiConfigurationError: If no project is configured, or no API
            key is available.
        ConceptsApiError: If the routing table does not route charite_demo to an
            endpoint, or the listing request fails.
    """
    resolved_project = _resolve_project(project)

    cached = _ALL_VAR_NAMES_CACHE.get(resolved_project)
    if cached is not None:
        return cached

    # Imported here rather than at module scope: this module is imported for
    # TABLES and GLOBAL_VARS on every charite_demo import, and that path must stay
    # free of the HTTP client and its dependencies.
    from corr_vars.concepts.client import ConceptsApiClient
    from corr_vars.concepts.routing import ApiRoute, load_routing

    routing = load_routing()
    route = routing.resolve(SOURCE)
    if not isinstance(route, ApiRoute):
        raise ConceptsApiConfigurationError(
            f"{SOURCE} does not route to a Concepts API endpoint "
            f"(routing.toml resolves it to {route!r}), but its variable "
            "definitions are only served remotely — there is no bundled "
            "vars.json to fall back to. Fix routing.toml."
        )

    client = ConceptsApiClient(route.url, project=resolved_project, api_key=None)
    names = client.list_concept_names(routing.taxonomy, source=SOURCE)

    _ALL_VAR_NAMES_CACHE[resolved_project] = names
    return names


GLOBAL_VARS = load_json("globals")
DEFAULT_VARS = load_json("defaults")
TABLES = _check_tables_type(load_json("tables"))
