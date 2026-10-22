"""charite_demo's local mapping configuration, plus a lazy view of its variable names.

``GLOBAL_VARS``, ``TABLES`` and ``DEFAULT_VARS`` are read from the bundled JSON
at import time and stay local — they are corr_vars configuration, not clinical
concepts.

``ALL_VAR_NAMES`` is different: charite_demo's variable definitions are served by the
Concepts API, so the list of names is a remote read. It is resolved through the
module ``__getattr__`` below (PEP 562), which means ``from ... import
ALL_VAR_NAMES`` still works but importing this package does not talk to anything.
The first attribute access needs a project (``CORR_CONCEPTS_PROJECT``) and an
API key (``CORR_CONCEPTS_API_KEY``); the result is cached for the process.
"""

from .loader import DEFAULT_VARS, GLOBAL_VARS, TABLES, fetch_all_var_names

from typing import Any

#: charite_demo ships no ``variables.py`` — a served ``py`` snippet is
#: compiled against the namespace :mod:`corr_vars.sources.charite_demo.py_env`
#: declares. The attribute is kept so that code walking the sources' mapping
#: modules sees the same shape everywhere.
variables = None

__all__ = [
    # Globals
    "GLOBAL_VARS",
    "DEFAULT_VARS",
    "TABLES",
    # Variable names, fetched from the Concepts API on first access
    "ALL_VAR_NAMES",
    "fetch_all_var_names",
    # Variable functions
    "variables",
]


def __getattr__(name: str) -> Any:
    """Resolve ``ALL_VAR_NAMES`` on first access.

    Args:
        name (str): Attribute being looked up.

    Returns:
        Any: The list of fully-qualified variable names, for ``ALL_VAR_NAMES``.

    Raises:
        AttributeError: For every other name, so ``getattr(mapping, "VARS", {})``
            and friends keep behaving like an ordinary module lookup.
    """
    if name == "ALL_VAR_NAMES":
        value = fetch_all_var_names()
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | {"ALL_VAR_NAMES"})
