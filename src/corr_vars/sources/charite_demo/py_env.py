"""Exec-namespace contract for ``py`` snippets served for the charite_demo source.

Mirrors the production contract the demo concepts were ported from, minus
``comorb_calc`` and ``classify_catheters`` (see
:mod:`corr_vars.sources.charite_demo.py_namespace`).
"""

from __future__ import annotations

import importlib
from pathlib import Path

from corr_vars.concepts.compile import CompiledVariableFunction, compile_snippet
from corr_vars.definitions.exceptions import ConceptsApiError

from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import ModuleType

#: Module holding the namespace a served snippet executes against — the import
#: header, module constants and shared helpers.
VARIABLES_MODULE: Final[str] = "corr_vars.sources.charite_demo.py_namespace"

#: Names bound by the import header of :data:`VARIABLES_MODULE`. A ``py`` snippet
#: served by the Concepts API is bare function source: it carries no imports, so
#: these are the names it expects to find already in its globals.
#:
#: TODO: ``classify_airway_devices`` ships as package code, but the curated
#: lookup tables inside ``classify/airway.py`` are *data* and should eventually
#: move into the versioned file library, reached with ``getfile``.
IMPORT_NAMES: Final[tuple[str, ...]] = (
    "datetime",
    "timedelta",
    "partial",
    "os",
    "Path",
    "pickle",
    "tempfile",
    "ibis",
    "np",
    "pd",
    "pl",
    "AsofJoinStrategy",
    "comorbidity",
    "logger",
    "__",
    "utils",
    "merge_consecutive",
    "interval_bucket_agg",
    "classify_airway_devices",
    "helpers",
    "TYPE_CHECKING",
    "Callable",
    "VariableContext",
)

#: Top-level functions in :data:`VARIABLES_MODULE` that are **not** variables:
#: they are shared helpers called by functions that are. A snippet for a variable
#: that calls one of these would raise ``NameError`` without them.
SHARED_HELPER_NAMES: Final[tuple[str, ...]] = (
    "_postcode_state",
    "fluids_all",
    "propofol_amount_mg",
    "propofol_rate_mg_kg_h",
)

#: Module-level constants in :data:`VARIABLES_MODULE` referenced from variable
#: functions. Path-valued constants are rebased onto the concept's file
#: directory by :func:`compile_py` so they follow ``__file__``; none of the
#: current ones is a path, since data files are reached through ``getfile``.
MODULE_CONSTANT_NAMES: Final[tuple[str, ...]] = (
    "ATMOSPHERIC_PRESSURE_MMHG",
    "WATER_VAPOUR_PRESSURE_MMHG",
    "RESPIRATORY_QUOTIENT",
    "ROOM_AIR_FIO2",
    "FIO2_LOOKBACK",
)

#: Every name :func:`build_py_namespace` exposes. Each has to exist in
#: :data:`VARIABLES_MODULE`, which is why ``getfile`` is *not* here: per-config
#: state, bound by the compiler and declared in
#: :data:`corr_vars.concepts.compile.INJECTED_NAMES` alongside ``__file__``.
NAMESPACE_NAMES: Final[tuple[str, ...]] = (
    IMPORT_NAMES + SHARED_HELPER_NAMES + MODULE_CONSTANT_NAMES
)


def variables_module() -> ModuleType:
    """Import :data:`VARIABLES_MODULE`.

    Imported lazily: it pulls in ibis, pandas and the charite_demo helpers, and
    importing it at module scope would create a cycle with
    :mod:`corr_vars.sources.charite_demo.helpers`.

    Returns:
        ModuleType: The imported module.
    """
    return importlib.import_module(VARIABLES_MODULE)


def mapping_dir() -> Path:
    """Return the directory a bundled path constant would be relative to.

    Nothing in :data:`VARIABLES_MODULE` is path-valued today — every data file a
    definition reads is reached with ``getfile``, so there is nothing left to
    rebase. The anchor is still handed to the compiler so that a constant
    reintroduced later keeps following ``__file__`` rather than silently
    pointing at a packaged copy.

    Returns:
        Path: Absolute path of the charite_demo package directory.
    """
    return Path(variables_module().__file__ or "").parent


def build_py_namespace() -> dict[str, Any]:
    """Build the execution globals a served ``py`` snippet needs.

    Returns exactly the names :data:`VARIABLES_MODULE` has in scope at import
    time — the import header, the shared
    top-level helpers that are not variables, and the module-level constants
    variable functions reference.

    Deliberately excluded: the variable functions themselves. A snippet must
    declare what it needs through ``requires`` rather than reach sideways into a
    sibling definition that may be served at a different version.

    Returns:
        dict[str, Any]: A fresh namespace dict, safe to mutate.

    Raises:
        ConceptsApiError: If a declared name is missing from the module.
            That means the import header or the set of shared helpers changed
            upstream without this module being updated, and every snippet that
            uses the missing name would otherwise fail with an opaque
            ``NameError`` deep inside user code.
    """
    module = variables_module()

    namespace: dict[str, Any] = {}
    missing: list[str] = []
    for name in NAMESPACE_NAMES:
        if not hasattr(module, name):
            missing.append(name)
            continue
        namespace[name] = getattr(module, name)

    if missing:
        raise ConceptsApiError(
            f"{VARIABLES_MODULE} no longer defines {missing}. "
            "corr_vars.sources.charite_demo.py_env declares the namespace served py "
            "snippets execute against; update NAMESPACE_NAMES to match."
        )

    return namespace


def compile_py(
    snippet: str,
    var_name: str,
    files_dir: Path | str,
    *,
    files_by_uuid: Mapping[str, Path] | None = None,
) -> CompiledVariableFunction:
    """Compile a served ``py`` snippet into a callable variable function.

    The snippet is executed against :func:`build_py_namespace` with ``__file__``
    set to ``<files_dir>/variables.py``. Three file-access idioms therefore work:

    - ``getfile("<uuid>")`` — the documented form, used by every
      definition and the one the browser's editor inserts. It is injected by
      :func:`~corr_vars.concepts.compile.compile_snippet`, not declared here,
      because it closes over the manifest of the config being compiled.
    - ``Path(__file__).parent / "postcode" / "postcode_mapping.csv"`` — the
      package-relative form, still supported for definitions authored elsewhere.
    - ``var.files["postcode/postcode_mapping.csv"]`` — the explicit accessor.

    Args:
        snippet (str): Bare function source text served by the Concepts API.
        var_name (str): Name of the function to extract from the snippet.
        files_dir (Path | str): Directory the concept's attached files were
            materialised into, mirroring their relative paths.
        files_by_uuid (Mapping[str, Path] | None): The config's file manifest,
            uuid to materialised path — what ``getfile`` resolves against.

    Returns:
        CompiledVariableFunction: The callable, with its attached files exposed
        as :attr:`~corr_vars.concepts.compile.CompiledVariableFunction.files`.

    Raises:
        ConceptsApiError: If the snippet does not parse, fails to execute, or
            defines no callable named `var_name`.
    """
    return compile_snippet(
        snippet,
        var_name,
        files_dir,
        namespace=build_py_namespace(),
        mapping_dir=mapping_dir(),
        files_by_uuid=files_by_uuid,
    )
