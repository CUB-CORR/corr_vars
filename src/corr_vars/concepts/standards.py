"""Publication standards: automated checks a definition must pass to be served.

A definition that loads locally does not automatically survive being published
and served back as a Concepts API concept. Local loading hands the variable
function straight out of its module, with the whole module namespace behind it;
a served ``py`` snippet is bare source text executed against the much smaller
namespace its source's ``py_env`` declares. Anything the function reaches for
that the namespace does not carry works locally and raises ``NameError``
remotely — at extraction time, in someone else's cohort.

This module is where those checks live. It is deliberately a registry rather
than a single function: the checks are the same whether they run

- in CI, over the definitions a source bundles today (see
  ``tests/integration/test_variable_standards.py``), or
- at publication time, over a config a user is trying to publish.

The second caller does not exist yet and its standards are still to be defined,
so only the source-scoped registry is populated. :func:`variable_standard` and
:func:`check_variable` are the seam it will attach to; both are live and empty.

Findings carry a severity. An ``error`` means the definition would fail when
served, so it must block. A ``warning`` records drift that is not yet a failure
— a declaration nothing uses, for instance — and is meant to be read, not
enforced.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import symtable
from pathlib import Path

from corr_vars import logger
from corr_vars.concepts.compile import INJECTED_NAMES as _INJECTED_NAMES

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence
    from types import ModuleType

Severity = Literal["error", "warning"]

#: Names the snippet compiler injects, on top of what ``py_env`` declares.
#: Re-exported from the compiler so the two can never drift: a name the compiler
#: binds but this standard does not know about would be reported as an
#: ``undeclared-name`` error on every definition that uses it.
INJECTED_NAMES: frozenset[str] = _INJECTED_NAMES


@dataclass(frozen=True, order=True)
class Finding:
    """One thing a standard objects to.

    Args:
        standard (str): Name of the standard that produced it.
        code (str): Stable machine-readable kind, e.g. ``"undeclared-name"``.
        severity (Severity): ``"error"`` blocks publication, ``"warning"`` does not.
        subject (str): What the finding is about — usually ``"<source>/<var>"``.
        message (str): Human-readable explanation, including the remedy.
    """

    standard: str
    code: str
    severity: Severity
    subject: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.subject}: {self.message} ({self.standard}/{self.code})"


@dataclass(frozen=True)
class Standard:
    """A registered check.

    Args:
        name (str): Unique name, used to select and to report.
        description (str): One line describing what it guarantees.
        check (Callable[..., list[Finding]]): The implementation.
    """

    name: str
    description: str
    check: Callable[..., list[Finding]]


_SOURCE_STANDARDS: dict[str, Standard] = {}
_VARIABLE_STANDARDS: dict[str, Standard] = {}


def _register(
    registry: dict[str, Standard], name: str, description: str
) -> Callable[[Callable[..., list[Finding]]], Callable[..., list[Finding]]]:
    def decorate(func: Callable[..., list[Finding]]) -> Callable[..., list[Finding]]:
        if name in registry:
            raise ValueError(f"A standard named {name!r} is already registered")
        registry[name] = Standard(name=name, description=description, check=func)
        return func

    return decorate


def source_standard(
    name: str, description: str
) -> Callable[[Callable[[str], list[Finding]]], Callable[[str], list[Finding]]]:
    """Register a standard that checks one source as a whole.

    Args:
        name (str): Unique name.
        description (str): One line describing what it guarantees.

    Returns:
        Callable: Decorator leaving the function otherwise unchanged.
    """
    return _register(_SOURCE_STANDARDS, name, description)  # type: ignore[return-value]


def variable_standard(
    name: str, description: str
) -> Callable[[Callable[..., list[Finding]]], Callable[..., list[Finding]]]:
    """Register a standard that checks a single candidate variable definition.

    The publication-time standards will live here. None are defined yet.

    Args:
        name (str): Unique name.
        description (str): One line describing what it guarantees.

    Returns:
        Callable: Decorator leaving the function otherwise unchanged.
    """
    return _register(_VARIABLE_STANDARDS, name, description)


def source_standards() -> tuple[Standard, ...]:
    """Return the registered source-scoped standards, in registration order."""
    return tuple(_SOURCE_STANDARDS.values())


def variable_standards() -> tuple[Standard, ...]:
    """Return the registered variable-scoped standards, in registration order."""
    return tuple(_VARIABLE_STANDARDS.values())


def check_source(source: str, *, only: Iterable[str] | None = None) -> list[Finding]:
    """Run every source-scoped standard against `source`.

    Args:
        source (str): Source key, e.g. ``"reprodicu"``.
        only (Iterable[str] | None): Restrict to these standard names.

    Returns:
        list[Finding]: All findings, errors first, then sorted for stable output.
    """
    selected = _select(_SOURCE_STANDARDS, only)
    findings = [f for standard in selected for f in standard.check(source)]
    return _ordered(findings)


def check_variable(source: str, name: str, config: Mapping[str, Any]) -> list[Finding]:
    """Run every variable-scoped standard against one candidate definition.

    The registry is empty until the publication-time standards are defined, so
    this returns no findings today. It exists so the call site can be written
    now and the standards added behind it.

    Args:
        source (str): Source the definition belongs to.
        name (str): Variable name.
        config (Mapping[str, Any]): The candidate config, as published.

    Returns:
        list[Finding]: All findings, errors first, then sorted for stable output.
    """
    findings = [
        f
        for standard in _VARIABLE_STANDARDS.values()
        for f in standard.check(source, name, config)
    ]
    return _ordered(findings)


def errors(findings: Iterable[Finding]) -> list[Finding]:
    """Return only the blocking findings."""
    return [f for f in findings if f.severity == "error"]


def _select(
    registry: Mapping[str, Standard], only: Iterable[str] | None
) -> list[Standard]:
    if only is None:
        return list(registry.values())
    names = list(only)
    unknown = [n for n in names if n not in registry]
    if unknown:
        raise KeyError(f"Unknown standard(s): {unknown}; known: {sorted(registry)}")
    return [registry[n] for n in names]


def _ordered(findings: Iterable[Finding]) -> list[Finding]:
    return sorted(findings, key=lambda f: (f.severity != "error", f))


# ---------------------------------------------------------------------------
# Static analysis of a source's variables.py
# ---------------------------------------------------------------------------


def _import_optional(name: str) -> ModuleType | None:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        return None


def _global_reads(table: symtable.SymbolTable) -> set[str]:
    """Names a function reads from module scope, including in nested scopes.

    ``symtable`` resolves scoping properly — parameters, assignments, walrus,
    comprehension and closure variables are all local or free rather than
    global — and it honours ``from __future__ import annotations``, so names
    used only in annotations are correctly absent (snippets are compiled with
    that same flag). Builtins are reported as global reads and are stripped by
    the caller.

    Args:
        table (symtable.SymbolTable): The function's symbol table.

    Returns:
        set[str]: Every module-scope name read anywhere inside it.
    """
    names = {
        symbol.get_name()
        for symbol in table.get_symbols()
        if symbol.is_global() and symbol.is_referenced()
    }
    for child in table.get_children():
        names |= _global_reads(child)
    return names


def _annotation_names(tree: ast.AST) -> set[str]:
    """Names used in annotation position anywhere in a module.

    Snippets are compiled with PEP 563, so annotations are strings and never
    evaluated — a name used only there cannot raise ``NameError`` and is not a
    finding. It is still a genuine reference though, so it counts as a read and
    keeps a declaration like ``VariableContext`` from being reported as unused.

    Args:
        tree (ast.AST): The parsed module.

    Returns:
        set[str]: Every name appearing inside an annotation.
    """
    annotations: list[ast.expr] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            for arg in [
                *args.posonlyargs,
                *args.args,
                *args.kwonlyargs,
                args.vararg,
                args.kwarg,
            ]:
                if arg is not None and arg.annotation is not None:
                    annotations.append(arg.annotation)
            if node.returns is not None:
                annotations.append(node.returns)
        elif isinstance(node, ast.AnnAssign):
            annotations.append(node.annotation)

    return {
        child.id
        for annotation in annotations
        for child in ast.walk(annotation)
        if isinstance(child, ast.Name)
    }


@source_standard(
    "py-snippet-namespace",
    "Every name a served py snippet reads is declared by the source's py_env.",
)
def py_snippet_namespace(source: str) -> list[Finding]:
    """Check a source's ``variables.py`` against its ``py_env`` declaration.

    Four ways the two can disagree, each of which breaks a served snippet or
    documents rot:

    - ``missing-declaration`` — ``py_env`` declares a name ``variables.py`` no
      longer defines. ``build_py_namespace`` would refuse to build.
    - ``undeclared-name`` — a variable function reads a module-level name the
      namespace does not carry. ``NameError`` when served.
    - ``sibling-variable`` — a variable function calls another variable
      function. Also a ``NameError`` when served, and deliberately so: a snippet
      must pull what it needs through ``requires`` rather than reach sideways
      into a definition that may be served at a different version.
    - ``unused-declaration`` (warning) — the namespace carries a name nothing
      reads.

    A source with no ``py_env`` module serves no snippets and is skipped.

    Args:
        source (str): Source key.

    Returns:
        list[Finding]: The disagreements found.
    """
    py_env = _import_optional(f"corr_vars.sources.{source}.py_env")
    if py_env is None:
        return []

    module = py_env.variables_module()
    path = Path(module.__file__ or "")
    if not path.is_file():
        logger.warning("Source %s has a py_env but no readable variables.py", source)
        return []

    declared = set(py_env.NAMESPACE_NAMES)
    findings = [
        Finding(
            standard="py-snippet-namespace",
            code="missing-declaration",
            severity="error",
            subject=f"{source}/py_env",
            message=(
                f"py_env declares {name!r}, which {module.__name__} no longer defines. "
                "Remove it from NAMESPACE_NAMES or restore the definition."
            ),
        )
        for name in sorted(declared)
        if not hasattr(module, name)
    ]

    mapping = _import_optional(f"corr_vars.sources.{source}.mapping")
    variable_names: set[str] = set(getattr(mapping, "VARS", {}).get("variables", {}))

    text = path.read_text()
    table = symtable.symtable(text, str(path), "exec")
    functions = {
        child.get_name(): child
        for child in table.get_children()
        if child.get_type() == "function"
    }

    allowed = declared | set(dir(builtins)) | INJECTED_NAMES
    read_by_anyone: set[str] = _annotation_names(ast.parse(text))

    for name, func_table in sorted(functions.items()):
        reads = _global_reads(func_table)
        read_by_anyone |= reads
        if name not in variable_names:
            # Shared helpers are handed out as the module's own function objects,
            # so they keep reading from the module namespace and need nothing
            # declared. Their reads still count towards "used".
            continue

        # A snippet defines its own function name before the body ever runs, so a
        # self-reference (recursion, or `f.__name__`) resolves.
        for missing in sorted(reads - allowed - {name}):
            if missing in variable_names:
                findings.append(
                    Finding(
                        standard="py-snippet-namespace",
                        code="sibling-variable",
                        severity="error",
                        subject=f"{source}/{name}",
                        message=(
                            f"reads {missing!r}, another variable of this source. Served "
                            "as a snippet it raises NameError: sibling definitions are "
                            "kept out of the namespace on purpose, since each is served "
                            f"at its own version. Declare {missing!r} in 'requires' and "
                            "read it from var.required_vars, or inline what it does."
                        ),
                    )
                )
            else:
                findings.append(
                    Finding(
                        standard="py-snippet-namespace",
                        code="undeclared-name",
                        severity="error",
                        subject=f"{source}/{name}",
                        message=(
                            f"reads module-level name {missing!r}, which "
                            f"corr_vars.sources.{source}.py_env does not declare. Served "
                            "as a snippet it raises NameError. Add it to IMPORT_NAMES, "
                            "SHARED_HELPER_NAMES or MODULE_CONSTANT_NAMES."
                        ),
                    )
                )

    findings.extend(
        Finding(
            standard="py-snippet-namespace",
            code="unused-declaration",
            severity="warning",
            subject=f"{source}/py_env",
            message=(
                f"declares {name!r}, which no definition in {module.__name__} reads. "
                "Harmless, but it is dead weight in every snippet's namespace."
            ),
        )
        for name in sorted(declared - read_by_anyone)
    )

    return _ordered(findings)


def format_findings(findings: Sequence[Finding]) -> str:
    """Render findings as one line each, for a test failure or a CLI.

    Args:
        findings (Sequence[Finding]): What to render.

    Returns:
        str: One line per finding, or a short note when there are none.
    """
    if not findings:
        return "no findings"
    return "\n".join(f"  {finding}" for finding in findings)
