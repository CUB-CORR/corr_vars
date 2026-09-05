from __future__ import annotations
import __future__ as future_features

import importlib
from pathlib import Path

from corr_vars.concepts.files import discover_files
from corr_vars.definitions.exceptions import ConceptsApiError, VariableDefinitionError

from types import FunctionType, ModuleType
from typing import TYPE_CHECKING, Any, Protocol

#: Compiler flag for PEP 563 string annotations, applied to every snippet.
_ANNOTATIONS_FLAG = future_features.annotations.compiler_flag

#: Names the snippet compiler binds itself, on top of what a source's ``py_env``
#: declares. ``getfile`` is one of them: it closes over *this* definition's file
#: manifest, so it cannot come from a module the way the rest of the namespace
#: does. The ``py-snippet-namespace`` standard reads this set, which is why a
#: definition may call ``getfile`` without declaring anything.
INJECTED_NAMES: frozenset[str] = frozenset(
    {"__file__", "__name__", "__builtins__", "getfile"}
)

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

    from corr_vars.core.cohort import Cohort

    from collections.abc import Callable, Mapping, MutableMapping
    from corr_vars.definitions.typing import VariableContext


class PyCompilerProtocol(Protocol):
    def __call__(
        self,
        snippet: str,
        var_name: str,
        files_dir: Path | str,
        *,
        files_by_uuid: Mapping[str, Path] | None = None,
    ) -> CompiledVariableFunction:
        """Compile a bare ``py`` snippet into a callable variable function.

        Args:
            snippet (str): Function source text as served by the Concepts API.
            var_name (str): Name of the function to pull out of the snippet.
            files_dir (Path | str): Directory the concept's attached files were
                materialised into.
            files_by_uuid (Mapping[str, Path] | None): This config's file
                manifest, uuid to materialised path — what ``getfile`` resolves
                against.

        Returns:
            CompiledVariableFunction: The callable, ready to be passed as ``py``.
        """
        ...


def make_getfile(
    files_by_uuid: Mapping[str, Path], var_name: str
) -> Callable[[str], Path]:
    """Build the ``getfile`` a served snippet calls to reach its data files.

    A snippet names a data file by uuid — ``pd.read_csv(getfile("2f3a…"))`` —
    and never by path, because the file lives in the source's library and the
    config version it belongs to pins one version of it. The returned callable
    resolves only against `files_by_uuid`, the manifest served *with this
    config*: a definition can reach exactly the files it was published against
    and nothing else, so it can never silently read another definition's data.

    The return value is a :class:`pathlib.Path`, matching every other path
    corr_vars hands a snippet (``var.files[...]``, the rebased module constants,
    ``Path(__file__).parent``). Both pandas and polars readers accept it.

    Args:
        files_by_uuid (Mapping[str, Path]): This config's manifest, uuid to the
            materialised location of the pinned bytes.
        var_name (str): Variable the snippet defines, for the error message.

    Returns:
        Callable[[str], Path]: The ``getfile`` to inject into the namespace.
    """

    def getfile(uuid: str) -> Path:
        """Return the local path of the data file `uuid` names.

        Args:
            uuid (str): The file's id, as it appears in the definition.

        Returns:
            Path: Location of the pinned bytes on disk.

        Raises:
            VariableDefinitionError: If `uuid` is not in this definition's
                manifest.
        """
        try:
            return files_by_uuid[uuid]
        except (KeyError, TypeError):
            known = ", ".join(sorted(files_by_uuid)) or "none"
            raise VariableDefinitionError(
                f"getfile({uuid!r}) in the definition of {var_name!r} names no file "
                f"this definition pins. Files pinned by this version: {known}. "
                "Either the uuid is wrong, or the file was added to the source's "
                "library after this version was published."
            ) from None

    return getfile


class CompiledVariableFunction:
    """A ``py`` function compiled from Concepts API source text.

    Satisfies :class:`~corr_vars.definitions.typing.VariableCallable` and carries
    the attached data files alongside, so
    :class:`~corr_vars.core.steps.PyFuncStep` can expose them as
    ``var.files["postcode/postcode_mapping.csv"]``.

    Args:
        func (Any): The function object extracted from the compiled snippet.
        var_name (str): Name the function was published under.
        files (Mapping[str, Path]): Attached files keyed by relative path.
        files_dir (Path): Directory the files were materialised into; it is also
            the parent of the synthetic ``__file__`` the snippet sees.
        files_by_uuid (Mapping[str, Path] | None): The same files keyed by uuid
            — the manifest ``getfile`` resolves against.
        source (str): The original snippet text, kept for debugging.
    """

    def __init__(
        self,
        func: Any,
        *,
        var_name: str,
        files: Mapping[str, Path],
        files_dir: Path,
        files_by_uuid: Mapping[str, Path] | None = None,
        source: str = "",
    ) -> None:
        self.func = func
        self.var_name = var_name
        self.files: dict[str, Path] = dict(files)
        self.files_by_uuid: dict[str, Path] = dict(files_by_uuid or {})
        self.files_dir = files_dir
        self.source = source
        self.__name__ = var_name
        self.__doc__ = getattr(func, "__doc__", None)

    def __call__(
        self, var: VariableContext, cohort: Cohort
    ) -> pl.DataFrame | pd.DataFrame:
        return self.func(var=var, cohort=cohort)

    def __repr__(self) -> str:
        return (
            f"CompiledVariableFunction(var_name={self.var_name!r}, "
            f"files={sorted(self.files)}, files_dir={str(self.files_dir)!r})"
        )


def _rebind_to_rebased_paths(
    namespace: MutableMapping[str, Any], rebased: Mapping[str, Path]
) -> None:
    """Make rebased path constants visible inside the namespace's helper functions.

    A shared helper such as ``_postcode_state`` enters the namespace as the
    function object defined in ``variables.py``, so it reads its module-level
    constants from *that* module's globals — rebasing the namespace copy would
    never reach it, and the helper would keep reading the packaged data file
    while the snippet around it reads the concept's attached one. Each affected
    helper is therefore replaced by a clone bound to a globals dict carrying the
    rebased values.

    The clone snapshots its module's globals, so a later reload of the defining
    module is not reflected. Helpers that reference none of the rebased names are
    left untouched.

    Args:
        namespace (MutableMapping[str, Any]): Execution globals, mutated in place.
        rebased (Mapping[str, Path]): Constants that were rebased onto the
            concept's file directory.
    """
    if not rebased:
        return

    for name, value in list(namespace.items()):
        if not isinstance(value, FunctionType):
            continue
        globals_ = value.__globals__
        overlap = {key: val for key, val in rebased.items() if key in globals_}
        if not overlap:
            continue

        clone = FunctionType(
            value.__code__,
            {**globals_, **overlap},
            value.__name__,
            value.__defaults__,
            value.__closure__,
        )
        clone.__kwdefaults__ = value.__kwdefaults__
        clone.__qualname__ = value.__qualname__
        clone.__module__ = value.__module__
        clone.__doc__ = value.__doc__
        clone.__dict__.update(value.__dict__)
        namespace[name] = clone


def compile_snippet(
    snippet: str,
    var_name: str,
    files_dir: Path | str,
    *,
    namespace: MutableMapping[str, Any],
    mapping_dir: Path | None = None,
    files_by_uuid: Mapping[str, Path] | None = None,
) -> CompiledVariableFunction:
    """Execute `snippet` against `namespace` and extract the function `var_name`.

    ``__file__`` is set to ``<files_dir>/variables.py`` before execution. That is
    what keeps the existing ``Path(__file__).parent / "postcode" / ...`` idiom
    working unchanged: the snippet resolves its data relative to the directory
    the concept's attached files were materialised into.

    Args:
        snippet (str): Bare function source text.
        var_name (str): Name of the function to extract.
        files_dir (Path | str): Directory holding the concept's attached files.
        namespace (MutableMapping[str, Any]): Execution globals, normally built
            by the source's ``py_env.build_py_namespace()``.
        mapping_dir (Path | None): Directory the source's bundled path constants
            are relative to. Constants pointing inside it are rebased onto
            `files_dir` so they follow ``__file__``, and any helper in `namespace`
            that reads one is rebound to see the rebased value.
        files_by_uuid (Mapping[str, Path] | None): This config's file manifest,
            uuid to materialised path. Bound into the namespace as ``getfile``,
            the accessor served snippets use to reach a data file.

    Returns:
        CompiledVariableFunction: The compiled callable with its files attached.

    Raises:
        ConceptsApiError: If the snippet fails to compile or execute, or does not
            define a callable named `var_name`.
    """
    from corr_vars.concepts.files import rebase_paths

    directory = Path(files_dir)
    module_file = directory / "variables.py"

    if mapping_dir is not None:
        rebased = rebase_paths(namespace, mapping_dir, directory)
        namespace.update(rebased)
        _rebind_to_rebased_paths(namespace, rebased)

    manifest = dict(files_by_uuid or {})
    namespace["__file__"] = str(module_file)
    namespace["__name__"] = f"corr_vars.concepts.compiled.{var_name}"
    # Bound last, and unconditionally: a snippet that calls getfile() against a
    # config pinning no files must fail saying so, not with a bare NameError.
    namespace["getfile"] = make_getfile(manifest, var_name)

    try:
        # A snippet is bare function source: the ``from __future__ import annotations``
        # its module carries is not part of it, so without this flag every annotation
        # would be evaluated at def time and a ``TYPE_CHECKING``-only name such as
        # ``Cohort`` would raise NameError. Compiled with ``dont_inherit`` so the
        # semantics come from here rather than from this module's own header.
        code = compile(
            snippet,
            str(module_file),
            "exec",
            flags=_ANNOTATIONS_FLAG,
            dont_inherit=True,
        )
    except SyntaxError as exc:
        raise ConceptsApiError(
            f"The py snippet served for {var_name!r} does not parse: {exc}"
        ) from exc

    try:
        exec(
            code, namespace
        )  # noqa: S102 - executing served definitions is the feature
    except Exception as exc:
        raise ConceptsApiError(
            f"The py snippet served for {var_name!r} failed to execute: {exc!r}"
        ) from exc

    func = namespace.get(var_name)
    if func is None:
        defined = sorted(
            name
            for name, value in namespace.items()
            if callable(value) and not name.startswith("__")
        )
        raise ConceptsApiError(
            f"The py snippet served for {var_name!r} defines no function of that "
            f"name. Defined callables: {defined}."
        )
    if not callable(func):
        raise ConceptsApiError(
            f"The py snippet served for {var_name!r} binds {var_name!r} to a "
            f"{type(func).__name__}, not a function."
        )

    return CompiledVariableFunction(
        func,
        var_name=var_name,
        files=discover_files(directory),
        files_dir=directory,
        files_by_uuid=manifest,
        source=snippet,
    )


def get_py_compiler(source: str) -> PyCompilerProtocol | None:
    """Return the ``compile_py`` entry point of a source's ``py_env`` module.

    Sources that serve ``py`` snippets ship a ``py_env`` module declaring the
    namespace their ``variables.py`` import header provides. Sources without one
    cannot execute served snippets.

    Args:
        source (str): Source name, e.g. ``"reprodicu"``.

    Returns:
        PyCompilerProtocol | None: The compiler, or ``None`` if the source has
        no ``py_env`` module.
    """
    try:
        module: ModuleType = importlib.import_module(
            f"corr_vars.sources.{source}.py_env"
        )
    except ModuleNotFoundError:
        return None
    return getattr(module, "compile_py", None)
