from __future__ import annotations

import builtins
import hashlib
import symtable
import textwrap
from pathlib import Path

import httpx
import polars as pl
import pytest

from corr_vars.concepts import standards
from corr_vars.concepts.client import ConceptFile, ConceptsApiClient
from corr_vars.concepts.compile import (
    INJECTED_NAMES,
    CompiledVariableFunction,
    compile_snippet,
    get_py_compiler,
)
from corr_vars.concepts.files import (
    CACHE_DIR_ENV_VAR,
    cache_root,
    discover_files,
    materialise_files,
    rebase_paths,
)
from corr_vars.concepts.spec import VariableSpec, VersionSelector
from corr_vars.definitions.exceptions import (
    ConceptsApiError,
    VariableDefinitionError,
)
from corr_vars.utils.time import TimeWindow

import dataclasses
from corr_vars.definitions.typing import VariableContext
from types import SimpleNamespace

#: A stand-in for a source's ``py_env`` module. A source that serves ``py``
#: snippets ships one: it declares the namespace a bare snippet executes against
#: and exposes a ``compile_py`` entry point built on :func:`compile_snippet`.
#: The tests below exercise that contract without depending on any one source.
ATMOSPHERIC_PRESSURE_MMHG = 760.0


def _shared_helper(cohort: object, column: str, mapping_path: Path) -> pl.DataFrame:
    """A package-code helper a snippet may hand a materialised file path to."""
    return pl.read_csv(mapping_path).select(column)


NAMESPACE_NAMES: tuple[str, ...] = (
    "pl",
    "Path",
    "VariableContext",
    "ATMOSPHERIC_PRESSURE_MMHG",
    "_shared_helper",
)


def build_py_namespace() -> dict[str, object]:
    return {name: globals()[name] for name in NAMESPACE_NAMES}


def compile_py(
    snippet: str,
    var_name: str,
    files_dir: Path | str,
    *,
    files_by_uuid: dict[str, Path] | None = None,
) -> CompiledVariableFunction:
    return compile_snippet(
        snippet,
        var_name,
        files_dir,
        namespace=build_py_namespace(),
        files_by_uuid=files_by_uuid,
    )


LATEST = VersionSelector("latest")
SPEC = VariableSpec(name="demo_var", taxonomy="corr_v1", version=LATEST)


@pytest.fixture
def cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(CACHE_DIR_ENV_VAR, str(tmp_path / "cache"))
    return tmp_path / "cache"


def file_client(payloads: dict[str, bytes]) -> ConceptsApiClient:
    def handler(request: httpx.Request) -> httpx.Response:
        prefix = "/files/"
        path = request.url.path.split(prefix, 1)[1]
        return httpx.Response(200, content=payloads[path])

    return ConceptsApiClient(
        "https://concepts.test",
        project="demo",
        api_key="k",
        transport=httpx.MockTransport(handler),
        backoff=0.0,
    )


def file_uuid(path: str) -> str:
    """A stable fake uuid per path, so a test can name a file it laid out."""
    return f"uuid-{hashlib.sha256(path.encode()).hexdigest()[:12]}"


def concept_file(path: str, content: bytes, uuid: str | None = None) -> ConceptFile:
    return ConceptFile(
        path=path,
        uuid=uuid or file_uuid(path),
        version_no=1,
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        media_type="text/csv",
        url=f"https://concepts.test/files/{path}",
    )


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

SIMPLE_SNIPPET = textwrap.dedent("""
    def demo_var(var, cohort):
        return pl.DataFrame({"value": [1, 2, 3]})
    """)


def make_context(**overrides: object) -> VariableContext:
    kwargs: dict[str, object] = {
        "var_name": "demo_var",
        "dynamic": True,
        "time_window": TimeWindow("icu_admission", "icu_discharge"),
        "required_vars": {},
        "data": None,
    }
    kwargs.update(overrides)
    return VariableContext(**kwargs)  # type: ignore[arg-type]


class TestCompilePy:
    def test_compiles_and_calls(self, tmp_path: Path) -> None:
        func = compile_py(SIMPLE_SNIPPET, "demo_var", tmp_path)
        assert isinstance(func, CompiledVariableFunction)
        result = func(var=make_context(), cohort=None)  # type: ignore[arg-type]
        assert result.to_dicts() == [{"value": 1}, {"value": 2}, {"value": 3}]

    def test_sets_dunder_file_into_files_dir(self, tmp_path: Path) -> None:
        snippet = textwrap.dedent("""
            def demo_var(var, cohort):
                return Path(__file__)
            """)
        func = compile_py(snippet, "demo_var", tmp_path)
        assert func(var=make_context(), cohort=None) == tmp_path / "variables.py"  # type: ignore[arg-type]

    def test_dunder_file_parent_resolves_attached_files(self, tmp_path: Path) -> None:
        (tmp_path / "postcode").mkdir()
        (tmp_path / "postcode" / "postcode_mapping.csv").write_text("a\n1\n")
        snippet = textwrap.dedent("""
            def demo_var(var, cohort):
                return pl.read_csv(Path(__file__).parent / "postcode" / "postcode_mapping.csv")
            """)
        func = compile_py(snippet, "demo_var", tmp_path)
        assert func(var=make_context(), cohort=None).to_dicts() == [{"a": 1}]  # type: ignore[arg-type]

    def test_var_files_resolves_attached_files(self, tmp_path: Path) -> None:
        (tmp_path / "postcode").mkdir()
        (tmp_path / "postcode" / "postcode_mapping.csv").write_text("a\n2\n")
        snippet = textwrap.dedent("""
            def demo_var(var, cohort):
                return pl.read_csv(var.files["postcode/postcode_mapping.csv"])
            """)
        func = compile_py(snippet, "demo_var", tmp_path)
        context = make_context(files=func.files)
        assert func(var=context, cohort=None).to_dicts() == [{"a": 2}]  # type: ignore[arg-type]

    def test_files_attribute_lists_materialised_paths(self, tmp_path: Path) -> None:
        (tmp_path / "nested").mkdir()
        (tmp_path / "nested" / "a.csv").write_text("x\n")
        (tmp_path / "b.csv").write_text("y\n")
        func = compile_py(SIMPLE_SNIPPET, "demo_var", tmp_path)
        assert sorted(func.files) == ["b.csv", "nested/a.csv"]

    def test_module_constants_are_available(self, tmp_path: Path) -> None:
        snippet = textwrap.dedent("""
            def demo_var(var, cohort):
                return ATMOSPHERIC_PRESSURE_MMHG
            """)
        func = compile_py(snippet, "demo_var", tmp_path)
        assert func(var=make_context(), cohort=None) == 760.0  # type: ignore[arg-type]

    def test_shared_helper_is_available(self, tmp_path: Path) -> None:
        snippet = textwrap.dedent("""
            def demo_var(var, cohort):
                return _shared_helper
            """)
        func = compile_py(snippet, "demo_var", tmp_path)
        assert callable(func(var=make_context(), cohort=None))  # type: ignore[arg-type]

    def test_syntax_error_is_reported(self, tmp_path: Path) -> None:
        with pytest.raises(ConceptsApiError, match="does not parse"):
            compile_py("def demo_var(:\n", "demo_var", tmp_path)

    def test_execution_error_is_reported(self, tmp_path: Path) -> None:
        with pytest.raises(ConceptsApiError, match="failed to execute"):
            compile_py("raise RuntimeError('boom')", "demo_var", tmp_path)

    def test_missing_function_is_reported(self, tmp_path: Path) -> None:
        snippet = "def other_name(var, cohort):\n    return None\n"
        with pytest.raises(ConceptsApiError, match="defines no function"):
            compile_py(snippet, "demo_var", tmp_path)

    def test_non_callable_binding_is_reported(self, tmp_path: Path) -> None:
        with pytest.raises(ConceptsApiError, match="not a function"):
            compile_py("demo_var = 3\n", "demo_var", tmp_path)

    def test_snippet_cannot_see_sibling_variable_functions(
        self, tmp_path: Path
    ) -> None:
        snippet = textwrap.dedent("""
            def demo_var(var, cohort):
                return some_other_variable
            """)
        func = compile_py(snippet, "demo_var", tmp_path)
        with pytest.raises(NameError):
            func(var=make_context(), cohort=None)  # type: ignore[arg-type]

    def test_type_checking_only_annotations_do_not_break_a_snippet(
        self, tmp_path: Path
    ) -> None:
        # `Cohort` is a `TYPE_CHECKING`-only import for the namespace module, so it
        # is not in the namespace. The snippet carries no `from __future__ import
        # annotations` of its own, so without the compiler flag the annotation
        # would be evaluated at def time and raise NameError.
        snippet = textwrap.dedent("""
            def demo_var(var: VariableContext, cohort: Cohort) -> pl.DataFrame:
                return pl.DataFrame({"value": [1]})
            """)
        func = compile_py(snippet, "demo_var", tmp_path)
        assert func(var=make_context(), cohort=None).to_dicts() == [{"value": 1}]  # type: ignore[arg-type]

    def test_annotation_semantics_do_not_depend_on_the_compiler_module(
        self, tmp_path: Path
    ) -> None:
        snippet = (
            "def demo_var(var: Undefined, cohort) -> AlsoUndefined:\n    return 1\n"
        )
        func = compile_snippet(snippet, "demo_var", tmp_path, namespace={})
        assert func.func.__annotations__["var"] == "Undefined"

    def test_helpers_see_rebased_path_constants(self, tmp_path: Path) -> None:
        # A shared helper enters the namespace as its module's function object and
        # keeps reading that module's globals, so rebasing the namespace copy of a
        # path constant has to reach into the helper too — otherwise the snippet
        # reads the concept's attached file while the helper it calls reads the
        # packaged one.
        anchor = tmp_path / "mapping"
        anchor.mkdir()
        module_globals: dict[str, object] = {"DATA": anchor / "postcode" / "m.csv"}
        exec("def helper():\n    return DATA\n", module_globals)  # noqa: S102

        files_dir = tmp_path / "files"
        files_dir.mkdir()
        namespace: dict[str, object] = {
            "DATA": module_globals["DATA"],
            "helper": module_globals["helper"],
        }
        snippet = "def demo_var(var, cohort):\n    return helper()\n"

        func = compile_snippet(
            snippet, "demo_var", files_dir, namespace=namespace, mapping_dir=anchor
        )
        assert func(var=make_context(), cohort=None) == files_dir / "postcode" / "m.csv"  # type: ignore[arg-type]
        # The module's own function is untouched — only the namespace copy is rebound.
        assert module_globals["helper"]() == anchor / "postcode" / "m.csv"  # type: ignore[operator]

    def test_helpers_without_rebased_names_are_not_rebound(
        self, tmp_path: Path
    ) -> None:
        anchor = tmp_path / "mapping"
        anchor.mkdir()
        module_globals: dict[str, object] = {"OTHER": Path("/elsewhere/x.csv")}
        exec("def helper():\n    return OTHER\n", module_globals)  # noqa: S102
        helper = module_globals["helper"]

        namespace: dict[str, object] = {"helper": helper, "DATA": anchor / "m.csv"}
        compile_snippet(
            "def demo_var(var, cohort):\n    return 1\n",
            "demo_var",
            tmp_path / "files",
            namespace=namespace,
            mapping_dir=anchor,
        )
        assert namespace["helper"] is helper

    def test_a_rewritten_snippet_reaches_its_file_and_its_helper(
        self, tmp_path: Path
    ) -> None:
        # The real shape of a definition after the getfile rewrite: the snippet
        # resolves its mapping file by uuid and hands the path to a shared
        # helper, which is package code and has no getfile of its own.
        uuid = "f5497211-1667-58ef-a16c-fb97b95b3987"
        target = tmp_path / "postcode" / "postcode_mapping.csv"
        target.parent.mkdir()
        target.write_text("postal_code_digits12,state_de\n10,Berlin\n")

        snippet = textwrap.dedent(f"""
            def demo_var(var, cohort):
                mapping_path = getfile({uuid!r})
                return _shared_helper(cohort, "state_de", mapping_path)
            """)
        func = compile_py(snippet, "demo_var", tmp_path, files_by_uuid={uuid: target})

        cohort = SimpleNamespace(
            primary_key="icu_stay_id",
            obs=pl.DataFrame({"icu_stay_id": [1], "postcode": [10115]}),
        )
        result = func(var=make_context(), cohort=cohort)  # type: ignore[arg-type]
        assert result.to_dicts() == [{"state_de": "Berlin"}]

    def test_compile_snippet_without_mapping_dir_leaves_paths_alone(
        self, tmp_path: Path
    ) -> None:
        namespace: dict[str, object] = {"Path": Path, "MY_PATH": Path("/elsewhere/x")}
        snippet = "def demo_var(var, cohort):\n    return MY_PATH\n"
        func = compile_snippet(snippet, "demo_var", tmp_path, namespace=namespace)
        assert func(var=make_context(), cohort=None) == Path("/elsewhere/x")  # type: ignore[arg-type]


class TestGetfile:
    """``getfile("<uuid>")`` — how a served snippet reaches a data file.

    Data files live in the source's library and a config version pins one
    version of each; the snippet names one by uuid and gets back a path. The
    manifest it resolves against is the one served with *that* config, so a
    definition can only reach the bytes it was published against.
    """

    UUID = "2f3a9c1e-5b7d-4a10-9f2c-8e6d1b4a7c30"

    def _snippet(self, uuid: str = UUID) -> str:
        return textwrap.dedent(f"""
            def demo_var(var, cohort):
                return pl.read_csv(getfile({uuid!r}))
            """)

    def test_resolves_to_the_materialised_file(self, tmp_path: Path) -> None:
        target = tmp_path / "postcode" / "postcode_mapping.csv"
        target.parent.mkdir()
        target.write_text("a\n1\n")

        func = compile_py(
            self._snippet(), "demo_var", tmp_path, files_by_uuid={self.UUID: target}
        )
        assert func(var=make_context(), cohort=None).to_dicts() == [{"a": 1}]  # type: ignore[arg-type]

    def test_returns_a_path(self, tmp_path: Path) -> None:
        target = tmp_path / "a.csv"
        target.write_text("a\n1\n")
        func = compile_py(
            "def demo_var(var, cohort):\n    return getfile(UUID)\n".replace(
                "UUID", repr(self.UUID)
            ),
            "demo_var",
            tmp_path,
            files_by_uuid={self.UUID: target},
        )
        result = func(var=make_context(), cohort=None)  # type: ignore[arg-type]
        assert isinstance(result, Path)
        assert result == target

    def test_manifest_is_exposed_on_the_compiled_function(self, tmp_path: Path) -> None:
        target = tmp_path / "a.csv"
        func = compile_py(
            SIMPLE_SNIPPET, "demo_var", tmp_path, files_by_uuid={self.UUID: target}
        )
        assert func.files_by_uuid == {self.UUID: target}

    def test_unknown_uuid_names_the_variable_and_the_uuid(self, tmp_path: Path) -> None:
        func = compile_py(
            self._snippet("not-a-known-uuid"),
            "demo_var",
            tmp_path,
            files_by_uuid={self.UUID: tmp_path / "a.csv"},
        )
        with pytest.raises(VariableDefinitionError) as excinfo:
            func(var=make_context(), cohort=None)  # type: ignore[arg-type]
        message = str(excinfo.value)
        assert "not-a-known-uuid" in message
        assert "demo_var" in message
        # It also says what the definition *does* pin, so the fix is obvious.
        assert self.UUID in message

    def test_a_config_pinning_no_files_still_answers(self, tmp_path: Path) -> None:
        # Not a NameError: a snippet calling getfile() against an empty manifest
        # has to be told the manifest is empty.
        func = compile_py(self._snippet(), "demo_var", tmp_path)
        with pytest.raises(VariableDefinitionError, match="none"):
            func(var=make_context(), cohort=None)  # type: ignore[arg-type]

    def test_getfile_is_declared_as_an_injected_name(self) -> None:
        # Without this the py-snippet-namespace standard reports every
        # definition that calls getfile as reading an undeclared name.
        assert "getfile" in INJECTED_NAMES
        assert "getfile" in standards.INJECTED_NAMES
        assert "getfile" not in NAMESPACE_NAMES

    def test_the_publication_standard_accepts_a_snippet_using_getfile(
        self, tmp_path: Path
    ) -> None:
        # The standard reads a source's namespace module and flags any module-level
        # name a variable function reads that the namespace does not carry.
        # `getfile` must pass that check the way `__file__` does.
        source = textwrap.dedent("""
            def demo_var(var, cohort):
                return pl.read_csv(getfile("2f3a"))
            """)
        path = tmp_path / "variables.py"
        path.write_text(source)
        table = symtable.symtable(source, str(path), "exec")
        func_table = next(
            child for child in table.get_children() if child.get_name() == "demo_var"
        )
        reads = {
            name
            for name in func_table.get_identifiers()
            if func_table.lookup(name).is_global()
        }
        assert "getfile" in reads
        assert not reads - set(NAMESPACE_NAMES) - set(dir(builtins)) - INJECTED_NAMES


class TestGetPyCompiler:
    def test_source_without_py_env_returns_none(self) -> None:
        assert get_py_compiler("reprodicu") is None

    def test_unknown_source_returns_none(self) -> None:
        assert get_py_compiler("not_a_source") is None


# ---------------------------------------------------------------------------
# The file cache
# ---------------------------------------------------------------------------


class TestFileCache:
    def test_cache_root_honours_the_env_var(self, cache_dir: Path) -> None:
        assert cache_root() == cache_dir

    def test_materialise_mirrors_relative_paths(self, cache_dir: Path) -> None:
        content = b"postal_code,state\n10115,Berlin\n"
        file = concept_file("postcode/postcode_mapping.csv", content)
        client = file_client({"postcode/postcode_mapping.csv": content})

        files_dir, resolved = materialise_files(
            [file], spec=SPEC, source="demo_source", client=client
        )

        target = files_dir / "postcode" / "postcode_mapping.csv"
        assert target.read_bytes() == content
        # Laid out under its path (that is what ``__file__`` points into), but
        # keyed by uuid, since that is all a snippet's getfile() call says.
        assert resolved == {file.uuid: target}

    def test_a_manifest_entry_without_a_uuid_is_fatal(self, cache_dir: Path) -> None:
        file = ConceptFile(path="a.csv", sha256="x" * 64)
        client = file_client({"a.csv": b"x"})
        with pytest.raises(ConceptsApiError, match="carries no uuid"):
            materialise_files([file], spec=SPEC, source="demo_source", client=client)

    def test_blobs_are_content_addressed_and_downloaded_once(
        self, cache_dir: Path
    ) -> None:
        content = b"shared"
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(200, content=content)

        client = ConceptsApiClient(
            "https://concepts.test",
            project="demo",
            api_key="k",
            transport=httpx.MockTransport(handler),
            backoff=0.0,
        )
        file = concept_file("a.csv", content)
        other_spec = VariableSpec(name="other_var", taxonomy="corr_v1", version=LATEST)

        materialise_files([file], spec=SPEC, source="demo_source", client=client)
        materialise_files([file], spec=other_spec, source="demo_source", client=client)

        assert calls["n"] == 1

    def test_checksum_mismatch_is_fatal(self, cache_dir: Path) -> None:
        file = dataclasses.replace(
            concept_file("a.csv", b"actual content"), sha256="0" * 64
        )
        client = file_client({"a.csv": b"actual content"})
        with pytest.raises(ConceptsApiError, match="checksum"):
            materialise_files([file], spec=SPEC, source="demo_source", client=client)

    @pytest.mark.parametrize("path", ["/etc/passwd", "../escape.csv", "a/../../b", ""])
    def test_unsafe_paths_are_rejected(self, cache_dir: Path, path: str) -> None:
        file = ConceptFile(path=path, uuid=file_uuid(path), sha256="x")
        client = file_client({})
        with pytest.raises(ConceptsApiError):
            materialise_files([file], spec=SPEC, source="demo_source", client=client)

    def test_discover_files_on_missing_directory(self, tmp_path: Path) -> None:
        assert discover_files(tmp_path / "nope") == {}

    def test_rebase_paths_only_touches_paths_under_the_anchor(self) -> None:
        anchor = Path("/pkg/mapping")
        rebased = rebase_paths(
            {
                "inside": anchor / "postcode" / "m.csv",
                "outside": Path("/other/x.csv"),
                "not_a_path": "string",
            },
            anchor,
            Path("/files"),
        )
        assert rebased == {"inside": Path("/files/postcode/m.csv")}
