from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

from corr_vars import __, logger
from corr_vars.definitions.exceptions import ConceptsApiError

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from corr_vars.concepts.client import ConceptFile, ConceptsApiClient
    from corr_vars.concepts.spec import VariableSpec

    from collections.abc import Iterable, Mapping

CACHE_DIR_ENV_VAR: Final[str] = "CORR_VARS_CACHE_DIR"
BLOB_DIRNAME: Final[str] = "blobs"
MATERIALISED_DIRNAME: Final[str] = "materialised"

#: Path segment standing in for a concept id the API did not report.
UNKNOWN_CONCEPT_DIRNAME: Final[str] = "_unknown"


def cache_root() -> Path:
    """Return the corr_vars cache directory for concept assets.

    Honours ``CORR_VARS_CACHE_DIR`` and otherwise follows the XDG base
    directory convention (``$XDG_CACHE_HOME`` or ``~/.cache``).

    Returns:
        Path: The cache root. The directory is not created here.
    """
    override = os.environ.get(CACHE_DIR_ENV_VAR)
    if override:
        return Path(override).expanduser()

    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "corr_vars" / "concepts"


def _safe_relative_path(path: str) -> Path:
    """Validate an API-supplied relative file path.

    Args:
        path (str): The ``path`` field of a :class:`ConceptFile`.

    Returns:
        Path: The validated relative path.

    Raises:
        ConceptsApiError: If the path is empty, absolute, or escapes its
            directory via ``..``.
    """
    if not path:
        raise ConceptsApiError("Concept file entry has an empty path.")

    candidate = Path(path)
    if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
        raise ConceptsApiError(
            f"Concept file path {path!r} must be relative and must not contain '..'."
        )
    return candidate


def blob_path(sha256: str) -> Path:
    """Return the content-addressed cache location for a blob.

    Args:
        sha256 (str): Hex digest of the content.

    Returns:
        Path: ``<cache_root>/blobs/<first two hex chars>/<digest>``.
    """
    return cache_root() / BLOB_DIRNAME / sha256[:2] / sha256


def store_blob(sha256: str, content: bytes) -> Path:
    """Write `content` into the content-addressed blob cache.

    Args:
        sha256 (str): Expected hex digest, as reported by the API.
        content (bytes): The downloaded bytes.

    Returns:
        Path: Location of the cached blob.

    Raises:
        ConceptsApiError: If the content does not hash to `sha256`.
    """
    actual = hashlib.sha256(content).hexdigest()
    if sha256 and actual != sha256:
        raise ConceptsApiError(
            f"Downloaded concept file does not match its checksum "
            f"(expected {sha256}, got {actual})."
        )

    target = blob_path(actual)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".partial")
    tmp.write_bytes(content)
    os.replace(tmp, target)
    return target


def materialised_dir(
    spec: VariableSpec, source: str, concept_id: str | int | None = None
) -> Path:
    """Return the per-concept directory the attached files are laid out in.

    The concept id is part of the path because a name may resolve to several
    concepts, each with its own files under the same name and source.

    Args:
        spec (VariableSpec): The resolved variable reference.
        source (str): Source key the files belong to.
        concept_id (str | int | None): Id of the concept the files belong to.

    Returns:
        Path: ``<cache_root>/materialised/<taxonomy>/<source>/<name>/<concept_id>/<version>``.
    """
    return (
        cache_root()
        / MATERIALISED_DIRNAME
        / spec.taxonomy
        / source
        / spec.name
        / (UNKNOWN_CONCEPT_DIRNAME if concept_id is None else str(concept_id))
        / str(spec.version)
    )


def materialise_files(
    files: Iterable[ConceptFile],
    *,
    spec: VariableSpec,
    source: str,
    client: ConceptsApiClient,
    concept_id: str | int | None = None,
) -> tuple[Path, dict[str, Path]]:
    """Lay a config's manifest out on disk, keyed by file uuid.

    Content is cached by sha256, so a file that has been downloaded once is
    never fetched again, whichever concept or version references it — which is
    also what makes a second resolve of the same definition work offline. The
    bytes are still laid out under the manifest's ``path``, because that is what
    ``__file__`` points into and what ``var.files["postcode/…"]`` is keyed by;
    the *lookup* key returned here is the uuid, since that is the only thing a
    served snippet says (``getfile("<uuid>")``).

    Args:
        files (Iterable[ConceptFile]): Manifest entries from the concept
            document's source config.
        spec (VariableSpec): The resolved variable reference.
        source (str): Source key the files belong to.
        client (ConceptsApiClient): Client used to download missing blobs.
        concept_id (str | int | None): Id of the concept the files belong to,
            which keeps two concepts sharing a name in separate directories.

    Returns:
        tuple[Path, dict[str, Path]]: The per-concept directory, and a mapping
        from each file's uuid to its absolute location inside it.

    Raises:
        ConceptsApiError: If an entry carries no uuid, if a path is unsafe, or
            if a download fails its checksum.
    """
    target_dir = materialised_dir(spec, source, concept_id)
    resolved: dict[str, Path] = {}

    for file in files:
        if not file.uuid:
            raise ConceptsApiError(
                f"Concept file {file.path!r} of {spec} carries no uuid, so no "
                "getfile() call could ever reach it."
            )
        relative = _safe_relative_path(file.path)
        destination = target_dir / relative

        cached = blob_path(file.sha256) if file.sha256 else None
        if cached is None or not cached.exists():
            content = client.fetch_file(
                spec.taxonomy, spec.name, spec.version, file, concept_id
            )
            cached = store_blob(
                file.sha256 or hashlib.sha256(content).hexdigest(), content
            )
            logger.debug(
                __(
                    "Cached concept file {path} for {spec} ({size} bytes)",
                    path=file.path,
                    spec=spec,
                    size=len(content),
                )
            )

        if (
            not destination.exists()
            or destination.stat().st_size != cached.stat().st_size
        ):
            destination.parent.mkdir(parents=True, exist_ok=True)
            _link_or_copy(cached, destination)

        resolved[file.uuid] = destination

    if files:
        target_dir.mkdir(parents=True, exist_ok=True)

    return target_dir, resolved


def _link_or_copy(source: Path, destination: Path) -> None:
    """Hard-link `source` to `destination`, copying when linking is unavailable.

    Args:
        source (Path): Blob in the content-addressed cache.
        destination (Path): Location inside the per-variable directory.
    """
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def discover_files(files_dir: Path | str) -> dict[str, Path]:
    """List every file under `files_dir`, keyed by its relative path.

    Args:
        files_dir (Path | str): A per-variable directory produced by
            :func:`materialise_files`.

    Returns:
        dict[str, Path]: Mapping from posix relative path to absolute location.
        Empty when the directory does not exist.
    """
    root = Path(files_dir)
    if not root.is_dir():
        return {}

    return {
        path.relative_to(root).as_posix(): path
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def rebase_paths(
    namespace: Mapping[str, object], anchor: Path, files_dir: Path
) -> dict[str, Path]:
    """Rebase namespace constants that point into the bundled mapping directory.

    Module-level path constants such as
    ``POSTCODE_MAPPING_PATH = Path(__file__).parent / "postcode" / "postcode_mapping.csv"``
    are evaluated when ``variables.py`` is imported and therefore keep pointing
    at the packaged copy. When a snippet runs against materialised files those
    constants must follow ``__file__`` into `files_dir`.

    Args:
        namespace (Mapping[str, object]): Candidate namespace entries.
        anchor (Path): Directory the bundled constants are relative to.
        files_dir (Path): Directory the constants should point into instead.

    Returns:
        dict[str, Path]: Only the entries that were rebased.
    """
    rebased: dict[str, Path] = {}
    for name, value in namespace.items():
        if not isinstance(value, Path):
            continue
        try:
            relative = value.relative_to(anchor)
        except ValueError:
            continue
        rebased[name] = files_dir / relative
    return rebased
