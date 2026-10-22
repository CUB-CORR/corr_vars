"""Download the charite_demo dataset for offline use.

Reading the dataset straight from its URL transfers only what a query touches,
but every query pays the network round-trips. :func:`download` copies the
dataset (or a subset of its tables) into a local folder once; pass that folder
as the source's ``path`` afterwards.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import httpx
from tqdm.auto import tqdm

from corr_vars import __, logger
from corr_vars.sources.charite_demo.config import DEFAULT_PATH
from corr_vars.sources.charite_demo.connector import (
    MANIFEST_FILE,
    fetch_manifest,
    is_url,
    normalize_path,
    validate_file_path,
)
from corr_vars.sources.charite_demo.helpers import clear_connections

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

#: Timeout (seconds) for connecting and for each read while downloading a file.
DOWNLOAD_TIMEOUT: float = 60.0

#: Chunk size (bytes) files are streamed in.
CHUNK_SIZE: int = 1 << 20

#: A valid ``dataset`` name: the subfolder the table files live in.
_DATASET_PATTERN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")


def download(
    dest: str | Path,
    base_url: str | None = None,
    tables: Iterable[str] | None = None,
) -> str:
    """Download the charite_demo dataset into a local folder.

    Files already present are skipped: when the manifest records a ``sha256``
    for a file (``manifest["files"][path]["sha256"]``) the local copy must match
    it, otherwise its size must match the size the server reports. A partially
    downloaded file is resumed with a range request, so an interrupted download
    can simply be restarted, and every downloaded file is checked against its
    ``sha256`` when the manifest has one.

    The published file names carry a content hash, so a new release adds new
    files. Parquet files under the dataset's own subfolder (``<dest>/<dataset>/``,
    e.g. ``charite_mockdata/``) that the new manifest does not list are removed
    after the download; nothing outside that subfolder is touched. The folder
    gets its own ``manifest.json`` listing exactly the tables on disk.

    The release also carries a validation set (manifest key ``validation``,
    folder ``validation/``: the generator's parameters per ICU stay and case,
    fully synthetic) for checking concept implementations. Cohorts do not need
    it, so it is not downloaded; read it from the release URL if needed.

    With `tables`, the download refreshes only those tables: the other tables of
    an existing local copy stay listed (as long as their files are present and
    match their recorded size and sha256), and only stale files of the
    requested tables are removed.

    Every file path in the manifest must be a ``.parquet`` file under
    ``<dataset>/``; a manifest listing any other path (absolute, with ``..``,
    a backslash or a scheme) is rejected before anything is written.

    .. code-block:: python

        from corr_vars.sources import charite_demo

        path = charite_demo.download("~/data/charite_mockdata")
        cohort = Cohort(obs_level="icu_stay", sources={"charite_demo": {"path": path}})

    Args:
        dest (str | Path): Target folder (``~`` is expanded); created if needed.
        base_url (str | None): Dataset URL. Defaults to the published dataset.
        tables (Iterable[str] | None): Tables to download. Defaults to all.

    Returns:
        str: The absolute path of `dest`, usable as the source's ``path``.

    Raises:
        ValueError: If `base_url` is not a URL, `tables` names a table the
            manifest does not list, or the manifest lists an unsafe file path.
        ConnectionError: If the manifest cannot be fetched.
        httpx.HTTPError: If a file cannot be downloaded or fails its checksum.
    """
    base = normalize_path(base_url or DEFAULT_PATH)
    if not is_url(base):
        raise ValueError(f"base_url must be an http(s) URL, got {base_url!r}")
    target = Path(dest).expanduser().resolve()

    manifest = fetch_manifest(base)
    available = manifest["tables"]
    wanted: list[str] | None = None
    if tables is not None:
        wanted = list(dict.fromkeys(tables))
        unknown = sorted(set(wanted) - set(available))
        if unknown:
            raise ValueError(
                f"Unknown table(s) {unknown}; the manifest lists {sorted(available)}"
            )
        manifest = {**manifest, "tables": {t: available[t] for t in wanted}}

    dataset = manifest.get("dataset")
    if not isinstance(dataset, str) or not _DATASET_PATTERN.fullmatch(dataset):
        raise ValueError(
            f"The charite_demo manifest at {base} names no valid 'dataset' folder: "
            f"{dataset!r}"
        )
    files = [f for info in manifest["tables"].values() for f in info.get("files", [])]
    # Validate every path before anything is written.
    for rel in files:
        _check_target_path(target, rel, dataset)
    index = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}

    target.mkdir(parents=True, exist_ok=True)
    # A subset refresh keeps the other tables of an existing local copy.
    kept_tables, kept_index = (
        _kept_tables(target, dataset, exclude=set(wanted))
        if wanted is not None
        else ({}, {})
    )
    # The cached connector of this folder would point at replaced files.
    clear_connections(str(target))

    with httpx.Client(
        timeout=httpx.Timeout(DOWNLOAD_TIMEOUT), follow_redirects=True
    ) as client:
        for rel in tqdm(files, desc="charite_demo", unit="file"):
            info = index.get(rel) if isinstance(index.get(rel), dict) else {}
            _download_file(client, f"{base}/{rel}", target / rel, info.get("sha256"))

    # The local manifest describes exactly what is on disk (the validation set
    # under "validation" is not downloaded, so it is not listed).
    local = {k: v for k, v in manifest.items() if k != "validation"}
    local["tables"] = {**kept_tables, **manifest["tables"]}
    local_index = {**kept_index, **{f: index[f] for f in files if f in index}}
    if local_index:
        local["files"] = local_index
    else:
        local.pop("files", None)
    # Written last: a folder with a manifest is a complete dataset.
    (target / MANIFEST_FILE).write_text(json.dumps(local, indent=2))
    keep = [f for info in local["tables"].values() for f in info.get("files", [])]
    removed = _remove_stale(target, dataset, keep, tables=wanted)
    logger.info(
        __(
            "Downloaded {n} charite_demo file(s) to {target} ({removed} stale file(s) removed)",
            n=len(files),
            target=target,
            removed=removed,
        )
    )
    return str(target)


def _check_target_path(target: Path, rel: object, dataset: str) -> Path:
    """Validate a manifest file path and return where it is stored under `target`.

    Args:
        target (Path): The resolved download folder.
        rel (object): The manifest-relative path.
        dataset (str): The manifest's ``dataset`` name.

    Returns:
        Path: ``target / rel``.

    Raises:
        ValueError: If `rel` is unsafe (see
            :func:`~corr_vars.sources.charite_demo.connector.validate_file_path`)
            or would resolve outside `target`.
    """
    rel = validate_file_path(rel, dataset)
    path = target / rel
    if not path.resolve().is_relative_to(target):
        raise ValueError(
            f"Refusing unsafe file path {rel!r} in the charite_demo manifest: "
            f"it resolves outside {target}."
        )
    return path


def _kept_tables(
    target: Path, dataset: str, exclude: set[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Tables of an existing local copy that a subset download should keep.

    A table is kept when it is not being downloaded and every one of its files
    is present, safe and matches the size and sha256 its index entry records.

    Args:
        target (Path): The resolved download folder.
        dataset (str): The dataset name of the release being downloaded.
        exclude (set[str]): Tables being downloaded now.

    Returns:
        tuple: ``(tables, files)`` entries to carry over into the new manifest.
    """
    try:
        old = json.loads((target / MANIFEST_FILE).read_text())
    except (OSError, ValueError):
        return {}, {}
    if not isinstance(old, dict) or not isinstance(old.get("tables"), dict):
        return {}, {}
    old_index = old.get("files") if isinstance(old.get("files"), dict) else {}
    tables: dict[str, Any] = {}
    index: dict[str, Any] = {}
    for table, info in old["tables"].items():
        if table in exclude or not isinstance(info, dict):
            continue
        files = info.get("files", [])
        try:
            paths = [_check_target_path(target, f, dataset) for f in files]
        except (TypeError, ValueError) as error:
            logger.warning(
                __("Dropping local table {table}: {error}", table=table, error=error)
            )
            continue
        entries = {f: old_index[f] for f in files if isinstance(old_index.get(f), dict)}
        if not paths or not all(
            _matches(path, entries.get(f, {})) for f, path in zip(files, paths)
        ):
            logger.info(__("Dropping incomplete local table {table}", table=table))
            continue
        tables[table] = info
        index.update(entries)
    return tables, index


def _matches(path: Path, entry: dict[str, Any]) -> bool:
    """Whether `path` exists and matches the ``bytes``/``sha256`` of its index entry."""
    if not path.is_file():
        return False
    size = entry.get("bytes")
    if isinstance(size, int) and path.stat().st_size != size:
        return False
    sha256 = entry.get("sha256")
    return not sha256 or _sha256(path) == sha256


def _sha256(path: Path) -> str:
    """Hex sha256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_stale(
    target: Path, dataset: object, files: list[str], tables: list[str] | None = None
) -> int:
    """Delete Parquet files under ``<target>/<dataset>/`` that `files` does not list.

    Only ``*.parquet`` files (and their ``*.parquet.part`` leftovers) inside the
    dataset's own subfolder are candidates; anything else is left alone, and
    nothing is removed when the subfolder cannot be determined safely. With
    `tables`, only files of those tables (``<table>[-<hash>].parquet`` or
    anything under ``<table>/``) are candidates.

    Args:
        target (Path): The download folder.
        dataset (object): The manifest's ``dataset`` name (the subfolder).
        files (list[str]): Manifest-relative paths that must be kept.
        tables (list[str] | None): Restrict removal to these tables' files.

    Returns:
        int: Number of files removed.
    """
    if not isinstance(dataset, str) or not _DATASET_PATTERN.fullmatch(dataset):
        return 0
    if not files or not all(f.startswith(f"{dataset}/") for f in files):
        return 0
    root = (target / dataset).resolve()
    if not root.is_dir() or root.parent != target:
        return 0
    keep = {(target / f).resolve() for f in files}
    removed = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        if not (path.name.endswith(".parquet") or path.name.endswith(".parquet.part")):
            continue
        # a ``.part`` of a listed file is a resumable download, not a stale file
        owner = path.with_name(path.name.removesuffix(".part"))
        if owner.resolve() in keep:
            continue
        if tables is not None and not _belongs_to(owner.relative_to(root), tables):
            continue
        path.unlink()
        removed += 1
        logger.debug(__("Removed stale {path}", path=path))
    for d in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
        if not any(d.iterdir()):
            d.rmdir()
    return removed


def _belongs_to(rel: Path, tables: list[str]) -> bool:
    """Whether a file under the dataset folder holds data of one of `tables`."""
    if len(rel.parts) > 1:
        return rel.parts[0] in tables
    return any(
        re.fullmatch(rf"{re.escape(t)}(-[0-9a-f]+)?\.parquet", rel.name) for t in tables
    )


def _download_file(
    client: httpx.Client, url: str, path: Path, sha256: str | None = None
) -> None:
    """Download one file, skipping it when complete and resuming it when partial.

    Args:
        client (httpx.Client): Client to download with.
        url (str): File URL.
        path (Path): Local destination.
        sha256 (str | None): Expected hex sha256 of the file, if the manifest
            records one. A local copy that does not match is downloaded again.

    Raises:
        httpx.HTTPError: If the download is incomplete or fails the checksum.
    """
    if sha256 and path.is_file():
        if _sha256(path) == sha256:
            logger.debug(__("Skipping {path}: checksum matches", path=path))
            return
        logger.info(__("Re-downloading {path}: checksum mismatch", path=path))
        path.unlink()
    head = client.head(url)
    head.raise_for_status()
    size = int(head.headers.get("content-length", -1))
    if not sha256 and size >= 0 and path.is_file() and path.stat().st_size == size:
        logger.debug(__("Skipping {path}: already complete", path=path))
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    offset = partial.stat().st_size if partial.is_file() else 0
    if size >= 0 and offset > size:
        offset = 0
    if not (size >= 0 and offset == size):
        _fetch(client, url, partial, offset)

    if size >= 0 and partial.stat().st_size != size:
        raise httpx.HTTPError(
            f"Incomplete download of {url}: got {partial.stat().st_size} of {size} bytes"
        )
    if sha256 and _sha256(partial) != sha256:
        partial.unlink()
        raise httpx.HTTPError(
            f"Checksum mismatch for {url}; the partial file was removed"
        )
    partial.replace(path)


def _fetch(client: httpx.Client, url: str, partial: Path, offset: int) -> None:
    """Stream `url` into `partial`, resuming at `offset` when the server allows.

    A ``416 Range Not Satisfiable`` answer to a resume means the partial file
    is already complete; it is left for the caller to verify. A server that
    ignores the range answers ``200`` with the whole file, which then replaces
    the partial file.

    Args:
        client (httpx.Client): Client to download with.
        url (str): File URL.
        partial (Path): The ``.part`` file to write.
        offset (int): Bytes already in `partial`.

    Raises:
        httpx.HTTPError: If the request fails.
    """
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with client.stream("GET", url, headers=headers) as response:
        if offset and response.status_code == 416:
            logger.debug(__("{url}: partial file already complete", url=url))
            return
        response.raise_for_status()
        mode = "ab" if offset and response.status_code == 206 else "wb"
        with partial.open(mode) as fh:
            for chunk in response.iter_bytes(CHUNK_SIZE):
                fh.write(chunk)
