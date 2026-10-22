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
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

#: Timeout (seconds) for connecting and for each read while downloading a file.
DOWNLOAD_TIMEOUT: float = 60.0

#: Chunk size (bytes) files are streamed in.
CHUNK_SIZE: int = 1 << 20


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
    gets its own ``manifest.json`` listing exactly the downloaded tables.

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
        ValueError: If `base_url` is not a URL, or `tables` names a table the
            manifest does not list.
        ConnectionError: If the manifest cannot be fetched.
        httpx.HTTPError: If a file cannot be downloaded or fails its checksum.
    """
    base = normalize_path(base_url or DEFAULT_PATH)
    if not is_url(base):
        raise ValueError(f"base_url must be an http(s) URL, got {base_url!r}")
    target = Path(dest).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)

    manifest = fetch_manifest(base)
    available = manifest["tables"]
    if tables is not None:
        wanted = list(dict.fromkeys(tables))
        unknown = sorted(set(wanted) - set(available))
        if unknown:
            raise ValueError(
                f"Unknown table(s) {unknown}; the manifest lists {sorted(available)}"
            )
        manifest = {**manifest, "tables": {t: available[t] for t in wanted}}

    files = [f for info in manifest["tables"].values() for f in info.get("files", [])]
    index = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
    with httpx.Client(
        timeout=httpx.Timeout(DOWNLOAD_TIMEOUT), follow_redirects=True
    ) as client:
        for rel in tqdm(files, desc="charite_demo", unit="file"):
            info = index.get(rel) if isinstance(index.get(rel), dict) else {}
            _download_file(
                client, f"{base}/{rel.lstrip('/')}", target / rel, info.get("sha256")
            )

    # The local manifest describes exactly what was downloaded (the evaluation
    # sidecar under "ground_truth" is not downloaded, so it is not listed).
    local = {k: v for k, v in manifest.items() if k != "ground_truth"}
    if index:
        local["files"] = {f: index[f] for f in files if f in index}
    # Written last: a folder with a manifest is a complete dataset.
    (target / MANIFEST_FILE).write_text(json.dumps(local, indent=2))
    removed = _remove_stale(target, manifest.get("dataset"), files)
    logger.info(
        __(
            "Downloaded {n} charite_demo file(s) to {target} ({removed} stale file(s) removed)",
            n=len(files),
            target=target,
            removed=removed,
        )
    )
    return str(target)


def _sha256(path: Path) -> str:
    """Hex sha256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_stale(target: Path, dataset: object, files: list[str]) -> int:
    """Delete Parquet files under ``<target>/<dataset>/`` that `files` does not list.

    Only ``*.parquet`` files (and their ``*.parquet.part`` leftovers) inside the
    dataset's own subfolder are candidates; anything else is left alone, and
    nothing is removed when the subfolder cannot be determined safely.

    Args:
        target (Path): The download folder.
        dataset (object): The manifest's ``dataset`` name (the subfolder).
        files (list[str]): Manifest-relative paths that must be kept.

    Returns:
        int: Number of files removed.
    """
    if not isinstance(dataset, str) or not re.fullmatch(
        r"[A-Za-z0-9_][A-Za-z0-9_.-]*", dataset
    ):
        return 0
    if not files or not all(f.lstrip("/").startswith(f"{dataset}/") for f in files):
        return 0
    root = (target / dataset).resolve()
    if not root.is_dir() or root.parent != target:
        return 0
    keep = {(target / f.lstrip("/")).resolve() for f in files}
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
        path.unlink()
        removed += 1
        logger.debug(__("Removed stale {path}", path=path))
    for d in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
        if not any(d.iterdir()):
            d.rmdir()
    return removed


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
    headers = {"Range": f"bytes={offset}-"} if offset else {}

    with client.stream("GET", url, headers=headers) as response:
        response.raise_for_status()
        # A server that ignores the range answers 200 with the whole file.
        mode = "ab" if offset and response.status_code == 206 else "wb"
        with partial.open(mode) as fh:
            for chunk in response.iter_bytes(CHUNK_SIZE):
                fh.write(chunk)

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
