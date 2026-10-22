"""Download the charite_demo dataset for offline use.

Reading the dataset straight from its URL transfers only what a query touches,
but every query pays the network round-trips. :func:`download` copies the
dataset (or a subset of its tables) into a local folder once; pass that folder
as the source's ``path`` afterwards.
"""

from __future__ import annotations

import json
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

    Files already present with the size the server reports are skipped, and a
    partially downloaded file is resumed with a range request, so an interrupted
    download can simply be restarted. The folder gets its own ``manifest.json``
    listing exactly the downloaded tables.

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
        httpx.HTTPError: If a file cannot be downloaded.
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
    with httpx.Client(
        timeout=httpx.Timeout(DOWNLOAD_TIMEOUT), follow_redirects=True
    ) as client:
        for rel in tqdm(files, desc="charite_demo", unit="file"):
            _download_file(client, f"{base}/{rel.lstrip('/')}", target / rel)

    # Written last: a folder with a manifest is a complete dataset.
    (target / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
    logger.info(
        __(
            "Downloaded {n} charite_demo file(s) to {target}",
            n=len(files),
            target=target,
        )
    )
    return str(target)


def _download_file(client: httpx.Client, url: str, path: Path) -> None:
    """Download one file, skipping it when complete and resuming it when partial.

    Args:
        client (httpx.Client): Client to download with.
        url (str): File URL.
        path (Path): Local destination.
    """
    head = client.head(url)
    head.raise_for_status()
    size = int(head.headers.get("content-length", -1))
    if size >= 0 and path.is_file() and path.stat().st_size == size:
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
    partial.replace(path)
