"""Download a manifest entry and refuse it when the SHA256 does not match."""

from __future__ import annotations

import hashlib
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from latentcodecbench.data.manifest import ClipSpec


class HashMismatchError(RuntimeError):
    """Downloaded bytes did not match the manifest digest. The file was removed."""

    def __init__(self, clip_id: str, expected: str, actual: str) -> None:
        super().__init__(
            f"{clip_id}: SHA256 mismatch (expected {expected}, got {actual}). The file was deleted."
        )
        self.clip_id = clip_id
        self.expected = expected
        self.actual = actual


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_clip(clip: ClipSpec, raw_dir: Path, *, timeout: float = 120.0) -> Path:
    """Download ``clip`` into ``raw_dir / {id}.src`` and verify its SHA256.

    A file that is already present and already matches is kept. A file that
    is present and does not match is deleted and downloaded again. A download
    that does not match is deleted and raises :class:`HashMismatchError`.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / f"{clip.id}.src"
    if destination.exists() and sha256_file(destination) == clip.sha256:
        return destination
    if destination.exists():
        destination.unlink()
    temporary = destination.with_suffix(".src.partial")
    try:
        _download(clip, temporary, timeout=timeout)
        actual = sha256_file(temporary)
        if actual != clip.sha256:
            temporary.unlink(missing_ok=True)
            raise HashMismatchError(clip.id, clip.sha256, actual)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _download(clip: ClipSpec, destination: Path, *, timeout: float) -> None:
    parsed = urlparse(clip.source_url)
    if parsed.scheme == "file":
        _copy_file(Path(parsed.path), destination, clip)
        return
    if parsed.scheme in {"", "path"} or clip.source_url.startswith("/"):
        _copy_file(Path(clip.source_url), destination, clip)
        return
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"{clip.id}: unsupported URL scheme {parsed.scheme!r}")
    request = urllib.request.Request(clip.source_url)
    if clip.byte_length is not None:
        end = clip.byte_start + clip.byte_length - 1
        request.add_header("Range", f"bytes={clip.byte_start}-{end}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if clip.byte_length is not None and response.status != 206:
                raise RuntimeError(
                    f"{clip.id}: server returned HTTP {response.status} for a byte range. "
                    "Refusing to read an unbounded body."
                )
            with destination.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{clip.id}: download failed with HTTP {exc.code}") from exc
    if clip.byte_length is not None and destination.stat().st_size != clip.byte_length:
        raise RuntimeError(
            f"{clip.id}: downloaded {destination.stat().st_size} bytes, "
            f"manifest asks for {clip.byte_length}"
        )


def _copy_file(source: Path, destination: Path, clip: ClipSpec) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    if clip.byte_length is None and clip.byte_start == 0:
        shutil.copyfile(source, destination)
        return
    if clip.byte_length is None:
        length = source.stat().st_size - clip.byte_start
    else:
        length = clip.byte_length
    with source.open("rb") as src, destination.open("wb") as dst:
        src.seek(clip.byte_start)
        remaining = length
        while remaining > 0:
            chunk = src.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            dst.write(chunk)
            remaining -= len(chunk)
    if destination.stat().st_size != length:
        raise RuntimeError(f"{clip.id}: local source is shorter than the manifest range")
