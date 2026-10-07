"""Download verification: a bad hash is deleted, a good hash is kept."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from latentcodecbench.data.fetch import HashMismatchError, fetch_clip
from latentcodecbench.data.manifest import ClipSpec


def _clip(path: Path, digest: str, *, byte_length: int | None = None) -> ClipSpec:
    return ClipSpec(
        id="sample",
        source_url=path.as_uri(),
        sha256=digest,
        license="test",
        license_url="https://example.com/license",
        native_width=2,
        native_height=2,
        native_fps_num=25,
        native_fps_den=1,
        category="test",
        byte_length=byte_length,
    )


def test_matching_hash_is_kept(tmp_path: Path) -> None:
    payload = b"canonical-bytes"
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    fetched = fetch_clip(_clip(source, digest), tmp_path / "raw")
    assert fetched.read_bytes() == payload
    again = fetch_clip(_clip(source, digest), tmp_path / "raw")
    assert again == fetched


def test_mismatch_deletes_the_file(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"not-the-bytes-we-promised")
    raw = tmp_path / "raw"
    with pytest.raises(HashMismatchError, match="sample"):
        fetch_clip(_clip(source, "a" * 64), raw)
    assert not (raw / "sample.src").exists()
    assert not (raw / "sample.src.partial").exists()


def test_byte_range_is_honored(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"HEADER" + b"BODY")
    body = b"BODY"
    digest = hashlib.sha256(body).hexdigest()
    clip = ClipSpec(
        id="ranged",
        source_url=source.as_uri(),
        sha256=digest,
        license="test",
        license_url="https://example.com/license",
        native_width=2,
        native_height=2,
        native_fps_num=1,
        native_fps_den=1,
        category="test",
        byte_start=6,
        byte_length=4,
    )
    fetched = fetch_clip(clip, tmp_path / "raw")
    assert fetched.read_bytes() == body
