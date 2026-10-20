"""Download verification: a bad hash is deleted, a good hash is kept."""

from __future__ import annotations

import hashlib
import io
import urllib.request
from pathlib import Path

import pytest

from latentcodecbench import __version__
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


def test_https_download_sends_a_descriptive_user_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"commons-bytes"
    digest = hashlib.sha256(payload).hexdigest()
    captured: list[urllib.request.Request] = []

    class _Body:
        status = 200

        def __init__(self) -> None:
            self._buffer = io.BytesIO(payload)

        def read(self, size: int = -1) -> bytes:
            return self._buffer.read(size)

        def __enter__(self) -> _Body:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

    def fake_urlopen(request: urllib.request.Request, timeout: float = 120.0) -> _Body:
        captured.append(request)
        assert timeout == 120.0
        return _Body()

    monkeypatch.setattr("latentcodecbench.data.fetch.urllib.request.urlopen", fake_urlopen)
    clip = ClipSpec(
        id="street_50",
        source_url="https://upload.wikimedia.org/wikipedia/commons/3/30/50_Street_video_vc.ogv",
        sha256=digest,
        license="CC BY 2.0",
        license_url="https://creativecommons.org/licenses/by/2.0/",
        native_width=1440,
        native_height=1080,
        native_fps_num=25,
        native_fps_den=1,
        category="ugc",
    )
    fetched = fetch_clip(clip, tmp_path / "raw")
    assert fetched.read_bytes() == payload
    assert len(captured) == 1
    header = captured[0].get_header("User-agent")
    expected = f"LatentCodecBench/{__version__} (https://github.com/Shuvam1024/LatentCodecBench)"
    assert header == expected
