"""Manifest validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from latentcodecbench.data.manifest import load_manifest

_GOOD = """\
version: 1
canonical:
  width: 832
  height: 480
  num_frames: 33
  pix_fmt: yuv420p
clips:
  - id: sample
    source_url: file:///tmp/sample.y4m
    sha256: {sha}
    license: test
    license_url: https://example.com/license
    native_width: 16
    native_height: 16
    native_fps_num: 25
    native_fps_den: 1
    category: test
    note: a fixture
"""


def test_loads_a_valid_manifest(tmp_path: Path) -> None:
    path = tmp_path / "clips.yaml"
    path.write_text(_GOOD.format(sha="ab" * 32), encoding="utf-8")
    manifest = load_manifest(path)
    assert manifest.clips[0].id == "sample"
    assert manifest.clips[0].native_fps == 25


def test_rejects_a_bad_digest(tmp_path: Path) -> None:
    path = tmp_path / "clips.yaml"
    path.write_text(_GOOD.format(sha="abcd"), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        load_manifest(path)


def test_checked_in_manifest_is_the_phase1_set() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = load_manifest(root / "manifests" / "clips.yaml")
    assert len(manifest.clips) == 34
    assert len({clip.id for clip in manifest.clips}) == 34
    ranged = [clip for clip in manifest.clips if clip.byte_length is not None]
    assert len(ranged) == 27
    archives = {clip.id: clip.inner_path for clip in manifest.clips if clip.archive == "7z"}
    assert archives == {
        "shaken_dry": "ShakeNDry_3840x2160_120fps_420_10bit_YUV.y4m",
        "bosphorus": "Bosphorus_3840x2160_120fps_420_10bit_YUV.y4m",
        "honey_bee": "HoneyBee_3840x2160_120fps_420_10bit_YUV.y4m",
    }
    compressed = {"street_50", "lawn_mower", "rust_laser", "volcano"}
    assert compressed <= {clip.id for clip in manifest.clips}


def test_rejects_the_wrong_canonical_shape(tmp_path: Path) -> None:
    path = tmp_path / "clips.yaml"
    text = _GOOD.format(sha="ab" * 32).replace("width: 832", "width: 640")
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="canonical"):
        load_manifest(path)
