"""Load a baseline-run config.

The checked-in file is ``configs/classic.yaml``. It repeats the CRF
ladders and presets in :mod:`latentcodecbench.codecs.baseline`. A test
locks the two copies together so a sweep cannot drift from the ADR.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_KNOWN = frozenset({"libx264", "libx265", "libsvtav1"})


@dataclass(frozen=True)
class CodecConfig:
    """One encoder in a sweep."""

    name: str
    preset: str
    crf: tuple[int, ...]


@dataclass(frozen=True)
class RunConfig:
    """What ``lcb baselines run`` will encode."""

    version: int
    codecs: tuple[CodecConfig, ...]


def load_config(path: Path) -> RunConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must be a mapping")
    if raw.get("version") != 1:
        raise ValueError(f"{path} has version {raw.get('version')!r}, want 1")
    codecs_raw = raw.get("codecs")
    if not isinstance(codecs_raw, dict) or not codecs_raw:
        raise ValueError(f"{path} has no codecs")
    codecs: list[CodecConfig] = []
    for name, body in codecs_raw.items():
        if name not in _KNOWN:
            raise ValueError(f"{path} codec {name!r} is not supported")
        if not isinstance(body, dict):
            raise ValueError(f"{path} codec {name} must be a mapping")
        crf_raw = body.get("crf")
        if not isinstance(crf_raw, list) or len(crf_raw) < 4:
            raise ValueError(f"{path} codec {name} needs at least 4 CRF points")
        crf = tuple(int(point) for point in crf_raw)
        if list(crf) != sorted(crf):
            raise ValueError(f"{path} codec {name} CRF points must be increasing")
        preset = body.get("preset")
        if preset is None:
            raise ValueError(f"{path} codec {name} is missing preset")
        codecs.append(CodecConfig(name=str(name), preset=str(preset), crf=crf))
    return RunConfig(version=1, codecs=tuple(codecs))
