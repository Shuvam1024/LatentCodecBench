"""Clip manifests, verified downloads, and preprocessing.

``lcb data fetch`` reads the YAML manifest, downloads each source, and
deletes the file if its SHA256 does not match. Surviving files are scaled
and cropped to the canonical 832x480, 33-frame y4m clip. The video bytes
themselves are not part of the repository.
"""

from latentcodecbench.data.canonical import CANONICAL_FRAMES, CANONICAL_HEIGHT, CANONICAL_WIDTH
from latentcodecbench.data.fetch import HashMismatchError, fetch_clip
from latentcodecbench.data.manifest import ClipSpec, Manifest, load_manifest
from latentcodecbench.data.preprocess import preprocess_clip

__all__ = [
    "CANONICAL_FRAMES",
    "CANONICAL_HEIGHT",
    "CANONICAL_WIDTH",
    "ClipSpec",
    "HashMismatchError",
    "Manifest",
    "fetch_clip",
    "load_manifest",
    "preprocess_clip",
]
