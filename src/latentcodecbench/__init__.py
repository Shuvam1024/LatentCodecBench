"""LatentCodecBench: open video VAEs measured as codecs.

Modules follow the project phases:

- :mod:`latentcodecbench.data` — clip manifests and preprocessing
- :mod:`latentcodecbench.metrics` — distortion and temporal metrics
- :mod:`latentcodecbench.codecs` — x264, x265, and SVT-AV1 baselines
- :mod:`latentcodecbench.vaes` — open video VAE wrappers
- :mod:`latentcodecbench.analysis` — rate-distortion and codec-leak studies
- :mod:`latentcodecbench.postfilter` — placeholder for a decoder-side post-filter
"""

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:
    __version__ = version("latentcodecbench")
except PackageNotFoundError:
    __version__ = "0.0.1"
