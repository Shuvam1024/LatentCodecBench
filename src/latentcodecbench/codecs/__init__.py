"""Classic codec baselines.

``encode_y4m`` drives ffmpeg's libx264, libx265, and libsvtav1 at a fixed
preset and one thread, and records bits per pixel, kilobits per second,
and encode and decode time. CRF ladders live next to the presets.
``decode_y4m`` writes pixels, and ``degrade_sources`` re-encodes the
study ladder for Phase 3. The runners expect the pinned ffmpeg from
``scripts/install-ffmpeg.sh``.
"""

from latentcodecbench.codecs.baseline import (
    BENCHMARK_PRESET,
    SVT_AV1_CRF,
    X264_CRF,
    X265_CRF,
    EncodeError,
    RatePoint,
    decode_y4m,
    encode_y4m,
    encoder_args,
)
from latentcodecbench.codecs.degrade import STUDY_LADDER, degrade_sources
from latentcodecbench.codecs.rate import bits_per_pixel, kilobits_per_second

__all__ = [
    "BENCHMARK_PRESET",
    "STUDY_LADDER",
    "SVT_AV1_CRF",
    "X264_CRF",
    "X265_CRF",
    "EncodeError",
    "RatePoint",
    "bits_per_pixel",
    "decode_y4m",
    "degrade_sources",
    "encode_y4m",
    "encoder_args",
    "kilobits_per_second",
]
