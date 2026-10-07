"""Classic codec baselines.

``encode_y4m`` drives ffmpeg's libx264, libx265, and libsvtav1 at a fixed
preset and one thread, and records bits per pixel, kilobits per second,
and encode and decode time. CRF ladders live next to the presets. The
runners expect the pinned ffmpeg from ``scripts/install-ffmpeg.sh``.
"""

from latentcodecbench.codecs.baseline import (
    BENCHMARK_PRESET,
    SVT_AV1_CRF,
    X264_CRF,
    X265_CRF,
    EncodeError,
    RatePoint,
    encode_y4m,
    encoder_args,
)
from latentcodecbench.codecs.rate import bits_per_pixel, kilobits_per_second

__all__ = [
    "BENCHMARK_PRESET",
    "SVT_AV1_CRF",
    "X264_CRF",
    "X265_CRF",
    "EncodeError",
    "RatePoint",
    "bits_per_pixel",
    "encode_y4m",
    "encoder_args",
    "kilobits_per_second",
]
