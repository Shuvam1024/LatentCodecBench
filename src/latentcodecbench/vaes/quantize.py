"""Turn a posterior mean into a zstd bitstream and back into a latent.

The quantizer is uniform and per channel, after the published
normalization in :mod:`latentcodecbench.vaes.scale`. Codes are stored as
the smallest unsigned integer that holds them and compressed with zstd.
A 24-byte header in front of that payload is side information, and
``bits`` counts both. The factorized Gaussian cross-entropy is a separate
estimate: it is not a file size. See ADR-0010.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from latentcodecbench.codecs.rate import bits_per_pixel, kilobits_per_second
from latentcodecbench.vaes.scale import LatentScale, denormalize, normalize

BIT_DEPTHS: tuple[int, ...] = (2, 4, 6, 8, 10, 12)
QUANT_LOW = -4.0
QUANT_HIGH = 4.0
ZSTD_LEVEL = 3
_MAGIC = b"LCBQ"
_HEADER = struct.Struct("<4sBBHHHHHff")


class QuantizeError(ValueError):
    """The latent could not be quantized or the payload could not be read."""


@dataclass(frozen=True)
class LatentPayload:
    """One compressed latent and the rate of that file."""

    bit_depth: int
    channels: int
    channels_kept: int
    frames: int
    height: int
    width: int
    qmin: float
    qmax: float
    codes: npt.NDArray[np.uint16]
    compressed: bytes
    payload: bytes
    bits: int
    entropy_bits: float


def measure_latent(
    latent: npt.NDArray[np.floating[Any]],
    scale: LatentScale,
    bit_depth: int,
    *,
    channels: int | None = None,
    low: float = QUANT_LOW,
    high: float = QUANT_HIGH,
    level: int = ZSTD_LEVEL,
) -> tuple[LatentPayload, npt.NDArray[np.float64]]:
    """Quantize ``latent``, compress the codes, and rebuild the decoder input.

    ``latent`` is ``(channels, frames, height, width)``, the posterior mean.
    ``channels`` keeps a prefix and leaves the rest at 0 in the tensor that
    ``decode`` receives. The returned array is that tensor.
    """
    if bit_depth not in BIT_DEPTHS:
        raise QuantizeError(f"bit depth must be one of {BIT_DEPTHS}, got {bit_depth}")
    if not low < high:
        raise QuantizeError(f"quantizer range must be increasing, got {low} .. {high}")
    array = np.asarray(latent, dtype=np.float64)
    if array.ndim != 4:
        raise QuantizeError(f"latent must be 4D, got {array.ndim}")
    count, frames, height, width = (int(v) for v in array.shape)
    if min(frames, height, width) < 1:
        raise QuantizeError("latent axes must be positive")
    kept = count if channels is None else channels
    if kept < 1 or kept > count:
        raise QuantizeError(f"channels must be in 1..{count}, got {kept}")
    normalized = normalize(array, scale)
    codes = quantize_codes(normalized[:kept], bit_depth, low=low, high=high)
    stored = _store_codes(codes, bit_depth)
    compressed = _compress(stored.tobytes(), level)
    header = _HEADER.pack(
        _MAGIC,
        1,
        bit_depth,
        count,
        kept,
        frames,
        height,
        width,
        float(low),
        float(high),
    )
    payload = header + compressed
    restored_kept = denormalize(
        dequantize_codes(codes, bit_depth, low=low, high=high),
        _prefix_scale(scale, kept),
    )
    restored = np.zeros_like(array)
    restored[:kept] = restored_kept
    result = LatentPayload(
        bit_depth=bit_depth,
        channels=count,
        channels_kept=kept,
        frames=frames,
        height=height,
        width=width,
        qmin=float(low),
        qmax=float(high),
        codes=codes,
        compressed=compressed,
        payload=payload,
        bits=8 * len(payload),
        entropy_bits=gaussian_cross_entropy_bits(codes, bit_depth, low=low, high=high),
    )
    return result, restored


def quantize_codes(
    normalized: npt.NDArray[np.floating[Any]],
    bit_depth: int,
    *,
    low: float,
    high: float,
) -> npt.NDArray[np.uint16]:
    """Nearest reconstruction point on ``[low, high]``, as an integer code."""
    levels = 1 << bit_depth
    step = (high - low) / (levels - 1)
    clipped = np.clip(np.asarray(normalized, dtype=np.float64), low, high)
    index = np.rint((clipped - low) / step).astype(np.int64)
    return np.clip(index, 0, levels - 1).astype(np.uint16)


def dequantize_codes(
    codes: npt.NDArray[np.integer[Any]],
    bit_depth: int,
    *,
    low: float,
    high: float,
) -> npt.NDArray[np.float64]:
    """Map codes back to normalized values. Endpoints land on ``low`` and ``high``."""
    levels = 1 << bit_depth
    step = (high - low) / (levels - 1)
    return low + np.asarray(codes, dtype=np.float64) * step


def gaussian_cross_entropy_bits(
    codes: npt.NDArray[np.integer[Any]],
    bit_depth: int,
    *,
    low: float,
    high: float,
) -> float:
    """Cross-entropy, in bits, of the codes under a factorized standard normal.

    The prior is fixed before the clip: mean 0 and variance 1 in normalized
    space, so its parameters add nothing to the count. Outer bins include
    the tails. This is an estimate of a codebook, not the size of a file.
    """
    levels = 1 << bit_depth
    step = (high - low) / (levels - 1)
    edges = np.empty(levels + 1, dtype=np.float64)
    edges[0] = -np.inf
    edges[-1] = np.inf
    edges[1:-1] = low + (np.arange(1, levels, dtype=np.float64) - 0.5) * step
    mass = norm.cdf(edges[1:]) - norm.cdf(edges[:-1])
    probabilities = np.maximum(mass, np.finfo(np.float64).tiny)
    log_prob = np.log2(probabilities)
    flat = np.asarray(codes, dtype=np.int64).ravel()
    if flat.size == 0 or int(flat.min()) < 0 or int(flat.max()) >= levels:
        raise QuantizeError("codes fall outside the bit depth")
    return float(-log_prob[flat].sum())


def pixel_rate(
    bits: int,
    *,
    width: int,
    height: int,
    frames: int,
    fps_num: int,
    fps_den: int,
) -> tuple[float, float]:
    """``bpp`` and ``kbps`` from pixel geometry, the same denominator as a classic row."""
    return (
        bits_per_pixel(bits, width, height, frames),
        kilobits_per_second(bits, frames, fps_num, fps_den),
    )


def parse_payload(payload: bytes) -> tuple[LatentPayload, npt.NDArray[np.uint16]]:
    """Read a payload written by :func:`measure_latent` and return its codes.

    The returned payload's ``entropy_bits`` is recomputed. ``bits`` is
    ``8 * len(payload)``.
    """
    if len(payload) < _HEADER.size:
        raise QuantizeError("payload is shorter than the header")
    header = _HEADER.unpack_from(payload)
    magic, version, bit_depth, channels, kept, frames, height, width, low, high = header
    if magic != _MAGIC or version != 1:
        raise QuantizeError("payload header is not a latent quantizer file")
    if bit_depth not in BIT_DEPTHS:
        raise QuantizeError(f"payload bit depth {bit_depth} is not supported")
    compressed = payload[_HEADER.size :]
    raw = _decompress(compressed)
    dtype = np.dtype("<u2") if bit_depth > 8 else np.dtype("<u1")
    count = kept * frames * height * width
    if len(raw) != count * dtype.itemsize:
        raise QuantizeError("compressed codes do not match the header shape")
    codes = np.frombuffer(raw, dtype=dtype).astype(np.uint16).reshape(kept, frames, height, width)
    entropy = gaussian_cross_entropy_bits(codes, int(bit_depth), low=float(low), high=float(high))
    described = LatentPayload(
        bit_depth=int(bit_depth),
        channels=int(channels),
        channels_kept=int(kept),
        frames=int(frames),
        height=int(height),
        width=int(width),
        qmin=float(low),
        qmax=float(high),
        codes=codes,
        compressed=compressed,
        payload=payload,
        bits=8 * len(payload),
        entropy_bits=entropy,
    )
    return described, codes


def _prefix_scale(scale: LatentScale, kept: int) -> LatentScale:
    return LatentScale(name=scale.name, shift=scale.shift[:kept], scale=scale.scale[:kept])


def _store_codes(codes: npt.NDArray[np.uint16], bit_depth: int) -> npt.NDArray[Any]:
    if bit_depth > 8:
        return np.ascontiguousarray(codes, dtype="<u2")
    return np.ascontiguousarray(codes, dtype="<u1")


def _compress(data: bytes, level: int) -> bytes:
    import zstandard as zstd

    compressor: Any = zstd.ZstdCompressor(level=level)
    compressed: bytes = compressor.compress(data)
    return compressed


def _decompress(data: bytes) -> bytes:
    import zstandard as zstd

    decompressor: Any = zstd.ZstdDecompressor()
    raw: bytes = decompressor.decompress(data)
    return raw
