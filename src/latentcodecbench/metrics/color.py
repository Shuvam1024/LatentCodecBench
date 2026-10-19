"""BT.709 limited-range YCbCr to RGB.

Canonical clips are 8-bit 4:2:0 with studio (limited) levels: luma
occupies 16..235 and chroma is centered at 128. PSNR-RGB and LPIPS need
RGB, so this is the one conversion those metrics share. VMAF stays in
YUV inside ffmpeg and does not use this module.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# Limited-range scale, then the BT.709 inverse matrix.
# Y goes 16..235 across 0..255, chroma excursions are 16..240.
_Y_SCALE = 255.0 / 219.0
_C_SCALE = 255.0 / 224.0
_R_FROM_CR = 1.5748
_G_FROM_CB = 0.18732427293064877
_G_FROM_CR = 0.46812427293064877
_B_FROM_CB = 1.8556


def upsample_420(plane: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Repeat each 4:2:0 sample across its 2x2 block.

    ``plane`` is ``(frames, height / 2, width / 2)``.
    """
    return np.repeat(np.repeat(plane, 2, axis=1), 2, axis=2)


def ycbcr_to_rgb(
    luma: npt.NDArray[np.uint8],
    cb: npt.NDArray[np.uint8],
    cr: npt.NDArray[np.uint8],
) -> npt.NDArray[np.float64]:
    """Return float RGB in 0..255, shape ``(frames, height, width, 3)``.

    Chroma is nearest-neighbor upsampled. Values are clipped into 0..255.
    """
    if luma.ndim != 3:
        raise ValueError(f"luma must be (frames, height, width), got {luma.shape}")
    cb_full = upsample_420(cb).astype(np.float64)
    cr_full = upsample_420(cr).astype(np.float64)
    if cb_full.shape != luma.shape or cr_full.shape != luma.shape:
        raise ValueError(f"chroma {cb.shape} does not upsample to luma {luma.shape}")
    y = (luma.astype(np.float64) - 16.0) * _Y_SCALE
    pb = (cb_full - 128.0) * _C_SCALE
    pr = (cr_full - 128.0) * _C_SCALE
    red = y + _R_FROM_CR * pr
    green = y - _G_FROM_CB * pb - _G_FROM_CR * pr
    blue = y + _B_FROM_CB * pb
    rgb = np.stack((red, green, blue), axis=-1)
    return np.clip(rgb, 0.0, 255.0)


def rgb_to_ycbcr(
    rgb: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Invert :func:`ycbcr_to_rgb` and return 4:2:0 planes.

    ``rgb`` is ``(frames, height, width, 3)`` in 0..255. Luma is rounded
    per pixel. Chroma is the rounded mean of each 2x2 block, because the
    canonical file is 4:2:0 and the forward path repeated each chroma
    sample across that block.
    """
    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError(f"rgb must be (frames, height, width, 3), got {rgb.shape}")
    frames, height, width, _channels = rgb.shape
    if height % 2 or width % 2:
        raise ValueError(f"4:2:0 needs even sides, got {width}x{height}")
    red = rgb[..., 0].astype(np.float64)
    green = rgb[..., 1].astype(np.float64)
    blue = rgb[..., 2].astype(np.float64)
    # Solve the forward matrix for y, then for the chroma that red and blue imply.
    # green = y - a * (blue - y) - b * (red - y), with a and b the green row.
    green_from_blue = _G_FROM_CB / _B_FROM_CB
    green_from_red = _G_FROM_CR / _R_FROM_CR
    scale = 1.0 + green_from_blue + green_from_red
    y = (green + green_from_blue * blue + green_from_red * red) / scale
    pr = (red - y) / _R_FROM_CR
    pb = (blue - y) / _B_FROM_CB
    luma = np.clip(np.rint(y / _Y_SCALE + 16.0), 0.0, 255.0).astype(np.uint8)
    cb = _downsample_420(pb / _C_SCALE + 128.0)
    cr = _downsample_420(pr / _C_SCALE + 128.0)
    if luma.shape != (frames, height, width):
        raise ValueError(f"luma shape {luma.shape} != {(frames, height, width)}")
    return luma, cb, cr


def _downsample_420(plane: npt.NDArray[np.float64]) -> npt.NDArray[np.uint8]:
    frames, height, width = plane.shape
    blocks = plane.reshape(frames, height // 2, 2, width // 2, 2)
    return np.clip(np.rint(blocks.mean(axis=(2, 4))), 0.0, 255.0).astype(np.uint8)
