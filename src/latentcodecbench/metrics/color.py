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
