"""The one clip shape every later stage is allowed to assume.

Wan-style video VAEs want a frame count of ``4k+1``. LTX-Video wants
``8k+1``. 33 frames satisfies both (``k = 8`` and ``k = 4``). 832x480 is
divisible by 32, which covers the spatial factors 8, 16, and 32 used by
Wan2.1, Wan2.2, and LTX-Video. See ADR-0002.
"""

from __future__ import annotations

CANONICAL_WIDTH = 832
CANONICAL_HEIGHT = 480
CANONICAL_FRAMES = 33
CANONICAL_PIX_FMT = "yuv420p"
