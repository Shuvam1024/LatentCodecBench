"""Rate of a video elementary stream.

Bits per pixel divides video-packet bits by the number of displayed
samples. Container overhead is left out on purpose: at a very low rate
the mp4 wrapper would otherwise dominate the number. Kilobits per second
uses the source frame rate, ``frames * fps_den / fps_num``, rather than
whatever timescale the muxer wrote.
"""

from __future__ import annotations


def bits_per_pixel(bits: int, width: int, height: int, frames: int) -> float:
    """Video bits divided by width x height x frames."""
    samples = width * height * frames
    if samples < 1:
        raise ValueError("bpp needs a positive frame area")
    if bits < 0:
        raise ValueError("bits must be non-negative")
    return bits / samples


def kilobits_per_second(bits: int, frames: int, fps_num: int, fps_den: int) -> float:
    """Average kilobits per second at the source frame rate."""
    if frames < 1 or fps_num < 1 or fps_den < 1:
        raise ValueError("kbps needs frames and a positive frame rate")
    duration = frames * fps_den / fps_num
    return (bits / duration) / 1000.0
