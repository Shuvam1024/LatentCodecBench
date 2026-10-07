"""Round-trip tests for the y4m reader."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from latentcodecbench.data.y4m import parse_header, read_y4m, write_y4m


def test_round_trip(tmp_path: Path) -> None:
    luma = np.arange(2 * 4 * 6, dtype=np.uint8).reshape(2, 4, 6)
    u_plane = np.arange(2 * 2 * 3, dtype=np.uint8).reshape(2, 2, 3)
    v_plane = np.full((2, 2, 3), 128, dtype=np.uint8)
    path = tmp_path / "tiny.y4m"
    write_y4m(path, luma, u_plane, v_plane, fps_num=50, fps_den=1)
    info, got_luma, got_u, got_v = read_y4m(path)
    assert info.width == 6
    assert info.height == 4
    assert info.fps_num == 50
    assert info.colorspace == "420mpeg2"
    assert np.array_equal(got_luma, luma)
    assert np.array_equal(got_u, u_plane)
    assert np.array_equal(got_v, v_plane)


def test_missing_colorspace_tag_means_420jpeg() -> None:
    blob = b"YUV4MPEG2 W1920 H1080 F50:1 Ip A1:1\nFRAME\n"
    info = parse_header(blob)
    assert info.colorspace == "420jpeg"
    assert info.width == 1920
    assert info.height == 1080
    assert info.fps_num == 50
    assert info.fps_den == 1
    assert info.payload_bytes == 1920 * 1080 * 3 // 2
