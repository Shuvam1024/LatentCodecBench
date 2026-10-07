"""Classic codec baselines.

Phase 1 drives ffmpeg's x264, x265, and SVT-AV1 encoders across a CRF
sweep and records bits per pixel plus encode and decode time. The runners
expect ``ffmpeg`` on ``PATH``, which continuous integration installs.
"""
