"""Clip manifests, verified downloads, and preprocessing.

Phase 1 turns a list of source videos into one fixed clip set the rest of
the benchmark shares. This package will own:

- the YAML manifest (UVG, Xiph derf 1080p, and a few user-generated clips)
- downloads checked against SHA256
- crop, resize, and frame-count rules that every VAE under test accepts

Nothing here reads a real video yet. Tests that need pixels use
:mod:`latentcodecbench.testing`.
"""
