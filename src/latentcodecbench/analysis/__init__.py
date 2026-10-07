"""Rate-distortion curves, BD-rate, and the codec-leak study.

Phase 1 stores one results table and draws the classic-codec curves.
Phase 2 puts quantized VAE latents on the same axes. Phase 3 measures
how far a re-encoded source moves the latent, and whether a linear probe
can read the source QP back out.
"""
