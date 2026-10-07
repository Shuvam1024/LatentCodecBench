"""Video VAE wrappers.

Phase 2 treats Wan2.1, Wan2.2, LTX-Video, and HunyuanVideo as codecs.
Each wrapper will expose encode, decode, the spatiotemporal compression
factor, and the legal frame count. Weights load through Diffusers.

Torch and Diffusers are optional extras (``gpu`` and ``vae``). Importing
this package does not import them, so a CPU checkout stays light.
"""
