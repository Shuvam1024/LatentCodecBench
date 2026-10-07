"""Decoder-side post-filter.

Phase 4 trains a small convolutional network, in the style of a learned
codec post-filter, on pairs of VAE reconstructions and original frames.
A multiply-accumulate counter will reject any model over 5k MACs per
pixel. Training runs in a Colab notebook. Export to ONNX, and a numerical
parity check against the PyTorch weights, stay in this package.
"""
