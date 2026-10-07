"""Distortion and temporal metrics.

Phase 1 scores decoded frames with PSNR, SSIM, LPIPS, and VMAF (through
ffmpeg's libvmaf), plus temporal flicker and a warping error. Each metric
will have a unit test against a known value before it is used in a sweep.
"""
