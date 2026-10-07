# ADR-0004: How a decoded clip is scored

- Status: Accepted
- Date: 2026-10-07

## Context

Phase 1 has to put x264, x265, and SVT-AV1 on one rate-distortion plot, and later phases have to drop VAE reconstructions on the same axes. The metrics have to be defined once, with a unit test that does not need a video, so a refactor cannot quietly change the number.

The candidates are the usual set: PSNR, SSIM, VMAF, LPIPS, plus something temporal. Optical-flow warping error is the temporal metric a codec paper often wants, and it pulls in a flow network.

## Decision

`score_y4m` reports six fields.

- **PSNR-Y and PSNR-RGB.** `10 log10(255^2 / MSE)`. One MSE over every sample in the clip, then one logarithm. Per-frame PSNR averaged afterwards hides a single bad frame. RGB is BT.709, limited range (luma 16..235, chroma centered at 128), with chroma repeated onto its 2×2 block. The canonical y4m is studio-range 4:2:0, so a full-range matrix would paint black as dark gray.
- **SSIM**, on luma only. Wang, Bovik, Sheikh, and Simoncelli, 2004: 11×11 Gaussian, sigma 1.5, `k1 = 0.01`, `k2 = 0.03`, `L = 255`. The window is normalized and the variances are the population form (no `N-1` correction). A clip is the mean of per-frame scores. Chroma SSIM is not reported; VMAF and PSNR-RGB cover appearance beyond luma structure.
- **VMAF**, ffmpeg `libvmaf`, distorted input first, `n_threads=1`, the pooled mean. Identical clips are asserted to score above 99 on textured input. The model does not return a bitwise 100.
- **Temporal flicker.** Mean absolute difference between the reference frame-delta and the distorted frame-delta, on luma. Identical clips score 0. A constant offset on every frame scores 0, because the deltas were preserved.
- **LPIPS** (Zhang et al., 2018, AlexNet) when `torch` and `lpips` import. Otherwise the field is `None`. It is not in the default environment and not in CI.

Optical-flow warping error is future work. It would measure whether distorted motion matches reference motion, which flicker cannot do: a pan in the wrong direction can have the same delta energy. A flow model is a heavy dependency for a CPU baseline, and the delta already catches flicker, freezes, and temporal pumping. The learn note says this in plain language.

## Consequences

Rate-distortion curves in the next step can plot PSNR-Y and VMAF against bits per pixel without redefining the y-axis. A missing torch install does not fail a codec sweep.

PSNR-RGB depends on the color matrix. A later change to full range would move every RGB number and should be a new ADR, not a silent edit.

Flicker will not rank a codec that preserves energy but points motion the wrong way. That ranking waits on the warping metric.

## Alternatives

**Mean of per-frame PSNR.** Common in logs, and it compresses outliers. One bad frame should move the clip score. Pooling MSE does that.

**MS-SSIM, or SSIM on RGB.** Closer to perception in some still-image papers, and a second implementation to lock down before any codec has been run. Luma SSIM plus VMAF is the pair codec comparisons already know how to read.

**Require LPIPS.** It needs PyTorch, which ADR-0001 kept out of the default install so CI and the codec sweep stay small.

**Ship a flow network now.** Honest temporal geometry, and a download plus a GPU-shaped dependency in the CPU phase. The gap is documented so it is a planned metric rather than a forgotten one.
