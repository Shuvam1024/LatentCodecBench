# LatentCodecBench: Project Plan

**Question:** Open video VAEs (the "codec" inside every latent video generator) vs. classic codecs (x264, x265, SVT-AV1): how do they compare on rate-distortion, temporal stability, and throughput? Do classic-codec artifacts in training video leak into VAE latents? Can a tiny decoder-side post-filter recover quality at a fixed compute budget?

**Compute:** CPU (cloud dev VM, GitHub Actions) for code, tests, and codec baselines. Colab Pro GPU (L4/A100) for VAE runs and training, through notebooks in `notebooks/`.

**Practices:** one small PR per step, conventional commits, CI must pass, tests for every module, `docs/DEVLOG.md` entry per PR, `docs/decisions/` ADRs for design choices, `docs/learn/` short concept notes for interview prep.

## Phase 0: Foundation
- 0.1 Repo skeleton: `src/latentcodecbench/` package, `pyproject.toml` (uv), ruff, mypy, pytest, pre-commit, MIT license, `.gitignore`.
- 0.2 CI: GitHub Actions running lint + type check + tests on CPU with tiny synthetic clips; ffmpeg installed in CI.
- 0.3 Docs: README (pitch, status, layout), this plan, DEVLOG, ADR template + ADR-0001 (tooling), CONTRIBUTING.
- 0.4 Tracking: one GitHub issue per phase with step checklists, labels per phase.

## Phase 1: Data and classic codec baselines (CPU)
- 1.1 Clip set: YAML manifest of ~50 clips (UVG, Xiph derf 1080p, a few UGC), download script with SHA256 verification, fixed preprocessing (crop/resize, frame count compatible with VAEs, e.g. 4k+1 frames).
- 1.2 Metrics: PSNR, SSIM, LPIPS, VMAF (ffmpeg libvmaf), temporal flicker / warping error; unit tests against known values.
- 1.3 Codec baselines: ffmpeg x264 / x265 / SVT-AV1 CRF sweeps; bits per pixel, encode/decode time.
- 1.4 Results pipeline: config files, run IDs, a single results table (parquet/CSV) schema, deterministic seeds.
- 1.5 First figure: rate-distortion curves for classic codecs + BD-rate utility (tested).

## Phase 2: Video VAEs as codecs (Colab GPU)
- 2.1 VAE wrapper interface (encode/decode, compression factors, frame constraints) for Wan2.1, Wan2.2, LTX-Video, HunyuanVideo via Diffusers; CPU tests with tiny random-init configs.
- 2.2 Reconstruction benchmark: quality metrics, VRAM, encode/decode fps, chunk latency (Colab notebook).
- 2.3 Giving latents a bitrate: per-channel N-bit quantization + entropy coding (zstd upper bound, then a factorized/Laplace model).
- 2.4 R-D curves and BD-rate vs. SVT-AV1; throughput table.

## Phase 3: Codec-leak study
- 3.1 Re-encode sources at QP 20 to 50 (x264, SVT-AV1).
- 3.2 Encode through each VAE; latent distance to clean latents, artifact reconstruction.
- 3.3 Linear probe predicting QP from latents, cross-validated with confidence intervals.
- 3.4 Write-up: implications for training-data filtering.

## Phase 4: Decoder-side post-filter
- 4.1 Paired dataset (VAE reconstruction, original).
- 4.2 compsr-style tiny CNN with a MAC counter test enforcing <= 5k MACs/pixel.
- 4.3 Train on Colab; report dB gain, BD-rate, ms/frame.
- 4.4 ONNX export + numerical parity test.
- 4.5 Optional: Wan2.1-1.3B generations with/without the filter, VBench quality subset.

## Phase 5: Write-up and release
- 5.1 README with results table, figures, one-command reproduction.
- 5.2 `docs/REPORT.md` technical report with CIs and limitations.
- 5.3 Make repo public, tag v1.0, pin on profile, short post draft.
