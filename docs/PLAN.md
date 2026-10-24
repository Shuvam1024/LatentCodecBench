# LatentCodecBench: Project Plan

**Question:** Open video VAEs (the "codec" inside every latent video generator) vs. classic codecs (x264, x265, SVT-AV1): how do they compare on rate-distortion, temporal stability, and throughput? Do classic-codec artifacts in training video leak into VAE latents? Whether a lightweight decoder-side post-filter can recover quality is an open direction; the design is not decided.

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

Wrappers: `AutoencoderKLWan` for Wan2.1 and for the Wan2.2 TI2V-5B VAE, `AutoencoderKLLTXVideo`, `AutoencoderKLHunyuanVideo`. The canonical 832×480×33 clips already meet each model's frame rule (`4k+1` or `8k+1`) and spatial rule (divisible by 8, 16, and 32), so this phase does not resize them. Steps 2.1 and 2.2 run on CPU, with a one-clip GPU smoke if a free accelerator is offered. A paid Colab plan starts at 2.3. The fallback when a 16 GB accelerator runs out of memory is an L4-class 24 GB GPU.

- 2.1 Wrapper and CPU shape tests. One encode/decode interface reports compression factors, latent shape, and whether the length is legal. CI stays on CPU, uses tiny random-init configs, and keeps the hub offline. ADR-0007 pins Diffusers 0.41.0.
  Acceptance: on Python 3.11 and 3.12, a 832×480×33 tensor yields 16×9×60×104 (Wan2.1 and HunyuanVideo), 48×9×30×52 (Wan2.2 TI2V-5B), and 128×5×15×26 (LTX-Video). The job fails if a checkpoint is fetched. `docs/learn/04-latent-shape.md` explains the causal frame rule.

- 2.2 Pins, licenses, and a VAE-only load. `from_pretrained(..., subfolder="vae")` on `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` (Apache-2.0; the same VAE Wan2.2-T2V-A14B ships), `Wan-AI/Wan2.2-TI2V-5B-Diffusers` (Apache-2.0, the 4×16×16 VAE), `Lightricks/LTX-Video` (license tag `other`; the pinned `vae/` file is the 0.9 weight under RAIL-M, and the LTXV Open Weights License 0.X covers 0.9.6 and later), and `hunyuanvideo-community/HunyuanVideo` (Tencent Hunyuan Community License on the `tencent/HunyuanVideo` weights; that agreement says it does not apply in the EU, the UK, or South Korea). ADR-0008 records repo id, revision, license, and dtype. `notebooks/vae_smoke.ipynb` encodes and decodes one 832×480×33 clip per VAE and records peak VRAM, time, PSNR, and whether tiling was on.
  Acceptance: offline random-init tests still pass; the smoke log shows a reconstruction from the VAE subfolder with the transformer unloaded.

- 2.3 Reconstruction sweep. This is the step that needs Colab Pro when the free accelerator is unavailable or a forward exceeds 16 GB. `notebooks/vae_sweep.ipynb` runs all 34 clips through all four VAEs, reuses PSNR, SSIM, VMAF, and flicker, and records peak VRAM, encode/decode time, and per-chunk time where the VAE encodes in chunks. Tiling stays off unless a clip runs out of memory, and that flag is a column (ADR-0009). HunyuanVideo is the exception: the T4 smoke ran it out of memory until spatial tiling was enabled, so the sweep enables that tiling before the first forward, on an L4 and on a T4. The runtime is bfloat16 on an L4 and float16 on a T4. The notebook installs the package from the git SHA being measured and writes a CSV the classic reader can load, with VAE columns added in `docs/results/SCHEMA.md`. `scripts/merge-vae-recon.py` copies that CSV into `docs/results/` once it is uploaded. The PR that brings results back commits the CSV. Video, latents, and checkpoints stay out of git.
  Acceptance: 136 rows, one per clip and VAE, rate columns empty, `run_id` `{utc}-{git_sha12}-{config_sha12}`, torch and diffusers versions pinned in the run.

- 2.4 Latent bitrate. Posterior mean, then per-channel uniform quantization at bit depths 2, 4, 6, 8, 10, and 12. Apply the published per-channel mean and std, or `scaling_factor`, before the quantizer and invert it after. The first rate is the zstd size of the integer tensor, side information included in `bits`. A factorized Laplace or Gaussian prior comes second; its parameters are fixed before the test clip, or their size is counted (ADR-0010). If the low bit depths still sit above the classic ladder, a channel-count knob repeats the sweep on a prefix of channels, with the decoder filling the rest with zeros. `bpp` divides by `width * height * frames`. `docs/learn/05-latent-bitrate.md` and `docs/learn/06-a-prior-costs-bits.md`.
  Acceptance: each bit depth adds rows with `bits` and `bpp`; a unit test locks that denominator on a synthetic latent; the DEVLOG says these VAEs were trained to reconstruct, and that a cross-entropy without a bitstream is an entropy estimate.

- 2.5 RD curves, BD-rate, and throughput. Same overlap rule as ADR-0006, anchor libx264, plus the comparison against libsvtav1. The throughput table is encode seconds, decode seconds, and peak VRAM.
  Acceptance: `lcb plot rd` reads the VAE table, writes figures under `docs/figures/`, and the README reports mean BD-rate together with mean overlap.

## Phase 3: Codec-leak study
- 3.1 Degraded sources and the propagation metrics. Three CRF points from each Phase 1 ladder (x264, x265, SVT-AV1): latent distance, luma PSNR of the degraded reconstruction against the clean source and against the degraded source, and blocking/ringing retention. ADR-0012. `scripts/degrade-sources.py` writes the y4m files. `notebooks/artifact_propagation.ipynb` is the GPU driver. Measured rows land in step 3.2.
- 3.2 Encode through each VAE; latent distance to clean latents, artifact reconstruction. Done: `docs/results/artifact-propagation.csv` (1224 rows), `scripts/summarize-artifact-propagation.py`, results in `docs/learn/10-artifact-propagation.md`.
- 3.3 Linear probe of source CRF from latent statistics. The feature schema, the leave-clip-out fit, and the Colab driver are in. ADR-0013. `notebooks/latent_probe_features.ipynb` encodes each source and writes the feature CSV. `lcb probe fit` writes the table, the figure, and `docs/learn/14-linear-probe.md`. The feature CSV has not been measured, so the result stays in Future work.
- 3.4 Write-up: implications for training-data filtering. Done: `docs/learn/11-training-data-filter.md` reads the per-clip PSNR change against the clean reconstruction, with the latent shift beside it. `scripts/summarize-filtering.py` reprints that section, and a test checks the note against the CSVs.

## Phase 5: Write-up and release
- 5.1 README with results table, figures, one-command reproduction. Done: `uv run lcb reproduce` rebuilds the headline in the README and in `docs/learn/12-reproducing-the-tables.md` from the committed CSVs, checks the long tables, and redraws the figures into a temporary directory. It does not re-run a codec or a VAE.
- 5.2 `docs/REPORT.md` technical report with CIs and limitations. Done: the report's measured section is a 95% Student-t interval across clips, rebuilt from the committed CSVs. `docs/learn/13-confidence-intervals.md` says what that interval is claiming. `lcb reproduce` checks both.
- 5.3 Public release. Done: the repository is public, tagged v1.0.0, with a GitHub release. Pinning the repository on a GitHub profile is a manual account setting and is not part of this tag.

## Future work

- **3.3 Linear probe.** The fitting code and the encode-only Colab driver are in. The feature CSV has not been measured, so there is no result yet. The probe asks whether source CRF, within one codec, is linearly readable from the latent. That is a stronger claim than the latent having moved. The sweep needs a GPU.
- **Phase 4, decoder-side post-filter.** A lightweight filter after the decoder, to recover quality. The design is still to be decided. It has not been run. It would show whether a cheap correction after decoding can raise the reconstruction the VAE already measured.
