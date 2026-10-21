# ADR-0010: Latent bitrate is a zstd file, and the prior is an estimate

- Status: Accepted
- Date: 2026-10-20

## Context

Step 2.3 scored the four VAEs with empty rate columns. A later BD-rate needs `bits`, `bpp`, and `kbps` on the same pixel denominator as x264. The latent those models hand back is a posterior mean, a grid of floats, not a bitstream.

The plan asks for per-channel uniform quantization at bit depths 2, 4, 6, 8, 10, and 12, after the published normalization, with the zstd size of the integer tensor as the first rate and the header counted in `bits`. A factorized Gaussian comes second. Its parameters are fixed before the test clip, or their size is counted. `bpp` divides by `width * height * frames`.

These VAEs were trained to reconstruct pixels. They were not trained to survive this quantizer.

## Decision

`normalize` applies the map the Diffusers 0.41.0 pipeline uses before the diffusion model, and `denormalize` is its inverse, applied before `decode`.

- Wan2.1 and Wan2.2: `(latent - latents_mean) / latents_std`, per channel, from the pinned `vae/config.json`.
- HunyuanVideo: `latent * scaling_factor`, with `scaling_factor` 0.476986.
- LTX-Video: `(latent - latents_mean) * scaling_factor / latents_std`. The config's `scaling_factor` is 1. The mean and standard deviation are the buffers in the pinned VAE weights.

The quantizer then clips that normalized grid to `[-4, 4]` and picks the nearest of `2^depth` reconstruction points, including both endpoints. Depths are 2, 4, 6, 8, 10, and 12. Codes are stored little-endian in `uint8` up to 8 bits and `uint16` after that, in C order, and compressed with zstd level 3. A 24-byte header (magic, depth, full channel count, channels kept, latent shape, and the two range endpoints) is prepended and counted. `bits` is eight times the length of header plus compressed payload. `bpp` and `kbps` use `latentcodecbench.codecs.rate`, so the denominator is the canonical pixel count, not the latent grid.

`crf` in the results table is this bit depth. The column is the operating point the classic reader already groups on. `preset` is `zstd`.

`entropy_bits` is the cross-entropy of those codes under a factorized standard normal in the normalized space. Mean 0 and variance 1 are fixed before the clip, so the prior's parameters add no bits. Outer bins include the tails. That number is an estimate of a codebook. It is not a file, and it is not copied into `bits`.

`channels` in `configs/vae-bitrate.yaml` is null, which keeps every channel. Setting it to a positive integer codes a prefix and writes 0 into the remaining channels of the tensor `decode` receives. That knob is for a later repeat if the low depths still sit above the classic ladder. It is not the first sweep.

`lpips` stays empty. The metrics list is PSNR-Y, PSNR-RGB, SSIM, VMAF, and flicker.

The 816 rows (34 clips × 4 VAEs × 6 depths) are produced by `notebooks/vae_bitrate.ipynb` on a GPU. `scripts/merge-vae-bitrate.py` copies an uploaded CSV into `docs/results/vae-bitrate.csv`. That file is the L4 sweep: 816 rows, run id `20261020T234203Z-8cb0b7cd9bc4-17ec7df8f1e6`. The per-depth means are in `docs/learn/05-latent-bitrate.md`.

## Consequences

`bits` is a real file size. zstd on a padded integer tensor is a compressor, not the arithmetic coder a Gaussian prior would justify, so the file can be larger than `entropy_bits`. Comparing the two says how much is left on the table. It does not make the estimate a bitstream.

`[-4, 4]` covers a unit normal out past four standard deviations and clips anything beyond that. A range fit to one clip would move with the clip, and the min and max would have to be counted as side information.

One encode is shared by the six depths of a clip. `encode_seconds` on each row is that shared forward, repeated, plus the depth has its own `decode_seconds`. Summing the six encode times counts the encode six times.

Hunyuan spatial tiling stays on from the first forward, as in ADR-0009.

## Alternatives

**Bit-pack the codes, then compress.** A 2-bit code does not need a byte. zstd is then asked to remove the padding. The plan asks for the size of the integer tensor, so the dtype is the smallest unsigned integer that holds the code, and the padding is part of what zstd sees.

**Per-clip min and max.** The step size would track the latent. The range is side information, and two clips of the same VAE would no longer share a quantizer. The fixed range is a property of the config, hashed with it.

**Fit a Laplace on the test clip and report the likelihood.** That prior saw the clip it is scoring. ADR-0006's spirit is that the rate is something you could have sent. A prior fit on the test clip is not that, unless the parameter bytes are added. The standard normal is fixed first, so there is nothing to add. A cross-entropy without a payload is still an estimate: nobody wrote the bits.
