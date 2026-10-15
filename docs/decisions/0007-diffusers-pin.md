# ADR-0007: Pin Diffusers 0.41.0 for the VAE wrappers

- Status: Accepted
- Date: 2026-10-15

## Context

The VAE wrapper has to call a real Diffusers class, because a hand-rolled encoder would not be the model the benchmark claims to measure. Diffusers moves constructor arguments between releases. Wan2.2's VAE config carries `patch_size`, `scale_factor_spatial`, and `scale_factor_temporal`, which the class gained before 0.35. The lockfile already resolved `diffusers>=0.32` to 0.41.0, released 2026-10-06.

CI has to construct those classes on CPU with random weights. The default PyPI torch wheel is a CUDA build, which is the wrong artifact for a runner with no GPU.

## Decision

The `vae` extra pins `diffusers==0.41.0`. Torch in that extra is the CPU wheel from `https://download.pytorch.org/whl/cpu`, recorded in `uv.lock`.

The tag checked for this pin is [v0.41.0](https://github.com/huggingface/diffusers/releases/tag/v0.41.0). On that tag:

- `AutoencoderKLWan` takes `z_dim`, `scale_factor_temporal` (default 4), `scale_factor_spatial` (default 8), and `patch_size`. Encode walks the clip as one frame and then groups of 4 (`1 + (num_frame - 1) // 4` in `autoencoder_kl_wan.py`).
- `AutoencoderKLLTXVideo` sets spatial compression to `patch_size * 2 ** sum(spatio_temporal_scaling)` and temporal compression to `patch_size_t * 2 ** sum(spatio_temporal_scaling)`. The published defaults are patch 4, temporal patch 1, and three spatiotemporal scales, which is 32 and 8. `encoder_causal` defaults to true and `decoder_causal` to false.
- `AutoencoderKLHunyuanVideo` takes `latent_channels=16`, `spatial_compression_ratio=8`, and `temporal_compression_ratio=4`.

Tiny configs keep those factors and channel counts and shrink the block width. `from_pretrained` is not on this path. CI sets `HF_HUB_OFFLINE=1`.

## Consequences

A later Diffusers release is a deliberate bump: re-read the three class files, confirm the factors, and update this ADR. The CPU wheel is what CI imports. A CUDA machine that wants its own torch installs that outside this lock; the wrapper reads whatever tensor it is given.

Checkpoint choice, license, and dtype stay with the following ADR, when a weight file is actually loaded.

## Alternatives

**Leave the extra at `diffusers>=0.32`.** The solver already picked 0.41.0, and an unpinned floor would let a later release change the constructor under a frozen-looking extra until someone re-locks. The pin makes the class file a known text.

**The CUDA wheel in CI.** It imports on CPU and downloads several gigabytes of libraries the shape test never calls. The CPU index is the wheel the test runs.
