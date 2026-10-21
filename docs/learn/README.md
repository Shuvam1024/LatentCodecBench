# Concept notes

Short notes for interview prep. Each one is about a page, in plain language, and cites the source it depends on.

| Note | Topic |
| --- | --- |
| [00 — Why VAEs are codecs](00-why-vaes-are-codecs.md) | Latents, the 4×8×8 / 4×16×16 / 8×32×32 trade-off, rate-distortion, and why a codec bitstream is a bad latent |
| [01 — What the metrics miss](01-what-the-metrics-miss.md) | PSNR, SSIM, VMAF, and LPIPS, and the kind of error each one cannot see |
| [02 — CRF, presets, and bits per pixel](02-crf-presets-and-bpp.md) | Why CRF is not a shared scale, what the preset is holding still, and a worked bpp example |
| [03 — RD curves and BD-rate](03-rd-curves-and-bd-rate.md) | Reading a curve, the -50% worked example, and why BD-PSNR is not zero on that example |
| [04 — What shape a latent is](04-latent-shape.md) | Channels, the causal `4k+1` / `8k+1` rule, and the 832×480×33 grid for each VAE |
| [05 — How a latent becomes a file](05-latent-bitrate.md) | Published normalization, uniform rungs, zstd, the pixel denominator, and the L4 bitrate table |
| [06 — A prior costs bits](06-a-prior-costs-bits.md) | A fixed unit normal, why its cross-entropy is an estimate, and the channel knob |
| [07 — Pins, licenses, and loading only the VAE](07-pins-and-licenses.md) | A revision plus `subfolder="vae"`, the license on each weight file, and which file is large for a 16 GB card |
| [08 — Reconstruction, before a bitrate](08-reconstruction.md) | Phase 1 metrics on the real clips, Hunyuan tiled from the start, and why the rate columns are empty |
| [09 — The VAEs on the same axes as x264](09-vae-rd-curves.md) | BD-rate against the classic ladder, the depth-2 cut, the plateau near 8 bits, and LTX's lower ceiling |
