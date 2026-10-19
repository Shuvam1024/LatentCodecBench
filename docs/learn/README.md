# Concept notes

Short notes for interview prep. Each one is about a page, in plain language, and cites the source it depends on.

| Note | Topic |
| --- | --- |
| [00 — Why VAEs are codecs](00-why-vaes-are-codecs.md) | Latents, the 4×8×8 / 4×16×16 / 8×32×32 trade-off, rate-distortion, and why a codec bitstream is a bad latent |
| [01 — What the metrics miss](01-what-the-metrics-miss.md) | PSNR, SSIM, VMAF, and LPIPS, and the kind of error each one cannot see |
| [02 — CRF, presets, and bits per pixel](02-crf-presets-and-bpp.md) | Why CRF is not a shared scale, what the preset is holding still, and a worked bpp example |
| [03 — RD curves and BD-rate](03-rd-curves-and-bd-rate.md) | Reading a curve, the -50% worked example, and why BD-PSNR is not zero on that example |
| [04 — What shape a latent is](04-latent-shape.md) | Channels, the causal `4k+1` / `8k+1` rule, and the 832×480×33 grid for each VAE |
| [07 — Pins, licenses, and loading only the VAE](07-pins-and-licenses.md) | A revision plus `subfolder="vae"`, the license on each weight file, and which file is large for a 16 GB card |

Notes 05 and 06 are the bitrate notes named in the plan. This one is 07 so those names stay put.
