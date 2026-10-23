# Reproducing the published numbers

The GPU sweeps are not a command on this machine. The rows they wrote are the CSVs in `docs/results/`. The command that belongs on a CPU checks that those rows still produce the pages.

```bash
uv run lcb reproduce
```

Run it from the repository root. It reads the classic baseline CSV, the VAE reconstruction CSV, the latent-bitrate CSV, and the artifact-propagation CSV. It rebuilds the headline below, the BD-rate table in note 09, the propagation table in note 10, the training-filter section in note 11, the measured section of `docs/REPORT.md`, and the reconstruction intervals in note 13. It fails if a published page no longer contains that text, or if `vae-rd-bdrate.csv` or `artifact-propagation-summary.csv` no longer matches its renderer. It redraws the figures into a temporary directory and checks that the PNGs are non-empty. It does not replace the committed PNGs. Matplotlib does not promise the same bytes on a second draw.

What the command cannot do is the measurement. `lcb data fetch` downloads the clips. `lcb baselines run` repeats the classic sweep, and it needs the pinned ffmpeg. The VAE rows come from the notebooks on a GPU. Re-running those is a different job from checking the pages.

## Why the lock is a contained string

The learn notes are prose around a generated table. The check is that the generated table is still in the note, not that the note is only the table. A sentence can explain a row. A hand-typed number in place of the row would fail the check.

## The headline

| Codec | Against | Metric | BD-rate | Clips in the mean | Mean overlap | Under 75% |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| x265 | x264 | PSNR-Y | -24.66% | 34 | 85.37% | 0 |
| SVT-AV1 | x264 | PSNR-Y | -41.99% | 34 | 80.98% | 7 |
| x265 | x264 | VMAF | -20.44% | 34 | 84.45% | 2 |
| SVT-AV1 | x264 | VMAF | -44.86% | 34 | 77.66% | 12 |
| wan2_1 | x264 | PSNR-Y | +824.24% | 33 | 32.27% | 33 |
| wan2_1 | x264 | VMAF | +643.08% | 33 | 61.88% | 23 |
| hunyuan | x264 | PSNR-Y | +725.09% | 32 | 35.30% | 32 |
| hunyuan | x264 | VMAF | +583.16% | 30 | 60.80% | 24 |
| ltx | x264 | PSNR-Y | +397.92% | 26 | 20.30% | 26 |
| ltx | x264 | VMAF | +205.95% | 25 | 41.95% | 25 |
| wan2_2 | x264 | PSNR-Y | +628.61% | 33 | 42.94% | 28 |
| wan2_2 | x264 | VMAF | +568.29% | 33 | 66.02% | 24 |

| VAE | Clips | Mean PSNR-Y | Mean VMAF |
| --- | ---: | ---: | ---: |
| wan2_1 | 34 | 34.57 dB | 86.40 |
| hunyuan | 34 | 35.11 dB | 87.63 |
| ltx | 34 | 32.30 dB | 74.56 |
| wan2_2 | 34 | 35.35 dB | 87.13 |

## Source

`src/latentcodecbench/analysis/published.py` is the check. Notes 09, 10, and 11 are the long tables. [ADR-0006](../decisions/0006-rd-curves-and-bd-rate.md) and [ADR-0011](../decisions/0011-vae-bd-rate.md) are the BD-rate rules.
