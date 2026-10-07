# LatentCodecBench

[![CI](https://github.com/Shuvam1024/LatentCodecBench/actions/workflows/ci.yml/badge.svg)](https://github.com/Shuvam1024/LatentCodecBench/actions/workflows/ci.yml)

LatentCodecBench measures the variational autoencoders inside open video generators as if they were codecs. It compares Wan2.1, Wan2.2, LTX-Video, and HunyuanVideo with x264, x265, and SVT-AV1 on rate-distortion, temporal stability, and throughput; checks whether compression artifacts in the source video show up in the latent; and tests whether a tiny decoder-side post-filter can buy back quality under a fixed compute budget.

## Research questions

1. How do these video VAEs compare with x264, x265, and SVT-AV1 on rate-distortion, temporal stability, and throughput?
2. Do classic-codec artifacts in the source video leak into the VAE latent?
3. Can a tiny decoder-side post-filter recover quality at a fixed compute budget?

## Status

| Phase | Scope | Status |
| --- | --- | --- |
| 0. Foundation | Package, CI, docs | In review |
| 1. Data and classic codecs | Clips, metrics, x264 / x265 / SVT-AV1 | Not started |
| 2. Video VAEs | Wan, LTX-Video, HunyuanVideo as codecs | Not started |
| 3. Codec leak | Source-codec artifacts in the latent | Not started |
| 4. Post-filter | Tiny decoder-side network, fixed compute | Not started |
| 5. Write-up | Report, figures, public release | Not started |

The full step list is in [docs/PLAN.md](docs/PLAN.md). Design notes are in [docs/decisions/](docs/decisions/). Short concept notes are in [docs/learn/](docs/learn/).

## Repository layout

CPU code, tests, and codec baselines live in the package. GPU runs (VAE inference, post-filter training) live in notebooks and run on Colab.

```text
src/latentcodecbench/    importable package
  data/                  clip manifests and preprocessing (phase 1)
  metrics/               PSNR, SSIM, LPIPS, VMAF, temporal metrics (phase 1)
  codecs/                x264, x265, SVT-AV1 (phase 1)
  vaes/                  Wan, LTX-Video, HunyuanVideo wrappers (phase 2)
  analysis/              rate-distortion, BD-rate, codec-leak probes (phases 1–3)
  postfilter/            decoder-side network (phase 4)
  testing.py             synthetic clips for tests
  cli.py                 `lcb` entry point
tests/                   pytest
notebooks/               Colab drivers for GPU work
docs/                    plan, devlog, ADRs, concept notes
```

`/data/`, `/results/`, and `/checkpoints/` are gitignored.

## Quickstart

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone https://github.com/Shuvam1024/LatentCodecBench.git
cd LatentCodecBench
uv sync
uv run lcb --version
uv run pytest
```

`uv sync` installs the CPU environment from `uv.lock`: NumPy, Ruff, mypy, and pytest. PyTorch and Diffusers are optional, for later phases:

```bash
uv sync --extra gpu   # PyTorch
uv sync --extra vae   # PyTorch, Diffusers, and the libraries that load the VAEs
```

ffmpeg is required once codec baselines exist. CI installs it. Locally, install it from your system package manager.

## License

MIT. Copyright (c) 2026 Shuvam Mukherjee. See [LICENSE](LICENSE).
