# ADR-0001: Tooling and the CPU / GPU split

- Status: Accepted
- Date: 2026-10-07

## Context

LatentCodecBench is a research repository that has to stay reviewable as it grows. Two kinds of work share it. Classic codecs, metrics, and tests run on a CPU and need ffmpeg. Video VAE inference and post-filter training need a GPU, model weights, and more RAM than GitHub Actions provides. The tooling has to make the CPU path reproducible in CI without pulling PyTorch into every clone.

## Decision

- **uv** manages Python, the virtualenv, and the lockfile. `uv.lock` is committed. CI runs `uv sync --frozen`.
- **Ruff** is both the linter and the formatter. Pre-commit runs the same hooks locally.
- **mypy** in strict mode type-checks `src/` and `tests/`.
- **pytest** is the test runner. A helper in `latentcodecbench.testing` builds a tiny NumPy clip so tests do not need a video file.
- **src layout.** The importable package lives in `src/latentcodecbench`, and the `lcb` script is an entry point in `pyproject.toml`.
- **Optional extras.** `gpu` is PyTorch. `vae` is PyTorch plus Diffusers and the libraries it needs to load Wan, LTX-Video, and HunyuanVideo. The default environment does not install either extra.
- **CPU CI, GPU notebooks.** GitHub Actions installs ffmpeg and runs Ruff, mypy, and pytest on Python 3.11 and 3.12. Weight-bearing runs go in `notebooks/` on Colab. Notebooks call the library; they are not the only copy of the logic.

## Consequences

CI stays small and can fail on a lint, a type, or a regression without a GPU. A fresh clone reproduces the CPU environment from the lockfile.

The cost is that the VAE import path is untested until Phase 2 adds CPU tests against tiny random-init configs. A notebook can drift from the library if logic is pasted into it instead of imported. The contributor guide says not to do that.

`uv.lock` records the optional extras as well as the default set, so a later `uv sync --extra vae --frozen` resolves the same versions CI would. The lockfile is larger because of that.

## Alternatives

**pip-tools or Poetry.** Both can lock a project. uv is one binary that also installs Python itself, which is what the CI job needs in order to test 3.11 and 3.12 without a separate setup action. The lockfile format is the one the workflow calls.

**Black, isort, and Flake8.** Three tools and three version pins for a job Ruff does in one process. The pre-commit config would be longer and easier to let drift from CI.

**A flat layout** (`latentcodecbench/` at the repository root). Tests can then import the checkout instead of the installed package, which hides a broken `pyproject.toml` until someone installs it. The src layout makes `uv run pytest` import the installed copy.

**PyTorch in the default dependencies.** Honest, in the sense that the project will need it, and painful for every other step: CI minutes, laptop disk, and the codec baselines, which never import it. Extras keep that cost opt-in.

**Running VAEs in GitHub Actions on a GPU runner.** Those runners are paid, slow to start, and still a bad place to download tens of gigabytes of weights on every pull request. Colab is where that work already has to happen for training. CI's job is the CPU contract.
