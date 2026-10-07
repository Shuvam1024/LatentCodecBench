# Development log

A short note for every pull request: what changed, and why it is shaped this way.

## 2026-10-07 — Phase 0, foundation

The repository is a skeleton. Later phases should land as small, tested changes on top of it.

- `src/latentcodecbench` is a src-layout package with one module per phase: `data`, `metrics`, `codecs`, `vaes`, `analysis`, and `postfilter`. They are stubs. The docstring on each one says what will live there, so the layout does not have to be recovered from the plan.
- `lcb --version` is the console script. It is the smallest entry point that proves packaging, install, and the CLI wiring all work.
- [uv](https://docs.astral.sh/uv/) owns the environment, and `uv.lock` is committed. The default install is NumPy plus the dev tools. PyTorch and Diffusers are the `gpu` and `vae` extras, so continuous integration does not download them.
- Ruff formats and lints, mypy type-checks, and pytest runs the suite. Pre-commit runs the same Ruff hooks on commit.
- GitHub Actions runs that suite on Ubuntu for Python 3.11 and 3.12, and installs ffmpeg from apt. Tests use a synthetic clip from `latentcodecbench.testing`, not a downloaded video.
- `notebooks/` is reserved for Colab GPU runs. Actions has no GPU, so pretending otherwise would make CI either skip the real work or time out downloading weights.
- Docs added with this change: the project plan, this log, an ADR template, ADR-0001 on the tooling, a concept-note index with a first note on latents, and `CONTRIBUTING.md`.

## 2026-10-07 — Phase 1.1, clip set

The benchmark needs one picture for every codec and every VAE. This change defines that picture and the list of sources.

- Canonical clip is 832×480, 33 frames, 8-bit 4:2:0 y4m. 33 frames is both `4k+1` and `8k+1`. 832 and 480 are divisible by 32. ADR-0002 records why, and why the set is 34 clips rather than 50.
- `manifests/clips.yaml` has the URL, license, native geometry, and SHA256. `lcb data fetch` deletes a download that does not match. For the large derf y4m files the digest is of a 33-frame byte range, not of the whole multi-gigabyte object. `DATA.md` says so per clip.
- Preprocess is lanczos, cover-then-center-crop, still in YUV. UVG arrives as a 7z; ffmpeg reads the member from a pipe and stops after 33 frames.
- `data/` stays gitignored. CI installs `p7zip-full` next to ffmpeg so the 7z path is tested on a tiny archive, not on UVG.
