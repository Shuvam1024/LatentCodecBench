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

## 2026-10-07 — Phase 1.2, distortion metrics

A decoded clip now has a number next to it, and each number is checked against a case that can be computed by hand or by another library.

- PSNR pools one MSE over the whole clip, then takes the log. Averaging per-frame PSNR would hide a bad frame. Luma and RGB are both reported. RGB uses the BT.709 limited-range matrix, because the canonical y4m is studio-range YUV. A 2×2 plane with one sample off by 1 is the unit test: MSE 0.25, PSNR about 54.15 dB.
- SSIM is Wang 2004 on luma: 11×11 Gaussian, sigma 1.5, the paper's C1 and C2. Flat fields have a closed form. A half-replaced frame is locked to the value scikit-image returns with the same window. scikit-image is not a dependency.
- VMAF is ffmpeg libvmaf, distorted first, one thread, pooled mean. Identical textured clips score above 99. They do not score a bitwise 100: the model uses integer features. Ubuntu's ffmpeg has no libvmaf, so CI switched to a pinned static build. ADR-0003.
- LPIPS is optional. Missing torch or lpips raises `MetricUnavailable`, and the score helper records `None`. CI does not install torch.
- Temporal flicker is the mean absolute disagreement of frame-to-frame deltas. A constant brightness shift scores 0, because the deltas did not change. Optical-flow warping error would also catch motion that points the wrong way. That needs a flow model, so it is future work rather than a dependency of this baseline. ADR-0004.

## 2026-10-07 — Phase 1.3, codec baselines

x264, x265, and SVT-AV1 can now be run as a function instead of a shell one-liner, with the knobs written down.

- Six CRF points each. x264 and x265 use 18, 22, 26, 30, 34, 38. SVT-AV1 uses 28, 35, 42, 49, 56, 63. The low end of x264 used to be 43, and SVT-AV1 used to stop at 60. Those two curves barely shared a quality range, so the ladders now end where the measured ranges meet. The numbers are still not a shared scale. ADR-0005.
- Presets: x264 `medium`, x265 `medium`, SVT-AV1 preset 6. On `crowd_run` at one thread those were 0.7 s, 1.5 s, and 2.0 s. Preset 8 was faster and was not chosen. Tests use `ultrafast` so CI stays on a 64×64 clip.
- One thread, and no `-tune psnr`. This ffmpeg's libsvtav1 defaults to a PSNR tune, so the wrapper sets `tune=0` (visual quality) explicitly.
- Rate is the sum of video packet sizes. Bits per pixel divides by width × height × frames, so a 60 fps clip and a 24 fps clip share an x-axis. Decode time is a one-thread decode to null, not the time to write a reconstructed y4m.
