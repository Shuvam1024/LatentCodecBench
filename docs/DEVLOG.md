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

## 2026-10-07 — Phase 1.4 and 1.5, results and the first RD curves

The sweep ran, on all 34 clips, and the table is small enough to commit.

- `lcb baselines run` writes one CSV. The run id is the UTC time, the git SHA of the code that ran, and the SHA256 of the config file. A second invocation skips operating points already in the file. ADR-0006. The schema is `docs/results/SCHEMA.md`.
- BD-rate is a PCHIP integral. A half-rate curve is exactly -50%. The published four-point vector from the `bjontegaard` package (`pchip`) is locked at -4.417% BD-rate. The number reported for the set is the mean of the per-clip BD-rates, anchor x264.
- `lcb plot rd` draws mean PSNR-Y and mean VMAF against bits per pixel, log x-axis.
- The sweep is 612 encodes (34 × 3 × 6). On this machine that finished in one sitting: the presets timed earlier were about 1–2 seconds per clip, and one VMAF was about 0.3 seconds. No subset.
- BD-rate of x265 against x264: -27.45% on PSNR-Y, -26.20% on VMAF. SVT-AV1 against x264: -39.20% on PSNR-Y, -42.63% on VMAF. All 34 clips had some shared range. The overlap fraction on that first ladder was under 75% for SVT-AV1 on 32 of 34 clips, for both metrics. LPIPS is empty because PyTorch was not installed. The 2026-10-08 entry replaces this table.

## 2026-10-08 — CRF ladders retuned, overlap recorded

The first SVT-AV1 ladder stopped near 33 dB PSNR-Y and VMAF 71. x264 CRF 43 reached about 28 dB and VMAF 34. Calling that "every clip overlapped" hid a narrow intersection. ADR-0005.

- x264 and x265 are now 18, 22, 26, 30, 34, 38. SVT-AV1 is 28, 35, 42, 49, 56, 63. CRF 63 is the end of the SVT-AV1 dial, and on this set it averages about 30.6 dB and VMAF 56, next to x264 CRF 38. CRF 36 and CRF 40 were encoded on all 34 clips and covered less of the same range.
- `docs/results/overlap.csv` stores the fraction for each clip, test codec, and metric. `below_0_75` is 1 when the fraction is under 75%. The mean of the fractions is reported with the BD-rate. A narrow pair stays in the mean.
- The sweep ran again: 612 encodes, run id `20261008T010001Z-1a9353d785d3-baa565d2b8a8`, on the pinned September ffmpeg. Points present in both tables (x264 and x265 at CRF 18 and 38, SVT-AV1 at CRF 28) match the previous bits, PSNR, and VMAF, so the pin did not move those rows. The BD-rates moved because the ladders did.
- x265 against x264: -24.66% on PSNR-Y, mean overlap 85.37%, none under 75%. -20.44% on VMAF, mean overlap 84.45%, 2 under 75% (`controlled_burn`, `rust_laser`). SVT-AV1 against x264: -41.99% on PSNR-Y, mean overlap 80.98%, 7 under 75%. -44.86% on VMAF, mean overlap 77.66%, 12 under 75%. No clip was skipped. The short pairs are mostly easy clips, where CRF 63 has not reached x264 CRF 38. LPIPS is still empty.
- The four camera clips are re-encodes of Theora or VP9. That is in `DATA.md`, and the mean includes them.

## 2026-10-15 — Phase 2.1, VAE wrapper and latent shape

The four VAEs now share one encode/decode object. It does not load weights.

- `VideoVae` takes pixels as `(batch, 3, frames, height, width)` in `[-1, 1]` and returns the posterior mean. Decode returns pixels. The legal length is `temporal_factor * k + 1`, checked before the forward.
- The geometry is the published one: Wan2.1 and HunyuanVideo are 16 channels, 4×8×8; Wan2.2 TI2V-5B is 48 channels, 4×16×16; LTX-Video is 128 channels, 8×32×32. An 832×480×33 clip is `16×9×60×104`, `48×9×30×52`, and `128×5×15×26`. `docs/learn/04-latent-shape.md`.
- CI builds those classes with random weights and narrow blocks. The block width is what got shrunk. The channel count and the downsample schedule are the real ones, because those are what set the grid. `from_pretrained` is patched to raise, and the job sets `HF_HUB_OFFLINE=1`.
- Diffusers is pinned to 0.41.0. The three class files on tag `v0.41.0` were read before the pin: Wan's scale factors and the group-of-4 encode loop, LTX's `patch * 2^3` formula, Hunyuan's 16 / 8 / 4. ADR-0007. Torch for the extra is the CPU wheel, so the runner is not asked for a CUDA build.
- The LTX encoder is causal and its decoder is not. The frame rule still follows the encoder. A 32-frame clip is rejected for every model in the set.

## 2026-10-19 — Phase 2.2, checkpoint pins and a VAE-only load

The wrapper can now load the published VAE file, and only that file.

- Each pin is a repo, a revision, and `subfolder="vae"`. The sha256 of `vae/diffusion_pytorch_model.safetensors` is stored next to the revision. All four headers are float32. ADR-0008. `load_vae` calls `from_pretrained` on the VAE class. The transformer is not requested. CI still patches that call so a download fails the job; the load test checks the arguments and does not fetch.
- Wan2.1 and Wan2.2 TI2V-5B are Apache-2.0. The A14B repo's VAE file has the same sha256 as Wan2.1, so the benchmark loads Wan2.1 for that geometry. LTX `vae/` was last changed on 2024-12-11. The model card maps 2B version 0.9 to the RAIL-M file (academic or research use). The Open Weights License 0.X in that repo covers 0.9.6 and later. Hunyuan's community repo has no license tag; the parent `tencent/HunyuanVideo` license is the Tencent Hunyuan Community License Agreement, which excludes the EU, the UK, and South Korea.
- `notebooks/vae_smoke.ipynb` installs the repo and Diffusers 0.41.0 on top of the torch already in the runtime, then encodes and decodes one seeded 832×480×33 clip. It writes `results/vae-smoke.json` and a CSV with peak VRAM, encode time, decode time, and PSNR. A T4 loads float16. Capability 8 and above loads bfloat16. An out-of-memory forward turns tiling on, retries once, and records the flag. `docs/learn/07-pins-and-licenses.md`.

## 2026-10-19 — T4 smoke, and the torchao install break

The notebook ran on a free Colab T4, float16, Python 3.13. The first attempt failed for all four VAEs: `cannot import name 'FqnToConfig' from 'torchao.quantization'`. Colab preinstalls torchao 0.10.0. Diffusers 0.41.0 imports that name whenever torchao is installed, and 0.10.0 does not define it. The VAE load does not quantize, so the install cell now runs `pip uninstall -y torchao` before Diffusers. If torchao was already imported, the session has to be restarted. The cell also accepts a `latentcodecbench-*.whl` uploaded to `/content`, so a private repo does not need a token.

The second run loaded each VAE from `subfolder="vae"` and reconstructed one seeded 832×480×33 noise clip. PSNR is about 10 dB because the input is noise. That number is a load check. The latent shapes match step 2.1. `docs/results/vae-smoke-t4.csv`.

| VAE | Peak VRAM | Encode | Decode | PSNR | Latent shape | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| wan2_1 | 4.58 GiB | 8.09 s | 10.40 s | 9.80 dB | 16×9×60×104 | spatial tiling off |
| hunyuan | 4.19 GiB | 17.04 s | 32.22 s | 10.51 dB | 16×9×60×104 | out of memory until spatial tiling; framewise already on |
| ltx | 2.28 GiB | 0.86 s | 0.86 s | 10.75 dB | 128×5×15×26 | framewise off, spatial tiling off |
| wan2_2 | 5.87 GiB | 3.73 s | 14.64 s | 10.06 dB | 48×9×30×52 | spatial tiling off |

Hunyuan is the 16 GB problem. Step 2.3 turns spatial tiling on for it from the start, or runs it on an L4. The other three finished inside 6 GiB on this clip with tiling off.

Wan2.2 printed `config attributes {'clip_output': False} ... will be ignored`. The pinned `vae/config.json` was written for Diffusers `0.35.0.dev0`, which took `clip_output`. Release 0.41.0 removed the argument in commit `58d2b10` and always clamps the RGB tensor to `[-1, 1]` after `unpatchify`. The official `Wan2_2_VAE.decode` clamps the same way. The inner `WanVAE_.decode` does not. This benchmark's pixels are in `[-1, 1]`, so the dropped key does not move the smoke. The warning is not a skipped weight.

## 2026-10-19 — Phase 2.3, the reconstruction sweep

The Colab notebook that scores the real clips is in place. This change is the driver, the schema, and the merge script. The measured table is the next entry.

- `notebooks/vae_sweep.ipynb` uninstalls torchao, installs a wheel from `/content` when one is there, and otherwise installs the git ref. It fetches the 34 clips through `fetch_clip` and `preprocess_clip`, so the manifest SHA256 is checked, and it scores each clip with the Phase 1 metrics against the source y4m. The pinned ffmpeg is the September build from ADR-0003. Output is `/content/results/vae-recon.csv`, written after every row. A `(clip_id, vae)` already in the file is skipped.
- HunyuanVideo calls `enable_tiling()` before the first clip. The T4 smoke is why. The other three stay untiled unless a forward runs out of memory, then that clip is retried once. The flags, peak VRAM, encode time, decode time, and the mean `encoder.forward` time are columns. An L4 runs bfloat16. A T4 runs float16. ADR-0009. `docs/learn/08-reconstruction.md`.
- The classic reader loads the file. `crf`, `bits`, `bpp`, and `kbps` are empty. The VAE columns are omitted from a classic CSV, so the Phase 1 table is unchanged. `configs/vae-recon.yaml` is the hashed config, and the same bytes are packaged as `latentcodecbench/vaes/recon_config.yaml` because a wheel does not contain `configs/`. `manifests/clips.yaml` is not in the wheel.
- `scripts/merge-vae-recon.py` copies an uploaded CSV into `docs/results/vae-recon.csv`. It refuses a different config hash, a filled-in rate column, and a Hunyuan row with spatial tiling off.
- `fetch_clip` sends `User-Agent: LatentCodecBench/<version> (https://github.com/Shuvam1024/LatentCodecBench)`, because upload.wikimedia.org rejects Python's default agent with HTTP 403 (`street_50`, `lawn_mower`, `rust_laser`, `volcano`).
- A local wheel is installed with `--force-reinstall --no-deps`. `pip install --upgrade` was skipping a rebuilt `0.0.1`, so the session kept the previous code. The hatch hook writes `git rev-parse HEAD` into `latentcodecbench._build_info.GIT_SHA` for that wheel and restores the empty source file afterwards. The sweep reads the recorded commit. `GIT_SHA` in the cell, and `LATENTCODECBENCH_GIT_SHA`, still override it.
- The sweep mounts Google Drive at `/content/drive/MyDrive/LatentCodecBench/` unless `USE_DRIVE` is set false. Canonical clips prepared off Colab live at `data/clips/{id}.y4m` (flat, 832×480, 33 frames, yuv420p). The same filenames under `LatentCodecBench/clips/` are copied into `/content/data/clips` before fetch, so a file already there is not downloaded. `life`, `shaken_dry`, and `bosphorus` are the downloads that were dropping the runtime. `fetch_clip` now gives each attempt a 1200s wall clock and retries once. The CSV is `LatentCodecBench/results/vae-recon.csv`, rewritten and flushed after every row. A restart skips `(clip_id, vae)` pairs already in that file.
- A Colab L4 run printed `crowd_run wan2_1` through `rush_field_cuts wan2_1` and then went silent, while the Drive CSV already held all 34 `wan2_1` rows and the first 14 `hunyuan` rows. The log line was printed before the forward, a finished row was never confirmed, and a pair already stored was not mentioned. The CSV is rewritten sorted by clip id, so its order is not the log order. `life` is a finished `wan2_1` row; the next pair is `controlled_burn hunyuan`. `lpips` is empty because this pass does not compute it. `fsync` and `os.replace` on the Drive FUSE mount can block with no traceback. The sweep now prints `resume`, `skip`, `run`, and `done clip vae t=..s`, writes the table on local disk, copies it to Drive in a child that can be killed, and a per-row watchdog logs a heartbeat and moves on when the row exceeds 900s.

## 2026-10-20 — Phase 2.3, the L4 reconstruction table

`docs/results/vae-recon.csv` now holds the finished sweep: 136 rows, 34 clips times `wan2_1`, `hunyuan`, `ltx`, and `wan2_2`. `scripts/merge-vae-recon.py` copied the Colab file. The config hash matches `configs/vae-recon.yaml`. Rate columns stay empty. Hunyuan is spatially tiled on every row; the other three are not, and no row was retried. Every row is bfloat16 on an NVIDIA L4.

`lpips` is blank on every row because step 2.3 does not score it. `configs/vae-recon.yaml` lists `psnr_y`, `psnr_rgb`, `ssim`, `vmaf`, and `flicker`. The row builder writes `lpips` as empty even when torch is installed. ADR-0004 keeps LPIPS out of the default environment.

The run id is `20261020T135251Z-5e6307722225-8af5cb8e9b39`. It was minted on `5e63077`. The 34 `wan2_1` rows and the first 14 `hunyuan` rows still record that commit. The remaining 88 rows, written after the logging change, record `67da1f4`. One run, one config.

The rate-distortion figures stay the classic curves. Those rows have no `crf` and no `bpp`, so `lcb plot rd` would skip every one of them.

| VAE | Mean PSNR-Y | Median PSNR-Y | Mean VMAF | Median VMAF | Mean time (s) |
| --- | --- | --- | --- | --- | --- |
| wan2_1 | 34.57 | 34.10 | 86.40 | 89.21 | 10.76 |
| hunyuan | 35.11 | 34.96 | 87.63 | 89.69 | 24.54 |
| ltx | 32.30 | 31.98 | 74.56 | 78.62 | 0.88 |
| wan2_2 | 35.35 | 34.45 | 87.13 | 90.85 | 9.49 |

Mean time is encode plus decode. The three lowest PSNR-Y clips are `lawn_mower`, `park_joy`, and `crowd_run` for `wan2_1` and `ltx`; `lawn_mower`, `park_joy`, and `red_kayak` for `hunyuan`; `red_kayak`, `park_joy`, and `crowd_run` for `wan2_2`. The lowest VMAF on each VAE is `red_kayak`.

## 2026-10-20 — Phase 2.4, latent bitrate

The reconstruction table has no rate. This change defines the file a posterior mean becomes, and the columns that file fills. The 816 measured rows are not in the commit. They need the VAE, which is a Colab GPU run.

- The published normalization is the one Diffusers 0.41.0 applies around the latent. Wan uses per-channel mean and standard deviation. Hunyuan multiplies by 0.476986. LTX uses the mean and standard deviation stored in the pinned weights, and a `scaling_factor` of 1. The numbers are packaged in `latent_scale.json`.
- Uniform quantization on `[-4, 4]` at bit depths 2, 4, 6, 8, 10, and 12. Codes are a little-endian integer tensor, `uint8` or `uint16`, then zstd level 3. A 24-byte header is side information and is included in `bits`. `bpp` divides by `width * height * frames`, locked by a unit test on a synthetic latent. `crf` stores the bit depth. `preset` is `zstd`.
- `entropy_bits` is the cross-entropy of the codes under a factorized standard normal whose parameters were fixed before the clip. These VAEs were trained to reconstruct. A cross-entropy without a bitstream is an entropy estimate, so that number stays out of `bits`.
- `channels: null` keeps every channel. A later config can code a prefix and hand the decoder zeros for the rest, if the low depths still sit above the classic ladder.
- `notebooks/vae_bitrate.ipynb` is the GPU driver. `scripts/merge-vae-bitrate.py` copies the CSV back. `lpips` stays empty. ADR-0010. `docs/learn/05-latent-bitrate.md`. `docs/learn/06-a-prior-costs-bits.md`.

## 2026-10-21 — Phase 2.4, the L4 bitrate table

`docs/results/vae-bitrate.csv` now holds the finished sweep: 816 rows, 34 clips times `wan2_1`, `hunyuan`, `ltx`, and `wan2_2`, at bit depths 2, 4, 6, 8, 10, and 12. `scripts/merge-vae-bitrate.py` copied the Colab file. The config hash matches `configs/vae-bitrate.yaml`. Every row is bfloat16 on an NVIDIA L4. Run id `20261020T234203Z-8cb0b7cd9bc4-17ec7df8f1e6`. `lpips` is empty. `entropy_bits` stays the cross-entropy estimate and is not added into `bits`.

`scripts/summarize-vae-bitrate.py` prints mean PSNR-Y, mean VMAF, and median bpp for each VAE and depth. The table and the reading of it are in `docs/learn/05-latent-bitrate.md`. Quality levels off from about depth 8. Depth 2 is about 17 to 18 dB. At depth 12 the mean PSNR-Y of each VAE is within 0.1 dB of `docs/results/vae-recon.csv`. Hunyuan on `sintel_trailer` is 2.44 dB lower, and the latent tensors are not in the repo, so the `[-4, 4]` clipping account of that gap is a hypothesis. Wan2.1 on the same clip drops 0.48 dB from depth 8 to depth 10.
