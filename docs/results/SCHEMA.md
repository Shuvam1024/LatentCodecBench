# Results table

`docs/results/classic-baselines.csv` is the tidy table for the classic-codec sweep. One row is one codec at one CRF on one clip. The file is CSV so a diff shows the number that moved. Parquet would be smaller and harder to review; this table is a few hundred rows.

Rows are sorted by `clip_id`, then `codec`, then `crf`.

| Column | Meaning |
| --- | --- |
| `run_id` | `{utc}-{git_sha12}-{config_sha12}`. UTC is when the table was created. The git SHA is the revision that ran the sweep, twelve hex characters. The config SHA is the SHA256 of the config file, twelve hex characters. |
| `git_sha` | Full SHA of that revision. |
| `config_sha256` | Full SHA256 of the config file bytes. |
| `clip_id` | Id from `manifests/clips.yaml`. |
| `codec` | `libx264`, `libx265`, or `libsvtav1`. |
| `preset` | The preset string passed to ffmpeg. |
| `crf` | The CRF integer. Not comparable across codecs. |
| `width`, `height`, `frames` | Geometry of the canonical clip. |
| `fps_num`, `fps_den` | Frame rate copied from the y4m header. |
| `bits` | Sum of video-packet sizes, in bits. The mp4 wrapper is not included. |
| `bpp` | `bits / (width * height * frames)`. |
| `kbps` | `bits / duration / 1000`, with duration `frames * fps_den / fps_num`. |
| `encode_seconds` | Wall clock of the encoder process, one thread. |
| `decode_seconds` | Wall clock of a one-thread decode to null. Writing the reconstructed y4m for the metrics is not included. |
| `psnr_y`, `psnr_rgb` | Pooled-MSE PSNR, dB. See ADR-0004. |
| `ssim` | Mean luma SSIM. |
| `vmaf` | libvmaf pooled mean. |
| `flicker` | Mean absolute disagreement of frame-to-frame luma deltas. |
| `lpips` | AlexNet LPIPS, or empty when torch or lpips is not installed. |

A second run against the same file skips a `(clip_id, codec, crf)` that is already present and keeps the original `run_id`. A file whose `config_sha256` does not match the config on the command line is left alone.

BD-rate is not a column. It is computed from the rows, per clip, then averaged. `lcb plot rd` prints that mean and the mean overlap.

`docs/results/overlap.csv` is the per-clip overlap that mean is taken from. One row is one clip, one test codec, and one metric, against libx264.

| Column | Meaning |
| --- | --- |
| `clip_id` | Id from the results table. |
| `anchor` | `libx264`. |
| `codec` | The test codec, `libx265` or `libsvtav1`. |
| `metric` | `psnr_y` or `vmaf`. |
| `overlap` | Intersection of the two metric ranges, divided by their union. This is the fraction the `bjontegaard` package compares with 0.75. |
| `below_0_75` | `1` when `overlap` is under 0.75, otherwise `0`. |

A pair under 0.75 is still included in the mean BD-rate when the ranges share any interval. The flag says the integral only covers part of the combined curve. A pair with no shared range is omitted from the BD-rate mean and still listed here with overlap `0`.

`docs/results/vae-smoke-t4.csv` is the one-clip Colab T4 load check from step 2.2. One row is one VAE. The clip is seeded uniform noise, 832×480×33, values in `[-1, 1]`, float16, Python 3.13. PSNR uses a peak of 2. A value near 10 dB is what that noise reconstructs to. It is a load and shape check. The latent shapes match step 2.1.

| Column | Meaning |
| --- | --- |
| `name` | `wan2_1`, `hunyuan`, `ltx`, or `wan2_2`. |
| `peak_vram_gib` | `torch.cuda.max_memory_allocated` for the successful forward, in GiB. |
| `encode_seconds`, `decode_seconds` | Wall clock of that forward, after a CUDA sync. |
| `psnr_db` | Pooled PSNR of the reconstruction against the noise input. |
| `latent_shape` | `channels x frames x height x width`. |
| `notes` | Empty, or the tiling retry. Hunyuan ran out of memory until spatial tiling was on. Framewise encoding was already on. |

`docs/results/vae-recon.csv` is the reconstruction sweep from step 2.3 (136 rows, copied in by `scripts/merge-vae-recon.py`). One row is one VAE on one clip. The classic columns above are all present, so `read_results` loads the file. `codec` is `wan2_1`, `hunyuan`, `ltx`, or `wan2_2`. `preset` is `reconstruction`. `crf`, `bits`, `bpp`, and `kbps` are empty: this pass does not quantize the latent. `lpips` is empty because this config does not score it. Rows are sorted by `clip_id`, then `codec`, then `crf`, and an empty `crf` sorts first.

The config hash is the SHA256 of `configs/vae-recon.yaml`. A second run that sees a different hash leaves the file alone. A `(clip_id, codec)` already in the file is skipped, and the original `run_id` is kept.

| Column | Meaning |
| --- | --- |
| `peak_vram_bytes` | `torch.cuda.max_memory_allocated` on the forward that succeeded. |
| `spatial_tiling` | `1` when `use_tiling` was on for that forward. Hunyuan is `1` from the start. On 832×480 that switch tiles, because the class minimum is 256. |
| `framewise_encoding`, `framewise_decoding` | The class flags after the tiling policy. Hunyuan's defaults are on. |
| `retried_with_tiling` | `1` when the first forward ran out of memory and the clip was scored after tiling was turned on. |
| `per_chunk_seconds` | Mean wall clock of `encoder.forward` during the encode, after a CUDA sync. |
| `chunk_count` | How many of those calls ran. Wan uses `1 + (frames - 1) // 4`. |
| `torch_version`, `diffusers_version` | The versions in the runtime that produced the row. |
| `runtime_dtype` | `bfloat16` on compute capability 8 and above (an L4). `float16` on a T4. |
| `gpu_name` | `torch.cuda.get_device_name`. |
| `vae_revision` | The pinned revision from ADR-0008. |
| `latent_shape` | `channels x frames x height x width`, without the batch axis. |

`docs/results/vae-smoke-t4.csv` is a different file. Its PSNR uses a peak of 2 on noise. The reconstruction table uses the Phase 1 metrics on the canonical clips.

`docs/results/vae-bitrate.csv` is the latent-bitrate sweep from step 2.4 (816 rows, copied in by `scripts/merge-vae-bitrate.py`). One row is one VAE, one clip, and one bit depth. `codec` is the VAE name. `preset` is `zstd`. `crf` is the quantizer bit depth: 2, 4, 6, 8, 10, or 12. It is not an x264 CRF. `bits` is eight times the length of the 24-byte header plus the zstd payload. `bpp` is `bits / (width * height * frames)`. `kbps` uses the clip frame rate. `entropy_bits` is the factorized standard-normal cross-entropy of the codes, in bits. It is an estimate, not a file size, and it is not added into `bits`. `lpips` is empty. The VAE columns match the reconstruction table. Rows are sorted by `clip_id`, then `codec`, then `crf`. The run id is `20261020T234203Z-8cb0b7cd9bc4-17ec7df8f1e6`. `scripts/summarize-vae-bitrate.py` prints the mean PSNR-Y, mean VMAF, and median bpp at each depth. ADR-0010.

The config hash is the SHA256 of `configs/vae-bitrate.yaml`. A second run that sees a different hash leaves the file alone. A `(clip_id, codec, crf)` already in the file is skipped.

`docs/results/vae-rd-bdrate.csv` is the per-clip BD-rate of each VAE against `libx264`, `libx265`, and `libsvtav1`, on `psnr_y` and `vmaf`. `status` is `defined`, `no_overlap`, `nonmonotonic`, or `too_few_points`. `bd_rate` is a percent from the same PCHIP as the classic table, and it is empty unless `status` is `defined`. `overlap` is the shared fraction of the two quality ranges. `below_0_75` is `1` when a defined pair is under 75%, and empty when the rate is undefined. Depth 2 is not a sample in this file. The mean curves, which do include depth 2, are `rd-psnr.png` and `rd-vmaf.png`. `rd-bdrate-clips.png` is the per-clip PSNR-Y rate against x264. `scripts/summarize-vae-rd.py` reprints the means. ADR-0011.

`artifact-propagation.csv` is the Phase 3 table. It is not committed yet. The Colab notebook writes it on local disk, one row per clip, VAE, and degraded source. Rows sort by `clip_id`, `vae`, `source_codec`, `source_crf`. The resume key is that tuple. A Drive copy of the same file is merged on that key at startup: the local row wins when both sides have it. A file whose `config_sha256` does not match the study ladder is left alone. ADR-0012.

| Column | Meaning |
| --- | --- |
| `run_id`, `git_sha`, `config_sha256` | Same idea as the classic table. The config hash is the study ladder and the benchmark presets. |
| `clip_id`, `vae` | Canonical clip id, and `wan2_1`, `hunyuan`, `ltx`, or `wan2_2`. |
| `source_codec`, `source_crf`, `preset` | The codec that degraded the source, its CRF, and the Phase 1 preset. |
| `width`, `height`, `frames` | Geometry of the canonical clip. |
| `latent_mse`, `latent_relative_l2`, `latent_cosine` | Distance from the clean posterior mean to the degraded one. |
| `psnr_recon_clean` | Luma PSNR of the degraded reconstruction against the clean source. |
| `psnr_recon_degraded` | Luma PSNR of that reconstruction against the degraded source. |
| `psnr_degraded_clean` | Luma PSNR of the degraded source against the clean source. |
| `blocking_clean`, `blocking_degraded`, `blocking_recon` | 8x8 boundary jump minus interior jump, on luma. |
| `blocking_retention` | `(recon - clean) / (degraded - clean)`. Empty when the gap is under `1e-6`. |
| `ringing_clean`, `ringing_degraded`, `ringing_recon` | Mean absolute second difference beside strong edges. |
| `ringing_retention` | Same ratio as blocking. Empty when the gap is under `1e-6`. |
| `clean_encode_seconds` | Wall clock of the clean encode. Shared by the nine rows of that clip and VAE. |
| `encode_seconds`, `decode_seconds` | Wall clock of the degraded encode and its decode. |
| `runtime_dtype`, `gpu_name` | The runtime that produced the row. |
