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

BD-rate is not a column. It is computed from the rows, per clip, then averaged. `lcb plot rd` prints that mean.
