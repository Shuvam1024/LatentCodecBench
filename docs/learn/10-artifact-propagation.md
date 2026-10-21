# Does source compression survive the VAE?

Phase 2 encoded clean clips. Phase 3 takes the same clips after x264, x265, or SVT-AV1 and asks whether that damage is still in the latent, and whether the reconstruction removes it or keeps it. The measured rows are a later Colab run. This note is the definition those rows will use. ADR-0012.

## The ladder

The Phase 1 sweep used six CRF points per codec. This study keeps three: the best, the middle step, and the softest.

| Codec | CRF points | Preset |
| --- | --- | --- |
| x264 | 18, 30, 38 | medium |
| x265 | 18, 30, 38 | medium |
| SVT-AV1 | 28, 49, 63 | 6 |

Each number is still that codec's own dial. x264 CRF 30 is not SVT-AV1 CRF 30. The presets are the ones timed in the classic sweep, so a degraded file is the same encoder setting as the matching row in `docs/results/classic-baselines.csv`.

`scripts/degrade-sources.py` reads `data/clips/*.y4m` and writes `data/degraded/{clip}__{codec}_crf{crf}.y4m`. A file that already has the source width, height, frame count, and frame rate is skipped. The y4m files stay out of git, with the rest of `data/`.

Nine files per clip, 34 clips, 306 files. On the timings from the classic sweep (about 1–2 seconds a clip at one thread), building them is a CPU job of a few minutes to a few tens of minutes, depending on the machine.

## What gets measured

The clean source is encoded once per clip per VAE. That posterior mean is the reference latent. Each degraded source is encoded and decoded. The clean reconstruction is not decoded: both pixel scores look at the reconstruction of the degraded input.

Latent distance is three numbers between those posterior means.

- MSE, the mean squared element difference. A uniform shift of `c` has MSE `c²`.
- Relative L2, `||clean - other|| / ||clean||`. When the clean tensor is all zeros, the denominator would be zero, so the number is `||clean - other||` itself.
- Cosine. Identical tensors score 1. A zero tensor against anything else scores 0. Two zero tensors score 1. A flat tensor shifted by a constant stays at cosine 1, because the direction did not change. The relative L2 is what records that shift.

Pixel quality is luma PSNR, the same pooled-MSE definition as Phase 1.

- Reconstruction against the clean source. This falls when the VAE keeps the source damage, and also when the VAE adds its own error.
- Reconstruction against the degraded source. This stays high when the VAE is copying the damaged input.
- Degraded source against the clean source. This is how hard the codec hit the clip before the VAE saw it.

Blocking is a single luma number. Take the mean absolute difference across every 8×8 block boundary, subtract the mean absolute difference of the neighboring pixels inside the blocks, and pool horizontal and vertical. A constant frame scores 0. A smooth ramp scores 0, because boundary and interior jumps match. A hard 8×8 edge scores high. 832 and 480 are both divisible by 8, so every canonical clip has a score.

Ringing looks beside a strong edge. A strong edge is a neighbor difference of at least 12 levels on the 0–255 scale. The two samples that make that edge are left out. The samples one step farther away contribute their absolute second difference. A pure step has flat neighbors, so the score is 0. A one-pixel overshoot next to the step makes the score rise. No strong edge scores 0.

Retention puts those artifact scores on one scale:

`(score(recon) - score(clean)) / (score(degraded) - score(clean))`

0 means the reconstruction is back at the clean artifact level. The VAE removed the damage. 1 means the reconstruction matches the degraded input. The VAE kept it. Above 1 means the reconstruction is more damaged than the input it was given. When the degraded score is within `1e-6` of the clean score, the codec did not add that artifact and the ratio is left empty.

## Where the rows go

The sweep writes `/content/results/artifact-propagation.csv` on the Colab disk and flushes that file after every row. Drive is a separate copy of that same file: `My Drive/LatentCodecBench/results/artifact-propagation.csv`. At startup the notebook downloads it through the Drive API and merges the two CSVs. After each new row the same Drive file is updated by its file id, in a child process with a 60 second timeout. A timeout is logged, and the loop keeps going. The local file is the one the loop trusts.

The resume key is `(clip_id, vae, source_codec, source_crf)`. A key on both sides keeps the local row. A key that only Drive has is copied in. A CSV built from a different ladder is refused, and the local file is left as it was.

Clean clips are the y4m files already in `/content/data/clips`. When that directory is empty, the notebook downloads `My Drive/LatentCodecBench/canonical-clips.tar` and extracts it, then encodes the nine degraded variants on the Colab disk. The pinned ffmpeg is already installed for that encode.

Credentials are `/content/latentcodecbench/drive-credentials.json`. The first start asks for one Google account, then refreshes an access token with `google.auth.default` on the Drive scope and saves it. Colab's preinstalled Drive libraries are used as installed. Restart session keeps `/content`, so the next start loads the file and does not ask again. A new VM asks once more. If the prompt is dismissed, the log says `DRIVE AUTH UNAVAILABLE` and the sweep still writes the local CSV. Every row reads the credential file again. Interrupt the sweep, re-run the auth cell, then re-run the sweep: finished rows are skipped, and the next push uploads the whole local CSV. The notebook does not call `drive.mount`. `files.download` of the local CSV is the copy to take home when Drive is not.

## L4 time

The estimate uses the mean encode and mean decode from `docs/results/vae-recon.csv`: 34 clips, bfloat16, NVIDIA L4. Per VAE the count is one clean encode plus nine degraded encode-and-decode passes:

`34 * (encode + 9 * (encode + decode))`

| VAE | Mean encode | Mean decode | L4 estimate |
| --- | --- | --- | --- |
| wan2_1 | 4.02 s | 6.73 s | 57 min |
| hunyuan | 8.70 s | 15.83 s | 2.2 h |
| ltx | 0.42 s | 0.46 s | 5 min |
| wan2_2 | 2.02 s | 7.47 s | 50 min |

Those means are the reconstruction sweep, so this is a forecast, not a new measurement. Luma PSNR and the artifact scores run on CPU after the decode and are not in the table. Four sources (`street_50`, `lawn_mower`, `rust_laser`, `volcano`) were already Theora or VP9, so their degraded files are a second generation. They stay in the set.
