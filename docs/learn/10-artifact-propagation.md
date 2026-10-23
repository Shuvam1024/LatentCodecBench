# Does source compression survive the VAE?

Phase 2 encoded clean clips. Phase 3 takes the same clips after x264, x265, or SVT-AV1 and asks whether that damage is still in the latent, and whether the reconstruction removes it or keeps it. The first half of this note defines the measurement. The second half, [Results](#results), reads the 1224 measured rows in `docs/results/artifact-propagation.csv`. ADR-0012.

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

Credentials are `/content/latentcodecbench/drive-credentials.json`. The first start asks for one Google account, then refreshes an access token with `google.auth.default` on the Drive scope and saves it. Colab's preinstalled Drive libraries are used as installed. Restart session keeps `/content`, so the next start loads the file and does not ask again. A new VM asks once more. If the prompt is dismissed, the log says `DRIVE AUTH UNAVAILABLE` and the sweep still writes the local CSV. Every row reads the credential file again. Interrupt the sweep, re-run the auth cell, then re-run the sweep: finished rows are skipped. Every push first downloads the Drive file and merges it into the local CSV, and it refuses to upload when the local file would still have fewer rows than Drive. An earlier version uploaded the local file as it was. After a restart that skipped the resume cell, a short local file came close to replacing the full Drive table. The notebook does not call `drive.mount`. `files.download` of the local CSV is the copy to take home when Drive is not.

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

## Results

`docs/results/artifact-propagation.csv` has 1224 rows: 34 clips, 4 VAEs, 9 degraded sources. One run id, `20261021T214039Z-933d37d99d88-3d42134b58f8`, one config hash, bfloat16 on an NVIDIA L4. No key is duplicated. `scripts/summarize-artifact-propagation.py --write` produces the table below, `docs/results/artifact-propagation-summary.csv`, and the three figures. A test checks that this table still matches the CSV.

Every cell is a median over the 34 clips. Retention cells are `median [Q1, Q3]`. "Additive gap" is the measured PSNR minus what two independent errors would give: the clean-source reconstruction PSNR from `docs/results/vae-recon.csv` and the degraded-source PSNR, added in MSE. Positive means the two errors overlap.

| VAE | Source | CRF | PSNR degraded | PSNR recon vs clean | Additive gap | Latent rel. L2 | Latent cosine | Blocking retention | Ringing retention |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wan2_1 | x264 | 18 | 41.39 dB | 34.40 dB | +0.99 dB | 0.138 | 0.9909 | -0.42 [-1.60, 0.06] | 2.16 [0.58, 3.61] |
| wan2_1 | x264 | 30 | 34.17 dB | 33.33 dB | +1.73 dB | 0.361 | 0.9524 | -0.21 [-0.68, 0.42] | 1.40 [0.98, 1.87] |
| wan2_1 | x264 | 38 | 30.28 dB | 30.26 dB | +1.39 dB | 0.527 | 0.9096 | 0.02 [-0.37, 0.18] | 1.01 [0.95, 1.14] |
| wan2_1 | x265 | 18 | 42.16 dB | 34.42 dB | +0.93 dB | 0.148 | 0.9908 | -0.47 [-1.73, 0.06] | 2.09 [1.00, 3.24] |
| wan2_1 | x265 | 30 | 35.26 dB | 33.53 dB | +1.64 dB | 0.320 | 0.9617 | -0.22 [-1.09, 0.16] | 1.20 [0.90, 1.46] |
| wan2_1 | x265 | 38 | 31.20 dB | 31.11 dB | +1.43 dB | 0.454 | 0.9326 | -0.00 [-0.61, 0.16] | 1.03 [0.95, 1.13] |
| wan2_1 | SVT-AV1 | 28 | 40.47 dB | 34.35 dB | +1.07 dB | 0.185 | 0.9869 | -0.37 [-2.69, 0.80] | 2.38 [1.11, 4.54] |
| wan2_1 | SVT-AV1 | 49 | 36.01 dB | 33.52 dB | +1.64 dB | 0.312 | 0.9649 | -0.08 [-1.12, 1.17] | 1.39 [0.89, 2.55] |
| wan2_1 | SVT-AV1 | 63 | 30.83 dB | 30.35 dB | +1.29 dB | 0.511 | 0.9215 | 0.30 [0.07, 0.43] | 1.03 [0.93, 1.13] |
| hunyuan | x264 | 18 | 41.39 dB | 35.15 dB | +1.00 dB | 0.099 | 0.9952 | 0.04 [-0.17, 0.39] | 2.34 [1.64, 4.08] |
| hunyuan | x264 | 30 | 34.17 dB | 33.39 dB | +1.59 dB | 0.219 | 0.9780 | 0.20 [0.04, 0.64] | 1.62 [1.20, 1.89] |
| hunyuan | x264 | 38 | 30.28 dB | 30.30 dB | +1.19 dB | 0.324 | 0.9519 | 0.48 [0.30, 0.72] | 1.12 [1.05, 1.27] |
| hunyuan | x265 | 18 | 42.16 dB | 35.15 dB | +0.93 dB | 0.098 | 0.9955 | -0.03 [-0.31, 0.34] | 2.54 [1.72, 3.54] |
| hunyuan | x265 | 30 | 35.25 dB | 33.81 dB | +1.52 dB | 0.191 | 0.9831 | 0.08 [-0.02, 0.42] | 1.33 [1.12, 1.56] |
| hunyuan | x265 | 38 | 31.20 dB | 31.17 dB | +1.18 dB | 0.275 | 0.9670 | 0.29 [0.20, 0.48] | 1.06 [1.02, 1.13] |
| hunyuan | SVT-AV1 | 28 | 40.40 dB | 35.11 dB | +1.01 dB | 0.120 | 0.9934 | -0.08 [-0.63, 0.41] | 3.31 [2.31, 5.12] |
| hunyuan | SVT-AV1 | 49 | 36.01 dB | 33.88 dB | +1.60 dB | 0.197 | 0.9812 | 0.37 [-0.03, 0.68] | 1.56 [1.15, 2.31] |
| hunyuan | SVT-AV1 | 63 | 30.57 dB | 30.41 dB | +1.13 dB | 0.318 | 0.9524 | 0.54 [0.39, 0.77] | 1.06 [0.99, 1.15] |
| ltx | x264 | 18 | 41.39 dB | 32.13 dB | +0.66 dB | 0.144 | 0.9897 | -1.70 [-3.32, -0.80] | 7.56 [3.49, 10.86] |
| ltx | x264 | 30 | 34.17 dB | 32.24 dB | +1.69 dB | 0.373 | 0.9397 | -0.68 [-1.32, -0.02] | 3.12 [1.98, 4.23] |
| ltx | x264 | 38 | 30.28 dB | 30.38 dB | +1.92 dB | 0.536 | 0.8821 | -0.40 [-0.75, -0.12] | 1.48 [1.26, 1.89] |
| ltx | x265 | 18 | 42.16 dB | 32.14 dB | +0.60 dB | 0.140 | 0.9905 | -1.52 [-2.84, -0.95] | 8.05 [3.25, 10.91] |
| ltx | x265 | 30 | 35.26 dB | 32.20 dB | +1.48 dB | 0.322 | 0.9541 | -0.90 [-2.01, -0.47] | 2.55 [1.68, 3.01] |
| ltx | x265 | 38 | 31.20 dB | 30.91 dB | +1.74 dB | 0.451 | 0.9142 | -0.68 [-2.13, -0.19] | 1.37 [1.13, 1.60] |
| ltx | SVT-AV1 | 28 | 40.41 dB | 32.14 dB | +0.71 dB | 0.168 | 0.9864 | -2.93 [-7.24, -1.09] | 9.92 [1.51, 16.60] |
| ltx | SVT-AV1 | 49 | 36.01 dB | 32.14 dB | +1.38 dB | 0.311 | 0.9565 | -0.69 [-3.89, 1.77] | 3.38 [1.43, 5.39] |
| ltx | SVT-AV1 | 63 | 30.63 dB | 29.92 dB | +1.77 dB | 0.496 | 0.9064 | -0.21 [-1.00, 0.05] | 1.42 [1.08, 1.71] |
| wan2_2 | x264 | 18 | 41.39 dB | 34.82 dB | +1.22 dB | 0.188 | 0.9828 | -0.22 [-0.96, 0.03] | 1.66 [0.94, 3.43] |
| wan2_2 | x264 | 30 | 34.17 dB | 33.49 dB | +1.69 dB | 0.410 | 0.9216 | -0.05 [-0.52, 0.41] | 1.31 [0.97, 1.69] |
| wan2_2 | x264 | 38 | 30.28 dB | 30.30 dB | +1.25 dB | 0.556 | 0.8612 | 0.26 [0.03, 0.36] | 1.03 [0.98, 1.13] |
| wan2_2 | x265 | 18 | 42.16 dB | 34.82 dB | +1.10 dB | 0.196 | 0.9817 | -0.34 [-1.26, -0.11] | 1.93 [1.12, 2.26] |
| wan2_2 | x265 | 30 | 35.25 dB | 33.80 dB | +1.73 dB | 0.362 | 0.9372 | -0.24 [-0.60, 0.07] | 1.12 [0.92, 1.23] |
| wan2_2 | x265 | 38 | 31.20 dB | 31.19 dB | +1.38 dB | 0.487 | 0.8979 | 0.00 [-0.42, 0.31] | 0.99 [0.95, 1.05] |
| wan2_2 | SVT-AV1 | 28 | 40.41 dB | 34.83 dB | +1.27 dB | 0.234 | 0.9746 | -0.72 [-2.04, -0.12] | 1.88 [1.33, 3.12] |
| wan2_2 | SVT-AV1 | 49 | 36.44 dB | 33.90 dB | +1.79 dB | 0.365 | 0.9400 | -0.04 [-0.95, 0.97] | 1.27 [1.06, 1.64] |
| wan2_2 | SVT-AV1 | 63 | 30.83 dB | 30.46 dB | +1.37 dB | 0.531 | 0.8846 | 0.33 [0.02, 0.53] | 1.01 [0.97, 1.10] |

### Why medians and not means

A retention ratio divides by `score(degraded) - score(clean)`. When the codec barely changed a score, that gap is close to zero and the ratio explodes. `station2` through Wan2.1 at SVT-AV1 CRF 28 has a blocking gap of about 0.00001 code values and a retention of -1403. One clip like that moves a 34-clip mean by about 40. The overall mean blocking retention is -3.0 and the standard deviation is 45.5, while the median is -0.10.

Two options were on the table: drop rows whose gap is under some floor, or use statistics that do not care about the tails. A floor needs a number, and 111 rows have a blocking gap under 0.01 code values while 68 have a ringing gap under 0.1, so the choice of number would decide which rows count. The table uses the median and the interquartile range instead. `artifact-propagation-summary.csv` also has the pooled ratio, `sum(recon - clean) / sum(degraded - clean)` over the clips. Pooling weights each clip by how much artifact the codec actually added, so a near-zero gap contributes almost nothing. The pooled ratio and the median have the same sign and the same trend across CRF in every group. Example: Hunyuan through x264 is 0.04, 0.20, 0.48 by median and -0.05, 0.15, 0.47 pooled.

### What matches expectations

- The latent moves further from the clean latent as the CRF rises. Relative L2 increases from the lowest to the highest CRF on all 408 clip, VAE, and codec curves. Hunyuan moves least: x264 18, 30, 38 give 0.099, 0.219, 0.324. Wan2.2 moves most: 0.188, 0.410, 0.556. The cosine never goes below a median of 0.86 (Wan2.2, x264 CRF 38).
- Degraded-source PSNR falls with CRF on every clip and codec. At matched CRF, x265 is about 0.8 to 1.1 dB above x264, as in Phase 1.
- The reconstruction is always closer to the degraded input than to the clean source. `psnr_recon_degraded` is above `psnr_recon_clean` on all 1224 rows. The VAE reproduces what it was given. It does not restore what the codec removed.
- At the harshest point the codec loss dominates. Through x264 CRF 38 the reconstruction is 30.26 to 30.38 dB against the clean source, and the degraded input itself is 30.28 dB. Against each clip's clean-source reconstruction, the top CRF costs a median 1.8 to 5.0 dB depending on VAE and codec. LTX loses least because its clean reconstruction is already the weakest (31.98 dB median).

### What is surprising

- **Mild compression costs nothing, and slightly helps.** At x264 and x265 CRF 18 and SVT-AV1 CRF 28 the median per-clip difference between the reconstruction (scored against the clean source) and the reconstruction of the clean source itself is -0.03 to +0.23 dB, positive on 47% to 91% of clips per group. Hunyuan is flat (-0.03 to +0.08 dB). The three others gain +0.10 to +0.23 dB. Codec noise at that level sits under the VAE's own error, and the encoder's smoothing removes some grain the VAE could not reproduce anyway. The effect is small, and both sweeps are bfloat16, so a few hundredths of a dB is within run-to-run noise. The direction holds for Wan2.1, Wan2.2, and LTX.
- **The errors overlap.** The additive gap is +0.60 to +1.92 dB in every group. Both the codec and the VAE throw away high-frequency detail, so the second one removes less that the first did not already remove.
- **The ringing score goes down under compression.** On 87.7% of rows the degraded source has a lower ringing score than the clean source. The score is the second difference beside a strong edge. At these CRFs the codecs soften edges more than they add overshoot, so the score is acting as a near-edge detail measure. A ringing retention above 1 therefore means the reconstruction is smoother beside edges than the degraded input. LTX is the extreme: 7.56 to 9.92 at the mildest CRF points. Read the column as edge softening, not as ringing kept.
- **There is very little 8×8 blocking to retain.** The clean score is 0.0063 code values. The degraded medians are 0.02 to 0.20 code values, highest at x264 CRF 38 (0.195) and SVT-AV1 CRF 63 (0.167). All three encoders deblock in-loop, and x265 and SVT-AV1 use variable block sizes rather than a fixed 8×8 grid. Hunyuan keeps a growing share of what there is: 0.04, 0.20, 0.48 for x264 18, 30, 38. LTX is negative everywhere: its reconstruction has smaller jumps across the 8×8 grid than the clean source does (median blocking score -0.11 against +0.006), so the ratio says it overshoots past clean in the smooth direction. Wan2.1 and Wan2.2 sit near zero, turning slightly positive at the highest CRF.

### The SVT-AV1 degraded files were not the same for every VAE

Every VAE should see the same degraded file for a clip and CRF, so `psnr_degraded_clean` should agree to the last digit across VAEs. x264 does, on all 34 clips. x265 differs on 17 clips by up to 0.057 dB. SVT-AV1 differs on 18 clips by up to 1.35 dB. The median SVT-AV1 CRF 49 value is 36.01 dB for three VAEs and 36.44 dB for Wan2.2.

The pattern points at a re-encode between sessions. For each affected clip there are only two versions of the file, and they split the VAEs into contiguous blocks in the order the sweep ran (for example the first four clips alphabetically differ only on Wan2.2, and `rush_field_cuts` through `shaken_dry` differ only on Hunyuan). The same clips change for x265 and SVT-AV1 together. That is what happens when `/content/data/degraded` was wiped by a new VM and `degrade_sources` rebuilt it: x264 at these settings reproduced bit for bit, while x265 and SVT-AV1 did not. The data cannot tell whether the cause is a different CPU instruction set on the new host (both encoders pick SIMD kernels at run time) or thread scheduling. The ffmpeg build is pinned in both sessions.

Each row is still internally consistent: latent distance, both reconstruction PSNRs, and the artifact scores all use the file that VAE actually saw. What breaks is the cross-VAE comparison at one SVT-AV1 point on those 18 clips, which carries up to 1.35 dB of source difference. The medians move by much less (the CRF 49 example is the largest at 0.43 dB). A rerun should build the degraded set once, record each file's SHA256, and copy it to Drive with the clean tarball.

### Hunyuan on `sintel_trailer` (step 2.4)

Step 2.4 found Hunyuan's depth-12 reconstruction 2.44 dB under the unquantized one on `sintel_trailer`. The existing data supports the clipping explanation but cannot prove it. On every other clip the quantization loss shrinks by roughly 10× for each two bits (median 0.11 dB at depth 8, 0.01 dB at depth 10, 0.00 dB at depth 12), which is what rounding error does. On `sintel_trailer` it is 2.38, 2.43, 2.44 dB at depths 8, 10, 12: a floor that more bits do not lower. A floor independent of step size is the signature of values saturating at the ends of `[-4, 4]`, and Hunyuan is normalized by `scaling_factor` alone, with no per-channel shift. `dinner` shows a smaller floor of the same shape (0.29, 0.24, 0.22 dB). Phase 3 does not quantize, so its rows say nothing more. Confirming it needs one GPU encode: the fraction of normalized Hunyuan latent values outside `[-4, 4]` on `sintel_trailer` and `dinner`, against a clip with no floor.

### What this means for training data

A VAE trained on compressed video sees latents that sit measurably away from the clean latent, and the reconstruction keeps the damaged signal instead of restoring the clean one. At the mild end (x264/x265 CRF 18, SVT-AV1 CRF 28) the pixel cost is below the VAE's own error, so filtering that data out buys nothing measurable here. At the harsh end the latent shift is three to four times larger and the codec loss passes straight through. A filter on source quality belongs at roughly the middle of these ladders and above.
