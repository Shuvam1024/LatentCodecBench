# ADR-0012: Source-artifact study uses a three-point Phase 1 ladder

- Status: Accepted
- Date: 2026-10-21

## Context

Phase 2 measured each VAE on the clean canonical clips. Phase 3 asks a different question: when the source is already compressed, does that damage show up in the latent, and does the reconstruction keep it?

The Phase 1 ladders are six CRF points per codec. Running every point through four VAEs, with an encode of the clean source and an encode plus decode of each degraded source, is a long L4 job. The last bitrate sweep also lost time to `drive.mount` re-consent and to FUSE calls that never returned. A restart had to mount again, and a hung copy stalled the loop.

## Decision

The study ladder is three points from each Phase 1 ladder in ADR-0005. x264 and x265 use CRF 18, 30, and 38. SVT-AV1 uses CRF 28, 49, and 63. Those are the two ends and the middle step. Presets stay `medium`, `medium`, and SVT-AV1 preset 6. `scripts/degrade-sources.py` writes the decoded y4m under `data/degraded/` and skips a file whose geometry already matches the source. The y4m files stay gitignored with the rest of `/data/`.

Each VAE row records:

- Latent distance from the clean posterior mean to the degraded one: MSE, relative L2 (`||diff|| / ||clean||`, or `||diff||` when the clean norm is 0), and cosine.
- Luma PSNR of the reconstruction of the degraded source against the clean source, and against the degraded source. The clean reconstruction is not decoded. The degraded-versus-clean PSNR is stored beside them so the input damage has a number.
- Blocking: mean absolute jump across 8×8 boundaries minus the mean absolute jump inside those blocks, horizontal and vertical pooled.
- Ringing: mean absolute second difference on the samples one step outside a strong edge. A strong edge is an absolute neighbor difference of at least 12 on the 0–255 scale. The two samples that form the edge are excluded, so a pure step scores 0.
- Retention `(score(recon) - score(clean)) / (score(degraded) - score(clean))`. 0 means the artifact is back at the clean level. 1 means it matches the degraded input. The ratio is empty when the degraded score is within `1e-6` of the clean score.

Quality in this table is luma PSNR, the pooled-MSE definition from ADR-0004. VMAF can be added when the rows are scored. It is not part of this definition, and it is not part of the runtime estimate.

The GPU driver writes `/content/results/artifact-propagation.csv` on Colab local disk and flushes after every row. Drive is a second copy, reached through the Drive API. At startup the notebook downloads `My Drive/LatentCodecBench/results/artifact-propagation.csv` and merges it into the local file. The resume key is `(clip_id, vae, source_codec, source_crf)`. A key on both sides keeps the local row. A key only on Drive is filled in. Two config hashes are refused, and the local file is left as it was. After each new row the same Drive file is updated by file id, in a child process with a 60 second timeout. A timeout leaves the local file in place and the loop continues. Clean clips come from `/content/data/clips` when those y4m files are already there, otherwise from `My Drive/LatentCodecBench/canonical-clips.tar`. The nine degraded variants are encoded on the Colab disk with the pinned ffmpeg. Credentials live at `/content/latentcodecbench/drive-credentials.json`, which survives Restart session. `authenticate_user` runs when that file is missing or no longer usable. That is one account prompt. The notebook then calls `google.auth.default` with the Drive scope and `credentials.refresh(Request())`, and saves the access token. Colab's preinstalled Drive libraries are left as installed. The install cell does not upgrade them. Each sweep row re-reads the file. A row that cannot authenticate stays local, and a later row uploads the whole local CSV once the file is usable again. A later Restart session loads the file. If auth is unavailable the sweep logs `DRIVE AUTH UNAVAILABLE` and keeps the local CSV. The notebook does not call `drive.mount`. `files.download` of the local CSV is the fallback.

The L4 estimate uses the mean encode and decode from `docs/results/vae-recon.csv` (34 clips, bfloat16, NVIDIA L4):

`34 * (encode + 9 * (encode + decode))`

That is one clean encode per clip and nine degraded encode-and-decode passes. Wan2.1 is about 57 min, Hunyuan about 2.2 h, LTX about 5 min, and Wan2.2 about 50 min. Pixel metrics run on CPU and are outside that estimate.

## Consequences

Nine variants times 34 clips is 306 degraded y4m files per machine, and 1224 VAE rows once all four models have run. A full six-point ladder would about double the GPU time. The three points still span the Phase 1 range: the best CRF, a middle CRF, and the softest CRF on each codec's own scale. CRF numbers stay incomparable across codecs, as in ADR-0005.

Four of the 34 sources are already Theora or VP9 re-encodes. Their degraded files are a second generation. The table includes them with the rest of the set.

A restart downloads the Drive CSV and merges it into the local file. The local row wins on a shared key, so a row that has not been uploaded yet is kept. The upload after every row keeps that window to one row. A hung API call dies at the timeout. Missing auth is logged and the loop continues on local disk.

## Alternatives

**QP 20 to 50 on x264 and SVT-AV1 only.** That is a new scale, and it drops x265. The Phase 1 ladders already have measured operating points. Reusing a subset keeps the degraded sources on the same CRF numbers as the classic table.

**All six Phase 1 points.** That is the full overlap study. It doubles the degraded encodes and the VAE forwards. Three points answer whether the damage survives. The interior points can be filled later if the ends are ambiguous.

**Write the CSV straight to Drive.** That is how the bitrate sweep lost rows: the mount prompt and a stuck FUSE write stopped the loop. Local disk is the file the loop trusts. Drive is a copy.

**Call `drive.mount` at the top of the notebook.** A restarted runtime asks for consent again, and the cell blocks until a person clicks. The notebook uses the Drive API and a credential file on `/content` instead. Restart session reuses that file.

**Copy through `/content/drive` when that directory is already mounted.** A fresh runtime never has the directory, so the backup never runs. An API call by file id does not need a mount.

**Replace the local CSV with the Drive file at startup.** A row written locally and not yet uploaded would disappear. The merge keeps the local row for a shared key and fills keys only the Drive file has.

## Addendum, 2026-10-23: reading the results

The 1224 measured rows are in `docs/results/artifact-propagation.csv`. Three decisions came out of reading them.

**Summary statistic.** Retention ratios are summarized by median and interquartile range, with the pooled ratio `sum(recon - clean) / sum(degraded - clean)` alongside. A near-zero denominator produced per-clip values down to -1403, so a mean is meaningless. A denominator floor was rejected because its threshold would decide which rows count: 111 rows have a blocking gap under 0.01 code values. The pooled ratio weights clips by the artifact the codec actually added, and it agrees with the median in sign and trend in every group.

**What ringing measures here.** On 87.7% of rows the degraded source scores lower than the clean source. At these CRFs the encoders soften edges more than they overshoot, so the score is reported as near-edge detail. A retention above 1 is read as "smoother than the degraded input", not "more ringing".

**Degraded files must be built once.** `psnr_degraded_clean` differs across VAEs for the same clip and point on 17 clips for x265 (up to 0.057 dB) and 18 for SVT-AV1 (up to 1.35 dB). x264 matches everywhere. The split follows sweep order, which fits `/content/data/degraded` being rebuilt on a new VM, with x265 and SVT-AV1 producing different bytes there. Rows stay valid on their own. Cross-VAE comparisons at one SVT-AV1 point on those clips carry up to 1.35 dB of source difference. A rerun builds the degraded set once, records each file's SHA256, and ships it with the clean tarball.

**Drive upload merges first.** Every upload now downloads the Drive CSV, merges it into the local file, and refuses to upload a file with fewer rows than Drive. An upload of an unmerged local file came close to replacing the full table after a restart that skipped the resume cell.
