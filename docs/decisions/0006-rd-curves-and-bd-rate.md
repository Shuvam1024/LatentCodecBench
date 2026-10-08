# ADR-0006: Results table, RD curves, and BD-rate

- Status: Accepted
- Date: 2026-10-07
- Revised: 2026-10-08

## Context

The sweep is 34 clips times 3 codecs times 6 CRF points: 612 encodes. That has to land as one table a reviewer can diff, one pair of figures, and one number that says whether x265 or SVT-AV1 beat x264. The number people expect is BD-rate. There are two common ways to compute it, and they do not agree on a curve that bends.

## Decision

The table is CSV, at `docs/results/classic-baselines.csv`. Columns and sort order are in `docs/results/SCHEMA.md`. Sort is `clip_id`, `codec`, `crf`. Parquet would not review as a diff, and this table is a few hundred rows.

A run id is `{utc}-{git_sha12}-{config_sha12}`. The git SHA is the revision that executed the sweep. The config SHA is the SHA256 of the config file. Re-running with the same config appends only missing `(clip, codec, CRF)` rows and keeps the original run id. A different config against the same file is refused.

BD-rate and BD-PSNR use a PCHIP interpolant (piecewise cubic Hermite), which is what the JCT-VC common test conditions use. BD-rate integrates `log(rate)` against distortion over the overlapping distortion range, then reports `exp(mean log gap) - 1` as a percent. BD-PSNR integrates distortion against `log(rate)` over the overlapping log-rate range. The anchor is libx264. The reported number is the mean of the per-clip BD-rates. A clip with no overlap is skipped and counted.

Overlap is stored per clip, per test codec, and per metric in `docs/results/overlap.csv`. The fraction is the intersection of the two metric ranges divided by their union, which is what the `bjontegaard` package compares with 0.75 before it warns. `below_0_75` is 1 for a pair under that line. The mean of those fractions is reported next to the BD-rate. A narrow overlap is a property of the number, so the pair stays in the mean and is flagged. It is not treated as if the curves covered each other.

The implementation is locked to two cases. A test curve at exactly half the anchor rate and the same PSNR samples has BD-rate -50%. The four-point vector shipped with the `bjontegaard` package, method `pchip`, has BD-rate -4.417485350589045% and BD-PSNR 0.11969278542479489.

Figures are the mean, across clips, of bits per pixel and of PSNR-Y or VMAF at each CRF. The x-axis is logarithmic. `lcb plot rd` writes `docs/figures/rd-psnr.png` and `docs/figures/rd-vmaf.png` and prints the mean BD-rates.

## Consequences

The CSV that gets committed names the code revision it was measured with. The commit that adds the CSV is a child of that revision, so the SHA in the file is the parent, not the commit that contains the file.

CRF is stored so a row can be repeated. It is not the x-axis. x264 CRF 28 and SVT-AV1 CRF 28 are different operating points; the plot does not align them.

A non-monotonic curve (a higher rate with a lower PSNR) cannot be interpolated and is reported as a skip rather than a silent NaN.

## Alternatives

**A cubic polynomial through the log-rate samples.** The original Bjontegaard fit. Four points determine it, and it can loop outside them. PCHIP stays inside the samples and matches the CTC spreadsheet. The `bjontegaard` package documents that agreement.

**One BD-rate on the pooled curve** (mean bpp, mean PSNR, then one integral). That hides a clip where the ranking flips. Per-clip then mean is what a sequence-level report does.

**Parquet.** Better once the table has VAE rows in the millions of numbers. It is the wrong format for the first table a reviewer will open.
