# ADR-0011: VAE BD-rate excludes depth 2 and does not extrapolate

- Status: Accepted
- Date: 2026-10-21

## Context

Step 2.4 filled `bpp` on the four VAEs. Step 2.5 puts those rows on the same axes as the Phase 1 codecs and asks for a BD-rate. The integral already exists: ADR-0006 locks a PCHIP of `log(rate)` against distortion, per clip, then the mean. Two facts about the VAE curves make a naive call to that function a bad report.

Depth 2 reconstructs at about 17 to 18 dB. x264's softest CRF on this set is about 30 dB. The depth-2 point sits under the classic ladder. It widens the union, which shrinks the overlap fraction, and it bends the first cubic. The integral's lower end is still the classic floor, so the point is not a shared operating point.

Several clips also turn down at the top of the ladder. A higher bit depth comes back with a lower PSNR or VMAF than depth 8. `sintel_trailer` does this on Wan2.1. ADR-0006 already refuses a non-monotonic curve rather than inventing a NaN. The VAE ceiling is also below the classic ceiling on most clips. Extending the VAE curve up to x264's best PSNR would invent a rate the sweep never measured.

## Decision

BD-rate of each VAE against libx264, libx265, and libsvtav1 uses `bd_rate` from ADR-0006. No second interpolant.

Depth 2 is omitted from the samples that enter the integral. It stays on the rate-distortion figure.

After that omission, a clip is defined only when both curves have at least four points, rate and quality are strictly increasing together, and the quality ranges overlap. Otherwise the clip is undefined and contributes no percent:

- `too_few_points`, fewer than four samples remain.
- `nonmonotonic`, quality falls or stalls as rate rises, or two samples share a rate.
- `no_overlap`, the quality ranges do not meet.

The integral stops at the shared quality range. Qualities past the last measured sample are left blank. A defined pair whose overlap is under 75% stays in the mean and is flagged, as in ADR-0006. The mean is the mean of the defined clips. The overlap printed next to it is the mean of those same clips.

`bpp` is the zstd file from ADR-0010. `entropy_bits` is not added into the rate.

## Consequences

The percents are large and positive, and on PSNR-Y every defined pair against x264 is under the 75% overlap line. The number describes the band where the VAE's ceiling meets the lower part of the classic curve. Quoting it without the overlap, or as a claim about the top of the classic curve, overreads it.

A clip such as Hunyuan on `sintel_trailer` is missing from the mean because the ladder is not monotonic. The gap between its depth-12 PSNR and the unquantized reconstruction stays a hypothesis about clipping at `[-4, 4]`. The latents are not in the repo.

## Alternatives

**Keep depth 2 in the integral.** The shared band does not grow, because the classic curve never goes that low. The cubic on the way into that band does change, and the overlap fraction gets worse for a point that is not a usable picture.

**Drop the one sample that went backwards and integrate the rest.** That hides the reversal. The clip is undefined instead, and the figure still shows the point.

**Extrapolate each VAE up to the anchor's best quality.** That is a rate at a distortion the VAE did not produce. The table leaves that region blank.

**One BD-rate on the pooled mean curve.** ADR-0006 already rejected this. A clip where the curves miss each other would disappear into the average.
