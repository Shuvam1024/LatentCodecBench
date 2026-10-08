# ADR-0005: Classic codec baselines

- Status: Accepted
- Date: 2026-10-07
- Revised: 2026-10-08

## Context

The VAE curves need something familiar on the same axes. That something is x264, x265, and SVT-AV1, each at several rates, with the rate measured the same way and the timing measured on one thread. CRF is the knob ffmpeg exposes for "quality" on all three. The knob is not a shared scale: x264 CRF 28 and SVT-AV1 CRF 28 do not mean the same distortion or the same rate.

A preset changes speed and, a little, the rate at a given CRF. An unfixed preset makes encode time a property of whoever last edited the command.

## Decision

Six CRF points per codec. The ends of each ladder are the quality range the three codecs share. Interior points are an even step across that span, so the curve has six samples and no seventh encode.

| Codec | CRF points | Preset |
| --- | --- | --- |
| libx264 | 18, 22, 26, 30, 34, 38 | `medium` |
| libx265 | 18, 22, 26, 30, 34, 38 | `medium` |
| libsvtav1 | 28, 35, 42, 49, 56, 63 | `6` |

The first ladders were x264 and x265 at 18, 23, 28, 33, 38, 43 and SVT-AV1 at 20, 28, 36, 44, 52, 60. On the 34-clip set that SVT-AV1 ladder bottomed out near 33.2 dB PSNR-Y and VMAF 71, while x264 CRF 43 reached about 27.7 dB and VMAF 34. Every clip had some shared range. The shared fraction — intersection of the two metric ranges divided by their union, which is what the `bjontegaard` package compares with 75% — was under 75% on 32 of 34 clips for PSNR-Y and on 32 of 34 for VMAF. A BD-rate over that sliver does not describe the low-rate end of x264.

SVT-AV1's CRF stops at 63. Encoding the whole set at CRF 63, on the pinned ffmpeg, landed at about 30.6 dB PSNR-Y and VMAF 56 on average. x264 CRF 38 is about 30.2 dB and VMAF 54. CRF 36 and CRF 40 were encoded on the same 34 clips. Ending the anchor at 36 leaves its worst point cleaner than SVT-AV1 on the hard clips, so the union grows upward. Ending at 40 falls past SVT-AV1 again. Dropping x264 and x265's CRF 43, and starting SVT-AV1 at 28 rather than 20, was the endpoint pair with the fewest (clip, metric) pairs under 75% among the floors that were measured. SVT-AV1 CRF 20 sits above x264 CRF 18 (about 43.0 dB against 41.7 dB), so that top point widens the union without widening the intersection.

x264 and x265 keep one ladder. Their CRF scales match, and splitting them would make the anchor a different experiment from the HEVC curve. The step is 4 from 18 to 38, and 7 from 28 to 63. Moving an interior point does not change the overlap fraction. It only changes how densely the integral is sampled.

Easy clips still refuse to fall apart under SVT-AV1 at CRF 63. A talking-head clip can sit near VMAF 80 there while x264 CRF 38 is near VMAF 64. Those pairs stay under 75%. The results table stores the fraction and flags them. They are not dropped, and they are not described as a full overlap.

The comparison remains the rate-distortion curve. CRF numbers are not a shared scale.

Presets were timed on `crowd_run` (832×480, 33 frames, one thread) with the pinned ffmpeg:

- x264 `medium`, CRF 28: 0.7 s
- x265 `medium`, CRF 28: 1.5 s
- SVT-AV1 preset 8, CRF 35: 1.2 s
- SVT-AV1 preset 6, CRF 35: 2.0 s

x265 `medium` is cheap enough at this resolution that a faster preset is not worth the quality it gives up. SVT-AV1 preset 6 is the slower of the two practical CPU presets and the one the sweep uses. Tests pass `ultrafast` or SVT preset `12` so CI does not spend benchmark time.

Threads are pinned:

- x264: `-threads 1 -x264-params sliced-threads=0`
- x265: `-x265-params pools=1:frame-threads=1`
- SVT-AV1: `-svtav1-params lp=1:tune=0`
- decode: `-threads 1`, to null

`tune=0` is SVT-AV1's visual-quality tune. On this ffmpeg build the wrapper's default tune is PSNR (the log says so). x264 and x265 are not passed `-tune psnr` either. A PSNR tune would climb the PSNR axis and is the wrong default when VMAF is also reported.

Rate is video-packet bits from ffprobe, not the mp4 file size. Bits per pixel divides that by width × height × frames. Kilobits per second uses the source frame rate. Decode time is the decoder writing nothing (`-f null`). A caller that needs pixels decodes a second time; that write is not part of `decode_seconds`.

## Consequences

A 34-clip sweep is 34 × 18 encodes. At the timings above that is on the order of ten minutes of encode, before VMAF. The full set fits on this machine.

CRF numbers must not be quoted as if they were comparable across codecs. The results table stores CRF so a row can be repeated, and the plot uses bits per pixel.

Encode time is wall clock of one ffmpeg process. Another job on the same CPU will move it. The thread flags make the encoder itself not the thing that changes.

## Alternatives

**Fixed bitrates (ABR) instead of CRF.** A shared kbps ladder looks cleaner in a table and fights the encoder: each codec hits the target differently, and short 33-frame clips give the rate controller very little time. CRF lets each codec do what it does, and bpp makes the x-axis honest.

**`-tune psnr` everywhere.** Higher PSNR at the same CRF, and a worse VMAF, which is then an artifact of the tune rather than of the codec. Leave the tune off, except for the SVT default that had to be turned away from PSNR.

**SVT-AV1 preset 8, or x265 `faster`.** Both finished the sample clip quicker. The slower settings still finish a clip in about two seconds, so the sweep keeps the stronger preset.

**File size as the rate.** The moov atom is a few kilobytes. At CRF 38 that is a visible fraction of the file. Packet sizes are the bits the codec spent.

**Leave the first ladders in place and only warn about overlap.** The warning would have been honest, and the integral would still have been taken over a band that misses the low-rate part of x264. Moving the samples is what makes the number about the same pictures.
