# ADR-0005: Classic codec baselines

- Status: Accepted
- Date: 2026-10-07

## Context

The VAE curves need something familiar on the same axes. That something is x264, x265, and SVT-AV1, each at several rates, with the rate measured the same way and the timing measured on one thread. CRF is the knob ffmpeg exposes for "quality" on all three. The knob is not a shared scale: x264 CRF 28 and SVT-AV1 CRF 28 do not mean the same distortion or the same rate.

A preset changes speed and, a little, the rate at a given CRF. An unfixed preset makes encode time a property of whoever last edited the command.

## Decision

Six CRF points per codec, high quality down to low:

| Codec | CRF points | Preset |
| --- | --- | --- |
| libx264 | 18, 23, 28, 33, 38, 43 | `medium` |
| libx265 | 18, 23, 28, 33, 38, 43 | `medium` |
| libsvtav1 | 20, 28, 36, 44, 52, 60 | `6` |

SVT-AV1's CRF runs 0–63, so its ladder is shifted up relative to x264. The comparison is the rate-distortion curve, not the CRF label.

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

**File size as the rate.** The moov atom is a few kilobytes. At CRF 43 that is a visible fraction of the file. Packet sizes are the bits the codec spent.
