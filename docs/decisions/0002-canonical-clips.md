# ADR-0002: Canonical clip format and how many clips

- Status: Accepted
- Date: 2026-10-07

## Context

Every later stage, classic codecs and video VAEs, has to see the same pixels. The VAEs also constrain the shape. Wan2.1 and HunyuanVideo use a temporal factor of 4, Wan2.2's high-compression VAE uses 4 in time and 16 in space, and LTX-Video uses 8 in time and 32 in space. A causal video VAE encodes the first frame on its own and then groups of that temporal factor, so a legal length is `4k+1` or `8k+1`.

The sources are not that shape. Xiph derf HD files are 1080p or 720p y4m, often about a gigabyte because they are raw. UVG sequences are 4K 10-bit y4m inside multi-gigabyte 7z archives. A few user-generated clips are short, already-compressed camera videos with a Creative Commons license.

The plan asked for about 50 clips. This machine can download and encode more than that, but only if we do not keep the full raw files.

## Decision

Canonical clip: **832×480, 33 frames, 8-bit 4:2:0, stored as y4m**.

- 33 = `8*4+1` = `4*8+1`, so it satisfies both frame rules.
- 832 and 480 are divisible by 32, so they satisfy spatial factors 8, 16, and 32.
- y4m is raw samples plus a header. A lossless codec such as FFV1 would also be lossless, but a second codec is another place for an off-by-one. y4m is the bytes.
- Scaling is lanczos, cover-then-center-crop. The picture is not stretched and not letterboxed. The source stays in YUV; preprocess does not round-trip through RGB. The output is tagged BT.709, because the sources are HD even though the stored height is 480.

The manifest records a SHA256 of the bytes `lcb data fetch` actually downloads. For a derf y4m that is hundreds of megabytes of frames we will never use, that is an HTTP byte range: the header plus 33 frames. The fetcher requires HTTP 206 and deletes the file if the digest or the length does not match. For a UVG 7z and for the short user-generated files, the digest is of the whole file. UVG's digest is the one published next to the mirror. Preprocess streams the y4m member out of the 7z and stops after 33 frames, then converts 10-bit to 8-bit 4:2:0.

The set is 34 clips: 27 from the Xiph derf HD collection (natural scenes, three conference sequences, and the Sintel trailer), 3 from UVG (ShakeNDry, Bosphorus, HoneyBee), and 4 Creative Commons camera clips. That is inside the planned 30 to 50. It spans high motion, water, foliage, scene cuts, screen-like conference content, animation, portrait phone video, and a 4K handheld landscape. The other UVG archives are the same kind of multi-gigabyte 7z; three sequences already cover the slow-texture, water, and camera-motion end of that set, and adding the rest would mostly repeat those categories at a large download cost.

User-generated clips are already compressed (Theora or VP9). The canonical y4m inherits that. Derf and UVG do not. `DATA.md` says which is which, because Phase 3 will care.

Video bytes stay in `data/`, which is gitignored.

## Consequences

Codecs and VAEs can share one directory of y4m files. A test can build a tiny source and check that preprocess emits 832×480×33 without downloading anything.

The derf digest is not the digest of the whole published file. Re-fetching the same range is reproducible. Re-hashing the full multi-gigabyte object is a different check, and the manifest does not claim to be that check.

Center-crop throws away the edges of a portrait phone video. That is the trade for a single shape.

UVG preprocess needs `7z` on `PATH`. CI installs `p7zip-full` for that.

## Alternatives

**Full-file downloads of every derf y4m.** Honest in the simplest sense, and tens of gigabytes of frames the benchmark never reads. The byte range is the same pixels, checked.

**FFV1 in Matroska.** Lossless and smaller than y4m, and another decoder in the path. y4m keeps the canonical file as samples.

**50 clips, including every UVG sequence.** The extra UVG files are the same 4K 10-bit material. The diversity in this set comes from the derf scenes and the phone clips, not from a fourth 4K boat.

**Padding short clips up to 33 frames.** Invented frames would show up as freezes in the temporal metrics. Clips shorter than 33 frames were left out.
