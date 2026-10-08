# ADR-0003: Pin a static ffmpeg with libvmaf

- Status: Accepted
- Date: 2026-10-07
- Revised: 2026-10-08

## Context

PSNR and SSIM are NumPy. VMAF is not: the implementation this benchmark trusts is Netflix's libvmaf, driven by ffmpeg's `libvmaf` filter. The codec baselines need `libx264`, `libx265`, and `libsvtav1` from the same binary, so encode and metric see one ffmpeg.

Ubuntu 24.04's `ffmpeg` package (the one `apt` installs on GitHub-hosted runners today) is built with libx264, libx265, and libsvtav1. It is not built with libvmaf. It does ship a `vmafmotion` filter, which is only the motion feature, not a VMAF score. Installing `libvmaf-dev` does not turn that package into a libvmaf-enabled ffmpeg.

BtbN's FFmpeg-Builds publishes a static GPL binary that has all four. The retention policy on that project is short for daily tags: the last 14 daily builds are kept, and the last build of each month is kept for two years. The floating `latest` tag moves every day. A daily URL such as the 2026-10-06 autobuild 404s about two weeks later, and CI breaks on a file that no longer exists.

## Decision

CI and the documented local setup install one pinned static build, and put it on `PATH` ahead of any distro ffmpeg.

The pin is the last build of September 2026 (`autobuild-2026-09-30-13-08`), the newest month-end autobuild. Month-end is the artifact BtbN keeps for two years. Inside that release, the linux64-gpl build from the newest release branch is preferred to the master build. Release-branch filenames start with `ffmpeg-n` (here, n9.0). Master filenames start with `ffmpeg-N-`.

- Build: `n9.0.2-17-g2a571b6068`, linux64-gpl, release branch 9.0, dated 2026-09-30.
- URL: `https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-09-30-13-08/ffmpeg-n9.0.2-17-g2a571b6068-linux64-gpl-9.0.tar.xz`
- SHA256: `68ee646831adaae2495618346f3bba94ff207ff83bbd34d643e7004730d66269`

That binary has `libx264`, `libx265`, `libsvtav1`, and `libvmaf`. `scripts/install-ffmpeg.sh` downloads the archive, refuses it if the checksum does not match, and extracts it. The workflow checks the four features after install. `p7zip-full` still comes from apt, because the static build is ffmpeg only and the clip fetcher needs `7z`.

GitHub Actions caches the extracted tree with `actions/cache`. The cache key is the SHA256 above, so a normal pull request does not download the archive again. A changed checksum is a cache miss, and the next job fills the new entry.

`LATENTCODECBENCH_FFMPEG` overrides the binary the library calls, so a developer can point at this build without hiding the distro one from the rest of the machine.

### How to bump the pin

1. Open the newest month-end release on BtbN/FFmpeg-Builds. Those tags look like `autobuild-YYYY-MM-DD` and fall on the last day of a month. Skip the daily tags and the floating `latest` tag.
2. From that release, take `linux64-gpl`. Prefer a file whose name starts with `ffmpeg-n` (a release branch such as n8 or n9) over one whose name starts with `ffmpeg-N-` (master).
3. Download it, record `sha256sum`, and confirm `libx264`, `libx265`, `libsvtav1`, and `libvmaf` are present.
4. Put the URL and checksum in `scripts/install-ffmpeg.sh`, and put the same checksum in the `actions/cache` key in `.github/workflows/ci.yml`. The key and the script have to move together. A stale key restores the previous tree and skips the download.
5. Make it its own commit. Encoder and libvmaf changes can move bits and VMAF. Spot-check a row already in `docs/results/classic-baselines.csv` before trusting the old table. If the bits or the VMAF of that row change, re-run the sweep.

## Consequences

VMAF tests run in CI. A future runner image that changes the apt build cannot silently drop an encoder. The URL stays valid until September 2028 under the published retention rule.

The binary is GPL. The benchmark invokes it as a subprocess. It is not linked into the Python package, and it is not committed.

The September 2026 n9.0 build produced the same packet sizes and the same VMAF as the October 6 master build on the rows that were re-encoded while moving this pin (`crowd_run` at x264 CRF 28, x265 CRF 28, and SVT-AV1 CRF 36; `sintel_trailer` at x264 CRF 23, x265 CRF 23, and SVT-AV1 CRF 28). The checked-in results table was left as it was.

## Alternatives

**Stay on apt and skip VMAF in CI.** The metric the plan names would be untested on every pull request. A broken log parser would show up only on a laptop that happened to have libvmaf.

**Build ffmpeg from source in the workflow.** Reproducible, and slow: a from-source GPL build with libvmaf and SVT-AV1 is a long compile on every cold cache. The static tarball is the same components, already built, checked by hash.

**Pin a daily build, or the floating `latest` tag.** `latest` changes under the same URL, so a green run today and a green run next week are not the same binary. A daily tag 404s once it ages out of the 14-build window.

**Call the `vmaf` Python package.** Another implementation to version, and it still needs the model files. libvmaf via ffmpeg is the tool codec papers mean by "VMAF".
