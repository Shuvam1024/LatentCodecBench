# ADR-0003: Pin a static ffmpeg with libvmaf

- Status: Accepted
- Date: 2026-10-07

## Context

PSNR and SSIM are NumPy. VMAF is not: the implementation this benchmark trusts is Netflix's libvmaf, driven by ffmpeg's `libvmaf` filter. The codec baselines in the next step need `libx264`, `libx265`, and `libsvtav1` from the same binary, so encode and metric see one ffmpeg.

Ubuntu 24.04's `ffmpeg` package (the one `apt` installs on GitHub-hosted runners today) is built with libx264, libx265, and libsvtav1. It is not built with libvmaf. It does ship a `vmafmotion` filter, which is only the motion feature, not a VMAF score. Installing `libvmaf-dev` does not turn that package into a libvmaf-enabled ffmpeg.

## Decision

CI and the documented local setup install one pinned static build, and put it on `PATH` ahead of any distro ffmpeg.

- Build: BtbN FFmpeg-Builds, GPL, `N-127222-g151814650f` (autobuild 2026-10-06).
- URL: `https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-10-06-13-06/ffmpeg-N-127222-g151814650f-linux64-gpl.tar.xz`
- SHA256: `0223797ce0ece4b58a11af30cd69318bcdaf1fad395dbaeb163e1abc588706a0`

`scripts/install-ffmpeg.sh` downloads that archive, refuses it if the checksum does not match, and extracts it. The workflow then checks that `libx264`, `libx265`, `libsvtav1`, and `libvmaf` are actually present. `p7zip-full` still comes from apt, because the static build is ffmpeg only and the clip fetcher needs `7z`.

`LATENTCODECBENCH_FFMPEG` overrides the binary the library calls, so a developer can point at this build without hiding the distro one from the rest of the machine.

## Consequences

VMAF tests run in CI instead of skipping. A future runner image that changes the apt build cannot silently drop an encoder.

The pin is a moving upstream if we never bump it. Bumping is a one-line change in the script plus a new checksum, and it should be its own commit because VMAF numbers can shift a little between libvmaf versions.

The binary is GPL. The benchmark invokes it as a subprocess. It is not linked into the Python package, and it is not committed.

## Alternatives

**Stay on apt and skip VMAF in CI.** The metric the plan names would be untested on every pull request. A broken log parser would show up only on a laptop that happened to have libvmaf.

**Build ffmpeg from source in the workflow.** Reproducible, and slow: a from-source GPL build with libvmaf and SVT-AV1 is a long compile on every cold cache. The static tarball is the same components, already built, checked by hash.

**Call the `vmaf` Python package.** Another implementation to version, and it still needs the model files. libvmaf via ffmpeg is the tool codec papers mean by "VMAF".
