# Changelog

## 1.0.0 — 2026-10-23

First public release of the benchmark. The Python package version stays 0.0.1; this tag is the repository release.

- Classic baselines on 34 clips. Against x264, x265 BD-rate is −24.66% on PSNR-Y and −20.44% on VMAF. SVT-AV1 is −41.99% on PSNR-Y and −44.86% on VMAF.
- Four VAEs as codecs, unquantized reconstruction on the same 34 clips. Mean PSNR-Y is 34.57 dB (Wan2.1), 35.11 dB (Hunyuan), 32.30 dB (LTX), and 35.35 dB (Wan2.2). PSNR-Y BD-rate against x264, on the clips where the curves overlap, is +824.24% (Wan2.1, 33 clips), +725.09% (Hunyuan, 32), +397.92% (LTX, 26), and +628.61% (Wan2.2, 33).
- Source compression moves the latent. Relative L2 rises with CRF on all 408 curves in the 1224-row propagation table.
- A linear probe of source QP or CRF, and a lightweight decoder-side post-filter, are future work. The post-filter design is not chosen.
