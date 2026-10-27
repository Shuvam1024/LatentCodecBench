# Changelog

## Unreleased

- Step 3.3 is measured. The feature table is `docs/results/latent-probe-features.csv.gz`, 1360 rows, uncompressed SHA256 `c759c29aad3a04dedaad52b795524a96aee00a9d266c6365d6bfadb60f0ace0f`. Leave-clip-out ridge reads the study rung from the latent. Pooled across codecs, accuracy is 0.703 (95% CI [0.661, 0.743]) against chance 0.333 and a pixel baseline of 0.471 (95% CI [0.420, 0.520]). Per-VAE latent accuracies are 0.663, 0.725, 0.725, and 0.699 for Wan2.1, Hunyuan, LTX, and Wan2.2. Note 14 and the report have the intervals and the caveats.
- Phase 4 stays an open direction: a lightweight decoder-side post-filter to recover quality. The design is not chosen.

## 1.0.0 — 2026-10-23

First public release of the benchmark. The Python package version stays 0.0.1; this tag is the repository release.

- Classic baselines on 34 clips. Against x264, x265 BD-rate is −24.66% on PSNR-Y and −20.44% on VMAF. SVT-AV1 is −41.99% on PSNR-Y and −44.86% on VMAF.
- Four VAEs as codecs, unquantized reconstruction on the same 34 clips. Mean PSNR-Y is 34.57 dB (Wan2.1), 35.11 dB (Hunyuan), 32.30 dB (LTX), and 35.35 dB (Wan2.2). PSNR-Y BD-rate against x264, on the clips where the curves overlap, is +824.24% (Wan2.1, 33 clips), +725.09% (Hunyuan, 32), +397.92% (LTX, 26), and +628.61% (Wan2.2, 33).
- Source compression moves the latent. Relative L2 rises with CRF on all 408 curves in the 1224-row propagation table.
- A linear probe of source QP or CRF, and a lightweight decoder-side post-filter, are future work. The post-filter design is not chosen.
