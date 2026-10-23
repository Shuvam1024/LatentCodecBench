# LatentCodecBench

## Question

How do the VAEs inside open video generators compare with x264, x265, and SVT-AV1 on rate-distortion, and does compression in the source still show up in the latent?

## What was measured

The clip is the canonical 832×480, 33-frame y4m. Classic codecs use the Phase 1 CRF lists, presets, and one thread. The VAEs are scored twice: an unquantized reconstruction, then a uniform quantizer at several bit depths whose rate is the zstd file. Depth 2 stays out of the BD-rate integral. Phase 3 re-encodes each clip at three rungs of each codec and compares the clean latent with the degraded one.

The anchors, the overlap rule, and the refusal to extrapolate are [ADR-0006](decisions/0006-rd-curves-and-bd-rate.md) and [ADR-0011](decisions/0011-vae-bd-rate.md). The propagation definitions are [ADR-0012](decisions/0012-artifact-propagation.md).

## Results

Reconstruction means use 34 clips per VAE. The interval is a 95% Student-t interval for that mean.

| VAE | Clips | Mean PSNR-Y | 95% CI | Mean VMAF | 95% CI |
| --- | ---: | ---: | --- | ---: | --- |
| wan2_1 | 34 | 34.57 dB | [32.24 dB, 36.90 dB] | 86.40 | [83.17, 89.62] |
| hunyuan | 34 | 35.11 dB | [32.90 dB, 37.32 dB] | 87.63 | [84.80, 90.46] |
| ltx | 34 | 32.30 dB | [30.10 dB, 34.50 dB] | 74.56 | [68.27, 80.85] |
| wan2_2 | 34 | 35.35 dB | [32.70 dB, 37.99 dB] | 87.13 | [83.56, 90.70] |

BD-rate intervals use only the clips whose rate is defined. The anchor is x264.

| Codec | Metric | Clips in the mean | Mean BD-rate | 95% CI |
| --- | --- | ---: | ---: | --- |
| x265 | PSNR-Y | 34 | -24.66% | [-29.58%, -19.73%] |
| SVT-AV1 | PSNR-Y | 34 | -41.99% | [-46.48%, -37.50%] |
| x265 | VMAF | 34 | -20.44% | [-26.32%, -14.55%] |
| SVT-AV1 | VMAF | 34 | -44.86% | [-49.12%, -40.59%] |
| wan2_1 | PSNR-Y | 33 | +824.24% | [+641.64%, +1006.83%] |
| wan2_1 | VMAF | 33 | +643.08% | [+505.02%, +781.14%] |
| hunyuan | PSNR-Y | 32 | +725.09% | [+577.40%, +872.79%] |
| hunyuan | VMAF | 30 | +583.16% | [+450.48%, +715.85%] |
| ltx | PSNR-Y | 26 | +397.92% | [+226.71%, +569.13%] |
| ltx | VMAF | 25 | +205.95% | [+156.04%, +255.85%] |
| wan2_2 | PSNR-Y | 33 | +628.61% | [+514.24%, +742.97%] |
| wan2_2 | VMAF | 33 | +568.29% | [+443.13%, +693.45%] |

Degraded-source PSNR across VAEs: x264 matches on every clip; x265 differs on 17 clips by up to 0.057 dB; SVT-AV1 differs on 18 clips by up to 1.345 dB.

The same means without intervals are the headline in the README. Note 09 is the long BD-rate table, including x265 and SVT-AV1 as anchors, and it prints the median beside the mean. Note 10 reads the propagation rows. Note 11 is the training-data filter.

Every classic interval in the table lies below zero. Every VAE interval lies above zero. The VAE intervals are wide, which is why the median in note 09 sits below the mean: a hard clip pulls the average. On VMAF, LTX's reconstruction interval does not meet the other three. On PSNR-Y the four reconstruction intervals overlap, so that ordering of means is not a separation.

## Limitations

The interval treats the clips as a sample from a larger population of the same kind. The list is designed, not drawn. Read the interval as clip-to-clip spread of the mean. Do not read it as a claim about all video, and do not read it as the error of one forward. Repeating a forward in the same dtype prints the same score. The width is the clips disagreeing.

Undefined BD-rate clips are outside the interval. Note 09 counts them. Pairs under 75% overlap stay in the mean, and the README headline counts those pairs.

The rate in the integral is the zstd file on a uniform quantizer. It is not a trained entropy coder. A coder that spent fewer bits would move a curve left. It would not raise the reconstruction the unquantized sweep already measured. LPIPS was not scored.

Hunyuan ran with spatial tiling. The VAE tables are bfloat16 on an L4.

The degraded-source mismatch is the last line of the results. A cross-VAE comparison on a codec that does not match carries that source gap. x264 is the column that matches.

Clips whose source was already compressed stay in the means. The latent tensors are not in the repo, so a clipping account of one clip's quantization floor stays a hypothesis. Note 10 records that hypothesis.

## Future work

A linear probe that predicts source QP or CRF from the latent has not been run. It would test whether the source quantizer is linearly readable, a stronger claim than the latent distance in the results above. It needs a GPU.

A lightweight decoder-side post-filter to recover quality is an open direction. The design is still to be decided. It has not been run. It would test whether a cheap correction after the decoder can raise the reconstruction this report already measured.

## Reproduce

From the repository root, `uv run lcb reproduce` rebuilds the tables in this report from the committed CSVs and fails if this page has drifted. It does not re-run a codec or a VAE.
