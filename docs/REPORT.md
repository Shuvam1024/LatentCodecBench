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

## Source rung

The linear probe in [ADR-0013](decisions/0013-linear-probe.md) is fit on `docs/results/latent-probe-features.csv.gz` (1360 rows, run `20261024T182512Z-5112c2552a5c-bcc52c3cc55f`, uncompressed SHA256 `c759c29aad3a04dedaad52b795524a96aee00a9d266c6365d6bfadb60f0ace0f`). Each model is leave-clip-out ridge inside one VAE and one codec. The pooled rows average those clip scores. They do not fit one CRF across codecs. Chance on the rung task is one third. The pixel model uses the six luma numbers. The tables below round to three decimals. `docs/results/latent-probe.csv` stores the full float, and note 14 has the per-codec MAE beside the accuracy.

On every per-codec classification, and on every pooled row, the latent interval sits above the pixel interval, and both sit above chance. Pooled latent accuracy is 0.663 for Wan2.1, 0.725 for Hunyuan, 0.725 for LTX, 0.699 for Wan2.2, and 0.703 across VAEs.

Classification accuracy pooled across codecs. The model was still fit inside one codec. A clip's score is the mean of its per-codec accuracies, and the `all` row is that mean taken across VAEs as well. The interval resamples clips. Chance is the per-codec majority rung, pooled the same way.

| VAE | Source | Model | Clips | Accuracy | 95% CI |
| --- | --- | --- | ---: | ---: | --- |
| wan2_1 | pooled | latent | 34 | 0.663 | [0.611, 0.719] |
| wan2_1 | pooled | pixel | 34 | 0.474 | [0.418, 0.526] |
| wan2_1 | pooled | chance | 34 | 0.333 | [0.333, 0.333] |
| hunyuan | pooled | latent | 34 | 0.725 | [0.657, 0.794] |
| hunyuan | pooled | pixel | 34 | 0.474 | [0.425, 0.523] |
| hunyuan | pooled | chance | 34 | 0.333 | [0.333, 0.333] |
| ltx | pooled | latent | 34 | 0.725 | [0.676, 0.775] |
| ltx | pooled | pixel | 34 | 0.461 | [0.408, 0.510] |
| ltx | pooled | chance | 34 | 0.333 | [0.333, 0.333] |
| wan2_2 | pooled | latent | 34 | 0.699 | [0.641, 0.755] |
| wan2_2 | pooled | pixel | 34 | 0.474 | [0.425, 0.523] |
| wan2_2 | pooled | chance | 34 | 0.333 | [0.333, 0.333] |
| all | pooled | latent | 34 | 0.703 | [0.661, 0.743] |
| all | pooled | pixel | 34 | 0.471 | [0.420, 0.520] |
| all | pooled | chance | 34 | 0.333 | [0.333, 0.333] |

Clean versus degraded, one VAE at a time, on the same ridge and the same leave-clip-out split. Chance is the majority training class. A tie takes the clean class. The `all` row averages clip scores across VAEs.

| VAE | Source | Model | Clips | Accuracy | 95% CI |
| --- | --- | --- | ---: | ---: | --- |
| wan2_1 | clean-degraded | latent | 34 | 0.982 | [0.962, 0.997] |
| wan2_1 | clean-degraded | pixel | 34 | 0.900 | [0.900, 0.900] |
| wan2_1 | clean-degraded | chance | 34 | 0.900 | [0.900, 0.900] |
| hunyuan | clean-degraded | latent | 34 | 0.982 | [0.962, 0.997] |
| hunyuan | clean-degraded | pixel | 34 | 0.900 | [0.900, 0.900] |
| hunyuan | clean-degraded | chance | 34 | 0.900 | [0.900, 0.900] |
| ltx | clean-degraded | latent | 34 | 0.862 | [0.785, 0.924] |
| ltx | clean-degraded | pixel | 34 | 0.900 | [0.900, 0.900] |
| ltx | clean-degraded | chance | 34 | 0.900 | [0.900, 0.900] |
| wan2_2 | clean-degraded | latent | 34 | 0.953 | [0.903, 0.991] |
| wan2_2 | clean-degraded | pixel | 34 | 0.900 | [0.900, 0.900] |
| wan2_2 | clean-degraded | chance | 34 | 0.900 | [0.900, 0.900] |
| all | clean-degraded | latent | 34 | 0.945 | [0.920, 0.966] |
| all | clean-degraded | pixel | 34 | 0.900 | [0.900, 0.900] |
| all | clean-degraded | chance | 34 | 0.900 | [0.900, 0.900] |

Codec identity on the degraded rows only, three codecs, same ridge and same leave-clip-out split. The label is the codec, not the CRF. Chance is the majority training codec, and a tie takes the first codec on the study ladder. The `all` row averages clip scores across VAEs.

| VAE | Source | Model | Clips | Accuracy | 95% CI |
| --- | --- | --- | ---: | ---: | --- |
| wan2_1 | codec-id | latent | 34 | 0.536 | [0.480, 0.601] |
| wan2_1 | codec-id | pixel | 34 | 0.418 | [0.386, 0.454] |
| wan2_1 | codec-id | chance | 34 | 0.333 | [0.333, 0.333] |
| hunyuan | codec-id | latent | 34 | 0.660 | [0.595, 0.725] |
| hunyuan | codec-id | pixel | 34 | 0.399 | [0.366, 0.435] |
| hunyuan | codec-id | chance | 34 | 0.333 | [0.333, 0.333] |
| ltx | codec-id | latent | 34 | 0.529 | [0.474, 0.582] |
| ltx | codec-id | pixel | 34 | 0.431 | [0.392, 0.471] |
| ltx | codec-id | chance | 34 | 0.333 | [0.333, 0.333] |
| wan2_2 | codec-id | latent | 34 | 0.552 | [0.493, 0.611] |
| wan2_2 | codec-id | pixel | 34 | 0.399 | [0.366, 0.435] |
| wan2_2 | codec-id | chance | 34 | 0.333 | [0.333, 0.333] |
| all | codec-id | latent | 34 | 0.569 | [0.529, 0.609] |
| all | codec-id | pixel | 34 | 0.412 | [0.382, 0.444] |
| all | codec-id | chance | 34 | 0.333 | [0.333, 0.333] |

`source_sha256` disagrees across VAEs on 75 of 306 degraded (clip, codec, CRF) points (x265 36, SVT-AV1 39). x264 agrees. The rung model is fit inside one codec, so a within-codec comparison does not mix those files. A cross-VAE comparison at one of those points does.

Clean versus degraded is a different question. Nine of the ten sources are degraded, so chance is 0.900 and a high accuracy can be the majority class. The pixel model ties that chance. Latent accuracy is above it for Wan2.1, Hunyuan, Wan2.2, and the four VAEs together (0.945). LTX is not: 0.862, and that interval meets 0.900.

Codec identity on the degraded rows is above chance. Pooled latent accuracy is 0.569 against 0.333. That is below the rung readout. Hunyuan is the VAE whose rung interval and codec-identity interval meet. The other three separate, with the rung higher. Point MAE in the per-codec table is lower for the latent model than for the pixel model on every group. Those MAE intervals meet for Wan2.1 on SVT-AV1 and for Hunyuan on SVT-AV1. The other ten groups separate, with the latent MAE lower.

![Leave-clip-out probe of source CRF](results/latent-probe.png)

## Future work

A lightweight decoder-side post-filter to recover quality is an open direction. The design is still to be decided. It has not been run. It would test whether a cheap correction after the decoder can raise the reconstruction this report already measured.

## Reproduce

From the repository root, `uv run lcb reproduce` rebuilds the tables in this report from the committed CSVs and fails if this page has drifted. It does not re-run a codec or a VAE.
