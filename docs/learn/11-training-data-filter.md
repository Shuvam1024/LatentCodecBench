# When would you filter compressed training video?

Open video models are often trained on video that x264, x265, or AV1 has already compressed. A filter is the cheap response: drop a clip when the source looks too damaged. Whether that drop buys anything depends on which loss the VAE is trained with. Note 10 measured the leak. This note is the filter those rows support. [ADR-0012](../decisions/0012-artifact-propagation.md).

## Two losses, two answers

A reconstruction loss scores pixels. Take the degraded source, encode and decode it with the VAE, and score the result against the clean source. Subtract the PSNR of the clean source run through the same VAE. That difference is ΔPSNR. Positive means the compressed source reconstructed higher than the clean source did, so throwing the clip away does not raise pixel PSNR.

A latent model scores the code. Relative L2 is how far the degraded posterior mean sits from the clean one, divided by the norm of the clean one. A ΔPSNR near zero does not mean that code stayed put. The table keeps both numbers, because a pixel filter and a latent filter are different decisions.

The three rungs are the study ladder: the ends of each Phase 1 CRF list and the middle step. CRF is not a shared scale. The middle rung of x264 and the middle rung of SVT-AV1 are the same kind of point on their own ladders, not the same quality. Copying an x264 CRF onto AV1 is not the rule this table supports.

## What the rows support

1224 rows, 34 clips, 4 VAEs, 9 degraded sources. Median ΔPSNR is the median over clips of reconstruction PSNR against the clean source, minus that clip's clean-source reconstruction PSNR. Positive means the compressed source scored higher.

| VAE | Source | CRF | Median ΔPSNR | IQR | Clips above clean recon | Median latent rel. L2 |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| wan2_1 | x264 | 18 | +0.111 dB | [-0.072, +0.227] | 24 of 34 | 0.138 |
| wan2_1 | x264 | 30 | -1.201 dB | [-2.349, -0.385] | 4 of 34 | 0.361 |
| wan2_1 | x264 | 38 | -4.242 dB | [-6.019, -2.629] | 0 of 34 | 0.527 |
| wan2_1 | x265 | 18 | +0.194 dB | [+0.083, +0.281] | 31 of 34 | 0.148 |
| wan2_1 | x265 | 30 | -0.720 dB | [-1.652, -0.022] | 8 of 34 | 0.320 |
| wan2_1 | x265 | 38 | -3.549 dB | [-5.127, -1.781] | 0 of 34 | 0.454 |
| wan2_1 | SVT-AV1 | 28 | +0.111 dB | [+0.013, +0.202] | 26 of 34 | 0.185 |
| wan2_1 | SVT-AV1 | 49 | -0.398 dB | [-1.154, +0.042] | 11 of 34 | 0.312 |
| wan2_1 | SVT-AV1 | 63 | -4.022 dB | [-5.245, -2.358] | 0 of 34 | 0.511 |
| hunyuan | x264 | 18 | +0.016 dB | [-0.246, +0.121] | 19 of 34 | 0.099 |
| hunyuan | x264 | 30 | -1.719 dB | [-2.663, -0.865] | 2 of 34 | 0.219 |
| hunyuan | x264 | 38 | -4.955 dB | [-6.560, -3.522] | 0 of 34 | 0.324 |
| hunyuan | x265 | 18 | +0.078 dB | [-0.081, +0.167] | 23 of 34 | 0.098 |
| hunyuan | x265 | 30 | -1.305 dB | [-2.052, -0.466] | 2 of 34 | 0.191 |
| hunyuan | x265 | 38 | -4.369 dB | [-5.485, -2.581] | 0 of 34 | 0.275 |
| hunyuan | SVT-AV1 | 28 | -0.027 dB | [-0.286, +0.099] | 16 of 34 | 0.120 |
| hunyuan | SVT-AV1 | 49 | -0.955 dB | [-1.751, -0.248] | 4 of 34 | 0.197 |
| hunyuan | SVT-AV1 | 63 | -4.914 dB | [-6.213, -2.924] | 0 of 34 | 0.318 |
| ltx | x264 | 18 | +0.111 dB | [+0.056, +0.213] | 31 of 34 | 0.144 |
| ltx | x264 | 30 | -0.149 dB | [-0.719, +0.044] | 12 of 34 | 0.373 |
| ltx | x264 | 38 | -2.253 dB | [-3.392, -1.120] | 0 of 34 | 0.536 |
| ltx | x265 | 18 | +0.123 dB | [+0.067, +0.237] | 31 of 34 | 0.140 |
| ltx | x265 | 30 | -0.010 dB | [-0.411, +0.059] | 15 of 34 | 0.322 |
| ltx | x265 | 38 | -1.792 dB | [-2.611, -0.864] | 0 of 34 | 0.451 |
| ltx | SVT-AV1 | 28 | +0.108 dB | [+0.040, +0.223] | 31 of 34 | 0.168 |
| ltx | SVT-AV1 | 49 | +0.009 dB | [-0.203, +0.125] | 18 of 34 | 0.311 |
| ltx | SVT-AV1 | 63 | -1.935 dB | [-3.272, -0.986] | 0 of 34 | 0.496 |
| wan2_2 | x264 | 18 | +0.103 dB | [-0.034, +0.227] | 23 of 34 | 0.188 |
| wan2_2 | x264 | 30 | -1.434 dB | [-2.836, -0.560] | 3 of 34 | 0.410 |
| wan2_2 | x264 | 38 | -4.785 dB | [-6.686, -2.954] | 1 of 34 | 0.556 |
| wan2_2 | x265 | 18 | +0.225 dB | [+0.120, +0.310] | 31 of 34 | 0.196 |
| wan2_2 | x265 | 30 | -0.953 dB | [-2.024, -0.148] | 7 of 34 | 0.362 |
| wan2_2 | x265 | 38 | -4.151 dB | [-6.055, -2.077] | 1 of 34 | 0.487 |
| wan2_2 | SVT-AV1 | 28 | +0.173 dB | [+0.005, +0.252] | 25 of 34 | 0.234 |
| wan2_2 | SVT-AV1 | 49 | -0.586 dB | [-1.526, +0.128] | 12 of 34 | 0.365 |
| wan2_2 | SVT-AV1 | 63 | -4.355 dB | [-6.116, -2.536] | 0 of 34 | 0.531 |

The mildest rung is x264 CRF 18, x265 CRF 18, SVT-AV1 CRF 28. Across 12 groups the median ΔPSNR runs from -0.027 dB (hunyuan, SVT-AV1 CRF 28) to +0.225 dB (wan2_2, x265 CRF 18). Clips above the clean reconstruction run from 16 of 34 to 31 of 34. Median latent relative L2 runs from 0.098 (hunyuan, x265 CRF 18) to 0.234 (wan2_2, SVT-AV1 CRF 28).

The middle rung is x264 CRF 30, x265 CRF 30, SVT-AV1 CRF 49. Across 12 groups the median ΔPSNR runs from -1.719 dB (hunyuan, x264 CRF 30) to +0.009 dB (ltx, SVT-AV1 CRF 49). Clips above the clean reconstruction run from 2 of 34 to 18 of 34. Median latent relative L2 runs from 0.191 (hunyuan, x265 CRF 30) to 0.410 (wan2_2, x264 CRF 30).

The harshest rung is x264 CRF 38, x265 CRF 38, SVT-AV1 CRF 63. Across 12 groups the median ΔPSNR runs from -4.955 dB (hunyuan, x264 CRF 38) to -1.792 dB (ltx, x265 CRF 38). Clips above the clean reconstruction run from 0 of 34 to 1 of 34. Median latent relative L2 runs from 0.275 (hunyuan, x265 CRF 38) to 0.556 (wan2_2, x264 CRF 38).

Ratio of the x264 median latent relative L2 at the harshest rung to the mildest rung: wan2_1 3.80 (0.527 / 0.138), hunyuan 3.27 (0.324 / 0.099), ltx 3.71 (0.536 / 0.144), wan2_2 2.97 (0.556 / 0.188).

The smallest median latent relative L2 in the table is 0.098 (hunyuan, x265 CRF 18). Relative L2 rises with CRF on 408 of 408 clip, VAE, and codec curves.

Read the mildest rung before writing a filter. Its median pixel change crosses zero. On some groups the compressed source reconstructs higher than the clean source. A filter that drops that rung drops reconstructions that are not worse. The latent has still moved. That shift is the smallest in the table, and it grows with CRF on every curve. A filter aimed at pixel PSNR can keep the mildest rung. A filter aimed at "the latent equals the clean latent" has no rung here where that is true.

The middle rung is where the pixel change turns down for Wan2.1, Hunyuan, and Wan2.2. LTX is the group that stays nearest zero there, and it turns down at the harshest rung. The harshest rung is a loss on every group. That is the rung a training filter should drop if it drops anything. The x264 column is the one to trust for a cutoff: that encoder reproduced its files. The x265 and SVT-AV1 columns agree on the direction. They are weaker evidence for a cross-VAE comparison, because those files were not bit-identical across VAEs. Note 10 records that mismatch.

## What this does not decide

It does not say a generator's samples get better. Nothing in the table was generated. The scores are reconstructions of the input.

It does not choose the cutoff from a linear probe. Note 14 fits that probe on the source rung. This note uses the pixel change and the latent distance.

It does not set a threshold between two rungs. The three points are the ones that were measured. A cutoff inside a rung would be a new experiment.

Clips whose source was already Theora or VP9 stay in the medians. Their "clean" reconstruction is itself a second generation.

## Source

The rows are `docs/results/artifact-propagation.csv` and the clean reconstruction is `docs/results/vae-recon.csv`. `scripts/summarize-filtering.py` reprints the section above. A test checks that this note still contains that printout. Note 10 is the measurement. Note 02 is why a CRF does not travel from one codec to another.
