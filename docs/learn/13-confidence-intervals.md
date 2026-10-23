# What the confidence interval is claiming

A mean over clips has a spread. The 95% Student-t interval is the mean plus or minus a critical value times the sample standard deviation over the square root of the number of clips. The critical value is the two-sided 95% point of a Student-t with one fewer degree of freedom than the number of clips.

It answers a narrow question: if these clips were a sample from a larger population of the same kind, where would that population's mean plausibly sit. It does not answer how accurately one clip was measured. The forward is deterministic. Running it again in the same dtype prints the same score. The width is the clips disagreeing with each other.

A designed benchmark is not that sample. The sources were picked. Use the interval as a check on whether two means are still apart once clip spread is admitted. Do not read it as a claim about all video.

## The reconstruction intervals

| VAE | Clips | Mean PSNR-Y | 95% CI | Mean VMAF | 95% CI |
| --- | ---: | ---: | --- | ---: | --- |
| wan2_1 | 34 | 34.57 dB | [32.24 dB, 36.90 dB] | 86.40 | [83.17, 89.62] |
| hunyuan | 34 | 35.11 dB | [32.90 dB, 37.32 dB] | 87.63 | [84.80, 90.46] |
| ltx | 34 | 32.30 dB | [30.10 dB, 34.50 dB] | 74.56 | [68.27, 80.85] |
| wan2_2 | 34 | 35.35 dB | [32.70 dB, 37.99 dB] | 87.13 | [83.56, 90.70] |

On VMAF, LTX's interval does not meet the other three. On PSNR-Y the four intervals overlap, so the ordering of those means is not a separation.

The BD-rate intervals are in [docs/REPORT.md](../REPORT.md). Every classic interval there lies below zero, and every VAE interval lies above zero. Those VAE intervals are wide. A hard clip pulls the mean, which is why note 09 also prints the median.

## Source

`student_t_interval` in `src/latentcodecbench/analysis/intervals.py` is the calculation. The report section that must keep matching the CSVs is the block above. [ADR-0006](../decisions/0006-rd-curves-and-bd-rate.md) is the overlap rule the BD-rate interval inherits.
