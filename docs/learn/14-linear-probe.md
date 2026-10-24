# Is the source rung readable in the latent?

Note 10 showed that a compressed source moves the posterior mean: relative L2 rises with CRF. A distance is not a reader. A linear probe asks a sharper question. Given the latent, and no other clip, can a linear function name the CRF the codec used, and can it name which of the three study rungs that CRF is? [ADR-0013](../decisions/0013-linear-probe.md).

## What is stored

The posterior mean is not the feature. It is a different shape for each VAE, and it is about a megabyte. The row stores the per-channel mean, standard deviation, and mean absolute value, padded to 128 channels, and one high-frequency energy. The same block is stored for the difference from the clean posterior mean. The clean clip is its own row, and that difference is zero. Six luma numbers from the y4m that was encoded are stored beside them. Those six are the baseline that never sees the latent. `source_sha256` is the hash of that y4m, because x265 and SVT-AV1 have not always reproduced their own files.

A row is about 10–15 KB of CSV. The study is 34 clips, four VAEs, and ten sources: 1360 rows. The notebook encodes and does not decode.

## How the fit refuses to cheat

The fit is one VAE and one codec at a time. CRF 38 on x264 and CRF 63 on SVT-AV1 are both the harsh end of their own ladder. They are not the same label, so they are not one model.

Each fold holds out a whole clip. The three rungs of that clip are scored by a model that has not seen the clip. A split that mixed the rows of one clip would train on the clip it then claims to predict.

The model is ridge with penalty 1 after the training columns are z-scored. The intercept is not penalized. A column that does not vary in the training clips is dropped, which is how the padding channels of a 16-channel VAE leave the fit. Classification is the same ridge onto the three rung indicators. The predicted rung is the indicator with the largest score.

Chance predicts the mean training CRF, and for the rung it predicts the most common training rung, breaking a tie toward the mildest. One of each rung makes that accuracy one third. The pixel model is the same ridge on the six luma numbers. If the latent model does not beat both, the latent is not adding a linear readout that the pixels, or the bare average, already have.

The number in the table is the mean across clips. The interval resamples clips 1000 times and takes the 2.5 and 97.5 percentiles. It is the clip-to-clip spread of that mean. It is not the error of one encode.

## Measured table

The feature CSV is written on Colab to `My Drive/LatentCodecBench/results/latent-probe-features.csv`. It is not in this repository yet, so the block below has no measured numbers. Copy the CSV in and run:

```bash
uv run lcb probe fit \
  --features docs/results/latent-probe-features.csv \
  --out docs/results/latent-probe.csv \
  --figure docs/results/latent-probe.png \
  --note docs/learn/14-linear-probe.md
```

<!-- probe-results:start -->

The feature table has not been measured. This section is replaced by `lcb probe fit`.

<!-- probe-results:end -->
