# ADR-0013: A linear probe reads source CRF from latent statistics

- Status: Accepted
- Date: 2026-10-24

## Context

Phase 3 measured that a compressed source moves the posterior mean. Relative L2 rises with CRF on every clip, VAE, and codec. That is a distance. It does not say whether the source rung is readable: a distance can grow without the three rungs being separable by a linear function of the latent.

The raw posterior mean is the wrong object to store. Wan2.1 is about 3.6 MB at float32, and the four VAEs do not share a shape. The probe has to resume on Colab, copy through Drive, and fit on a CPU. A row has to stay a few kilobytes.

CRF is not a shared scale. x264 CRF 38 and SVT-AV1 CRF 63 are both the harsh end of their own ladders, not the same number. A model that pools codecs would be scored on a label that the study already refuses to compare.

## Decision

The feature row is fixed-length. For the posterior mean, and again for its difference from the clean posterior mean, store the per-channel mean, standard deviation, and mean absolute value, padded to 128 channels, plus one high-frequency energy (the mean square of the first difference along width). Three scalars sit beside the difference: MSE, relative L2, and cosine. The clean clip is its own row, with that difference equal to zero. The y4m that was encoded also contributes six luma numbers: mean, standard deviation, blocking, ringing, spatial highpass energy, and the mean absolute frame step. Those six are the pixel baseline. `source_sha256` is the SHA256 of that y4m. The padded row is about 10–15 KB of CSV, not the latent.

The sweep encodes only. It does not decode. Rows are 34 clips, four VAEs, and ten sources (clean plus the nine study points): 1360. The degraded set is built once per VM and a finished y4m is kept, which is as deterministic as the encoder on that machine. x265 and SVT-AV1 have disagreed across VMs before, so the hash is the record of which pixels were encoded. The CSV is `My Drive/LatentCodecBench/results/latent-probe-features.csv`. The upload merges first and refuses a file with fewer rows than Drive, the same rule as ADR-0012.

The fit is per VAE and per codec. Clean rows are not targets. Regression predicts CRF. Classification predicts the rung, which is the index of that CRF on the study ladder (mildest, middle, harshest). Cross-validation is leave-clip-out: the three rungs of a clip never train the model that scores them. The model is ridge with penalty 1 on z-scored columns and an unpenalized intercept. The classifier is the same ridge onto the three rung indicators, and the predicted rung is the largest score. A column with no variance in the training clips is dropped, which removes the padding channels of a narrower VAE.

Chance uses no features. Regression chance predicts the mean training CRF. Classification chance predicts the most common training rung and breaks a tie toward the mildest rung. With one row per rung per clip, that accuracy is one third. The pixel model is the same ridge on the six luma numbers.

The reported number is the mean across clips of the clip's own score (MAE of its CRFs, or accuracy of its rungs). The 95% interval is the 2.5 and 97.5 percentiles of that mean over 1000 bootstrap resamples of the clips, seed 0. `lcb probe fit` writes the table, the figure, and the marked section of `docs/learn/14-linear-probe.md`.

## Consequences

A column that is constant inside one VAE does not enter that VAE's fit. The stored width is still 128 so one CSV header covers every model. The linear probe can say the rung is readable. It cannot say a nonlinear probe would fail, and it cannot compare a CRF across codecs.

The feature CSV is produced on a GPU and is not in the repository until that run is copied in. Until then the learn note has no measured table. The fit does not re-encode anything.

Leave-clip-out spends one encode of the clean clip whenever a clip still has a missing row, and it does not replace a clean row that is already stored. A resumed clip therefore uses a newly encoded clean latent as the reference for its new degraded rows. The stored clean row stays the one written first.

## Alternatives

**The raw posterior mean.** One row would be about a megabyte, the shapes differ, and Drive resume of 1360 tensors is a different system. Channel statistics are the same kind of summary for every VAE.

**Logistic regression.** It is the same linear family. A Newton step on several hundred columns, per class, per fold, is the slow part, and the clip count is smaller than the column count. Ridge onto the rung indicators is one linear solve per class and is the classifier this probe reports.

**A random split of rows.** The three rungs of one clip share the clip. A row split trains on that clip and then scores it. Leave-clip-out does not.

**One model across codecs.** The label would be a CRF, and CRF is not a shared scale. The fit stays inside one codec.

**A searched penalty.** An inner loop over clips would retune λ on every fold. λ = 1 on z-scored columns keeps the fit to one solve per fold and makes the number repeatable.

**Decode the reconstruction and probe the pixels of the VAE output.** That asks whether the damage is visible after the decoder. This step asks whether it is readable in the latent. The pixel baseline is the source y4m, not the reconstruction.

## Addendum, 2026-10-27

The feature table is in the repository as `docs/results/latent-probe-features.csv.gz`. It is gzip of the Colab CSV. The uncompressed SHA256 is `c759c29aad3a04dedaad52b795524a96aee00a9d266c6365d6bfadb60f0ace0f`. 1360 rows, one config `bcc52c3cc55f5c45c7e45a0643a4e7dec6e4e6e7b533e55f4acf7e632f74d9ed`, git `5112c2552a5c7e198f38996399a6e496f135e7b4`, run `20261024T182512Z-5112c2552a5c-bcc52c3cc55f`. `source_crf` and `preset` are empty on the 136 clean rows only.

`lcb probe fit` wrote `docs/results/latent-probe.csv` and `docs/results/latent-probe.png`, and filled the marked section of note 14. The per-codec fits are unchanged from this ADR. The note also pools classification accuracy across codecs by averaging clip scores, and it reports clean versus degraded and codec identity with the same ridge and the same split. Those two are readouts, not a change to the rung task. Pooling does not fit one model on a shared CRF.
