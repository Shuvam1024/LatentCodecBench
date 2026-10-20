# A prior costs bits, unless you fixed it first

zstd gives a file. A probability model gives a number that *would be* a file if someone built the coder. Those are different claims, and this benchmark keeps them in different columns.

## What the Gaussian is claiming

After the published normalization, the grid is supposed to look like independent standard normals: mean 0, variance 1, no correlation between neighbors. "Factorized" means the model refuses to look at neighbors. The probability of a whole latent is the product of the probabilities of its cells.

The quantizer cut `[-4, 4]` into bins. The center of each bin is a rung. The probability of a rung is the normal curve's area over that bin. The two end bins also swallow everything past -4 and past 4, so the areas sum to one. The cross-entropy is the average, over the rungs you actually sent, of minus log base 2 of that area, summed rather than averaged so the unit is bits for the whole clip. That sum is `entropy_bits`.

Mean 0 and variance 1 were chosen before any test clip existed. There is no extra file that says "the variance was 1.07 on this clip." If the variance had been fit on the clip, those parameter bytes would have to sit in `bits` as well. A prior that saw the test clip and then forgot to count its own description is not a rate.

## Why it is still not a bitstream

Cross-entropy is the size of an ideal code for that model. It assumes a coder that can spend fractional bits, and it assumes the model is the one the coder uses. This repository does not write that coder. Nothing in the zstd payload is an arithmetic-coded Gaussian. `entropy_bits` can be smaller than `bits`, and often will be, because zstd does not know the bins are a normal. It can also be larger, if the latent is nothing like a unit normal and zstd finds some other repetition.

You cannot hand `entropy_bits` to a decoder and get the clip back. You can hand it the zstd file. The column is an estimate of how cheap the latent would be under a dumb independent normal. The DEVLOG says the same thing in one line: a cross-entropy without a bitstream is an entropy estimate.

These models were trained to reconstruct. A unit normal is the diffusion pipeline's convenience, not a proof that the posterior mean is that normal. When the estimate and the file disagree, the file is the rate this step reports for BD-rate later. The estimate is the comment on how much a matched coder might still save.

## The channel knob

A prefix of channels is a cruder rate switch than bit depth. `channels` in the config, left null for the first sweep, keeps the first K channels in the file and tells the decoder to fill the rest with zeros. Zeros here are zeros in the tensor `decode` sees, after the kept channels have been mapped back to raw latent space. The dropped channels are not "the mean." They are empty. Their absence is already in the header, as `channels_kept`, so it is counted.

That knob exists so a later run can ask whether throwing channels away moves the low bit depths down onto the classic ladder. It is not turned on until a full six-depth file says the ladder is still out of reach.

## Source

The factorized-prior split, and the rule that a test-clip parameter must be counted or fixed beforehand, is [ADR-0010](../decisions/0010-latent-bitrate.md). Note 05 is the file the decoder can actually read.
