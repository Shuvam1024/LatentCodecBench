# How a latent becomes a file

A reconstruction score answers "how close are the pixels?" It does not answer "how many bits did that cost?" The posterior mean is a grid of floats. Floats are not a codec. Step 2.4 turns that grid into a file, then turns the file back into a grid the decoder can read.

## Put the grid on a published scale

Each model already publishes the scale its diffusion pipeline uses. Wan2.1 and Wan2.2 subtract a per-channel mean and divide by a per-channel standard deviation. HunyuanVideo multiplies by one number, 0.476986. LTX-Video subtracts a per-channel mean and divides by a per-channel standard deviation; its extra `scaling_factor` is 1. Those numbers are pinned with the weights. They are not estimated from the clip under test, so they are not side information.

After that map, a typical value sits near zero. The quantizer is then the same for every clip.

## Snap each value to a rung

The rung ladder runs from -4 to 4. Two bits means four rungs. Twelve bits means 4096. A value past either end is clipped to the end rung, then the nearest rung wins. The decoder stores the rung number, not the original float.

The integer tensor is those rung numbers in order: channels, then frames, then height, then width. Depths up to 8 use one byte per number. Depths 10 and 12 use two bytes, little-endian. A lot of those bytes are predictable. zstd level 3 compresses them. The compressed blob is the payload.

A 24-byte header sits in front and is counted too. It records the bit depth, how many channels the decoder expects, how many of those were actually stored, the latent shape, and the two ends of the ladder. Without it the blob is just bytes. `bits` is eight times the header plus the blob. That is a file size. You could write the bytes to disk and get the same number from `len`.

## Bits per pixel still means pixels

The classic table divides video-packet bits by `width * height * frames`. This table uses that same function. A Wan2.1 latent for an 832×480×33 clip is 16×9×60×104, which is far fewer numbers than the pixels. Dividing by the latent grid would make every VAE look cheaper than it is, and it would not line up with x264. The denominator is the pixels the viewer sees.

A worked step. Suppose the header and the zstd blob add up to 80,000 bytes, so `bits` is 640,000. The clip is 832×480×33 = 13,178,880 pixels. `bpp` is 640,000 / 13,178,880 ≈ 0.0486. That is the same arithmetic as note 02. `kbps` then uses the clip's own frame rate, again the same helper as the classic row.

`crf` in this table is the bit depth, 2 through 12. It is not x264's CRF. The preset column says `zstd`, so a reader can tell the rows apart from `reconstruction` and from `medium`.

## What is left for a GPU

Quantizing a numpy array is CPU work, and the tests do it on a synthetic grid. Producing the posterior mean, and decoding the restored grid back to pixels for PSNR and VMAF, is the VAE. That forward is the Colab notebook `notebooks/vae_bitrate.ipynb`. One encode is shared by the six depths. Each depth has its own decode. The notebook writes `vae-bitrate.csv`. The merge script copies it into `docs/results/` after the run. This note does not contain those rows.

## Source

The normalization formulas are the ones in Diffusers 0.41.0: Wan's pipeline divides by the reciprocal of `latents_std` before decode, HunyuanVideo divides by `scaling_factor`, and LTX's `_denormalize_latents` multiplies by `latents_std / scaling_factor` and adds `latents_mean`. The decision to count the zstd file, header included, and to divide by pixels, is [ADR-0010](../decisions/0010-latent-bitrate.md).
