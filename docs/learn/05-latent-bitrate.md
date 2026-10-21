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

Quantizing a numpy array is CPU work, and the tests do it on a synthetic grid. Producing the posterior mean, and decoding the restored grid back to pixels for PSNR and VMAF, is the VAE. That forward is the Colab notebook `notebooks/vae_bitrate.ipynb`. One encode is shared by the six depths. Each depth has its own decode. The notebook writes `vae-bitrate.csv`. `scripts/merge-vae-bitrate.py` copies it to `docs/results/vae-bitrate.csv`.

## What the L4 run measured

The checked-in file is 816 rows: 34 clips, four VAEs, and bit depths 2, 4, 6, 8, 10, and 12. One config hash. Run id `20261020T234203Z-8cb0b7cd9bc4-17ec7df8f1e6`, from git `8cb0b7cd9bc474d254e2488bd02cb43d627f5f66`. The runtime is bfloat16 on an NVIDIA L4. `lpips` is empty on every row. `entropy_bits` is the cross-entropy from note 06. It is an estimate, and it is not added into `bits`.

`scripts/summarize-vae-bitrate.py` rebuilds the table from the CSV. Each number is the mean PSNR-Y, the mean VMAF, or the median bits per pixel at that depth. With 34 clips the median is the average of the 17th and 18th bpp values.

| VAE | Depth | Mean PSNR-Y | Mean VMAF | Median bpp |
| --- | --- | --- | --- | --- |
| wan2_1 | 2 | 17.89 | 14.87 | 0.0537 |
| wan2_1 | 4 | 27.43 | 43.06 | 0.1223 |
| wan2_1 | 6 | 33.37 | 78.67 | 0.2575 |
| wan2_1 | 8 | 34.49 | 85.79 | 0.4141 |
| wan2_1 | 10 | 34.57 | 86.34 | 0.6396 |
| wan2_1 | 12 | 34.57 | 86.39 | 0.7780 |
| hunyuan | 2 | 18.16 | 21.23 | 0.0399 |
| hunyuan | 4 | 26.75 | 41.22 | 0.1101 |
| hunyuan | 6 | 33.43 | 78.71 | 0.2431 |
| hunyuan | 8 | 34.90 | 86.85 | 0.4078 |
| hunyuan | 10 | 35.02 | 87.50 | 0.6182 |
| hunyuan | 12 | 35.03 | 87.54 | 0.7468 |
| ltx | 2 | 17.11 | 15.85 | 0.0254 |
| ltx | 4 | 28.19 | 53.41 | 0.0624 |
| ltx | 6 | 31.85 | 71.98 | 0.0973 |
| ltx | 8 | 32.21 | 73.75 | 0.1340 |
| ltx | 10 | 32.24 | 73.86 | 0.2080 |
| ltx | 12 | 32.24 | 73.87 | 0.2424 |
| wan2_2 | 2 | 17.11 | 17.98 | 0.0500 |
| wan2_2 | 4 | 28.85 | 46.49 | 0.1298 |
| wan2_2 | 6 | 34.27 | 80.15 | 0.2358 |
| wan2_2 | 8 | 35.26 | 86.52 | 0.3420 |
| wan2_2 | 10 | 35.33 | 87.00 | 0.5262 |
| wan2_2 | 12 | 35.34 | 87.04 | 0.6273 |

Quality levels off from about depth 8 on every VAE. Depth 6 to depth 8 is still 1.12 dB on Wan2.1, 1.47 dB on Hunyuan, 0.36 dB on LTX, and 0.99 dB on Wan2.2. From depth 8 to depth 12 the same means move 0.08, 0.13, 0.03, and 0.08 dB. At depth 8 the means are Wan2.2 35.26 dB and VMAF 86.52, Hunyuan 34.90 and 86.85, Wan2.1 34.49 and 85.79, LTX 32.21 and 73.75. LTX's median bpp there is 0.1340, about a third of Wan2.1 (0.4141) and Hunyuan (0.4078), and about two fifths of Wan2.2 (0.3420). Its ceiling is the lower one. Mean PSNR-Y at depth 2 is 17.11 to 18.16 dB.

At depth 12 the mean PSNR-Y of each VAE is within 0.1 dB of the unquantized reconstruction in `docs/results/vae-recon.csv`. Hunyuan on `sintel_trailer` is the clip that misses by a wide margin: 48.06 dB against 50.50 dB, 2.44 dB lower. Depth 8 on that pair is already 48.12 dB, 2.38 dB under the same reconstruction. A value past either end of [-4, 4] is stuck on the end rung at every depth, so a synthetic clip with extreme latents would look like this once the rungs are fine. The latent tensors stay out of git, with the clips and the checkpoints, so that clipping account is a hypothesis. Eight other pairs are 0.12 to 0.38 dB lower at depth 12, six of them LTX. The mean for each VAE still stays inside 0.1 dB.

Wan2.1 on `sintel_trailer` is a smaller dip at the top of the ladder: 50.76 dB at depth 8, 50.28 dB at depth 10 (0.48 dB lower), and 50.17 dB at depth 12. The unquantized row is 50.24 dB, so depth 12 is 0.07 dB under it.

## Source

The normalization formulas are the ones in Diffusers 0.41.0: Wan's pipeline divides by the reciprocal of `latents_std` before decode, HunyuanVideo divides by `scaling_factor`, and LTX's `_denormalize_latents` multiplies by `latents_std / scaling_factor` and adds `latents_mean`. The decision to count the zstd file, header included, and to divide by pixels, is [ADR-0010](../decisions/0010-latent-bitrate.md).
