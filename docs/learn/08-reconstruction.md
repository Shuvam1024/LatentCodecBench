# Reconstruction, before a bitrate

A video VAE is trained to turn pixels into a grid and the grid back into pixels. The first measurement is how close those pixels come back, with the grid left alone. No rounding, no entropy coder, no bitstream. The rate columns in this table are empty on purpose. Notes 05 and 06 are reserved for the step that puts bits on the latent.

The clip and the score are the ones Phase 1 already uses. The source is the canonical 832×480×33 y4m. PSNR-Y is luma, peak 255, one MSE over the whole clip. SSIM is luma. VMAF is libvmaf through the pinned ffmpeg. Flicker is the mean absolute disagreement of frame-to-frame luma deltas. The smoke notebook's figure near 10 dB was a different number: uniform noise in `[-1, 1]`, peak 2, one clip, a load check.

The path from the file to the network is the BT.709 limited-range matrix. Luma 16..235 and chroma centered at 128 become RGB in 0..255, then `[-1, 1]`, which is the range the wrapper documents. A limited-range YUV triple can fall outside that cube. Those samples are clipped, because that is the range the VAEs were trained on. Where the triple already lands inside the cube, the trip back to 4:2:0 restores the luma. Chroma in the file is one sample per 2x2 block, so the way back averages that block. `score_y4m` then compares the reconstructed y4m with the source, and the file is deleted.

## Tiling is a column

HunyuanVideo enables spatial tiling before the first clip. The T4 smoke ran that model out of memory until `enable_tiling()` was called. Its framewise switch is already on, in chunks of 16 frames, and 33 frames is above that minimum. The spatial minimum is 256 pixels on a side. 832×480 is above it, so the flag being on means the forward really tiles. Wan2.1, LTX-Video, and Wan2.2 start with tiling off. One out-of-memory error turns it on and retries that clip once. `spatial_tiling`, `framewise_encoding`, `framewise_decoding`, and `retried_with_tiling` are how a later reader tells those runs apart.

Per-chunk time is the mean time spent inside `encoder.forward` on the encode that succeeded. Wan's encoder is called once per group of four frames, which is nine calls on a 33-frame clip.

## Where the 136 rows go

34 clips times four VAEs is 136 rows. The notebook writes `/content/results/vae-recon.csv` after each one and skips a pair it has already stored, so a restarted runtime continues. An L4 runs the weights in bfloat16. A T4 runs them in float16. Torch, Diffusers, the GPU name, and the VAE revision sit on the row, next to the run id `{utc}-{git_sha12}-{config_sha12}`.

`scripts/merge-vae-recon.py` is what copies that CSV into `docs/results/vae-recon.csv` after it is downloaded. The y4m clips, the latents, and the checkpoints stay out of git. ADR-0009 records the tiling rule and the empty rate columns.
