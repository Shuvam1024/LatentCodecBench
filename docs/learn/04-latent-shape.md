# What shape a latent is

A video VAE turns a clip into a grid. The grid has four sizes: how many numbers sit in each cell (the channels), how many cells run along time, and how many run down and across. Encode asks for pixels and returns that grid. Decode asks for the grid and returns pixels. The wrapper in this repository reports the grid as `(channels, frames, height, width)`, and the tensor you actually get has a batch axis in front of that.

The four models in the benchmark, as Diffusers 0.41.0 constructs them:

| VAE | Channels | Time | Space | 832×480×33 becomes |
| --- | --- | --- | --- | --- |
| Wan2.1 | 16 | 4 | 8 | 16×9×60×104 |
| HunyuanVideo | 16 | 4 | 8 | 16×9×60×104 |
| Wan2.2 TI2V-5B | 48 | 4 | 16 | 48×9×30×52 |
| LTX-Video | 128 | 8 | 32 | 128×5×15×26 |

Read the middle two columns as "this many pixel frames become one latent frame" and "this many pixels on a side become one cell." Wan2.2's 16 in space is a patch of 2 followed by the same 8× stack Wan2.1 uses. The diffusion transformer in that model patches the latent again, down to 32×32 in space. The VAE itself stops at 16. LTX's 32 and 8 are `patch_size * 2^3` and `patch_size_t * 2^3`: the patch is 4 by 4 by 1, and three blocks each halve time and space.

## Why the frame count has a leftover 1

These encoders are causal. The first frame is encoded on its own, because nothing has happened yet for a temporal filter to look at. Every later frame is packed with `temporal_factor - 1` of its neighbors. A legal length is therefore `temporal_factor * k + 1`: 4k+1 for Wan and Hunyuan, 8k+1 for LTX. Thirty-three frames is `4*8+1` and `8*4+1`, so one clip satisfies both rules. The latent frame count is `(frames - 1) / temporal_factor + 1`, which is 9 for a factor of 4 and 5 for a factor of 8.

The same rule is why 32 frames is refused. A multiple of 4 would be natural for a codec GOP and is one frame short of what the causal conv wants.

832 and 480 are divisible by 32, and so by 16 and by 8. A width of 800 fails LTX, whose cells are 32 pixels wide, and passes Wan2.1, whose cells are 8. The canonical clip was chosen so every model sees the same pixels. ADR-0002.

The LTX encoder is causal. The LTX decoder is not: `encoder_causal` is true and `decoder_causal` is false in the class defaults. The length rule still comes from the encoder. HunyuanVideo's paper writes the same idea as a clip of shape `(T+1)` becoming a latent of shape `(T/4 + 1)`.

## Channels travel with the grid

The channel count is part of the shape and part of the rate, later. Sixteen channels at 4×8×8 and 128 channels at 8×32×32 are different bargains: the second grid has fewer cells and a longer vector in each one. Step 2.4 is where those cells get a bit cost. Until then the shape is only the geometry the decoder expects, and a Wan2.1 decoder cannot read a 48-channel Wan2.2 latent.

The tiny networks in CI use these same factors and channel counts with very narrow residual blocks. Narrow blocks change the parameter count. They do not change the grid. That is why a random-init forward of an 832×480×33 tensor is allowed to stand in for the shape check, and why that forward is still required to match the table above.

## Source

Diffusers 0.41.0, tag `v0.41.0`:

- [`autoencoder_kl_wan.py`](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/models/autoencoders/autoencoder_kl_wan.py): `scale_factor_temporal` 4, `scale_factor_spatial` 8, and the encode loop `1 + (num_frame - 1) // 4`. Wan2.2 sets the spatial factor to 16 and `patch_size` to 2 in its VAE config.
- [`autoencoder_kl_ltx.py`](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/models/autoencoders/autoencoder_kl_ltx.py): the compression-ratio formula, and `encoder_causal` / `decoder_causal`.
- [`autoencoder_kl_hunyuan_video.py`](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/models/autoencoders/autoencoder_kl_hunyuan_video.py): `spatial_compression_ratio` 8, `temporal_compression_ratio` 4, `latent_channels` 16.

HunyuanVideo paper, arXiv:2412.03603. A clip `(T+1) × 3 × H × W` is compressed with `c_t = 4`, `c_s = 8`, and `C = 16`.

Wan2.2 TI2V-5B model card: the VAE compression is `4×16×16`. The further `4×32×32` figure is the transformer's patchify on top of that VAE.
