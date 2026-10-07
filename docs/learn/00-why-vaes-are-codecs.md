# Why VAEs are codecs

A video codec and a video VAE do the same two-step job. An encoder turns a clip into a smaller representation. A decoder turns that representation back into pixels. In x264, x265, or SVT-AV1 the representation is a bitstream: prediction modes, transform coefficients, and an entropy-coded payload. In Wan2.1, Wan2.2, LTX-Video, and HunyuanVideo the representation is a latent, a grid of vectors. The generator never sees pixels. It samples that grid, and the VAE decoder is the only stage that paints frames. Asking how good those frames are, and how many bits the latent stands in for, is codec evaluation.

The shape of the grid is a compression setting, in the same way a codec has a profile. These models publish three different spatiotemporal factors. Read the numbers as temporal × height × width: how many frames, rows, and columns collapse into one latent cell.

Wan2.1 and HunyuanVideo use **4×8×8**. Four frames become one latent frame, and each 8×8 block becomes one spatial cell. The grid stays relatively fine, so reconstruction is easier and the generator has more tokens to process. Wan2.2's high-compression VAE moves to **4×16×16** and reports an overall compression rate of 64: time is compressed the same way, space is four times coarser. LTX-Video uses **8×32×32**. Eight frames and a 32×32 patch share one cell. Generation gets cheap because attention now runs on a tiny grid, and the decoder is asked to invent texture the latent never stored.

The factor alone is not the bit budget. Each cell is a vector, and these models use more channels than the three in RGB, so a harsher downsample does not shrink the tensor by the full `4 × 8 × 8`. Sander Dieleman calls the combined shrink the tensor size reduction factor: input elements divided by latent elements. You can downsample harder and widen the channels, and the tensor barely changes size. Several shapes with the same reduction reconstruct about equally well, which is why a video VAE can split its budget differently across time, space, and channels. Change the reduction itself and two things move together: reconstruction quality, and how easy the latent is to model. A fatter latent reconstructs better and leaves the generator more bits to learn. A thinner one is cheaper and loses detail. That is a rate-distortion trade-off.

Rate-distortion is also why the comparison with classic codecs is a fair one. A codec is scored by rate (bits per pixel) against distortion (PSNR, SSIM, VMAF, and, in this project, temporal flicker). A VAE latent is not a bitstream until each channel is quantized and entropy-coded. That step comes later. Once it exists, the latent has a rate, distortion is measured on decoded frames the same way, and BD-rate can say which curve wins.

Dieleman adds the axis that pure compression does not have: modelability. Lossy compression only cares that a decoder can invert the file. A generator also has to learn the distribution of the representation. A representation can be compact and still be a bad latent, if the structure a network could have used was thrown away.

That is why a codec bitstream is a poor stand-in for a latent. Codecs are built to remove redundancy. Motion compensation and transforms strip structure the decoder knows how to put back. The last stage, entropy coding (CABAC in HEVC, the symbol coder in AV1), assigns short codewords to common symbols. Nothing is deleted, but neighborhood structure is gone: a one-step change in a residual can reshuffle a long stretch of the bitstream. Dieleman uses entropy coding as the example of what not to feed a generator. It saves rate and destroys the regular layout that convolutions and attention exploit. Image latents from a model like Stable Diffusion still look like blurry pictures. An AV1 packet does not.

So this benchmark treats the VAE as the codec under test, and keeps x264, x265, and SVT-AV1 as baselines that emit bitstreams. It does not try to run a generator on DCT coefficients or on the bitstream itself.

One smaller constraint is codec-like too. These video VAEs are causal. The first frame is encoded on its own, and later frames are grouped by the temporal factor, so legal clip lengths look like `4n+1` or, for an 8× temporal VAE, `8n+1`. Preprocessing has to respect that, the way a codec test respects GOP rules.

## Source

Sander Dieleman, “Generative modelling in latent space,” 15 April 2025. <https://sander.ai/2025/04/15/latents.html>

The compression factors above are the ones published for Wan2.1 (4×8×8), HunyuanVideo (4×8×8), the Wan2.2 high-compression VAE (4×16×16, overall rate 64), and LTX-Video (8×32×32). Dieleman's essay is the source for the tensor size reduction factor, the rate-distortion-modelability trade-off, and the argument against entropy-coded bitstreams as latents.
