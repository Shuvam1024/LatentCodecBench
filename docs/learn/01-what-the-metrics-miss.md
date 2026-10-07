# What the metrics miss

A decoded frame is a pile of samples. A metric turns two piles, the reference and the decode, into one number. None of the numbers this benchmark reports is "quality." Each one is blind to a different kind of mistake, which is why the table has more than one column.

## PSNR

PSNR starts from the squared error. Subtract the two frames, square, average, and compare that average to the largest sample you could have stored. For 8-bit video that peak is 255, and the score in decibels is

```text
PSNR = 10 * log10(255^2 / MSE)
```

Identical frames have no error, so the score is infinite. A 2×2 patch with a single sample off by one has MSE 0.25 and PSNR about 54.15 dB. Ten levels of error on every sample is 100 MSE and about 28.1 dB. Bigger is better. A jump from 40 dB to 41 dB is a real reduction in error, and a jump from 20 dB to 21 dB is a much larger one in absolute pixels, because the logarithm squeezes the bad end.

What it misses is everything that is not an average of squares. Shift the picture by one pixel and almost every sample is "wrong," so PSNR collapses, even though a person still sees the same scene. Blur and a sprinkle of noise can share an MSE and look nothing alike. A single frame that falls apart is diluted if you average per-frame PSNR; this project pools the MSE first, so that frame still moves the clip score. PSNR also has no idea that time exists. A clip that flashes can win on PSNR frame by frame.

We report it twice. PSNR-Y is the luma plane, which is where codec papers usually quote the number. PSNR-RGB converts studio-range YUV to BT.709 RGB first, so a chroma mistake is visible. The conversion is part of the score: a different matrix would be a different metric.

## SSIM

SSIM (Wang, Bovik, Sheikh, Simoncelli, 2004) asks whether local structure survived. Around each pixel it looks at an 11×11 window, compares the average brightness, the variance, and the covariance of the two frames, and combines those with two small constants so a flat wall does not divide by zero. The constants are the paper's: `k1 = 0.01`, `k2 = 0.03`, peak 255. The window is a Gaussian with sigma 1.5. The score sits near 1 when the patch has the same shape, and near 0 when it does not. A constant offset on a flat field still has a closed form, which is how the unit test is checked. A textured frame with half of its columns replaced is locked to the value scikit-image computes with the same window.

What it misses: it is still a still-image metric, averaged over frames here. It does not know that a block moved. It is also almost indifferent to a small brightness shift when the edges line up, which is often what you want and sometimes what you do not (a whole scene crushed toward gray can keep its edges). We run it on luma only. Color errors have to show up in PSNR-RGB or in looking at the frames.

## VMAF

VMAF is Netflix's attempt to predict a subjective score. It computes a few features that each describe a different failure (detail loss, additive impairment, motion) and fuses them with a model trained on people rating compressed video. The output is roughly 0 to 100. Higher is better. ffmpeg's libvmaf is the implementation used here: the distorted clip is the first input, the reference is the second, and the recorded number is the pooled mean.

What it misses: it is a model of the videos and the raters it was trained on, mostly professionally shot material with codec artifacts. A failure mode those raters never saw (a VAE that invents a texture, a temporal pump at a frequency the features smear out) can get a flattering score. It is also not an identity function. Two identical clips of a test pattern score above 99, not a perfect 100, because the features are integer approximations. Treating 100 as a bitwise check will fail a correct encode. And because the fusion was fit to opinion scores, a paper that "optimizes VMAF" can climb the number without climbing the thing a different audience would have said.

## LPIPS

LPIPS (Zhang et al., 2018) measures distance in the feature maps of a classifier, here AlexNet, instead of in pixels. Patches that people call similar tend to land near each other in that space, even when the squared error is large. It is an extra column. The default environment has no PyTorch, and a missing install records "not computed" rather than failing the run.

What it misses: the network was trained on photographs and image classification, not on video and not on codec artifacts specifically. It has no temporal term. Two consecutive frames can each look fine to LPIPS while the cut between them flickers. It also depends on which backbone you pick. AlexNet LPIPS and VGG LPIPS do not agree on every pair, so a number without the network name is not comparable.

## The temporal hole

All four of those are, at heart, opinions about frames. A codec can win every one of them and still pump: blockiness that appears and disappears, a fine texture that freezes for a frame, a face that swims. The flicker score in this repo is a cheap check on that. Take how much the reference changed from one frame to the next, take how much the decode changed, and average the absolute disagreement. Identical clips score 0. Adding 7 to every pixel of every frame also scores 0, because the changes were preserved even though the level moved (PSNR would have punished that offset; flicker correctly does not).

What flicker misses is direction. A pan that goes the wrong way can change by about as many levels as the true pan, and the deltas then look similar. The metric that would catch it compares the reference, warped by optical flow, to the decode. That needs a flow field per frame pair. It is the right next temporal metric, and it is not in this phase: a flow network is a large dependency for a CPU baseline, and the delta already flags freezes and pumping. When it is added, it should sit beside flicker, not replace the frame metrics.

## Source

Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli, "Image Quality Assessment: From Error Visibility to Structural Similarity," IEEE Transactions on Image Processing, 2004.

Zhi Li, Anne Aaron, Ioannis Katsavounidis, Anush Moorthy, and Megha Manohara, "Toward a Practical Perceptual Video Quality Metric," Netflix Technology Blog, 6 June 2016. <https://netflixtechblog.com/toward-a-practical-perceptual-video-quality-metric-653f208b9652>

Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang, "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric," CVPR 2018.
