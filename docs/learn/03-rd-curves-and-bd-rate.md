# RD curves and BD-rate

Two codecs, each run at several qualities, give two clouds of points. An RD curve is those points with rate on the x-axis and distortion on the y-axis, joined in order. In this benchmark the rate is bits per pixel, on a log axis, because a step from 0.02 bpp to 0.04 bpp is the same kind of move as a step from 0.2 to 0.4, and a linear axis would crush the low-rate end into the border. The y-axis is PSNR-Y or VMAF. Up is better. Left is cheaper.

A curve that sits up and to the left of another curve is the better codec on that clip: more quality at the same rate, or the same quality at a lower rate. Real curves cross. One codec wins at low rate and loses at high rate, or wins on PSNR and loses on VMAF. Reading the picture is the point of drawing it. A single number is what you quote after you have looked.

## BD-rate

BD-rate, for Bjontegaard delta, is that number. Take the two curves. Keep only the range of quality where both of them have a sample. Along that overlap, ask how much rate the test codec spends compared with the anchor, on average, in the log domain. Turn the average back into a percent.

The integral is not a straight line between the endpoints and not a single polynomial through all four points. It is a PCHIP, a piecewise cubic that passes through the samples and does not swing past them. That is the curve the JCT-VC common test conditions use. A global polynomial can loop between the points and invent a rate nobody measured.

The sign is easy to get backwards once. Negative means the test codec needs fewer bits than the anchor for the same quality. -20% means about a fifth less rate. Positive means the test is more expensive. The anchor here is x264, so a negative number for x265 is x265 winning.

## A worked case where the answer is -50%

Give the anchor four points. Rate 1000, 500, 200, 80 at PSNR 42, 38, 34, 30 dB. Give the test codec exactly half the rate at the same four PSNR values: 500, 250, 100, 40.

The quality ranges are identical, 30 dB to 42 dB, so the overlap is the whole curve. At every quality, the test rate is the anchor rate divided by two. The log of the rate is therefore lower by exactly `log(1/2)`, everywhere, not just at the samples. The average gap in the log domain is `log(1/2)`. Exponentiate and subtract one:

```text
exp(log(1/2)) - 1 = 0.5 - 1 = -0.5
```

BD-rate is -50%. The unit test locks that. It does not depend on the cubic, because a constant gap survives any interpolant that passes through the samples.

BD-PSNR is the same idea with the axes swapped: at a given rate, how many dB the test is ahead, averaged over the overlap of the two rate ranges. On this same example it is not zero. The test curve is shifted left, so at a rate both curves cover, the test has already reached a higher PSNR. The rate ranges only partly overlap (the anchor never goes as low as 40, the test never goes as high as 1000), and on the overlap the gap is about 3.18 dB. The unit test locks 3.1801024775791302, which is what the `bjontegaard` package returns for this input with method `pchip`. A horizontal win and a vertical win are the same picture measured two ways, and they are not interchangeable numbers.

## A published vector that is not a toy

The `bjontegaard` package ships a four-point example that is not a pure scale:

```text
anchor rate  9487.76, 4593.60, 2486.44, 1358.24
anchor PSNR  40.037,  38.615,  36.845,  34.851
test rate    9787.80, 4469.00, 2451.52, 1356.24
test PSNR    40.121,  38.651,  36.970,  34.987
```

PCHIP BD-rate on that vector is -4.417485350589045%. BD-PSNR is 0.11969278542479489 dB. The test spends a little more rate at the top point and a little less further down, and it is slightly ahead in PSNR throughout, so the average comes out as a small saving. The implementation in this repo is locked to those digits. If a refactor changes the interpolant, this test fails before any video is encoded.

## How this repo reports it

BD-rate is computed per clip, then averaged across clips. One number for the whole set can hide a clip where the ranking flips; the mean of the per-clip numbers is what a sequence report usually quotes, and the per-clip values can be recomputed from the CSV. A clip whose curves do not overlap is skipped and counted, not given a NaN that poisons the mean.

The figure next to that number is the mean curve: at each CRF, average bits per pixel and average PSNR (or VMAF) across clips, then plot. CRF is not the x-axis. x264's CRF 28 and SVT-AV1's CRF 28 are not the same operating point, so the curves are not lined up by CRF index. They are lined up by the rate that actually came out.

## Source

Gisle Bjontegaard, "Calculation of average PSNR differences between RD-curves," ITU-T SG16/Q6, VCEG-M33, 2001. The piecewise-cubic form used here is the one in the JCT-VC common test conditions, matched by the `pchip` method of the `bjontegaard` package (FAU-LMS). ADR-0006 records the choice and the two locked numbers.
