# CRF, presets, and why bits per pixel

A codec has too many switches to plot. This benchmark turns two of them and measures one number on the x-axis. The two switches are CRF and the preset. The number is bits per pixel.

## CRF

CRF means constant rate factor. You hand the encoder a single quality target and it spends bits where the picture is hard: a flat wall gets almost nothing, a crowd in motion gets a lot. The file size is whatever fell out. That is the opposite of a bitrate cap, where you hand over a budget and the encoder decides which frames have to look worse.

x264 and x265 use a scale where low numbers are better. CRF 18 is close to visually lossless on this kind of footage. CRF 23 is a common default. CRF 28 is visibly softer. CRF 38 is a damaged picture, and CRF 43 is worse still. The six points in the sweep are 18, 22, 26, 30, 34, 38. The ladder stops at 38 so the low end still meets SVT-AV1, whose highest CRF does not reach the damage x264 does at 43. ADR-0005 has the overlap numbers behind that stop.

SVT-AV1's dial has the same name and a different meaning. Its CRF runs from 0 to 63. CRF 35 there is a middle setting, not "worse than x264's 28." The ladder used here is 28, 35, 42, 49, 56, 63. It starts at 28, where the quality is about x264's CRF 18, and it ends at 63 because that is the end of the dial. Quoting "CRF 28" without the codec name is a comparison that was never made.

Why CRF and not "1 megabit per second for everyone"? A 33-frame clip is about a second, sometimes less. A bitrate controller spends the start of a clip learning the scene, and on a clip this short it is still learning when the clip ends. CRF does not have that warmup. Each codec also misses a bitrate target in its own way, so a shared kbps ladder would mix encoder behavior into the rate. CRF lets the codec allocate, and the measured rate is what actually landed.

## Presets

The preset is the speed/effort knob, held fixed so the timing means something. `medium` on x264 is the reference point most people already have in their head: not the slowest, not a live stream. x265 `medium` on an 832×480, 33-frame clip took about 1.5 seconds on one thread. That is cheap enough that dropping to `faster` would only hide a little quality to save a fraction of a second. SVT-AV1 preset 6 took about 2 seconds on the same clip; preset 8 took about 1.2. Preset 6 is the one in the sweep. Lower SVT presets exist and are much slower. They are a different experiment.

Tests do not use these presets. CI encodes a 64×64 noise clip at `ultrafast` (and SVT preset 12) to check that a lower CRF spends more bits. The benchmark command is the one that passes `medium` and `6`.

Threads are part of the preset, in practice. x264 is told `-threads 1` and not to use sliced threads. x265 is one pool and one frame thread. SVT-AV1 is `lp=1`, one logical processor. Decode is also one thread. A second run on a busier machine can still move the wall clock. The encoder itself is not allowed to.

One more switch is deliberately absent. `-tune psnr` tells x264 and x265 to chase the squared error, which raises PSNR and often hurts the picture VMAF was trained on. It is not passed. SVT-AV1 on this ffmpeg build defaults to a PSNR tune (the encoder log says so). The wrapper sets `tune=0`, which is SVT's visual-quality tune, so the three codecs are not silently optimizing different scores.

## Bits per pixel

The y-axis of a rate-distortion plot will be PSNR or VMAF. The x-axis has to be a rate that means the same thing for a 24 fps animation and a 60 fps conference clip. Kilobits per second does not. A 60 fps clip has more than twice as many frames to code as a 24 fps clip of the same scene, so the same "difficulty per frame" shows up as a much higher kbps. Divide by the number of samples instead:

```text
bpp = video_bits / (width * height * frames)
```

`video_bits` is the sum of the video packets ffprobe reports, times 8. The mp4 wrapper is not included. At CRF 38 the wrapper is a few kilobytes and the codec spent about that much too, so counting the file size would flatter or punish the container rather than the codec.

A worked step. Suppose the video packets of one 832×480 clip, 33 frames, add up to 80,000 bytes. That is 640,000 bits. The frame holds 832 × 480 = 399,360 samples, and the clip holds 33 of them: 13,178,880 samples. bpp = 640,000 / 13,178,880 ≈ 0.0486 bits per pixel. At 50 fps those 33 frames last 0.66 seconds, so the same bitstream is 640,000 / 0.66 / 1000 ≈ 970 kbps. At 24 fps the same 33 frames last 33/24 = 1.375 seconds, and the same packets are 640,000 / 1.375 / 1000 ≈ 465 kbps. The bits did not change. The clock did. bpp stays 0.0486 either way, which is why the plot uses it. kbps is stored beside it so a point can still be read as a stream rate at that clip's own frame rate.

## Source

The CRF scale and the preset names are the ones documented by the x264 and x265 projects and by SVT-AV1's ffmpeg wrapper. The decision to hold presets fixed, to refuse `-tune psnr`, and to count packet bits rather than the file is [ADR-0005](../decisions/0005-codec-baselines.md). The timings in that note are one clip, `crowd_run`, on the pinned ffmpeg build.
