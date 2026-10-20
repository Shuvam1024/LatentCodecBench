# ADR-0009: Reconstruction sweep, with Hunyuan tiled from the start

- Status: Accepted
- Date: 2026-10-19

## Context

Step 2.3 scores the four pinned VAEs on the 34 canonical clips before any latent is quantized. The number that belongs next to a classic row is the Phase 1 score against the source y4m: PSNR-Y, PSNR-RGB, SSIM, VMAF, and flicker. Peak VRAM, encode time, decode time, and the time spent inside the encoder's chunk loop have to be stored with that score, or the later throughput table has nothing to read.

The T4 smoke ran HunyuanVideo out of memory with spatial tiling off, then finished once `enable_tiling()` was called. Framewise encoding on that class is already on. Wan2.1, LTX-Video, and Wan2.2 finished on the same card with tiling off. An L4 is the target runtime for the full sweep. A T4 remains the fallback.

The plan asks for a CSV the classic reader can load, with the rate columns empty, because this pass has no bitstream.

## Decision

`notebooks/vae_sweep.ipynb` runs the 136 forwards. The runtime dtype follows the card: bfloat16 from compute capability 8 (an L4 is 8.9), float16 on capability 7 (a T4 is 7.5).

HunyuanVideo calls `enable_tiling()` before the first clip, on every card. The other three keep the class default. A forward that raises an out-of-memory error turns tiling on and retries that clip once. `spatial_tiling`, `framewise_encoding`, `framewise_decoding`, and `retried_with_tiling` are columns. Hunyuan's spatial minimum is 256 pixels on a side, and 832×480 is above that, so the flag being on means the forward is tiled.

Per-chunk time is the mean wall clock of `encoder.forward` during the successful encode, after a CUDA sync. Wan calls that encoder once per group of four frames (`1 + (frames - 1) // 4`).

The table is `docs/results/vae-recon.csv` once the Colab file has been merged. `codec` is the VAE name. `crf`, `bits`, `bpp`, and `kbps` are empty. `preset` is `reconstruction`. The classic columns stay required, and the VAE columns are added when a row carries them, so `docs/results/classic-baselines.csv` still loads. `read_results` accepts an empty rate cell. The run id is `{utc}-{git_sha12}-{config_sha12}` of `configs/vae-recon.yaml`. Torch, Diffusers, the runtime dtype, the GPU name, and the VAE revision are columns on each row.

The notebook writes `MyDrive/LatentCodecBench/results/vae-recon.csv` after every row and skips a `(clip_id, vae)` that is already present. `USE_DRIVE = False` keeps that file at `/content/results/vae-recon.csv`. Each write is flushed and renamed into place, so a dead runtime leaves the previous rows. Canonical clips prepared off Colab use the flat layout `data/clips/{id}.y4m` and can be uploaded to `LatentCodecBench/clips/` before the fetch. `scripts/merge-vae-recon.py` copies an uploaded file into `docs/results/vae-recon.csv`. That file is the L4 sweep: 136 rows, run id `20261020T135251Z-5e6307722225-8af5cb8e9b39`. `lpips` stays empty. The metrics list is PSNR-Y, PSNR-RGB, SSIM, VMAF, and flicker. The sweep config is also packaged at `latentcodecbench/vaes/recon_config.yaml`, because a wheel does not contain `configs/`. The clip manifest is not packaged.

Pixels move through the BT.709 limited-range matrix in `metrics/color.py`. The source y4m becomes RGB in `[-1, 1]`. The decode is converted back, chroma is the rounded mean of each 2x2 block, and `score_y4m` scores that file against the source. The reconstructed y4m is deleted after the score.

## Consequences

A Hunyuan number from this sweep is a tiled reconstruction. The column says so. Comparing it with a later untiled run means reading that flag.

An empty rate cell is a reconstruction, so BD-rate skips it. Step 2.4 is what fills `bits` and `bpp`.

The manifest has to be uploaded to `/content/manifests/clips.yaml`, or cloned, because the wheel does not contain it. The ffmpeg pin is downloaded by the notebook from the same URL and SHA256 as `scripts/install-ffmpeg.sh`.

## Alternatives

**Tile every VAE.** Wan2.1, LTX-Video, and Wan2.2 finished the smoke clip with tiling off. Turning the switch on for them would measure a different forward than the one that fits.

**Leave Hunyuan untiled and require an L4.** The smoke already showed the failure mode on 16 GB. The notebook still targets an L4, and the same file has to run if the runtime handed out is a T4.

**A second CSV schema.** The plan asks the classic reader to load the reconstruction table. Optional columns, and empty rate cells, keep one reader.
