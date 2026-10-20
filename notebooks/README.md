# Notebooks

GPU work runs here, on Colab, not in GitHub Actions.

Actions provides a CPU and, in CI, ffmpeg. That covers classic codecs, metrics, and tests on synthetic clips. Wan, LTX-Video, and HunyuanVideo need a GPU and large weight files. Those runs belong in notebooks so a Colab Pro runtime (L4 or A100) can execute them.

A notebook is a driver, not the source of the implementation.

- Library code stays in `src/latentcodecbench`. A notebook should call it.
- Do not commit checkpoints, datasets, or saved cell output.
- `.ipynb_checkpoints/` is gitignored. So are `/data/`, `/results/`, and `/checkpoints/`.

`vae_smoke.ipynb` is the step 2.2 load check. It installs this repo on top of the torch Colab already has, loads each pinned VAE from `subfolder="vae"`, and writes `results/vae-smoke.json`. The install cell removes Colab's torchao 0.10.0, which is too old for Diffusers 0.41.0. A wheel dropped in `/content` installs the package without a token. The T4 measurement is `docs/results/vae-smoke-t4.csv`. HunyuanVideo needed spatial tiling to fit in 16 GB. The pins and the licenses are ADR-0008.

`vae_sweep.ipynb` is the step 2.3 reconstruction sweep: 34 clips times four VAEs. It uses the same install, fetches the clip set with the manifest hashes, and writes `MyDrive/LatentCodecBench/results/vae-recon.csv` after every row. Canonical clips prepared off Colab (`data/clips/{id}.y4m`) can be uploaded to `LatentCodecBench/clips/` so the fetch skips them. The metrics are the Phase 1 scores against the source. Hunyuan spatial tiling is on from the first forward. An L4 runs bfloat16 and a T4 runs float16. The wheel does not contain `manifests/clips.yaml`; the fetch cell takes that file from `/content/manifests/clips.yaml` or clones it. `scripts/merge-vae-recon.py` copies the downloaded CSV into `docs/results/`. ADR-0009.

`vae_bitrate.ipynb` is the step 2.4 bitrate sweep: the same clips and VAEs, at bit depths 2, 4, 6, 8, 10, and 12. It reuses the reconstruction install and the pinned BtbN ffmpeg n9.0.2 (`LATENTCODECBENCH_FFMPEG`, sha256 checked, `libvmaf` required), reads canonical clips from `LatentCodecBench/clips/` when Drive is mounted, and writes `LatentCodecBench/results/vae-bitrate.csv`. `crf` is the bit depth. `bits` is the zstd file, header included. `entropy_bits` is the unit-normal estimate. `scripts/merge-vae-bitrate.py` copies the CSV into `docs/results/`. ADR-0010.
