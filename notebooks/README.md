# Notebooks

GPU work runs here, on Colab, not in GitHub Actions.

Actions provides a CPU and, in CI, ffmpeg. That covers classic codecs, metrics, and tests on synthetic clips. Wan, LTX-Video, and HunyuanVideo need a GPU and large weight files. Those runs belong in notebooks so a Colab Pro runtime (L4 or A100) can execute them.

A notebook is a driver, not the source of the implementation.

- Library code stays in `src/latentcodecbench`. A notebook should call it.
- Do not commit checkpoints, datasets, or saved cell output.
- `.ipynb_checkpoints/` is gitignored. So are `/data/`, `/results/`, and `/checkpoints/`.

`vae_smoke.ipynb` is the step 2.2 load check. It installs this repo on top of the torch Colab already has, loads each pinned VAE from `subfolder="vae"`, and writes `results/vae-smoke.json`. Run it on a GPU runtime. Wan2.1 is the model expected to finish on a free T4. Wan2.2 TI2V-5B is the one most likely to need a larger GPU. The pins and the licenses are ADR-0008.
