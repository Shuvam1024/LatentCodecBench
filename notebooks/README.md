# Notebooks

GPU work runs here, on Colab, not in GitHub Actions.

Actions provides a CPU and, in CI, ffmpeg. That covers classic codecs, metrics, and tests on synthetic clips. Wan, LTX-Video, and HunyuanVideo need a GPU and large weight files. Those runs belong in notebooks so a Colab Pro runtime (L4 or A100) can execute them.

A notebook is a driver, not the source of the implementation.

- Library code stays in `src/latentcodecbench`. A notebook should call it.
- Do not commit checkpoints, datasets, or saved cell output.
- `.ipynb_checkpoints/` is gitignored. So are `/data/`, `/results/`, and `/checkpoints/`.
