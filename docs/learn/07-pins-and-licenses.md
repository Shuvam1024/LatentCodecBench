# Pins, licenses, and loading only the VAE

A reconstruction is a claim about one weight file. The way to name that file is a repo, a git revision, and a subfolder. `main` on Hugging Face is a moving label. A revision is 40 hex characters, and it still means the same commit after somebody uploads a new README. ADR-0008 writes those three strings down for Wan2.1, the Wan2.2 TI2V-5B VAE, LTX-Video, and HunyuanVideo, plus the sha256 of `vae/diffusion_pytorch_model.safetensors` so the pin can be checked against the bytes.

The loader calls the VAE class, not the pipeline:

```python
AutoencoderKLWan.from_pretrained(repo, subfolder="vae", revision=revision)
```

`subfolder="vae"` asks for `vae/config.json` and the VAE weight. The diffusion transformer stays on the hub. That split is the point of this phase. The VAE is the codec: pixels in, a grid out, pixels back. The transformer is the generator that samples the grid. Measuring the generator would answer a different question and download a much larger file.

## The license belongs to the file

The four files are float32 safetensors. The license is not one line for the whole benchmark.

Wan2.1 and Wan2.2 TI2V-5B carry Apache-2.0 on the model card. Wan2.2-T2V-A14B ships the Wan2.1 VAE: the sha256 of the weight file is the same. The benchmark loads the 1.3B repo for that VAE and the TI2V-5B repo for the 48-channel VAE.

LTX-Video's `vae/` file was last changed on 2024-12-11. The model card maps "2B version 0.9" to `ltx-video-2b-v0.9.license.txt`, a RAIL-M license dated 22 November 2024. Permitted Purpose in that file is academic or research use. The same repository also contains the LTXV Open Weights License 0.X, dated 15 April 2025, which the card applies to version 0.9.6 and later. The pinned file is the older one.

Hunyuan's Diffusers weights are `hunyuanvideo-community/HunyuanVideo`, a conversion of `tencent/HunyuanVideo`. The community card does not set a license tag. The parent repository's license is the Tencent Hunyuan Community License Agreement, release date 3 December 2024. It says the agreement does not apply in the European Union, the United Kingdom, or South Korea. Section 5(b) says the works and their outputs are not to be used to improve any other AI model, other than Tencent Hunyuan or a derivative of it.

## What a 16 GB card is being asked to hold

The weight file is the part that can be counted before a run. Peak memory during the forward is larger, and this note does not pretend to have measured it.

| VAE | Float32 file | Half precision | What the class does with time |
| --- | --- | --- | --- |
| Wan2.1 | 0.47 GiB | 0.24 GiB | Encode already walks the clip in groups of 4. Spatial tiling defaults off. |
| HunyuanVideo | 0.92 GiB | 0.46 GiB | Framewise encoding defaults on, in chunks of 16 frames. |
| LTX-Video | 1.56 GiB | 0.78 GiB | Framewise encoding defaults off, so one 33-frame forward holds the whole clip. |
| Wan2.2 TI2V-5B | 2.62 GiB | 1.31 GiB | Same group-of-4 encode as Wan2.1, and a much wider decoder. |

A free Colab T4 has 16 GB and compute capability 7.5. It has float16 tensor cores. The smoke therefore loads float16 on a T4, and bfloat16 on capability 8 and above (L4, A100). The table above is the size of the file. A T4 run of one seeded 832×480×33 clip, float16, measured the forward: Wan2.1 peaked at 4.58 GiB with spatial tiling off, LTX-Video at 2.28 GiB with framewise encoding off, and Wan2.2 TI2V-5B at 5.87 GiB with spatial tiling off. HunyuanVideo ran out of memory with only its default framewise chunks, then finished at 4.19 GiB after spatial tiling was turned on. On a 16 GB card, step 2.3 keeps spatial tiling on for Hunyuan, or moves that VAE to an L4. `docs/results/vae-smoke-t4.csv`.

## Source

ADR-0008, which lists the revisions and the license URLs. Diffusers 0.41.0, tag `v0.41.0`, `from_pretrained(..., subfolder="vae")` on `AutoencoderKLWan`, `AutoencoderKLLTXVideo`, and `AutoencoderKLHunyuanVideo`. The safetensors headers of the four pinned files, read 2026-10-19, are float32.
