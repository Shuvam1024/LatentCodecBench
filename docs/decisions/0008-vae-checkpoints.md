# ADR-0008: Pin the four VAE checkpoints

- Status: Accepted
- Date: 2026-10-19

## Context

Step 2.1 locked the latent shape on random weights. A reconstruction number needs the published weight file, and that file has to stay the same file the next time the notebook runs. Hugging Face `main` moves. The license is a property of the weight file, and the four repos do not share one.

The benchmark measures the VAE. The diffusion transformer is a different model and a much larger download. Diffusers can load the VAE class from `subfolder="vae"`.

The bytes below were read from the safetensors header (first bytes of `vae/diffusion_pytorch_model.safetensors`) and from the Hugging Face file metadata on 2026-10-19. Every header in this set says float32.

## Decision

`latentcodecbench.vaes.checkpoints` pins the repo, the revision, and `subfolder="vae"` for each model. `load_vae` calls `from_pretrained` on the VAE class with that revision. It does not request the transformer.

| Name | Repo and revision | License | File |
| --- | --- | --- | --- |
| wan2_1 | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` @ `0fad780a534b6463e45facd96134c9f345acfa5b` | Apache-2.0, on the model card | 507,591,892 bytes, sha256 `d6e524b3fffede1787a74e81b30976dce5400c4439ba64222168e607ed19e793`, 194 float32 tensors. Weight commit `5af5c01aff2e88e1a9d47bcd3dbba93822597f8e` |
| wan2_2 | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` @ `b8fff7315c768468a5333511427288870b2e9635` | Apache-2.0, on the model card | 2,818,777,808 bytes, sha256 `62cd18f19438e35b32ac63020e2852f566e9b02f46b6cdbd87972a356e3c6f4b`, 196 float32 tensors. Weight commit `4c6ca6c2ded5c79550a3ca25555efc561112891a` |
| ltx | `Lightricks/LTX-Video` @ `8984fa25007f376c1a299016d0957a37a2f797bb` | RAIL-M, `ltx-video-2b-v0.9.license.txt` | 1,676,798,532 bytes, sha256 `265ca87cb5dff5e37f924286e957324e282fe7710a952a7dafc0df43883e2010`, 190 float32 tensors. Weight commit `05be24f065a268e3b7881b63a114dbf52ce68fd1` (2024-12-11) |
| hunyuan | `hunyuanvideo-community/HunyuanVideo` @ `e8c2aaa66fe3742a32c11a6766aecbf07c56e773` | Tencent Hunyuan Community License Agreement | 985,943,868 bytes, sha256 `7c68a6295f9034a88225fbafb1f3258291a08d57a1fdb938233fa57b1b8f4883`, 248 float32 tensors. Weight commit `bb5ecc0b6f0df267dd70ae80b0562301fc23096d` |

Wan2.2-T2V-A14B (`Wan-AI/Wan2.2-T2V-A14B-Diffusers` @ `5be7df9619b54f4e2667b2755bc6a756675b5cd7`) publishes the same VAE bytes as Wan2.1: the sha256 matches. The two `vae/config.json` files differ in `_diffusers_version` (`0.33.0.dev0` on Wan2.1, `0.35.0.dev0` on A14B) and match on the architecture fields, including `z_dim` 16 and `latents_mean`. The benchmark loads the Wan2.1 repo for that VAE.

LTX-Video's model card maps "2B version 0.9" to [`ltx-video-2b-v0.9.license.txt`](https://huggingface.co/Lightricks/LTX-Video/blob/8984fa25007f376c1a299016d0957a37a2f797bb/ltx-video-2b-v0.9.license.txt). That file is a RAIL-M license dated 22 November 2024. Its Permitted Purpose is academic or research use. The weight file's last change is 2024-12-11, which is the date of the diffusers conversion of that 0.9 checkpoint. The [LTXV Open Weights License 0.X](https://huggingface.co/Lightricks/LTX-Video/blob/8984fa25007f376c1a299016d0957a37a2f797bb/LTX-Video-Open-Weights-License-0.X.txt) is in the same repo and, by its own text, applies to versions released since 15 April 2025 (v0.9.6). The card points 0.9.6 and later at that file. The pinned `vae/` file is the 0.9 weight.

`hunyuanvideo-community/HunyuanVideo` is a Diffusers conversion of [`tencent/HunyuanVideo`](https://huggingface.co/tencent/HunyuanVideo). The community card has no license tag. The license recorded here is the [Tencent Hunyuan Community License Agreement](https://huggingface.co/tencent/HunyuanVideo/blob/6204ad6aea1a77ff5aba337c88278bb9500eb37d/LICENSE) on the parent repo at revision `6204ad6aea1a77ff5aba337c88278bb9500eb37d`, HunyuanVideo release date 3 December 2024. The agreement says it does not apply in the European Union, the United Kingdom, or South Korea. Section 5(b) says the works and their outputs are not to be used to improve any other AI model, other than Tencent Hunyuan or a derivative of it.

The stored dtype is float32 for all four. The Colab smoke casts at load: float16 when the GPU's compute capability is below 8 (a T4 is 7.5), bfloat16 from capability 8 upward (L4, A100), and float32 on CPU. CI does not call `load_vae`.

`notebooks/vae_smoke.ipynb` encodes and decodes one seeded 832×480×33 clip through each pin and writes peak VRAM, encode time, decode time, PSNR, and whether tiling was on.

## Consequences

A new weight is a new revision in `checkpoints.py` and in this ADR, checked against the file sha256. The random-init shape tests stay offline.

Wan and Wan2.2 are Apache-2.0. LTX 0.9 is limited to academic or research use. Hunyuan's agreement is limited to its Territory, which excludes the EU, the UK, and South Korea, and it restricts using outputs to train another model.

Weight size is not peak memory. A Colab T4, float16, Python 3.13, encoded and decoded one seeded 832×480×33 clip. Peaks were 4.58 GiB for Wan2.1, 2.28 GiB for LTX-Video, and 5.87 GiB for Wan2.2 TI2V-5B, all with spatial tiling off. HunyuanVideo ran out of memory on the default framewise path and finished at 4.19 GiB after spatial tiling. `docs/results/vae-smoke-t4.csv`. On a 16 GB card, step 2.3 tiles Hunyuan from the start, or runs it on an L4.

The Wan2.2 `vae/config.json` still has `clip_output: false`. That key belonged to the class in the config's Diffusers version, `0.35.0.dev0`. Release 0.41.0 removed the argument (commit `58d2b10`, "fix vae patches"). `from_pretrained` warns and drops the key. Decode still clamps the RGB tensor to `[-1, 1]` after `unpatchify`. The official `Wan2_2_VAE.decode` in Wan-Video/Wan2.2 `wan/modules/vae2_2.py` clamps the same way after the inner decode. The inner `WanVAE_.decode` returns the unclamped tensor. This benchmark stores pixels in `[-1, 1]`, so the dropped key does not move the smoke.

## Alternatives

**Pin `main`.** The notebook would follow whatever the repo uploaded last, and a PSNR from one week would not be the same checkpoint the next week.

**Load the pipeline and read `.vae`.** That downloads the transformer as well. The transformer is out of scope for this phase, and on a free Colab disk it is the download most likely to fail.

**Record the LTX Open Weights License 0.X for `vae/`.** That license covers 0.9.6 and later. The pinned file was last changed in December 2024, and the model card maps 2B version 0.9 to the RAIL-M file. Using the newer license would describe a different checkpoint.
