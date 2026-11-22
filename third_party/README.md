# third_party/training

Model code for Wan2.2-Fun-5B-InP (transformer, VAE, umT5 text encoder) and the LoRA network, vendored from the
public release of **Dexterous World Models (DWM)**: <https://github.com/snuvclab/dwm>, directory `training/`, commit
`f62f309f024a3982a7e0a06d5b526c2caa659d37` ("Initial public release"). The DWM repository is distributed under the
Apache License 2.0 (see the `LICENSE` file of that repository).

Only the modules that Ego2Wrist imports (`training.wan.models` for `AutoencoderKLWan3_8`, `WanT5EncoderModel`,
`WanTransformer3DModel`, and `training.wan.utils.lora_utils.create_network`) and their imports are included; the
files are unmodified copies of that commit. Several of them are themselves derived from other projects and keep their
original headers:

- `wan/models/wan_transformer3d.py`, `wan/models/wan_text_encoder.py`, `wan/models/wan_vae3_8.py`,
  `wan/dist/fsdp.py`: Wan2.1 / Wan2.2 (Alibaba Wan Team, <https://github.com/Wan-Video>), Apache License 2.0
- `wan/utils/fm_solvers.py`, `wan/utils/fm_solvers_unipc.py`: adapted from Hugging Face diffusers schedulers
  (Apache License 2.0)
- `wan/utils/lora_utils.py`: based on microsoft/LoRA, cloneofsimo/lora and kohya_ss (see the file header)

`e2w/model.py` puts this directory on `sys.path`, so `import training` resolves here.
