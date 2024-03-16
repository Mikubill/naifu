# Naifu

naifu (or naifu-diffusion) is designed for training generative models with various configurations and features. The code in the main branch of this repository is under development and subject to change as new features are added.

## Installation

To get started with Naifu, follow these steps to install the necessary dependencies:
Clone the Naifu repository:
```bash
git clone --depth 1 https://github.com/mikubill/naifu
```

Navigate to the cloned repository:
```bash
cd naifu
```

Install the required Python packages:
```bash
pip install -r requirements.txt
```

Make sure you have a compatible version of Python installed (Python 3.6 or above) before running the installation command.

## Usage

Naifu provides a flexible and intuitive way to train models using various configurations. To train a model, use the trainer.py script and provide the desired configuration file as an argument.

```bash
python trainer.py --config config/<config_file>
python trainer.py config/<config_file>
```

Replace `<config_file>` with one of the available configuration files listed below.

## Configurations

Choose the appropriate configuration file based on training objectives and environment.

Train SDXL (Stable Diffusion XL) model
```bash
# stabilityai/stable-diffusion-xl-base-1.0
python trainer.py config/train_sdxl.yaml

# use original sgm denoiser and loss weighting
python trainer.py config/train_sdxl_original.yaml
```

Train SDXL refiner (Stable Diffusion XL refiner) model
```bash
# stabilityai/stable-diffusion-xl-refiner-1.0
python trainer.py config/train_refiner.yaml
```

Train original Stable Diffusion 1.4 or 1.5 model
```bash
# runwayml/stable-diffusion-v1-5
# Note: will save in diffusers format
python trainer.py config/train_sd15.yaml
```

Train SDXL model with diffusers backbone
```bash
# stabilityai/stable-diffusion-xl-base-1.0
# Note: will save in diffusers format
python trainer.py config/train_diffusers.yaml
```

Train SDXL model with LyCORIS.
```bash
# Based on the work available at KohakuBlueleaf/LyCORIS
pip install lycoris_lora toml
python trainer.py config/train_lycoris.yaml
```

Use fairscale strategy for distributed data parallel sharded training
```bash
pip install fairscale
python trainer.py config/train_fairscale.yaml
```

Train SDXL model with Diffusion DPO  
Paper: Diffusion Model Alignment Using Direct Preference Optimization ([arxiv:2311.12908](https://arxiv.org/abs/2311.12908))
```bash
# dataset: yuvalkirstain/pickapic_v2
# Be careful tuning the resolution and dpo_betas!
# will save in diffusers format
python trainer.py config/train_dpo_diffusers.yaml # diffusers backend
python trainer.py config/train_dpo.yaml # sgm backend
```

Train Pixart-Alpha model  
Paper: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis ([arxiv:2310.00426](https://arxiv.org/abs/2310.00426))
```bash
# PixArt-alpha/PixArt-XL-2-1024-MS
python trainer.py config/train_pixart.yaml
```

Train SDXL-LCM model  
Paper: Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference ([arxiv:2310.04378](https://arxiv.org/abs/2310.04378))
```bash
python trainer.py config/train_lcm.yaml
```

Train StableCascade model ([Sai](https://github.com/Stability-AI/StableCascade/))
```bash
# currently only stage_c (w/ or w/o text encoder)
python trainer.py config/train_cascade_stage_c.yaml
```

Train GPT2 model
```bash
# currently only stage_c (w/ or w/o text encoder)
python trainer.py config/train_gpt2.yaml
```

## Preparing Datasets

Each configuration file may have different dataset requirements. Make sure to check the specific configuration file for any dataset specifications or requirements.

You can use your dataset directly for training. Simply point the configuration file to the location of your dataset. If you want to reduce the VRAM usage during training, you can encode your dataset to latents using the `encode_latents.py` script.

```bash
# prepare images in input_path
python encode_latents.py -i <input_path> -o <encoded_path>
```

## Other branches

* sgm - Uses the [sgm](https://github.com/Stability-AI/generative-models) to train SDXL models. 
* main-archived - Contains the original naifu-diffusion code for training Stable Diffusion 1.x models.
