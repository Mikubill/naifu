# Naifu

naifu-diffusion (or naifu) is designed for training generative models with various configurations and features. The code in the main branch of this repository is under development and subject to change as new features are added.

Other branches in the repository include:
* sgm - Uses the [sgm](https://github.com/Stability-AI/generative-models) to train SDXL models. 
* main-archived - Contains the original naifu-diffusion code for training Stable Diffusion 1.x models.

## Installation

To install the necessary dependencies, simply run:

```bash
git clone https://github.com/mikubill/naifu-diffusion
pip install -r requirements.txt
```

## Usage

You can train the image model using different configurations by running the `trainer.py` script with the appropriate configuration file.

```bash
python trainer.py --config config/<config_file>
python trainer.py config/<config_file>
```

Replace `<config_file>` with one of the available configuration files listed below.

## Available Configurations

Choose the appropriate configuration file based on training objectives and environment.

Train SDXL (Stable Diffusion XL) model
```bash
# stabilityai/stable-diffusion-xl-base-1.0
python trainer.py config/train.yaml
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
python trainer.py config/train_dpo_hfdataset.yaml
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

## Preparing Datasets

Each configuration file may have different dataset requirements. Make sure to check the specific configuration file for any dataset specifications or requirements.

You can use your dataset directly for training. Simply point the configuration file to the location of your dataset. If you want to reduce the VRAM usage during training, you can encode your dataset to latents using the `encode_latents.py` script.

```bash
# prepare images in input_path
python encode_latents.py -i <input_path> -o <encoded_path>
```