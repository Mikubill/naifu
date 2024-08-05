# Naifu

naifu (or naifu-diffusion) is designed for training generative models with various configurations and features. The code in the main branch of this repository is under development and subject to change as new features are added.

## Installation

To get started with Naifu, follow these steps to install the necessary dependencies:

```bash
# Clone the Naifu repository:
git clone --depth 1 https://github.com/mikubill/naifu

# Install the required Python packages:
cd naifu && pip install -r requirements.txt
```

Make sure you have a compatible version of Python installed (Python 3.9 or above).

## Usage

Naifu provides a flexible and intuitive way to train models using various configurations. To train a model, use the trainer.py script and provide the desired configuration file as an argument.

```bash
python trainer.py --config config/<config_file>

# or (same as --config)
python trainer.py config/<config_file>
```

Replace `<config_file>` with one of the available configuration files listed below.

## Configurations

Choose the appropriate configuration file based on training objectives and environment.

Train SDXL (Stable Diffusion XL) model
```bash
# prepare image data (to latents)
python scripts/encode_latents_xl.py -i <input_path> -o <encoded_path>

# sd_xl_base_1.0_0.9vae.safetensors
python trainer.py config/train_sdxl.yaml

# For huggingface model support
# stabilityai/stable-diffusion-xl-base-1.0
python trainer.py config/train_diffusers.yaml

# use original sgm loss module
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

Train with [Phi-1.5/2](https://huggingface.co/microsoft) model
```bash
python trainer.py config/train_phi2.yaml
```

Train language models ([LLaMA](https://github.com/facebookresearch/llama), [Qwen](https://huggingface.co/Qwen), [Gemma](https://huggingface.co/google) etc)
```bash
# Note that prepare data in sharegpt/chatml format, or define your own dataset in data/text_dataset.py
# See example dataset for reference: function-calling-sharegpt
python trainer.py config/train_general_llm.yaml
```

Train language models with lora or qlora (For example, [Mistral](https://huggingface.co/mistralai))
```bash
python trainer.py config/train_mistral_lora.yaml
```

## Other branches

* sgm - Uses the [sgm](https://github.com/Stability-AI/generative-models) to train SDXL models.
* sd3 - Trainer for SD3 models - use with caution: may produce undesired result
* hydit - Trainer for hunyuan dit models (v1.1 and v1.2)
* main-archived - Contains the original naifu-diffusion code for training Stable Diffusion 1.x models.

For branches without documentation, please follow the installation instructions provided above.
