# Naifu Diffusion

Naifu Diffusion is the name for this project of finetuning Stable Diffusion on images and captions.

This branch is aiming to train stable diffusion model stably with diffusers. To make use of all new features, such as SDXL Training and efficient/experiment strategies, checkout sgm branch.

Colab demo: https://colab.research.google.com/drive/1Xf1tnsP4fu8y5MoYbK1pz08jmyMiTrvv

## Features

The trainer has integrated several features:

* Aspect Ratio Bucket and Custom Batch
* Utilizing Hidden States of CLIP’s Penultimate Layer
* Nai-style Tag Processing (w/ Tag Fliter and Cliper)
* Extending the Stable Diffusion Token Limit by 3x
* Lora/Locon Training
* Min-SNR Weighting Strategy
* Offset Noise and Input Perturbation

## Usage

Clone repo

```bash
git clone https://github.com/Mikubill/naifu-diffusion
cd naifu-diffusion
```

Fulfill deps

```bash
# by conda
conda env create -f environment.yaml
conda activate nd

# OR by pip
pip install -r requirements.txt
```

Start training.

```bash
# test
python trainer.py --config train.yaml
```

## Experiments

Train [LoRA](https://arxiv.org/abs/2106.09685)

```bash
python trainer.py --config experiment/lora.yaml

## extract 
python experiment/extract_lora.py --src last.ckpt
```

Train [LoCon](https://github.com/KohakuBlueleaf/LoCon)

```bash
python trainer.py --config experiment/locon.yaml

## extract 
python experiment/extract_lora.py --src last.ckpt
```

Train [Textual Inversion](https://textual-inversion.github.io)

```bash
python trainer.py --config experiment/textual_inversion.yaml
```

Convert any checkpoint to safetensors
```bash
python scripts/sd_to_safetensors.py --src input.ckpt --dst output.safetensors
```
