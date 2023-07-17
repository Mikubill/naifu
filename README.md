# Naifu Diffusion

Naifu Diffusion is the name for this project of finetuning Stable Diffusion on images and captions.


## Features

The trainer has integrated several features:

* Aspect Ratio Bucket and Custom Batch
* Utilizing Hidden States of CLIP’s Penultimate Layer
* Nai-style Tag Processing (w/ Tag Fliter and Cliper)
* Lora/Locon Training
* Min-SNR Weighting Strategy
* Offset Noise and Input Perturbation

And several experimental features:

* SDXL Training w/ Lora/Locon
* Less vram usage 
* Image and Caption Cache
* Use optimized sgm library instead of diffusers impl

## Usage

Clone repo

```bash
git clone https://github.com/Mikubill/naifu-diffusion
cd naifu-diffusion
```

Start training

```bash
# Fulfill deps
pip install -r requirements.txt

# Start training.
python trainer.py --config train.yaml
```

No need to convert any model: we have done it for you.