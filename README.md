# Naifu Diffusion

Naifu Diffusion is the name for this project of finetuning Stable Diffusion on images and captions.

Still under testing, see `config/test.yaml` for any configuration.

Colab example: https://colab.research.google.com/drive/1Xf1tnsP4fu8y5MoYbK1pz08jmyMiTrvv

Currently implemented features:

- [x] Aspect Ratio Bucket and Custom Batch
- [x] Using Hidden States of CLIP’s Penultimate Layer
- [x] Nai-style tag processing
- [x] Extending the Stable Diffusion Token Limit by 3x (beta)

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
python trainer.py --config config/test.yaml

# For multi-gpu
python trainer.py --config config/multi-gpu.yaml

# Disitrubuted
python trainer.py --config config/distributed.yaml
```

Convert checkpoint files to use in SD-based webui

```bash
python scripts/convert_to_sd.py --src /path/last.ckpt --dst /path/last.ckpt
```

## Experiments

Train [LoCon](https://github.com/KohakuBlueleaf/LoCon)

```bash
python trainer.py --config experiment/locon.yaml

## extract 
python experiment/extract_lora.py --src last.ckpt --dst output_locon.pt
```

Train [LoRA](https://arxiv.org/abs/2106.09685)

```bash
python trainer.py --config experiment/lora.yaml

## extract 
python experiment/extract_lora.py --src last.ckpt --dst output_lora.pt
```

Train [Textual Inversion](https://textual-inversion.github.io)

```bash
python trainer.py --config experiment/textual_inversion.yaml
```

Convert any checkpoint to safetensors
```bash
python scripts/sd_to_safetensors.py --src input.ckpt --dst output.safetensors
```
