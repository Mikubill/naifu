# Naifu Diffusion

Naifu Diffusion is the name for this project of finetuning Stable Diffusion on images and captions.

See `config/test.yaml` for any configuration.

Colab example: https://colab.research.google.com/drive/1Xf1tnsP4fu8y5MoYbK1pz08jmyMiTrvv

## Usage

<details>
      <summary>deployment notes</summary>
      There is no need to prepare datasets and models by default, the script will download automatically.
</details>

Clone repo

```bash
git clone https://github.com/Mikubill/naifu-diffusion
cd naifu-diffusion
```

Fulfill deps

```bash
# by conda
conda env create -f environment.yaml
conda activate nai

# OR by pip
pip install -r requirements.txt
```
Start training.

```bash
# test
torchrun trainer.py --model_path=/tmp/model --config config/test.yaml

# For multi-gpu
torchrun trainer.py --model_path=/tmp/model --config config/multi-gpu.yaml

# Disitrubuted
torchrun trainer.py --model_path=/tmp/model --config config/distributed.yaml
```
