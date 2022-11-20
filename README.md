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
conda activate nd

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

Convert checkpoint files to use in SD-based webui

```bash
python scripts/convert_to_sd.py --src /path/last.ckpt --dst /path/last.ckpt
```