# Naifu Diffusion

Naifu Diffusion is the name for this project of finetuning Stable Diffusion on images and captions.

See `test-run.yaml` for any configuration.

Colab example: https://colab.research.google.com/drive/1Xf1tnsP4fu8y5MoYbK1pz08jmyMiTrvv

## Usage

Fulfill deps

```bash
# by conda
conda env create -f environment.yaml
conda activate ldm

# OR by pip
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Start training

```bash
torchrun trainer.py --model_path=/tmp/model --config test-run.yaml
```

## BibTeX

```
@misc{rombach2021highresolution,
      title={High-Resolution Image Synthesis with Latent Diffusion Models}, 
      author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
      year={2021},
      eprint={2112.10752},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

```