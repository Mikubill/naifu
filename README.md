# Naifu-hydit

Trainer for HunyuanDiT models - v1.1 and v1.2 supported

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

To train a model, use the trainer.py script and provide the desired configuration file as an argument.

```bash
python trainer.py --config config/<config_file>

# or (same as --config)
python trainer.py config/<config_file>
```

Replace `<config_file>` with one of the available configuration files listed below.
