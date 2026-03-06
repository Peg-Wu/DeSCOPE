# Installation

## Step 1: Set up a python environment

We recommend creating a virtual Python environment with [Anaconda](https://docs.anaconda.com/free/anaconda/install/linux/):

- Required version: `python >= 3.10`

```bash
conda create -n descope python=3.10
conda activate descope
```

## Step 2: Install pytorch

Install `PyTorch` based on your system configuration. Refer to [PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/).

For the exact command, for example:

- You may choose any version to install, but make sure the PyTorch version is not too old.
- We recommend `torch ≥ 2.6`.

```bash
# Installation Example: torch v2.7.1
# CUDA 11.8
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.6
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
# CUDA 12.8
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

## Step 3: Install deepspeed (optional)

Install `DeepSpeed` based on your system configuration. Refer to [DeepSpeed installation instructions](https://www.deepspeed.ai/tutorials/advanced-install/).

For the exact command, for example:

```bash
pip install deepspeed
```

## Step 4: Install descope and dependencies

To install `descope`, run:

```bash
pip install descopex
```

Or install from `github`:

```bash
git clone https://github.com/Peg-Wu/DeSCOPE.git
cd DeSCOPE
pip install [-e] .
```

Check if installation was successful:

```python
import descope
descope.welcome()
```

