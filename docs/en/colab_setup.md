# Colab Setup

This guide lists the steps for setting up OpenTAD on Google Colab.

## Environment

- **Python**: 3.10.12
- **PyTorch**: 2.0.1 (CUDA 11.8)

## Install system packages

```bash
sudo apt-get update
sudo apt-get install -y build-essential
```

If your Colab runtime does not have CUDA 11.8, install it with:

```bash
sudo apt-get install -y cuda-toolkit-11-8
```

## Install Python packages

```bash
pip install openmim
mim install mmcv==2.0.1
mim install mmaction2==1.1.0
pip install -r requirements.txt
```

These commands build the local C++/CUDA extensions.

## Training example

Run training with one GPU using `torchrun`:

```bash
torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    tools/train.py configs/actionformer/thumos_i3d.py
```
