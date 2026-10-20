# RETBind

PyTorch training code for **RETBind**, a self-supervised multimodal foundation model for retinal image analysis using paired **OCT** and **UWF fundus** images.

## Overview

RETBind learns joint retinal representations from paired OCT and UWF images with a shared Vision Transformer encoder and a masked reconstruction objective.

## Installation
Clone the repository and install the required packages:
```bash
pip install -r requirements.txt
```

## Dataset Structure
Prepare the dataset in the following format:
```
data/
└── {patient_id}/
    └── {date}/
        └── {eye}/
            ├── uwf.JPG
            └── oct.JPG
```
- `{patient_id}`: patient identifier
- `{date}`: acquisition date
- `{eye}`: `L` or `R`
- `uwf.JPG`: ultra-widefield fundus image
- `oct.JPG`: OCT image

## Training
### Single GPU
Run training on a single GPU:
```bash
python train.py --config configs/train.yaml
```
### Multi-GPU
Run distributed training with `torchrun`:
```bash
torchrun --nproc_per_node=4 train.py --config configs/train.yaml
```
Adjust `--nproc_per_node` to the number of GPUs available.
