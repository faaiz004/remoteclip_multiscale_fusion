# RemoteCLIP – Multi-Scale & Dual-Branch Extensions

This repo contains two lightweight upgrades to RemoteCLIP:

* **Multi-Scale Encoder** (`src/model/multiscale_encoder.py`)  
  – adds a Transformer head that fuses intermediate ResNet feature maps (Improvement 1).

* **Dual-Branch Patch-Scene Fusion** (`src/model/*` + `src/train/train_fusion_dual_gpu.py`)  
  – combines object-level embeddings from cropped patches with scene-level embeddings (Improvement 2).

All code comes verbatim from the original notebooks; only the module layout has changed.

## Quick start
```bash
conda env create -f environment.yml
conda activate remoteclip-fusion

# (1) Fine-tune multi-scale branch on a single P4 (Colab)
python -m src.train.train_multiscale

# (2) Pre-train patch encoder on xView crops
python -m src.train.train_patch_encoder

# (3) Dual-GPU fusion fine-tune (needs 2×P4 on Kaggle)
python -m src.train.train_fusion_dual_gpu
