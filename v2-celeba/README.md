# Native 1-Bit Diffusion on CelebA (W1A16 Ablation Study)

This directory implements a highly optimized, modular, and resilient PyTorch training and ablation pipeline for a **1-Bit Native Diffusion Model (W1A16)** operating on the RGB CelebA dataset. 

---

## Technical Methodology

Our implementation builds upon the twin theoretical pillars of **Structural Dominance** and **Topological Stability**:

1.  **Structural Dominance (Mean-Centered Weights):**
    Convolutional filters are binarized on-the-fly via the standard Straight-Through Estimator (STE) but are centered per-filter by subtracting the mean first:
    $$\bar{W}_o = W_o - \text{mean}(W_o)$$
    $$\hat{W}_o = \text{sign}(\bar{W}_o) \cdot \text{mean}(|\bar{W}_o|)$$
    This removes isotropic DC offsets, allowing the binary representation to capture the orientation/geometry of the target score field.

2.  **Topological Stability (Pre-Activation):**
    Residual block convolutions are preceded by BatchNorm and activation (`BN -> SiLU -> Conv`). This centers activation inputs around zero, ensuring that the gradient through the sign function does not collapse or suffer from dead neurons.

3.  **High-Precision Boundaries:**
    Input projection ($3 \to 64$ channels) and output projection ($64 \to 3$ channels) remain in full-precision (FP32/FP16) to avoid quantization noise at the boundaries.

---

## Directory Layout

```
v2-celeba/
├── data/
│   └── dataset-samples/       # preprocessed dataset visual sanity check
├── code/
│   ├── config.py              # settings, hyperparams, and paths
│   ├── logger.py              # logging setup & VRAM monitor utilities
│   ├── data_loader.py         # CelebA loader, center crop, and 10k subset
│   ├── models/
│   │   ├── binarization.py    # BitConv2d_Std implementation
│   │   ├── embed.py           # Sinusoidal position embeddings
│   │   ├── blocks.py          # ResBlock16 and ResBlock1Bit
│   │   └── unet.py            # U-Net container classes (FP16 & W1A16)
│   ├── samplers/
│   │   ├── schedule.py        # linear noise schedule variables
│   │   └── ddpm.py            # standard 1000-step DDPM sampler
│   ├── trainers/
│   │   ├── base.py            # AMP mixed precision trainer
│   │   └── ptq.py             # post-training binarization converter
│   ├── test_train.py          # Phase 2 micro-batch overfit executable
│   └── run_ablation.py        # Phase 3 ablation runner script
└── README.md                  # academic documentation
```

---

## Execution Guide

To run the various phases of the ablation study, execute the following commands from the repository root:

### **Phase 1: Data Preparation & Verification**
Load the CelebA dataset (falling back to manual image directories or synthetic procedural faces if download limits are hit), center-crop, downsample to 32x32, and save a batch of 32 sample images to disk.
```bash
python -m v2-celeba.code.data_loader
```
*Outputs saved to:* `v2-celeba/data/dataset-samples/preprocessed_samples.png`

### **Phase 2: Overfitting Micro-Batch (Tiny Test Train)**
Train/overfit a 1-bit weights model (`ResUNet_W1A16`) on a micro-batch of exactly 100 images for 300 epochs. Generates final samples from the overfit checkpoint to verify that 3-channel spatial geometry is successfully captured by the binarized score field.
```bash
python -m v2-celeba.code.test_train
```
*Outputs saved to:* `v2-celeba/outputs/tiny_original.png`, `v2-celeba/outputs/tiny_overfit_samples.png`, and `v2-celeba/checkpoints/w1a16_tiny_overfit.pth`

### **Phase 3: Truncated Ablation Study**
Automate the comparative study across a truncated subset of 10,000 images:
1. Train an **FP16 Baseline** model for 20 epochs.
2. Train a **W1A16 Native** model (binarized weights during training) for 20 epochs.
3. Perform **Post-Training Quantization (PTQ)** on the FP16 Baseline to construct a **W1A16 PTQ** model.
4. Draw 16 samples from each model and save them as comparison grids.
```bash
python -m v2-celeba.code.run_ablation
```
*Outputs saved to:* checkpoints (`.pth`) under `checkpoints/` and grids (`.png`) under `outputs/`.
