# Native Binarization — Project Context for Claude Code

## What This Project Is
Research project: **"One Bit is All You Need to Diffuse"** — training binary (1-bit) neural networks natively for diffusion-based image generation, without teacher models or distillation. The key insight is that binary weights trained from scratch (native binarization) drastically outperform post-training quantization (PTQ).

**Paper**: `v1-mnist-paper/paper.tex` (conference paper, completed)
**GitHub**: https://github.com/nihal-gazi/Native-Binarization

---

## Repo Structure

```
Native-Binarization/
├── v1-mnist/              # Phase 1 (DONE) — MNIST grayscale diffusion
│   ├── models/architectures.py   # All model definitions (FP16, W1A16, W1A1)
│   ├── code/Trainers/            # Training scripts
│   ├── code/Benchmarks/          # FID + Legibility evaluation
│   ├── code/Quantizers/          # PTQ: FP16 → binary conversion
│   ├── assets/                   # Generated sample images
│   └── run_all_benchmarks.py
│
├── v1-mnist-paper/        # Conference paper (IEEEtran, completed)
│   ├── paper.tex
│   └── references.bib
│
├── v2-cifar10/            # Phase 2 (IN PROGRESS) — CIFAR-10 color diffusion + DDIM
│   ├── models/
│   │   ├── layers.py      # Binary layers: BitConv2d_Std, BitConv2d_BNN, BinaryTanh
│   │   └── unet.py        # ResUNet variants: FP16, W1A16, W1A1 (3ch, [128,256,512])
│   ├── samplers/
│   │   ├── schedule.py    # DiffusionSchedule (linear beta, all derived quantities)
│   │   ├── ddpm.py        # Standard 1000-step DDPM sampler
│   │   └── ddim.py        # DDIM sampler (50/100/200 steps, eta=0 deterministic)
│   ├── trainers/
│   │   ├── train.py       # Unified trainer (--variant fp16/w1a16/w1a1)
│   │   └── quantize.py    # PTQ converter (FP16 → W1A16 or W1A1)
│   ├── benchmarks/
│   │   ├── fid.py         # FID using InceptionV3 features
│   │   └── classifier_score.py  # CIFAR-10 classifier confidence (+ judge trainer)
│   ├── generate.py        # Sample generation + grid visualization
│   └── run_all_benchmarks.py    # Master benchmark runner (3-run averages)
│
├── v3-latent/             # Phase 3 (TODO) — Binary VAE + Binary Latent Diffusion
│   ├── vae/               # Binary VAE (encoder/decoder)
│   └── diffusion/         # Latent-space DDIM with binary U-Net
│
├── TODO.md                # Detailed 4-phase roadmap with checkboxes
├── requirements.txt       # torch, torchvision, scipy, numpy, matplotlib
└── .gitignore
```

---

## Key Technical Concepts

### Binarization Variants
| Variant | Weights | Activations | Conv Layer | Activation Fn |
|---------|---------|-------------|------------|---------------|
| **FP16** | Float32/16 | Float32/16 | nn.Conv2d | SiLU |
| **W1A16** | 1-bit (binary) | Float16 | BitConv2d_Std (mean-centered) | SiLU |
| **W1A1** | 1-bit | 1-bit | BitConv2d_BNN (no centering) | BinaryTanh (sign) |

### Core Design Principles
1. **Boundary convention**: First conv (input) and last conv (output) are ALWAYS full-precision
2. **Pre-activation ordering**: BN → Activation → Conv (not Conv → BN → Act)
3. **Structural Dominance**: Mean-centered binarization preserves weight direction geometry
4. **STE (Straight-Through Estimator)**: `(w_bin - w).detach() + w` trick for gradient flow
5. **Training is always DDPM** (epsilon prediction MSE loss) — DDIM only changes sampling

### DDIM vs DDPM
- Same training loss — only sampling differs
- DDIM: deterministic, uses timestep subsequence (50 steps instead of 1000)
- Critical for binary models: fewer steps = less error accumulation
- Speedup stacks: binary (58x) × DDIM (20x) × latent (10-50x)

---

## Current Status — What To Work On

### Phase 2 code is WRITTEN but UNTESTED (no GPU available on this machine)
All v2-cifar10 code has been written and needs:
1. **Verify imports and forward passes** work on a GPU machine
2. **Train FP16 baseline first** to confirm architecture works:
   ```bash
   cd v2-cifar10
   python -m trainers.train --variant fp16 --epochs 200 --batch_size 128 --lr 1e-4
   ```
3. **Then train binary variants**:
   ```bash
   python -m trainers.train --variant w1a16 --epochs 300 --batch_size 128 --lr 1e-4
   python -m trainers.train --variant w1a1  --epochs 300 --batch_size 128 --lr 5e-5
   ```
4. **PTQ baselines**:
   ```bash
   python -m trainers.quantize --source checkpoints/fp16/fp16_best.pth --target w1a16
   python -m trainers.quantize --source checkpoints/fp16/fp16_best.pth --target w1a1
   ```
5. **Run benchmarks**: `python run_all_benchmarks.py`

### Phase 3 is scaffold only — needs implementation
See TODO.md Phase 3 for the full plan. Key: binary VAE + binary latent DDIM.

---

## V1 Benchmark Results (MNIST, for reference)

| Model | FID (↓) | Legibility (↑) |
|-------|---------|-----------------|
| FP16 Native | 29.60 ± 0.23 | 90.79 ± 0.51% |
| W1A16 Native | 24.24 ± 1.15 | 89.45 ± 0.73% |
| W1A16 PTQ | 191.94 ± 1.37 | 64.92 ± 1.16% |
| W1A1 Native | 83.98 ± 1.66 | 81.02 ± 0.54% |
| W1A1 PTQ | 223.00 ± 1.89 | 63.04 ± 0.74% |

**Key finding**: Native W1A16 *beats* FP16 on FID (24.24 vs 29.60), while PTQ collapses.

---

## V2 Architecture Details (CIFAR-10)

- **Input**: 3×32×32 RGB, normalized to [-1, 1]
- **Channel widths**: [128, 256, 512] (wider than v1's [64, 128, 256])
- **Downsample path**: 32→16→8→4 (3 levels + bottleneck)
- **Time embedding**: Sinusoidal(64) → Linear(256) → Act → Linear(256)
- **Skip connections**: Concatenation (not addition)
- **Noise schedule**: Linear β from 1e-4 to 0.02, T=1000 steps
- **DDIM**: Uniform timestep subsequence, eta=0 (deterministic)

---

## Important Notes
- This is a research project aiming for a journal paper (IEEE TPAMI / NeurIPS / ICML)
- Always use 3-run mean ± std for benchmark numbers
- CIFAR-10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- If W1A1 pixel-space CIFAR-10 fails badly, the latent-space route (Phase 3) becomes mandatory
- The judge classifier for classifier_score needs to be trained once before benchmarking
