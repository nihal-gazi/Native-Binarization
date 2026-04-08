# Native Binarization - Roadmap

## Phase 1: MNIST Conference Paper (DONE)
- [x] Binary ResUNet architecture (FP16, W1A16, W1A1)
- [x] Mean-centered weight binarization (BitConv2d_Std)
- [x] Pre-activation block ordering (BN -> Act -> Conv)
- [x] DDPM training on MNIST (28x28, grayscale)
- [x] W1A16 hardware ablation (Native vs PTQ)
- [x] FID + Legibility Score benchmarks (3-run averages)
- [x] Conference paper written and paraphrased
- [x] Code available at github.com/nihal-gazi/Native-Binarization

## Phase 2: CIFAR-10 Color Diffusion with DDIM (`v2-cifar10/`)

### 2.1 Architecture
- [ ] Port binary layers (BitConv2d_Std, BitConv2d_BNN, BinaryTanh) from v1
- [ ] Scale U-Net for CIFAR-10: 3-channel input, widths [128, 256, 512], 3 downsamples
- [ ] Build FP16, W1A16, and W1A1 variants
- [ ] Keep boundary layer convention (first/last conv in FP16)

### 2.2 DDIM Sampler
- [ ] Implement DDIM sampling loop (deterministic, eta=0)
- [ ] Support configurable step counts (50, 100, 200, 1000)
- [ ] Keep DDPM sampler for comparison
- [ ] Training stays DDPM loss (no change) -- only inference uses DDIM

### 2.3 Training
- [ ] CIFAR-10 data loader (3x32x32 RGB, normalized to [-1,1])
- [ ] FP16 baseline trainer
- [ ] W1A16 native trainer (BitConv2d_Std)
- [ ] W1A1 native trainer (BitConv2d_BNN + BinaryTanh)
- [ ] PTQ quantizer (binarize trained FP16 model post-hoc)
- [ ] Hyperparameter search: lr, batch size, epochs, noise schedule

### 2.4 Evaluation
- [ ] FID on CIFAR-10 test set (N=10,000 generated, 50,000 real)
- [ ] Inception Score (IS)
- [ ] CIFAR-10 classifier confidence (analogous to Legibility Score)
- [ ] Compare: FP16 vs W1A16-Native vs W1A16-PTQ vs W1A1-Native vs W1A1-PTQ
- [ ] Ablation: DDIM steps (50 vs 100 vs 200 vs 1000)
- [ ] 3-run mean +/- std for all metrics
- [ ] Save sample grids as PNG for paper figures

## Phase 3: Binary Latent Diffusion with VAE (`v3-latent/`)

### 3.1 VAE Architecture
- [ ] Design compact convolutional VAE for CIFAR-10
  - Encoder: 3x32x32 -> latent z (e.g. 4x8x8 or 8x4x4)
  - Decoder: latent z -> 3x32x32
- [ ] FP16 VAE baseline (train first, verify reconstruction quality)
- [ ] Binary VAE (W1A16): BitConv2d_Std in encoder/decoder, FP16 boundaries
- [ ] Binary VAE (W1A1): fully binary encoder/decoder
- [ ] VAE loss: reconstruction (MSE or L1) + KL divergence

### 3.2 Binary Latent Diffusion
- [ ] Train DDPM in VAE's latent space (not pixel space)
- [ ] Binary U-Net operates on latent dims (e.g. 4x8x8)
- [ ] DDIM sampling in latent space, then decode with VAE decoder
- [ ] Full pipeline: encode dataset -> train binary DDIM on latents -> sample -> decode

### 3.3 Experiments
- [ ] FP16 VAE + FP16 Latent DDIM (upper bound)
- [ ] FP16 VAE + W1A16 Latent DDIM (binary denoiser only)
- [ ] W1A16 VAE + W1A16 Latent DDIM (fully binary pipeline)
- [ ] Compare all against pixel-space results from Phase 2
- [ ] Measure: FID, IS, wall-clock time, memory footprint

## Phase 4: Journal Paper

### Key results needed
- [ ] CIFAR-10 pixel-space: native binary matches/approaches FP16
- [ ] CIFAR-10 latent-space: binary VAE + binary DDIM generates recognizable color images
- [ ] DDIM step reduction: show binary + 50-step DDIM beats FP16 + 1000-step DDPM
- [ ] Compute stacked speedup: binary (58x) * DDIM (20x) * latent (10-50x)
- [ ] Real hardware benchmarks (if possible: XNOR kernels or FPGA)

### Paper structure
- [ ] Extend current conference paper or write new journal submission
- [ ] Add CIFAR-10 results section
- [ ] Add latent diffusion section
- [ ] Add DDIM analysis section
- [ ] Formal bounds on structural dominance angular error
- [ ] Compare against BiDM, Q-Diffusion, PTQD
- [ ] Target: IEEE TPAMI, IJCV, or NeurIPS/ICML

## Notes
- Training objective is always DDPM (epsilon prediction) -- DDIM only changes sampling
- Boundary layers (first conv, last conv) always stay FP16
- Start v2 with FP16 CIFAR-10 baseline to verify architecture before binarizing
- If pixel-space CIFAR-10 fails at W1A1, latent-space route becomes mandatory
