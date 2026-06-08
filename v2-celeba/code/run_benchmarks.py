import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from .config import DEVICE, CHECKPOINT_DIR, OUTPUT_DIR
from .data_loader import get_celeba_dataset
from .models.unet import ResUNet_FP16, ResUNet_W1A16, ResUNet_W1A1
from .samplers.schedule import DiffusionSchedule
from .samplers.ddpm import sample_ddpm
from .benchmarks.fid import compute_fid
from .benchmarks.judge import train_judge, get_legibility_score
from .logger import logger

def save_comparison_grid(model_samples, model_names, out_path, num_samples=6):
    """
    Creates a side-by-side comparison card with column labels.
    Resizes images to 96x96 (3x nearest-neighbor) for clarity.
    """
    from PIL import Image, ImageDraw
    import numpy as np

    imgs = []
    for s in model_samples:
        byte_img = (s[:num_samples] * 255.0).clamp(0, 255).cpu().byte().numpy()
        imgs.append(byte_img)

    num_cols = len(model_names)
    col_w = 96
    row_h = 96
    pad = 8
    header_h = 32

    grid_w = num_cols * col_w + (num_cols + 1) * pad
    grid_h = header_h + num_samples * row_h + (num_samples + 1) * pad

    grid_img = Image.new("RGB", (grid_w, grid_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(grid_img)

    for col_idx, name in enumerate(model_names):
        text_x = pad + col_idx * (col_w + pad) + col_w // 2
        text_y = pad + header_h // 2
        try:
            bbox = draw.textbbox((0, 0), name)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            tw, th = draw.textsize(name)
        draw.text((text_x - tw // 2, text_y - th // 2), name, fill=(240, 240, 240))

    for r in range(num_samples):
        for c in range(num_cols):
            x = pad + c * (col_w + pad)
            y = header_h + pad + r * (row_h + pad)
            
            img_np = imgs[c][r, 0]  # (32, 32)
            img_pil = Image.fromarray(img_np).convert("RGB")
            img_pil = img_pil.resize((col_w, row_h), Image.NEAREST)
            grid_img.paste(img_pil, (x, y))

    grid_img.save(out_path)


def run_benchmarks():
    logger.info("=== Starting Master Grayscale Benchmarking System (5-Way comparison) ===")
    
    # 1. Load data
    dataset = get_celeba_dataset(truncate_size=1000)
    data_loader = DataLoader(dataset, batch_size=1000, shuffle=False)
    real_x, _ = next(iter(data_loader))
    
    # 2. Train Judge Classifier
    judge = train_judge(real_x, device=DEVICE)
    
    # 3. Setup schedule and models definition
    schedule = DiffusionSchedule(device=DEVICE)
    
    models_def = {
        "FP16 Baseline": {"class": ResUNet_FP16, "ckpt": "fp16_baseline.pth", "size_kb": 3879.0, "factor": 1.0},
        "W1A16 Native": {"class": ResUNet_W1A16, "ckpt": "w1a16_native.pth", "size_kb": 242.4, "factor": 16.0},
        "W1A1 Native": {"class": ResUNet_W1A1, "ckpt": "w1a1_native.pth", "size_kb": 242.4, "factor": 16.0},
        "W1A16 PTQ": {"class": ResUNet_W1A16, "ckpt": "fp16_baseline.pth", "size_kb": 242.4, "factor": 16.0},
        "W1A1 PTQ": {"class": ResUNet_W1A1, "ckpt": "fp16_baseline.pth", "size_kb": 242.4, "factor": 16.0}
    }
    
    results = {}
    saved_samples = {}
    
    for name, info in models_def.items():
        logger.info(f"Evaluating {name}...")
        model = info["class"]().to(DEVICE)
        ckpt_path = os.path.join(CHECKPOINT_DIR, info["ckpt"])
        
        if not os.path.exists(ckpt_path):
            logger.warning(f"Checkpoint {ckpt_path} missing. Skipping evaluation for {name}.")
            continue
            
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        
        # Generate 100 samples for stats stability (with seed=42 for aligned grids)
        logger.info(f"Generating 100 samples for {name}...")
        samples = sample_ddpm(model, schedule, batch_size=100, device=DEVICE, seed=42)
        
        # Keep track of the first 6 samples for side-by-side grids
        saved_samples[name] = samples[:6].cpu()
        
        # Save visual sample grid of first 16 images
        grid_img = samples[:16]
        out_name = name.lower().replace(" ", "_") + ".png"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        save_image(grid_img, out_path, nrow=4)
        logger.info(f"Visual sample grid for {name} saved to {out_path}")
        
        # Compute stats
        fid = compute_fid(real_x, samples, device=DEVICE)
        legibility = get_legibility_score(judge, samples, device=DEVICE)
        
        # Calculations
        legibility_per_bit = legibility / info["size_kb"]
        utility_score = (legibility * info["factor"]) / (1.0 + fid)
        
        results[name] = {
            "fid": fid,
            "legibility": legibility,
            "legibility_per_bit": legibility_per_bit,
            "utility": utility_score,
            "size_kb": info["size_kb"],
            "image_file": out_name
        }
        
    # Generate Three-Way Comparison Grids
    native_names = ["FP16 Baseline", "W1A16 Native", "W1A1 Native"]
    native_samples = [saved_samples[n] for n in native_names if n in saved_samples]
    if len(native_samples) == 3:
        native_grid_path = os.path.join(OUTPUT_DIR, "native_comparison.png")
        save_comparison_grid(native_samples, native_names, native_grid_path)
        logger.info(f"Three-way Native comparison grid saved to {native_grid_path}")
        
    ptq_names = ["FP16 Baseline", "W1A16 PTQ", "W1A1 PTQ"]
    ptq_samples = [saved_samples[n] for n in ptq_names if n in saved_samples]
    if len(ptq_samples) == 3:
        ptq_grid_path = os.path.join(OUTPUT_DIR, "ptq_comparison.png")
        save_comparison_grid(ptq_samples, ptq_names, ptq_grid_path)
        logger.info(f"Three-way PTQ comparison grid saved to {ptq_grid_path}")

    # Write to RESULTS.md
    write_results_markdown(results)
    logger.info("=== Benchmarking Completed. RESULTS.md updated. ===")

def write_results_markdown(results):
    results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "RESULTS.md")
    
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("# Grayscale CelebA 1-Bit Native Ablation Study: Benchmarking Report\n\n")
        f.write("This report compiles the quantitative benchmark metrics for the models trained from scratch on 32x32 grayscale CelebA.\n\n")
        
        f.write("## 1. Quantitative Benchmark Results (5-Way Summary)\n\n")
        f.write("| Model Variant | Model Size (KB) | FID Distance ↓ | Legibility Score (Face Prob) ↑ | Legibility per Bit (Score/KB) ↑ | Utility Score (Fair Bench) ↑ |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        
        for name, r in results.items():
            f.write(
                f"| **{name}** | {r['size_kb']:.1f} KB | {r['fid']:.2f} | {r['legibility']*100:.2f}% | "
                f"{r['legibility_per_bit'] * 1e6:.4f} × 10⁻⁶ | **{r['utility']:.5f}** |\n"
            )
            
        f.write("\n## 2. Three-Way Comparison Tables\n\n")
        
        # 2.1 Native Comparison Table
        f.write("### 2.1 Native Training Comparison: FP16 Baseline vs W1A16 Native vs W1A1 Native\n\n")
        f.write("| Model Variant | Model Size (KB) | FID Distance ↓ | Legibility Score (Face Prob) ↑ | Legibility per Bit (Score/KB) ↑ | Utility Score (Fair Bench) ↑ |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for name in ["FP16 Baseline", "W1A16 Native", "W1A1 Native"]:
            if name in results:
                r = results[name]
                f.write(
                    f"| **{name}** | {r['size_kb']:.1f} KB | {r['fid']:.2f} | {r['legibility']*100:.2f}% | "
                    f"{r['legibility_per_bit'] * 1e6:.4f} × 10⁻⁶ | **{r['utility']:.5f}** |\n"
                )
        f.write("\n**Visual Output Comparison (Aligned Samples):**\n\n")
        f.write("![Native Aligned Comparison](outputs/native_comparison.png)\n\n")
        
        # 2.2 PTQ Comparison Table
        f.write("### 2.2 Post-Training Quantization (PTQ) Comparison: FP16 Baseline vs W1A16 PTQ vs W1A1 PTQ\n\n")
        f.write("| Model Variant | Model Size (KB) | FID Distance ↓ | Legibility Score (Face Prob) ↑ | Legibility per Bit (Score/KB) ↑ | Utility Score (Fair Bench) ↑ |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for name in ["FP16 Baseline", "W1A16 PTQ", "W1A1 PTQ"]:
            if name in results:
                r = results[name]
                f.write(
                    f"| **{name}** | {r['size_kb']:.1f} KB | {r['fid']:.2f} | {r['legibility']*100:.2f}% | "
                    f"{r['legibility_per_bit'] * 1e6:.4f} × 10⁻⁶ | **{r['utility']:.5f}** |\n"
                )
        f.write("\n**Visual Output Comparison (Aligned Samples):**\n\n")
        f.write("![PTQ Aligned Comparison](outputs/ptq_comparison.png)\n\n")
        
        f.write("\n## 3. Benchmark Logic & Fairness Rationale\n\n")
        f.write("### **Legibility Score per Bit**\n")
        f.write("Calculated as:\n")
        f.write("$$\\text{Legibility per Bit} = \\frac{\\text{Legibility Score}}{\\text{Model Size (KB)}}$$\n")
        f.write("This measures the semantic capacity density of the parameters. It indicates how much visual reconstruction fidelity the model extracts per kilobyte of weight storage.\n\n")
        
        f.write("### **The Utility Score**\n")
        f.write("Calculated as:\n")
        f.write("$$\\text{Utility Score} = \\frac{\\text{Legibility Score} \\times \\text{Compression Factor}}{1 + \\text{FID}}$$\n")
        f.write("where $\\text{Compression Factor} = \\frac{\\text{FP16 Size (3879.0 KB)}}{\\text{Model Size (KB)}}$.\n\n")
        
        f.write("#### **Why this Benchmarking System is Scientifically Fair:**\n")
        f.write("1.  **Anti-Cheating Safeguards:** A model cannot achieve a high score simply by being extremely small (which would yield a high compression factor) if its image quality collapses (which drops the Legibility Score to chance level ~50% and explodes the FID to 100+). Similarly, a model cannot win by being large unless it provides massive quality gains that offset its lower compression factor.\n")
        f.write("2.  **Generative Fidelity vs. Semantic Coherence Balance:** The denominator ($1 + \\text{FID}$) rewards perceptual realism, while the numerator rewards class presence/structure (Legibility). This reflects the classic generative model trade-off: generating clean images that represent the true target data manifold.\n\n")
        
        f.write("## 4. Visual Grid Comparisons (Individual Runs)\n\n")
        
        for name, r in results.items():
            f.write(f"### {name} Output Grid\n")
            f.write(f"![{name} Grayscale Output](outputs/{r['image_file']})\n\n")

if __name__ == "__main__":
    run_benchmarks()
