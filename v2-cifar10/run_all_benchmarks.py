"""
Master benchmark runner for CIFAR-10 binary diffusion experiments.

Runs FID and Classifier Score for all model variants across different
sampling configurations. Results are printed as a formatted table.

Usage:
    python run_all_benchmarks.py
"""

import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmarks.fid import evaluate_fid
from benchmarks.classifier_score import evaluate_classifier_score


# ---- Configuration ----

# Define all experiments to run
EXPERIMENTS = [
    # (variant, method, checkpoint_path, sampler, steps)
    ('fp16',  'native', 'checkpoints/fp16/fp16_best.pth',           'ddim', 50),
    ('fp16',  'native', 'checkpoints/fp16/fp16_best.pth',           'ddim', 100),
    ('fp16',  'native', 'checkpoints/fp16/fp16_best.pth',           'ddpm', 1000),
    ('w1a16', 'native', 'checkpoints/w1a16/w1a16_best.pth',         'ddim', 50),
    ('w1a16', 'native', 'checkpoints/w1a16/w1a16_best.pth',         'ddim', 100),
    ('w1a16', 'ptq',    'checkpoints/w1a16_ptq/w1a16_ptq.pth',      'ddim', 50),
    ('w1a1',  'native', 'checkpoints/w1a1/w1a1_best.pth',           'ddim', 50),
    ('w1a1',  'native', 'checkpoints/w1a1/w1a1_best.pth',           'ddim', 100),
    ('w1a1',  'ptq',    'checkpoints/w1a1_ptq/w1a1_ptq.pth',        'ddim', 50),
]

N_RUNS = 3  # Number of runs per experiment for mean ± std


def run_benchmark(variant, method, checkpoint, sampler, steps, data_dir='./data'):
    """Run FID and classifier score for one configuration."""

    if not os.path.exists(checkpoint):
        print(f"SKIP: {checkpoint} not found")
        return None

    class Args:
        pass

    # FID
    fid_args = Args()
    fid_args.checkpoint = checkpoint
    fid_args.variant = variant
    fid_args.sampler = sampler
    fid_args.steps = steps
    fid_args.n_gen = 10000
    fid_args.n_real = 10000
    fid_args.gen_batch = 256
    fid_args.data_dir = data_dir
    fid = evaluate_fid(fid_args)

    # Classifier score
    cs_args = Args()
    cs_args.checkpoint = checkpoint
    cs_args.variant = variant
    cs_args.sampler = sampler
    cs_args.steps = steps
    cs_args.n_samples = 2000
    cs_args.gen_batch = 128
    cs_args.judge_path = './checkpoints/cifar10_judge.pth'
    cs_args.data_dir = data_dir
    cs = evaluate_classifier_score(cs_args)

    return {'fid': fid, 'classifier_score': cs}


def main():
    import numpy as np

    results = {}
    for variant, method, ckpt, sampler, steps in EXPERIMENTS:
        key = f"{variant}_{method}_{sampler}_{steps}"
        fids, css = [], []

        for run in range(1, N_RUNS + 1):
            print(f"\n{'='*60}")
            print(f"Run {run}/{N_RUNS}: {key}")
            print(f"{'='*60}")

            result = run_benchmark(variant, method, ckpt, sampler, steps)
            if result is None:
                break
            fids.append(result['fid'])
            css.append(result['classifier_score'])

        if fids:
            results[key] = {
                'variant': variant,
                'method': method,
                'sampler': sampler,
                'steps': steps,
                'fid_mean': float(np.mean(fids)),
                'fid_std': float(np.std(fids)),
                'cs_mean': float(np.mean(css)),
                'cs_std': float(np.std(css)),
            }

    # Print summary table
    print(f"\n\n{'='*80}")
    print("CIFAR-10 Binary Diffusion Benchmark Results")
    print(f"{'='*80}")
    print(f"{'Config':<35} {'FID':>15} {'Classifier Score':>20}")
    print(f"{'-'*80}")
    for key, r in results.items():
        fid_str = f"{r['fid_mean']:.2f} ± {r['fid_std']:.2f}"
        cs_str = f"{r['cs_mean']*100:.2f}% ± {r['cs_std']*100:.2f}%"
        config = f"{r['variant'].upper()} {r['method']} {r['sampler'].upper()}-{r['steps']}"
        print(f"{config:<35} {fid_str:>15} {cs_str:>20}")

    # Save results to JSON
    os.makedirs('./benchmarks', exist_ok=True)
    out_path = f'./benchmarks/results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == '__main__':
    main()
