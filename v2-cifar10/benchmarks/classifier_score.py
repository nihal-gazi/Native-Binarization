"""
Classifier confidence score for CIFAR-10 diffusion models.

Analogous to the Legibility Score from the MNIST paper: generates samples,
passes them through a pre-trained CIFAR-10 classifier, and reports the
average maximum softmax confidence.

Higher confidence = model generates images that look like real CIFAR-10 classes.

Usage:
    python -m benchmarks.classifier_score \
        --checkpoint checkpoints/fp16/fp16_best.pth \
        --variant fp16 --sampler ddim --steps 50
"""

import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_model
from samplers import DiffusionSchedule, ddpm_sample, ddim_sample


# ---------------------------------------------------------------------------
# CIFAR-10 classifier (simple ResNet-style for judging generated images)
# ---------------------------------------------------------------------------

class CIFAR10Classifier(nn.Module):
    """
    Compact CNN classifier for CIFAR-10 evaluation.
    Trained separately on real CIFAR-10 to serve as a "judge".
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # 32 -> 16

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),  # 16 -> 8

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),  # 8 -> 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train_judge(data_dir='./data', epochs=30, save_path='./checkpoints/cifar10_judge.pth'):
    """Train the CIFAR-10 judge classifier on real data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1),
    ])

    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True,
                                  transform=transform_train)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True,
                                 transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)

    model = CIFAR10Classifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
        scheduler.step()

        # Test accuracy
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                test_correct += (logits.argmax(1) == labels).sum().item()
                test_total += labels.size(0)

        print(f"Epoch {epoch:2d} | Train acc: {100*correct/total:.1f}% | "
              f"Test acc: {100*test_correct/test_total:.1f}%")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Judge classifier saved: {save_path} (test acc: {100*test_correct/test_total:.1f}%)")
    return model


@torch.no_grad()
def evaluate_classifier_score(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load diffusion model
    model = build_model(variant=args.variant).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(sd)
    model.eval()
    print(f"Loaded {args.variant.upper()} from {args.checkpoint}")

    # Load judge classifier
    judge = CIFAR10Classifier().to(device)
    if not os.path.exists(args.judge_path):
        print("Judge classifier not found. Training one...")
        judge = train_judge(data_dir=args.data_dir, save_path=args.judge_path)
    else:
        judge.load_state_dict(torch.load(args.judge_path, map_location=device,
                                          weights_only=True))
    judge.eval()
    print(f"Judge classifier loaded from {args.judge_path}")

    # Schedule
    schedule = DiffusionSchedule(timesteps=1000, device=device)

    # Generate samples
    print(f"Generating {args.n_samples} samples...")
    all_confs = []
    remaining = args.n_samples
    while remaining > 0:
        n = min(args.gen_batch, remaining)
        if args.sampler == 'ddim':
            samples = ddim_sample(model, schedule, n, num_steps=args.steps,
                                   device=device)
        else:
            samples = ddpm_sample(model, schedule, n, device=device)

        logits = judge(samples)
        probs = F.softmax(logits, dim=1)
        max_conf, _ = probs.max(dim=1)
        all_confs.append(max_conf.cpu())
        remaining -= n

    all_confs = torch.cat(all_confs)
    mean_conf = all_confs.mean().item()
    std_conf = all_confs.std().item()

    print(f"\n{'='*50}")
    print(f"Classifier Score ({args.variant.upper()}, {args.sampler.upper()}, "
          f"{args.steps} steps):")
    print(f"  Mean confidence: {mean_conf*100:.2f}% ± {std_conf*100:.2f}%")
    print(f"{'='*50}")

    return mean_conf


def main():
    parser = argparse.ArgumentParser(description='Classifier Confidence Score')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--variant', type=str, required=True,
                        choices=['fp16', 'w1a16', 'w1a1'])
    parser.add_argument('--sampler', type=str, default='ddim',
                        choices=['ddpm', 'ddim'])
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--n_samples', type=int, default=2000)
    parser.add_argument('--gen_batch', type=int, default=128)
    parser.add_argument('--judge_path', type=str,
                        default='./checkpoints/cifar10_judge.pth')
    parser.add_argument('--data_dir', type=str, default='./data')
    args = parser.parse_args()

    evaluate_classifier_score(args)


if __name__ == '__main__':
    main()
