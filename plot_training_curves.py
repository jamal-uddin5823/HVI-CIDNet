"""
Training Loss Curves Visualization Script

Generates publication-quality plots showing:
1. Training loss curves for all models
2. PSNR/SSIM evolution during training
3. Learning rate schedules

This script parses results directly from train.log file.

Usage:
    # Generate training curves
    python plot_training_curves.py

    # Custom log file
    python plot_training_curves.py --train_log=./train.log

    # Custom output directory
    python plot_training_curves.py --output_dir=./figures
"""

import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict

# Use non-interactive backend for HPC
matplotlib.use('Agg')

# Set style for publication-quality figures
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13


def parse_train_log(filepath):
    """Parse training log file

    Expected log format:
        ===> Epoch[N]: Loss: X.XXXX || Learning rate: lr=X.XXXXXe-XX
        ====> Avg.PSNR: X.XX dB
        ====> Avg.SSIM: X.XXXX
    """
    training_data = {
        'baseline': {'epochs': [], 'losses': [], 'psnr': [], 'ssim': [], 'lr': []},
        'face_loss3': {'epochs': [], 'losses': [], 'psnr': [], 'ssim': [], 'lr': []},
        'face_loss5': {'epochs': [], 'losses': [], 'psnr': [], 'ssim': [], 'lr': []},
    }

    if not os.path.exists(filepath):
        print(f"Warning: Log file not found: {filepath}")
        return {}

    # Detect which model section we're in based on context
    current_model = 'baseline'  # Default

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # Find model boundaries based on training commands
    for i, line in enumerate(lines):
        # Detect model switches based on training commands
        if 'face_loss3' in line and ('Training' in line or 'Starting' in line or 'Run' in line):
            current_model = 'face_loss3'
        elif 'face_loss5' in line and ('Training' in line or 'Starting' in line or 'Run' in line):
            current_model = 'face_loss5'
        elif 'baseline' in line and ('Training' in line or 'Starting' in line or 'Run' in line):
            current_model = 'baseline'

        # Parse epoch loss
        match = re.search(r'Epoch\[(\d+)\]:\s+Loss:\s+([\d.]+)\s+\|\|.*?lr=([\d.e+-]+)', line)
        if match:
            epoch = int(match.group(1))
            loss = float(match.group(2))
            lr = float(match.group(3))

            # Avoid duplicate epochs
            if epoch not in training_data[current_model]['epochs']:
                training_data[current_model]['epochs'].append(epoch)
                training_data[current_model]['losses'].append(loss)
                training_data[current_model]['lr'].append(lr)

        # Parse PSNR
        match = re.search(r'Avg\.PSNR:\s+([\d.]+)', line)
        if match:
            psnr = float(match.group(1))
            if training_data[current_model]['epochs']:
                if len(training_data[current_model]['psnr']) < len(training_data[current_model]['epochs']):
                    training_data[current_model]['psnr'].append(psnr)
                else:
                    training_data[current_model]['psnr'][-1] = psnr

        # Parse SSIM
        match = re.search(r'Avg\.SSIM:\s+([\d.]+)', line)
        if match:
            ssim = float(match.group(1))
            if training_data[current_model]['epochs']:
                if len(training_data[current_model]['ssim']) < len(training_data[current_model]['epochs']):
                    training_data[current_model]['ssim'].append(ssim)
                else:
                    training_data[current_model]['ssim'][-1] = ssim

    # Filter out empty models
    return {k: v for k, v in training_data.items() if v['epochs']}


def plot_training_loss_curves(training_data, output_dir):
    """Plot training loss curves for all models"""
    model_labels = {
        'baseline': 'Baseline (λ=0.0)',
        'face_loss3': 'Face Loss (λ=0.3)',
        'face_loss5': 'Face Loss (λ=0.5)'
    }

    colors = {
        'baseline': '#0173B2',
        'face_loss3': '#029E73',
        'face_loss5': '#CC78BC'
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax = axes[0]
    for model in ['baseline', 'face_loss3', 'face_loss5']:
        if model in training_data and training_data[model]['losses']:
            epochs = training_data[model]['epochs']
            losses = training_data[model]['losses']
            ax.plot(epochs, losses, '-', label=model_labels[model],
                   color=colors[model], linewidth=2, alpha=0.8)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('(a) Training Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Learning rate curve
    ax = axes[1]
    for model in ['baseline', 'face_loss3', 'face_loss5']:
        if model in training_data and training_data[model]['lr']:
            epochs = training_data[model]['epochs']
            lrs = training_data[model]['lr']
            ax.plot(epochs, lrs, '-', label=model_labels[model],
                   color=colors[model], linewidth=2, alpha=0.8)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('(b) Learning Rate Schedule')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.suptitle('Training Progress', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'training_loss_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_quality_evolution(training_data, output_dir):
    """Plot PSNR/SSIM evolution during training"""
    model_labels = {
        'baseline': 'Baseline (λ=0.0)',
        'face_loss3': 'Face Loss (λ=0.3)',
        'face_loss5': 'Face Loss (λ=0.5)'
    }

    colors = {
        'baseline': '#0173B2',
        'face_loss3': '#029E73',
        'face_loss5': '#CC78BC'
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # PSNR evolution
    ax = axes[0]
    for model in ['baseline', 'face_loss3', 'face_loss5']:
        if model in training_data and training_data[model]['psnr']:
            epochs = training_data[model]['epochs'][:len(training_data[model]['psnr'])]
            psnr = training_data[model]['psnr']
            if len(epochs) == len(psnr):
                ax.plot(epochs, psnr, 'o-', label=model_labels[model],
                       color=colors[model], linewidth=2, markersize=4, alpha=0.8)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('(a) Validation PSNR Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # SSIM evolution
    ax = axes[1]
    for model in ['baseline', 'face_loss3', 'face_loss5']:
        if model in training_data and training_data[model]['ssim']:
            epochs = training_data[model]['epochs'][:len(training_data[model]['ssim'])]
            ssim = training_data[model]['ssim']
            if len(epochs) == len(ssim):
                ax.plot(epochs, ssim, 'o-', label=model_labels[model],
                       color=colors[model], linewidth=2, markersize=4, alpha=0.8)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('SSIM')
    ax.set_title('(b) Validation SSIM Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.85, 1])

    plt.suptitle('Image Quality Evolution During Training', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'quality_evolution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate training curve visualizations')
    parser.add_argument('--train_log', type=str, default='./train.log',
                       help='Path to training log file')
    parser.add_argument('--output_dir', type=str, default='./figures',
                       help='Output directory for figures')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("Training Curves Visualization")
    print("=" * 70)
    print(f"Train log:        {args.train_log}")
    print(f"Output directory: {args.output_dir}")
    print()

    # Load training data
    training_data = parse_train_log(args.train_log)

    if not training_data:
        print("No training data found in log!")
        print("  Make sure train.log exists and contains training logs.")
        return 1

    print(f"Found training data for {len(training_data)} model(s):")
    for model_name in sorted(training_data.keys()):
        epochs = len(training_data[model_name]['epochs'])
        print(f"  - {model_name}: {epochs} epochs")
    print()

    # Generate plots
    print("Generating plots...")
    plot_training_loss_curves(training_data, args.output_dir)
    plot_quality_evolution(training_data, args.output_dir)

    print()
    print("=" * 70)
    print("Visualization complete!")
    print("=" * 70)
    print(f"\nFigures saved to: {args.output_dir}")

    return 0


if __name__ == '__main__':
    exit(main())
