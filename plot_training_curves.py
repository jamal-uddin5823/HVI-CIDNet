"""
Training Curves Visualization Script for Thesis

Generates publication-quality plots showing:
1. Total loss curves over training for all models
2. Individual loss component curves (L1, SSIM, Perceptual, Edge, Face Loss)
3. Validation metrics (PSNR, SSIM, LPIPS) over epochs

This script parses training metrics from markdown files generated during training.

Usage:
    # Plot all training curves from multilevel training
    python plot_training_curves.py --results_dir=./results/training

    # Plot specific models only
    python plot_training_curves.py --models=baseline,face_loss3,face_loss5

    # Custom output directory
    python plot_training_curves.py --output_dir=./figures
"""

import os
import re
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from collections import defaultdict

# Use non-interactive backend for server environments
matplotlib.use('Agg')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.labelsize'] = 11
matplotlib.rcParams['axes.titlesize'] = 12
matplotlib.rcParams['xtick.labelsize'] = 9
matplotlib.rcParams['ytick.labelsize'] = 9
matplotlib.rcParams['legend.fontsize'] = 9
matplotlib.rcParams['figure.titlesize'] = 13


class TrainingMetricsParser:
    """Parse training metrics from markdown files"""

    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.metrics = defaultdict(lambda: defaultdict(list))

    def parse_markdown_file(self, filepath):
        """Parse a metrics markdown file

        Expected format:
        dataset: lfw/
        lr: 0.0001
        ...
        | Epochs | PSNR | SSIM | LPIPS |
        |--------|------|------|-------|
        | 10 | 20.5 | 0.85 | 0.15 |
        ...
        """
        if not os.path.exists(filepath):
            return None

        metrics = {
            'epochs': [],
            'psnr': [],
            'ssim': [],
            'lpips': []
        }

        with open(filepath, 'r') as f:
            content = f.read()

        # Extract hyperparameters
        lr_match = re.search(r'lr:\s*([\d.]+)', content)
        if lr_match:
            metrics['lr'] = float(lr_match.group(1))

        batch_match = re.search(r'batch size:\s*(\d+)', content)
        if batch_match:
            metrics['batch_size'] = int(batch_match.group(1))

        fr_weight_match = re.search(r'FR_weight:\s*([\d.]+)', content)
        if fr_weight_match:
            metrics['fr_weight'] = float(fr_weight_match.group(1))
        else:
            metrics['fr_weight'] = 0.0

        # Parse the table
        in_table = False
        for line in content.split('\n'):
            if '| Epochs |' in line:
                in_table = True
                continue
            if in_table and line.startswith('|'):
                parts = [p.strip() for p in line.split('|')[1:-1]]
                if len(parts) >= 4:
                    try:
                        epoch = int(parts[0])
                        psnr = float(parts[1])
                        ssim = float(parts[2])
                        lpips = float(parts[3])

                        metrics['epochs'].append(epoch)
                        metrics['psnr'].append(psnr)
                        metrics['ssim'].append(ssim)
                        metrics['lpips'].append(lpips)
                    except (ValueError, IndexError):
                        continue

        return metrics if metrics['epochs'] else None

    def parse_json_file(self, filepath):
        """Parse a metrics JSON file (loss history)

        Expected format:
        {
            "loss_history": {
                "total": [0.5, 0.45, ...],
                "l1": [0.3, 0.28, ...],
                "ssim": [0.1, 0.09, ...],
                ...
            },
            "epochs": [1, 2, ...],
            "validation": {
                "psnr": [20.5, 21.0, ...],
                ...
            }
        }
        """
        if not os.path.exists(filepath):
            return None

        with open(filepath, 'r') as f:
            try:
                data = json.load(f)
                return data
            except json.JSONDecodeError:
                return None

    def discover_metrics(self):
        """Find all metrics files in the results directory"""
        metrics_files = []

        # Find markdown files
        for md_file in self.results_dir.glob('metrics*.md'):
            metrics_files.append(('markdown', md_file))

        # Find JSON files
        for json_file in self.results_dir.glob('loss_history*.json'):
            metrics_files.append(('json', json_file))

        return metrics_files

    def load_all_metrics(self):
        """Load all available metrics"""
        results = {}

        # Try to load from model-specific directories
        for model_dir in ['baseline', 'face_loss3', 'face_loss5']:
            model_path = self.results_dir / model_dir
            if model_path.exists():
                # Check for JSON loss history
                json_file = model_path / 'loss_history.json'
                if json_file.exists():
                    data = self.parse_json_file(json_file)
                    if data:
                        results[model_dir] = data

        # Also check for markdown files
        for md_file in self.results_dir.glob('metrics*.md'):
            data = self.parse_markdown_file(md_file)
            if data:
                # Try to determine model from filename or FR_weight
                fr_weight = data.get('fr_weight', 0.0)
                if fr_weight == 0.0:
                    model_name = 'baseline'
                elif fr_weight == 0.3:
                    model_name = 'face_loss3'
                elif fr_weight == 0.5:
                    model_name = 'face_loss5'
                else:
                    model_name = f'fr_weight_{fr_weight}'

                if model_name not in results:
                    results[model_name] = {}
                results[model_name]['validation'] = data

        return results


def plot_loss_curves(metrics_data, output_dir):
    """Plot total loss curves for all models"""
    if not metrics_data:
        print("  No loss history data found, skipping loss curves")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'baseline': '#1f77b4', 'face_loss3': '#2ca02c', 'face_loss5': '#d62728'}
    labels = {'baseline': 'Baseline (FR=0.0)', 'face_loss3': 'Face Loss 3 (FR=0.3)', 'face_loss5': 'Face Loss 5 (FR=0.5)'}

    for model_name, data in metrics_data.items():
        if 'loss_history' in data and 'total' in data['loss_history']:
            losses = data['loss_history']['total']
            epochs = data.get('epochs', list(range(1, len(losses) + 1)))

            # Downsample if too many points
            if len(losses) > 100:
                step = len(losses) // 100
                epochs = epochs[::step]
                losses = losses[::step]

            ax.plot(epochs, losses, label=labels.get(model_name, model_name),
                   color=colors.get(model_name, None), linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Training Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'training_loss_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_loss_components(metrics_data, output_dir):
    """Plot individual loss components for each model"""
    if not metrics_data:
        print("  No loss component data found, skipping component plots")
        return

    loss_components = ['l1', 'ssim', 'perceptual', 'edge', 'face']

    for model_name, data in metrics_data.items():
        if 'loss_history' not in data:
            continue

        loss_hist = data['loss_history']

        # Find which components are available
        available_components = [c for c in loss_components if c in loss_hist]

        if not available_components:
            continue

        n_components = len(available_components)
        n_cols = min(3, n_components)
        n_rows = (n_components + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_components == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)

        axes = axes.flatten() if n_components > 1 else [axes]

        for idx, component in enumerate(available_components):
            ax = axes[idx]
            losses = loss_hist[component]
            epochs = data.get('epochs', list(range(1, len(losses) + 1)))

            # Downsample if needed
            if len(losses) > 100:
                step = len(losses) // 100
                epochs = epochs[::step]
                losses = losses[::step]

            ax.plot(epochs, losses, linewidth=2, color='#1f77b4')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(f'{component.capitalize()} Loss')
            ax.set_title(f'{component.capitalize()} Loss ({model_name})')
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(len(available_components), len(axes)):
            axes[idx].set_visible(False)

        label = model_name.replace('_', ' ').title()
        fig.suptitle(f'Loss Components - {label}', fontsize=14, fontweight='bold')

        plt.tight_layout()
        output_path = os.path.join(output_dir, f'loss_components_{model_name}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()


def plot_validation_metrics(metrics_data, output_dir):
    """Plot validation metrics (PSNR, SSIM, LPIPS) over epochs"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = {'baseline': '#1f77b4', 'face_loss3': '#2ca02c', 'face_loss5': '#d62728'}
    labels = {'baseline': 'Baseline (FR=0.0)', 'face_loss3': 'Face Loss 3 (FR=0.3)', 'face_loss5': 'Face Loss 5 (FR=0.5)'}

    # PSNR
    ax = axes[0]
    for model_name, data in metrics_data.items():
        if 'validation' in data and 'psnr' in data['validation']:
            psnr = data['validation']['psnr']
            epochs = data['validation'].get('epochs', list(range(1, len(psnr) + 1)))
            ax.plot(epochs, psnr, 'o-', label=labels.get(model_name, model_name),
                   color=colors.get(model_name, None), linewidth=2, markersize=6)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Validation PSNR')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # SSIM
    ax = axes[1]
    for model_name, data in metrics_data.items():
        if 'validation' in data and 'ssim' in data['validation']:
            ssim = data['validation']['ssim']
            epochs = data['validation'].get('epochs', list(range(1, len(ssim) + 1)))
            ax.plot(epochs, ssim, 'o-', label=labels.get(model_name, model_name),
                   color=colors.get(model_name, None), linewidth=2, markersize=6)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('SSIM')
    ax.set_title('Validation SSIM')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # LPIPS
    ax = axes[2]
    for model_name, data in metrics_data.items():
        if 'validation' in data and 'lpips' in data['validation']:
            lpips = data['validation']['lpips']
            epochs = data['validation'].get('epochs', list(range(1, len(lpips) + 1)))
            ax.plot(epochs, lpips, 'o-', label=labels.get(model_name, model_name),
                   color=colors.get(model_name, None), linewidth=2, markersize=6)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('LPIPS')
    ax.set_title('Validation LPIPS (Lower is Better)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'validation_metrics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate training visualization for thesis')
    parser.add_argument('--results_dir', type=str, default='./results/training',
                       help='Directory containing training metrics')
    parser.add_argument('--output_dir', type=str, default='./figures',
                       help='Output directory for figures')
    parser.add_argument('--models', type=str, default=None,
                       help='Comma-separated list of models to plot (e.g., baseline,face_loss3)')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("="*70)
    print("Training Curves Visualization")
    print("="*70)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory:  {args.output_dir}")
    print()

    # Parse metrics
    parser = TrainingMetricsParser(args.results_dir)
    metrics_data = parser.load_all_metrics()

    if not metrics_data:
        print("No training metrics found!")
        print(f"  Looking for: {args.results_dir}/metrics*.md or loss_history*.json")
        print()
        print("To generate metrics during training, ensure train.py saves:")
        print("  - results/training/metrics_*.md (already implemented)")
        print("  - results/training/loss_history_*.json (need to implement)")
        return 1

    print(f"Found metrics for {len(metrics_data)} model(s):")
    for model_name in sorted(metrics_data.keys()):
        print(f"  - {model_name}")
    print()

    # Generate plots
    print("Generating plots...")
    plot_loss_curves(metrics_data, args.output_dir)
    plot_loss_components(metrics_data, args.output_dir)
    plot_validation_metrics(metrics_data, args.output_dir)

    print()
    print("="*70)
    print("Visualization complete!")
    print("="*70)
    print(f"\nFigures saved to: {args.output_dir}")

    return 0


if __name__ == '__main__':
    exit(main())
