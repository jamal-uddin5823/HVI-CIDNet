"""
Verification Analysis Visualization Script for Thesis

Generates publication-quality plots showing:
1. ROC curves (TPR vs FPR) for each model
2. Genuine vs. Impostor score distributions (histograms)
3. Score separation analysis
4. Threshold analysis

This script parses verification_scores.json files exported by eval_face_verification.py.

Usage:
    # Plot verification analysis from evaluation results
    python plot_verification_analysis.py --results_dir=./results/multilevel_evaluations

    # Plot specific model
    python plot_verification_analysis.py --models=face_loss3

    # Custom output directory
    python plot_verification_analysis.py --output_dir=./figures
"""

import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from collections import defaultdict
from scipy import stats

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


def load_scores(results_dir, models=None):
    """Load verification scores from JSON files

    Args:
        results_dir: Base directory containing results
        models: List of models to load (None = all)

    Returns:
        dict: {model: {difficulty: scores_dict}}
    """
    if models is None:
        models = ['baseline', 'face_loss3', 'face_loss5']

    results = defaultdict(lambda: defaultdict(dict))
    base_dir = Path(results_dir)

    for model in models:
        model_dir = base_dir / model
        if not model_dir.exists():
            continue

        # Look for JSON files in difficulty subdirectories or directly in model dir
        difficulties = ['easy', 'medium', 'hard', 'mixed']

        # Check for difficulty subdirectories
        has_difficulties = any((model_dir / d).exists() for d in difficulties)

        if has_difficulties:
            for diff in difficulties:
                json_file = model_dir / diff / 'verification_scores.json'
                if json_file.exists():
                    with open(json_file, 'r') as f:
                        results[model][diff] = json.load(f)
        else:
            # Load directly from model directory
            json_file = model_dir / 'verification_scores.json'
            if json_file.exists():
                with open(json_file, 'r') as f:
                    results[model]['mixed'] = json.load(f)

    return results


def plot_roc_curves(scores_data, output_dir, difficulty='mixed'):
    """Plot ROC curves for all models on a specific difficulty level"""
    model_labels = {
        'baseline': 'Baseline (FR=0.0)',
        'face_loss3': 'Face Loss 0.3',
        'face_loss5': 'Face Loss 0.5'
    }

    colors = {
        'baseline': '#1f77b4',
        'face_loss3': '#2ca02c',
        'face_loss5': '#d62728'
    }

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot enhanced images
    for model in ['baseline', 'face_loss3', 'face_loss5']:
        if model not in scores_data or difficulty not in scores_data[model]:
            continue

        data = scores_data[model][difficulty]
        if 'roc_data' not in data or 'enhanced' not in data['roc_data']:
            continue

        roc = data['roc_data']['enhanced']
        fpr = np.array(roc['fpr'])
        tpr = np.array(roc['tpr'])

        # Sort by FPR for clean plotting
        idx = np.argsort(fpr)
        fpr = fpr[idx]
        tpr = tpr[idx]

        ax.plot(fpr, tpr, linewidth=2, label=model_labels[model],
               color=colors[model])

    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')

    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_title(f'ROC Curves - {difficulty.capitalize()} Test Set')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'roc_curves_{difficulty}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_score_distributions(scores_data, output_dir, difficulty='mixed'):
    """Plot genuine vs. impostor score distributions for each model"""
    model_labels = {
        'baseline': 'Baseline',
        'face_loss3': 'Face Loss 0.3',
        'face_loss5': 'Face Loss 0.5'
    }

    for model in ['baseline', 'face_loss3', 'face_loss5']:
        if model not in scores_data or difficulty not in scores_data[model]:
            continue

        data = scores_data[model][difficulty]

        if 'genuine_scores_enhanced' not in data or 'impostor_scores_enhanced' not in data:
            continue

        genuine = np.array(data['genuine_scores_enhanced'])
        impostor = np.array(data['impostor_scores_enhanced'])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Enhanced scores
        bins = np.linspace(0, 1, 50)
        ax1.hist(genuine, bins=bins, alpha=0.6, label='Genuine Pairs', color='#2ca02c', density=True)
        ax1.hist(impostor, bins=bins, alpha=0.6, label='Impostor Pairs', color='#d62728', density=True)

        # Add KDE curves
        if len(genuine) > 1:
            from scipy.stats import gaussian_kde
            kde_genuine = gaussian_kde(genuine)
            x_grid = np.linspace(0, 1, 200)
            ax1.plot(x_grid, kde_genuine(x_grid), 'g-', linewidth=2, label='Genuine KDE')

        if len(impostor) > 1:
            from scipy.stats import gaussian_kde
            kde_impostor = gaussian_kde(impostor)
            x_grid = np.linspace(0, 1, 200)
            ax1.plot(x_grid, kde_impostor(x_grid), 'r-', linewidth=2, label='Impostor KDE')

        ax1.set_xlabel('Similarity Score')
        ax1.set_ylabel('Density')
        ax1.set_title(f'({chr(97+0)}) Score Distributions - {model_labels[model]}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Compute statistics
        genuine_mean = np.mean(genuine)
        genuine_std = np.std(genuine)
        impostor_mean = np.mean(impostor)
        impostor_std = np.std(impostor)

        # Separation metrics
        separation = abs(genuine_mean - impostor_mean) / np.sqrt((genuine_std**2 + impostor_std**2) / 2)

        # Box plot
        data_to_plot = [genuine, impostor]
        bp = ax2.boxplot(data_to_plot, labels=['Genuine', 'Impostor'], patch_artist=True)

        for patch, color in zip(bp['boxes'], ['#2ca02c', '#d62728']):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax2.set_ylabel('Similarity Score')
        ax2.set_title(f'({chr(97+1)}) Score Distribution - {model_labels[model]}')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim([0, 1])

        # Add statistics text
        stats_text = f'Genuine: {genuine_mean:.3f} ± {genuine_std:.3f}\n'
        stats_text += f'Impostor: {impostor_mean:.3f} ± {impostor_std:.3f}\n'
        stats_text += f'Separation (d\'): {separation:.3f}'
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        output_path = os.path.join(output_dir, f'score_distributions_{model}_{difficulty}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()


def plot_roc_by_difficulty(scores_data, output_dir):
    """Plot ROC curves for all models across all difficulty levels"""
    model_labels = {
        'baseline': 'Baseline',
        'face_loss3': 'Face Loss 0.3',
        'face_loss5': 'Face Loss 0.5'
    }

    colors = {
        'baseline': '#1f77b4',
        'face_loss3': '#2ca02c',
        'face_loss5': '#d62728'
    }

    difficulties = ['easy', 'medium', 'hard', 'mixed']

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, difficulty in enumerate(difficulties):
        ax = axes[idx]

        for model in ['baseline', 'face_loss3', 'face_loss5']:
            if model not in scores_data or difficulty not in scores_data[model]:
                continue

            data = scores_data[model][difficulty]
            if 'roc_data' not in data or 'enhanced' not in data['roc_data']:
                continue

            roc = data['roc_data']['enhanced']
            fpr = np.array(roc['fpr'])
            tpr = np.array(roc['tpr'])

            # Sort by FPR for clean plotting
            sort_idx = np.argsort(fpr)
            fpr = fpr[sort_idx]
            tpr = tpr[sort_idx]

            ax.plot(fpr, tpr, linewidth=2, label=model_labels[model],
                   color=colors[model])

        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.3)

        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'({chr(97+idx)}) {difficulty.capitalize()} Test Set')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    plt.suptitle('ROC Curves Across Difficulty Levels', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'roc_curves_all_difficulties.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate verification analysis visualizations')
    parser.add_argument('--results_dir', type=str, default='./results/multilevel_evaluations',
                       help='Directory containing evaluation results')
    parser.add_argument('--output_dir', type=str, default='./figures',
                       help='Output directory for figures')
    parser.add_argument('--models', type=str, default=None,
                       help='Comma-separated list of models to plot')
    parser.add_argument('--difficulty', type=str, default='mixed',
                       help='Difficulty level for single-difficulty plots')

    args = parser.parse_args()

    models = args.models.split(',') if args.models else None
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*70)
    print("Verification Analysis Visualization")
    print("="*70)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory:  {args.output_dir}")
    print(f"Difficulty:        {args.difficulty}")
    print()

    # Load scores
    scores_data = load_scores(args.results_dir, models)

    if not scores_data:
        print("No verification scores found!")
        print(f"  Looking for: {args.results_dir}/<model>/verification_scores.json")
        print()
        print("Make sure to run eval_face_verification.py with score export enabled.")
        return 1

    print(f"Found scores for {len(scores_data)} model(s):")
    for model_name in sorted(scores_data.keys()):
        difficulties = list(scores_data[model_name].keys())
        print(f"  - {model_name}: {difficulties}")
    print()

    # Generate plots
    print("Generating plots...")
    plot_roc_curves(scores_data, args.output_dir, args.difficulty)
    plot_score_distributions(scores_data, args.output_dir, args.difficulty)
    plot_roc_by_difficulty(scores_data, args.output_dir)

    print()
    print("="*70)
    print("Visualization complete!")
    print("="*70)
    print(f"\nFigures saved to: {args.output_dir}")

    return 0


if __name__ == '__main__':
    exit(main())
