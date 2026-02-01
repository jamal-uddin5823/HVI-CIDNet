"""
Multi-Level Results Visualization Script for Thesis

Generates publication-quality plots showing:
1. Model comparison across difficulty levels (grouped bar charts)
2. Performance degradation curves (easy -> medium -> hard)
3. Image quality vs. difficulty analysis
4. Multi-level generalization analysis

This script parses evaluation results from all model × difficulty combinations.

Usage:
    # Plot all multi-level results
    python plot_multilevel_results.py --results_dir=./results/multilevel_evaluations

    # Plot specific difficulty levels
    python plot_multilevel_results.py --difficulties=easy,medium,hard

    # Custom output directory
    python plot_multilevel_results.py --output_dir=./figures
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


def parse_verification_results(filepath):
    """Parse face verification results file

    Returns:
        dict: Parsed metrics
    """
    metrics = {}

    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r') as f:
        content = f.read()

    # Check if pairs-based or legacy evaluation
    is_pairs_based = 'Pairs Protocol' in content or 'Genuine pairs:' in content

    try:
        if is_pairs_based:
            # Parse pairs-based metrics
            # Genuine pair similarity
            match = re.search(r'Enhanced avg similarity:\s+([\d.]+)', content)
            if match:
                metrics['genuine_similarity'] = float(match.group(1))

            # EER
            match = re.search(r'Enhanced:\s+([\d.]+)%.*?EER', content)
            if match:
                metrics['eer'] = float(match.group(1))

            # TAR @ FAR = 0.1%
            match = re.search(r'TAR @ FAR=0\.1%.*?Enhanced:\s+([\d.]+)%', content, re.DOTALL)
            if match:
                metrics['tar_001'] = float(match.group(1))

            # TAR @ FAR = 1%
            match = re.search(r'TAR @ FAR=1%.*?Enhanced:\s+([\d.]+)%', content, re.DOTALL)
            if match:
                metrics['tar_01'] = float(match.group(1))

        # Common metrics (both modes)
        match = re.search(r'Average PSNR:\s+([\d.]+)', content)
        if match:
            metrics['psnr'] = float(match.group(1))

        match = re.search(r'Average SSIM:\s+([\d.]+)', content)
        if match:
            metrics['ssim'] = float(match.group(1))

        metrics['is_pairs_based'] = is_pairs_based

    except Exception as e:
        print(f"Warning: Error parsing {filepath}: {e}")
        return None

    return metrics


def load_json_scores(filepath):
    """Load verification scores from JSON file"""
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return None


def discover_results(results_dir):
    """Discover all evaluation results in the directory

    Expected structure:
    results_dir/
        baseline/
            face_verification_results.txt
            verification_scores.json
        face_loss3/
            ...
        face_loss5/
            ...
    Or with difficulty subdirectories:
    results_dir/
        baseline/
            easy/
                face_verification_results.txt
            medium/
                ...
            hard/
                ...
            mixed/
                ...
    """
    results = defaultdict(lambda: defaultdict(dict))

    base_dir = Path(results_dir)

    # Check for difficulty-based structure
    models = ['baseline', 'face_loss3', 'face_loss5']
    difficulties = ['easy', 'medium', 'hard', 'mixed']

    for model in models:
        model_dir = base_dir / model
        if not model_dir.exists():
            continue

        # Check if there are difficulty subdirectories
        has_difficulties = any((model_dir / d).exists() for d in difficulties)

        if has_difficulties:
            # Load from difficulty subdirectories
            for diff in difficulties:
                diff_dir = model_dir / diff
                if diff_dir.exists():
                    results_file = diff_dir / 'face_verification_results.txt'
                    json_file = diff_dir / 'verification_scores.json'

                    if results_file.exists():
                        metrics = parse_verification_results(results_file)
                        if metrics:
                            results[model][diff] = metrics

                    if json_file.exists():
                        scores = load_json_scores(json_file)
                        if scores:
                            results[model][diff]['scores'] = scores
        else:
            # Load directly from model directory
            results_file = model_dir / 'face_verification_results.txt'
            json_file = model_dir / 'verification_scores.json'

            if results_file.exists():
                metrics = parse_verification_results(results_file)
                if metrics:
                    # Assume 'mixed' if no difficulty specified
                    results[model]['mixed'] = metrics

            if json_file.exists():
                scores = load_json_scores(json_file)
                if scores and 'mixed' in results[model]:
                    results[model]['mixed']['scores'] = scores

    return results


def plot_model_comparison_by_difficulty(results, output_dir):
    """Plot grouped bar charts: metrics for each model × difficulty combination"""
    models = ['baseline', 'face_loss3', 'face_loss5']
    difficulties = ['easy', 'medium', 'hard', 'mixed']

    model_labels = {
        'baseline': 'Baseline',
        'face_loss3': 'Face Loss 0.3',
        'face_loss5': 'Face Loss 0.5'
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # EER comparison
    ax = axes[0, 0]
    x = np.arange(len(difficulties))
    width = 0.25

    for i, model in enumerate(models):
        eers = []
        for diff in difficulties:
            if diff in results[model] and 'eer' in results[model][diff]:
                eers.append(results[model][diff]['eer'])
            else:
                eers.append(0)

        ax.bar(x + i * width, eers, width, label=model_labels[model], alpha=0.8)

    ax.set_xlabel('Difficulty Level')
    ax.set_ylabel('EER (%)')
    ax.set_title('(a) Equal Error Rate by Difficulty')
    ax.set_xticks(x + width)
    ax.set_xticklabels(difficulties)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # TAR @ FAR = 1% comparison
    ax = axes[0, 1]
    for i, model in enumerate(models):
        tars = []
        for diff in difficulties:
            if diff in results[model] and 'tar_01' in results[model][diff]:
                tars.append(results[model][diff]['tar_01'])
            else:
                tars.append(0)

        ax.bar(x + i * width, tars, width, label=model_labels[model], alpha=0.8)

    ax.set_xlabel('Difficulty Level')
    ax.set_ylabel('TAR @ FAR=1% (%)')
    ax.set_title('(b) True Accept Rate by Difficulty')
    ax.set_xticks(x + width)
    ax.set_xticklabels(difficulties)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Genuine similarity comparison
    ax = axes[1, 0]
    for i, model in enumerate(models):
        sims = []
        for diff in difficulties:
            if diff in results[model] and 'genuine_similarity' in results[model][diff]:
                sims.append(results[model][diff]['genuine_similarity'])
            else:
                sims.append(0)

        ax.bar(x + i * width, sims, width, label=model_labels[model], alpha=0.8)

    ax.set_xlabel('Difficulty Level')
    ax.set_ylabel('Genuine Similarity')
    ax.set_title('(c) Genuine Similarity by Difficulty')
    ax.set_xticks(x + width)
    ax.set_xticklabels(difficulties)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])

    # PSNR comparison
    ax = axes[1, 1]
    for i, model in enumerate(models):
        psnrs = []
        for diff in difficulties:
            if diff in results[model] and 'psnr' in results[model][diff]:
                psnrs.append(results[model][diff]['psnr'])
            else:
                psnrs.append(0)

        ax.bar(x + i * width, psnrs, width, label=model_labels[model], alpha=0.8)

    ax.set_xlabel('Difficulty Level')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('(d) Image Quality (PSNR) by Difficulty')
    ax.set_xticks(x + width)
    ax.set_xticklabels(difficulties)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'multilevel_model_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_degradation_curves(results, output_dir):
    """Plot performance degradation curves (easy -> medium -> hard)"""
    models = ['baseline', 'face_loss3', 'face_loss5']
    difficulties = ['easy', 'medium', 'hard']

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

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # EER degradation
    ax = axes[0]
    for model in models:
        eers = []
        for diff in difficulties:
            if diff in results[model] and 'eer' in results[model][diff]:
                eers.append(results[model][diff]['eer'])
            else:
                eers.append(None)

        if any(v is not None for v in eers):
            ax.plot(difficulties, eers, 'o-', label=model_labels[model],
                   color=colors[model], linewidth=2, markersize=8)

    ax.set_xlabel('Difficulty Level')
    ax.set_ylabel('EER (%)')
    ax.set_title('(a) Verification Degradation (EER)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # TAR degradation
    ax = axes[1]
    for model in models:
        tars = []
        for diff in difficulties:
            if diff in results[model] and 'tar_01' in results[model][diff]:
                tars.append(results[model][diff]['tar_01'])
            else:
                tars.append(None)

        if any(v is not None for v in tars):
            ax.plot(difficulties, tars, 'o-', label=model_labels[model],
                   color=colors[model], linewidth=2, markersize=8)

    ax.set_xlabel('Difficulty Level')
    ax.set_ylabel('TAR @ FAR=1% (%)')
    ax.set_title('(b) Verification Degradation (TAR)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'degradation_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_quality_vs_difficulty(results, output_dir):
    """Plot image quality metrics vs. difficulty level"""
    models = ['baseline', 'face_loss3', 'face_loss5']
    difficulties = ['easy', 'medium', 'hard', 'mixed']

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

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # PSNR vs difficulty
    ax = axes[0]
    for model in models:
        psnrs = []
        for diff in difficulties:
            if diff in results[model] and 'psnr' in results[model][diff]:
                psnrs.append(results[model][diff]['psnr'])
            else:
                psnrs.append(None)

        if any(v is not None for v in psnrs):
            ax.plot(difficulties, psnrs, 'o-', label=model_labels[model],
                   color=colors[model], linewidth=2, markersize=8)

    ax.set_xlabel('Difficulty Level')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('(a) Image Quality (PSNR) vs. Difficulty')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # SSIM vs difficulty
    ax = axes[1]
    for model in models:
        ssims = []
        for diff in difficulties:
            if diff in results[model] and 'ssim' in results[model][diff]:
                ssims.append(results[model][diff]['ssim'])
            else:
                ssims.append(None)

        if any(v is not None for v in ssims):
            ax.plot(difficulties, ssims, 'o-', label=model_labels[model],
                   color=colors[model], linewidth=2, markersize=8)

    ax.set_xlabel('Difficulty Level')
    ax.set_ylabel('SSIM')
    ax.set_title('(b) Image Quality (SSIM) vs. Difficulty')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'quality_vs_difficulty.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate multi-level comparison visualizations')
    parser.add_argument('--results_dir', type=str, default='./results/multilevel_evaluations',
                       help='Directory containing multilevel evaluation results')
    parser.add_argument('--output_dir', type=str, default='./figures',
                       help='Output directory for figures')
    parser.add_argument('--difficulties', type=str, default='easy,medium,hard,mixed',
                       help='Comma-separated list of difficulty levels')

    args = parser.parse_args()

    difficulties = args.difficulties.split(',')
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*70)
    print("Multi-Level Results Visualization")
    print("="*70)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory:  {args.output_dir}")
    print(f"Difficulty levels: {difficulties}")
    print()

    # Discover and load results
    results = discover_results(args.results_dir)

    if not results:
        print("No evaluation results found!")
        print(f"  Looking for: {args.results_dir}/<model>/face_verification_results.txt")
        print()
        print("Expected structure:")
        print("  results/multilevel_evaluations/")
        print("    baseline/")
        print("      easy/face_verification_results.txt")
        print("      medium/face_verification_results.txt")
        print("      ...")
        print("    face_loss3/")
        print("      ...")
        return 1

    print(f"Found results for {len(results)} model(s):")
    for model_name in sorted(results.keys()):
        print(f"  - {model_name}: {list(results[model_name].keys())}")
    print()

    # Generate plots
    print("Generating plots...")
    plot_model_comparison_by_difficulty(results, args.output_dir)
    plot_degradation_curves(results, args.output_dir)
    plot_quality_vs_difficulty(results, args.output_dir)

    print()
    print("="*70)
    print("Visualization complete!")
    print("="*70)
    print(f"\nFigures saved to: {args.output_dir}")

    return 0


if __name__ == '__main__':
    exit(main())
