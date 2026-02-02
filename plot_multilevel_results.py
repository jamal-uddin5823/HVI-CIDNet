"""
Multi-Level Results Visualization Script for Thesis

Generates publication-quality plots showing:
1. Model comparison across difficulty levels (grouped bar charts)
2. Performance degradation curves (easy -> medium -> hard)
3. Image quality vs. difficulty analysis
4. Multi-level generalization analysis

This script parses results directly from face_eval.log file.

Usage:
    # Plot all multi-level results
    python plot_multilevel_results.py

    # Custom log file
    python plot_multilevel_results.py --eval_log=./face_eval.log

    # Custom output directory
    python plot_multilevel_results.py --output_dir=./figures
"""

import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
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


def parse_face_eval_log(filepath):
    """Parse face verification results from eval log file

    Expected log format:
        Evaluating: <model> on <difficulty>
        ...
        Equal Error Rate (EER):
            Enhanced:   X.XX%
        True Accept Rate @ FAR=0.1%:
            Enhanced:   XX.XX%
        True Accept Rate @ FAR=1%:
            Enhanced:   XX.XX%
        Genuine Pair Scores:
            Enhanced avg similarity:   X.XXXX
        Average PSNR: X.XX dB
        Average SSIM: X.XXXX
    """
    results = defaultdict(lambda: defaultdict(dict))

    if not os.path.exists(filepath):
        print(f"Warning: Log file not found: {filepath}")
        return {}

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Split into evaluation sections
    sections = re.split(r'Evaluating:\s+(\S+)\s+on\s+(\w+)', content)

    for i in range(1, len(sections), 2):
        if i + 1 >= len(sections):
            break

        model = sections[i].strip()
        difficulty = sections[i + 1]

        # Extract the section content
        if i + 2 < len(sections):
            section_content = sections[i + 2]
        else:
            section_content = sections[-1]

        # Parse EER
        match = re.search(r'Equal Error Rate.*?Enhanced:\s+([\d.]+)%', section_content, re.DOTALL)
        if match:
            results[model][difficulty]['eer'] = float(match.group(1))

        # Parse TAR @ FAR=0.1%
        match = re.search(r'TAR.*?FAR=0\.1%.*?Enhanced:\s+([\d.]+)%', section_content, re.DOTALL)
        if match:
            results[model][difficulty]['tar_001'] = float(match.group(1))

        # Parse TAR @ FAR=1%
        match = re.search(r'TAR.*?FAR=1%.*?Enhanced:\s+([\d.]+)%', section_content, re.DOTALL)
        if match:
            results[model][difficulty]['tar_01'] = float(match.group(1))

        # Parse genuine similarity
        match = re.search(r'Enhanced avg similarity:\s+([\d.]+)', section_content)
        if match:
            results[model][difficulty]['genuine_similarity'] = float(match.group(1))

        # Parse PSNR
        match = re.search(r'Average PSNR:\s+([\d.]+)', section_content)
        if match:
            results[model][difficulty]['psnr'] = float(match.group(1))

        # Parse SSIM
        match = re.search(r'Average SSIM:\s+([\d.]+)', section_content)
        if match:
            results[model][difficulty]['ssim'] = float(match.group(1))

    return dict(results)


def plot_model_comparison_by_difficulty(results, output_dir):
    """Plot grouped bar charts: metrics for each model × difficulty combination"""
    models = ['baseline', 'face_loss3', 'face_loss5']
    difficulties = ['easy', 'medium', 'hard', 'mixed']

    model_labels = {
        'baseline': 'Baseline',
        'face_loss3': 'Face Loss 0.3',
        'face_loss5': 'Face Loss 0.5'
    }

    colors = {
        'baseline': '#0173B2',
        'face_loss3': '#029E73',
        'face_loss5': '#CC78BC'
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # EER comparison
    ax = axes[0, 0]
    x = np.arange(len(difficulties))
    width = 0.25

    for i, model in enumerate(models):
        eers = []
        for diff in difficulties:
            if model in results and diff in results[model] and 'eer' in results[model][diff]:
                eers.append(results[model][diff]['eer'])
            else:
                eers.append(0)

        ax.bar(x + i * width, eers, width, label=model_labels[model],
               color=colors[model], alpha=0.8, edgecolor='black', linewidth=1)

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
            if model in results and diff in results[model] and 'tar_01' in results[model][diff]:
                tars.append(results[model][diff]['tar_01'])
            else:
                tars.append(0)

        ax.bar(x + i * width, tars, width, label=model_labels[model],
               color=colors[model], alpha=0.8, edgecolor='black', linewidth=1)

    ax.set_xlabel('Difficulty Level')
    ax.set_ylabel('TAR @ FAR=1% (%)')
    ax.set_title('(b) True Accept Rate by Difficulty')
    ax.set_xticks(x + width)
    ax.set_xticklabels(difficulties)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 100])

    # Genuine similarity comparison
    ax = axes[1, 0]
    for i, model in enumerate(models):
        sims = []
        for diff in difficulties:
            if model in results and diff in results[model] and 'genuine_similarity' in results[model][diff]:
                sims.append(results[model][diff]['genuine_similarity'])
            else:
                sims.append(0)

        ax.bar(x + i * width, sims, width, label=model_labels[model],
               color=colors[model], alpha=0.8, edgecolor='black', linewidth=1)

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
            if model in results and diff in results[model] and 'psnr' in results[model][diff]:
                psnrs.append(results[model][diff]['psnr'])
            else:
                psnrs.append(0)

        ax.bar(x + i * width, psnrs, width, label=model_labels[model],
               color=colors[model], alpha=0.8, edgecolor='black', linewidth=1)

    ax.set_xlabel('Difficulty Level')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('(d) Image Quality (PSNR) by Difficulty')
    ax.set_xticks(x + width)
    ax.set_xticklabels(difficulties)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Model Comparison Across Difficulty Levels', fontsize=14, fontweight='bold')
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
        'baseline': '#0173B2',
        'face_loss3': '#029E73',
        'face_loss5': '#CC78BC'
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # EER degradation
    ax = axes[0]
    for model in models:
        eers = []
        for diff in difficulties:
            if model in results and diff in results[model] and 'eer' in results[model][diff]:
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
            if model in results and diff in results[model] and 'tar_01' in results[model][diff]:
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
    ax.set_ylim([90, 100])

    plt.suptitle('Performance Degradation Across Difficulty Levels', fontsize=13, fontweight='bold')
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
        'baseline': '#0173B2',
        'face_loss3': '#029E73',
        'face_loss5': '#CC78BC'
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # PSNR vs difficulty
    ax = axes[0]
    for model in models:
        psnrs = []
        for diff in difficulties:
            if model in results and diff in results[model] and 'psnr' in results[model][diff]:
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
            if model in results and diff in results[model] and 'ssim' in results[model][diff]:
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
    ax.set_ylim([0.6, 1])

    plt.suptitle('Image Quality vs. Difficulty Level', fontsize=13, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'quality_vs_difficulty.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate multi-level comparison visualizations')
    parser.add_argument('--eval_log', type=str, default='./face_eval.log',
                       help='Path to face evaluation log file')
    parser.add_argument('--output_dir', type=str, default='./figures',
                       help='Output directory for figures')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("Multi-Level Results Visualization")
    print("=" * 70)
    print(f"Eval log:         {args.eval_log}")
    print(f"Output directory:  {args.output_dir}")
    print()

    # Load results from log
    results = parse_face_eval_log(args.eval_log)

    if not results:
        print("No evaluation results found in log!")
        print("  Make sure face_eval.log exists and contains evaluation results.")
        return 1

    print(f"Found results for {len(results)} model(s):")
    for model_name in sorted(results.keys()):
        difficulties = list(results[model_name].keys())
        print(f"  - {model_name}: {difficulties}")
    print()

    # Generate plots
    print("Generating plots...")
    plot_model_comparison_by_difficulty(results, args.output_dir)
    plot_degradation_curves(results, args.output_dir)
    plot_quality_vs_difficulty(results, args.output_dir)

    print()
    print("=" * 70)
    print("Visualization complete!")
    print("=" * 70)
    print(f"\nFigures saved to: {args.output_dir}")

    return 0


if __name__ == '__main__':
    exit(main())
