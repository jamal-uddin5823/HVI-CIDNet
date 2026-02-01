"""
Thesis Summary Figure Script

Generates publication-quality main results figures for thesis:
1. 2×2 main results subplot (EER, TAR@FAR=1%, TAR@FAR=0.1%, Genuine Similarity)
2. Comprehensive multi-level comparison
3. Trade-off analysis (Pareto frontier)
4. Quality vs. verification correlation

These figures are designed for direct inclusion in the thesis document.

Usage:
    # Generate all thesis figures
    python plot_thesis_summary.py --results_dir=./results/multilevel_evaluations

    # Generate specific figure type
    python plot_thesis_summary.py --figure_type=main_results

    # Custom output directory
    python plot_thesis_summary.py --output_dir=./figures/thesis
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

# Use non-interactive backend
matplotlib.use('Agg')

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
matplotlib.rcParams['font.size'] =11
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 13
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['figure.titlesize'] = 14

# Define thesis color scheme (colorblind-friendly)
THESIS_COLORS = {
    'baseline': '#0173B2',      # Blue
    'face_loss3': '#029E73',    # Green
    'face_loss5': '#CC78BC',    # Purple/Magenta
    'easy': '#ECE133',          # Yellow
    'medium': '#F39700',        # Orange
    'hard': '#D55E00'           # Red-Orange
}

MODEL_LABELS = {
    'baseline': 'Baseline (λ=0.0)',
    'face_loss3': 'Face Loss (λ=0.3)',
    'face_loss5': 'Face Loss (λ=0.5)'
}


def parse_verification_results(filepath):
    """Parse face verification results file"""
    metrics = {}

    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r') as f:
        content = f.read()

    try:
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

        # PSNR
        match = re.search(r'Average PSNR:\s+([\d.]+)', content)
        if match:
            metrics['psnr'] = float(match.group(1))

        # SSIM
        match = re.search(r'Average SSIM:\s+([\d.]+)', content)
        if match:
            metrics['ssim'] = float(match.group(1))

    except Exception as e:
        print(f"Warning: Error parsing {filepath}: {e}")
        return None

    return metrics


def load_results(results_dir):
    """Load all evaluation results"""
    results = {}
    base_dir = Path(results_dir)

    models = ['baseline', 'face_loss3', 'face_loss5']

    for model in models:
        model_dir = base_dir / model
        if not model_dir.exists():
            continue

        results_file = model_dir / 'face_verification_results.txt'
        if results_file.exists():
            results[model] = parse_verification_results(results_file)

    return results


def plot_main_results_2x2(results, output_dir):
    """Generate 2×2 main results figure for thesis

    This is the key figure showing:
    - (a) EER comparison
    - (b) TAR@FAR=1% comparison
    - (c) TAR@FAR=0.1% comparison
    - (d) Genuine Similarity comparison
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    models = ['baseline', 'face_loss3', 'face_loss5']
    x = np.arange(len(models))
    width = 0.6

    # (a) EER
    ax = axes[0, 0]
    eers = []
    for model in models:
        if model in results and 'eer' in results[model]:
            eers.append(results[model]['eer'])
        else:
            eers.append(0)

    bars = ax.bar(x, eers, width, color=[THESIS_COLORS[m] for m in models], alpha=0.8, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Equal Error Rate (%)', fontsize=12)
    ax.set_title('(a) Equal Error Rate', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=0, fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, max(eers) * 1.2 if eers else 10])

    # Add value labels on bars
    for i, (bar, eer) in enumerate(zip(bars, eers)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{eer:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # (b) TAR @ FAR = 1%
    ax = axes[0, 1]
    tars = []
    for model in models:
        if model in results and 'tar_01' in results[model]:
            tars.append(results[model]['tar_01'])
        else:
            tars.append(0)

    bars = ax.bar(x, tars, width, color=[THESIS_COLORS[m] for m in models], alpha=0.8, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('True Accept Rate (%)', fontsize=12)
    ax.set_title('(b) TAR @ FAR = 1%', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=0, fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 100])

    for bar, tar in zip(bars, tars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{tar:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # (c) TAR @ FAR = 0.1%
    ax = axes[1, 0]
    tars = []
    for model in models:
        if model in results and 'tar_001' in results[model]:
            tars.append(results[model]['tar_001'])
        else:
            tars.append(0)

    bars = ax.bar(x, tars, width, color=[THESIS_COLORS[m] for m in models], alpha=0.8, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('True Accept Rate (%)', fontsize=12)
    ax.set_title('(c) TAR @ FAR = 0.1%', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=0, fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 100])

    for bar, tar in zip(bars, tars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{tar:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # (d) Genuine Similarity
    ax = axes[1, 1]
    sims = []
    for model in models:
        if model in results and 'genuine_similarity' in results[model]:
            sims.append(results[model]['genuine_similarity'])
        else:
            sims.append(0)

    bars = ax.bar(x, sims, width, color=[THESIS_COLORS[m] for m in models], alpha=0.8, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_title('(d) Genuine Pair Similarity', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=0, fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])

    for bar, sim in zip(bars, sims):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{sim:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle('Face Verification Performance Comparison', fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'thesis_main_results.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_quality_verification_tradeoff(results, output_dir):
    """Plot Pareto frontier: PSNR vs. EER trade-off"""
    models = ['baseline', 'face_loss3', 'face_loss5']

    fig, ax = plt.subplots(figsize=(10, 8))

    psnr_values = []
    eer_values = []
    fr_weights = []
    colors = []

    for model in models:
        if model in results:
            if 'psnr' in results[model] and 'eer' in results[model]:
                psnr_values.append(results[model]['psnr'])
                eer_values.append(results[model]['eer'])

                if model == 'baseline':
                    fr_weights.append(0.0)
                elif model == 'face_loss3':
                    fr_weights.append(0.3)
                else:
                    fr_weights.append(0.5)

                colors.append(THESIS_COLORS[model])

    # Create scatter plot
    scatter = ax.scatter(psnr_values, eer_values, c=fr_weights, s=200,
                        cmap='viridis', alpha=0.8, edgecolors='black', linewidths=1.5)

    # Add labels
    for i, (model, psnr, eer) in enumerate(zip(models, psnr_values, eer_values)):
        ax.annotate(MODEL_LABELS[model], (psnr, eer),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Face Recognition Loss Weight (λ)', fontsize=11)
    cbar.set_ticks([0.0, 0.3, 0.5])

    ax.set_xlabel('PSNR (dB)', fontsize=12)
    ax.set_ylabel('Equal Error Rate (%)', fontsize=12)
    ax.set_title('Quality-Verification Trade-off', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'thesis_tradeoff_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_comprehensive_summary(results, output_dir):
    """Create comprehensive summary with multiple subplots"""
    models = ['baseline', 'face_loss3', 'face_loss5']

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # (a) EER
    ax1 = fig.add_subplot(gs[0, 0])
    eers = [results[m]['eer'] if m in results and 'eer' in results[m] else 0 for m in models]
    bars = ax1.bar(range(len(models)), eers, color=[THESIS_COLORS[m] for m in models],
                  alpha=0.8, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('EER (%)')
    ax1.set_title('(a) Equal Error Rate', fontweight='bold')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=15, fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, eer in zip(bars, eers):
        if eer > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{eer:.2f}%', ha='center', va='bottom', fontsize=9)

    # (b) TAR @ FAR = 1%
    ax2 = fig.add_subplot(gs[0, 1])
    tars = [results[m]['tar_01'] if m in results and 'tar_01' in results[m] else 0 for m in models]
    bars = ax2.bar(range(len(models)), tars, color=[THESIS_COLORS[m] for m in models],
                  alpha=0.8, edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('TAR (%)')
    ax2.set_title('(b) TAR @ FAR = 1%', fontweight='bold')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=15, fontsize=9)
    ax2.set_ylim([0, 100])
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, tar in zip(bars, tars):
        if tar > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{tar:.1f}%', ha='center', va='bottom', fontsize=9)

    # (c) Genuine Similarity
    ax3 = fig.add_subplot(gs[0, 2])
    sims = [results[m]['genuine_similarity'] if m in results and 'genuine_similarity' in results[m] else 0
            for m in models]
    bars = ax3.bar(range(len(models)), sims, color=[THESIS_COLORS[m] for m in models],
                  alpha=0.8, edgecolor='black', linewidth=1.2)
    ax3.set_ylabel('Similarity')
    ax3.set_title('(c) Genuine Similarity', fontweight='bold')
    ax3.set_xticks(range(len(models)))
    ax3.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=15, fontsize=9)
    ax3.set_ylim([0, 1])
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, sim in zip(bars, sims):
        if sim > 0:
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{sim:.3f}', ha='center', va='bottom', fontsize=9)

    # (d) PSNR
    ax4 = fig.add_subplot(gs[1, 0])
    psnrs = [results[m]['psnr'] if m in results and 'psnr' in results[m] else 0 for m in models]
    bars = ax4.bar(range(len(models)), psnrs, color=[THESIS_COLORS[m] for m in models],
                  alpha=0.8, edgecolor='black', linewidth=1.2)
    ax4.set_ylabel('PSNR (dB)')
    ax4.set_title('(d) Image Quality (PSNR)', fontweight='bold')
    ax4.set_xticks(range(len(models)))
    ax4.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=15, fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, psnr in zip(bars, psnrs):
        if psnr > 0:
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{psnr:.2f}', ha='center', va='bottom', fontsize=9)

    # (e) SSIM
    ax5 = fig.add_subplot(gs[1, 1])
    ssims = [results[m]['ssim'] if m in results and 'ssim' in results[m] else 0 for m in models]
    bars = ax5.bar(range(len(models)), ssims, color=[THESIS_COLORS[m] for m in models],
                  alpha=0.8, edgecolor='black', linewidth=1.2)
    ax5.set_ylabel('SSIM')
    ax5.set_title('(e) Image Quality (SSIM)', fontweight='bold')
    ax5.set_xticks(range(len(models)))
    ax5.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=15, fontsize=9)
    ax5.set_ylim([0, 1])
    ax5.grid(True, alpha=0.3, axis='y')
    for bar, ssim in zip(bars, ssims):
        if ssim > 0:
            ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{ssim:.3f}', ha='center', va='bottom', fontsize=9)

    # (f) Trade-off: PSNR vs EER
    ax6 = fig.add_subplot(gs[1, 2])
    valid_data = [(results[m]['psnr'], results[m]['eer']) for m in models
                 if m in results and 'psnr' in results[m] and 'eer' in results[m]]
    if valid_data:
        psnr_vals, eer_vals = zip(*valid_data)
        fr_weights = [0.0, 0.3, 0.5][:len(psnr_vals)]
        scatter = ax6.scatter(psnr_vals, eer_vals, c=fr_weights, s=150,
                            cmap='viridis', alpha=0.8, edgecolors='black', linewidths=1.5)
        for m, psnr, eer in zip(models, psnr_vals, eer_vals):
            ax6.annotate(MODEL_LABELS[m], (psnr, eer),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        cbar = plt.colorbar(scatter, ax=ax6)
        cbar.set_label('FR Weight (λ)', fontsize=9)
    ax6.set_xlabel('PSNR (dB)')
    ax6.set_ylabel('EER (%)')
    ax6.set_title('(f) Quality-Verification Trade-off', fontweight='bold')
    ax6.grid(True, alpha=0.3)

    fig.suptitle('Comprehensive Results Summary', fontsize=16, fontweight='bold')

    output_path = os.path.join(output_dir, 'thesis_comprehensive_summary.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate thesis summary figures')
    parser.add_argument('--results_dir', type=str, default='./results/multilevel_evaluations',
                       help='Directory containing evaluation results')
    parser.add_argument('--output_dir', type=str, default='./figures/thesis',
                       help='Output directory for thesis figures')
    parser.add_argument('--figure_type', type=str, default='all',
                       choices=['all', 'main_results', 'tradeoff', 'comprehensive'],
                       help='Type of figure to generate')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("="*70)
    print("Thesis Summary Figure Generator")
    print("="*70)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory:  {args.output_dir}")
    print(f"Figure type:       {args.figure_type}")
    print()

    # Load results
    results = load_results(args.results_dir)

    if not results:
        print("No evaluation results found!")
        print(f"  Looking for: {args.results_dir}/<model>/face_verification_results.txt")
        return 1

    print(f"Found results for {len(results)} model(s):")
    for model_name in sorted(results.keys()):
        if results[model_name]:
            print(f"  - {model_name}")
    print()

    # Generate figures
    print("Generating thesis figures...")

    if args.figure_type in ['all', 'main_results']:
        print("  Generating main results (2x2)...")
        plot_main_results_2x2(results, args.output_dir)

    if args.figure_type in ['all', 'tradeoff']:
        print("  Generating trade-off analysis...")
        plot_quality_verification_tradeoff(results, args.output_dir)

    if args.figure_type in ['all', 'comprehensive']:
        print("  Generating comprehensive summary...")
        plot_comprehensive_summary(results, args.output_dir)

    print()
    print("="*70)
    print("Thesis figures generated successfully!")
    print("="*70)
    print(f"\nFigures saved to: {args.output_dir}")
    print("\nThese figures are ready for inclusion in your thesis:")
    print("  - thesis_main_results.png: Main 2x2 results figure")
    print("  - thesis_tradeoff_analysis.png: Pareto frontier analysis")
    print("  - thesis_comprehensive_summary.png: Complete results summary")
    print("\nAll figures are 300 DPI for publication quality.")

    return 0


if __name__ == '__main__':
    exit(main())
