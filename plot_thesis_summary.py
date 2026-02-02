"""
Thesis Summary Figure Script

Generates publication-quality main results figures for thesis:
1. 2×2 main results subplot (EER, TAR@FAR=1%, TAR@FAR=0.1%, Genuine Similarity)
2. Comprehensive multi-level comparison
3. Trade-off analysis (Pareto frontier)
4. Quality vs. verification correlation

This version parses results directly from face_eval.log file.

Usage:
    # Generate all thesis figures
    python plot_thesis_summary.py

    # Custom log file
    python plot_thesis_summary.py --eval_log=./face_eval.log

    # Custom output directory
    python plot_thesis_summary.py --output_dir=./figures/thesis
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

# Set publication-quality style
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

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
    # Each section starts with "Evaluating: <model> on <difficulty>"
    sections = re.split(r'Evaluating:\s+(\S+)\s+on\s+(\w+)', content)

    for i in range(1, len(sections), 2):
        if i + 1 >= len(sections):
            break

        model = sections[i].strip()
        difficulty = sections[i + 1]

        # Extract the section content (until next "Evaluating:" or end)
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


def plot_main_results_2x2(results, output_dir, difficulty='mixed'):
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
        if model in results and difficulty in results[model] and 'eer' in results[model][difficulty]:
            eers.append(results[model][difficulty]['eer'])
        else:
            eers.append(0)

    bars = ax.bar(x, eers, width, color=[THESIS_COLORS[m] for m in models],
                  alpha=0.8, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Equal Error Rate (%)', fontsize=12)
    ax.set_title('(a) Equal Error Rate', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], rotation=0, fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, max(eers) * 1.2 if eers else 10])

    for i, (bar, eer) in enumerate(zip(bars, eers)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{eer:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # (b) TAR @ FAR = 1%
    ax = axes[0, 1]
    tars = []
    for model in models:
        if model in results and difficulty in results[model] and 'tar_01' in results[model][difficulty]:
            tars.append(results[model][difficulty]['tar_01'])
        else:
            tars.append(0)

    bars = ax.bar(x, tars, width, color=[THESIS_COLORS[m] for m in models],
                  alpha=0.8, edgecolor='black', linewidth=1.2)
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
        if model in results and difficulty in results[model] and 'tar_001' in results[model][difficulty]:
            tars.append(results[model][difficulty]['tar_001'])
        else:
            tars.append(0)

    bars = ax.bar(x, tars, width, color=[THESIS_COLORS[m] for m in models],
                  alpha=0.8, edgecolor='black', linewidth=1.2)
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
        if model in results and difficulty in results[model] and 'genuine_similarity' in results[model][difficulty]:
            sims.append(results[model][difficulty]['genuine_similarity'])
        else:
            sims.append(0)

    bars = ax.bar(x, sims, width, color=[THESIS_COLORS[m] for m in models],
                  alpha=0.8, edgecolor='black', linewidth=1.2)
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

    plt.suptitle(f'Face Verification Performance Comparison ({difficulty.capitalize()} Test Set)',
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f'thesis_main_results_{difficulty}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_quality_verification_tradeoff(results, output_dir, difficulty='mixed'):
    """Plot Pareto frontier: PSNR vs. EER trade-off"""
    models = ['baseline', 'face_loss3', 'face_loss5']

    fig, ax = plt.subplots(figsize=(10, 8))

    psnr_values = []
    eer_values = []
    fr_weights = []

    for model in models:
        if model in results and difficulty in results[model]:
            if 'psnr' in results[model][difficulty] and 'eer' in results[model][difficulty]:
                psnr_values.append(results[model][difficulty]['psnr'])
                eer_values.append(results[model][difficulty]['eer'])

                if model == 'baseline':
                    fr_weights.append(0.0)
                elif model == 'face_loss3':
                    fr_weights.append(0.3)
                else:
                    fr_weights.append(0.5)

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
    output_path = os.path.join(output_dir, f'thesis_tradeoff_analysis_{difficulty}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_comprehensive_summary(results, output_dir, difficulty='mixed'):
    """Create comprehensive summary with multiple subplots"""
    models = ['baseline', 'face_loss3', 'face_loss5']

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # (a) EER
    ax1 = fig.add_subplot(gs[0, 0])
    eers = [results[m][difficulty]['eer'] if m in results and difficulty in results[m] and 'eer' in results[m][difficulty] else 0
            for m in models]
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
    tars = [results[m][difficulty]['tar_01'] if m in results and difficulty in results[m] and 'tar_01' in results[m][difficulty] else 0
            for m in models]
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
    sims = [results[m][difficulty]['genuine_similarity'] if m in results and difficulty in results[m] and 'genuine_similarity' in results[m][difficulty] else 0
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
    psnrs = [results[m][difficulty]['psnr'] if m in results and difficulty in results[m] and 'psnr' in results[m][difficulty] else 0
             for m in models]
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
    ssims = [results[m][difficulty]['ssim'] if m in results and difficulty in results[m] and 'ssim' in results[m][difficulty] else 0
             for m in models]
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
    valid_data = [(results[m][difficulty]['psnr'], results[m][difficulty]['eer'])
                  for m in models
                  if m in results and difficulty in results[m] and 'psnr' in results[m][difficulty] and 'eer' in results[m][difficulty]]
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

    fig.suptitle(f'Comprehensive Results Summary ({difficulty.capitalize()} Test Set)', fontsize=16, fontweight='bold')

    output_path = os.path.join(output_dir, f'thesis_comprehensive_summary_{difficulty}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate thesis summary figures')
    parser.add_argument('--eval_log', type=str, default='./face_eval.log',
                       help='Path to face evaluation log file')
    parser.add_argument('--output_dir', type=str, default='./figures/thesis',
                       help='Output directory for thesis figures')
    parser.add_argument('--difficulty', type=str, default='mixed',
                       choices=['easy', 'medium', 'hard', 'mixed'],
                       help='Difficulty level to plot')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("Thesis Summary Figure Generator")
    print("=" * 70)
    print(f"Eval log:         {args.eval_log}")
    print(f"Output directory:  {args.output_dir}")
    print(f"Difficulty:       {args.difficulty}")
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

    # Generate figures
    print("Generating thesis figures...")
    plot_main_results_2x2(results, args.output_dir, args.difficulty)
    plot_quality_verification_tradeoff(results, args.output_dir, args.difficulty)
    plot_comprehensive_summary(results, args.output_dir, args.difficulty)

    print()
    print("=" * 70)
    print("Thesis figures generated successfully!")
    print("=" * 70)
    print(f"\nFigures saved to: {args.output_dir}")
    print("\nGenerated figures:")
    print(f"  - thesis_main_results_{args.difficulty}.png: Main 2x2 results figure")
    print(f"  - thesis_tradeoff_analysis_{args.difficulty}.png: Pareto frontier analysis")
    print(f"  - thesis_comprehensive_summary_{args.difficulty}.png: Complete results summary")
    print("\nAll figures are 300 DPI for publication quality.")

    return 0


if __name__ == '__main__':
    exit(main())
