"""
Ablation Study Visualization Script

Generates publication-quality plots for thesis from ablation study results.

This script creates:
1. FR weight vs. verification metrics (EER, TAR@FAR)
2. FR weight vs. image quality (PSNR, SSIM)
3. TAR/FAR curves for all configurations
4. Genuine vs. Impostor score distributions
5. Trade-off plots (quality vs. verification)
6. Comparison bar charts

Usage:
    python plot_ablation_results.py --results_dir=./results/ablation
    python plot_ablation_results.py --results_dir=./results/ablation --output_dir=./figures
"""

import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

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


def parse_results_file(filepath):
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

            # Impostor pair similarity
            match = re.search(r'Impostor.*?Enhanced avg similarity:\s+([\d.]+)', content, re.DOTALL)
            if match:
                metrics['impostor_similarity'] = float(match.group(1))

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
        else:
            # Parse legacy metrics
            match = re.search(r'Enhanced.*?GT:\s+([\d.]+)', content)
            if match:
                metrics['similarity'] = float(match.group(1))

            match = re.search(r'Improvement:\s+([\d.]+)', content)
            if match:
                metrics['improvement'] = float(match.group(1))

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


def extract_fr_weight(config_name):
    """Extract FR weight from configuration name"""
    if 'baseline' in config_name.lower():
        return 0.0

    match = re.search(r'fr_weight_([\d.]+)', config_name)
    if match:
        return float(match.group(1))

    match = re.search(r'([\d.]+)', config_name)
    if match:
        return float(match.group(1))

    return None


def plot_fr_weight_vs_metrics(results, output_dir):
    """Plot FR weight vs. verification metrics"""
    configs = sorted(results.keys(), key=lambda x: extract_fr_weight(x) or 0)
    fr_weights = [extract_fr_weight(c) for c in configs]

    # Check if pairs-based
    is_pairs = results[configs[0]].get('is_pairs_based', False)

    if is_pairs:
        # Pairs-based metrics
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: EER
        eers = [results[c].get('eer', np.nan) for c in configs]
        ax1.plot(fr_weights, eers, 'o-', linewidth=2, markersize=8, color='#d62728')
        ax1.set_xlabel('Face Recognition Loss Weight')
        ax1.set_ylabel('Equal Error Rate (%)')
        ax1.set_title('EER vs. FR Loss Weight (Lower is Better)')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)

        # Plot 2: Genuine Similarity
        gen_sims = [results[c].get('genuine_similarity', np.nan) for c in configs]
        ax2.plot(fr_weights, gen_sims, 'o-', linewidth=2, markersize=8, color='#2ca02c')
        ax2.set_xlabel('Face Recognition Loss Weight')
        ax2.set_ylabel('Genuine Pair Similarity')
        ax2.set_title('Genuine Similarity vs. FR Loss Weight (Higher is Better)')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])

        # Plot 3: TAR @ FAR = 1%
        tar_01 = [results[c].get('tar_01', np.nan) for c in configs]
        ax3.plot(fr_weights, tar_01, 'o-', linewidth=2, markersize=8, color='#1f77b4')
        ax3.set_xlabel('Face Recognition Loss Weight')
        ax3.set_ylabel('TAR @ FAR=1% (%)')
        ax3.set_title('True Accept Rate @ FAR=1% vs. FR Loss Weight')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 100])

        # Plot 4: TAR @ FAR = 0.1%
        tar_001 = [results[c].get('tar_001', np.nan) for c in configs]
        ax4.plot(fr_weights, tar_001, 'o-', linewidth=2, markersize=8, color='#ff7f0e')
        ax4.set_xlabel('Face Recognition Loss Weight')
        ax4.set_ylabel('TAR @ FAR=0.1% (%)')
        ax4.set_title('True Accept Rate @ FAR=0.1% vs. FR Loss Weight')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([0, 100])

        plt.tight_layout()
        output_path = os.path.join(output_dir, 'fr_weight_vs_verification_metrics.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    else:
        # Legacy metrics
        fig, ax = plt.subplots(figsize=(8, 6))
        sims = [results[c].get('similarity', np.nan) for c in configs]
        ax.plot(fr_weights, sims, 'o-', linewidth=2, markersize=8, color='#2ca02c')
        ax.set_xlabel('Face Recognition Loss Weight')
        ax.set_ylabel('Face Similarity')
        ax.set_title('Face Similarity vs. FR Loss Weight')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        plt.tight_layout()
        output_path = os.path.join(output_dir, 'fr_weight_vs_similarity.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()


def plot_fr_weight_vs_quality(results, output_dir):
    """Plot FR weight vs. image quality metrics"""
    configs = sorted(results.keys(), key=lambda x: extract_fr_weight(x) or 0)
    fr_weights = [extract_fr_weight(c) for c in configs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # PSNR
    psnrs = [results[c].get('psnr', np.nan) for c in configs]
    ax1.plot(fr_weights, psnrs, 'o-', linewidth=2, markersize=8, color='#9467bd')
    ax1.set_xlabel('Face Recognition Loss Weight')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_title('PSNR vs. FR Loss Weight')
    ax1.grid(True, alpha=0.3)

    # SSIM
    ssims = [results[c].get('ssim', np.nan) for c in configs]
    ax2.plot(fr_weights, ssims, 'o-', linewidth=2, markersize=8, color='#8c564b')
    ax2.set_xlabel('Face Recognition Loss Weight')
    ax2.set_ylabel('SSIM')
    ax2.set_title('SSIM vs. FR Loss Weight')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'fr_weight_vs_image_quality.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_comparison_bars(results, output_dir):
    """Plot comparison bar charts for all configurations"""
    configs = sorted(results.keys(), key=lambda x: extract_fr_weight(x) or 0)
    config_labels = [c.replace('_', ' ').title() for c in configs]

    is_pairs = results[configs[0]].get('is_pairs_based', False)

    if is_pairs:
        # Create 2x2 subplot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        x = np.arange(len(configs))
        width = 0.6

        # EER
        eers = [results[c].get('eer', 0) for c in configs]
        bars1 = ax1.bar(x, eers, width, color='#d62728', alpha=0.8)
        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('Equal Error Rate (%)')
        ax1.set_title('EER Comparison (Lower is Better)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(config_labels, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        # Add value labels on bars
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{eers[i]:.2f}%', ha='center', va='bottom', fontsize=8)

        # Genuine Similarity
        gen_sims = [results[c].get('genuine_similarity', 0) for c in configs]
        bars2 = ax2.bar(x, gen_sims, width, color='#2ca02c', alpha=0.8)
        ax2.set_xlabel('Configuration')
        ax2.set_ylabel('Genuine Similarity')
        ax2.set_title('Genuine Pair Similarity (Higher is Better)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(config_labels, rotation=45, ha='right')
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3, axis='y')
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{gen_sims[i]:.3f}', ha='center', va='bottom', fontsize=8)

        # TAR @ FAR = 1%
        tar_01 = [results[c].get('tar_01', 0) for c in configs]
        bars3 = ax3.bar(x, tar_01, width, color='#1f77b4', alpha=0.8)
        ax3.set_xlabel('Configuration')
        ax3.set_ylabel('TAR @ FAR=1% (%)')
        ax3.set_title('True Accept Rate @ FAR=1%')
        ax3.set_xticks(x)
        ax3.set_xticklabels(config_labels, rotation=45, ha='right')
        ax3.set_ylim([0, 100])
        ax3.grid(True, alpha=0.3, axis='y')
        for i, bar in enumerate(bars3):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{tar_01[i]:.1f}%', ha='center', va='bottom', fontsize=8)

        # PSNR
        psnrs = [results[c].get('psnr', 0) for c in configs]
        bars4 = ax4.bar(x, psnrs, width, color='#9467bd', alpha=0.8)
        ax4.set_xlabel('Configuration')
        ax4.set_ylabel('PSNR (dB)')
        ax4.set_title('Image Quality (PSNR)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(config_labels, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3, axis='y')
        for i, bar in enumerate(bars4):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{psnrs[i]:.2f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        output_path = os.path.join(output_dir, 'comparison_bars.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()


def plot_quality_vs_verification_tradeoff(results, output_dir):
    """Plot trade-off between image quality and verification accuracy"""
    configs = sorted(results.keys(), key=lambda x: extract_fr_weight(x) or 0)

    is_pairs = results[configs[0]].get('is_pairs_based', False)

    if is_pairs:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # PSNR vs EER
        psnrs = [results[c].get('psnr', np.nan) for c in configs]
        eers = [results[c].get('eer', np.nan) for c in configs]
        fr_weights = [extract_fr_weight(c) for c in configs]

        scatter1 = ax1.scatter(psnrs, eers, c=fr_weights, s=100, cmap='viridis', alpha=0.7)
        ax1.set_xlabel('PSNR (dB)')
        ax1.set_ylabel('Equal Error Rate (%)')
        ax1.set_title('Trade-off: Image Quality vs. Verification Accuracy')
        ax1.grid(True, alpha=0.3)
        # Annotate points with FR weights
        for i, (p, e, w) in enumerate(zip(psnrs, eers, fr_weights)):
            ax1.annotate(f'λ={w}', (p, e), xytext=(5, 5), textcoords='offset points', fontsize=8)
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('FR Loss Weight')

        # PSNR vs Genuine Similarity
        gen_sims = [results[c].get('genuine_similarity', np.nan) for c in configs]
        scatter2 = ax2.scatter(psnrs, gen_sims, c=fr_weights, s=100, cmap='viridis', alpha=0.7)
        ax2.set_xlabel('PSNR (dB)')
        ax2.set_ylabel('Genuine Pair Similarity')
        ax2.set_title('Trade-off: Image Quality vs. Face Similarity')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        for i, (p, s, w) in enumerate(zip(psnrs, gen_sims, fr_weights)):
            ax2.annotate(f'λ={w}', (p, s), xytext=(5, 5), textcoords='offset points', fontsize=8)
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('FR Loss Weight')

        plt.tight_layout()
        output_path = os.path.join(output_dir, 'quality_vs_verification_tradeoff.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()


def create_summary_figure(results, output_dir):
    """Create comprehensive summary figure for thesis"""
    configs = sorted(results.keys(), key=lambda x: extract_fr_weight(x) or 0)
    fr_weights = [extract_fr_weight(c) for c in configs]

    is_pairs = results[configs[0]].get('is_pairs_based', False)

    if not is_pairs:
        print("⚠ Summary figure requires pairs-based evaluation results")
        return

    # Create 2x3 subplot
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    # Plot 1: EER
    eers = [results[c].get('eer', np.nan) for c in configs]
    ax1.plot(fr_weights, eers, 'o-', linewidth=2, markersize=8, color='#d62728')
    ax1.set_xlabel('FR Loss Weight')
    ax1.set_ylabel('EER (%)')
    ax1.set_title('(a) Equal Error Rate')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Plot 2: Genuine Similarity
    gen_sims = [results[c].get('genuine_similarity', np.nan) for c in configs]
    ax2.plot(fr_weights, gen_sims, 'o-', linewidth=2, markersize=8, color='#2ca02c')
    ax2.set_xlabel('FR Loss Weight')
    ax2.set_ylabel('Similarity')
    ax2.set_title('(b) Genuine Pair Similarity')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    # Plot 3: TAR @ FAR=1%
    tar_01 = [results[c].get('tar_01', np.nan) for c in configs]
    ax3.plot(fr_weights, tar_01, 'o-', linewidth=2, markersize=8, color='#1f77b4')
    ax3.set_xlabel('FR Loss Weight')
    ax3.set_ylabel('TAR (%)')
    ax3.set_title('(c) TAR @ FAR=1%')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 100])

    # Plot 4: PSNR
    psnrs = [results[c].get('psnr', np.nan) for c in configs]
    ax4.plot(fr_weights, psnrs, 'o-', linewidth=2, markersize=8, color='#9467bd')
    ax4.set_xlabel('FR Loss Weight')
    ax4.set_ylabel('PSNR (dB)')
    ax4.set_title('(d) Image Quality (PSNR)')
    ax4.grid(True, alpha=0.3)

    # Plot 5: SSIM
    ssims = [results[c].get('ssim', np.nan) for c in configs]
    ax5.plot(fr_weights, ssims, 'o-', linewidth=2, markersize=8, color='#8c564b')
    ax5.set_xlabel('FR Loss Weight')
    ax5.set_ylabel('SSIM')
    ax5.set_title('(e) Image Quality (SSIM)')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, 1])

    # Plot 6: Trade-off (PSNR vs EER)
    scatter = ax6.scatter(psnrs, eers, c=fr_weights, s=100, cmap='viridis', alpha=0.7)
    ax6.set_xlabel('PSNR (dB)')
    ax6.set_ylabel('EER (%)')
    ax6.set_title('(f) Quality-Verification Trade-off')
    ax6.grid(True, alpha=0.3)
    for i, (p, e, w) in enumerate(zip(psnrs, eers, fr_weights)):
        ax6.annotate(f'{w}', (p, e), xytext=(5, 5), textcoords='offset points', fontsize=8)
    cbar = plt.colorbar(scatter, ax=ax6)
    cbar.set_label('FR Weight')

    fig.suptitle('Ablation Study: Face Recognition Loss Weight Analysis', fontsize=14, fontweight='bold')

    output_path = os.path.join(output_dir, 'summary_figure.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate ablation study visualizations')
    parser.add_argument('--results_dir', type=str, default='./results/ablation',
                       help='Directory containing ablation results')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for figures (default: results_dir/figures)')
    args = parser.parse_args()

    # Setup output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, 'figures')

    os.makedirs(args.output_dir, exist_ok=True)

    print("="*70)
    print("Ablation Study Visualization Generator")
    print("="*70)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory:  {args.output_dir}")
    print()

    # Discover configuration directories
    if not os.path.exists(args.results_dir):
        print(f"✗ Results directory not found: {args.results_dir}")
        return 1

    configs = []
    for item in os.listdir(args.results_dir):
        item_path = os.path.join(args.results_dir, item)
        if os.path.isdir(item_path):
            result_file = os.path.join(item_path, 'face_verification_results.txt')
            if os.path.exists(result_file):
                configs.append(item)

    if not configs:
        print(f"✗ No configuration results found in {args.results_dir}")
        print("  Expected structure: results_dir/<config>/face_verification_results.txt")
        return 1

    print(f"Found {len(configs)} configurations:")
    for config in sorted(configs):
        print(f"  • {config}")
    print()

    # Parse all results
    print("Parsing results...")
    results = {}
    for config in configs:
        result_file = os.path.join(args.results_dir, config, 'face_verification_results.txt')
        parsed = parse_results_file(result_file)
        if parsed:
            results[config] = parsed
            print(f"  ✓ {config}")
        else:
            print(f"  ✗ {config} (failed to parse)")

    if not results:
        print("\n✗ No valid results parsed")
        return 1

    print(f"\nSuccessfully parsed {len(results)} configurations")
    print()

    # Generate plots
    print("Generating visualizations...")
    print()

    try:
        plot_fr_weight_vs_metrics(results, args.output_dir)
        plot_fr_weight_vs_quality(results, args.output_dir)
        plot_comparison_bars(results, args.output_dir)
        plot_quality_vs_verification_tradeoff(results, args.output_dir)
        create_summary_figure(results, args.output_dir)
    except Exception as e:
        print(f"\n✗ Error generating plots: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print()
    print("="*70)
    print("Visualization complete!")
    print("="*70)
    print(f"\nGenerated figures saved to: {args.output_dir}")
    print()
    print("Files created:")
    for filename in sorted(os.listdir(args.output_dir)):
        if filename.endswith('.png'):
            filepath = os.path.join(args.output_dir, filename)
            print(f"  • {filename}")
    print()
    print("These figures are ready to include in your thesis!")
    print()

    return 0


if __name__ == '__main__':
    exit(main())
