#!/usr/bin/env python3
"""
Generate Comprehensive Thesis Results with Statistical Analysis

This script:
1. Loads evaluation results from all 4 models
2. Computes statistical significance tests
3. Generates comparison tables
4. Creates publication-ready visualizations
5. Identifies best configuration with statistical proof

Usage:
    python generate_thesis_results.py --results_dir ./results/full_evaluation
"""

import os
import argparse
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

# Statistical tests
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    print("Warning: scipy not available. Statistical tests will be skipped.")
    print("Install with: pip install scipy")
    SCIPY_AVAILABLE = False

# Plotting
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib not available. Visualizations will be skipped.")
    print("Install with: pip install matplotlib")
    MATPLOTLIB_AVAILABLE = False


def parse_result_file(result_file):
    """Parse face verification results file"""
    results = {
        'num_genuine': 0,
        'num_impostor': 0,
        'genuine_mean_low': 0,
        'genuine_std_low': 0,
        'genuine_mean_enhanced': 0,
        'genuine_std_enhanced': 0,
        'impostor_mean_low': 0,
        'impostor_std_low': 0,
        'impostor_mean_enhanced': 0,
        'impostor_std_enhanced': 0,
        'eer_low': 0,
        'eer_enhanced': 0,
        'tar_001_low': 0,
        'tar_001_enhanced': 0,
        'tar_01_low': 0,
        'tar_01_enhanced': 0,
        'psnr_mean': 0,
        'ssim_mean': 0,
    }

    if not os.path.exists(result_file):
        return None

    with open(result_file, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        line = line.strip()

        # Parse counts
        if 'Genuine pairs:' in line:
            results['num_genuine'] = int(line.split(':')[1].strip())
        elif 'Impostor pairs:' in line:
            results['num_impostor'] = int(line.split(':')[1].strip())

        # Parse genuine pair scores
        elif 'Low-light avg similarity:' in line and i > 0 and 'Genuine' in lines[i-1]:
            parts = line.split()
            results['genuine_mean_low'] = float(parts[3])
            results['genuine_std_low'] = float(parts[5])
        elif 'Enhanced avg similarity:' in line and i > 0 and 'Genuine' in lines[i-2]:
            parts = line.split()
            results['genuine_mean_enhanced'] = float(parts[3])
            results['genuine_std_enhanced'] = float(parts[5])

        # Parse impostor pair scores
        elif 'Low-light avg similarity:' in line and i > 0 and 'Impostor' in lines[i-1]:
            parts = line.split()
            results['impostor_mean_low'] = float(parts[3])
            results['impostor_std_low'] = float(parts[5])
        elif 'Enhanced avg similarity:' in line and i > 0 and 'Impostor' in lines[i-2]:
            parts = line.split()
            results['impostor_mean_enhanced'] = float(parts[3])
            results['impostor_std_enhanced'] = float(parts[5])

        # Parse EER
        elif 'Low-light:' in line and 'EER' in lines[i-1]:
            parts = line.split()
            results['eer_low'] = float(parts[1].strip('%'))
        elif 'Enhanced:' in line and 'EER' in lines[i-1]:
            parts = line.split()
            results['eer_enhanced'] = float(parts[1].strip('%'))

        # Parse TAR@FAR
        elif 'Low-light:' in line and 'TAR @ FAR=0.1%' in lines[i-1]:
            parts = line.split()
            results['tar_001_low'] = float(parts[1].strip('%'))
        elif 'Enhanced:' in line and 'TAR @ FAR=0.1%' in lines[i-2]:
            parts = line.split()
            results['tar_001_enhanced'] = float(parts[1].strip('%'))
        elif 'Low-light:' in line and 'TAR @ FAR=1%' in lines[i-1]:
            parts = line.split()
            results['tar_01_low'] = float(parts[1].strip('%'))
        elif 'Enhanced:' in line and 'TAR @ FAR=1%' in lines[i-2]:
            parts = line.split()
            results['tar_01_enhanced'] = float(parts[1].strip('%'))

        # Parse image quality
        elif 'Average PSNR:' in line:
            parts = line.split()
            results['psnr_mean'] = float(parts[2])
        elif 'Average SSIM:' in line:
            parts = line.split()
            results['ssim_mean'] = float(parts[2])

    return results


def compute_improvement(baseline, model):
    """Compute improvements over baseline"""
    improvements = {
        'genuine_sim': model['genuine_mean_enhanced'] - baseline['genuine_mean_enhanced'],
        'impostor_sim': model['impostor_mean_enhanced'] - baseline['impostor_mean_enhanced'],
        'eer': baseline['eer_enhanced'] - model['eer_enhanced'],  # Lower is better
        'tar_001': model['tar_001_enhanced'] - baseline['tar_001_enhanced'],
        'tar_01': model['tar_01_enhanced'] - baseline['tar_01_enhanced'],
        'psnr': model['psnr_mean'] - baseline['psnr_mean'],
        'ssim': model['ssim_mean'] - baseline['ssim_mean'],
    }
    return improvements


def statistical_significance(baseline, model):
    """
    Compute statistical significance using paired t-test

    Note: This is approximate since we don't have raw pair-wise scores.
    For proper significance testing, we'd need the raw similarity scores for each pair.

    Here we approximate using the reported means and standard deviations.
    """
    if not SCIPY_AVAILABLE:
        return {'available': False}

    results = {}

    # Approximate t-test for genuine pair similarity improvement
    # Using two-sample t-test as approximation
    n1 = baseline['num_genuine']
    n2 = model['num_genuine']

    if n1 > 0 and n2 > 0:
        # Genuine pair improvement
        mean_diff = model['genuine_mean_enhanced'] - baseline['genuine_mean_enhanced']
        pooled_std = np.sqrt((baseline['genuine_std_enhanced']**2 + model['genuine_std_enhanced']**2) / 2)

        if pooled_std > 0:
            # Compute t-statistic
            t_stat = mean_diff / (pooled_std * np.sqrt(2/n1))
            # Two-tailed p-value
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n1-1))

            results['genuine_similarity'] = {
                'mean_diff': mean_diff,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        else:
            results['genuine_similarity'] = {'error': 'Zero standard deviation'}

        # EER improvement
        eer_diff = baseline['eer_enhanced'] - model['eer_enhanced']
        results['eer_improvement'] = {
            'improvement': eer_diff,
            'baseline': baseline['eer_enhanced'],
            'model': model['eer_enhanced']
        }

    results['available'] = True
    return results


def generate_comparison_table(results_dict, output_file):
    """Generate formatted comparison table"""

    # Get all configurations from results_dict
    configs = sorted(results_dict.keys())

    with open(output_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write("COMPREHENSIVE ABLATION STUDY RESULTS\n")
        f.write("Face Recognition Perceptual Loss for Low-Light Face Enhancement\n")
        f.write("="*100 + "\n\n")

        # Table 1: Verification Performance
        f.write("TABLE 1: FACE VERIFICATION PERFORMANCE\n")
        f.write("-"*120 + "\n")
        f.write(f"{'Configuration':<25} | {'Genuine Sim':<12} | {'Impostor Sim':<12} | {'EER (%)':<10} | {'TAR@0.1%(%)':<12} | {'TAR@1%(%)':<12}\n")
        f.write("-"*120 + "\n")

        for config in configs:
            if config in results_dict and results_dict[config]:
                r = results_dict[config]
                f.write(f"{config:<25} | {r['genuine_mean_enhanced']:>12.4f} | {r['impostor_mean_enhanced']:>12.4f} | "
                       f"{r['eer_enhanced']:>10.2f} | {r['tar_001_enhanced']:>12.2f} | {r['tar_01_enhanced']:>12.2f}\n")
            else:
                f.write(f"{config:<25} | {'N/A':<12} | {'N/A':<12} | {'N/A':<10} | {'N/A':<12} | {'N/A':<12}\n")

        f.write("\n")

        # Table 2: Image Quality
        f.write("TABLE 2: IMAGE QUALITY METRICS\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Configuration':<25} | {'PSNR (dB)':<15} | {'SSIM':<15}\n")
        f.write("-"*70 + "\n")

        for config in configs:
            if config in results_dict and results_dict[config]:
                r = results_dict[config]
                f.write(f"{config:<25} | {r['psnr_mean']:>15.2f} | {r['ssim_mean']:>15.4f}\n")
            else:
                f.write(f"{config:<25} | {'N/A':<15} | {'N/A':<15}\n")

        f.write("\n")

        # Table 3: Improvements over Baseline (per d_weight)
        # Group baselines by d_weight
        baselines = {c: results_dict[c] for c in configs if c.startswith('baseline_') and c in results_dict}
        
        if baselines:
            f.write("TABLE 3: IMPROVEMENTS OVER BASELINE (per D_weight)\n")
            f.write("-"*120 + "\n")
            f.write(f"{'Configuration':<25} | {'Baseline':<20} | {'ΔGenuine':<12} | {'ΔImpostor':<12} | {'ΔEER':<10} | {'ΔTAR@1%':<12}\n")
            f.write("-"*120 + "\n")

            # For each FR config, compare to baseline with same d_weight
            for config in configs:
                if not config.startswith('baseline_') and config in results_dict and results_dict[config]:
                    # Extract d_weight from config name (e.g., "fr_weight_0.5_d1" -> "d1")
                    parts = config.split('_')
                    d_weight_part = [p for p in parts if p.startswith('d')]
                    if d_weight_part:
                        d_suffix = d_weight_part[0]
                        baseline_key = f"baseline_{d_suffix}"
                        
                        if baseline_key in baselines:
                            baseline = baselines[baseline_key]
                            imp = compute_improvement(baseline, results_dict[config])
                            f.write(f"{config:<25} | {baseline_key:<20} | {imp['genuine_sim']:>+12.4f} | {imp['impostor_sim']:>+12.4f} | "
                                   f"{imp['eer']:>+10.2f} | {imp['tar_01']:>+12.2f}\n")

            f.write("\n")
            f.write("Note: Each FR configuration is compared to baseline with the same D_weight.\n")
            f.write("      Δ indicates change from baseline. For EER, negative is better (lower error).\n")
            f.write("      For all other metrics, positive is better.\n\n")

        # Statistical Significance
        baselines = {c: results_dict[c] for c in configs if c.startswith('baseline_') and c in results_dict}
        
        if SCIPY_AVAILABLE and baselines:
            f.write("="*100 + "\n")
            f.write("STATISTICAL SIGNIFICANCE ANALYSIS (vs baseline with same D_weight)\n")
            f.write("="*100 + "\n\n")

            for config in configs:
                if not config.startswith('baseline_') and config in results_dict and results_dict[config]:
                    # Find matching baseline
                    parts = config.split('_')
                    d_weight_part = [p for p in parts if p.startswith('d')]
                    if d_weight_part:
                        d_suffix = d_weight_part[0]
                        baseline_key = f"baseline_{d_suffix}"
                        
                        if baseline_key in baselines:
                            baseline = baselines[baseline_key]
                            f.write(f"\n{config} vs {baseline_key}:\n")
                            f.write("-"*60 + "\n")

                            sig = statistical_significance(baseline, results_dict[config])
                            if sig['available'] and 'genuine_similarity' in sig:
                                gs = sig['genuine_similarity']
                                if 'p_value' in gs:
                                    f.write(f"  Genuine Similarity Improvement: {gs['mean_diff']:+.4f}\n")
                                    f.write(f"  t-statistic: {gs['t_statistic']:.4f}\n")
                                    f.write(f"  p-value: {gs['p_value']:.6f}\n")
                                    f.write(f"  Significant (p < 0.05): {'YES ✓' if gs['significant'] else 'NO ✗'}\n")

                            if 'eer_improvement' in sig:
                                eer = sig['eer_improvement']
                                f.write(f"\n  EER Improvement: {eer['improvement']:+.2f}%\n")
                                f.write(f"    Baseline EER: {eer['baseline']:.2f}%\n")
                                f.write(f"    Model EER: {eer['model']:.2f}%\n")

        # Key Findings
        f.write("\n" + "="*100 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("="*100 + "\n\n")

        baselines = {c: results_dict[c] for c in configs if c.startswith('baseline_') and c in results_dict}
        
        if baselines:
            # Find best configuration overall
            best_eer_config = min(configs, key=lambda c: results_dict[c]['eer_enhanced'] if c in results_dict and results_dict[c] else float('inf'))
            best_tar_config = max(configs, key=lambda c: results_dict[c]['tar_01_enhanced'] if c in results_dict and results_dict[c] else 0)
            best_genuine_config = max(configs, key=lambda c: results_dict[c]['genuine_mean_enhanced'] if c in results_dict and results_dict[c] else 0)
            
            # Also find best baseline
            best_baseline = min(baselines.keys(), key=lambda c: baselines[c]['eer_enhanced'])

            f.write(f"1. Best Baseline: {best_baseline}\n")
            f.write(f"   EER: {baselines[best_baseline]['eer_enhanced']:.2f}%\n")
            f.write(f"   TAR@FAR=1%: {baselines[best_baseline]['tar_01_enhanced']:.2f}%\n\n")
            
            f.write(f"2. Best Overall Configuration (Lowest EER): {best_eer_config}\n")
            f.write(f"   EER: {results_dict[best_eer_config]['eer_enhanced']:.2f}%\n")
            f.write(f"   TAR@FAR=1%: {results_dict[best_eer_config]['tar_01_enhanced']:.2f}%\n\n")

            f.write(f"3. Best TAR@FAR=1%: {best_tar_config}\n")
            f.write(f"   TAR: {results_dict[best_tar_config]['tar_01_enhanced']:.2f}%\n")
            f.write(f"   EER: {results_dict[best_tar_config]['eer_enhanced']:.2f}%\n\n")

            f.write(f"4. Best Genuine Similarity: {best_genuine_config}\n")
            f.write(f"   Similarity: {results_dict[best_genuine_config]['genuine_mean_enhanced']:.4f}\n\n")

            # Overall recommendation
            f.write("5. RECOMMENDED CONFIGURATION: ")
            # Typically, we'd choose based on best TAR@FAR or EER
            if best_eer_config == best_tar_config:
                f.write(f"{best_eer_config}\n")
                f.write(f"   Reason: Best performance on both EER and TAR@FAR metrics\n")
            else:
                f.write(f"{best_tar_config}\n")
                f.write(f"   Reason: Best TAR@FAR=1% (most relevant for real-world verification)\n")
            
            # Show improvement over best baseline
            # Find the baseline with matching d_weight
            parts = best_tar_config.split('_')
            d_weight_part = [p for p in parts if p.startswith('d')]
            if d_weight_part and not best_tar_config.startswith('baseline_'):
                d_suffix = d_weight_part[0]
                matching_baseline = f"baseline_{d_suffix}"
                if matching_baseline in baselines:
                    imp = compute_improvement(baselines[matching_baseline], results_dict[best_tar_config])
                    f.write(f"\n   Improvement over {matching_baseline}:\n")
                    f.write(f"     EER: {imp['eer']:+.2f}% (lower is better)\n")
                    f.write(f"     TAR@FAR=1%: {imp['tar_01']:+.2f}% (higher is better)\n")
                    f.write(f"     Genuine Similarity: {imp['genuine_sim']:+.4f}\n")

        f.write("\n" + "="*100 + "\n")


def plot_results(results_dict, output_dir):
    """Generate publication-quality plots"""

    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Skipping plots.")
        return

    configs = ['baseline', 'fr_weight_0.3', 'fr_weight_0.5', 'fr_weight_1.0']
    config_labels = ['Baseline', 'FR=0.3', 'FR=0.5', 'FR=1.0']

    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Verification Metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Face Verification Performance Comparison', fontsize=16, fontweight='bold')

    # Genuine similarity
    genuine_sims = [results_dict[c]['genuine_mean_enhanced'] if c in results_dict and results_dict[c] else 0 for c in configs]
    axes[0, 0].bar(config_labels, genuine_sims, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    axes[0, 0].set_ylabel('Cosine Similarity', fontsize=12)
    axes[0, 0].set_title('Genuine Pair Similarity (Higher is Better)', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylim([0, 1.0])
    axes[0, 0].grid(axis='y', alpha=0.3)

    # Impostor similarity
    impostor_sims = [results_dict[c]['impostor_mean_enhanced'] if c in results_dict and results_dict[c] else 0 for c in configs]
    axes[0, 1].bar(config_labels, impostor_sims, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    axes[0, 1].set_ylabel('Cosine Similarity', fontsize=12)
    axes[0, 1].set_title('Impostor Pair Similarity (Lower is Better)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylim([0, 1.0])
    axes[0, 1].grid(axis='y', alpha=0.3)

    # EER
    eers = [results_dict[c]['eer_enhanced'] if c in results_dict and results_dict[c] else 0 for c in configs]
    axes[1, 0].bar(config_labels, eers, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    axes[1, 0].set_ylabel('EER (%)', fontsize=12)
    axes[1, 0].set_title('Equal Error Rate (Lower is Better)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)

    # TAR@FAR
    tars = [results_dict[c]['tar_01_enhanced'] if c in results_dict and results_dict[c] else 0 for c in configs]
    axes[1, 1].bar(config_labels, tars, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    axes[1, 1].set_ylabel('TAR (%)', fontsize=12)
    axes[1, 1].set_title('True Accept Rate @ FAR=1% (Higher is Better)', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylim([0, 100])
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'verification_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Image Quality
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Image Quality Metrics', fontsize=16, fontweight='bold')

    psnrs = [results_dict[c]['psnr_mean'] if c in results_dict and results_dict[c] else 0 for c in configs]
    axes[0].bar(config_labels, psnrs, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    axes[0].set_ylabel('PSNR (dB)', fontsize=12)
    axes[0].set_title('Peak Signal-to-Noise Ratio', fontsize=12, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)

    ssims = [results_dict[c]['ssim_mean'] if c in results_dict and results_dict[c] else 0 for c in configs]
    axes[1].bar(config_labels, ssims, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    axes[1].set_ylabel('SSIM', fontsize=12)
    axes[1].set_title('Structural Similarity Index', fontsize=12, fontweight='bold')
    axes[1].set_ylim([0, 1.0])
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'image_quality.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 3: Trade-off Analysis (if baseline exists)
    if 'baseline' in results_dict and results_dict['baseline']:
        fig, ax = plt.subplots(figsize=(10, 8))

        for i, config in enumerate(configs):
            if config in results_dict and results_dict[config]:
                r = results_dict[config]
                ax.scatter(r['eer_enhanced'], r['tar_01_enhanced'],
                          s=300, label=config_labels[i],
                          marker='o', alpha=0.7)
                ax.annotate(config_labels[i],
                           (r['eer_enhanced'], r['tar_01_enhanced']),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=10, fontweight='bold')

        ax.set_xlabel('Equal Error Rate (%) [Lower is Better]', fontsize=12, fontweight='bold')
        ax.set_ylabel('TAR @ FAR=1% (%) [Higher is Better]', fontsize=12, fontweight='bold')
        ax.set_title('Verification Performance Trade-off\n(Ideal point: Bottom-right)',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'tradeoff_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"✓ Plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive thesis results')
    parser.add_argument('--results_dir', type=str, default='./results/full_evaluation',
                       help='Directory containing evaluation results')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (defaults to results_dir)')

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.results_dir

    print("="*80)
    print("COMPREHENSIVE THESIS RESULTS GENERATOR")
    print("="*80)
    print()

    # Load results from all configurations
    configs = ['baseline', 'fr_weight_0.3', 'fr_weight_0.5', 'fr_weight_1.0']
    results_dict = {}

    print("Loading evaluation results...")
    
    # Look for all d_weight variations
    d_weights = ['0.5', '1', '1.5']
    
    for config in configs:
        for d_weight in d_weights:
            config_key = f"{config}_d{d_weight}"
            result_file = os.path.join(args.results_dir, config_key, 'face_verification_results.txt')
            
            if os.path.exists(result_file):
                results = parse_result_file(result_file)
                if results:
                    results_dict[config_key] = results
                    print(f"  ✓ {config_key}: ({results['num_genuine']} genuine + {results['num_impostor']} impostor pairs)")
                else:
                    print(f"  ⚠ Failed to parse: {config_key}")

    if not results_dict:
        print("\n✗ Error: No results found!")
        print(f"   Check that evaluation results exist in: {args.results_dir}")
        return 1

    print()
    print(f"Loaded {len(results_dict)} configuration(s)")
    print()

    # Generate comparison table
    table_file = os.path.join(args.output_dir, 'thesis_results_summary.txt')
    print(f"Generating comparison table...")
    generate_comparison_table(results_dict, table_file)
    print(f"✓ Saved to: {table_file}")
    print()

    # Generate plots
    plot_dir = os.path.join(args.output_dir, 'plots')
    print(f"Generating publication-quality plots...")
    plot_results(results_dict, plot_dir)
    print()

    # Display summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    with open(table_file, 'r') as f:
        content = f.read()
        # Print key findings section
        if 'KEY FINDINGS' in content:
            key_findings = content.split('KEY FINDINGS')[1].split('='*100)[0]
            print(key_findings)

    print("="*80)
    print("✓ All results generated successfully!")
    print("="*80)
    print()
    print("Files created:")
    print(f"  • {table_file}")
    if MATPLOTLIB_AVAILABLE:
        print(f"  • {plot_dir}/verification_metrics.png")
        print(f"  • {plot_dir}/image_quality.png")
        print(f"  • {plot_dir}/tradeoff_analysis.png")
    print()

    return 0


if __name__ == '__main__':
    exit(main())
