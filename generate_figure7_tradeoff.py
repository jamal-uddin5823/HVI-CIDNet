"""
Figure 7: Quality-Performance Trade-off Analysis (4-panel)
Panels: A) PSNR vs EER, B) SSIM vs EER, C) PSNR vs TAR, D) SSIM vs TAR
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import json
import numpy as np

def generate_figure7():
    """Generate Figure 7: Quality-Performance Trade-off (4 panels)"""

    # Load extracted data
    with open('thesis_data_extracted.json', 'r') as f:
        data = json.load(f)

    eval_data = data['evaluation']

    # Setup figure with 2x2 grid
    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Color scheme (colorblind-friendly)
    colors = {
        'baseline': '#0173B2',      # Blue
        'face_loss3': '#029E73',    # Green
        'face_loss5': '#DE8F05'     # Orange
    }

    # Marker styles for difficulty levels
    markers = {
        'easy': 'o',      # Circle
        'medium': '^',    # Triangle
        'hard': 's',      # Square
        'mixed': 'D'      # Diamond
    }

    marker_sizes = {
        'easy': 200,
        'medium': 150,
        'hard': 120,
        'mixed': 180
    }

    difficulties = ['easy', 'medium', 'hard', 'mixed']
    model_names = ['baseline', 'face_loss3', 'face_loss5']

    # Collect all data points for axis limit calculation
    all_psnr = []
    all_ssim = []
    all_eer = []
    all_tar = []

    for model_name in model_names:
        for diff in difficulties:
            all_psnr.append(eval_data[model_name][diff]['psnr'])
            all_ssim.append(eval_data[model_name][diff]['ssim'])
            all_eer.append(eval_data[model_name][diff]['eer'])
            all_tar.append(eval_data[model_name][diff]['tar_001'])

    # Calculate axis limits with padding
    psnr_min, psnr_max = min(all_psnr) - 1, max(all_psnr) + 1
    ssim_min, ssim_max = min(all_ssim) - 0.02, max(all_ssim) + 0.02
    eer_min, eer_max = -0.05, max(all_eer) + 0.15
    tar_min, tar_max = min(all_tar) - 1, max(all_tar) + 0.5

    # ========== PANEL A: PSNR vs. EER ==========
    ax_a = fig.add_subplot(gs[0, 0])

    for model_name in model_names:
        for diff in difficulties:
            psnr = eval_data[model_name][diff]['psnr']
            eer = eval_data[model_name][diff]['eer']

            print(f"Panel A - {model_name}/{diff}: PSNR={psnr:.2f}, EER={eer:.2f}")

            ax_a.scatter(psnr, eer,
                        c=colors[model_name],
                        marker=markers[diff],
                        s=marker_sizes[diff],
                        edgecolors='black',
                        linewidths=1.5,
                        alpha=0.8)

    ax_a.set_xlabel('PSNR (dB)', fontsize=12, fontweight='bold')
    ax_a.set_ylabel('EER (%)', fontsize=12, fontweight='bold')
    ax_a.set_title('A', fontsize=14, fontweight='bold', loc='left')
    ax_a.grid(True, alpha=0.3, linestyle='--')
    ax_a.set_xlim(psnr_min, psnr_max)
    ax_a.set_ylim(eer_min, eer_max)

    # Add ideal region annotation (lower-right: high PSNR, low EER)
    ax_a.annotate('Ideal', xy=(0.85, 0.15), xycoords='axes fraction',
                 fontsize=10, color='green', fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))

    # ========== PANEL B: SSIM vs. EER ==========
    ax_b = fig.add_subplot(gs[0, 1])

    for model_name in model_names:
        for diff in difficulties:
            ssim = eval_data[model_name][diff]['ssim']
            eer = eval_data[model_name][diff]['eer']

            print(f"Panel B - {model_name}/{diff}: SSIM={ssim:.4f}, EER={eer:.2f}")

            ax_b.scatter(ssim, eer,
                        c=colors[model_name],
                        marker=markers[diff],
                        s=marker_sizes[diff],
                        edgecolors='black',
                        linewidths=1.5,
                        alpha=0.8)

    ax_b.set_xlabel('SSIM', fontsize=12, fontweight='bold')
    ax_b.set_ylabel('EER (%)', fontsize=12, fontweight='bold')
    ax_b.set_title('B', fontsize=14, fontweight='bold', loc='left')
    ax_b.grid(True, alpha=0.3, linestyle='--')
    ax_b.set_xlim(ssim_min, ssim_max)
    ax_b.set_ylim(eer_min, eer_max)

    # Add ideal region annotation (lower-right: high SSIM, low EER)
    ax_b.annotate('Ideal', xy=(0.85, 0.15), xycoords='axes fraction',
                 fontsize=10, color='green', fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))

    # ========== PANEL C: PSNR vs. TAR@FAR=0.1% ==========
    ax_c = fig.add_subplot(gs[1, 0])

    for model_name in model_names:
        for diff in difficulties:
            psnr = eval_data[model_name][diff]['psnr']
            tar = eval_data[model_name][diff]['tar_001']

            print(f"Panel C - {model_name}/{diff}: PSNR={psnr:.2f}, TAR={tar:.1f}")

            ax_c.scatter(psnr, tar,
                        c=colors[model_name],
                        marker=markers[diff],
                        s=marker_sizes[diff],
                        edgecolors='black',
                        linewidths=1.5,
                        alpha=0.8)

    ax_c.set_xlabel('PSNR (dB)', fontsize=12, fontweight='bold')
    ax_c.set_ylabel('TAR @ FAR=0.1% (%)', fontsize=12, fontweight='bold')
    ax_c.set_title('C', fontsize=14, fontweight='bold', loc='left')
    ax_c.grid(True, alpha=0.3, linestyle='--')
    ax_c.set_xlim(psnr_min, psnr_max)
    ax_c.set_ylim(tar_min, tar_max)

    # Add ideal region annotation (upper-right: high PSNR, high TAR)
    ax_c.annotate('Ideal', xy=(0.85, 0.85), xycoords='axes fraction',
                 fontsize=10, color='green', fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))

    # ========== PANEL D: SSIM vs. TAR@FAR=0.1% ==========
    ax_d = fig.add_subplot(gs[1, 1])

    for model_name in model_names:
        for diff in difficulties:
            ssim = eval_data[model_name][diff]['ssim']
            tar = eval_data[model_name][diff]['tar_001']

            print(f"Panel D - {model_name}/{diff}: SSIM={ssim:.4f}, TAR={tar:.1f}")

            ax_d.scatter(ssim, tar,
                        c=colors[model_name],
                        marker=markers[diff],
                        s=marker_sizes[diff],
                        edgecolors='black',
                        linewidths=1.5,
                        alpha=0.8)

    ax_d.set_xlabel('SSIM', fontsize=12, fontweight='bold')
    ax_d.set_ylabel('TAR @ FAR=0.1% (%)', fontsize=12, fontweight='bold')
    ax_d.set_title('D', fontsize=14, fontweight='bold', loc='left')
    ax_d.grid(True, alpha=0.3, linestyle='--')
    ax_d.set_xlim(ssim_min, ssim_max)
    ax_d.set_ylim(tar_min, tar_max)

    # Add ideal region annotation (upper-right: high SSIM, high TAR)
    ax_d.annotate('Ideal', xy=(0.85, 0.85), xycoords='axes fraction',
                 fontsize=10, color='green', fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))

    # ========== LEGENDS ==========
    # Create custom legend handles for models
    model_labels = {
        'baseline': 'Baseline',
        'face_loss3': 'Face Loss (FR=0.3)',
        'face_loss5': 'Face Loss (FR=0.5)'
    }

    model_handles = [Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=colors[m],
                           markersize=10,
                           label=model_labels[m],
                           markeredgecolor='black',
                           markeredgewidth=1.5)
                    for m in model_names]

    # Create custom legend handles for difficulties
    diff_handles = [Line2D([0], [0], marker=markers[d], color='w',
                          markerfacecolor='gray',
                          markersize=8,
                          label=d.capitalize(),
                          markeredgecolor='black',
                          markeredgewidth=1.5)
                   for d in difficulties]

    # Add model legend to Panel A (upper left)
    legend1_a = ax_a.legend(handles=model_handles, loc='upper left',
                           title='Model', frameon=True, fontsize=10,
                           title_fontsize=10)

    # Add difficulty legend to Panel A (upper right)
    ax_a.legend(handles=diff_handles, loc='upper right',
               title='Difficulty', frameon=True, fontsize=10,
               title_fontsize=10)
    ax_a.add_artist(legend1_a)  # Re-add first legend

    # Add model legend to Panel C (lower left)
    legend1_c = ax_c.legend(handles=model_handles, loc='lower left',
                           title='Model', frameon=True, fontsize=10,
                           title_fontsize=10)

    # Add difficulty legend to Panel C (lower right)
    ax_c.legend(handles=diff_handles, loc='lower right',
               title='Difficulty', frameon=True, fontsize=10,
               title_fontsize=10)
    ax_c.add_artist(legend1_c)  # Re-add first legend

    # ========== OVERALL TITLE ==========
    plt.suptitle('Quality-Performance Trade-off Analysis',
                fontsize=16, fontweight='bold', y=0.995)

    # ========== SAVE OUTPUTS ==========
    plt.savefig('figures/figure7_quality_tradeoff.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/figure7_quality_tradeoff.png', dpi=300, bbox_inches='tight')

    print("\n✓ Figure 7 saved successfully!")
    print("  - figures/figure7_quality_tradeoff.pdf")
    print("  - figures/figure7_quality_tradeoff.png")

    plt.close()

if __name__ == '__main__':
    import os
    os.makedirs('figures', exist_ok=True)
    generate_figure7()
