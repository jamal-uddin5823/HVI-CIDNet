"""
Figure 6: Main results - Verification performance across difficulty levels
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import numpy as np

def generate_figure6():
    """Generate Figure 6: Verification performance"""

    # Load extracted data
    with open('thesis_data_extracted.json', 'r') as f:
        data = json.load(f)

    eval_data = data['evaluation']

    # Setup figure with 2 panels (top: EER, bottom: TAR@FAR)
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.3)

    # Color scheme
    colors = {
        'baseline': '#0173B2',
        'face_loss3': '#029E73',
        'face_loss5': '#DE8F05'
    }

    difficulties = ['easy', 'medium', 'hard', 'mixed']
    difficulty_labels = ['Easy\n(1% light)', 'Medium\n(5% light)', 'Hard\n(10% light)', 'Mixed\n(All levels)']

    # ==============================================================
    # PANEL A: Equal Error Rate (EER)
    # ==============================================================
    ax1 = fig.add_subplot(gs[0, 0])

    x = np.arange(len(difficulties))
    width = 0.25

    # Extract EER values
    eer_baseline = [eval_data['baseline'][d]['eer'] for d in difficulties]
    eer_face_loss3 = [eval_data['face_loss3'][d]['eer'] for d in difficulties]
    eer_face_loss5 = [eval_data['face_loss5'][d]['eer'] for d in difficulties]

    # VALIDATION CHECK
    print("EER values:")
    print(f"  Baseline: {eer_baseline}")
    print(f"  Face Loss 3: {eer_face_loss3}")
    print(f"  Face Loss 5: {eer_face_loss5}")

    # Plot bars
    bars1 = ax1.bar(x - width, eer_baseline, width, label='Baseline',
                   color=colors['baseline'], edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar(x, eer_face_loss3, width, label='Face Loss (FR=0.3)',
                   color=colors['face_loss3'], edgecolor='black', linewidth=0.5)
    bars3 = ax1.bar(x + width, eer_face_loss5, width, label='Face Loss (FR=0.5)',
                   color=colors['face_loss5'], edgecolor='black', linewidth=0.5)

    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}%',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')

    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)

    ax1.set_xlabel('Difficulty Level', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Equal Error Rate (%)', fontsize=13, fontweight='bold')
    ax1.set_title('(A) Verification Performance: Equal Error Rate (Lower is Better)',
                 fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(difficulty_labels, fontsize=11)
    ax1.legend(loc='upper left', frameon=True, fontsize=11)
    ax1.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, max(eer_baseline + eer_face_loss3 + eer_face_loss5) * 1.3)

    # Add annotations for key improvements
    # Medium difficulty: 62% reduction (0.65 → 0.25)
    ax1.annotate('62% reduction\n(0.65→0.25%)',
                xy=(1 + width, eer_face_loss5[1]), xytext=(1.5, 0.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, color='red', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # ==============================================================
    # PANEL B: TAR @ FAR 0.1% (Strict Security)
    # ==============================================================
    ax2 = fig.add_subplot(gs[1, 0])

    # Extract TAR@FAR 0.1% values
    tar_baseline = [eval_data['baseline'][d]['tar_001'] for d in difficulties]
    tar_face_loss3 = [eval_data['face_loss3'][d]['tar_001'] for d in difficulties]
    tar_face_loss5 = [eval_data['face_loss5'][d]['tar_001'] for d in difficulties]

    print("TAR@FAR 0.1% values:")
    print(f"  Baseline: {tar_baseline}")
    print(f"  Face Loss 3: {tar_face_loss3}")
    print(f"  Face Loss 5: {tar_face_loss5}")

    # Plot bars
    bars1 = ax2.bar(x - width, tar_baseline, width, label='Baseline',
                   color=colors['baseline'], edgecolor='black', linewidth=0.5)
    bars2 = ax2.bar(x, tar_face_loss3, width, label='Face Loss (FR=0.3)',
                   color=colors['face_loss3'], edgecolor='black', linewidth=0.5)
    bars3 = ax2.bar(x + width, tar_face_loss5, width, label='Face Loss (FR=0.5)',
                   color=colors['face_loss5'], edgecolor='black', linewidth=0.5)

    # Add value labels
    def add_value_labels_tar(bars):
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')

    add_value_labels_tar(bars1)
    add_value_labels_tar(bars2)
    add_value_labels_tar(bars3)

    ax2.set_xlabel('Difficulty Level', fontsize=13, fontweight='bold')
    ax2.set_ylabel('True Accept Rate @ FAR=0.1% (%)', fontsize=13, fontweight='bold')
    ax2.set_title('(B) Strict Security Performance: TAR @ FAR=0.1% (Higher is Better)',
                 fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(difficulty_labels, fontsize=11)
    ax2.legend(loc='lower left', frameon=True, fontsize=11)
    ax2.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(90, 102)

    # Add annotations
    ax2.annotate('+3.2%\nimprovement',
                xy=(1 + width, tar_face_loss5[1]), xytext=(1.5, 94),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=10, color='green', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    plt.suptitle('Figure 6: Face Verification Performance Across Difficulty Levels',
                fontsize=16, fontweight='bold', y=0.995)

    plt.savefig('figures/figure6_verification_performance.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/figure6_verification_performance.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 6 saved")
    plt.close()

if __name__ == '__main__':
    import os
    os.makedirs('figures', exist_ok=True)
    generate_figure6()
