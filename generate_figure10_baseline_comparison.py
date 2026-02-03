"""
Figure 10: Impact of enhancement - low-light baseline vs. enhanced
"""
import matplotlib.pyplot as plt
import json
import numpy as np

def generate_figure10():
    """Generate Figure 10: Before/after enhancement comparison"""

    # Load extracted data
    with open('thesis_data_extracted.json', 'r') as f:
        data = json.load(f)

    eval_data = data['evaluation']

    # Setup figure
    fig, ax = plt.subplots(figsize=(12, 8))

    difficulties = ['easy', 'medium', 'hard', 'mixed']
    difficulty_labels = ['Easy\n(1% light)', 'Medium\n(5% light)', 'Hard\n(10% light)', 'Mixed\n(All levels)']

    x = np.arange(len(difficulties))
    width = 0.28

    # Extract EER values
    eer_lowlight = [eval_data['baseline'][d]['low_light_eer'] for d in difficulties]
    eer_baseline = [eval_data['baseline'][d]['eer'] for d in difficulties]
    eer_face_loss5 = [eval_data['face_loss5'][d]['eer'] for d in difficulties]

    # VALIDATION CHECK
    print("Low-light EER:", eer_lowlight)
    print("Baseline (enhanced) EER:", eer_baseline)
    print("Face Loss 5 (enhanced) EER:", eer_face_loss5)

    # Plot bars
    bars1 = ax.bar(x - width, eer_lowlight, width, label='Low-light (no enhancement)',
                  color='#616161', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, eer_baseline, width, label='Enhanced (Baseline)',
                  color='#0173B2', edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + width, eer_face_loss5, width, label='Enhanced (Face Loss)',
                  color='#DE8F05', edgecolor='black', linewidth=0.5)

    # Add value labels
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)

    # Labels
    ax.set_xlabel('Difficulty Level', fontsize=13, fontweight='bold')
    ax.set_ylabel('Equal Error Rate (%)', fontsize=13, fontweight='bold')
    ax.set_title('Figure 10: Impact of Enhancement on Face Verification',
                fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(difficulty_labels, fontsize=11)
    ax.legend(loc='upper left', frameon=True, fontsize=11)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(eer_lowlight) * 1.15)

    # Add annotations for dramatic improvements
    # Medium difficulty: 48.95% → 0.25% (48.7% reduction)
    ax.annotate('Enhancement reduces\nEER by 48.7%\n(48.95→0.25%)',
               xy=(1 + width, eer_face_loss5[1]), xytext=(1.8, 35),
               arrowprops=dict(arrowstyle='->', color='green', lw=2.5),
               fontsize=11, color='green', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Add summary text box
    improvements = [(eer_lowlight[i] - eer_face_loss5[i]) / eer_lowlight[i] * 100
                   for i in range(len(difficulties))]

    textstr = f'''Enhancement Impact:
Easy:   {eer_lowlight[0]:.2f}% → {eer_face_loss5[0]:.2f}% ({improvements[0]:.1f}% improvement)
Medium: {eer_lowlight[1]:.2f}% → {eer_face_loss5[1]:.2f}% ({improvements[1]:.1f}% improvement)
Hard:   {eer_lowlight[2]:.2f}% → {eer_face_loss5[2]:.2f}% ({improvements[2]:.1f}% improvement)
Mixed:  {eer_lowlight[3]:.2f}% → {eer_face_loss5[3]:.2f}% ({improvements[3]:.1f}% improvement)

Average: {np.mean(improvements):.1f}% improvement
Face loss provides additional 0.4-0.65% EER reduction'''

    ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.savefig('figures/figure10_enhancement_impact.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/figure10_enhancement_impact.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 10 saved")
    plt.close()

if __name__ == '__main__':
    import os
    os.makedirs('figures', exist_ok=True)
    generate_figure10()
