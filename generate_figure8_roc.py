"""
Figure 8: ROC curves by difficulty level (2x2 grid)
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import numpy as np

def generate_figure8():
    """Generate Figure 8: ROC curves"""

    # Load extracted data
    with open('thesis_data_extracted.json', 'r') as f:
        data = json.load(f)

    roc_data = data['roc']

    # Setup figure with 2x2 grid
    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Color scheme
    colors = {
        'baseline': '#0173B2',
        'face_loss3': '#029E73',
        'face_loss5': '#DE8F05'
    }

    difficulties = ['easy', 'medium', 'hard', 'mixed']
    difficulty_titles = {
        'easy': 'Easy (1% Light)',
        'medium': 'Medium (5% Light)',
        'hard': 'Hard (10% Light)',
        'mixed': 'Mixed (All Levels)'
    }

    # Plot each difficulty in a subplot
    for idx, diff in enumerate(difficulties):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])

        # Plot ROC curves for each model
        for model_name, color in colors.items():
            if model_name in roc_data and diff in roc_data[model_name]:
                model_roc = roc_data[model_name][diff]

                tpr = model_roc['tpr']
                fpr = model_roc['fpr']

                # VALIDATION CHECK
                print(f"{model_name}/{diff}: TPR/FPR points = {len(tpr)}")
                assert len(tpr) == len(fpr), f"{model_name}/{diff}: TPR/FPR mismatch!"

                # Plot ROC curve
                label = model_name.replace('_', ' ').title()
                if model_name == 'face_loss3':
                    label = 'Face Loss (FR=0.3)'
                elif model_name == 'face_loss5':
                    label = 'Face Loss (FR=0.5)'

                ax.plot(fpr, tpr, color=color, linewidth=2.5, label=label)

                # Calculate AUC
                from sklearn.metrics import auc
                auc_score = auc(fpr, tpr)
                print(f"  AUC = {auc_score:.4f}")
            else:
                print(f"WARNING: {model_name}/{diff} ROC data missing!")

        # Plot diagonal (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1.5, label='Random')

        # Mark special points (FAR=0.1%, FAR=1%)
        ax.axvline(x=0.001, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
        ax.axvline(x=0.01, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)

        # Labels
        ax.set_xlabel('False Accept Rate (FAR)', fontsize=11, fontweight='bold')
        ax.set_ylabel('True Accept Rate (TAR)', fontsize=11, fontweight='bold')
        ax.set_title(f'{difficulty_titles[diff]}', fontsize=12, fontweight='bold')
        ax.set_xscale('log')
        ax.set_xlim(1e-4, 1)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(loc='lower right', fontsize=9, frameon=True)

        # Add annotation for FAR thresholds
        ax.text(0.001, 0.05, 'FAR=0.1%', rotation=90, fontsize=8, color='gray', alpha=0.7)
        ax.text(0.01, 0.05, 'FAR=1%', rotation=90, fontsize=8, color='gray', alpha=0.7)

    plt.suptitle('Figure 8: ROC Curves Across Difficulty Levels',
                fontsize=16, fontweight='bold', y=0.995)

    plt.savefig('figures/figure8_roc_curves.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/figure8_roc_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 8 saved")
    plt.close()

if __name__ == '__main__':
    import os
    os.makedirs('figures', exist_ok=True)

    # Install sklearn if not available (for AUC calculation)
    try:
        from sklearn.metrics import auc
    except ImportError:
        print("Installing scikit-learn for AUC calculation...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'scikit-learn'])

    generate_figure8()
