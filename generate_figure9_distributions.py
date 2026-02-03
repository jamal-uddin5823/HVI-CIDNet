"""
Figure 9: Genuine vs. impostor score distributions (2x2 grid)
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import numpy as np

def generate_figure9():
    """Generate Figure 9: Score distributions"""

    # Load extracted data
    with open('thesis_data_extracted.json', 'r') as f:
        data = json.load(f)

    roc_data = data['roc']
    eval_data = data['evaluation']

    # Setup figure with 2x2 grid
    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    difficulties = ['easy', 'medium', 'hard', 'mixed']
    difficulty_titles = {
        'easy': 'Easy (1% Light)',
        'medium': 'Medium (5% Light)',
        'hard': 'Hard (10% Light)',
        'mixed': 'Mixed (All Levels)'
    }

    # Compare baseline vs. face_loss5 (most improvement)
    models_to_compare = ['baseline', 'face_loss5']

    # Plot each difficulty
    for idx, diff in enumerate(difficulties):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])

        # Overlay distributions for baseline and face_loss5
        for model_name in models_to_compare:
            if model_name in roc_data and diff in roc_data[model_name]:
                model_roc = roc_data[model_name][diff]

                genuine_scores = model_roc['genuine_scores_enhanced']
                impostor_scores = model_roc['impostor_scores_enhanced']

                # VALIDATION CHECK
                print(f"{model_name}/{diff}: Genuine={len(genuine_scores)}, Impostor={len(impostor_scores)}")

                # Color scheme
                if model_name == 'baseline':
                    gen_color = '#90CAF9'   # Light blue
                    imp_color = '#FFAB91'   # Light red
                    alpha = 0.5
                    label_suffix = ' (Baseline)'
                else:  # face_loss5
                    gen_color = '#81C784'   # Light green
                    imp_color = '#E57373'   # Darker red
                    alpha = 0.6
                    label_suffix = ' (Face Loss)'

                # Plot histograms
                ax.hist(genuine_scores, bins=50, alpha=alpha, color=gen_color,
                       edgecolor='black', linewidth=0.5, density=True,
                       label=f'Genuine{label_suffix}')
                ax.hist(impostor_scores, bins=50, alpha=alpha, color=imp_color,
                       edgecolor='black', linewidth=0.5, density=True,
                       label=f'Impostor{label_suffix}')

        # Add EER threshold line
        eer_thresh_baseline = eval_data['baseline'][diff].get('eer_threshold', 0.5)
        eer_thresh_face = eval_data['face_loss5'][diff].get('eer_threshold', 0.5)

        # Note: EER thresholds not in evaluation_data.json, use mean as proxy
        # If available from verification_scores.json metrics, use those

        # Labels
        ax.set_xlabel('Cosine Similarity Score', fontsize=11, fontweight='bold')
        ax.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
        ax.set_title(f'{difficulty_titles[diff]}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9, frameon=True)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xlim(0, 1)

        # Add statistics text box
        if 'face_loss5' in roc_data and diff in roc_data['face_loss5']:
            genuine_mean = np.mean(roc_data['face_loss5'][diff]['genuine_scores_enhanced'])
            genuine_std = np.std(roc_data['face_loss5'][diff]['genuine_scores_enhanced'])
            impostor_mean = np.mean(roc_data['face_loss5'][diff]['impostor_scores_enhanced'])
            impostor_std = np.std(roc_data['face_loss5'][diff]['impostor_scores_enhanced'])

            textstr = f'''Face Loss Stats:
Genuine: {genuine_mean:.3f} ± {genuine_std:.3f}
Impostor: {impostor_mean:.3f} ± {impostor_std:.3f}
Separation: {genuine_mean - impostor_mean:.3f}'''

            ax.text(0.98, 0.97, textstr, transform=ax.transAxes,
                   fontsize=8, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.suptitle('Figure 9: Genuine vs. Impostor Score Distributions',
                fontsize=16, fontweight='bold', y=0.995)

    plt.savefig('figures/figure9_score_distributions.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/figure9_score_distributions.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 9 saved")
    plt.close()

if __name__ == '__main__':
    import os
    os.makedirs('figures', exist_ok=True)
    generate_figure9()
