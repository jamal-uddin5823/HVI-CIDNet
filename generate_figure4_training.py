"""
Figure 4: Training loss curves (epochs 1-40)
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import numpy as np

def generate_figure4():
    """Generate Figure 4: Training curves"""

    # Load extracted data
    with open('thesis_data_extracted.json', 'r') as f:
        data = json.load(f)

    training_data = data['training']

    # Setup figure with 2 panels
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3)

    # Color scheme (colorblind-friendly)
    colors = {
        'baseline': '#0173B2',      # Blue
        'face_loss3': '#029E73',    # Green
        'face_loss5': '#DE8F05'     # Orange
    }

    # ==============================================================
    # PANEL A: Total Loss Comparison
    # ==============================================================
    ax1 = fig.add_subplot(gs[0, 0])

    for model_name, color in colors.items():
        model_data = training_data[model_name]
        epochs = model_data['epochs']
        losses = model_data['losses']

        # VALIDATION CHECK
        assert len(epochs) > 0, f"{model_name}: No epoch data!"
        assert len(losses) == len(epochs), f"{model_name}: Epoch/loss mismatch!"

        print(f"{model_name}: {len(epochs)} epochs, loss range {min(losses):.4f}-{max(losses):.4f}")

        # Plot with different line styles
        if model_name == 'baseline':
            ax1.plot(epochs, losses, color=color, linewidth=2, label='Baseline', linestyle='-')
        elif model_name == 'face_loss3':
            ax1.plot(epochs, losses, color=color, linewidth=2, label='Face Loss (FR=0.3)', linestyle='-')
        else:  # face_loss5
            ax1.plot(epochs, losses, color=color, linewidth=2, label='Face Loss (FR=0.5)', linestyle='-')

    ax1.set_xlabel('Training Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total Loss', fontsize=12, fontweight='bold')
    ax1.set_title('(A) Training Loss Convergence (Epochs 1-40)', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', frameon=True, fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0, 41)

    # Add annotations
    ax1.annotate('Initial spike\n(face loss initialization)',
                xy=(1, 6.0), xytext=(5, 5.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=9, color='red')

    ax1.annotate('Convergence region',
                xy=(35, 0.32), xytext=(25, 0.8),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=9)

    # ==============================================================
    # PANEL B: Learning Rate Schedule
    # ==============================================================
    ax2 = fig.add_subplot(gs[0, 1])

    # Plot loss curves again (lighter)
    for model_name, color in colors.items():
        model_data = training_data[model_name]
        epochs = model_data['epochs']
        losses = model_data['losses']
        ax2.plot(epochs, losses, color=color, linewidth=1.5, alpha=0.4, label=f'{model_name} loss')

    # Overlay learning rate schedule (secondary axis)
    ax2_lr = ax2.twinx()

    # Use baseline LR schedule (same for all models)
    baseline_data = training_data['baseline']
    epochs = baseline_data['epochs']
    lrs = baseline_data['lr']

    ax2_lr.plot(epochs, lrs, color='red', linewidth=2.5, linestyle='--',
               label='Learning Rate', zorder=10)

    # Labels
    ax2.set_xlabel('Training Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Total Loss', fontsize=12, fontweight='bold', color='gray')
    ax2_lr.set_ylabel('Learning Rate', fontsize=12, fontweight='bold', color='red')
    ax2.set_title('(B) Loss vs. Learning Rate Schedule', fontsize=13, fontweight='bold')

    ax2.tick_params(axis='y', labelcolor='gray')
    ax2_lr.tick_params(axis='y', labelcolor='red')

    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(0, 41)

    # Legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_lr.get_legend_handles_labels()
    ax2_lr.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

    # Annotations
    ax2_lr.annotate('Warmup phase\n(epochs 1-3)',
                   xy=(2, 6.67e-5), xytext=(6, 8e-5),
                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                   fontsize=9, color='red')

    ax2_lr.annotate('Cosine annealing\n(epochs 4+)',
                   xy=(20, 9.5e-5), xytext=(25, 1.1e-4),
                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                   fontsize=9, color='red')

    plt.suptitle('Figure 4: Training Dynamics (Epochs 1-40)',
                fontsize=15, fontweight='bold', y=1.02)

    plt.savefig('figures/figure4_training_curves.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/figure4_training_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 4 saved")
    plt.close()

if __name__ == '__main__':
    import os
    os.makedirs('figures', exist_ok=True)
    generate_figure4()
