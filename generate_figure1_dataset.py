"""
Figure 1: Multi-level dataset generation illustration
Shows example images at each difficulty level
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import numpy as np
import os

def generate_figure1():
    """Generate Figure 1: Dataset methodology"""

    # Setup figure
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    # ==============================================================
    # ROW 1: Example Images
    # ==============================================================
    # Choose a person with images at all difficulties
    person = 'Ben_Affleck'  # Adjust based on available data
    image_num = '0001'

    base_dir = 'datasets/LFW_multilevel'

    # Load images
    images = []
    titles = ['Original (GT)', 'Easy (1% light)', 'Medium (5% light)', 'Hard (10% light)']
    paths = [
        f'{base_dir}/test_easy/high/{person}/{person}_{image_num}_easy_easy.png',  # Original (GT)
        f'{base_dir}/test_easy/low/{person}/{person}_{image_num}_easy.png',
        f'{base_dir}/test_medium/low/{person}/{person}_{image_num}_medium.png',
        f'{base_dir}/test_hard/low/{person}/{person}_{image_num}_hard.png'
    ]

    for i, (path, title) in enumerate(zip(paths, titles)):
        ax = fig.add_subplot(gs[0, i])

        if os.path.exists(path):
            img = Image.open(path)
            ax.imshow(img)
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, f'Image not found:\n{path}',
                   ha='center', va='center', fontsize=8)
            ax.axis('off')
            print(f"WARNING: {path} not found")

    # ==============================================================
    # ROW 2: Physics Pipeline Diagram (text-based)
    # ==============================================================
    ax_pipeline = fig.add_subplot(gs[1, :])
    ax_pipeline.axis('off')

    pipeline_text = """
    Physics-Based Synthesis Pipeline:

    sRGB Image → Linear RGB → Light Reduction → Noise Addition → White Balance → Linear/sRGB
    (Input)      (γ⁻¹)      (* reduction_factor)  (Poisson-Gaussian) (Per-channel gain)  (γ or RAW)
    """

    ax_pipeline.text(0.5, 0.5, pipeline_text, ha='center', va='center',
                    fontsize=9, family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # ==============================================================
    # ROW 3: Difficulty Parameters Table
    # ==============================================================
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis('off')

    # Table data
    table_data = [
        ['Level', 'Light %', 'Noise Type', 'WB Shift', 'Raw Sensor Mode', 'Gamma Correction'],
        ['Easy', '1%', 'None', 'No', 'False', 'Applied'],
        ['Medium', '5%', 'Poisson-Gaussian\n(σ_shot=1.0, σ_read=0.005)', 'No', 'True', 'Skipped'],
        ['Hard', '10%', 'High P-G Noise\n(σ_shot=2.0, σ_read=0.015)', 'Yes (±10%)', 'True', 'Skipped']
    ]

    table = ax_table.table(cellText=table_data, loc='center', cellLoc='center',
                          colWidths=[0.12, 0.1, 0.25, 0.15, 0.18, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2.5)

    # Style header row
    for i in range(6):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style data rows
    colors = ['#f0f0f0', '#d4f1d4', '#fff4d4', '#ffd4d4']
    for row_idx, color in enumerate(colors[1:], start=1):
        for col_idx in range(6):
            table[(row_idx, col_idx)].set_facecolor(color)

    plt.suptitle('Figure 1: Multi-Level Low-Light Dataset Generation',
                fontsize=14, fontweight='bold', y=0.98)

    plt.savefig('figures/figure1_dataset_methodology.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/figure1_dataset_methodology.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 1 saved")
    plt.close()

if __name__ == '__main__':
    os.makedirs('figures', exist_ok=True)
    generate_figure1()
