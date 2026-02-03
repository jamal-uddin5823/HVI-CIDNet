"""
Figure: Gallery of Enhanced Images - Multiple Samples
Shows 3-4 different people enhanced at medium difficulty level
to demonstrate consistent improvement across different faces
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import numpy as np
import os
import glob
import random

def find_multiple_samples(base_dir, num_samples=3, difficulty='medium'):
    """
    Find multiple sample images that exist across all models
    Returns: list of (person_name, image_filename) tuples
    """
    models = ['baseline', 'face_loss3', 'face_loss5']

    # Get all person directories from baseline
    baseline_path = f'{base_dir}/baseline/{difficulty}/enhanced'
    if not os.path.exists(baseline_path):
        print(f"WARNING: {baseline_path} not found")
        return []

    persons = [d for d in os.listdir(baseline_path)
               if os.path.isdir(os.path.join(baseline_path, d))]

    valid_samples = []

    # Find persons that exist in all models
    for person in persons:
        found_all = True
        for model in models:
            path = f'{base_dir}/{model}/{difficulty}/enhanced/{person}'
            if not os.path.exists(path):
                found_all = False
                break

        if found_all:
            # Get first image from this person
            images = glob.glob(f'{baseline_path}/{person}/*.png')
            if images:
                valid_samples.append((person, os.path.basename(images[0])))

        if len(valid_samples) >= num_samples:
            break

    return valid_samples[:num_samples]

def generate_enhanced_gallery(num_samples=3, difficulty='medium'):
    """
    Generate gallery showing multiple samples enhanced by different models
    """

    # Setup paths
    base_dir = 'results/multilevel_evaluations'
    dataset_dir = 'datasets/LFW_multilevel'

    # Find sample images
    samples = find_multiple_samples(base_dir, num_samples=num_samples, difficulty=difficulty)

    if not samples:
        print(f"ERROR: Could not find sample images for {difficulty} difficulty!")
        return

    print(f"Found {len(samples)} samples for {difficulty} difficulty:")
    for person, img in samples:
        print(f"  - {person}/{img}")

    # Setup figure: rows = num_samples, columns = 5 (low-light, 3 models, GT)
    fig = plt.figure(figsize=(18, 5 * len(samples)))
    gs = gridspec.GridSpec(len(samples), 5, figure=fig, hspace=0.2, wspace=0.1)

    column_titles = ['Low-Light Input', 'Baseline Enhanced', 'Face Loss (FR=0.3)', 'Face Loss (FR=0.5)', 'Ground Truth']

    # Add column headers
    for col_idx, title in enumerate(column_titles):
        ax = plt.subplot(gs[0, col_idx])
        ax.text(0.5, 1.3, title, ha='center', va='bottom',
               fontsize=14, fontweight='bold', transform=ax.transAxes)

    difficulty_label = {
        'easy': 'Easy (1% Light)',
        'medium': 'Medium (5% Light)',
        'hard': 'Hard (10% Light)',
        'mixed': 'Mixed'
    }[difficulty]

    # Process each sample
    for row_idx, (person, image_filename) in enumerate(samples):
        # Column 0: Low-light input
        ax = plt.subplot(gs[row_idx, 0])
        low_light_path = f'{dataset_dir}/test_{difficulty}/low/{person}/{image_filename}'

        if os.path.exists(low_light_path):
            img = Image.open(low_light_path)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, 'Not found', ha='center', va='center', fontsize=10)
            print(f"WARNING: {low_light_path} not found")

        # Add person name as ylabel
        person_display = person.replace('_', ' ')
        ax.set_ylabel(person_display, fontsize=11, fontweight='bold', rotation=90, labelpad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2.0)

        # Columns 1-3: Enhanced by each model
        models = ['baseline', 'face_loss3', 'face_loss5']
        for model_idx, model in enumerate(models):
            ax = plt.subplot(gs[row_idx, model_idx + 1])
            enhanced_path = f'{base_dir}/{model}/{difficulty}/enhanced/{person}/{image_filename}'

            if os.path.exists(enhanced_path):
                img = Image.open(enhanced_path)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, 'Not found', ha='center', va='center', fontsize=10)
                print(f"WARNING: {enhanced_path} not found")

            ax.set_xticks([])
            ax.set_yticks([])

            # Highlight face_loss5 with thicker border
            if model == 'face_loss5':
                for spine in ax.spines.values():
                    spine.set_edgecolor('#DE8F05')  # Orange
                    spine.set_linewidth(3.0)
            else:
                for spine in ax.spines.values():
                    spine.set_edgecolor('gray')
                    spine.set_linewidth(1.0)

        # Column 4: Ground truth
        ax = plt.subplot(gs[row_idx, 4])
        gt_path = f'{dataset_dir}/test_{difficulty}/high/{person}/{image_filename}'

        if os.path.exists(gt_path):
            img = Image.open(gt_path)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, 'Not found', ha='center', va='center', fontsize=10)
            print(f"WARNING: {gt_path} not found")

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('green')
            spine.set_linewidth(2.5)

    # Add figure title
    plt.suptitle(f'Enhanced Image Gallery - {difficulty_label} Difficulty Level',
                fontsize=17, fontweight='bold', y=0.998)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='white', edgecolor='black', linewidth=2.0, label='Low-Light Input'),
        Patch(facecolor='white', edgecolor='gray', linewidth=1.0, label='Baseline / Face Loss 0.3'),
        Patch(facecolor='white', edgecolor='#DE8F05', linewidth=3.0, label='Face Loss 0.5 (Best)'),
        Patch(facecolor='white', edgecolor='green', linewidth=2.5, label='Ground Truth')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
              bbox_to_anchor=(0.5, -0.015), fontsize=12, frameon=True)

    # Save figure
    os.makedirs('figures', exist_ok=True)
    output_name = f'figure_enhanced_gallery_{difficulty}'
    plt.savefig(f'figures/{output_name}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'figures/{output_name}.png', dpi=300, bbox_inches='tight')
    print(f"✓ {output_name} saved")
    plt.close()

if __name__ == '__main__':
    import sys

    # Allow specification of difficulty level
    if len(sys.argv) >= 2:
        difficulty = sys.argv[1]
        if difficulty not in ['easy', 'medium', 'hard', 'mixed']:
            print(f"Invalid difficulty: {difficulty}")
            print("Valid options: easy, medium, hard, mixed")
            sys.exit(1)
    else:
        difficulty = 'medium'  # Default to medium

    print(f"Generating enhanced image gallery for {difficulty} difficulty...")
    generate_enhanced_gallery(num_samples=3, difficulty=difficulty)
    print(f"\nTo generate for other difficulties, run:")
    print(f"  python generate_figure_enhanced_gallery.py <difficulty>")
    print(f"  Example: python generate_figure_enhanced_gallery.py hard")
