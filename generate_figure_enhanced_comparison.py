"""
Figure: Visual Comparison of Enhanced Images Across Models and Difficulty Levels
Shows how each model (baseline, face_loss3, face_loss5) enhances the same image
at different difficulty levels
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import numpy as np
import os
import glob

def find_sample_image(base_dir, person_name=None):
    """
    Find a sample image that exists across all models and difficulties
    Returns: person name and image filename
    """
    difficulties = ['easy', 'medium', 'hard', 'mixed']
    models = ['baseline', 'face_loss3', 'face_loss5']

    # If person name provided, try that first
    if person_name:
        test_path = f'{base_dir}/baseline/easy/enhanced/{person_name}'
        if os.path.exists(test_path):
            images = glob.glob(f'{test_path}/*.png')
            if images:
                return person_name, os.path.basename(images[0])

    # Otherwise, search for a common person across all models and difficulties
    baseline_easy_path = f'{base_dir}/baseline/easy/enhanced'
    if not os.path.exists(baseline_easy_path):
        print(f"WARNING: {baseline_easy_path} not found")
        return None, None

    # Get all person directories
    persons = [d for d in os.listdir(baseline_easy_path)
               if os.path.isdir(os.path.join(baseline_easy_path, d))]

    # Find a person that exists in all models and difficulties
    for person in persons:
        found_all = True
        for model in models:
            for diff in difficulties:
                path = f'{base_dir}/{model}/{diff}/enhanced/{person}'
                if not os.path.exists(path):
                    found_all = False
                    break
            if not found_all:
                break

        if found_all:
            # Get first image from this person
            images = glob.glob(f'{baseline_easy_path}/{person}/*.png')
            if images:
                return person, os.path.basename(images[0])

    return None, None

def generate_enhanced_comparison(figure_name='figure_enhanced_comparison',
                                 person_name=None,
                                 image_filename=None):
    """Generate visual comparison of enhanced images"""

    # Setup paths
    base_dir = 'results/multilevel_evaluations'
    dataset_dir = 'datasets/LFW_multilevel'

    # Find a suitable sample image
    if not person_name or not image_filename:
        person_name, image_filename = find_sample_image(base_dir)
        if not person_name:
            print("ERROR: Could not find a suitable sample image!")
            print("Please specify person_name and image_filename manually")
            return

    print(f"Using sample: {person_name}/{image_filename}")

    # Extract base filename (without difficulty suffix)
    # e.g., "George_W_Bush_0001_easy.png" -> "George_W_Bush_0001"
    base_name = image_filename.replace('_easy.png', '').replace('_medium.png', '').replace('_hard.png', '').replace('_mixed.png', '')

    # Setup figure with 4 rows (difficulties) × 5 columns (low-light, 3 models, GT)
    fig = plt.figure(figsize=(18, 16))
    gs = gridspec.GridSpec(4, 5, figure=fig, hspace=0.15, wspace=0.1)

    difficulties = ['easy', 'medium', 'hard', 'mixed']
    difficulty_labels = {
        'easy': 'Easy (1% Light)',
        'medium': 'Medium (5% Light)',
        'hard': 'Hard (10% Light)',
        'mixed': 'Mixed (All Levels)'
    }

    models = ['baseline', 'face_loss3', 'face_loss5']
    model_labels = {
        'baseline': 'Baseline',
        'face_loss3': 'Face Loss (FR=0.3)',
        'face_loss5': 'Face Loss (FR=0.5)'
    }

    column_titles = ['Low-Light Input', 'Baseline Enhanced', 'Face Loss 0.3', 'Face Loss 0.5', 'Ground Truth']

    # Add column headers (only once at the top)
    for col_idx, title in enumerate(column_titles):
        ax = fig.add_subplot(gs[0, col_idx])
        ax.text(0.5, 1.15, title, ha='center', va='bottom',
               fontsize=13, fontweight='bold', transform=ax.transAxes)

    # Process each difficulty level
    for row_idx, diff in enumerate(difficulties):
        # Determine the actual filename for this difficulty
        if diff == 'easy':
            actual_filename = f'{base_name}_easy.png'
        elif diff == 'medium':
            actual_filename = f'{base_name}_medium.png'
        elif diff == 'hard':
            actual_filename = f'{base_name}_hard.png'
        else:  # mixed
            # Mixed might use any suffix, try to find it
            mixed_path = f'{base_dir}/baseline/mixed/enhanced/{person_name}'
            if os.path.exists(mixed_path):
                candidates = glob.glob(f'{mixed_path}/{base_name}*.png')
                if candidates:
                    actual_filename = os.path.basename(candidates[0])
                else:
                    actual_filename = f'{base_name}_mixed.png'
            else:
                actual_filename = f'{base_name}_mixed.png'

        # Column 0: Low-light input
        ax = fig.add_subplot(gs[row_idx, 0])
        low_light_path = f'{dataset_dir}/test_{diff}/low/{person_name}/{actual_filename}'

        if os.path.exists(low_light_path):
            img = Image.open(low_light_path)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, f'Not found:\n{low_light_path}',
                   ha='center', va='center', fontsize=8, wrap=True)
            print(f"WARNING: {low_light_path} not found")

        ax.set_ylabel(difficulty_labels[diff], fontsize=12, fontweight='bold', rotation=90, labelpad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)

        # Columns 1-3: Enhanced by each model
        for model_idx, model in enumerate(models):
            ax = fig.add_subplot(gs[row_idx, model_idx + 1])
            enhanced_path = f'{base_dir}/{model}/{diff}/enhanced/{person_name}/{actual_filename}'

            if os.path.exists(enhanced_path):
                img = Image.open(enhanced_path)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, f'Not found:\n{enhanced_path}',
                       ha='center', va='center', fontsize=8, wrap=True)
                print(f"WARNING: {enhanced_path} not found")

            ax.set_xticks([])
            ax.set_yticks([])

            # Highlight face_loss5 with thicker border
            if model == 'face_loss5':
                for spine in ax.spines.values():
                    spine.set_edgecolor('#DE8F05')  # Orange
                    spine.set_linewidth(2.5)
            else:
                for spine in ax.spines.values():
                    spine.set_edgecolor('gray')
                    spine.set_linewidth(1.0)

        # Column 4: Ground truth
        ax = fig.add_subplot(gs[row_idx, 4])
        gt_path = f'{dataset_dir}/test_{diff}/high/{person_name}/{actual_filename}'

        if os.path.exists(gt_path):
            img = Image.open(gt_path)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, f'Not found:\n{gt_path}',
                   ha='center', va='center', fontsize=8, wrap=True)
            print(f"WARNING: {gt_path} not found")

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('green')
            spine.set_linewidth(2.0)

    # Add figure title
    plt.suptitle(f'Visual Comparison: Enhanced Images Across Models and Difficulty Levels\n({person_name})',
                fontsize=16, fontweight='bold', y=0.995)

    # Add legend for border colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='white', edgecolor='black', linewidth=1.5, label='Low-Light Input'),
        Patch(facecolor='white', edgecolor='gray', linewidth=1.0, label='Baseline / Face Loss 0.3'),
        Patch(facecolor='white', edgecolor='#DE8F05', linewidth=2.5, label='Face Loss 0.5 (Best)'),
        Patch(facecolor='white', edgecolor='green', linewidth=2.0, label='Ground Truth')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
              bbox_to_anchor=(0.5, -0.02), fontsize=11, frameon=True)

    # Save figure
    os.makedirs('figures', exist_ok=True)
    plt.savefig(f'figures/{figure_name}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'figures/{figure_name}.png', dpi=300, bbox_inches='tight')
    print(f"✓ {figure_name} saved")
    plt.close()

if __name__ == '__main__':
    import sys

    # Allow manual specification of person and image
    if len(sys.argv) >= 3:
        person = sys.argv[1]
        image = sys.argv[2]
        print(f"Using manually specified image: {person}/{image}")
        generate_enhanced_comparison(person_name=person, image_filename=image)
    else:
        print("Auto-detecting suitable sample image...")
        generate_enhanced_comparison()
        print("\nTo use a specific image, run:")
        print("  python generate_figure_enhanced_comparison.py <person_name> <image_filename>")
        print("  Example: python generate_figure_enhanced_comparison.py George_W_Bush George_W_Bush_0001_easy.png")
