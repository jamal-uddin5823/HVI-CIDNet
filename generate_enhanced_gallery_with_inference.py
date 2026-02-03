"""
Generate enhanced image gallery by running inference with trained models
Shows multiple people enhanced at one difficulty level
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import numpy as np
import os
import glob
import torch
import torch.nn as nn
from torchvision import transforms
import sys

# Add parent directory to path to import model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def load_model(checkpoint_path, device='cuda'):
    """Load a trained model from checkpoint"""
    from model import CIDNet

    model = CIDNet().to(device)

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        print(f"✓ Model loaded successfully")
        return model
    else:
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return None

def enhance_image(model, image_path, device='cuda'):
    """Enhance a single image using the model"""
    img = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        enhanced_tensor = model(img_tensor)

    enhanced_tensor = enhanced_tensor.squeeze(0).cpu().clamp(0, 1)
    enhanced_np = (enhanced_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    enhanced_img = Image.fromarray(enhanced_np)

    return enhanced_img

def find_multiple_samples(dataset_dir, difficulty='medium', num_samples=3):
    """Find multiple sample images from the test set"""
    test_dir = f'{dataset_dir}/test_{difficulty}/low'

    if not os.path.exists(test_dir):
        print(f"ERROR: Test directory not found: {test_dir}")
        return []

    # Get person directories
    persons = [d for d in os.listdir(test_dir)
               if os.path.isdir(os.path.join(test_dir, d))]

    if not persons:
        print(f"ERROR: No person directories found in {test_dir}")
        return []

    samples = []
    for person in persons[:num_samples]:
        images = glob.glob(f'{test_dir}/{person}/*.png')
        if images:
            samples.append((person, os.path.basename(images[0])))

        if len(samples) >= num_samples:
            break

    return samples

def generate_gallery_with_inference(
    checkpoint_paths,
    difficulty='medium',
    num_samples=3,
    device='cuda'
):
    """Generate gallery by running inference"""

    # Check device
    if device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        device = 'cpu'

    print(f"Using device: {device}")

    # Dataset directory
    dataset_dir = 'datasets/LFW_multilevel'

    # Find sample images
    samples = find_multiple_samples(dataset_dir, difficulty, num_samples)

    if not samples:
        print(f"ERROR: Could not find sample images for {difficulty} difficulty!")
        return

    print(f"\nFound {len(samples)} samples for {difficulty} difficulty:")
    for person, img in samples:
        print(f"  - {person}/{img}")

    # Load models
    models = {}
    for model_name, ckpt_path in checkpoint_paths.items():
        if ckpt_path and os.path.exists(ckpt_path):
            models[model_name] = load_model(ckpt_path, device)
        else:
            print(f"WARNING: Checkpoint for {model_name} not found: {ckpt_path}")
            models[model_name] = None

    if not any(models.values()):
        print("ERROR: No models could be loaded!")
        return

    # Setup figure
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
        print(f"\nProcessing {person}/{image_filename}...")

        # Column 0: Low-light input
        ax = plt.subplot(gs[row_idx, 0])
        low_light_path = f'{dataset_dir}/test_{difficulty}/low/{person}/{image_filename}'

        if os.path.exists(low_light_path):
            low_light_img = Image.open(low_light_path).convert('RGB')
            ax.imshow(low_light_img)
        else:
            ax.text(0.5, 0.5, 'Not found', ha='center', va='center', fontsize=10)
            print(f"  WARNING: {low_light_path} not found")
            low_light_img = None

        person_display = person.replace('_', ' ')
        ax.set_ylabel(person_display, fontsize=11, fontweight='bold', rotation=90, labelpad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2.0)

        # Columns 1-3: Enhanced by each model
        model_names = ['baseline', 'face_loss3', 'face_loss5']
        for model_idx, model_name in enumerate(model_names):
            ax = plt.subplot(gs[row_idx, model_idx + 1])

            if models[model_name] is not None and low_light_img is not None:
                try:
                    enhanced_img = enhance_image(models[model_name], low_light_path, device)
                    ax.imshow(enhanced_img)
                    print(f"  ✓ Enhanced with {model_name}")
                except Exception as e:
                    ax.text(0.5, 0.5, 'Enhancement\nfailed', ha='center', va='center', fontsize=10)
                    print(f"  ERROR enhancing with {model_name}: {str(e)}")
            else:
                ax.text(0.5, 0.5, 'Model not\navailable', ha='center', va='center', fontsize=10)

            ax.set_xticks([])
            ax.set_yticks([])

            if model_name == 'face_loss5':
                for spine in ax.spines.values():
                    spine.set_edgecolor('#DE8F05')
                    spine.set_linewidth(3.0)
            else:
                for spine in ax.spines.values():
                    spine.set_edgecolor('gray')
                    spine.set_linewidth(1.0)

        # Column 4: Ground truth
        ax = plt.subplot(gs[row_idx, 4])
        gt_path = f'{dataset_dir}/test_{difficulty}/high/{person}/{image_filename}'

        if os.path.exists(gt_path):
            gt_img = Image.open(gt_path).convert('RGB')
            ax.imshow(gt_img)
        else:
            ax.text(0.5, 0.5, 'Not found', ha='center', va='center', fontsize=10)
            print(f"  WARNING: {gt_path} not found")

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('green')
            spine.set_linewidth(2.5)

    # Add title
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

    # Save
    os.makedirs('figures', exist_ok=True)
    output_name = f'figure_enhanced_gallery_{difficulty}_inference'
    plt.savefig(f'figures/{output_name}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'figures/{output_name}.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ {output_name} saved to figures/")
    plt.close()

if __name__ == '__main__':
    # Define checkpoint paths
    checkpoint_paths = {
        'baseline': 'checkpoints/baseline/best_model.pth',
        'face_loss3': 'checkpoints/face_loss3/best_model.pth',
        'face_loss5': 'checkpoints/face_loss5/best_model.pth'
    }

    difficulty = 'medium'  # Default

    # Parse arguments
    if len(sys.argv) >= 2:
        # First arg could be difficulty or checkpoint dir
        if sys.argv[1] in ['easy', 'medium', 'hard', 'mixed']:
            difficulty = sys.argv[1]
        else:
            # Assume it's checkpoint directory
            ckpt_dir = sys.argv[1]
            checkpoint_paths = {
                'baseline': f'{ckpt_dir}/baseline/best_model.pth',
                'face_loss3': f'{ckpt_dir}/face_loss3/best_model.pth',
                'face_loss5': f'{ckpt_dir}/face_loss5/best_model.pth'
            }

    if len(sys.argv) >= 3:
        # Second arg is difficulty
        if sys.argv[2] in ['easy', 'medium', 'hard', 'mixed']:
            difficulty = sys.argv[2]

    # Check checkpoints
    found_any = False
    for model_name, path in checkpoint_paths.items():
        if os.path.exists(path):
            print(f"Found checkpoint: {path}")
            found_any = True
        else:
            print(f"Checkpoint not found: {path}")

    if not found_any:
        print("\n" + "="*70)
        print("ERROR: No model checkpoints found!")
        print("="*70)
        print("\nPlease specify the checkpoint directory:")
        print(f"  python {sys.argv[0]} <checkpoint_directory> <difficulty>")
        print("\nExample:")
        print(f"  python {sys.argv[0]} checkpoints medium")
        print(f"  python {sys.argv[0]} medium  # (uses default checkpoint path)")
        sys.exit(1)

    # Generate gallery
    print("\n" + "="*70)
    print(f"Generating Enhanced Image Gallery - {difficulty.capitalize()} Difficulty")
    print("="*70)

    generate_gallery_with_inference(
        checkpoint_paths=checkpoint_paths,
        difficulty=difficulty,
        num_samples=3,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print("\n" + "="*70)
    print("✓ Generation complete!")
    print("="*70)
