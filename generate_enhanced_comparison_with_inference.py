"""
Generate enhanced image comparison by running inference with trained models
Loads models, enhances test images, and creates visual comparison
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
    from net.CIDNet import CIDNet  # Import your model architecture

    model = CIDNet().to(device)

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Handle different checkpoint formats
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
    # Load image
    img = Image.open(image_path).convert('RGB')

    # Transform to tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    img_tensor = transform(img).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        enhanced_tensor = model(img_tensor)

    # Convert back to PIL Image
    enhanced_tensor = enhanced_tensor.squeeze(0).cpu().clamp(0, 1)
    enhanced_np = (enhanced_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    enhanced_img = Image.fromarray(enhanced_np)

    return enhanced_img

def find_sample_image(dataset_dir, difficulty='easy'):
    """Find a sample image from the test set"""
    test_dir = f'{dataset_dir}/test_{difficulty}/low'

    if not os.path.exists(test_dir):
        print(f"ERROR: Test directory not found: {test_dir}")
        return None, None

    # Get first person directory
    persons = [d for d in os.listdir(test_dir)
               if os.path.isdir(os.path.join(test_dir, d))]

    if not persons:
        print(f"ERROR: No person directories found in {test_dir}")
        return None, None

    person = persons[0]

    # Get first image
    images = glob.glob(f'{test_dir}/{person}/*.png')
    if not images:
        print(f"ERROR: No images found in {test_dir}/{person}")
        return None, None

    return person, os.path.basename(images[0])

def generate_comparison_with_inference(
    checkpoint_paths,
    person_name=None,
    image_filename=None,
    device='cuda'
):
    """
    Generate visual comparison by running inference

    Args:
        checkpoint_paths: dict with keys 'baseline', 'face_loss3', 'face_loss5'
        person_name: Person to use (auto-detect if None)
        image_filename: Image filename (auto-detect if None)
        device: 'cuda' or 'cpu'
    """

    # Check device availability
    if device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        device = 'cpu'

    print(f"Using device: {device}")

    # Dataset directory
    dataset_dir = 'datasets/LFW_multilevel'
    difficulties = ['easy', 'medium', 'hard', 'mixed']

    # Auto-detect sample image if not provided
    if not person_name or not image_filename:
        print("Auto-detecting sample image...")
        person_name, image_filename = find_sample_image(dataset_dir, difficulty='easy')
        if not person_name:
            return

    print(f"Using sample: {person_name}/{image_filename}")

    # Extract base filename (without difficulty suffix)
    base_name = image_filename.replace('_easy.png', '').replace('_medium.png', '').replace('_hard.png', '').replace('_mixed.png', '')

    # Load models
    models = {}
    model_labels = {
        'baseline': 'Baseline',
        'face_loss3': 'Face Loss (FR=0.3)',
        'face_loss5': 'Face Loss (FR=0.5)'
    }

    for model_name, ckpt_path in checkpoint_paths.items():
        if ckpt_path and os.path.exists(ckpt_path):
            models[model_name] = load_model(ckpt_path, device)
        else:
            print(f"WARNING: Checkpoint for {model_name} not found: {ckpt_path}")
            models[model_name] = None

    # Check if at least one model loaded
    if not any(models.values()):
        print("ERROR: No models could be loaded!")
        return

    # Setup figure
    fig = plt.figure(figsize=(18, 16))
    gs = gridspec.GridSpec(4, 5, figure=fig, hspace=0.15, wspace=0.1)

    difficulty_labels = {
        'easy': 'Easy (1% Light)',
        'medium': 'Medium (5% Light)',
        'hard': 'Hard (10% Light)',
        'mixed': 'Mixed (All Levels)'
    }

    column_titles = ['Low-Light Input', 'Baseline Enhanced', 'Face Loss 0.3', 'Face Loss 0.5', 'Ground Truth']

    # Add column headers
    for col_idx, title in enumerate(column_titles):
        ax = fig.add_subplot(gs[0, col_idx])
        ax.text(0.5, 1.15, title, ha='center', va='bottom',
               fontsize=13, fontweight='bold', transform=ax.transAxes)

    # Process each difficulty level
    for row_idx, diff in enumerate(difficulties):
        print(f"\nProcessing {diff} difficulty...")

        # Determine filename for this difficulty
        if diff == 'easy':
            actual_filename = f'{base_name}_easy.png'
        elif diff == 'medium':
            actual_filename = f'{base_name}_medium.png'
        elif diff == 'hard':
            actual_filename = f'{base_name}_hard.png'
        else:  # mixed
            # Try to find mixed image
            mixed_dir = f'{dataset_dir}/test_mixed/low/{person_name}'
            if os.path.exists(mixed_dir):
                candidates = glob.glob(f'{mixed_dir}/{base_name}*.png')
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
            low_light_img = Image.open(low_light_path).convert('RGB')
            ax.imshow(low_light_img)
        else:
            ax.text(0.5, 0.5, 'Image not found', ha='center', va='center', fontsize=9)
            print(f"  WARNING: {low_light_path} not found")
            low_light_img = None

        ax.set_ylabel(difficulty_labels[diff], fontsize=12, fontweight='bold', rotation=90, labelpad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)

        # Columns 1-3: Enhanced by each model
        model_names = ['baseline', 'face_loss3', 'face_loss5']
        for model_idx, model_name in enumerate(model_names):
            ax = fig.add_subplot(gs[row_idx, model_idx + 1])

            if models[model_name] is not None and low_light_img is not None:
                try:
                    # Generate enhanced image
                    enhanced_img = enhance_image(models[model_name], low_light_path, device)
                    ax.imshow(enhanced_img)
                    print(f"  ✓ Enhanced with {model_name}")
                except Exception as e:
                    ax.text(0.5, 0.5, f'Enhancement\nfailed', ha='center', va='center', fontsize=9)
                    print(f"  ERROR enhancing with {model_name}: {str(e)}")
            else:
                ax.text(0.5, 0.5, 'Model not\navailable', ha='center', va='center', fontsize=9)

            ax.set_xticks([])
            ax.set_yticks([])

            # Highlight face_loss5 with thicker border
            if model_name == 'face_loss5':
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
            gt_img = Image.open(gt_path).convert('RGB')
            ax.imshow(gt_img)
        else:
            ax.text(0.5, 0.5, 'GT not found', ha='center', va='center', fontsize=9)
            print(f"  WARNING: {gt_path} not found")

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('green')
            spine.set_linewidth(2.0)

    # Add figure title
    plt.suptitle(f'Visual Comparison: Enhanced Images Across Models and Difficulty Levels\n({person_name})',
                fontsize=16, fontweight='bold', y=0.995)

    # Add legend
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
    output_name = 'figure_enhanced_comparison_inference'
    plt.savefig(f'figures/{output_name}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'figures/{output_name}.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ {output_name} saved to figures/")
    plt.close()

if __name__ == '__main__':
    # Define checkpoint paths
    checkpoint_paths = {
        'baseline': 'weights/multilevel/baseline/epoch_40.pth',
        'face_loss3': 'weights/multilevel/face_loss3/epoch_40.pth',
        'face_loss5': 'weights/multilevel/face_loss5/epoch_40.pth'
    }

    # Allow command-line override
    if len(sys.argv) >= 2:
        # User can specify checkpoint directory
        ckpt_dir = sys.argv[1]
        checkpoint_paths = {
            'baseline': f'{ckpt_dir}/multilevel/baseline/epoch_40.pth',
            'face_loss3': f'{ckpt_dir}/multilevel/face_loss3/epoch_40.pth',
            'face_loss5': f'{ckpt_dir}/multilevel/face_loss5/epoch_40.pth'
        }

    # Check if any checkpoints exist
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
        print(f"  python {sys.argv[0]} <checkpoint_directory>")
        print("\nExample:")
        print(f"  python {sys.argv[0]} weights")
        print("\nExpected structure:")
        print("  weights/")
        print("    └── multilevel/")
        print("        ├── baseline/epoch_40.pth")
        print("        ├── face_loss3/epoch_40.pth")
        print("        └── face_loss5/epoch_40.pth")
        sys.exit(1)

    # Generate comparison
    print("\n" + "="*70)
    print("Generating Enhanced Image Comparison with Model Inference")
    print("="*70)

    generate_comparison_with_inference(
        checkpoint_paths=checkpoint_paths,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print("\n" + "="*70)
    print("✓ Generation complete!")
    print("="*70)
