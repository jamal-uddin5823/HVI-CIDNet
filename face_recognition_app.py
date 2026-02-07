"""
Gradio Face Recognition Demo Application

This application provides three demonstration modes:
1. Well-Lit Face Recognition - Test face recognition on well-lit images
2. Low-Light Face Recognition - Enhancement + recognition pipeline
3. Synthetic Low-Light Generation - Generate synthetic low-light images
"""

import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import os
from pathlib import Path
from typing import List, Tuple, Optional
import time
import argparse

# Import local modules
from recognizers import FaceRecognizerFactory
from face_database import FaceDatabase
from net.CIDNet import CIDNet
from data.lowlight_synthesis import synthesize_low_light_image


# Global variables for models and database
face_database = None
enhancement_models = {}
current_enhancement_model = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# Utility Functions
# ============================================================================

def find_enhancement_weights(base_dir='weights'):
    """Find all .pth files, excluding training checkpoints"""
    weights = []
    base_path = Path(base_dir)

    if not base_path.exists():
        return []

    for root, dirs, files in os.walk(base_path):
        # Skip training directories
        if 'train' in root.lower():
            continue

        for file in files:
            if file.endswith('.pth'):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, base_dir)
                weights.append(rel_path)

    return sorted(weights)


def group_enhancement_weights(weights: List[str]) -> dict:
    """Group enhancement weights by category"""
    grouped = {
        'Recommended': [],
        'Face Recognition (Thesis)': [],
        'LOLv2': [],
        'Specialized': []
    }

    for weight in weights:
        weight_lower = weight.lower()

        if 'generalization' in weight_lower or 'sice' in weight_lower:
            grouped['Recommended'].append(weight)
        elif 'face_loss' in weight_lower or 'multilevel' in weight_lower or 'baseline' in weight_lower:
            grouped['Face Recognition (Thesis)'].append(weight)
        elif 'lolv2' in weight_lower or 'lol_v2' in weight_lower:
            grouped['LOLv2'].append(weight)
        else:
            grouped['Specialized'].append(weight)

    return grouped


def load_enhancement_model(weight_path: str):
    """Load enhancement model from checkpoint"""
    global current_enhancement_model

    # Check cache
    if weight_path in enhancement_models:
        current_enhancement_model = enhancement_models[weight_path]
        return current_enhancement_model

    # Load model
    model = CIDNet()
    full_path = Path('weights') / weight_path

    if not full_path.exists():
        raise ValueError(f"Weight file not found: {full_path}")

    checkpoint = torch.load(full_path, map_location=device)

    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    # Cache model
    enhancement_models[weight_path] = model
    current_enhancement_model = model

    return model


def enhance_image(image: np.ndarray, model, gamma: float = 1.0) -> np.ndarray:
    """Enhance low-light image using CIDNet"""
    # Convert to tensor
    img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    img_tensor = img_tensor.to(device)

    # Enhance
    with torch.no_grad():
        enhanced_tensor = model(img_tensor)

    # Apply gamma adjustment if needed
    if gamma != 1.0:
        enhanced_tensor = torch.pow(enhanced_tensor, 1.0 / gamma)

    # Convert back to numpy
    enhanced = enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    enhanced = np.clip(enhanced, 0, 1)

    return enhanced


def format_gallery_results(matches: List[Tuple[str, float, str]],
                           query_image: Optional[np.ndarray] = None) -> List[Tuple[np.ndarray, str]]:
    """Format match results for Gradio gallery"""
    gallery_images = []

    # Add query image if provided
    if query_image is not None:
        gallery_images.append((query_image, "Query Image"))

    # Add matches
    for i, (person_id, score, img_path) in enumerate(matches):
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)

            # Format label
            label = f"#{i+1}: {person_id}\nScore: {score:.4f}"

            gallery_images.append((img_array, label))
        except Exception as e:
            print(f"Warning: Failed to load {img_path}: {e}")

    return gallery_images


# ============================================================================
# Tab 1: Well-Lit Face Recognition
# ============================================================================

def recognize_face_welllit(image: np.ndarray, model_name: str, top_k: int,
                          use_face_detection: bool) -> List[Tuple[np.ndarray, str]]:
    """Recognize face in well-lit image"""
    if image is None:
        return []

    try:
        # Convert to PIL
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image.astype(np.uint8))
        else:
            image_pil = image

        # Create recognizer
        recognizer = FaceRecognizerFactory.create(model_name, device=device)

        # Extract embedding
        embedding = recognizer.get_embedding(image_pil, use_face_detection=use_face_detection)

        # Match against database
        matches = face_database.match(embedding, top_k=top_k, threshold=0.0)

        # Format results
        gallery = format_gallery_results(matches, query_image=image)

        return gallery

    except Exception as e:
        print(f"Error in face recognition: {e}")
        import traceback
        traceback.print_exc()
        return [(np.zeros((100, 100, 3), dtype=np.uint8), f"Error: {str(e)}")]


# ============================================================================
# Tab 2: Low-Light Face Recognition (Enhancement + Recognition)
# ============================================================================

def recognize_face_lowlight(image: np.ndarray, enhancement_weight: str, gamma: float,
                           recognition_model: str, top_k: int) -> Tuple[np.ndarray, List[Tuple[np.ndarray, str]], str]:
    """Enhance low-light image and perform face recognition"""
    if image is None:
        return None, [], "No image provided"

    try:
        start_time = time.time()

        # Load enhancement model
        load_start = time.time()
        enhancement_model = load_enhancement_model(enhancement_weight)
        load_time = time.time() - load_start

        # Enhance image
        enhance_start = time.time()
        image_normalized = image.astype(np.float32) / 255.0
        enhanced = enhance_image(image_normalized, enhancement_model, gamma=gamma)
        enhanced_uint8 = (enhanced * 255).astype(np.uint8)
        enhance_time = time.time() - enhance_start

        # Create recognizer
        recog_start = time.time()
        recognizer = FaceRecognizerFactory.create(recognition_model, device=device)

        # Extract embedding from enhanced image
        enhanced_pil = Image.fromarray(enhanced_uint8)
        embedding = recognizer.get_embedding(enhanced_pil, use_face_detection=False)

        # Match against database
        matches = face_database.match(embedding, top_k=top_k, threshold=0.0)
        recog_time = time.time() - recog_start

        # Format results
        gallery = format_gallery_results(matches)

        total_time = time.time() - start_time

        # Create timing breakdown
        timing_info = (f"⏱️ Processing Time Breakdown:\n"
                      f"• Model Loading: {load_time:.3f}s\n"
                      f"• Enhancement: {enhance_time:.3f}s\n"
                      f"• Recognition: {recog_time:.3f}s\n"
                      f"• Total: {total_time:.3f}s")

        return enhanced_uint8, gallery, timing_info

    except Exception as e:
        print(f"Error in low-light recognition: {e}")
        import traceback
        traceback.print_exc()
        error_msg = f"❌ Error: {str(e)}"
        return None, [(np.zeros((100, 100, 3), dtype=np.uint8), error_msg)], error_msg


# ============================================================================
# Tab 3: Generate Synthetic Low-Light Images
# ============================================================================

# Difficulty parameters from generate_multilevel_training_sets.py
DIFFICULTY_PARAMS = {
    'Easy': {
        'reduction_factor': 0.01,  # 1% light
        'apply_noise': False,
        'apply_white_balance': False,
        'apply_blur': False,
        'raw_sensor_mode': False  # Gamma correction ON
    },
    'Medium': {
        'reduction_factor': 0.05,  # 5% light
        'apply_noise': True,
        'shot_noise_scale': 1.0,
        'read_noise_std': 0.005,
        'gain': 1.5,
        'apply_white_balance': False,
        'apply_blur': False,
        'raw_sensor_mode': True
    },
    'Hard': {
        'reduction_factor': 0.10,  # 10% light
        'apply_noise': True,
        'shot_noise_scale': 2.0,
        'read_noise_std': 0.015,
        'gain': 3.0,
        'apply_white_balance': True,
        'wb_variation': 0.1,
        'apply_blur': False,
        'raw_sensor_mode': True
    }
}


def generate_lowlight_images(image: np.ndarray, difficulty: str,
                            show_comparison: bool) -> List[Tuple[np.ndarray, str]]:
    """Generate synthetic low-light images at different difficulty levels"""
    if image is None:
        return []

    try:
        # Normalize image
        image_normalized = image.astype(np.float32) / 255.0

        results = []

        # Original image
        results.append((image, "Original (Well-Lit)"))

        # Generate for selected difficulty
        if difficulty == "All Levels":
            difficulties_to_generate = ['Easy', 'Medium', 'Hard']
        else:
            difficulties_to_generate = [difficulty]

        for diff in difficulties_to_generate:
            params = DIFFICULTY_PARAMS[diff]

            # Generate low-light image
            lowlight = synthesize_low_light_image(
                image_normalized,
                apply_light_reduction=True,
                **params,
                output_format='numpy'
            )

            # Convert to uint8
            lowlight_uint8 = (np.clip(lowlight, 0, 1) * 255).astype(np.uint8)

            # Create label
            label = f"{diff} Degradation\n"
            if diff == 'Easy':
                label += "1% light, no noise, gamma ON"
            elif diff == 'Medium':
                label += "5% light, Poisson-Gaussian noise"
            elif diff == 'Hard':
                label += "10% light, high noise, WB shift"

            results.append((lowlight_uint8, label))

        return results

    except Exception as e:
        print(f"Error generating low-light images: {e}")
        import traceback
        traceback.print_exc()
        return [(np.zeros((100, 100, 3), dtype=np.uint8), f"Error: {str(e)}")]


# ============================================================================
# Gradio Interface
# ============================================================================

def create_interface(db_path: str, recognizer_type: str = 'AdaFace'):
    """Create Gradio interface"""
    global face_database

    # Initialize face database
    print(f"Loading face database from: {db_path}")
    recognizer = FaceRecognizerFactory.create(recognizer_type, device=device)
    face_database = FaceDatabase(
        db_path=db_path,
        recognizer=recognizer,
        device=device,
        use_face_detection=False
    )
    print(f"Database loaded: {face_database}")

    # Find available enhancement weights
    available_weights = find_enhancement_weights('weights')
    if len(available_weights) == 0:
        print("Warning: No enhancement weights found in 'weights/' directory")
        available_weights = ['No weights found']

    # Set default weight
    default_weight = 'multilevel/face_loss5/epoch_40.pth'
    if default_weight not in available_weights and len(available_weights) > 0:
        default_weight = available_weights[0]

    # Create Gradio interface with three tabs
    with gr.Blocks(title="Face Recognition Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔍 Face Recognition Demo Application")
        gr.Markdown("*Low-Light Face Recognition Enhancement - Thesis Demonstration*")

        with gr.Tabs():
            # Tab 1: Well-Lit Face Recognition
            with gr.Tab("✨ Well-Lit Face Recognition"):
                gr.Markdown("### Test face recognition on well-lit images")

                with gr.Row():
                    with gr.Column(scale=1):
                        input_image_welllit = gr.Image(
                            label="Upload Face Image",
                            type="numpy",
                            height=300
                        )

                        model_selector = gr.Dropdown(
                            choices=["AdaFace", "InsightFace"],
                            value="AdaFace",
                            label="Recognition Model"
                        )

                        topk_slider = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1,
                            label="Top-K Matches"
                        )

                        face_detection_toggle = gr.Checkbox(
                            label="Enable Face Detection",
                            value=False,
                            info="Use for full scene images (disable for pre-cropped faces)"
                        )

                        recognize_btn = gr.Button("🔍 Recognize Face", variant="primary")

                    with gr.Column(scale=2):
                        output_gallery_welllit = gr.Gallery(
                            label="Top Matches",
                            columns=4,
                            rows=2,
                            height=350,
                            object_fit="contain"
                        )

                recognize_btn.click(
                    fn=recognize_face_welllit,
                    inputs=[input_image_welllit, model_selector, topk_slider, face_detection_toggle],
                    outputs=output_gallery_welllit
                )

            # Tab 2: Low-Light Face Recognition
            with gr.Tab("🌙 Low-Light Face Recognition"):
                gr.Markdown("### Enhancement + Recognition Pipeline for Low-Light Images")

                with gr.Row():
                    with gr.Column(scale=1):
                        input_image_lowlight = gr.Image(
                            label="Upload Low-Light Face Image",
                            type="numpy",
                            height=300
                        )

                        enhancement_selector = gr.Dropdown(
                            choices=available_weights,
                            value=default_weight,
                            label="Enhancement Model"
                        )

                        gamma_slider = gr.Slider(
                            minimum=1.0,
                            maximum=3.0,
                            value=1.0,
                            step=0.1,
                            label="Gamma Adjustment (Brightness Boost)"
                        )

                        recognition_model_selector = gr.Dropdown(
                            choices=["AdaFace", "InsightFace"],
                            value="AdaFace",
                            label="Recognition Model"
                        )

                        topk_slider_lowlight = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1,
                            label="Top-K Matches"
                        )

                        enhance_recognize_btn = gr.Button("🚀 Enhance & Recognize", variant="primary")

                    with gr.Column(scale=2):
                        enhanced_output = gr.Image(
                            label="Enhanced Image",
                            type="numpy",
                            height=250,
                            show_download_button=True
                        )

                        output_gallery_lowlight = gr.Gallery(
                            label="Top Matches",
                            columns=4,
                            rows=2,
                            height=300,
                            object_fit="contain"
                        )

                        timing_output = gr.Textbox(
                            label="Processing Time",
                            lines=5
                        )

                enhance_recognize_btn.click(
                    fn=recognize_face_lowlight,
                    inputs=[input_image_lowlight, enhancement_selector, gamma_slider,
                           recognition_model_selector, topk_slider_lowlight],
                    outputs=[enhanced_output, output_gallery_lowlight, timing_output]
                )

            # Tab 3: Generate Synthetic Low-Light Images
            with gr.Tab("🎨 Generate Synthetic Low-Light"):
                gr.Markdown("### Generate Synthetic Low-Light Images for Demonstration")

                with gr.Row():
                    with gr.Column(scale=1):
                        input_image_synthetic = gr.Image(
                            label="Upload Well-Lit Face Image",
                            type="numpy",
                            height=300
                        )

                        difficulty_selector = gr.Dropdown(
                            choices=["Easy", "Medium", "Hard", "All Levels"],
                            value="All Levels",
                            label="Difficulty Level"
                        )

                        comparison_toggle = gr.Checkbox(
                            label="Show Side-by-Side Comparison",
                            value=True,
                            info="Display original alongside degraded versions"
                        )

                        generate_btn = gr.Button("🎨 Generate Low-Light Images", variant="primary")

                        gr.Markdown("""
                        **Difficulty Levels:**
                        - **Easy**: 1% light, no noise, gamma correction ON
                        - **Medium**: 5% light, Poisson-Gaussian noise
                        - **Hard**: 10% light, high noise, white balance shift
                        """)

                    with gr.Column(scale=2):
                        output_gallery_synthetic = gr.Gallery(
                            label="Generated Low-Light Images",
                            columns=3,
                            rows=2,
                            height=400,
                            object_fit="contain"
                        )

                generate_btn.click(
                    fn=generate_lowlight_images,
                    inputs=[input_image_synthetic, difficulty_selector, comparison_toggle],
                    outputs=output_gallery_synthetic
                )

        # Footer
        gr.Markdown("""
        ---
        **System Information:**
        - Device: {}
        - Database: {} people, {} total embeddings
        - Enhancement Weights: {} available
        """.format(
            device.upper(),
            len(face_database.person_ids),
            sum(len(v) for v in face_database.embeddings.values()),
            len(available_weights)
        ))

    return demo


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Face Recognition Demo Application")
    parser.add_argument('--db_path', type=str, default='face_database',
                       help='Path to face database directory')
    parser.add_argument('--recognizer', type=str, default='AdaFace',
                       choices=['AdaFace', 'InsightFace'],
                       help='Default recognizer to use for database')
    parser.add_argument('--port', type=int, default=7860,
                       help='Port to run Gradio server on')
    parser.add_argument('--share', action='store_true',
                       help='Create public share link')

    args = parser.parse_args()

    # Create interface
    demo = create_interface(
        db_path=args.db_path,
        recognizer_type=args.recognizer
    )

    # Launch
    print(f"\n{'='*70}")
    print(f"🚀 Launching Face Recognition Demo Application")
    print(f"{'='*70}")
    print(f"Database: {args.db_path}")
    print(f"Recognizer: {args.recognizer}")
    print(f"Device: {device.upper()}")
    print(f"Port: {args.port}")
    print(f"{'='*70}\n")

    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share
    )


if __name__ == "__main__":
    main()
