"""
Gradio Face Recognition Demo App

A web interface for low-light face enhancement and face recognition demonstration.
This app combines CIDNet enhancement with face recognition matching.

Features:
    - Low-light image enhancement using CIDNet
    - Face recognition using AdaFace or InsightFace
    - Face database matching with top-K results
    - Multiple enhancement model weights support
    - Adjustable enhancement parameters (gamma, alpha_s, alpha_i)

Usage:
    python face_recognition_app.py --port 7863 --device cuda

Requirements:
    - face_database.py: Face database management
    - recognizers.py: Face recognizer wrappers
    - net.CIDNet: Enhancement model
"""

import argparse
import os
import platform
from pathlib import Path

import gradio as gr
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from net.CIDNet import CIDNet
from face_database import FaceDatabase
from recognizers import AdaFaceRecognizer, InsightFaceRecognizer, get_recognizer


# Global models (loaded at startup)
enhancer = None
recognizer = None
face_db = None
device = 'cuda'

# Available weights
available_weights = []

# Face database path
DEFAULT_DB_PATH = './face_database'


def find_pth_files(directory):
    """Find all .pth files in directory excluding train subdirectories"""
    pth_files = []
    for root, dirs, files in os.walk(directory):
        if 'train' in root.split(os.sep):
            continue
        for file in files:
            if file.endswith('.pth'):
                pth_files.append(os.path.join(root, file))
    return pth_files


def remove_weights_prefix(paths, base_dir="weights"):
    """Remove the weights directory prefix from paths for display"""
    os_name = platform.system()
    if os_name.lower() == 'windows':
        sep = '\\'
    else:
        sep = '/'

    prefix = base_dir + sep
    cleaned_paths = [path.replace(prefix, '') if path.startswith(prefix) else path
                     for path in paths]
    return cleaned_paths


def load_enhancer_model(model_path):
    """Load CIDNet enhancer model"""
    global enhancer

    full_path = os.path.join('weights', model_path) if not os.path.isabs(model_path) else model_path

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Model weights not found: {full_path}")

    enhancer = CIDNet().to(device)
    enhancer.trans.gated = True
    enhancer.trans.gated2 = True

    state_dict = torch.load(full_path, map_location=device)
    enhancer.load_state_dict(state_dict)
    enhancer.eval()

    return enhancer


def load_recognizer_model(recognizer_type, face_weights_path=None):
    """Load face recognizer model"""
    global recognizer

    recognizer_type = recognizer_type.lower()

    if recognizer_type == 'adaface':
        recognizer = AdaFaceRecognizer(
            arch='ir_50',
            weights_path=face_weights_path,
            device=device
        )
    elif recognizer_type == 'insightface':
        recognizer = InsightFaceRecognizer(device=device)
    else:
        raise ValueError(f"Unknown recognizer type: {recognizer_type}")

    return recognizer


def load_face_database(db_path):
    """Load or create face database"""
    global face_db, recognizer

    if recognizer is None:
        raise RuntimeError("Recognizer not loaded. Please select a recognizer first.")

    # Create database if it doesn't exist
    if not os.path.exists(db_path):
        print(f"Database path does not exist: {db_path}")
        print("Creating empty database. Please add face images to continue.")
        face_db = FaceDatabase(db_path, recognizer, device=device)
    else:
        face_db = FaceDatabase(db_path, recognizer, device=device)

    return face_db


def preprocess_image(input_img):
    """Convert PIL image to tensor and pad to multiple of 8"""
    pil2tensor = transforms.Compose([transforms.ToTensor()])
    input_tensor = pil2tensor(input_img)

    factor = 8
    h, w = input_tensor.shape[1], input_tensor.shape[2]
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    input_tensor = F.pad(input_tensor.unsqueeze(0), (0, padw, 0, padh), 'reflect')

    return input_tensor


def process_query(
    query_image,
    enhancer_weights,
    recognizer_type,
    top_k,
    gamma,
    alpha_s,
    alpha_i,
    face_db_path,
    face_weights_path
):
    """Process query image through enhancement and recognition pipeline

    Args:
        query_image: Input low-light image (PIL Image)
        enhancer_weights: Path to enhancer weights
        recognizer_type: Type of face recognizer
        top_k: Number of top matches to return
        gamma: Gamma correction parameter
        alpha_s: Saturation alpha parameter
        alpha_i: Illumination alpha parameter
        face_db_path: Path to face database
        face_weights_path: Path to face recognizer weights

    Returns:
        tuple: (enhanced_image, matches_info)
    """
    global enhancer, recognizer, face_db, device

    torch.set_grad_enabled(False)

    if query_image is None:
        return None, "No image provided"

    try:
        # Step 1: Load/reload models if needed
        if enhancer is None or enhancer_weights != getattr(process_query, '_last_weights', None):
            enhancer = load_enhancer_model(enhancer_weights)
            process_query._last_weights = enhancer_weights

        if recognizer is None or recognizer_type != getattr(process_query, '_last_recognizer', None):
            load_recognizer_model(recognizer_type, face_weights_path)
            process_query._last_recognizer = recognizer_type

        # Step 2: Preprocess input
        input_tensor = preprocess_image(query_image)

        # Step 3: Enhance low-light image
        with torch.no_grad():
            enhancer.trans.alpha_s = alpha_s
            enhancer.trans.alpha = alpha_i

            input_device = input_tensor.to(device)
            enhanced = enhancer(input_device ** gamma)
            enhanced = torch.clamp(enhanced, 0, 1)

        # Crop back to original size
        h, w = input_tensor.shape[1], input_tensor.shape[2]
        enhanced = enhanced[:, :, :h, :w]

        # Convert to PIL
        enhanced_img = transforms.ToPILImage()(enhanced.squeeze(0))

        # Step 4: Load/reload face database if needed
        if face_db is None or face_db_path != getattr(process_query, '_last_db_path', None):
            face_db = load_face_database(face_db_path)
            process_query._last_db_path = face_db_path

        # Step 5: Extract face embedding
        with torch.no_grad():
            face_embedding = recognizer.get_embedding(enhanced)

        # Step 6: Match against database
        results = face_db.match(face_embedding.squeeze(0), top_k=top_k)

        # Step 7: Format results
        if not results:
            matches_info = "No matches found. Database may be empty or no similar faces."
        else:
            matches_info = f"<h3>Top {len(results)} Matches</h3><table style='width:100%'>"
            matches_info += "<tr><th>Rank</th><th>Person ID</th><th>Confidence</th></tr>"

            for rank, (person_id, confidence, img_path) in enumerate(results, 1):
                conf_pct = confidence * 100
                color = "green" if conf_pct > 70 else "orange" if conf_pct > 50 else "red"
                matches_info += f"<tr><td>{rank}</td><td>{person_id}</td>"
                matches_info += f"<td><span style='color:{color}'>{conf_pct:.2f}%</span></td></tr>"

            matches_info += "</table>"

            # Add reference image path
            if results[0][2]:
                matches_info += f"<p><small>Reference: {results[0][2]}</small></p>"

        return enhanced_img, matches_info

    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n\n{traceback.format_exc()}"
        return None, error_msg


def create_interface():
    """Create Gradio interface"""
    global available_weights

    # Find available weights
    weights_dir = 'weights'
    if os.path.exists(weights_dir):
        all_weights = find_pth_files(weights_dir)
        available_weights = remove_weights_prefix(all_weights, weights_dir)

    if not available_weights:
        available_weights = ["SICE.pth", "generalization.pth"]
        print("Warning: No weights found in weights/ directory")

    # Default selections
    default_weights = "SICE.pth" if "SICE.pth" in available_weights else available_weights[0] if available_weights else ""

    with gr.Blocks(title="Face Recognition Demo") as interface:
        gr.Markdown("# Low-Light Face Recognition Demo")
        gr.Markdown("Upload a low-light face image to enhance and match against the face database.")

        with gr.Row():
            with gr.Column(scale=1):
                # Input image
                query_image = gr.Image(
                    label="Low-Light Query Image",
                    type="pil",
                    height=300
                )

                # Configuration
                gr.Markdown("### Configuration")

                enhancer_weights = gr.Dropdown(
                    choices=available_weights,
                    value=default_weights,
                    label="Enhancer Weights",
                    info="Select CIDNet model weights"
                )

                recognizer_type = gr.Radio(
                    choices=["AdaFace", "InsightFace"],
                    value="AdaFace",
                    label="Face Recognizer",
                    info="Select face recognition model"
                )

                top_k = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Top-K Matches",
                    info="Number of matches to display"
                )

                # Enhancement parameters
                gr.Markdown("### Enhancement Parameters")

                gamma = gr.Slider(
                    minimum=0.1,
                    maximum=5.0,
                    value=1.0,
                    step=0.01,
                    label="Gamma",
                    info="Lower is lighter. Best range: [0.5, 2.5]"
                )

                alpha_s = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=1.0,
                    step=0.01,
                    label="Alpha-s (Saturation)",
                    info="Higher is more saturated"
                )

                alpha_i = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=1.0,
                    step=0.01,
                    label="Alpha-i (Illumination)",
                    info="Higher is lighter"
                )

                # Advanced settings
                with gr.Accordion("Advanced Settings", open=False):
                    face_db_path = gr.Textbox(
                        value=DEFAULT_DB_PATH,
                        label="Face Database Path",
                        info="Path to face database directory"
                    )

                    face_weights_path = gr.Textbox(
                        value="",
                        label="Face Recognizer Weights (Optional)",
                        info="Path to face recognizer weights file"
                    )

                # Process button
                process_btn = gr.Button("Process", variant="primary", size="lg")

            with gr.Column(scale=1):
                # Outputs
                enhanced_image = gr.Image(
                    label="Enhanced Image",
                    type="pil",
                    height=300
                )

                matches_output = gr.HTML(
                    label="Matching Results",
                    value="<p>Upload an image and click Process to see results.</p>"
                )

        # Event handlers
        process_btn.click(
            fn=process_query,
            inputs=[
                query_image,
                enhancer_weights,
                recognizer_type,
                top_k,
                gamma,
                alpha_s,
                alpha_i,
                face_db_path,
                face_weights_path
            ],
            outputs=[enhanced_image, matches_output]
        )

        # Examples
        gr.Markdown("### Examples")
        gr.Markdown("<small>Tip: The face database should be organized as `face_database/person_name/*.jpg`</small>")

    return interface


def main():
    parser = argparse.ArgumentParser(description='Face Recognition Demo App')
    parser.add_argument('--port', type=int, default=7863, help='Server port')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--share', action='store_true', help='Create public link')
    parser.add_argument('--db_path', type=str, default=DEFAULT_DB_PATH,
                       help='Path to face database')

    args = parser.parse_args()

    global device

    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        device = 'cpu'
    else:
        device = args.device

    print(f"Using device: {device}")

    # Create interface
    interface = create_interface()

    # Launch
    print(f"\nStarting Face Recognition Demo on port {args.port}")
    print(f"Face database path: {args.db_path}")

    interface.launch(
        server_port=args.port,
        share=args.share,
        server_name="0.0.0.0"
    )


if __name__ == "__main__":
    main()
