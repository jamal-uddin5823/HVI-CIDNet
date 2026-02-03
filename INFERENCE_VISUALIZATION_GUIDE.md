# Enhanced Image Visualization with Model Inference

These scripts load your trained models and generate enhanced images on-the-fly for visualization.

---

## Overview

**New Scripts** (generate images via inference):
1. `generate_enhanced_comparison_with_inference.py` - Single face across all difficulties
2. `generate_enhanced_gallery_with_inference.py` - Multiple faces at one difficulty

**What they do**:
1. Load trained model checkpoints (baseline, face_loss3, face_loss5)
2. Take low-light test images as input
3. Run inference to generate enhanced versions
4. Display side-by-side comparison

---

## Prerequisites

### Required Files

**Model Checkpoints**:
```
checkpoints/
├── baseline/best_model.pth
├── face_loss3/best_model.pth
└── face_loss5/best_model.pth
```

**Test Dataset**:
```
datasets/LFW_multilevel/
├── test_easy/
│   ├── low/{person_name}/{image.png}
│   └── high/{person_name}/{image.png}
├── test_medium/
├── test_hard/
└── test_mixed/
```

**Model Architecture**:
- The script expects a `model.py` file with `CIDNet` class definition
- Ensure `model.py` is in the same directory or Python path

---

## Usage

### Script 1: Single Face Comparison

**Basic usage** (auto-detect checkpoint and image):
```bash
python generate_enhanced_comparison_with_inference.py
```

**Specify checkpoint directory**:
```bash
python generate_enhanced_comparison_with_inference.py /path/to/checkpoints
```

**What it generates**:
- Loads all three models
- Picks a random person from test set
- Enhances at easy, medium, hard, mixed difficulties
- Saves to `figures/figure_enhanced_comparison_inference.pdf`

---

### Script 2: Multiple Face Gallery

**Basic usage** (medium difficulty, auto-detect checkpoints):
```bash
python generate_enhanced_gallery_with_inference.py
```

**Specify difficulty**:
```bash
python generate_enhanced_gallery_with_inference.py medium
python generate_enhanced_gallery_with_inference.py hard
```

**Specify checkpoint directory and difficulty**:
```bash
python generate_enhanced_gallery_with_inference.py /path/to/checkpoints medium
```

**What it generates**:
- Loads all three models
- Picks 3 different people from test set
- Enhances at specified difficulty
- Saves to `figures/figure_enhanced_gallery_{difficulty}_inference.pdf`

---

## Running on HPC

### Option 1: Automatic (via master script)

The master script will run both visualization scripts automatically:

```bash
./generate_all_thesis_figures.sh
```

This assumes checkpoints are in `checkpoints/` directory. If they're elsewhere, edit the script:

```bash
# Line 45-48 in generate_all_thesis_figures.sh
python generate_enhanced_comparison_with_inference.py /your/checkpoint/path
python generate_enhanced_gallery_with_inference.py /your/checkpoint/path medium
```

### Option 2: Manual (run individually)

```bash
# Find your checkpoint directory
ls -la checkpoints/
ls -la saved_models/
ls -la experiments/

# Run with correct path
python generate_enhanced_comparison_with_inference.py checkpoints
python generate_enhanced_gallery_with_inference.py checkpoints medium
```

---

## GPU vs CPU

**Default**: Uses CUDA (GPU) if available, falls back to CPU

**Expected runtime**:
- GPU: ~10-30 seconds per figure (3 models × 4 difficulties × inference)
- CPU: ~2-5 minutes per figure

**To force CPU**:
```python
# Edit line ~240 in the scripts:
device='cpu'  # Change from 'cuda'
```

---

## Troubleshooting

### Issue 1: "No model checkpoints found"

**Cause**: Checkpoint files not in expected location

**Solution**:
```bash
# Find where checkpoints are stored
find . -name "*.pth" -o -name "*.pt" | grep -E "(baseline|face_loss)"

# Use that directory
python generate_enhanced_comparison_with_inference.py /path/to/actual/checkpoints
```

### Issue 2: "Cannot import model"

**Cause**: `model.py` not found or CIDNet class not defined

**Solution**:
```bash
# Check if model.py exists
ls -la model.py

# Check class name
grep "class.*Net" model.py

# If class name is different (e.g., MyModel instead of CIDNet), edit the scripts:
# Line ~13 in both scripts:
from model import MyModel  # Change CIDNet to your class name
model = MyModel().to(device)  # Change CIDNet to your class name
```

### Issue 3: "RuntimeError: state_dict mismatch"

**Cause**: Checkpoint saved with different model architecture

**Solution**:
```python
# The script tries multiple loading strategies:
# 1. checkpoint['model_state_dict']
# 2. checkpoint['state_dict']
# 3. checkpoint directly

# If still failing, check checkpoint structure:
import torch
ckpt = torch.load('checkpoints/baseline/best_model.pth')
print(ckpt.keys())  # See what keys exist

# Then modify load_model() function accordingly
```

### Issue 4: "CUDA out of memory"

**Cause**: GPU doesn't have enough memory for model

**Solution**:
```bash
# Option 1: Use CPU
python generate_enhanced_comparison_with_inference.py checkpoints

# Option 2: Process one image at a time (already implemented)
# Option 3: Use smaller batch size (script uses batch_size=1 already)
```

### Issue 5: Enhanced images look wrong (too dark/bright)

**Cause**: Model output range mismatch

**Solution**:
```python
# Check if model outputs [0,1] or [-1,1] or [0,255]
# Adjust in enhance_image() function (line ~30):

# If model outputs [-1,1]:
enhanced_tensor = (enhanced_tensor + 1) / 2  # Scale to [0,1]

# If model outputs [0,255]:
enhanced_tensor = enhanced_tensor / 255.0  # Scale to [0,1]
```

### Issue 6: "Test directory not found"

**Cause**: Dataset not in expected location

**Solution**:
```python
# Edit dataset_dir in the scripts (line ~95):
dataset_dir = 'your/actual/dataset/path/LFW_multilevel'

# Or create a symlink:
ln -s /path/to/your/dataset datasets/LFW_multilevel
```

---

## Expected Output

### Console Output

```
========================================================================
Generating Enhanced Image Comparison with Model Inference
========================================================================
Using device: cuda
Found checkpoint: checkpoints/baseline/best_model.pth
Found checkpoint: checkpoints/face_loss3/best_model.pth
Found checkpoint: checkpoints/face_loss5/best_model.pth

Loading checkpoint: checkpoints/baseline/best_model.pth
✓ Model loaded successfully
Loading checkpoint: checkpoints/face_loss3/best_model.pth
✓ Model loaded successfully
Loading checkpoint: checkpoints/face_loss5/best_model.pth
✓ Model loaded successfully

Auto-detecting sample image...
Using sample: George_W_Bush/George_W_Bush_0001_easy.png

Processing easy difficulty...
  ✓ Enhanced with baseline
  ✓ Enhanced with face_loss3
  ✓ Enhanced with face_loss5

Processing medium difficulty...
  ✓ Enhanced with baseline
  ✓ Enhanced with face_loss3
  ✓ Enhanced with face_loss5

Processing hard difficulty...
  ✓ Enhanced with baseline
  ✓ Enhanced with face_loss3
  ✓ Enhanced with face_loss5

Processing mixed difficulty...
  ✓ Enhanced with baseline
  ✓ Enhanced with face_loss3
  ✓ Enhanced with face_loss5

✓ figure_enhanced_comparison_inference saved to figures/

========================================================================
✓ Generation complete!
========================================================================
```

### Generated Files

```
figures/
├── figure_enhanced_comparison_inference.pdf
├── figure_enhanced_comparison_inference.png
├── figure_enhanced_gallery_medium_inference.pdf
└── figure_enhanced_gallery_medium_inference.png
```

---

## Customization

### Change Number of Faces in Gallery

```python
# Edit line ~235 in generate_enhanced_gallery_with_inference.py:
generate_gallery_with_inference(
    checkpoint_paths=checkpoint_paths,
    difficulty=difficulty,
    num_samples=5,  # Change from 3 to 5
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
```

### Use Specific Person

```python
# Modify find_sample_image() to return specific person:
# Line ~48 in generate_enhanced_comparison_with_inference.py:
def find_sample_image(dataset_dir, difficulty='easy'):
    person = 'George_W_Bush'  # Hardcode your desired person
    image_filename = 'George_W_Bush_0001_easy.png'
    return person, image_filename
```

### Generate for All Difficulties (Gallery)

```bash
for diff in easy medium hard mixed; do
    python generate_enhanced_gallery_with_inference.py checkpoints $diff
done
```

This creates 4 separate gallery figures, one for each difficulty.

---

## Integration with Thesis

### Recommended Usage

1. **Figure for Results Section**: Use comparison figure (single face, all difficulties)
   - Shows how models perform across difficulty spectrum
   - Caption: "Visual comparison of enhanced images. Face Loss (FR=0.5) produces sharper facial features and better color restoration, especially at medium and hard difficulty levels."

2. **Figure for Qualitative Analysis**: Use gallery figure (multiple faces, medium difficulty)
   - Shows consistency across different subjects
   - Caption: "Enhanced image gallery at medium difficulty. Face Loss models demonstrate consistent improvement in facial clarity across diverse subjects."

### Figure Placement

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{figures/figure_enhanced_comparison_inference.pdf}
\caption{Visual comparison of enhanced images across models and difficulty levels.
Low-light inputs are enhanced by three models: Baseline, Face Loss (FR=0.3), and
Face Loss (FR=0.5). Ground truth images are shown for reference. Face Loss models
preserve facial details and produce more natural-looking results.}
\label{fig:enhanced_comparison}
\end{figure}
```

---

## Performance Notes

- First run will be slower (model loading)
- Subsequent runs on same HPC session may benefit from cached models
- Each inference takes ~50-200ms on GPU, ~500-2000ms on CPU
- Total time: ~1-2 minutes per figure on GPU, ~5-10 minutes on CPU

---

## Differences from Previous Scripts

**Old scripts** (`generate_figure_enhanced_comparison.py`):
- ✗ Expected pre-generated enhanced images
- ✗ Failed if images not in specific directory structure
- ✗ Couldn't generate new enhancements

**New scripts** (`generate_enhanced_comparison_with_inference.py`):
- ✓ Generate enhancements on-the-fly
- ✓ Only need model checkpoints and test dataset
- ✓ More flexible and self-contained
- ✓ Guaranteed to show actual model outputs

---

## Next Steps

After generating figures:

1. **Review figures** for visual quality
   ```bash
   ls -lh figures/figure_*inference*
   ```

2. **Download from HPC** to your local machine
   ```bash
   scp user@hpc:~/code/figures/figure_*inference* ./thesis_figures/
   ```

3. **Include in LaTeX** thesis document

4. **Add captions** describing what readers should observe

5. **Reference in text** when discussing qualitative results

---

## Questions or Issues?

If you encounter problems:
1. Check console output for error messages
2. Verify checkpoint files exist and are loadable
3. Ensure test dataset is accessible
4. Check GPU memory usage (`nvidia-smi`)
5. Try CPU mode if GPU issues persist
