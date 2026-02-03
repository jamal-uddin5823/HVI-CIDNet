# Enhanced Image Visualization Scripts

Two scripts to generate visual comparisons of enhanced images across models.

---

## Script 1: Single Face Across All Difficulties

**File**: `generate_figure_enhanced_comparison.py`

**What it shows**: One person's face enhanced at all difficulty levels (easy, medium, hard, mixed)

**Layout**:
```
                Low-Light | Baseline | Face Loss 0.3 | Face Loss 0.5 | Ground Truth
Easy (1%)       [image]   | [image]  | [image]       | [image]       | [image]
Medium (5%)     [image]   | [image]  | [image]       | [image]       | [image]
Hard (10%)      [image]   | [image]  | [image]       | [image]       | [image]
Mixed           [image]   | [image]  | [image]       | [image]       | [image]
```

**Usage**:

Auto-detect suitable image:
```bash
python generate_figure_enhanced_comparison.py
```

Use specific person/image:
```bash
python generate_figure_enhanced_comparison.py George_W_Bush George_W_Bush_0001_easy.png
```

**Output**: `figures/figure_enhanced_comparison.pdf` and `.png`

**Best for**: Showing how the same face is enhanced differently across difficulty levels

---

## Script 2: Multiple Faces at One Difficulty

**File**: `generate_figure_enhanced_gallery.py`

**What it shows**: Multiple different people (3 faces) enhanced at one difficulty level

**Layout**:
```
                Low-Light | Baseline | Face Loss 0.3 | Face Loss 0.5 | Ground Truth
Person 1        [image]   | [image]  | [image]       | [image]       | [image]
Person 2        [image]   | [image]  | [image]       | [image]       | [image]
Person 3        [image]   | [image]  | [image]       | [image]       | [image]
```

**Usage**:

Generate for medium difficulty (default):
```bash
python generate_figure_enhanced_gallery.py
```

Generate for specific difficulty:
```bash
python generate_figure_enhanced_gallery.py easy
python generate_figure_enhanced_gallery.py medium
python generate_figure_enhanced_gallery.py hard
python generate_figure_enhanced_gallery.py mixed
```

**Output**: `figures/figure_enhanced_gallery_{difficulty}.pdf` and `.png`

**Best for**: Demonstrating consistent improvement across different faces

---

## Which One to Use in Thesis?

### Recommended: Use BOTH

1. **Figure_enhanced_comparison**: In the "Results" section to show comprehensive performance across all difficulty levels
   - Caption: "Visual comparison of enhanced images across difficulty levels. Face Loss (FR=0.5) produces sharper, more natural-looking enhancements while preserving facial details."

2. **Figure_enhanced_gallery_medium**: In the "Results" or "Qualitative Analysis" section to show consistency
   - Caption: "Enhanced image gallery at medium difficulty level (5% light). Face Loss models consistently improve facial clarity across different subjects."

### Alternative: Use Script 2 for Multiple Difficulties

Generate galleries for each difficulty:
```bash
python generate_figure_enhanced_gallery.py easy
python generate_figure_enhanced_gallery.py medium
python generate_figure_enhanced_gallery.py hard
```

Then include the most impressive one (likely medium or hard) in the thesis.

---

## Troubleshooting

### Issue: "Could not find a suitable sample image"

**Cause**: Enhanced images not in expected directory structure

**Solution**: Check where enhanced images are stored:
```bash
ls results/multilevel_evaluations/baseline/medium/enhanced/
```

If they're in a different location, update the `base_dir` variable in the scripts:
```python
# Line ~15 in both scripts
base_dir = 'results/multilevel_evaluations'  # Update this path
```

### Issue: "Not found" text appears instead of images

**Cause**: File paths don't match actual structure

**Solution**:
1. Check if enhanced images exist:
   ```bash
   find results/multilevel_evaluations -name "*.png" | head -10
   ```

2. Check dataset structure:
   ```bash
   ls datasets/LFW_multilevel/test_medium/low/
   ```

3. Manually specify a known person/image:
   ```bash
   python generate_figure_enhanced_comparison.py <person_name> <image_filename>
   ```

### Issue: Images are there but look identical

**Cause**: Models may not have learned significantly different enhancements

**Solution**:
- Choose a more challenging difficulty (medium or hard)
- Zoom in on facial details in the figure
- Add quality metrics (PSNR, SSIM) as text overlays

---

## Expected File Structure

The scripts expect this directory structure:

```
results/multilevel_evaluations/
├── baseline/
│   ├── easy/enhanced/{person_name}/{image.png}
│   ├── medium/enhanced/{person_name}/{image.png}
│   ├── hard/enhanced/{person_name}/{image.png}
│   └── mixed/enhanced/{person_name}/{image.png}
├── face_loss3/
│   └── (same structure)
└── face_loss5/
    └── (same structure)

datasets/LFW_multilevel/
├── test_easy/
│   ├── low/{person_name}/{image.png}
│   └── high/{person_name}/{image.png}
├── test_medium/
│   └── (same structure)
├── test_hard/
│   └── (same structure)
└── test_mixed/
    └── (same structure)
```

If your structure is different, modify the path variables in the scripts.

---

## Customization Options

### Change Number of Samples (Script 2)

```python
# In generate_figure_enhanced_gallery.py, line ~130
generate_enhanced_gallery(num_samples=5, difficulty='medium')  # Change to 4 or 5 samples
```

### Add Quality Metrics Overlay

Add PSNR/SSIM values to each enhanced image:

```python
# After loading image, add text annotation:
from PIL import ImageDraw, ImageFont
draw = ImageDraw.Draw(img)
psnr_value = 28.5  # Get from evaluation_data.json
draw.text((10, 10), f'PSNR: {psnr_value:.2f}', fill='white')
```

### Zoom In on Facial Region

Crop images to focus on face:

```python
# After loading image:
img = img.crop((50, 30, 200, 180))  # Adjust coordinates
```

---

## Integration with Main Script

These scripts are automatically run by `generate_all_thesis_figures.sh`:

```bash
./generate_all_thesis_figures.sh
```

This generates:
- `figure_enhanced_comparison.pdf/png` (single face, all difficulties)
- `figure_enhanced_gallery_medium.pdf/png` (3 faces, medium difficulty)

To disable, comment out lines in the master script:
```bash
# echo "  - Enhanced comparison..."
# python generate_figure_enhanced_comparison.py
```
