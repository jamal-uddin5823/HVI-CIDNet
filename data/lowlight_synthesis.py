"""
Physics-based Low-Light Image Synthesis

This module implements realistic low-light image simulation using physics-based
sensor noise models and camera characteristics. Unlike simple brightness reduction,
this approach simulates the actual photon arrival process and sensor behavior.

Key Features:
1. Linear Space Transformation: Converts sRGB to linear RGB for realistic light reduction
2. Poisson-Gaussian Noise: Models shot noise (signal-dependent) and read noise (signal-independent)
3. White Balance Simulation: Simulates auto-white-balance failures in low lightP
4. Blur/Detail Loss: Simulates motion blur and denoising artifacts

References:
- Foi et al. (2008) "Practical Poissonian-Gaussian Noise Modeling and Fitting for Single-Image Raw-Data"
- Hasinoff et al. (2016) "Burst Photography for High Dynamic Range and Low-Light Imaging on Mobile Cameras"
"""

import os
import numpy as np
from typing import Union, Tuple, Optional
import cv2
from PIL import Image


# ============================================================================
# 1. Linear Space Transformation
# ============================================================================

def srgb_to_linear(img: np.ndarray) -> np.ndarray:
    """
    Convert sRGB image to linear RGB space.

    In sRGB space, pixel values are gamma-encoded for display. In linear space,
    pixel values are proportional to physical light intensity, which is necessary
    for realistic light reduction simulation.

    The sRGB to linear conversion follows the official sRGB specification:
    - For dark values (≤ 0.04045): linear = srgb / 12.92
    - For bright values (> 0.04045): linear = ((srgb + 0.055) / 1.055)^2.4

    Args:
        img: Input image in sRGB space, range [0, 1], shape (H, W, C) or (H, W)

    Returns:
        Image in linear RGB space, range [0, 1], same shape as input

    Example:
        >>> srgb_img = np.random.rand(256, 256, 3)
        >>> linear_img = srgb_to_linear(srgb_img)
    """
    # Ensure float type
    img = img.astype(np.float32)

    # Apply sRGB to linear transformation
    # Two different formulas for dark and bright regions
    linear = np.where(
        img <= 0.04045,
        img / 12.92,
        np.power((img + 0.055) / 1.055, 2.4)
    )

    return linear


def linear_to_srgb(img: np.ndarray) -> np.ndarray:
    """
    Convert linear RGB image back to sRGB space for display.

    This is the inverse of srgb_to_linear(). Applies gamma encoding to
    make the image suitable for display on standard monitors.

    The linear to sRGB conversion:
    - For dark values (≤ 0.0031308): srgb = linear * 12.92
    - For bright values (> 0.0031308): srgb = 1.055 * linear^(1/2.4) - 0.055

    Args:
        img: Input image in linear RGB space, range [0, 1], shape (H, W, C) or (H, W)

    Returns:
        Image in sRGB space, range [0, 1], same shape as input

    Example:
        >>> linear_img = np.random.rand(256, 256, 3)
        >>> srgb_img = linear_to_srgb(linear_img)
    """
    # Ensure float type
    img = img.astype(np.float32)

    # Clip to valid range first
    img = np.clip(img, 0, 1)

    # Apply linear to sRGB transformation
    srgb = np.where(
        img <= 0.0031308,
        img * 12.92,
        1.055 * np.power(img, 1.0 / 2.4) - 0.055
    )

    return srgb


def reduce_light_intensity(
    img_linear: np.ndarray,
    reduction_factor: float
) -> np.ndarray:
    """
    Reduce light intensity in linear space.

    This simulates a reduction in exposure time or ISO. In linear space,
    multiplying by a factor directly corresponds to reducing the amount
    of light captured by the sensor.

    Args:
        img_linear: Image in linear RGB space, range [0, 1]
        reduction_factor: Factor to reduce light by.
                         E.g., 0.1 means 10% of original light (90% darker)
                         Range: (0, 1]

    Returns:
        Darkened image in linear space, range [0, 1]

    Example:
        >>> linear_img = srgb_to_linear(img)
        >>> dark_img = reduce_light_intensity(linear_img, 0.1)  # 90% darker
    """
    assert 0 < reduction_factor <= 1, "reduction_factor must be in (0, 1]"

    return img_linear * reduction_factor


# ============================================================================
# 2. Poisson-Gaussian Noise Model
# ============================================================================

def add_poisson_gaussian_noise(
    img_linear: np.ndarray,
    shot_noise_scale: float = 1.0,
    read_noise_std: float = 0.01,
    gain: float = 1.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Add realistic sensor noise using Poisson-Gaussian model.

    This is the most accurate model of real sensor noise, consisting of:

    1. Shot Noise (Poisson): Noise from random photon arrivals
       - Variance proportional to signal intensity (√I behavior)
       - Brighter regions have more noise
       - Models the quantum nature of light

    2. Read Noise (Gaussian): Electronic noise from sensor readout
       - Constant variance across the image
       - Visible even in pure black regions
       - Independent of signal intensity

    The model: y = gain * Poisson(x / gain) + Gaussian(0, σ)

    Args:
        img_linear: Input image in linear space, range [0, 1]
        shot_noise_scale: Scale factor for shot noise (default: 1.0)
                         Higher = more shot noise
        read_noise_std: Standard deviation of read noise (default: 0.01)
                       Typical range: 0.001 - 0.02
        gain: Sensor gain/ISO (default: 1.0)
              Higher gain amplifies both signal and noise
        seed: Random seed for reproducibility

    Returns:
        Noisy image in linear space

    Example:
        >>> # Simulate low-light with high ISO
        >>> dark_img = reduce_light_intensity(linear_img, 0.05)
        >>> noisy_img = add_poisson_gaussian_noise(dark_img,
        ...                                        shot_noise_scale=2.0,
        ...                                        read_noise_std=0.015,
        ...                                        gain=4.0)
    """
    if seed is not None:
        np.random.seed(seed)

    # Scale image to higher range for Poisson noise (prevents quantization)
    # Poisson noise needs reasonable count values
    scale = 255.0
    img_scaled = img_linear * scale / gain

    # 1. Shot noise (Poisson)
    # Poisson distribution parameter λ = signal intensity
    # For computational efficiency, we approximate with Gaussian when λ is large
    if shot_noise_scale > 0:
        # Use Poisson sampling
        # clip to avoid negative values which would cause issues
        img_scaled = np.clip(img_scaled, 0, None)

        # For performance: use Gaussian approximation for high intensities
        # Poisson(λ) ≈ Gaussian(λ, λ) when λ is large
        shot_noisy = np.random.poisson(img_scaled).astype(np.float32)
        shot_noisy *= shot_noise_scale
        # Blend back
        img_with_shot = shot_noisy * gain / scale
    else:
        img_with_shot = img_linear

    # 2. Read noise (Gaussian)
    # This is constant across the image (signal-independent)
    if read_noise_std > 0:
        read_noise = np.random.normal(0, read_noise_std, img_linear.shape).astype(np.float32)
        img_noisy = img_with_shot + read_noise
    else:
        img_noisy = img_with_shot

    # Clip to valid range
    img_noisy = np.clip(img_noisy, 0, 1)

    return img_noisy


def add_simplified_sensor_noise(
    img_linear: np.ndarray,
    noise_intensity: float = 0.02,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Simplified version of sensor noise for quick experiments.

    This uses a signal-dependent Gaussian noise model as approximation:
    noise_std = noise_intensity * sqrt(signal)

    While not as accurate as the full Poisson-Gaussian model, this is
    faster and captures the key property that noise increases with signal.

    Args:
        img_linear: Input image in linear space, range [0, 1]
        noise_intensity: Controls overall noise level (default: 0.02)
        seed: Random seed for reproducibility

    Returns:
        Noisy image in linear space
    """
    if seed is not None:
        np.random.seed(seed)

    # Signal-dependent noise: std proportional to sqrt(signal)
    noise_std = noise_intensity * np.sqrt(np.maximum(img_linear, 1e-6))
    noise = np.random.normal(0, 1, img_linear.shape).astype(np.float32) * noise_std

    img_noisy = img_linear + noise
    img_noisy = np.clip(img_noisy, 0, 1)

    return img_noisy


# ============================================================================
# 3. White Balance Simulation
# ============================================================================

def simulate_white_balance_failure(
    img: np.ndarray,
    r_gain: Optional[float] = None,
    g_gain: Optional[float] = None,
    b_gain: Optional[float] = None,
    variation_range: float = 0.2,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Simulate auto-white-balance failure in low light.

    In low-light conditions, cameras often fail to correctly estimate the
    scene's color temperature, resulting in color casts (yellowish, bluish,
    or greenish tints).

    This function simulates this by applying different gains to each color channel.

    Args:
        img: Input image in linear or sRGB space, range [0, 1], shape (H, W, 3)
        r_gain: Red channel gain. If None, randomly sampled from [1-var, 1+var]
        g_gain: Green channel gain. If None, randomly sampled from [1-var, 1+var]
        b_gain: Blue channel gain. If None, randomly sampled from [1-var, 1+var]
        variation_range: Range of random variation (default: 0.2)
        seed: Random seed for reproducibility

    Returns:
        Image with simulated white balance error

    Example:
        >>> # Simulate warm (yellowish) color cast
        >>> wb_img = simulate_white_balance_failure(img, r_gain=1.2, b_gain=0.8)

        >>> # Random color cast
        >>> wb_img = simulate_white_balance_failure(img, variation_range=0.3)
    """
    if seed is not None:
        np.random.seed(seed)

    assert img.ndim == 3 and img.shape[2] == 3, "Image must be RGB (H, W, 3)"

    # Sample random gains if not provided
    if r_gain is None:
        r_gain = np.random.uniform(1 - variation_range, 1 + variation_range)
    if g_gain is None:
        g_gain = np.random.uniform(1 - variation_range, 1 + variation_range)
    if b_gain is None:
        b_gain = np.random.uniform(1 - variation_range, 1 + variation_range)

    # Apply per-channel gains
    img_wb = img.copy()
    img_wb[:, :, 0] *= r_gain  # Red
    img_wb[:, :, 1] *= g_gain  # Green
    img_wb[:, :, 2] *= b_gain  # Blue

    # Clip to valid range
    img_wb = np.clip(img_wb, 0, 1)

    return img_wb


def simulate_color_temperature_shift(
    img: np.ndarray,
    temperature_shift: float,
) -> np.ndarray:
    """
    Simulate color temperature shift in Kelvin.

    Positive shift = warmer (more yellow/orange)
    Negative shift = cooler (more blue)

    Args:
        img: Input image in linear or sRGB space, range [0, 1], shape (H, W, 3)
        temperature_shift: Temperature shift in Kelvin
                          Typical range: -500 to +500
                          Positive = warmer, Negative = cooler

    Returns:
        Image with color temperature shift applied

    Example:
        >>> # Warm color cast (like tungsten lighting)
        >>> warm_img = simulate_color_temperature_shift(img, 300)

        >>> # Cool color cast (like overcast daylight)
        >>> cool_img = simulate_color_temperature_shift(img, -200)
    """
    assert img.ndim == 3 and img.shape[2] == 3, "Image must be RGB (H, W, 3)"

    # Simplified color temperature model
    # Based on approximation of blackbody radiation
    # Positive shift -> increase red, decrease blue
    # Negative shift -> decrease red, increase blue

    # Normalize shift to reasonable gain range
    shift_factor = temperature_shift / 1000.0  # Convert to manageable range

    # Calculate per-channel gains
    r_gain = 1.0 + shift_factor * 0.3
    g_gain = 1.0
    b_gain = 1.0 - shift_factor * 0.3

    # Apply gains
    img_shifted = img.copy()
    img_shifted[:, :, 0] *= r_gain
    img_shifted[:, :, 1] *= g_gain
    img_shifted[:, :, 2] *= b_gain

    img_shifted = np.clip(img_shifted, 0, 1)

    return img_shifted


# ============================================================================
# 4. Blur / Loss of Detail
# ============================================================================

def apply_low_light_blur(
    img: np.ndarray,
    kernel_size: int = 3,
    sigma: float = 0.5
) -> np.ndarray:
    """
    Apply subtle blur to simulate loss of detail in low light.

    Low-light images often appear softer due to:
    1. Motion blur from longer exposure times
    2. Aggressive denoising by camera ISP
    3. Lower effective resolution due to noise

    Args:
        img: Input image, range [0, 1]
        kernel_size: Size of Gaussian kernel (default: 3)
                    Odd number, typically 3 or 5
        sigma: Standard deviation of Gaussian kernel (default: 0.5)
              Higher = more blur. Typical range: 0.3 - 1.5

    Returns:
        Blurred image

    Example:
        >>> # Subtle blur
        >>> blurred = apply_low_light_blur(img, kernel_size=3, sigma=0.5)

        >>> # More pronounced blur
        >>> blurred = apply_low_light_blur(img, kernel_size=5, sigma=1.0)
    """
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

    return blurred


def apply_motion_blur(
    img: np.ndarray,
    kernel_size: int = 9,
    angle: float = 0.0
) -> np.ndarray:
    """
    Apply directional motion blur.

    Simulates camera shake or subject motion during long exposure.

    Args:
        img: Input image, range [0, 1]
        kernel_size: Length of motion blur (default: 9)
        angle: Direction of motion in degrees (default: 0.0)
               0 = horizontal, 90 = vertical

    Returns:
        Motion blurred image

    Example:
        >>> # Horizontal motion blur
        >>> blurred = apply_motion_blur(img, kernel_size=9, angle=0)

        >>> # Diagonal motion blur
        >>> blurred = apply_motion_blur(img, kernel_size=11, angle=45)
    """
    # Create motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size

    # Rotate kernel to desired angle
    M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))

    # Renormalize
    kernel = kernel / np.sum(kernel)

    # Apply convolution
    blurred = cv2.filter2D(img, -1, kernel)

    return blurred


# ============================================================================
# 5. Main Pipeline
# ============================================================================

def synthesize_low_light_image(
    img: Union[np.ndarray, Image.Image, str],
    apply_light_reduction: bool = True,
    apply_noise: bool = False,  # DEFAULT CHANGED: Supervisor guidance - no information destruction
    apply_white_balance: bool = False,  # DEFAULT CHANGED: Supervisor guidance - no color destruction
    apply_blur: bool = False,  # DEFAULT CHANGED: Supervisor guidance - no spatial information loss
    reduction_factor: float = 0.05,  # DEFAULT CHANGED: Lower for challenging scenarios (5% light)
    shot_noise_scale: float = 1.5,
    read_noise_std: float = 0.01,
    gain: float = 2.0,
    wb_variation: float = 0.2,
    blur_sigma: float = 0.5,
    blur_type: str = 'gaussian',
    motion_blur_angle: float = 0.0,
    output_format: str = 'numpy',
    seed: Optional[int] = None
) -> Union[np.ndarray, Image.Image]:
    """
    Physically accurate low-light image synthesis with 4 core transformations.

    The transformations are applied in the physically correct order:
    1. Blur (optional) - Scene/optical effect before sensor
    2. to_linear - Convert sRGB to linear space
    3. Reduce Light - Exposure control
    4. White Balance - Analog channel gains (in linear space)
    5. Noise - Sensor noise (Poisson + Gaussian)
    6. to_srgb - ISP gamma correction

    Args:
        img: Input image (numpy array, PIL Image, or file path)
        apply_light_reduction: Apply light reduction (default: True)
        apply_noise: Apply Poisson-Gaussian noise (default: True)
        apply_white_balance: Apply white balance failure (default: True)
        apply_blur: Apply blur (default: True)
        reduction_factor: Light reduction factor, range (0, 1] (default: 0.1)
        shot_noise_scale: Shot noise scale (default: 1.5)
        read_noise_std: Read noise std deviation (default: 0.01)
        gain: Sensor gain/ISO (default: 2.0)
        wb_variation: White balance variation range (default: 0.2)
        blur_sigma: Blur sigma (default: 0.5)
        blur_type: 'gaussian' or 'motion' (default: 'gaussian')
        motion_blur_angle: Angle for motion blur in degrees (default: 0.0)
        output_format: 'numpy' or 'pil' (default: 'numpy')
        seed: Random seed for reproducibility

    Returns:
        Synthesized low-light image

    Example:
        >>> # All transformations
        >>> low_light = synthesize_low_light_image('image.jpg')

        >>> # Only light reduction
        >>> low_light = synthesize_low_light_image(
        ...     img,
        ...     apply_noise=False,
        ...     apply_white_balance=False,
        ...     apply_blur=False,
        ...     reduction_factor=0.05
        ... )

        >>> # Only noise
        >>> low_light = synthesize_low_light_image(
        ...     img,
        ...     apply_light_reduction=False,
        ...     apply_white_balance=False,
        ...     apply_blur=False
        ... )

        >>> # Custom combination
        >>> low_light = synthesize_low_light_image(
        ...     img,
        ...     apply_light_reduction=True,
        ...     apply_noise=True,
        ...     apply_white_balance=False,
        ...     apply_blur=True,
        ...     reduction_factor=0.1,
        ...     blur_type='motion',
        ...     motion_blur_angle=45
        ... )
    """
    # Load and normalize input image
    if isinstance(img, str):
        img = Image.open(img).convert('RGB')

    if isinstance(img, Image.Image):
        img_array = np.array(img).astype(np.float32) / 255.0
    else:
        img_array = img.astype(np.float32)
        if img_array.max() > 1.0:
            img_array = img_array / 255.0

    # Ensure RGB
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)

    if seed is not None:
        np.random.seed(seed)

    # --- PHYSICALLY ACCURATE ORDER ---

    # Step 1: Blur (optional) - happens before sensor
    if apply_blur:
        if blur_type == 'motion':
            img_array = apply_motion_blur(
                img_array,
                kernel_size=int(blur_sigma * 6 + 1),  # Convert sigma to kernel size
                angle=motion_blur_angle
            )
        else:  # gaussian
            img_array = apply_low_light_blur(img_array, sigma=blur_sigma)

    # Step 2: Convert to linear space
    img_linear = srgb_to_linear(img_array)

    # Step 3: Reduce light (exposure control)
    if apply_light_reduction:
        img_linear = reduce_light_intensity(img_linear, reduction_factor)

    # Step 4: White balance (analog gains in linear space)
    if apply_white_balance:
        img_linear = simulate_white_balance_failure(
            img_linear,
            variation_range=wb_variation,
            seed=seed
        )

    # Step 5: Add sensor noise
    if apply_noise:
        img_linear = add_poisson_gaussian_noise(
            img_linear,
            shot_noise_scale=shot_noise_scale,
            read_noise_std=read_noise_std,
            gain=gain,
            seed=seed
        )

    # Step 6: Convert back to sRGB
    img_srgb = linear_to_srgb(img_linear)

    # Final clip
    img_srgb = np.clip(img_srgb, 0, 1)

    # Return in requested format
    if output_format == 'pil':
        img_uint8 = (img_srgb * 255).astype(np.uint8)
        return Image.fromarray(img_uint8)
    else:
        return img_srgb


def create_low_light_dataset(
    image_paths: list,
    output_dir: str,
    num_variants_per_image: int = 3,
    reduction_factors: Optional[list] = None,
    save_pairs: bool = False,
    seed: Optional[int] = None
) -> None:
    """
    Create a dataset of low-light images from a list of input images.

    Generates multiple low-light variants with different degradation levels.

    Args:
        image_paths: List of paths to input images
        output_dir: Directory to save synthesized images
        num_variants_per_image: Number of variants per image (default: 3)
        reduction_factors: List of reduction factors. If None, uses [0.05, 0.1, 0.2]
        save_pairs: If True, saves original and low-light side-by-side
        seed: Random seed base (will increment for each variant)

    Example:
        >>> create_low_light_dataset(
        ...     image_paths=['img1.jpg', 'img2.jpg'],
        ...     output_dir='./low_light_dataset',
        ...     num_variants_per_image=5
        ... )
    """
    import os
    from pathlib import Path
    from tqdm import tqdm

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Default reduction factors
    if reduction_factors is None:
        reduction_factors = [0.05, 0.1, 0.2]

    for img_idx, img_path in enumerate(image_paths):
        img_name = Path(img_path).stem

        # Load original image
        original = Image.open(img_path).convert('RGB')

        for var_idx in range(num_variants_per_image):
            # Get current seed
            current_seed = seed + img_idx * 1000 + var_idx if seed is not None else None

            # Select reduction factor (cycle through list)
            reduction_factor = reduction_factors[var_idx % len(reduction_factors)]

            # Vary parameters
            if current_seed is not None:
                np.random.seed(current_seed)

            shot_noise = np.random.uniform(1.0, 2.5)
            read_noise = np.random.uniform(0.005, 0.02)
            gain = np.random.uniform(1.5, 4.0)
            wb_variation = np.random.uniform(0.15, 0.25)
            blur_sigma = np.random.uniform(0.3, 0.7)

            # Synthesize low-light image
            low_light = synthesize_low_light_image(
                original,
                reduction_factor=reduction_factor,
                shot_noise_scale=shot_noise,
                read_noise_std=read_noise,
                gain=gain,
                wb_variation=wb_variation,
                blur_sigma=blur_sigma,
                seed=current_seed,
                output_format='pil'
            )

            # Generate filename
            output_name = f"{img_name}_lowlight_{var_idx:02d}_rf{int(reduction_factor*100):02d}.png"
            low_light.save(os.path.join(output_dir, output_name))

            # Save side-by-side comparison
            if save_pairs:
                combined = Image.new('RGB', (original.width * 2, original.height))
                combined.paste(original, (0, 0))
                combined.paste(low_light, (original.width, 0))
                pair_name = f"{img_name}_pair_{var_idx:02d}.png"
                combined.save(os.path.join(output_dir, pair_name))

        print(f"Processed {img_idx + 1}/{len(image_paths)}: {img_path}")

    print(f"Dataset created in {output_dir}")


# ============================================================================
# Utility Functions
# ============================================================================

def estimate_noise_parameters(
    dark_img: np.ndarray,
    bright_img: np.ndarray,
    num_samples: int = 1000
) -> Tuple[float, float]:
    """
    Estimate shot noise and read noise from a pair of images.

    This can be used to calibrate the noise model to match a specific camera.

    Args:
        dark_img: Dark image (low signal), range [0, 1]
        bright_img: Bright image (high signal), range [0, 1]
        num_samples: Number of random pixels to sample

    Returns:
        (shot_noise_scale, read_noise_std) estimated parameters
    """
    # Sample random pixels
    h, w = dark_img.shape[:2]
    y_coords = np.random.randint(0, h, num_samples)
    x_coords = np.random.randint(0, w, num_samples)

    dark_samples = dark_img[y_coords, x_coords]
    bright_samples = bright_img[y_coords, x_coords]

    # Estimate read noise from dark regions (constant noise floor)
    read_noise_std = np.std(dark_samples)

    # Estimate shot noise from bright regions (signal-dependent)
    bright_variance = np.var(bright_samples)
    bright_mean = np.mean(bright_samples)
    shot_noise_scale = np.sqrt(bright_variance - read_noise_std**2) / np.sqrt(bright_mean)

    return float(shot_noise_scale), float(read_noise_std)


if __name__ == "__main__":
    # Example usage demonstrating the flexible transformation pipeline
    print("Low-Light Image Synthesis Module")
    print("=" * 70)

    # Load test image
    test_img_path = "../data/test/image.png"
    if not os.path.exists(test_img_path):
        print(f"Warning: {test_img_path} not found, using random test image")
        test_img = np.random.rand(256, 256, 3).astype(np.float32) * 0.5 + 0.3
    else:
        test_img = Image.open(test_img_path).convert('RGB')
        test_img = np.array(test_img).astype(np.float32) / 255.0

    # Create output directory
    os.makedirs("../out/tests", exist_ok=True)

    print(f"\nInput image shape: {test_img.shape}")
    print(f"Input range: [{test_img.min():.3f}, {test_img.max():.3f}]")

    # Save original test image for comparison
    original_img = (test_img * 255).astype(np.uint8)
    Image.fromarray(original_img).save("../out/tests/original_test_image.png")
    print("Saved: tests/original_test_image.png")
    print("=" * 70)

    # Example 1: Default pipeline (enable all transformations)
    print("\n[Example 1] Default pipeline (all transformations)")
    low_light_1 = synthesize_low_light_image(
        test_img,
        apply_light_reduction=True,
        apply_noise=True,
        apply_white_balance=True,
        apply_blur=True,
        seed=42
    )
    print(f"Output range: [{low_light_1.min():.3f}, {low_light_1.max():.3f}]")

    # Save output image
    output_1 = (low_light_1 * 255).astype(np.uint8)
    Image.fromarray(output_1).save("../out/tests/example_1_default_pipeline.png")
    print("Saved: tests/example_1_default_pipeline.png")

    # Example 2: Minimal pipeline - only light reduction
    print("\n[Example 2] Minimal: Only light reduction, no noise or blur")
    low_light_2 = synthesize_low_light_image(
        test_img,
        apply_light_reduction=True,
        apply_noise=False,
        apply_white_balance=False,
        apply_blur=False,
        reduction_factor=0.05,
        seed=42
    )
    print(f"Output range: [{low_light_2.min():.3f}, {low_light_2.max():.3f}]")

    # Save output image
    output_2 = (low_light_2 * 255).astype(np.uint8)
    Image.fromarray(output_2).save("../out/tests/example_2_light_reduction_only.png")
    print("Saved: tests/example_2_light_reduction_only.png")

    # Example 3: High noise with custom parameters
    print("\n[Example 3] High noise: Custom Poisson-Gaussian noise parameters")
    low_light_3 = synthesize_low_light_image(
        test_img,
        apply_light_reduction=True,
        apply_noise=True,
        apply_white_balance=False,
        apply_blur=False,
        reduction_factor=0.1,
        shot_noise_scale=2.0,
        read_noise_std=0.015,
        gain=3.0,
        seed=42
    )
    print(f"Output range: [{low_light_3.min():.3f}, {low_light_3.max():.3f}]")

    # Save output image
    output_3 = (low_light_3 * 255).astype(np.uint8)
    Image.fromarray(output_3).save("../out/tests/example_3_high_noise.png")
    print("Saved: tests/example_3_high_noise.png")

    # Example 4: White balance simulation
    print("\n[Example 4] Color effects: White balance failure simulation")
    low_light_4 = synthesize_low_light_image(
        test_img,
        apply_light_reduction=True,
        apply_noise=False,
        apply_white_balance=True,
        apply_blur=False,
        reduction_factor=0.15,
        wb_variation=0.25,
        seed=42
    )
    print(f"Output range: [{low_light_4.min():.3f}, {low_light_4.max():.3f}]")

    # Save output image
    output_4 = (low_light_4 * 255).astype(np.uint8)
    Image.fromarray(output_4).save("../out/tests/example_4_white_balance_failure.png")
    print("Saved: tests/example_4_white_balance_failure.png")

    # Example 5: Motion blur
    print("\n[Example 5] Blur effects: Motion blur simulation")
    low_light_5 = synthesize_low_light_image(
        test_img,
        apply_light_reduction=True,
        apply_noise=False,
        apply_white_balance=False,
        apply_blur=True,
        reduction_factor=0.1,
        blur_type='motion',
        blur_sigma=1.5,  # This will be converted to kernel_size
        motion_blur_angle=30,
        seed=42
    )
    print(f"Output range: [{low_light_5.min():.3f}, {low_light_5.max():.3f}]")

    # Save output image
    output_5 = (low_light_5 * 255).astype(np.uint8)
    Image.fromarray(output_5).save("../out/tests/example_5_motion_blur.png")
    print("Saved: tests/example_5_motion_blur.png")

    # Example 6: Everything combined - extreme degradation
    print("\n[Example 6] Maximum degradation: All effects combined")
    low_light_6 = synthesize_low_light_image(
        test_img,
        apply_light_reduction=True,
        apply_noise=True,
        apply_white_balance=True,
        apply_blur=True,
        reduction_factor=0.05,
        shot_noise_scale=2.5,
        read_noise_std=0.02,
        gain=4.0,
        wb_variation=0.3,
        blur_type='gaussian',
        blur_sigma=0.7,
        seed=42
    )
    print(f"Output range: [{low_light_6.min():.3f}, {low_light_6.max():.3f}]")

    # Save output image
    output_6 = (low_light_6 * 255).astype(np.uint8)
    Image.fromarray(output_6).save("../out/tests/example_6_maximum_degradation.png")
    print("Saved: tests/example_6_maximum_degradation.png")

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("\nKey insight: Transformations are NOT mutually exclusive.")
    print("You can combine multiple noise types, blur types, and color adjustments")
    print("by simply listing them in the transformations list.")
    print("=" * 70)