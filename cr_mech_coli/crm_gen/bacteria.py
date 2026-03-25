"""
Per-cell brightness assignment for synthetic bacteria.

Computes an additive brightness adjustment for each cell in a synthetic image,
either by matching measured intensities from a real microscope image or by
interpolating between configurable brightness bounds based on cell age.
Optionally overlays multi-scale spatial noise for intra-cell variation.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, zoom, label
from typing import Dict

import cr_mech_coli as crm


def extract_original_brightness(
    original_image: np.ndarray, original_mask: np.ndarray, colors: list
) -> Dict[int, float]:
    """
    Extract mean brightness for each original bacterium.

    Handles color reuse in the original mask by using connected component
    analysis to separate bacteria with the same color.

    Args:
        original_image (np.ndarray): Original microscope image (grayscale or RGB).
        original_mask (np.ndarray): Original labeled mask - can be grayscale (H x W) with
            integer labels or RGB (H x W x 3) with unique colors per cell.
        colors (list): List of colors/labels from crm.extract_positions(), maps cell index to
            color (RGB tuple) or label (integer).

    Returns:
        Dict[int, float]: Dictionary mapping cell_index -> mean_brightness.
    """
    # Convert image to grayscale if needed
    if len(original_image.shape) == 3:
        gray_image = np.mean(original_image, axis=2).astype(np.float32)
    else:
        gray_image = original_image.astype(np.float32)

    # Normalize to 0-255 range based on image dtype
    if original_image.dtype == np.uint16:
        gray_image = gray_image / 256.0  # Scale 16-bit to 8-bit range
    elif original_image.dtype == np.float32 or original_image.dtype == np.float64:
        if gray_image.max() <= 1.0:
            gray_image = gray_image * 255.0  # Scale 0-1 float to 0-255

    # Check if mask is grayscale (2D) or RGB (3D)
    is_grayscale_mask = len(original_mask.shape) == 2

    brightness_map = {}

    for cell_idx, color in enumerate(colors):
        # Find pixels with this color/label in original mask
        if is_grayscale_mask:
            # Grayscale mask: color is an integer label
            color_mask = original_mask == color
        else:
            # RGB mask: color is a tuple/array
            color = np.array(color)
            color_mask = np.all(original_mask == color, axis=2)

        if not np.any(color_mask):
            brightness_map[cell_idx] = 0.0
            continue

        # Use connected components to handle color/label reuse
        labeled_components, num_components = label(color_mask)

        if num_components > 1:
            # Find largest component (most likely the correct bacterium)
            component_sizes = [
                np.sum(labeled_components == i) for i in range(1, num_components + 1)
            ]
            largest_component = np.argmax(component_sizes) + 1
            component_mask = labeled_components == largest_component
        else:
            component_mask = color_mask

        # Extract mean brightness from original image
        if np.any(component_mask):
            mean_brightness = np.mean(gray_image[component_mask])
            brightness_map[cell_idx] = float(mean_brightness)
        else:
            brightness_map[cell_idx] = 0.0

    return brightness_map


def create_synthetic_brightness_map(
    synthetic_mask: np.ndarray, brightness_map: Dict[int, float], n_cells: int
) -> np.ndarray:
    """
    Create brightness map for synthetic bacteria.

    Maps each synthetic bacterium to its target brightness value extracted
    from the corresponding original bacterium.

    Args:
        synthetic_mask (np.ndarray): Synthetic RGB mask where cell i has color
            crm.counter_to_color(i+1).
        brightness_map (Dict[int, float]): Mapping from cell_index -> mean_brightness.
        n_cells (int): Number of cells.

    Returns:
        np.ndarray: Brightness values at each pixel (H x W), dtype=float32.
    """
    h, w, _ = synthetic_mask.shape
    brightness_image = np.zeros((h, w), dtype=np.float32)

    for cell_idx in range(n_cells):
        # Synthetic mask color for cell i is counter_to_color(i+1)
        synthetic_color = np.array(crm.counter_to_color(cell_idx + 1))

        # Find pixels belonging to this cell
        cell_mask = np.all(synthetic_mask == synthetic_color, axis=2)

        # Apply brightness from original
        if cell_idx in brightness_map and np.any(cell_mask):
            brightness_image[cell_mask] = brightness_map[cell_idx]

    return brightness_image


def add_brightness_noise(
    brightness_map: np.ndarray,
    synthetic_mask: np.ndarray,
    noise_strength: float,
    seed: int = None,
) -> np.ndarray:
    """
    Add Perlin-like noise variation around mean brightness values.

    Creates smooth spatial variations in brightness within each bacterium,
    making the result look more natural.

    Args:
        brightness_map (np.ndarray): Target brightness map (H x W).
        synthetic_mask (np.ndarray): Synthetic RGB mask to identify bacteria regions.
        noise_strength (float): Strength of noise variation (0-1). Typical range: 0.1-0.3.
        seed (int): Random seed for reproducibility. If None, uses random state.

    Returns:
        np.ndarray: Brightness map with added noise (H x W), dtype=float32.
    """
    if noise_strength <= 0.0:
        return brightness_map

    rng = np.random.default_rng(seed)

    h, w = brightness_map.shape

    # Create multi-scale noise (2 octaves for subtle variation)
    noise = np.zeros((h, w), dtype=np.float32)

    for octave in range(2):
        frequency = 2**octave
        amplitude = 0.5**octave

        # Coarse noise grid
        octave_h = max(h // (frequency * 8), 4)
        octave_w = max(w // (frequency * 8), 4)
        octave_noise = rng.standard_normal((octave_h, octave_w))
        octave_noise = gaussian_filter(octave_noise, sigma=1.0)

        # Resize to full resolution
        zoom_factors = (h / octave_h, w / octave_w)
        octave_noise = zoom(octave_noise, zoom_factors, order=3)

        noise += octave_noise * amplitude

    # Normalize noise to [-1, +1]
    noise = (noise - noise.mean()) / (noise.std() + 1e-10)

    # Apply noise additively (scaled by noise_strength and local brightness)
    bacteria_mask = np.any(synthetic_mask > 0, axis=2)
    result = brightness_map.copy()

    # Scale noise by local brightness mean and noise_strength
    # This makes brighter areas have proportionally larger variations
    result[bacteria_mask] = result[bacteria_mask] + (
        noise[bacteria_mask] * noise_strength * result[bacteria_mask]
    )

    # Ensure non-negative values
    result = np.clip(result, 0.0, None)

    return result


def apply_original_brightness(
    synthetic_image: np.ndarray,
    synthetic_mask: np.ndarray,
    original_image: np.ndarray,
    original_mask: np.ndarray,
    colors: list,
    noise_strength: float = 0.0,
    seed: int = None,
) -> np.ndarray:
    """
    Apply brightness from original bacteria to synthetic bacteria.

    Extracts mean brightness from each original bacterium and applies it
    to the corresponding synthetic bacterium. This creates realistic
    brightness variations that match the original microscope image.

    Args:
        synthetic_image (np.ndarray): Synthetic microscope image (H x W x C or H x W),
            uint8 or float.
        synthetic_mask (np.ndarray): Synthetic RGB labeled mask (H x W x 3) where cell i
            has color crm.counter_to_color(i+1).
        original_image (np.ndarray): Original microscope image (grayscale or RGB).
        original_mask (np.ndarray): Original RGB labeled mask (H x W x 3) - may have
            color reuse.
        colors (list): List of colors from crm.extract_positions(), maps cell index to
            original mask color.
        noise_strength (float): Strength of Perlin-like noise variation (0 = no noise,
            just mean). Typical range: 0.1-0.3 for subtle variation.
        seed (int): Random seed for reproducibility. If None, uses random state.

    Returns:
        np.ndarray: Additive brightness adjustment map (same shape as synthetic_image).
            Values to be added to the image to match original brightness.
    """
    # Remember input dtype
    input_dtype = synthetic_image.dtype
    is_uint8 = input_dtype == np.uint8
    n_cells = len(colors)

    # Handle empty case
    if n_cells == 0:
        if len(synthetic_image.shape) == 3:
            return np.zeros_like(synthetic_image, dtype=np.float32)
        else:
            return np.zeros(synthetic_image.shape, dtype=np.float32)

    # Step 1: Extract mean brightness from original bacteria
    original_brightness = extract_original_brightness(
        original_image, original_mask, colors
    )

    # Step 2: Create brightness map for synthetic bacteria
    target_brightness = create_synthetic_brightness_map(
        synthetic_mask, original_brightness, n_cells
    )

    # Step 3: Add optional Perlin noise
    if noise_strength > 0.0:
        target_brightness = add_brightness_noise(
            target_brightness, synthetic_mask, noise_strength, seed
        )

    # Step 4: Compute adjustment (target - current)
    # Get current synthetic brightness
    if len(synthetic_image.shape) == 3:
        current_brightness = np.mean(synthetic_image, axis=2).astype(np.float32)
    else:
        current_brightness = synthetic_image.astype(np.float32)

    # Calculate adjustment only for bacteria regions
    bacteria_mask = np.any(synthetic_mask > 0, axis=2)

    adjustment = np.zeros_like(target_brightness, dtype=np.float32)
    adjustment[bacteria_mask] = (
        target_brightness[bacteria_mask] - current_brightness[bacteria_mask]
    )

    # Scale for float images in [0, 1] range
    if not is_uint8 and synthetic_image.max() <= 1.0:
        adjustment = adjustment / 255.0

    # Match image dimensions (grayscale vs RGB)
    if len(synthetic_image.shape) == 3:
        # RGB - expand to 3 channels
        adjustment = np.stack([adjustment, adjustment, adjustment], axis=2)

    return adjustment


def compute_age_based_brightness(
    synthetic_mask: np.ndarray,
    cell_ages: Dict,
    cell_colors: Dict,
    brightness_range: tuple = (0.8, 0.3),
    max_age: int = None,
) -> np.ndarray:
    """
    Compute brightness map based on cell age.

    Creates a brightness map where younger cells are brighter and older cells
    are darker, using linear interpolation between brightness_range values.

    Args:
        synthetic_mask (np.ndarray): Synthetic RGB mask (H x W x 3) where each cell has
            a unique color.
        cell_ages (Dict): Dictionary mapping cell identifier to age (in iterations).
            Can use CellIdentifier objects or any hashable key.
        cell_colors (Dict): Dictionary mapping cell identifier to RGB color tuple.
            Must have same keys as cell_ages.
        brightness_range (tuple): (young_brightness, old_brightness). Young cells (age=0)
            get the first value, oldest cells get the second value. Linear interpolation
            in between.
        max_age (int): Maximum age for normalization. If None, uses max observed age from
            cell_ages. When max_age=0 or all cells have same age, all cells get
            young_brightness.

    Returns:
        np.ndarray: Brightness values at each pixel (H x W), dtype=float32.
            Values are in range [0, 1] representing brightness multiplier.
    """
    h, w = synthetic_mask.shape[:2]
    brightness_image = np.zeros((h, w), dtype=np.float32)

    if len(cell_ages) == 0:
        return brightness_image

    rng = np.random.default_rng()

    # Determine max age for normalization
    ages = list(cell_ages.values())
    if max_age is None:
        max_age = max(ages) if ages else 0

    young_brightness, old_brightness = brightness_range

    for cell_id, age in cell_ages.items():
        if cell_id not in cell_colors:
            continue

        color = cell_colors[cell_id]
        color_array = np.array(color)

        # Find pixels belonging to this cell
        cell_mask = np.all(synthetic_mask == color_array, axis=2)

        if not np.any(cell_mask):
            continue

        # Compute normalized age [0, 1]
        if max_age > 0:
            normalized_age = min(age / max_age, 1.0)
        else:
            normalized_age = 0.0

        # Linear interpolation: young (0) -> young_brightness, old (1) -> old_brightness
        brightness = young_brightness + normalized_age * (
            old_brightness - young_brightness
        )

        # Vary brightness between cells by adding small random variation (±10% of the range)
        variation = rng.uniform(0.9, 1.1)

        brightness_image[cell_mask] = brightness * variation

    return brightness_image


def apply_age_based_brightness(
    synthetic_image: np.ndarray,
    synthetic_mask: np.ndarray,
    cell_ages: Dict,
    cell_colors: Dict,
    brightness_range: tuple = (0.8, 0.3),
    max_age: int = None,
    noise_strength: float = 0.0,
    seed: int = None,
) -> np.ndarray:
    """
    Apply age-based brightness to synthetic bacteria image.

    Creates realistic brightness variations based on cell age, where older
    cells appear darker than younger cells.

    Args:
        synthetic_image (np.ndarray): Synthetic microscope image (H x W x C or H x W),
            uint8 or float.
        synthetic_mask (np.ndarray): Synthetic RGB labeled mask (H x W x 3).
        cell_ages (Dict): Dictionary mapping cell identifier to age (in iterations).
        cell_colors (Dict): Dictionary mapping cell identifier to RGB color tuple.
        brightness_range (tuple): (young_brightness, old_brightness).
        max_age (int): Maximum age for normalization. If None, uses max observed age.
        noise_strength (float): Strength of Perlin-like noise variation (0 = no noise).
        seed (int): Random seed for reproducibility. If None, uses random state.

    Returns:
        np.ndarray: Additive brightness adjustment map (same shape as synthetic_image).
    """
    input_dtype = synthetic_image.dtype
    is_uint8 = input_dtype == np.uint8

    if len(cell_ages) == 0:
        if len(synthetic_image.shape) == 3:
            return np.zeros_like(synthetic_image, dtype=np.float32)
        else:
            return np.zeros(synthetic_image.shape, dtype=np.float32)

    # Compute target brightness based on age
    target_brightness = compute_age_based_brightness(
        synthetic_mask, cell_ages, cell_colors, brightness_range, max_age
    )

    # Scale to 0-255 range for consistency with original brightness function
    target_brightness = target_brightness * 255.0

    # Add optional noise
    if noise_strength > 0.0:
        target_brightness = add_brightness_noise(
            target_brightness, synthetic_mask, noise_strength, seed
        )

    # Get current synthetic brightness
    if len(synthetic_image.shape) == 3:
        current_brightness = np.mean(synthetic_image, axis=2).astype(np.float32)
    else:
        current_brightness = synthetic_image.astype(np.float32)

    # Calculate adjustment only for bacteria regions
    bacteria_mask = np.any(synthetic_mask > 0, axis=2)

    adjustment = np.zeros_like(target_brightness, dtype=np.float32)
    adjustment[bacteria_mask] = (
        target_brightness[bacteria_mask] - current_brightness[bacteria_mask]
    )

    # Scale for float images in [0, 1] range
    if not is_uint8 and synthetic_image.max() <= 1.0:
        adjustment = adjustment / 255.0

    # Match image dimensions (grayscale vs RGB)
    if len(synthetic_image.shape) == 3:
        adjustment = np.stack([adjustment, adjustment, adjustment], axis=2)

    return adjustment
