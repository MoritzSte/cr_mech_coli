"""
Phase contrast microscopy background generation.

This module provides functions to generate realistic synthetic backgrounds
that mimic the appearance of phase contrast microscopy images.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from typing import Tuple


def create_base_background(
    shape: Tuple[int, int],
    noise_scale: int = 20,
    base_brightness: float = 0.6,
    gradient_strength: float = 0.05,
    perlin_scale: int = 4,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Create a base background with subtle gradients mimicking illumination variations.

    Args:
        shape (Tuple[int, int]): Shape of the background image (height, width).
        noise_scale (int): Factor controlling illumination variation scale.
        base_brightness (float): Base brightness level (0.0 to 1.0).
        gradient_strength (float): Strength of illumination gradients (0.0 to 1.0).
        perlin_scale (int): Number of octaves for Perlin-like noise.
        rng (np.random.Generator): Random number generator. If None, creates a new one.

    Returns:
        np.ndarray: Base background image with values between 0 and 1.
    """
    if rng is None:
        rng = np.random.default_rng()

    height, width = shape

    # Create multi-scale noise (Perlin-like) for fine-grained illumination variations
    smooth_gradient = np.zeros(shape, dtype=np.float64)

    # Define octave parameters
    num_octaves = perlin_scale  # Number of frequency layers
    persistence = 0.5  # Amplitude decay factor (each octave has half the amplitude)

    for octave in range(num_octaves):
        # Scale factor: 1, 2, 4, 8 (each octave doubles frequency)
        frequency = 2**octave
        # Amplitude factor: 1.0, 0.5, 0.25, 0.125 (each octave halves amplitude)
        amplitude = persistence**octave

        # Create noise at this frequency scale
        # Higher frequency = smaller grid = finer details
        octave_height = max(height // (noise_scale * frequency), 4)
        octave_width = max(width // (noise_scale * frequency), 4)

        # Generate random noise for this octave
        octave_noise = rng.standard_normal((octave_height, octave_width))

        # Smooth it slightly to create coherent blobs
        octave_noise = gaussian_filter(octave_noise, sigma=10.0)

        # Resize to full resolution using interpolation
        zoom_factors = (height / octave_noise.shape[0], width / octave_noise.shape[1])
        octave_noise = zoom(octave_noise, zoom_factors, order=3)

        # Add weighted octave to accumulated gradient
        smooth_gradient += octave_noise * amplitude

    # Normalize to [-1, 1] and scale by gradient strength
    smooth_gradient = (smooth_gradient - smooth_gradient.mean()) / (
        smooth_gradient.std() + 1e-10
    )
    smooth_gradient = smooth_gradient * gradient_strength

    # Create base background with gradient
    background = np.full(shape, base_brightness, dtype=np.float64) + smooth_gradient

    # Clip to valid range
    background = np.clip(background, 0.0, 1.0)

    return background


def add_darker_spots(
    background: np.ndarray,
    num_spots_range: Tuple[int, int] = (0, 50),
    spot_intensity: float = 0.15,
    spot_size_range: Tuple[float, float] = (2.0, 8.0),
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Add randomly distributed darker spots to simulate debris or artifacts.

    Args:
        background (np.ndarray): Input background image.
        num_spots_range (Tuple[int, int]): Range for number of dark spots (min, max).
            When min == max, exactly that number of spots is created.
        spot_intensity (float): Intensity of dark spots (how much darker than background,
            0.0 to 1.0).
        spot_size_range (Tuple[float, float]): Range of spot sizes (sigma values for
            Gaussian blobs).
        rng (np.random.Generator): Random number generator. If None, creates a new one.

    Returns:
        np.ndarray: Background with added dark spots.
    """
    if rng is None:
        rng = np.random.default_rng()

    result = background.copy()
    height, width = background.shape

    # Determine actual number of spots
    if num_spots_range[0] == num_spots_range[1]:
        num_spots = num_spots_range[0]
    else:
        num_spots = rng.integers(num_spots_range[0], num_spots_range[1] + 1)

    for _ in range(num_spots):
        # Random position
        y = rng.integers(0, height)
        x = rng.integers(0, width)

        # Random size
        sigma = rng.uniform(spot_size_range[0], spot_size_range[1])

        # Random intensity variation (80% to 100% of specified intensity)
        intensity = spot_intensity * rng.uniform(0.8, 1.0)

        # Create Gaussian spot
        y_grid, x_grid = np.ogrid[:height, :width]
        distance = np.sqrt((y_grid - y) ** 2 + (x_grid - x) ** 2)
        gaussian_spot = np.exp(-(distance**2) / (2 * sigma**2))

        # Subtract dark spot from background
        result = result - gaussian_spot * intensity

    # Clip to valid range
    result = np.clip(result, 0.0, 1.0)

    return result


def add_lighter_spots(
    background: np.ndarray,
    num_spots_range: Tuple[int, int] = (0, 30),
    spot_intensity: float = 0.12,
    spot_size_range: Tuple[float, float] = (3.0, 12.0),
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Add randomly distributed lighter spots to simulate phase artifacts or bright debris.

    Args:
        background (np.ndarray): Input background image.
        num_spots_range (Tuple[int, int]): Range for number of light spots (min, max).
            When min == max, exactly that number of spots is created.
        spot_intensity (float): Intensity of light spots (how much brighter than background,
            0.0 to 1.0).
        spot_size_range (Tuple[float, float]): Range of spot sizes (sigma values for
            Gaussian blobs).
        rng (np.random.Generator): Random number generator. If None, creates a new one.

    Returns:
        np.ndarray: Background with added light spots.
    """
    if rng is None:
        rng = np.random.default_rng()

    result = background.copy()
    height, width = background.shape

    # Determine actual number of spots
    if num_spots_range[0] == num_spots_range[1]:
        num_spots = num_spots_range[0]
    else:
        num_spots = rng.integers(num_spots_range[0], num_spots_range[1] + 1)

    for _ in range(num_spots):
        # Random position
        y = rng.integers(0, height)
        x = rng.integers(0, width)

        # Random size (lighter spots tend to be slightly larger)
        sigma = rng.uniform(spot_size_range[0], spot_size_range[1])

        # Random intensity variation (70% to 100% of specified intensity)
        intensity = spot_intensity * rng.uniform(0.7, 1.0)

        # Create Gaussian spot
        y_grid, x_grid = np.ogrid[:height, :width]
        distance = np.sqrt((y_grid - y) ** 2 + (x_grid - x) ** 2)
        gaussian_spot = np.exp(-(distance**2) / (2 * sigma**2))

        # Add bright spot to background
        result = result + gaussian_spot * intensity

    # Clip to valid range
    result = np.clip(result, 0.0, 1.0)

    return result


def add_fine_texture(
    background: np.ndarray,
    texture_strength: float = 0.02,
    texture_scale: float = 1.5,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Add fine texture to simulate microscopic surface variations and optical artifacts.

    Args:
        background (np.ndarray): Input background image.
        texture_strength (float): Strength of the texture (0.0 to 1.0).
        texture_scale (float): Smoothness of texture (higher = smoother).
        rng (np.random.Generator): Random number generator. If None, creates a new one.

    Returns:
        np.ndarray: Background with added fine texture.
    """
    if rng is None:
        rng = np.random.default_rng()

    height, width = background.shape

    # Create fine random texture
    texture = rng.standard_normal((height, width))
    texture = gaussian_filter(texture, sigma=texture_scale)

    # Normalize and scale
    texture = (texture - texture.mean()) / (texture.std() + 1e-10)
    texture = texture * texture_strength

    # Add to background
    result = background + texture

    # Clip to valid range
    result = np.clip(result, 0.0, 1.0)

    return result


def add_gaussian_blur(background: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Apply Gaussian blur to the background.

    Args:
        background (np.ndarray): Input background image.
        sigma (float): Standard deviation for Gaussian kernel.

    Returns:
        np.ndarray: Blurred background image.
    """
    return gaussian_filter(background, sigma=sigma)


def generate_phase_contrast_background(
    shape: Tuple[int, int] = (512, 512),
    noise_scale: int = 20,
    base_brightness: float = 0.6,
    gradient_strength: float = 0.05,
    perlin_scale: int = 4,
    num_dark_spots_range: Tuple[int, int] = (0, 50),
    dark_spot_intensity: float = 0.15,
    dark_spot_size_range: Tuple[float, float] = (2.0, 8.0),
    num_light_spots_range: Tuple[int, int] = (0, 30),
    light_spot_intensity: float = 0.12,
    light_spot_size_range: Tuple[float, float] = (3.0, 12.0),
    texture_strength: float = 0.02,
    texture_scale: float = 1.5,
    blur_sigma: float = 3.0,
    seed: int = None,
    return_uint8: bool = True,
) -> np.ndarray:
    """
    Generate a complete phase contrast microscopy background with all features.

    This is the main method that combines all steps to create a realistic
    phase contrast microscope background.

    Args:
        shape (Tuple[int, int]): Shape of the background image (height, width).
        noise_scale (int): Factor controlling illumination variation scale.
        base_brightness (float): Base brightness level (0.0=black to 1.0=white).
        gradient_strength (float): Strength of illumination gradients (0.0 to 1.0).
        perlin_scale (int): Number of octaves for Perlin-like noise.
        num_dark_spots_range (Tuple[int, int]): Range for number of dark spots (min, max).
            When min == max, exactly that number of spots is created.
        dark_spot_intensity (float): How much darker the dark spots are (0.0 to 1.0).
        dark_spot_size_range (Tuple[float, float]): Size range for dark spots.
        num_light_spots_range (Tuple[int, int]): Range for number of light spots (min, max).
            When min == max, exactly that number of spots is created.
        light_spot_intensity (float): How much brighter the light spots are (0.0 to 1.0).
        light_spot_size_range (Tuple[float, float]): Size range for light spots.
        texture_strength (float): Strength of fine texture (0.0 to 1.0).
        texture_scale (float): Smoothness of texture (higher = smoother).
        blur_sigma (float): Standard deviation for Gaussian blur to simulate optical effects.
        seed (int): Random seed for reproducibility. If None, uses random state.
        return_uint8 (bool): If True, return uint8 array (0-255), otherwise float (0.0-1.0).

    Returns:
        np.ndarray: Complete synthetic phase contrast background.
    """

    # Create RNG from seed
    rng = np.random.default_rng(seed)

    # Step 1: Create base background with gradients
    background = create_base_background(
        shape=shape,
        noise_scale=noise_scale,
        base_brightness=base_brightness,
        gradient_strength=gradient_strength,
        perlin_scale=perlin_scale,
        rng=rng,
    )

    # Step 2: Add darker spots (debris, artifacts)
    background = add_darker_spots(
        background=background,
        num_spots_range=num_dark_spots_range,
        spot_intensity=dark_spot_intensity,
        spot_size_range=dark_spot_size_range,
        rng=rng,
    )

    # Step 3: Add lighter spots (bright artifacts, phase halos)
    background = add_lighter_spots(
        background=background,
        num_spots_range=num_light_spots_range,
        spot_intensity=light_spot_intensity,
        spot_size_range=light_spot_size_range,
        rng=rng,
    )

    # Step 4: Add fine texture
    background = add_fine_texture(
        background=background,
        texture_strength=texture_strength,
        texture_scale=texture_scale,
        rng=rng,
    )

    # Step 5: Apply Gaussian blur to simulate optical effects
    background = add_gaussian_blur(background, sigma=blur_sigma)

    # Convert to uint8 if requested
    if return_uint8:
        background = (background * 255).astype(np.uint8)

    return background
